"""Graph-attention extractor shapes and the neighbour-observation layout on the grid."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from drs.config import SimConfig
from drs.gnn import GraphAttentionExtractor, neighbour_layout


def test_extractor_masks_absent_neighbours() -> None:
    base, full = neighbour_layout(4)
    space = gym.spaces.Box(-5, 5, shape=(full,), dtype=np.float32)
    ext = GraphAttentionExtractor(space, embed_dim=16)
    x = torch.zeros(3, full)
    x[0, base + 48] = 1.0  # one neighbour present (arm N)
    out = ext(x)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()
    # with no neighbours the aggregated half is exactly zero (tanh(0))
    assert torch.allclose(out[1, 16:], torch.zeros(16))


@pytest.mark.slow
def test_grid_env_neighbour_observations() -> None:
    from drs.env import TrafficSignalEnv

    cfg = SimConfig.load()
    env = TrafficSignalEnv(cfg, "grid2x2", "offpeak", [13], episode_seconds=30, neighbour_obs=True)
    obs = env.reset()
    base, full = neighbour_layout(cfg.signal.n_phases)
    assert obs.shape == (4, full)
    presence = obs[:, base + 48 :: 49]
    assert presence.shape == (4, 4) and presence.sum(axis=1).tolist() == [
        2,
        2,
        2,
        2,
    ]  # each grid junction has two internal arms
    env.close()
