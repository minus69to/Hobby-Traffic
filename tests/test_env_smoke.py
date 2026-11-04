"""Slow smoke tests that start SUMO (marked `slow`)."""

from __future__ import annotations

import numpy as np
import pytest

from drs.config import SimConfig
from drs.controllers import FixedTimeController, MaxPressureController
from drs.env import TrafficSignalEnv
from drs.metrics import episode_metrics
from drs.runner import run_episode


@pytest.mark.slow
def test_short_episode_fixed_time_and_max_pressure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = SimConfig.load()
    for ctrl in (FixedTimeController(cfg), MaxPressureController()):
        env = TrafficSignalEnv(cfg, "single", "offpeak", [13], episode_seconds=300, record=True, tripinfo_dir=tmp_path)
        rec = run_episode(env, ctrl)
        m = episode_metrics(rec, 60, 300)
        assert m["throughput"] > 0
        assert m["shield_interventions"] == 0
        assert rec.shield_stats["C"]["switches"] > 0


@pytest.mark.slow
def test_observation_and_mask_shapes() -> None:
    cfg = SimConfig.load()
    env = TrafficSignalEnv(cfg, "single", "offpeak", [42], episode_seconds=60)
    obs = env.reset()
    assert obs.shape == (1, env.obs_dim)
    assert env.action_masks().shape == (1, 4)
    assert env.action_masks()[0].tolist() == [True, False, False, False]
    obs2, rew, done, infos = env.step([0])
    assert obs2.shape == obs.shape and rew.shape == (1,) and not done
    assert np.isfinite(obs2).all()
    env.close()
