"""Graph-attention policy encoder for the 2x2 grid (plain PyTorch, parameter-shared across junctions).

Each junction's observation can be extended with the raw local features of its four neighbours
(``TrafficSignalEnv(neighbour_obs=True)``).  The extractor embeds the junction and each neighbour with
a shared MLP, aggregates neighbours with masked attention (external arms are masked out) and
concatenates the result with the junction's own embedding.  Used as ``features_extractor_class`` of
MaskablePPO so the rest of the pipeline (mask, shield, evaluation) is unchanged.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from drs.sumo_sim import observation_size

LOCAL_DIM = 48  # lane queues (12) + lane waits (12) + class counts (24): the part shared with neighbours
N_NEIGHBOURS = 4


def neighbour_layout(n_phases: int = 4) -> tuple[int, int]:
    """(base observation size, size with neighbour features appended)."""
    base = observation_size(12, n_phases, 4)
    return base, base + N_NEIGHBOURS * (LOCAL_DIM + 1)  # +1 presence flag per neighbour


class GraphAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, embed_dim: int = 64, n_phases: int = 4) -> None:
        base, full = neighbour_layout(n_phases)
        assert observation_space.shape is not None and observation_space.shape[0] == full, (
            "neighbour_obs must be enabled"
        )
        super().__init__(observation_space, features_dim=2 * embed_dim)
        self.base = base
        self.node = nn.Sequential(
            nn.Linear(LOCAL_DIM, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim), nn.Tanh()
        )
        self.self_proj = nn.Sequential(nn.Linear(base, embed_dim), nn.Tanh())
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim**-0.5

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        own = obs[:, : self.base]
        neigh = obs[:, self.base :].view(obs.shape[0], N_NEIGHBOURS, LOCAL_DIM + 1)
        present = neigh[:, :, -1] > 0.5  # [B, 4]
        h_self = self.self_proj(own)  # [B, E]
        h_n = self.node(neigh[:, :, :LOCAL_DIM])  # [B, 4, E]
        q = self.q(h_self).unsqueeze(1)  # [B, 1, E]
        k = self.k(h_n)
        v = self.v(h_n)
        att = (q * k).sum(-1) * self.scale  # [B, 4]
        att = att.masked_fill(~present, -1e9)
        w = torch.softmax(att, dim=-1) * present.float()  # zero weight when no neighbour at all
        agg = (w.unsqueeze(-1) * v).sum(1)  # [B, E]
        out: torch.Tensor = torch.cat([h_self, torch.tanh(agg)], dim=-1)
        return out
