"""Run one controller for one episode and return the record."""

from __future__ import annotations

from drs.controllers.base import Controller
from drs.env import EpisodeRecord, TrafficSignalEnv


def run_episode(env: TrafficSignalEnv, controller: Controller, scenario: str | None = None, seed: int | None = None) -> EpisodeRecord:
    obs = env.reset(scenario=scenario, seed=seed)
    controller.reset(env)
    done = False
    while not done:
        masks = env.action_masks()
        actions = controller.act(env.last_states, obs, masks, env.shields)
        obs, _, done, _ = env.step(actions)
    return env.finish()
