"""Multi-process vectorised environment where every junction is one "agent slot".

Each worker process owns one :class:`TrafficSignalEnv` (one SUMO instance).  A network with
``n_j`` junctions contributes ``n_j`` slots, so a parameter-shared policy sees
``n_procs * n_j`` independent observation/action pairs per step, exactly like SB3's
``SubprocVecEnv`` does for single-agent environments.  ``env_method("action_masks")`` returns one
mask per slot, which is what ``sb3_contrib.MaskablePPO`` expects.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from drs.env import TrafficSignalEnv


def _worker(remote: Connection, parent_remote: Connection, env_fn: Callable[[], TrafficSignalEnv]) -> None:
    parent_remote.close()
    env = env_fn()
    obs: np.ndarray | None = None
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rew, done, infos = env.step(data)
                if done:
                    rec = env.finish()
                    for k, info in enumerate(infos):
                        info["terminal_observation"] = obs[k]
                        info["TimeLimit.truncated"] = True  # episodes end by time limit: bootstrap V(s_T)
                        info["episode_reward"] = rec.total_reward
                        info["scenario"] = rec.scenario
                        info["shield"] = rec.shield_stats.get(info["junction"], {})
                    obs = env.reset()
                remote.send((obs, rew, done, infos, env.action_masks()))
            elif cmd == "reset":
                obs = env.reset()
                remote.send((obs, env.action_masks()))
            elif cmd == "masks":
                remote.send(env.action_masks())
            elif cmd == "set_scenario_weights":
                env.set_scenario_weights(data)
                remote.send(True)
            elif cmd == "get_attr":
                remote.send(getattr(env, data, None))
            elif cmd == "n_agents":
                remote.send(env.n_agents)
            elif cmd == "close":
                env.close()
                remote.send(True)
                break
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        env.close()


class MultiJunctionVecEnv(VecEnv):
    """VecEnv over `n_procs` worker processes, each exposing `n_agents` junction slots."""

    def __init__(
        self, env_fns: Sequence[Callable[[], TrafficSignalEnv]], n_agents: int, obs_dim: int, n_phases: int
    ) -> None:
        self.n_procs = len(env_fns)
        self.n_agents = n_agents
        ctx = mp.get_context("forkserver")
        self.remotes: list[Connection] = []
        self.processes = []
        for fn in env_fns:
            remote, work_remote = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(work_remote, remote, fn), daemon=True)
            p.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(p)
        obs_space = gym.spaces.Box(-5.0, 5.0, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Discrete(n_phases)
        super().__init__(self.n_procs * n_agents, obs_space, act_space)
        self._masks = np.ones((self.num_envs, n_phases), dtype=bool)
        self._actions: np.ndarray | None = None

    # ------------------------------------------------------------------ VecEnv API
    def reset(self) -> VecEnvObs:
        for r in self.remotes:
            r.send(("reset", None))
        results = [r.recv() for r in self.remotes]
        self._masks = np.concatenate([m for _, m in results])
        return np.concatenate([o for o, _ in results]).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(self.n_procs, self.n_agents)
        for r, a in zip(self.remotes, self._actions):
            r.send(("step", a.tolist()))

    def step_wait(self) -> VecEnvStepReturn:
        results = [r.recv() for r in self.remotes]
        obs = np.concatenate([o for o, _, _, _, _ in results]).astype(np.float32)
        rews = np.concatenate([r for _, r, _, _, _ in results]).astype(np.float32)
        dones = np.concatenate([[d] * self.n_agents for _, _, d, _, _ in results])
        infos = [info for _, _, _, inf, _ in results for info in inf]
        self._masks = np.concatenate([m for _, _, _, _, m in results])
        return obs, rews, dones, infos

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
                r.recv()
            except (EOFError, BrokenPipeError):  # pragma: no cover
                pass
        for p in self.processes:
            p.join(timeout=10)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        if attr_name == "action_masks":
            return [self.action_masks] * self.num_envs
        for r in self.remotes:
            r.send(("get_attr", attr_name))
        vals = [r.recv() for r in self.remotes]
        return [v for v in vals for _ in range(self.n_agents)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def env_method(
        self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any
    ) -> list[Any]:
        if method_name == "action_masks":
            return list(self._masks)
        if method_name == "set_scenario_weights":
            for r in self.remotes:
                r.send(("set_scenario_weights", method_args[0]))
            return [r.recv() for r in self.remotes]
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False] * self.num_envs

    def action_masks(self) -> np.ndarray:
        return self._masks

    def get_images(self) -> Sequence[np.ndarray | None]:  # pragma: no cover
        return [None] * self.num_envs
