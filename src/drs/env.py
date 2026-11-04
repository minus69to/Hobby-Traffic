"""Traffic-signal control environment with the safety shield in the loop.

`TrafficSignalEnv` controls every junction of a network at a fixed decision interval.  It is a
multi-agent environment with a shared observation/action layout per junction (so one policy can be
parameter-shared across junctions); for the `single` network it has exactly one agent.  A
Gymnasium wrapper for the single-junction case is provided in :class:`SingleJunctionGymEnv`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from drs.config import SimConfig
from drs.demand import ensure_routes
from drs.shield import SafetyShield
from drs.sumo_sim import JunctionState, SumoSim, observation_size, state_to_vector
from drs.topology import DIRS

MAX_JUNCTIONS = 4


@dataclass
class RewardWeights:
    w_wait_delta: float = 1.0
    w_pressure: float = 0.2
    w_ped_wait: float = 0.3
    w_switch: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> RewardWeights:
        d = d or {}
        return RewardWeights(**{k: float(v) for k, v in d.items() if k in RewardWeights.__dataclass_fields__})


@dataclass
class SensorNoise:
    """Gaussian noise on normalised readings plus a fraction of dropped (zeroed) readings."""

    gaussian_std: float = 0.0
    dropout: float = 0.0

    def apply(self, s: JunctionState, rng: np.random.Generator) -> JunctionState:
        if self.gaussian_std <= 0 and self.dropout <= 0:
            return s

        def noisy(x: np.ndarray, scale: float) -> np.ndarray:
            y = x + rng.normal(0.0, self.gaussian_std * scale, size=x.shape)
            if self.dropout > 0:
                y = np.where(rng.random(x.shape) < self.dropout, 0.0, y)
            return np.clip(y, 0.0, None)

        s.lane_halting = noisy(s.lane_halting, 40.0)
        s.lane_max_wait = noisy(s.lane_max_wait, 300.0)
        s.lane_count = noisy(s.lane_count, 40.0)
        s.class_counts = noisy(s.class_counts, 20.0)
        s.ped_waiting = noisy(s.ped_waiting, 10.0)
        s.out_lane_halting = {k: float(noisy(np.array([v]), 40.0)[0]) for k, v in s.out_lane_halting.items()}
        return s


@dataclass
class EpisodeRecord:
    """Everything the evaluator and the replay exporter need from one episode."""

    scenario: str
    seed: int
    network: str
    per_second: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    tls_states: dict[str, list[str]] = field(default_factory=dict)
    shield_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    interventions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    ped_wait_seconds: float = 0.0
    ped_wait_samples: int = 0
    total_reward: float = 0.0
    tripinfo: Path | None = None


class TrafficSignalEnv:
    """Multi-junction control loop around :class:`SumoSim` and one :class:`SafetyShield` per junction."""

    def __init__(
        self,
        cfg: SimConfig,
        network: str,
        scenarios: dict[str, float] | str,
        seeds: list[int],
        *,
        reward: RewardWeights | None = None,
        episode_seconds: int | None = None,
        step_length: float | None = None,
        sensor_noise: SensorNoise | None = None,
        rng_seed: int = 0,
        record: bool = False,
        tripinfo_dir: Path | None = None,
        fcd_path: Path | None = None,
        step_hook: Callable[[SumoSim, dict[str, str]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.scenario_weights = {scenarios: 1.0} if isinstance(scenarios, str) else dict(scenarios)
        self.seeds = list(seeds)
        self.reward_w = reward or RewardWeights()
        self.episode_seconds = episode_seconds or cfg.sumo.episode_seconds
        self.step_length = step_length or cfg.sumo.step_length
        self.decision_interval = cfg.sumo.decision_interval
        self.noise = sensor_noise
        self.rng = np.random.default_rng(rng_seed)
        self.record = record
        self.tripinfo_dir = tripinfo_dir
        self.fcd_path = fcd_path
        self.step_hook = step_hook
        self.sim: SumoSim | None = None
        self.shields: dict[str, SafetyShield] = {}
        self.junction_ids: list[str] = []
        self.n_phases = cfg.signal.n_phases
        self.obs_dim = observation_size(12, self.n_phases, MAX_JUNCTIONS)
        self.scenario = ""
        self.seed = 0
        self._prev_wait: dict[str, float] = {}
        self._episode_steps = 0
        self.max_steps = int(self.episode_seconds // self.decision_interval)
        self.record_data: EpisodeRecord | None = None
        self.last_states: dict[str, JunctionState] = {}
        self._episode_counter = 0
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def n_agents(self) -> int:
        return len(self.junction_ids)

    def _pick_scenario(self) -> str:
        names = list(self.scenario_weights)
        w = np.array([self.scenario_weights[n] for n in names], dtype=float)
        return str(self.rng.choice(names, p=w / w.sum()))

    def _tripinfo_path(self) -> Path | None:
        if self.tripinfo_dir is None and not self.record:
            return None
        if self.tripinfo_dir is None:
            if self._tmpdir is None:
                self._tmpdir = tempfile.TemporaryDirectory(prefix="drs_")
            base = Path(self._tmpdir.name)
        else:
            base = self.tripinfo_dir
            base.mkdir(parents=True, exist_ok=True)
        return base / f"tripinfo_{self.network}_{self.scenario}_s{self.seed}_{self._episode_counter}.xml"

    def set_scenario_weights(self, weights: dict[str, float]) -> None:
        self.scenario_weights = dict(weights)

    # ------------------------------------------------------------------ gym-like API
    def reset(self, scenario: str | None = None, seed: int | None = None) -> np.ndarray:
        self.close()
        self.scenario = scenario or self._pick_scenario()
        self.seed = int(seed if seed is not None else self.rng.choice(self.seeds))
        self._episode_counter += 1
        scen = self.cfg.scenario(self.scenario)
        if self.noise is None and scen.get("sensor_noise"):
            self._episode_noise: SensorNoise | None = SensorNoise(**scen["sensor_noise"])
        else:
            self._episode_noise = self.noise
        route = ensure_routes(self.cfg, self.network, self.scenario, self.seed)
        self.sim = SumoSim(
            self.cfg,
            self.network,
            route,
            self.seed,
            episode_seconds=self.episode_seconds,
            step_length=self.step_length,
            tripinfo=self._tripinfo_path(),
            fcd=self.fcd_path,
        )
        self.sim.start()
        self.junction_ids = self.sim.junction_ids
        self.shields = {jid: SafetyShield(self.sim.tables[jid], self.cfg.signal) for jid in self.junction_ids}
        self._episode_steps = 0
        self.record_data = EpisodeRecord(self.scenario, self.seed, self.network, tripinfo=self.sim.tripinfo)
        if self.record:
            for jid in self.junction_ids:
                self.record_data.per_second[jid] = []
                self.record_data.tls_states[jid] = []
        self._sense()
        self._prev_wait = {jid: s.total_acc_wait for jid, s in self.last_states.items()}
        return self._observations()

    def _sense(self) -> None:
        assert self.sim is not None
        states = {jid: self.sim.junction_state(jid) for jid in self.junction_ids}
        if self._episode_noise is not None:
            states = {jid: self._episode_noise.apply(s, self.rng) for jid, s in states.items()}
        # neighbour pressure: pressure of the junction at the far end of each arm
        pressures = {jid: s.pressure() for jid, s in states.items()}
        for jid, s in states.items():
            np_ = np.zeros(4)
            for k, d in enumerate(DIRS):
                arm = self.sim.spec.junctions[jid].arms[d]
                if not arm.external:
                    np_[k] = pressures.get(arm.neighbour, 0.0)
            s.neighbour_pressure = np_
        self.last_states = states

    def _observations(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for k, jid in enumerate(self.junction_ids):
            sh = self.shields[jid]
            onehot = np.zeros(self.n_phases)
            onehot[sh.phase if not sh.in_clearance else sh.target] = 1.0
            jonehot = np.zeros(MAX_JUNCTIONS)
            jonehot[k % MAX_JUNCTIONS] = 1.0
            obs[k] = state_to_vector(
                self.last_states[jid], onehot, sh.time_in_green, sh.in_clearance, jonehot, self.cfg.signal.max_green
            )
        return obs

    def action_masks(self) -> np.ndarray:
        return np.stack([self.shields[jid].valid_actions() for jid in self.junction_ids])

    def step(self, actions: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray, bool, list[dict[str, Any]]]:
        assert self.sim is not None, "call reset() first"
        switched = np.zeros(self.n_agents)
        for k, jid in enumerate(self.junction_ids):
            sh = self.shields[jid]
            before = sh.mode
            sh.request(int(actions[k]))
            switched[k] = 1.0 if (before != sh.mode) else 0.0
        self._advance(self.decision_interval)
        self._sense()
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        infos: list[dict[str, Any]] = []
        for k, jid in enumerate(self.junction_ids):
            s = self.last_states[jid]
            dwait = s.total_acc_wait - self._prev_wait[jid]
            self._prev_wait[jid] = s.total_acc_wait
            w = self.reward_w
            r = (
                -w.w_wait_delta * dwait / 100.0
                - w.w_pressure * max(s.pressure(), 0.0) / 100.0
                - w.w_ped_wait * float(s.ped_waiting.sum()) / 10.0
                - w.w_switch * switched[k]
            )
            rewards[k] = r
            infos.append({"junction": jid, "queue": float(s.lane_halting.sum()), "pressure": s.pressure()})
        self._episode_steps += 1
        done = self._episode_steps >= self.max_steps
        if self.record_data is not None:
            self.record_data.total_reward += float(rewards.sum())
        return self._observations(), rewards, done, infos

    def _advance(self, seconds: int) -> None:
        assert self.sim is not None
        n = round(seconds / self.step_length)
        for _ in range(n):
            shown = {jid: self.shields[jid].tick(self.step_length) for jid in self.junction_ids}
            for jid, state in shown.items():
                self.sim.set_state(jid, state)
            self.sim.step()
            if self.step_hook is not None:
                self.step_hook(self.sim, shown)
            if self.record and self.record_data is not None:
                t = self.sim.time
                if abs(t - round(t)) < 1e-6:  # once per simulated second
                    for jid in self.junction_ids:
                        m = self.sim.per_second_metrics(jid)
                        m["t"] = round(t)
                        self.record_data.per_second[jid].append(m)
                        self.record_data.tls_states[jid].append(self.shields[jid].current_state())
                        if t > self.cfg.sumo.warmup_seconds:
                            self.record_data.ped_wait_seconds += m["ped_waiting"]
                            self.record_data.ped_wait_samples += 1

    def finish(self) -> EpisodeRecord:
        """Collect shield statistics and close SUMO (flushes tripinfo output)."""
        rec = self.record_data or EpisodeRecord(self.scenario, self.seed, self.network)
        for jid, sh in self.shields.items():
            rec.shield_stats[jid] = sh.stats.as_dict()
            rec.interventions[jid] = [i.__dict__ for i in sh.log]
        self.close()
        return rec

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None


class SingleJunctionGymEnv(gym.Env[np.ndarray, int]):
    """Gymnasium view of :class:`TrafficSignalEnv` for networks with exactly one junction."""

    metadata = {"render_modes": []}

    def __init__(self, env: TrafficSignalEnv) -> None:
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(env.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(env.n_phases)  # type: ignore[assignment]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs = self.env.reset()
        return obs[0], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, rew, done, infos = self.env.step([int(action)])
        return obs[0], float(rew[0]), done, False, infos[0]

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env.action_masks()[0])

    def close(self) -> None:
        self.env.close()
