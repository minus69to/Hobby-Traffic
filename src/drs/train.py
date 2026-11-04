"""Train MaskablePPO on the shielded traffic-signal environment.

Run ``python -m drs.train --config configs/train.yaml``.  Writes, per seed, under ``out_dir/seed<N>``:
``best/`` (policy + VecNormalize stats, chosen by validation return), ``final/``, ``tb/`` (TensorBoard),
``progress.csv`` and ``train_meta.json`` with wall-clock time.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.env import RewardWeights, TrafficSignalEnv
from drs.vec_env import MultiJunctionVecEnv


class EnvFactory:
    """Picklable factory so worker processes (forkserver) can build their own environment."""

    def __init__(
        self,
        sim_config: str,
        network: str,
        scenarios: dict[str, float],
        seeds: list[int],
        reward: RewardWeights,
        rng_seed: int,
        episode_seconds: int | None,
    ) -> None:
        self.sim_config = sim_config
        self.network = network
        self.scenarios = scenarios
        self.seeds = seeds
        self.reward = reward
        self.rng_seed = rng_seed
        self.episode_seconds = episode_seconds

    def __call__(self) -> TrafficSignalEnv:
        cfg = SimConfig.load(self.sim_config)
        return TrafficSignalEnv(
            cfg,
            self.network,
            self.scenarios,
            self.seeds,
            reward=self.reward,
            rng_seed=self.rng_seed,
            episode_seconds=self.episode_seconds,
        )


def make_env_fn(
    cfg: SimConfig,
    network: str,
    scenarios: dict[str, float],
    seeds: list[int],
    reward: RewardWeights,
    rng_seed: int,
    episode_seconds: int | None,
) -> Callable[[], TrafficSignalEnv]:
    return EnvFactory(cfg.path, network, scenarios, seeds, reward, rng_seed, episode_seconds)


def n_junctions(cfg: SimConfig, network: str) -> int:
    return len(cfg.networks[network]["junctions"])


def evaluate_policy(model: MaskablePPO, vec: VecNormalize, n_episodes: int = 1) -> dict[str, float]:
    """Deterministic rollout on the validation VecEnv; returns mean per-slot episode return and stats."""
    obs: Any = vec.reset()
    returns: list[float] = []
    queues: list[float] = []
    shield_interventions = 0
    episodes_done = 0
    ep_ret = np.zeros(vec.num_envs)
    while episodes_done < n_episodes:
        masks = get_action_masks(vec)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, rew, dones, infos = vec.step(actions)
        ep_ret += vec.get_original_reward() if hasattr(vec, "get_original_reward") else rew
        queues.extend(float(i["queue"]) for i in infos)
        if dones.any():
            returns.extend(ep_ret.tolist())
            shield_interventions += sum(int(i.get("shield", {}).get("interventions", 0)) for i in infos)
            ep_ret[:] = 0
            episodes_done += 1
    return {
        "return": float(np.mean(returns)),
        "mean_queue": float(np.mean(queues)) if queues else 0.0,
        "shield_interventions": float(shield_interventions),
    }


class CurriculumAndEvalCallback(BaseCallback):
    def __init__(
        self,
        curriculum: list[dict[str, Any]],
        total_timesteps: int,
        eval_vec: VecNormalize,
        eval_every: int,
        eval_episodes: int,
        out_dir: Path,
        t_start: float,
    ) -> None:
        super().__init__()
        self.curriculum = curriculum
        self.total = total_timesteps
        self.eval_vec = eval_vec
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.out_dir = out_dir
        self.stage = -1
        self.best = -np.inf
        self.last_eval = 0
        self.t_start = t_start
        self.rows: list[dict[str, Any]] = []
        self.ep_returns: list[float] = []

    def _apply_stage(self) -> None:
        frac = self.num_timesteps / self.total
        stage = next((i for i, c in enumerate(self.curriculum) if frac < float(c["until"])), len(self.curriculum) - 1)
        if stage != self.stage:
            self.stage = stage
            self.training_env.env_method("set_scenario_weights", dict(self.curriculum[stage]["scenarios"]))
            if self.verbose:
                print(f"[curriculum] step {self.num_timesteps}: stage {stage} {self.curriculum[stage]['scenarios']}")

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_reward" in info:
                self.ep_returns.append(float(info["episode_reward"]))
        self._apply_stage()
        if self.num_timesteps - self.last_eval >= self.eval_every:
            self.last_eval = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        assert isinstance(self.model, MaskablePPO)
        train_vec = self.model.get_vec_normalize_env()
        assert train_vec is not None
        self.eval_vec.obs_rms = train_vec.obs_rms
        self.eval_vec.training = False
        stats = evaluate_policy(self.model, self.eval_vec, self.eval_episodes)
        wall = time.time() - self.t_start
        row = {
            "timesteps": self.num_timesteps,
            "wall_s": round(wall, 1),
            **{k: round(v, 4) for k, v in stats.items()},
            "train_ep_return_mean": round(float(np.mean(self.ep_returns[-8:])), 3) if self.ep_returns else None,
        }
        self.rows.append(row)
        self.logger.record("eval/return", stats["return"])
        self.logger.record("eval/mean_queue", stats["mean_queue"])
        self.logger.record("eval/shield_interventions", stats["shield_interventions"])
        with (self.out_dir / "progress.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        if stats["return"] > self.best:
            self.best = stats["return"]
            save_policy(self.model, train_vec, self.out_dir / "best")
        print(
            f"[eval] steps={self.num_timesteps} return={stats['return']:.2f} "
            f"queue={stats['mean_queue']:.1f} wall={wall:.0f}s",
            flush=True,
        )


def save_policy(model: MaskablePPO, vec: VecNormalize, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "policy.zip")
    vec.save(str(path / "vecnormalize.pkl"))


def train_seed(tcfg: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = SimConfig.load(tcfg["sim_config"])
    network = str(tcfg["network"])
    out_dir = REPO_ROOT / str(tcfg["out_dir"]) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reward = RewardWeights.from_dict(tcfg.get("reward"))
    episode_seconds = tcfg.get("episode_seconds")
    n_envs = int(tcfg["n_envs"])
    train_seeds = [int(s) for s in cfg.seeds["train"]]
    val_seeds = [int(s) for s in cfg.seeds["validation"]]
    first_stage = dict(tcfg["curriculum"][0]["scenarios"])
    nj = n_junctions(cfg, network)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    fns = [
        make_env_fn(cfg, network, first_stage, train_seeds, reward, seed * 100 + i, episode_seconds)
        for i in range(n_envs)
    ]
    from drs.sumo_sim import observation_size

    obs_dim = observation_size(12, cfg.signal.n_phases, 4)
    venv = MultiJunctionVecEnv(fns, nj, obs_dim, cfg.signal.n_phases)
    vec = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=float(tcfg["ppo"]["gamma"]))
    eval_scen = {"morning_peak": 1.0}
    eval_venv = MultiJunctionVecEnv(
        [make_env_fn(cfg, network, eval_scen, val_seeds, reward, 999, episode_seconds)],
        nj,
        obs_dim,
        cfg.signal.n_phases,
    )
    eval_vec = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=5.0, training=False)
    p = tcfg["ppo"]
    lr0 = float(p["learning_rate"])
    lr: float | Callable[[float], float] = (lambda f: lr0 * f) if p.get("lr_linear_decay", True) else lr0
    act = torch.nn.Tanh if str(p.get("activation", "tanh")) == "tanh" else torch.nn.ReLU
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        n_steps=int(p["n_steps"]),
        batch_size=int(p["batch_size"]),
        n_epochs=int(p["n_epochs"]),
        learning_rate=lr,
        gamma=float(p["gamma"]),
        gae_lambda=float(p["gae_lambda"]),
        clip_range=float(p["clip_range"]),
        ent_coef=float(p["ent_coef"]),
        vf_coef=float(p.get("vf_coef", 0.5)),
        max_grad_norm=float(p.get("max_grad_norm", 0.5)),
        policy_kwargs={
            "net_arch": {"pi": list(p["policy_layers"]), "vf": list(p["policy_layers"])},
            "activation_fn": act,
        },
        tensorboard_log=str(out_dir / "tb"),
        seed=seed,
        device="cpu",
        verbose=0,
    )
    t0 = time.time()
    cb = CurriculumAndEvalCallback(
        list(tcfg["curriculum"]),
        int(tcfg["total_timesteps"]),
        eval_vec,
        int(tcfg["eval_every_steps"]),
        int(tcfg.get("eval_episodes", 1)),
        out_dir,
        t0,
    )
    cb.verbose = 1
    model.learn(total_timesteps=int(tcfg["total_timesteps"]), callback=cb, tb_log_name="ppo")
    wall = time.time() - t0
    save_policy(model, vec, out_dir / "final")
    if not (out_dir / "best").exists():
        save_policy(model, vec, out_dir / "best")
    meta = {
        "seed": seed,
        "network": network,
        "total_timesteps": int(tcfg["total_timesteps"]),
        "n_envs": n_envs,
        "n_junctions": nj,
        "wall_clock_s": round(wall, 1),
        "env_steps_per_s": round(int(tcfg["total_timesteps"]) / wall, 2),
        "best_validation_return": cb.best if cb.best > -np.inf else None,
        "eval_rows": cb.rows,
        "ppo": p,
        "reward": reward.__dict__,
        "curriculum": tcfg["curriculum"],
        "train_seeds": train_seeds,
        "validation_seeds": val_seeds,
        "obs_dim": obs_dim,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    vec.close()
    eval_vec.close()
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--seed", type=int, default=None, help="train only this seed")
    args = parser.parse_args()
    tcfg = load_yaml(args.config)
    seeds = [args.seed] if args.seed is not None else [int(s) for s in tcfg["seeds"]]
    for s in seeds:
        meta = train_seed(tcfg, s)
        print(f"seed {s}: wall {meta['wall_clock_s']} s, best validation return {meta['best_validation_return']}")


if __name__ == "__main__":
    main()
