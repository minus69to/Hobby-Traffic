"""Measure simulation speed for one hour of peak traffic on the single junction (M0 requirement)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from drs.config import REPO_ROOT, SimConfig
from drs.controllers import FixedTimeController
from drs.env import TrafficSignalEnv
from drs.runner import run_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sim.yaml")
    parser.add_argument("--network", default="single")
    parser.add_argument("--scenario", default="morning_peak")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="results/speedtest.json")
    args = parser.parse_args()
    cfg = SimConfig.load(args.config)
    env = TrafficSignalEnv(cfg, args.network, args.scenario, [args.seed], record=True)
    ctrl = FixedTimeController(cfg)
    t0 = time.perf_counter()
    rec = run_episode(env, ctrl)
    wall = time.perf_counter() - t0
    sim_steps = cfg.sumo.episode_seconds / cfg.sumo.step_length
    out = {
        "network": args.network,
        "scenario": args.scenario,
        "simulated_seconds": cfg.sumo.episode_seconds,
        "wall_seconds": round(wall, 2),
        "sim_steps_per_second": round(sim_steps / wall, 1),
        "decision_steps_per_second": round(cfg.sumo.episode_seconds / cfg.sumo.decision_interval / wall, 1),
        "shield_stats": rec.shield_stats,
        "ped_wait_samples": rec.ped_wait_samples,
    }
    Path(REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
