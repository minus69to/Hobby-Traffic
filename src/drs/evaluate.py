"""Evaluate controllers on every scenario with disjoint evaluation seeds.

Run ``python -m drs.evaluate --config configs/eval.yaml``.  Outputs under ``out_dir``:
``eval_summary.json`` (means, std, 95 % bootstrap CIs, percentage change vs fixed-time, per
episode rows), ``main_table.csv`` and ``raw/`` (per-episode metrics and per-second time series).
Every number in the README, report and website is read from these files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.controllers import FixedTimeController, MaxPressureController
from drs.controllers.base import Controller
from drs.controllers.ppo import OnnxController, PPOController
from drs.env import TrafficSignalEnv
from drs.metrics import METRIC_KEYS, episode_metrics
from drs.runner import run_episode

LOWER_IS_BETTER = {k: k not in ("throughput",) for k in METRIC_KEYS}


def build_controller(cfg: SimConfig, spec: dict[str, Any], name: str) -> Controller:
    kind = str(spec["type"])
    if kind == "fixed_time":
        c: Controller = FixedTimeController(cfg)
    elif kind == "max_pressure":
        c = MaxPressureController()
    elif kind == "ppo":
        c = PPOController(str(spec["model"]), name=name)
    elif kind == "onnx":
        c = OnnxController(str(spec["model"]), str(spec["meta"]), name=name)
    else:
        raise KeyError(kind)
    c.name = name
    return c


def _job(args: tuple[str, str, dict[str, Any], str, str, int, int, int | None, str]) -> dict[str, Any]:
    sim_config, network, ctrl_spec, ctrl_name, scenario, seed, ep_index, episode_seconds, out_dir = args
    cfg = SimConfig.load(sim_config)
    raw_dir = REPO_ROOT / out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = TrafficSignalEnv(
        cfg,
        network,
        scenario,
        [seed],
        episode_seconds=episode_seconds,
        record=True,
        tripinfo_dir=raw_dir,
        rng_seed=seed,
    )
    ctrl = build_controller(cfg, ctrl_spec, ctrl_name)
    t0 = time.perf_counter()
    rec = run_episode(env, ctrl, scenario=scenario, seed=seed)
    ep_secs = episode_seconds or cfg.sumo.episode_seconds
    m: dict[str, Any] = episode_metrics(rec, cfg.sumo.warmup_seconds, ep_secs)
    m.update(
        {
            "controller": ctrl_name,
            "scenario": scenario,
            "seed": seed,
            "episode": ep_index,
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    )
    if ep_index == 0:  # keep one per-second time series per (controller, scenario) for the curves
        series = {jid: rows for jid, rows in rec.per_second.items()}
        with gzip.open(raw_dir / f"timeseries_{ctrl_name}_{scenario}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"controller": ctrl_name, "scenario": scenario, "seed": seed, "series": series}, fh)
    if rec.tripinfo is not None and rec.tripinfo.exists():
        rec.tripinfo.unlink()  # keep the raw directory small
    (raw_dir / f"episode_{ctrl_name}_{scenario}_s{seed}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def bootstrap_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_pct_change(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Percentage change of mean(a) vs mean(b) with a paired bootstrap CI (same seeds)."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    base = b.mean()
    pct = (a.mean() / base - 1.0) * 100.0 if base else 0.0
    if n < 2 or not base:
        return float(pct), float(pct), float(pct)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratios = (a[idx].mean(axis=1) / np.maximum(b[idx].mean(axis=1), 1e-9) - 1.0) * 100.0
    return float(pct), float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def aggregate(
    episodes: list[dict[str, Any]],
    controllers: list[str],
    scenarios: list[str],
    n_boot: int,
    baseline: str = "fixed_time",
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    pooled = _pool_ppo(episodes)
    ctrl_names = [*controllers, *(["ppo"] if pooled else [])]
    episodes = episodes + pooled
    summary: dict[str, Any] = {}
    for ctrl in ctrl_names:
        summary[ctrl] = {}
        for scen in scenarios:
            rows = sorted(
                [e for e in episodes if e["controller"] == ctrl and e["scenario"] == scen],
                key=lambda e: (e["seed"], e.get("sub", "")),
            )
            base_rows = sorted(
                [e for e in episodes if e["controller"] == baseline and e["scenario"] == scen], key=lambda e: e["seed"]
            )
            if not rows:
                continue
            entry: dict[str, Any] = {"n": len(rows)}
            for k in METRIC_KEYS:
                x = np.array([float(r[k]) for r in rows])
                lo, hi = bootstrap_ci(x, n_boot, rng)
                item: dict[str, Any] = {
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                    "ci_low": lo,
                    "ci_high": hi,
                }
                if base_rows and ctrl != baseline:
                    b = np.array([float(r[k]) for r in base_rows])
                    if ctrl == "ppo":  # pooled: compare each seed's episodes to the same evaluation seed
                        b = np.array([float(next(br for br in base_rows if br["seed"] == r["seed"])[k]) for r in rows])
                    pct, plo, phi = paired_pct_change(x, b, n_boot, rng)
                    item.update({"pct_vs_fixed": pct, "pct_ci_low": plo, "pct_ci_high": phi})
                entry[k] = item
            summary[ctrl][scen] = entry
    return summary


def _pool_ppo(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in episodes:
        if e["controller"].startswith("ppo_"):
            d = dict(e)
            d["sub"] = e["controller"]
            d["controller"] = "ppo"
            out.append(d)
    return out


def write_table(summary: dict[str, Any], path: Path) -> None:
    cols = [
        "controller",
        "scenario",
        "n",
        *(f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "ci_low", "ci_high", "pct_vs_fixed")),
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ctrl, scens in summary.items():
            for scen, entry in scens.items():
                row: list[Any] = [ctrl, scen, entry["n"]]
                for k in METRIC_KEYS:
                    it = entry[k]
                    row += [
                        round(it["mean"], 3),
                        round(it["ci_low"], 3),
                        round(it["ci_high"], 3),
                        round(it.get("pct_vs_fixed", float("nan")), 2),
                    ]
                w.writerow(row)


def run(ecfg: dict[str, Any], only_controllers: list[str] | None = None) -> dict[str, Any]:
    cfg = SimConfig.load(ecfg["sim_config"])
    network = str(ecfg["network"])
    out_dir = Path(ecfg["out_dir"])
    (REPO_ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    scenarios = [str(s) for s in ecfg["scenarios"]]
    eval_seeds = [int(s) for s in cfg.seeds["eval"]][: int(ecfg["episodes_per_scenario"])]
    episode_seconds = ecfg.get("episode_seconds")
    controllers: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for name, spec in ecfg["controllers"].items():
        if only_controllers and name not in only_controllers:
            continue
        model = spec.get("model")
        if model and not (REPO_ROOT / model).exists() and not (REPO_ROOT / model / "best").exists():
            skipped[name] = f"model not found: {model}"
            continue
        controllers[name] = spec
    jobs = [
        (cfg.path, network, spec, name, scen, seed, k, episode_seconds, str(out_dir))
        for name, spec in controllers.items()
        for scen in scenarios
        for k, seed in enumerate(eval_seeds)
    ]
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    with ctx.Pool(int(ecfg.get("n_workers", 4))) as pool:
        episodes = pool.map(_job, jobs, chunksize=1)
    wall = time.time() - t0
    summary = aggregate(episodes, list(controllers), scenarios, int(ecfg.get("bootstrap_samples", 2000)))
    result = {
        "meta": {
            "network": network,
            "scenarios": scenarios,
            "controllers": list(controllers),
            "skipped_controllers": skipped,
            "episodes_per_scenario": len(eval_seeds),
            "eval_seeds": eval_seeds,
            "episode_seconds": episode_seconds or cfg.sumo.episode_seconds,
            "warmup_seconds": cfg.sumo.warmup_seconds,
            "metrics": list(METRIC_KEYS),
            "lower_is_better": LOWER_IS_BETTER,
            "wall_clock_s": round(wall, 1),
            "n_episodes": len(episodes),
            "note": "Synthetic Dhaka-inspired demand in SUMO; no real-world data or validation.",
        },
        "summary": summary,
        "episodes": episodes,
    }
    (REPO_ROOT / out_dir / "eval_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_table(summary, REPO_ROOT / out_dir / "main_table.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--controllers", nargs="*", default=None)
    args = parser.parse_args()
    res = run(load_yaml(args.config), args.controllers)
    print(f"{res['meta']['n_episodes']} episodes in {res['meta']['wall_clock_s']} s -> {args.config}")
    for ctrl, scens in res["summary"].items():
        for scen, e in scens.items():
            mw = e["mean_waiting_time"]
            print(
                f"{ctrl:14s} {scen:13s} wait {mw['mean']:7.1f} [{mw['ci_low']:.1f}, {mw['ci_high']:.1f}]  "
                f"vs fixed {mw.get('pct_vs_fixed', 0):+.1f}%"
            )


if __name__ == "__main__":
    main()
