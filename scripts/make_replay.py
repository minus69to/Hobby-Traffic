"""Export synchronised replays (positions, headings, speeds, classes, signal states) for the web app.

For every (scenario, controller) pair in ``replay`` of the evaluation config, one episode with the
replay seed is simulated at a 0.5 s step, sampled every 0.5 s, quantised to 5 cm, delta-encoded and
written gzip-compressed to ``web/public/data/replays/<network>_<scenario>_<controller>.drsr.gz``
next to a JSON manifest with scenario metadata and per-second aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.env import TrafficSignalEnv
from drs.evaluate import build_controller
from drs.metrics import episode_metrics
from drs.replay import FrameRecorder, write_replay
from drs.runner import run_episode

OUT_DIR = REPO_ROOT / "web" / "public" / "data" / "replays"


def make_one(
    ecfg: dict[str, Any], network: str, scenario: str, ctrl_name: str, seed: int, dt: float, out_dir: Path
) -> dict[str, Any]:
    cfg = SimConfig.load(ecfg["sim_config"])
    rec_holder: dict[str, FrameRecorder] = {}

    def hook(sim: Any, states: dict[str, str]) -> None:
        if "r" not in rec_holder:
            rec_holder["r"] = FrameRecorder(dt, sim.junction_ids)
        rec_holder["r"].on_step(sim, states)

    env = TrafficSignalEnv(
        cfg,
        network,
        scenario,
        [seed],
        step_length=dt,
        record=True,
        step_hook=hook,
        rng_seed=seed,
        tripinfo_dir=out_dir / "tmp",
    )
    ctrl = build_controller(cfg, ecfg["controllers"][ctrl_name], ctrl_name)
    t0 = time.perf_counter()
    ep = run_episode(env, ctrl, scenario=scenario, seed=seed)
    metrics = episode_metrics(ep, cfg.sumo.warmup_seconds, cfg.sumo.episode_seconds)
    if ep.tripinfo is not None and ep.tripinfo.exists():
        ep.tripinfo.unlink()
    scen = cfg.scenario(scenario)
    meta = {
        "network": network,
        "scenario": scenario,
        "controller": ctrl_name,
        "seed": seed,
        "episode_seconds": cfg.sumo.episode_seconds,
        "warmup_seconds": cfg.sumo.warmup_seconds,
        "step_length": dt,
    }
    name = f"{network}_{scenario}_{ctrl_name}"
    size = write_replay(rec_holder["r"], meta, out_dir / f"{name}.drsr.gz")
    manifest = {
        **meta,
        "file": f"{name}.drsr.gz",
        "size_bytes": size,
        "n_frames": rec_holder["r"].n_frames,
        "n_tracks": len(rec_holder["r"].tracks),
        "incident": scen.get("incident"),
        "arms_vph": scen.get("arms"),
        "metrics": {k: round(v, 3) for k, v in metrics.items()},
        "shield_stats": ep.shield_stats,
        "interventions": ep.interventions,
        "per_second": ep.per_second,
        "wall_s": round(time.perf_counter() - t0, 1),
        "note": "Synthetic Dhaka-inspired demand; numbers come from this replay run, not from real data.",
    }
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--network", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--controllers", nargs="*", default=None)
    args = parser.parse_args()
    ecfg = load_yaml(args.config)
    rp = ecfg["replay"]
    network = args.network or str(ecfg["network"])
    index_path = OUT_DIR / "index.json"
    index: dict[str, Any] = (
        json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"replays": []}
    )
    for scenario in args.scenarios or rp["scenarios"]:
        for ctrl in args.controllers or rp["controllers"]:
            m = make_one(ecfg, network, scenario, ctrl, int(rp["seed"]), float(rp.get("sample_interval", 0.5)), OUT_DIR)
            index["replays"] = [r for r in index["replays"] if r["file"] != m["file"]] + [
                {
                    k: m[k]
                    for k in ("network", "scenario", "controller", "seed", "file", "size_bytes", "n_frames", "n_tracks")
                }
            ]
            print(
                f"{m['file']}: {m['size_bytes'] / 1e6:.2f} MB, {m['n_frames']} frames, "
                f"{m['n_tracks']} tracks, {m['wall_s']} s"
            )
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    (OUT_DIR / "tmp").rmdir() if (OUT_DIR / "tmp").exists() and not any((OUT_DIR / "tmp").iterdir()) else None


if __name__ == "__main__":
    main()
