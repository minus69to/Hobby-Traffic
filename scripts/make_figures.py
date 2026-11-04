"""Generate the figures for README, report and website from ``results/`` (never from hard-coded numbers).

Outputs in ``figures/``: ``wait_by_scenario.png`` (bars with 95 % CIs), ``fairness_by_class.png``,
``wait_over_time.png`` (from the per-second time series of the evaluation episodes),
``training_curves.png`` (validation return vs environment steps per seed) and ``edge_latency.png``.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from drs.config import REPO_ROOT  # noqa: E402

COLOURS = {"fixed_time": "#f59e0b", "max_pressure": "#60a5fa", "ppo": "#34d399"}
LABELS = {"fixed_time": "Fixed-time (Webster)", "max_pressure": "Max-pressure", "ppo": "Safe PPO (pooled seeds)"}
SCEN_LABELS = {
    "offpeak": "Off-peak",
    "morning_peak": "Morning peak",
    "evening_peak": "Evening peak",
    "incident": "Incident",
    "surge": "Surge",
    "noisy": "Noisy sensors",
}
CLASSES = ["car", "motorcycle", "cng", "rickshaw", "bus", "truck"]


def _style() -> None:
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def bars(summary: dict[str, Any], scenarios: list[str], metric: str, ylabel: str, out: Path, title: str) -> None:
    ctrls = [c for c in ("fixed_time", "max_pressure", "ppo") if c in summary]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / max(len(ctrls), 1)
    for i, c in enumerate(ctrls):
        xs, ys, lo, hi = [], [], [], []
        for k, s in enumerate(scenarios):
            e = summary[c].get(s)
            if not e:
                continue
            st = e[metric]
            xs.append(k + (i - (len(ctrls) - 1) / 2) * w)
            ys.append(st["mean"])
            lo.append(st["mean"] - st["ci_low"])
            hi.append(st["ci_high"] - st["mean"])
        ax.bar(xs, ys, width=w, color=COLOURS[c], label=LABELS[c], yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCEN_LABELS.get(s, s) for s in scenarios])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fairness(summary: dict[str, Any], scenario: str, out: Path) -> None:
    ctrls = [c for c in ("fixed_time", "max_pressure", "ppo") if c in summary and scenario in summary[c]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / max(len(ctrls), 1)
    for i, c in enumerate(ctrls):
        e = summary[c][scenario]
        ys = [e[f"wait_{cls}"]["mean"] for cls in CLASSES]
        lo = [e[f"wait_{cls}"]["mean"] - e[f"wait_{cls}"]["ci_low"] for cls in CLASSES]
        hi = [e[f"wait_{cls}"]["ci_high"] - e[f"wait_{cls}"]["mean"] for cls in CLASSES]
        ax.bar([k + (i - (len(ctrls) - 1) / 2) * w for k in range(len(CLASSES))], ys, width=w, color=COLOURS[c], label=LABELS[c], yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("mean waiting time (s)")
    ax.set_title(f"Waiting time per vehicle class, {SCEN_LABELS.get(scenario, scenario)}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def wait_over_time(raw_dir: Path, scenario: str, out: Path, warmup: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for c in ("fixed_time", "max_pressure", "ppo_s13", "ppo_s42", "ppo_s2026"):
        p = raw_dir / f"timeseries_{c}_{scenario}.json.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            d = json.load(fh)
        rows = next(iter(d["series"].values()))
        ts = [r["t"] for r in rows]
        ys = [r["vehicles_waiting"] for r in rows]
        base = c.split("_s")[0] if c.startswith("ppo") else c
        ax.plot(ts, ys, color=COLOURS.get(base, "#888"), lw=0.9, alpha=0.9 if base != "ppo" else 0.6, label=LABELS.get(base, c) if c in ("fixed_time", "max_pressure", "ppo_s13") else None)
    ax.axvspan(0, warmup, color="grey", alpha=0.12, label="warm-up (excluded)")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("vehicles waiting")
    ax.set_title(f"Vehicles waiting over time, {SCEN_LABELS.get(scenario, scenario)} (evaluation seed 1)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def training_curves(models_dir: Path, out: Path) -> list[dict[str, Any]]:
    runs = []
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for meta_path in sorted(models_dir.glob("seed*/train_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = meta.get("eval_rows", [])
        if rows:
            ax.plot([r["timesteps"] for r in rows], [r["return"] for r in rows], marker="o", ms=3, lw=1, label=f"seed {meta['seed']}")
        runs.append(
            {
                "seed": meta["seed"],
                "network": meta["network"],
                "total_timesteps": meta["total_timesteps"],
                "wall_clock_s": meta["wall_clock_s"],
                "env_steps_per_s": meta["env_steps_per_s"],
                "best_validation_return": meta["best_validation_return"],
            }
        )
    ax.set_xlabel("environment steps")
    ax.set_ylabel("validation return (morning peak)")
    ax.set_title("MaskablePPO training, single junction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return runs


def edge_latency(bench: dict[str, Any], out: Path) -> None:
    if not bench.get("int8", {}).get("n_calls"):
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    names = ["float32", "int8"]
    p50 = [bench["float"]["latency_p50_ms"], bench["int8"]["latency_p50_ms"]]
    p95 = [bench["float"]["latency_p95_ms"], bench["int8"]["latency_p95_ms"]]
    ax.bar([0, 1], p50, width=0.35, label="p50", color="#60a5fa")
    ax.bar([0.4, 1.4], p95, width=0.35, label="p95", color="#f59e0b")
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels([f"{n}\n{bench[k]['size_bytes'] / 1024:.0f} kB" for n, k in zip(names, ("float", "int8"))])
    ax.set_ylabel("latency (ms), 1 thread")
    ax.set_title("ONNX Runtime inference latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--models", default="models/single")
    args = parser.parse_args()
    _style()
    res = REPO_ROOT / args.results
    fig_dir = REPO_ROOT / args.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = res / "eval_summary.json"
    if summary_path.exists():
        ev = json.loads(summary_path.read_text(encoding="utf-8"))
        scenarios = ev["meta"]["scenarios"]
        bars(ev["summary"], scenarios, "mean_waiting_time", "mean waiting time (s)", fig_dir / "wait_by_scenario.png", "Mean vehicle waiting time (95 % bootstrap CI)")
        bars(ev["summary"], scenarios, "ped_mean_wait", "pedestrian mean wait (s)", fig_dir / "ped_wait_by_scenario.png", "Pedestrian waiting time (95 % bootstrap CI)")
        bars(ev["summary"], scenarios, "throughput", "vehicles completed per episode", fig_dir / "throughput_by_scenario.png", "Throughput (95 % bootstrap CI)")
        scen = "morning_peak" if "morning_peak" in scenarios else scenarios[0]
        fairness(ev["summary"], scen, fig_dir / "fairness_by_class.png")
        wait_over_time(res / "raw", scen, fig_dir / "wait_over_time.png", int(ev["meta"]["warmup_seconds"]))
        if "incident" in scenarios:
            wait_over_time(res / "raw", "incident", fig_dir / "wait_over_time_incident.png", int(ev["meta"]["warmup_seconds"]))
        print("wrote evaluation figures")
    runs = training_curves(REPO_ROOT / args.models, fig_dir / "training_curves.png")
    (res / "training_summary.json").write_text(
        json.dumps({"runs": runs, "note": "Wall-clock on 4 CPU cores in the development VM (see ENVIRONMENT.md)."}, indent=1), encoding="utf-8"
    )
    bench_path = res / "edge_benchmark.json"
    if bench_path.exists():
        edge_latency(json.loads(bench_path.read_text(encoding="utf-8")), fig_dir / "edge_latency.png")
    print(f"figures in {fig_dir.relative_to(REPO_ROOT)}, training summary with {len(runs)} runs")


if __name__ == "__main__":
    main()
