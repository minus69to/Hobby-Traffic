"""Collect the PPO-from-scratch vs BC+PPO learning curves into results/imitation.json and a figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drs.config import REPO_ROOT

RUNS = {
    "ppo_scratch": REPO_ROOT / "models" / "single" / "seed13",
    "bc_ppo": REPO_ROOT / "models" / "single_bc" / "seed13",
}


def main() -> None:
    out: dict[str, object] = {
        "runs": {},
        "note": (
            "Validation return on a held-out morning-peak seed vs environment steps and wall-clock; "
            "same seed, budget and hyper-parameters."
        ),
    }
    runs: dict[str, object] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, d in RUNS.items():
        meta_path = d / "train_meta.json"
        progress = d / "progress.csv"
        entry: dict[str, Any]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entry = {
                "eval_rows": [
                    {"timesteps": r["timesteps"], "wall_s": r["wall_s"], "return": r["return"]}
                    for r in meta["eval_rows"]
                ],
                "wall_clock_s": meta["wall_clock_s"],
                "best_validation_return": meta["best_validation_return"],
                "completed": True,
            }
        elif progress.exists():  # run interrupted: use the evaluation rows logged so far
            with progress.open(encoding="utf-8") as fh:
                rows = [{k: float(v) for k, v in r.items() if v not in ("", None)} for r in csv.DictReader(fh)]
            entry = {
                "eval_rows": [
                    {"timesteps": int(r["timesteps"]), "wall_s": r["wall_s"], "return": r["return"]} for r in rows
                ],
                "wall_clock_s": rows[-1]["wall_s"] if rows else 0.0,
                "best_validation_return": max(r["return"] for r in rows) if rows else None,
                "completed": False,
                "note": f"interrupted after {int(rows[-1]['timesteps']) if rows else 0} steps (VM restart)",
            }
        else:
            continue
        bc_meta = d.parent / "bc_meta.json"
        if bc_meta.exists():
            bm = json.loads(bc_meta.read_text(encoding="utf-8"))
            entry["bc_val_acc"] = bm["history"][-1]["val_acc"]
            entry["bc_samples"] = bm["n_samples"]
            entry["bc_wall_s"] = bm["wall_s"]
        runs[name] = entry
        rows = entry["eval_rows"]
        axes[0].plot([r["timesteps"] for r in rows], [r["return"] for r in rows], marker="o", ms=3, label=name)
        axes[1].plot([r["wall_s"] / 60 for r in rows], [r["return"] for r in rows], marker="o", ms=3, label=name)
    axes[0].set_xlabel("environment steps")
    axes[1].set_xlabel("wall-clock (min)")
    for ax in axes:
        ax.set_ylabel("validation return")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("PPO from scratch vs behaviour-cloning warm start (seed 13)")
    fig.tight_layout()
    Path(REPO_ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(REPO_ROOT / "figures" / "imitation_curves.png", dpi=130)
    out["runs"] = runs
    (REPO_ROOT / "results" / "imitation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"imitation.json with runs: {list(runs)}")


if __name__ == "__main__":
    main()
