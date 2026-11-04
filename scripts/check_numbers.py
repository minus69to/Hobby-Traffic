"""Verify that every number shown in README, report and the built website comes from ``results/``.

1. README.md and docs/report.md must equal the rendering of their templates (all numbers injected
   from results files, see scripts/render_docs.py).
2. The built website bundle (web/dist) must embed the current ``results/eval_summary.json`` and
   ``results/edge_benchmark.json`` values (a handful of exact numbers are searched for).
3. The replay manifests shipped with the site must be internally consistent with their binaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from drs.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from render_docs import render_all  # noqa: E402


def check_docs() -> list[str]:
    problems = []
    for target, text in render_all(write=False).items():
        if (REPO_ROOT / target).read_text(encoding="utf-8") != text:
            problems.append(f"{target} differs from its rendered template (run `make docs`)")
    return problems


def check_web_bundle() -> list[str]:
    dist = REPO_ROOT / "web" / "dist" / "assets"
    if not dist.exists():
        return ["web/dist not built (run `make web`)"]
    bundle = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in dist.glob("*.js"))
    problems = []
    ev = json.loads((REPO_ROOT / "results" / "eval_summary.json").read_text(encoding="utf-8"))
    probes = []
    for ctrl, scens in ev["summary"].items():
        for scen, entry in scens.items():
            probes.append((f"{ctrl}/{scen}/mean_waiting_time", json.dumps(entry["mean_waiting_time"]["mean"])))
    edge_path = REPO_ROOT / "results" / "edge_benchmark.json"
    if edge_path.exists():
        edge = json.loads(edge_path.read_text(encoding="utf-8"))
        if edge.get("int8", {}).get("n_calls"):
            probes.append(("edge/int8/latency_p50_ms", json.dumps(edge["int8"]["latency_p50_ms"])))
            probes.append(("edge/int8/size_bytes", json.dumps(edge["int8"]["size_bytes"])))
    for name, literal in probes:
        if literal not in bundle:
            problems.append(f"website bundle does not contain {name}={literal} (rebuild with `make web`)")
    return problems


def check_replays() -> list[str]:
    from drs.replay import read_replay

    rdir = REPO_ROOT / "web" / "public" / "data" / "replays"
    problems = []
    for manifest in rdir.glob("*.json"):
        if manifest.name == "index.json":
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        rep = read_replay(rdir / m["file"])
        if rep["header"]["n_frames"] != m["n_frames"] or len(rep["tracks"]) != m["n_tracks"]:
            problems.append(f"{manifest.name}: manifest frames/tracks do not match the binary")
        if rep["header"]["scenario"] != m["scenario"] or rep["header"]["controller"] != m["controller"]:
            problems.append(f"{manifest.name}: scenario/controller mismatch with the binary header")
    return problems


def main() -> None:
    problems = check_docs() + check_web_bundle() + check_replays()
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        raise SystemExit(1)
    print("all numbers in README, report and the website bundle match results/")


if __name__ == "__main__":
    main()
