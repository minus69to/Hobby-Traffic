"""Render README.md and docs/report.md from templates, filling every number from ``results/``.

Placeholders look like ``[[ppo.morning_peak.mean_waiting_time.pct]]`` and are resolved against
``results/eval_summary.json``, ``results/edge_benchmark.json``, ``results/training_summary.json``,
``results/speedtest.json`` and ``results/service_integration.json``.  ``scripts/check_numbers.py``
re-renders the templates and fails if the committed documents differ, which guarantees that no
number in the documents was typed by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from drs.config import REPO_ROOT

TEMPLATES = {
    "README.md": REPO_ROOT / "docs" / "templates" / "README.md.in",
    "docs/report.md": REPO_ROOT / "docs" / "templates" / "report.md.in",
}
PLACEHOLDER = re.compile(r"\[\[([a-zA-Z0-9_.:+-]+)\]\]")


def load(name: str) -> dict[str, Any]:
    p = REPO_ROOT / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


class Resolver:
    """Resolve dotted keys with formatting suffixes: ``a.b.c:fmt`` where fmt is e.g. ``.1f`` or ``pct``."""

    def __init__(self) -> None:
        self.ev = load("eval_summary.json")
        self.edge = load("edge_benchmark.json")
        self.train = load("training_summary.json")
        self.speed = load("speedtest.json")
        self.service = load("service_integration.json")
        self.missing: list[str] = []

    def value(self, key: str) -> Any:
        parts = key.split(".")
        root = parts[0]
        if root == "eval":  # eval.<controller>.<scenario>.<metric>.<field>
            ctrl, scen, metric, field_ = parts[1], parts[2], parts[3], parts[4] if len(parts) > 4 else "mean"
            return self.ev["summary"][ctrl][scen][metric][field_]
        if root == "n":  # eval meta
            return self.ev["meta"][parts[1]]
        if root == "edge":
            cur: Any = self.edge
            for p in parts[1:]:
                cur = cur[p]
            return cur
        if root == "train":
            runs = self.train.get("runs", [])
            if parts[1] == "n_seeds":
                return len(runs)
            if parts[1] == "total_wall_h":
                return sum(r["wall_clock_s"] for r in runs) / 3600.0
            if parts[1] == "steps_per_seed":
                return runs[0]["total_timesteps"] if runs else 0
            if parts[1] == "mean_steps_per_s":
                return sum(r["env_steps_per_s"] for r in runs) / len(runs) if runs else 0
            if parts[1] == "seeds":
                return ", ".join(str(r["seed"]) for r in runs)
            if parts[1] == "wall_min_per_seed":
                return sum(r["wall_clock_s"] for r in runs) / 60.0 / len(runs) if runs else 0
            raise KeyError(key)
        if root == "speed":
            return self.speed[parts[1]]
        if root == "service":
            return self.service[parts[1]]
        raise KeyError(key)

    def render(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            spec = m.group(1)
            key, _, fmt = spec.partition(":")
            try:
                v = self.value(key)
            except (KeyError, IndexError, TypeError):
                self.missing.append(spec)
                return "n/a"
            if fmt == "pct":
                return f"{v:+.1f}%"
            if fmt == "int":
                return f"{round(v):,}"
            if fmt == "kb":
                return f"{v / 1024:.1f} kB"
            if fmt == "pct100":
                return f"{v * 100:.2f}%"
            if fmt:
                return format(v, fmt)
            return str(v)

        return PLACEHOLDER.sub(repl, text)


def render_all(write: bool = True) -> dict[str, str]:
    r = Resolver()
    out: dict[str, str] = {}
    for target, tpl in TEMPLATES.items():
        if not tpl.exists():
            continue
        rendered = r.render(tpl.read_text(encoding="utf-8"))
        out[target] = rendered
        if write:
            (REPO_ROOT / target).write_text(rendered, encoding="utf-8")
    if r.missing:
        print(f"WARNING unresolved placeholders: {sorted(set(r.missing))}")
    return out


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safe Deep RL Traffic Signals for Dhaka-like Mixed Traffic: technical report</title>
<style>
body{{max-width:860px;margin:2rem auto;padding:0 1rem;font-family:system-ui,sans-serif;
line-height:1.55;color:#22182e;background:#fbf5ec}}
img{{max-width:100%}} table{{border-collapse:collapse;font-size:.9rem;width:100%}}
th,td{{border-bottom:1px solid #ccc;padding:.3rem .5rem;text-align:left}}
code{{background:#f0e8dc;padding:0 .2em;border-radius:3px}} a{{color:#b4570e}}
@media (prefers-color-scheme: dark){{body{{background:#130f1f;color:#f3ecff}}
code{{background:#2a2040}} a{{color:#ffb347}}}}
</style></head><body><p><a href="./">Back to the site</a></p>{body}</body></html>
"""


def render_html(markdown_text: str, out: Path) -> bool:
    """Convert the report to a standalone HTML page (figure links rewritten for the website)."""
    try:
        import markdown
    except ImportError:
        return False
    body = markdown.markdown(markdown_text.replace("](../figures/", "](figures/"), extensions=["tables", "fenced_code"])
    out.write_text(HTML_TEMPLATE.format(body=body), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if committed files differ")
    args = parser.parse_args()
    out = render_all(write=not args.check)
    if args.check:
        bad = [t for t, text in out.items() if (REPO_ROOT / t).read_text(encoding="utf-8") != text]
        if bad:
            raise SystemExit(f"documents differ from their rendered templates: {bad}")
        print("documents match the rendered templates")
    else:
        if "docs/report.md" in out and render_html(out["docs/report.md"], REPO_ROOT / "docs" / "report.html"):
            web_pub = REPO_ROOT / "web" / "public"
            (web_pub / "report.html").write_text(
                (REPO_ROOT / "docs" / "report.html").read_text(encoding="utf-8"), encoding="utf-8"
            )
            fig_dst = web_pub / "figures"
            fig_dst.mkdir(exist_ok=True)
            for f in (REPO_ROOT / "figures").glob("*.png"):
                (fig_dst / f.name).write_bytes(f.read_bytes())
        print(f"rendered {', '.join(out)}")


if __name__ == "__main__":
    main()
