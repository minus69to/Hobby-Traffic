"""Export the SUMO network geometry to JSON so the 3D scene matches the simulation exactly.

Writes ``web/public/data/geometry_<network>.json`` with lanes (polylines, widths, pedestrian flag),
junction polygons, crossings, walking areas, stop lines per arm, signal-head link indexes and the
four-phase table.  Coordinates are SUMO network coordinates (metres, y up north).
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from drs.config import REPO_ROOT, SimConfig
from drs.signal import load_phase_tables
from drs.topology import DIRS, build_spec

OUT_DIR = REPO_ROOT / "web" / "public" / "data"


def parse_shape(s: str | None) -> list[list[float]]:
    if not s:
        return []
    return [[round(float(a), 2) for a in p.split(",")[:2]] for p in s.split()]


def export(cfg: SimConfig, network: str) -> Path:
    net_xml = REPO_ROOT / "sim" / "networks" / f"{network}.net.xml"
    root = ET.parse(net_xml).getroot()
    spec = build_spec(network, cfg.networks)
    arm_dirs = {j.id: {a.inbound_edge: d for d, a in j.arms.items()} for j in spec.junctions.values()}
    tables = load_phase_tables(net_xml, arm_dirs)
    loc = root.find("location")
    offset = [float(v) for v in str(loc.get("netOffset", "0,0")).split(",")] if loc is not None else [0.0, 0.0]
    lanes: list[dict[str, Any]] = []
    for edge in root.iter("edge"):
        func = edge.get("function", "normal")
        for lane in edge.iter("lane"):
            lanes.append(
                {
                    "id": lane.get("id"),
                    "edge": edge.get("id"),
                    "function": func,
                    "pedestrian": lane.get("allow") == "pedestrian",
                    "width": float(lane.get("width", 3.2)),
                    "length": float(lane.get("length", 0)),
                    "shape": parse_shape(lane.get("shape")),
                }
            )
    junctions = []
    for j in root.iter("junction"):
        if j.get("type") in ("internal",):
            continue
        junctions.append(
            {
                "id": j.get("id"),
                "type": j.get("type"),
                "x": float(j.get("x", 0)),
                "y": float(j.get("y", 0)),
                "shape": parse_shape(j.get("shape")),
            }
        )
    # stop lines + signal heads: end of each inbound vehicle lane
    lane_by_id = {ln["id"]: ln for ln in lanes}
    signals = []
    for jid, junction in spec.junctions.items():
        table = tables[jid]
        for d, arm in junction.arms.items():
            veh_lanes = [ln for ln in lanes if ln["edge"] == arm.inbound_edge and not ln["pedestrian"]]
            ends = [ln["shape"][-1] for ln in veh_lanes]
            prevs = [ln["shape"][-2] for ln in veh_lanes]
            if not ends:
                continue
            cx = sum(e[0] for e in ends) / len(ends)
            cy = sum(e[1] for e in ends) / len(ends)
            hx = cx - sum(p[0] for p in prevs) / len(prevs)
            hy = cy - sum(p[1] for p in prevs) / len(prevs)
            heading = math.degrees(math.atan2(hy, hx))
            links = {lk.turn: lk.index for lk in table.links if lk.arm == d and lk.turn != "c"}
            crossing = next((lk.index for lk in table.links if lk.arm == d and lk.turn == "c"), None)
            crossing_lane = lane_by_id.get(f"{table.crossings[crossing]}_0") if crossing is not None else None
            signals.append(
                {
                    "junction": jid,
                    "arm": d,
                    "stop_line": [[round(e[0], 2), round(e[1], 2)] for e in ends],
                    "centre": [round(cx, 2), round(cy, 2)],
                    "heading_deg": round(heading, 2),
                    "vehicle_links": links,
                    "crossing_link": crossing,
                    "crossing_shape": crossing_lane["shape"] if crossing_lane else [],
                    "inbound_lanes": [ln["id"] for ln in veh_lanes],
                }
            )
    phases = {
        jid: {
            "n_links": t.n_links,
            "phases": [{"index": p.index, "name": p.name, "state": p.state, "has_walk": p.has_walk} for p in t.phases],
            "links": [
                {"index": lk.index, "from": lk.from_lane, "to": lk.to_lane, "arm": lk.arm, "turn": lk.turn}
                for lk in t.links
            ],
            "incoming_lanes": t.incoming_lanes,
            "lane_phases": t.lane_phases(),
        }
        for jid, t in tables.items()
    }
    xs = [p[0] for ln in lanes for p in ln["shape"]]
    ys = [p[1] for ln in lanes for p in ln["shape"]]
    data = {
        "network": network,
        "net_offset": offset,
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "junction_centres": {jid: [j.x + offset[0], j.y + offset[1]] for jid, j in spec.junctions.items()},
        "arms": {
            jid: {
                d: {"inbound": a.inbound_edge, "outbound": a.outbound_edge, "external": a.external}
                for d, a in j.arms.items()
            }
            for jid, j in spec.junctions.items()
        },
        "directions": list(DIRS),
        "lanes": lanes,
        "junctions": junctions,
        "signals": signals,
        "phases": phases,
        "lefthand": cfg.sumo.lefthand,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"geometry_{network}.json"
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim.yaml")
    parser.add_argument("--networks", nargs="*", default=None)
    args = parser.parse_args()
    cfg = SimConfig.load(args.config)
    for net in args.networks or list(cfg.networks):
        p = export(cfg, net)
        print(f"wrote {p.relative_to(REPO_ROOT)} ({p.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
