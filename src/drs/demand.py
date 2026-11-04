"""Generate seeded SUMO route files for the Dhaka-like demand scenarios.

Run ``python -m drs.demand --config configs/sim.yaml``.  Flows are written per vehicle class and
per route (entry edge to exit edge, routes enumerated from the turning ratios), plus pedestrian
flows on every crossing.  All numbers are synthetic assumptions, see ``configs/sim.yaml``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drs.config import REPO_ROOT, SimConfig
from drs.topology import OPPOSITE, TURN_NAMES, NetworkSpec, build_spec

DEMAND_DIR = REPO_ROOT / "sim" / "demand"
MAX_ROUTE_JUNCTIONS = 4
MIN_ROUTE_PROB = 0.004


@dataclass(frozen=True)
class RouteFlow:
    """Expected hourly flow of all classes along one route (list of edge ids)."""

    edges: tuple[str, ...]
    veh_per_hour: float


def enumerate_routes(spec: NetworkSpec, arms_vph: dict[str, float], turns: list[float]) -> list[RouteFlow]:
    """Enumerate entry-to-exit routes following the turning ratios at every junction.

    `arms_vph` gives the inbound volume for the *external* arms by direction of the arm relative to
    the network (for the grid the entry arm direction of its junction).
    """
    routes: list[RouteFlow] = []
    turn_prob = dict(zip(TURN_NAMES, turns))
    for entry_edge, jid, arm_dir in spec.entry_edges:
        volume = float(arms_vph[arm_dir])
        stack: list[tuple[list[str], str, str, float, int]] = [([entry_edge], jid, arm_dir, 1.0, 0)]
        while stack:
            edges, cur_j, from_dir, prob, depth = stack.pop()
            junction = spec.junctions[cur_j]
            for turn in TURN_NAMES:
                p = prob * turn_prob[turn]
                if p < MIN_ROUTE_PROB:
                    continue
                target_dir = junction.turn_target(from_dir, turn)
                arm = junction.arms[target_dir]
                new_edges = [*edges, arm.outbound_edge]
                if arm.external or depth + 1 >= MAX_ROUTE_JUNCTIONS:
                    if arm.external:
                        routes.append(RouteFlow(tuple(new_edges), volume * p))
                    continue
                # continue into the neighbouring junction, arriving from the opposite side
                stack.append((new_edges, arm.neighbour, OPPOSITE[target_dir], p, depth + 1))
    return routes


def edge_volumes(routes: list[RouteFlow]) -> dict[str, float]:
    """Expected vehicles per hour on every edge, summed over routes."""
    vols: dict[str, float] = {}
    for r in routes:
        for e in r.edges:
            vols[e] = vols.get(e, 0.0) + r.veh_per_hour
    return vols


def _class_shares(cfg: SimConfig) -> dict[str, float]:
    """Vehicle type id -> share, splitting buses into standard and double-decker."""
    mix = cfg.vehicle_mix
    dd = float(cfg.raw.get("double_decker_share", 0.0))
    shares: dict[str, float] = {}
    for cls, share in mix.items():
        if cls == "bus":
            shares["bus"] = share * (1.0 - dd)
            shares["bus_dd"] = share * dd
        else:
            shares[cls] = share
    return shares


def write_routes(cfg: SimConfig, spec: NetworkSpec, scenario: str, seed: int, out_path: Path) -> dict[str, Any]:
    """Write a `.rou.xml` for (network, scenario, seed) and return a manifest with expected volumes."""
    scen = cfg.scenario(scenario)
    arms = {k: float(v) for k, v in scen["arms"].items()}
    turns = [float(t) for t in scen["turns"]]
    episode = cfg.sumo.episode_seconds
    routes = enumerate_routes(spec, arms, turns)
    shares = _class_shares(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    lines.append(f"  <!-- scenario={scenario} seed={seed} synthetic Dhaka-inspired demand -->")
    for i, r in enumerate(routes):
        lines.append(f'  <route id="r{i}" edges="{" ".join(r.edges)}"/>')
    fid = 0
    for i, r in enumerate(routes):
        for vtype, share in shares.items():
            vph = r.veh_per_hour * share
            if vph < 0.5:
                continue
            lines.append(
                f'  <flow id="f{fid}" type="{vtype}" route="r{i}" begin="0" end="{episode}" '
                f'vehsPerHour="{vph:.3f}" departLane="best" departSpeed="max" departPos="base"/>'
            )
            fid += 1
    # pedestrians: each crossing is used by people walking between the two adjacent external arms
    pph = float(scen["pedestrians_per_crossing"])
    pid = 0
    for junction in spec.junctions.values():
        dirs = list(junction.arms)
        for k, d in enumerate(dirs):
            nxt = dirs[(k + 1) % len(dirs)]
            a, b = junction.arms[d], junction.arms[nxt]
            for frm, to in ((a.inbound_edge, b.outbound_edge), (b.inbound_edge, a.outbound_edge)):
                lines.append(
                    f'  <personFlow id="p{pid}" type="pedestrian" begin="0" end="{episode}" '
                    f'personsPerHour="{pph / 2:.2f}">'
                )
                lines.append(f'    <walk from="{frm}" to="{to}"/>')
                lines.append("  </personFlow>")
                pid += 1
    # incident (listed last: SUMO loads route files incrementally and drops elements whose departure
    # precedes an earlier element, so the depart=1500 vehicle must follow the begin=0 flows)
    # one vehicle breaks down and blocks a lane for `duration` seconds
    if "incident" in scen:
        inc = scen["incident"]
        jid = next(iter(spec.junctions))
        arm = spec.junctions[jid].arms[str(inc["edge_suffix"])]
        exit_edge = spec.junctions[jid].arms[spec.junctions[jid].turn_target(arm.direction, "s")].outbound_edge
        lines.append(
            f'  <vehicle id="incident_truck" type="truck" depart="{int(inc["start"])}" '
            f'departLane="{int(inc["lane"])}" color="1,0,1">'
        )
        lines.append(f'    <route edges="{arm.inbound_edge} {exit_edge}"/>')
        lines.append(
            f'    <stop lane="{arm.inbound_edge}_{int(inc["lane"])}" endPos="{float(inc["end_pos"]):.1f}" '
            f'duration="{int(inc["duration"])}" parking="false"/>'
        )
        lines.append("  </vehicle>")
    lines.append("</routes>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "scenario": scenario,
        "seed": seed,
        "network": spec.name,
        "edge_vph": edge_volumes(routes),
        "arms_vph": arms,
        "turns": turns,
        "pedestrians_per_crossing": pph,
        "incident": scen.get("incident"),
        "sensor_noise": scen.get("sensor_noise"),
        "n_routes": len(routes),
        "n_flows": fid,
    }
    out_path.with_suffix(".json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def route_path(network: str, scenario: str, seed: int) -> Path:
    return DEMAND_DIR / network / f"{scenario}_s{seed}.rou.xml"


def ensure_routes(cfg: SimConfig, network: str, scenario: str, seed: int) -> Path:
    """Return the route file for (network, scenario, seed), generating it if missing."""
    p = route_path(network, scenario, seed)
    if not p.exists():
        spec = build_spec(network, cfg.networks)
        write_routes(cfg, spec, scenario, seed, p)
    return p


def generate_all(cfg: SimConfig) -> int:
    """Generate route files for every network, scenario and seed listed in the config."""
    n = 0
    seeds = cfg.seeds
    all_seeds = list(seeds["train"]) + list(seeds["validation"]) + list(seeds["eval"]) + [int(seeds["replay"])]
    for network in cfg.networks:
        spec = build_spec(network, cfg.networks)
        for scenario in cfg.scenarios:
            for seed in all_seeds:
                write_routes(cfg, spec, scenario, int(seed), route_path(network, scenario, int(seed)))
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim.yaml")
    args = parser.parse_args()
    cfg = SimConfig.load(args.config)
    print(f"wrote {generate_all(cfg)} route files under {DEMAND_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
