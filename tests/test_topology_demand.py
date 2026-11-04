"""Topology conventions, route enumeration and the Webster plan."""

from __future__ import annotations

from drs.config import SimConfig
from drs.controllers.fixed_time import webster_plan
from drs.demand import edge_volumes, enumerate_routes
from drs.signal import PhaseTable
from drs.topology import build_spec


def test_single_turn_targets_lefthand() -> None:
    cfg = SimConfig.load()
    spec = build_spec("single", cfg.networks)
    j = spec.junctions["C"]
    # arriving from the north the vehicle heads south: left turn -> east, right turn -> west
    assert j.turn_target("N", "s") == "S"
    assert j.turn_target("N", "l") == "E"
    assert j.turn_target("N", "r") == "W"
    assert j.arms["N"].inbound_edge == "N_C" and j.arms["N"].outbound_edge == "C_N"


def test_routes_conserve_volume() -> None:
    cfg = SimConfig.load()
    for network in ("single", "grid2x2"):
        spec = build_spec(network, cfg.networks)
        arms = {"N": 1000.0, "S": 800.0, "E": 600.0, "W": 400.0}
        routes = enumerate_routes(spec, arms, [0.2, 0.6, 0.2])
        total_in = sum(arms[d] for _, _, d in spec.entry_edges)
        assert abs(sum(r.veh_per_hour for r in routes) - total_in) / total_in < 0.05
        vols = edge_volumes(routes)
        for edge, _, _ in spec.entry_edges:
            assert edge in vols
        assert all(len(r.edges) >= 2 for r in routes)


def test_webster_plan_respects_bounds(phase_table: PhaseTable) -> None:
    cfg = SimConfig.load()
    plan = webster_plan(cfg, phase_table, {"N_C": 1250, "S_C": 900, "E_C": 1100, "W_C": 850}, [0.2, 0.6, 0.2])
    assert len(plan) == 4
    assert all(cfg.signal.min_green <= g <= cfg.signal.max_green for g in plan)
    assert plan[0] >= plan[1] and plan[2] >= plan[3]  # through phases carry more flow than right turns
    assert all(g % cfg.sumo.decision_interval == 0 for g in plan)
