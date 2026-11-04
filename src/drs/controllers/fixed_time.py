"""Webster-style fixed-time controller.

Cycle length (Webster 1958): ``C = (1.5 L + 5) / (1 - Y)`` where ``L`` is the total lost time per
cycle (``n_phases * (yellow + all_red)``) and ``Y`` the sum over phases of the critical flow ratio
``y_p = max_lane_flow_p / saturation_flow``.  Flows are the expected volumes from the demand
manifest (vehicles per hour per inbound edge converted to passenger-car units with the per-class
``pcu`` factors in ``configs/sim.yaml``) spread over the lanes that have green in the phase.
Greens are split proportionally to ``y_p``, clamped to ``[min_green, max_green]`` and rounded up to
the decision interval.  The cycle is clamped to [60, 180] s.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from drs.config import SimConfig
from drs.shield import SafetyShield
from drs.signal import PhaseTable
from drs.sumo_sim import JunctionState

SATURATION_PCU_PER_HOUR_PER_LANE = 1800.0


def average_pcu(cfg: SimConfig) -> float:
    mix = cfg.vehicle_mix
    return sum(float(cfg.vtypes[c]["pcu"]) * share for c, share in mix.items())


def webster_plan(
    cfg: SimConfig, table: PhaseTable, edge_vph: dict[str, float], turns: list[float]
) -> list[int]:
    """Green time (s) per phase for one junction."""
    sig = cfg.signal
    pcu = average_pcu(cfg)
    turn_share = dict(zip(("l", "s", "r"), turns))
    ys = []
    for p in table.phases:
        links = [lk for lk in table.phase_links(p.index) if lk.turn != "c"]
        lane_flow: dict[str, float] = {}
        for lk in links:
            edge = lk.from_edge
            n_lanes_for_turn = max(1, sum(1 for o in links if o.from_edge == edge and o.turn == lk.turn))
            flow = edge_vph.get(edge, 0.0) * pcu * turn_share[lk.turn] / n_lanes_for_turn
            lane_flow[lk.from_lane] = lane_flow.get(lk.from_lane, 0.0) + flow
        ys.append(max(lane_flow.values(), default=0.0) / SATURATION_PCU_PER_HOUR_PER_LANE)
    y_total = min(sum(ys), 0.9)
    lost = len(table.phases) * (sig.yellow + sig.all_red)
    cycle = (1.5 * lost + 5.0) / max(1.0 - y_total, 0.1)
    cycle = float(np.clip(cycle, 60.0, 180.0))
    effective = cycle - lost
    plan = []
    for y in ys:
        g = effective * (y / y_total) if y_total > 0 else effective / len(ys)
        g = float(np.clip(g, sig.min_green, sig.max_green))
        step = cfg.sumo.decision_interval
        plan.append(int(math.ceil(g / step) * step))
    return plan


class FixedTimeController:
    name = "fixed_time"

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.plans: dict[str, list[int]] = {}

    def reset(self, env: Any) -> None:
        from drs.demand import route_path

        manifest_path = route_path(env.network, env.scenario, env.seed).with_suffix(".json")
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.plans = {
            jid: webster_plan(self.cfg, env.sim.tables[jid], manifest["edge_vph"], manifest["turns"])
            for jid in env.junction_ids
        }

    def act(
        self, states: dict[str, JunctionState], obs: np.ndarray, masks: np.ndarray, shields: dict[str, SafetyShield]
    ) -> list[int]:
        out = []
        for jid, sh in shields.items():
            if sh.in_clearance:
                out.append(sh.target)
                continue
            if sh.time_in_green >= self.plans[jid][sh.phase]:
                out.append((sh.phase + 1) % sh.n_phases)
            else:
                out.append(sh.phase)
        return out
