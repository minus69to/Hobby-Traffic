"""Max-pressure controller (Varaiya 2013 style) with the same mask/shield constraints as PPO.

At every decision the phase with the largest pressure, i.e. the sum over its green movements of
(queue on the incoming lane - queue on the outgoing lane), is requested, restricted to the actions
the safety shield currently allows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drs.shield import SafetyShield
from drs.signal import PhaseTable
from drs.sumo_sim import JunctionState


def phase_pressures(table: PhaseTable, s: JunctionState) -> np.ndarray:
    lane_q = dict(zip(s.lanes, s.lane_halting))
    out = np.zeros(table.n_phases)
    for p in table.phases:
        for lk in table.phase_links(p.index):
            if lk.turn == "c":
                continue
            out[p.index] += lane_q.get(lk.from_lane, 0.0) - s.out_lane_halting.get(lk.to_lane, 0.0)
    return out


class MaxPressureController:
    name = "max_pressure"

    def __init__(self) -> None:
        self.tables: dict[str, PhaseTable] = {}

    def reset(self, env: Any) -> None:
        self.tables = dict(env.sim.tables)

    def act(
        self, states: dict[str, JunctionState], obs: np.ndarray, masks: np.ndarray, shields: dict[str, SafetyShield]
    ) -> list[int]:
        out = []
        for k, (jid, sh) in enumerate(shields.items()):
            pressures = phase_pressures(self.tables[jid], states[jid])
            # small hysteresis: keep the current phase unless another is clearly better
            pressures[sh.phase] += 1.0
            pressures[~masks[k]] = -np.inf
            out.append(int(np.argmax(pressures)))
        return out
