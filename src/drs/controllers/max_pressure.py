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
    """Pressure per phase: sum over its green vehicle movements of (queue in - queue out).

    Shared lanes (through + right turn) count for every phase that serves one of their movements.
    """
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
        self,
        states: dict[str, JunctionState],
        obs: np.ndarray,
        masks: np.ndarray,
        shields: dict[str, SafetyShield],
    ) -> list[int]:
        """Textbook max-pressure: request the admissible phase with the largest pressure.

        Known weakness in this lefthand layout: the protected right turn shares a lane with the
        through movement, so the through phases always dominate and the shared lane discharges only
        when the shield's max-green forces a rotation; this costs capacity at low demand. The
        behaviour is kept deliberately (it is the canonical algorithm) and reported as is.
        """
        out = []
        for k, (jid, sh) in enumerate(shields.items()):
            table = self.tables[jid]
            if sh.in_clearance:
                out.append(sh.target)
                continue
            pressures = phase_pressures(table, states[jid])
            if pressures.max() <= 0.0:
                # nothing queued anywhere: rest on a through phase instead of cycling empty phases
                through = [p.index for p in table.phases if p.has_walk] or [0]
                target = sh.phase if sh.phase in through else through[0]
                out.append(target if masks[k][target] else int(np.argmax(masks[k])))
                continue
            pressures[sh.phase] += 1.0  # small hysteresis
            pressures[~masks[k]] = -np.inf
            out.append(int(np.argmax(pressures)))
        return out
