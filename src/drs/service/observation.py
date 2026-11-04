"""Build the policy observation from a detector message, using the same layout as training."""

from __future__ import annotations

import numpy as np

from drs.config import SignalConfig
from drs.service.schemas import DetectorMessage
from drs.shield import SafetyShield
from drs.signal import PhaseTable
from drs.sumo_sim import VEHICLE_CLASSES, JunctionState, state_to_vector
from drs.topology import DIRS

MAX_JUNCTIONS = 4


def junction_state_from_message(msg: DetectorMessage, table: PhaseTable, lane_arm: dict[str, str]) -> JunctionState:
    lanes = table.incoming_lanes
    by_id = {lr.lane_id: lr for lr in msg.lanes}
    n = len(lanes)
    halting = np.zeros(n)
    max_wait = np.zeros(n)
    count = np.zeros(n)
    class_counts = np.zeros((len(DIRS), len(VEHICLE_CLASSES)))
    out_halting: dict[str, float] = {}
    for i, lane in enumerate(lanes):
        lr = by_id.get(lane)
        if lr is None:
            continue
        halting[i] = lr.halting
        max_wait[i] = lr.max_wait
        count[i] = lr.count
        arm_idx = DIRS.index(lane_arm[lane])
        for cls, v in lr.class_counts.items():
            if cls in VEHICLE_CLASSES:
                class_counts[arm_idx, VEHICLE_CLASSES.index(cls)] += v
        for lk in table.links:
            if lk.from_lane == lane and lk.turn != "c":
                out_halting[lk.to_lane] = lr.out_halting
    ped = np.zeros(4)
    for k, cr in enumerate(msg.crossings[:4]):
        ped[k] = cr.waiting
    neigh = np.array([msg.neighbour_pressure.get(d, 0.0) for d in DIRS])
    return JunctionState(
        junction=msg.junction_id,
        lanes=lanes,
        lane_halting=halting,
        lane_max_wait=max_wait,
        lane_count=count,
        class_counts=class_counts,
        out_lane_halting=out_halting,
        ped_waiting=ped,
        total_acc_wait=0.0,
        neighbour_pressure=neigh,
    )


def observation_vector(
    state: JunctionState, shield: SafetyShield, cfg: SignalConfig, junction_index: int = 0
) -> np.ndarray:
    onehot = np.zeros(cfg.n_phases)
    onehot[shield.target if shield.in_clearance else shield.phase] = 1.0
    jonehot = np.zeros(MAX_JUNCTIONS)
    jonehot[junction_index % MAX_JUNCTIONS] = 1.0
    return state_to_vector(state, onehot, shield.time_in_green, shield.in_clearance, jonehot, cfg.max_green)
