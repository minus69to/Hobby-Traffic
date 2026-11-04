"""Thin wrapper around libsumo/TraCI: starts SUMO, applies signal states, reads detector-like data.

Set ``LIBSUMO_AS_TRACI=1`` (the Makefile does) to use the in-process libsumo binding, which is
several times faster than socket TraCI.  Only one simulation per process is possible with libsumo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drs.config import REPO_ROOT, SimConfig, sumo_binary
from drs.signal import PhaseTable, load_phase_tables
from drs.topology import DIRS, NetworkSpec, build_spec

VEHICLE_CLASSES: tuple[str, ...] = ("car", "bus", "truck", "motorcycle", "cng", "rickshaw")
TYPE_TO_CLASS: dict[str, str] = {c: c for c in VEHICLE_CLASSES} | {"bus_dd": "bus"}
LANE_CAPACITY_VEH = 40.0  # normalisation constant: ~300 m / 7.5 m per queued vehicle
MAX_WAIT_NORM = 300.0  # s
CLASS_COUNT_NORM = 20.0
PED_NORM = 10.0
PRESSURE_NORM = 50.0


def traci_module() -> Any:
    """Return libsumo when ``LIBSUMO_AS_TRACI`` is set, else the socket-based traci module."""
    if os.environ.get("LIBSUMO_AS_TRACI", "1") not in ("0", "false", "False"):
        import libsumo

        return libsumo
    import traci

    return traci


@dataclass
class JunctionState:
    """Detector-like snapshot of one junction, the input to every controller."""

    junction: str
    lanes: list[str]
    lane_halting: np.ndarray  # vehicles with speed < 0.1 m/s per incoming lane
    lane_max_wait: np.ndarray  # s, longest consecutive wait on the lane
    lane_count: np.ndarray  # vehicles per incoming lane
    class_counts: np.ndarray  # [4 arms, n classes]
    out_lane_halting: dict[str, float]  # halting vehicles on downstream lanes
    ped_waiting: np.ndarray  # waiting pedestrians per corner walking area (4)
    total_acc_wait: float  # sum of accumulated waiting time of vehicles on incoming lanes
    neighbour_pressure: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def pressure(self) -> float:
        return float(self.lane_halting.sum() - sum(self.out_lane_halting.values()))


class SumoSim:
    """One SUMO run for a (network, scenario, seed) triple."""

    def __init__(
        self,
        cfg: SimConfig,
        network: str,
        route_file: Path,
        seed: int,
        *,
        episode_seconds: int | None = None,
        step_length: float | None = None,
        tripinfo: Path | None = None,
        fcd: Path | None = None,
        gui: bool = False,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.spec: NetworkSpec = build_spec(network, cfg.networks)
        self.route_file = route_file
        self.seed = seed
        self.episode_seconds = episode_seconds or cfg.sumo.episode_seconds
        self.step_length = step_length or cfg.sumo.step_length
        self.tripinfo = tripinfo
        self.fcd = fcd
        self.gui = gui
        self.extra_args = extra_args
        self.net_xml = REPO_ROOT / "sim" / "networks" / f"{network}.net.xml"
        self.vtypes = REPO_ROOT / "sim" / "vtypes" / "dhaka.add.xml"
        arm_dirs = {j.id: {a.inbound_edge: d for d, a in j.arms.items()} for j in self.spec.junctions.values()}
        self.tables: dict[str, PhaseTable] = load_phase_tables(self.net_xml, arm_dirs)
        self.junction_ids: list[str] = list(self.spec.junctions)
        self._t: Any = None
        self._lane_arm: dict[str, dict[str, str]] = {
            jid: {lane: self._arm_of_lane(jid, lane) for lane in self.tables[jid].incoming_lanes}
            for jid in self.junction_ids
        }
        self._out_lanes: dict[str, list[str]] = {
            jid: sorted({lk.to_lane for lk in self.tables[jid].links if lk.turn != "c"}) for jid in self.junction_ids
        }
        self._walkingareas: dict[str, list[str]] = {jid: [f":{jid}_w{i}" for i in range(4)] for jid in self.junction_ids}
        self.started = False

    def _arm_of_lane(self, jid: str, lane: str) -> str:
        edge = lane.rsplit("_", 1)[0]
        for d, arm in self.spec.junctions[jid].arms.items():
            if arm.inbound_edge == edge:
                return d
        raise KeyError(lane)

    # ------------------------------------------------------------------ lifecycle
    def command(self) -> list[str]:
        cmd = [
            sumo_binary("sumo-gui" if self.gui else "sumo"),
            "-n", str(self.net_xml),
            "-r", str(self.route_file),
            "-a", str(self.vtypes),
            "--seed", str(self.seed),
            "--step-length", f"{self.step_length:g}",
            "--lateral-resolution", f"{self.cfg.sumo.lateral_resolution:g}",
            "--begin", "0",
            "--end", str(self.episode_seconds + 1),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "1000",
            "--pedestrian.model", "striping",
            "--collision.action", "warn",
            "--default.emergencydecel", "9",
            "--xml-validation", "never",
        ]
        if self.tripinfo is not None:
            cmd += ["--tripinfo-output", str(self.tripinfo), "--tripinfo-output.write-unfinished", "true"]
        if self.fcd is not None:
            cmd += ["--fcd-output", str(self.fcd), "--fcd-output.attributes", "id,x,y,angle,speed,type,lane"]
        cmd += list(self.extra_args)
        return cmd

    def start(self) -> None:
        self._t = traci_module()
        self._t.start(self.command())
        self.started = True
        # show a sane state from the first step
        for jid in self.junction_ids:
            self._t.trafficlight.setRedYellowGreenState(jid, self.tables[jid].phases[0].state)

    def close(self) -> None:
        if self.started and self._t is not None:
            try:
                self._t.close()
            except Exception:  # pragma: no cover - libsumo raises if already closed
                pass
        self.started = False

    @property
    def time(self) -> float:
        return float(self._t.simulation.getTime())

    def step(self) -> None:
        self._t.simulationStep()

    def set_state(self, jid: str, state: str) -> None:
        self._t.trafficlight.setRedYellowGreenState(jid, state)

    # ------------------------------------------------------------------ sensing
    def junction_state(self, jid: str) -> JunctionState:
        t = self._t
        table = self.tables[jid]
        lanes = table.incoming_lanes
        n = len(lanes)
        halting = np.zeros(n)
        max_wait = np.zeros(n)
        count = np.zeros(n)
        class_counts = np.zeros((len(DIRS), len(VEHICLE_CLASSES)))
        total_acc_wait = 0.0
        for i, lane in enumerate(lanes):
            ids = t.lane.getLastStepVehicleIDs(lane)
            halting[i] = t.lane.getLastStepHaltingNumber(lane)
            count[i] = len(ids)
            arm_idx = DIRS.index(self._lane_arm[jid][lane])
            mw = 0.0
            for vid in ids:
                w = t.vehicle.getWaitingTime(vid)
                mw = max(mw, w)
                total_acc_wait += t.vehicle.getAccumulatedWaitingTime(vid)
                cls = TYPE_TO_CLASS.get(t.vehicle.getTypeID(vid))
                if cls is not None:
                    class_counts[arm_idx, VEHICLE_CLASSES.index(cls)] += 1
            max_wait[i] = mw
        out_halting = {lane: float(t.lane.getLastStepHaltingNumber(lane)) for lane in self._out_lanes[jid]}
        ped = np.zeros(4)
        for pid in t.person.getIDList():
            road = t.person.getRoadID(pid)
            if road.startswith(f":{jid}_w") and t.person.getWaitingTime(pid) > 0:
                k = int(road.rsplit("w", 1)[1])
                if k < 4:
                    ped[k] += 1
        return JunctionState(
            junction=jid,
            lanes=lanes,
            lane_halting=halting,
            lane_max_wait=max_wait,
            lane_count=count,
            class_counts=class_counts,
            out_lane_halting=out_halting,
            ped_waiting=ped,
            total_acc_wait=total_acc_wait,
        )

    def per_second_metrics(self, jid: str) -> dict[str, Any]:
        """Cheap aggregates recorded every simulation second (used by evaluation and replays)."""
        t = self._t
        table = self.tables[jid]
        queue_per_arm = dict.fromkeys(DIRS, 0.0)
        waiting_total = 0.0
        vehicles = 0
        for lane in table.incoming_lanes:
            h = t.lane.getLastStepHaltingNumber(lane)
            queue_per_arm[self._lane_arm[jid][lane]] += h
            waiting_total += t.lane.getWaitingTime(lane)
            vehicles += t.lane.getLastStepVehicleNumber(lane)
        ped_waiting = 0
        for pid in t.person.getIDList():
            if t.person.getWaitingTime(pid) > 0 and t.person.getRoadID(pid).startswith(f":{jid}_"):
                ped_waiting += 1
        return {
            "queue": queue_per_arm,
            "vehicles_waiting": float(sum(queue_per_arm.values())),
            "avg_wait": waiting_total / vehicles if vehicles else 0.0,
            "vehicles": vehicles,
            "ped_waiting": ped_waiting,
        }

    def vehicle_count(self) -> int:
        return int(self._t.vehicle.getIDCount())

    def person_count(self) -> int:
        return int(self._t.person.getIDCount())


def state_to_vector(
    s: JunctionState, phase_onehot: np.ndarray, time_in_phase: float, in_clearance: bool, junction_onehot: np.ndarray, max_green: float
) -> np.ndarray:
    """Flatten a `JunctionState` plus signal status into the policy observation vector."""
    parts = [
        np.clip(s.lane_halting / LANE_CAPACITY_VEH, 0, 2),
        np.clip(s.lane_max_wait / MAX_WAIT_NORM, 0, 2),
        np.clip(s.class_counts.reshape(-1) / CLASS_COUNT_NORM, 0, 2),
        phase_onehot,
        np.array([min(time_in_phase / max_green, 1.5), 1.0 if in_clearance else 0.0]),
        np.clip(s.ped_waiting / PED_NORM, 0, 2),
        np.clip(s.neighbour_pressure / PRESSURE_NORM, -2, 2),
        junction_onehot,
    ]
    return np.concatenate(parts).astype(np.float32)


def observation_size(n_lanes: int = 12, n_phases: int = 4, max_junctions: int = 4) -> int:
    return 2 * n_lanes + len(DIRS) * len(VEHICLE_CLASSES) + n_phases + 2 + 4 + 4 + max_junctions
