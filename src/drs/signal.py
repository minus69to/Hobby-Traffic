"""Phase tables and conflict matrices derived from the SUMO network file.

The four-phase plan for a 4-arm junction in a lefthand network:

0. north-south through (protected) + north-south left turns (yielding) + east/west crossings walk
1. north-south right turns (protected, they cross oncoming traffic)
2. east-west through + east-west left turns (yielding) + north/south crossings walk
3. east-west right turns

Yellow (3 s) and all-red (2 s) intervals are *not* phases: the safety shield inserts them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GREEN_CHARS = ("G", "g")


@dataclass(frozen=True)
class Link:
    index: int
    from_lane: str
    to_lane: str
    arm: str  # direction of the arm the vehicle (or pedestrian) comes from / crosses
    turn: str  # "l", "s", "r" for vehicles, "c" for crossings

    @property
    def from_edge(self) -> str:
        return self.from_lane.rsplit("_", 1)[0]


@dataclass(frozen=True)
class Phase:
    index: int
    name: str
    state: str  # SUMO red/yellow/green string
    has_walk: bool  # pedestrian crossings are green in this phase


@dataclass
class PhaseTable:
    """Links, phases and the conflict ("foes") matrix of one traffic light."""

    tls_id: str
    links: list[Link]
    phases: list[Phase]
    foes: np.ndarray  # bool matrix [n_links, n_links]
    incoming_lanes: list[str] = field(default_factory=list)  # vehicle lanes entering the junction
    crossings: dict[int, str] = field(default_factory=dict)  # link index -> crossing edge id

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def all_red(self) -> str:
        return "r" * self.n_links

    def yellow_state(self, cur: int, nxt: int) -> str:
        """Transition state: links losing green turn yellow, crossings losing walk turn red."""
        a, b = self.phases[cur].state, self.phases[nxt].state
        out = []
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca in GREEN_CHARS and cb not in GREEN_CHARS:
                out.append("r" if self.links[i].turn == "c" else "y")
            elif ca in GREEN_CHARS and cb in GREEN_CHARS:
                out.append(ca)
            else:
                out.append("r")
        return "".join(out)

    def is_conflict_free(self, state: str) -> bool:
        """True if no two mutually conflicting links are both protected green ('G')."""
        if len(state) != self.n_links:
            return False
        greens = [i for i, c in enumerate(state) if c == "G"]
        return not any(self.foes[i, j] for i in greens for j in greens if i < j)

    def phase_links(self, phase: int) -> list[Link]:
        return [lk for lk, c in zip(self.links, self.phases[phase].state) if c in GREEN_CHARS]

    def lane_phases(self) -> dict[str, list[int]]:
        """Incoming vehicle lane -> phases in which it has green (for the HUD/heatmap)."""
        out: dict[str, list[int]] = {}
        for p in self.phases:
            for lk in self.phase_links(p.index):
                if lk.turn != "c":
                    out.setdefault(lk.from_lane, []).append(p.index)
        return out


def _parse_foes(bits: str, n: int) -> list[bool]:
    # SUMO writes the bit for link j at string position n-1-j
    return [bits[n - 1 - j] == "1" for j in range(n)]


def load_phase_tables(net_xml: Path, arm_dirs: dict[str, dict[str, str]]) -> dict[str, PhaseTable]:
    """Build a `PhaseTable` for every traffic light in `net_xml`.

    `arm_dirs[tls_id]` maps the *inbound edge id* of each arm to its direction (N/E/S/W); the
    topology module provides it.
    """
    root = ET.parse(net_xml).getroot()
    crossing_arm: dict[str, str] = {}  # crossing edge id -> arm direction
    for edge in root.iter("edge"):
        if edge.get("function") == "crossing":
            for e in str(edge.get("crossingEdges", "")).split():
                for tls_id, dirs in arm_dirs.items():
                    if e in dirs:
                        crossing_arm[str(edge.get("id"))] = dirs[e]
    tables: dict[str, PhaseTable] = {}
    for tls_id, dirs in arm_dirs.items():
        links: dict[int, Link] = {}
        for con in root.iter("connection"):
            if con.get("tl") != tls_id:
                continue
            idx = int(str(con.get("linkIndex")))
            frm, to = str(con.get("from")), str(con.get("to"))
            from_lane = f"{frm}_{con.get('fromLane')}"
            to_lane = f"{to}_{con.get('toLane')}"
            if frm.startswith(":"):  # pedestrian crossing link
                links[idx] = Link(idx, from_lane, to_lane, crossing_arm.get(to, "?"), "c")
            else:
                links[idx] = Link(idx, from_lane, to_lane, dirs[frm], str(con.get("dir")))
        n = len(links)
        ordered = [links[i] for i in range(n)]
        foes = np.zeros((n, n), dtype=bool)
        for junction in root.iter("junction"):
            if junction.get("id") != tls_id:
                continue
            for req in junction.iter("request"):
                i = int(str(req.get("index")))
                foes[i] = _parse_foes(str(req.get("foes")), n)
        foes = foes | foes.T
        phases = _four_phase_plan(ordered)
        table = PhaseTable(
            tls_id=tls_id,
            links=ordered,
            phases=phases,
            foes=foes,
            incoming_lanes=sorted({lk.from_lane for lk in ordered if lk.turn != "c"}),
            crossings={lk.index: lk.to_lane.rsplit("_", 1)[0] for lk in ordered if lk.turn == "c"},
        )
        for p in phases:
            if not table.is_conflict_free(p.state):
                raise ValueError(f"phase {p.name} of {tls_id} has conflicting greens")
        tables[tls_id] = table
    return tables


def _four_phase_plan(links: list[Link]) -> list[Phase]:
    specs = [
        ("NS_through", {"N", "S"}, {"s": "G", "l": "g"}, {"E", "W"}),
        ("NS_right", {"N", "S"}, {"r": "G"}, set()),
        ("EW_through", {"E", "W"}, {"s": "G", "l": "g"}, {"N", "S"}),
        ("EW_right", {"E", "W"}, {"r": "G"}, set()),
    ]
    phases = []
    for i, (name, arms, turn_state, walk_arms) in enumerate(specs):
        chars = []
        for lk in links:
            if lk.turn == "c":
                chars.append("G" if lk.arm in walk_arms else "r")
            elif lk.arm in arms and lk.turn in turn_state:
                chars.append(turn_state[lk.turn])
            else:
                chars.append("r")
        phases.append(Phase(i, name, "".join(chars), bool(walk_arms)))
    return phases
