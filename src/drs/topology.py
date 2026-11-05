"""Abstract description of the signalised networks (junctions, arms, edges, turns).

Everything that needs to know "which edge is the northern approach of junction J" (network
building, demand generation, phase construction, observations, geometry export) uses this module,
so the convention lives in one place.

Conventions
-----------
* Directions are ``N``, ``E``, ``S``, ``W``.  Edge ids are ``<from_node>_<to_node>``.
* The **arm** ``N`` of a junction is the road leaving to the north; vehicles *on* that arm's
  inbound edge are heading south.
* Bangladesh drives on the left, so from the driver's point of view the **left turn is the easy
  (near-side) turn** and the right turn crosses oncoming traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIRS: tuple[str, ...] = ("N", "E", "S", "W")
VEC: dict[str, tuple[float, float]] = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
OPPOSITE: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
# Turn mapping for a vehicle *heading* in direction h (not the arm it comes from).
# Heading north: left is west (counter-clockwise), right is east (clockwise).
LEFT_OF: dict[str, str] = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF: dict[str, str] = {"N": "E", "E": "S", "S": "W", "W": "N"}
TURN_NAMES: tuple[str, ...] = ("l", "s", "r")


@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float
    signalised: bool


@dataclass(frozen=True)
class Arm:
    """One road connected to a junction, in direction `direction` from the junction."""

    junction: str
    direction: str
    neighbour: str  # node id at the far end
    inbound_edge: str  # edge whose vehicles enter the junction (from neighbour to junction)
    outbound_edge: str  # edge leaving the junction towards the neighbour
    external: bool  # neighbour is a boundary node (vehicles enter/leave the network here)

    @property
    def heading(self) -> str:
        """Heading of vehicles on the inbound edge (they travel from the arm towards the centre)."""
        return OPPOSITE[self.direction]


@dataclass
class Junction:
    id: str
    x: float
    y: float
    arms: dict[str, Arm] = field(default_factory=dict)

    def turn_target(self, from_dir: str, turn: str) -> str:
        """Direction of the arm a vehicle leaves by when arriving from `from_dir` and turning `turn`."""
        heading = OPPOSITE[from_dir]
        if turn == "s":
            return heading
        if turn == "l":
            return LEFT_OF[heading]
        if turn == "r":
            return RIGHT_OF[heading]
        raise ValueError(turn)


@dataclass
class NetworkSpec:
    name: str
    nodes: dict[str, Node]
    junctions: dict[str, Junction]
    lanes_per_approach: int
    speed_limit: float
    sidewalk_width: float

    @property
    def edges(self) -> list[tuple[str, str, str]]:
        """All (edge_id, from_node, to_node) tuples, each road once per direction."""
        seen: dict[str, tuple[str, str, str]] = {}
        for j in self.junctions.values():
            for arm in j.arms.values():
                seen[arm.inbound_edge] = (arm.inbound_edge, arm.neighbour, j.id)
                seen[arm.outbound_edge] = (arm.outbound_edge, j.id, arm.neighbour)
        return sorted(seen.values())

    @property
    def entry_edges(self) -> list[tuple[str, str, str]]:
        """(edge, junction, arm direction) for every edge entering the network from outside."""
        out = []
        for j in self.junctions.values():
            for d, arm in j.arms.items():
                if arm.external:
                    out.append((arm.inbound_edge, j.id, d))
        return out

    def junction_of_inbound(self, edge: str) -> tuple[str, str]:
        for j in self.junctions.values():
            for d, arm in j.arms.items():
                if arm.inbound_edge == edge:
                    return j.id, d
        raise KeyError(edge)


def _add_arm(spec_nodes: dict[str, Node], junction: Junction, direction: str, neighbour: str) -> None:
    n = spec_nodes[neighbour]
    junction.arms[direction] = Arm(
        junction=junction.id,
        direction=direction,
        neighbour=neighbour,
        inbound_edge=f"{neighbour}_{junction.id}",
        outbound_edge=f"{junction.id}_{neighbour}",
        external=not n.signalised,
    )


def build_single(cfg: dict[str, Any]) -> NetworkSpec:
    """One 4-arm junction `C` with boundary nodes N, E, S, W."""
    length = float(cfg.get("approach_length", 300.0))
    nodes = {"C": Node("C", 0.0, 0.0, True)}
    for d in DIRS:
        vx, vy = VEC[d]
        nodes[d] = Node(d, vx * length, vy * length, False)
    junction = Junction("C", 0.0, 0.0)
    for d in DIRS:
        _add_arm(nodes, junction, d, d)
    return NetworkSpec(
        name="single",
        nodes=nodes,
        junctions={"C": junction},
        lanes_per_approach=int(cfg.get("lanes_per_approach", 3)),
        speed_limit=float(cfg.get("speed_limit", 13.89)),
        sidewalk_width=float(cfg.get("sidewalk_width", 2.5)),
    )


def build_grid2x2(cfg: dict[str, Any]) -> NetworkSpec:
    """Four junctions J<row><col> on a 2x2 lattice with boundary nodes on the free sides."""
    spacing = float(cfg.get("spacing", 250.0))
    length = float(cfg.get("approach_length", 300.0))
    nodes: dict[str, Node] = {}
    junctions: dict[str, Junction] = {}
    coords = {"J00": (0, 0), "J01": (1, 0), "J10": (0, 1), "J11": (1, 1)}
    for jid, (c, r) in coords.items():
        nodes[jid] = Node(jid, c * spacing, r * spacing, True)
        junctions[jid] = Junction(jid, c * spacing, r * spacing)
    for jid, (c, r) in coords.items():
        for d in DIRS:
            vx, vy = VEC[d]
            nc, nr = c + int(vx), r + int(vy)
            neighbour = next((k for k, v in coords.items() if v == (nc, nr)), None)
            if neighbour is None:
                neighbour = f"B{jid[1:]}{d}"
                nodes[neighbour] = Node(neighbour, c * spacing + vx * length, r * spacing + vy * length, False)
            _add_arm(nodes, junctions[jid], d, neighbour)
    return NetworkSpec(
        name="grid2x2",
        nodes=nodes,
        junctions=junctions,
        lanes_per_approach=int(cfg.get("lanes_per_approach", 3)),
        speed_limit=float(cfg.get("speed_limit", 13.89)),
        sidewalk_width=float(cfg.get("sidewalk_width", 2.5)),
    )


def build_spec(name: str, networks_cfg: dict[str, Any]) -> NetworkSpec:
    cfg = networks_cfg[name]
    if name == "single":
        return build_single(cfg)
    if name == "grid2x2":
        return build_grid2x2(cfg)
    raise KeyError(f"unknown network {name}")
