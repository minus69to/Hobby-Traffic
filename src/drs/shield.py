"""Runtime safety shield: a finite-state machine between any controller and the signal heads.

Every controller in this repository (fixed-time, max-pressure, PPO, the MQTT edge service) only
*requests* a phase.  The shield decides what the lamps actually show:

* ``GREEN(phase)`` -> ``YELLOW(phase -> target)`` -> ``ALL_RED(target)`` -> ``GREEN(target)``
* a change is rejected before ``min_green`` seconds of green (and before ``ped_walk_min`` if
  pedestrians have walk), ignored during yellow / all-red, and forced after ``max_green``;
* phase indexes outside the table and states with conflicting protected greens are rejected;
* every intervention is logged with the simulation time, the request and the reason.

The same object also produces the **action mask** used by MaskablePPO, so the policy can only
pick what the shield would accept; when the two disagree the shield wins and logs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from drs.config import SignalConfig
from drs.signal import PhaseTable


class Mode(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass(frozen=True)
class Intervention:
    time: float
    kind: str
    requested: int | None
    applied: int
    reason: str


@dataclass
class ShieldStats:
    interventions: int = 0
    rejected_min_green: int = 0
    rejected_ped_walk: int = 0
    ignored_in_clearance: int = 0
    forced_max_green: int = 0
    rejected_invalid: int = 0
    switches: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class SafetyShield:
    table: PhaseTable
    cfg: SignalConfig
    initial_phase: int = 0
    log_interventions: bool = True
    mode: Mode = Mode.GREEN
    phase: int = 0
    target: int = 0
    time_in_mode: float = 0.0
    time: float = 0.0
    stats: ShieldStats = field(default_factory=ShieldStats)
    log: list[Intervention] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.initial_phase < self.table.n_phases:
            raise ValueError("initial phase out of range")
        self.phase = self.target = self.initial_phase
        self.mode = Mode.GREEN
        self.time_in_mode = 0.0

    # ------------------------------------------------------------------ queries
    @property
    def n_phases(self) -> int:
        return self.table.n_phases

    @property
    def in_clearance(self) -> bool:
        return self.mode is not Mode.GREEN

    @property
    def time_in_green(self) -> float:
        return self.time_in_mode if self.mode is Mode.GREEN else 0.0

    def current_state(self) -> str:
        if self.mode is Mode.GREEN:
            return self.table.phases[self.phase].state
        if self.mode is Mode.YELLOW:
            return self.table.yellow_state(self.phase, self.target)
        return self.table.all_red()

    def _walk_locked(self) -> bool:
        return self.table.phases[self.phase].has_walk and self.time_in_mode < self.cfg.ped_walk_min

    def valid_actions(self) -> np.ndarray:
        """Boolean mask over phase indexes; action == current phase means "keep"."""
        mask = np.zeros(self.n_phases, dtype=bool)
        if self.mode is not Mode.GREEN:
            mask[self.target] = True
            return mask
        if self.time_in_mode < self.cfg.min_green or self._walk_locked():
            mask[self.phase] = True
            return mask
        mask[:] = True
        if self.time_in_mode >= self.cfg.max_green and self.n_phases > 1:
            mask[self.phase] = False
        return mask

    # ------------------------------------------------------------------ control
    def _log(self, kind: str, requested: int | None, reason: str) -> None:
        self.stats.interventions += 1
        setattr(self.stats, kind, getattr(self.stats, kind) + 1)
        if self.log_interventions:
            self.log.append(Intervention(self.time, kind, requested, self.phase, reason))

    def _start_change(self, target: int) -> None:
        self.target = target
        if self.cfg.yellow > 0:
            self.mode = Mode.YELLOW
        elif self.cfg.all_red > 0:
            self.mode = Mode.ALL_RED
        else:  # pragma: no cover - zero clearance is only used in tests
            self.mode = Mode.GREEN
            self.phase = target
            self.stats.switches += 1
        self.time_in_mode = 0.0

    def request(self, phase: int | None) -> bool:
        """Controller asks for `phase` (None or the current phase = keep). Returns True if accepted."""
        if phase is None or (phase == self.phase and self.mode is Mode.GREEN):
            return True
        if not isinstance(phase, int | np.integer) or not 0 <= int(phase) < self.n_phases:
            self._log("rejected_invalid", None if phase is None else int(phase), "phase index out of range")
            return False
        phase = int(phase)
        if self.mode is not Mode.GREEN:
            if phase != self.target:
                self._log("ignored_in_clearance", phase, f"request during {self.mode.value}")
                return False
            return True
        if self.time_in_mode < self.cfg.min_green:
            self._log("rejected_min_green", phase, f"green for {self.time_in_mode:.0f}s < min_green")
            return False
        if self._walk_locked():
            self._log("rejected_ped_walk", phase, "pedestrian walk minimum not met")
            return False
        if not self.table.is_conflict_free(self.table.phases[phase].state):  # pragma: no cover
            self._log("rejected_invalid", phase, "conflicting greens")
            return False
        self._start_change(phase)
        return True

    def tick(self, dt: float = 1.0) -> str:
        """Return the state to show for the next `dt` seconds, then advance the FSM.

        The forced max-green change is checked *before* the state is produced, so a green is
        never displayed for longer than ``max_green`` seconds even if no controller ever calls
        :meth:`request`.
        """
        if self.mode is Mode.GREEN and self.time_in_mode >= self.cfg.max_green and self.n_phases > 1:
            nxt = (self.phase + 1) % self.n_phases
            self._log("forced_max_green", None, f"green reached max_green={self.cfg.max_green}s")
            self._start_change(nxt)
        state = self.current_state()
        self.time += dt
        self.time_in_mode += dt
        if self.mode is Mode.YELLOW and self.time_in_mode >= self.cfg.yellow:
            self.mode = Mode.ALL_RED
            self.time_in_mode = 0.0
            if self.cfg.all_red <= 0:
                self._enter_green()
        elif self.mode is Mode.ALL_RED and self.time_in_mode >= self.cfg.all_red:
            self._enter_green()
        return state

    def _enter_green(self) -> None:
        self.mode = Mode.GREEN
        self.phase = self.target
        self.time_in_mode = 0.0
        self.stats.switches += 1

    def reset(self, initial_phase: int = 0) -> None:
        self.initial_phase = initial_phase
        self.__post_init__()
        self.time = 0.0
        self.stats = ShieldStats()
        self.log = []
