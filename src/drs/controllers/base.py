"""Controller interface."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from drs.shield import SafetyShield
from drs.sumo_sim import JunctionState


class Controller(Protocol):
    """A controller maps junction states (and the shield's status) to phase requests."""

    name: str

    def reset(self, env: Any) -> None:
        """Called after the environment was reset (gives access to tables, scenario, seed)."""

    def act(
        self, states: dict[str, JunctionState], obs: np.ndarray, masks: np.ndarray, shields: dict[str, SafetyShield]
    ) -> list[int]:
        """Return one requested phase per junction, in `env.junction_ids` order."""
