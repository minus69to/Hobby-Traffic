"""Typed access to the YAML configuration files."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary (paths are resolved against the repo root)."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    return data


@dataclass(frozen=True)
class SignalConfig:
    """Signal timing constraints shared by the mask, the shield and the service."""

    n_phases: int = 4
    yellow: int = 3
    all_red: int = 2
    min_green: int = 10
    max_green: int = 60
    ped_walk_min: int = 8

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SignalConfig:
        return SignalConfig(
            n_phases=int(d.get("n_phases", 4)),
            yellow=int(d.get("yellow", 3)),
            all_red=int(d.get("all_red", 2)),
            min_green=int(d.get("min_green", 10)),
            max_green=int(d.get("max_green", 60)),
            ped_walk_min=int(d.get("ped_walk_min", 8)),
        )


@dataclass(frozen=True)
class SumoConfig:
    step_length: float = 1.0
    lateral_resolution: float = 0.8
    lefthand: bool = True
    episode_seconds: int = 3600
    warmup_seconds: int = 300
    decision_interval: int = 5
    ignore_junction_blocker: int = 30
    pedestrian_jamtime: int = 60

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SumoConfig:
        return SumoConfig(
            step_length=float(d.get("step_length", 1.0)),
            lateral_resolution=float(d.get("lateral_resolution", 0.8)),
            lefthand=bool(d.get("lefthand", True)),
            episode_seconds=int(d.get("episode_seconds", 3600)),
            warmup_seconds=int(d.get("warmup_seconds", 300)),
            decision_interval=int(d.get("decision_interval", 5)),
            ignore_junction_blocker=int(d.get("ignore_junction_blocker", 30)),
            pedestrian_jamtime=int(d.get("pedestrian_jamtime", 60)),
        )


@dataclass
class SimConfig:
    """Whole `configs/sim.yaml` with typed sub-sections and the raw dictionary."""

    raw: dict[str, Any]
    sumo: SumoConfig = field(default_factory=SumoConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    path: str = "configs/sim.yaml"

    @staticmethod
    def load(path: str | Path = "configs/sim.yaml") -> SimConfig:
        raw = load_yaml(path)
        return SimConfig(
            raw=raw,
            sumo=SumoConfig.from_dict(raw.get("sumo", {})),
            signal=SignalConfig.from_dict(raw.get("signal", {})),
            path=str(path),
        )

    @property
    def networks(self) -> dict[str, Any]:
        return dict(self.raw["networks"])

    @property
    def scenarios(self) -> dict[str, Any]:
        return dict(self.raw["scenarios"])

    @property
    def vtypes(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["vtypes"])

    @property
    def vehicle_mix(self) -> dict[str, float]:
        mix = {str(k): float(v) for k, v in self.raw["vehicle_mix"].items()}
        total = sum(mix.values())
        return {k: v / total for k, v in mix.items()}

    @property
    def seeds(self) -> dict[str, Any]:
        return dict(self.raw["seeds"])

    def scenario(self, name: str) -> dict[str, Any]:
        """Resolve a scenario including its `base` inheritance and `scale` factor."""
        scen = dict(self.scenarios[name])
        if "base" in scen:
            base = self.scenario(str(scen["base"]))
            merged = dict(base)
            merged.update({k: v for k, v in scen.items() if k != "base"})
            scen = merged
        if "scale" in scen:
            s = float(scen.pop("scale"))
            scen["arms"] = {k: float(v) * s for k, v in scen["arms"].items()}
            scen["pedestrians_per_crossing"] = float(scen["pedestrians_per_crossing"]) * s
        scen["name"] = name
        return scen


def sumo_binary(name: str = "sumo") -> str:
    """Locate a SUMO binary from `SUMO_HOME`, the active virtualenv or the PATH."""
    sumo_home = os.environ.get("SUMO_HOME")
    candidates = []
    if sumo_home:
        candidates.append(Path(sumo_home) / "bin" / name)
    try:
        import sumo as sumo_pkg  # type: ignore[import-not-found]

        candidates.append(Path(sumo_pkg.__file__).parent / "bin" / name)
    except ImportError:  # pragma: no cover - only when the pip package is missing
        pass
    candidates.append(Path(sys.executable).parent / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return name
