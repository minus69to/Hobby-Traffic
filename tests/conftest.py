"""Shared fixtures: phase table of the single junction (built from the committed network file)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

from drs.config import REPO_ROOT, SignalConfig, SimConfig
from drs.signal import PhaseTable, load_phase_tables
from drs.topology import build_spec


@pytest.fixture(scope="session")
def sim_cfg() -> SimConfig:
    return SimConfig.load("configs/sim.yaml")


@pytest.fixture(scope="session")
def ensure_network(sim_cfg: SimConfig) -> Path:
    net = REPO_ROOT / "sim" / "networks" / "single.net.xml"
    if not net.exists():
        from drs.netbuild import build_all

        build_all(sim_cfg)
    return net


@pytest.fixture(scope="session")
def phase_table(sim_cfg: SimConfig, ensure_network: Path) -> PhaseTable:
    spec = build_spec("single", sim_cfg.networks)
    arm_dirs = {j.id: {a.inbound_edge: d for d, a in j.arms.items()} for j in spec.junctions.values()}
    return load_phase_tables(ensure_network, arm_dirs)["C"]


@pytest.fixture(scope="session")
def signal_cfg() -> SignalConfig:
    return SignalConfig(n_phases=4, yellow=3, all_red=2, min_green=10, max_green=60, ped_walk_min=8)
