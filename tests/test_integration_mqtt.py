"""Integration test: mosquitto + controller service + SUMO bridge for 5 simulated minutes."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.signal import load_phase_tables
from drs.topology import build_spec

pytestmark = pytest.mark.integration


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.skipif(shutil.which("mosquitto") is None, reason="mosquitto broker not installed")
def test_full_loop_five_simulated_minutes() -> None:
    port = free_port()
    http_port = free_port()
    tmp = Path(tempfile.mkdtemp(prefix="drs_mqtt_"))
    conf = tmp / "mosquitto.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), LIBSUMO_AS_TRACI="1")
    broker = subprocess.Popen(["mosquitto", "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    override = json.dumps({"mqtt": {"port": port}, "http": {"port": http_port}, "watchdog": {"stale_after_s": 30}})
    service = None
    try:
        time.sleep(0.5)
        service = subprocess.Popen(
            [sys.executable, "-m", "drs.service.app", "--config", "configs/service.yaml", "--override", override],
            cwd=REPO_ROOT,
            env=env,
        )
        time.sleep(2.5)
        from drs.service.bridge import run_bridge

        scfg = load_yaml("configs/service.yaml")
        scfg["mqtt"]["port"] = port
        stats = asyncio.run(
            run_bridge(
                scfg, "offpeak", 300, 10001, wait_s=3.0, log_path=REPO_ROOT / "results" / "service_integration.json"
            )
        )
        assert stats["steps"] == 300
        assert stats["commands"] >= 280, stats
        assert stats["misses"] <= 20
        assert sum(1 for s in stats["states"] if "y" in s) >= 3
        cfg = SimConfig.load()
        spec = build_spec("single", cfg.networks)
        arm_dirs = {j.id: {a.inbound_edge: d for d, a in j.arms.items()} for j in spec.junctions.values()}
        table = load_phase_tables(REPO_ROOT / "sim" / "networks" / "single.net.xml", arm_dirs)["C"]
        assert all(table.is_conflict_free(s) for s in stats["states"])
        health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{http_port}/health", timeout=5).read())
        assert health["decisions"] >= 55
    finally:
        if service is not None:
            service.terminate()
            service.wait(timeout=10)
        broker.terminate()
        broker.wait(timeout=10)
