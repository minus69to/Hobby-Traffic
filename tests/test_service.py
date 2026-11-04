"""Unit tests for the edge service core (no broker needed)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from drs.config import REPO_ROOT, SimConfig
from drs.service.core import ControllerCore, CoreConfig
from drs.service.schemas import CrossingReading, DetectorMessage, LaneReading
from drs.service.sidecar import Metrics, make_app

SMALL = REPO_ROOT / "models" / "small" / "seed13"


def make_msg(core: ControllerCore, t: float, halting: float = 3.0) -> DetectorMessage:
    lanes = [
        LaneReading(
            lane_id=lane, halting=halting, max_wait=20.0, count=halting + 2, class_counts={"car": 2, "rickshaw": 1}
        )
        for lane in core.table.incoming_lanes
    ]
    return DetectorMessage(
        junction_id="C",
        timestamp=t,
        lanes=lanes,
        crossings=[CrossingReading(crossing_id=f":C_w{k}", waiting=1) for k in range(4)],
    )


@pytest.fixture(scope="module")
def sim_cfg() -> SimConfig:
    return SimConfig.load()


def test_schema_roundtrip_and_version() -> None:
    m = DetectorMessage(
        junction_id="C", timestamp=1.0, lanes=[LaneReading(lane_id="N_C_1", halting=1, max_wait=2, count=3)]
    )
    j = m.model_dump_json()
    assert '"schema_version":"1.0"' in j
    assert DetectorMessage.model_validate_json(j) == m
    with pytest.raises(ValueError):
        DetectorMessage.model_validate_json(j.replace('"1.0"', '"9.9"'))


def test_fallback_without_model_is_fixed_time_and_safe(sim_cfg: SimConfig) -> None:
    core = ControllerCore(sim_cfg, CoreConfig(fallback_green_s=20), None, None)
    out = core.on_detectors(make_msg(core, 0.0))
    assert out.alarms and out.alarms[0].kind == "model_failure"
    assert core.controller == "fixed_time"
    states = []
    for t in range(1, 120):
        out = core.on_detectors(make_msg(core, float(t)))
        states.append(out.commands[0].state)
        assert core.table.is_conflict_free(out.commands[0].state)
    assert any("y" in s for s in states)
    assert core.shield.stats.switches >= 3


@pytest.mark.skipif(not (SMALL / "policy_int8.onnx").exists(), reason="small ONNX model not exported")
def test_rl_decisions_with_onnx(sim_cfg: SimConfig) -> None:
    core = ControllerCore(sim_cfg, CoreConfig(), SMALL / "policy_int8.onnx", SMALL / "onnx_meta.json")
    assert core.model_loaded and core.controller == "rl"
    for t in range(0, 60):
        out = core.on_detectors(make_msg(core, float(t)))
        assert out.commands[0].controller == "rl"
        assert core.table.is_conflict_free(out.commands[0].state)
    assert core.decisions == 12
    assert core.status(60.0).inference_ms_p50 >= 0


def test_watchdog_stale_detectors_falls_back(sim_cfg: SimConfig) -> None:
    core = ControllerCore(sim_cfg, CoreConfig(stale_after_s=5), None, None)
    core.on_detectors(make_msg(core, 0.0))
    core.model_loaded = True  # pretend the model works so a fallback is observable
    core.controller = "rl"
    core.alarm_state = None
    out = core.watchdog(now_wall=time.monotonic() + 10.0)
    assert core.controller == "fixed_time"
    assert out.alarms and out.alarms[0].kind == "stale_detectors"
    assert out.commands and core.table.is_conflict_free(out.commands[0].state)
    out2 = core.on_detectors(make_msg(core, 1.0))  # data is back -> recovery alarm
    assert any(a.kind == "recovered" for a in out2.alarms)
    assert core.controller == "rl"


def test_inference_deadline_triggers_fallback(sim_cfg: SimConfig) -> None:
    class SlowPolicy:
        def choose(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
            time.sleep(0.02)
            return np.array([0])

    core = ControllerCore(sim_cfg, CoreConfig(inference_deadline_ms=1.0), None, None)
    core.policy = SlowPolicy()
    core.model_loaded = True
    core.controller = "rl"
    core.alarm_state = None
    out = core.on_detectors(make_msg(core, 0.0))
    assert core.controller == "fixed_time"
    assert any(a.kind == "inference_deadline" for a in out.alarms)


def test_sidecar_endpoints(sim_cfg: SimConfig) -> None:
    from fastapi.testclient import TestClient

    core = ControllerCore(sim_cfg, CoreConfig(), None, None)
    core.on_detectors(make_msg(core, 0.0))
    client = TestClient(make_app(core, Metrics()))
    h = client.get("/health").json()
    assert h["controller"] == "fixed_time" and h["decisions"] == 1
    m = client.get("/metrics").text
    assert "drs_controller_mode" in m and "drs_shield_interventions_total" in m
