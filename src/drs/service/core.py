"""Transport-independent controller logic (used by the MQTT app and by the unit tests).

The core consumes detector messages, advances the safety shield by the simulated/real time that
passed, takes a decision every ``decision_interval`` seconds with the ONNX policy (or the fixed-time
fallback) and returns the messages to publish.  A watchdog switches to the fallback when the model
is unavailable, inference misses its deadline or detector data is stale.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drs.config import REPO_ROOT, SimConfig
from drs.service.observation import junction_state_from_message, observation_vector
from drs.service.schemas import AlarmMessage, CommandMessage, DetectorMessage, StatusMessage
from drs.shield import SafetyShield
from drs.signal import PhaseTable, load_phase_tables
from drs.topology import build_spec


@dataclass
class CoreConfig:
    junction_id: str = "C"
    network: str = "single"
    decision_interval: float = 5.0
    stale_after_s: float = 15.0
    inference_deadline_ms: float = 50.0
    fallback_green_s: float = 30.0


@dataclass
class CoreOutputs:
    commands: list[CommandMessage] = field(default_factory=list)
    status: StatusMessage | None = None
    alarms: list[AlarmMessage] = field(default_factory=list)


class ControllerCore:
    def __init__(
        self, sim_cfg: SimConfig, ccfg: CoreConfig, model_onnx: str | Path | None, model_meta: str | Path | None
    ) -> None:
        self.cfg = sim_cfg
        self.ccfg = ccfg
        spec = build_spec(ccfg.network, sim_cfg.networks)
        net_xml = REPO_ROOT / "sim" / "networks" / f"{ccfg.network}.net.xml"
        arm_dirs = {j.id: {a.inbound_edge: d for d, a in j.arms.items()} for j in spec.junctions.values()}
        self.table: PhaseTable = load_phase_tables(net_xml, arm_dirs)[ccfg.junction_id]
        self.lane_arm = {lane: arm_dirs[ccfg.junction_id][lane.rsplit("_", 1)[0]] for lane in self.table.incoming_lanes}
        self.shield = SafetyShield(self.table, sim_cfg.signal)
        self.junction_index = list(spec.junctions).index(ccfg.junction_id)
        self.policy: Any = None
        self.model_loaded = False
        self.model_error = ""
        if model_onnx is not None and model_meta is not None:
            try:
                from drs.controllers.ppo import OnnxController

                self.policy = OnnxController(model_onnx, model_meta)
                self.model_loaded = True
            except Exception as e:
                self.model_error = f"{type(e).__name__}: {e}"
        self.controller: str = "rl" if self.model_loaded else "fixed_time"
        self.last_msg_time: float | None = None
        self.last_wall = time.monotonic()
        self.since_decision = self.ccfg.decision_interval  # decide on the first message
        self.decision_id = 0
        self.decisions = 0
        self.fallbacks = 0
        self.latencies: list[float] = []
        self.alarm_state: str | None = None if self.model_loaded else "model_failure"
        self.shadow_ok = True  # last (shadow) inference met the deadline
        self.last_tick_wall = time.monotonic()
        self.last_decided = False
        self.last_inference_ms: float | None = None
        self.pending_alarm: list[AlarmMessage] = (
            []
            if self.model_loaded
            else [
                AlarmMessage(
                    junction_id=ccfg.junction_id,
                    timestamp=0.0,
                    kind="model_failure",
                    detail=self.model_error or "no model configured",
                )
            ]
        )

    # ------------------------------------------------------------------ helpers
    def _fixed_time_request(self) -> int | None:
        sh = self.shield
        if sh.in_clearance:
            return None
        if sh.time_in_green >= self.ccfg.fallback_green_s:
            return (sh.phase + 1) % sh.n_phases
        return None

    def _alarm(self, kind: str, detail: str, ts: float) -> AlarmMessage:
        return AlarmMessage(junction_id=self.ccfg.junction_id, timestamp=ts, kind=kind, detail=detail)  # type: ignore[arg-type]

    def _enter_fallback(self, kind: str, detail: str, ts: float, out: CoreOutputs) -> None:
        if self.controller != "fixed_time":
            self.fallbacks += 1
        self.controller = "fixed_time"
        if self.alarm_state != kind:
            self.alarm_state = kind
            out.alarms.append(self._alarm(kind, detail, ts))

    def _maybe_recover(self, ts: float, out: CoreOutputs) -> None:
        """Return to RL control once data is fresh and (after a deadline miss) a shadow inference met the deadline."""
        if self.controller != "fixed_time" or not self.model_loaded:
            return
        if self.alarm_state is None and self.shadow_ok:
            self.controller = "rl"
            self.shadow_ok = False
            out.alarms.append(self._alarm("recovered", "detector data fresh and inference within deadline", ts))

    def _command(self, ts: float, requested: int | None, state: str) -> CommandMessage:
        return CommandMessage(
            junction_id=self.ccfg.junction_id,
            timestamp=ts,
            phase=self.shield.target if self.shield.in_clearance else self.shield.phase,
            requested_phase=requested,
            state=state,
            shield_mode=self.shield.mode.value,  # type: ignore[arg-type]
            controller=self.controller,  # type: ignore[arg-type]
            decision_id=self.decision_id,
        )

    # ------------------------------------------------------------------ main entry points
    def on_detectors(self, msg: DetectorMessage) -> CoreOutputs:
        out = CoreOutputs(alarms=list(self.pending_alarm))
        self.pending_alarm = []
        if self.last_msg_time is None:
            dt = 1.0
        elif msg.timestamp <= self.last_msg_time:
            dt = 0.0  # duplicate or out-of-order message: never advance the timers twice
        else:
            dt = float(np.clip(msg.timestamp - self.last_msg_time, 0.0, 5.0))
        self.last_msg_time = max(msg.timestamp, self.last_msg_time or msg.timestamp)
        now = time.monotonic()
        self.last_wall = now
        self.last_tick_wall = now
        if self.alarm_state == "stale_detectors":
            self.alarm_state = None  # data is back; the controller recovers below if the model is healthy
            out.alarms.append(self._alarm("recovered", "detector data fresh again", msg.timestamp))
        self._maybe_recover(msg.timestamp, out)
        self.since_decision += dt
        requested: int | None = None
        self.last_decided = False
        self.last_inference_ms = None
        if self.since_decision >= self.ccfg.decision_interval:
            self.since_decision = 0.0
            self.decision_id += 1
            self.last_decided = True
            state = junction_state_from_message(msg, self.table, self.lane_arm)
            mask = self.shield.valid_actions()
            action: int | None = None
            if self.model_loaded and self.policy is not None:
                # the policy runs whenever the model is loaded; in fallback it runs in shadow so that
                # recovery is only granted after an in-deadline inference
                obs = observation_vector(state, self.shield, self.cfg.signal, self.junction_index)
                t0 = time.perf_counter()
                try:
                    action = int(self.policy.choose(obs.reshape(1, -1), mask.reshape(1, -1))[0])
                    ms = (time.perf_counter() - t0) * 1000.0
                    self.latencies.append(ms)
                    self.last_inference_ms = ms
                    if ms > self.ccfg.inference_deadline_ms:
                        self.shadow_ok = False
                        self._enter_fallback(
                            "inference_deadline",
                            f"inference took {ms:.1f} ms > {self.ccfg.inference_deadline_ms} ms",
                            msg.timestamp,
                            out,
                        )
                    else:
                        self.shadow_ok = True
                        if self.alarm_state == "inference_deadline":
                            self.alarm_state = None
                except Exception as e:
                    self.model_loaded = False
                    self.shadow_ok = False
                    self._enter_fallback("model_failure", f"{type(e).__name__}: {e}", msg.timestamp, out)
            self._maybe_recover(msg.timestamp, out)
            requested = action if (self.controller == "rl" and action is not None) else self._fixed_time_request()
            self.shield.request(requested)
            self.decisions += 1
        state_str = self.shield.tick(dt)
        out.commands.append(self._command(msg.timestamp, requested, state_str))
        out.status = self.status(msg.timestamp)
        return out

    def watchdog(self, now_wall: float | None = None, ts: float | None = None) -> CoreOutputs:
        """Wall-clock check; call periodically. Switches to fixed-time when detectors go stale.

        While data is stale the shield is advanced by the real elapsed wall time (clipped to 5 s), so
        its timers keep their physical meaning whatever the watchdog tick is.
        """
        out = CoreOutputs()
        now_wall = time.monotonic() if now_wall is None else now_wall
        age = now_wall - self.last_wall
        if age > self.ccfg.stale_after_s:
            t = ts if ts is not None else (self.last_msg_time or 0.0)
            self._enter_fallback("stale_detectors", f"no detector data for {age:.0f} s", t, out)
            elapsed = float(np.clip(now_wall - self.last_tick_wall, 0.0, 5.0))
            self.last_tick_wall = now_wall
            self.shield.request(self._fixed_time_request())
            state = self.shield.tick(elapsed if elapsed > 0 else 1.0)
            out.commands.append(self._command(t, None, state))
        return out

    def status(self, ts: float) -> StatusMessage:
        lat = float(np.percentile(self.latencies[-500:], 50)) if self.latencies else 0.0
        return StatusMessage(
            junction_id=self.ccfg.junction_id,
            timestamp=ts,
            controller=self.controller,  # type: ignore[arg-type]
            model_loaded=self.model_loaded,
            last_detector_age_s=round(time.monotonic() - self.last_wall, 2),
            inference_ms_p50=lat,
            decisions=self.decisions,
            fallbacks=self.fallbacks,
            shield_interventions=self.shield.stats.interventions,
            shield_stats=self.shield.stats.as_dict(),
        )
