"""FastAPI sidecar exposing /health and Prometheus /metrics for the controller service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from drs.service.core import ControllerCore


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.decisions = Counter("drs_decisions_total", "Controller decisions", ["controller"], registry=self.registry)
        self.latency = Histogram(
            "drs_inference_latency_seconds",
            "Policy inference latency",
            registry=self.registry,
            buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
        )
        self.fallbacks = Counter("drs_fallbacks_total", "Fallbacks to fixed-time", ["reason"], registry=self.registry)
        self.interventions = Gauge(
            "drs_shield_interventions_total", "Safety shield interventions", registry=self.registry
        )
        self.mode = Gauge("drs_controller_mode", "1 = RL, 0 = fixed-time fallback", registry=self.registry)
        self.detector_age = Gauge(
            "drs_detector_age_seconds", "Age of the last detector message", registry=self.registry
        )


def make_app(core: ControllerCore, metrics: Metrics) -> FastAPI:
    app = FastAPI(title="drs controller sidecar", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        st = core.status(core.last_msg_time or 0.0)
        return {
            "status": "ok" if core.controller == "rl" and core.alarm_state is None else "degraded",
            "alarm": core.alarm_state,
            "controller": st.controller,
            "model_loaded": st.model_loaded,
            "last_detector_age_s": st.last_detector_age_s,
            "decisions": st.decisions,
            "fallbacks": st.fallbacks,
            "shield_interventions": st.shield_interventions,
        }

    @app.get("/metrics")
    def prom() -> PlainTextResponse:
        metrics.interventions.set(core.shield.stats.interventions)
        metrics.mode.set(1 if core.controller == "rl" else 0)
        st = core.status(core.last_msg_time or 0.0)
        metrics.detector_age.set(st.last_detector_age_s)
        return PlainTextResponse(generate_latest(metrics.registry).decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    return app
