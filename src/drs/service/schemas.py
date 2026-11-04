"""Versioned JSON message schemas exchanged over MQTT (pydantic v2).

Topics (``<id>`` is the junction id):

* ``drs/junction/<id>/detectors`` -> :class:`DetectorMessage` (from the field / simulator bridge)
* ``drs/junction/<id>/command``   -> :class:`CommandMessage` (from the controller service)
* ``drs/junction/<id>/status``    -> :class:`StatusMessage`
* ``drs/junction/<id>/alarm``     -> :class:`AlarmMessage`

These schemas are original to this project and are not modelled on any real deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION: Literal["1.0"] = "1.0"


class LaneReading(BaseModel):
    lane_id: str
    halting: float = Field(ge=0, description="vehicles with speed < 0.1 m/s")
    max_wait: float = Field(ge=0, description="longest consecutive wait on the lane, s")
    count: float = Field(ge=0, description="vehicles on the lane")
    class_counts: dict[str, float] = Field(default_factory=dict, description="vehicles per class on the lane")
    out_halting: float = Field(default=0.0, ge=0, description="halting vehicles on the downstream lane (for pressure)")


class CrossingReading(BaseModel):
    crossing_id: str
    waiting: float = Field(ge=0, description="pedestrians waiting at this corner")


class DetectorMessage(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    junction_id: str
    timestamp: float = Field(description="seconds (simulation time or unix time)")
    lanes: list[LaneReading]
    crossings: list[CrossingReading] = Field(default_factory=list)
    neighbour_pressure: dict[str, float] = Field(
        default_factory=dict, description="pressure of the neighbouring junction per arm (N/E/S/W)"
    )


class CommandMessage(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    junction_id: str
    timestamp: float
    phase: int = Field(ge=0, description="phase currently shown or being transitioned to")
    requested_phase: int | None = Field(default=None, description="what the policy asked for (None = keep)")
    state: str = Field(description="SUMO-style red/yellow/green string the shield produced for the next second")
    shield_mode: Literal["green", "yellow", "all_red"]
    controller: Literal["rl", "fixed_time"]
    decision_id: int


class StatusMessage(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    junction_id: str
    timestamp: float
    controller: Literal["rl", "fixed_time"]
    model_loaded: bool
    last_detector_age_s: float
    inference_ms_p50: float
    decisions: int
    fallbacks: int
    shield_interventions: int
    shield_stats: dict[str, int]


class AlarmMessage(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    junction_id: str
    timestamp: float
    kind: Literal["model_failure", "inference_deadline", "stale_detectors", "recovered"]
    detail: str
