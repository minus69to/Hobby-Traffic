"""Parse SUMO tripinfo output and compute the evaluation metrics of one episode."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from drs.env import EpisodeRecord
from drs.sumo_sim import TYPE_TO_CLASS, VEHICLE_CLASSES

METRIC_KEYS: tuple[str, ...] = (
    "mean_travel_time",
    "p95_travel_time",
    "mean_waiting_time",
    "mean_time_loss",
    "mean_queue_length",
    "throughput",
    "ped_mean_wait",
    "shield_interventions",
    "phase_switches_per_hour",
    *(f"wait_{c}" for c in VEHICLE_CLASSES),
)


def parse_tripinfo(path: Path, warmup: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (vehicle trips, pedestrian trips) that departed after `warmup`."""
    vehicles: list[dict[str, Any]] = []
    persons: list[dict[str, Any]] = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "tripinfo":
            depart = float(el.get("depart", 0))
            if depart >= warmup:
                vehicles.append(
                    {
                        "id": el.get("id"),
                        "type": el.get("vType"),
                        "duration": float(el.get("duration", 0)),
                        "waiting": float(el.get("waitingTime", 0)),
                        "time_loss": float(el.get("timeLoss", 0)),
                        "arrived": float(el.get("arrival", -1)) >= 0,
                    }
                )
            el.clear()
        elif el.tag == "personinfo":
            depart = float(el.get("depart", 0))
            if depart >= warmup:
                walk = el.find("walk")
                persons.append(
                    {
                        "id": el.get("id"),
                        "waiting": float((walk if walk is not None else el).get("waitingTime", 0)),
                        "time_loss": float((walk if walk is not None else el).get("timeLoss", 0)),
                    }
                )
            el.clear()
    return vehicles, persons


def episode_metrics(rec: EpisodeRecord, warmup: float, episode_seconds: float) -> dict[str, float]:
    """Metrics of one episode from the tripinfo file plus the recorded per-second data."""
    assert rec.tripinfo is not None, "episode was run without tripinfo output"
    vehicles, persons = parse_tripinfo(rec.tripinfo, warmup)
    vehicles = [v for v in vehicles if v["id"] != "incident_truck"]
    durations = np.array([v["duration"] for v in vehicles]) if vehicles else np.zeros(1)
    waits = np.array([v["waiting"] for v in vehicles]) if vehicles else np.zeros(1)
    losses = np.array([v["time_loss"] for v in vehicles]) if vehicles else np.zeros(1)
    out: dict[str, float] = {
        "mean_travel_time": float(durations.mean()),
        "p95_travel_time": float(np.percentile(durations, 95)),
        "mean_waiting_time": float(waits.mean()),
        "mean_time_loss": float(losses.mean()),
        "throughput": float(sum(1 for v in vehicles if v["arrived"])),
        "ped_mean_wait": float(np.mean([p["waiting"] for p in persons])) if persons else 0.0,
        "n_vehicles": float(len(vehicles)),
        "n_pedestrians": float(len(persons)),
    }
    for c in VEHICLE_CLASSES:
        ws = [v["waiting"] for v in vehicles if TYPE_TO_CLASS.get(str(v["type"])) == c]
        out[f"wait_{c}"] = float(np.mean(ws)) if ws else 0.0
    queues = []
    for rows in rec.per_second.values():
        queues.append(np.mean([r["vehicles_waiting"] for r in rows if r["t"] > warmup]) if rows else 0.0)
    out["mean_queue_length"] = float(np.mean(queues)) if queues else 0.0
    out["shield_interventions"] = float(sum(s.get("interventions", 0) for s in rec.shield_stats.values()))
    switches = sum(s.get("switches", 0) for s in rec.shield_stats.values())
    hours = max(episode_seconds, 1.0) / 3600.0
    out["phase_switches_per_hour"] = float(switches / hours / max(len(rec.shield_stats), 1))
    return out
