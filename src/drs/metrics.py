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
    "mean_depart_delay",
    "unfinished_share",
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
            depart_delay = float(el.get("departDelay", 0))
            scheduled = depart - depart_delay if depart >= 0 else -1.0
            if scheduled >= warmup or (depart < 0 and float(el.get("departDelay", -1)) >= warmup):
                vehicles.append(
                    {
                        "id": el.get("id"),
                        "type": el.get("vType"),
                        "duration": float(el.get("duration", 0)),
                        "waiting": float(el.get("waitingTime", 0)),
                        "time_loss": float(el.get("timeLoss", 0)),
                        "depart_delay": max(depart_delay, 0.0),
                        "departed": depart >= 0,
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
    """Metrics of one episode from the tripinfo file plus the recorded per-second data.

    Travel/waiting-time means use completed trips only (unfinished trips would enter with truncated
    values); the share of unfinished or never-inserted vehicles is reported separately, and the time
    vehicles spent queued before insertion (``departDelay``) is reported as ``mean_depart_delay``.
    """
    assert rec.tripinfo is not None, "episode was run without tripinfo output"
    vehicles, persons = parse_tripinfo(rec.tripinfo, warmup)
    vehicles = [v for v in vehicles if v["id"] != "incident_truck"]
    finished = [v for v in vehicles if v["arrived"]]
    durations = np.array([v["duration"] for v in finished]) if finished else np.array([np.nan])
    waits = np.array([v["waiting"] for v in finished]) if finished else np.array([np.nan])
    losses = np.array([v["time_loss"] for v in finished]) if finished else np.array([np.nan])
    hours = max(episode_seconds - warmup, 1.0) / 3600.0
    out: dict[str, float] = {
        "mean_travel_time": float(np.nanmean(durations)),
        "p95_travel_time": float(np.nanpercentile(durations, 95)),
        "mean_waiting_time": float(np.nanmean(waits)),
        "mean_time_loss": float(np.nanmean(losses)),
        "mean_depart_delay": float(np.mean([v["depart_delay"] for v in vehicles])) if vehicles else float("nan"),
        "unfinished_share": float(1.0 - len(finished) / len(vehicles)) if vehicles else float("nan"),
        "throughput": float(len(finished) / hours),
        "ped_mean_wait": float(np.mean([p["waiting"] for p in persons])) if persons else float("nan"),
        "n_vehicles": float(len(vehicles)),
        "n_finished": float(len(finished)),
        "n_pedestrians": float(len(persons)),
    }
    for c in VEHICLE_CLASSES:
        ws = [v["waiting"] for v in finished if TYPE_TO_CLASS.get(str(v["type"])) == c]
        out[f"wait_{c}"] = float(np.mean(ws)) if ws else float("nan")
    queues = []
    for rows in rec.per_second.values():
        vals = [r["vehicles_waiting"] for r in rows if r["t"] > warmup]
        queues.append(float(np.mean(vals)) if vals else float("nan"))
    out["mean_queue_length"] = float(np.nanmean(queues)) if queues and not all(np.isnan(queues)) else float("nan")
    out["shield_interventions"] = float(sum(s.get("interventions", 0) for s in rec.shield_stats.values()))
    switches = sum(s.get("switches", 0) for s in rec.shield_stats.values())
    out["phase_switches_per_hour"] = float(
        switches / (max(episode_seconds, 1.0) / 3600.0) / max(len(rec.shield_stats), 1)
    )
    return out
