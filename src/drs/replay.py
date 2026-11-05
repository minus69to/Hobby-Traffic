"""Compact replay recording for the web visualisation.

Format ``DRSR`` version 1 (little endian), written gzip-compressed as ``<name>.drsr.gz``:

* ``b"DRSR"``, ``uint32 version``, ``uint32 header_len``, UTF-8 JSON header
* ``int16[n_pos * 2]`` delta-encoded x, y in 5 cm units (first sample of each track absolute,
  stored in the header as ``x0, y0``), tracks concatenated in header order
* ``uint8[n_pos]`` heading (0..255 maps to 0..360 degrees, SUMO convention: 0 = north, clockwise)
* ``uint8[n_pos]`` speed in 0.25 m/s units (capped at 63.75 m/s)
* ``uint8[n_frames * n_junctions]`` index into ``header.tls_states`` (unique state strings)

The JSON manifest next to it holds scenario metadata and the per-second aggregate metrics.
Sampling: every ``dt`` seconds (0.5 s by default; the replay run uses a 0.5 s SUMO step); frame ``f``
holds the positions at simulation time ``f * dt`` and the lamp state shown during ``[f*dt, (f+1)*dt)``.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drs.sumo_sim import TYPE_TO_CLASS, SumoSim

MAGIC = b"DRSR"
VERSION = 1
QUANT = 0.05  # metres
SPEED_QUANT = 0.25  # m/s
CLASSES: tuple[str, ...] = ("car", "bus", "bus_dd", "truck", "motorcycle", "cng", "rickshaw", "pedestrian")


@dataclass
class Track:
    id: str
    cls: str
    first: int
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)


class FrameRecorder:
    """Collects vehicle, pedestrian and signal states every `dt` seconds through libsumo."""

    def __init__(self, dt: float, junction_ids: list[str]) -> None:
        self.dt = dt
        self.junction_ids = junction_ids
        self.tracks: dict[str, Track] = {}
        self.tls: list[list[str]] = []
        self.n_frames = 0
        self._subscribed: set[str] = set()
        self._psubscribed: set[str] = set()

    def on_step(self, sim: SumoSim, states: dict[str, str]) -> None:
        t = sim._t
        time = sim.time
        if abs(time / self.dt - round(time / self.dt)) > 1e-6:
            return
        f = self.n_frames
        tc = t.constants
        for vid in t.simulation.getDepartedIDList():
            t.vehicle.subscribe(vid, [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED])
            self._subscribed.add(vid)
        for vid, res in t.vehicle.getAllSubscriptionResults().items():
            tr = self.tracks.get(vid)
            if tr is None:
                try:
                    vtype = t.vehicle.getTypeID(vid)
                except Exception:
                    continue
                cls = vtype if vtype in CLASSES else TYPE_TO_CLASS.get(vtype, "car")
                tr = Track(vid, cls, f)
                self.tracks[vid] = tr
            x, y = res[tc.VAR_POSITION]
            tr.xs.append(x)
            tr.ys.append(y)
            tr.angles.append(res[tc.VAR_ANGLE])
            tr.speeds.append(res[tc.VAR_SPEED])
        for pid in t.person.getIDList():
            if pid not in self._psubscribed:
                t.person.subscribe(pid, [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED])
                self._psubscribed.add(pid)
        for pid, res in t.person.getAllSubscriptionResults().items():
            if tc.VAR_POSITION not in res:
                continue
            x, y = res[tc.VAR_POSITION]
            if x < -1e6:  # not yet departed
                continue
            tr = self.tracks.get(pid)
            if tr is None:
                tr = Track(pid, "pedestrian", f)
                self.tracks[pid] = tr
            while len(tr.xs) < f - tr.first:  # gap (should not happen) -> repeat the last sample
                tr.xs.append(tr.xs[-1])
                tr.ys.append(tr.ys[-1])
                tr.angles.append(tr.angles[-1])
                tr.speeds.append(0.0)
            tr.xs.append(x)
            tr.ys.append(y)
            tr.angles.append(res[tc.VAR_ANGLE])
            tr.speeds.append(res[tc.VAR_SPEED])
        self.tls.append([states[j] for j in self.junction_ids])
        self.n_frames += 1


def encode(rec: FrameRecorder, meta: dict[str, Any]) -> bytes:
    """Serialise a recorder into the DRSR binary (uncompressed)."""
    tracks = [tr for tr in rec.tracks.values() if len(tr.xs) > 0]
    tracks.sort(key=lambda tr: (tr.first, tr.id))
    header_tracks = []
    pos_chunks: list[np.ndarray] = []
    ang_chunks: list[np.ndarray] = []
    spd_chunks: list[np.ndarray] = []
    for tr in tracks:
        q = np.round(np.column_stack([tr.xs, tr.ys]) / QUANT).astype(np.int64)
        deltas = np.diff(q, axis=0, prepend=q[:1])
        deltas[0] = 0
        deltas = np.clip(deltas, -32768, 32767).astype("<i2")
        pos_chunks.append(deltas.reshape(-1))
        ang_chunks.append((np.round(np.mod(np.asarray(tr.angles), 360.0) / 360.0 * 256.0) % 256).astype(np.uint8))
        spd_chunks.append(np.clip(np.round(np.asarray(tr.speeds) / SPEED_QUANT), 0, 255).astype(np.uint8))
        header_tracks.append(
            {
                "id": tr.id,
                "cls": CLASSES.index(tr.cls),
                "first": tr.first,
                "n": len(tr.xs),
                "x0": int(q[0, 0]),
                "y0": int(q[0, 1]),
            }
        )
    uniq: list[str] = []
    idx: dict[str, int] = {}
    tls_idx = np.zeros((rec.n_frames, len(rec.junction_ids)), dtype=np.uint8)
    for f, row in enumerate(rec.tls):
        for j, s in enumerate(row):
            if s not in idx:
                idx[s] = len(uniq)
                uniq.append(s)
            tls_idx[f, j] = idx[s]
    header = {
        **meta,
        "dt": rec.dt,
        "n_frames": rec.n_frames,
        "quant": QUANT,
        "speed_quant": SPEED_QUANT,
        "classes": list(CLASSES),
        "junctions": rec.junction_ids,
        "tls_states": uniq,
        "tracks": header_tracks,
        "n_pos": int(sum(len(tr.xs) for tr in tracks)),
    }
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = b"".join(
        [
            MAGIC,
            struct.pack("<II", VERSION, len(hb)),
            hb,
            (np.concatenate(pos_chunks) if pos_chunks else np.zeros(0, "<i2")).tobytes(),
            (np.concatenate(ang_chunks) if ang_chunks else np.zeros(0, np.uint8)).tobytes(),
            (np.concatenate(spd_chunks) if spd_chunks else np.zeros(0, np.uint8)).tobytes(),
            tls_idx.tobytes(),
        ]
    )
    return body


def decode(blob: bytes) -> dict[str, Any]:
    """Inverse of :func:`encode` (used by tests and `scripts/check_numbers.py`)."""
    assert blob[:4] == MAGIC, "bad magic"
    version, hlen = struct.unpack("<II", blob[4:12])
    assert version == VERSION
    header = json.loads(blob[12 : 12 + hlen].decode("utf-8"))
    off = 12 + hlen
    n_pos = header["n_pos"]
    pos = np.frombuffer(blob, "<i2", count=n_pos * 2, offset=off).reshape(-1, 2)
    off += n_pos * 4
    ang = np.frombuffer(blob, np.uint8, count=n_pos, offset=off)
    off += n_pos
    spd = np.frombuffer(blob, np.uint8, count=n_pos, offset=off)
    off += n_pos
    nf, nj = header["n_frames"], len(header["junctions"])
    tls = np.frombuffer(blob, np.uint8, count=nf * nj, offset=off).reshape(nf, nj)
    tracks = []
    cursor = 0
    for tr in header["tracks"]:
        n = tr["n"]
        d = pos[cursor : cursor + n].astype(np.int64)
        d[0] = [tr["x0"], tr["y0"]]
        xy = np.cumsum(d, axis=0) * header["quant"]
        tracks.append(
            {
                "id": tr["id"],
                "cls": header["classes"][tr["cls"]],
                "first": tr["first"],
                "xy": xy,
                "angle": ang[cursor : cursor + n].astype(float) * 360.0 / 256.0,
                "speed": spd[cursor : cursor + n].astype(float) * header["speed_quant"],
            }
        )
        cursor += n
    return {"header": header, "tracks": tracks, "tls": [[header["tls_states"][k] for k in row] for row in tls]}


def write_replay(rec: FrameRecorder, meta: dict[str, Any], path: Path) -> int:
    blob = encode(rec, meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as fh:
        fh.write(blob)
    return path.stat().st_size


def read_replay(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as fh:
        return decode(fh.read())
