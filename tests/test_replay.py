"""Schema and round-trip tests for the DRSR replay format and the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drs.config import REPO_ROOT
from drs.replay import CLASSES, QUANT, FrameRecorder, Track, decode, encode

REPLAY_DIR = REPO_ROOT / "web" / "public" / "data" / "replays"


def _synthetic_recorder() -> FrameRecorder:
    rec = FrameRecorder(0.5, ["C"])
    rng = np.random.default_rng(1)
    for k in range(5):
        tr = Track(f"v{k}", CLASSES[k % len(CLASSES)], first=k)
        x, y = rng.uniform(0, 600, 2)
        for _ in range(20 + k):
            x += rng.uniform(-5, 5)
            y += rng.uniform(-5, 5)
            tr.xs.append(x)
            tr.ys.append(y)
            tr.angles.append(float(rng.uniform(0, 360)))
            tr.speeds.append(float(rng.uniform(0, 14)))
        rec.tracks[tr.id] = tr
    rec.n_frames = 30
    rec.tls = [["gGGGrrrr" if f % 2 else "rrrryyyy"] for f in range(30)]
    return rec


def test_round_trip_within_quantisation() -> None:
    rec = _synthetic_recorder()
    blob = encode(rec, {"scenario": "test"})
    out = decode(blob)
    assert out["header"]["n_frames"] == 30
    assert len(out["tracks"]) == 5
    for tr in out["tracks"]:
        src = rec.tracks[tr["id"]]
        assert tr["first"] == src.first
        assert np.allclose(tr["xy"][:, 0], src.xs, atol=QUANT / 2 + 1e-9)
        assert np.allclose(tr["xy"][:, 1], src.ys, atol=QUANT / 2 + 1e-9)
        assert np.allclose(tr["speed"], src.speeds, atol=0.25)
        diff = np.abs((tr["angle"] - np.asarray(src.angles) + 180) % 360 - 180)
        assert diff.max() <= 360 / 256 + 1e-9
    assert out["tls"][1] == ["gGGGrrrr"]
    assert out["tls"][0] == ["rrrryyyy"]


def test_header_schema() -> None:
    rec = _synthetic_recorder()
    out = decode(encode(rec, {"scenario": "s", "controller": "c"}))
    h = out["header"]
    for key in ("dt", "n_frames", "quant", "speed_quant", "classes", "junctions", "tls_states", "tracks", "n_pos"):
        assert key in h
    assert all(set(t) == {"id", "cls", "first", "n", "x0", "y0"} for t in h["tracks"])
    assert h["n_pos"] == sum(t["n"] for t in h["tracks"])


@pytest.mark.skipif(not REPLAY_DIR.exists() or not list(REPLAY_DIR.glob("*.json")), reason="no exported replays")
def test_exported_manifests_are_consistent() -> None:
    for manifest in REPLAY_DIR.glob("*.json"):
        if manifest.name == "index.json":
            continue
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("network", "scenario", "controller", "seed", "file", "n_frames", "metrics", "per_second", "shield_stats"):
            assert key in m, (manifest.name, key)
        assert (REPLAY_DIR / m["file"]).exists()
        assert m["size_bytes"] < 10_000_000
        for jid, rows in m["per_second"].items():
            assert rows[0]["t"] >= 1 and all("queue" in r and "avg_wait" in r for r in rows)
        assert "shield_interventions" in m["metrics"]
