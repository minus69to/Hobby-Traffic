"""Behaviour cloning on synthetic data and ONNX export parity on the small model."""

from __future__ import annotations

import numpy as np
import pytest

from drs.config import REPO_ROOT
from drs.imitation import behaviour_clone

SMALL = REPO_ROOT / "models" / "small" / "seed13"


def test_behaviour_clone_learns_a_masked_rule() -> None:
    rng = np.random.default_rng(0)
    n, d, a = 2000, 66, 4
    obs = rng.random((n, d)).astype(np.float32)
    masks = rng.random((n, a)) > 0.3
    masks[np.arange(n), rng.integers(0, a, n)] = True
    # teacher: pick the valid action whose "feature" obs[:, k] is largest
    scores = np.where(masks, obs[:, :a], -1.0)
    actions = scores.argmax(axis=1)
    res = behaviour_clone({"obs": obs, "masks": masks, "actions": actions}, [64, 64], epochs=25, lr=3e-3, seed=0)
    assert res["history"][-1]["val_acc"] > 0.8
    assert set(res["state_dict"]) >= {"policy_net.0.weight", "action_net.weight"}


@pytest.mark.skipif(not (SMALL / "best" / "policy.zip").exists(), reason="small model not trained")
def test_onnx_export_matches_torch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from drs.export_onnx import benchmark, export

    exp = export(SMALL, tmp_path)
    bench = benchmark(exp, n_obs=500, n_calls=200)
    assert bench["float"]["agreement_vs_torch"] > 0.999
    assert bench["int8"]["agreement_vs_float"] > 0.9
    assert (tmp_path / "policy_int8.onnx").stat().st_size < (tmp_path / "policy.onnx").stat().st_size
