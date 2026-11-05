"""Forecasting windows, naive baselines and the GRU/TCN forward passes on synthetic series."""

from __future__ import annotations

import numpy as np

from drs.forecast import (
    HORIZONS,
    GRUForecaster,
    TCNForecaster,
    errors_with_ci,
    make_windows,
    naive_predictions,
    predict,
    train_model,
)


def test_windows_and_models() -> None:
    rng = np.random.default_rng(0)
    series = [np.abs(rng.normal(5, 2, size=(600, 16))).astype(np.float32) for _ in range(3)]
    x, y = make_windows(series, 12, 60, HORIZONS)
    assert x.shape[1:] == (60, 16) and y.shape[1:] == (2, 12)
    naive = naive_predictions(x, 12, 2)
    assert naive["last_value"].shape == y.shape
    ep = np.concatenate([np.full(len(make_windows([s], 12, 60, HORIZONS)[0]), k) for k, s in enumerate(series)])
    res = errors_with_ci(naive["last_value"], y, ep, n_boot=50)
    assert set(res) == {"h30", "h60"} and res["h30"]["mae"] >= 0
    for model in (GRUForecaster(16, 12, 2, hidden=8), TCNForecaster(16, 12, 2, channels=8, levels=2)):
        losses = train_model(model, x[:200], y[:200], epochs=2, lr=1e-3, seed=0, scale=20.0)
        assert len(losses) == 2 and np.isfinite(losses).all()
        assert predict(model, x[:10], 20.0).shape == (10, 2, 12)
