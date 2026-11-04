"""Queue forecasting (supervised deep learning): GRU and TCN vs naive baselines.

1. ``collect``: run controllers on training seeds and log per-lane queue (halting vehicles) every
   second; held-out seeds give the test set.
2. Windows of ``history`` seconds of all lanes' queues (plus the signal state one-hot) predict each
   lane's queue 30 s and 60 s ahead.
3. Models: last value, moving average (naive), a small GRU and a Temporal Convolutional Network in
   plain PyTorch (CPU).  MAE / RMSE with bootstrap CIs over held-out episodes go to
   ``results/forecast.json``.

Run ``python -m drs.forecast --config configs/forecast.yaml``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from drs.config import REPO_ROOT, SimConfig, load_yaml
from drs.controllers import FixedTimeController, MaxPressureController
from drs.controllers.base import Controller
from drs.env import TrafficSignalEnv
from drs.runner import run_episode

HORIZONS = (30, 60)


def collect_episode(
    cfg: SimConfig, network: str, scenario: str, seed: int, controller: str, model_dir: str | None
) -> np.ndarray:
    """Return an array [T, n_lanes + n_phases] of per-second lane queues and the phase one-hot."""
    env = TrafficSignalEnv(cfg, network, scenario, [seed], record=True, rng_seed=seed)
    ctrl: Controller
    if controller == "fixed_time":
        ctrl = FixedTimeController(cfg)
    elif controller == "max_pressure":
        ctrl = MaxPressureController()
    else:
        from drs.controllers.ppo import PPOController

        ctrl = PPOController(model_dir or "models/final/single_seed13")
    rec = run_episode(env, ctrl, scenario=scenario, seed=seed)
    jid = next(iter(rec.per_second))
    rows = rec.per_second[jid]
    states = rec.tls_states[jid]
    q = np.array([r["lane_halting"] for r in rows], dtype=np.float32)
    n_phases = cfg.signal.n_phases
    ph = np.zeros((len(rows), n_phases), dtype=np.float32)
    table_states = [p.state for p in env.sim.tables[jid].phases] if env.sim else []
    for i, st in enumerate(states[: len(rows)]):
        if st in table_states:
            ph[i, table_states.index(st)] = 1.0
    return np.concatenate([q, ph], axis=1)


def make_windows(
    series: list[np.ndarray], n_lanes: int, history: int, horizons: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    hmax = max(horizons)
    for s in series:
        for t in range(history, len(s) - hmax, 5):
            xs.append(s[t - history : t])
            ys.append(np.stack([s[t + h - 1, :n_lanes] for h in horizons]))
    return np.stack(xs), np.stack(ys)


class GRUForecaster(nn.Module):
    def __init__(self, n_in: int, n_lanes: int, n_h: int, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_lanes * n_h)
        self.n_lanes, self.n_h = n_lanes, n_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        out: torch.Tensor = self.head(h[-1]).view(-1, self.n_h, self.n_lanes)
        return out


class TCNForecaster(nn.Module):
    """Causal dilated 1-D convolutions with residual connections."""

    def __init__(self, n_in: int, n_lanes: int, n_h: int, channels: int = 48, levels: int = 4, k: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = n_in
        self.pads: list[int] = []
        for i in range(levels):
            d = 2**i
            layers.append(nn.Conv1d(c_in, channels, k, dilation=d))
            self.pads.append((k - 1) * d)
            c_in = channels
        self.convs = nn.ModuleList(layers)
        self.res = nn.ModuleList([nn.Conv1d(n_in if i == 0 else channels, channels, 1) for i in range(levels)])
        self.head = nn.Linear(channels, n_lanes * n_h)
        self.n_lanes, self.n_h = n_lanes, n_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # [B, C, T]
        for conv, res, pad in zip(self.convs, self.res, self.pads):
            y = torch.relu(conv(nn.functional.pad(h, (pad, 0))))
            h = y + res(h)
        out: torch.Tensor = self.head(h[:, :, -1]).view(-1, self.n_h, self.n_lanes)
        return out


def train_model(
    model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, lr: float, seed: int, scale: float
) -> list[float]:
    torch.manual_seed(seed)
    xt = torch.as_tensor(x / scale)
    yt = torch.as_tensor(y / scale)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xt)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        model.train()
        for i in range(0, n, 128):
            b = idx[i : i + 128]
            loss = nn.functional.l1_loss(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            tot += float(loss) * len(b)
        losses.append(tot / n)
    return losses


def predict(model: nn.Module, x: np.ndarray, scale: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = np.concatenate([model(torch.as_tensor(x[i : i + 512] / scale)).numpy() for i in range(0, len(x), 512)])
    return np.asarray(out * scale)


def naive_predictions(x: np.ndarray, n_lanes: int, n_h: int, window: int = 30) -> dict[str, np.ndarray]:
    last = np.repeat(x[:, -1:, :n_lanes], n_h, axis=1)
    ma = np.repeat(x[:, -window:, :n_lanes].mean(axis=1, keepdims=True), n_h, axis=1)
    return {"last_value": last, "moving_average": ma}


def errors_with_ci(pred: np.ndarray, y: np.ndarray, episode_ids: np.ndarray, n_boot: int = 1000) -> dict[str, Any]:
    """MAE/RMSE per horizon with bootstrap CIs over held-out episodes."""
    rng = np.random.default_rng(0)
    eps = np.unique(episode_ids)
    out: dict[str, Any] = {}
    for h in range(y.shape[1]):
        err = pred[:, h] - y[:, h]
        per_ep_mae = np.array([np.abs(err[episode_ids == e]).mean() for e in eps])
        per_ep_mse = np.array([(err[episode_ids == e] ** 2).mean() for e in eps])
        boots_mae = rng.choice(per_ep_mae, size=(n_boot, len(eps))).mean(axis=1)
        boots_rmse = np.sqrt(rng.choice(per_ep_mse, size=(n_boot, len(eps))).mean(axis=1))
        out[f"h{HORIZONS[h]}"] = {
            "mae": float(np.abs(err).mean()),
            "mae_ci": [float(np.percentile(boots_mae, 2.5)), float(np.percentile(boots_mae, 97.5))],
            "rmse": float(np.sqrt((err**2).mean())),
            "rmse_ci": [float(np.percentile(boots_rmse, 2.5)), float(np.percentile(boots_rmse, 97.5))],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/forecast.yaml")
    args = parser.parse_args()
    fc = load_yaml(args.config)
    cfg = SimConfig.load(fc["sim_config"])
    out_dir = REPO_ROOT / str(fc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cache = out_dir / "series.npz"
    network = str(fc["network"])
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        train_series = list(z["train"])
        test_series = list(z["test"])
    else:
        train_series, test_series = [], []
        for split, seeds, store in (
            ("train", cfg.seeds["train"][: int(fc["train_episodes"])], train_series),
            ("test", cfg.seeds["eval"][: int(fc["test_episodes"])], test_series),
        ):
            for i, seed in enumerate(seeds):
                scen = list(fc["scenarios"])[i % len(fc["scenarios"])]
                ctrl = list(fc["controllers"])[i % len(fc["controllers"])]
                store.append(collect_episode(cfg, network, scen, int(seed), ctrl, fc.get("ppo_model")))
                print(f"[collect] {split} seed={seed} {scen} {ctrl} T={len(store[-1])}", flush=True)
        np.savez_compressed(cache, train=np.array(train_series, dtype=object), test=np.array(test_series, dtype=object))
    n_lanes = 12
    history = int(fc.get("history", 60))
    xtr, ytr = make_windows(train_series, n_lanes, history, HORIZONS)
    xte, yte = make_windows(test_series, n_lanes, history, HORIZONS)
    ep_ids = np.concatenate(
        [np.full(len(make_windows([s], n_lanes, history, HORIZONS)[0]), k) for k, s in enumerate(test_series)]
    )
    scale = float(fc.get("scale", 20.0))
    results: dict[str, Any] = {
        "n_train_windows": len(xtr),
        "n_test_windows": len(xte),
        "history_s": history,
        "horizons_s": list(HORIZONS),
        "models": {},
    }
    for name, pred in naive_predictions(xte, n_lanes, len(HORIZONS)).items():
        results["models"][name] = errors_with_ci(pred, yte, ep_ids)
    models: dict[str, nn.Module] = {
        "gru": GRUForecaster(xtr.shape[2], n_lanes, len(HORIZONS)),
        "tcn": TCNForecaster(xtr.shape[2], n_lanes, len(HORIZONS)),
    }
    for name, model in models.items():
        t1 = time.time()
        losses = train_model(model, xtr, ytr, int(fc.get("epochs", 15)), float(fc.get("lr", 1e-3)), 0, scale)
        res = errors_with_ci(predict(model, xte, scale), yte, ep_ids)
        res["train_loss"] = losses
        res["wall_s"] = round(time.time() - t1, 1)
        res["n_params"] = sum(p.numel() for p in model.parameters())
        results["models"][name] = res
        torch.save(model.state_dict(), out_dir / f"{name}.pt")
        print(
            f"[forecast] {name}: " + ", ".join(f"{h}: MAE {r['mae']:.2f}" for h, r in res.items() if h.startswith("h")),
            flush=True,
        )
    results["wall_s"] = round(time.time() - t0, 1)
    results["note"] = (
        "Queue (halting vehicles) per lane forecast from the last 60 s of all lanes + phase; synthetic simulation data."
    )
    (REPO_ROOT / "results" / "forecast.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {h: round(v["mae"], 2) for h, v in m.items() if h.startswith("h")}
                for k, m in results["models"].items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
