"""Imitation-learning warm start (behaviour cloning from the max-pressure controller).

1. ``collect``: roll out a controller on training seeds and store (observation, mask, action) tuples.
2. ``behaviour_clone``: train the policy MLP with a masked cross-entropy loss.
3. ``drs.train`` with ``bc_init: <file>`` loads the cloned weights (and the dataset's observation
   statistics into VecNormalize) before PPO fine-tuning, so learning curves of PPO from scratch and
   BC+PPO can be compared on the same budget.

Run ``python -m drs.imitation --config configs/train_bc.yaml``.
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


def collect(
    cfg: SimConfig,
    network: str,
    scenarios: dict[str, float],
    seeds: list[int],
    n_episodes: int,
    teacher: str,
    episode_seconds: int | None,
) -> dict[str, np.ndarray]:
    ctrl: Controller = MaxPressureController() if teacher == "max_pressure" else FixedTimeController(cfg)
    env = TrafficSignalEnv(cfg, network, scenarios, seeds, episode_seconds=episode_seconds, rng_seed=7)
    obs_l, mask_l, act_l = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        ctrl.reset(env)
        done = False
        while not done:
            masks = env.action_masks()
            actions = ctrl.act(env.last_states, obs, masks, env.shields)
            obs_l.append(obs.copy())
            mask_l.append(masks.copy())
            act_l.append(np.asarray(actions))
            obs, _, done, _ = env.step(actions)
        env.finish()
    env.close()
    return {"obs": np.concatenate(obs_l), "masks": np.concatenate(mask_l), "actions": np.concatenate(act_l)}


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, layers: list[int], n_actions: int) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = obs_dim
        for h in layers:
            mods += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.policy_net = nn.Sequential(*mods)
        self.action_net = nn.Linear(d, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.action_net(self.policy_net(x))
        return out


def behaviour_clone(
    data: dict[str, np.ndarray],
    layers: list[int],
    epochs: int,
    lr: float,
    seed: int,
    clip: float = 5.0,
    eps: float = 1e-8,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    obs = data["obs"].astype(np.float32)
    mean = obs.mean(axis=0)
    var = obs.var(axis=0) + eps
    x = torch.as_tensor(np.clip((obs - mean) / np.sqrt(var), -clip, clip))
    masks = torch.as_tensor(data["masks"].astype(bool))
    y = torch.as_tensor(data["actions"].astype(np.int64))
    n_actions = int(masks.shape[1])
    model = PolicyMLP(x.shape[1], layers, n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    val = idx[: max(1, n // 10)]
    tr = idx[max(1, n // 10) :]
    history = []
    for ep in range(epochs):
        rng.shuffle(tr)
        model.train()
        tot = 0.0
        for i in range(0, len(tr), 256):
            b = tr[i : i + 256]
            logits = model(x[b]).masked_fill(~masks[b], -1e9)
            loss = nn.functional.cross_entropy(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            vl = model(x[val]).masked_fill(~masks[val], -1e9)
            acc = float((vl.argmax(1) == y[val]).float().mean())
        history.append({"epoch": ep, "train_loss": tot / len(tr), "val_acc": acc})
    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "obs_mean": mean,
        "obs_var": var,
        "history": history,
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_bc.yaml")
    args = parser.parse_args()
    tcfg = load_yaml(args.config)
    cfg = SimConfig.load(tcfg["sim_config"])
    bc = tcfg["bc"]
    out_dir = REPO_ROOT / str(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = collect(
        cfg,
        str(tcfg["network"]),
        dict(bc["scenarios"]),
        [int(s) for s in cfg.seeds["train"]],
        int(bc["episodes"]),
        str(bc.get("teacher", "max_pressure")),
        tcfg.get("episode_seconds"),
    )
    np.savez_compressed(out_dir / "bc_dataset.npz", **data)
    res = behaviour_clone(
        data, list(tcfg["ppo"]["policy_layers"]), int(bc["epochs"]), float(bc["lr"]), int(tcfg["seeds"][0])
    )
    torch.save(
        {"state_dict": res["state_dict"], "obs_mean": res["obs_mean"], "obs_var": res["obs_var"]},
        out_dir / "bc_policy.pt",
    )
    meta = {
        "n_samples": res["n"],
        "history": res["history"],
        "wall_s": round(time.time() - t0, 1),
        "teacher": bc.get("teacher", "max_pressure"),
    }
    (out_dir / "bc_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"BC: {res['n']} samples, final val acc {res['history'][-1]['val_acc']:.3f}, {meta['wall_s']} s")


if __name__ == "__main__":
    main()
