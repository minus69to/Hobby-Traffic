"""Controllers backed by a trained MaskablePPO policy (PyTorch) or its ONNX export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from drs.config import REPO_ROOT
from drs.shield import SafetyShield
from drs.sumo_sim import JunctionState


class PPOController:
    """Deterministic action selection with the trained policy and its saved observation statistics."""

    name = "ppo"

    def __init__(self, model_dir: str | Path, name: str | None = None) -> None:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import VecNormalize

        d = Path(model_dir)
        if not d.is_absolute():
            d = REPO_ROOT / d
        if not (d / "policy.zip").exists() and (d / "best" / "policy.zip").exists():
            d = d / "best"
        self.model = MaskablePPO.load(str(d / "policy.zip"), device="cpu")
        import pickle

        with (d / "vecnormalize.pkl").open("rb") as fh:
            vec: VecNormalize = pickle.load(fh)  # noqa: S301 - our own file
        self.obs_mean = vec.obs_rms.mean.astype(np.float32)
        self.obs_var = vec.obs_rms.var.astype(np.float32)
        self.clip = float(vec.clip_obs)
        self.eps = float(vec.epsilon)
        if name:
            self.name = name

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + self.eps), -self.clip, self.clip).astype(np.float32)

    def reset(self, env: Any) -> None:
        return None

    def act(
        self, states: dict[str, JunctionState], obs: np.ndarray, masks: np.ndarray, shields: dict[str, SafetyShield]
    ) -> list[int]:
        actions, _ = self.model.predict(self.normalise(obs), action_masks=masks, deterministic=True)
        return [int(a) for a in np.atleast_1d(actions)]


class OnnxController:
    """Same decision rule with the exported (optionally int8) ONNX graph; used by the edge service."""

    name = "onnx"

    def __init__(self, onnx_path: str | Path, meta_path: str | Path, name: str | None = None) -> None:
        import onnxruntime as ort

        p = Path(onnx_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        m = Path(meta_path)
        if not m.is_absolute():
            m = REPO_ROOT / m
        meta = json.loads(m.read_text(encoding="utf-8"))
        self.obs_mean = np.asarray(meta["obs_mean"], dtype=np.float32)
        self.obs_var = np.asarray(meta["obs_var"], dtype=np.float32)
        self.clip = float(meta["clip_obs"])
        self.eps = float(meta["epsilon"])
        self.normalise_in_graph = bool(meta.get("normalisation_folded", True))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        if name:
            self.name = name

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        if not self.normalise_in_graph:
            x = np.clip((x - self.obs_mean) / np.sqrt(self.obs_var + self.eps), -self.clip, self.clip)
        out = self.session.run(None, {self.input_name: x.reshape(-1, x.shape[-1])})[0]
        return np.asarray(out)

    def choose(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        lg = self.logits(obs)
        lg = np.where(masks.reshape(lg.shape), lg, -1e9)
        return np.argmax(lg, axis=-1)

    def reset(self, env: Any) -> None:
        return None

    def act(
        self, states: dict[str, JunctionState], obs: np.ndarray, masks: np.ndarray, shields: dict[str, SafetyShield]
    ) -> list[int]:
        return [int(a) for a in self.choose(obs, masks)]
