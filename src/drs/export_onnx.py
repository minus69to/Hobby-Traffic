"""Export the trained MaskablePPO policy to ONNX (opset 17), quantise to int8 and benchmark.

Run ``python -m drs.export_onnx --config configs/eval.yaml [--model models/final/single_seed13]``.
Writes ``policy.onnx``, ``policy_int8.onnx`` and ``onnx_meta.json`` next to the model and the
benchmark numbers to ``results/edge_benchmark.json``.  Observation normalisation (VecNormalize
statistics) is folded into the graph, so the service feeds raw observations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drs.config import REPO_ROOT, load_yaml

OPSET = 17


class PolicyWithNorm(nn.Module):
    """Raw observation -> normalised -> policy MLP -> action logits."""

    def __init__(self, policy: Any, mean: np.ndarray, var: np.ndarray, clip: float, eps: float) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.sqrt(torch.as_tensor(var, dtype=torch.float32) + eps), requires_grad=False)
        self.clip = clip
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        out: torch.Tensor = self.action_net(self.policy_net(x))
        return out


def load_model(model_dir: Path) -> tuple[Any, np.ndarray, np.ndarray, float, float]:
    from sb3_contrib import MaskablePPO

    if not (model_dir / "policy.zip").exists() and (model_dir / "best" / "policy.zip").exists():
        model_dir = model_dir / "best"
    model = MaskablePPO.load(str(model_dir / "policy.zip"), device="cpu")
    with (model_dir / "vecnormalize.pkl").open("rb") as fh:
        vec = pickle.load(fh)
    return model, vec.obs_rms.mean, vec.obs_rms.var, float(vec.clip_obs), float(vec.epsilon)


def sample_observations(n: int, mean: np.ndarray, var: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random observations around the training distribution (clipped to the observation range)."""
    std = np.sqrt(var + 1e-8)
    x = mean + rng.normal(size=(n, len(mean))) * std * 1.5
    return np.clip(x, 0.0, 2.0).astype(np.float32)


def export(model_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model, mean, var, clip, eps = load_model(model_dir)
    wrapper = PolicyWithNorm(model.policy, mean, var, clip, eps).eval()
    obs_dim = int(mean.shape[0])
    dummy = torch.zeros(1, obs_dim)
    onnx_path = out_dir / "policy.onnx"
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            opset_version=OPSET,
            input_names=["obs"],
            output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            opset_version=OPSET,
            input_names=["obs"],
            output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        )
    import onnx

    onnx.checker.check_model(str(onnx_path))
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "policy_int8.onnx"
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
    meta = {
        "obs_dim": obs_dim,
        "n_actions": int(model.action_space.n),
        "obs_mean": mean.tolist(),
        "obs_var": var.tolist(),
        "clip_obs": clip,
        "epsilon": eps,
        "normalisation_folded": True,
        "opset": OPSET,
        "source_model": str(model_dir.relative_to(REPO_ROOT))
        if model_dir.is_relative_to(REPO_ROOT)
        else str(model_dir),
    }
    (out_dir / "onnx_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {
        "model": model,
        "wrapper": wrapper,
        "meta": meta,
        "onnx": onnx_path,
        "int8": int8_path,
        "mean": mean,
        "var": var,
    }


def benchmark(exp: dict[str, Any], n_obs: int = 10_000, n_calls: int = 10_000, seed: int = 0) -> dict[str, Any]:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    obs = sample_observations(n_obs, exp["mean"], exp["var"], rng)
    masks = rng.random((n_obs, exp["meta"]["n_actions"])) > 0.3
    masks[np.arange(n_obs), rng.integers(0, exp["meta"]["n_actions"], n_obs)] = True
    with torch.no_grad():
        torch_logits = exp["wrapper"](torch.as_tensor(obs)).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    out: dict[str, Any] = {}
    sessions = {
        "float": ort.InferenceSession(str(exp["onnx"]), so, providers=["CPUExecutionProvider"]),
        "int8": ort.InferenceSession(str(exp["int8"]), so, providers=["CPUExecutionProvider"]),
    }
    logits = {}
    for name, sess in sessions.items():
        logits[name] = np.concatenate([sess.run(None, {"obs": obs[i : i + 512]})[0] for i in range(0, n_obs, 512)])

    def masked_argmax(lg: np.ndarray) -> np.ndarray:
        return np.argmax(np.where(masks, lg, -1e9), axis=1)

    def agreement(a: np.ndarray, b: np.ndarray) -> float:
        return float((masked_argmax(a) == masked_argmax(b)).mean())

    for name, sess in sessions.items():
        single = obs[:1]
        for _ in range(200):
            sess.run(None, {"obs": single})
        lat = np.empty(n_calls)
        for i in range(n_calls):
            x = obs[i % n_obs : i % n_obs + 1]
            t0 = time.perf_counter_ns()
            sess.run(None, {"obs": x})
            lat[i] = (time.perf_counter_ns() - t0) / 1e6
        out[name] = {
            "size_bytes": Path(exp["onnx"] if name == "float" else exp["int8"]).stat().st_size,
            "agreement_vs_torch": agreement(logits[name], torch_logits),
            "agreement_vs_float": agreement(logits[name], logits["float"]),
            "max_abs_logit_diff_vs_torch": float(np.abs(logits[name] - torch_logits).max()),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_mean_ms": float(lat.mean()),
            "n_calls": n_calls,
        }
    out.update(
        {
            "model": exp["meta"]["source_model"],
            "opset": OPSET,
            "obs_dim": exp["meta"]["obs_dim"],
            "n_actions": exp["meta"]["n_actions"],
            "n_random_obs": n_obs,
            "threads": 1,
            "cpu": platform.processor() or platform.machine(),
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "note": (
                "Single-thread CPU latency in the development VM approximates a small edge CPU; "
                "it is not a measurement on a Jetson or other target device."
            ),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--model", default=None, help="model directory (default: first ppo controller in the eval config)"
    )
    parser.add_argument("--out", default="results/edge_benchmark.json")
    parser.add_argument("--n", type=int, default=10_000)
    args = parser.parse_args()
    model_dir = args.model
    if model_dir is None:
        ecfg = load_yaml(args.config)
        model_dir = next(spec["model"] for spec in ecfg["controllers"].values() if spec["type"] == "ppo")
    exp = export(REPO_ROOT / model_dir)
    bench = benchmark(exp, n_obs=args.n, n_calls=args.n)
    (REPO_ROOT / args.out).write_text(json.dumps(bench, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in bench.items() if k in ("float", "int8")}, indent=1))


if __name__ == "__main__":
    main()
