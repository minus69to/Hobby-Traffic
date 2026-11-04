# ENVIRONMENT

Recorded on 2026-11-04 in the development VM (CPU only).

| Item | Value |
|---|---|
| CPU / RAM | 4 vCPU, 16 GB RAM, no GPU |
| OS | Linux 6.18 (Debian-based container) |
| Python | 3.13.18 (virtualenv in `.venv`, created with `uv`) |
| SUMO | eclipse-sumo 1.28.0, libsumo 1.28.0, traci 1.28.0, sumolib 1.28.0 (pip) |
| RL | stable-baselines3 2.9.2, sb3-contrib 2.9.2, gymnasium 1.4.0 |
| torch | 2.15.0 (the CUDA wheel was installed because the CPU-only index was unreachable through the VM proxy; it runs on CPU) |
| ONNX | onnx 1.23.2, onnxruntime 1.31.1 |
| Testing / lint | pytest, hypothesis 6.170, ruff 0.17.3, mypy 2.4.1 |
| Service | fastapi 0.162, uvicorn, aiomqtt 2.5.1, paho-mqtt 2.1, pydantic 2.13, prometheus-client |
| Node | 22.23.4, npm 10.9.9 |
| ffmpeg | system package |
| mosquitto | apt package (local broker for tests); Docker daemon not available in this VM |

## Simulation speed test (M0)

`python scripts/speedtest.py` runs one hour of simulated morning-peak traffic on the `single`
network with the sublane model (`--lateral-resolution 0.8`), pedestrians (striping model) and
per-second metric recording under the fixed-time controller. Result is in `results/speedtest.json`:

- ~100 simulation steps per second per process (1 s step length), i.e. ~20 decision steps/s.
- Without the sublane model SUMO is ~8x faster; the sublane model is kept because lane-sharing and
  weaving is the defining feature of the traffic being modelled.

Training budget derived from this: 300 000 environment steps per seed (1.5 M simulation steps),
4 worker processes per seed, about 1 hour wall-clock per seed on 4 cores.
