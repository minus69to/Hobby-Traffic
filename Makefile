# Dhaka RL Signals - developer entry points
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export LIBSUMO_AS_TRACI=1
export PYTHONPATH := src

.PHONY: setup networks demand speedtest lint test train train-grid eval report replay geometry export service web web-dev media reproduce check-numbers clean

setup:
	uv venv .venv --python 3.13 || python3 -m venv .venv
	.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
	.venv/bin/pip install -e ".[dev]" tensorboard playwright
	cd web && npm ci

networks:
	$(PY) -m drs.netbuild --config configs/sim.yaml

demand:
	$(PY) -m drs.demand --config configs/sim.yaml

speedtest:
	$(PY) scripts/speedtest.py --config configs/sim.yaml

lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
	.venv/bin/mypy

test:
	$(PY) -m pytest tests -m "not slow and not integration" -x

test-all:
	$(PY) -m pytest tests -x

train:
	$(PY) -m drs.train --config configs/train.yaml

train-grid:
	$(PY) -m drs.train --config configs/train_grid.yaml

eval:
	$(PY) -m drs.evaluate --config configs/eval.yaml

report:
	$(PY) scripts/make_figures.py

replay:
	$(PY) scripts/make_replay.py --config configs/eval.yaml

geometry:
	$(PY) scripts/export_geometry.py --config configs/sim.yaml

export:
	$(PY) -m drs.export_onnx --config configs/eval.yaml

service:
	$(PY) -m drs.service.app --config configs/service.yaml

web:
	cd web && npm run build

web-dev:
	cd web && npm run dev

media:
	$(PY) scripts/capture_media.py

check-numbers:
	$(PY) scripts/check_numbers.py

# 15-minute small-budget end-to-end run (networks, demand, short training, mini evaluation)
reproduce:
	$(PY) -m drs.netbuild --config configs/sim.yaml
	$(PY) -m drs.demand --config configs/sim.yaml
	$(PY) -m drs.train --config configs/train_small.yaml
	$(PY) -m drs.evaluate --config configs/eval_small.yaml

clean:
	rm -rf sim/out results/raw logs/*.log
