#!/usr/bin/env bash
# Pipeline run after a single-junction seed finishes: finalise the checkpoint, export ONNX, make the
# web replays for the first PPO seed, rebuild the site and run the smoke test. Usage: scripts/after_seed.sh 13
set -euo pipefail
SEED=${1:?seed}
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src LIBSUMO_AS_TRACI=1
python scripts/finalize_models.py
python -m drs.export_onnx --model "models/final/single_seed${SEED}" --out results/edge_benchmark.json
if [ "$SEED" = "13" ]; then
  python scripts/make_replay.py --scenarios morning_peak incident --controllers fixed_time ppo_s13
  python scripts/make_figures.py
  (cd web && npm run build && npx playwright test)
fi
echo "AFTER_SEED_${SEED}_DONE"
