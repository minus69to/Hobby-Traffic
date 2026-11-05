#!/usr/bin/env bash
# Final chain after the evaluation: figures, training summary, docs, website build, number check.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src LIBSUMO_AS_TRACI=1
python scripts/make_figures.py --models models/single
python scripts/collect_imitation.py
python scripts/render_docs.py
(cd web && npm run build)
python scripts/check_numbers.py
echo FINISH_DONE
