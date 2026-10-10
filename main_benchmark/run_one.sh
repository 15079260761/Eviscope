#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: run_one.sh METHOD DATASET PHYSICAL_GPU RUN_ID" >&2
  exit 2
fi
METHOD="$1"; DATASET="$2"; PHYSICAL_GPU="$3"; RUN_ID="$4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
EXP_ROOT="$PROJECT_ROOT/main_benchmark"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$EXP_ROOT/logs/$RUN_ID"
echo "[launch] method=$METHOD dataset=$DATASET physical_gpu=$PHYSICAL_GPU run=$RUN_ID"
echo "[launch] $(date --iso-8601=seconds)"
python -u "$EXP_ROOT/run_experiment.py" \
  --method "$METHOD" --dataset "$DATASET" --device cuda:0 \
  --run-id "$RUN_ID" --epochs "${EPOCHS:-2}" \
  2>&1 | tee "$EXP_ROOT/logs/$RUN_ID/${METHOD}_${DATASET}.log"
