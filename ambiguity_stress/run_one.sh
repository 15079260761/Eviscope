#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: run_one.sh METHOD DATASET PHYSICAL_GPU RUN_ID CHECKPOINT_RUN_ID" >&2; exit 2; }
METHOD="$1"; DATASET="$2"; PHYSICAL_GPU="$3"; RUN_ID="$4"; CHECKPOINT_RUN_ID="$5"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
EXP_ROOT="$PROJECT_ROOT/ambiguity_stress"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$EXP_ROOT/logs/$RUN_ID"
echo "[launch] method=$METHOD dataset=$DATASET physical_gpu=$PHYSICAL_GPU run=$RUN_ID checkpoint_run=$CHECKPOINT_RUN_ID"
python -u "$EXP_ROOT/run_method.py" --method "$METHOD" --dataset "$DATASET" --device cuda:0 \
  --run-id "$RUN_ID" --checkpoint-run-id "$CHECKPOINT_RUN_ID" \
  2>&1 | tee "$EXP_ROOT/logs/$RUN_ID/${METHOD}_${DATASET}.log"
