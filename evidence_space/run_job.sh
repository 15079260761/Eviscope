#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 7 ]] || { echo "usage: run_job.sh DIRECTION METHOD DATASET GPU RUN_ID EXP1_RUN_ID VENV"; exit 2; }
DIRECTION="$1"; METHOD="$2"; DATASET="$3"; GPU="$4"; RUN_ID="$5"; EXP1_RUN_ID="$6"; VENV="$7"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"; EXP="$PROJECT_ROOT/evidence_space"
cd "$PROJECT_ROOT"; source "$VENV/bin/activate"; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$EXP/logs/$RUN_ID"
if [[ "$DIRECTION" == benchmark ]]; then
  python -u "$EXP/benchmark_side_test_settings/run.py" --method "$METHOD" --dataset "$DATASET" --run-id "$RUN_ID" --exp1-run-id "$EXP1_RUN_ID" --device cuda:0 --resume-verified 2>&1 | tee -a "$EXP/logs/$RUN_ID/benchmark_${METHOD}_${DATASET}.log"
else
  [[ "$METHOD" == eviret ]] || { echo "Section 2.7 training ablation is EviRet-only"; exit 2; }
  for variant in anchor_only sub dis sub_dis full_eviret; do
    echo "[training-ablation queue] EviRet/$DATASET/$variant"
    python -u "$EXP/training_side_train_settings/run.py" --dataset "$DATASET" --variant "$variant" --run-id "$RUN_ID" --device cuda:0 --resume-verified 2>&1 | tee -a "$EXP/logs/$RUN_ID/training_eviret_${DATASET}_${variant}.log"
  done
fi
