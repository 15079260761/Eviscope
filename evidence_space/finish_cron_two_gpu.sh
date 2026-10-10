#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: finish_cron_two_gpu.sh GPU_LANE[0|1] [RUN_ID]"
  exit 2
}

GPU_LANE="$1"
RUN_ID="${2:-exp2_v2}"
[[ "$GPU_LANE" == 0 || "$GPU_LANE" == 1 ]] || {
  echo "GPU_LANE must be 0 or 1"
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
EXP="$PROJECT_ROOT/evidence_space"
EXP1_RUN_ID="${EXP1_RUN_ID:-exp1_full_v1}"
LOG_ROOT="$EXP/logs/$RUN_ID"

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_LANE"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p "$LOG_ROOT"

run_logged() {
  local label="$1"
  shift
  echo "================================================================"
  echo "[START] $label physical_gpu=$GPU_LANE $(date --iso-8601=seconds)"
  echo "================================================================"
  "$@" 2>&1 | tee -a "$LOG_ROOT/${label}.log"
  echo "[SUCCESS] $label physical_gpu=$GPU_LANE $(date --iso-8601=seconds)"
}

run_benchmark() {
  local method="$1"
  run_logged "recovery_benchmark_${method}_cron" \
    python -u "$EXP/benchmark_side_test_settings/run.py" \
      --method "$method" --dataset cron --run-id "$RUN_ID" \
      --exp1-run-id "$EXP1_RUN_ID" --device cuda:0 --resume-verified
}

run_variant() {
  local variant="$1"
  run_logged "recovery_training_eviret_cron_${variant}" \
    python -u "$EXP/training_side_train_settings/run.py" \
      --dataset cron --variant "$variant" --run-id "$RUN_ID" \
      --device cuda:0 --resume-verified
}

if [[ "$GPU_LANE" == 0 ]]; then
  run_benchmark cross_encoder
  run_variant anchor_only
else
  run_benchmark eviret
  run_variant sub
  run_variant dis
  run_variant sub_dis
  run_variant full_eviret
fi

echo "[ALL SUCCESS] Cron recovery lane $GPU_LANE completed"
