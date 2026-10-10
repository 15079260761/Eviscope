#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
CONDITION="${2:-}"
GPU_SLOT="${3:-}"
RUN_TOKEN="${4:-20260926_163208}"

[[ "$METHOD" == "cronkgqa" || "$METHOD" == "ctrn" ]] || {
  echo "Usage: bash launch_condition.sh {cronkgqa|ctrn} {clean|full} {0|1} [run_token]" >&2
  exit 2
}
[[ "$CONDITION" == "clean" || "$CONDITION" == "full" ]] || {
  echo "Usage: bash launch_condition.sh {cronkgqa|ctrn} {clean|full} {0|1} [run_token]" >&2
  exit 2
}
[[ "$GPU_SLOT" == "0" || "$GPU_SLOT" == "1" ]] || {
  echo "Usage: bash launch_condition.sh {cronkgqa|ctrn} {clean|full} {0|1} [run_token]" >&2
  exit 2
}

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TKBC_CHECKPOINT="$PROJECT_ROOT/Baselines/CronKGQA-main/models/wikidata_big/kg_embeddings/tcomplex_17dec.ckpt"

[[ -f "$VENV_DIR/bin/activate" ]] || { echo "[FATAL] missing venv: $VENV_DIR" >&2; exit 1; }
[[ -f "$TKBC_CHECKPOINT" ]] || { echo "[FATAL] missing checkpoint: $TKBC_CHECKPOINT" >&2; exit 1; }
source "$VENV_DIR/bin/activate"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_SLOT"
export GPU_SLOT RUN_TOKEN
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export SEED=20260926 EPOCHS=3
export PYTHON_BIN="$VENV_DIR/bin/python"
export TKBC_CHECKPOINT
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

echo "============================================================"
echo "[launch] method=$METHOD condition=$CONDITION physical_gpu=$GPU_SLOT"
echo "[launch] run_token=$RUN_TOKEN"
echo "[launch] project=$PROJECT_ROOT"
echo "[launch] realtime output + 60-second heartbeat enabled"
echo "============================================================"

cd "$PROJECT_ROOT"
exec bash "$EXPERIMENT_DIR/run_table2_method.sh" "$METHOD" "$CONDITION"
