#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
TEMPOQR_DIR="$PROJECT_ROOT/Baselines/TempoQR-main"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
RESULTS_ROOT="$EXPERIMENT_DIR/results/$RUN_TOKEN"

[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] inputs not prepared"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export TEMPOQR_LM_MODEL="${TEMPOQR_LM_MODEL:-distilbert-base-uncased}"
cd "$TEMPOQR_DIR"
"$PYTHON_BIN" -u "$EXPERIMENT_DIR/runtime_preflight.py" \
  --tempoqr-root "$TEMPOQR_DIR" --prepared "$TEMPOQR_DIR/eviscope_data" \
  --kg-root "$ORIG_ROOT/wikidata_big/kg" --tkbc-checkpoint "$TKBC_CHECKPOINT" \
  --output "$RESULTS_ROOT/runtime_preflight.json"
printf '%s\n' "ready $(date -Iseconds)" > "$RESULTS_ROOT/.runtime_ready"
echo "[SUCCESS] all three TempoQR CUDA forward/backward branches passed"
