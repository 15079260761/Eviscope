#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
SUBGTR_DIR="$PROJECT_ROOT/Baselines/SubGTR-main"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"

[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] inputs not prepared"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export SUBGTR_LM_MODEL="${SUBGTR_LM_MODEL:-distilbert-base-uncased}"
cd "$SUBGTR_DIR"
"$PYTHON_BIN" -u "$EXPERIMENT_DIR/runtime_preflight.py" \
  --subgtr-root "$SUBGTR_DIR" --prepared "$SUBGTR_DIR/eviscope_data" \
  --kg-root "$ORIG_ROOT/wikidata_big/kg" --tkbc-checkpoint "$TKBC_CHECKPOINT" \
  --output "$RESULTS_ROOT/runtime_preflight.json"
printf '%s\n' "ready $(date -Iseconds)" > "$RESULTS_ROOT/.runtime_ready"
echo "[SUCCESS] CUDA/model/data forward+backward preflight passed"
