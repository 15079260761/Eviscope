#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
EVI_ROOT="${EVI_ROOT:-$PROJECT_ROOT/data/Evi/Evi_cronquestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SEED="${SEED:-20260926}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
MULTIQA_DIR="$BASELINES_DIR/MultiTQ-main/MultiQA"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

required=(
  "$ORIG_ROOT/wikidata_big/questions/train.pickle"
  "$ORIG_ROOT/wikidata_big/kg/tkbc_processed_data/wikidata_big/ent_id"
  "$EVI_ROOT/data/questions/test.jsonl"
  "$EVI_ROOT/data/descriptors/test.jsonl"
  "$EVI_ROOT/data/private_labels/test.jsonl"
  "$EVI_ROOT/data/facts/facts.jsonl"
  "$EVI_ROOT/audit/leakage_report.json"
  "$TKBC_CHECKPOINT"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "[FATAL] missing: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] invalid Python: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$RESULTS_ROOT"
cd "$MULTIQA_DIR"
"$PYTHON_BIN" -u prepare_eviscope.py \
  --orig-root "$ORIG_ROOT" --evi-root "$EVI_ROOT" \
  --output-root eviscope_data --train-limit 10000 --dev-limit 2000 --seed "$SEED"

"$PYTHON_BIN" -u "$EXPERIMENT_DIR/audit_protocol.py" \
  --orig-root "$ORIG_ROOT" --evi-root "$EVI_ROOT" \
  --prepared "$MULTIQA_DIR/eviscope_data" \
  --output "$RESULTS_ROOT/protocol_audit.json"
printf '%s\n' "ready $(date -Iseconds)" > "$RESULTS_ROOT/.inputs_ready"
echo "[SUCCESS] inputs ready: $RESULTS_ROOT"
