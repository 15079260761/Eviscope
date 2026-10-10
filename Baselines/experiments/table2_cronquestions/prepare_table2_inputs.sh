#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
EVI_ROOT="${EVI_ROOT:-$PROJECT_ROOT/data/Evi/Evi_cronquestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SEED="${SEED:-20260926}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN must be set}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT must be set}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CRONQUESTIONS_DATA_ROOT="$ORIG_ROOT"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

require_file() {
  [[ -f "$1" ]] || { echo "[FATAL] missing required file: $1" >&2; exit 1; }
}

require_file "$ORIG_ROOT/wikidata_big/questions/train.pickle"
require_file "$ORIG_ROOT/wikidata_big/kg/tkbc_processed_data/wikidata_big/ent_id"
require_file "$EVI_ROOT/data/questions/test.jsonl"
require_file "$EVI_ROOT/data/private_labels/test.jsonl"
require_file "$EVI_ROOT/data/descriptors/test.jsonl"
require_file "$EVI_ROOT/data/facts/facts.jsonl"
require_file "$EVI_ROOT/construction_config.json"
require_file "$TKBC_CHECKPOINT"
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] invalid Python: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$RESULTS_ROOT"
"$PYTHON_BIN" -u "$EXPERIMENT_DIR/preflight_table2.py" \
  --checkpoint "$TKBC_CHECKPOINT" --orig-root "$ORIG_ROOT" --evi-root "$EVI_ROOT"

for method in CronKGQA-main CTRN-main; do
  cd "$BASELINES_DIR/$method"
  "$PYTHON_BIN" -u prepare_eviscope.py \
    --orig-root "$ORIG_ROOT" --evi-root "$EVI_ROOT" \
    --output-root eviscope_data \
    --train-limit 10000 --dev-limit 2000 --seed "$SEED"
done

"$PYTHON_BIN" -u "$EXPERIMENT_DIR/audit_table2_protocol.py" \
  --orig-root "$ORIG_ROOT" \
  --evi-root "$EVI_ROOT" \
  --cron-prepared "$BASELINES_DIR/CronKGQA-main/eviscope_data" \
  --ctrn-prepared "$BASELINES_DIR/CTRN-main/eviscope_data" \
  --output "$RESULTS_ROOT/protocol_audit.json"

printf '%s\n' "prepared $(date -Iseconds)" > "$RESULTS_ROOT/.inputs_ready"
echo "[SUCCESS] paired inputs and protocol audit ready: $RESULTS_ROOT"
