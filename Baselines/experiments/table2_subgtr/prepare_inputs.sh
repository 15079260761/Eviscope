#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
SUBGTR_DIR="$BASELINES_DIR/SubGTR-main"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
EVI_ROOT="${EVI_ROOT:-$PROJECT_ROOT/data/Evi/Evi_cronquestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SEED="${SEED:-20260927}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"

required=(
  "$ORIG_ROOT/wikidata_big/questions/train.pickle"
  "$ORIG_ROOT/wikidata_big/questions/valid.pickle"
  "$ORIG_ROOT/wikidata_big/questions/test.pickle"
  "$ORIG_ROOT/wikidata_big/kg/full.txt"
  "$ORIG_ROOT/wikidata_big/kg/tkbc_processed_data/wikidata_big/ent_id"
  "$EVI_ROOT/data/questions/test.jsonl"
  "$EVI_ROOT/data/descriptors/test.jsonl"
  "$EVI_ROOT/data/private_labels/test.jsonl"
  "$EVI_ROOT/data/facts/facts.jsonl"
  "$EVI_ROOT/audit/leakage_report.json"
  "$SUBGTR_DIR/ComplexCronQuestions/train.pickle"
  "$SUBGTR_DIR/ComplexCronQuestions/valid.pickle"
  "$SUBGTR_DIR/saved_pkl/e2rt.pkl"
  "$SUBGTR_DIR/saved_pkl/e2tr.pkl"
  "$SUBGTR_DIR/saved_pkl/event2time.pkl"
  "$TKBC_CHECKPOINT"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "[FATAL] missing: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] invalid Python: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$RESULTS_ROOT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$SUBGTR_DIR"
"$PYTHON_BIN" -u prepare_eviscope.py \
  --orig-root "$ORIG_ROOT" --evi-root "$EVI_ROOT" \
  --complex-root "$SUBGTR_DIR/ComplexCronQuestions" \
  --output-root "$SUBGTR_DIR/eviscope_data" \
  --train-limit 10000 --dev-limit 2000 --seed "$SEED"

"$PYTHON_BIN" -u "$EXPERIMENT_DIR/audit_protocol.py" \
  --evi-root "$EVI_ROOT" --orig-root "$ORIG_ROOT" \
  --prepared "$SUBGTR_DIR/eviscope_data" --subgtr-root "$SUBGTR_DIR" \
  --output "$RESULTS_ROOT/protocol_audit.json"
printf '%s\n' "ready $(date -Iseconds)" > "$RESULTS_ROOT/.inputs_ready"
echo "[SUCCESS] SubGTR inputs and protocol audit ready: $RESULTS_ROOT"
