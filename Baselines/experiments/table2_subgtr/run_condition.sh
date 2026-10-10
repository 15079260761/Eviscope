#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
VIEW="${2:-}"
[[ "$VARIANT" == "wikidata_big" || "$VARIANT" == "wikidata_big_complex" ]] || {
  echo "Usage: run_condition.sh {wikidata_big|wikidata_big_complex} {clean|full}" >&2; exit 2; }
[[ "$VIEW" == "clean" || "$VIEW" == "full" ]] || {
  echo "Usage: run_condition.sh {wikidata_big|wikidata_big_complex} {clean|full}" >&2; exit 2; }

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
SUBGTR_DIR="$BASELINES_DIR/SubGTR-main"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
EVI_ROOT="${EVI_ROOT:-$PROJECT_ROOT/data/Evi/Evi_cronquestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
GPU_SLOT="${GPU_SLOT:?GPU_SLOT must be 0 or 1}"
SEED="${SEED:-20260927}"
EPOCHS="${EPOCHS:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
CHECKPOINT="$SUBGTR_DIR/models/eviscope_${VARIANT}_10k_seed${SEED}_${RUN_TOKEN}.ckpt"
TRAINING_MARKER="${CHECKPOINT}.training_complete.json"
OUTPUT_DIR="$RESULTS_ROOT/subgtr_${VARIANT}/$VIEW"
LABELS="$SUBGTR_DIR/eviscope_data/evaluation/test_labels_full_intersection.jsonl"
STATUS_PREFIX="$RESULTS_ROOT/.${VARIANT}_${VIEW}"

[[ "$GPU_SLOT" == 0 || "$GPU_SLOT" == 1 ]] || { echo "[FATAL] invalid GPU_SLOT"; exit 2; }
[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] run prepare_inputs.sh first"; exit 1; }
[[ -f "$RESULTS_ROOT/.runtime_ready" ]] || { echo "[FATAL] run runtime_preflight.sh first"; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] invalid Python: $PYTHON_BIN"; exit 1; }
[[ -f "$LABELS" ]] || { echo "[FATAL] missing labels: $LABELS"; exit 1; }
mkdir -p "$OUTPUT_DIR" "$(dirname "$CHECKPOINT")"
rm -f "${STATUS_PREFIX}_failed" "${STATUS_PREFIX}_complete"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "failed exit=$rc $(date -Iseconds)" > "${STATUS_PREFIX}_failed"; fi' EXIT

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export SUBGTR_KG_ROOT="$ORIG_ROOT/wikidata_big/kg"
export SUBGTR_QUESTIONS_ROOT="$SUBGTR_DIR/eviscope_data/$VARIANT/questions"
export SUBGTR_LM_MODEL="${SUBGTR_LM_MODEL:-distilbert-base-uncased}"

visible() {
  local label="$1"; shift
  echo; echo "============================================================"
  echo "[stage] $label"; echo "[stage] $(date -Iseconds)"
  echo "============================================================"
  "$@" &
  local pid=$!
  ( while sleep 60; do kill -0 "$pid" 2>/dev/null || exit 0
      echo "[heartbeat] $(date '+%F %T') $label"
      nvidia-smi -i "$GPU_SLOT" --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    done ) &
  local heartbeat=$!
  local rc=0; wait "$pid" || rc=$?
  kill "$heartbeat" 2>/dev/null || true; wait "$heartbeat" 2>/dev/null || true
  [[ $rc -eq 0 ]] || { echo "[FAILED] $label exit=$rc" >&2; return "$rc"; }
  echo "[done] $label $(date -Iseconds)"
}

if [[ "$VIEW" == "full" ]]; then
  echo "[wait] SubGTR/$VARIANT/Full waits for its Clean window to finish the shared 10k training"
  ticks=0
  while [[ ! -s "$TRAINING_MARKER" ]]; do
    [[ ! -f "$RESULTS_ROOT/.${VARIANT}_clean_failed" ]] || {
      echo "[FATAL] $VARIANT/Clean failed; Full will not use an incomplete checkpoint" >&2; exit 1; }
    sleep 20; ticks=$((ticks + 1))
    (( ticks % 3 != 0 )) || echo "[wait-heartbeat] $(date '+%F %T') $VARIANT/Full still waiting"
  done
fi

(
  exec 9>"/tmp/eviscope_subgtr_gpu${GPU_SLOT}.lock"
  echo "[queue] SubGTR/$VARIANT/$VIEW waiting for physical GPU $GPU_SLOT"
  flock 9
  echo "[gpu] SubGTR/$VARIANT/$VIEW acquired physical GPU $GPU_SLOT at $(date -Iseconds)"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_SLOT"
  cd "$SUBGTR_DIR"
  if [[ "$VIEW" == "clean" && ! -s "$TRAINING_MARKER" ]]; then
    visible "SubGTR/$VARIANT 10k training ($EPOCHS epochs)" \
      "$PYTHON_BIN" -u run_eviscope_subgtr.py \
        --mode train --variant "$VARIANT" --tkbc-checkpoint "$TKBC_CHECKPOINT" \
        --checkpoint "$CHECKPOINT" --epochs "$EPOCHS" --seed "$SEED" \
        --batch-size 16 --eval-batch-size 16
  else
    echo "[train] reusing completed $VARIANT checkpoint: $CHECKPOINT"
  fi
  [[ -s "$CHECKPOINT" && -s "$TRAINING_MARKER" ]] || {
    echo "[FATAL] complete SubGTR training artifact is missing" >&2; exit 1; }
  visible "SubGTR/$VARIANT/$VIEW evidence inference" \
    "$PYTHON_BIN" -u run_eviscope_subgtr.py \
      --mode eval --variant "$VARIANT" --tkbc-checkpoint "$TKBC_CHECKPOINT" \
      --checkpoint "$CHECKPOINT" --eviscope-root "$EVI_ROOT" --view "$VIEW" \
      --predictions-out "$OUTPUT_DIR/predictions.jsonl" --seed "$SEED" \
      --eval-batch-size 16
)

for k in 1 10; do
  visible "SubGTR/$VARIANT/$VIEW official evaluation K=$k" \
    env PYTHONPATH="$EVI_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -u -m evaluator.evaluate \
      --labels "$LABELS" --facts "$EVI_ROOT/data/facts/facts.jsonl" \
      --config "$EVI_ROOT/construction_config.json" \
      --submission "$OUTPUT_DIR/predictions.jsonl" \
      --output "$OUTPUT_DIR/scores_k${k}.json" -k "$k" --stratified
done
printf '%s\n' "complete $(date -Iseconds)" > "${STATUS_PREFIX}_complete"
trap - EXIT
echo "[SUCCESS] SubGTR/$VARIANT/$VIEW complete"

all_done=true
for marker in .wikidata_big_clean_complete .wikidata_big_full_complete \
              .wikidata_big_complex_clean_complete .wikidata_big_complex_full_complete; do
  [[ -f "$RESULTS_ROOT/$marker" ]] || all_done=false
done
if [[ "$all_done" == true ]] && mkdir "$RESULTS_ROOT/.summary_lock" 2>/dev/null; then
  "$PYTHON_BIN" -u "$EXPERIMENT_DIR/summarize.py" --results-root "$RESULTS_ROOT"
fi
