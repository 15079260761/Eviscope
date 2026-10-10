#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
CONDITION="${2:-}"
[[ "$METHOD" == "cronkgqa" || "$METHOD" == "ctrn" ]] || {
  echo "Usage: bash run_table2_method.sh {cronkgqa|ctrn} {clean|full}" >&2; exit 2;
}
[[ "$CONDITION" == "clean" || "$CONDITION" == "full" ]] || {
  echo "Usage: bash run_table2_method.sh {cronkgqa|ctrn} {clean|full}" >&2; exit 2;
}

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$BASELINES_DIR/.." && pwd)"
ORIG_ROOT="${ORIG_ROOT:-$PROJECT_ROOT/data/Ori/CronQuestions}"
EVI_ROOT="${EVI_ROOT:-$PROJECT_ROOT/data/Evi/Evi_cronquestions}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SEED="${SEED:-20260926}"
EPOCHS="${EPOCHS:-3}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN must be set}"
GPU_SLOT="${GPU_SLOT:?GPU_SLOT must be 0 or 1}"
[[ "$GPU_SLOT" == "0" || "$GPU_SLOT" == "1" ]] || { echo "[FATAL] invalid GPU_SLOT"; exit 2; }
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT must be set}"
CHECKPOINT_NAME="eviscope_10k_seed${SEED}_${RUN_TOKEN}"
STATUS_PREFIX="$RESULTS_ROOT/.${METHOD}_${CONDITION}"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CRONQUESTIONS_DATA_ROOT="$ORIG_ROOT"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] inputs not prepared" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[FATAL] invalid Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$TKBC_CHECKPOINT" ]] || { echo "[FATAL] missing TComplEx checkpoint" >&2; exit 1; }
for required in "$EVI_ROOT/tools/src/common.py" "$EVI_ROOT/tools/src/bm25.py" \
  "$EVI_ROOT/tools/src/encoder.py" "$EVI_ROOT/tools/evaluator/metrics.py"; do
  [[ -f "$required" ]] || { echo "[FATAL] incomplete evaluator package: $required" >&2; exit 1; }
done

if [[ "$METHOD" == "cronkgqa" ]]; then
  METHOD_DIR="$BASELINES_DIR/CronKGQA-main"
  ENTRY="train_qa_model.py"
  MODEL="embedkgqa"
else
  METHOD_DIR="$BASELINES_DIR/CTRN-main"
  ENTRY="train.py"
  MODEL="ctrn"
fi
MODEL_CHECKPOINT="$METHOD_DIR/models/wikidata_big/qa_models/${CHECKPOINT_NAME}.ckpt"
LABELS="$METHOD_DIR/eviscope_data/evaluation/test_labels_full_intersection.jsonl"
[[ -f "$LABELS" ]] || { echo "[FATAL] missing paired evaluation labels" >&2; exit 1; }

mkdir -p "$RESULTS_ROOT"
rm -f "${STATUS_PREFIX}_failed" "${STATUS_PREFIX}_complete"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s\n" "failed $(date -Iseconds) exit=$rc" > "${STATUS_PREFIX}_failed"; fi' EXIT

# Full uses the exact checkpoint produced/reused by Clean.  It waits without
# occupying GPU 1, so another ready experiment can use that device meanwhile.
if [[ "$CONDITION" == "full" ]]; then
  echo "[wait] $METHOD/full waiting for shared trained checkpoint"
  wait_ticks=0
  while [[ ! -s "$MODEL_CHECKPOINT" ]]; do
    [[ ! -f "$RESULTS_ROOT/.${METHOD}_clean_failed" ]] || {
      echo "[FATAL] $METHOD/clean failed before producing the checkpoint" >&2; exit 1;
    }
    sleep 20
    wait_ticks=$((wait_ticks + 1))
    if (( wait_ticks % 3 == 0 )); then
      echo "[wait-heartbeat] $(date '+%F %T') $METHOD/full still waiting for Clean training"
    fi
  done
  echo "[wait] shared checkpoint is ready: $MODEL_CHECKPOINT"
fi

# Cross-session device lock: CronKGQA and CTRN may be launched together, but
# only one process can own a physical GPU at a time.
GPU_LOCK="/tmp/eviscope_table2_gpu${GPU_SLOT}.lock"
exec 9>"$GPU_LOCK"
echo "[queue] $METHOD/$CONDITION waiting for physical GPU $GPU_SLOT"
flock 9
echo "[start] $METHOD/$CONDITION acquired physical GPU $GPU_SLOT at $(date -Iseconds)"
nvidia-smi -i "$GPU_SLOT" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader || true

run_visible() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo "[stage] $label"
  echo "[stage] command start: $(date -Iseconds)"
  echo "============================================================"
  "$@" &
  local worker_pid=$!
  (
    while sleep 60; do
      kill -0 "$worker_pid" 2>/dev/null || exit 0
      echo "[heartbeat] $(date '+%F %T') $label is still running"
      nvidia-smi -i "$GPU_SLOT" --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true
    done
  ) &
  local heartbeat_pid=$!
  local rc=0
  wait "$worker_pid" || rc=$?
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if [[ "$rc" -ne 0 ]]; then
    echo "[FAILED] $label exit=$rc at $(date -Iseconds)" >&2
    return "$rc"
  fi
  echo "[done] $label at $(date -Iseconds)"
}

train_if_needed() {
  if [[ "$CONDITION" != "clean" ]]; then
    echo "[train] Full reuses the shared $METHOD checkpoint; no duplicate training"
    return 0
  fi
  if [[ -s "$MODEL_CHECKPOINT" ]]; then
    echo "[resume] shared trained checkpoint exists; skipping training: $MODEL_CHECKPOINT"
    return 0
  fi
  cd "$METHOD_DIR"
  export CRONQUESTIONS_QUESTIONS_ROOT="$PWD/eviscope_data"
  if [[ "$METHOD" == "cronkgqa" ]]; then
    run_visible "$METHOD shared training" \
      "$PYTHON_BIN" -u "$ENTRY" --mode train --model "$MODEL" --dataset_name wikidata_big \
      --tkbc_model_file "$TKBC_CHECKPOINT" --train_split train_10k --eval_split valid_2k \
      --max_epochs "$EPOCHS" --valid_freq 1 --eval_k 1 --batch_size 64 \
      --valid_batch_size 32 --lr 0.00002 --frozen 1 --lm_frozen 1 \
      --num_workers 2 --seed "$SEED" --save_to "$CHECKPOINT_NAME"
  else
    run_visible "$METHOD shared training" \
      "$PYTHON_BIN" -u "$ENTRY" --mode train --model "$MODEL" --dataset_name wikidata_big \
      --tkbc_model_file "$TKBC_CHECKPOINT" --train_split train_10k --test valid_2k \
      --eval_split valid_2k --max_epochs "$EPOCHS" --valid_freq 1 --eval_k 1 \
      --batch_size 64 --valid_batch_size 32 --lr 0.0002 --frozen 1 --lm_frozen 1 \
      --num_workers 2 --seed "$SEED" --save_to "$CHECKPOINT_NAME"
  fi
  [[ -s "$MODEL_CHECKPOINT" ]] || { echo "[FATAL] training produced no checkpoint" >&2; exit 1; }
}

evaluate_condition() {
  local output_dir="$RESULTS_ROOT/$METHOD/$CONDITION"
  mkdir -p "$output_dir"
  cd "$METHOD_DIR"
  export CRONQUESTIONS_QUESTIONS_ROOT="$PWD/eviscope_data"
  local common=(--mode eval --model "$MODEL" --dataset_name wikidata_big
    --tkbc_model_file "$TKBC_CHECKPOINT" --train_split train_10k
    --eval_split test_eviscope --valid_batch_size 32 --num_workers 2
    --seed "$SEED" --load_from "$CHECKPOINT_NAME" --eviscope_root "$EVI_ROOT"
    --eviscope_view "$CONDITION" --evidence_predictions_out "$output_dir/predictions.jsonl")
  [[ "$METHOD" != "ctrn" ]] || common+=(--test test_eviscope)
  run_visible "$METHOD/$CONDITION model inference" \
    "$PYTHON_BIN" -u "$ENTRY" "${common[@]}"
  for k in 1 10; do
    run_visible "$METHOD/$CONDITION official evaluation K=$k" \
      env PYTHONPATH="$EVI_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -u -m evaluator.evaluate \
        --labels "$LABELS" --facts "$EVI_ROOT/data/facts/facts.jsonl" \
        --config "$EVI_ROOT/construction_config.json" \
        --submission "$output_dir/predictions.jsonl" \
        --output "$output_dir/scores_k${k}.json" -k "$k" --stratified
  done
}

train_if_needed
evaluate_condition
printf '%s\n' "completed $(date -Iseconds)" > "${STATUS_PREFIX}_complete"
trap - EXIT
echo "[SUCCESS] $METHOD/$CONDITION finished on physical GPU $GPU_SLOT"

required=(.cronkgqa_clean_complete .cronkgqa_full_complete .ctrn_clean_complete .ctrn_full_complete)
for marker in "${required[@]}"; do
  [[ -f "$RESULTS_ROOT/$marker" ]] || exit 0
done

LOCK_DIR="$RESULTS_ROOT/.summary_lock"
if mkdir "$LOCK_DIR" 2>/dev/null; then
  "$PYTHON_BIN" -u "$EXPERIMENT_DIR/summarize_table2.py" --results-root "$RESULTS_ROOT" \
    || { printf '%s\n' "summary failed" > "$RESULTS_ROOT/.summary_failed"; exit 1; }
  printf '%s\n' "summarized $(date -Iseconds)" > "$RESULTS_ROOT/.summary_complete"
fi
