#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
SEED="${SEED:-20260927}"
EPOCHS="${EPOCHS:-3}"
RESULTS_ROOT="$EXPERIMENT_DIR/results/$RUN_TOKEN"

command -v tmux >/dev/null || { echo "[FATAL] tmux is not installed"; exit 1; }
[[ -f "$RESULTS_ROOT/.inputs_ready" && -f "$RESULTS_ROOT/.runtime_ready" ]] || {
  echo "[FATAL] preparation/runtime preflight has not passed"; exit 1; }
active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$active" ]] || { echo "[FATAL] active GPU processes: $active" >&2; exit 1; }

launch_configuration() {
  local configuration="$1" gpu="$2"
  local session="tempoqr_${configuration}_${RUN_TOKEN}"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "[FATAL] tmux session exists: $session" >&2; exit 1; }
  tmux new-session -d -s "$session" -n clean-gpu${gpu}
  tmux new-window -t "$session" -n full-gpu${gpu}
  local common="cd '$PROJECT_ROOT' && source '$VENV_ACTIVATE' && export RUN_TOKEN='$RUN_TOKEN' SEED='$SEED' EPOCHS='$EPOCHS' PYTHON_BIN='$PYTHON_BIN' TKBC_CHECKPOINT='$TKBC_CHECKPOINT' GPU_SLOT='$gpu' &&"
  tmux send-keys -t "$session:clean-gpu${gpu}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' '$configuration' clean; exec bash" C-m
  tmux send-keys -t "$session:full-gpu${gpu}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' '$configuration' full; exec bash" C-m
  echo "[launched] $session: clean/full on physical GPU $gpu"
}

launch_configuration soft_cronkgqa 0
launch_configuration soft_tempoqr 1
echo "查看 Soft/CronKGQA: tmux attach -t tempoqr_soft_cronkgqa_${RUN_TOKEN}"
echo "查看 Soft/TempoQR:  tmux attach -t tempoqr_soft_tempoqr_${RUN_TOKEN}"
