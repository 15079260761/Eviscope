#!/usr/bin/env bash
set -euo pipefail

GPU_SLOT="${1:-}"
[[ "$GPU_SLOT" == 0 || "$GPU_SLOT" == 1 ]] || {
  echo "Usage: launch_third.sh {0|1}" >&2; exit 2; }
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
SEED="${SEED:-20260927}"
EPOCHS="${EPOCHS:-3}"
RESULTS_ROOT="$EXPERIMENT_DIR/results/$RUN_TOKEN"
SESSION="tempoqr_hard_tempoqr_${RUN_TOKEN}"

command -v tmux >/dev/null || { echo "[FATAL] tmux is not installed"; exit 1; }
[[ -f "$RESULTS_ROOT/.inputs_ready" && -f "$RESULTS_ROOT/.runtime_ready" ]] || {
  echo "[FATAL] preparation/runtime preflight has not passed"; exit 1; }
active="$(nvidia-smi -i "$GPU_SLOT" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$active" ]] || { echo "[FATAL] GPU $GPU_SLOT is still busy: $active" >&2; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "[FATAL] tmux session exists: $SESSION" >&2; exit 1; }

tmux new-session -d -s "$SESSION" -n clean-gpu${GPU_SLOT}
tmux new-window -t "$SESSION" -n full-gpu${GPU_SLOT}
common="cd '$PROJECT_ROOT' && source '$VENV_ACTIVATE' && export RUN_TOKEN='$RUN_TOKEN' SEED='$SEED' EPOCHS='$EPOCHS' PYTHON_BIN='$PYTHON_BIN' TKBC_CHECKPOINT='$TKBC_CHECKPOINT' GPU_SLOT='$GPU_SLOT' &&"
tmux send-keys -t "$SESSION:clean-gpu${GPU_SLOT}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' hard_tempoqr clean; exec bash" C-m
tmux send-keys -t "$SESSION:full-gpu${GPU_SLOT}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' hard_tempoqr full; exec bash" C-m
echo "[launched] Hard/TempoQR Clean+Full on GPU $GPU_SLOT"
echo "查看: tmux attach -t $SESSION"
