#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"
RUN_TOKEN="${RUN_TOKEN:?RUN_TOKEN is required}"
TKBC_CHECKPOINT="${TKBC_CHECKPOINT:?TKBC_CHECKPOINT is required}"
SEED="${SEED:-20260926}"
EPOCHS="${EPOCHS:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"

command -v tmux >/dev/null || { echo "[FATAL] tmux is not installed"; exit 1; }
[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] inputs are not prepared"; exit 1; }

launch_session() {
  local method="$1"
  local session="${method}_${RUN_TOKEN}"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "[FATAL] tmux session already exists: $session" >&2; exit 1; }
  tmux new-session -d -s "$session" -n clean-gpu0
  tmux new-window -t "$session" -n full-gpu1
  local common="cd '$PROJECT_ROOT' && source '$VENV_ACTIVATE' && export RUN_TOKEN='$RUN_TOKEN' SEED='$SEED' EPOCHS='$EPOCHS' PYTHON_BIN='$PYTHON_BIN' TKBC_CHECKPOINT='$TKBC_CHECKPOINT' RESULTS_ROOT='$RESULTS_ROOT' &&"
  tmux send-keys -t "$session:clean-gpu0" "$common GPU_SLOT=0 bash '$EXPERIMENT_DIR/run_condition.sh' '$method' clean; exec bash" C-m
  tmux send-keys -t "$session:full-gpu1" "$common GPU_SLOT=1 bash '$EXPERIMENT_DIR/run_condition.sh' '$method' full; exec bash" C-m
  echo "[launched] $session: clean-gpu0, full-gpu1"
}

launch_session multiqa
launch_session rtqa
echo
echo "查看 MultiQA: tmux attach -t multiqa_${RUN_TOKEN}"
echo "查看 RTQA:    tmux attach -t rtqa_${RUN_TOKEN}"
echo "窗口切换: Ctrl-b n    脱离: Ctrl-b d"
