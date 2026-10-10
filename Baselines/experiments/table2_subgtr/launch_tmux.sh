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
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_DIR/results/$RUN_TOKEN}"

command -v tmux >/dev/null || { echo "[FATAL] tmux is not installed"; exit 1; }
[[ -f "$RESULTS_ROOT/.inputs_ready" ]] || { echo "[FATAL] inputs are not prepared"; exit 1; }
[[ -f "$RESULTS_ROOT/.runtime_ready" ]] || { echo "[FATAL] runtime preflight has not passed"; exit 1; }
active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$active_gpu_pids" ]] || {
  echo "[FATAL] existing GPU compute processes detected; refusing a conflicting launch: $active_gpu_pids" >&2
  exit 1
}

launch_variant() {
  local variant="$1" gpu="$2" short="$3"
  local session="subgtr_${short}_${RUN_TOKEN}"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "[FATAL] tmux session already exists: $session" >&2; exit 1; }
  tmux new-session -d -s "$session" -n clean-gpu${gpu}
  tmux new-window -t "$session" -n full-gpu${gpu}
  local common="cd '$PROJECT_ROOT' && source '$VENV_ACTIVATE' && export RUN_TOKEN='$RUN_TOKEN' SEED='$SEED' EPOCHS='$EPOCHS' PYTHON_BIN='$PYTHON_BIN' TKBC_CHECKPOINT='$TKBC_CHECKPOINT' RESULTS_ROOT='$RESULTS_ROOT' GPU_SLOT='$gpu' &&"
  tmux send-keys -t "$session:clean-gpu${gpu}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' '$variant' clean; exec bash" C-m
  tmux send-keys -t "$session:full-gpu${gpu}" "$common bash '$EXPERIMENT_DIR/run_condition.sh' '$variant' full; exec bash" C-m
  echo "[launched] $session: clean-gpu${gpu}, full-gpu${gpu}"
}

# One configuration per physical GPU; its Full window waits for the single
# Clean-trained checkpoint, so no two jobs can contend for the same GPU.
launch_variant wikidata_big 0 standard
launch_variant wikidata_big_complex 1 complex
echo
echo "查看 standard: tmux attach -t subgtr_standard_${RUN_TOKEN}"
echo "查看 complex:  tmux attach -t subgtr_complex_${RUN_TOKEN}"
echo "窗口切换: Ctrl-b n    脱离: Ctrl-b d"
