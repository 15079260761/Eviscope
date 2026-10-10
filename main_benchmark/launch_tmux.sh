#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
RUN_ID="${1:-exp1_v1}"
EXP_ROOT="$PROJECT_ROOT/main_benchmark"
command -v tmux >/dev/null || { echo "tmux is not installed: apt-get update && apt-get install -y tmux"; exit 1; }
test -x "$VENV/bin/python" || { echo "missing virtual environment: $VENV"; exit 1; }

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
if [[ ! -f "$EXP_ROOT/prepared/cron/audit.json" || ! -f "$EXP_ROOT/prepared/icews/audit.json" ]]; then
  python -u "$EXP_ROOT/prepare_protocol.py" --dataset all
else
  echo "[prepare] reusing existing frozen manifests; audit follows"
fi
python -u "$EXP_ROOT/audit_protocol.py"

launch_dataset() {
  local dataset="$1" physical_gpu="$2" session="eviscope_exp1_${1}_gpu${2}"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "session already exists: $session (kill it explicitly before relaunching)" >&2; exit 1;
  }
  local methods=(bm25 dual_encoder cross_encoder eviret)
  local windows=(bm25 dual cross eviret)
  tmux new-session -d -s "$session" -n "${windows[0]}"
  for i in 1 2 3; do tmux new-window -t "$session" -n "${windows[$i]}"; done
  for i in 0 1 2 3; do
    local wait_prefix=""
    local signal_suffix=""
    if (( i > 0 )); then wait_prefix="echo '[queue] ${methods[$i]}/$dataset waiting behind ${methods[$((i-1))]} on physical GPU $physical_gpu'; tmux wait-for ${session}_stage_${i}; echo '[queue] acquired physical GPU $physical_gpu'; "; fi
    if (( i < 3 )); then signal_suffix=" && tmux wait-for -S ${session}_stage_$((i+1))"; fi
    local command="cd '$PROJECT_ROOT' && source '$VENV/bin/activate' && ${wait_prefix}bash '$EXP_ROOT/run_one.sh' '${methods[$i]}' '$dataset' '$physical_gpu' '$RUN_ID'${signal_suffix}; exec bash"
    tmux send-keys -t "$session:${windows[$i]}" "$command" C-m
  done
  echo "created $session: bm25 -> dual_encoder -> cross_encoder -> eviret"
}

launch_dataset cron 0
launch_dataset icews 1
echo "GPU 0 and GPU 1 now run in parallel; jobs within each GPU are serialized."
echo "attach: tmux attach -t eviscope_exp1_cron_gpu0"
echo "attach: tmux attach -t eviscope_exp1_icews_gpu1"
