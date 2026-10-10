#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"; VENV="${VENV:-$PROJECT_ROOT/.venv}"; RUN_ID="${1:-exp2_v2}"; EXP1_RUN_ID="${2:-exp1_full_v1}"; EXP="$PROJECT_ROOT/evidence_space"
command -v tmux >/dev/null || { echo "install tmux first"; exit 1; }; test -x "$VENV/bin/python" || { echo "missing $VENV"; exit 1; }
cd "$PROJECT_ROOT"; source "$VENV/bin/activate"
if [[ ! -f "$EXP/prepared/cron/audit.json" || ! -f "$EXP/prepared/icews/audit.json" ]]; then python -u "$EXP/prepare_protocol.py" --dataset all; fi
python -u "$EXP/audit_protocol.py"
for method in dual_encoder cross_encoder eviret; do
  for dataset in cron icews; do test -f "$PROJECT_ROOT/main_benchmark/checkpoints/$EXP1_RUN_ID/$method/$dataset/config.json" || { echo "missing Exp1 checkpoint: $method/$dataset/$EXP1_RUN_ID"; exit 1; }; done
done
launch_dataset() {
  local dataset="$1"
  local gpu="$2"
  local session="eviscope_${RUN_ID}_${dataset}_gpu${gpu}"
  tmux has-session -t "$session" 2>/dev/null && { echo "session exists: $session"; exit 1; }
  local directions=(benchmark benchmark benchmark benchmark training); local methods=(bm25 dual_encoder cross_encoder eviret eviret); local names=(test-bm25 test-dual test-cross test-eviret train-eviret-ablation)
  tmux new-session -d -s "$session" -n "${names[0]}"; for i in 1 2 3 4; do tmux new-window -t "$session" -n "${names[$i]}"; done
  for i in 0 1 2 3 4; do
    local wait="" next=""; if ((i>0)); then wait="echo '[queue] ${names[$i]} waiting on GPU $gpu'; tmux wait-for ${session}_stage_${i}; echo '[queue] ${names[$i]} acquired GPU $gpu'; "; fi; if ((i<4)); then next=" && tmux wait-for -S ${session}_stage_$((i+1))"; fi
    local snapshot="$EXP/logs/$RUN_ID/run_job_${dataset}_${names[$i]}.snapshot.sh"
    mkdir -p "$EXP/logs/$RUN_ID"
    cp "$EXP/run_job.sh" "$snapshot"
    bash -n "$snapshot"
    local cmd="cd '$PROJECT_ROOT' && source '$VENV/bin/activate' && ${wait}bash '$snapshot' '${directions[$i]}' '${methods[$i]}' '$dataset' '$gpu' '$RUN_ID' '$EXP1_RUN_ID' '$VENV'${next}; exec bash"
    tmux send-keys -t "$session:${names[$i]}" "$cmd" C-m
  done
  echo "created $session"
}
launch_dataset cron 0; launch_dataset icews 1
echo "attach: tmux attach -t eviscope_${RUN_ID}_cron_gpu0"; echo "attach: tmux attach -t eviscope_${RUN_ID}_icews_gpu1"
