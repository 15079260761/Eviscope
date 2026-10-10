#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
RUN_ID="${1:-exp3_v1}"
CHECKPOINT_RUN_ID="${2:-exp1_full_v1}"
ATTEMPT_TAG="${3:-retry2}"
EXP_ROOT="$PROJECT_ROOT/ambiguity_stress"
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
python -u "$EXP_ROOT/audit_protocol.py" --dataset all
for ITEM in "cron:0" "icews:1"; do
  DATASET="${ITEM%%:*}"; GPU="${ITEM##*:}"
  SESSION="eviscope_${RUN_ID}_eviret_${ATTEMPT_TAG}_${DATASET}_gpu${GPU}"
  tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "session already exists: $SESSION" >&2
    exit 1
  }
  COMMAND="cd '$PROJECT_ROOT' && source '$VENV/bin/activate' && bash '$EXP_ROOT/run_one.sh' eviret '$DATASET' '$GPU' '$RUN_ID' '$CHECKPOINT_RUN_ID'; exec bash"
  tmux new-session -d -s "$SESSION" -n "eviret-${DATASET}" "$COMMAND"
  echo "[launched] EviRet/$DATASET on physical GPU $GPU: $SESSION"
done
echo "attach Cron : tmux attach -t eviscope_${RUN_ID}_eviret_${ATTEMPT_TAG}_cron_gpu0"
echo "attach ICEWS: tmux attach -t eviscope_${RUN_ID}_eviret_${ATTEMPT_TAG}_icews_gpu1"
