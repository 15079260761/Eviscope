#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
RUN_ID="${1:-exp3_v1}"
CHECKPOINT_RUN_ID="${2:-exp1_full_v1}"
EXP_ROOT="$PROJECT_ROOT/ambiguity_stress"
command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
test -x "$VENV/bin/python" || { echo "missing virtual environment: $VENV" >&2; exit 1; }
cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
python -u "$EXP_ROOT/audit_protocol.py" --dataset all
for DATASET in cron icews; do
  for METHOD in dual_encoder cross_encoder eviret; do
    CHECKPOINT="$PROJECT_ROOT/main_benchmark/checkpoints/$CHECKPOINT_RUN_ID/$METHOD/$DATASET"
    test -f "$CHECKPOINT/config.json" || { echo "missing checkpoint config: $CHECKPOINT/config.json" >&2; exit 1; }
    test -f "$CHECKPOINT/training_metadata.json" || { echo "missing checkpoint metadata: $CHECKPOINT/training_metadata.json" >&2; exit 1; }
  done
done
for ITEM in "cron:0" "icews:1"; do
  DATASET="${ITEM%%:*}"; GPU="${ITEM##*:}"; SESSION="eviscope_${RUN_ID}_${DATASET}_gpu${GPU}"
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
  COMMAND="cd '$PROJECT_ROOT' && source '$VENV/bin/activate' && bash '$EXP_ROOT/run_dataset.sh' '$DATASET' '$GPU' '$RUN_ID' '$CHECKPOINT_RUN_ID'; exec bash"
  tmux new-session -d -s "$SESSION" -n "${DATASET}-all4" "$COMMAND"
  echo "[launched] $SESSION (GPU $GPU): BM25 -> Dual Encoder -> Cross-Encoder -> EviRet"
done
echo "attach Cron : tmux attach -t eviscope_${RUN_ID}_cron_gpu0"
echo "attach ICEWS: tmux attach -t eviscope_${RUN_ID}_icews_gpu1"
