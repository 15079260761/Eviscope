#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 ]] || { echo "usage: run_dataset.sh DATASET PHYSICAL_GPU RUN_ID CHECKPOINT_RUN_ID" >&2; exit 2; }
DATASET="$1"; GPU="$2"; RUN_ID="$3"; CHECKPOINT_RUN_ID="$4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EXP_ROOT="$PROJECT_ROOT/ambiguity_stress"
for METHOD in bm25 dual_encoder cross_encoder eviret; do
  echo "================================================================"
  echo "[stage] $METHOD/$DATASET $(date --iso-8601=seconds)"
  bash "$EXP_ROOT/run_one.sh" "$METHOD" "$DATASET" "$GPU" "$RUN_ID" "$CHECKPOINT_RUN_ID"
done
echo "[SUCCESS] all four methods complete for $DATASET"
