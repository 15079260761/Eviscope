#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "main_benchmark"
sys.path.insert(0, str(EXP1))
from _shared.data import prepare_dataset
MODES = {
    "cron": {"clean": "clean", "substitute": "substitute_only", "distractor": "distractor_only", "full": "full"},
    "icews": {"clean": "clean_control", "substitute": "substitute_only", "distractor": "distractor_only", "full": "full_eviscope"},
}
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=("cron", "icews", "all"), default="all"); parser.add_argument("--output", type=Path, default=HERE / "prepared"); args = parser.parse_args()
    for dataset in (("cron", "icews") if args.dataset == "all" else (args.dataset,)):
        audit = prepare_dataset(dataset, args.output / dataset, modes=MODES[dataset])
        print(json.dumps({"dataset": dataset, "selected": audit["eligible_and_selected"], "modes": audit["modes"]}, ensure_ascii=False), flush=True)
if __name__ == "__main__": main()
