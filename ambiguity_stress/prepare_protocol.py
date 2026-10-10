#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from protocol import prepare_dataset

HERE = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe paired pools for Experiment 3")
    parser.add_argument("--dataset", choices=("cron", "icews", "all"), default="all")
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    args = parser.parse_args()
    for dataset in (("cron", "icews") if args.dataset == "all" else (args.dataset,)):
        audit = prepare_dataset(dataset, args.output / dataset)
        print(f"[SUCCESS] prepared {dataset}: N={audit['selected_test_questions']} "
              f"Q33={audit['train_quantiles']['q33']:.6f} Q66={audit['train_quantiles']['q66']:.6f}", flush=True)

if __name__ == "__main__":
    main()
