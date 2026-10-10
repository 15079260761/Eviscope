#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared.data import prepare_dataset, prepared_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cron", "icews", "all"), default="all")
    parser.add_argument("--output", type=Path, default=prepared_root())
    args = parser.parse_args()
    datasets = ("cron", "icews") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        print(json.dumps(prepare_dataset(dataset, args.output / dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
