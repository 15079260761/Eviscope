"""Prepare the leakage-safe paired CronQuestions inputs used by Table 2."""
import argparse
import json
from pathlib import Path

from eviscope_adapter import prepare_questions


parser = argparse.ArgumentParser()
parser.add_argument("--orig-root", type=Path, required=True)
parser.add_argument("--evi-root", type=Path, required=True)
parser.add_argument("--output-root", type=Path, default=Path("eviscope_data"))
parser.add_argument("--train-limit", type=int, default=10_000)
parser.add_argument("--dev-limit", type=int, default=2_000)
parser.add_argument("--seed", type=int, default=20260926)
args = parser.parse_args()

manifest = prepare_questions(args.orig_root, args.evi_root, args.output_root,
                             args.train_limit, args.dev_limit, args.seed)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
