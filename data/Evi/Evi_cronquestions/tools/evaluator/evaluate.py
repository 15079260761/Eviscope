"""Official evaluator CLI (spec sections 17/25).

Usage:
  python evaluator/evaluate.py \
      --labels <private_labels.jsonl> --facts <facts.jsonl> \
      --config <construction_config.json> \
      --submission <submission.jsonl> --output <scores.json> [-k 10] [--stratified]

The frozen encoder and BS threshold come from the released construction
config; the evaluator never re-fits anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import iter_jsonl, read_json, write_json
from src.encoder import FrozenEncoder
from evaluator.metrics import evaluate_all, stratified_summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="EviScope official evaluator")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="frozen construction config (from the release)")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--stratified", action="store_true")
    args = parser.parse_args(argv)

    frozen = read_json(args.config)
    hdm_cfg = frozen.get("hdm", {})
    eval_cfg = frozen.get("evaluation", {})
    tau = float(eval_cfg.get("bs_tau", 0.0))
    encoder = FrozenEncoder({"encoder": hdm_cfg.get("encoder", {})}, args.facts)
    facts = {}
    for ordinal, fact in enumerate(iter_jsonl(args.facts)):
        index = int(fact.get("fact_index", ordinal))
        facts[index] = fact
        facts[str(fact["fact_id"])] = fact
    questions = list(iter_jsonl(args.labels))
    submissions = list(iter_jsonl(args.submission))
    summary = evaluate_all(questions, submissions, args.k, encoder, tau, facts)
    if args.stratified:
        summary["stratified"] = stratified_summary(questions, summary["per_question"])
    summary["evaluator"] = {"encoder_revision": encoder.revision, "bs_tau": tau}
    write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_question", "stratified")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
