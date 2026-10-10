"""Prepare the leakage-safe paired inputs shared by all TempoQR variants."""
import argparse
import json
from pathlib import Path

from eviscope_adapter import prepare_questions

p = argparse.ArgumentParser()
p.add_argument("--orig-root", type=Path, required=True)
p.add_argument("--evi-root", type=Path, required=True)
p.add_argument("--output-root", type=Path, default=Path("eviscope_data"))
p.add_argument("--train-limit", type=int, default=10_000)
p.add_argument("--dev-limit", type=int, default=2_000)
p.add_argument("--seed", type=int, default=20260927)
args = p.parse_args()

manifest = prepare_questions(args.orig_root, args.evi_root, args.output_root,
                             args.train_limit, args.dev_limit, args.seed)
manifest["tempoqr_configurations"] = [
    "soft_cronkgqa", "soft_tempoqr", "hard_tempoqr",
]
manifest["shared_training_cohort"] = True
(args.output_root / "tempoqr_preparation_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
