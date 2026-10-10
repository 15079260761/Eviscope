"""Fail-closed audit for the paired MultiQA/RTQA experiment."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--orig-root", type=Path, required=True)
parser.add_argument("--evi-root", type=Path, required=True)
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

def iter_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

sys.path.insert(0, str(args.evi_root))
from tools.candidate_space.runtime.pool_views import ReleaseFactIndex, materialize_pool

fact_index = ReleaseFactIndex(args.evi_root / "data/facts/facts.jsonl")
descriptors = iter_jsonl(args.evi_root / "data/descriptors/test.jsonl")
labels = iter_jsonl(args.evi_root / "data/private_labels/test.jsonl")
full_ids, pool_sizes = set(), {"clean": set(), "full": set()}
for descriptor, label in zip(descriptors, labels, strict=True):
    if descriptor["paired_id"] != label["paired_id"]:
        raise SystemExit("Descriptor/label order mismatch")
    if descriptor.get("mode_eligibility", {}).get("full"):
        full_ids.add(str(descriptor["paired_id"]))
        for view in pool_sizes:
            pool_sizes[view].add(materialize_pool(
                descriptor, label, view, fact_index)["pool_size"])

expected = {"train_10k": 10000, "valid_2k": 2000, "test_eviscope": len(full_ids)}
split_ids, split_report = {}, {}
for name, count in expected.items():
    path = args.prepared / "wikidata_big/questions" / f"{name}.pickle"
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    if len(rows) != count:
        raise SystemExit(f"{name}: expected {count}, found {len(rows)}")
    ids = {str(row["_eviscope_paired_id"]) for row in rows}
    if len(ids) != len(rows):
        raise SystemExit(f"{name}: duplicate paired IDs")
    if name == "test_eviscope":
        if ids != full_ids:
            raise SystemExit("Test is not the exact Clean/Full intersection")
        if any(row.get("answers") for row in rows):
            raise SystemExit("Test gold answers leaked into model input")
    split_ids[name] = ids
    split_report[name] = {"rows": len(rows), "sha256": sha256(path)}
if (split_ids["train_10k"] & split_ids["valid_2k"] or
        split_ids["train_10k"] & split_ids["test_eviscope"] or
        split_ids["valid_2k"] & split_ids["test_eviscope"]):
    raise SystemExit("Cross-split paired ID leakage")
leakage = json.loads((args.evi_root / "audit/leakage_report.json").read_text(encoding="utf-8"))
if leakage.get("status") != "pass":
    raise SystemExit("Benchmark leakage audit is not PASS")
if pool_sizes != {"clean": {1000}, "full": {1000}}:
    raise SystemExit(f"Unexpected pool sizes: {pool_sizes}")
report = {
    "status": "pass", "protocol": "paired_clean_full_v1",
    "methods": {
        "MultiQA": "trained once on train_10k; shared checkpoint",
        "RTQA": "published BGE-M3 retrieval stage; inference only",
    },
    "splits": split_report, "test_questions": len(full_ids),
    "pool_sizes": {key: sorted(value) for key, value in pool_sizes.items()},
    "test_gold_in_model_input": False, "release_leakage_audit": "pass",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
