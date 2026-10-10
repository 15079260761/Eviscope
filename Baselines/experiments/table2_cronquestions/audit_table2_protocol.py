"""Fail-closed checks for the paired CronKGQA/CTRN Table-2 protocol."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--orig-root", type=Path, required=True)
parser.add_argument("--evi-root", type=Path, required=True)
parser.add_argument("--cron-prepared", type=Path, required=True)
parser.add_argument("--ctrn-prepared", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


names = {"train": "train_10k", "dev": "valid_2k", "test": "test_eviscope"}
report = {"status": "pass", "checks": {}, "mode_eligibility_counts": {}}
seen = {}


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def iter_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


sys.path.insert(0, str(args.evi_root))
from tools.candidate_space.runtime.pool_views import (  # noqa: E402
    MODES, ReleaseFactIndex, materialize_pool,
)

EXPERIMENT_MODES = ("clean", "full")

facts = ReleaseFactIndex(args.evi_root / "data" / "facts" / "facts.jsonl")
descriptors = iter_jsonl(args.evi_root / "data" / "descriptors" / "test.jsonl")
private_labels = iter_jsonl(args.evi_root / "data" / "private_labels" / "test.jsonl")
full_ids = set()
pool_sizes = {mode: set() for mode in EXPERIMENT_MODES}
for descriptor, labels in zip(descriptors, private_labels, strict=True):
    if descriptor.get("paired_id") != labels.get("paired_id"):
        raise SystemExit("Test descriptor/private-label pairing mismatch")
    eligibility = descriptor.get("mode_eligibility", {})
    for mode in MODES:
        if eligibility.get(mode):
            report["mode_eligibility_counts"][mode] = \
                report["mode_eligibility_counts"].get(mode, 0) + 1
    if eligibility.get("full"):
        full_ids.add(str(descriptor["paired_id"]))
        for mode in EXPERIMENT_MODES:
            pool = materialize_pool(descriptor, labels, mode, facts)
            pool_sizes[mode].add(int(pool["pool_size"]))
if not full_ids:
    raise SystemExit("No full-intersection Test cohort")

expected = {"train": 10_000, "dev": 2_000, "test": len(full_ids)}
report["evaluation_protocol"] = "paired_clean_full_v1"
report["evaluation_cohort_size"] = len(full_ids)
report["pool_modes"] = list(EXPERIMENT_MODES)
report["pool_sizes_by_mode"] = {mode: sorted(values) for mode, values in pool_sizes.items()}


kg_root = args.orig_root / "wikidata_big" / "kg" / "tkbc_processed_data" / "wikidata_big"
ent2id = load_pickle(kg_root / "ent_id")
ts2id = load_pickle(kg_root / "ts_id")


def time_key(value):
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(part) for part in value)
    return (int(str(value).strip()), 0, 0)


def validate_model_mappings(rows, split):
    mapped = {"question_entities": 0, "question_times": 0,
              "answer_entities": 0, "answer_times": 0}
    for row_index, row in enumerate(rows):
        for entity in row.get("entities", []):
            if entity not in ent2id:
                raise SystemExit(f"Unmapped question entity in {split}[{row_index}]: {entity!r}")
            mapped["question_entities"] += 1
        for value in row.get("times", []):
            key = time_key(value)
            if key not in ts2id:
                raise SystemExit(f"Unmapped question time in {split}[{row_index}]: {value!r} -> {key!r}")
            mapped["question_times"] += 1
        if split == "test":
            continue
        answer_type = row.get("answer_type")
        for answer in row.get("answers", []):
            if answer_type == "entity":
                if answer not in ent2id:
                    raise SystemExit(f"Unmapped entity answer in {split}[{row_index}]: {answer!r}")
                mapped["answer_entities"] += 1
            elif answer_type == "time":
                key = time_key(answer)
                if key not in ts2id:
                    raise SystemExit(f"Unmapped time answer in {split}[{row_index}]: {answer!r} -> {key!r}")
                mapped["answer_times"] += 1
            else:
                raise SystemExit(f"Unknown answer_type in {split}[{row_index}]: {answer_type!r}")
    return mapped


for split, name in names.items():
    left = args.cron_prepared / "wikidata_big" / "questions" / f"{name}.pickle"
    right = args.ctrn_prepared / "wikidata_big" / "questions" / f"{name}.pickle"
    if sha256(left) != sha256(right):
        raise SystemExit(f"Prepared inputs differ between methods: {split}")
    rows = load_pickle(left)
    if len(rows) != expected[split]:
        raise SystemExit(f"Unexpected {split} size: {len(rows)}")
    ids = {row["_eviscope_paired_id"] for row in rows}
    if len(ids) != len(rows) or any(row["_eviscope_source_split"] != split for row in rows):
        raise SystemExit(f"Invalid identity/split inheritance: {split}")
    if split == "test" and any(row.get("answers") for row in rows):
        raise SystemExit("Test model input contains gold answers")
    if split == "test" and {str(value) for value in ids} != full_ids:
        raise SystemExit("Test inputs are not exactly the paired clean/full cohort")
    seen[split] = ids
    report["checks"][split] = {
        "rows": len(rows),
        "sha256": sha256(left),
        "model_dictionary_mappings": validate_model_mappings(rows, split),
    }
if any(seen[a] & seen[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
    raise SystemExit("Cross-split paired_id overlap")

leakage = json.loads((args.evi_root / "audit" / "leakage_report.json").read_text(encoding="utf-8"))
if leakage.get("status") != "pass":
    raise SystemExit("Released benchmark leakage audit is not PASS")

for prepared in (args.cron_prepared, args.ctrn_prepared):
    labels_path = prepared / "evaluation" / "test_labels_full_intersection.jsonl"
    label_ids = {str(row["paired_id"]) for row in iter_jsonl(labels_path)}
    if label_ids != full_ids or len(label_ids) != len(full_ids):
        raise SystemExit(f"Invalid full-intersection evaluation labels: {labels_path}")
report["checks"]["release_leakage_audit"] = "pass"
report["checks"]["test_gold_in_model_input"] = False
report["checks"]["clean_full_pairing"] = "pass"
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
