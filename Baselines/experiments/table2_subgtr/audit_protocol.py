"""Fail-closed audit for both SubGTR configurations and paired Test views."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--evi-root", type=Path, required=True)
p.add_argument("--orig-root", type=Path, required=True)
p.add_argument("--prepared", type=Path, required=True)
p.add_argument("--subgtr-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()


def load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def iter_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def surface(row):
    value = row.get("question") or (row.get("paraphrases") or [""])[0]
    return " ".join(str(value).lower().split())


def family(row):
    return (tuple(sorted(map(str, row.get("entities", [])))),
            tuple(sorted(map(str, row.get("relations", [])))),
            tuple(sorted(map(str, row.get("times", [])))),
            str(row.get("type")), str(row.get("answer_type")))


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
if len(full_ids) != 1691:
    raise SystemExit(f"Expected the frozen 1,691-question paired cohort, found {len(full_ids)}")
if pool_sizes != {"clean": {1000}, "full": {1000}}:
    raise SystemExit(f"Unexpected candidate pool sizes: {pool_sizes}")

split_report, loaded = {}, {}
for variant in ("wikidata_big", "wikidata_big_complex"):
    loaded[variant] = {}
    for split, count in (("train_10k", 10000), ("valid_2k", 2000),
                         ("test_eviscope", 1691)):
        path = args.prepared / variant / "questions" / f"{split}.pickle"
        rows = load(path)
        if len(rows) != count:
            raise SystemExit(f"{variant}/{split}: expected {count}, found {len(rows)}")
        if split == "test_eviscope":
            ids = {str(row.get("_eviscope_paired_id")) for row in rows}
            if ids != full_ids or len(ids) != len(rows):
                raise SystemExit(f"{variant}: Test is not the exact paired cohort")
            if any(row.get("answers") for row in rows):
                raise SystemExit(f"{variant}: Test gold leaked into model inputs")
        loaded[variant][split] = rows
        split_report[f"{variant}/{split}"] = {"rows": len(rows), "sha256": digest(path)}

std_test = loaded["wikidata_big"]["test_eviscope"]
complex_test = loaded["wikidata_big_complex"]["test_eviscope"]
if [row.get("_eviscope_paired_id") for row in std_test] != [
        row.get("_eviscope_paired_id") for row in complex_test]:
    raise SystemExit("The two SubGTR configurations do not share identical Test order")
test_surfaces, test_families = {surface(x) for x in std_test}, {family(x) for x in std_test}
for variant in ("wikidata_big", "wikidata_big_complex"):
    train, dev = loaded[variant]["train_10k"], loaded[variant]["valid_2k"]
    if {surface(x) for x in train} & test_surfaces:
        raise SystemExit(f"{variant}: Train/Test exact-surface leakage")
    if {surface(x) for x in dev} & test_surfaces:
        raise SystemExit(f"{variant}: Dev/Test exact-surface leakage")
    if {surface(x) for x in train} & {surface(x) for x in dev}:
        raise SystemExit(f"{variant}: Train/Dev surface leakage")
    if variant == "wikidata_big_complex":
        if {family(x) for x in train} & test_families or {family(x) for x in dev} & test_families:
            raise SystemExit("wikidata_big_complex: semantic-family leakage")
    else:
        train_ids = {str(x.get("_eviscope_paired_id")) for x in train}
        dev_ids = {str(x.get("_eviscope_paired_id")) for x in dev}
        if train_ids & dev_ids or train_ids & full_ids or dev_ids & full_ids:
            raise SystemExit("wikidata_big: inherited source-split ID leakage")

kg = args.orig_root / "wikidata_big/kg"
ent2id = load(kg / "tkbc_processed_data/wikidata_big/ent_id")
rel2id = load(kg / "tkbc_processed_data/wikidata_big/rel_id")
ts2id = load(kg / "tkbc_processed_data/wikidata_big/ts_id")
event2time = load(args.subgtr_root / "saved_pkl/event2time.pkl")
for variant, splits in loaded.items():
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            missing_e = [x for x in row.get("entities", []) if x not in ent2id]
            missing_r = [x for x in row.get("relations", []) if x not in rel2id]
            missing_t = [x for x in row.get("times", [])
                         if (int(str(x)[:4]), 0, 0) not in ts2id]
            event = row.get("annotation", {}).get("event_head")
            missing_event = bool(event and str(event).startswith("Q") and not event2time.get(event))
            if missing_e or missing_r or missing_t or missing_event:
                raise SystemExit(
                    f"Incompatible model input {variant}/{split}[{index}]: "
                    f"entities={missing_e}, relations={missing_r}, times={missing_t}, event={event}")

leakage = json.loads((args.evi_root / "audit/leakage_report.json").read_text(encoding="utf-8"))
if leakage.get("status") != "pass":
    raise SystemExit("Benchmark leakage audit is not PASS")
report = {
    "status": "pass", "protocol": "subgtr_standard_and_complex_paired_clean_full_v1",
    "configurations": ["wikidata_big", "wikidata_big_complex"],
    "views": ["clean", "full"], "formal_runs": 4,
    "train_samples_per_configuration": 10000, "dev_samples_per_configuration": 2000,
    "test_questions_per_run": len(full_ids),
    "candidate_pool_sizes": {key: sorted(value) for key, value in pool_sizes.items()},
    "shared_test_order": True, "test_gold_in_model_input": False,
    "complex_validation_overlap_removed": True,
    "release_leakage_audit": "pass", "splits": split_report,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
