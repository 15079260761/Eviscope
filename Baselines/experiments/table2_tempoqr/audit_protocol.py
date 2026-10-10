"""Fail-closed audit for the six paired TempoQR evaluations."""
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
if len(full_ids) != 1691:
    raise SystemExit(f"Expected 1,691 paired Test questions, found {len(full_ids)}")
if pool_sizes != {"clean": {1000}, "full": {1000}}:
    raise SystemExit(f"Unexpected candidate pool sizes: {pool_sizes}")

expected = {"train_10k": 10000, "valid_2k": 2000, "test_eviscope": 1691}
rows_by_split, split_report = {}, {}
for split, expected_count in expected.items():
    path = args.prepared / "wikidata_big/questions" / f"{split}.pickle"
    rows = load(path)
    if len(rows) != expected_count:
        raise SystemExit(f"{split}: expected {expected_count}, found {len(rows)}")
    ids = {str(row.get("_eviscope_paired_id")) for row in rows}
    if len(ids) != len(rows):
        raise SystemExit(f"{split}: duplicate paired IDs")
    if split == "test_eviscope":
        if ids != full_ids:
            raise SystemExit("Test is not the exact Clean/Full intersection")
        if any(row.get("answers") for row in rows):
            raise SystemExit("Test gold answers leaked into model input")
    rows_by_split[split] = rows
    split_report[split] = {"rows": len(rows), "sha256": sha256(path)}
ids = {split: {str(row["_eviscope_paired_id"]) for row in rows}
       for split, rows in rows_by_split.items()}
if ids["train_10k"] & ids["valid_2k"] or ids["train_10k"] & ids["test_eviscope"] or ids["valid_2k"] & ids["test_eviscope"]:
    raise SystemExit("Cross-split paired-ID leakage")

kg = args.orig_root / "wikidata_big/kg/tkbc_processed_data/wikidata_big"
ent2id, rel2id, ts2id = load(kg / "ent_id"), load(kg / "rel_id"), load(kg / "ts_id")
for split, rows in rows_by_split.items():
    for index, row in enumerate(rows):
        missing_e = [x for x in row.get("entities", []) if x not in ent2id]
        missing_r = [x for x in row.get("relations", []) if x not in rel2id]
        missing_t = [x for x in row.get("times", [])
                     if (int(str(x)[:4]), 0, 0) not in ts2id]
        if missing_e or missing_r or missing_t:
            raise SystemExit(f"Incompatible {split}[{index}]: e={missing_e}, r={missing_r}, t={missing_t}")

leakage = json.loads((args.evi_root / "audit/leakage_report.json").read_text(encoding="utf-8"))
if leakage.get("status") != "pass":
    raise SystemExit("Benchmark leakage audit is not PASS")
report = {
    "status": "pass", "protocol": "tempoqr_three_config_paired_clean_full_v1",
    "configurations": {
        "soft_cronkgqa": {"model": "QA_cronkgqa", "supervision": "soft"},
        "soft_tempoqr": {"model": "QA_TempoQR", "supervision": "soft"},
        "hard_tempoqr": {"model": "QA_TempoQR", "supervision": "hard"},
    },
    "views": ["clean", "full"], "formal_runs": 6,
    "train_samples_per_configuration": 10000,
    "dev_samples_per_configuration": 2000,
    "test_questions_per_run": 1691,
    "candidate_pool_sizes": {key: sorted(value) for key, value in pool_sizes.items()},
    "test_gold_in_model_input": False, "release_leakage_audit": "pass",
    "splits": split_report,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
