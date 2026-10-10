# Public-release and local-retention guide

## Public data asset: required

- `questions/{train,dev,test}.jsonl`
- `supervision/{train,dev}.jsonl`
- `facts/facts.jsonl`
- `candidate_space/descriptors/{train,dev,test}.jsonl`
- `candidate_space/candidate_policy.json`
- `candidate_space/runtime/`
- `evaluator/`
- `src/` (the evaluator imports its frozen encoder and JSONL utilities)
- `statistics/`
- `audit/leakage_report.json`
- `audit/split_consistency_report.json`
- `audit/evidence_validation_report.json`
- `audit/hdm_validation_report.json`
- `audit/release_integrity_report.json`
- `construction_config.json`, `README.md`, `LICENSES.md`, and `VERSION`

## Keep locally, but do not publish for a blind official test

- `private_labels/test.jsonl`

This file is required by the official evaluator. Publishing it does not alter
the already frozen split, but it makes future blind evaluation impossible
because users can tune directly against the test answers and evidence labels.
Publish it only if the project intentionally adopts a fully open, non-blind
test protocol.

## Recommended convenience file

- `cohorts/four_way_common.jsonl`

This is the primary 23,569-question cohort for strictly paired four-way
comparisons. It is derivable from supervision/private labels but is small and
helps users avoid selecting the wrong cohort.

## Optional and reproducible files

- `cohorts/clean_all.jsonl`
- `cohorts/substitute_effect.jsonl`
- `cohorts/distractor_effect.jsonl`

These lists can be reconstructed from `evidence_availability`. They may be
omitted from a minimal public asset, while keeping `four_way_common.jsonl`.

## Construction provenance: archive separately

- `manifests/cronquestions_source_split.jsonl`
- `manifests/paired_question_manifest.jsonl`
- `manifests/exclusion_manifest.jsonl`
- `manifests/manifest_hashes.json`
- `manifests/source_manifest.yaml`
- `audit/construction_log.jsonl`

These files are not consumed by model training, candidate materialization, or
evaluation. They are useful for forensic reconstruction, so retain one cold
archive, but they need not be included in the routine experiment download.

## Redundant configuration copy

- `construction_config.yaml` duplicates the machine-readable JSON config and
  may be omitted if `construction_config.json` is retained.

