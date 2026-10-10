# EviScope-ICEWS-Actor V2

V2 expands the official release from 23,453 to 48,122 anchor-valid questions without changing the frozen 23,453-question four-view experimental cohort. All source splits are inherited. Anchor sufficiency/minimality, answer reproduction, semantic-family isolation, Train-only evidence thresholds, verified evidence roles, and public/private test separation remain enforced.

## View eligibility

Each question declares `mode_eligibility` in its descriptor. Clean is available for all 48,122 questions; substitute-only for 46,504; distractor-only for 24,948; and full EviScope for the unchanged 23,453-question paired cohort. A missing optional role never creates an empty or fabricated evidence list.

## Compatibility

Existing four-view experiments remain tied to `cohorts/four_way_common` and do not require rerunning. New full-release statistics must use V2 counts and must distinguish release size from four-view eligibility. Runtime pools are materialized deterministically from the single fact catalog and descriptors.

## Audits

See `audit/full_semantic_audit.json`, `audit/v2_integrity_report.json`, `audit/public_private_boundary.json`, and `statistics/release_statistics.json`.
The local `construction_builder/` directory contains reproducibility code and is excluded from the immutable release manifest.
