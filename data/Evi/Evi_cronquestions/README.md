# EviScope-cronquestions (3.0.0-cron-compact)

- Construction spec: EviScope数据基准从零构建方案.md (paper-verbatim Evidence Space Reconstruction / Hybrid Difficulty Modeling / BS@K, source splits inherited, train-only threshold fitting).
- Scale: actual valid samples after all validation gates — see statistics/source_statistics.json for per-split candidate/excluded/retained counts and reasons (完整目标规模说明).
- Storage: facts/facts.jsonl stores the fact catalog once. Per-split candidate_space/descriptors/*.jsonl files contain tiny deterministic sampling recipes and store zero neutral-fact lists. Trusted supervision uses dense fact indices. The runtime materializer emits an unlabeled fixed-size candidate list for clean, substitute_only, distractor_only or full.
- Supervision: supervision/ contains Train/Dev labels. Test answers and gold evidence remain in private_labels/test.jsonl for the official evaluator; questions/test.jsonl is model-safe.
- Example (trusted preprocessing): python -m candidate_space.runtime.pool_views --descriptors candidate_space/descriptors/test.jsonl --labels private_labels/test.jsonl --facts facts/facts.jsonl --mode full --output work/test_full.jsonl. Expose only candidate_fact_indices (or use --emit-fact-ids) to the model, never the trusted labels.
- Audits: audit/leakage_report.json must be status=pass for release.
