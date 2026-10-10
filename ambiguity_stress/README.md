# Experiment 3 — Controlled Ambiguity Stress Test

This directory implements Section 3 of `文章基础上补充的新实验（小规模）/实验说明.md` for **CronQuestions** and **ICEWS-Actor**.

## Frozen protocol

- Methods: BM25, Dual Encoder, Cross-Encoder, EviRet.
- Neural methods reuse the already-trained Experiment-1 `full` checkpoint. There is no retraining and no fallback to a pretrained base model.
- Training-only empirical terciles define Low `[0,Q33)`, Medium `[Q33,Q66)`, and High `[Q66,1]`.
- Threshold fitting uses the same deterministic 8,000-question full-eligible training cohort as Experiment 1.
- A test question is retained only if its curated, verified distractors contain at least five items in every tercile.
- Every retained question has the same anchors and same neutral background in all conditions. Exactly five distractors are changed; every pool has exactly 1,000 unique facts.
- Substitutes are excluded from these controlled pools and labels because this experiment isolates distractor ambiguity only.
- H@K, BS@K, and Cov@K use the frozen paper evaluator; nBS@K is reported only as the declared auxiliary metric.
- The two main tables exactly follow the requested paper layout and use H@1/BS@1. Complete K=1/10 metrics are retained separately.
- In the generated tables, every delta is explicitly defined as `High - Low`: a positive sign means the High condition scored higher, a negative sign means it scored lower, and values rounding to zero are printed as `0.0000` rather than `-0.0000`.

Cron compact labels do not publish per-distractor scores, so their scores are exactly reconstructed as the maximum anchor–distractor cosine using the release-compatible frozen hashing-token-IDF encoder. ICEWS uses its published `similarity_score`.

## Outputs

- Raw runs: `method_results/<run-id>/<method>/<dataset>/<low|medium|high>/`
- Main tables: `results/<run-id>/table_exp3_cron.{csv,md}` and `table_exp3_icews.{csv,md}`
- Auxiliary metrics: `results/<run-id>/all_metrics_k1_k10.csv`

Run `prepare_protocol.py`, `audit_protocol.py`, then `launch_tmux.sh`. The preparation audit records the actual strict paired cohort size; it is never padded with invalid or synthetic distractors merely to reach 2,000 questions.
