# Experiment 1 — Main benchmark evaluation

## Scope

Four retrieval baselines are evaluated on the **full condition only** for both
released EviScope benchmarks. This is 4 methods × 2 datasets = 8 experimental
runs. Clean, substitute-only and distractor-only comparisons belong to
Experiment 2 and are intentionally neither prepared nor evaluated here.

Methods:

1. `bm25`: lexical BM25 with whole-catalog IDF; no training.
2. `dual_encoder`: vanilla Contriever dual encoder, trained with anchors and
   random/in-batch negatives only.
3. `cross_encoder`: MiniLM cross encoder, trained with anchor positives and
   three random negatives per question.
4. `eviret`: the same Contriever backbone as the dual encoder, with
   evidence-group multi-positive sampling, released distractor negatives and
   random/in-batch negatives. Its per-question HDM weight is the documented
   `1 + (D-D_train_min)/(D_train_max-D_train_min+epsilon)` and uses training
   statistics only.

All neural weights and tokenizers are loaded with `local_files_only=True` from
the root `models` directory.  A missing local model is therefore a hard error,
not a hidden network download.

## Frozen unified protocol

- Seed: `20260927`.
- Candidate pool: exactly 1,000 unique facts per question in the full condition.
- Test cohort: full-eligible questions only. Cron uses all 1,691
  eligible test questions; ICEWS uses a deterministic 2,000-question stratified
  sample from 3,735 eligible questions.
- Stratification: joint question type, HDM difficulty, anchor-count bin,
  substitute-count bin and distractor-count bin, with proportional largest
  remainder allocation and rare-stratum preservation.
- Cron's full-eligible cohort is intrinsically skewed toward complex/time-join
  questions (it contains only 41 simple eligible test questions). We use the
  complete cohort rather than duplicate rare questions or discard most complex
  questions. The generated audit records every stratum count so this source
  property is explicit rather than hidden by resampling.
- Training/dev caps: 8,000/1,000 full-eligible questions, sampled using the
  same rule.  Default neural training is two epochs.
- Training is performed once per method and dataset and evaluated only on full.
- Trusted preprocessing materializes official pools and writes only question
  text and unlabeled candidate indices to `prepared/*/model_input`. Private test
  labels remain in `prepared/*/trusted` and are opened only after predictions
  have been frozen.
- The downloaded ICEWS package stores the fact catalog uncompressed although
  its descriptors retain the immutable `.jsonl.gz` artifact hash. The adapter
  permits this packaging-only mismatch only when `source_lineage.json` attests
  that exact descriptor hash, the frozen full semantic audit has status `pass`,
  and the dense fact count agrees; both hashes are recorded in `audit.json`.
- Cron compact supervision does not publish substitute-to-anchor edges. For
  EviRet training only, each training substitute is attached to its nearest
  anchor with the benchmark's frozen hashing-token-IDF encoder. ICEWS uses its
  explicit released edges. This reconstruction never uses test labels.

## Metrics

At K=1 and K=10 the evaluator reports:

- Hits@K: whether at least one verified anchor or substitute is retrieved.
- BS@K: thresholded cosine maximum-weight one-to-one matching divided by the
  full number of anchors.
- Cov@K: number of anchors with a **positive selected matching edge**, divided
  by the number of anchors. Zero-weight Hungarian assignments do not count.
- nBS@K: the same matching numerator divided by `min(K, |F*(q)|)`.

BS/Cov use the release-compatible frozen hashing-token-IDF encoder and the
dataset thresholds (`0.871848` for Cron, `0.349259` for ICEWS). There is no rank
discount and no double credit.

## Output

Each `results/RUN_ID/METHOD/DATASET/full` directory contains
`predictions.jsonl`, `metrics_k1.json`, `metrics_k10.json` and per-question metric
files. `summary.json` and `run_manifest.json` record the fixed-checkpoint audit.
After all jobs finish, `summarize_results.py` produces the Experiment 1 table as
`table_exp1_full.csv` and `table_exp1_full.md`.

## Server workflow

```bash
cd /path/to/Eviscope
bash main_benchmark/setup_env.sh
source .venv/bin/activate
bash main_benchmark/launch_tmux.sh exp1_v1
```

Two tmux sessions are created: Cron on physical GPU 0 and ICEWS on physical GPU
1. Their four windows are visible, but a `tmux wait-for` chain guarantees only
one job is active per GPU. If a job fails, the chain stops instead of silently
running later methods on incomplete state.

After both sessions report success:

```bash
source .venv/bin/activate
python main_benchmark/summarize_results.py --run-id exp1_v1
```
