# Experiment 2 — Evidence-Space Component Validation

This experiment keeps two causal questions separate.

## A. `benchmark_side_test_settings`

The Experiment 1 checkpoint is frozen. Only two rankings are produced: the
released `substitute_only` pool without the distractor challenge, and `full`
with the challenge. Original/+Sub share the first ranking; +Dis/Full share the
second. The evaluator recognizes substitutes only for +Sub/Full. Thus S changes
evaluation only and D changes the candidate environment only.

## B. `training_side_train_settings`

This is the exact Section 2.7 EviRet training ablation. The Full test benchmark
is fixed and five freshly trained EviRet variants are compared:

| Variant | Substitute positives | Constructed distractor negatives | HDM |
|---|---:|---:|---:|
| `anchor_only` | no | no | no |
| `sub` | yes | no | no |
| `dis` | no | yes | no |
| `sub_dis` | yes | yes | no |
| `full_eviret` | yes | yes | yes |

Every variant consumes the same 8,000 unique questions. Existing checkpoints
or result directories cause a hard failure: the runner never silently reuses
or overwrites a training run. Each result manifest records the training-ID and
training-label hashes, flags, epoch count, batch size and expected optimizer
steps.

Both directions use Cron and ICEWS, seed 20260927, 1,000 candidates, the
paper-defined Hits/BS/Cov, and auxiliary nBS. Under the released anchor-only
BS definition, changing substitute validity while holding rankings fixed can
change Hits but cannot change BS/Cov/nBS; identical Original/+Sub values for
those three metrics are therefore structural and explicitly marked. Run `prepare_protocol.py`,
`audit_protocol.py`, then `launch_tmux.sh`; `summarize_results.py` emits two
separate tables.

## Server launch

The second argument must be the completed Experiment 1 run ID because the
benchmark-side table freezes those exact neural checkpoints.

```bash
cd /path/to/Eviscope
source .venv/bin/activate
bash evidence_space/launch_tmux.sh exp2_v2 exp1_full_v1
```

This creates `eviscope_exp2_v2_cron_gpu0` and `eviscope_exp2_v2_icews_gpu1`. Each has
five visible windows (four benchmark-side methods and one sequential five-row
EviRet training-ablation window), serialized within its physical GPU. The two
datasets run in parallel.

After both sessions finish:

```bash
python evidence_space/audit_results.py --run-id exp2_v2
python evidence_space/summarize_results.py --run-id exp2_v2
```

Outputs are `table_2a_benchmark_side.{csv,md}` (the CSV contains explicit
`@1` and `@10` columns), `table_2a_benchmark_side_all_metrics.md`, and
`table_2b_training_ablation.{csv,md}`. The old `exp2_v1` four-method by
four-condition training table is exploratory and must not be presented as the
Section 2.7 experiment.
