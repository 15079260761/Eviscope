# CronKGQA / CTRN × paired Clean/Full conditions

This directory adapts the original CronKGQA and CTRN implementations to the
rebuilt compact CronQuestions benchmark.  It targets the four Table-2 columns
Hits@1, Hits@10, BS@1 and BS@10; it does not claim that new values must equal
the numbers in the original paper because the benchmark was rebuilt.

## Controlled comparison

- Train once per method on a deterministic, type/answer-type stratified 10,000
  example subset of the retained Train split.
- Select the checkpoint on a deterministic 2,000 example Dev subset and train
  for three epochs by default.
- Evaluate every method checkpoint on the exact same full-intersection Test
  cohort: questions eligible for both Clean and Full. Question wording,
  order, candidate-pool size and sampling seed are identical in every cell.
- `clean = anchor + neutral` is the controlled Orig condition.
- `full = anchor + substitute + distractor + neutral`.
- The released benchmark contains 1,691 Clean/Full paired Test questions.
  Using all 24,543 questions for Clean but only 1,691 for Full would be an
  unpaired, mechanism-confounded comparison and is intentionally prohibited.
- The legacy models score entities/times.  The adapter ranks each candidate
  fact by the maximum native score of its subject, object, start and end
  endpoints (mean endpoint score is only a deterministic tie-breaker).  It
  never uses anchor/substitute/distractor roles as model features.
- Prepared Test model inputs have empty `answers`.  Private Test labels are
  opened only by trusted candidate assembly and by the official evaluator.

`audit_table2_protocol.py` fails closed unless the two methods receive
byte-identical prepared inputs, split identities are disjoint, Test model
inputs contain no gold answers, both pools materialize for every selected
Test question, and the release leakage audit is `pass`.

## Required external checkpoint

Both original methods require a pretrained TComplEx checkpoint compatible
with the released `wikidata_big` dictionaries.  Set `TKBC_CHECKPOINT` to its
absolute path.  The runner deliberately stops if it is absent; silently
training against random or mismatched KG embeddings would invalidate the
comparison.

## Environment

Run `bash Baselines/experiments/table2_cronquestions/setup_table2_env.sh`
once on the AutoDL host. It creates an isolated environment on the data disk,
inherits the AutoDL image's matched PyTorch/CUDA stack and installs the other
Python dependencies, caches the
exact DistilBERT assets used by both methods under the project data disk, and
fails if PyTorch cannot see the GPU.

## Two-GPU execution

First run `prepare_table2_inputs.sh` once. Each method has separate Clean and
Full processes, but both conditions reuse that method's one trained checkpoint.
The runner uses cross-session GPU file locks so CronKGQA and CTRN jobs queue
safely on two physical GPUs instead of oversubscribing them.

## Outputs

Predictions and official reports are written under this experiment's `results/`.
After all conditions finish, `results/<run-token>/table2.md` contains four
paper-ready paired rows. Keep `protocol_audit.json`, prediction manifests,
score JSON files, seed, epoch count and model checkpoints with the record.

The default is one fixed seed (`20260926`) because this is the requested fast
run.  For final uncertainty reporting, rerun the complete paired protocol with
additional seeds and report mean ± standard deviation; never mix an Orig run
from one seed with an Evi run from another.
