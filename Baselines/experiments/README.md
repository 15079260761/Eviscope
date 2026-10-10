# Experiment packages

Cross-method experiments live here instead of as loose files in `Baselines/`.
Each experiment owns one named subdirectory containing its shared protocol,
environment specification, audits, orchestration and result summarization.

Method-specific model, dataset and inference changes remain inside the
corresponding original method directory. A method may expose a small adapter
that imports the shared protocol from its owning experiment package.

Current package:

- `table2_cronquestions/`: CronKGQA and CTRN on paired Orig/Evi
  CronQuestions views for EviScope Table 2.

For later adaptations, create another descriptive experiment subdirectory;
do not add orchestration scripts directly under `Baselines/`.
