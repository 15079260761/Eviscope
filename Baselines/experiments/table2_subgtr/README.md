# SubGTR paired Clean/Full experiment

This adapter preserves both SubGTR configurations reported in Table 2:
`wikidata_big` and `wikidata_big_complex`. Each configuration is trained once
on a deterministic, stratified 10,000-question sample from its own published
training source, selects its checkpoint on a leakage-audited 2,000-question
development sample, and is evaluated on the same 1,691-question paired Test
cohort under Clean and Full candidate views.

Clean and Full share the same trained checkpoint within a configuration. The
only experimental intervention is the role-blind evidence pool:

- Clean: Anchor + Neutral
- Full: Anchor + Substitute + Distractor + Neutral

Both pools contain exactly 1,000 candidates per question. Test answers are
removed from all model-input pickles and are read only by the official
evaluator. The original SubGTR aware module, time-sensitivity layer, and
top-100 symbolic subgraph reranking remain enabled.

The formal matrix contains four runs:

1. `wikidata_big / clean`
2. `wikidata_big / full`
3. `wikidata_big_complex / clean`
4. `wikidata_big_complex / full`

Run `prepare_inputs.sh`, then `runtime_preflight.sh`, and only launch tmux after
both fail-closed checks pass. Runtime output is unbuffered and printed directly
in each tmux window; no log redirection is used.
