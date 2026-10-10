# EviScope-CronQuestions statistics for the paper

All values below were recomputed from the released JSONL files on 2026-09-26.
The question, label, and descriptor rows are pairwise aligned in every split.

## Dataset scale

| Statistic | Value |
|---|---:|
| Source questions | 410,000 |
| Retained questions | 327,047 |
| Excluded questions | 82,953 |
| Overall retention rate | 79.77% |
| Normalized facts | 326,701 |
| Candidate-pool size per question and mode | 1,000 |
| Stored per-question neutral IDs | 0 |

## Retained questions by type

| Question type | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Simple Entity | 87,878 | 7,461 | 7,524 | 102,863 |
| Simple Time | 59,103 | 4,999 | 4,853 | 68,955 |
| First/Last | 101,575 | 9,418 | 9,413 | 120,406 |
| Before/After | 4,206 | 527 | 524 | 5,257 |
| Time Join | 25,080 | 2,257 | 2,229 | 29,566 |
| **Total** | **277,842** | **24,662** | **24,543** | **327,047** |

## Mode-eligible paired questions

| Candidate-space mode | Train | Dev | Test | Total | Share of release |
|---|---:|---:|---:|---:|---:|
| Clean-Control | 277,842 | 24,662 | 24,543 | 327,047 | 100.00% |
| Substitute-Only | 20,275 | 1,674 | 1,696 | 23,645 | 7.23% |
| Distractor-Only | 277,771 | 24,662 | 24,538 | 326,971 | 99.98% |
| Full-EviScope | 20,204 | 1,674 | 1,691 | 23,569 | 7.21% |

The strict four-way comparison must use the 23,569 Full-EviScope-eligible
questions, because exactly the same paired IDs can then be evaluated under all
four candidate-space modes.

## Hybrid difficulty distribution

| HDM level | Train | Dev | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| Simple | 137,566 | 12,105 | 11,973 | 161,644 | 49.43% |
| Medium | 55,034 | 4,991 | 5,057 | 65,082 | 19.90% |
| Complex | 85,242 | 7,566 | 7,513 | 100,321 | 30.67% |
| **Total** | **277,842** | **24,662** | **24,543** | **327,047** | **100.00%** |

Mean HDM quantities over the release are `C(q)=7.767544`, `A(q)=0.744818`,
and `D(q)=8.673346`. The ambiguity threshold is `tau_amb=0.724076`; the
Train-fitted HDM boundaries are `b1=1.0` and `b2=2.0`.

## Evidence statistics

| Statistic | Construction-time verified candidates | Compact released role references |
|---|---:|---:|
| Anchor | 2,540,352 occurrences | 2,513,738 unique per-question references |
| Substitute | 208,706 | 127,378 (maximum 32 per question) |
| Distractor | 22,491,076 | 11,163,542 (maximum 64 per question) |

The compact-column values describe what an experiment actually reads. The
construction-time values describe all evidence candidates that passed the
verification stages before the per-question release caps.

## Answer type

| Answer type | Train | Dev | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| Entity | 159,436 | 14,781 | 14,814 | 189,031 | 57.80% |
| Time | 118,406 | 9,881 | 9,729 | 138,016 | 42.20% |
| **Total** | **277,842** | **24,662** | **24,543** | **327,047** | **100.00%** |

## Integrity summary

- Leakage audit: PASS.
- Final release integrity audit: PASS.
- Cross-split paired-ID overlaps: 0.
- Cross-split question-surface overlaps: 0.
- Cross-split semantic-family overlaps: 0.
- Candidate descriptors audited: 327,047.
- Dynamically replayed during Stage 12: 5,032 questions.
- All thresholds and HDM boundaries were fitted on Train only.

## HDM consistency note requiring resolution before publication

For 26,614 questions (8.14% of the release), the frozen `reasoning_demand.C_q`
is exactly one larger than the number of unique released
`anchor_fact_indices`: Train 22,851, Dev 1,918, and Test 1,845. Construction
counted a repeated anchor occurrence, whereas compact export deduplicated fact
indices. Of the affected questions, 26,599 are labelled Complex and 15 Medium.

This is not a split leak and does not affect candidate-pool materialization,
but the paper must not claim that the released unique anchor set exactly
recomputes the current HDM values until one of the following is done:

1. define `C(q)` as evidence-slot multiplicity and preserve that multiplicity
   explicitly in the released labels; or
2. recompute HDM from unique anchor facts and update difficulty labels and
   statistics.
