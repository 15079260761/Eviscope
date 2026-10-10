# EviScope CronQuestions baseline summary

All rows use the same paired test cohort (`n = 1,691`). `Clean` is the
anchor-plus-neutral view, `Full` adds both substitute and distractor facts,
and `Delta = Full - Clean`. Values are macro averages over questions.

| Method | Hits@1 Clean | Hits@1 Full | Delta | Hits@10 Clean | Hits@10 Full | Delta | BS@1 Clean | BS@1 Full | Delta | BS@10 Clean | BS@10 Full | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CronKGQA | 0.441 | 0.313 | -0.128 | 0.708 | 0.697 | -0.012 | 0.031 | 0.011 | -0.020 | 0.140 | 0.082 | -0.058 |
| CTRN | 0.871 | 0.704 | -0.167 | 0.923 | 0.897 | -0.025 | 0.093 | 0.039 | -0.054 | 0.290 | 0.179 | -0.111 |
| SubGTR (wikidata_big) | 0.811 | 0.581 | -0.230 | 0.879 | 0.870 | -0.009 | 0.075 | 0.039 | -0.036 | 0.263 | 0.158 | -0.105 |
| SubGTR (w_big_complex) | 0.252 | 0.160 | -0.092 | 0.410 | 0.370 | -0.041 | 0.067 | 0.038 | -0.029 | 0.153 | 0.117 | -0.036 |
| TempoQR (Soft, CronKGQA) | 0.721 | 0.475 | -0.246 | 0.843 | 0.828 | -0.015 | 0.055 | 0.025 | -0.030 | 0.222 | 0.120 | -0.102 |
| TempoQR (Soft, TempoQR) | 0.868 | 0.707 | -0.160 | 0.928 | 0.882 | -0.046 | 0.091 | 0.038 | -0.053 | 0.300 | 0.184 | -0.116 |
| TempoQR (Hard, TempoQR) | 0.864 | 0.720 | -0.144 | 0.921 | 0.889 | -0.033 | 0.094 | 0.042 | -0.052 | 0.300 | 0.181 | -0.118 |
| MultiQA | 0.836 | 0.681 | -0.156 | 0.915 | 0.880 | -0.035 | 0.086 | 0.037 | -0.049 | 0.277 | 0.177 | -0.100 |
| **RTQA** | **0.975** | **0.876** | -0.098 | **0.996** | **0.988** | -0.008 | **0.169** | **0.100** | -0.069 | **0.364** | **0.288** | -0.076 |

## Interpretation notes

- RTQA is the strongest method in both views on all four primary metrics.
- Every method degrades from Clean to Full. The relatively small Hits@10
  changes, combined with much larger BS@10 changes, indicate that augmented
  evidence more strongly harms complete evidence reconstruction than the
  presence of at least one answer-supporting fact in the top ten.
- The low absolute BS values are largely constrained by the selected cohort.
  It has 59.72 anchors per question on average (median 48, maximum 500), and
  71.61% of questions have more than ten anchors. Therefore, even with perfect
  similarity and matching, the cohort-level upper bounds are only 0.176842 for
  BS@1 and 0.428692 for BS@10.
- In contrast, the complete released test set (`n = 24,543`) has 7.47 anchors
  per question on average and median 1, with upper bounds 0.829827 and
  0.933512. Restricting evaluation to questions eligible for the Full view
  therefore introduces a strong multi-anchor selection effect.
- The frozen BS threshold is high (`tau = 0.871848`). Consequently, a valid
  substitute may satisfy the answer-level Hits definition while receiving a
  zero BS edge when its embedding cosine does not exceed the threshold. This
  occurs for roughly 4% of questions on average at Full Hits@1, depending on
  the method; it is secondary to the much larger anchor-denominator effect.
- Raw BS remains the primary metric required by the paper formulation. A
  ceiling-normalized BS can be reported only as an additional diagnostic, not
  as a replacement for raw BS.

## Additional locally recomputed metrics

`nBS@K = S_K / min(K, |F*(q)|)`. Corrected Coverage counts only positive
edges actually selected by the optimal one-to-one matching. The values below
were recomputed from saved predictions; no model was retrained or rerun.

| Method | nBS@1 Clean | nBS@1 Full | nBS@10 Clean | nBS@10 Full | Cov@1 Clean | Cov@1 Full | Cov@10 Clean | Cov@10 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CronKGQA | 0.441 | 0.304 | 0.470 | 0.332 | 0.031 | 0.011 | 0.140 | 0.083 |
| CTRN | 0.872 | 0.671 | 0.830 | 0.502 | 0.093 | 0.040 | 0.290 | 0.180 |
| SubGTR (wikidata_big) | 0.811 | 0.572 | 0.781 | 0.437 | 0.075 | 0.039 | 0.263 | 0.158 |
| SubGTR (w_big_complex) | 0.252 | 0.159 | 0.242 | 0.176 | 0.067 | 0.038 | 0.153 | 0.118 |
| TempoQR (Soft, CronKGQA) | 0.721 | 0.461 | 0.708 | 0.376 | 0.055 | 0.026 | 0.222 | 0.120 |
| TempoQR (Soft, TempoQR) | 0.868 | 0.673 | 0.851 | 0.561 | 0.091 | 0.039 | 0.300 | 0.185 |
| TempoQR (Hard, TempoQR) | 0.864 | 0.676 | 0.839 | 0.524 | 0.094 | 0.043 | 0.300 | 0.182 |
| MultiQA | 0.836 | 0.643 | 0.807 | 0.530 | 0.086 | 0.037 | 0.277 | 0.177 |
| RTQA | 0.975 | 0.812 | 0.882 | 0.677 | 0.169 | 0.101 | 0.364 | 0.289 |

## Why the paired cohort has many anchors

Question count and anchor count measure different objects. CronQuestions has
set-valued questions: one question can have many correct entities, and each
answer generally contributes at least one supporting anchor. In the paired
cohort, the mean number of gold answers is 59.50 and the mean number of
anchors is 59.72; 1,321 of 1,691 questions have exactly one anchor per gold
answer.

The requirement that a question be eligible for the Full view introduces a
strong selection effect. Full eligibility requires verified substitute and
distractor evidence. A question with many anchors has many more opportunities
to supply both roles. In the complete released test set, Full eligibility is
0.68% for one-anchor questions, 10.00% for 2--10 anchors, 35.96% for 11--50
anchors, and 72.52% for more than 50 anchors. The resulting 1,691-question
paired cohort is therefore small in question count but dense in answers and
anchors.

It is not true that K is smaller than the anchor count for every question.
For K=10 this holds for 71.61% of the paired cohort; for the remaining 28.39%,
`min(K, |F*|)` automatically reduces to the anchor count. Accordingly, nBS is
best described as a capacity-normalized companion metric, while raw BS and
Coverage retain their role as full-evidence reconstruction/coverage metrics.





> 由于固定的 Top-K 列表至多只能与 \(\min(K,|F^*(q)|)\) 个锚点事实进行一对一匹配，原始 BS@K 在多锚点问题上的理论上限显著低于 1。为区分完整证据覆盖率与 Top-K 容量内的证据质量，我们进一步引入 capacity-normalized BS，即以可达到的最大匹配数 \(\min(K,|F^*(q)|)\) 作为归一化分母。

还需要修正一点：并不是所有问题都满足 
$$
K<|F^*|
$$
。

对于 K=10：

- 71.61% 的问题满足 `anchor > 10`；
- 28.39% 的问题满足 `anchor ≤ 10`。

这正是公式使用
$$
 \min(K,|F^*(q)|) 
$$
而不是直接使用 K 的原因：

- 当 `anchor > K`，nBS 衡量 Top-K 容量利用质量；
- 当 `anchor ≤ K`，分母仍然是实际 anchor 数，不会因为证据需求较少而错误惩罚模型。

因此三个指标应当共同解释：

- **Hits@K**：是否至少找到一个有效答案事实；
- **nBS@K**：Top-K 可用匹配容量内的证据质量；
- **Cov@K / BS@K**：对完整 gold evidence 集合的覆盖与重构程度。

nBS 适合作为补充指标解释低 BS，但不应替代原始 BS 和 Coverage。
