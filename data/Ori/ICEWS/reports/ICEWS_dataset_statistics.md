# ICEWS 数据集统计

统计对象：`data/ICEWS/unified_kg_icews_actor.csv` 与
`data/ICEWS/unified_kg_icews_actor_questions_all.csv`。CSV 与 JSON 是同一数据的两种格式，
因此只统计 CSV，避免重复计数。

## 数据集总体规模

| Statistic | Count |
|---|---:|
| Temporal facts | 108,005 |
| Unique entities (subject/object union) | 61,314 |
| Unique subjects | 50,862 |
| Unique objects | 16,217 |
| Relation types | 1 |
| Questions | 89,372 |

唯一关系为 `Affiliation To`。事实 ID 与问题 ID 均无重复。

## 论文主表：问题类别与数据划分

| Difficulty | Question category | Train | Dev | Test | Total |
|---|---|---:|---:|---:|---:|
| Simple | Temporal constraint retrieval | 5,881 | 2,037 | 2,072 | 9,990 |
|  | Timeline recovery | 12,101 | 3,957 | 3,922 | 19,980 |
| **Simple subtotal** |  | **17,982** | **5,994** | **5,994** | **29,970** |
| Medium | Timeline recovery + temporal constraint retrieval | 7,049 | 2,315 | 2,316 | 11,680 |
|  | Timeline recovery + timeline recovery | 8,941 | 3,015 | 3,014 | 14,970 |
| **Medium subtotal** |  | **15,990** | **5,330** | **5,330** | **26,650** |
| Complex | Timeline position retrieval × 3 | 14,315 | 3,642 | 1,967 | 19,924 |
|  | Timeline position retrieval × 2 + temporal constraint retrieval | 9,166 | 2,332 | 1,330 | 12,828 |
| **Complex subtotal** |  | **23,481** | **5,974** | **3,297** | **32,752** |
| **Total** |  | **57,453** | **17,298** | **14,621** | **89,372** |

划分比例为 Train 64.29%、Dev 19.36%、Test 16.36%。源文件将 Dev 写作
`validation`，本文为与常见论文表格保持一致而显示为 Dev。

## 答案类型分布

| Answer type | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Duration | 2,996 | 1,002 | 997 | 4,995 |
| Object | 11,090 | 3,332 | 2,827 | 17,249 |
| Relation duration | 6,562 | 1,951 | 1,458 | 9,971 |
| Relation ranking | 3,571 | 898 | 512 | 4,981 |
| Relation union/intersection | 13,123 | 3,808 | 3,011 | 19,942 |
| Subject | 11,006 | 3,352 | 2,891 | 17,249 |
| End timestamp | 3,032 | 950 | 1,013 | 4,995 |
| Time range | 3,038 | 985 | 972 | 4,995 |
| Start timestamp | 3,035 | 1,020 | 940 | 4,995 |
| **Total** | **57,453** | **17,298** | **14,621** | **89,372** |

## LaTeX（booktabs）

```latex
\begin{table}[t]
\centering
\caption{Statistics of question categories in EviScope-ICEWS.}
\label{tab:icews_statistics}
\begin{tabular}{llrrrr}
\toprule
Difficulty & Question category & Train & Dev & Test & Total \\
\midrule
Simple
 & Temporal constraint retrieval & 5,881 & 2,037 & 2,072 & 9,990 \\
 & Timeline recovery & 12,101 & 3,957 & 3,922 & 19,980 \\
\cmidrule(lr){2-6}
 & \textit{Subtotal} & 17,982 & 5,994 & 5,994 & 29,970 \\
\midrule
Medium
 & Timeline recovery + temporal constraint retrieval & 7,049 & 2,315 & 2,316 & 11,680 \\
 & Timeline recovery + timeline recovery & 8,941 & 3,015 & 3,014 & 14,970 \\
\cmidrule(lr){2-6}
 & \textit{Subtotal} & 15,990 & 5,330 & 5,330 & 26,650 \\
\midrule
Complex
 & Timeline position retrieval $\times 3$ & 14,315 & 3,642 & 1,967 & 19,924 \\
 & Timeline position retrieval $\times 2$ + temporal constraint retrieval
 & 9,166 & 2,332 & 1,330 & 12,828 \\
\cmidrule(lr){2-6}
 & \textit{Subtotal} & 23,481 & 5,974 & 3,297 & 32,752 \\
\midrule
\multicolumn{2}{l}{\textbf{Total}} & \textbf{57,453} & \textbf{17,298}
& \textbf{14,621} & \textbf{89,372} \\
\bottomrule
\end{tabular}
\end{table}
```

## 数据质量说明

- `data_description.txt` 声称划分名称为 `train/dev/test`，但 CSV 实际使用
  `train/validation/test`。
- 日期字段的原始显式范围为 `0017-01-10` 至 `3018-03-23`。其中共有 10 个日期的
  年份早于 1000 或晚于 2099（年份为 17、970、2203、2989、2994、2996、2998、3018），
  很可能含录入或清洗异常，因此不建议未经核查就把该范围写成数据集覆盖年代。
- 49,857 条事实使用开放起点，66,969 条事实使用开放终点。
- 原始 `question_type` 字段含拼写 `constrainted` 和 `retrievel`；表中只对显示名称做了
  英文拼写规范化，计数严格依据原始值。
