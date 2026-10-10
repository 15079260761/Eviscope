# ICEWS-Actor V2 reproducible construction

Run from this directory:

```bash
python -B rebuild_v2.py
```

The orchestrator reads the immutable source at `data/Ori/ICEWS`, reruns the
original 00--09 evidence pipeline, preserves all original quality and leakage
gates, relaxes only the requirement that every released ICEWS question have
both optional evidence roles, applies the frozen V1 HDM threshold/boundaries,
builds compact lazy views, verifies all available modes, refreshes the release
manifest, and removes multi-gigabyte transient intermediates.  Do not invoke
`run_pipeline.py --stage all`: its legacy stage 10 materializes expanded pools,
whereas V2 intentionally uses the equivalent compact descriptor protocol.

The mapping in `v1_id_mapping.jsonl` is used only to preserve the exact V1
four-way question IDs. New V2 IDs are deterministic hashes of source IDs.

# Original pipeline documentation

严格对照 `EviScope数据基准从零构建方案.md`（以原论文为方法参照、工程细节
按该方案补充）从零实现的构建流水线，并遵守《完整目标规模说明》补充要求：
**最终规模 = 两个子基准通过全部验证后实际保留的有效样本之和，绝不人为补齐、
降低标准或凑论文数量**；Train/Dev/Test 的候选、排除、保留数量与排除原因均如实
分阶段、分 split 报告。

## 与规范章节的对应

| 规范章节 | 实现 |
|---|---|
| §4/§5 源划分继承 | `01_lock_source_splits`：`split_EviScope(q)=split_source(q)`，无任何重划分脚本；ICEWS 源 split 值 `validation` 映射为 `dev`；CronQuestions 源 uniq_id 存在跨行复用（已验证 350,000 行中仅 183,280 个唯一值），paired_id 以稳定后缀消歧并记录 |
| §6 清单冻结 | `manifests/<dataset>/`：source_manifest.yaml（含各成员 SHA256）、source_split、paired、exclusion、quarantine 清单 + 清单哈希；后续阶段只读冻结清单 |
| §7 事实规范化 | `02_normalize_facts`：fact_id 含 `time_granularity`；嵌套 `time{time_type,time_start,time_end,time_granularity,raw_time}`；`canonical_fact_text`；别名/逆关系表版本化；完全重复去重；禁止合成事实 |
| §8 问题规范化 | `03_normalize_questions`：paired_id=question_id、`assert benchmark_split == source_split`、仅表面规范化；ICEWS 使用与事件链逐行对齐的原始 `question`，已确认污染的 `paraphrased_question` 只保留作来源审计，不作为模型输入；CronQuestions 读取逻辑不变 |
| §9 family 职责 | `05_build_family_audit`：family_id 仅用于审计；ICEWS 语义实例签名 = 类型+答案类型+时间算子+事件链（答案不参与签名，使冲突标签能够相撞并被隔离），相同模型输入或语义实例跨 split 时整组隔离，同 split 冲突表面整组隔离、真正重复只留首条；CronQuestions 的原有 family 和近重复行为保持不变 |
| §10 Anchor | `04_extract_anchor_facts`：充分性 `Execute(q,F*)=A*` + 最小性 `Execute(q,F*\{fi})≠A*` 逐 anchor 字面检验（ICEWS 逐事件删除执行；Cron 用 F*\{fi} 受限池执行），输出 anchor_count/proof_verified/minimality_verified/verification_version |
| §11/§12 Substitute | `06_recall_candidates`（实体-别名/关系表/BM25/时间邻域多路召回）+ `07_score_substitutability`：`Sub=I_valid·(αS_sem+βS_rel+γS_time)`，α+β+γ=1 冻结；七项硬约束门（否定/核心实体/主客体角色/关系逻辑/时间/答案类型/答案保持）逐项记录；答案保持=字面替换检验（Cron 单步程序在 Anchor 最小性下等价于程序成员资格——有证明记录于注释；ICEWS 逐事件位置字面 swap 执行）；F^sub={Sub≥τ_eq}，每条保存三分数+阈值+valid_gate+answer_preservation+verification |
| §13 Distractor | `08_build_distractors`：`{max_i Sim≥τ_sim ∧ max_i Sub<τ_min ∧ Support=0}`；Support 检查（Cron=支持集成员资格；ICEWS=全位置 swap 检验，能支持者回流 Substitute 流）；顶层类别 Confusable/Contradictory + 子类；不确定者标 `uncertain` 不入正式标签 |
| §13.1 保留条件 | `09_build_evidence_spaces`：源 split 可追溯、Anchor 非空且验证通过、≥1 验证 Substitute、≥1 验证 Distractor、分数与类别完整、配对视图一致；报告 substitute_anchor_coverage；ICEWS 的完整 KG 是共享只读检索语料，孤立事实 ID 跨 split 重叠只报告、不删除，强制隔离对象是 paired_id、模型可见问句表面、语义 family 和问题-证据角色关联；CronQuestions 的既有事实角色净化行为不变 |
| §14 阈值冻结 | 全部阈值 **只在 Train 拟合**（τ_eq/τ_min=Train 有效门候选 Sub 中位数；τ_sim=候选最大 S_sem 中位数；τ_amb=Train anchor 最近非自身余弦中位数；bs_tau=Train anchor–substitute 余弦中位数），Dev 仅确认一次（分布记录，不重拟合），Test 原样消费冻结值；存在未解析 LOCK_BEFORE_BUILD 时流水线终止 |
| §15 HDM | `11_compute_hdm`：**论文原式** `C(q)=|F*(q)|`、`N(fi)={f'∈全量规范化事实空间 | cos≥τ_amb}`（邻域含自身，公式用 |N|−1）、`A(q)=(1/N)Σ(|N(fi)|−1)/|F*|`、`D=C(1+A)`；全空间语义邻域由冻结编码器精确计算（按 distinct anchor 记忆化）；三档边界=Train 三分位，Dev 确认、Test 沿用；连续值+离散等级均发布 |
| §16 配对池 | `10_build_paired_pools`：source_control_pool 与 eviscope_pool 共享 paired_id/split/规模/Anchor/背景采样/种子/格式；EviScope 池以 Substitute/Distractor 替换等量中性背景，绝不删除 Anchor |
| §17 BS@K | `evaluator/`：**论文原式** `M_ij=cos(e(fi),e(rj)) if >τ else 0`、最大权一对一匹配（匈牙利，双侧互斥）、分母=Anchor 数、**无排名折扣**、无重复奖励；Hits@K=检索事实可导出正确答案（Anchor/Substitute 成员，非答案字符串出现）；Coverage@K=被匹配 Anchor 比例；三者同时报告 |
| §19 分 split 构造 | 04/06/07/08 每次只加载一个 split 的问题标签，链式冻结输出；阈值拟合只用 Train |
| §20/§21/§22 审计 | `12_run_audits`：split 一致性、paired_id 跨 split、Test 标签进入 Train、未验证 Substitute、支持答案的 Distractor、Test 私有标签泄露、配对视图一致性、HDM 可复算、BS 单元测试等强制项全部为 0 才允许发布；另报实体/关系/时间重叠与各阶段过滤量 |
| §23/§25 目录与顺序 | `src/00..13_stage.py` + `run_pipeline.py`，configs 5 个文件，manifests/、evaluator/、tests/、logs/ |
| §26 发布结构 | `data/release/<dataset>/`：README/LICENSES/VERSION/source_manifest.yaml/construction_config/manifests/facts/source_view/eviscope_view/candidate_pools/{source_control,eviscope}/private_labels/evaluator/statistics/{source,evidence,hdm,paired}_statistics.json/audit/{split_consistency,evidence_validation,hdm_validation,leakage}_report.json+construction_log.jsonl |

## 冻结编码器（§12.3/§14/§15）

`hashing_token_idf_v1`（revision 1.0.0，含语料哈希）：canonical_fact_text 词元
sha256 哈希到固定维度、语料 IDF 加权、稀疏、L2 归一。跨机器逐字节可复现、无
下载依赖；支持 numpy 快速路径与纯标准库回退。τ_amb/bs_tau 的余弦均由此计算。

## 规模说明（完整目标规模说明补充）

`statistics/source_statistics.json` 逐 split 报告：源候选数、保留数、排除数、
排除原因分布、与论文参考规模的对照；任何阶段不得为凑数放宽验证。
ICEWS 正式全量发布另设失败式验收门：总量 ≥20,000、Train ≥10,000、Dev/Test
各 ≥2,500；低于门槛时流水线中止而不是补样或降低证据有效性要求。显式使用
`--limit-questions`/`--limit-facts` 的 pilot 不受发布门槛约束，也不能作为正式发布。

## 运行

```bash
cd eviscope_builder_v2

# pilot（小规模端到端）
python src/run_pipeline.py --dataset icews_actor --limit-questions 4000
python src/run_pipeline.py --dataset cronquestions --limit-questions 5000

# 完整构建（云端，见 docs/cloud_run_instructions.md）
bash run_all.sh
# 或分数据集 / 分阶段（§25 顺序）
python src/run_pipeline.py --dataset icews_actor --stage 00   # … 到 13
```

评测（§17/§25）：

```bash
python evaluator/evaluate.py --labels data/release/icews_actor/private_labels/test.jsonl \
    --facts data/release/icews_actor/facts/facts.jsonl \
    --config <construction_config.json> \
    --submission my_submission.jsonl --output scores.json -k 10 --stratified
```

提交格式：每行 `{"question_id": ..., "ranked_fact_ids": [...]}`。

## 依赖

Python 3.10+ 标准库；`pyyaml`（可选，内置 fallback 解析器）与 `numpy`
（可选，HDM 邻域快速路径；缺失时自动回退纯 Python）。`run_all.sh` 自动检测。
