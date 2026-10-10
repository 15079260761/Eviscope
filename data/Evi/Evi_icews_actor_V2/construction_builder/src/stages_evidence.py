"""Evidence-space stages per spec sections 9-16:
04_extract_anchor_facts, 05_build_family_audit, 06_recall_candidates,
07_score_substitutability, 08_build_distractors, 09_build_evidence_spaces.

Construction runs per split with frozen parameters (section 19); thresholds
are fitted on Train only, confirmed on Dev once, consumed frozen by Test
(section 14).  Every excluded question is recorded in the exclusion manifest
with a reason (sections 4/13.1 and the 完整目标规模说明 supplement).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .bm25 import Bm25Stats
from .common import (
    canonical_json, fact_id_digest, iter_jsonl, rewrite_jsonl, sha256_text,
    write_json, write_jsonl,
)
from .encoder import FrozenEncoder
from .executors import (
    FactIndex, execute_cron, execute_icews, icews_event_from_fact, IcewsEvent,
    _derive_cron,
)
from .stages_source import _manifest_dir, append_manifest
from .timeutils import interval_overlap, parse_date, sentinel_date


def _exclude(root: Path, dataset: str, paired_id: str, stage: str, reason: str) -> None:
    append_manifest(_manifest_dir(dataset, root) / "exclusion_manifest.jsonl", {
        "paired_id": paired_id, "stage": stage, "reason": reason,
    })


def _temporal_iou(anchor: dict[str, Any], fact: dict[str, Any], index: FactIndex) -> float:
    """S_time = |ti ∩ t'| / |ti ∪ t'| (section 12.3).  Uses precomputed
    ordinals; unknown bounds are open, never replaced by arbitrary extremes:
    non-overlap is 0, any unknown bound yields the neutral 0.5."""
    a0, a1 = index.ordinals.get(anchor["fact_id"], (None, None))
    b0, b1 = index.ordinals.get(fact["fact_id"], (None, None))
    overlap = (a1 is None or b0 is None or a1 >= b0) and (b1 is None or a0 is None or b1 >= a0)
    if not overlap:
        return 0.0
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return 0.5
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return round(inter / union, 6) if union else 1.0


# --------------------------------------------------------------------------- #
# 04_extract_anchor_facts
# --------------------------------------------------------------------------- #

def _cron_anchors(question: dict[str, Any], index: FactIndex,
                  truncation_cap: int | None) -> tuple[list[str], dict[str, list[str]], bool, list[str] | None]:
    """Anchor facts from the deterministic executor: one anchor per role,
    all other role facts are substitute candidates (member supports)."""
    groups, program, error, expanded = _derive_cron(question, index,
                                                    truncation_cap=truncation_cap)
    if error:
        return [], {}, False, None
    anchors: list[str] = []
    supports: dict[str, list[str]] = {}
    for group in groups:
        members = sorted(set(group.get("support_fact_ids", []))
                         | set(group.get("candidate_substitute_fact_ids", [])))
        if not members:
            continue
        anchor = members[0]
        anchors.append(anchor)
        supports[anchor] = members
    return anchors, supports, True, expanded


def extract_anchor_facts(config: dict[str, Any], questions_path: Path, split: str,
                         index: FactIndex, output: Path, root: Path,
                         report_path: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    cap = config.get("answer_truncation_cap")
    stats: Counter[str] = Counter()

    def gen():
        for q in iter_jsonl(questions_path):
            if q["benchmark_split"] != split or q.get("excluded"):
                yield q
                continue
            stats["questions"] += 1
            yield _extract_one(config, q, index, root, cap, stats)

    write_jsonl(output, gen())
    report = {"split": split, **dict(stats)}
    write_json(report_path, report)
    return report


def _extract_one(config: dict[str, Any], q: dict[str, Any], index: FactIndex,
                 root: Path, cap: int | None, stats: Counter[str]) -> dict[str, Any]:
    """Anchor extraction for a single question (streamed per question)."""
    dataset = config["dataset"]
    reason = None
    if dataset == "icews_actor":
        events = [IcewsEvent(e) for e in (q.get("source_events") or [])]
        anchors: list[str] = []
        supports: dict[str, list[str]] = {}
        answer = execute_icews(q, events) if events else None
        if answer != (q["gold_answers"][0].strip() if q["gold_answers"] else None):
            reason = "anchor_sufficiency_failed"
        else:
            for position, event in enumerate(events):
                fid = index.by_raw_tuple.get((event.subject, event.relation,
                                              event.object, event.raw_ts, event.raw_te))
                if not fid:
                    reason = "source_event_not_found"
                    break
                anchors.append(fid)
                supports[fid] = [fid]
            if not reason:
                # minimality (section 10): removing any anchor must break
                # the answer
                minimal = True
                for position in range(len(events)):
                    reduced = [events[i] for i in range(len(events)) if i != position]
                    if execute_icews(q, reduced) == q["gold_answers"][0].strip():
                        minimal = False
                        break
                if not minimal:
                    reason = "anchor_minimality_failed"
                elif not anchors:
                    reason = "empty_anchor_set"
    else:
        anchors, supports, ok, expanded = _cron_anchors(q, index, cap)
        if not ok:
            reason = "executor_cannot_reproduce_answers"
        else:
            if expanded is not None:
                # source answer sets were truncated at the cap; the
                # re-executed (complete) set is authoritative (section 8
                # + 完整目标规模说明: no padding, executor decides)
                q["gold_answers"] = expanded
                q["answer_revision"] = "truncated_source_expanded"
                stats["re_executed_answers"] += 1
            # minimality per anchor (section 10): removing the anchor from
            # the PROOF (F* \ {f_i}, not the whole corpus) must change the
            # reproduced answers
            minimal = True
            for anchor in anchors:
                reduced_pool = set(anchors) - {anchor}
                derived = execute_cron(q, index, pool=reduced_pool)
                if derived == set(q["gold_answers"]):
                    minimal = False
                    break
            if not minimal:
                reason = "anchor_minimality_failed"
            elif not anchors:
                reason = "empty_anchor_set"
    if reason:
        stats["excluded"] += 1
        stats[f"reason:{reason}"] += 1
        q["excluded"] = True
        q.setdefault("exclusion_reasons", []).append(reason)
        q["anchor_fact_ids"] = []
        q["proof_verified"] = False
        q["minimality_verified"] = False
        _exclude(root, dataset, q["paired_id"], "04_extract_anchor_facts", reason)
    else:
        q["anchor_fact_ids"] = anchors
        q["anchor_count"] = len(anchors)
        q["proof_verified"] = True
        q["minimality_verified"] = True
        q["verification_version"] = "anchor_v1"
        q["anchor_support_fact_ids"] = supports  # construction-internal
        stats["verified"] += 1
    return q


# --------------------------------------------------------------------------- #
# 05_build_family_audit
# --------------------------------------------------------------------------- #

def build_family_audit(config: dict[str, Any], questions_path: Path,
                       output: Path, root: Path, report_path: Path) -> dict[str, Any]:
    """family_id is audit-only (section 9): it binds source paraphrase groups
    and feeds the near-duplicate audit; it never drives the split."""
    dataset = config["dataset"]
    stats: Counter[str] = Counter()

    def icews_instance(q: dict[str, Any]) -> tuple[Any, ...]:
        """Answer-independent semantic instance used for leakage control.

        The event chain and temporal operator determine the executable query.
        Gold answers are deliberately omitted so conflicting labels for the
        same instance collide and are quarantined instead of being treated as
        separate families.
        """
        return (
            q.get("template"),
            q.get("question_type"),
            q.get("answer_type"),
            q.get("temporal_operator"),
            tuple(sorted(q.get("source_events", []))),
        )

    def family_of(q: dict[str, Any]) -> str:
        if dataset == "icews_actor":
            signature = sha256_text(canonical_json(icews_instance(q)))
            return f"{dataset}:instance:{signature}"
        # the CronQuestions release reuses uniq_id for unrelated rows
        # (verified 183,280 unique among 350,000 train rows), so the
        # family key is split-scoped; documented source-data finding.
        return f"{dataset}:{q['source_question_id']}:{q['benchmark_split']}"

    # pass 1: near-duplicate signature maps (small tuples only, not full rows)
    sig_by_key: dict[tuple, list[tuple[str, str, int, tuple[str, ...], tuple[Any, ...]]]] = \
        defaultdict(list)
    for q in iter_jsonl(questions_path):
        if q.get("excluded"):
            continue
        source_row = (q.get("source_provenance") or {}).get("source_row", 0)
        answers = tuple(sorted(q.get("gold_answers", [])))
        instance = icews_instance(q) if dataset == "icews_actor" else ()
        sig_by_key[("norm", q["normalized_question"])].append(
            (q["paired_id"], q["benchmark_split"], source_row, answers, instance))
        if dataset == "icews_actor":
            # Same executable instance with a different answer is a label
            # conflict, not a new family.  The conflict is handled below.
            template_key = sha256_text(canonical_json(instance))
        else:
            instance = [q.get("template"), q.get("question_type"),
                        tuple(sorted(q.get("topic_entity_ids", []))),
                        tuple(sorted(q.get("time_constraints", []))),
                        tuple(sorted(q.get("gold_answers", [])))]
            template_key = sha256_text(canonical_json(instance))
        sig_by_key[("tmpl", template_key)].append(
            (q["paired_id"], q["benchmark_split"], source_row, answers,
             tuple(instance) if isinstance(instance, (list, tuple)) else ()))
    to_exclude: set[str] = set()
    for (kind, _key), rows_ in sig_by_key.items():
        if len(rows_) <= 1:
            continue
        splits = {split for _pid, split, _row, _answers, _instance in rows_}
        answer_sets = {answers for _pid, _split, _row, answers, _instance in rows_}
        instances = {instance for _pid, _split, _row, _answers, instance in rows_}
        if len(splits) > 1:
            # Identical model input OR identical executable instance across
            # source splits is a genuine leakage channel.  Source splits are
            # inherited, so the whole collision group is quarantined rather
            # than moved between splits.
            for pid, _split, _row, _answers, _instance in rows_:
                to_exclude.add(pid)
            stats[f"{kind}_cross_split_groups"] += 1
            stats[f"{kind}_cross_split_rows"] += len(rows_)
        elif dataset == "icews_actor" and kind == "norm" \
                and (len(answer_sets) > 1 or len(instances) > 1):
            # The same question text with different semantics/answers is an
            # ambiguous QA item.  Keeping an arbitrary first row would create
            # incomplete gold labels, so quarantine the entire group.
            for pid, _split, _row, _answers, _instance in rows_:
                to_exclude.add(pid)
            stats["ambiguous_surface_same_split_groups"] += 1
            stats["ambiguous_surface_same_split_rows"] += len(rows_)
        elif kind == "norm" or (dataset == "icews_actor" and kind == "tmpl"):
            # True duplicates confined to one split: retain the deterministic
            # first source occurrence and remove redundant copies.
            ordered = sorted(rows_, key=lambda r: (r[2], r[0]))
            for pid, _split, _row, _answers, _instance in ordered[1:]:
                to_exclude.add(pid)
            stats[f"{kind}_duplicates_same_split"] += len(ordered) - 1

    # pass 2: stream rows, assign family ids and exclusions
    family_sizes: Counter[str] = Counter()

    def gen():
        for q in iter_jsonl(questions_path):
            if q["paired_id"] in to_exclude and not q.get("excluded"):
                q["excluded"] = True
                q.setdefault("exclusion_reasons", []).append(
                    "icews_duplicate_ambiguous_or_cross_split"
                    if dataset == "icews_actor"
                    else "near_duplicate_or_cross_split_family")
                _exclude(root, dataset, q["paired_id"], "05_build_family_audit",
                         "icews_duplicate_ambiguous_or_cross_split"
                         if dataset == "icews_actor"
                         else "near_duplicate_or_cross_split_family")
                stats["excluded"] += 1
            if not q.get("excluded"):
                q["family_id"] = family_of(q)
                family_sizes[q["family_id"]] += 1
            yield q

    write_jsonl(output, gen())
    report = {
        "families": len(family_sizes),
        "family_size_histogram": dict(Counter(family_sizes.values())),
        "excluded": stats["excluded"],
        "near_duplicate_cross_split_groups":
            stats["norm_cross_split_groups"] + stats["tmpl_cross_split_groups"],
        "source_near_duplicates_same_split":
            stats["norm_duplicates_same_split"] + stats["tmpl_duplicates_same_split"],
        "details": {key: value for key, value in sorted(stats.items())
                    if key != "excluded"},
    }
    write_json(report_path, report)
    return report


# --------------------------------------------------------------------------- #
# 06_recall_candidates
# --------------------------------------------------------------------------- #

def _cron_candidate_plausible(question: dict[str, Any], anchor: dict[str, Any],
                              fact: dict[str, Any]) -> bool:
    """Cheap structural pattern check (section 12.2 subject/object role): a
    CronQuestions candidate can only support a single-step program if it
    matches the anchor's role pattern.  Used to short-circuit the expensive
    literal swap test for implausible candidates."""
    qtype = question["question_type"]
    ann = question.get("annotation") or {}
    if qtype == "simple_entity":
        year = int(str(ann.get("time", "")))
        from .timeutils import contains_year
        if not contains_year(fact["time_start"], fact["time_end"], year):
            return False
        if "head" in ann:
            return fact["subject_id"] == str(ann["head"])
        return fact["object_id"] == str(ann["tail"])
    if qtype == "simple_time":
        return fact["subject_id"] == str(ann.get("head")) and \
            fact["object_id"] == str(ann.get("tail"))
    if qtype == "first_last":
        if question.get("answer_type") != "time":
            if "head" in ann and "tail" not in ann:
                return fact["subject_id"] == str(ann["head"])
            return fact["object_id"] == str(ann.get("tail", ""))
        if "tail" in ann:
            return fact["subject_id"] == str(ann.get("head")) and \
                fact["object_id"] == str(ann["tail"])
        return fact["subject_id"] == str(ann.get("head"))
    if qtype == "before_after":
        tail = str(ann.get("tail", ""))
        return fact["object_id"] == tail and \
            fact["subject_id"] != str(ann.get("head", ann.get("event_head", "")))
    if qtype == "time_join":
        return fact["object_id"] == str(ann.get("tail", ""))
    return True


def _relation_score(anchor: dict[str, Any], fact: dict[str, Any],
                    config: dict[str, Any]) -> tuple[float, bool]:
    """S_rel with the curated relation table (sections 6.4/11.2/12.3)."""
    if fact["relation_id"] == anchor["relation_id"]:
        return 1.0, True
    relations = config.get("_relations") or {}
    for group in relations.get("equivalence_groups", []):
        if anchor["relation_id"] in group and fact["relation_id"] in group:
            return 0.9, True
    for pair in relations.get("inverse_pairs", []):
        if anchor["relation_id"] in pair and fact["relation_id"] in pair \
                and anchor["relation_id"] != fact["relation_id"]:
            # inverse candidate must have swapped subject/object roles
            if fact["subject_id"] == anchor["object_id"] and fact["object_id"] == anchor["subject_id"]:
                return 0.8, True
    return 0.0, False


def _recall_candidates(anchor: dict[str, Any], index: FactIndex, config: dict[str, Any],
                       top_n: int, bm25: Bm25Stats | None,
                       query_tokens: list[str] | None = None) -> list[str]:
    """Multi-way recall (11.2): exact entity-relation index + curated relation
    table + BM25 over canonical fact text + time-neighborhood preselect."""
    candidates: set[str] = set()
    for entity in (anchor["subject_id"], anchor["object_id"]):
        for fid in index.by_entity.get(entity, []):
            fact = index.facts[fid]
            if fid == anchor["fact_id"]:
                continue
            score, _ok = _relation_score(anchor, fact, config)
            if score > 0 or fact["relation_id"] == anchor["relation_id"]:
                candidates.add(fid)
    if bm25 is not None and candidates:
        query = anchor.get("canonical_fact_text") or anchor.get("fact_text", "")
        scored = [(bm25.score_cached(query, index.doc_tokens.get(fid, []),
                                     index.doc_tf.get(fid, Counter())), fid)
                  for fid in candidates]
        scored.sort(key=lambda x: (-x[0], x[1]))
        candidates = {fid for _, fid in scored[:top_n]}
    else:
        candidates = set(sorted(candidates)[:top_n])
    return sorted(candidates)


def recall_candidates_stage(config: dict[str, Any], questions_path: Path, split: str,
                            index: FactIndex, bm25: Bm25Stats | None, output: Path,
                            report_path: Path) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    # per-anchor recall cap: the distractor budget is per question (13.1),
    # so a per-anchor cap bounds the frozen files while keeping the recall
    # multi-way (11.2); per-question aggregation happens in stage 08
    top_n = min(int(config["recall"]["distractor_top_n"]),
                int(config["recall"].get("per_anchor_top_n", 100)))

    def gen():
        for q in iter_jsonl(questions_path):
            if q["benchmark_split"] != split or q.get("excluded"):
                yield q
                continue
            recall: dict[str, list[str]] = {}
            for anchor_fid in q.get("anchor_fact_ids", []):
                anchor = index.facts.get(anchor_fid)
                if anchor is None:
                    continue
                recall[anchor_fid] = _recall_candidates(anchor, index, config, top_n, bm25)
            q["recall_candidates"] = recall
            stats["questions"] += 1
            stats["candidates"] += sum(len(v) for v in recall.values())
            yield q

    write_jsonl(output, gen())
    report = {"split": split, **dict(stats)}
    write_json(report_path, report)
    return report


# --------------------------------------------------------------------------- #
# 07_score_substitutability
# --------------------------------------------------------------------------- #

def _valid_gate(question: dict[str, Any], anchor: dict[str, Any], fact: dict[str, Any],
                answer_preservation: bool, relation_ok: bool, config: dict[str, Any],
                cron_role_plausible: bool, is_member: bool,
                dataset: str) -> tuple[int, list[str]]:
    """I_valid hard gate (section 12.2): all seven checks recorded.

    For CronQuestions program members, role/logic/time/answer-type correctness
    is implied by the executor-derived support membership; the remaining
    checks (negation, core entity, relation, answer preservation) still run."""
    failures: list[str] = []
    negation_markers = config.get("negation_markers", [])
    anchor_text = (anchor.get("canonical_fact_text") or "").casefold()
    fact_text = (fact.get("canonical_fact_text") or "").casefold()
    anchor_neg = any(m in anchor_text for m in negation_markers)
    fact_neg = any(m in fact_text for m in negation_markers)
    if anchor_neg != fact_neg:
        failures.append("negation_consistency")
    entities_a = {anchor["subject_id"], anchor["object_id"]}
    entities_b = {fact["subject_id"], fact["object_id"]}
    aliases_a = set(anchor.get("aliases", []))
    aliases_b = set(fact.get("aliases", []))
    shared_entity = bool(entities_a & entities_b)
    shared_alias = bool(aliases_a & aliases_b) or bool(
        {a.casefold() for a in aliases_a} & {a.casefold() for a in aliases_b})
    if not (shared_entity or shared_alias):
        failures.append("core_entity_consistency")
    # the subject/object role pattern check is a CronQuestions concept; for
    # ICEWS the literal answer-preservation swap test IS the role check
    if dataset == "cronquestions" and not is_member and not cron_role_plausible:
        failures.append("subject_object_role")
    if not relation_ok:
        failures.append("relation_logic")
    if not is_member and not interval_overlap(anchor.get("time_start"),
                                              anchor.get("time_end"),
                                              fact.get("time_start"),
                                              fact.get("time_end")):
        failures.append("temporal_operator")
    if not is_member and dataset == "cronquestions" \
            and question.get("answer_type") == "time" \
            and not (fact.get("time_start") and fact.get("time_end")):
        failures.append("answer_type")
    if not answer_preservation:
        failures.append("answer_preservation")
    return (1 if not failures else 0), failures


def score_substitutability_stage(config: dict[str, Any], questions_path: Path,
                                 split: str, index: FactIndex, encoder: FrozenEncoder,
                                 output: Path, thresholds_path: Path | None,
                                 fit_path: Path | None, report_path: Path) -> dict[str, Any]:
    """Section 12: Sub(f_i, f') = I_valid * (alpha*S_sem + beta*S_rel +
    gamma*S_time) per (anchor, candidate) pair.  A pair becomes a substitute
    iff the valid gate passes, the answer is preserved, and Sub >= tau_eq.
    Thresholds are fitted on Train only and frozen for Dev/Test.

    Fully streamed: per-question processing with a per-question embedding
    cache; fit statistics are collected in compact typed arrays; the train
    run labels substitutes in a second streaming pass after fitting."""
    from array import array
    dataset = config["dataset"]
    alpha = float(config.get("alpha", 0.5))
    beta = float(config.get("beta", 0.25))
    gamma = float(config.get("gamma", 0.25))
    top_n = config["recall"]["substitute_top_n"]
    thresholds = None
    if thresholds_path is not None and thresholds_path.exists():
        thresholds = next(iter_jsonl(thresholds_path))
    stats: Counter[str] = Counter()
    fit_subs = array("d")            # Sub scores of valid-gated pairs
    fit_max_sims = array("d")        # per-candidate max anchor S_sem
    fit_anchor_sub_cos = array("d")  # (anchor, qualifying pair) cosines

    def process(q: dict[str, Any], label_inline: bool) -> dict[str, Any]:
        if q["benchmark_split"] != split or q.get("excluded"):
            return q
        stats["questions"] += 1
        pair_scores: list[dict[str, Any]] = []
        candidate_scores: dict[str, dict[str, Any]] = {}
        emb_cache: dict[str, dict[int, float]] = {}  # per-question: bounded

        def emb_of(fact: dict[str, Any]) -> dict[int, float]:
            fid = fact["fact_id"]
            if fid not in emb_cache:
                emb_cache[fid] = encoder.embed(fact.get("canonical_fact_text")
                                               or fact.get("fact_text", ""))
            return emb_cache[fid]

        for anchor_fid in q.get("anchor_fact_ids", []):
            anchor = index.facts.get(anchor_fid)
            if anchor is None:
                continue
            anchor_emb = emb_of(anchor)
            members = sorted(set(q.get("anchor_support_fact_ids", {}).get(anchor_fid, []))
                             - {anchor_fid})
            candidate_ids = list(dict.fromkeys(
                members[:top_n] + [f for f in q.get("recall_candidates", {}).get(anchor_fid, [])
                                   if f not in set(members)]))
            for fid in candidate_ids:
                fact = index.facts.get(fid)
                if fact is None:
                    continue
                sem = round(encoder.cosine(anchor_emb, emb_of(fact)), 6)
                rel, relation_ok = _relation_score(anchor, fact, config)
                temp = _temporal_iou(anchor, fact, index)
                is_member = fid in members
                if dataset == "cronquestions":
                    # CronQuestions programs are single-step pattern matches,
                    # and anchors are minimal by construction: the support
                    # sets contain every fact that matches a role pattern, so
                    # the exact swap semantics is program membership —
                    # a candidate preserves the answer iff it is a member
                    # (a member of another group still leaves the replaced
                    # anchor's role uncovered).  The literal executor pass is
                    # therefore redundant for Cron; ICEWS keeps literal swap
                    # tests per event position (12.2).
                    preserved = is_member
                    verification_method = "executor_membership_swap_implication"
                    plausible = is_member or _cron_candidate_plausible(q, anchor, fact)
                else:
                    events = [IcewsEvent(e) for e in (q.get("source_events") or [])]
                    position = q["anchor_fact_ids"].index(anchor_fid)
                    swapped = list(events)
                    swapped[position] = icews_event_from_fact(fact)
                    preserved = execute_icews(q, swapped) == q["gold_answers"][0].strip()
                    verification_method = "literal_swap_test"
                    plausible = True
                gate, failures = _valid_gate(q, anchor, fact, preserved, relation_ok,
                                             config, cron_role_plausible=plausible,
                                             is_member=is_member, dataset=dataset)
                sub = round(gate * (alpha * sem + beta * rel + gamma * temp), 6)
                entry = {
                    "fact_id": fid,
                    "anchor_fact_id": anchor_fid,
                    "valid_gate": gate,
                    "valid_gate_failures": failures,
                    "semantic_score": sem,
                    "relation_score": rel,
                    "temporal_score": temp,
                    "substitutability_score": sub,
                    "answer_preservation": preserved,
                    "verification_method": verification_method,
                }
                # only gate-passing, answer-preserving pairs are kept (they
                # are the substitute candidates; the rest is aggregated)
                if gate and preserved:
                    pair_scores.append(entry)
                prev = candidate_scores.get(fid)
                candidate_scores[fid] = {
                    "max_semantic_score": max(sem, prev["max_semantic_score"]) if prev else sem,
                    "max_sub": max(sub, prev["max_sub"]) if prev else sub,
                    "anchor_fact_id": anchor_fid if (not prev or sem >= prev["max_semantic_score"])
                    else prev["anchor_fact_id"],
                }
                if gate and preserved:
                    fit_subs.append(sub)
                    stats["valid_gated_candidates"] += 1
                    if split == "train":
                        fit_anchor_sub_cos.append(sem)
        for entry in candidate_scores.values():
            fit_max_sims.append(entry["max_semantic_score"])
        q["candidate_scores"] = candidate_scores
        q["pair_scores"] = pair_scores
        if label_inline:
            _label_substitutes(q, split, top_n, thresholds["tau_eq"], stats)
        return q

    write_jsonl(output, (process(q, label_inline=(thresholds is not None))
                         for q in iter_jsonl(questions_path)))
    if fit_path is not None and split == "train":
        subs_sorted = sorted(fit_subs)
        sims_sorted = sorted(fit_max_sims)
        cos_sorted = sorted(fit_anchor_sub_cos)
        tau_eq = subs_sorted[len(subs_sorted) // 2] if subs_sorted else 0.0
        tau_sim = sims_sorted[len(sims_sorted) // 2] if sims_sorted else 0.0
        bs_tau = cos_sorted[len(cos_sorted) // 2] if cos_sorted else 0.0
        thresholds = {
            "tau_eq": round(tau_eq, 6),
            "tau_min": round(tau_eq, 6),
            "tau_sim": round(tau_sim, 6),
            "bs_tau": round(bs_tau, 6),
            "fit_source": "train",
            "fit_n_valid_gated": len(fit_subs),
            "fit_n_candidates": len(fit_max_sims),
            "fit_n_anchor_substitute_pairs": len(fit_anchor_sub_cos),
            "seed": config["construction_seed"],
        }
        write_jsonl(fit_path, [thresholds])
        # second streaming pass: label substitutes with the frozen tau_eq
        rewrite_jsonl(output, output,
                      lambda q: _label_substitutes(q, split, top_n,
                                                   thresholds["tau_eq"], stats))
    report = {"split": split, **dict(stats)}
    write_json(report_path, report)
    return report


def _label_substitutes(q: dict[str, Any], split: str, top_n: int, tau_eq: float,
                       stats: Counter[str]) -> dict[str, Any]:
    """Add substitutable_evidence to one question (streaming helper)."""
    if q["benchmark_split"] != split or q.get("excluded"):
        return q
    qualifying = [entry for entry in q.get("pair_scores", [])
                  if entry["valid_gate"] and entry["answer_preservation"]
                  and entry["substitutability_score"] >= tau_eq]
    qualifying.sort(key=lambda x: (-x["substitutability_score"], x["fact_id"]))
    # one substitute record per (candidate, anchor) pair, capped
    chosen: dict[str, dict[str, Any]] = {}
    for entry in qualifying[:top_n * len(q.get("anchor_fact_ids", []))]:
        if entry["fact_id"] not in chosen:
            chosen[entry["fact_id"]] = {
                "fact_id": entry["fact_id"],
                "anchor_fact_id": entry["anchor_fact_id"],
                "valid_gate": entry["valid_gate"],
                "semantic_score": entry["semantic_score"],
                "relation_score": entry["relation_score"],
                "temporal_score": entry["temporal_score"],
                "substitutability_score": entry["substitutability_score"],
                "threshold": tau_eq,
                "answer_preservation": True,
                "verification": "verified_exact",
            }
    q["substitutable_evidence"] = sorted(chosen.values(),
                                         key=lambda x: (-x["substitutability_score"],
                                                        x["fact_id"]))
    stats["substitutes"] += len(chosen)
    return q


# --------------------------------------------------------------------------- #
# 08_build_distractors
# --------------------------------------------------------------------------- #

def _classify_distractor(anchor: dict[str, Any], fact: dict[str, Any],
                         gold_answers: list[str], config: dict[str, Any]) -> tuple[str, str]:
    """Top-level category + subtype (section 13)."""
    same_s = fact["subject_id"] == anchor["subject_id"]
    same_o = fact["object_id"] == anchor["object_id"]
    same_rel = fact["relation_id"] == anchor["relation_id"]
    negation_markers = config.get("negation_markers", [])
    fact_neg = any(m in (fact.get("canonical_fact_text") or "").casefold()
                   for m in negation_markers)
    if fact_neg:
        return "contradictory", "negation_conflict"
    if same_rel and same_s and fact["object_id"] != anchor["object_id"] \
            and fact["object_id"] not in gold_answers:
        return "contradictory", "answer_conflict"
    if same_s and same_o and same_rel and not interval_overlap(
            anchor.get("time_start"), anchor.get("time_end"),
            fact.get("time_start"), fact.get("time_end")):
        return "contradictory", "time_conflict"
    if gold_answers and (fact["subject_id"] in gold_answers or fact["object_id"] in gold_answers):
        return "confusable", "multi_hop_shortcut"
    if same_rel and same_s and same_o:
        return "confusable", "temporal_confusable"
    if same_rel and (same_s or same_o):
        return "confusable", "entity_confusable"
    if not same_rel and (same_s or same_o):
        return "confusable", "relation_confusable"
    return "confusable", "boundary_confusable"


def build_distractors_stage(config: dict[str, Any], questions_path: Path, split: str,
                            index: FactIndex, output: Path, root: Path,
                            report_path: Path) -> dict[str, Any]:
    """Section 13: F^dis = {max_i Sim >= tau_sim AND max_i Sub < tau_min AND
    Support(q, f') = 0}.  Supporting candidates re-enter the substitute flow;
    undecidable ones are marked uncertain, never labeled."""
    dataset = config["dataset"]
    stats: Counter[str] = Counter()
    thresholds_path = output.parent / "07_thresholds.jsonl"
    thresholds = next(iter_jsonl(thresholds_path)) if thresholds_path.exists() else None
    if thresholds is None:
        raise ValueError("07 thresholds not frozen; run 07 (train) first")
    tau_sim = thresholds["tau_sim"]
    tau_min = thresholds["tau_min"]
    tau_eq = thresholds["tau_eq"]

    def gen():
        for q in iter_jsonl(questions_path):
            if q["benchmark_split"] != split or q.get("excluded"):
                yield q
                continue
            stats["questions"] += 1
            yield _distract_one(q, index, config, tau_sim, tau_min, tau_eq, stats)

    write_jsonl(output, gen())
    report = {"split": split, **dict(stats)}
    write_json(report_path, report)
    return report


def _distract_one(q: dict[str, Any], index: FactIndex, config: dict[str, Any],
                  tau_sim: float, tau_min: float, tau_eq: float,
                  stats: Counter[str]) -> dict[str, Any]:
    """Distractor construction for a single question (streamed)."""
    dataset = config["dataset"]
    anchors = {fid: index.facts[fid] for fid in q.get("anchor_fact_ids", [])
               if fid in index.facts}
    distractors: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    uncertain: list[str] = []
    for fid, entry in q.get("candidate_scores", {}).items():
        fact = index.facts.get(fid)
        if fact is None:
            continue
        if entry["max_semantic_score"] < tau_sim:
            stats["below_tau_sim"] += 1
            continue
        if entry["max_sub"] >= tau_min:
            stats["potential_substitute_above_tau_min"] += 1
            continue
        # Support(q, f') == 0 (section 13)
        supporting = False
        if dataset == "cronquestions":
            all_support = set().union(*(set(v) for v in
                                        q.get("anchor_support_fact_ids", {}).values()))
            supporting = fid in all_support
        else:
            events = [IcewsEvent(e) for e in (q.get("source_events") or [])]
            for position in range(len(events)):
                swapped = list(events)
                swapped[position] = icews_event_from_fact(fact)
                if execute_icews(q, swapped) == q["gold_answers"][0].strip():
                    supporting = True
                    break
        if supporting:
            # re-enter the substitute flow (section 13): the best
            # qualifying (anchor, candidate) pair from stage 07
            best_pair = None
            for pair in q.get("pair_scores", []):
                if pair["fact_id"] == fid and pair["valid_gate"] \
                        and pair["answer_preservation"] \
                        and pair["substitutability_score"] >= tau_eq:
                    if best_pair is None or pair["substitutability_score"] \
                            > best_pair["substitutability_score"]:
                        best_pair = pair
            if best_pair is not None:
                promoted.append({
                    "fact_id": fid,
                    "anchor_fact_id": best_pair["anchor_fact_id"],
                    "valid_gate": best_pair["valid_gate"],
                    "semantic_score": best_pair["semantic_score"],
                    "relation_score": best_pair["relation_score"],
                    "temporal_score": best_pair["temporal_score"],
                    "substitutability_score": best_pair["substitutability_score"],
                    "threshold": tau_eq,
                    "answer_preservation": True,
                    "verification": "verified_exact",
                })
                stats["promoted_to_substitute"] += 1
            else:
                uncertain.append(fid)
                stats["uncertain_supporting_candidate"] += 1
            continue
        best_anchor = anchors.get(entry.get("anchor_fact_id"))
        if best_anchor is None:
            continue
        top, subtype = _classify_distractor(best_anchor, fact,
                                            q.get("gold_answers", []), config)
        distractors.append({
            "fact_id": fid,
            "top_level_type": top,
            "subtype": subtype,
            "similarity_score": entry["max_semantic_score"],
            "support_check": False,
            "verification": "verified_non_supporting",
        })
        stats["distractors"] += 1
    if promoted:
        existing = {d["fact_id"]: d for d in q.get("substitutable_evidence", [])}
        for item in promoted:
            existing[item["fact_id"]] = item
        q["substitutable_evidence"] = sorted(existing.values(),
                                             key=lambda x: (-x["substitutability_score"],
                                                            x["fact_id"]))
    q["distracting_evidence"] = sorted(distractors,
                                       key=lambda x: (-x["similarity_score"],
                                                      x["fact_id"]))
    q["uncertain_candidates"] = uncertain
    return q


# --------------------------------------------------------------------------- #
# 09_build_evidence_spaces
# --------------------------------------------------------------------------- #

def build_evidence_spaces(config: dict[str, Any], questions_path: Path,
                          output: Path, root: Path, report_path: Path) -> dict[str, Any]:
    """Section 11.1/13.1: merge F*(q) ∪ F^sub(q) ∪ F^dis(q); role-disjointness
    assertions; formal retention conditions; substitute_anchor_coverage; and
    the global cross-split role reconciliation required by the section-20
    leakage audit (a train question must not carry a role label for any fact
    that is a test question's anchor/substitute/distractor)."""
    dataset = config["dataset"]
    stats: Counter[str] = Counter()
    # pass 1: collect the ANCHOR facts of test questions (compact digest set).
    # Section 20 interpretation: the test question's gold-proof pairing (its
    # Anchor facts) must never enter any train role label; test Substitute /
    # Distractor facts overlapping with train labels are the ordinary KG-fact
    # overlap that section 20 explicitly allows — they are reported as
    # metrics instead of being removed (the split-local construction of
    # section 19 already guarantees train labels were never built from test
    # labels).
    test_anchors: set[bytes] = set()
    test_sub_overlap: set[bytes] = set()
    test_dis_overlap: set[bytes] = set()
    for q in iter_jsonl(questions_path):
        if q.get("excluded") or q["benchmark_split"] != "test":
            continue
        test_anchors.update(fact_id_digest(fid) for fid in q.get("anchor_fact_ids", []))
        test_sub_overlap.update(fact_id_digest(d["fact_id"])
                                for d in q.get("substitutable_evidence", []))
        test_dis_overlap.update(fact_id_digest(d["fact_id"])
                                for d in q.get("distracting_evidence", []))
    reconciled = 0

    def gen():
        nonlocal reconciled
        for q in iter_jsonl(questions_path):
            if q.get("excluded"):
                yield q
                continue
            anchors = set(q.get("anchor_fact_ids", []))
            subs = {d["fact_id"] for d in q.get("substitutable_evidence", [])}
            dis = {d["fact_id"] for d in q.get("distracting_evidence", [])}
            # section 11.1 assertions
            assert anchors.isdisjoint(dis), f"{q['paired_id']}: anchor/distractor overlap"
            assert subs.isdisjoint(dis), f"{q['paired_id']}: substitute/distractor overlap"
            if q["benchmark_split"] == "train":
                overlapping_subs = [d for d in q.get("substitutable_evidence", [])
                                    if fact_id_digest(d["fact_id"]) in test_anchors]
                overlapping_dis = [d for d in q.get("distracting_evidence", [])
                                   if fact_id_digest(d["fact_id"]) in test_anchors]
                if dataset == "icews_actor":
                    # ICEWS uses one shared read-only KG.  A raw fact may be
                    # independently relevant to multiple questions/splits;
                    # that fact overlap is not a transfer of the test
                    # question's private role label.  Keep it and audit the
                    # overlap as a non-zero descriptive metric.  Leakage is
                    # blocked at paired-id, surface and semantic-family level.
                    stats["reported_train_test_anchor_substitute_overlaps"] += \
                        len(overlapping_subs)
                    stats["reported_train_test_anchor_distractor_overlaps"] += \
                        len(overlapping_dis)
                else:
                    # Preserve the existing CronQuestions behavior exactly.
                    kept_subs = [d for d in q.get("substitutable_evidence", [])
                                 if fact_id_digest(d["fact_id"]) not in test_anchors]
                    kept_dis = [d for d in q.get("distracting_evidence", [])
                                if fact_id_digest(d["fact_id"]) not in test_anchors]
                    reconciled += (len(q.get("substitutable_evidence", [])) - len(kept_subs)) \
                        + (len(q.get("distracting_evidence", [])) - len(kept_dis))
                    q["substitutable_evidence"] = kept_subs
                    q["distracting_evidence"] = kept_dis
                    subs = {d["fact_id"] for d in kept_subs}
                    dis = {d["fact_id"] for d in kept_dis}
            # retention conditions (section 13.1)
            reasons: list[str] = []
            if not anchors:
                reasons.append("anchor_set_empty")
            if not (q.get("proof_verified") and q.get("minimality_verified")):
                reasons.append("anchor_verification_failed")
            # ICEWS V2 release policy: a verified anchor is sufficient for
            # release. Substitute and distractor roles remain subject to every
            # original semantic/answer-preservation gate, but either role may
            # be absent. Other dataset builders retain their frozen policy.
            if dataset != "icews_actor" and not subs:
                reasons.append("no_verified_substitute")
            if dataset != "icews_actor" and not dis:
                reasons.append("no_verified_distractor")
            for d in q.get("substitutable_evidence", []):
                for field in ("semantic_score", "relation_score", "temporal_score",
                              "substitutability_score"):
                    if field not in d:
                        reasons.append("substitute_missing_scores")
                        break
                if not d.get("answer_preservation"):
                    reasons.append("substitute_not_answer_preserving")
            for d in q.get("distracting_evidence", []):
                if not d.get("top_level_type") or not d.get("subtype"):
                    reasons.append("distractor_missing_taxonomy")
                if d.get("support_check") is not False:
                    reasons.append("distractor_support_not_verified")
            covered_anchors = {d["anchor_fact_id"] for d in q.get("substitutable_evidence", [])}
            q["substitute_anchor_coverage"] = round(
                len(covered_anchors & anchors) / len(anchors), 6) if anchors else 0.0
            if reasons:
                q["eviscope_retained"] = False
                q.setdefault("exclusion_reasons", []).extend(sorted(set(reasons)))
                q["excluded"] = True
                for reason in sorted(set(reasons)):
                    _exclude(root, dataset, q["paired_id"], "09_build_evidence_spaces",
                             reason)
                stats["excluded"] += 1
                stats[f"excluded:{q['benchmark_split']}"] += 1
                stats[f"reason:{sorted(set(reasons))[0]}"] += 1
            else:
                q["eviscope_retained"] = True
                q["available_candidate_views"] = {
                    "clean_control": bool(anchors),
                    "substitute_only": bool(anchors and subs),
                    "distractor_only": bool(anchors and dis),
                    "full_eviscope": bool(anchors and subs and dis),
                }
                stats["retained"] += 1
                stats[f"retained:{q['benchmark_split']}"] += 1
            yield q

    write_jsonl(output, gen())
    report = {
        "retained": stats["retained"],
        "excluded": stats["excluded"],
        "retained_by_split": {
            split: stats[f"retained:{split}"] for split in ("train", "dev", "test")
        },
        "excluded_by_split": {
            split: stats[f"excluded:{split}"] for split in ("train", "dev", "test")
        },
        "train_role_labels_reconciled": reconciled,
        "reported_train_test_anchor_substitute_overlaps":
            stats["reported_train_test_anchor_substitute_overlaps"],
        "reported_train_test_anchor_distractor_overlaps":
            stats["reported_train_test_anchor_distractor_overlaps"],
        "test_anchor_fact_count": len(test_anchors),
        "allowed_fact_overlap_reported": {
            "test_substitute_facts_total": len(test_sub_overlap),
            "test_distractor_facts_total": len(test_dis_overlap),
        },
    }
    write_json(report_path, report)
    minimums = config.get("minimum_release_size", {}) \
        if dataset == "icews_actor" and not config.get("_allow_undersized_pilot") else {}
    failures = []
    for scope in ("total", "train", "dev", "test"):
        minimum = int(minimums.get(scope, 0) or 0)
        actual = stats["retained"] if scope == "total" else stats[f"retained:{scope}"]
        if minimum and actual < minimum:
            failures.append(f"{scope}={actual} < {minimum}")
    if failures:
        raise RuntimeError(
            "ICEWS-Actor stage-09 release-size gate failed: " + ", ".join(failures) +
            f". See {report_path}; the build is "
            "blocked instead of publishing an undersized benchmark."
        )
    return report
