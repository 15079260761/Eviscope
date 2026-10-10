"""Core stages: 02 seed questions, 03 anchor proofs, 04 families, 05 split.

Every stage only reads the frozen output of the previous stage (construction
document section 4).  All outputs carry rule_version and construction hashes
for traceability (section 2.7).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    canonical_json, iter_jsonl, normalize_question, placeholder_template,
    sha256_text, stable_id, write_json, write_jsonl,
)
from .executors import (
    FactIndex, execute_cron, execute_icews, IcewsEvent, _derive_cron,
)
from .sources import iter_cron_questions, iter_icews_questions
from .timeutils import BOT, EOT, parse_date, sentinel_date


# --------------------------------------------------------------------------- #
# stage 02: prepare seed questions
# --------------------------------------------------------------------------- #

def prepare_seed_questions(config: dict[str, Any], archive: Path, output: Path,
                           quarantine: Path, limit: int | None) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows = iter_cron_questions(config, archive, limit) if config["dataset"] == "cronquestions" \
        else iter_icews_questions(config, archive, limit)

    def emit():
        with output.open("w", encoding="utf-8", newline="\n") as good, \
                quarantine.open("w", encoding="utf-8", newline="\n") as bad:
            for row in rows:
                stats["input_questions"] += 1
                question = " ".join(str(row.get("question", "")).split())
                answers = sorted(set(str(x).strip() for x in row.get("answers", []) if str(x).strip()))
                reason = None
                if not question:
                    reason = "empty_question"
                elif not answers:
                    reason = "empty_answers"
                elif not row.get("question_type"):
                    reason = "missing_question_type"
                qid = stable_id(config["dataset"] + "q", row.get("source_split"), row.get("source_question_id"))
                if qid in seen:
                    reason = "duplicate_source_question_id"
                if reason:
                    stats["quarantined_questions"] += 1
                    stats[f"reason:{reason}"] += 1
                    bad.write(canonical_json({"reason": reason, "source": row}) + "\n")
                    continue
                seen.add(qid)
                prepared = {
                    "question_id": qid,
                    "family_id": None,
                    "source_dataset": config["dataset"],
                    "source_question_id": row.get("source_question_id"),
                    "source_split": row.get("source_split"),
                    "benchmark_split": None,
                    "question": question,
                    "normalized_question": normalize_question(question),
                    "question_type": row.get("question_type"),
                    "temporal_operator": row.get("temporal_operator"),
                    "topic_entity_ids": row.get("entities", []),
                    "time_constraints": row.get("times", []),
                    "relations": row.get("relations", []),
                    "gold_answers": answers,
                    "answer_type": row.get("answer_type"),
                    "template": row.get("template"),
                    "annotation": row.get("annotation", {}),
                    "source_events": row.get("source_events", []),
                    "source_kg_id": row.get("source_kg_id"),
                    "question_level": row.get("question_level"),
                    "logic_program": None,
                    "evidence_groups": [],
                    "distractors": [],
                    "answer_revision": None,
                    "construction": {"rule_version": config["rule_version"]},
                }
                good.write(canonical_json(prepared) + "\n")
                stats["accepted_questions"] += 1

    emit()
    return dict(stats)


# --------------------------------------------------------------------------- #
# stage 03: extract minimal anchor proofs
# --------------------------------------------------------------------------- #

def _cron_groups_to_anchors(question: dict[str, Any], index: FactIndex,
                            groups: list[dict[str, Any]], pool=None):
    """Convert support groups into anchor groups with minimality checks."""
    out: list[dict[str, Any]] = []
    for group in groups:
        supports = group.get("support_fact_ids", [])
        role = group.get("role")
        candidates = sorted(set(supports) | set(group.get("candidate_substitute_fact_ids", [])))
        if not candidates:
            continue
        # anchor: one fact per group; the group's supports are pre-minimal by
        # construction (per-answer roles / set-cover groups / reference facts)
        anchor = candidates[0]
        # minimality check (8.2): removing the anchor must break gold answers
        others = [f for f in candidates if f != anchor]
        pool_all = set().union(*(g.get("support_fact_ids", []) for g in groups)) if pool is None else set(pool)
        reduced_pool = (pool_all - {anchor}) | set(others)
        derived = execute_cron(question, index, pool=reduced_pool)
        minimal = derived != set(question["gold_answers"])
        out.append({
            "group_id": stable_id("g", question["question_id"], role, anchor),
            "role": str(role),
            "anchor_fact_ids": [anchor],
            "candidate_substitute_fact_ids": sorted(others),
            "substitute_fact_ids": [],
            "support_fact_ids": sorted(candidates),
            "minimality_verified": minimal,
        })
    return out


def extract_anchor_proofs(config: dict[str, Any], seeds_path: Path, index: FactIndex,
                          output: Path, quarantine: Path) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as good, \
            quarantine.open("w", encoding="utf-8", newline="\n") as bad:
        for question in iter_jsonl(seeds_path):
            stats["input_questions"] += 1
            reason = None
            if config["dataset"] == "icews":
                events = [IcewsEvent(e) for e in (question.get("source_events") or [])]
                groups: list[dict[str, Any]] = []
                program = {"op": "source_event_program",
                           "question_type": question["question_type"],
                           "temporal_operator": question.get("temporal_operator")}
                for position, event in enumerate(events):
                    fid = index.by_raw_tuple.get(
                        (event.subject, event.relation, event.object, event.raw_ts, event.raw_te))
                    if not fid:
                        reason = "source_event_not_found"
                        break
                    groups.append({
                        "group_id": stable_id("g", question["question_id"], position),
                        "role": position,
                        "anchor_fact_ids": [fid],
                        "candidate_substitute_fact_ids": [],
                        "substitute_fact_ids": [],
                        "support_fact_ids": [fid],
                        "minimality_verified": None,
                    })
                if not groups and not reason:
                    reason = "empty_source_proof"
                if not reason:
                    # sufficiency + reproducibility (7.3 / 8.3)
                    answer = execute_icews(question, events)
                    if answer != question["gold_answers"][0].strip():
                        reason = "executor_answer_mismatch"
            else:
                cap = config.get("answer_truncation_cap")
                groups, program, reason, expanded = _derive_cron(
                    question, index, truncation_cap=cap)
                if not reason:
                    if expanded is not None:
                        # source answer sets were truncated at the cap; the
                        # re-executed (complete) set is authoritative (7.1)
                        question["gold_answers"] = expanded
                        question["answer_revision"] = "truncated_source_expanded"
                        stats["re_executed_answers"] += 1
                    groups = _cron_groups_to_anchors(question, index, groups)
            if reason:
                stats["quarantined_questions"] += 1
                stats[f"reason:{reason}"] += 1
                bad.write(canonical_json({"reason": reason, "question_id": question["question_id"]}) + "\n")
                continue
            question["logic_program"] = program
            question["evidence_groups"] = groups
            good.write(canonical_json(question) + "\n")
            stats["verified_questions"] += 1
            stats["anchor_facts"] += sum(len(g["anchor_fact_ids"]) for g in groups)
    return dict(stats)


# --------------------------------------------------------------------------- #
# stage 04: build question families (union-find)
# --------------------------------------------------------------------------- #

class UnionFind:
    def __init__(self, values) -> None:
        self.parent = {v: v for v in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _coarse_equivalent_key(subject: str, relation: str, obj: str,
                           inverse_pairs: list, equivalence_groups: list) -> str | None:
    """Coarse-equivalent anchor key: subject/object swap with an inverse
    relation from the curated relation table (section 9.2)."""
    inv = {tuple(sorted(p)) for p in inverse_pairs}
    for pair in inverse_pairs:
        if relation == pair[0]:
            return canonical_json([obj, pair[1], subject])
        if relation == pair[1]:
            return canonical_json([obj, pair[0], subject])
    for group in equivalence_groups:
        if relation in group:
            base = sorted(group)
            return canonical_json([subject, base[0], obj])
    return None


def build_families(config: dict[str, Any], input_path: Path, output: Path) -> dict[str, Any]:
    questions = list(iter_jsonl(input_path))
    uf = UnionFind(q["question_id"] for q in questions)
    signatures: dict[tuple, str] = {}
    relations = config.get("_relations") or {}
    inverse_pairs = relations.get("inverse_pairs") or []
    equivalence_groups = relations.get("equivalence_groups") or []
    index = None  # coarse-equivalent edges need the fact index
    anchors_by_q: dict[str, list[dict[str, Any]]] = {}

    # question -> anchors (fact ids + triples) for anchor-sharing edges
    facts_path = input_path.parent / "01_facts.jsonl"
    if facts_path.exists():
        index = FactIndex(facts_path)

    for q in questions:
        qid = q["question_id"]
        anchor_ids = sorted(fid for g in q["evidence_groups"] for fid in g["anchor_fact_ids"])
        anchors_by_q[qid] = [index.facts[fid] for fid in anchor_ids] if index else []
        # template instance (9.1): same template + same parameters.  Cron
        # parameters are the annotated entities/times; ICEWS parameters are
        # the source event chain.
        if q.get("source_dataset") == "icews":
            instance_params = [q.get("template"), q.get("question_type"),
                               sorted(q.get("source_events", []))]
        else:
            instance_params = [q.get("template"), q.get("question_type"),
                               tuple(sorted(q.get("topic_entity_ids", []))),
                               tuple(sorted(q.get("time_constraints", [])))]
        keys: list[tuple] = [
            ("proof", sha256_text(canonical_json(anchor_ids))),
            ("normalized", q["normalized_question"]),
            ("template_instance", sha256_text(canonical_json(instance_params))),
        ]
        for anchor in anchor_ids:
            keys.append(("anchor", anchor))
        for key in keys:
            if key in signatures:
                uf.union(qid, signatures[key])
            else:
                signatures[key] = qid

    # anchor -- coarse equivalent fact edges (9.2)
    if index and (inverse_pairs or equivalence_groups):
        coarse: dict[str, str] = {}
        for q in questions:
            for fact in anchors_by_q[q["question_id"]]:
                key = _coarse_equivalent_key(
                    fact["subject_id"], fact["relation_id"], fact["object_id"],
                    inverse_pairs, equivalence_groups)
                if key is not None:
                    if key in coarse:
                        uf.union(q["question_id"], coarse[key])
                    else:
                        coarse[key] = q["question_id"]

    components: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        components[uf.find(q["question_id"])].append(q["question_id"])
    family_by_q: dict[str, str] = {}
    for members in components.values():
        fid = stable_id("family", sorted(members))
        for qid in members:
            family_by_q[qid] = fid
    for q in questions:
        q["family_id"] = family_by_q[q["question_id"]]
    write_jsonl(output, questions)
    sizes = Counter(len(x) for x in components.values())
    return {"questions": len(questions), "families": len(components),
            "family_size_histogram": dict(sizes),
            "giant_families_ge_100": sum(1 for m in components.values() if len(m) >= 100),
            "max_family_size": max((len(m) for m in components.values()), default=0)}


# --------------------------------------------------------------------------- #
# stage 05: stratified family split
# --------------------------------------------------------------------------- #

def _time_bucket(question: dict[str, Any]) -> str:
    years: list[int] = []
    for t in question.get("time_constraints", []):
        try:
            years.append(int(str(t)[:4]))
        except ValueError:
            pass
    for event in question.get("source_events", []):
        parts = str(event).split("|")
        if len(parts) == 5:
            for raw in (parts[3], parts[4]):
                d = parse_date(raw if raw not in (BOT, EOT) else "")
                if d:
                    years.append(d.year)
    if not years:
        return "none"
    decade = min(years) // 10 * 10
    return f"{decade:04d}s"


def _entity_freq_bucket(question: dict[str, Any], entity_freq: dict[str, int]) -> str:
    freqs = [entity_freq.get(e, 1) for e in question.get("topic_entity_ids", [])]
    if question.get("source_dataset") == "cronquestions":
        ann = question.get("annotation") or {}
        for key in ("head", "tail", "event_head"):
            value = ann.get(key)
            if value:
                freqs.append(entity_freq.get(str(value), 1))
    freqs += [1]
    return f"log{int(math.log10(max(freqs)))}"


def split_families(config: dict[str, Any], input_path: Path, index: FactIndex | None,
                   output: Path) -> dict[str, Any]:
    questions = list(iter_jsonl(input_path))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_family[q["family_id"]].append(q)
    entity_freq: dict[str, int] = {}
    if index is not None:
        entity_freq = {e: len(fids) for e, fids in index.by_entity.items()}
    strata: dict[tuple, list[str]] = defaultdict(list)
    for family, members in by_family.items():
        q = members[0]
        strata[(
            q["question_type"],
            len(q["evidence_groups"]),
            q.get("temporal_operator") or "none",
            "multi" if len(q["gold_answers"]) > 1 else "single",
            canonical_json(sorted(q.get("relations", []))),
            _time_bucket(q),
            _entity_freq_bucket(q, entity_freq),
        )].append(family)
    ratios = config["split"]["ratios"]
    seed = config["construction_seed"]
    family_split: dict[str, str] = {}
    names = ("train", "dev", "test")
    for key, families in strata.items():
        ordered = sorted(families, key=lambda f: sha256_text(f"{seed}|{canonical_json(list(key))}|{f}"))
        n = len(ordered)
        boundaries = [round(n * ratios[0]), round(n * (ratios[0] + ratios[1]))]
        for position, family in enumerate(ordered):
            chosen = names[0] if position < boundaries[0] else names[1] if position < boundaries[1] else names[2]
            family_split[family] = chosen
    counts: Counter[str] = Counter()
    for q in questions:
        q["benchmark_split"] = family_split[q["family_id"]]
        counts[q["benchmark_split"]] += 1
    write_jsonl(output, questions)
    return {
        "questions": len(questions),
        "families": len(by_family),
        "question_counts": dict(counts),
        "family_counts": dict(Counter(family_split.values())),
        "strata_count": len(strata),
    }
