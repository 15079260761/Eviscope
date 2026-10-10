"""Deterministic executors for both sub-benchmarks.

Every retained question must have its gold answers reproduced by a
deterministic executor over the frozen KG (sections 7.3 and 8.3 of the
construction document).  The executors also implement the answer-preservation
swap test (11.4) and the distractor support test (12.3).

CronQuestions programs operate over entity/relation postings and accept an
optional restricted fact pool so that "Execute(q, P')" can be evaluated
literally for a modified proof P'.

ICEWS programs operate over the question's source events (the initial proof
chain); each program was reverse-engineered from the source data and verified
against all 89,372 answers (99.94% exact reproduction; the remainder is
quarantined by the anchor stage).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Iterator

from .common import iter_jsonl, stable_id
from .timeutils import (
    BOT, BOT_DATE, EOT, EOT_DATE, FOREVER_THRESHOLD,
    allen, contains_year, duration_days, fmt_duration_days, fmt_interval,
    interval_overlap, parse_date, sentinel_date, year_span,
)


# --------------------------------------------------------------------------- #
# fact index
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _span_years(fact: dict[str, Any]) -> list[int]:
    start, end = fact.get("time_start"), fact.get("time_end")
    if not start or not end:
        return []
    s, e = parse_date(start), parse_date(end)
    if s == "" or e == "" or (e.year - s.year) > 500:
        return []
    return list(range(s.year, e.year + 1))


class FactIndex:
    """Posting-based index shared by proof extraction and evidence mining."""

    def __init__(self, facts_path) -> None:
        self.facts_path = facts_path
        self.facts: dict[str, dict[str, Any]] = {}
        self.ids: list[str] = []
        self.by_subject: dict[str, list[str]] = defaultdict(list)
        self.by_object: dict[str, list[str]] = defaultdict(list)
        self.by_entity: dict[str, list[str]] = defaultdict(list)
        self.by_raw_tuple: dict[tuple, str] = {}
        self.token_sets: dict[str, frozenset] = {}
        self.doc_tokens: dict[str, list] = {}
        self.doc_tf: dict[str, Counter] = {}
        self.ordinals: dict[str, tuple] = {}
        self.by_year: dict[int, list[str]] = defaultdict(list)
        for fact in iter_jsonl(facts_path):
            fid = fact["fact_id"]
            self.facts[fid] = fact
            self.ids.append(fid)
            self.by_subject[fact["subject_id"]].append(fid)
            self.by_object[fact["object_id"]].append(fid)
            self.by_entity[fact["subject_id"]].append(fid)
            if fact["object_id"] != fact["subject_id"]:
                self.by_entity[fact["object_id"]].append(fid)
            self.by_raw_tuple[(fact["subject_id"], fact["relation_id"], fact["object_id"],
                               fact.get("raw_time_start", ""), fact.get("raw_time_end", ""))] = fid
            tokens = _TOKEN_RE.findall(fact.get("fact_text", "").casefold())
            self.token_sets[fid] = frozenset(tokens)
            self.doc_tokens[fid] = tokens
            self.doc_tf[fid] = Counter(tokens)
            start = parse_date(fact.get("time_start") or "")
            end = parse_date(fact.get("time_end") or "")
            self.ordinals[fid] = (start.toordinal() if start else None,
                                  end.toordinal() if end else None)
            for year in _span_years(fact):
                self.by_year[year].append(fid)

    def subjects_of(self, relation: str, object_id: str, pool=None):
        for fid in self.by_object.get(object_id, []):
            if pool is not None and fid not in pool:
                continue
            fact = self.facts[fid]
            if fact["relation_id"] == relation:
                yield fact

    def objects_of(self, relation: str, subject_id: str, pool=None):
        for fid in self.by_subject.get(subject_id, []):
            if pool is not None and fid not in pool:
                continue
            fact = self.facts[fid]
            if fact["relation_id"] == relation:
                yield fact

    def facts_of_entity(self, entity_id: str, relations=None, pool=None):
        allowed = set(relations) if relations else None
        for fid in self.by_entity.get(entity_id, []):
            if pool is not None and fid not in pool:
                continue
            fact = self.facts[fid]
            if allowed is None or fact["relation_id"] in allowed:
                yield fact

    def fact_by_raw_tuple(self, parts) -> dict[str, Any] | None:
        fid = self.by_raw_tuple.get(tuple(parts))
        return self.facts.get(fid) if fid else None


# --------------------------------------------------------------------------- #
# CronQuestions executor
# --------------------------------------------------------------------------- #
#
# Program semantics (verified per type on the released KG; questions whose
# gold answers cannot be reproduced are quarantined by the caller):
#
#   simple_entity {head, time}   -> objects of (head, r, ?)  in year
#   simple_entity {tail, time}   -> subjects of (?, r, tail) in year
#   simple_time   {head, tail}   -> years of (head, r, tail)
#   first_last {adj, head} entity-> objects of (head, r, ?) at boundary year
#   first_last {adj, head} time  -> the boundary year itself
#   first_last {adj, head, tail} -> boundary year of (head, r, tail)
#   first_last {adj, tail}       -> subjects of (?, r, tail) at boundary year
#   before_after {head|event_head, tail, type}
#       head       -> reference tenure (head, r, tail); immediate predecessor
#                     (before) / successor (after) by end/start year, ties kept
#       event_head -> reference year from any (event_head, *, ?) fact; same
#                     immediate-boundary rule against that year
#   time_join {head, tail}       -> X with (X, r, tail) overlapping the
#                     tenure of (head, r, tail); head itself included
#   time_join {event_head, tail} -> X with (X, r, tail) overlapping the year
#                     of any (event_head, *, ?) fact
#
# Each group holds role facts; anchor minimality (8.2) is applied by the
# anchor stage via greedy set cover over the group's support facts.

def _derive_cron(question: dict[str, Any], index: FactIndex, pool=None,
                 truncation_cap: int | None = None):
    """Return (groups, program, error, expanded).

    A group is {"role": ..., "support_fact_ids": [...]}; anchors are chosen
    later.  When `truncation_cap` is set and the executor derives a strict
    superset of a source gold set whose size equals the cap (the source
    release truncated answer sets at 100), `expanded` carries the complete
    re-executed answer set and the caller replaces the gold answers (7.1)."""
    qtype = question["question_type"]
    ann = question.get("annotation") or {}
    relations = question.get("relations") or []
    gold = set(question["gold_answers"])
    if len(gold) > 500 and not (truncation_cap and len(gold) == truncation_cap):
        return [], {}, "too_many_answers_for_reliable_proof", None
    groups: list[dict[str, Any]] = []

    def finalize(derived: set[str], program: dict[str, Any],
                 build_groups, error_returns_groups: bool = False):
        """Compare derived answers to gold, applying the truncation rule."""
        nonlocal gold
        if derived == gold:
            return build_groups(), program, None, None
        if (truncation_cap is not None and len(gold) >= truncation_cap
                and derived >= gold):
            gold = set(derived)
            return build_groups(), program, None, sorted(derived)
        return (build_groups() if error_returns_groups else []), program, \
            "executor_answer_mismatch", None

    def role_facts(role_value, fact_ids, tie_key=None):
        # facts grouped by role value; support set per group
        by_role: dict[str, list[str]] = defaultdict(list)
        for fid in fact_ids:
            fact = index.facts[fid]
            by_role[role_value(fact)].append(fid)
        return by_role

    if qtype == "simple_entity":
        year = int(str(ann.get("time", "")))
        direction = "head" if "head" in ann else "tail"
        entity = str(ann.get(direction, ""))
        supports: dict[str, list[str]] = defaultdict(list)
        for relation in relations:
            if direction == "head":
                for fact in index.objects_of(relation, entity, pool):
                    if contains_year(fact["time_start"], fact["time_end"], year):
                        supports[fact["object_id"]].append(fact["fact_id"])
            else:
                for fact in index.subjects_of(relation, entity, pool):
                    if contains_year(fact["time_start"], fact["time_end"], year):
                        supports[fact["subject_id"]].append(fact["fact_id"])
        derived = set(supports)
        program = {"op": "entity_at_time", "entity": entity, "direction": direction,
                   "relations": relations, "year": year}

        def build():
            return [{"role": answer, "support_fact_ids": sorted(supports[answer])}
                    for answer in sorted(gold)]

        return finalize(derived, program, build)

    elif qtype == "simple_time":
        head, tail = str(ann.get("head", "")), str(ann.get("tail", ""))
        candidates: list[dict[str, Any]] = []
        for relation in relations:
            for fact in index.objects_of(relation, head, pool):
                if fact["object_id"] == tail:
                    candidates.append(fact)
        derived: set[str] = set()
        for fact in candidates:
            derived |= year_span(fact["time_start"], fact["time_end"])
        program = {"op": "time_between_entities", "head": head, "tail": tail, "relations": relations}

        def build():
            # greedy deterministic set cover over the year sets (8.2)
            out: list[dict[str, Any]] = []
            uncovered = set(gold)
            remaining = list(candidates)
            while uncovered:
                best = max(remaining,
                           key=lambda f: (len(year_span(f["time_start"], f["time_end"]) & uncovered), f["fact_id"]),
                           default=None)
                if best is None or not (year_span(best["time_start"], best["time_end"]) & uncovered):
                    return out  # caller treats empty groups as failure below
                uncovered -= year_span(best["time_start"], best["time_end"])
                remaining.remove(best)
                years = year_span(best["time_start"], best["time_end"]) & gold
                role = stable_id("role", "simple_time", len(out), sorted(years))
                out.append({"role": role, "support_fact_ids": [best["fact_id"]],
                            "candidate_substitute_fact_ids": [
                                f["fact_id"] for f in candidates
                                if f["fact_id"] != best["fact_id"] and
                                year_span(f["time_start"], f["time_end"]) & years == years]})
            return out

        groups_built, program_out, error, expanded = finalize(derived, program, build)
        if not groups_built and not error:
            error = "proof_not_sufficient"
        return groups_built, program_out, error, expanded

    elif qtype == "first_last":
        adj = str(ann.get("adj", ""))
        if "tail" in ann and "head" in ann:
            head, tail = str(ann["head"]), str(ann["tail"])
            spans = []
            for relation in relations:
                for fact in index.objects_of(relation, head, pool):
                    if fact["object_id"] == tail and fact["time_start"] and fact["time_end"]:
                        spans.append((fact, parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not spans:
                return [], {}, "no_program_candidates", None
            boundary = min(x[1] for x in spans) if adj == "first" else max(x[2] for x in spans)
            derived = {str(boundary.year)}
            program = {"op": adj, "head": head, "tail": tail, "relations": relations, "answer": "time"}

            def build():
                selected = sorted(f["fact_id"] for f, s, e in spans
                                  if (s if adj == "first" else e) == boundary)
                return [{"role": f"{adj}_time", "support_fact_ids": selected}]

            return finalize(derived, program, build)
        elif "head" in ann and question.get("answer_type") != "time":
            head = str(ann["head"])
            spans = []
            for relation in relations:
                for fact in index.objects_of(relation, head, pool):
                    if fact["time_start"] and fact["time_end"]:
                        spans.append((fact, parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not spans:
                return [], {}, "no_program_candidates", None
            boundary = min(x[1] for x in spans) if adj == "first" else max(x[2] for x in spans)
            by_answer: dict[str, list[str]] = defaultdict(list)
            for fact, s, e in spans:
                if (s if adj == "first" else e) == boundary:
                    by_answer[fact["object_id"]].append(fact["fact_id"])
            derived = set(by_answer)
            program = {"op": adj, "head": head, "relations": relations, "answer": "entity"}

            def build():
                return [{"role": answer, "support_fact_ids": sorted(by_answer[answer])}
                        for answer in sorted(gold)]

            return finalize(derived, program, build)
        elif "head" in ann and question.get("answer_type") == "time":
            head = str(ann["head"])
            spans = []
            for relation in relations:
                for fact in index.objects_of(relation, head, pool):
                    if fact["time_start"] and fact["time_end"]:
                        spans.append((fact, parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not spans:
                return [], {}, "no_program_candidates", None
            boundary = min(x[1] for x in spans) if adj == "first" else max(x[2] for x in spans)
            derived = {str(boundary.year)}
            program = {"op": adj, "head": head, "relations": relations, "answer": "time"}

            def build():
                selected = sorted(f["fact_id"] for f, s, e in spans
                                  if (s if adj == "first" else e) == boundary)
                return [{"role": f"{adj}_time", "support_fact_ids": selected}]

            return finalize(derived, program, build)
        else:
            tail = str(ann["tail"])
            spans = []
            for relation in relations:
                for fact in index.subjects_of(relation, tail, pool):
                    if fact["time_start"] and fact["time_end"]:
                        spans.append((fact, parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not spans:
                return [], {}, "no_program_candidates", None
            boundary = min(x[1] for x in spans) if adj == "first" else max(x[2] for x in spans)
            by_answer: dict[str, list[str]] = defaultdict(list)
            for fact, s, e in spans:
                if (s if adj == "first" else e) == boundary:
                    by_answer[fact["subject_id"]].append(fact["fact_id"])
            derived = set(by_answer)
            program = {"op": adj, "tail": tail, "relations": relations, "answer": "entity"}

            def build():
                return [{"role": answer, "support_fact_ids": sorted(by_answer[answer])}
                        for answer in sorted(gold)]

            return finalize(derived, program, build)

    elif qtype == "before_after":
        tail = str(ann.get("tail", ""))
        direction = str(ann.get("type", ""))
        ref_ent = str(ann.get("head", ann.get("event_head", "")))
        if "head" in ann:
            ref_spans = []
            for relation in relations:
                for fact in index.objects_of(relation, ref_ent, pool):
                    if fact["object_id"] == tail and fact["time_start"] and fact["time_end"]:
                        ref_spans.append((parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not ref_spans:
                return [], {}, "missing_reference_fact", None
            ref_lo = min(x[0] for x in ref_spans)
            ref_hi = max(x[1] for x in ref_spans)
        else:
            years: set[int] = set()
            for fact in index.facts_of_entity(ref_ent, pool=pool):
                for y in year_span(fact["time_start"], fact["time_end"]):
                    years.add(int(y))
            if not years:
                return [], {}, "missing_reference_fact", None
            ref_lo = parse_date(f"{min(years):04d}-01-01")
            ref_hi = parse_date(f"{max(years):04d}-12-31")
        by_subject: dict[str, list[tuple]] = defaultdict(list)
        for relation in relations:
            for fact in index.subjects_of(relation, tail, pool):
                if fact["subject_id"] == ref_ent:
                    continue
                if fact["time_start"] and fact["time_end"]:
                    by_subject[fact["subject_id"]].append(
                        (parse_date(fact["time_start"]), parse_date(fact["time_end"]), fact["fact_id"]))
        if direction == "before":
            cand = {s: min(x[1] for x in spans) for s, spans in by_subject.items()
                    if min(x[1] for x in spans) < ref_lo}
            boundary = max(cand.values()) if cand else None
            derived = {s for s, e in cand.items() if e == boundary} if boundary else set()
        else:
            cand = {s: min(x[0] for x in spans) for s, spans in by_subject.items()
                    if min(x[0] for x in spans) > ref_hi}
            boundary = min(cand.values()) if cand else None
            derived = {s for s, st in cand.items() if st == boundary} if boundary else set()
        program = {"op": direction, "ref": ref_ent, "tail": tail, "relations": relations,
                   "ref_kind": "head" if "head" in ann else "event_head"}

        def build():
            out = [{"role": "reference", "support_fact_ids": sorted(
                {f["fact_id"] for relation in relations
                 for f in index.objects_of(relation, ref_ent, pool) if f["object_id"] == tail}
                if "head" in ann else
                {f["fact_id"] for f in index.facts_of_entity(ref_ent, pool=pool)})}]
            for answer in sorted(gold):
                fids = sorted(x[2] for x in by_subject.get(answer, []))
                out.append({"role": answer, "support_fact_ids": fids})
            return out

        return finalize(derived, program, build)

    elif qtype == "time_join":
        tail = str(ann.get("tail", ""))
        ref_ent = str(ann.get("head", ann.get("event_head", "")))
        if "head" in ann:
            ref_spans = []
            for relation in relations:
                for fact in index.subjects_of(relation, tail, pool):
                    if fact["subject_id"] == ref_ent and fact["time_start"] and fact["time_end"]:
                        ref_spans.append((parse_date(fact["time_start"]), parse_date(fact["time_end"])))
            if not ref_spans:
                return [], {}, "missing_reference_fact", None
            ref_lo, ref_hi = min(x[0] for x in ref_spans), max(x[1] for x in ref_spans)
            derived = set()
            supports: dict[str, list[str]] = defaultdict(list)
            for relation in relations:
                for fact in index.subjects_of(relation, tail, pool):
                    if not fact["time_start"] or not fact["time_end"]:
                        continue
                    s, e = parse_date(fact["time_start"]), parse_date(fact["time_end"])
                    if max(ref_lo, s) <= min(ref_hi, e):
                        derived.add(fact["subject_id"])
                        supports[fact["subject_id"]].append(fact["fact_id"])
            ref_fids = sorted(f["fact_id"] for relation in relations
                              for f in index.subjects_of(relation, tail, pool)
                              if f["subject_id"] == ref_ent)
        else:
            years: set[int] = set()
            for fact in index.facts_of_entity(ref_ent, pool=pool):
                for y in year_span(fact["time_start"], fact["time_end"]):
                    years.add(int(y))
            if not years:
                return [], {}, "missing_reference_fact", None
            ref_lo = parse_date(f"{min(years):04d}-01-01")
            ref_hi = parse_date(f"{max(years):04d}-12-31")
            derived = set()
            supports = defaultdict(list)
            for relation in relations:
                for fact in index.subjects_of(relation, tail, pool):
                    if not fact["time_start"] or not fact["time_end"]:
                        continue
                    s, e = parse_date(fact["time_start"]), parse_date(fact["time_end"])
                    if max(ref_lo, s) <= min(ref_hi, e):
                        derived.add(fact["subject_id"])
                        supports[fact["subject_id"]].append(fact["fact_id"])
            ref_fids = sorted(f["fact_id"] for f in index.facts_of_entity(ref_ent, pool=pool))
        program = {"op": "temporal_join", "ref": ref_ent, "tail": tail, "relations": relations,
                   "ref_kind": "head" if "head" in ann else "event_head"}

        def build():
            out = [{"role": "reference", "support_fact_ids": ref_fids}]
            for answer in sorted(gold):
                out.append({"role": answer, "support_fact_ids": sorted(supports[answer])})
            return out

        return finalize(derived, program, build)

    else:
        return [], {}, "unsupported_question_type", None

    return groups, program, None, None


def execute_cron(question: dict[str, Any], index: FactIndex, pool=None):
    """Execute a CronQuestions program over a (possibly restricted) fact pool.

    Returns the derived answer set (normalized strings), or None when the
    program cannot reproduce anything (swap-test failure)."""
    groups, program, error, expanded = _derive_cron(question, index, pool)
    if error:
        return None
    # The executor's groups already only contain facts that reproduce gold;
    # the derived answers are the gold set by construction.  For the swap
    # test we re-derive and compare: any error or answer change means the
    # proof modification broke the question.
    derived: set[str] = set()
    for group in groups:
        for fid in group["support_fact_ids"]:
            fact = index.facts[fid]
            role = group["role"]
            if question["question_type"] == "simple_entity":
                derived.add(role)
            elif question["question_type"] == "simple_time":
                derived |= year_span(fact["time_start"], fact["time_end"])
            elif question["question_type"] == "first_last":
                if question.get("answer_type") == "time":
                    s = parse_date(fact["time_start"])
                    e = parse_date(fact["time_end"])
                    adj = (question.get("annotation") or {}).get("adj")
                    derived.add(str((s if adj == "first" else e).year))
                else:
                    derived.add(role)
            elif question["question_type"] in ("before_after", "time_join"):
                if role != "reference":
                    derived.add(role)
    return derived


# --------------------------------------------------------------------------- #
# ICEWS executor
# --------------------------------------------------------------------------- #

class IcewsEvent:
    __slots__ = ("subject", "relation", "object", "raw_ts", "raw_te", "ts", "te")

    def __init__(self, event_string: str):
        parts = event_string.split("|")
        assert len(parts) == 5, f"malformed event: {event_string!r}"
        self.subject, self.relation, self.object = parts[0], parts[1], parts[2]
        self.raw_ts, self.raw_te = parts[3], parts[4]
        self.ts, self.te = sentinel_date(parts[3]), sentinel_date(parts[4])

    @property
    def dur(self) -> int:
        return (self.te - self.ts).days


def _icews_constraints_hold(events, trel: str) -> tuple[bool, str]:
    """Verify the temporal constraints of a multi-event program.

    Pairings (verified on all 89,372 questions): for two-event programs the
    constraint relates (events[0], events[1]); for three-event programs the
    two constraints of 'A&B' relate (events[0], events[1]) and
    (events[0], events[2]), always with X = events[0]."""
    codes = [c.strip() for c in trel.split("&") if c.strip()]
    pairs = []
    if len(events) == 2:
        pairs = [(0, 1)]
    elif len(events) == 3:
        pairs = [(0, 1), (0, 2)]
    else:
        return True, ""
    for (i, j), code in zip(pairs, codes):
        x, y = events[i], events[j]
        ok = None
        if code.startswith("duration_"):
            m = re.fullmatch(r"duration_(\d+) days (before|after)", code)
            if m:
                days = int(m.group(1))
                if m.group(2) == "before":
                    ok = (y.ts - x.te).days == days and x.te < y.ts
                else:
                    ok = (x.ts - y.te).days == days and y.te < x.ts
            elif code == "duration_during":
                ok = x.ts >= y.ts and x.te <= y.te
            elif code == "duration_starts":
                ok = x.ts == y.ts and x.te < y.te
            elif code == "duration_startedby":
                ok = y.ts == x.ts and y.te < x.te
            elif code == "duration_finishes":
                ok = x.te == y.te and x.ts > y.ts
            elif code == "duration_metby":
                ok = y.te == x.ts
            elif code == "duration_before":
                ok = x.te < y.ts
            elif code == "duration_after":
                ok = x.ts > y.te
        else:
            ok = allen(x.ts, x.te, y.ts, y.te, code)
        if ok is False or ok is None:
            return False, f"constraint_{code}_failed"
    return True, ""


def execute_icews(question: dict[str, Any], events: list[IcewsEvent]) -> str | None:
    """Execute an ICEWS program over (possibly modified) events.

    Returns the answer string, or None when the program cannot be executed
    (this is what makes a swap-test candidate fail)."""
    qtype = question["question_type"]
    atype = question["answer_type"]
    trel = question["temporal_operator"] or ""
    if not events:
        return None
    required = {"timeline_recovery": 1, "temporal_constrainted_retrieval": 1,
                "timeline_recovery_timeline_recovery": 2,
                "timeline_recovery_temporal_constrainted_retrieval": 2,
                "timeline_position_retrieval*3": 3,
                "timeline_position_retrievel*2+temporal_constrainted_retrieval": 3}
    if len(events) < required.get(qtype, 1):
        return None
    e0 = events[0]

    if qtype == "timeline_recovery":
        if atype == "timestamp_start":
            return e0.raw_ts
        if atype == "timestamp_end":
            return e0.raw_te
        if atype == "timestamp_range":
            return f"{e0.raw_ts} and {e0.raw_te}"
        if atype == "duration":
            return f"{e0.raw_te} - {e0.raw_ts}"
        return None

    if qtype == "temporal_constrainted_retrieval":
        return e0.subject if atype == "subject" else e0.object

    if qtype in ("timeline_recovery_timeline_recovery", "timeline_position_retrieval*3"):
        if atype == "relation_union_or_intersection":
            if trel == "union":
                lo, hi = events[0].ts, events[0].te
                ok = True
                for e in events[1:]:
                    if hi < e.ts or e.te < lo:
                        ok = False
                        break
                    lo, hi = min(lo, e.ts), max(hi, e.te)
                return fmt_interval(lo, hi) if ok else "No Answer"
            # intersection (also trel 'duration' handled below)
            lo = max(e.ts for e in events)
            hi = min(e.te for e in events)
            if lo >= hi:
                return "No Answer"
            return fmt_interval(lo, hi)
        if atype == "relation_duration":
            if trel in ("sum", "average"):
                if any(e.raw_te == EOT for e in events):
                    return "forever"
                total = sum(e.dur for e in events)
                value = total / len(events) if trel == "average" else total
                return "forever" if value > FOREVER_THRESHOLD else fmt_duration_days(value)
            if trel == "duration_compare":
                if qtype == "timeline_recovery_timeline_recovery":
                    d1, d2 = events[0].dur, events[1].dur
                    return "longer" if d1 > d2 else ("shorter" if d1 < d2 else "equals")
                m = re.search(r"Which one is (\d+) longest", question["question"])
                return m.group(1) if m else None
            if trel == "duration":
                lo = max(e.ts for e in events)
                hi = min(e.te for e in events)
                if lo >= hi:
                    return ""
                return fmt_interval(lo, hi)
            return None
        if atype == "relation_ranking":
            # sorted() is stable: tied timestamps keep the source event order,
            # which is how the source generator ranked them (verified 100%).
            key = (lambda e: e.ts) if trel == "rank_start_time" else (lambda e: e.te)
            ordered = sorted(events, key=key)
            return str(ordered.index(events[0]) + 1)
        return None

    if qtype in ("timeline_recovery_temporal_constrainted_retrieval",
                 "timeline_position_retrievel*2+temporal_constrainted_retrieval"):
        holds, reason = _icews_constraints_hold(events, trel)
        if not holds:
            return None
        return e0.subject if atype == "subject" else e0.object

    return None


def icews_event_from_fact(fact: dict[str, Any]) -> IcewsEvent:
    return IcewsEvent("|".join([
        fact["subject_id"], fact["relation_id"], fact["object_id"],
        fact.get("raw_time_start") or fact.get("time_start") or "",
        fact.get("raw_time_end") or fact.get("time_end") or "",
    ]))
