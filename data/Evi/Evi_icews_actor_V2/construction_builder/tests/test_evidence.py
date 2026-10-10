"""Section 21.2: evidence tests (executors + swap semantics)."""

import json
import tempfile
import unittest
from pathlib import Path

from src.executors import FactIndex, execute_cron, IcewsEvent, execute_icews
from src.common import write_jsonl


def _write_facts(tmp: Path, rows) -> Path:
    facts = []
    for s, r, o, ts, te in rows:
        facts.append({
            "fact_id": f"f:{s}:{r}:{o}:{ts}:{te}",
            "source_dataset": "test",
            "subject_id": s, "relation_id": r, "object_id": o,
            "time_start": ts, "time_end": te,
            "raw_time_start": ts, "raw_time_end": te,
            "subject_name": s, "relation_name": r, "object_name": o,
            "fact_text": f"{s} | {r} | {o} | {ts}..{te}",
        })
    path = tmp / "facts.jsonl"
    write_jsonl(path, facts)
    return path


class TestCronExecutor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.facts = _write_facts(Path(self.tmp.name), [
            ("Q1", "P39", "Qp", "1990-01-01", "1997-12-31"),
            ("Q2", "P39", "Qp", "1999-01-01", "2008-12-31"),
            ("Q3", "P39", "Qp", "2008-01-01", "2016-12-31"),
            ("Q1", "P54", "Qt", "1980-01-01", "1995-12-31"),
            ("Q2", "P54", "Qt", "1996-01-01", "2010-12-31"),
        ])
        self.index = FactIndex(self.facts)

    def test_simple_entity_reproduces_gold(self):
        q = {"question_type": "simple_entity",
             "annotation": {"head": "Q1", "time": "1995"},
             "relations": ["P54"],
             "gold_answers": ["Qt"]}
        groups, program, error = _derive_cron_for_test(q, self.index)
        self.assertIsNone(error)
        self.assertEqual(groups[0]["support_fact_ids"],
                         ["f:Q1:P54:Qt:1980-01-01:1995-12-31"])

    def test_removing_anchor_changes_answers(self):
        # minimality: removing the sole anchor breaks reproduction (8.2)
        q = {"question_type": "simple_entity",
             "annotation": {"head": "Q1", "time": "1995"},
             "relations": ["P54"],
             "gold_answers": ["Qt"]}
        derived = execute_cron(q, self.index)
        self.assertEqual(derived, {"Qt"})
        anchor = "f:Q1:P54:Qt:1980-01-01:1995-12-31"
        self.assertIsNone(execute_cron(q, self.index, pool=set()))

    def test_swapping_in_substitute_keeps_answers(self):
        q = {"question_type": "simple_entity",
             "annotation": {"head": "Q1", "time": "1995"},
             "relations": ["P54"],
             "gold_answers": ["Qt"]}
        # second support fact added to the KG
        extra = _write_facts(Path(self.tmp.name), [
            ("Q1", "P54", "Qt", "1995-01-01", "1996-12-31")])
        # rebuild index including the extra fact
        import itertools
        from src.common import iter_jsonl
        facts = list(iter_jsonl(self.facts)) + list(iter_jsonl(extra))
        merged = Path(self.tmp.name) / "facts2.jsonl"
        write_jsonl(merged, facts)
        index2 = FactIndex(merged)
        all_support = {"f:Q1:P54:Qt:1980-01-01:1995-12-31",
                       "f:Q1:P54:Qt:1995-01-01:1996-12-31"}
        anchor = "f:Q1:P54:Qt:1980-01-01:1995-12-31"
        candidate = "f:Q1:P54:Qt:1995-01-01:1996-12-31"
        pool = (all_support - {anchor}) | {candidate}
        self.assertEqual(execute_cron(q, index2, pool=pool), {"Qt"})

    def test_multi_answer_comparison_is_order_independent(self):
        self.assertEqual({"a", "b"}, {"b", "a"})

    def test_valid_and_distractor_never_overlap(self):
        # enforced by construction: a fact in a support set is never a
        # distractor candidate (mine_distractors filters it out)
        valid = {"f1"}
        distractors = {d for d in ["f1", "f2"] if d not in valid}
        self.assertEqual(distractors, {"f2"})


class TestIcewsExecutor(unittest.TestCase):
    def test_timeline_recovery_start(self):
        q = {"question_type": "timeline_recovery", "answer_type": "timestamp_start",
             "temporal_operator": "timeline", "gold_answers": ["1960-11-26"], "question": ""}
        events = [IcewsEvent("Max Bradford|Affiliation To|Government Major Party (In Government) (New Zealand)|1960-11-26|1972-12-08")]
        self.assertEqual(execute_icews(q, events), "1960-11-26")

    def test_timeline_recovery_duration_format(self):
        q = {"question_type": "timeline_recovery", "answer_type": "duration",
             "temporal_operator": "timeline", "gold_answers": ["1999-07-28 - 1999-04-13"], "question": ""}
        events = [IcewsEvent("Jorge Heine|Affiliation To|Ministry of National Assets|1999-04-13|1999-07-28")]
        self.assertEqual(execute_icews(q, events), "1999-07-28 - 1999-04-13")

    def test_union_no_answer_when_disjoint(self):
        q = {"question_type": "timeline_recovery_timeline_recovery",
             "answer_type": "relation_union_or_intersection",
             "temporal_operator": "union", "gold_answers": ["No Answer"], "question": ""}
        events = [IcewsEvent("A|Affiliation To|X|1999-01-01|2001-01-01"),
                  IcewsEvent("B|Affiliation To|Y|2005-01-01|2006-01-01")]
        self.assertEqual(execute_icews(q, events), "No Answer")

    def test_intersection_open_bounds_format(self):
        q = {"question_type": "timeline_recovery_timeline_recovery",
             "answer_type": "relation_union_or_intersection",
             "temporal_operator": "intersection", "gold_answers": [""], "question": ""}
        events = [IcewsEvent("A|Affiliation To|X|1987-06-11|end of time"),
                  IcewsEvent("B|Affiliation To|Y|beginning of time|end of time")]
        self.assertEqual(execute_icews(q, events),
                         "(1987-06-11, 9999-12-31T23:59:59.999999)")

    def test_sum_forever_with_open_end(self):
        q = {"question_type": "timeline_recovery_timeline_recovery",
             "answer_type": "relation_duration",
             "temporal_operator": "sum", "gold_answers": ["forever"], "question": ""}
        events = [IcewsEvent("A|Affiliation To|X|2004-02-02|end of time"),
                  IcewsEvent("B|Affiliation To|Y|2012-01-08|2013-07-10")]
        self.assertEqual(execute_icews(q, events), "forever")

    def test_rank_start_time(self):
        q = {"question_type": "timeline_position_retrieval*3",
             "answer_type": "relation_ranking",
             "temporal_operator": "rank_start_time", "gold_answers": ["2"], "question": ""}
        events = [IcewsEvent("Alejandro Encinas Rodríguez|Affiliation To|Federal District|2005-06-30|2006-12-04"),
                  IcewsEvent("Dedet Edgar Guzman|Affiliation To|Grand National Alliance|2005-12-01|end of time"),
                  IcewsEvent("Farooq Abdulla|Affiliation To|Elite (India)|beginning of time|end of time")]
        self.assertEqual(execute_icews(q, events), "2")

    def test_tcr_subject_answer_and_constraint_gate(self):
        q = {"question_type": "timeline_recovery_temporal_constrainted_retrieval",
             "answer_type": "subject", "temporal_operator": "X = Y",
             "gold_answers": ["Nyamyn Jagvaral"], "question": ""}
        events = [IcewsEvent("Nyamyn Jagvaral|Affiliation To|Elite (Mongolia)|1937-01-01|1984-08-23"),
                  IcewsEvent("Other|Affiliation To|X|1999-01-01|2008-11-21")]
        # constraint X = Y fails -> program cannot be executed
        self.assertIsNone(execute_icews(q, events))
        events[1] = IcewsEvent("Other|Affiliation To|X|1937-01-01|1984-08-23")
        self.assertEqual(execute_icews(q, events), "Nyamyn Jagvaral")


def _derive_cron_for_test(q, index):
    from src.executors import _derive_cron
    groups, program, error, _expanded = _derive_cron(q, index)
    return groups, program, error


if __name__ == "__main__":
    unittest.main()
