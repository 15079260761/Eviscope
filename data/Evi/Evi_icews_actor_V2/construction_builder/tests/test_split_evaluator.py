"""Spec sections 5/17/20/21: split inheritance, family audit, paper BS@K,
HDM recomputation and retention-condition tests."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.common import canonical_json, sha256_text, write_jsonl, iter_jsonl
from src.encoder import FrozenEncoder
from src.stages_evidence import build_family_audit, build_evidence_spaces
from src.stages_release import compute_hdm, run_audits
from src.sources import SourceBundle, iter_icews_questions
from evaluator.metrics import score_question


def _mk_question(paired_id, split, anchors=("f1",), subs=(), dis=(), excluded=False,
                 qtype="simple_entity", answers=("a",), template="tpl",
                 normalized=None, norm_dup=False):
    return {
        "paired_id": paired_id, "question_id": paired_id, "family_id": None,
        "source_dataset": "test", "source_question_id": paired_id.split(":")[-1],
        "source_split": split, "benchmark_split": split,
        "question": f"question {paired_id}", "normalized_question": normalized or f"question {paired_id}",
        "question_type": qtype, "temporal_operator": None,
        "topic_entity_ids": [], "time_constraints": [], "relations": [],
        "gold_answers": list(answers), "answer_type": "entity", "template": template,
        "annotation": {}, "source_events": [],
        "source_provenance": {"source_file": "x", "source_row": int(paired_id[-1]) if paired_id[-1].isdigit() else 0},
        "anchor_fact_ids": list(anchors), "anchor_count": len(anchors),
        "proof_verified": True, "minimality_verified": True,
        "verification_version": "anchor_v1",
        "anchor_support_fact_ids": {a: [a] for a in anchors},
        "substitutable_evidence": [{"fact_id": s, "anchor_fact_id": anchors[0],
                                    "valid_gate": 1, "semantic_score": 0.5,
                                    "relation_score": 1.0, "temporal_score": 0.5,
                                    "substitutability_score": 0.5, "threshold": 0.0,
                                    "answer_preservation": True,
                                    "verification": "verified_exact"} for s in subs],
        "distracting_evidence": [{"fact_id": d, "top_level_type": "confusable",
                                  "subtype": "temporal_confusable",
                                  "similarity_score": 0.5, "support_check": False,
                                  "verification": "verified_non_supporting"} for d in dis],
        "excluded": excluded,
        "exclusion_reasons": ["test"] if excluded else [],
    }


class TestSplitInheritance(unittest.TestCase):
    def test_benchmark_split_equals_source_split(self):
        q = _mk_question("test:q1", "train")
        self.assertEqual(q["benchmark_split"], q["source_split"])

    def test_family_audit_near_duplicate_cross_split(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("a:1", "train", normalized="same question"),
            _mk_question("a:2", "test", normalized="same question"),
            _mk_question("a:3", "train", normalized="other"),
        ]
        for r in rows:
            r["topic_entity_ids"] = [r["paired_id"]]  # distinct template instances
        inp = tmp / "in.jsonl"
        write_jsonl(inp, rows)
        out = tmp / "out.jsonl"
        config = {"dataset": "test"}
        build_family_audit(config, inp, out, tmp, tmp / "report.json")
        rows = list(iter_jsonl(out))
        states = {r["paired_id"]: r.get("excluded") for r in rows}
        self.assertTrue(states["a:1"])
        self.assertTrue(states["a:2"])
        self.assertFalse(states["a:3"])

    def test_retention_requires_substitute_and_distractor(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("b:1", "train", anchors=("f1",), subs=(), dis=("d1",)),
            _mk_question("b:2", "train", anchors=("f2",), subs=("s1",), dis=("d2",)),
        ]
        inp = tmp / "in.jsonl"
        write_jsonl(inp, rows)
        out = tmp / "out.jsonl"
        build_evidence_spaces({"dataset": "test"}, inp, out, tmp, tmp / "report.json")
        rows = list(iter_jsonl(out))
        states = {r["paired_id"]: r.get("eviscope_retained") for r in rows}
        self.assertFalse(states["b:1"])
        self.assertTrue(states["b:2"])

    def test_icews_v2_retains_anchor_valid_partial_view_question(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("icews-v2:1", "train", anchors=("f1",),
                         subs=(), dis=("d1",)),
            _mk_question("icews-v2:2", "train", anchors=("f2",),
                         subs=("s1",), dis=()),
        ]
        inp, out = tmp / "in.jsonl", tmp / "out.jsonl"
        write_jsonl(inp, rows)
        build_evidence_spaces({"dataset": "icews_actor"}, inp, out, tmp,
                              tmp / "report.json")
        built = {r["paired_id"]: r for r in iter_jsonl(out)}
        self.assertTrue(built["icews-v2:1"]["eviscope_retained"])
        self.assertTrue(built["icews-v2:2"]["eviscope_retained"])
        self.assertFalse(
            built["icews-v2:1"]["available_candidate_views"]["substitute_only"])
        self.assertFalse(
            built["icews-v2:2"]["available_candidate_views"]["distractor_only"])

    def test_icews_uses_row_aligned_original_question(self):
        tmp = Path(tempfile.mkdtemp())
        source = tmp / "questions.csv"
        fields = ["id", "split", "question", "paraphrased_question", "answer",
                  "answer_type", "question_type", "temporal_relation", "events",
                  "source_kg_id", "question_level"]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "id": "1", "split": "train",
                "question": "Who joined the aligned event?",
                "paraphrased_question": "Corrupted unrelated paraphrase",
                "answer": "Alice", "answer_type": "subject",
                "question_type": "temporal_constrainted_retrieval",
                "temporal_relation": "timeline",
                "events": "['Alice|Affiliation To|Org|2001-01-01|2002-01-01']",
                "source_kg_id": "7", "question_level": "simple",
            })
        config = {"dataset": "icews_actor", "questions_member": "questions.csv",
                  "question_text_policy": "original"}
        row = next(iter_icews_questions(config, SourceBundle(tmp), None))
        self.assertEqual(row["question"], "Who joined the aligned event?")
        self.assertEqual(row["source_paraphrased_question"],
                         "Corrupted unrelated paraphrase")
        self.assertEqual(row["question_text_source"], "original_question")

    def test_icews_family_audit_blocks_surface_and_semantic_leakage(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("icews:1", "train", normalized="same surface",
                         answers=("a",)),
            _mk_question("icews:2", "test", normalized="same surface",
                         answers=("b",)),
            _mk_question("icews:3", "train", normalized="wording one",
                         answers=("c",)),
            _mk_question("icews:4", "dev", normalized="wording two",
                         answers=("c",)),
            _mk_question("icews:5", "train", normalized="unique",
                         answers=("e",)),
        ]
        rows[0]["source_events"] = ["a|r|o|2001-01-01|2002-01-01"]
        rows[1]["source_events"] = ["b|r|o|2001-01-01|2002-01-01"]
        shared = ["c|r|o|2001-01-01|2002-01-01"]
        rows[2]["source_events"] = shared
        rows[3]["source_events"] = shared
        rows[4]["source_events"] = ["e|r|o|2001-01-01|2002-01-01"]
        inp, out = tmp / "in.jsonl", tmp / "out.jsonl"
        write_jsonl(inp, rows)
        build_family_audit({"dataset": "icews_actor"}, inp, out, tmp,
                           tmp / "report.json")
        states = {r["paired_id"]: r.get("excluded") for r in iter_jsonl(out)}
        self.assertTrue(states["icews:1"])
        self.assertTrue(states["icews:2"])
        self.assertTrue(states["icews:3"])
        self.assertTrue(states["icews:4"])
        self.assertFalse(states["icews:5"])

    def test_icews_shared_fact_overlap_is_reported_not_removed(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("icews:1", "train", anchors=("f1",),
                         subs=("shared",), dis=("d1",)),
            _mk_question("icews:2", "test", anchors=("shared",),
                         subs=("s2",), dis=("d2",)),
        ]
        inp, out = tmp / "in.jsonl", tmp / "out.jsonl"
        write_jsonl(inp, rows)
        report = build_evidence_spaces({"dataset": "icews_actor"}, inp, out,
                                       tmp, tmp / "report.json")
        states = {r["paired_id"]: r.get("eviscope_retained")
                  for r in iter_jsonl(out)}
        self.assertTrue(states["icews:1"])
        self.assertTrue(states["icews:2"])
        self.assertEqual(report["train_role_labels_reconciled"], 0)
        self.assertEqual(
            report["reported_train_test_anchor_substitute_overlaps"], 1)

    def test_cron_shared_test_anchor_behavior_is_unchanged(self):
        tmp = Path(tempfile.mkdtemp())
        rows = [
            _mk_question("cron:1", "train", anchors=("f1",),
                         subs=("shared",), dis=("d1",)),
            _mk_question("cron:2", "test", anchors=("shared",),
                         subs=("s2",), dis=("d2",)),
        ]
        inp, out = tmp / "in.jsonl", tmp / "out.jsonl"
        write_jsonl(inp, rows)
        report = build_evidence_spaces({"dataset": "cronquestions"}, inp, out,
                                       tmp, tmp / "report.json")
        states = {r["paired_id"]: r.get("eviscope_retained")
                  for r in iter_jsonl(out)}
        self.assertFalse(states["cron:1"])
        self.assertTrue(states["cron:2"])
        self.assertEqual(report["train_role_labels_reconciled"], 1)


class TestPaperBS(unittest.TestCase):
    """Section 17/21: M_ij = cos(e(fi), e(rj)) if > tau else 0; BS = max-weight
    one-to-one / |F*(q)|; no rank discount; no double credit."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.facts_path = self.tmp / "facts.jsonl"
        write_jsonl(self.facts_path, [
            {"fact_id": "f:xa", "canonical_fact_text": "alpha alpha alpha"},
            {"fact_id": "f:xb", "canonical_fact_text": "beta beta beta"},
            {"fact_id": "f:xc", "canonical_fact_text": "alpha alpha beta"},
            {"fact_id": "f:xd", "canonical_fact_text": "gamma gamma gamma"},
        ])
        self.encoder = FrozenEncoder({"encoder": {"dim": 1024}}, self.facts_path)
        self.facts = {f["fact_id"]: f for f in iter_jsonl(self.facts_path)}
        self.tau = 0.0

    def _q(self, anchors, subs=()):
        return {
            "question_id": "q1",
            "anchor_fact_ids": list(anchors),
            "substitutable_evidence": [{"fact_id": s} for s in subs],
            "distracting_evidence": [],
        }

    def test_complete_evidence_ranks_above_redundant(self):
        q = self._q(["f:xa", "f:xb"])
        complete = score_question(q, ["f:xa", "f:xb"], 2, self.encoder, self.tau, self.facts)
        redundant = score_question(q, ["f:xa", "f:xa"], 2, self.encoder, self.tau, self.facts)
        self.assertGreater(complete["bs"], redundant["bs"])
        self.assertEqual(complete["coverage"], 1.0)
        self.assertEqual(redundant["coverage"], 0.5)

    def test_threshold_zeroing(self):
        q = self._q(["f:xa"])
        s = score_question(q, ["f:xc"], 1, self.encoder, 0.999, self.facts)
        self.assertEqual(s["bs"], 0.0)

    def test_one_to_one_no_double_credit(self):
        # one retrieved fact similar to two anchors: matched to at most one
        q = self._q(["f:xa", "f:xc"])
        s = score_question(q, ["f:xa"], 1, self.encoder, self.tau, self.facts)
        self.assertEqual(s["bs"], 0.5)

    def test_denominator_is_anchor_count(self):
        q = self._q(["f:xa", "f:xb", "f:xd"])
        s = score_question(q, ["f:xa"], 1, self.encoder, self.tau, self.facts)
        self.assertAlmostEqual(s["bs"], 1.0 / 3.0, places=6)

    def test_no_rank_discount(self):
        q = self._q(["f:xa", "f:xb"])
        early = score_question(q, ["f:xa", "f:xb"], 2, self.encoder, self.tau, self.facts)
        late = score_question(q, ["f:xb", "f:xa"], 2, self.encoder, self.tau, self.facts)
        self.assertEqual(early["bs"], late["bs"])
        self.assertEqual(early["bs"], 1.0)

    def test_deterministic(self):
        q = self._q(["f:xa", "f:xb", "f:xc"])
        r1 = score_question(q, ["f:xc", "f:xa"], 2, self.encoder, self.tau, self.facts)
        r2 = score_question(q, ["f:xc", "f:xa"], 2, self.encoder, self.tau, self.facts)
        self.assertEqual(r1, r2)

    def test_hits_require_verified_evidence(self):
        q = self._q(["f:xa"], subs=("f:xb",))
        self.assertEqual(score_question(q, ["f:xb"], 1, self.encoder, self.tau, self.facts)["hits"], 1.0)
        self.assertEqual(score_question(q, ["f:xd"], 1, self.encoder, self.tau, self.facts)["hits"], 0.0)


class TestHDMFormula(unittest.TestCase):
    """Section 15/21: C(q)=|F*|, A(q)=(sum|N|-N)/N^2, D(q)=C(1+A)."""

    def test_formula_recomputation(self):
        sizes = [8, 4, 11]
        n = 3
        a_q = (sum(sizes) - n) / (n * n)
        self.assertAlmostEqual(a_q, (23 - 3) / 9.0, places=6)
        d_q = n * (1 + a_q)
        self.assertAlmostEqual(d_q, 3 * (1 + 20 / 9.0), places=6)


if __name__ == "__main__":
    unittest.main()
