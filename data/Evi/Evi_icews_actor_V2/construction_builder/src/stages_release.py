"""Release stages per spec sections 15-27:
10_build_paired_pools, 11_compute_hdm, 12_run_audits, 13_export_release.

HDM uses the paper's formulas verbatim (section 15):
  C(q) = |F*(q)|
  N(f_i) = {f' in full normalized fact space | cos(e(f_i), e(f')) >= tau_amb}
  A(q) = (1/N) * sum_i (|N(f_i)| - 1) / |F*(q)|,   N = |F*(q)|
  D(q) = C(q) * (1 + A(q))
with train-fitted tau_amb and train tercile boundaries (Dev confirms once,
Test consumes the frozen values).
"""

from __future__ import annotations

import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    canonical_json, fact_id_digest, iter_jsonl, rewrite_jsonl, sha256_file,
    sha256_text, utc_now, write_json, write_jsonl,
)
from .encoder import FrozenEncoder, NeighborIndex
from .executors import FactIndex
from .stages_source import _manifest_dir


# --------------------------------------------------------------------------- #
# 10_build_paired_pools
# --------------------------------------------------------------------------- #

def build_paired_pools(config: dict[str, Any], questions_path: Path, index: FactIndex,
                       output_dir: Path, report_path: Path,
                       questions_output: Path) -> dict[str, Any]:
    """Section 16: source_control_pool(q) and eviscope_pool(q) share paired_id,
    split, size, anchors, background sampling rules, seed and serialization.
    The EviScope pool replaces equal neutral background with
    Substitute/Distractor facts and never removes anchors."""
    pool_size = config["candidate_pool"]["size"]
    seed = config["construction_seed"]
    stats: Counter[str] = Counter()
    control_dir = output_dir / "source_control"
    eviscope_dir = output_dir / "eviscope"
    control_dir.mkdir(parents=True, exist_ok=True)
    eviscope_dir.mkdir(parents=True, exist_ok=True)
    control_handle = (control_dir / "candidate_pools.jsonl").open(
        "w", encoding="utf-8", newline="\n")
    eviscope_handle = (eviscope_dir / "candidate_pools.jsonl").open(
        "w", encoding="utf-8", newline="\n")

    def gen():
        for q in iter_jsonl(questions_path):
            if not q.get("eviscope_retained"):
                yield q
                continue
            anchors = list(q.get("anchor_fact_ids", []))
            subs = [d["fact_id"] for d in q.get("substitutable_evidence", [])]
            dis = [d["fact_id"] for d in q.get("distracting_evidence", [])]
            rng = random.Random(int(sha256_text(f"{seed}|{q['paired_id']}")[:16], 16))
            # background: facts whose interval overlaps the question's time span
            q_years: set[int] = set()
            for event in q.get("source_events", []):
                parts = str(event).split("|")
                if len(parts) == 5:
                    for raw in (parts[3], parts[4]):
                        try:
                            q_years.add(int(raw[:4]))
                        except ValueError:
                            pass
            for t in q.get("time_constraints", []):
                try:
                    q_years.add(int(str(t)[:4]))
                except ValueError:
                    pass
            background: list[str] = []
            for y in q_years:
                background += index.by_year.get(y, [])
            control = list(dict.fromkeys(anchors))
            attempts = 0
            while len(control) < min(pool_size, len(index.ids)) and attempts < pool_size * 20:
                fid = background[rng.randrange(len(background))] if background \
                    else index.ids[rng.randrange(len(index.ids))]
                attempts += 1
                if fid not in control:
                    control.append(fid)
            rng.shuffle(control)
            control_id = f"control:{sha256_text(f'{seed}|{q['paired_id']}')[:16]}"
            eviscope_id = f"eviscope:{sha256_text(f'{seed}|{q['paired_id']}')[:16]}"
            # EviScope pool: anchors preserved, equal neutral background
            # replaced by substitutes + distractors (section 16)
            replacements = list(dict.fromkeys(subs + dis))
            background_slots = [fid for fid in control if fid not in set(anchors)]
            eviscope = list(control)
            for slot_index in range(len(background_slots) - 1, -1, -1):
                if not replacements:
                    break
                eviscope[eviscope.index(background_slots[slot_index])] = replacements.pop(0)
            assert set(anchors) <= set(eviscope), "anchors must never be removed from pools"
            assert set(anchors) <= set(control), "control pool must keep anchors"
            control_handle.write(canonical_json({
                "candidate_pool_id": control_id,
                "paired_id": q["paired_id"],
                "benchmark_split": q["benchmark_split"],
                "pool_kind": "source_control",
                "pool_size": len(control),
                "anchor_fact_ids": anchors,
                "candidate_fact_ids": control,
                "shuffle_seed": seed,
                "pool_version": "1.0.0",
            }) + "\n")
            eviscope_handle.write(canonical_json({
                "candidate_pool_id": eviscope_id,
                "paired_id": q["paired_id"],
                "benchmark_split": q["benchmark_split"],
                "pool_kind": "eviscope",
                "pool_size": len(eviscope),
                "anchor_fact_ids": anchors,
                "candidate_fact_ids": eviscope,
                "shuffle_seed": seed,
                "pool_version": "1.0.0",
            }) + "\n")
            q["control_pool_id"] = control_id
            q["eviscope_pool_id"] = eviscope_id
            stats["pools"] += 2
            yield q

    write_jsonl(questions_output, gen())
    control_handle.close()
    eviscope_handle.close()
    report = dict(stats)
    write_json(report_path, report)
    return report


# --------------------------------------------------------------------------- #
# 11_compute_hdm
# --------------------------------------------------------------------------- #

def compute_hdm(config: dict[str, Any], questions_path: Path, index: FactIndex,
                encoder: FrozenEncoder, output: Path, tau_fit_path: Path | None,
                boundaries_path: Path | None, root: Path,
                report_path: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    neighbor_index = NeighborIndex(encoder, index.facts_path, index.ids, index.facts)
    tau_amb = None
    if tau_fit_path is not None and tau_fit_path.exists():
        loaded = next(iter_jsonl(tau_fit_path))
        tau_amb = loaded.get("tau_amb")
    else:
        # fit on Train: median anchor->nearest-non-self cosine (section 14);
        # streaming pass 0 collects the fit values only
        from array import array
        train_top = array("d")
        seen_anchors: set[str] = set()
        for q in iter_jsonl(questions_path):
            if q["benchmark_split"] != "train" or not q.get("eviscope_retained"):
                continue
            for fid in q.get("anchor_fact_ids", []):
                if fid in seen_anchors:
                    continue
                seen_anchors.add(fid)
                fact = index.facts.get(fid)
                if fact is None:
                    continue
                emb = encoder.embed(fact.get("canonical_fact_text") or fact.get("fact_text", ""))
                train_top.append(neighbor_index.top_neighbor_cosine(emb, fid))
        sorted_top = sorted(train_top)
        tau_amb = sorted_top[len(sorted_top) // 2] if sorted_top else 0.0
        if tau_fit_path is not None:
            write_jsonl(tau_fit_path, [{"tau_amb": round(tau_amb, 6),
                                        "fit_source": "train",
                                        "fit_n_anchors": len(train_top),
                                        "encoder_revision": encoder.revision}])
    # neighborhood sizes (memoized per distinct anchor); streaming pass 1
    # computes C/A/D for every question and writes the file
    neighborhood_cache: dict[str, int] = {}
    from array import array
    train_d = array("d")

    def compute(q: dict[str, Any]) -> dict[str, Any]:
        if not q.get("eviscope_retained"):
            return q
        anchors = q.get("anchor_fact_ids", [])
        sizes: list[int] = []
        computable = True
        for fid in anchors:
            if fid not in neighborhood_cache:
                fact = index.facts.get(fid)
                if fact is None:
                    computable = False
                    break
                emb = encoder.embed(fact.get("canonical_fact_text") or fact.get("fact_text", ""))
                if not emb:
                    computable = False
                    break
                neighborhood_cache[fid] = neighbor_index.neighborhood_size(emb, tau_amb)
            sizes.append(neighborhood_cache[fid])
        if not computable:
            q["eviscope_retained"] = False
            q["excluded"] = True
            q.setdefault("exclusion_reasons", []).append("hdm_not_computable")
            return q
        n = len(anchors)
        c_q = n
        # A(q) = (sum_i |N(fi)| - N) / N^2   (paper formula, section 15.3)
        a_q = (sum(sizes) - n) / (n * n) if n else 0.0
        d_q = c_q * (1 + a_q)
        q["reasoning_demand"] = {"anchor_count": n, "C_q": n}
        q["ambiguity_pressure"] = {
            "candidate_universe": "full_normalized_fact_space",
            "neighborhood_sizes": sizes,
            "tau_amb": round(tau_amb, 6),
            "A_q": round(a_q, 6),
        }
        q["hybrid_difficulty"] = {
            "D_q": round(d_q, 6),
            "difficulty_level": None,
            "boundary_version": "hdm_v1",
        }
        if q["benchmark_split"] == "train":
            train_d.append(d_q)
        return q

    write_jsonl(output, (compute(q) for q in iter_jsonl(questions_path)))
    boundaries = None
    if boundaries_path is not None and boundaries_path.exists():
        boundaries = next(iter_jsonl(boundaries_path))
    else:
        ordered = sorted(train_d)
        b1 = ordered[int((len(ordered) - 1) * 0.333)] if ordered else 0.0
        b2 = ordered[int((len(ordered) - 1) * 0.667)] if ordered else 0.0
        boundaries = {"b1": b1, "b2": b2, "fit_source": "train",
                      "boundary_version": "hdm_v1", "n_train": len(train_d)}
        if boundaries_path is not None:
            write_jsonl(boundaries_path, [boundaries])
    # streaming pass 2: fill difficulty levels with the frozen boundaries
    level_counts: Counter[str] = Counter()

    def level(q: dict[str, Any]) -> dict[str, Any]:
        if q.get("eviscope_retained") and "hybrid_difficulty" in q:
            d = q["hybrid_difficulty"]["D_q"]
            value = "simple" if d <= boundaries["b1"] else \
                ("medium" if d <= boundaries["b2"] else "complex")
            q["hybrid_difficulty"]["difficulty_level"] = value
            level_counts[value] += 1
        return q

    rewrite_jsonl(output, output, level)
    report = {
        "tau_amb": round(tau_amb, 6),
        "boundaries": boundaries,
        "encoder_revision": encoder.revision,
        "neighborhood_cache_size": len(neighborhood_cache),
        "level_counts": dict(level_counts),
    }
    write_json(report_path, report)
    return report


# --------------------------------------------------------------------------- #
# 12_run_audits
# --------------------------------------------------------------------------- #

def run_audits(config: dict[str, Any], questions_path: Path, facts_path: Path,
               pools_path: Path, source_report: dict[str, Any], encoder: FrozenEncoder,
               output: Path, hdm_output: Path, evidence_output: Path,
               split_output: Path) -> dict[str, Any]:
    """Section 20/21/22 audits.  Fully streamed: one pass over the questions
    (per-question checks + small global sets), plus streaming passes over the
    two candidate-pool files (never loaded into memory)."""
    fact_ids = {f["fact_id"] for f in iter_jsonl(facts_path)}
    errors: Counter[str] = Counter()
    paired_splits: dict[str, set[str]] = defaultdict(set)
    icews_surface_splits: dict[str, set[str]] = defaultdict(set)
    icews_family_splits: dict[str, set[str]] = defaultdict(set)
    test_facts: dict[str, set[bytes]] = {"anchor": set(), "sub": set(), "dis": set()}
    train_label_facts: set[bytes] = set()
    retained_by_split: dict[str, set[str]] = defaultdict(set)
    exclusion_reasons: Counter[str] = Counter()
    n_questions = 0
    n_retained = 0
    n_anchors = n_subs = n_dis = 0
    coverage_sum = 0.0

    def check(condition: bool, label: str) -> None:
        if not condition:
            errors[label] += 1

    for q in iter_jsonl(questions_path):
        n_questions += 1
        paired_splits[q["paired_id"]].add(q["benchmark_split"])
        # section 5 / 20: benchmark split MUST equal source split
        check(q["benchmark_split"] == q["source_split"], "split_mismatch")
        if not q.get("eviscope_retained"):
            exclusion_reasons.update(q.get("exclusion_reasons", []))
            continue
        n_retained += 1
        retained_by_split[q["benchmark_split"]].add(q["paired_id"])
        if config["dataset"] == "icews_actor":
            icews_surface_splits[q.get("normalized_question", "")].add(
                q["benchmark_split"])
            icews_family_splits[q.get("family_id", "")].add(q["benchmark_split"])
        anchors = set(q.get("anchor_fact_ids", []))
        subs = {d["fact_id"] for d in q.get("substitutable_evidence", [])}
        dis = {d["fact_id"] for d in q.get("distracting_evidence", [])}
        check(bool(anchors), "empty_anchor_set")
        check(q.get("proof_verified") and q.get("minimality_verified"),
              "anchor_verification_missing")
        # ICEWS V2 permits anchor-valid questions with either optional evidence
        # role absent. Available roles are still fully verified below; missing
        # roles control view eligibility and are descriptive, not audit failures.
        if config["dataset"] != "icews_actor":
            check(bool(subs), "no_substitute")
            check(bool(dis), "no_distractor")
        check(not (anchors & dis), "anchor_distractor_overlap")
        check(not (subs & dis), "substitute_distractor_overlap")
        check((anchors | subs | dis) <= fact_ids, "unknown_evidence_fact_id")
        for d in q.get("substitutable_evidence", []):
            check(d.get("verification") == "verified_exact" and d.get("answer_preservation"),
                  "unverified_substitute_in_gold")
            check(all(k in d for k in ("semantic_score", "relation_score",
                                       "temporal_score", "substitutability_score",
                                       "threshold")), "substitute_missing_scores")
        for d in q.get("distracting_evidence", []):
            check(d.get("verification") == "verified_non_supporting"
                  and d.get("support_check") is False, "supporting_fact_labeled_distractor")
            check(bool(d.get("top_level_type")) and bool(d.get("subtype")),
                  "distractor_missing_taxonomy")
        if q["benchmark_split"] == "test":
            test_facts["anchor"].update(fact_id_digest(fid) for fid in anchors)
            test_facts["sub"].update(fact_id_digest(fid) for fid in subs)
            test_facts["dis"].update(fact_id_digest(fid) for fid in dis)
        if q["benchmark_split"] == "train":
            train_label_facts.update(fact_id_digest(fid) for fid in (subs | dis))
        # section 21: HDM recomputation
        check(q.get("reasoning_demand", {}).get("C_q") == len(q.get("anchor_fact_ids", [])),
              "hdm_c_mismatch")
        sizes = q.get("ambiguity_pressure", {}).get("neighborhood_sizes", [])
        n_anchors_q = len(q.get("anchor_fact_ids", []))
        if n_anchors_q and sizes:
            a_recomputed = (sum(sizes) - n_anchors_q) / (n_anchors_q * n_anchors_q)
            check(abs(a_recomputed - q["ambiguity_pressure"]["A_q"]) < 1e-6,
                  "hdm_a_not_recomputable")
            d_recomputed = n_anchors_q * (1 + a_recomputed)
            check(abs(d_recomputed - q["hybrid_difficulty"]["D_q"]) < 1e-6,
                  "hdm_d_formula_violated")
        else:
            check(False, "hdm_neighborhood_missing")
        check(q["hybrid_difficulty"].get("difficulty_level") in
              ("simple", "medium", "complex"), "hdm_level_missing")
        n_anchors += len(anchors)
        n_subs += len(subs)
        n_dis += len(dis)
        coverage_sum += q.get("substitute_anchor_coverage", 0.0)
    # section 20: the test question's GOLD-PROOF pairing (its Anchor facts)
    # must never carry a train role label — mandatory zero.  Test Substitute /
    # Distractor facts overlapping train labels are the ordinary KG-fact
    # overlap that section 20 explicitly allows; they are reported below.
    allowed_overlap = {
        "test_substitute_facts_in_train_labels": len(test_facts["sub"] & train_label_facts),
        "test_distractor_facts_in_train_labels": len(test_facts["dis"] & train_label_facts),
    }
    test_anchor_overlap = len(test_facts["anchor"] & train_label_facts)
    if config["dataset"] == "icews_actor":
        # ICEWS facts live in one shared read-only KG.  Fact-id overlap is
        # allowed by the construction spec; leakage is checked on question
        # identity, model-visible surface and semantic family instead.
        allowed_overlap["test_anchor_facts_in_train_labels"] = test_anchor_overlap
        errors["icews_question_surface_cross_split"] = sum(
            1 for splits in icews_surface_splits.values() if len(splits) > 1)
        errors["icews_semantic_family_cross_split"] = sum(
            1 for splits in icews_family_splits.values() if len(splits) > 1)
    else:
        # Preserve the existing CronQuestions audit contract.
        errors["test_anchor_facts_in_train_labels"] = test_anchor_overlap
    errors["paired_id_cross_split"] = sum(1 for s in paired_splits.values() if len(s) > 1)
    # threshold sources (section 20/21)
    thresholds = next(iter_jsonl(output.parent / "07_thresholds.jsonl")) \
        if (output.parent / "07_thresholds.jsonl").exists() else None
    hdm_boundaries = next(iter_jsonl(output.parent / "11_hdm_boundaries.jsonl")) \
        if (output.parent / "11_hdm_boundaries.jsonl").exists() else None
    check(thresholds is not None and thresholds.get("fit_source") == "train",
          "threshold_not_train_fitted")
    check(hdm_boundaries is not None and hdm_boundaries.get("fit_source") == "train",
          "hdm_boundaries_not_train_fitted")
    # paired views: stream the pool files (anchors-preserved is asserted at
    # construction time in stage 10; here we verify existence, uniqueness,
    # split consistency and complete coverage)
    pool_ids: dict[str, set[str]] = {}
    for kind in ("source_control", "eviscope"):
        pool_file = pools_path / kind / "candidate_pools.jsonl"
        ids: set[str] = set()
        if pool_file.exists():
            for row in iter_jsonl(pool_file):
                pid = row.get("paired_id")
                if pid in ids:
                    errors[f"pool_duplicate_paired_id_{kind}"] += 1
                ids.add(pid)
                split = row.get("benchmark_split")
                if pid is not None and pid not in retained_by_split.get(split or "", ()):
                    errors[f"pool_row_not_retained_{kind}"] += 1
        else:
            errors[f"pool_file_missing_{kind}"] += 1
        pool_ids[kind] = ids
    for kind in ("source_control", "eviscope"):
        for split, ids in retained_by_split.items():
            missing = ids - pool_ids.get(kind, set())
            errors[f"pool_missing_{kind}"] += len(missing)
    mandatory = {key: int(value) for key, value in errors.items()}
    split_report = {
        "status": "pass" if not any(mandatory.values()) else "fail",
        "dataset": config["dataset"],
        "retained": n_retained,
        "questions": n_questions,
        "mandatory_checks": mandatory,
        "thresholds": thresholds,
        "hdm_boundaries": hdm_boundaries,
        "encoder_revision": encoder.revision,
    }
    leakage_report = {
        "status": "pass" if not any(mandatory.values()) else "fail",
        "dataset": config["dataset"],
        "mandatory_checks": mandatory,
        "allowed_nonzero_metrics": allowed_overlap,
        "thresholds": thresholds,
        "hdm_boundaries": hdm_boundaries,
        "encoder_revision": encoder.revision,
        "notes": [
            "split consistency, evidence validation, HDM and BS checks per "
            "sections 20/21/22; any non-zero mandatory item blocks release.",
            "Entity/relation/time/fact overlap across splits is allowed and "
            "reported separately. For ICEWS-Actor, shared raw fact IDs are "
            "descriptive overlap metrics while paired IDs, model-visible "
            "question surfaces and semantic families must remain split-"
            "disjoint. CronQuestions retains its existing stricter fact-role "
            "audit contract.",
        ],
    }
    write_json(split_output, split_report)
    write_json(output, leakage_report)
    evidence_report = {
        "retained": n_retained,
        "excluded": n_questions - n_retained,
        "exclusion_reasons": dict(exclusion_reasons),
        "substitute_anchor_coverage_mean": round(coverage_sum / n_retained, 6)
        if n_retained else 0.0,
        "anchors": n_anchors,
        "substitutes": n_subs,
        "distractors": n_dis,
        "per_split": {
            split: {"retained": len(retained_by_split.get(split, set())),
                    "excluded": errors.get("split_mismatch", 0)}  # replaced below
            for split in ("train", "dev", "test")},
    }
    # per-split excluded counts need a second quick streaming pass
    per_split_questions: Counter[str] = Counter()
    for q in iter_jsonl(questions_path):
        per_split_questions[q["benchmark_split"]] += 1
    evidence_report["per_split"] = {
        split: {
            "retained": len(retained_by_split.get(split, set())),
            "excluded": per_split_questions.get(split, 0)
            - len(retained_by_split.get(split, set())),
        }
        for split in ("train", "dev", "test")}
    write_json(evidence_output, evidence_report)
    hdm_report = {
        "encoder_revision": encoder.revision,
        "tau_amb": next(iter_jsonl(output.parent / "11_tau_amb.jsonl"))["tau_amb"]
        if (output.parent / "11_tau_amb.jsonl").exists() else None,
        "hdm_checks": {k: v for k, v in mandatory.items() if k.startswith("hdm_")},
    }
    write_json(hdm_output, hdm_report)
    return leakage_report


# --------------------------------------------------------------------------- #
# 13_export_release
# --------------------------------------------------------------------------- #

_PRIVATE_FIELDS = {"gold_answers", "logic_program", "anchor_fact_ids",
                   "substitutable_evidence", "distracting_evidence",
                   "ambiguity_pressure", "hybrid_difficulty", "reasoning_demand",
                   "answer_revision", "candidate_scores", "pair_scores",
                   "recall_candidates", "anchor_support_fact_ids",
                   "uncertain_candidates", "excluded", "eviscope_retained"}
_INTERNAL = {"anchor_support_fact_ids", "candidate_scores", "pair_scores",
             "recall_candidates", "uncertain_candidates", "excluded",
             "eviscope_retained", "source_split_raw", "question_sha256",
             "answer_sha256", "_source_file", "_source_row"}


def _strip(question: dict[str, Any], private: bool) -> dict[str, Any]:
    keys = set(_INTERNAL)
    if private:
        keys |= _PRIVATE_FIELDS
    return {key: value for key, value in question.items() if key not in keys}


def export_release(config: dict[str, Any], questions_path: Path, facts_path: Path,
                   pools_dir: Path, output_root: Path, source_report: dict[str, Any],
                   encoder: FrozenEncoder, version: str, builder_root: Path) -> dict[str, Any]:
    dataset = config["dataset"]
    minimums = config.get("minimum_release_size", {}) \
        if dataset == "icews_actor" and not config.get("_allow_undersized_pilot") else {}
    if minimums:
        preflight: Counter[str] = Counter()
        for q in iter_jsonl(questions_path):
            if q.get("eviscope_retained"):
                preflight["total"] += 1
                preflight[q["benchmark_split"]] += 1
        failures = []
        for scope in ("total", "train", "dev", "test"):
            minimum = int(minimums.get(scope, 0) or 0)
            if minimum and preflight[scope] < minimum:
                failures.append(f"{scope}={preflight[scope]} < {minimum}")
        if failures:
            raise RuntimeError(
                "ICEWS-Actor export blocked by release-size gate: "
                + ", ".join(failures)
            )
    target = output_root / dataset
    for sub in ("source_view", "eviscope_view", "private_labels", "facts",
                "candidate_pools", "statistics", "audit", "manifests"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(facts_path, target / "facts" / "facts.jsonl")
    # single streaming pass: write each retained question to its view files
    # and accumulate all statistics in counters (never load the whole file)
    counts: Counter[str] = Counter()
    split_candidates: Counter[str] = Counter()
    split_exclusion_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    n_anchors = n_subs = n_dis = 0
    top_levels: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    sum_c = sum_a = sum_d = 0.0
    control_ids: set[str] = set()
    eviscope_ids: set[str] = set()
    paired_ids: set[str] = set()

    handles = {}
    for split in ("train", "dev", "test"):
        handles[f"source:{split}"] = (target / "source_view" / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n")
        handles[f"eviscope:{split}"] = (target / "eviscope_view" / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n")
    handles["eviscope:test_public"] = (target / "eviscope_view" / "test_public.jsonl").open(
        "w", encoding="utf-8", newline="\n")
    handles["private:test"] = (target / "private_labels" / "test.jsonl").open(
        "w", encoding="utf-8", newline="\n")

    try:
        for q in iter_jsonl(questions_path):
            split = q.get("benchmark_split")
            split_candidates[split] += 1
            if not q.get("eviscope_retained"):
                split_exclusion_reasons[split].update(q.get("exclusion_reasons", []))
                continue
            counts[split] += 1
            paired_ids.add(q["paired_id"])
            control_ids.add(q["control_pool_id"])
            eviscope_ids.add(q["eviscope_pool_id"])
            n_anchors += len(q.get("anchor_fact_ids", []))
            subs = q.get("substitutable_evidence", [])
            dis = q.get("distracting_evidence", [])
            n_subs += len(subs)
            n_dis += len(dis)
            top_levels.update(d["top_level_type"] for d in dis)
            subtypes.update(d["subtype"] for d in dis)
            level_counts[q["hybrid_difficulty"]["difficulty_level"]] += 1
            sum_c += q["reasoning_demand"]["C_q"]
            sum_a += q["ambiguity_pressure"]["A_q"]
            sum_d += q["hybrid_difficulty"]["D_q"]
            source_row = _strip(q, private=False)
            eviscope_row = _strip(q, private=False)
            handles[f"source:{split}"].write(canonical_json(source_row) + "\n")
            handles[f"eviscope:{split}"].write(canonical_json(eviscope_row) + "\n")
            if split == "test":
                handles["eviscope:test_public"].write(
                    canonical_json(_strip(q, private=True)) + "\n")
                handles["private:test"].write(canonical_json(eviscope_row) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for kind in ("source_control", "eviscope"):
        src = pools_dir / kind / "candidate_pools.jsonl"
        if src.exists():
            dst = target / "candidate_pools" / kind / "candidate_pools.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    total_retained = sum(counts.values())
    total_candidates = sum(split_candidates.values())
    # statistics (完整目标规模说明 supplement: candidate/excluded/retained
    # per split with reasons, never padded)
    per_split = {}
    for split in ("train", "dev", "test"):
        per_split[split] = {
            "source_candidates": split_candidates.get(split, 0),
            "retained": counts.get(split, 0),
            "excluded": split_candidates.get(split, 0) - counts.get(split, 0),
            "exclusion_reasons": dict(split_exclusion_reasons.get(split, Counter())),
        }
    source_statistics = {
        "dataset": dataset,
        "source_counts": config.get("source_counts", {}),
        "source_split_counts_actual": source_report.get("source_split_counts_mapped", {}),
        "target_benchmark_total_reference": config.get("target_benchmark_total"),
        "per_split": per_split,
        "total_retained": total_retained,
        "total_excluded": total_candidates - total_retained,
    }
    write_json(target / "statistics" / "source_statistics.json", source_statistics)
    evidence_statistics = {
        "anchors": n_anchors,
        "substitutes": n_subs,
        "distractors": n_dis,
        "distractor_top_levels": dict(top_levels),
        "distractor_subtypes": dict(subtypes),
    }
    write_json(target / "statistics" / "evidence_statistics.json", evidence_statistics)
    hdm_statistics = {
        "encoder_revision": encoder.revision,
        "level_counts": dict(level_counts),
        "mean_C_q": round(sum_c / total_retained, 6) if total_retained else 0.0,
        "mean_A_q": round(sum_a / total_retained, 6) if total_retained else 0.0,
        "mean_D_q": round(sum_d / total_retained, 6) if total_retained else 0.0,
    }
    write_json(target / "statistics" / "hdm_statistics.json", hdm_statistics)
    paired_statistics = {
        "paired_question_count": total_retained,
        "control_pools": len(control_ids),
        "eviscope_pools": len(eviscope_ids),
        "view_row_count": {"source_view": total_retained, "eviscope_view": total_retained},
        "paired_id_uniqueness_verified": len(paired_ids) == total_retained,
    }
    write_json(target / "statistics" / "paired_statistics.json", paired_statistics)
    # manifests copy (section 6/26)
    mdir = _manifest_dir(dataset, builder_root)
    for name in ("source_manifest.yaml", f"{dataset}_source_split.jsonl",
                 "paired_question_manifest.jsonl", "exclusion_manifest.jsonl",
                 "quarantine_manifest.jsonl", "manifest_hashes.json"):
        src = mdir / name
        if src.exists():
            shutil.copyfile(src, target / "manifests" / name)
    # audit copies
    interim = questions_path.parent
    for name, out in (("12_leakage_report.json", "leakage_report.json"),
                      ("12_evidence_validation_report.json", "evidence_validation_report.json"),
                      ("12_hdm_validation_report.json", "hdm_validation_report.json"),
                      ("12_split_consistency_report.json", "split_consistency_report.json"),
                      ("construction_log.jsonl", "construction_log.jsonl")):
        src = interim / name
        if src.exists():
            shutil.copyfile(src, target / "audit" / out)
    (target / "VERSION").write_text(version + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        f"# EviScope-{dataset} ({version})\n\n"
        "- Construction spec: EviScope数据基准从零构建方案.md (paper-verbatim "
        "Evidence Space Reconstruction / Hybrid Difficulty Modeling / BS@K, "
        "source splits inherited, train-only threshold fitting).\n"
        "- Scale: actual valid samples after all validation gates — see "
        "statistics/source_statistics.json for per-split candidate/excluded/"
        "retained counts and reasons (完整目标规模说明).\n"
        "- Views: source_view/ and eviscope_view/ share paired_id and splits; "
        "candidate_pools/{source_control,eviscope}/ are paired pools; "
        "private_labels/ holds test supervision for the official evaluator "
        "(evaluator/ package, frozen encoder and bs_tau in "
        "construction_config.yaml).\n"
        "- Audits: audit/leakage_report.json must be status=pass for release.\n",
        encoding="utf-8")
    (target / "LICENSES.md").write_text(
        f"# Licenses and source notices\n\n- Source: {config['source_release']}\n"
        f"- License: {config['license']}\n"
        "- This benchmark redistributes normalized facts and questions derived "
        "from the sources above; retain the original attributions.\n",
        encoding="utf-8")
    # frozen construction config (section 14/26): dataset config + auxiliary
    # sections + TRAIN-FITTED threshold values, with the encoder revision
    from .common import dump_yaml, read_json
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    thresholds_path = questions_path.parent / "07_thresholds.jsonl"
    if thresholds_path.exists():
        thresholds = next(iter_jsonl(thresholds_path))
        resolved.setdefault("evidence_scoring", {}).update({
            "tau_eq": thresholds.get("tau_eq"),
            "tau_min": thresholds.get("tau_min"),
            "tau_sim": thresholds.get("tau_sim"),
        })
        resolved.setdefault("evaluation", {}).update({"bs_tau": thresholds.get("bs_tau")})
    tau_path = questions_path.parent / "11_tau_amb.jsonl"
    if tau_path.exists():
        tau_entry = next(iter_jsonl(tau_path))
        resolved.setdefault("hdm", {}).update({"tau_amb": tau_entry.get("tau_amb")})
    resolved["encoder_revision"] = encoder.revision
    resolved["threshold_fit_source"] = "train"
    write_json(target / "construction_config.json", resolved)
    with (target / "construction_config.yaml").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(dump_yaml(resolved) + "\n")
    return {"release_dir": str(target), "question_counts": dict(counts),
            "total_retained": total_retained}
