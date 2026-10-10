from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / "main_benchmark"
sys.path.insert(0, str(EXP1))
from _shared.config import SEED, dataset_spec, model_paths
from _shared.io import iter_jsonl, write_json, write_jsonl
from _shared.metrics import FrozenHashEncoder, evaluate
CONDITIONS = ("original", "sub", "dis", "full")
def load_facts(path: Path) -> list[dict]:
    facts = list(iter_jsonl(path))
    if any(int(row.get("fact_index", -1)) != i for i, row in enumerate(facts)): raise ValueError("fact catalog is not dense")
    return facts
def load_prepared(dataset: str):
    root = HERE / "prepared" / dataset
    questions = list(iter_jsonl(root / "model_input" / "test_questions.jsonl")); labels = list(iter_jsonl(root / "trusted" / "test_labels.jsonl")); train = list(iter_jsonl(root / "trusted" / "train_labels.jsonl"))
    pools = {mode: {row["question_id"]: row["candidate_fact_indices"] for row in iter_jsonl(root / "model_input" / "test_pools" / f"{mode}.jsonl")} for mode in ("clean", "substitute", "distractor", "full")}
    ids = {row["question_id"] for row in questions}
    if any(set(pool) != ids for pool in pools.values()): raise ValueError("condition pools do not share the paired cohort")
    return questions, labels, train, pools
def labels_for_condition(labels: list[dict], condition: str) -> list[dict]:
    valid = condition in ("sub", "full")
    return [{**row, "substitutes": row["substitutes"] if valid else []} for row in labels]
def save_and_evaluate(output: Path, predictions: list[dict], labels: list[dict], facts: list[dict], encoder: FrozenHashEncoder, tau: float) -> dict:
    write_jsonl(output / "predictions.jsonl", predictions); result = {}
    for k in (1, 10):
        summary, per_question = evaluate(predictions, labels, facts, encoder, tau, k); result[f"k{k}"] = summary; write_json(output / f"metrics_k{k}.json", summary); write_jsonl(output / f"per_question_k{k}.jsonl", per_question)
    return result
def make_retriever(method: str, source: Path | None, facts: list[dict], device: str):
    if method == "bm25":
        from bm25 import BM25Retriever
        return BM25Retriever(facts)
    if method in ("dual_encoder", "eviret"):
        from dual_encoder import DualEncoderRetriever
        return DualEncoderRetriever(source, device)
    from cross_encoder import CrossEncoderRetriever
    return CrossEncoderRetriever(source, device)
def score_union(method: str, retriever, question: str, candidates: list[int], facts: list[dict], dense_cache=None, batch_size: int = 128):
    if method == "bm25": return retriever.score(question, candidates)
    if method in ("dual_encoder", "eviret"): return retriever.score(question, candidates, dense_cache)
    return retriever.score(question, candidates, facts, batch_size)
def top10(scores: dict[int, float], candidates: list[int]) -> list[int]:
    return sorted(candidates, key=lambda index: (-scores[index], index))[:10]
