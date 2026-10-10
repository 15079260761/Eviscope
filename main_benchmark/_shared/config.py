from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEED = 20260927
POOL_SIZE = 1000
TEST_TARGET = 2000
TRAIN_TARGET = 8000
DEV_TARGET = 1000
TOP_K = (1, 10)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    release_dir: Path
    facts: Path
    questions_dir: Path
    supervision_dir: Path
    private_test: Path
    descriptors_dir: Path
    modes: dict[str, str]
    bs_tau: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_spec(key: str, root: Path | None = None) -> DatasetSpec:
    root = root or project_root()
    if key == "cron":
        release = root / "data" / "Evi" / "Evi_cronquestions"
        return DatasetSpec(
            key=key, release_dir=release,
            facts=release / "data" / "facts" / "facts.jsonl",
            questions_dir=release / "data" / "questions",
            supervision_dir=release / "data" / "supervision",
            private_test=release / "data" / "private_labels" / "test.jsonl",
            descriptors_dir=release / "data" / "descriptors",
            modes={"full": "full"},
            bs_tau=0.871848,
        )
    if key == "icews":
        release = root / "data" / "Evi" / "Evi_icews_actor_V2"
        return DatasetSpec(
            key=key, release_dir=release,
            facts=release / "data" / "facts.jsonl",
            questions_dir=release / "data" / "questions",
            supervision_dir=release / "data" / "supervision",
            private_test=release / "private_labels" / "test.jsonl",
            descriptors_dir=release / "candidate_space" / "descriptors",
            modes={"full": "full_eviscope"},
            bs_tau=0.349259,
        )
    raise ValueError(f"unknown dataset: {key}")


def model_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    return {
        "dual": root / "models" / "contriever-msmarco",
        "cross": root / "models" / "ms-marco-MiniLM-L6-v2",
    }
