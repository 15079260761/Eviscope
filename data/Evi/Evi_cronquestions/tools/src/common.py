"""Shared deterministic I/O, hashing and configuration helpers.

The builder only requires the Python standard library.  Configurations are
YAML per the construction document (section 19); PyYAML is used when
available and a small built-in loader covers the flat schema otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

def builder_root() -> Path:
    """Absolute path of eviscope_builder_v2/ (this file lives in src/)."""
    return Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Absolute path of the repository root (Eviscope_rebuild/)."""
    return builder_root().parent


# --------------------------------------------------------------------------- #
# hashing / canonical serialization
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fact_id(dataset: str, subject_id: str, relation_id: str,
                   object_id: str, time_start: str | None, time_end: str | None,
                   granularity: str = "") -> str:
    """Stable fact id per section 7 of the construction document: dataset +
    canonical s/r/o + normalized time bounds + time granularity."""
    payload = [dataset, subject_id, relation_id, object_id, time_start, time_end,
               granularity]
    return "sha256:" + sha256_text(canonical_json(payload))


def stable_id(prefix: str, *parts: Any) -> str:
    """Deterministic id derived from a semantic prefix plus stable parts."""
    return f"{prefix}_{sha256_text(canonical_json(list(parts)))[:24]}"


def normalize_question(text: str) -> str:
    """Level-1 deterministic normalization: casefold + punctuation strip."""
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def placeholder_template(text: str) -> str:
    """Level-1 template signature: entities/dates/years replaced by placeholders.

    Q-ids, Wikidata P-ids and 4-digit years are replaced so that surface
    variations of the same template instance collapse to one signature.
    """
    text = text.casefold().strip()
    text = re.sub(r"\bq\d+\b", "<ent>", text)
    text = re.sub(r"\bp\d+\b", "<rel>", text)
    text = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", "<time>", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# atomic file I/O
# --------------------------------------------------------------------------- #

@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent,
        prefix=path.name + ".", suffix=".tmp",
    )
    temp_path = Path(handle.name)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temp_path, path)
    except Exception:
        handle.close()
        temp_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: Any) -> None:
    with atomic_text_writer(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with atomic_text_writer(path) as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


# --------------------------------------------------------------------------- #
# configuration loading (YAML with stdlib-only fallback)
# --------------------------------------------------------------------------- #

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file.  Uses PyYAML when installed, else a conservative
    built-in parser that supports the flat mapping/list/comment schema used by
    the shipped configs (no flow syntax, no multi-line strings)."""
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return value
    except ImportError:
        return _load_yaml_minimal(path)


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text in ("", "~", "null"):
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def _load_yaml_minimal(path: Path) -> dict[str, Any]:
    """Minimal indentation-based YAML subset parser (mappings, lists, scalars,
    full-line comments, and list items that are dicts with 'key: value')."""
    root: dict[str, Any] = {}
    stack: list[tuple[str, Any]] = []  # (indent, container)
    stack.append((-1, root))
    list_indent = -1
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            content = line.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if content.startswith("- "):
                rest = content[2:].strip()
                if ":" in rest and not rest.startswith(("{", "[", '"', "'")):
                    key, _, val = rest.partition(":")
                    item = {key.strip(): _parse_scalar(val)}
                else:
                    item = _parse_scalar(rest)
                if not isinstance(parent, list):
                    raise ValueError(f"YAML list item without list context at {path}: {content}")
                parent.append(item)
                list_indent = indent
                stack.append((indent, item))
            elif ":" in content:
                key, _, val = content.partition(":")
                key = key.strip()
                val = val.strip()
                if val == "":
                    child: Any = {}
                elif val.startswith("[") and val.endswith("]"):
                    child = [_parse_scalar(x) for x in val[1:-1].split(",")]
                else:
                    child = _parse_scalar(val)
                if isinstance(parent, list):
                    raise ValueError(f"Unexpected mapping line in list context at {path}: {content}")
                parent[key] = child
                stack.append((indent, child))
            else:
                raise ValueError(f"Unparseable YAML line at {path}: {content}")
    return root


def load_config(dataset: str, builder_root_path: Path | None = None) -> dict[str, Any]:
    root = builder_root_path or builder_root()
    path = root / "configs" / f"{dataset}.yaml"
    config = load_yaml(path)
    config["_config_path"] = str(path.resolve())
    relations_path = root / "configs" / "relations.yaml"
    if relations_path.exists():
        config["_relations"] = load_yaml(relations_path)
    else:
        config["_relations"] = {"equivalence_groups": [], "inverse_pairs": []}
    # auxiliary frozen configs (sections 12-17)
    for name, prefix in (("evidence_scoring", "evidence_scoring"),
                         ("hdm", "hdm"), ("evaluation", "evaluation")):
        aux = root / "configs" / f"{name}.yaml"
        if aux.exists():
            loaded = load_yaml(aux)
            config[prefix] = loaded.get(prefix, loaded)
            config[f"_{prefix}_path"] = str(aux.resolve())
    return config


def load_aux_config(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Auxiliary section config (e.g. evidence_scoring) merged over the
    dataset config."""
    merged = dict(config.get(section, {}) or {})
    for key, value in (config.get(section, {}) or {}).items():
        if key not in merged:
            merged[key] = value
    return merged


def dump_yaml(value: Any, indent: int = 0) -> str:
    """Minimal deterministic YAML emitter for flat dict/list/scalar trees
    (used for the frozen manifests and construction config, sections 6/26)."""
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key in sorted(value):
            item = value[key]
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict) and item:
                inner = dump_yaml(item, indent + 1)
                first, rest = inner.split("\n", 1)
                lines.append(f"{pad}- {first}")
                lines.append(rest)
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(value)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in ":#{}[]&*!|>'\"%@`\n"):
        return json.dumps(text, ensure_ascii=False)
    return text


def resolve_archive(config: dict[str, Any], override: str | None = None) -> Path:
    root = builder_root()
    candidate = Path(override) if override else (root / config["archive_path"])
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Source archive does not exist: {candidate}")
    return candidate


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def rewrite_jsonl(input_path: Path, output_path: Path, mutator) -> int:
    """Stream a JSONL file through `mutator(row) -> row` and atomically
    replace the original.  Used for two-pass stages whose second pass adds a
    field after global fitting (memory-safe for very large files).

    The final replace retries briefly: on Windows, antivirus/indexer handles
    on the freshly written destination can transiently block the rename."""
    import time
    tmp_path = output_path.with_name(output_path.name + ".rewrite.tmp")
    count = write_jsonl(tmp_path, (mutator(row) for row in iter_jsonl(input_path)))
    for attempt in range(10):
        try:
            os.replace(tmp_path, output_path)
            return count
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1.0)
    return count


def fact_id_digest(fact_id: str) -> bytes:
    """Compact hashable form of a fact id for large membership sets."""
    if fact_id.startswith("sha256:"):
        try:
            return bytes.fromhex(fact_id.split(":", 1)[1])
        except ValueError:
            pass
    return fact_id.encode("utf-8")


def append_log(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(entry) + "\n")
