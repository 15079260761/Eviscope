"""Source verification, safe archive/directory readers and fact
normalization.

Implements stage 00 (verify_sources) and stage 01 (normalize_facts) of the
construction document.  Only the standard library is used; pickle loading is
restricted to primitive containers (executable globals are rejected).

Sources may be either a zip archive or an extracted directory (the cloud copy
uses the extracted form).  A SourceBundle presents both uniformly.
"""

from __future__ import annotations

import ast
import csv
import io
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .common import (
    canonical_json, sha256_file, stable_fact_id, utc_now, write_json,
)
from .timeutils import interval_overlap


class SourceBundle:
    """Uniform read access to a zip archive or an extracted directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kind = "zip" if path.is_file() and zipfile.is_zipfile(path) else "directory"
        if self.kind == "zip":
            self._zip = zipfile.ZipFile(path)
        elif not path.is_dir():
            raise FileNotFoundError(f"Source is neither a zip nor a directory: {path}")

    def _resolve_member(self, member: str) -> str:
        """Zip members carry the top-level folder prefix ('ICEWS/...'); an
        extracted directory may or may not keep it."""
        if self.kind == "zip":
            return member
        candidates = [member]
        if "/" in member:
            candidates.append(member.split("/", 1)[1])
        for candidate in candidates:
            if (self.path / candidate).is_file():
                return candidate
        raise FileNotFoundError(f"Missing source member: {member}")

    def names(self) -> set[str]:
        if self.kind == "zip":
            return set(self._zip.namelist())
        names: set[str] = set()
        for item in self.path.rglob("*"):
            if item.is_file():
                names.add(item.relative_to(self.path).as_posix())
        return names

    def open(self, member: str):
        if self.kind == "zip":
            return self._zip.open(member)
        return (self.path / self._resolve_member(member)).open("rb")

    def sha256(self, member: str) -> str:
        if self.kind == "zip":
            import hashlib
            digest = hashlib.sha256()
            with self._zip.open(member) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        return sha256_file(self.path / self._resolve_member(member))

    def member_size(self, member: str) -> int:
        if self.kind == "zip":
            info = self._zip.getinfo(member)
            return info.file_size
        return (self.path / self._resolve_member(member)).stat().st_size

    def close(self) -> None:
        if self.kind == "zip":
            self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def resolve_source(config: dict[str, Any], override: str | None, root: Path) -> SourceBundle:
    """Locate the source: --archive override, config archive_path (zip), or the
    extracted directory next to it (directory fallback for cloud copies)."""
    if override:
        return SourceBundle(Path(override).expanduser().resolve())
    archive_candidate = (root / config["archive_path"]).expanduser().resolve()
    if archive_candidate.is_file():
        return SourceBundle(archive_candidate)
    dir_candidate = archive_candidate.with_suffix("")
    if dir_candidate.is_dir():
        return SourceBundle(dir_candidate)
    raise FileNotFoundError(
        f"Source not found (tried zip {archive_candidate} and directory {dir_candidate})")


# --------------------------------------------------------------------------- #
# stage 00: verify sources
# --------------------------------------------------------------------------- #

class RestrictedUnpickler(pickle.Unpickler):
    """Load CronQuestions primitive containers while blocking executable globals.

    The alias pickle legitimately contains ``set`` objects; these immutable
    builtins are the only globals allowed — anything else is rejected."""

    _SAFE = {"builtins.set", "builtins.frozenset"}

    def find_class(self, module: str, name: str) -> Any:  # pragma: no cover - security path
        key = f"{module}.{name}"
        if key in self._SAFE:
            return {"builtins.set": set, "builtins.frozenset": frozenset}[key]
        raise pickle.UnpicklingError(f"Unsafe pickle global blocked: {key}")


def verify_source(config: dict[str, Any], bundle: SourceBundle, output: Path) -> dict[str, Any]:
    """Verify source integrity, required members and provenance locks."""
    report: dict[str, Any] = {
        "dataset": config["dataset"],
        "checked_at": utc_now(),
        "source_path": str(bundle.path),
        "source_kind": bundle.kind,
        "required_members": config["required_members"],
        "missing_members": [],
        "member_sha256": {},
        "member_sizes": {},
        "release_locks": {},
        "status": "pass",
    }
    names = bundle.names()
    report["missing_members"] = []
    for member in config["required_members"]:
        try:
            bundle._resolve_member(member)
            report["member_sha256"][member] = bundle.sha256(member)
            report["member_sizes"][member] = bundle.member_size(member)
        except FileNotFoundError:
            report["missing_members"].append(member)
    for key in ("source_release", "license", "temporal_coverage"):
        value = str(config.get(key, ""))
        locked = bool(value) and "LOCK_BEFORE" not in value and "VERIFY_BEFORE" not in value
        report["release_locks"][key] = {"value": value, "locked": locked}
    if report["missing_members"]:
        report["status"] = "fail"
    elif any(not item["locked"] for item in report["release_locks"].values()):
        report["status"] = "pilot_only"
    write_json(output, report)
    if report["status"] == "fail":
        raise ValueError(f"Source verification failed; see {output}")
    return report


# --------------------------------------------------------------------------- #
# stage 01: fact normalization
# --------------------------------------------------------------------------- #

def _canonical_text(text: str) -> str:
    """Deterministic canonical surface form (section 7): casefolded,
    punctuation normalized, whitespace collapsed.  Active/passive
    canonicalization requires the curated relation table; without it the
    canonical form is the normalized surface form (documented)."""
    normalized = re.sub(r"[^\w\s|]", " ", text.casefold())
    return " ".join(normalized.split())


def _parse_iso_date(raw: str, *, boundary: str, minimum: int, maximum: int):
    """Return (normalized, error).  Year-only sources become full year spans."""
    value = str(raw).strip()
    if value.casefold() in {"", "none", "null", "beginning of time", "end of time"}:
        return None, None
    try:
        parsed = re.fullmatch(r"(\d{1,4})", value)
        if parsed:
            year = int(value)
            if not minimum <= year <= maximum:
                return None, "year_out_of_allowed_range"
            return (f"{year:04d}-01-01" if boundary == "start" else f"{year:04d}-12-31"), None
        from datetime import date as _date
        d = _date.fromisoformat(value)
        if not minimum <= d.year <= maximum:
            return None, "year_out_of_allowed_range"
        return d.isoformat(), None
    except ValueError:
        return None, "invalid_iso_date"


def _load_tsv_mapping(bundle: SourceBundle, member: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with bundle.open(member) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            key, separator, value = line.rstrip("\r\n").partition("\t")
            if separator:
                mapping[key] = value
    return mapping


def normalize_facts(config: dict[str, Any], bundle: SourceBundle, facts_path: Path,
                    quarantine_path: Path, statistics_path: Path,
                    max_records: int | None = None) -> dict[str, Any]:
    """Stream source facts to the common five-field temporal schema (6.1/6.5)."""
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    seen: set[str] = set()

    with facts_path.open("w", encoding="utf-8", newline="\n") as good, \
            quarantine_path.open("w", encoding="utf-8", newline="\n") as bad:
        if config["dataset"] == "icews_actor":
            source = io.TextIOWrapper(bundle.open(config["facts_member"]), encoding="utf-8-sig", newline="")
            records = ((line, row) for line, row in enumerate(csv.DictReader(source), 2))
            entity_names: dict[str, str] = {}
            relation_names: dict[str, str] = {}
            granularity = "day"
        else:
            entity_names = _load_tsv_mapping(bundle, config["entity_map_member"])
            relation_names = _load_tsv_mapping(bundle, config["relation_map_member"])
            alias_map: dict[str, list[str]] = {}
            if config.get("alias_map_member"):
                with bundle.open(config["alias_map_member"]) as raw:
                    loaded = RestrictedUnpickler(raw).load()
                if isinstance(loaded, dict):
                    alias_map = {str(k): [str(a) for a in v] for k, v in loaded.items()
                                 if isinstance(v, (list, tuple, set))}
            source = io.TextIOWrapper(bundle.open(config["facts_member"]), encoding="utf-8", newline="")

            def cron_rows():
                for line, text in enumerate(source, 1):
                    parts = text.rstrip("\r\n").split("\t")
                    if len(parts) != 5:
                        yield line, {"_malformed": text.rstrip("\r\n")}
                    else:
                        yield line, dict(zip(("subject", "predicate", "object", "start_time", "end_time"), parts))

            records = cron_rows()
            granularity = "year"

        try:
            for line_number, row in records:
                if max_records is not None and counters["input_rows"] >= max_records:
                    break
                counters["input_rows"] += 1
                if "_malformed" in row:
                    reason = "wrong_column_count"
                    rejection_reasons[reason] += 1
                    bad.write(canonical_json({"source_line": line_number, "reason": reason, "raw": row}) + "\n")
                    continue

                subject = str(row.get("subject", "")).strip()
                relation = str(row.get("predicate", "")).strip()
                obj = str(row.get("object", "")).strip()
                if not subject or not relation or not obj:
                    reason = "missing_entity_or_relation"
                    rejection_reasons[reason] += 1
                    bad.write(canonical_json({"source_line": line_number, "reason": reason, "raw": row}) + "\n")
                    continue

                start, start_error = _parse_iso_date(
                    row.get("start_time", ""), boundary="start",
                    minimum=int(config["valid_year_min"]), maximum=int(config["valid_year_max"]))
                end, end_error = _parse_iso_date(
                    row.get("end_time", ""), boundary="end",
                    minimum=int(config["valid_year_min"]), maximum=int(config["valid_year_max"]))
                error = start_error or end_error
                if not error and start is not None and end is not None and start > end:
                    error = "inverted_interval"
                if error:
                    rejection_reasons[error] += 1
                    bad.write(canonical_json({"source_line": line_number, "reason": error, "raw": row}) + "\n")
                    continue

                raw_start = str(row.get("start_time", ""))
                raw_end = str(row.get("end_time", ""))
                fact_id = stable_fact_id(config["dataset"], subject, relation, obj,
                                         start, end, granularity)
                if fact_id in seen:
                    counters["duplicate_facts"] += 1
                    continue
                seen.add(fact_id)

                if config["dataset"] == "icews_actor":
                    subject_name = subject
                    object_name = obj
                    relation_name = relation
                    subject_aliases: list[str] = []
                    try:
                        meta = ast.literal_eval(row.get("subject_json") or "{}")
                        if isinstance(meta, dict):
                            aliases = meta.get("Aliases", "")
                            subject_aliases = [a.strip() for a in str(aliases).split("||") if a.strip()]
                    except (ValueError, SyntaxError):
                        subject_aliases = []
                else:
                    subject_name = entity_names.get(subject, subject)
                    object_name = entity_names.get(obj, obj)
                    relation_name = relation_names.get(relation, relation)
                    subject_aliases = alias_map.get(subject, []) if alias_map else []

                interval = f"{start or '?'}..{end or '?'}"
                fact_text = f"{subject_name} | {relation_name} | {object_name} | {interval}"
                time_type = "point" if (start is not None and start == end) else "interval"
                raw_time = raw_start if raw_start == raw_end else f"{raw_start}..{raw_end}"
                fact = {
                    "fact_id": fact_id,
                    "source_dataset": config["dataset"],
                    "source_record_id": str(row.get("id") or f"{config['facts_member']}:{line_number}"),
                    "subject_id": subject,
                    "relation_id": relation,
                    "object_id": obj,
                    "time": {
                        "time_type": time_type,
                        "time_start": start,
                        "time_end": end,
                        "time_granularity": granularity,
                        "raw_time": raw_time,
                    },
                    "time_start": start,
                    "time_end": end,
                    "raw_time_start": raw_start,
                    "raw_time_end": raw_end,
                    "subject_name": subject_name,
                    "relation_name": relation_name,
                    "object_name": object_name,
                    "aliases": subject_aliases,
                    "fact_text": fact_text,
                    "canonical_fact_text": _canonical_text(fact_text),
                    "provenance": {
                        "source_archive": bundle.path.name,
                        "source_file": config["facts_member"],
                        "source_row": line_number,
                    },
                    "construction": {"rule_version": config["rule_version"]},
                }
                good.write(canonical_json(fact) + "\n")
                counters["accepted_facts"] += 1
                if start is None:
                    counters["unknown_start"] += 1
                if end is None:
                    counters["unknown_end"] += 1
        finally:
            source.close()

    stats = {
        "dataset": config["dataset"],
        "max_records": max_records,
        **dict(counters),
        "quarantined_facts": sum(rejection_reasons.values()),
        "rejection_reasons": dict(rejection_reasons),
        "output": str(facts_path),
    }
    write_json(statistics_path, stats)
    return stats


# --------------------------------------------------------------------------- #
# question readers (stage 02 input)
# --------------------------------------------------------------------------- #

def parse_icews_events(cell: str) -> list[str]:
    """Parse the source 'events' column.

    The source generator serialized a Python list of 's|r|o|ts|te' strings
    where entity names containing commas (or apostrophes) break the naive
    list repr; fragments are re-merged on the '|' arity rule, which was
    verified to resolve 89,372/89,372 questions against the KG."""
    s = cell.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return []
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts = re.split(r"['\"]\s*,\s*['\"]", inner)
    parts = [p.strip() for p in parts]
    parts = [p[1:] if p[:1] in "'\"" else p for p in parts]
    parts = [p[:-1] if p[-1:] in "'\"" else p for p in parts]
    parts = [p.replace("\\'", "'").replace('\\"', '"') for p in parts]
    merged: list[str] = []
    for part in parts:
        if merged and merged[-1].count("|") < 4:
            merged[-1] = merged[-1] + ", " + part
        else:
            merged.append(part)
    return merged


def iter_icews_questions(config: dict[str, Any], bundle: SourceBundle, limit: int | None):
    with bundle.open(config["questions_member"]) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                return
            events = parse_icews_events(row.get("events") or "")
            original_question = row.get("question") or ""
            paraphrased_question = row.get("paraphrased_question") or ""
            # The locked ICEWS-Actor release contains a corrupted paraphrase
            # column: some paraphrases are repeated across thousands of rows
            # whose events and answers are unrelated.  The deterministic
            # source question is generated from the row's event chain and is
            # therefore the only admissible benchmark surface.  Keep the
            # paraphrase for provenance/auditing, never as model input.
            policy = config.get("question_text_policy", "original")
            if policy != "original":
                raise ValueError(
                    "ICEWS-Actor formal builds require question_text_policy=original; "
                    "the source paraphrased_question column is not row-aligned"
                )
            yield {
                "source_question_id": str(row.get("id", index)),
                "source_split": row.get("split"),
                "question": original_question,
                "original_question": original_question,
                "source_paraphrased_question": paraphrased_question,
                "question_text_source": "original_question",
                "answers": [row.get("answer", "")],
                "answer_type": row.get("answer_type"),
                "question_type": row.get("question_type"),
                "temporal_operator": row.get("temporal_relation"),
                "template": row.get("question_type"),
                "entities": [],
                "relations": [],
                "times": [],
                "annotation": {},
                "source_events": events,
                "source_kg_id": str(row.get("source_kg_id", "")),
                "question_level": row.get("question_level"),
                "_source_file": config["questions_member"],
                "_source_row": index,
            }


def iter_cron_questions(config: dict[str, Any], bundle: SourceBundle, limit: int | None):
    emitted = 0
    for source_split, member in config["questions_members"].items():
        with bundle.open(member) as raw:
            rows = RestrictedUnpickler(raw).load()
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in {member}, got {type(rows).__name__}")
        for row in rows:
            if limit is not None and emitted >= limit:
                return
            emitted += 1
            # Sets are sorted to make repeated builds byte-for-byte reproducible.
            yield {
                "source_question_id": str(row.get("uniq_id", emitted)),
                "source_split": source_split,
                "question": (row.get("paraphrases") or [row.get("question", "")])[0],
                "original_question": row.get("question", ""),
                "answers": sorted(str(x) for x in row.get("answers", set())),
                "answer_type": row.get("answer_type"),
                "question_type": row.get("type"),
                "temporal_operator": (row.get("annotation") or {}).get("type") or (row.get("annotation") or {}).get("adj"),
                "template": row.get("template"),
                "entities": sorted(str(x) for x in row.get("entities", set())),
                "relations": sorted(str(x) for x in row.get("relations", set())),
                "times": sorted(str(x) for x in row.get("times", set())),
                "annotation": row.get("annotation") or {},
                "source_events": [],
                "paraphrases": list(row.get("paraphrases") or []),
                "_source_file": member,
                "_source_row": emitted,
            }
