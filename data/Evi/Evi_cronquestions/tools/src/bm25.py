"""Dependency-free tokenization helper used by the frozen evaluator."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return deterministic, case-folded word tokens."""
    return _TOKEN.findall(text.casefold())
