#!/usr/bin/env python3
"""Refresh the immutable-file SHA-256 manifest after audit generation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.release_root.resolve()
    manifest = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(item for item in root.rglob("*")
                       if item.is_file() and item != manifest
                       and "construction_builder" not in item.relative_to(root).parts
                       and "__pycache__" not in item.relative_to(root).parts):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest_files={len(lines)} path={manifest}")


if __name__ == "__main__":
    main()
