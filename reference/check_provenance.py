#!/usr/bin/env python3
"""Verify every vendored file in reference/llama2c/ against PROVENANCE.md.

Checks bidirectionally:
1. Every row in PROVENANCE.md's table must exist on disk with the exact byte
   size and exact 64-hex sha256 digest.
2. No unlisted files or unexpected subdirectories may exist in the directory
   (ignoring only `PROVENANCE.md` itself and Python's `__pycache__/`).

Stdlib only.
Usage:
    python3 reference/check_provenance.py
    python3 reference/check_provenance.py --dir /path/to/scratch/llama2c
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO / "reference" / "llama2c"

ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|$"
)


def parse_table(provenance: Path) -> dict[str, tuple[int, str]]:
    entries: dict[str, tuple[int, str]] = {}
    for line in provenance.read_text().splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            name, size_str, digest = m.group(1), m.group(2), m.group(3)
            if name in entries:
                raise ValueError(f"duplicate entry in PROVENANCE.md: {name}")
            entries[name] = (int(size_str), digest)
    if not entries:
        raise ValueError(f"no 64-hex sha256 rows parsed from {provenance}")
    return entries


def check_dir(target_dir: Path) -> list[str]:
    provenance = target_dir / "PROVENANCE.md"
    if not provenance.exists():
        return [f"{provenance} does not exist"]

    try:
        expected = parse_table(provenance)
    except ValueError as e:
        return [str(e)]

    failures: list[str] = []

    # 1. Every listed file must exist with exact size and sha256.
    for name, (exp_size, exp_sha) in sorted(expected.items()):
        path = target_dir / name
        if not path.is_file():
            failures.append(f"missing vendored file: {name}")
            continue
        data = path.read_bytes()
        if len(data) != exp_size:
            failures.append(f"{name}: byte size {len(data)} != expected {exp_size}")
        got_sha = hashlib.sha256(data).hexdigest()
        if got_sha != exp_sha:
            failures.append(f"{name}: sha256 {got_sha} != expected {exp_sha}")

    # 2. No unlisted files or unexpected subdirectories (ignore PROVENANCE.md and __pycache__).
    for child in sorted(target_dir.iterdir()):
        if child.name == "PROVENANCE.md":
            continue
        if child.is_dir() and child.name == "__pycache__":
            continue
        if child.is_dir():
            failures.append(f"unexpected subdirectory in vendored dir: {child.name}/")
        elif child.name not in expected:
            failures.append(f"unlisted file in vendored dir: {child.name}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="directory containing PROVENANCE.md and vendored files",
    )
    args = parser.parse_args()

    target = args.dir.resolve()
    failures = check_dir(target)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}", file=sys.stderr)
        print(f"FAILED: {len(failures)} provenance problem(s)", file=sys.stderr)
        return 1

    count = len(parse_table(target / "PROVENANCE.md"))
    print(f"verified {count} vendored files in {target} (exact size + 64-hex sha256, no unlisted files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
