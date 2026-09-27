#!/usr/bin/env python3
"""Check that the Verus proof still describes the code it claims to describe.

## Why this exists

`verus/kv_cache_bounds.rs` is a Verus *model* of the indexing arithmetic, not a
verification of the crate: `src/ops.rs` and `src/model.rs` contain no `verus!`
macro and are never compiled by Verus. That gap is stated as the first caveat
in `verus/README.md`, and it has a sharp edge — if someone edits an index
expression, the proof keeps verifying, faithfully, about code that no longer
exists. A green Verus run would say nothing.

This script closes most of that edge. The proof and its README cite the Rust
source by line number, and this checks that each cited line still contains the
construct the citation is about. So the common failure mode — someone changes
`h / kv_mul` to something else and the line numbers shift — becomes a red build
instead of a stale claim.

## What it does and does not check

It checks **that the cited lines still contain what they are cited for**. It
cannot check that the proof's reasoning is *correct*, that it covers every index
expression, or that the transcription is faithful in any deeper sense. A
sufficiently determined edit that preserves the anchor substring would still
slip through. It is a tripwire on the most likely accident, not a proof of
correspondence.

## Why the check is bidirectional

* Every citation found in the prose must have an entry in `EXPECTED` below. A
  new citation with no expectation is a hard failure, so the table cannot
  silently stop covering part of the file.
* Every entry in `EXPECTED` must actually be cited. A leftover entry is also a
  failure, so the table cannot rot into describing code nothing refers to.

Stdlib only, so the CI job that runs it needs no extra packages.

Usage:
    python3 verus/check_citations.py
Exit codes: 0 all citations resolve, 1 at least one does not, 2 bad usage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VERUS = REPO / "verus"
PROSE = ("kv_cache_bounds.rs", "README.md")

# (source file, first line, last line) -> a substring that must appear somewhere
# in that range of `src/<file>`.
#
# The substrings are identifiers and expressions, never whitespace or
# punctuation layout, so reformatting the file does not trip the check while
# changing what the expression *computes* does.
EXPECTED: dict[tuple[str, int, int], str] = {
    # --- src/ops.rs ---------------------------------------------------------
    ("ops.rs", 315, 345): "pub fn try_new(dim: usize",
    ("ops.rs", 321, 323): 'return Err("dim must be positive"',
    ("ops.rs", 324, 326): "must be divisible by n_heads",
    ("ops.rs", 334, 334): "let head_size = dim / n_heads;",
    ("ops.rs", 350, 352): "self.head_size * self.n_kv_heads",
    ("ops.rs", 356, 358): "self.n_heads / self.n_kv_heads",
    ("ops.rs", 362, 364): "self.head_size * self.n_heads",
    ("ops.rs", 397, 454): "pub fn attention(",
    ("ops.rs", 412, 425): "debug_assert_eq!(out.len(), dims.q_dim())",
    ("ops.rs", 422, 422): "let limit = pos + 1;",
    ("ops.rs", 422, 425): "let limit = pos + 1;",
    ("ops.rs", 423, 425): "key_cache.len() >= limit * kv_dim",
    ("ops.rs", 428, 428): "let kv = h / kv_mul;",
    ("ops.rs", 429, 429): "&q[h * head_size..(h + 1) * head_size]",
    ("ops.rs", 430, 430): "&mut scores[..limit]",
    ("ops.rs", 432, 436): "let s = ",
    ("ops.rs", 433, 433): "&key_cache[t * kv_dim + kv * head_size..",
    ("ops.rs", 440, 440): "&mut out[h * head_size..(h + 1) * head_size]",
    ("ops.rs", 443, 444): "&value_cache[t * kv_dim + kv * head_size..",
    # --- src/model.rs -------------------------------------------------------
    ("model.rs", 123, 144): "fn validate(&self) -> Result<AttentionDims, LoadError>",
    ("model.rs", 249, 253): "fn checked_tensor_len(",
    ("model.rs", 603, 607): "checked_mul(config.seq_len)",
    ("model.rs", 614, 614): "q: vec![0.0; q_dim]",
    ("model.rs", 619, 619): "scores: vec![0.0; config.seq_len]",
    ("model.rs", 620, 621): "key_cache: vec![0.0; cache]",
    ("model.rs", 658, 662): "positions must be filled in order",
    ("model.rs", 671, 676): "if pos >= cfg.seq_len {",
    ("model.rs", 682, 682): "let kv_layer_stride = cfg.seq_len * kv_dim;",
    ("model.rs", 711, 713): "let base = l * kv_layer_stride + pos * kv_dim;",
    ("model.rs", 711, 724): "let base = l * kv_layer_stride + pos * kv_dim;",
    ("model.rs", 718, 719): "let layer_start = l * kv_layer_stride;",
}

# `ops.rs:433` and `ops.rs:443-444` are cited; the ranges below are also cited.
CITE = re.compile(r"\b(ops|model)\.rs:(\d+)(?:-(\d+))?")


def find_citations() -> set[tuple[str, int, int]]:
    found: set[tuple[str, int, int]] = set()
    for name in PROSE:
        text = (VERUS / name).read_text()
        for m in CITE.finditer(text):
            first = int(m.group(2))
            found.add((m.group(1) + ".rs", first, int(m.group(3) or m.group(2))))
    return found


def main() -> int:
    sources: dict[str, list[str]] = {}
    for name in ("ops.rs", "model.rs"):
        path = REPO / "src" / name
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        sources[name] = path.read_text().splitlines()

    cited = find_citations()
    if not cited:
        print("error: no citations found; the regex or the files changed", file=sys.stderr)
        return 2

    failures: list[str] = []

    # 1. Every citation must resolve to the construct it is cited for.
    for key in sorted(cited):
        name, first, last = key
        if key not in EXPECTED:
            failures.append(
                f"{name}:{first}" + (f"-{last}" if last != first else "")
                + " is cited but has no entry in EXPECTED; add one so the check "
                "keeps covering it"
            )
            continue
        anchor = EXPECTED[key]
        lines = sources[name]
        if first < 1 or last > len(lines):
            failures.append(
                f"{name}:{first}-{last} is outside the file (1..{len(lines)}); "
                "the source shrank or the citation is stale"
            )
            continue
        body = "\n".join(lines[first - 1 : last])
        if anchor not in body:
            failures.append(
                f"{name}:{first}-{last} no longer contains {anchor!r}.\n"
                f"      the Verus proof describes the old code here. Re-derive the "
                f"proof against the current source, or restore the code.\n"
                f"      was: {lines[first - 1].strip()[:100]!r}"
            )

    # 2. Every expectation must actually be cited, or the table has rotted.
    for key in sorted(EXPECTED):
        if key not in cited:
            name, first, last = key
            failures.append(
                f"{name}:{first}-{last} has an EXPECTED entry but is not cited "
                "anywhere; the table has drifted from the prose"
            )

    print(f"checked {len(cited)} citations against src/")
    if failures:
        print()
        for f in failures:
            print(f"  {f}")
        print(f"\nFAILED: {len(failures)} citation problem(s)", file=sys.stderr)
        return 1

    print("all citations still resolve to the constructs they are cited for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
