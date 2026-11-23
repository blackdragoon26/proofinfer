#!/usr/bin/env python3
"""Bidirectional cross-file verification of proofinfer's Verus proof citations.

Checks three invariants so `verus/kv_cache_bounds.rs` and `verus/README.md`
cannot silently drift from `src/ops.rs` and `src/model.rs`:

1. Symbol-scoped `src/` verification: every symbol cited in `verus/` prose
   (syntax: `ops.rs::Symbol` or `model.rs::Symbol`) must resolve to that Rust
   item's body (or `@doc` comment block) and contain every required normalized
   expression (including full lower+upper slice bounds and `checked_mul` chains)
   after stripping `//` comments and string literals.
2. Cross-file `verus/` verification: for every entry with paired Verus `exec_*`
   expressions, the verified `exec_*` body in `verus/kv_cache_bounds.rs` must
   also contain those exact normalized expressions — preventing silent weakening
   on the Verus side.
3. Bidirectional prose<->registry coverage: every symbol cited in `verus/` prose
   must exist in `REGISTRY`, and every `REGISTRY` key must be cited in prose.

Stdlib only.
Usage:
    python3 verus/check_citations.py
    python3 verus/check_citations.py --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VERUS = REPO / "verus"
PROSE = ("kv_cache_bounds.rs", "README.md")

CITE_RE = re.compile(r"\b((?:ops|model)\.rs::[A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)*(?:@doc)?)")


@dataclasses.dataclass(frozen=True)
class Check:
    src_file: str
    symbol: str  # e.g. "AttentionDims::try_new", "attention", "State::forward@doc"
    src_exprs: tuple[str, ...]
    verus_fn: str | None = None
    verus_exprs: tuple[str, ...] = ()


REGISTRY: dict[str, Check] = {
    # --- src/ops.rs ---------------------------------------------------------
    "ops.rs::AttentionDims::try_new": Check(
        src_file="ops.rs",
        symbol="AttentionDims::try_new",
        src_exprs=(
            "if n_heads == 0 || n_kv_heads == 0",
            "if dim == 0",
            "if dim % n_heads != 0",
            "if n_heads % n_kv_heads != 0",
            "let head_size = dim / n_heads;",
            "if head_size % 2 != 0",
        ),
    ),
    "ops.rs::AttentionDims::kv_dim": Check(
        src_file="ops.rs",
        symbol="AttentionDims::kv_dim",
        src_exprs=("self.head_size * self.n_kv_heads",),
        verus_fn="exec_attention_read",
        verus_exprs=("let kv_dim = head_size * n_kv_heads;",),
    ),
    "ops.rs::AttentionDims::kv_mul": Check(
        src_file="ops.rs",
        symbol="AttentionDims::kv_mul",
        src_exprs=("self.n_heads / self.n_kv_heads",),
        verus_fn="exec_attention_read",
        verus_exprs=("let kv_mul = n_heads / n_kv_heads;",),
    ),
    "ops.rs::AttentionDims::q_dim": Check(
        src_file="ops.rs",
        symbol="AttentionDims::q_dim",
        src_exprs=("self.head_size * self.n_heads",),
    ),
    "ops.rs::attention": Check(
        src_file="ops.rs",
        symbol="attention",
        src_exprs=(
            "debug_assert_eq!(out.len(), dims.q_dim());",
            "debug_assert_eq!(q.len(), dims.q_dim());",
            "let limit = pos + 1;",
            "debug_assert!(scores.len() >= limit);",
            "debug_assert!(key_cache.len() >= limit * kv_dim);",
            "debug_assert!(value_cache.len() >= limit * kv_dim);",
            "let kv = h / kv_mul;",
            "&q[h * head_size..(h + 1) * head_size]",
            "&mut scores[..limit]",
            "&key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
            "&mut out[h * head_size..(h + 1) * head_size]",
            "&value_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
        ),
        verus_fn="exec_attention_read",
        verus_exprs=(
            "let kv_dim = head_size * n_kv_heads;",
            "let kv_mul = n_heads / n_kv_heads;",
            "let limit = pos + 1;",
            "let kv = h / kv_mul;",
            "&q[h * head_size..(h + 1) * head_size]",
            "&out[h * head_size..(h + 1) * head_size]",
            "&scores[..limit]",
            "&key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
            "&value_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
        ),
    ),
    # --- src/model.rs -------------------------------------------------------
    "model.rs::Config::validate": Check(
        src_file="model.rs",
        symbol="Config::validate",
        src_exprs=(
            "if self.dim == 0",
            "if self.hidden_dim == 0",
            "if self.n_layers == 0",
            "if self.vocab_size == 0",
            "if self.seq_len == 0",
            "self.attention_dims()",
        ),
    ),
    "model.rs::checked_tensor_len": Check(
        src_file="model.rs",
        symbol="checked_tensor_len",
        src_exprs=(".try_fold(1usize, |acc, &d| acc.checked_mul(d))",),
    ),
    "model.rs::State::new": Check(
        src_file="model.rs",
        symbol="State::new",
        src_exprs=(
            "let dims = config.validate()?;",
            ".checked_mul(config.seq_len)",
            ".and_then(|n| n.checked_mul(kv_dim))",
            'q: try_zeroed_f32_vec("q", q_dim)?',
            'scores: try_zeroed_f32_vec("scores", config.seq_len)?',
            'key_cache: try_zeroed_f32_vec("kv cache", cache)?',
            'value_cache: try_zeroed_f32_vec("kv cache", cache)?',
        ),
    ),
    "model.rs::State::forward@doc": Check(
        src_file="model.rs",
        symbol="State::forward@doc",
        src_exprs=("positions must be filled in order",),
    ),
    "model.rs::State::forward": Check(
        src_file="model.rs",
        symbol="State::forward",
        src_exprs=(
            "if pos >= cfg.seq_len {",
            "let kv_layer_stride = cfg.seq_len * kv_dim;",
            "let base = l * kv_layer_stride + pos * kv_dim;",
            "self.key_cache[base..base + kv_dim].copy_from_slice(&self.k);",
            "self.value_cache[base..base + kv_dim].copy_from_slice(&self.v);",
            "let layer_start = l * kv_layer_stride;",
            "let layer_end = layer_start + kv_layer_stride;",
            "&self.key_cache[layer_start..layer_end]",
            "&self.value_cache[layer_start..layer_end]",
        ),
        verus_fn="exec_forward_cache_write_and_layer_slice",
        verus_exprs=(
            "let kv_dim = head_size * n_kv_heads;",
            "let kv_layer_stride = seq_len * kv_dim;",
            "let base = l * kv_layer_stride + pos * kv_dim;",
            "&key_cache[base..base + kv_dim]",
            "&value_cache[base..base + kv_dim]",
            "let layer_start = l * kv_layer_stride;",
            "let layer_end = layer_start + kv_layer_stride;",
            "&key_cache[layer_start..layer_end]",
            "&value_cache[layer_start..layer_end]",
        ),
    ),
}


def strip_comments_and_strings(code: str, *, preserve_strings: bool = False) -> str:
    """Strip `//` line comments and (optionally) double-quoted string contents."""
    out: list[str] = []
    i = 0
    n = len(code)
    in_str = False
    while i < n:
        ch = code[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                if preserve_strings:
                    out.append(code[i : i + 2])
                i += 2
                continue
            if ch == '"':
                in_str = False
                out.append('"')
            elif preserve_strings:
                out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            while i < n and code[i] != "\n":
                i += 1
            continue
        if ch == '"':
            in_str = True
            out.append('"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize(text: str) -> str:
    return " ".join(text.split())


def extract_brace_block(text: str, open_brace_idx: int) -> str:
    """Return substring from `open_brace_idx` through its matching `}`."""
    masked = strip_comments_and_strings(text[open_brace_idx:], preserve_strings=False)
    depth = 0
    # Walk raw and masked in lockstep via character scan on raw text.
    i = open_brace_idx
    n = len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx : i + 1]
        i += 1
    _ = masked
    raise ValueError("unbalanced braces while extracting Rust block")


def find_item_scope(source: str, symbol: str) -> str:
    """Locate a Rust fn body (or `@doc` comment block) by `fn` or `Type::fn`."""
    is_doc = symbol.endswith("@doc")
    sym = symbol[:-4] if is_doc else symbol

    region = source
    if "::" in sym:
        type_name, fn_name = sym.split("::", 1)
        impl_re = re.compile(rf"^\s*impl\b[^{{]*\b{re.escape(type_name)}\b[^{{]*\{{", re.M)
        m_impl = impl_re.search(source)
        if not m_impl:
            raise ValueError(f"impl block for {type_name} not found")
        region = extract_brace_block(source, m_impl.end() - 1)
    else:
        fn_name = sym

    fn_re = re.compile(rf"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+{re.escape(fn_name)}\b", re.M)
    m_fn = fn_re.search(region)
    if not m_fn:
        raise ValueError(f"function {symbol} not found")

    if is_doc:
        # Collect contiguous `///` doc-comment lines immediately preceding `fn`.
        lines = region[: m_fn.start()].splitlines()
        doc_lines: list[str] = []
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith("///"):
                doc_lines.append(stripped[3:].strip())
            elif stripped.startswith("#[") or stripped == "":
                continue
            else:
                break
        return "\n".join(reversed(doc_lines))

    # Find opening brace of the function body (handles Verus `requires`/`ensures`
    # clauses before `{` as well as standard Rust `fn` signatures).
    scan_idx = m_fn.end()
    # Skip parameter list `(...)` first so braces inside attributes/generics don't confuse us.
    paren = region.find("(", scan_idx)
    if paren != -1:
        pdepth = 0
        i = paren
        while i < len(region):
            if region[i] == "(":
                pdepth += 1
            elif region[i] == ")":
                pdepth -= 1
                if pdepth == 0:
                    scan_idx = i + 1
                    break
            i += 1
    brace_idx = region.find("{", scan_idx)
    if brace_idx == -1:
        raise ValueError(f"opening brace for {symbol} not found")
    return extract_brace_block(region, brace_idx)


def find_prose_citations(prose_texts: dict[str, str]) -> set[str]:
    found: set[str] = set()
    for text in prose_texts.values():
        for m in CITE_RE.finditer(text):
            found.add(m.group(1))
    return found


def verify(
    sources: dict[str, str],
    verus_source: str,
    prose_texts: dict[str, str],
) -> tuple[int, list[str]]:
    cited = find_prose_citations(prose_texts)
    failures: list[str] = []
    if not cited:
        return 0, ["no citations found in verus/ prose"]

    for key in sorted(cited):
        if key not in REGISTRY:
            failures.append(f"{key} is cited in prose but has no entry in REGISTRY")
            continue
        chk = REGISTRY[key]
        src_text = sources[chk.src_file]
        try:
            raw_scope = find_item_scope(src_text, chk.symbol)
        except ValueError as e:
            failures.append(f"{key}: {e}")
            continue
        if chk.symbol.endswith("@doc"):
            norm_scope = normalize(raw_scope)
        else:
            norm_scope = normalize(strip_comments_and_strings(raw_scope, preserve_strings=True))
        for expr in chk.src_exprs:
            if normalize(expr) not in norm_scope:
                failures.append(
                    f"{key}: src/{chk.src_file}::{chk.symbol} missing expression {expr!r}"
                )

        if chk.verus_fn is not None:
            try:
                v_raw = find_item_scope(verus_source, chk.verus_fn)
            except ValueError as e:
                failures.append(f"{key}: verus/kv_cache_bounds.rs::{chk.verus_fn}: {e}")
                continue
            v_norm = normalize(strip_comments_and_strings(v_raw, preserve_strings=True))
            for expr in chk.verus_exprs:
                if normalize(expr) not in v_norm:
                    failures.append(
                        f"{key}: verus/kv_cache_bounds.rs::{chk.verus_fn} missing expression {expr!r}"
                    )

    for key in sorted(REGISTRY):
        if key not in cited:
            failures.append(f"{key} is in REGISTRY but is not cited anywhere in verus/ prose")

    return len(cited), failures


def run_self_test(
    sources: dict[str, str],
    verus_source: str,
    prose_texts: dict[str, str],
) -> int:
    """Prove the checker rejects mutations on BOTH `src/` and `verus/` sides."""
    _, base_failures = verify(sources, verus_source, prose_texts)
    if base_failures:
        print("self-test aborted: unmutated baseline already fails:", file=sys.stderr)
        for f in base_failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    cases: list[tuple[str, dict[str, str], str]] = [
        (
            "src-upper-bound-mutated",
            {
                **sources,
                "ops.rs": sources["ops.rs"].replace(
                    "t * kv_dim + (kv + 1) * head_size",
                    "t * kv_dim + (kv + 2) * head_size",
                    1,
                ),
            },
            verus_source,
        ),
        (
            "src-checked-mul-kv-dim-mutated",
            {
                **sources,
                "model.rs": sources["model.rs"].replace(
                    ".and_then(|n| n.checked_mul(kv_dim))",
                    ".map(|n| n.wrapping_mul(kv_dim))",
                    1,
                ),
            },
            verus_source,
        ),
        (
            "src-commented-out-anchor",
            {
                **sources,
                "ops.rs": sources["ops.rs"].replace(
                    "let kv = h / kv_mul;",
                    "// let kv = h / kv_mul;\n        let kv = h % dims.n_kv_heads;",
                    1,
                ),
            },
            verus_source,
        ),
        (
            "src-layer-end-mutated",
            {
                **sources,
                "model.rs": sources["model.rs"].replace(
                    "let layer_end = layer_start + kv_layer_stride;",
                    "let layer_end = layer_start + kv_dim;",
                    1,
                ),
            },
            verus_source,
        ),
        (
            "verus-exec-slice-mutated",
            sources,
            verus_source.replace(
                "&key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
                "&key_cache[0..head_size]",
                1,
            ),
        ),
    ]

    failed = 0
    for name, mut_sources, mut_verus in cases:
        if mut_sources == sources and mut_verus == verus_source:
            print(f"  FAIL  self-test {name}: mutation was a no-op", file=sys.stderr)
            failed += 1
            continue
        _, errs = verify(mut_sources, mut_verus, prose_texts)
        if not errs:
            print(f"  FAIL  self-test {name}: mutation slipped past checker", file=sys.stderr)
            failed += 1
        else:
            print(f"  ok    self-test {name} caught ({len(errs)} error(s))")

    if failed:
        print(f"SELF-TEST FAILED: {failed}/{len(cases)} blind spot(s)", file=sys.stderr)
        return 1
    print(f"self-test passed: {len(cases)}/{len(cases)} mutations rejected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run in-memory negative tests proving mutations on src/ and verus/ fail",
    )
    args = parser.parse_args()

    sources: dict[str, str] = {}
    for name in ("ops.rs", "model.rs"):
        path = REPO / "src" / name
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
        sources[name] = path.read_text()

    prose_texts = {name: (VERUS / name).read_text() for name in PROSE}
    verus_source = prose_texts["kv_cache_bounds.rs"]

    if args.self_test:
        return run_self_test(sources, verus_source, prose_texts)

    count, failures = verify(sources, verus_source, prose_texts)
    print(f"checked {count} symbol-scoped bidirectional citations against src/ and verus/")
    if failures:
        print()
        for f in failures:
            print(f"  {f}")
        print(f"\nFAILED: {len(failures)} citation problem(s)", file=sys.stderr)
        return 1

    print("all citations resolve across both src/ and verus/kv_cache_bounds.rs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
