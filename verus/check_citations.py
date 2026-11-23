#!/usr/bin/env python3
"""Bidirectional cross-file verification of proofinfer's Verus proof citations.

Checks three invariants so `verus/kv_cache_bounds.rs` and `verus/README.md`
cannot silently drift from `src/ops.rs` and `src/model.rs`:

1. Symbol-scoped token verification in `src/`: every symbol cited in `verus/`
   prose (`ops.rs::Symbol` or `model.rs::Symbol`) must resolve to that Rust
   item's body (searching only inherent `impl Type` blocks for methods, or the
   `@doc` comment block) and contain every required expression as an exact
   contiguous Rust token sequence — stripping `//` and nested `/* ... */`
   comments, treating string literals as atomic tokens (so code inside a string
   literal cannot satisfy a code guard), ignoring braces inside character
   literals (`'}'`, `'\n'`) without breaking lifetimes (`'a`, `'_`), and
   requiring full token boundaries (so `0x1` cannot match `0`).
2. Cross-file `verus/` verification: for every entry with paired Verus `exec_*`
   expressions, the verified `exec_*` body in `verus/kv_cache_bounds.rs` must
   also contain those exact token sequences — preventing silent weakening on
   the Verus side.
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


@dataclasses.dataclass(frozen=True)
class Token:
    kind: str  # IDENT, NUM, STR, CHAR, OP
    text: str
    start: int
    end: int


MULTI_OPS = (
    "..=",
    "==>",
    "&&&",
    "|||",
    "..",
    "::",
    "==",
    "!=",
    "<=",
    ">=",
    "&&",
    "||",
    "->",
    "=>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
)


def tokenize_rust(code: str) -> list[Token]:
    """Tokenize Rust/Verus source, skipping `//` and nested `/* ... */` comments.

    String literals (`"..."`, `b"..."`, `r#"..."#`) and character literals
    (`'x'`, `'\\n'`, `'}'`, `b'}'`) are emitted as single atomic tokens so:
    - braces inside comments, strings, or character literals never affect scope
      extraction;
    - code text hidden inside a string literal cannot satisfy a code-token
      citation;
    - lifetimes (`'a`, `'_`) are NOT confused with character literals;
    - numeric/identifier tokens match only on full token boundaries (`0x1` != `0`).
    """
    tokens: list[Token] = []
    i = 0
    n = len(code)

    while i < n:
        ch = code[i]

        # Whitespace.
        if ch.isspace():
            i += 1
            continue

        # Line comment `// ...`.
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            i += 2
            while i < n and code[i] != "\n":
                i += 1
            continue

        # Nested block comment `/* ... */`.
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if code[i] == "/" and i + 1 < n and code[i + 1] == "*":
                    depth += 1
                    i += 2
                elif code[i] == "*" and i + 1 < n and code[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue

        # Raw string literal `r"..."`, `r#"..."#`, `br"..."`, `br#"..."#`.
        raw_prefix = 0
        if ch == "r":
            raw_prefix = 1
        elif ch == "b" and i + 1 < n and code[i + 1] == "r":
            raw_prefix = 2
        if raw_prefix:
            j = i + raw_prefix
            hashes = 0
            while j < n and code[j] == "#":
                hashes += 1
                j += 1
            if j < n and code[j] == '"':
                start = i
                j += 1
                closer = '"' + "#" * hashes
                end_idx = code.find(closer, j)
                i = n if end_idx == -1 else end_idx + len(closer)
                tokens.append(Token("STR", code[start:i], start, i))
                continue

        # Regular or byte string literal `"..."`, `b"..."`.
        str_prefix = 1 if ch == '"' else (2 if ch == "b" and i + 1 < n and code[i + 1] == '"' else 0)
        if str_prefix:
            start = i
            i += str_prefix
            while i < n:
                if code[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if code[i] == '"':
                    i += 1
                    break
                i += 1
            tokens.append(Token("STR", code[start:i], start, i))
            continue

        # Character or byte-character literal (`'x'`, `'\n'`, `'\u{...}'`, `b'}'`)
        # distinguished from lifetimes (`'a`, `'_`, `'static`).
        char_prefix = 1 if ch == "'" else (2 if ch == "b" and i + 1 < n and code[i + 1] == "'" else 0)
        if char_prefix:
            q = i + char_prefix - 1
            if q + 1 < n and code[q + 1] == "\\":
                # Escaped char literal: scan to closing quote on the same line.
                j = q + 2
                if j < n:
                    j += 1
                while j < n and code[j] != "'" and code[j] != "\n":
                    j += 1
                if j < n and code[j] == "'":
                    start = i
                    i = j + 1
                    tokens.append(Token("CHAR", code[start:i], start, i))
                    continue
            elif q + 2 < n and code[q + 2] == "'" and code[q + 1] not in ("'", "\n"):
                start = i
                i = q + 3
                tokens.append(Token("CHAR", code[start:i], start, i))
                continue
            # Otherwise it is a lifetime (`'a`, `'_`): emit `'` as punctuation.
            if char_prefix == 1:
                tokens.append(Token("OP", "'", i, i + 1))
                i += 1
                continue

        # Numeric literal (decimal, hex/bin/oct, float, type suffix like `1usize`, `4usize`).
        if ch.isdigit():
            start = i
            i += 1
            while i < n:
                c = code[i]
                if c.isalnum() or c == "_":
                    i += 1
                elif c == "." and i + 1 < n and code[i + 1].isdigit():
                    i += 1
                elif c in ("+", "-") and i > start and code[i - 1] in ("e", "E"):
                    i += 1
                else:
                    break
            tokens.append(Token("NUM", code[start:i], start, i))
            continue

        # Identifier or keyword.
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (code[i].isalnum() or code[i] == "_"):
                i += 1
            tokens.append(Token("IDENT", code[start:i], start, i))
            continue

        # Multi-character operators / punctuation.
        matched_op = None
        for op in MULTI_OPS:
            if code.startswith(op, i):
                matched_op = op
                break
        if matched_op is not None:
            tokens.append(Token("OP", matched_op, i, i + len(matched_op)))
            i += len(matched_op)
            continue

        # Single-character punctuation.
        tokens.append(Token("OP", ch, i, i + 1))
        i += 1

    return tokens


def match_balanced(tokens: list[Token], open_idx: int, open_ch: str, close_ch: str) -> int:
    """Return index of matching `close_ch` token for `tokens[open_idx]`."""
    depth = 0
    for idx in range(open_idx, len(tokens)):
        t = tokens[idx]
        if t.kind == "OP" and t.text == open_ch:
            depth += 1
        elif t.kind == "OP" and t.text == close_ch:
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"unbalanced {open_ch!r}..{close_ch!r} while scanning Rust tokens")


def find_inherent_impl_ranges(tokens: list[Token], type_name: str) -> list[tuple[int, int]]:
    """Return `(open_brace_idx, close_brace_idx)` for all inherent `impl TypeName` blocks.

    Trait impls (`impl Drop for TypeName`) are excluded because `for` appears
    before the receiver type instead of `type_name` immediately following
    `impl [<...>]`.
    """
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].kind == "IDENT" and tokens[i].text == "impl":
            j = i + 1
            # Optional generic params `impl<'a, T>`.
            if j < n and tokens[j].kind == "OP" and tokens[j].text == "<":
                j = match_balanced(tokens, j, "<", ">") + 1
            # Must directly name `type_name` (no trait + `for`).
            if j < n and tokens[j].kind == "IDENT" and tokens[j].text == type_name:
                k = j + 1
                if k < n and tokens[k].kind == "OP" and tokens[k].text == "<":
                    k = match_balanced(tokens, k, "<", ">") + 1
                # Skip optional `where ...` clause up to `{`, rejecting `for`.
                is_trait = False
                while k < n and not (tokens[k].kind == "OP" and tokens[k].text == "{"):
                    if tokens[k].kind == "IDENT" and tokens[k].text == "for":
                        is_trait = True
                        break
                    k += 1
                if not is_trait and k < n and tokens[k].kind == "OP" and tokens[k].text == "{":
                    close_idx = match_balanced(tokens, k, "{", "}")
                    ranges.append((k, close_idx))
                    i = close_idx + 1
                    continue
        i += 1
    return ranges


def find_fn_in_token_range(
    tokens: list[Token],
    start_idx: int,
    end_idx: int,
    fn_name: str,
) -> tuple[int, int, int] | None:
    """Locate `fn fn_name` within `tokens[start_idx:end_idx]`.

    Returns `(fn_token_idx, body_open_idx, body_close_idx)` or `None`.
    """
    i = start_idx
    while i < end_idx:
        if (
            tokens[i].kind == "IDENT"
            and tokens[i].text == "fn"
            and i + 1 < end_idx
            and tokens[i + 1].kind == "IDENT"
            and tokens[i + 1].text == fn_name
        ):
            j = i + 2
            # Optional generics `<'a, T>`.
            if j < end_idx and tokens[j].kind == "OP" and tokens[j].text == "<":
                j = match_balanced(tokens, j, "<", ">") + 1
            # Parameter list `(...)`.
            if j < end_idx and tokens[j].kind == "OP" and tokens[j].text == "(":
                j = match_balanced(tokens, j, "(", ")") + 1
            # Scan forward lexically for the function body `{ ... }`, skipping
            # any `{ ... }` sub-blocks inside Verus `requires`/`ensures` clauses
            # (e.g. `ensures match res { ... },`).
            while j < end_idx:
                if tokens[j].kind == "OP" and tokens[j].text == ";":
                    break
                if tokens[j].kind == "OP" and tokens[j].text == "{":
                    close_idx = match_balanced(tokens, j, "{", "}")
                    nxt = tokens[close_idx + 1] if close_idx + 1 < end_idx else None
                    if nxt is not None and (
                        (nxt.kind == "OP" and nxt.text in (",", "{"))
                        or (nxt.kind == "IDENT" and nxt.text in ("ensures", "requires", "decreases"))
                    ):
                        j = close_idx + 1
                        continue
                    return (i, j, close_idx)
                j += 1
        i += 1
    return None


def extract_doc_before(source: str, byte_pos: int) -> str:
    """Collect contiguous `///` doc-comment lines immediately preceding `byte_pos`."""
    lines = source[:byte_pos].splitlines()
    doc_lines: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("///"):
            doc_lines.append(stripped[3:].strip())
        elif stripped.startswith("#[") or stripped == "" or stripped.startswith("pub"):
            continue
        else:
            break
    return " ".join(" ".join(reversed(doc_lines)).split())


def find_item_tokens(source: str, symbol: str) -> tuple[list[Token], str | None]:
    """Resolve `symbol` in `source`.

    For `@doc` symbols, returns `([], normalized_doc_text)`.
    For code symbols, returns `(body_tokens, None)`.
    """
    is_doc = symbol.endswith("@doc")
    sym = symbol[:-4] if is_doc else symbol
    all_tokens = tokenize_rust(source)

    found: tuple[int, int, int] | None = None
    if "::" in sym:
        type_name, fn_name = sym.split("::", 1)
        impl_ranges = find_inherent_impl_ranges(all_tokens, type_name)
        if not impl_ranges:
            raise ValueError(f"inherent impl block for {type_name} not found")
        for lo, hi in impl_ranges:
            found = find_fn_in_token_range(all_tokens, lo + 1, hi, fn_name)
            if found is not None:
                break
        if found is None:
            raise ValueError(f"associated function {symbol} not found in inherent impl {type_name}")
    else:
        found = find_fn_in_token_range(all_tokens, 0, len(all_tokens), sym)
        if found is None:
            raise ValueError(f"function {symbol} not found")

    fn_idx, open_idx, close_idx = found
    if is_doc:
        # Walk back past visibility modifiers (`pub`, `pub(crate)`) to start of fn line.
        line_start = source.rfind("\n", 0, all_tokens[fn_idx].start)
        anchor_pos = 0 if line_start == -1 else line_start
        return [], extract_doc_before(source, anchor_pos)

    return all_tokens[open_idx : close_idx + 1], None


def find_token_subseq(haystack: list[Token], needle: list[Token]) -> tuple[int, int] | None:
    """Find exact contiguous token-sequence match; return `(start_byte, end_byte)`."""
    m = len(needle)
    if m == 0 or m > len(haystack):
        return None
    needle_key = [(t.kind, t.text) for t in needle]
    for i in range(len(haystack) - m + 1):
        if all(
            haystack[i + k].kind == needle_key[k][0] and haystack[i + k].text == needle_key[k][1]
            for k in range(m)
        ):
            return haystack[i].start, haystack[i + m - 1].end
    return None


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
            body_tokens, doc_text = find_item_tokens(src_text, chk.symbol)
        except ValueError as e:
            failures.append(f"{key}: {e}")
            continue
        if doc_text is not None:
            for expr in chk.src_exprs:
                if " ".join(expr.split()) not in doc_text:
                    failures.append(
                        f"{key}: src/{chk.src_file}::{chk.symbol} missing doc text {expr!r}"
                    )
        else:
            for expr in chk.src_exprs:
                needle = tokenize_rust(expr)
                if find_token_subseq(body_tokens, needle) is None:
                    failures.append(
                        f"{key}: src/{chk.src_file}::{chk.symbol} missing expression {expr!r}"
                    )

        if chk.verus_fn is not None:
            try:
                v_tokens, _ = find_item_tokens(verus_source, chk.verus_fn)
            except ValueError as e:
                failures.append(f"{key}: verus/kv_cache_bounds.rs::{chk.verus_fn}: {e}")
                continue
            for expr in chk.verus_exprs:
                needle = tokenize_rust(expr)
                if find_token_subseq(v_tokens, needle) is None:
                    failures.append(
                        f"{key}: verus/kv_cache_bounds.rs::{chk.verus_fn} missing expression {expr!r}"
                    )

    for key in sorted(REGISTRY):
        if key not in cited:
            failures.append(f"{key} is in REGISTRY but is not cited anywhere in verus/ prose")

    return len(cited), failures


def mutate_token_span(source: str, symbol: str, target_expr: str, replacement: str) -> str:
    """Locate `target_expr` inside `symbol` using the verifier's own tokenizer and replace it."""
    body_tokens, _ = find_item_tokens(source, symbol)
    span = find_token_subseq(body_tokens, tokenize_rust(target_expr))
    if span is None:
        raise RuntimeError(f"self-test setup failed: {target_expr!r} not found in {symbol}")
    start, end = span
    return source[:start] + replacement + source[end:]


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

    cases: list[tuple[str, str, dict[str, str], str]] = [
        (
            "src-upper-bound-mutated",
            "ops.rs::attention",
            {
                **sources,
                "ops.rs": mutate_token_span(
                    sources["ops.rs"],
                    "attention",
                    "t * kv_dim + (kv + 1) * head_size",
                    "t * kv_dim + (kv + 2) * head_size",
                ),
            },
            verus_source,
        ),
        (
            "src-checked-mul-kv-dim-mutated",
            "model.rs::State::new",
            {
                **sources,
                "model.rs": mutate_token_span(
                    sources["model.rs"],
                    "State::new",
                    ".and_then(|n| n.checked_mul(kv_dim))",
                    ".map(|n| n.wrapping_mul(kv_dim))",
                ),
            },
            verus_source,
        ),
        (
            "src-commented-out-anchor",
            "ops.rs::attention",
            {
                **sources,
                "ops.rs": mutate_token_span(
                    sources["ops.rs"],
                    "attention",
                    "let kv = h / kv_mul;",
                    "// let kv = h / kv_mul;\n        let kv = h % dims.n_kv_heads;",
                ),
            },
            verus_source,
        ),
        (
            "src-block-commented-anchor",
            "ops.rs::AttentionDims::try_new",
            {
                **sources,
                "ops.rs": mutate_token_span(
                    sources["ops.rs"],
                    "AttentionDims::try_new",
                    "if head_size % 2 != 0",
                    "/* /* if head_size % 2 != 0 */ */ if false",
                ),
            },
            verus_source,
        ),
        (
            "src-string-literal-spoof",
            "ops.rs::AttentionDims::try_new",
            {
                **sources,
                "ops.rs": mutate_token_span(
                    sources["ops.rs"],
                    "AttentionDims::try_new",
                    "if dim == 0",
                    'let _ = "if dim == 0"; if dim <= 1',
                ),
            },
            verus_source,
        ),
        (
            "src-token-boundary-prefix",
            "ops.rs::AttentionDims::try_new",
            {
                **sources,
                "ops.rs": mutate_token_span(
                    sources["ops.rs"],
                    "AttentionDims::try_new",
                    "if dim == 0",
                    "if dim == 0x1",
                ),
            },
            verus_source,
        ),
        (
            "src-layer-end-mutated",
            "model.rs::State::forward",
            {
                **sources,
                "model.rs": mutate_token_span(
                    sources["model.rs"],
                    "State::forward",
                    "let layer_end = layer_start + kv_layer_stride;",
                    "let layer_end = layer_start + kv_dim;",
                ),
            },
            verus_source,
        ),
        (
            "verus-exec-slice-mutated",
            "ops.rs::attention",
            sources,
            mutate_token_span(
                verus_source,
                "exec_attention_read",
                "&key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size]",
                "&key_cache[0..head_size]",
            ),
        ),
    ]

    failed = 0
    for name, expected_key, mut_sources, mut_verus in cases:
        if mut_sources == sources and mut_verus == verus_source:
            print(f"  FAIL  self-test {name}: mutation was a no-op", file=sys.stderr)
            failed += 1
            continue
        _, errs = verify(mut_sources, mut_verus, prose_texts)
        if not errs:
            print(f"  FAIL  self-test {name}: mutation slipped past checker", file=sys.stderr)
            failed += 1
        elif not any(expected_key in e for e in errs):
            print(
                f"  FAIL  self-test {name}: failed without naming {expected_key}: {errs}",
                file=sys.stderr,
            )
            failed += 1
        else:
            print(f"  ok    self-test {name} caught ({expected_key})")

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
