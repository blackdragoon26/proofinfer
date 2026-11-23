# KV-cache index safety in Verus

This directory holds a machine-checked proof of the property named in
CONTEXT.md section 10, question 7:

> KV-cache and attention indexing never goes out of bounds for any
> `pos < seq_len`.

- `kv_cache_bounds.rs` — the Verus development. **24 verification conditions,
  all discharged by Verus.** See "Verification status" below for the exact
  command and its real output.
- `check_citations.py` — bidirectional, symbol-scoped citation verifier tying
  this document and `kv_cache_bounds.rs` to `src/ops.rs` and `src/model.rs`.
- This file — what is proved, where it lands in the Rust source, what is
  *not* proved, and how to re-run the check.

---

## 1. The property, precisely

**The shapes.** `n_layers`, `n_heads`, `n_kv_heads`, `head_size`, `seq_len` are
integers, and `kv_dim = head_size * n_kv_heads` and
`kv_layer_stride = seq_len * kv_dim`. The KV cache is
`n_layers * kv_layer_stride` floats, so layer `l` occupies
`[l * kv_layer_stride, (l + 1) * kv_layer_stride)`.

**The structural preconditions**, transcribed from the checks the loader
actually performs: `n_layers > 0`, `n_heads > 0`, `n_kv_heads > 0`,
`seq_len > 0`, `head_size > 0`, and `n_heads % n_kv_heads == 0`.

**Claim 1 — the shapes follow from the loader.**
`lemma_shapes_from_config` shows that anything passing
`model.rs::Config::validate` and `ops.rs::AttentionDims::try_new` yields a
shape tuple satisfying the preconditions above. The only non-obvious part is
`head_size > 0`, which the loader never checks explicitly: it computes
`head_size = dim / n_heads` only after rejecting `dim == 0` and
`dim % n_heads != 0`, and a divisor of a positive number is positive.

**Claim 2 — the attention read is in bounds** (`lemma_attention_row_in_layer`).
For every `t` with `0 <= t <= pos` and every `kv` with `kv < n_kv_heads`:

```text
t*kv_dim + kv*head_size  <  t*kv_dim + (kv+1)*head_size  <=  (pos+1)*kv_dim  <=  kv_layer_stride
```

That is: the half-open range
`[t*kv_dim + kv*head_size, t*kv_dim + (kv+1)*head_size)` is non-empty and lies
inside the `pos + 1` live rows, hence inside the layer. Supporting facts:
`lemma_limit_rows_fit_in_layer` (the `pos + 1` live rows fit in a layer,
because `pos < seq_len`) and `lemma_gqa_head_in_range` (`kv = h / kv_mul` is in
`0..n_kv_heads` for `h < n_heads`).

**Claim 3 — the per-layer write is in bounds** (`lemma_cache_write_in_cache`).
For every `l < n_layers` and `pos < seq_len`:

```text
l*kv_layer_stride + pos*kv_dim + kv_dim  <=  cache_len
l*kv_layer_stride + pos*kv_dim + kv_dim  <=  l*kv_layer_stride + kv_layer_stride
```

The second is what makes the attention read immediately afterwards observe the
value just written. Supporting fact: `lemma_layer_slice_in_cache`, the
per-layer slice `[l*stride, l*stride + stride)` is inside the whole cache.

**Claim 4 — the composition** (`lemma_attention_row_in_cache`) is the same
bound with the layer offset folded in, which is the form the CONTEXT.md
question is phrased in.

**Claim 5 — the remaining index expressions.**
`lemma_query_head_in_q` (`&q[h*head_size..(h+1)*head_size]` and
`&mut out[h*head_size..(h+1)*head_size]` in `ops.rs::attention`) and
`lemma_scores_window_in_scratch` (`&mut scores[..limit]` in `ops.rs::attention`).

## 2. Where each claim lands in the Rust source

Citations are symbol-scoped rather than line-numbered and are verified
bidirectionally against both `src/` and `verus/kv_cache_bounds.rs` by
`verus/check_citations.py`.

| Rust symbol | Expression | Verus |
|---|---|---|
| `ops.rs::attention` | `&q[h*head_size..(h+1)*head_size]`, `&mut out[h*head_size..(h+1)*head_size]` | `lemma_query_head_in_q`, `exec_attention_read` |
| `ops.rs::attention` | `&mut scores[..limit]` | `lemma_scores_window_in_scratch`, `exec_attention_read` |
| `ops.rs::attention` | `&key_cache[t*kv_dim + kv*head_size .. t*kv_dim + (kv+1)*head_size]` | `lemma_attention_row_in_layer`, `exec_attention_read` |
| `ops.rs::attention` | `&value_cache[t*kv_dim + kv*head_size .. t*kv_dim + (kv+1)*head_size]` | `lemma_attention_row_in_layer`, `exec_attention_read` |
| `ops.rs::attention` | `let kv = h / kv_mul` | `lemma_gqa_head_in_range`, `exec_attention_read` |
| `ops.rs::attention` | `let limit = pos + 1` | `lemma_limit_rows_fit_in_layer`, `exec_attention_read` |
| `ops.rs::attention` | `debug_assert!`s on buffer lengths | the `requires` of `exec_attention_read` |
| `ops.rs::AttentionDims::kv_dim` / `ops.rs::AttentionDims::kv_mul` / `ops.rs::AttentionDims::q_dim` | `kv_dim` / `kv_mul` / `q_dim` | `Shapes::kv_dim`, `kv_mul`, `q_dim` |
| `ops.rs::AttentionDims::try_new` | shape validation checks | `Shapes::valid`, `lemma_shapes_from_config` |
| `model.rs::Config::validate` | non-zero config checks + `attention_dims()` | `Shapes::valid`, `lemma_shapes_from_config` |
| `model.rs::State::forward` | `let kv_layer_stride = cfg.seq_len * kv_dim` | `Shapes::kv_layer_stride`, `exec_forward_cache_write_and_layer_slice` |
| `model.rs::State::forward` | `let base = l*stride + pos*kv_dim; key_cache/value_cache[base..base+kv_dim]` | `lemma_cache_write_in_cache`, `exec_forward_cache_write_and_layer_slice` |
| `model.rs::State::forward` | `layer_start`/`layer_end`, `&self.key_cache[layer_start..layer_end]` / `&self.value_cache[layer_start..layer_end]` | `lemma_layer_slice_in_cache`, `exec_forward_cache_write_and_layer_slice` |
| `model.rs::State::forward` | `if pos >= cfg.seq_len { return Err(..) }` | the `pos < seq_len` hypothesis everywhere |
| `model.rs::State::new` | `checked_mul` cache size and `try_zeroed_f32_vec` allocations | the overflow preconditions of the `exec fn`s |

Section 7 of `kv_cache_bounds.rs` goes one step further: `exec_attention_read`
and `exec_forward_cache_write_and_layer_slice` are `exec fn`s containing the
index expressions **copied verbatim** from the table above, operating on real
`Vec<f32>`. Verus discharges the genuine slice-indexing obligations there,
including its implicit no-overflow check on the index arithmetic. So the
arithmetic above is not just asserted about strings of symbols: the same
expressions are accepted by the verifier as real Rust slicing.

## 3. What is inside the lemma, and what is not

**Inside.**

- The bounds of every KV-cache index expression, for all shapes satisfying the
  structural preconditions and all `pos < seq_len`.
- That those structural preconditions follow from the conditions the loader
  checks (transcribed by hand — see below).
- That the GQA head mapping `h / kv_mul` lands inside `0..n_kv_heads`.
- That the index arithmetic cannot overflow `usize`, both via Verus's own
  no-overflow check on the real slices and via the explicit
  `lemma_index_fits_in_usize`.

**Outside. None of the following is proved, and nothing here should be read as
claiming otherwise.**

1. **This is not a verification of the crate.** `src/ops.rs` and
   `src/model.rs` contain no `verus!` macro and are not compiled by Verus. The
   development is a *separate model file*. To verify the crate itself you would
   need `vargo` and to annotate the real functions; that was not attempted.
2. **The correspondence between the model and the code is manual, bounded by
   a syntactic citation check.** The `Shapes` preconditions and the `exec fn`
   bodies were transcribed by a human. If someone changed the control flow or
   context around an expression without changing the cited text, **this file
   would keep verifying.** This is the single most important caveat: the proof
   is a proof *about a transcription*.

   `verus/check_citations.py` closes the syntactic drift mode on *both* sides:
   it strips `//` comments and string literals, resolves each symbol cited in
   this file and in `kv_cache_bounds.rs`, asserts every required normalized
   expression (including full lower and upper slice bounds and `checked_mul`
   chains) appears in `src/`, asserts the paired `exec_*` body in
   `verus/kv_cache_bounds.rs` contains the matching normalized expressions, and
   runs `--self-test` in CI to prove mutations on either side fail. What that
   check does *not* do is verify semantic equivalence across un-cited code: an
   edit that preserves a cited expression's text while altering how variables
   are initialized elsewhere, or a gap in which expressions were transcribed at
   all, would still pass.
3. **The loader's `checked_mul` is assumed, not proved.** The overflow
   preconditions in the `exec fn`s (`head_size * n_kv_heads <= usize::MAX`,
   `(n_layers * seq_len) * (head_size * n_kv_heads) <= usize::MAX`,
   `model.rs::State::new`) are taken as `requires`. Nothing here shows that
   `model.rs::checked_tensor_len` or `model.rs::State::new` really do return
   `Err` rather than wrap.
4. **The `pos < seq_len` guard is assumed, not proved.**
   `model.rs::State::forward`'s guard is read as a precondition of `forward`,
   and the proof is conditional on it.
5. **The `debug_assert!`s in `ops.rs::attention` are compiled out in release.**
   They are *not* what makes the code safe. The safety argument is the
   arithmetic in `lemma_limit_rows_fit_in_layer` plus
   `lemma_layer_slice_in_cache`: a layer slice is `seq_len * kv_dim` long, and
   `pos < seq_len` makes `pos + 1` rows fit. That argument is real, but it is
   carried out against the model, so the connection to the real call site is
   one of the manual correspondences in point 2.
6. **No float semantics whatsoever.** `dot`, `rmsnorm`, `rope`, `softmax`,
   `silu`, `matmul` and the eight-accumulator reduction are not modelled, not
   reasoned about, and not verified. `attention`'s arithmetic is treated as
   irrelevant to index safety, which it is, but that is a modelling decision,
   not a proof.
7. **The composition across the `forward` -> `attention` boundary is manual.**
   `exec_attention_read` and `exec_forward_cache_write_and_layer_slice` are two
   separate functions with separate `requires`. Nothing machine-checks that the
   slice `forward` passes satisfies `attention`'s precondition; that is
   `lemma_attention_row_in_layer` at the model level plus the manual
   correspondence.
8. **`out` and `scores` are modelled as `&Vec<f32>`, not `&mut Vec<f32>`.** The
   real signature is `&mut`. Writing to them cannot affect whether an index is
   in bounds, and Verus correctly refuses to carry `v.len()` across a loop
   iteration for a `&mut`, but the mutability is genuinely dropped rather than
   proved irrelevant.
9. **Bounds safety does not catch wrong-but-in-bounds bugs.** Mutant #3
   (`h / kv_mul` -> `h % n_kv_heads`) is also in range and would sail through
   this proof. The differential test is what catches that one. Stating
   `lemma_gqa_head_in_range` is about totality of the mapping, not about it
   being the *right* mapping.
10. **Order of positions is not modelled.** That positions must be fed
    `0, 1, 2, ...` in order is a caller obligation
    (`model.rs::State::forward@doc`), not a checked precondition. A hole left
    by feeding position 7 before position 3 reads as zeros and is perfectly in
    bounds.
11. **Bounds safety is not memory safety.** Nothing here covers use-after-free,
    aliasing, or anything the Rust compiler does not already cover.

## 4. Reproducing the check

Verus is not a dependency of this crate and is not vendored. It is obtained as
a prebuilt release; `cargo install vargo` is **not** needed.

### Linux x86_64 (matching CI)

```sh
# 1. Prebuilt Verus for Linux/x86_64 (~485 MB).
curl -fSL -o verus-x86-linux.zip \
  'https://github.com/verus-lang/verus/releases/download/release%2F0.2026.09.20.aef82ed/verus-0.2026.09.20.aef82ed-x86-linux.zip'
echo '7b870fa12bc589015c2fab60a8b3d9f07c7b1adb3444eb0fadffcbf7f0447b33  verus-x86-linux.zip' \
  | sha256sum --check --strict -
unzip -q verus-x86-linux.zip -d verus-dist

# 2. Verus requires a matching rustc (1.98.1 for this release).
#    Using an isolated RUSTUP_HOME keeps your host toolchain untouched:
RUSTUP_HOME="$PWD/.verus-rustup" rustup toolchain install 1.98.1-x86_64-unknown-linux-gnu

# 3. Check citations (including self-test) and run Verus:
python3 verus/check_citations.py --self-test
python3 verus/check_citations.py
RUSTUP_HOME="$PWD/.verus-rustup" ./verus-dist/verus-x86-linux/verus \
  --crate-type lib --no-cheating verus/kv_cache_bounds.rs
```

### macOS arm64

```sh
# 1. Prebuilt Verus for macOS/arm64 (449 MB).
curl -fSL -o verus-arm64-macos.zip \
  'https://github.com/verus-lang/verus/releases/download/release%2F0.2026.09.20.aef82ed/verus-0.2026.09.20.aef82ed-arm64-macos.zip'
echo '3f89fd250d1e9792ed6d0c7c3ad72c03c02fdbca3f0638987af6e153d69377bc  verus-arm64-macos.zip' \
  | shasum -a 256 --check -
unzip -q verus-arm64-macos.zip -d verus-dist

# 2. Install the matching rustc into an isolated RUSTUP_HOME:
RUSTUP_HOME="$PWD/.verus-rustup" rustup toolchain install 1.98.1-aarch64-apple-darwin

# 3. Check citations and run Verus:
python3 verus/check_citations.py --self-test
python3 verus/check_citations.py
RUSTUP_HOME="$PWD/.verus-rustup" ./verus-dist/verus-arm64-macos/verus \
  --crate-type lib --no-cheating verus/kv_cache_bounds.rs
```

`--crate-type lib` is required because the file has no `main`. `--no-cheating`
rejects `assume`, `admit` and `external_body`; the file uses none, and this
flag makes that machine-checked rather than a promise in a comment.

## 5. Verification status

**Machine-checked and currently passing.** This is a real run, not a
transcription of one.

```
$ verus --version
Verus
  Version: 0.2026.09.20.aef82ed
  Profile: release
  Platform: linux_x86_64
  Toolchain: 1.98.1-x86_64-unknown-linux-gnu

$ verus --crate-type lib --no-cheating verus/kv_cache_bounds.rs
verification results:: 24 verified, 0 errors
$ echo $?
0
```

- Verus `0.2026.09.20.aef82ed`, prebuilt x86-linux release, sha256
  `7b870fa12bc589015c2fab60a8b3d9f07c7b1adb3444eb0fadffcbf7f0447b33` (and
  arm64-macOS sha256
  `3f89fd250d1e9792ed6d0c7c3ad72c03c02fdbca3f0638987af6e153d69377bc`).
- rustc `1.98.1`, installed into an isolated `RUSTUP_HOME`. No `vargo`, no
  `cargo install`, nothing added to `Cargo.toml` or `Cargo.lock`.
- `verus/check_citations.py --self-test` and `verus/check_citations.py` both
  pass, verifying symbol-scoped expressions across both `src/` and
  `verus/kv_cache_bounds.rs`.
- `grep` finds no `assume`, `admit`, or `external_body` in the file, and
  `--no-cheating` passes, which is the stronger of the two claims.

**What "verified" means here, and what it does not.** 24 verification
conditions passed, covering the five claims in section 1. That is a proof about
the model in `kv_cache_bounds.rs` and about the real slice expressions
transcribed into its section 7. It is not a proof about the compiled
`src/ops.rs` or `src/model.rs`, and section 3 lists what that excludes. The
honest one-line summary: **the indexing arithmetic is machine-checked by Verus,
and `verus/check_citations.py` checks that every cited `src/` and `exec_*`
expression still matches textually; semantic equivalence across un-cited code
remains a manual correspondence.**
