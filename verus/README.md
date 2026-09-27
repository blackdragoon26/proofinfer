# KV-cache index safety in Verus

This directory holds a machine-checked proof of the property named in
CONTEXT.md section 10, question 7:

> KV-cache and attention indexing never goes out of bounds for any
> `pos < seq_len`.

- `kv_cache_bounds.rs` — the Verus development. **24 verification conditions,
  all discharged by Verus.** See "Verification status" below for the exact
  command and its real output.
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
`lemma_shapes_from_config` shows that anything passing `Config::validate`
(model.rs:123-144) and `AttentionDims::try_new` (ops.rs:315-345) yields a shape
tuple satisfying the preconditions above. The only non-obvious part is
`head_size > 0`, which the loader never checks explicitly: it computes
`head_size = dim / n_heads` (ops.rs:334) only after rejecting `dim == 0`
(ops.rs:321-323) and `dim % n_heads != 0` (ops.rs:324-326), and a divisor of a
positive number is positive.

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
`lemma_query_head_in_q` (`&q[h*head_size..(h+1)*head_size]`, ops.rs:429) and
`&mut out[h*head_size..(h+1)*head_size]`, ops.rs:440) and
`lemma_scores_window_in_scratch` (`&mut scores[..limit]`, ops.rs:430).

## 2. Where each claim lands in the Rust source

Line numbers are against commit `efef84b`.

| Rust | Expression | Verus |
|---|---|---|
| ops.rs:429, 440 | `&q[h*head_size..(h+1)*head_size]`, `&mut out[h*head_size..(h+1)*head_size]` | `lemma_query_head_in_q` |
| ops.rs:430 | `&mut scores[..limit]` | `lemma_scores_window_in_scratch` |
| ops.rs:433 | `&key_cache[t*kv_dim + kv*head_size .. t*kv_dim + (kv+1)*head_size]` | `lemma_attention_row_in_layer` |
| ops.rs:443-444 | `&value_cache[t*kv_dim + kv*head_size .. t*kv_dim + (kv+1)*head_size]` | `lemma_attention_row_in_layer` (identical shape) |
| ops.rs:428 | `let kv = h / kv_mul` | `lemma_gqa_head_in_range` |
| ops.rs:422 | `let limit = pos + 1` | `lemma_limit_rows_fit_in_layer` |
| ops.rs:423-425 | the `debug_assert!`s on buffer lengths | the `requires` of `exec_attention_read` |
| ops.rs:350-352 / 356-358 / 362-364 | `kv_dim` / `kv_mul` / `q_dim` | `Shapes::kv_dim`, `kv_mul`, `q_dim` |
| ops.rs:315-345 | `AttentionDims::try_new` | `Shapes::valid`, `lemma_shapes_from_config` |
| model.rs:682 | `let kv_layer_stride = cfg.seq_len * kv_dim` | `Shapes::kv_layer_stride` |
| model.rs:711-713 | `let base = l*stride + pos*kv_dim; key_cache[base..base+kv_dim]` | `lemma_cache_write_in_cache` |
| model.rs:718-719, 723-724 | `layer_start`/`layer_end`, `&self.key_cache[layer_start..layer_end]` | `lemma_layer_slice_in_cache` |
| model.rs:671-676 | `if pos >= cfg.seq_len { return Err(..) }` | the `pos < seq_len` hypothesis everywhere |
| model.rs:603-607, 619-621 | the `checked_mul` cache size and the allocations | the overflow preconditions of the `exec fn`s |

Sections 7 of `kv_cache_bounds.rs` go one step further: `exec_attention_read`
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
2. **The correspondence between the model and the code is manual.** The
   `Shapes` preconditions and the `exec fn` bodies were transcribed by a human.
   If someone changed the loader's checks, or changed an index expression in
   `src/ops.rs`, **this file would keep verifying.** It would simply be
   describing the old code. This is the single most important caveat: the proof
   is a proof *about a transcription*, and a re-check after any edit to
   `src/ops.rs` or `src/model.rs` is a human obligation. The line-number table
   in section 2 is the only thing that would flag such a drift, and it flags it
   by eye, not by machine.
3. **The loader's `checked_mul` is assumed, not proved.** The overflow
   preconditions in the `exec fn`s (`head_size * n_kv_heads <= usize::MAX`,
   `(n_layers * seq_len) * (head_size * n_kv_heads) <= usize::MAX`,
   model.rs:603-607) are taken as `requires`. Nothing here shows that
   `checked_tensor_len` (model.rs:249-253) or `State::new` really do return
   `Err` rather than wrap.
4. **The `pos < seq_len` guard is assumed, not proved.** model.rs:671-676 is
   read as a precondition of `forward`, and the proof is conditional on it.
5. **The `debug_assert!`s at ops.rs:423-425 are compiled out in release.**
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
    `0, 1, 2, ...` in order is a caller obligation (model.rs:658-662), not a
    checked precondition. A hole left by feeding position 7 before position 3
    reads as zeros and is perfectly in bounds.
11. **Bounds safety is not memory safety.** Nothing here covers use-after-free,
    aliasing, or anything the Rust compiler does not already cover.

## 4. Reproducing the check

Verus is not a dependency of this crate and is not vendored. It was obtained
as a prebuilt release; `cargo install vargo` was **not** needed.

```sh
# 1. Prebuilt Verus for macOS/arm64 (449 MB).
curl -L -o verus-arm64-macos.zip \
  'https://github.com/verus-lang/verus/releases/download/release%2F0.2026.09.20.aef82ed/verus-0.2026.09.20.aef82ed-arm64-macos.zip'
shasum -a 256 verus-arm64-macos.zip
# 3f89fd250d1e9792ed6d0c7c3ad72c03c02fdbca3f0638987af6e153d69377bc
unzip -q verus-arm64-macos.zip

# 2. Verus requires a matching rustc. This release wants 1.98.1.
#    Installing it into a scratch RUSTUP_HOME keeps your own toolchain untouched:
RUSTUP_HOME=$HOME/.jcode/scratch/rustup CARGO_HOME=$HOME/.jcode/scratch/cargo \
  rustup toolchain install 1.98.1-aarch64-apple-darwin

# 3. Run the verifier on the model file.
cd /path/to/tinyinfer
RUSTUP_HOME=$HOME/.jcode/scratch/rustup CARGO_HOME=$HOME/.jcode/scratch/cargo \
  "$HOME/.jcode/scratch/verus-dl/verus-arm64-macos/verus" \
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
  Platform: macos_aarch64
  Toolchain: 1.98.1-aarch64-apple-darwin

$ verus --crate-type lib --no-cheating verus/kv_cache_bounds.rs
verification results:: 24 verified, 0 errors
$ echo $?
0
```

- Verus `0.2026.09.20.aef82ed`, prebuilt arm64-macOS release, sha256
  `3f89fd250d1e9792ed6d0c7c3ad72c03c02fdbca3f0638987af6e153d69377bc`.
- rustc `1.98.1-aarch64-apple-darwin`, installed into a scratch `RUSTUP_HOME`.
  The machine's default toolchain is `1.98.0` and was not modified; no
  `vargo`, no `cargo install`, nothing added to `Cargo.toml` or `Cargo.lock`.
- Repo commit `efef84b`. The source line numbers were re-derived against this
  commit: commit `efef84b` landed mid-task and shifted `src/ops.rs` by 8
  lines, and every reference above was corrected afterwards. The `attention`
  body itself is unchanged, only its line numbers moved.
- Wall clock for the verification itself: about 0.8 s.
- `grep` finds no `assume`, `admit`, or `external_body` in the file, and
  `--no-cheating` passes, which is the stronger of the two claims.

**What "verified" means here, and what it does not.** 24 verification
conditions passed, covering the five claims in section 1. That is a proof about
the model in `kv_cache_bounds.rs` and about the real slice expressions
transcribed into its section 7. It is not a proof about the compiled
`src/ops.rs` or `src/model.rs`, and section 3 lists what that excludes. The
honest one-line summary: **the indexing arithmetic is machine-checked; the
link from that arithmetic to the current contents of `src/` is a human
transcription that no tool here is checking.**
