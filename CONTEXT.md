# CONTEXT.md

Design notes for `proofinfer`. This is the working document I keep open while
building, and the thing I would hand to a reviewer who asks "why is it like
this?".

## 1. What this is

A Llama-architecture inference engine in Rust with **zero external crates**,
plus the machinery that proves it computes the right thing.

The engine is the easy half. The interesting half is the *testing story*:
a differential harness that compares this engine's per-position logits against
an independent reference implementation, and a mutation checker that proves the
harness would have caught real bugs.

## 2. Non-negotiable rules for this repo

1. **Zero external dependencies.** `Cargo.lock` contains exactly one package.
   Arg parsing, file IO, f32 math, the tokenizer, the CLI: all `std` only.
2. **No fake green.** No `|| true`, no `continue-on-error`, no skipped
   assertions, no tolerances widened until a failure disappears. If a check
   fails, the bug is in my code, not in the check.
3. **Do not tune numbers to match the reference.** A mismatch means I have a
   real bug. Find it.
4. **Every non-obvious decision gets a comment explaining the reasoning**, not
   restating the code.

## 3. Checkpoint format (llama2.c legacy "v0")

Little-endian throughout. Header is 7 x i32:

```
dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
```

A **negative** `vocab_size` means the classifier is not tied to the token
embedding: take the absolute value and expect a trailing `wcls` tensor.

Then f32 tensors, per-layer tensors concatenated across layers:

| tensor           | shape                          |
|------------------|--------------------------------|
| `token_embedding`| `vocab x dim`                  |
| `rms_att`        | `layers x dim`                 |
| `wq`             | `layers x dim x dim`           |
| `wk`             | `layers x kv_dim x dim`        |
| `wv`             | `layers x kv_dim x dim`        |
| `wo`             | `layers x dim x dim`           |
| `rms_ffn`        | `layers x dim`                 |
| `w1`             | `layers x hidden x dim`        |
| `w2`             | `layers x dim x hidden`        |
| `w3`             | `layers x hidden x dim`        |
| `rms_final`      | `dim`                          |
| `freq_cis_real`  | `seq_len x head_size/2`  SKIP  |
| `freq_cis_imag`  | `seq_len x head_size/2`  SKIP  |
| `wcls`           | `vocab x dim` (untied only)    |

`head_size = dim / n_heads`, `kv_dim = head_size * n_kv_heads`.

**The RoPE tables are skipped on purpose.** The file *contains* `freq_cis`, and
I could just read it. Computing RoPE myself in `ops::rope` means the
differential test also exercises my RoPE maths instead of trusting a precomputed
table that both implementations would then share. A shared precomputed table is
a shared blind spot.

### The loader treats the file as hostile input

A checkpoint arrives from disk and its header decides how many bytes we are
about to allocate. Every size product uses `checked_mul`, so an overflow is an
`Err`, never a panic and never a silent wraparound that would hand out a
short `Vec` that later gets indexed out of bounds. Trailing bytes are an error
too, which is what catches a tensor-order mistake.

## 4. The forward pass

```
x = token_embedding[token]
for l in layers:
    xb  = rmsnorm(x, rms_att[l])
    q   = wq[l] @ xb
    k   = wk[l] @ xb
    v   = wv[l] @ xb
    rope(q, pos); rope(k, pos)
    key_cache[l][pos] = k; value_cache[l][pos] = v
    for h in heads:
        kv = h / (n_heads / n_kv_heads)          # grouped-query attention
        score[t] = dot(q_h, key_cache[l][t][kv]) / sqrt(head_size)   for t in 0..=pos
        softmax(score)
        out_h = sum_t score[t] * value_cache[l][t][kv]
    x += wo[l] @ out
    xb = rmsnorm(x, rms_ffn[l])
    x += w2[l] @ (silu(w1[l] @ xb) * (w3[l] @ xb))   # SwiGLU
x  = rmsnorm(x, rms_final)
logits = classifier @ x
```

Kernels:

- `rmsnorm(x, w) = x / sqrt(mean(x^2) + 1e-5) * w`
- `softmax`: subtract the max before `exp` (see Q3 in the interview list)
- `rope`: within each head, rotate pair `(2i, 2i+1)` by `pos * 10000^(-2i/head_size)`
- `silu(x) = x / (1 + exp(-x))`
- `matmul(out, x, W, n, d)` where `W` is `d x n`, row-major

All scratch and both caches are allocated once in `State` and reused. Positions
must be fed `0, 1, 2, ...` in order, because every step appends to the cache.

## 5. Tokenizer

`tokenizer.bin`: `i32 max_token_length`, then per token `f32 score, i32 len,
bytes`. 32000 tokens. Ids 0/1/2 are `<unk>`/BOS/EOS; ids 3..=258 are the byte
fallbacks for `0x00..0xFF`.

`encode` follows llama2.c exactly: BOS, then the SentencePiece dummy-prefix
space, then per UTF-8 codepoint a vocab lookup or byte fallback, then greedily
merge the highest-scoring adjacent pair until nothing merges.

## 6. Differential testing

`reference/diff_test.py` builds six small configs, randomises their weights,
exports each with the reference `legacy_export`, runs the reference model in
PyTorch and this engine in Rust, and requires:

```
|engine - ref| <= 1e-4 + 1e-4 * |ref|      at every position, every logit
argmax agrees                               at every position
```

The two implementations share nothing but the file format. The reference does a
batched forward pass with a causal mask; this engine decodes one token at a
time with a KV cache, computes its own RoPE, and uses index-mapped GQA. That is
the whole reason the test is worth anything: agreeing on logits means the two
structurally different implementations agree on the maths.

### Gotcha: default init hides bugs

llama2.c initialises weights with `std=0.02`, which makes the logits nearly
flat. Several genuinely wrong implementations still land inside `1e-4` of a flat
reference. The harness therefore re-randomises: matrices as
`randn / sqrt(fan_in)`, norm weights as `1 + 0.5 * randn`. The reference must
also be told to return logits for *every* position rather than only the last
(pass `targets=`), and `legacy_export` has to be wrapped in
`contextlib.redirect_stdout` or it pollutes the report.

## 7. Mutation testing

A test that cannot fail is decoration. `reference/mutation_check.py` copies the
crate, injects one realistic bug at a time, rebuilds, and requires the
differential harness to exit non-zero. It asserts each replacement snippet
occurs *exactly once*, so a refactor cannot silently turn a mutant into a no-op
that trivially "passes".

Ten mutants: RoPE exponent `2i`->`i`; RoPE skipped on k; GQA mapping
`h / kv_mul` -> `h % n_kv_heads`; attention window `pos + 1` -> `pos.max(1)`;
attention scale removed; RMSNorm eps `1e-5` -> `1e-27`; SwiGLU gate/up swapped;
untied classifier ignored; residual `x += d` -> `x = d`; RoPE theta 10000 ->
500000.

Required result: **10/10 caught**. If one survives, the suite has a blind spot
and the suite is what gets fixed.

The per-mutant output names *which configurations* caught each bug, and that is
the part worth reading: `gqa-modulo-mapping` is caught by `gqa-2x` and `gqa-4x`
and nothing else, because modulo and division agree when `kv_heads == n_heads`.
`ignore-untied-classifier` is caught by `untied-cls` alone. If every mutant had
been caught by all six, the suite would have had no discrimination and the
number would have hidden that.

### The same discipline applied to the Rust tests

`reference/mutation_check.py` proves the *differential harness* can fail. It says
nothing about whether `cargo test` can. `reference/mutation_check_tests.py`
closes that: 21 realistic bugs injected into `src/`, each required to make
`cargo test` fail. **21/21 caught**, and running it the first time found two
real holes in the suite:

- Every RoPE test was a *structural* invariant, and a structural invariant is
  satisfied by any angle at all. `2 * i` -> `i` therefore passed all four. Fixed
  by a test that computes the expected angle from the documented schedule.
- Nothing covered generation stopping on BOS. The byte-identity check had found
  that bug by hand and the differential harness cannot see it because it never
  looks at text. Fixed with an engineered checkpoint plus its mirror.

A third finding was a test passing for the wrong reason:
`state_rejects_a_config_whose_cache_would_overflow` used `i32::MAX` as `dim`,
which is odd, so `State::new` returned `InvalidHeader` for the odd head size and
never reached the arithmetic the test is named after. "Assert some error" was
not enough; it now asserts the specific `Overflow { what: "kv cache" }`.

Lesson worth keeping: a suite that has only ever been seen passing is
indistinguishable from a suite that cannot fail. Every check in this repository
has now been exercised in a state where it had to go red.

## 8. Performance

Measured on an Apple M3 (rustc 1.98.0, Apple clang 21.0.0), stories15M, 248
generated greedy tokens, 5 runs, median, single threaded. Full table and the
reasoning are in the README; the two findings worth keeping here are:

**The eight-accumulator dot product is necessary but not sufficient.** It is
necessary because f32 addition is not associative, so LLVM may not split a
serial `sum()` into parallel partial sums, and the resulting loop-carried
dependency caps throughput at one add per multiply. It is *not* sufficient
because writing the eight lanes as an indexed inner loop
(`acc[lane] += a[i+lane] * b[i+lane]`) leaves eight independent **scalar**
chains: 2.2 → 3.3 Gelem/s at dim 288, which reads like success and is a quarter
of the available win. Rewriting the identical arithmetic as `chunks_exact(8)`
gets 14.6 Gelem/s, because it hands the SLP pass one contiguous chunk to pack.
154 → 208 → 835 tok/s across the three variants, all measured in one session.
"Use eight accumulators" is advice that gets you most of the way and looks like
it got all of it.

`-C target-cpu=native` adds 0.2% on aarch64 (835 → 837), inside the noise. The
loop *shape*, not the target flag, is what unlocked the vectorisation.

**`run.c -Ofast` is still 1.17x faster** (974 vs 835 tok/s), because
`-ffast-math` lets clang reassociate anywhere, not just in one dot product.
This is reported rather than hidden. Part of that gap is reachable by writing
the loop in a shape the compiler likes; the rest is not, and the cost of not
taking it is a smaller trusted surface for the numerics.

Two bugs were found only by the byte-identical comparison against `run.c`,
neither reachable from the differential test (which never looks at text):

- Generation stopped on **EOS**; `run.c` stops on **BOS**, which is the
  document delimiter these models are trained with. On stories15M our output
  ran past the end of the first story and the texts diverged by 112 bytes.
- Output was decoded and printed in one call. `run.c` prints *per token*
  through `safe_printf`, which drops single non-printable bytes. Bulk decoding
  emits them, so the byte comparison needs `print_tokens` + `safe_print`.

`bench.py` also has to give `run.c` a larger step budget than we use:
`run.c`'s `-n` counts total forward passes from position 0, prompt included,
while ours counts generated tokens only. Comparing equal numbers made one side
stop early, and the outputs then differed by length for a reason that had
nothing to do with the arithmetic.

## 9. Trusted computing base

For the record, since it is the honest answer to "how much do you actually
trust this?": the Rust compiler and `std`, the f32 semantics of the CPU, the
checkpoint parsing code, and the reference implementation being tested against.
Everything above those is tested; those are assumed and stated.

## 10. Interview questions this has to answer without notes

1. Why does a KV cache make decoding O(n) per token rather than O(n^2)?
2. What does GQA save, and why is the head mapping `h / (n_heads / n_kv_heads)`?
3. Why subtract the max in softmax?
4. Why did the 8-accumulator dot product speed things up, and why did it change
   the numbers?
5. Why is `1e-4 + 1e-4 * |ref|` a reasonable tolerance, and what changes for
   fp16?
6. The diff test passed the first run. Why believe it can fail at all? (The
   mutation check.)
7. What would you verify first in Verus or Lean, and what is the property?
   (Index safety of the KV-cache slicing in `forward` for any `pos < seq_len`.)
   This one is no longer hypothetical: `verus/` proves it, 24 conditions, all
   discharged by Verus under `--no-cheating`. What it does *not* do is verify
   the crate. It is a model of the indexing arithmetic whose correspondence to
   `src/ops.rs` and `src/model.rs` is a human transcription, and a refactor
   would leave it verifying a description of code that no longer exists. That
   edge is now partly blunted: `verus/check_citations.py` checks, in CI, that
   every source line the proof cites still holds the construct it is cited for,
   and it is negative-tested against exactly the edits it is meant to catch.
8. What is in the TCB?
