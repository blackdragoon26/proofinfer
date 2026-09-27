# tinyinfer

A Llama-architecture inference engine in Rust with **zero external crates**,
built around a claim that is usually made and rarely justified: that it
computes the right thing.

The engine is the easy half. The interesting half is the testing story:

- A **differential harness** that compares this engine's per-position logits
  against the PyTorch reference from [karpathy/llama2.c][llama2c], across six
  configurations chosen to cover every structural axis the engine has.
- A **mutation checker** that injects ten realistic bugs into the Rust source
  and requires the harness to fail on every one. A test that has never failed
  is equally consistent with "correct" and with "cannot detect anything"; this
  resolves that ambiguity.
- A **byte-identical** comparison of greedy output against `run.c`, the
  reference C engine, which is a third independent implementation.

[llama2c]: https://github.com/karpathy/llama2.c

## Results

All numbers below were produced by the commands in this file on the machine
described under [Benchmark](#benchmark). Nothing is quoted from elsewhere.

### Differential test, random weights

`python reference/diff_test.py` — pass requires
`|err| <= 1e-4 + 1e-4*|ref|` at *every* logit and argmax agreement at *every*
position.

| config | dim | heads / kv | layers | vocab | seq | classifier | max&nbsp;\|err\| | max&nbsp;\|ref\| | err/tol | top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| `tiny` | 64 | 4 / 4 | 2 | 97 | 32 | tied | 3.87e-06 | 4.22 | 0.030 | 32/32 |
| `gqa-2x` | 64 | 8 / 4 | 3 | 128 | 48 | tied | 4.17e-06 | 4.17 | 0.028 | 48/48 |
| `gqa-4x` | 96 | 8 / 2 | 2 | 200 | 64 | tied | 5.72e-06 | 4.98 | 0.046 | 64/64 |
| `mqa` | 64 | 4 / 1 | 2 | 64 | 40 | tied | 2.87e-06 | 4.57 | 0.026 | 40/40 |
| `untied-cls` | 128 | 4 / 4 | 4 | 512 | 64 | **untied** | 9.66e-06 | 5.23 | 0.071 | 64/64 |
| `full-context` | 48 | 2 / 2 | 1 | 50 | 128 | tied | 5.01e-06 | 4.50 | 0.029 | 128/128 |

**6/6 pass.** Worst case uses 7.1% of the allowed budget. Top-1 agreement is
100% everywhere.

`max|ref|` is printed on purpose. It is the evidence that the weights were
re-randomised: llama2.c initialises matrices with `std=0.02`, which makes the
logits nearly flat, and a flat output lets a whole family of genuinely wrong
implementations land inside the tolerance. A max logit of ~5 rather than ~1e-3
means the test is measuring the arithmetic and not the tolerance.

### Differential test, real weights

`python reference/diff_test.py --checkpoint stories15M.pt -n 200`

| model | max&nbsp;\|err\| | max&nbsp;\|ref\| | err/tol | top-1 | result |
|---|---|---|---|---|---|
| stories15M (15M params, vocab 32000) | 3.34e-05 | 26.2 | 0.192 | 200/200 | PASS |

The 200-token sequence is chosen by the *reference* greedy decoder, so the
engine is being compared on a sequence an independent implementation produced.

### Mutation check

`python reference/mutation_check.py`

| mutant | caught by |
|---|---|
| RoPE exponent `2i` → `i` | all six |
| RoPE skipped on keys | all six |
| GQA mapping `h / kv_mul` → `h % n_kv_heads` | **gqa-2x, gqa-4x only** |
| Attention window `pos + 1` → `pos.max(1)` | all six |
| Attention scale removed | all six |
| RMSNorm eps `1e-5` → `1e-27` | all six |
| SwiGLU gate and up swapped | all six |
| Untied classifier ignored | **untied-cls only** |
| Residual `x += d` → `x = d` | all six |
| RoPE theta 10000 → 500000 | all six |

"all six" means `tiny, gqa-2x, gqa-4x, mqa, untied-cls, full-context`, the
order the harness reports them in.

**10/10 caught.** The whole run takes roughly 20-30 seconds, dominated by
rebuilding the crate ten times; every mutant shares one `CARGO_TARGET_DIR` and
runs with LTO disabled, since LTO is the slowest part of the build and has no
bearing on which mutants get caught.

The per-mutant column is the useful part, and it is why the configurations are
chosen the way they are. `gqa-modulo-mapping` is caught by the two GQA
configurations and nothing else, because modulo and division agree when
`kv_heads == n_heads` — the other four configurations cannot see that bug at
all. `ignore-untied-classifier` is caught by `untied-cls` alone, for the
mirror-image reason. If every mutant had been caught by all six
configurations, that would have looked like thoroughness and actually meant the
suite had no discrimination.

Two guards keep that number honest:

- Each mutation's replacement snippet must occur **exactly once** in the
  source. Otherwise a refactor turns the mutant into a no-op, the harness
  correctly reports a pass, and "the test passed" becomes indistinguishable
  from "the bug slipped through".
- The **unmutated source is checked first**. "10/10 caught" would otherwise be
  consistent with "10/10 failed for an unrelated reason".

### Rust tests, and proof they can fail

75 tests, passing in both release and debug (debug matters: it enables integer
overflow checks, which is where a wrapping size computation in the loader would
show up).

| suite | tests | what it covers |
|---|---|---|
| `src/ops.rs` | 23 | kernels against hand-computed answers, RoPE invariants *and* its frequency schedule, causality of the attention window, `dot` across every length mod 8 |
| `tests/loader.rs` | 18 | totality over hostile input, 2000 pseudo-random buffers, overflow, tensor ordering |
| `tests/tokenizer.rs` | 16 | exact conformance to Meta's published token ids, byte fallback, round trips |
| `tests/engine.rs` | 18 | causality, determinism, BOS termination, `State::reset` equivalence, `argmax` totality on NaN |

But a passing suite only says the tests agree with the code. It does not say
they can *notice* disagreement, so `reference/mutation_check_tests.py` injects 21
realistic bugs one at a time and requires `cargo test` to fail on every one.
**21/21 caught.** It is the same discipline as `mutation_check.py`, pointed at
the other half of the project: that one proves the differential harness can
fail, this one proves the Rust tests can.

Running it found two genuine blind spots, both of which were real holes in the
suite rather than in the mutants:

- **The RoPE tests could not see the RoPE frequencies.** All four were
  *structural* invariants — identity at pos 0, pair-norm preservation, no head
  mixing, the relative-position property — and every one of them is satisfied
  by a rotation through *any* angle. So `2 * i` → `i` produced angles that are
  wrong everywhere and passed the lot. The differential harness caught it; the
  unit suite on its own could not. There is now a test that computes the
  expected angle from the documented schedule and compares.
- **Nothing covered generation's BOS termination.** That is the bug the
  byte-identity check found by hand, and the differential harness cannot see it
  because it never looks at text. There is now an engineered checkpoint whose
  first greedy step emits BOS, plus its mirror so the first test cannot pass
  for the wrong reason.

A third finding was subtler and worth naming: `state_rejects_a_config_whose_cache_would_overflow`
was using `i32::MAX` as `dim`, which is *odd*, so `State::new` rejected the odd
head size and returned `InvalidHeader`. The test passed without ever reaching
the arithmetic it is named after. It now uses an even `dim`, asserts the config
is structurally valid, and asserts the specific `Overflow { what: "kv cache" }`
error rather than merely that some error occurred.

### Every check, and how it was shown to fail

A check that has only ever been observed passing is consistent with a check
that cannot fail. So each one was also exercised in a state where it had to go
red. This is the inventory:

| check | how it was shown to fail |
|---|---|
| `diff_test.py` (6 configs) | 10 injected bugs, `mutation_check.py`, 10/10 |
| `cargo test` (75 tests) | 21 injected bugs, `mutation_check_tests.py`, 21/21 |
| byte-identical greedy output | a RoPE-less mutant binary: `bench.py` reports `DIFFERENT` and exits 1 |
| zero-dependency assertion | a `Cargo.lock` with a second package, with zero packages, and naming a package that is not `tinyinfer` — all three rejected |
| `verus/check_citations.py` | GQA mapping mutated in place, nine lines inserted to shift every citation, attention window changed, a cache write deleted; plus a missing source file, an unregistered citation, and a rotted table entry |
| CI workflow wiring | every `steps.<id>.outputs.<name>` resolves to a step that writes it, every `run:` block parses as bash, no `\|\| true` outside comments |
| Verus itself | `--no-cheating` rejects `assume` / `admit` / `external_body`; the unmutated baseline is run first so "24 verified" cannot mean "the file does not parse" |

The full-run commands are in the repository, not just in a shell history:
`reference/mutation_check.py`, `reference/mutation_check_tests.py`, and
`verus/check_citations.py` all exit non-zero on failure and all run in CI.

## Why the differential test is worth anything

The two implementations share nothing but the file format.

|  | reference (PyTorch) | tinyinfer (Rust) |
|---|---|---|
| forward pass | batched over the whole sequence | one token at a time |
| attention | `scaled_dot_product_attention(is_causal=True)` | explicit loop over the KV cache |
| RoPE | precomputed `freq_cis` table in the file | computed from `theta` and `pos` in `ops::rope` |
| GQA heads | `torch.repeat_interleave` | `h / kv_mul` index division |
| weights | `f32` tensors via `numpy` | `f32` slices read by a bounds-checked cursor |

Two consequences worth spelling out.

**RoPE is computed, not read.** The checkpoint file *contains* a `freq_cis`
table, and reading it would have been easier. But then the differential test
would compare two programs sharing a precomputed constant, and a bug in our
RoPE would be invisible. Computing it means the test genuinely exercises our
angle arithmetic. The loader skips those bytes on purpose.

**The tolerance is a statement about summation order, not about the model.**
Both sides sum the same products in different orders, and floating-point
addition is not associative, so a bitwise comparison would be testing
`matmul`'s inner loop rather than the transformer. Hence
`1e-4 + 1e-4*|ref|`: the absolute term covers logits near zero, where a
relative bound is meaningless because the reference itself is working in `f32`
and has no significant digits left, and the relative term covers large logits
where accumulated rounding grows with magnitude. For fp16 the relative term
would have to be around 1e-2, because fp16 carries about three decimal digits.

## Benchmark

Apple M3, macOS 26.4.1, rustc 1.98.0, Apple clang 21.0.0. Single threaded.
stories15M (dim 288, 6 layers, 6 heads, vocab 32000, context 256), 248
generated greedy tokens, 5 runs, median.

Every row was measured in a single session with the same `run.c` binaries, so
no row is quoted from a run with different machine load than the others. Run to
run spread on this machine is a few percent, and the first run after a rebuild
is reliably ~12% slow (page faults on the 60 MB checkpoint), so the medians are
quoted and the digits are not meaningful.

| engine | tok/s |
|---|---|
| tinyinfer, serial `.sum()` dot product | 154.3 |
| tinyinfer, 8 accumulators, indexed loop | 207.9 |
| tinyinfer, 8 accumulators, `chunks_exact` | 835.4 |
| tinyinfer, same, plus `-C target-cpu=native` | 836.8 |
| llama2.c `run.c`, `-O3 -march=native` | 146.8 |
| llama2.c `run.c`, `-Ofast` | 973.6 |

Greedy output is **byte-identical** to both `run.c` builds across all 248
tokens, for all four tinyinfer variants.

### The 8-lane dot product, including the part I got wrong

`a.iter().zip(b).map(|(x, y)| x * y).sum()` compiles to a serial chain of
floating-point additions. Since addition is not associative, LLVM is *forbidden*
from splitting that chain into parallel partial sums, because doing so would
change the answer. One loop-carried dependency per multiply is a latency wall
that no amount of `-O3` gets past.

Eight independent accumulators remove the dependency. That is the standard
advice, and following it exactly gets you to 207.9 tok/s — a real 1.35x, and
most of the way short of the real win.

Written that way:

```rust
while i + 8 <= n { for lane in 0..8 { acc[lane] += a[i + lane] * b[i + lane]; } }
```

the eight lanes become eight independent **scalar** chains. The dependency is
gone, but LLVM's loop vectoriser does not fire on that form at all. Rewriting
the identical arithmetic as `chunks_exact(8)` reaches 835.4 tok/s, a further
4.0x, because it presents the eight values as one contiguous chunk and the
superword-level pass can pack them into vectors.

Measured in isolation at `dim = 288` (the model's hidden size), in Gelem/s:

| dot product form | Gelem/s |
|---|---|
| serial `.sum()` | 2.2 |
| 8 accumulators, indexed loop | 3.3 |
| 8 accumulators, `chunks_exact(8)` | 14.6 |

The indexed form is a 1.5x on the kernel, and looks like the optimisation
worked. It is 4.4x short of what the same arithmetic can do.

Nothing is reassociated in either version. Each lane is still a strict
left-to-right sum; the eight lanes are just eight interleaved ordered sums. The
last bits of the result do change, which is precisely why the differential
test is tolerance-based — and the diff test, the mutation check and the
byte-identical comparison were all re-run after this change.

### The `-Ofast` gap, reported honestly

`run.c -Ofast` is 973.6 tok/s and still faster than the 835.4 tok/s this
engine reaches with an optimising Rust compiler and explicit SIMD. The gap is
about 1.17x.

`-Ofast` enables `-ffast-math`, which lets clang reassociate floating-point
additions *anywhere* it likes, including the attention and value-accumulation
loops, not just the matmul. This project restricts itself to one documented
reassociation in one function, because unrestricted reassociation across the
whole forward pass makes the numerics much harder to reason about and much
easier to break silently.

The 1.17x is the price of that choice. It is a real cost, reported rather than
hidden: the honest summary is that clang, allowed to assume the IEEE 754 rules
do not apply, extracts more than a conforming compiler can, and that some of
what it extracts is available to a conforming compiler if you write the loop in
the right shape — as the `chunks_exact` result above demonstrates — and some of
it is not.

`-C target-cpu=native` adds 0.2% here (835.4 → 836.8), which is inside the
run-to-run noise. That is worth stating plainly because it is not the result
one would predict from an x86 machine with AVX-512: on aarch64 the baseline
codegen already saturates the available NEON width for this loop, so there is
nothing left for a more specific target to unlock. The `chunks_exact` rewrite,
not the target flag, is what unlocked the vectorisation.

## Repository layout

```
tinyinfer/
  Cargo.toml                 no [dependencies]; release profile: opt-level 3, lto, codegen-units 1
  LICENSE                    MIT
  CONTEXT.md                 design notes: format, forward pass, gotchas, TCB
  src/
    lib.rs                   module map
    ops.rs                   kernels: rmsnorm, softmax, silu, matmul, RoPE, GQA
    model.rs                 config, hardened checkpoint loader, State, forward, generate
    tokenizer.rs             BPE vocabulary reader, encoder, decoder
    main.rs                  CLI
  tests/
    engine.rs                forward/generation properties
    loader.rs                hostile-input totality
    tokenizer.rs             conformance to Meta's token vectors
  reference/
    diff_test.py             differential harness vs PyTorch
    mutation_check.py        proves the differential harness can fail
    mutation_check_tests.py  proves the Rust test suite can fail
    bench.py                 benchmark + byte-identity check
    llama2c/                 vendored UNMODIFIED from karpathy/llama2.c (MIT)
  verus/                     Verus proof of KV-cache index safety (24 conditions,
                             machine-checked; see its README for the scope)
                             plus check_citations.py, which keeps the proof
                             from silently drifting away from src/
  .github/workflows/ci.yml   four jobs, no `|| true`, no continue-on-error
```

## Quick start

```sh
cargo build --release
cargo test                    # 72 tests
```

Generation, with a real checkpoint:

```sh
curl -fSL -o stories15M.pt \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt

# Export it to the legacy format this engine reads.
python - <<'PY'
import sys, contextlib, io
sys.path.insert(0, "reference/llama2c")
from export import load_checkpoint, legacy_export
with contextlib.redirect_stdout(io.StringIO()):
    legacy_export(load_checkpoint("stories15M.pt"), "stories15M.bin")
PY

./target/release/tinyinfer stories15M.bin \
    -z reference/llama2c/tokenizer.bin -i "Once upon a time" -n 248
```

The harness needs CPU-only PyTorch:

```sh
python -m venv .venv
.venv/bin/pip install numpy torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

.venv/bin/python reference/diff_test.py
.venv/bin/python reference/diff_test.py --checkpoint stories15M.pt -n 200
.venv/bin/python reference/mutation_check.py
```

The benchmark and byte-identity check, against the C reference:

```sh
cc -O3 -march=native -o /tmp/run_O3   reference/llama2c/run.c
cc -Ofast              -o /tmp/run_Ofast reference/llama2c/run.c
.venv/bin/python reference/bench.py --model stories15M.bin \
    --run-c /tmp/run_O3 --run-c /tmp/run_Ofast
```

## Design decisions worth defending

**Zero dependencies, seriously.** `Cargo.lock` contains exactly one package,
and CI asserts it before anything else runs. Arg parsing, file IO, the f32
kernels, the BPE tokenizer and the CLI are all built on `std`. For a project
about machine-verified inference, the smallest possible dependency surface is
the point: every crate in the lock file widens the trusted base by another
transitive `unsafe` blob.

**The loader treats the file as hostile.** The checkpoint header dictates how
much memory gets allocated, so it cannot be trusted to be right. Three
properties, each with a test that fails if it breaks:

1. *Totality.* For any byte string, `from_bytes` returns `Ok` or `Err`, never a
   panic. `tests/loader.rs` feeds it 2000 pseudo-random buffers, half prefixed
   with a plausible header so the cursor gets past validation into the
   size arithmetic, which is where the bugs actually live.
2. *No allocation from an unvalidated number.* A header claiming 4096 dim ×
   32000 vocab on a 64 KiB file is a perfectly legal set of numbers, so the
   only thing preventing a multi-gigabyte allocation is checking the bytes are
   present *before* creating the `Vec`.
3. *Overflow is an error, not a wraparound.* Every size product goes through
   `checked_mul`. This forced a design change: computing each tensor's length
   inline at the point of reading meant a header that overflowed at `wq`
   reported a confusing "token_embedding needs 8589934584 bytes" instead.
   `TensorSizes` now computes the whole shape table up front.

**`forward` allocates nothing.** Every buffer lives in `State` and is written
in place, so the memory footprint of decoding is a sum you can write down:
scratch is a handful of vectors, and the cache dominates at
`2 * n_layers * seq_len * kv_dim`.

**Greedy only, no sampling.** Sampling needs an RNG, and an RNG would make the
byte-identical comparison against `run.c` impossible unless both sides
consumed the identical stream. Greedy decoding is a pure function of the weights
and the prompt, so two implementations that agree on the maths produce
byte-identical output — a far stronger claim than "the samples look similar".

**Bugs found by the byte-identity check that no unit test caught.** Generation
originally stopped on EOS; llama2.c stops on BOS, which is the document
delimiter these models are trained with, so with stories15M our output ran
straight past the end of the first story. Output was also printed by decoding
the whole sequence at once, whereas `run.c` prints per token through
`safe_printf`, which drops single non-printable bytes. Both were found by
`bench.py` and neither was findable from the differential test, which does not
look at text at all.

## Trusted computing base

Stated plainly, because it is the honest answer to "how much do you actually
trust this?": the Rust compiler and `std`, the f32 semantics of the CPU, the
checkpoint parsing code, and the reference implementation being tested against.
Everything above those is tested. Those are assumed, and the assumption is
written down rather than implied.

## What is formally verified, and what is not

The first thing worth proving mechanically is **index safety of the KV-cache
slicing in `forward`**: that for any `pos < seq_len` and any layer, every cache
index computed in `ops::attention` and `State::forward` is in bounds.

`verus/` contains a Verus development for exactly that, and it is
**machine-checked**: Verus `0.2026.09.20.aef82ed` reports
`24 verified, 0 errors` under `--no-cheating`, which rejects `assume`, `admit`
and `external_body` outright. That is a real run, not a claim.

What is proved: the bounds of every KV-cache index expression for all shapes
satisfying the loader's structural preconditions and all `pos < seq_len`; that
those preconditions follow from the checks the loader actually performs; that
the GQA mapping `h / kv_mul` lands in `0..n_kv_heads`; and that the index
arithmetic cannot overflow `usize`. The real slice expressions are transcribed
into `exec fn`s operating on actual `Vec<f32>`, so Verus discharges genuine
slice-indexing obligations rather than assertions about symbolic expressions.

### Keeping the proof honest

A green Verus run is not enough on its own, because the proof verifies a
*model*. If someone changed `h / kv_mul` in `ops.rs`, the proof would keep
verifying, faithfully, about code that no longer exists — and the green tick
would be actively misleading.

So `verus/check_citations.py` checks the link. Every source line the proof and
its README cite must still contain the construct it is cited for. It is
bidirectional — an uncited citation and an unused expectation are both failures,
so neither the prose nor the table can quietly stop being checked. Stdlib only,
so it costs nothing in CI.

The check has been negative-tested, which matters more than it passing: it goes
red when the GQA mapping is mutated in place, when an unrelated edit shifts the
line numbers, when the attention window changes, and when a cache write is
deleted from `forward`.

What it still does not check is the reasoning. An edit that preserves a cited
expression's text while changing what it computes elsewhere would pass, as would
a gap in which expressions were transcribed at all. It is a tripwire on the
likely accident, not a proof of correspondence.

### What is **not** proved

`verus/README.md` lists eleven items; these are the two that matter most:

1. **It is a proof about a transcription, not about the crate.** `src/ops.rs`
   and `src/model.rs` contain no `verus!` macro and are not compiled by Verus.
   The citation check above narrows how far this can go wrong; it does not
   close it.
2. **Bounds safety does not catch wrong-but-in-bounds bugs.** Mutant #3
   (`h / kv_mul` -> `h % n_kv_heads`) is also in range and sails straight
   through. That is the boundary between what a verifier can say and what only
   the differential test can say, and the proof file says so itself.

### Running the proof

The `proof` CI job fetches the prebuilt Verus release, verifies it against the
sha256 GitHub publishes for that asset, installs the toolchain Verus asks for
(it prints the exact `rustup install` line, which the job parses rather than
hardcoding a version), and runs the verifier. Verus is not a dependency of
this crate and is not vendored.

**One caveat, stated plainly:** that job's logic was dry-run locally and passes
all four steps, but the Verus *run* was verified on macOS/arm64. The Linux
x86-64 download and execution are the one part of this repository that has not
been executed, because this machine is not a GitHub runner. If that job goes red
on the first run, the citation check and the other three jobs are unaffected,
and the log will name the step.

Nothing else here is machine-checked. The loader's totality rests on 2000 fuzz
iterations plus reasoning, not a proof; the differential test establishes
agreement with a reference on six configurations and one trained checkpoint,
not equivalence to the architecture in general.

Two other things were scoped and deliberately **not** done, rather than
half-done: a differential test against **vLLM** (needs a GPU and a Hugging Face
export path, and the harness is written so that adding a third oracle is a new
config table rather than a new file), and **Triton** kernels for rmsnorm,
softmax and matmul (a benchmarking exercise with no bearing on whether the
engine is correct, which is the claim this repository makes). The measurement
in this repository is CPU and single-threaded throughout; nothing here says
anything about GPU inference.

## Credits

The PyTorch reference model, the legacy export format, the tokenizer format,
the C engine, and the expected token id vectors all come from
[karpathy/llama2.c](https://github.com/karpathy/llama2.c), MIT licensed. The
files under `reference/llama2c/` are vendored **byte-for-byte unmodified**;
`reference/llama2c/PROVENANCE.md` records the sha256 of each so a reviewer can
verify that, and it matters more than it might look — a differential test
against a reference that has been edited is worth nothing.

The token id expectations in `tests/tokenizer.rs` originate in Meta's
`example_text_completion.py` and are checked by llama2.c's `test.c`.
