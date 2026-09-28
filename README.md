# proofinfer

A Llama inference engine in Rust with **zero external crates**, built around a
claim that is usually made and rarely justified: that it computes the right
thing.

The engine is the easy half. The rest is the evidence:

- **Differential testing** against the PyTorch reference from
  [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — 6 configurations,
  every logit at every position.
- **Byte-identical output** against `run.c`, a third independent implementation
  in C.
- **Mutation testing** on both halves of the suite, so neither the harness nor
  the tests can be quietly incapable of failing.
- A **machine-checked Verus proof** of KV-cache index safety.

## Results

Full detail, including how each check was shown to be able to fail, is in
[docs/testing.md](docs/testing.md).

| | |
|---|---|
| Differential, 6 configurations | 6/6 pass, max error 1.0e-5, top-1 100% |
| Real `stories15M` weights | 200/200 top-1, max error 3.3e-5 |
| Greedy output vs `run.c` | byte-identical at `-O3` and `-Ofast` |
| Mutation: differential harness | 10/10 caught |
| Mutation: Rust test suite | 21/21 caught |
| Verus `--no-cheating` | 24 verified, 0 errors |
| Rust tests | 75, green in debug **and** release |
| Throughput | 835 tok/s single-threaded (5.4x over the naive dot product) |

## Quick start

```sh
cargo build --release
cargo test                              # 75 tests, both profiles
```

Generate from text, with a real checkpoint:

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

./target/release/proofinfer stories15M.bin \
    -z reference/llama2c/tokenizer.bin -i "Once upon a time" -n 248
```

Run the harnesses (needs CPU-only PyTorch):

```sh
python -m venv .venv
.venv/bin/pip install numpy torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

.venv/bin/python reference/diff_test.py        # 6/6 vs PyTorch
.venv/bin/python reference/mutation_check.py    # can the harness fail?
.venv/bin/python reference/mutation_check_tests.py  # can the tests fail?
```

## Documentation

| | |
|---|---|
| [docs/testing.md](docs/testing.md) | the differential and mutation harnesses, and how every check was invalidated to prove it works |
| [docs/performance.md](docs/performance.md) | the 8-lane dot product, the measurement method, and the gap left by `-Ofast` |
| [docs/design.md](docs/design.md) | architecture, the decisions worth defending, trusted computing base, and what is *not* verified |
| [CONTEXT.md](CONTEXT.md) | working design notes kept during the build |
| [verus/README.md](verus/README.md) | the formal proof, its scope, and its eleven stated limitations |
| [site/](site/) | the landing page — static, generated from these docs, no build step |

## Honest limits

- The Verus proof is a proof **about a transcription**, not about the crate.
  `src/` contains no `verus!` macro. `verus/check_citations.py` narrows how far
  that can silently drift; it does not close the gap. See
  [docs/design.md](docs/design.md).
- `run.c -Ofast` is still about 1.2x faster. Reported with its cause rather
  than buried.
- Throughput figures are from an idle machine and are not reproducible under
  load; use them for ratios, not absolutes.
- **Not attempted:** vLLM differential testing and Triton kernels. Both need a
  GPU, and a half-finished GPU comparison is worth less than none.

## Credits

The PyTorch reference, the legacy export format, the tokenizer format, the C
engine, and the expected token id vectors all come from
[karpathy/llama2.c](https://github.com/karpathy/llama2.c) (MIT). The files under
`reference/llama2c/` are vendored **byte-for-byte unmodified**; sha256 for each
is recorded in `reference/llama2c/PROVENANCE.md`.

MIT licensed. See [LICENSE](LICENSE).
