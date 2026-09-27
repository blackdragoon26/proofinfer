#!/usr/bin/env python3
"""Differential test: tinyinfer (Rust) against the llama2.c reference (PyTorch).

The claim being tested is narrow and checkable: for the same weights and the
same token sequence, both implementations produce the same logits at every
position.

What makes it worth running is that the two implementations share nothing but
the file format:

* The reference does a **batched** forward pass over the whole sequence with a
  causal mask (`scaled_dot_product_attention(..., is_causal=True)`).
* tinyinfer decodes **one token at a time** with a KV cache, computing its own
  RoPE from `theta` and `pos` rather than reading the `freq_cis` table the file
  also contains, and mapping grouped-query heads by index division rather than by
  `torch.repeat_interleave`.

So agreement is evidence about the mathematics, not about a shared constant or
a shared code path. If either side has a bug in RoPE, in the GQA mapping, in
the attention window, in SwiGLU, or in the residual stream, the two will
disagree.

Why tolerance-based rather than bitwise: the two implementations sum the same
products in different orders (torch batches them, the Rust dot product is a
serial or 8-lane reduction), and floating-point addition is not associative. A
bitwise comparison would be testing the summation order, not the model.

Usage:
    python3 reference/diff_test.py
    python3 reference/diff_test.py --binary ./target/release/tinyinfer
    python3 reference/diff_test.py --config tiny --config mqa
    python3 reference/diff_test.py --checkpoint stories15M.pt -n 200
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
LLAMA2C = REPO / "reference" / "llama2c"
sys.path.insert(0, str(LLAMA2C))

# The vendored reference is imported by path. Its `tokenizer.py` needs
# sentencepiece, which this harness never uses, so it is never imported.
from export import legacy_export, load_checkpoint  # noqa: E402
from model import ModelArgs, Transformer  # noqa: E402

# Absolute + relative tolerance on every logit.
#
# The absolute term dominates for logits near zero, where a relative bound
# would be meaningless: the reference itself computes in f32, so a logit of
# magnitude 1e-6 has no significant digits left to compare. The relative term
# then takes over for the large logits, where accumulated rounding error grows
# with magnitude. Together they bound both "the answer is near zero and both
# sides agree it is" and "the answer is large and the relative error is small".
ABS_TOL = 1e-4
REL_TOL = 1e-4

# Llama 2's BOS id, fixed by the SentencePiece convention. The real-weights mode
# starts from it so the harness needs no tokenizer.
BOS_ID = 1


@dataclasses.dataclass(frozen=True)
class Config:
    """One differential-test configuration.

    `hidden` is not part of the published table; it is fixed at 2x dim, which
    keeps the FFN matrices small enough that the whole suite runs in seconds
    while still being big enough that a transposed weight would be obvious.
    """

    name: str
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    tied: bool

    @property
    def hidden_dim(self) -> int:
        return 2 * self.dim

    def describe(self) -> str:
        head_size = self.dim // self.n_heads
        kind = (
            "MHA"
            if self.n_kv_heads == self.n_heads
            else ("MQA" if self.n_kv_heads == 1 else f"GQA {self.n_heads // self.n_kv_heads}x")
        )
        return (
            f"dim={self.dim} layers={self.n_layers} heads={self.n_heads} "
            f"kv_heads={self.n_kv_heads} ({kind}, head_size={head_size}) "
            f"vocab={self.vocab_size} seq={self.seq_len} "
            f"classifier={'tied' if self.tied else 'untied'}"
        )


# Six configurations chosen to cover every structural axis the engine has:
# plain multi-head attention, two levels of GQA, multiquery, a separate
# classifier, and a long context that exercises the cache at depth.
CONFIGS: tuple[Config, ...] = (
    Config("tiny", 64, 2, 4, 4, 97, 32, True),
    Config("gqa-2x", 64, 3, 8, 4, 128, 48, True),
    Config("gqa-4x", 96, 2, 8, 2, 200, 64, True),
    Config("mqa", 64, 2, 4, 1, 64, 40, True),
    Config("untied-cls", 128, 4, 4, 4, 512, 64, False),
    Config("full-context", 48, 1, 2, 2, 50, 128, True),
)


@dataclasses.dataclass
class Result:
    name: str
    max_abs_err: float
    max_rel_to_tol: float  # max(err / allowed); <= 1 means passing
    max_abs_ref: float
    max_abs_got: float
    top1_agreed: int
    n_positions: int
    worst: Optional[str] = None
    top1_mismatches: int = 0

    @property
    def ok(self) -> bool:
        return self.max_rel_to_tol <= 1.0 and self.top1_mismatches == 0


def reinitialise(model: Transformer, seed: int) -> None:
    """Re-randomise the weights so the logits are not flat.

    This is not cosmetic. `Transformer.__init__` initialises every matrix with
    `std=0.02`, which makes the logits almost identical across the vocabulary.
    With near-constant logits, a whole family of genuinely wrong
    implementations still lands inside the tolerance: a wrong RoPE exponent, a
    missing attention scale, or a swapped SwiGLU gate all perturb a nearly flat
    output by a negligible amount. Re-randomising removes that slack, so the
    tolerance is testing the maths rather than the tolerance.

    Matrices get `randn / sqrt(fan_in)`, which is the standard fan-in scaling
    and keeps activations at roughly unit variance through the stack. Norm
    weights get `1 + 0.5 * randn` so they are near 1 but not exactly 1, because
    an all-ones norm weight would hide a bug that swapped the norm weight for a
    constant.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                # fan_in is the contracted dimension. For nn.Linear(dim, out)
                # the weight is (out, dim), so shape[1] is fan_in. For the
                # embedding it is (vocab, dim) and shape[1] is also the
                # contraction length.
                fan_in = p.shape[1]
                p.copy_(torch.randn(p.shape, generator=generator) / (fan_in**0.5))
            else:
                p.copy_(1.0 + 0.5 * torch.randn(p.shape, generator=generator))


def build_reference(cfg: Config, seed: int) -> Transformer:
    """Construct a reference model for `cfg` with fresh, non-flat weights."""
    torch.manual_seed(seed)
    args = ModelArgs(
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        max_seq_len=cfg.seq_len,
        multiple_of=32,
    )
    model = Transformer(args)

    if not cfg.tied:
        # `Transformer.__init__` ties `tok_embeddings.weight` to
        # `output.weight` by assignment, so they are literally the same tensor.
        # `legacy_export` decides tied-vs-untied with `torch.equal`, which would
        # see them as equal and write a tied file. Replacing the output weight
        # with a fresh Parameter breaks the aliasing first.
        model.output.weight = torch.nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.dim)
        )

    reinitialise(model, seed + 1)
    model.eval()
    return model


def reference_logits(model: Transformer, tokens: torch.Tensor) -> np.ndarray:
    """Logits for every position, as a `(seq_len, vocab)` array.

    The `targets=` argument is essential. `Transformer.forward` only projects
    the *last* position to logits when `targets is None`; that is a deliberate
    training-time optimisation and it means a plain `model(x)` call returns one
    row. Passing `targets=x` takes the branch that computes the full
    `(1, seq_len, vocab)` projection we need. Getting this wrong looks like a
    shape mismatch rather than a numerical one, which is a nasty way to lose an
    afternoon.
    """
    with torch.no_grad():
        logits = model(tokens, targets=tokens)
    assert logits.shape == (1, tokens.shape[1], model.vocab_size), logits.shape
    return logits[0].to(torch.float32).numpy()


def run_engine(binary: Path, checkpoint: Path, tokens: list[int], dump: Path) -> np.ndarray:
    """Run the Rust engine over `tokens` and return its `(n, vocab)` logits."""
    args = [
        str(binary),
        str(checkpoint),
        "--tokens",
        ",".join(str(t) for t in tokens),
        "--dump-logits",
        str(dump),
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"engine exited {proc.returncode}\n"
            f"  command: {' '.join(args)}\n"
            f"  stderr: {proc.stderr.strip()}"
        )

    raw = np.fromfile(dump, dtype="<f4")
    n = len(tokens)
    if raw.size == 0 or raw.size % n != 0:
        raise RuntimeError(
            f"engine wrote {raw.size} floats, which is not a whole number of "
            f"rows for {n} positions; the dump is malformed"
        )
    # The vocabulary size is implied by the row length rather than read back
    # from the checkpoint: the engine writes exactly `vocab` floats per
    # position, so a mismatch here means the shapes disagree and `compare`
    # will report it with both shapes named.
    return raw.reshape(n, raw.size // n)


def compare(name: str, ref: np.ndarray, got: np.ndarray) -> Result:
    """Compare two `(n, vocab)` logit matrices under the stated tolerance."""
    if ref.shape != got.shape:
        raise RuntimeError(f"{name}: shape mismatch {ref.shape} vs {got.shape}")

    err = np.abs(got - ref)
    allowed = ABS_TOL + REL_TOL * np.abs(ref)
    # Divide rather than compare directly so the report can say *how far* over
    # budget something is, which is the difference between a useful CI failure
    # and a bare "assertion failed".
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(allowed > 0, err / allowed, np.inf)

    ref_top1 = ref.argmax(axis=-1)
    got_top1 = got.argmax(axis=-1)
    mismatches = int((ref_top1 != got_top1).sum())

    worst = None
    if np.isfinite(ratio).any() and ratio.max() > 1.0:
        pos = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(ratio), ratio, -np.inf))), ratio.shape)
        worst = (
            f"position {int(pos[0])}, logit {int(pos[1])}: "
            f"ref={ref[pos]:.6g} got={got[pos]:.6g} "
            f"err={err[pos]:.3g} allowed={allowed[pos]:.3g}"
        )

    return Result(
        name=name,
        max_abs_err=float(err.max()),
        max_rel_to_tol=float(ratio.max()),
        max_abs_ref=float(np.abs(ref).max()),
        max_abs_got=float(np.abs(got).max()),
        top1_agreed=int((ref_top1 == got_top1).sum()),
        n_positions=ref.shape[0],
        worst=worst,
        top1_mismatches=mismatches,
    )


def run_random_config(cfg: Config, binary: Path, seed: int) -> Result:
    """The full round trip for one random-weight configuration."""
    model = build_reference(cfg, seed)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / f"{cfg.name}.bin"
        logits_path = Path(tmp) / f"{cfg.name}.f32"

        # legacy_export prints "wrote <path>" to stdout. Capture it so the
        # report stays readable and so a stray print cannot be mistaken for
        # harness output in CI logs.
        with contextlib.redirect_stdout(io.StringIO()):
            legacy_export(model, str(checkpoint))

        # A fresh generator so the token sequence depends only on the config and
        # the seed, never on how many random numbers the weight initialisation
        # happened to draw.
        gen = torch.Generator().manual_seed(seed + 2)
        tokens = torch.randint(
            0, cfg.vocab_size, (1, cfg.seq_len), generator=gen, dtype=torch.long
        )

        ref = reference_logits(model, tokens)
        ids = [int(t) for t in tokens[0].tolist()]
        got = run_engine(binary, checkpoint, ids, logits_path)

    return compare(cfg.name, ref, got)


def run_real_checkpoint(
    binary: Path, checkpoint: Path, n: int, workdir: Path
) -> Result:
    """Compare against a genuinely trained checkpoint.

    The random-weight suite proves the *maths* agrees. It cannot prove the engine
    loads and runs a real Llama 2 checkpoint, because a real one has weight
    distributions, activation magnitudes and a 32000-entry vocabulary that a
    synthetic checkpoint does not. This mode closes that gap.

    The token sequence is produced by the *reference* greedy decoder, so the
    engine is being asked about a sequence chosen by a completely independent
    implementation, and every position of it is compared.

    The prompt is a bare BOS. That is not laziness: this is a numerical
    comparison, so what the tokens *mean* is irrelevant - any valid id sequence
    exercises the same code paths. Using BOS alone means the harness needs no
    tokenizer at all, which keeps `sentencepiece` out of the differential-test
    dependency set and removes any question about the two sides having
    tokenised the same text differently. (`tests/tokenizer.rs` is where
    tokenisation is actually tested, against Meta's own vectors.)
    """
    model = load_checkpoint(str(checkpoint))
    model.eval()
    seq_len = model.params.max_seq_len
    vocab = model.params.vocab_size
    print(
        f"  checkpoint: dim={model.params.dim} layers={model.params.n_layers} "
        f"heads={model.params.n_heads} kv_heads={model.params.n_kv_heads} "
        f"vocab={vocab} max_seq_len={seq_len}",
        flush=True,
    )
    if n > seq_len:
        raise RuntimeError(
            f"-n {n} exceeds the checkpoint context ({seq_len}); the generated "
            f"sequence has to fit in the KV cache"
        )

    # Greedy-decode with the reference until the sequence is `n` long. This is
    # O(n^2) token-forwards because the reference has no KV cache, so it is the
    # slow part of this mode: on stories15M with n=200 it is tens of seconds,
    # and it is worth waiting for because every one of those positions is a
    # comparison the engine has to agree on.
    print(
        f"  reference greedy decode to {n} tokens (no KV cache, this is slow)...",
        flush=True,
    )
    started = time.time()
    tokens = [BOS_ID]
    while len(tokens) < n:
        idx = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            # No `targets=` here: the reference's inference-time path projects
            # only the final position, which is exactly the one row needed to
            # pick the next token.
            logits = model(idx)
        tokens.append(int(logits[0, -1].argmax().item()))
    print(f"  decoded {len(tokens)} tokens in {time.time() - started:.1f}s", flush=True)

    idx = torch.tensor([tokens], dtype=torch.long)
    ref = reference_logits(model, idx)

    legacy = workdir / f"{checkpoint.stem}.bin"
    with contextlib.redirect_stdout(io.StringIO()):
        legacy_export(model, str(legacy))
    got = run_engine(binary, legacy, tokens, workdir / f"{checkpoint.stem}.f32")

    return compare(f"real:{checkpoint.name}", ref, got)


def report(results: list[Result]) -> None:
    print()
    print(
        f"{'config':<16}{'max|err|':>12}{'max|ref|':>12}{'err/tol':>10}"
        f"{'top-1':>12}{'result':>9}"
    )
    print("-" * 71)
    for r in results:
        top1 = f"{r.top1_agreed}/{r.n_positions}"
        status = "PASS" if r.ok else "FAIL"
        print(
            f"{r.name:<16}{r.max_abs_err:>12.3e}{r.max_abs_ref:>12.3e}"
            f"{r.max_rel_to_tol:>10.3f}{top1:>12}{status:>9}"
        )
        if not r.ok and r.worst:
            print(f"{'':<16}  worst: {r.worst}")
        if r.top1_mismatches:
            print(
                f"{'':<16}  {r.top1_mismatches} position(s) where the argmax "
                f"disagrees with the reference"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "target" / "release" / "tinyinfer",
        help="path to the built tinyinfer binary",
    )
    parser.add_argument("--seed", type=int, default=1234, help="base RNG seed")
    parser.add_argument(
        "--config",
        action="append",
        help="run only this configuration (repeatable); default is all six",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="test against a real trained checkpoint instead of random weights",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=200,
        help="with --checkpoint, how many tokens to greedy-decode",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.exists():
        print(
            f"error: {binary} does not exist. Build it first:\n"
            f"  cargo build --release",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)

        if args.checkpoint:
            if not args.checkpoint.exists():
                print(f"error: no such checkpoint: {args.checkpoint}", file=sys.stderr)
                return 2
            print(f"real-weight mode: {args.checkpoint}")
            results = [run_real_checkpoint(binary, args.checkpoint, args.n, workdir)]
        else:
            selected = CONFIGS
            if args.config:
                wanted = set(args.config)
                unknown = wanted - {c.name for c in CONFIGS}
                if unknown:
                    print(
                        f"error: unknown config(s) {sorted(unknown)}; "
                        f"known: {[c.name for c in CONFIGS]}",
                        file=sys.stderr,
                    )
                    return 2
                selected = tuple(c for c in CONFIGS if c.name in wanted)

            print("random-weight mode")
            results = []
            for cfg in selected:
                print(f"  {cfg.name:<14} {cfg.describe()}", flush=True)
                r = run_random_config(cfg, binary, args.seed)
                print(
                    f"  {'':<14} max|err|={r.max_abs_err:.3e} "
                    f"max|ref|={r.max_abs_ref:.3f} err/tol={r.max_rel_to_tol:.3f} "
                    f"top-1={r.top1_agreed}/{r.n_positions}",
                    flush=True,
                )
                results.append(r)

        report(results)

    failed = [r for r in results if not r.ok]
    if failed:
        print(
            f"FAILED: {len(failed)}/{len(results)} configurations outside tolerance",
            file=sys.stderr,
        )
        return 1
    print(f"PASSED: {len(results)}/{len(results)} configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
