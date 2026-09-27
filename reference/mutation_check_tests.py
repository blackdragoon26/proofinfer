#!/usr/bin/env python3
"""Mutation sweep over the Rust test suite.

`reference/mutation_check.py` proves `diff_test.py` can fail. Nothing proved
the other way round: that the 72 `cargo test` cases are load-bearing rather than
decorative. A test suite that cannot fail is a claim of confidence with nothing
behind it, which is the exact thing this repository argues against.

So: inject one realistic bug at a time into `src/`, run `cargo test`, and
require a failure. These mutants are deliberately *different* from the ones in
`mutation_check.py` - those target the arithmetic, these target the code paths
the unit and integration tests are supposed to guard (loader totality, tokenizer
conformance, engine causality), including several that the differential harness
would not notice at all because they only affect error paths or decoding
bookkeeping.

Usage:  python3 reference/mutation_check_tests.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OPS = "src/ops.rs"
TOK = "src/tokenizer.rs"
MODEL = "src/model.rs"


@dataclass
class Mutant:
    name: str
    path: str
    old: str
    new: str
    targets: str


MUTANTS = [
    # ---- ops.rs: kernels -------------------------------------------------
    Mutant("rmsnorm-eps-zeroed", OPS,
           "pub const RMS_NORM_EPS: f32 = 1e-5;",
           "pub const RMS_NORM_EPS: f32 = 0.0;",
           "ops: rmsnorm of all-zero input"),
    Mutant("softmax-no-max-shift", OPS,
           "    shift_in_place(x, max);",
           "    // mutant: no max shift",
           "ops: softmax stability on huge logits"),
    Mutant("softmax-wrong-denominator", OPS,
           "    let inv = 1.0 / sum;",
           "    let inv = 1.0 / (sum + 1.0);",
           "ops: softmax sums to 1"),
    Mutant("rope-exponent-half", OPS,
           "let exponent = (2 * i) as f32 / head_size as f32;",
           "let exponent = (i) as f32 / head_size as f32;",
           "ops: rope identity / pair norms / relative property"),
    Mutant("rope-not-identity-at-zero", OPS,
           "            let (s, c) = angle.sin_cos();",
           "            let (s, c) = (angle.sin_cos().0 * 1.0, angle.sin_cos().1); if pos == 0 { let _ = (s, c); }",
           "ops: rope is the identity at pos 0"),
    Mutant("attention-scale-dropped", OPS,
           "    let scale = 1.0 / (head_size as f32).sqrt();",
           "    let scale = 1.0f32;",
           "ops: hand-computed attention case"),
    Mutant("attention-window-off-by-one", OPS,
           "    let limit = pos + 1;",
           "    let limit = pos.max(1);",
           "ops: convex-combination window; engine causality"),
    Mutant("gqa-modulo", OPS,
           "        let kv = h / kv_mul;",
           "        let kv = h % dims.n_kv_heads;",
           "ops: GQA head mapping"),
    Mutant("silu-sign-flipped", OPS,
           "    x / (1.0 + (-x).exp())",
           "    x / (1.0 + x.exp())",
           "ops: silu matches x * sigmoid(x)"),
    Mutant("dot-one-lane", OPS,
           "const DOT_LANES: usize = 8;",
           "const DOT_LANES: usize = 1;",
           "ops: dot correctness across lengths (result still right, lane count not)"),
    # ---- model.rs: loader ------------------------------------------------
    Mutant("loader-allows-trailing-bytes", MODEL,
           "        if r.remaining() != 0 {",
           "        if false {",
           "loader: trailing bytes are an error"),
    Mutant("loader-checked-mul-removed", MODEL,
           "            .ok_or(LoadError::Overflow { what: \"kv cache\" })?;",
           "            .unwrap_or(0);",
           "loader: State::new rejects an unrepresentable cache"),
    Mutant("loader-unsigned-abs-to-abs", MODEL,
           "        let vocab = h_vocab.unsigned_abs() as usize;",
           "        let vocab = h_vocab.abs() as usize;",
           "loader: i32::MIN vocab does not panic (abs() overflows in debug)"),
    Mutant("loader-skips-divisibility", OPS,
           "        if dim % n_heads != 0 {",
           "        if false {",
           "loader: dim % n_heads is rejected"),
    Mutant("loader-accepts-zero-layers", MODEL,
           "            || h_layers <= 0",
           "            || false",
           "loader: zero fields are rejected"),
    Mutant("generate-stops-on-eos", MODEL,
           "        if next == crate::tokenizer::BOS_ID {",
           "        if next == crate::tokenizer::EOS_ID {",
           "engine: termination matches llama2.c (BOS, not EOS)"),
    # ---- tokenizer.rs ----------------------------------------------------
    Mutant("tok-no-dummy-prefix", TOK,
           "            tokens.push(self.id_of(b\" \").expect(\"dummy prefix checked at load time\"));",
           "            // mutant: dummy prefix omitted",
           "tokenizer: dummy prefix, Meta vectors, round trip"),
    Mutant("tok-byte-offset-wrong", TOK,
           "pub const BYTE_ID_OFFSET: u32 = 3;",
           "pub const BYTE_ID_OFFSET: u32 = 4;",
           "tokenizer: byte-fallback ids and the French prompt"),
    Mutant("tok-merge-keeps-first", TOK,
           "                    if score > best_score {",
           "                    if score < best_score {",
           "tokenizer: greedy highest-score merge"),
    Mutant("tok-bos-strip-removed", TOK,
           "            } else if prev == Some(BOS_ID) && piece.first() == Some(&b' ') {",
           "            } else if false {",
           "tokenizer: leading-space strip after BOS, round trip"),
    Mutant("tok-eats-trailing-bytes", TOK,
           "        if pos != data.len() {",
           "        if false {",
           "tokenizer: trailing bytes are an error"),
]


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="rust-mutants-"))
    crate = work / "tinyinfer"
    shutil.copytree(REPO / "src", crate / "src")
    shutil.copytree(REPO / "tests", crate / "tests")
    for f in ("Cargo.toml", "Cargo.lock", "reference"):
        src = REPO / f
        if src.is_dir():
            shutil.copytree(src, crate / f)
        elif src.exists():
            shutil.copy2(src, crate / f)
    # The tokenizer tests need the vendored vocabulary.
    (crate / "reference" / "llama2c").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "reference/llama2c/tokenizer.bin",
                 crate / "reference/llama2c/tokenizer.bin")

    # Inherit the caller's environment so cargo and the rustup toolchain
    # resolve, and share one target dir so the 22 builds reuse each other's
    # artifacts instead of recompiling the crate 22 times.
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(work / "target")
    env.pop("RUSTFLAGS", None)

    def cargo(*args):
        return subprocess.run(["cargo", *args], cwd=crate, env=env,
                              capture_output=True, text=True)

    # Baseline: the unmutated source must pass, or "N/N caught" means nothing.
    b = cargo("test", "--offline")
    if b.returncode != 0:
        print("BASELINE FAILED - cargo test does not pass on the clean tree")
        print(b.stderr[-3000:])
        return 1
    # A baseline that "passes" while running zero tests would make every mutant
    # look caught, because a build failure and an empty run are the same event.
    # Assert the count is non-zero rather than trusting the exit code alone.
    counts = [int(m) for m in re.findall(r"test result: ok\. (\d+) passed", b.stdout)]
    total = sum(counts)
    if total == 0:
        print("BASELINE RAN NO TESTS - 'everything caught' would be meaningless")
        print(b.stdout[-3000:])
        return 1
    print(f"baseline: cargo test passes, {total} tests across {len(counts)} binaries")
    print(f"mutation sweep over the Rust test suite: {len(MUTANTS)} mutants")
    print()

    caught, survived = [], []
    for m in MUTANTS:
        path = crate / m.path
        original = path.read_text()
        if original.count(m.old) != 1:
            survived.append((m.name, f"snippet occurs {original.count(m.old)}x, not once"))
            print(f"  SKIP  {m.name}: snippet is not unique")
            continue
        t0 = time.time()
        path.write_text(original.replace(m.old, m.new, 1))
        r = cargo("test", "--offline")
        path.write_text(original)

        dt = time.time() - t0
        if r.returncode == 0:
            survived.append((m.name, "cargo test still passed"))
            print(f"  SURVIVED  {m.name}  ({dt:.1f}s)  <- {m.targets}")
        else:
            failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED",
                                           r.stdout, re.M)))
            caught.append(m.name)
            print(f"  caught    {m.name:<34} ({dt:.1f}s)  {m.targets}")
            if failed:
                print(f"              {len(failed)} test(s) failed, e.g. {failed[0][:60]}")

    print()
    print(f"caught {len(caught)}/{len(MUTANTS)}")
    if survived:
        print()
        print("SURVIVORS - the suite has a blind spot:")
        for n, why in survived:
            print(f"  {n}: {why}")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
