#!/usr/bin/env python3
"""Mutation sweep over the Rust test suite.

`reference/mutation_check.py` proves `diff_test.py` can fail. This script proves
the other half: that `cargo test` notices real defects rather than passing
decoratively.

Each mutant runs through a strict two-phase gate on a shared `CARGO_TARGET_DIR`:
1. `cargo test --offline --locked --no-run` MUST exit 0. A mutant that fails to
   compile is classified `COMPILE_ERROR` and fails the sweep (never counted as
   caught).
2. `cargo test --offline --locked` MUST exit > 0 and print at least one
   `test <name> ... FAILED` line. Exit 0 is `SURVIVED`, negative exit is
   `CRASHED_SIGNAL`, timeout is `TIMEOUT`, and non-zero with zero failed tests
   is `ZERO_TESTS_FAILED`.

Usage:  python3 reference/mutation_check_tests.py
"""

from __future__ import annotations

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

MUTANT_TIMEOUT_SECS = 60


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    targets: str


MUTANTS: tuple[Mutant, ...] = (
    # ---- ops.rs: kernels -------------------------------------------------
    Mutant(
        "rmsnorm-eps-zeroed",
        OPS,
        "pub const RMS_NORM_EPS: f32 = 1e-5;",
        "pub const RMS_NORM_EPS: f32 = 0.0;",
        "ops: rmsnorm of all-zero input",
    ),
    Mutant(
        "softmax-no-max-shift",
        OPS,
        "    shift_in_place(x, max);",
        "    // mutant: no max shift",
        "ops: softmax stability on huge logits",
    ),
    Mutant(
        "softmax-wrong-denominator",
        OPS,
        "    let inv = 1.0 / sum;",
        "    let inv = 1.0 / (sum + 1.0);",
        "ops: softmax sums to 1",
    ),
    Mutant(
        "rope-exponent-half",
        OPS,
        "let exponent = (2 * i) as f32 / head_size as f32;",
        "let exponent = (i) as f32 / head_size as f32;",
        "ops: rope frequency schedule",
    ),
    Mutant(
        "rope-not-identity-at-zero",
        OPS,
        "    let pos = pos as f32;",
        "    let pos = (pos + 1) as f32;",
        "ops: rope is the identity at pos 0",
    ),
    Mutant(
        "attention-scale-dropped",
        OPS,
        "    let scale = 1.0 / (head_size as f32).sqrt();",
        "    let scale = 1.0f32;",
        "ops: hand-computed attention case",
    ),
    Mutant(
        "attention-window-off-by-one",
        OPS,
        "    let limit = pos + 1;",
        "    let limit = pos.max(1);",
        "ops: convex-combination window; engine causality",
    ),
    Mutant(
        "gqa-modulo",
        OPS,
        "        let kv = h / kv_mul;",
        "        let kv = h % dims.n_kv_heads;",
        "ops: GQA head mapping",
    ),
    Mutant(
        "silu-sign-flipped",
        OPS,
        "    x / (1.0 + (-x).exp())",
        "    x / (1.0 + x.exp())",
        "ops: silu matches x * sigmoid(x)",
    ),
    Mutant(
        "dot-drops-last-lane",
        OPS,
        "    sum += acc[7];",
        "    // mutant: lane 7 dropped",
        "ops: dot exact powers-of-two and lane-boundary sweep",
    ),
    # ---- model.rs: loader & engine ---------------------------------------
    Mutant(
        "loader-allows-trailing-bytes",
        MODEL,
        "        if r.remaining() != 0 {",
        "        if false {",
        "loader: trailing bytes are an error",
    ),
    Mutant(
        "loader-checked-mul-removed",
        MODEL,
        '            .ok_or(LoadError::Overflow { what: "kv cache" })?;',
        "            .unwrap_or(0);",
        "loader: State::new rejects an unrepresentable cache",
    ),
    Mutant(
        "loader-unsigned-abs-to-abs",
        MODEL,
        "        let vocab = h_vocab.unsigned_abs() as usize;",
        "        let vocab = h_vocab.abs() as usize;",
        "loader: i32::MIN vocab does not panic (abs() overflows in debug)",
    ),
    Mutant(
        "loader-skips-divisibility",
        OPS,
        "        if dim % n_heads != 0 {",
        "        if false {",
        "loader: dim % n_heads is rejected",
    ),
    Mutant(
        "loader-accepts-zero-layers",
        MODEL,
        "            || h_layers <= 0",
        "            || false",
        "loader: zero fields are rejected",
    ),
    Mutant(
        "generate-stops-on-eos",
        MODEL,
        "        if next == crate::tokenizer::BOS_ID {",
        "        if next == crate::tokenizer::EOS_ID {",
        "engine: termination matches llama2.c (BOS, not EOS)",
    ),
    # ---- tokenizer.rs ----------------------------------------------------
    Mutant(
        "tok-no-dummy-prefix",
        TOK,
        '            tokens.push(self.id_of(b" ").expect("dummy prefix checked at load time"));',
        "            // mutant: dummy prefix omitted",
        "tokenizer: dummy prefix, Meta vectors, round trip",
    ),
    Mutant(
        "tok-byte-offset-wrong",
        TOK,
        "pub const BYTE_ID_OFFSET: u32 = 3;",
        "pub const BYTE_ID_OFFSET: u32 = 4;",
        "tokenizer: byte-fallback ids and the French prompt",
    ),
    Mutant(
        "tok-merge-keeps-first",
        TOK,
        "                    if score > best_score {",
        "                    if score < best_score {",
        "tokenizer: greedy highest-score merge",
    ),
    Mutant(
        "tok-bos-strip-removed",
        TOK,
        "            } else if prev == Some(BOS_ID) && piece.first() == Some(&b' ') {",
        "            } else if false {",
        "tokenizer: leading-space strip after BOS, round trip",
    ),
    Mutant(
        "tok-eats-trailing-bytes",
        TOK,
        "        if pos != data.len() {",
        "        if false {",
        "tokenizer: trailing bytes are an error",
    ),
)


class MtimeBumper:
    """Guarantee Cargo never skips a rebuild due to coarse filesystem mtimes."""

    def __init__(self) -> None:
        self._last = time.time()

    def write(self, path: Path, content: str) -> None:
        path.write_text(content)
        now = max(time.time(), self._last + 1.1)
        os.utime(path, (now, now))
        self._last = now


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="rust-mutants-"))
    try:
        crate = work / "proofinfer"
        shutil.copytree(REPO / "src", crate / "src")
        shutil.copytree(REPO / "tests", crate / "tests")
        for f in ("Cargo.toml", "Cargo.lock"):
            shutil.copy2(REPO / f, crate / f)
        # Copy only the vendored vocabulary required by tokenizer tests.
        (crate / "reference" / "llama2c").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO / "reference" / "llama2c" / "tokenizer.bin",
            crate / "reference" / "llama2c" / "tokenizer.bin",
        )

        env = dict(os.environ)
        env["CARGO_TARGET_DIR"] = str(work / "target")
        env.pop("RUSTFLAGS", None)

        def cargo(*args: str, timeout: int = MUTANT_TIMEOUT_SECS) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["cargo", *args],
                cwd=crate,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        # Baseline: the unmutated source must pass, or "N/N caught" means nothing.
        b = cargo("test", "--offline", "--locked", timeout=180)
        if b.returncode != 0:
            print("BASELINE FAILED - cargo test does not pass on the clean tree")
            print(b.stderr[-3000:])
            return 1
        counts = [int(m) for m in re.findall(r"test result: ok\. (\d+) passed", b.stdout)]
        total = sum(counts)
        if total == 0:
            print("BASELINE RAN NO TESTS - 'everything caught' would be meaningless")
            print(b.stdout[-3000:])
            return 1
        print(f"baseline: cargo test passes, {total} tests across {len(counts)} binaries")
        print(f"mutation sweep over the Rust test suite: {len(MUTANTS)} mutants")
        print()

        bumper = MtimeBumper()
        caught: list[str] = []
        failed_mutants: list[tuple[str, str, str]] = []

        for m in MUTANTS:
            path = crate / m.path
            original = path.read_text()
            if m.new == m.old:
                failed_mutants.append((m.name, "NO_OP", "replacement is identical to original"))
                print(f"  NO_OP            {m.name}: m.new == m.old")
                continue
            count = original.count(m.old)
            if count != 1:
                failed_mutants.append((m.name, "BAD_SNIPPET", f"snippet occurs {count}x, expected 1"))
                print(f"  BAD_SNIPPET      {m.name}: snippet occurs {count}x, not once")
                continue

            t0 = time.time()
            bumper.write(path, original.replace(m.old, m.new, 1))
            try:
                # Phase 1: mutant MUST compile cleanly.
                try:
                    build = cargo("test", "--offline", "--locked", "--no-run")
                except subprocess.TimeoutExpired:
                    dt = time.time() - t0
                    failed_mutants.append((m.name, "TIMEOUT", "compile timed out"))
                    print(f"  TIMEOUT          {m.name:<34} ({dt:.1f}s)  compile")
                    continue

                if build.returncode != 0:
                    dt = time.time() - t0
                    detail = build.stderr.strip()[-600:]
                    failed_mutants.append((m.name, "COMPILE_ERROR", detail))
                    print(f"  COMPILE_ERROR    {m.name:<34} ({dt:.1f}s)  mutant did not compile")
                    continue

                # Phase 2: tests MUST execute and fail at least one test assertion.
                try:
                    r = cargo("test", "--offline", "--locked")
                except subprocess.TimeoutExpired:
                    dt = time.time() - t0
                    failed_mutants.append((m.name, "TIMEOUT", "test execution timed out"))
                    print(f"  TIMEOUT          {m.name:<34} ({dt:.1f}s)  test run")
                    continue
            finally:
                bumper.write(path, original)

            dt = time.time() - t0
            if r.returncode == 0:
                failed_mutants.append((m.name, "SURVIVED", "cargo test still passed"))
                print(f"  SURVIVED         {m.name:<34} ({dt:.1f}s)  <- {m.targets}")
            elif r.returncode < 0:
                failed_mutants.append((m.name, "CRASHED_SIGNAL", f"terminated by signal {-r.returncode}"))
                print(f"  CRASHED_SIGNAL   {m.name:<34} ({dt:.1f}s)  signal {-r.returncode}")
            else:
                failed = sorted(set(re.findall(r"^test (\S+) \.\.\. FAILED", r.stdout, re.M)))
                if not failed:
                    failed_mutants.append(
                        (m.name, "ZERO_TESTS_FAILED", f"exited {r.returncode} without any FAILED test line")
                    )
                    print(f"  ZERO_TESTS_FAILED {m.name:<33} ({dt:.1f}s)  exit {r.returncode}")
                else:
                    caught.append(m.name)
                    print(f"  CAUGHT           {m.name:<34} ({dt:.1f}s)  {m.targets}")
                    print(f"                     {len(failed)} test(s) failed, e.g. {failed[0][:60]}")

        print()
        print(f"caught {len(caught)}/{len(MUTANTS)}")
        if failed_mutants:
            print()
            print("PROBLEMS - sweep did not achieve clean 100% caught:")
            for name, status, why in failed_mutants:
                print(f"  [{status}] {name}: {why}")
            return 1
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
