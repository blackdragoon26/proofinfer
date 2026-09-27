#!/usr/bin/env python3
"""Prove the differential harness can actually fail.

A test that has never failed is not evidence. `diff_test.py` passing on its
first run says nothing on its own - it is equally consistent with "the engine is
correct" and "the test cannot detect anything". This script resolves that
ambiguity by mutation testing: inject one realistic bug at a time into a copy of
the Rust source, rebuild, and require `diff_test.py` to exit non-zero.

If a mutant survives, the honest conclusion is not "that bug is too subtle" but
"the suite has a blind spot", and the fix is to the suite. Every mutant here is
a mistake someone could plausibly make while reading the code.

Two details make the result trustworthy rather than decorative:

* **Each snippet must occur exactly once.** If a refactor duplicated or renamed
  a line, `str.replace` would silently do nothing (or the wrong thing), the
  mutant would compile to identical behaviour, and the harness would correctly
  report a pass - which the naive reading of "the test passed" would score as
  "the mutant survived". Asserting the count turns that silent no-op into a
  loud failure, so a refactor cannot quietly disarm the whole check.
* **The unmutated source is checked first.** The baseline must pass. Otherwise
  "10/10 caught" could mean "10/10 failed for an unrelated reason".

Speed: every mutant shares one `CARGO_TARGET_DIR`, and LTO is disabled via
`CARGO_PROFILE_RELEASE_LTO=false`. LTO is what makes each build slow, and it has
no effect on which mutants the harness catches. Sharing the target directory
means dependencies and the standard library are compiled once for the whole run.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Only the files the mutations touch, plus what they need to build.
COPY = ("Cargo.toml", "Cargo.lock", "src")


@dataclasses.dataclass(frozen=True)
class Mutant:
    name: str
    path: str  # repo-relative
    old: str
    new: str
    why: str


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        name="rope-exponent",
        path="src/ops.rs",
        old="let exponent = (2 * i) as f32 / head_size as f32;",
        new="let exponent = (i) as f32 / head_size as f32;",
        why="RoPE pairs are (2i, 2i+1); using i instead of 2i halves every "
        "exponent, which changes all the rotation angles but keeps the code "
        "looking perfectly reasonable",
    ),
    Mutant(
        name="rope-skip-k",
        path="src/model.rs",
        old="            ops::rope(&mut self.k, dims.n_kv_heads, dims.head_size, pos);",
        new="            // MUTANT: RoPE intentionally not applied to k",
        why="forgetting the key rotation. Every value gets position "
        "information and every key does not, so queries match keys from the "
        "wrong positions",
    ),
    Mutant(
        name="gqa-modulo-mapping",
        path="src/ops.rs",
        old="        let kv = h / kv_mul;",
        new="        let kv = h % dims.n_kv_heads;",
        why="GQA head mapping by modulo instead of division. Both are valid "
        "ways to spread heads over KV heads, and both are self-consistent, so "
        "this is the single most dangerous class of GQA bug",
    ),
    Mutant(
        name="attention-window-off-by-one",
        path="src/ops.rs",
        old="    let limit = pos + 1;",
        new="    let limit = pos.max(1);",
        why="dropping the current token from its own attention. At pos 0 this "
        "reads uninitialised cache; at pos >= 1 it silently attends to one "
        "token too few",
    ),
    Mutant(
        name="no-attention-scale",
        path="src/ops.rs",
        old="    let scale = 1.0 / (head_size as f32).sqrt();",
        new="    let scale = 1.0f32;",
        why="dropping the 1/sqrt(head_size) factor. Standard dot-product "
        "attention practice; without it the logits grow with head size and "
        "softmax saturates",
    ),
    Mutant(
        name="rmsnorm-eps",
        path="src/ops.rs",
        old="pub const RMS_NORM_EPS: f32 = 1e-5;",
        new="pub const RMS_NORM_EPS: f32 = 1e-27;",
        why="an eps of 1e-27 underflows to a no-op next to any real mean "
        "square, and 1e-27 is a plausible typo for 1e-5",
    ),
    Mutant(
        name="swiglu-gate-swapped",
        path="src/model.rs",
        old="                self.hb[i] = ops::silu_value(self.hb[i]) * self.hb2[i];",
        new="                self.hb[i] = self.hb[i] * ops::silu_value(self.hb2[i]);",
        why="applying the nonlinearity to the up-projection instead of the "
        "gate. The two are the same shape, so nothing about the code makes the "
        "swap obvious",
    ),
    Mutant(
        name="ignore-untied-classifier",
        path="src/model.rs",
        old="            w.classifier(),",
        new="            &w.tok_embedding,",
        why="ignoring wcls and always using the token embedding. Invisible for "
        "every tied model, and the untied configuration exists precisely to "
        "catch it",
    ),
    Mutant(
        name="residual-overwrite",
        path="src/model.rs",
        old=(
            "            ops::matmul(&mut self.xb, &self.att, w.wo_layer(l), d, q_dim);\n"
            "            for i in 0..d {\n"
            "                self.x[i] += self.xb[i];\n"
            "            }"
        ),
        new=(
            "            ops::matmul(&mut self.xb, &self.att, w.wo_layer(l), d, q_dim);\n"
            "            for i in 0..d {\n"
            "                self.x[i] = self.xb[i];\n"
            "            }"
        ),
        why="assigning instead of accumulating the attention residual, which "
        "discards everything the earlier layers computed. A multi-layer model "
        "still produces output, just the wrong output",
    ),
    Mutant(
        name="rope-theta",
        path="src/ops.rs",
        old="pub const ROPE_THETA: f32 = 10000.0;",
        new="pub const ROPE_THETA: f32 = 500000.0;",
        why="a different RoPE base. The angles are all wrong but the code is "
        "still a textbook rotation, and nothing crashes",
    ),
)


@dataclasses.dataclass
class Outcome:
    name: str
    caught: bool
    detail: str
    seconds: float


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def engine_binary(target: Path) -> Path:
    """Where `cargo build` puts the binary.

    Not `crate/target/...`: CARGO_TARGET_DIR is redirected to one shared
    directory so the ten builds reuse each other's artifacts. Getting this wrong
    points at a stale or absent binary and quietly turns the whole check into a
    no-op, so it is computed in one place.
    """
    return target / "release" / "tinyinfer"


def check_baseline(crate: Path, target: Path, env: dict[str, str]) -> tuple[bool, str]:
    """Build and test the unmutated source. It has to pass first."""
    build = run(["cargo", "build", "--release"], crate, env)
    if build.returncode != 0:
        return False, f"baseline build failed:\n{build.stderr.strip()}"
    diff = run(
        [
            sys.executable,
            str(REPO / "reference" / "diff_test.py"),
            "--binary",
            str(engine_binary(target)),
        ],
        REPO,
        env,
    )
    if diff.returncode != 0:
        return False, f"baseline diff_test failed:\n{diff.stdout[-2000:]}"
    return True, "unmutated source passes"


def check_mutant(
    m: Mutant, crate: Path, target: Path, env: dict[str, str]
) -> Outcome:
    started = time.time()
    path = crate / m.path
    original = path.read_text()

    # The guard that makes this check meaningful. Without it, a refactor that
    # moved or duplicated a line would turn a mutant into a no-op, the mutant
    # would behave identically to the baseline, and diff_test would exit 0 -
    # indistinguishable, in the output, from "the bug slipped through".
    count = original.count(m.old)
    if count != 1:
        return Outcome(
            m.name,
            False,
            f"snippet occurs {count} times in {m.path}, expected exactly 1 "
            f"(the mutation is not well defined; fix the mutant or the source)",
            time.time() - started,
        )
    if m.new in original:
        return Outcome(
            m.name,
            False,
            f"replacement text already present in {m.path}; the mutant would be "
            f"a no-op",
            time.time() - started,
        )

    path.write_text(original.replace(m.old, m.new, 1))
    try:
        build = run(["cargo", "build", "--release"], crate, env)
        if build.returncode != 0:
            # A mutant that does not compile is not a *silent* bug, so it does
            # not prove anything about the harness. Report it as not-caught
            # rather than quietly counting it.
            return Outcome(
                m.name,
                False,
                f"mutant did not compile, so the harness never saw it:\n"
                f"{build.stderr.strip()[-800:]}",
                time.time() - started,
            )

        binary = engine_binary(target)
        diff = run(
            [
                sys.executable,
                str(REPO / "reference" / "diff_test.py"),
                "--binary",
                str(binary),
            ],
            REPO,
            env,
        )
        if diff.returncode == 0:
            return Outcome(
                m.name,
                False,
                "SURVIVED: diff_test passed with the bug injected, so the "
                "harness has a blind spot here",
                time.time() - started,
            )

        # Report which configurations noticed, because "it failed" is much less
        # informative than "the GQA configs noticed and nothing else did".
        failed_lines = [
            line.strip()
            for line in diff.stdout.splitlines()
            if line.strip() and "FAIL" in line
        ]
        detail = "caught"
        if failed_lines:
            detail += " by " + "; ".join(f.split()[0] for f in failed_lines[:4])
        return Outcome(m.name, True, detail, time.time() - started)
    finally:
        path.write_text(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary crate directories for inspection",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="run only these mutants (repeatable)",
    )
    args = parser.parse_args()

    selected = MUTANTS
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {m.name for m in MUTANTS}
        if unknown:
            print(
                f"error: unknown mutant(s) {sorted(unknown)}; "
                f"known: {[m.name for m in MUTANTS]}",
                file=sys.stderr,
            )
            return 2
        selected = tuple(m for m in MUTANTS if m.name in wanted)

    workdir = Path(tempfile.mkdtemp(prefix="tinyinfer-mutants-"))
    crate = workdir / "tinyinfer"
    crate.mkdir()
    for item in COPY:
        src = REPO / item
        if not src.exists():
            # Cargo.lock is not in the repo's first commit; cargo regenerates it.
            continue
        dst = crate / item
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # One shared target directory across every mutant: the standard library and
    # the crate's own dependencies (there are none) are built once. LTO off,
    # because LTO is the slowest part of the build and has no bearing on
    # whether a mutant is caught.
    target = workdir / "target"
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target)
    env["CARGO_PROFILE_RELEASE_LTO"] = "false"
    env.pop("RUSTFLAGS", None)

    try:
        print(f"mutation check: {len(selected)} mutants")
        print()

        ok, detail = check_baseline(crate, target, env)
        if not ok:
            print("BASELINE FAILED - the suite must pass before mutants mean anything")
            print(detail)
            return 1
        print(f"  baseline    OK  {detail}")
        print()

        outcomes: list[Outcome] = []
        for m in selected:
            print(f"  {m.name:<28} {m.why}", flush=True)
            print(f"  {'':<28} {m.path}", flush=True)
            outcome = check_mutant(m, crate, target, env)
            outcomes.append(outcome)
            mark = "caught  " if outcome.caught else "SURVIVED"
            print(
                f"  {'':<28} -> {mark}  {outcome.detail}  ({outcome.seconds:.1f}s)",
                flush=True,
            )
            print(flush=True)

        caught = [o for o in outcomes if o.caught]
        survived = [o for o in outcomes if not o.caught]

        print(f"{'=' * 72}")
        for o in outcomes:
            print(f"  {o.name:<28} {'caught' if o.caught else 'SURVIVED'}")
        print(f"{'=' * 72}")
        print(f"caught {len(caught)}/{len(outcomes)}")

        if survived:
            print(
                f"\nFAILED: {len(survived)} mutant(s) survived. The differential "
                f"harness has a blind spot; the fix is to the harness, not to the "
                f"mutants.",
                file=sys.stderr,
            )
            for o in survived:
                print(f"  - {o.name}: {o.detail}", file=sys.stderr)
            return 1

        return 0
    finally:
        if args.keep:
            print(f"\nkept workdir: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
