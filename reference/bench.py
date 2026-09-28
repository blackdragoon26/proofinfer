#!/usr/bin/env python3
"""Benchmark proofinfer against llama2.c's C engine, and check they agree.

Two things happen here, and the second is the more important one:

1. **Throughput.** Each engine greedily decodes the same prompt on the same
   legacy-format checkpoint, three times, and the median is reported. Median
   rather than mean because a single scheduler hiccup on a shared machine
   should not move the number.

2. **Byte-identical greedy output.** Greedy decoding is a pure function of the
   weights and the prompt: at every step both engines take the argmax, so if
   they agree on the maths they produce the same token ids, and therefore the
   same text. This is a much stronger claim than "the samples look similar",
   and it is a genuinely independent check because `run.c` is a third
   implementation with its own matmul, its own softmax and its own RoPE.

The comparison strips a single trailing newline from both sides. proofinfer
prints one so the shell prompt starts on a fresh line; `run.c` does not. Nothing
else is normalised - if the two engines disagree by so much as one token, this
fails.

Build the C reference first:

    cc -O3 -march=native -o /tmp/run_O3   reference/llama2c/run.c
    cc -Ofast              -o /tmp/run_Ofast reference/llama2c/run.c

Usage:
    python3 reference/bench.py --model /tmp/stories15M.bin
    python3 reference/bench.py --model /tmp/stories15M.bin --runs 5
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOKENIZER = REPO / "reference" / "llama2c" / "tokenizer.bin"

# run.c reports throughput on stderr as "achieved tok/s: <float>".
TPS_RE = re.compile(r"achieved tok/s:\s*([0-9.]+)")
# proofinfer reports it as "generated N tokens in T s (X tok/s, M tokens processed)".
TINY_TPS_RE = re.compile(r"\(([0-9.]+) tok/s")
# proofinfer reports the encoded prompt length on stderr.
PROMPT_TOKENS_RE = re.compile(r"prompt:\s*(\d+)\s*tokens")


@dataclasses.dataclass
class Engine:
    name: str
    cmd: list[str]
    pattern: re.Pattern[str]

    def run(self) -> tuple[float, bytes, str]:
        proc = subprocess.run(self.cmd, capture_output=True, text=False)
        stderr = proc.stderr.decode(errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.name} exited {proc.returncode}\n  stderr: {stderr.strip()}"
            )
        matches = self.pattern.findall(stderr)
        if not matches:
            raise RuntimeError(
                f"{self.name} did not report a throughput figure.\n"
                f"  stderr: {stderr.strip()}"
            )
        tps = float(matches[-1])
        # Strip exactly one trailing newline from each side so the CLI's
        # convenience newline is not counted as a text difference.
        return tps, proc.stdout.rstrip(b"\n"), stderr


def bench(engine: Engine, runs: int) -> tuple[list[float], bytes]:
    rates: list[float] = []
    output: bytes | None = None
    for _ in range(runs):
        tps, out, _stderr = engine.run()
        rates.append(tps)
        if output is None:
            output = out
        elif output != out:
            raise RuntimeError(
                f"{engine.name} produced different output on two runs of the "
                f"same input; greedy decoding is supposed to be deterministic"
            )
    assert output is not None
    return rates, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="legacy-format .bin")
    parser.add_argument("--binary", type=Path, default=REPO / "target/release/proofinfer")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument(
        "--n", type=int, default=0, help="tokens to generate; 0 means fill the context"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--run-c", type=Path, action="append", default=[])
    args = parser.parse_args()

    model = args.model.resolve()
    binary = args.binary.resolve()
    for path in (model, binary, TOKENIZER):
        if not path.exists():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2

    # The context length is in the checkpoint header; reading it keeps the
    # benchmark from asking for a run that cannot fit.
    import struct

    with model.open("rb") as f:
        header = struct.unpack("iiiiiii", f.read(28))
    seq_len = header[6]
    n = args.n if args.n else seq_len - 8
    if n <= 0 or n > seq_len:
        print(
            f"error: cannot generate {n} tokens into a context of {seq_len}",
            file=sys.stderr,
        )
        return 2

    print(f"model   {model.name}")
    print(f"prompt  {args.prompt!r}")
    print(f"decode  {n} generated greedy tokens, {args.runs} runs each, single threaded")
    print(f"cpu     {cpu_name()}")
    print()

    tiny_cmd = [
        str(binary),
        str(model),
        "-z",
        str(TOKENIZER),
        "-i",
        args.prompt,
        "-n",
        str(n),
    ]
    tiny = Engine("proofinfer", tiny_cmd, TINY_TPS_RE)

    # run.c's `-n` counts *total* forward passes from position 0, prompt
    # included: its loop is `while (pos < steps)` with pos starting at 0. Ours
    # counts generated tokens only. Giving both the same number would make one
    # of them stop early, and the outputs would then differ by length for a
    # reason that has nothing to do with the maths - which is exactly the kind
    # of false failure that trains you to ignore the check.
    #
    # So: learn the prompt length from proofinfer's own stderr, then ask run.c
    # for `prompt + n` total steps. Both then generate exactly `n` tokens.
    probe_tps, probe_out, probe_err = tiny.run()
    m = PROMPT_TOKENS_RE.search(probe_err)
    if not m:
        raise RuntimeError(
            f"could not read the encoded prompt length from proofinfer's stderr; "
            f"got:\n{probe_err.strip()}"
        )
    prompt_tokens = int(m.group(1))
    c_steps = prompt_tokens + n
    if c_steps > seq_len:
        raise RuntimeError(
            f"prompt ({prompt_tokens}) + {n} generated = {c_steps} exceeds the "
            f"context ({seq_len}); lower --n"
        )
    print(
        f"prompt is {prompt_tokens} tokens, so run.c is given -n {c_steps} "
        f"(its budget counts the prompt too)"
    )
    print()

    engines = [tiny]
    for c in args.run_c:
        p = Path(c)
        if not p.exists():
            print(f"error: {c} does not exist", file=sys.stderr)
            return 2
        engines.append(
            Engine(
                p.name,
                # -t 0.0 is what makes run.c greedy. Its default is 1.0, i.e.
                # sampling, which would be incomparable by construction.
                [
                    str(p),
                    str(model),
                    "-z",
                    str(TOKENIZER),
                    "-i",
                    args.prompt,
                    "-n",
                    str(c_steps),
                    "-t",
                    "0.0",
                ],
                TPS_RE,
            )
        )

    results: list[tuple[Engine, list[float], bytes]] = []
    for e in engines:
        if e is tiny:
            # The probe run above already measured it; reuse it as run 1 so the
            # engine is not needlessly run an extra time.
            rates, out = [probe_tps], probe_out
            for _ in range(args.runs - 1):
                tps, o, _e = e.run()
                rates.append(tps)
                if o != out:
                    raise RuntimeError(f"{e.name} was not deterministic across runs")
        else:
            rates, out = bench(e, args.runs)
        results.append((e, rates, out))
        print(
            f"  {e.name:<14} {statistics.median(rates):7.1f} tok/s  "
            f"(runs: {', '.join(f'{r:.1f}' for r in rates)})"
        )

    base_engine, _, base_out = results[0]
    print()
    print(f"byte-identical greedy output vs {base_engine.name}:")
    identical = True
    for e, _, out in results[1:]:
        same = out == base_out
        identical = identical and same
        print(f"  {e.name:<14} {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            print(f"    {base_engine.name}: {base_out[:200]!r}")
            print(f"    {e.name:<14}: {out[:200]!r}")

    print()
    if not identical:
        print("FAILED: greedy output diverges from the C reference", file=sys.stderr)
        return 1
    return 0


def cpu_name() -> str:
    """Best-effort CPU string. Never load-bearing, only for the report."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                return out.stdout.strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
