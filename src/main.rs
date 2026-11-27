//! Command-line interface.
//!
//! Two modes:
//!
//! ```text
//! proofinfer model.bin -z tokenizer.bin -i "Once upon a time" -n 256
//! proofinfer model.bin --tokens 1,450,2462 --dump-logits out.f32
//! ```
//!
//! The first is interactive greedy generation. The second is the mode the
//! differential harness uses: run an exact token sequence at positions `0..n`
//! and optionally write every step's logits as raw little-endian `f32`, one row
//! per position. The second mode deliberately does not load a tokenizer, so the
//! harness controls the tokenisation and there is no question about whether the
//! two implementations were fed the same thing.
//!
//! # Why a hand-written parser
//!
//! `clap` is the obvious choice and it is exactly what this crate does not
//! have. The argument grammar here is six flags, all of which take a value.
//! Writing the loop by hand costs about eighty lines and keeps `Cargo.lock` at
//! one package, which is the project's headline constraint. It also means an
//! unknown flag is a normal `Err` with a usage message rather than something
//! that has to be configured to be an error.

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use proofinfer::model::{self, SampleConfig, State, Weights};
use proofinfer::ops;
use proofinfer::tokenizer::Tokenizer;

const USAGE: &str = "\
proofinfer - a dependency-free Llama inference engine

USAGE:
    proofinfer <model.bin> -z <tokenizer.bin> -i <text> -n <count> [OPTIONS]
    proofinfer <model.bin> --tokens <t0,t1,...> [--dump-logits <out.f32>] [--threads <count>]

MODES:
    With -i/--input, the text is tokenised and up to <count> tokens are
    streamed to stdout. Decoding is greedy by default (--temp 0), matching
    run.c byte-for-byte. Tokens per second is reported on stderr.

    With --tokens, exactly those token ids are run at positions 0, 1, ... with
    no tokenisation and no sampling. This is the mode the differential test
    harness uses.

OPTIONS:
    -z, --tokenizer <path>    BPE vocabulary (tokenizer.bin)
    -i, --input <text>        prompt text
    -n, --num-tokens <count>  tokens to generate (default 256)
    -t, --threads <count>     worker threads for matmul (default 1)
        --temp <float>        sampling temperature; 0 = greedy (default 0.0)
        --top-p <float>       nucleus sampling cutoff in (0, 1] (default 0.9)
        --seed <u64>          PRNG seed when --temp > 0 (default 0)
        --tokens <list>       comma-separated token ids, run verbatim
        --dump-logits <path>  write every step's logits as raw f32, one row
                              per position, little-endian
    -h, --help                print this message
    -V, --version             print the version

EXAMPLES:
    proofinfer stories15M.bin -z tokenizer.bin -i \"Once upon a time\" -n 256
    proofinfer stories15M.bin -z tokenizer.bin -i \"Once upon a time\" -n 256 -t 4
    proofinfer stories15M.bin -z tokenizer.bin -i \"Once upon a time\" --temp 0.8 --top-p 0.9 --seed 42
    proofinfer tiny.bin --tokens 1,450,2462 --dump-logits out.f32
";

/// A command-line usage error. Not a model error: it means the invocation was
/// wrong, which is a different thing from the model being wrong.
#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage_err(msg: impl Into<String>) -> CliError {
    CliError(msg.into())
}

/// Parsed command line.
#[derive(Debug)]
struct Args {
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    input: Option<String>,
    num_tokens: usize,
    threads: usize,
    sample: SampleConfig,
    tokens: Option<Vec<u32>>,
    dump_logits: Option<PathBuf>,
}

/// What to do after parsing. `Help` and `Version` short-circuit `main`.
enum Action {
    Run(Box<Args>),
    Help,
    Version,
}

/// Parse `argv` (without the program name).
///
/// Both `--flag value` and `--flag=value` are accepted. Splitting on the first
/// `=` only for `--`-prefixed names keeps a positional path containing an `=`
/// (a legal character in a filename) from being mistaken for a flag.
fn parse_args(argv: &[String]) -> Result<Action, CliError> {
    let mut model: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut input: Option<String> = None;
    let mut num_tokens: Option<usize> = None;
    let mut threads: Option<usize> = None;
    let mut temp: Option<f32> = None;
    let mut top_p: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut tokens: Option<Vec<u32>> = None;
    let mut dump_logits: Option<PathBuf> = None;

    let mut it = argv.iter();
    while let Some(raw) = it.next() {
        // Own the strings so the borrow on `argv` ends and the iterator stays
        // free for the `take_value!` macro below.
        let raw = raw.clone();
        let (flag, inline): (String, Option<String>) = match raw.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (raw, None),
        };

        /// The value for a flag: the text after `=`, else the next argument.
        macro_rules! take_value {
            ($name:expr) => {
                match inline.clone() {
                    Some(v) => v,
                    None => it
                        .next()
                        .ok_or_else(|| usage_err(format!("{} needs a value", $name)))?
                        .clone(),
                }
            };
        }

        match flag.as_str() {
            "-h" | "--help" => return Ok(Action::Help),
            "-V" | "--version" => return Ok(Action::Version),
            "-z" | "--tokenizer" => tokenizer = Some(PathBuf::from(take_value!("-z"))),
            "-i" | "--input" => input = Some(take_value!("-i")),
            "-n" | "--num-tokens" => num_tokens = Some(parse_count(&flag, &take_value!("-n"))?),
            "-t" | "--threads" => threads = Some(parse_threads(&flag, &take_value!("-t"))?),
            "--temp" => temp = Some(parse_f32(&flag, &take_value!("--temp"))?),
            "--top-p" => top_p = Some(parse_f32(&flag, &take_value!("--top-p"))?),
            "--seed" => seed = Some(parse_u64(&flag, &take_value!("--seed"))?),
            "--tokens" => tokens = Some(parse_token_list(&take_value!("--tokens"))?),
            "--dump-logits" => dump_logits = Some(PathBuf::from(take_value!("--dump-logits"))),
            // A bare "-" is conventionally stdin; nothing here reads it, so it
            // is a positional that will fail the model-path check with a
            // sensible message rather than being silently ignored.
            f if f.starts_with('-') && f != "-" => {
                return Err(usage_err(format!("unknown option {f}")));
            }
            path => {
                if model.is_some() {
                    return Err(usage_err(format!(
                        "unexpected extra argument {path:?}: the model checkpoint path is \
                         the only positional argument"
                    )));
                }
                model = Some(PathBuf::from(path));
            }
        }
    }

    let model = model.ok_or_else(|| usage_err("missing the model checkpoint path"))?;
    let sample = SampleConfig {
        temperature: temp.unwrap_or(0.0),
        top_p: top_p.unwrap_or(0.9),
        seed: seed.unwrap_or(0),
    };
    if let Err(e) = sample.validate() {
        return Err(usage_err(e.to_string()));
    }
    let has_sampling_flag = temp.is_some() || top_p.is_some() || seed.is_some();

    let args = Args {
        model,
        tokenizer,
        input,
        num_tokens: num_tokens.unwrap_or(256),
        threads: threads.unwrap_or(1),
        sample,
        tokens,
        dump_logits,
    };

    // Mode consistency. `--tokens` and `-i` are two different ways of saying
    // what to feed the model, and silently preferring one would make a typo in
    // a harness script look like a numerical mismatch.
    match (&args.tokens, &args.input) {
        (Some(_), Some(_)) => {
            return Err(usage_err(
                "--tokens and -i/--input are mutually exclusive: --tokens already \
                 supplies the token ids",
            ))
        }
        (None, None) => {
            return Err(usage_err(
                "nothing to run: pass -i/--input with some text, or --tokens with \
                 a list of token ids",
            ))
        }
        _ => {}
    }

    if args.tokens.is_none() && args.tokenizer.is_none() {
        return Err(usage_err(
            "-z/--tokenizer is required when generating from text",
        ));
    }
    if args.dump_logits.is_some() && args.tokens.is_none() {
        return Err(usage_err(
            "--dump-logits only makes sense with --tokens: it records one row of \
             logits per input position, and generated tokens have no fixed \
             input sequence to record against",
        ));
    }
    if has_sampling_flag && args.tokens.is_some() {
        return Err(usage_err(
            "--temp, --top-p, and --seed only apply to text generation (-i/--input)",
        ));
    }

    Ok(Action::Run(Box::new(args)))
}

/// Parse a non-negative count, rejecting anything that is not a plain number.
fn parse_count(flag: &str, s: &str) -> Result<usize, CliError> {
    s.parse::<usize>()
        .map_err(|_| usage_err(format!("{flag} expects a non-negative integer, got {s:?}")))
}

/// Parse a positive thread count.
fn parse_threads(flag: &str, s: &str) -> Result<usize, CliError> {
    let n = s
        .parse::<usize>()
        .map_err(|_| usage_err(format!("{flag} expects a positive integer, got {s:?}")))?;
    if n == 0 {
        return Err(usage_err(format!("{flag} must be at least 1, got 0")));
    }
    Ok(n)
}

/// Parse a finite floating-point flag value.
fn parse_f32(flag: &str, s: &str) -> Result<f32, CliError> {
    let v = s
        .parse::<f32>()
        .map_err(|_| usage_err(format!("{flag} expects a number, got {s:?}")))?;
    if !v.is_finite() {
        return Err(usage_err(format!("{flag} must be finite, got {s:?}")));
    }
    Ok(v)
}

/// Parse a 64-bit unsigned integer flag value.
fn parse_u64(flag: &str, s: &str) -> Result<u64, CliError> {
    s.parse::<u64>()
        .map_err(|_| usage_err(format!("{flag} expects a non-negative integer, got {s:?}")))
}

/// Parse a comma-separated token id list such as `1,450,2462`.
///
/// Empty entries are an error rather than skipped. `"1,,2"` almost certainly
/// means a generated list had a hole in it, and silently dropping the hole
/// would shift every subsequent position.
fn parse_token_list(s: &str) -> Result<Vec<u32>, CliError> {
    if s.trim().is_empty() {
        return Err(usage_err("--tokens was given an empty list"));
    }
    s.split(',')
        .map(|part| {
            let part = part.trim();
            if part.is_empty() {
                return Err(usage_err(format!(
                    "--tokens contains an empty entry in {s:?}"
                )));
            }
            part.parse::<u32>()
                .map_err(|_| usage_err(format!("--tokens entry {part:?} is not a token id")))
        })
        .collect()
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("proofinfer: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Classify a stdout write/flush result.
///
/// `BrokenPipe` (for example piping into `head`) terminates output cleanly
/// with `Ok(false)` instead of panicking or printing a spurious error; any
/// other I/O error is propagated so `main` exits non-zero.
fn handle_stdout_io(res: io::Result<()>) -> Result<bool, Box<dyn Error>> {
    match res {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(Box::new(e)),
    }
}

fn run(argv: &[String]) -> Result<ExitCode, Box<dyn Error>> {
    let args = match parse_args(argv)? {
        Action::Help => {
            let mut out = io::stdout().lock();
            let _ = handle_stdout_io(out.write_all(USAGE.as_bytes()).and_then(|()| out.flush()))?;
            return Ok(ExitCode::SUCCESS);
        }
        Action::Version => {
            let mut out = io::stdout().lock();
            let _ = handle_stdout_io(
                writeln!(out, "proofinfer {}", env!("CARGO_PKG_VERSION"))
                    .and_then(|()| out.flush()),
            )?;
            return Ok(ExitCode::SUCCESS);
        }
        Action::Run(a) => *a,
    };

    ops::set_num_threads(args.threads);

    let weights = Weights::load(&args.model)?;
    let cfg = weights.config;
    eprintln!(
        "loaded {}: dim={} layers={} heads={} kv_heads={} vocab={} seq_len={}{}",
        args.model.display(),
        cfg.dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.vocab_size,
        cfg.seq_len,
        if weights.wcls.is_some() {
            " classifier=untied"
        } else {
            " classifier=tied"
        }
    );

    let mut state = State::new(&cfg)?;

    match &args.tokens {
        // --- exact-token mode ---------------------------------------------
        Some(tokens) => {
            if tokens.len() > cfg.seq_len {
                return Err(Box::new(model::RunError::TooManyTokens {
                    requested: tokens.len(),
                    seq_len: cfg.seq_len,
                }));
            }
            let mut dump = match &args.dump_logits {
                Some(path) => Some(BufWriter::new(create_file(path)?)),
                None => None,
            };

            let start = Instant::now();
            for (pos, &token) in tokens.iter().enumerate() {
                let logits = state.forward(&weights, token, pos)?;
                if let Some(w) = dump.as_mut() {
                    write_logits(w, logits)?;
                }
            }
            let elapsed = start.elapsed();

            if let Some(w) = dump.as_mut() {
                w.flush()?;
            }
            eprintln!(
                "ran {} tokens in {:.3}s ({:.1} tok/s)",
                tokens.len(),
                elapsed.as_secs_f64(),
                tokens.len() as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
            );
        }

        // --- generation mode ----------------------------------------------
        None => {
            let tokenizer_path = args
                .tokenizer
                .as_ref()
                .ok_or_else(|| usage_err("-z/--tokenizer is required when generating"))?;
            let tokenizer = Tokenizer::load(tokenizer_path)?;
            let text = args
                .input
                .as_deref()
                .ok_or_else(|| usage_err("-i/--input is required when generating"))?;

            // BOS is prepended because that is how the model was trained; the
            // tokenizer's dummy prefix then marks the following text as starting
            // a fresh word.
            let prompt = tokenizer.encode(text, true, false);
            if prompt.is_empty() {
                return Err(Box::new(model::RunError::EmptyPrompt));
            }
            eprintln!(
                "prompt: {} tokens, generating {}",
                prompt.len(),
                args.num_tokens
            );
            {
                let mut out = io::stdout().lock();
                if !handle_stdout_io(
                    print_tokens(&mut out, &tokenizer, &prompt, None).and_then(|()| out.flush()),
                )? {
                    return Ok(ExitCode::SUCCESS);
                }
            }

            let start = Instant::now();
            let mut prev = prompt.last().copied();
            let mut broken_pipe = false;
            let mut io_err: Option<io::Error> = None;
            let generated_count = {
                let mut out = io::stdout().lock();
                model::generate_sampled_stream(
                    &weights,
                    &mut state,
                    &prompt,
                    args.num_tokens,
                    args.sample,
                    |token| {
                        let bytes = tokenizer.decode_one_bytes(prev, token);
                        prev = Some(token);
                        match safe_print(&mut out, &bytes).and_then(|()| out.flush()) {
                            Ok(()) => true,
                            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                                broken_pipe = true;
                                false
                            }
                            Err(e) => {
                                io_err = Some(e);
                                false
                            }
                        }
                    },
                )?
            };
            let elapsed = start.elapsed();

            if let Some(e) = io_err {
                return Err(Box::new(e));
            }
            if broken_pipe {
                return Ok(ExitCode::SUCCESS);
            }
            {
                let mut out = io::stdout().lock();
                if !handle_stdout_io(out.write_all(b"\n").and_then(|()| out.flush()))? {
                    return Ok(ExitCode::SUCCESS);
                }
            }

            // Throughput counts the prompt as well as the generated tokens.
            //
            // The elapsed time covers every forward pass, including the ones
            // that consumed the prompt, so dividing by generated tokens alone
            // would charge the prompt's cost to nothing and flatter the number.
            // Counting `prompt.len() + generated_count` also matches llama2.c's
            // convention, which is what makes the two directly comparable in
            // `reference/bench.py`.
            let processed = prompt.len() + generated_count;
            let tps = processed as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
            eprintln!(
                "generated {} tokens in {:.3}s ({tps:.2} tok/s, {processed} tokens processed)",
                generated_count,
                elapsed.as_secs_f64()
            );
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Print a token sequence, one piece at a time, the way llama2.c does.
///
/// Decoding the whole sequence in one call and printing the result is simpler
/// but not equivalent. `run.c` prints *per token* through `safe_printf`, which
/// drops a piece that is a single byte which is neither printable nor
/// whitespace. Decoding in bulk would emit those bytes, and the output would
/// differ from the reference for reasons that have nothing to do with the
/// model. Going piece by piece is what makes the byte-identical comparison in
/// `reference/bench.py` a statement about the arithmetic.
///
/// `prev` is the token before this sequence, needed because the dummy-prefix
/// rule depends on it. Pass `None` for a sequence that starts at BOS.
fn print_tokens<W: Write>(
    out: &mut W,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    prev: Option<u32>,
) -> io::Result<()> {
    let mut previous = prev;
    for &token in tokens {
        let bytes = tokenizer.decode_one_bytes(previous, token);
        safe_print(out, &bytes)?;
        previous = Some(token);
    }
    Ok(())
}

/// Write one decoded raw-byte piece, skipping the ones llama2.c would skip.
///
/// A single non-printable, non-whitespace byte (including single non-ASCII
/// bytes `0x80..=0xFF` from byte-fallback tokens) is dropped before any UTF-8
/// conversion. The reasoning is the reference's: byte-fallback tokens can name
/// any byte at all, including control characters and half of a multi-byte UTF-8
/// sequence, and echoing those raw into a terminal is unpleasant and
/// occasionally terminal-escaping. Pieces of two bytes or more are written
/// verbatim even if they are not valid UTF-8, matching `run.c`'s `safe_printf`.
///
/// "Printable" is ASCII 0x20..=0x7E and "whitespace" is the C library's
/// `isspace` set. Both are spelled out rather than delegated to
/// `char::is_control` because the point is to match a specific C
/// implementation, not to have a defensible opinion about Unicode.
fn safe_print<W: Write>(out: &mut W, bytes: &[u8]) -> io::Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    if bytes.len() == 1 {
        let b = bytes[0];
        let printable = (0x20..=0x7e).contains(&b);
        let whitespace = matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
        if !(printable || whitespace) {
            return Ok(());
        }
    }
    out.write_all(bytes)
}

/// Create an output file, turning a path problem into a normal error.
fn create_file(path: &Path) -> Result<File, Box<dyn Error>> {
    File::create(path).map_err(|e| -> Box<dyn Error> {
        Box::new(CliError(format!("cannot create {}: {e}", path.display())))
    })
}

/// Write one position's logits as raw little-endian `f32`.
///
/// Serialised through a per-row byte buffer rather than four `write_all` calls
/// per element: a 32000-entry vocabulary at every position turns the naive form
/// into a syscall storm.
fn write_logits<W: Write>(w: &mut W, logits: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for &v in logits {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BrokenPipeWriter;

    impl Write for BrokenPipeWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"))
        }
    }

    #[test]
    fn safe_print_matches_run_c_on_single_and_multibyte_raw_bytes() {
        let mut out = Vec::new();
        // Printable ASCII and C whitespace pass through.
        safe_print(&mut out, b"A").unwrap();
        safe_print(&mut out, b"\n").unwrap();
        safe_print(&mut out, b"\t").unwrap();
        // Single control and non-ASCII bytes are dropped (not turned into U+FFFD).
        for b in [0x00u8, 0x07, 0x1b, 0x7f, 0x80, 0xc3, 0xff] {
            safe_print(&mut out, &[b]).unwrap();
        }
        // Multi-byte non-UTF-8 pieces are written verbatim.
        safe_print(&mut out, &[0xff, 0xfe]).unwrap();

        assert_eq!(out, b"A\n\t\xff\xfe");
    }

    #[test]
    fn broken_pipe_stops_output_without_error() {
        let mut w = BrokenPipeWriter;
        let res = safe_print(&mut w, b"hello");
        assert!(!handle_stdout_io(res).unwrap());
    }

    #[test]
    fn parse_args_accepts_threads_and_sampling_and_rejects_invalid_combinations() {
        let argv = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| (*s).to_string()).collect() };

        let action = parse_args(&argv(&[
            "m.bin",
            "-z",
            "t.bin",
            "-i",
            "hi",
            "-t",
            "4",
            "--temp",
            "0.8",
            "--top-p",
            "0.95",
            "--seed",
            "42",
        ]))
        .unwrap();
        match action {
            Action::Run(a) => {
                assert_eq!(a.threads, 4);
                assert_eq!(a.sample.temperature, 0.8);
                assert_eq!(a.sample.top_p, 0.95);
                assert_eq!(a.sample.seed, 42);
            }
            _ => panic!("expected Action::Run"),
        }

        // Zero threads is rejected.
        assert!(parse_args(&argv(&["m.bin", "-z", "t.bin", "-i", "hi", "-t", "0"])).is_err());
        // Negative temperature and out-of-range top-p are rejected.
        assert!(parse_args(&argv(&[
            "m.bin", "-z", "t.bin", "-i", "hi", "--temp", "-0.1"
        ]))
        .is_err());
        assert!(parse_args(&argv(&[
            "m.bin", "-z", "t.bin", "-i", "hi", "--top-p", "0.0"
        ]))
        .is_err());
        assert!(parse_args(&argv(&[
            "m.bin", "-z", "t.bin", "-i", "hi", "--top-p", "1.5"
        ]))
        .is_err());
        // Sampling flags cannot be combined with --tokens.
        assert!(parse_args(&argv(&["m.bin", "--tokens", "1,2", "--temp", "0.5"])).is_err());
    }
}
