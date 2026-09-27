//! Command-line interface.
//!
//! Two modes:
//!
//! ```text
//! tinyinfer model.bin -z tokenizer.bin -i "Once upon a time" -n 256
//! tinyinfer model.bin --tokens 1,450,2462 --dump-logits out.f32
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
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use tinyinfer::model::{self, State, Weights};
use tinyinfer::tokenizer::Tokenizer;

const USAGE: &str = "\
tinyinfer - a dependency-free Llama inference engine

USAGE:
    tinyinfer <model.bin> -z <tokenizer.bin> -i <text> -n <count>
    tinyinfer <model.bin> --tokens <t0,t1,...> [--dump-logits <out.f32>]

MODES:
    With -i/--input, the text is tokenised and <count> tokens are generated
    greedily. Tokens per second is reported on stderr.

    With --tokens, exactly those token ids are run at positions 0, 1, ... with
    no tokenisation and no sampling. This is the mode the differential test
    harness uses.

OPTIONS:
    -z, --tokenizer <path>    BPE vocabulary (tokenizer.bin)
    -i, --input <text>        prompt text
    -n, --num-tokens <count>  tokens to generate (default 256)
        --tokens <list>       comma-separated token ids, run verbatim
        --dump-logits <path>  write every step's logits as raw f32, one row
                              per position, little-endian
    -h, --help                print this message
    -V, --version             print the version

EXAMPLES:
    tinyinfer stories15M.bin -z tokenizer.bin -i \"Once upon a time\" -n 256
    tinyinfer tiny.bin --tokens 1,450,2462 --dump-logits out.f32
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
    let args = Args {
        model,
        tokenizer,
        input,
        num_tokens: num_tokens.unwrap_or(256),
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

    Ok(Action::Run(Box::new(args)))
}

/// Parse a non-negative count, rejecting anything that is not a plain number.
fn parse_count(flag: &str, s: &str) -> Result<usize, CliError> {
    s.parse::<usize>()
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
            eprintln!("tinyinfer: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: &[String]) -> Result<ExitCode, Box<dyn Error>> {
    let args = match parse_args(argv)? {
        Action::Help => {
            print!("{USAGE}");
            return Ok(ExitCode::SUCCESS);
        }
        Action::Version => {
            println!("tinyinfer {}", env!("CARGO_PKG_VERSION"));
            return Ok(ExitCode::SUCCESS);
        }
        Action::Run(a) => *a,
    };

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
            print!("{}", tokenizer.decode_text(&prompt));
            let _ = std::io::stdout().flush();

            let start = Instant::now();
            let generated = model::generate(&weights, &mut state, &prompt, args.num_tokens)?;
            let elapsed = start.elapsed();

            // Streamed token by token in a real implementation; printed in one
            // go here because the output is the deliverable rather than a
            // progress display.
            print!("{}", tokenizer.decode_text(&generated));
            println!();
            let _ = std::io::stdout().flush();

            let tps = generated.len() as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
            eprintln!(
                "generated {} tokens in {:.3}s ({tps:.2} tok/s)",
                generated.len(),
                elapsed.as_secs_f64()
            );
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// Create an output file, turning a path problem into a normal error.
fn create_file(path: &Path) -> Result<File, Box<dyn Error>> {
    File::create(path).map_err(|e| -> Box<dyn Error> {
        Box::new(CliError(format!("cannot create {}: {e}", path.display())))
    })
}

/// Write one position's logits as raw little-endian `f32`.
///
/// Serialised through a reusable byte buffer rather than four `write_all` calls
/// per element: a 32000-entry vocabulary at every position turns the naive form
/// into a syscall storm. The buffer is allocated once per row and refilled, so
/// the cost is one memcpy per element and one write per row.
fn write_logits<W: Write>(w: &mut W, logits: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for &v in logits {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_all(&bytes)?;
    Ok(())
}
