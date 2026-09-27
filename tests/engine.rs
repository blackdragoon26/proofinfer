//! Forward-pass and generation tests.
//!
//! These need no reference implementation, so they are cheap and can check
//! *properties* rather than specific values:
//!
//! * **Causality** is the big one. The logits computed at position `p` must not
//!   change when more tokens are fed afterwards. That is a defining property of
//!   a causal language model, and it is the property that a KV cache could
//!   plausibly get wrong - a cache that leaked a stale row, or an attention
//!   window that was too wide, would still produce plausible output but would
//!   fail this. It is checked here rather than left to the differential test,
//!   because the differential test can only catch it on the configurations it
//!   happens to run.
//!
//! * **Determinism**: greedy generation is a pure function of the weights and
//!   the prompt. Two runs must produce identical token ids, because the
//!   byte-identical comparison against llama2.c's `run.c` depends on it.

use tinyinfer::model::{argmax, generate, Config, State, Weights};

/// Build a small but well-formed legacy-format checkpoint.
///
/// Values come from a cheap deterministic pattern with a per-tensor phase
/// offset, so tensors are distinguishable from one another and a mix-up shows
/// up as a wrong value rather than a plausible one.
fn build_checkpoint(
    dim: i32,
    hidden: i32,
    layers: i32,
    heads: i32,
    kv_heads: i32,
    vocab: i32,
    seq: i32,
) -> Vec<u8> {
    let head_size = (dim / heads) as usize;
    let kv_dim = head_size * kv_heads as usize;
    let (d, h, l, v, s) = (
        dim as usize,
        hidden as usize,
        layers as usize,
        vocab.unsigned_abs() as usize,
        seq as usize,
    );

    let mut total = v * d // tok_embedding
        + l * d // rms_att
        + l * d * d // wq
        + l * kv_dim * d // wk
        + l * kv_dim * d // wv
        + l * d * d // wo
        + l * d // rms_ffn
        + l * h * d // w1
        + l * d * h // w2
        + l * h * d // w3
        + d // rms_final
        + 2 * s * (head_size / 2); // freq_cis, both tables
    if vocab < 0 {
        total += v * d; // wcls
    }

    let mut bytes = Vec::with_capacity(28 + total * 4);
    for f in [dim, hidden, layers, heads, kv_heads, vocab, seq] {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    // A sawtooth in [-1, 1]. Bounded on purpose: a checkpoint with huge
    // activations overflows f32 inside softmax and every assertion here would
    // become a test of overflow behaviour instead.
    for i in 0..total {
        let v = ((i % 41) as f32 / 20.0) - 1.0;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn tiny() -> Weights {
    Weights::from_bytes(&build_checkpoint(64, 128, 2, 4, 4, 48, 16)).expect("valid checkpoint")
}

fn gqa() -> Weights {
    Weights::from_bytes(&build_checkpoint(64, 96, 3, 8, 2, 48, 16)).expect("valid checkpoint")
}

fn untied() -> Weights {
    Weights::from_bytes(&build_checkpoint(64, 128, 2, 4, 4, -48, 16)).expect("valid checkpoint")
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

#[test]
fn forward_returns_one_logit_per_token() {
    let w = tiny();
    let mut s = State::new(&w.config).unwrap();
    let logits = s.forward(&w, 7, 0).unwrap();
    assert_eq!(logits.len(), w.config.vocab_size);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
}

#[test]
fn forward_rejects_a_token_outside_the_vocabulary() {
    let w = tiny();
    let mut s = State::new(&w.config).unwrap();
    let err = s.forward(&w, w.config.vocab_size as u32, 0).unwrap_err();
    assert!(matches!(
        err,
        tinyinfer::model::RunError::TokenOutOfRange { .. }
    ));
    // i32::MAX is a token id a fuzzer would happily produce.
    assert!(s.forward(&w, u32::MAX, 0).is_err());
    // The last valid id must still work.
    assert!(s.forward(&w, w.config.vocab_size as u32 - 1, 0).is_ok());
}

#[test]
fn forward_rejects_a_position_past_the_kv_cache() {
    let w = tiny();
    let mut s = State::new(&w.config).unwrap();
    let last = w.config.seq_len - 1;
    assert!(s.forward(&w, 0, last).is_ok());
    assert!(s.forward(&w, 0, last + 1).is_err());
    assert!(s.forward(&w, 0, usize::MAX).is_err());
}

#[test]
fn logits_do_not_depend_on_later_tokens() {
    // The causality property. Feed a short prefix, record the logits at every
    // position, then feed a longer sequence in the same State and re-read the
    // early positions. Nothing about a causal model may change.
    let w = gqa();

    // `long` is a sequence; `short` is its first five tokens. The causality
    // claim is that decoding only the prefix gives the same logits at those
    // positions as decoding all of it.
    let long: Vec<u32> = (0..12).map(|i| (i * 5 + 1) as u32 % 48).collect();
    let short = &long[..5];

    let mut prefix_state = State::new(&w.config).unwrap();
    let short_logits: Vec<Vec<f32>> = short
        .iter()
        .enumerate()
        .map(|(pos, &t)| prefix_state.forward(&w, t, pos).unwrap().to_vec())
        .collect();

    let mut full_state = State::new(&w.config).unwrap();
    let long_logits: Vec<Vec<f32>> = long
        .iter()
        .enumerate()
        .map(|(pos, &t)| full_state.forward(&w, t, pos).unwrap().to_vec())
        .collect();

    for (pos, (earlier, later)) in short_logits.iter().zip(long_logits.iter()).enumerate() {
        for (i, (a, b)) in earlier.iter().zip(later.iter()).enumerate() {
            assert_eq!(
                a, b,
                "logit {i} at position {pos} changed when later tokens were fed"
            );
        }
    }
}

#[test]
fn forward_is_deterministic_across_identical_runs() {
    let w = untied();
    let tokens: Vec<u32> = (0..8).map(|i| (i * 3) as u32 % 48).collect();

    let run_once = || {
        let mut s = State::new(&w.config).unwrap();
        let mut out = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            out.push(s.forward(&w, t, pos).unwrap().to_vec());
        }
        out
    };
    assert_eq!(run_once(), run_once());
}

#[test]
fn a_fresh_state_and_a_reset_state_agree() {
    // `reset` exists so a `State` can be reused. If it did not fully clear the
    // cache, the second sequence would be decoded against the first one's keys
    // and values.
    let w = tiny();
    let tokens: Vec<u32> = (0..6).map(|i| (i * 11) as u32 % 48).collect();

    let mut reused = State::new(&w.config).unwrap();
    for (pos, &t) in tokens.iter().enumerate() {
        reused.forward(&w, t, pos).unwrap();
    }
    let first = reused.logits.clone();

    reused.reset();
    for (pos, &t) in tokens.iter().enumerate() {
        reused.forward(&w, t, pos).unwrap();
    }
    let after_reset = reused.logits.clone();

    let mut fresh = State::new(&w.config).unwrap();
    for (pos, &t) in tokens.iter().enumerate() {
        fresh.forward(&w, t, pos).unwrap();
    }
    let virgin = fresh.logits.clone();

    assert_eq!(
        first, virgin,
        "sanity: a fresh state must reproduce the first run"
    );
    assert_eq!(
        after_reset, virgin,
        "a reset state must behave exactly like a fresh one"
    );
}

#[test]
fn untied_and_tied_models_produce_different_logits() {
    // If the untied `wcls` were silently ignored, the untied model would
    // compute the same logits as a tied one. This is mutant 8 in the mutation
    // check; the same guard belongs here as a cheap unit-level check.
    let tied = tiny();
    let untied = untied();
    assert!(tied.wcls.is_none());
    assert!(untied.wcls.is_some());

    let mut a = State::new(&tied.config).unwrap();
    let mut b = State::new(&untied.config).unwrap();
    let la = a.forward(&tied, 5, 0).unwrap().to_vec();
    let lb = b.forward(&untied, 5, 0).unwrap().to_vec();
    assert_ne!(la, lb, "the untied classifier must change the logits");
}

// ---------------------------------------------------------------------------
// Termination
// ---------------------------------------------------------------------------

/// Build a checkpoint engineered so that the very first greedy step emits BOS.
///
/// Every weight matrix is zero, so each layer contributes nothing and the
/// residual stream stays equal to the input token's embedding row. The final
/// rmsnorm is all ones, and the classifier is tied to the embedding, so the
/// logits are `embedding[v] . x_final` — the argmax is whichever embedding row
/// has the largest first component. That is the knob: token 0's row gets 1.0 to
/// drive the residual stream, BOS's row gets 2.0 to win the argmax, and every
/// other row stays zero.
fn bos_emitting_checkpoint() -> Weights {
    const DIM: usize = 4;
    const VOCAB: usize = 8;
    const LAYERS: usize = 1;
    const HEADS: usize = 2; // head_size 2, even
    const SEQ: usize = 8;
    const BOS: usize = 1;

    let head_size = DIM / HEADS;
    let kv_dim = head_size * HEADS;

    // Walk the tensors in file order with a running cursor rather than writing
    // out a hand-computed offset. The first version of this fixture did the
    // arithmetic by hand, got it wrong, and the test failed for a reason that
    // had nothing to do with what it was testing. A cursor cannot drift.
    let mut f: Vec<f32> = Vec::new();
    let take = |n: usize, f: &mut Vec<f32>| {
        let at = f.len();
        f.resize(at + n, 0.0);
        at
    };

    let emb = take(VOCAB * DIM, &mut f);
    take(LAYERS * DIM, &mut f); // rms_att
    take(LAYERS * DIM * DIM, &mut f); // wq
    take(LAYERS * kv_dim * DIM, &mut f); // wk
    take(LAYERS * kv_dim * DIM, &mut f); // wv
    take(LAYERS * DIM * DIM, &mut f); // wo
    take(LAYERS * DIM, &mut f); // rms_ffn
    take(LAYERS * DIM * DIM, &mut f); // w1 (hidden = dim)
    take(LAYERS * DIM * DIM, &mut f); // w2
    take(LAYERS * DIM * DIM, &mut f); // w3
    let rms_final = take(DIM, &mut f);
    take(2 * SEQ * (head_size / 2), &mut f); // freq_cis, both tables

    for v in f[rms_final..rms_final + DIM].iter_mut() {
        *v = 1.0;
    }
    // Control surface. Because `logits[v] = E[input] . E[v]`, the argmax is the
    // vocabulary row most similar to the *input* row, so putting the control
    // rows on different coordinates is what makes them independently
    // reachable. Three rows on one axis would not work: the argmax over
    // E[:, 0] depends only on the sign of x[0], so every positive input picks
    // the same winner.
    //
    //   E[0]   = (1,0,0,0)  ->  argmax over E[:,0] is BOS  (2.0)     : terminates
    //   E[BOS] = (2,0,0,0)
    //   E[2]   = (0,1,0,0)  ->  argmax over E[:,1] is token 2 (1.0)   : loops
    f[emb] = 1.0;
    f[emb + BOS * DIM] = 2.0;
    f[emb + 2 * DIM + 1] = 1.0;

    let mut bytes = Vec::with_capacity(28 + f.len() * 4);
    for field in [
        DIM as i32,
        DIM as i32, // hidden
        LAYERS as i32,
        HEADS as i32,
        HEADS as i32, // kv_heads
        VOCAB as i32,
        SEQ as i32,
    ] {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    for v in &f {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Weights::from_bytes(&bytes).expect("engineered checkpoint must load")
}

#[test]
fn generate_stops_on_bos_because_bos_delimits_documents() {
    // llama2.c's rule is `if (next == 1) break;` - BOS, not EOS - because BOS
    // is the document delimiter these models are trained with. Stopping on EOS
    // instead means never stopping on a BOS-delimited checkpoint: our text ran
    // straight past the end of the first story and diverged from `run.c` by
    // 112 bytes.
    //
    // The byte-identity check in reference/bench.py is what caught that, and the
    // differential harness cannot see it at all because it never looks at text.
    // A mutation sweep over this file found the same hole, so the rule is
    // pinned here too.
    let w = bos_emitting_checkpoint();
    let prompt = vec![0u32];

    // Sanity: the fixture really does predict BOS after token 0. Without this
    // the test could pass for the wrong reason, e.g. if `generate` always
    // returned nothing.
    let mut probe = State::new(&w.config).unwrap();
    let logits = probe.forward(&w, 0, 0).unwrap();
    assert_eq!(
        argmax(logits).0 as u32,
        tinyinfer::tokenizer::BOS_ID,
        "fixture is wrong: the first step must predict BOS"
    );

    let gen = generate(&w, &mut State::new(&w.config).unwrap(), &prompt, 5).unwrap();
    assert!(
        gen.is_empty(),
        "generation must stop immediately on BOS, got {gen:?}"
    );
}

#[test]
fn generate_keeps_going_when_the_next_token_is_not_bos() {
    // The mirror of the test above, so the first one cannot pass for the wrong
    // reason. Feeding token 2 makes token 2 the argmax, which is not a
    // terminator, so all five requested tokens are produced.
    let w = bos_emitting_checkpoint();

    let mut probe = State::new(&w.config).unwrap();
    let logits = probe.forward(&w, 2, 0).unwrap();
    assert_ne!(
        argmax(logits).0 as u32,
        tinyinfer::tokenizer::BOS_ID,
        "fixture is wrong: token 2 must not be a terminator"
    );

    let gen = generate(&w, &mut State::new(&w.config).unwrap(), &[2], 5).unwrap();
    assert_eq!(
        gen.len(),
        5,
        "nothing terminates, so all 5 tokens are produced"
    );
    assert!(
        gen.iter().all(|&t| t == 2),
        "expected a fixed point on token 2, got {gen:?}"
    );
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

#[test]
fn argmax_finds_the_largest_element() {
    assert_eq!(argmax(&[1.0, 5.0, 3.0]).0, 1);
    assert_eq!(argmax(&[-1.0, -5.0, -3.0]).0, 0);
    assert_eq!(argmax(&[7.0]).0, 0);
    assert_eq!(argmax(&[0.0; 8]).0, 0);
    // Ties resolve to the first, which keeps generation deterministic.
    assert_eq!(argmax(&[2.0, 2.0, 2.0]).0, 0);
}

#[test]
fn argmax_is_total_on_nan_and_never_out_of_bounds() {
    // A NaN must not cause a skip that returns a finite element, and must not
    // panic. Both properties matter for a debug-vs-release consistent answer.
    let (i, _) = argmax(&[1.0, f32::NAN, 3.0]);
    assert!(i < 3);

    let (i, _) = argmax(&[f32::NAN, 1.0]);
    assert!(i < 2);

    // All -inf is a degenerate but legal input.
    let (i, v) = argmax(&[f32::NEG_INFINITY; 3]);
    assert_eq!(i, 0);
    assert!(v.is_infinite());
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

#[test]
fn generate_returns_the_requested_number_of_tokens() {
    let w = tiny();
    let prompt = vec![1u32, 10, 20, 30];
    // A fresh State per count: `generate` consumes positions, so reusing one
    // would run the second case against a cache already filled by the first.
    for n in [0usize, 1, 5, 10] {
        let mut s = State::new(&w.config).unwrap();
        let gen = generate(&w, &mut s, &prompt, n).unwrap();
        assert_eq!(gen.len(), n, "asked for {n} tokens");
        assert!(gen.iter().all(|&t| (t as usize) < w.config.vocab_size));
    }
}

#[test]
fn generate_is_deterministic() {
    let w = gqa();
    let prompt = vec![1u32, 5, 9, 13];
    let once = generate(&w, &mut State::new(&w.config).unwrap(), &prompt, 12).unwrap();
    let twice = generate(&w, &mut State::new(&w.config).unwrap(), &prompt, 12).unwrap();
    assert_eq!(once, twice, "greedy decoding must be a pure function");
}

#[test]
fn generate_continues_the_argmax_of_the_last_prompt_position() {
    // The first generated token must be the argmax of the logits produced by
    // the final prompt token. This pins the forward-pass accounting: getting it
    // wrong would shift the whole continuation by one position.
    let w = tiny();
    let prompt = vec![1u32, 4, 9, 16];

    let mut probe = State::new(&w.config).unwrap();
    for (pos, &t) in prompt.iter().enumerate() {
        probe.forward(&w, t, pos).unwrap();
    }
    let expected_first = argmax(probe.logits.as_slice()).0 as u32;

    let mut s = State::new(&w.config).unwrap();
    let gen = generate(&w, &mut s, &prompt, 5).unwrap();
    assert_eq!(gen[0], expected_first);
}

#[test]
fn generate_rejects_a_prompt_longer_than_the_context() {
    let w = tiny();
    let too_long: Vec<u32> = (0..w.config.seq_len).map(|i| i as u32).collect();
    let err = generate(&w, &mut State::new(&w.config).unwrap(), &too_long, 1).unwrap_err();
    assert!(matches!(
        err,
        tinyinfer::model::RunError::TooManyTokens { .. }
    ));
    // Exactly filling the context is allowed.
    assert!(generate(&w, &mut State::new(&w.config).unwrap(), &too_long, 0).is_ok());
}

#[test]
fn generate_rejects_an_empty_prompt() {
    let w = tiny();
    let mut s = State::new(&w.config).unwrap();
    assert!(generate(&w, &mut s, &[], 5).is_err());
}

// ---------------------------------------------------------------------------
// State sizing
// ---------------------------------------------------------------------------

#[test]
fn state_scratch_matches_the_config() {
    let w = gqa();
    let s = State::new(&w.config).unwrap();
    let dims = w.config.attention_dims().unwrap();

    assert_eq!(s.x.len(), w.config.dim);
    assert_eq!(s.logits.len(), w.config.vocab_size);
    // The cache is the dominant allocation and is exactly
    // n_layers * seq_len * kv_dim per cache.
    assert_eq!(
        s.cache_len(),
        w.config.n_layers * w.config.seq_len * dims.kv_dim()
    );
}

#[test]
fn state_rejects_a_config_whose_cache_would_overflow() {
    // Structurally legal, and legal in the way that matters: `head_size` must
    // be *even* so the config gets past `AttentionDims::try_new` and actually
    // reaches the cache arithmetic. The first version of this test used
    // `i32::MAX`, which is odd, so `State::new` rejected the odd head size and
    // returned `InvalidHeader` - the test passed without ever exercising the
    // overflow it is named after. A mutation sweep caught that.
    let cfg = Config {
        dim: i32::MAX as usize - 1, // even, so head_size is even
        hidden_dim: 1,
        n_layers: i32::MAX as usize,
        n_heads: 1,
        n_kv_heads: 1,
        vocab_size: 1,
        seq_len: i32::MAX as usize,
    };
    // Sanity: the shape really is acceptable, so the only thing left that can
    // fail is the arithmetic.
    assert!(
        cfg.attention_dims().is_ok(),
        "fixture is wrong: the config must be structurally valid"
    );

    // Assert the *specific* error, not just that there is one. Otherwise a
    // future change that rejects this config for an unrelated reason would
    // keep the test green while the overflow path went untested.
    match State::new(&cfg) {
        Err(tinyinfer::model::LoadError::Overflow { what }) => {
            assert_eq!(what, "kv cache");
        }
        Err(other) => panic!("expected an overflow error, got {other:?}"),
        Ok(_) => panic!("an unrepresentable cache must not be accepted"),
    }
}
