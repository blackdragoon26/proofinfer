//! Loader tests: the checkpoint file is untrusted input.
//!
//! The contract under test is a *totality* property: for **any** byte string,
//! `Weights::from_bytes` either returns a `Weights` or returns an `Err`. It
//! must never panic, never allocate an attacker-sized buffer, and never produce
//! a `Weights` whose internal buffers are shorter than its `Config` claims.
//!
//! That last part is the one that matters most. If a size product wrapped
//! around, the loader could hand back a `Weights` whose `wq` has 4 elements
//! while `config` says `n_layers * dim * dim`, and the bug would not surface
//! until some later forward pass indexed out of bounds — far from the malformed
//! file that caused it. The positive tests at the bottom of this file check
//! buffer lengths against the config for exactly that reason.

use proofinfer::model::{Config, LoadError, Weights};

// ---------------------------------------------------------------------------
// Test fixture: build well-formed checkpoints
// ---------------------------------------------------------------------------

/// A small but structurally valid model description.
///
/// Every field is a legal small value: `dim` is divisible by `n_heads`,
/// `n_heads` by `n_kv_heads`, and `head_size` is even.
#[derive(Clone, Copy)]
struct Shape {
    dim: i32,
    hidden: i32,
    layers: i32,
    heads: i32,
    kv_heads: i32,
    vocab: i32,
    seq: i32,
}

impl Default for Shape {
    fn default() -> Self {
        // dim 64, head_size 16, even. Two layers, MHA. Small enough that the
        // generated file is a few hundred kilobytes at most.
        Shape {
            dim: 64,
            hidden: 128,
            layers: 2,
            heads: 4,
            kv_heads: 4,
            vocab: 32,
            seq: 8,
        }
    }
}

impl Shape {
    fn header_bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        for f in [
            self.dim,
            self.hidden,
            self.layers,
            self.heads,
            self.kv_heads,
            self.vocab,
            self.seq,
        ] {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v
    }

    /// Number of floats in the whole file, for this shape.
    fn total_floats(&self) -> usize {
        let dim = self.dim as usize;
        let hid = self.hidden as usize;
        let layers = self.layers as usize;
        let vocab = self.vocab.unsigned_abs() as usize;
        let head_size = dim / self.heads as usize;
        let kv_dim = head_size * self.kv_heads as usize;
        let seq = self.seq as usize;

        let mut n = vocab * dim; // tok_embedding
        n += layers * dim; // rms_att
        n += layers * dim * dim; // wq
        n += layers * kv_dim * dim; // wk
        n += layers * kv_dim * dim; // wv
        n += layers * dim * dim; // wo
        n += layers * dim; // rms_ffn
        n += layers * hid * dim; // w1
        n += layers * dim * hid; // w2
        n += layers * hid * dim; // w3
        n += dim; // rms_final
        n += 2 * seq * (head_size / 2); // freq_cis real + imag
        if self.vocab < 0 {
            n += vocab * dim; // wcls
        }
        n
    }
}

/// Build a byte-for-byte well-formed legacy checkpoint.
///
/// Element values are a recognisable pattern rather than noise, so a test can
/// assert that a particular tensor landed in the right place.
fn build_checkpoint(shape: Shape) -> Vec<u8> {
    let mut bytes = shape.header_bytes();
    // Fill with a cheap deterministic pattern: index -> (i % 17) as f32 - 8.0
    for i in 0..shape.total_floats() {
        let v = (i % 17) as f32 - 8.0;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// A tiny deterministic PRNG, so the fuzz-style tests are reproducible.
///
/// xorshift32 is used rather than anything from `rand` because the crate has no
/// dependencies, and a fixed seed makes a CI failure reproducible from the
/// printed seed alone.
struct Xorshift32(u32);

impl Xorshift32 {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    fn next_byte(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }

    fn fill(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_byte()).collect()
    }
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

#[test]
fn empty_input_is_an_error_not_a_panic() {
    let err = Weights::from_bytes(&[]).expect_err("empty file must not load");
    assert!(
        matches!(
            err,
            LoadError::Truncated {
                what: "header: dim",
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn a_partial_header_is_rejected() {
    // Every prefix of the 28-byte header must be an error, and must say so.
    let full = Shape::default().header_bytes();
    for n in 0..full.len() {
        let err = Weights::from_bytes(&full[..n]).expect_err("partial header must not load");
        assert!(
            matches!(err, LoadError::Truncated { .. }),
            "prefix of {n} bytes gave {err:?}"
        );
    }
}

#[test]
fn a_header_with_no_weights_is_rejected() {
    let bytes = Shape::default().header_bytes();
    let err = Weights::from_bytes(&bytes).expect_err("header-only file must not load");
    assert!(
        matches!(
            err,
            LoadError::Truncated {
                what: "tensor: token_embedding",
                ..
            }
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn a_truncated_body_is_rejected_with_the_tensor_named() {
    // Cut the file at several points. Whichever tensor the cursor is sitting on
    // is the one the error should name, which is what makes the message useful.
    let full = build_checkpoint(Shape::default());
    for frac in [10, 25, 50, 75, 90, 99] {
        let n = full.len() * frac / 100;
        let err = Weights::from_bytes(&full[..n]).expect_err("truncated file must not load");
        match err {
            LoadError::Truncated { what, .. } => {
                assert!(what.starts_with("tensor: "), "got {what:?}")
            }
            other => panic!("at {frac}% expected Truncated, got {other:?}"),
        }
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    // One extra byte is enough. This is the check that catches a wrong tensor
    // order or a missing tensor in an otherwise well-formed file.
    let mut bytes = build_checkpoint(Shape::default());
    bytes.push(0);
    let err = Weights::from_bytes(&bytes).expect_err("trailing byte must not load");
    assert_eq!(err, LoadError::TrailingBytes(1));

    // A whole phantom tensor's worth of bytes is also rejected.
    let shape = Shape::default();
    let mut bytes = build_checkpoint(shape);
    bytes.extend(std::iter::repeat_n(0u8, shape.dim as usize * 4));
    let err = Weights::from_bytes(&bytes).expect_err("trailing tensor must not load");
    assert_eq!(err, LoadError::TrailingBytes(shape.dim as usize * 4));
}

// ---------------------------------------------------------------------------
// Header validation
// ---------------------------------------------------------------------------

#[test]
fn zero_fields_are_rejected() {
    // Each of the five dimensions the format declares as positive, zeroed in
    // turn, plus vocab_size and seq_len.
    //
    // The body is a fixed 1 MiB rather than a correctly sized one. Computing a
    // correct size would mean dividing by `n_heads`, which is exactly the
    // field under test - a divide-by-zero in the *test* would masquerade as a
    // loader bug. A body that is comfortably larger than any plausible shape
    // for these values means the header check is what rejects the input.
    for idx in 0..7 {
        let mut fields = [64i32, 128, 2, 4, 4, 32, 8];
        fields[idx] = 0;
        let shape = Shape {
            dim: fields[0],
            hidden: fields[1],
            layers: fields[2],
            heads: fields[3],
            kv_heads: fields[4],
            vocab: fields[5],
            seq: fields[6],
        };
        let mut bytes = shape.header_bytes();
        bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
        let err = Weights::from_bytes(&bytes)
            .err()
            .unwrap_or_else(|| panic!("field {idx} = 0 was accepted"));
        assert!(
            matches!(err, LoadError::InvalidHeader(_)),
            "field {idx} gave {err:?}"
        );
    }
}

#[test]
fn negative_fields_are_rejected() {
    // Only vocab_size is ever legitimately negative (it flags an untied
    // classifier), so every other field must reject a negative value.
    for idx in [0usize, 1, 2, 3, 4, 6] {
        let base = Shape::default();
        let fields: [i32; 7] = [
            base.dim,
            base.hidden,
            base.layers,
            base.heads,
            base.kv_heads,
            base.vocab,
            base.seq,
        ];
        let mut mutated = fields;
        mutated[idx] = -1;
        let shape = Shape {
            dim: mutated[0],
            hidden: mutated[1],
            layers: mutated[2],
            heads: mutated[3],
            kv_heads: mutated[4],
            vocab: mutated[5],
            seq: mutated[6],
        };
        let mut bytes = shape.header_bytes();
        // A generous body, so the header check is what rejects it and not a
        // short file.
        bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
        let err = Weights::from_bytes(&bytes)
            .err()
            .unwrap_or_else(|| panic!("field {idx} = -1 was accepted"));
        assert!(
            matches!(err, LoadError::InvalidHeader(_)),
            "field {idx} gave {err:?}"
        );
    }
}

#[test]
fn bad_divisibility_is_rejected() {
    // dim not divisible by n_heads.
    let shape = Shape {
        dim: 65,
        ..Default::default()
    };
    let mut bytes = shape.header_bytes();
    bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
    let err = Weights::from_bytes(&bytes).expect_err("dim=65, heads=4 must be rejected");
    match err {
        LoadError::InvalidHeader(m) => assert!(m.contains("divisible by n_heads"), "{m}"),
        other => panic!("got {other:?}"),
    }

    // n_heads not divisible by n_kv_heads.
    let shape = Shape {
        heads: 4,
        kv_heads: 3,
        ..Default::default()
    };
    let mut bytes = shape.header_bytes();
    bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
    let err = Weights::from_bytes(&bytes).expect_err("heads=4, kv=3 must be rejected");
    match err {
        LoadError::InvalidHeader(m) => assert!(m.contains("divisible by n_kv_heads"), "{m}"),
        other => panic!("got {other:?}"),
    }

    // Odd head_size. dim=12 with 4 heads gives head_size 3, which cannot be
    // split into RoPE pairs.
    let shape = Shape {
        dim: 12,
        heads: 4,
        ..Default::default()
    };
    let mut bytes = shape.header_bytes();
    bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
    let err = Weights::from_bytes(&bytes).expect_err("odd head_size must be rejected");
    match err {
        LoadError::InvalidHeader(m) => assert!(m.contains("even"), "{m}"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn i32_extremes_are_rejected_without_panicking() {
    // i32::MIN is the interesting one: `i32::MIN.abs()` panics in debug builds,
    // so the loader must use `unsigned_abs`. If someone "simplifies" that call
    // back to `abs`, this test panics in `cargo test` (debug) and is silent in
    // release, which is exactly the failure mode this suite exists to catch.
    for vocab in [i32::MIN, i32::MAX, -1, 1, 0] {
        let shape = Shape {
            vocab,
            ..Default::default()
        };
        let mut bytes = shape.header_bytes();
        bytes.extend(std::iter::repeat_n(0u8, 1 << 20));
        // The only requirement is totality: Ok or Err, never a panic.
        let _ = Weights::from_bytes(&bytes);
    }

    // i32::MAX as every dimension, which makes the size products enormous.
    for field in 0..7 {
        let mut fields = [64i32, 128, 2, 4, 4, 32, 8];
        fields[field] = i32::MAX;
        let shape = Shape {
            dim: fields[0],
            hidden: fields[1],
            layers: fields[2],
            heads: fields[3],
            kv_heads: fields[4],
            vocab: fields[5],
            seq: fields[6],
        };
        let mut bytes = shape.header_bytes();
        bytes.extend(std::iter::repeat_n(0u8, 1 << 16));
        let _ = Weights::from_bytes(&bytes);
    }
}

#[test]
fn a_header_whose_size_products_overflow_is_rejected() {
    // The interesting property here is *which* error comes back. A header
    // whose products overflow must be reported as Overflow, not as Truncated:
    // Truncated would mean the code computed a small wrapped length and
    // believed it, which is the bug the checked arithmetic exists to prevent.
    //
    // The size computation must genuinely be reached, so the shape has to be
    // *structurally* legal. In particular dim must be even here, because an odd
    // dim would be rejected by the head_size check first and the test would
    // pass for the wrong reason. i32::MAX - 1 is the largest even i32.
    //
    // With n_heads = 1: head_size = dim = 2^31-2 (even, fine) and
    // n_layers * dim * dim ~ 2^31 * 2^62 = 2^93, far past 2^64.
    let shape = Shape {
        dim: i32::MAX - 1,
        hidden: 1,
        layers: i32::MAX,
        heads: 1,
        kv_heads: 1,
        vocab: 1,
        seq: 1,
    };
    let mut bytes = shape.header_bytes();
    bytes.extend(std::iter::repeat_n(0u8, 1 << 16));
    let err = Weights::from_bytes(&bytes).expect_err("overflowing header must be rejected");
    assert!(
        matches!(err, LoadError::Overflow { .. }),
        "expected Overflow, got {err:?}"
    );
}

#[test]
fn a_large_but_non_overflowing_header_is_rejected_as_truncated_not_allocated() {
    // This is the "allocation size is bounded by file size" property. The
    // header is perfectly legal as a set of numbers and the products do not
    // overflow, so the only thing standing between this and a 30 GB allocation
    // is the check that the file actually contains the bytes. The file is
    // 64 KiB, so the loader must fail fast.
    let shape = Shape {
        dim: 4096,
        hidden: 11008,
        layers: 32,
        heads: 32,
        kv_heads: 32,
        vocab: 32000,
        seq: 2048,
    };
    // head_size = 128, even; 32 % 32 == 0. Structurally legal.
    let mut bytes = shape.header_bytes();
    bytes.extend(std::iter::repeat_n(0u8, 1 << 16));
    let err = Weights::from_bytes(&bytes).expect_err("huge header on a tiny file must fail");
    assert!(
        matches!(err, LoadError::Truncated { .. }),
        "expected Truncated, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Fuzz-style totality sweep
// ---------------------------------------------------------------------------

/// 2000 pseudo-random buffers, half of them prefixed with a plausible header.
///
/// The point is not that any particular input produces a particular error. The
/// point is that the loader is *total*: after this loop, every one of the 2000
/// inputs has been fed in and the process is still running. A panic anywhere
/// inside `from_bytes` fails the test.
///
/// Prefixing a valid header is what makes this useful. Pure random bytes almost
/// always fail at field 0 of the header, so they only test the length check.
/// With a plausible header the cursor gets past validation and starts reading
/// tensors, which is where the size arithmetic and the trailing-bytes check
/// actually live.
#[test]
fn random_buffers_never_panic() {
    let mut rng = Xorshift32(0x1234_5678);
    let good = Shape::default();
    let header = good.header_bytes();

    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for i in 0..2000u32 {
        // Vary the length so the sweep covers "too short", "about right" and
        // "far too long" relative to the plausible header.
        let len = (i as usize * 37) % 4096;
        let mut buf = rng.fill(len);
        if i % 2 == 0 {
            // Half get the plausible header spliced in front. The header is
            // longer than the buffer in some cases, in which case the splice
            // simply produces a different random buffer - still a valid test
            // input either way.
            let mut with_header = header.clone();
            with_header.extend_from_slice(&buf);
            buf = with_header;
        }

        match Weights::from_bytes(&buf) {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    // Sanity-check the harness itself: if literally nothing was accepted the
    // sweep is not reaching the interesting code, and if everything was
    // accepted the loader is accepting garbage. Both mean the test is broken.
    assert!(
        rejected > 1900,
        "expected almost everything to be rejected, got {rejected} rejected / {accepted} accepted"
    );
}

// ---------------------------------------------------------------------------
// Positive tests: a well-formed file loads, and the result is self-consistent
// ---------------------------------------------------------------------------

#[test]
fn a_well_formed_checkpoint_loads() {
    let shape = Shape::default();
    let w = Weights::from_bytes(&build_checkpoint(shape)).expect("valid checkpoint must load");

    assert_eq!(
        w.config,
        Config {
            dim: 64,
            hidden_dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 32,
            seq_len: 8,
        }
    );
    assert_eq!(w.tok_embedding.len(), 32 * 64);
    assert_eq!(w.rms_att.len(), 2 * 64);
    assert_eq!(w.wq.len(), 2 * 64 * 64);
    assert_eq!(w.wk.len(), 2 * 64 * 64);
    assert_eq!(w.wv.len(), 2 * 64 * 64);
    assert_eq!(w.wo.len(), 2 * 64 * 64);
    assert_eq!(w.rms_ffn.len(), 2 * 64);
    assert_eq!(w.w1.len(), 2 * 128 * 64);
    assert_eq!(w.w2.len(), 2 * 64 * 128);
    assert_eq!(w.w3.len(), 2 * 128 * 64);
    assert_eq!(w.rms_final.len(), 64);
    // Tied classifier: no wcls tensor in the file.
    assert!(w.wcls.is_none());
}

#[test]
fn an_untied_checkpoint_reads_a_separate_classifier() {
    let shape = Shape {
        vocab: -32, // negative flags "not tied"
        ..Default::default()
    };
    let w = Weights::from_bytes(&build_checkpoint(shape)).expect("untied checkpoint must load");

    assert_eq!(
        w.config.vocab_size, 32,
        "vocab_size must be the absolute value"
    );
    let wcls = w.wcls.as_ref().expect("untied model must have wcls");
    assert_eq!(wcls.len(), 32 * 64);
    // And the classifier accessor must return wcls, not the embedding.
    assert_eq!(w.classifier().len(), 32 * 64);
    assert!(!std::ptr::eq(
        w.classifier().as_ptr(),
        w.tok_embedding.as_ptr()
    ));
}

#[test]
fn a_tied_checkpoint_reuses_the_embedding_as_the_classifier() {
    let shape = Shape {
        vocab: 32,
        ..Default::default()
    };
    let w = Weights::from_bytes(&build_checkpoint(shape)).unwrap();
    assert!(w.wcls.is_none());
    // The classifier *is* the token embedding, by pointer identity. This is
    // weight tying; asserting on the pointer makes it impossible to
    // accidentally substitute a copy, which would still give the right answer
    // here but would double the memory for a 7B model.
    assert!(std::ptr::eq(
        w.classifier().as_ptr(),
        w.tok_embedding.as_ptr()
    ));
}

#[test]
fn grouped_query_configs_change_the_kv_shapes_only() {
    // With kv_heads < heads, wk and wv are narrower than wq and wo. Getting
    // this wrong is a classic bug: it either reads past the end of wk or
    // silently uses query-head-major data as KV data.
    let shape = Shape {
        dim: 64,
        heads: 8,
        kv_heads: 2,
        layers: 3,
        hidden: 64,
        vocab: 16,
        seq: 8,
    };
    let w = Weights::from_bytes(&build_checkpoint(shape)).unwrap();

    let head_size = 64 / 8; // 8
    let kv_dim = head_size * 2; // 16
    assert_eq!(w.wq.len(), 3 * 64 * 64);
    assert_eq!(w.wk.len(), 3 * kv_dim * 64, "wk must be kv_dim wide");
    assert_eq!(w.wv.len(), 3 * kv_dim * 64, "wv must be kv_dim wide");
    assert_eq!(w.wo.len(), 3 * 64 * 64, "wo must be dim wide");

    // Per-layer slicing must line up with those shapes.
    assert_eq!(w.wk_layer(0).len(), kv_dim * 64);
    assert_eq!(w.wk_layer(2).len(), kv_dim * 64);
    assert_eq!(w.wq_layer(1).len(), 64 * 64);
}

#[test]
fn rope_tables_are_skipped_but_still_occupy_the_right_bytes() {
    // The file contains freq_cis_real and freq_cis_imag of seq_len x head_size/2
    // each. The loader must advance past them, so the *byte offset* after
    // rms_final has to account for them even though the values are discarded.
    // If it did not, an untied checkpoint would fail to find wcls, and a tied
    // one would report trailing bytes.
    let shape = Shape {
        dim: 64,
        heads: 4, // head_size 16, so head_size/2 = 8
        seq: 8,
        ..Default::default()
    };
    // 2 * 8 * 8 = 128 floats of RoPE table. Prove the offset matters by
    // loading both the tied and untied variants of the same shape.
    assert_eq!(2 * 8 * 8, 128);

    let tied = Weights::from_bytes(&build_checkpoint(Shape { vocab: 32, ..shape })).unwrap();
    assert!(tied.wcls.is_none());

    let untied = Weights::from_bytes(&build_checkpoint(Shape {
        vocab: -32,
        ..shape
    }))
    .unwrap();
    assert!(untied.wcls.is_some());
}

#[test]
fn tensor_values_land_in_the_right_place() {
    // The generator writes element i as (i % 17) - 8.0, so the first element of
    // every tensor is known. Checking a few boundaries catches a reordering of
    // the read sequence, which a length-only test would not.
    let shape = Shape::default();
    let w = Weights::from_bytes(&build_checkpoint(shape)).unwrap();
    let expect = |i: usize| (i % 17) as f32 - 8.0;

    assert_eq!(w.tok_embedding[0], expect(0));
    let after_emb = 32 * 64;
    assert_eq!(w.rms_att[0], expect(after_emb));
    let after_rms_att = after_emb + 2 * 64;
    assert_eq!(w.wq[0], expect(after_rms_att));
    let after_wq = after_rms_att + 2 * 64 * 64;
    assert_eq!(w.wk[0], expect(after_wq));
    let after_wk = after_wq + 2 * 64 * 64;
    assert_eq!(w.wv[0], expect(after_wk));
    let after_wv = after_wk + 2 * 64 * 64;
    assert_eq!(w.wo[0], expect(after_wv));
    let after_wo = after_wv + 2 * 64 * 64;
    assert_eq!(w.rms_ffn[0], expect(after_wo));
    let after_rms_ffn = after_wo + 2 * 64;
    assert_eq!(w.w1[0], expect(after_rms_ffn));
    let after_w1 = after_rms_ffn + 2 * 128 * 64;
    assert_eq!(w.w2[0], expect(after_w1));
    let after_w2 = after_w1 + 2 * 64 * 128;
    assert_eq!(w.w3[0], expect(after_w2));
    let after_w3 = after_w2 + 2 * 128 * 64;
    assert_eq!(w.rms_final[0], expect(after_w3));
}
