//! Tests for Q8_0 v2 checkpoint loading, quantized forward/generation, and
//! bit-for-bit parity between single-threaded and persistent-pool execution.

use proofinfer::model::{Config, LoadError, SampleConfig, State, Weights};
use proofinfer::ops;
use proofinfer::pool;
use proofinfer::quant::{
    generate_q80_sampled_stream, Checkpoint, QuantizedState, QuantizedWeights, V2_HEADER_BYTES,
    V2_MAGIC,
};

fn build_q80_checkpoint(cfg: &Config, shared: bool, group_size: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&V2_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    for field in [
        cfg.dim as i32,
        cfg.hidden_dim as i32,
        cfg.n_layers as i32,
        cfg.n_heads as i32,
        cfg.n_kv_heads as i32,
        cfg.vocab_size as i32,
        cfg.seq_len as i32,
    ] {
        bytes.extend_from_slice(&field.to_le_bytes());
    }
    bytes.push(u8::from(shared));
    bytes.extend_from_slice(&group_size.to_le_bytes());
    assert!(bytes.len() <= V2_HEADER_BYTES);
    bytes.resize(V2_HEADER_BYTES, 0);

    let (d, h, l, v) = (cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.vocab_size);
    let head_size = d / cfg.n_heads;
    let kv_dim = head_size * cfg.n_kv_heads;
    let gs = group_size as usize;

    let mut seed = 0u64;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((seed >> 33) as f32) / ((1u64 << 31) as f32);
        u * 2.0 - 1.0
    };

    // FP32 norms: rms_att, rms_ffn, rms_final
    for _ in 0..(l * d + l * d + d) {
        let val = 0.8 + 0.4 * next_f32().abs();
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    let mut push_q80 = |elems: usize| {
        let fp: Vec<f32> = (0..elems).map(|_| next_f32() * 0.5).collect();
        let mut q = vec![0i8; elems];
        let mut s = vec![0.0f32; elems / gs];
        ops::quantize_q80(&mut q, &mut s, &fp, gs);
        bytes.extend(q.iter().map(|&v| v as u8));
        for &sv in &s {
            bytes.extend_from_slice(&sv.to_le_bytes());
        }
    };

    // Order matches export.py::version2_export
    push_q80(v * d); // tok_embeddings
    for _ in 0..l {
        push_q80(d * d); // wq
    }
    for _ in 0..l {
        push_q80(kv_dim * d); // wk
    }
    for _ in 0..l {
        push_q80(kv_dim * d); // wv
    }
    for _ in 0..l {
        push_q80(d * d); // wo
    }
    for _ in 0..l {
        push_q80(h * d); // w1
    }
    for _ in 0..l {
        push_q80(d * h); // w2
    }
    for _ in 0..l {
        push_q80(h * d); // w3
    }
    if !shared {
        push_q80(v * d); // wcls
    }

    bytes
}

fn build_fp32_checkpoint(
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
    let mut total = v * d
        + l * d
        + l * d * d
        + l * kv_dim * d
        + l * kv_dim * d
        + l * d * d
        + l * d
        + l * h * d
        + l * d * h
        + l * h * d
        + d
        + 2 * s * (head_size / 2);
    if vocab < 0 {
        total += v * d;
    }
    let mut bytes = Vec::with_capacity(28 + total * 4);
    for f in [dim, hidden, layers, heads, kv_heads, vocab, seq] {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    for i in 0..total {
        let v = ((i % 41) as f32 / 20.0) - 1.0;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[test]
fn q80_loader_parses_tied_and_untied_and_auto_detects_format() {
    let tied_cfg = Config {
        dim: 64,
        hidden_dim: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 48,
        seq_len: 16,
    };
    let tied_bytes = build_q80_checkpoint(&tied_cfg, true, 32);
    let ckpt = Checkpoint::from_bytes(&tied_bytes).unwrap();
    assert!(!ckpt.is_untied());
    match ckpt {
        Checkpoint::Q80(w) => {
            assert_eq!(w.group_size, 32);
            assert!(w.wcls.is_none());
            assert_eq!(w.tok_embedding.len(), 48 * 64);
        }
        Checkpoint::Fp32(_) => panic!("expected Q80 checkpoint"),
    }

    let untied_cfg = Config {
        dim: 64,
        hidden_dim: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 48,
        seq_len: 16,
    };
    let untied_bytes = build_q80_checkpoint(&untied_cfg, false, 16);
    let ckpt_u = Checkpoint::from_bytes(&untied_bytes).unwrap();
    assert!(ckpt_u.is_untied());
    match ckpt_u {
        Checkpoint::Q80(w) => {
            assert_eq!(w.group_size, 16);
            assert!(w.wcls.is_some());
        }
        Checkpoint::Fp32(_) => panic!("expected untied Q80 checkpoint"),
    }
}

#[test]
fn q80_loader_rejects_malformed_headers_trailing_bytes_and_random_inputs() {
    let cfg = Config {
        dim: 64,
        hidden_dim: 128,
        n_layers: 1,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 32,
        seq_len: 8,
    };
    let valid = build_q80_checkpoint(&cfg, true, 32);

    // Trailing byte must fail.
    let mut trailing = valid.clone();
    trailing.push(0);
    assert!(matches!(
        QuantizedWeights::from_bytes(&trailing),
        Err(LoadError::TrailingBytes(1))
    ));

    // Truncated body must fail cleanly.
    assert!(matches!(
        QuantizedWeights::from_bytes(&valid[..valid.len() - 1]),
        Err(LoadError::Truncated { .. })
    ));

    // Unsupported version (v1 or v3) must fail.
    let mut bad_ver = valid.clone();
    bad_ver[4..8].copy_from_slice(&3i32.to_le_bytes());
    assert!(matches!(
        QuantizedWeights::from_bytes(&bad_ver),
        Err(LoadError::InvalidHeader(_))
    ));

    // Bad shared_classifier flag (2).
    let mut bad_flag = valid.clone();
    bad_flag[36] = 2;
    assert!(matches!(
        QuantizedWeights::from_bytes(&bad_flag),
        Err(LoadError::InvalidHeader(_))
    ));

    // Non-divisible or zero group_size.
    for gs in [0i32, -8, 12, 100_000] {
        let mut bad_gs = valid.clone();
        bad_gs[37..41].copy_from_slice(&gs.to_le_bytes());
        assert!(
            matches!(
                QuantizedWeights::from_bytes(&bad_gs),
                Err(LoadError::InvalidHeader(_))
            ),
            "expected InvalidHeader for group_size={gs}"
        );
    }

    // 500 hostile pseudo-random buffers prefixed with V2_MAGIC must never panic.
    let mut state = 0x1234_5678_9abc_def0u64;
    for i in 0..500usize {
        let len = (i * 17) % 512;
        let mut buf = Vec::with_capacity(len.max(8));
        buf.extend_from_slice(&V2_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        while buf.len() < len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            buf.push((state >> 56) as u8);
        }
        let _ = Checkpoint::from_bytes(&buf);
    }
}

#[test]
fn q80_forward_is_causal_and_bit_identical_across_persistent_pool_threads() {
    // Large enough vocab*dim to exceed pool::PARALLEL_MIN_OPS (131_072) so
    // background worker threads actually execute classifier rows.
    let cfg = Config {
        dim: 128,
        hidden_dim: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 1200,
        seq_len: 16,
    };
    let bytes = build_q80_checkpoint(&cfg, true, 32);
    let w = QuantizedWeights::from_bytes(&bytes).unwrap();
    let prompt = vec![1u32, 7, 19, 42];

    let mut s1 = QuantizedState::new(&w.config, w.group_size).unwrap();
    let mut single_logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        single_logits.push(s1.forward(&w, t, pos).unwrap().to_vec());
    }

    // Verify causality: prefix logits match when followed by more tokens.
    let mut s_long = QuantizedState::new(&w.config, w.group_size).unwrap();
    for (pos, &t) in [1u32, 7, 19, 42, 99, 123].iter().enumerate() {
        let l = s_long.forward(&w, t, pos).unwrap();
        if pos < prompt.len() {
            assert_eq!(l, single_logits[pos].as_slice());
        }
    }

    // Verify persistent WorkerPool matches single-threaded bit-for-bit.
    for threads in [1usize, 2, 4] {
        let mut sp = QuantizedState::new(&w.config, w.group_size).unwrap();
        pool::with_q80_pool(&w, threads, |p| {
            for (pos, &t) in prompt.iter().enumerate() {
                let got = pool::forward_q80_pooled(&mut sp, &w, p, t, pos).unwrap();
                for (i, (&a, &b)) in got.iter().zip(single_logits[pos].iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "Q8_0 mismatch at pos {pos} logit {i} with {threads} threads"
                    );
                }
            }
        });
    }

    // Verify pooled generation matches single-threaded Q8_0 generation.
    let mut single_gen = Vec::new();
    generate_q80_sampled_stream(
        &w,
        &mut QuantizedState::new(&w.config, w.group_size).unwrap(),
        &prompt,
        6,
        SampleConfig::default(),
        |t| {
            single_gen.push(t);
            true
        },
    )
    .unwrap();
    let mut pooled_gen = Vec::new();
    pool::generate_q80_pooled_stream(
        &w,
        &mut QuantizedState::new(&w.config, w.group_size).unwrap(),
        &prompt,
        6,
        SampleConfig::default(),
        4,
        |t| {
            pooled_gen.push(t);
            true
        },
    )
    .unwrap();
    assert_eq!(pooled_gen, single_gen);
}

#[test]
fn fp32_persistent_pool_is_bit_identical_to_single_threaded_forward() {
    // 1200 * 128 = 153_600 > PARALLEL_MIN_OPS, exercising worker threads.
    let bytes = build_fp32_checkpoint(128, 256, 2, 4, 2, 1200, 16);
    let w = Weights::from_bytes(&bytes).unwrap();
    let tokens = [1u32, 5, 11, 23];

    let mut s_ref = State::new(&w.config).unwrap();
    let ref_logits: Vec<Vec<f32>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| s_ref.forward(&w, t, pos).unwrap().to_vec())
        .collect();

    for threads in [2usize, 4] {
        let mut s_pool = State::new(&w.config).unwrap();
        pool::with_fp32_pool(&w, threads, |p| {
            for (pos, &t) in tokens.iter().enumerate() {
                let got = pool::forward_fp32_pooled(&mut s_pool, &w, p, t, pos).unwrap();
                for (i, (&a, &b)) in got.iter().zip(ref_logits[pos].iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "FP32 pooled mismatch at pos {pos} logit {i} with {threads} threads"
                    );
                }
            }
        });
    }
}
