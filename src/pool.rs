//! Zero-dependency persistent worker pool for row-partitioned matmul.
//!
//! Spawns `threads - 1` background workers once inside a `std::thread::scope`
//! and reuses them across every matmul of a forward or generation session.
//! Built purely on `std::sync` with zero `unsafe`: workers spin briefly on a
//! generation counter to avoid syscall wake latency during active decoding,
//! then park on a `Condvar` when idle.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, RwLock};

use crate::model::{select_next_token, Rng, RunError, SampleConfig, State, Weights};
use crate::ops::{self, QSlice};
use crate::quant::{QuantizedState, QuantizedWeights};

/// Minimum multiply-accumulate count (`n * d`) before dispatching to background
/// workers. Below this threshold the main thread runs inline so small
/// projections never pay synchronization overhead.
pub const PARALLEL_MIN_OPS: usize = 131_072;

const SPIN_ITERS: u32 = 4096;

struct Dispatch<M: Copy> {
    matrix: M,
    n: usize,
    x_f32: Vec<f32>,
    x_q: Vec<i8>,
    x_s: Vec<f32>,
}

struct Shared<M: Copy> {
    workers: usize,
    gen: AtomicU64,
    done: AtomicUsize,
    sleeping: AtomicUsize,
    shutdown: AtomicBool,
    job: RwLock<Dispatch<M>>,
    outputs: Vec<Mutex<Vec<f32>>>,
    wake_mu: Mutex<u64>,
    wake_cv: Condvar,
}

struct ShutdownGuard<'a, M: Copy>(&'a Shared<M>);

impl<'a, M: Copy> Drop for ShutdownGuard<'a, M> {
    fn drop(&mut self) {
        if self.0.workers == 0 {
            return;
        }
        self.0.shutdown.store(true, Ordering::Release);
        let next = self.0.gen.fetch_add(1, Ordering::SeqCst) + 1;
        let mut guard = self.0.wake_mu.lock().expect("pool wake lock poisoned");
        *guard = next;
        self.0.wake_cv.notify_all();
    }
}

/// Handle passed into a scoped session to dispatch row-partitioned matmuls
/// across persistent worker threads without per-call thread spawning.
pub struct WorkerPool<'a, M: Copy> {
    shared: &'a Shared<M>,
}

impl<'a, M: Copy> WorkerPool<'a, M> {
    /// Total participating threads (main thread + background workers).
    pub fn threads(&self) -> usize {
        self.shared.workers + 1
    }

    /// Execute a row-partitioned FP32 matmul across the pool, or inline when
    /// the matrix is below [`PARALLEL_MIN_OPS`].
    pub fn matmul_f32<L>(
        &self,
        matrix: M,
        o: &mut [f32],
        x: &[f32],
        n: usize,
        d: usize,
        run_local: L,
    ) where
        L: FnOnce(M, &[f32], usize, usize, &mut [f32]),
    {
        let total_threads = self.shared.workers + 1;
        if self.shared.workers == 0
            || n < total_threads * 4
            || n.saturating_mul(d) < PARALLEL_MIN_OPS
        {
            run_local(matrix, x, 0, n, o);
            return;
        }

        let chunk = n / total_threads + usize::from(n % total_threads != 0);
        {
            let mut job = self.shared.job.write().expect("pool job lock poisoned");
            job.matrix = matrix;
            job.n = n;
            job.x_f32.clear();
            job.x_f32.extend_from_slice(x);
            job.x_q.clear();
            job.x_s.clear();
        }

        self.dispatch_and_collect(
            n,
            chunk,
            |start, end, dst| run_local(matrix, x, start, end, dst),
            o,
        );
    }

    /// Execute a row-partitioned Q8_0 matmul across the pool, or inline when
    /// the matrix is below [`PARALLEL_MIN_OPS`].
    pub fn matmul_q80<L>(
        &self,
        matrix: M,
        o: &mut [f32],
        x: QSlice<'_>,
        n: usize,
        d: usize,
        run_local: L,
    ) where
        L: FnOnce(M, QSlice<'_>, usize, usize, &mut [f32]),
    {
        let total_threads = self.shared.workers + 1;
        if self.shared.workers == 0
            || n < total_threads * 4
            || n.saturating_mul(d) < PARALLEL_MIN_OPS
        {
            run_local(matrix, x, 0, n, o);
            return;
        }

        let chunk = n / total_threads + usize::from(n % total_threads != 0);
        {
            let mut job = self.shared.job.write().expect("pool job lock poisoned");
            job.matrix = matrix;
            job.n = n;
            job.x_f32.clear();
            job.x_q.clear();
            job.x_q.extend_from_slice(x.q);
            job.x_s.clear();
            job.x_s.extend_from_slice(x.s);
        }

        self.dispatch_and_collect(
            n,
            chunk,
            |start, end, dst| run_local(matrix, x, start, end, dst),
            o,
        );
    }

    fn dispatch_and_collect<L>(&self, n: usize, chunk: usize, run_local: L, o: &mut [f32])
    where
        L: FnOnce(usize, usize, &mut [f32]),
    {
        let shared = self.shared;
        shared.done.store(0, Ordering::Relaxed);
        let next_gen = shared.gen.fetch_add(1, Ordering::SeqCst) + 1;
        if shared.sleeping.load(Ordering::SeqCst) > 0 {
            let mut guard = shared.wake_mu.lock().expect("pool wake lock poisoned");
            *guard = next_gen;
            shared.wake_cv.notify_all();
        }

        let main_end = chunk.min(n);
        run_local(0, main_end, &mut o[0..main_end]);

        while shared.done.load(Ordering::Acquire) < shared.workers {
            std::hint::spin_loop();
        }

        for (w_idx, slot_mu) in shared.outputs.iter().enumerate() {
            let start = (w_idx + 1).saturating_mul(chunk).min(n);
            let end = (w_idx + 2).saturating_mul(chunk).min(n);
            if start < end {
                let buf = slot_mu.lock().expect("pool output lock poisoned");
                o[start..end].copy_from_slice(&buf[..end - start]);
            }
        }
    }
}

fn wait_for_gen<M: Copy>(shared: &Shared<M>, seen: u64) -> u64 {
    for _ in 0..SPIN_ITERS {
        let cur = shared.gen.load(Ordering::Acquire);
        if cur != seen {
            return cur;
        }
        std::hint::spin_loop();
    }
    shared.sleeping.fetch_add(1, Ordering::SeqCst);
    let mut guard = shared.wake_mu.lock().expect("pool wake lock poisoned");
    loop {
        let cur = shared.gen.load(Ordering::SeqCst);
        if cur != seen {
            drop(guard);
            shared.sleeping.fetch_sub(1, Ordering::SeqCst);
            return cur;
        }
        guard = shared.wake_cv.wait(guard).expect("pool wake cv poisoned");
    }
}

/// Spawn `threads.saturating_sub(1)` persistent workers for the duration of
/// `body`, guaranteeing clean shutdown and join when `body` returns or unwinds.
pub fn with_workers<M, W, B, R>(threads: usize, default_matrix: M, worker_fn: W, body: B) -> R
where
    M: Copy + Send + Sync,
    W: Fn(M, &[f32], &[i8], &[f32], usize, usize, &mut [f32]) + Sync,
    B: FnOnce(&WorkerPool<'_, M>) -> R,
{
    let workers = threads.saturating_sub(1);
    let outputs = (0..workers).map(|_| Mutex::new(Vec::new())).collect();
    let shared = Shared {
        workers,
        gen: AtomicU64::new(0),
        done: AtomicUsize::new(0),
        sleeping: AtomicUsize::new(0),
        shutdown: AtomicBool::new(false),
        job: RwLock::new(Dispatch {
            matrix: default_matrix,
            n: 0,
            x_f32: Vec::new(),
            x_q: Vec::new(),
            x_s: Vec::new(),
        }),
        outputs,
        wake_mu: Mutex::new(0),
        wake_cv: Condvar::new(),
    };

    if workers == 0 {
        let pool = WorkerPool { shared: &shared };
        return body(&pool);
    }

    let total_threads = workers + 1;
    std::thread::scope(|s| {
        let shared_ref = &shared;
        let worker_fn_ref = &worker_fn;
        for w_idx in 0..workers {
            s.spawn(move || {
                let mut seen_gen = 0u64;
                loop {
                    let cur = wait_for_gen(shared_ref, seen_gen);
                    seen_gen = cur;
                    if shared_ref.shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    let job = shared_ref.job.read().expect("pool job lock poisoned");
                    let n = job.n;
                    let chunk = n / total_threads + usize::from(n % total_threads != 0);
                    let start = (w_idx + 1).saturating_mul(chunk).min(n);
                    let end = (w_idx + 2).saturating_mul(chunk).min(n);
                    if start < end {
                        let mut out = shared_ref.outputs[w_idx]
                            .lock()
                            .expect("pool output lock poisoned");
                        let len = end - start;
                        if out.len() < len {
                            out.resize(len, 0.0);
                        }
                        worker_fn_ref(
                            job.matrix,
                            &job.x_f32,
                            &job.x_q,
                            &job.x_s,
                            start,
                            end,
                            &mut out[..len],
                        );
                    }
                    drop(job);
                    shared_ref.done.fetch_add(1, Ordering::Release);
                }
            });
        }

        let _shutdown = ShutdownGuard(shared_ref);
        let pool = WorkerPool { shared: shared_ref };
        body(&pool)
    })
}

/// Matrix selector passed to persistent FP32/Q8_0 pool workers.
#[derive(Debug, Clone, Copy)]
pub enum MatrixTarget {
    Wq(usize),
    Wk(usize),
    Wv(usize),
    Wo(usize),
    W1(usize),
    W2(usize),
    W3(usize),
    Classifier,
}

fn fp32_matrix_slice(w: &Weights, target: MatrixTarget) -> (&[f32], usize) {
    let d = w.config.dim;
    match target {
        MatrixTarget::Wq(l) => (w.wq_layer(l), d),
        MatrixTarget::Wk(l) => (w.wk_layer(l), d),
        MatrixTarget::Wv(l) => (w.wv_layer(l), d),
        MatrixTarget::Wo(l) => (w.wo_layer(l), w.attention_dims().q_dim()),
        MatrixTarget::W1(l) => (w.w1_layer(l), d),
        MatrixTarget::W2(l) => (w.w2_layer(l), w.config.hidden_dim),
        MatrixTarget::W3(l) => (w.w3_layer(l), d),
        MatrixTarget::Classifier => (w.classifier(), d),
    }
}

fn fp32_eval_rows(
    w: &Weights,
    target: MatrixTarget,
    x: &[f32],
    start: usize,
    end: usize,
    dst: &mut [f32],
) {
    let (mat, d) = fp32_matrix_slice(w, target);
    for (slot, row) in dst.iter_mut().zip(mat[start * d..end * d].chunks_exact(d)) {
        *slot = ops::dot(row, x);
    }
}

fn q80_matrix_slice<'a>(w: &'a QuantizedWeights, target: MatrixTarget) -> (QSlice<'a>, usize) {
    let d = w.config.dim;
    match target {
        MatrixTarget::Wq(l) => (w.wq_layer(l), d),
        MatrixTarget::Wk(l) => (w.wk_layer(l), d),
        MatrixTarget::Wv(l) => (w.wv_layer(l), d),
        MatrixTarget::Wo(l) => (w.wo_layer(l), w.attention_dims().q_dim()),
        MatrixTarget::W1(l) => (w.w1_layer(l), d),
        MatrixTarget::W2(l) => (w.w2_layer(l), w.config.hidden_dim),
        MatrixTarget::W3(l) => (w.w3_layer(l), d),
        MatrixTarget::Classifier => (w.classifier(), d),
    }
}

fn q80_eval_rows(
    w: &QuantizedWeights,
    target: MatrixTarget,
    x: QSlice<'_>,
    start: usize,
    _end: usize,
    dst: &mut [f32],
) {
    let (mat, d) = q80_matrix_slice(w, target);
    let gs = w.group_size;
    let row_groups = d / gs;
    for (offset, slot) in dst.iter_mut().enumerate() {
        let i = start + offset;
        let q_row = &mat.q[i * d..i * d + d];
        let s_row = &mat.s[i * row_groups..i * row_groups + row_groups];
        *slot = ops::dot_q80(QSlice { q: q_row, s: s_row }, x, d, gs);
    }
}

/// Run one FP32 forward step using a persistent [`WorkerPool`].
///
/// Produces bit-for-bit identical logits to [`State::forward`].
pub fn forward_fp32_pooled<'s>(
    state: &'s mut State,
    w: &Weights,
    pool: &WorkerPool<'_, MatrixTarget>,
    token: u32,
    pos: usize,
) -> Result<&'s [f32], RunError> {
    let cfg = &w.config;
    if state.config != *cfg {
        return Err(RunError::StateMismatch {
            detail: format!(
                "state config {:?} does not match weights config {:?}",
                state.config, cfg
            ),
        });
    }
    if token as usize >= cfg.vocab_size {
        return Err(RunError::TokenOutOfRange {
            token,
            vocab_size: cfg.vocab_size,
        });
    }
    if pos >= cfg.seq_len {
        return Err(RunError::PositionOutOfRange {
            pos,
            seq_len: cfg.seq_len,
        });
    }

    let dims = w.attention_dims();
    let d = cfg.dim;
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let kv_layer_stride = cfg.seq_len * kv_dim;

    let row = token as usize * d;
    state.x.copy_from_slice(&w.tok_embedding[row..row + d]);

    let run = |target, o: &mut [f32], x: &[f32], n, cols| {
        pool.matmul_f32(target, o, x, n, cols, |t, xf, s, e, dst| {
            fp32_eval_rows(w, t, xf, s, e, dst);
        });
    };

    for l in 0..cfg.n_layers {
        ops::rmsnorm(&mut state.xb, &state.x, w.rms_att_layer(l));
        run(MatrixTarget::Wq(l), &mut state.q, &state.xb, q_dim, d);
        run(MatrixTarget::Wk(l), &mut state.k, &state.xb, kv_dim, d);
        run(MatrixTarget::Wv(l), &mut state.v, &state.xb, kv_dim, d);

        ops::rope(&mut state.q, dims.n_heads, dims.head_size, pos);
        ops::rope(&mut state.k, dims.n_kv_heads, dims.head_size, pos);

        let base = l * kv_layer_stride + pos * kv_dim;
        state.key_cache[base..base + kv_dim].copy_from_slice(&state.k);
        state.value_cache[base..base + kv_dim].copy_from_slice(&state.v);

        let layer_start = l * kv_layer_stride;
        let layer_end = layer_start + kv_layer_stride;
        ops::attention(
            &mut state.att,
            &state.q,
            &state.key_cache[layer_start..layer_end],
            &state.value_cache[layer_start..layer_end],
            &dims,
            pos,
            &mut state.scores,
        );

        run(MatrixTarget::Wo(l), &mut state.xb, &state.att, d, q_dim);
        for i in 0..d {
            state.x[i] += state.xb[i];
        }

        ops::rmsnorm(&mut state.xb, &state.x, w.rms_ffn_layer(l));
        run(
            MatrixTarget::W1(l),
            &mut state.hb,
            &state.xb,
            cfg.hidden_dim,
            d,
        );
        run(
            MatrixTarget::W3(l),
            &mut state.hb2,
            &state.xb,
            cfg.hidden_dim,
            d,
        );
        for i in 0..cfg.hidden_dim {
            state.hb[i] = ops::silu_value(state.hb[i]) * state.hb2[i];
        }

        run(
            MatrixTarget::W2(l),
            &mut state.xb,
            &state.hb,
            d,
            cfg.hidden_dim,
        );
        for i in 0..d {
            state.x[i] += state.xb[i];
        }
    }

    ops::rmsnorm(&mut state.xb, &state.x, &w.rms_final);
    run(
        MatrixTarget::Classifier,
        &mut state.logits,
        &state.xb,
        cfg.vocab_size,
        d,
    );

    Ok(&state.logits)
}

/// Run one Q8_0 forward step using a persistent [`WorkerPool`].
///
/// Produces bit-for-bit identical logits to [`QuantizedState::forward`].
pub fn forward_q80_pooled<'s>(
    state: &'s mut QuantizedState,
    w: &QuantizedWeights,
    pool: &WorkerPool<'_, MatrixTarget>,
    token: u32,
    pos: usize,
) -> Result<&'s [f32], RunError> {
    let cfg = &w.config;
    if state.base.config != *cfg || state.group_size != w.group_size {
        return Err(RunError::StateMismatch {
            detail: format!(
                "quantized state config {:?} (gs={}) does not match weights {:?} (gs={})",
                state.base.config, state.group_size, cfg, w.group_size
            ),
        });
    }
    if token as usize >= cfg.vocab_size {
        return Err(RunError::TokenOutOfRange {
            token,
            vocab_size: cfg.vocab_size,
        });
    }
    if pos >= cfg.seq_len {
        return Err(RunError::PositionOutOfRange {
            pos,
            seq_len: cfg.seq_len,
        });
    }

    let dims = w.attention_dims();
    let d = cfg.dim;
    let hid = cfg.hidden_dim;
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let gs = w.group_size;
    let d_groups = d / gs;
    let hid_groups = hid / gs;
    let kv_layer_stride = cfg.seq_len * kv_dim;

    let row = token as usize * d;
    state.base.x.copy_from_slice(&w.tok_embedding[row..row + d]);

    let run = |target, o: &mut [f32], xq: &[i8], xs: &[f32], n, cols| {
        pool.matmul_q80(
            target,
            o,
            QSlice { q: xq, s: xs },
            n,
            cols,
            |t, x, start, end, dst| {
                q80_eval_rows(w, t, x, start, end, dst);
            },
        );
    };

    for l in 0..cfg.n_layers {
        ops::rmsnorm(&mut state.base.xb, &state.base.x, w.rms_att_layer(l));
        ops::quantize_q80(
            &mut state.xq[..d],
            &mut state.xs[..d_groups],
            &state.base.xb,
            gs,
        );
        run(
            MatrixTarget::Wq(l),
            &mut state.base.q,
            &state.xq[..d],
            &state.xs[..d_groups],
            q_dim,
            d,
        );
        run(
            MatrixTarget::Wk(l),
            &mut state.base.k,
            &state.xq[..d],
            &state.xs[..d_groups],
            kv_dim,
            d,
        );
        run(
            MatrixTarget::Wv(l),
            &mut state.base.v,
            &state.xq[..d],
            &state.xs[..d_groups],
            kv_dim,
            d,
        );

        ops::rope(&mut state.base.q, dims.n_heads, dims.head_size, pos);
        ops::rope(&mut state.base.k, dims.n_kv_heads, dims.head_size, pos);

        let base = l * kv_layer_stride + pos * kv_dim;
        state.base.key_cache[base..base + kv_dim].copy_from_slice(&state.base.k);
        state.base.value_cache[base..base + kv_dim].copy_from_slice(&state.base.v);

        let layer_start = l * kv_layer_stride;
        let layer_end = layer_start + kv_layer_stride;
        ops::attention(
            &mut state.base.att,
            &state.base.q,
            &state.base.key_cache[layer_start..layer_end],
            &state.base.value_cache[layer_start..layer_end],
            &dims,
            pos,
            &mut state.base.scores,
        );

        ops::quantize_q80(
            &mut state.xq[..q_dim],
            &mut state.xs[..q_dim / gs],
            &state.base.att,
            gs,
        );
        run(
            MatrixTarget::Wo(l),
            &mut state.base.xb,
            &state.xq[..q_dim],
            &state.xs[..q_dim / gs],
            d,
            q_dim,
        );
        for i in 0..d {
            state.base.x[i] += state.base.xb[i];
        }

        ops::rmsnorm(&mut state.base.xb, &state.base.x, w.rms_ffn_layer(l));
        ops::quantize_q80(
            &mut state.xq[..d],
            &mut state.xs[..d_groups],
            &state.base.xb,
            gs,
        );
        run(
            MatrixTarget::W1(l),
            &mut state.base.hb,
            &state.xq[..d],
            &state.xs[..d_groups],
            hid,
            d,
        );
        run(
            MatrixTarget::W3(l),
            &mut state.base.hb2,
            &state.xq[..d],
            &state.xs[..d_groups],
            hid,
            d,
        );
        for i in 0..hid {
            state.base.hb[i] = ops::silu_value(state.base.hb[i]) * state.base.hb2[i];
        }

        ops::quantize_q80(
            &mut state.xq[..hid],
            &mut state.xs[..hid_groups],
            &state.base.hb,
            gs,
        );
        run(
            MatrixTarget::W2(l),
            &mut state.base.xb,
            &state.xq[..hid],
            &state.xs[..hid_groups],
            d,
            hid,
        );
        for i in 0..d {
            state.base.x[i] += state.base.xb[i];
        }
    }

    ops::rmsnorm(&mut state.base.xb, &state.base.x, &w.rms_final);
    ops::quantize_q80(
        &mut state.xq[..d],
        &mut state.xs[..d_groups],
        &state.base.xb,
        gs,
    );
    run(
        MatrixTarget::Classifier,
        &mut state.base.logits,
        &state.xq[..d],
        &state.xs[..d_groups],
        cfg.vocab_size,
        d,
    );

    Ok(&state.base.logits)
}

/// Run a persistent FP32 worker pool over `body`.
pub fn with_fp32_pool<F, R>(w: &Weights, threads: usize, body: F) -> R
where
    F: FnOnce(&WorkerPool<'_, MatrixTarget>) -> R,
{
    with_workers(
        threads,
        MatrixTarget::Classifier,
        |target, xf, _xq, _xs, start, end, dst| fp32_eval_rows(w, target, xf, start, end, dst),
        body,
    )
}

/// Run a persistent Q8_0 worker pool over `body`.
pub fn with_q80_pool<F, R>(w: &QuantizedWeights, threads: usize, body: F) -> R
where
    F: FnOnce(&WorkerPool<'_, MatrixTarget>) -> R,
{
    with_workers(
        threads,
        MatrixTarget::Classifier,
        |target, _xf, xq, xs, start, end, dst| {
            q80_eval_rows(w, target, QSlice { q: xq, s: xs }, start, end, dst);
        },
        body,
    )
}

/// Stream up to `n` tokens from an FP32 model using a persistent worker pool.
pub fn generate_fp32_pooled_stream<F>(
    w: &Weights,
    state: &mut State,
    prompt: &[u32],
    n: usize,
    sample: SampleConfig,
    threads: usize,
    mut on_token: F,
) -> Result<usize, RunError>
where
    F: FnMut(u32) -> bool,
{
    sample.validate()?;
    if prompt.is_empty() {
        return Err(RunError::EmptyPrompt);
    }
    if prompt
        .len()
        .checked_add(n)
        .map_or(true, |total| total > w.config.seq_len)
    {
        return Err(RunError::TooManyTokens {
            requested: prompt.len().saturating_add(n),
            seq_len: w.config.seq_len,
        });
    }

    with_fp32_pool(w, threads, |pool| {
        let mut rng = Rng::new(sample.seed);
        let mut probs: Vec<f32> = Vec::new();
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        let mut next = 0u32;
        for (pos, &t) in prompt.iter().enumerate() {
            let logits = forward_fp32_pooled(state, w, pool, t, pos)?;
            if pos + 1 == prompt.len() {
                next =
                    select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
            }
        }

        let mut emitted = 0usize;
        let mut pos = prompt.len();
        while emitted < n {
            if next == crate::tokenizer::BOS_ID {
                break;
            }
            emitted += 1;
            if !on_token(next) {
                break;
            }
            if emitted == n {
                break;
            }
            let logits = forward_fp32_pooled(state, w, pool, next, pos)?;
            next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
            pos += 1;
        }
        Ok(emitted)
    })
}

/// Stream up to `n` tokens from a Q8_0 model using a persistent worker pool.
pub fn generate_q80_pooled_stream<F>(
    w: &QuantizedWeights,
    state: &mut QuantizedState,
    prompt: &[u32],
    n: usize,
    sample: SampleConfig,
    threads: usize,
    mut on_token: F,
) -> Result<usize, RunError>
where
    F: FnMut(u32) -> bool,
{
    sample.validate()?;
    if prompt.is_empty() {
        return Err(RunError::EmptyPrompt);
    }
    if prompt
        .len()
        .checked_add(n)
        .map_or(true, |total| total > w.config.seq_len)
    {
        return Err(RunError::TooManyTokens {
            requested: prompt.len().saturating_add(n),
            seq_len: w.config.seq_len,
        });
    }

    with_q80_pool(w, threads, |pool| {
        let mut rng = Rng::new(sample.seed);
        let mut probs: Vec<f32> = Vec::new();
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        let mut next = 0u32;
        for (pos, &t) in prompt.iter().enumerate() {
            let logits = forward_q80_pooled(state, w, pool, t, pos)?;
            if pos + 1 == prompt.len() {
                next =
                    select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
            }
        }

        let mut emitted = 0usize;
        let mut pos = prompt.len();
        while emitted < n {
            if next == crate::tokenizer::BOS_ID {
                break;
            }
            emitted += 1;
            if !on_token(next) {
                break;
            }
            if emitted == n {
                break;
            }
            let logits = forward_q80_pooled(state, w, pool, next, pos)?;
            next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
            pos += 1;
        }
        Ok(emitted)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    #[test]
    fn worker_pool_matches_single_threaded_across_multiple_dispatches() {
        // Large enough to exceed PARALLEL_MIN_OPS so workers actually execute.
        let n = 1024usize;
        let d = 192usize;
        let w: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.19).cos()).collect();
        let mut expected = vec![0.0f32; n];
        ops::matmul_parallel(&mut expected, &x, &w, n, d, 1);

        for threads in [1usize, 2, 4] {
            let mut got = vec![0.0f32; n];
            with_workers(
                threads,
                (),
                |(), xf, _xq, _xs, start, end, dst| {
                    for (slot, row) in dst.iter_mut().zip(w[start * d..end * d].chunks_exact(d)) {
                        *slot = ops::dot(row, xf);
                    }
                },
                |pool| {
                    assert_eq!(pool.threads(), threads);
                    // Dispatch multiple times to verify worker reuse across generations.
                    for _ in 0..4 {
                        pool.matmul_f32((), &mut got, &x, n, d, |(), xf, start, end, dst| {
                            for (slot, row) in
                                dst.iter_mut().zip(w[start * d..end * d].chunks_exact(d))
                            {
                                *slot = ops::dot(row, xf);
                            }
                        });
                    }
                },
            );
            for i in 0..n {
                assert_eq!(
                    got[i].to_bits(),
                    expected[i].to_bits(),
                    "row {i} mismatch at {threads} threads"
                );
            }
        }
    }
}
