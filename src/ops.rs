//! Numeric kernels for the Llama forward pass.
//!
//! Everything here is written against slices rather than owning buffers. The
//! `State` struct in [`crate::model`] allocates every one of these buffers once
//! and reuses them for every decoded token, so a decode step performs no
//! allocation at all. Keeping the kernels allocation-free is what makes that
//! possible.
//!
//! Signature convention, matching llama2.c: a kernel writes into its first
//! argument. Where an operation is elementwise and destructive, it takes a
//! single `&mut [f32]` and transforms it in place.

/// Epsilon added to the mean square before the square root in [`rmsnorm`].
///
/// This is *not* a numerical-stability fudge in the usual sense: it is the
/// `norm_eps` hyperparameter from the Llama 2 / llama2.c reference, which is
/// fixed at 1e-5. It keeps the gradient finite for a zero input. Matching the
/// reference exactly matters more than any reasoning of our own here, and the
/// differential test would catch us if we "improved" it.
pub const RMS_NORM_EPS: f32 = 1e-5;

/// Base of the RoPE geometric frequency progression.
///
/// 10000 is the value from the original RoPE paper, chosen so that the lowest
/// frequency wavelength equals the context length of the models it was tuned
/// for. It is a constant of the architecture, not a tunable.
pub const ROPE_THETA: f32 = 10000.0;

/// Multiply a slice by a scalar in place. Hot enough to be worth a helper.
#[inline(always)]
fn scale_in_place(v: &mut [f32], s: f32) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// Subtract a scalar from every element in place.
///
/// This deliberately exists as a separate function from [`scale_in_place`]
/// rather than being written as `scale_in_place(x, -max)`. I wrote it the
/// multiplicative way first, and the unit test
/// `softmax_normalises_and_survives_extreme_logits` caught it immediately:
/// `42.0 * -42.0` is `-1764.0`, not `0.0`, so the "shift" turned into a
/// magnitude blow-up and every exponent argument became hugely negative,
/// underflowing the whole row to zero and then producing `0/0 = NaN` in the
/// normalisation. Cheap unit tests earn their keep.
#[inline(always)]
fn shift_in_place(v: &mut [f32], s: f32) {
    for x in v.iter_mut() {
        *x -= s;
    }
}

/// Dot product of two equal-length slices.
///
/// This is deliberately the naive serial form for now. Floating-point addition
/// is *not* associative, so `a + b + c` and `a + (b + c)` can differ in the
/// last bits, and the compiler is therefore forbidden from reassociating the
/// loop-carried dependency chain. That serialisation is also what stops LLVM
/// emitting a SIMD reduction. See the optimised version in the performance
/// notes; the point of keeping this function in one place is that swapping the
/// implementation is a one-line change that the differential test immediately
/// re-validates.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Root-mean-square normalisation with a learned scale.
///
/// ```text
/// out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
/// ```
///
/// The mean square is accumulated in `f64` even though the data is `f32`. This
/// costs a negligible amount of time relative to the two matmuls that follow,
/// and it keeps the sum from accumulating error across a long hidden dimension.
/// The reference computes the same quantity in `f32` via torch, so this is
/// slightly *more* accurate than the thing we are comparing against, which is
/// fine because the comparison is tolerance-based, not bitwise.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), weight.len());

    let n = x.len() as f64;
    let mut ss = 0.0f64;
    for v in x {
        ss += (*v as f64) * (*v as f64);
    }
    let mean_ss = (ss / n) as f32;
    let scale = 1.0 / (mean_ss + RMS_NORM_EPS).sqrt();

    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place softmax.
///
/// The maximum is subtracted before exponentiating. This is the single most
/// important detail in the whole file for numerical stability: `exp(1000)`
/// overflows to infinity in `f32` (the max finite value is ~3.4e38, and `exp(88)`
/// already exceeds it), so a row of raw attention scores with any large
/// magnitude would produce `inf`, then `inf / inf = NaN` in the normalisation,
/// and a single `NaN` poisons every downstream value because attention output
/// feeds the residual stream. Because softmax is invariant to a constant shift
/// of its input, subtracting the max leaves the mathematical result unchanged
/// while guaranteeing every exponent argument is <= 0, hence every `exp` is in
/// `(0, 1]`, hence the sum is >= 1 and cannot overflow.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > max {
            max = *v;
        }
    }
    shift_in_place(x, max);

    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = v.exp();
        sum += *v;
    }
    // `sum` is at least 1.0 (the max element contributes exp(0) = 1) and at
    // most x.len() as f32, so this division is always well defined.
    let inv = 1.0 / sum;
    scale_in_place(x, inv);
}

/// SiLU / swish: `x * sigmoid(x)`.
///
/// Written as `x / (1 + exp(-x))` rather than the sigmoid form because the
/// exponential is evaluated once. For large positive `x` this underflows
/// `exp(-x)` towards zero (harmless, result tends to `x`); for large negative
/// `x`, `exp(-x)` overflows to `inf` and the result is `x/inf = -0.0`, which is
/// the correct limit. So there is no input for which this produces `NaN`, and
/// no clamping is needed.
#[inline]
pub fn silu_value(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply [`silu_value`] to one element in place.
#[inline]
pub fn silu(x: &mut f32) {
    *x = silu_value(*x);
}

/// Apply [`silu_value`] to every element in place.
pub fn silu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = silu_value(*x);
    }
}

/// `out = W @ x` where `W` is `d x n` stored row-major and `x` has length `n`.
///
/// Written as `out[i] = dot(W[i], x)`: each output element contracts a row of
/// `W` with the whole activation vector. The dimensions are passed explicitly
/// rather than inferred from slice lengths because a mismatch between the
/// declared shape and the actual buffer length is a bug in the *caller* that
/// should surface loudly, not something to paper over.
pub fn matmul(o: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    debug_assert_eq!(o.len(), n);
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(w.len(), n * d);

    for i in 0..n {
        let row = &w[i * d..i * d + d];
        o[i] = dot(row, x);
    }
}

/// Rotary positional embedding, applied in place to a packed `(n_heads,
/// head_size)` vector.
///
/// The layout is `[head0_element0, head0_element1, ..., head1_element0, ...]`.
/// Within each head, consecutive *pairs* of elements are rotated together, so
/// the pairing is `(2i, 2i+1)` and never crosses a head boundary. Each pair is
/// rotated by `pos * theta^(-2i/head_size)`, which makes the magnitude of each
/// pair rotation invariant and its angle proportional to the position
/// difference — the property that lets attention score two positions using
/// only a relative offset.
///
/// `theta^(-2i/head_size)` is written with the exponent as `(2*i) / head_size`
/// rather than folded into a precomputed table. That costs one `powf` per pair
/// per head per step, which is noise next to the matmuls, and in exchange the
/// differential test is actually testing our RoPE implementation. The
/// checkpoint file does carry a `freq_cis` table; we deliberately skip reading
/// it so that we are not comparing two programs that share a precomputed
/// constant.
pub fn rope(vec: &mut [f32], n_heads: usize, head_size: usize, pos: usize) {
    debug_assert_eq!(vec.len(), n_heads * head_size);
    debug_assert!(head_size % 2 == 0, "RoPE pairs cannot straddle heads");

    let half = head_size / 2;
    let pos = pos as f32;
    for h in 0..n_heads {
        let head = &mut vec[h * head_size..(h + 1) * head_size];
        for i in 0..half {
            let exponent = (2 * i) as f32 / head_size as f32;
            let freq = 1.0 / ROPE_THETA.powf(exponent);
            let angle = pos * freq;
            let (s, c) = angle.sin_cos();

            let v0 = head[2 * i];
            let v1 = head[2 * i + 1];
            head[2 * i] = v0 * c - v1 * s;
            head[2 * i + 1] = v0 * s + v1 * c;
        }
    }
}

/// The three shape parameters that govern attention.
///
/// Grouped into a struct rather than passed as three loose `usize` arguments so
/// that the call site reads `attention(..., &dims, pos, scores)` instead of
/// `attention(..., n_heads, n_kv_heads, head_size, pos, scores)`. At the call
/// site those three numbers are all derived from the model config and are easy
/// to transpose; in the reference C this is a real class of bug. The struct also
/// keeps the argument count inside clippy's limit without an `#[allow]`.
///
/// `try_new` is the single place where the structural validity rules live, and
/// the checkpoint loader calls it too so that the loader and the kernel cannot
/// disagree about what a legal configuration is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDims {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_size: usize,
}

impl AttentionDims {
    /// Derive attention dimensions from a model config, validating them.
    ///
    /// The rules, all of which come from the architecture rather than from this
    /// implementation:
    ///
    /// * `dim % n_heads == 0` — heads must tile the model dimension exactly,
    ///   otherwise some channels would be unowned by any head.
    /// * `n_heads % n_kv_heads == 0` — query heads are partitioned into equally
    ///   sized groups, each sharing one KV head. A non-divisible split would
    ///   leave some groups short a head.
    /// * `head_size` even — RoPE rotates element pairs `(2i, 2i+1)`, so an odd
    ///   head size would leave the last element unpaired and unrotated.
    pub fn try_new(dim: usize, n_heads: usize, n_kv_heads: usize) -> Result<Self, String> {
        if n_heads == 0 || n_kv_heads == 0 {
            return Err(format!(
                "head counts must be positive, got n_heads={n_heads}, n_kv_heads={n_kv_heads}"
            ));
        }
        if dim == 0 {
            return Err("dim must be positive".to_string());
        }
        if dim % n_heads != 0 {
            return Err(format!(
                "dim ({dim}) must be divisible by n_heads ({n_heads})"
            ));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(format!(
                "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            ));
        }
        let head_size = dim / n_heads;
        if head_size % 2 != 0 {
            return Err(format!(
                "head_size ({head_size}) must be even for RoPE pair rotation"
            ));
        }
        Ok(Self {
            n_heads,
            n_kv_heads,
            head_size,
        })
    }

    /// Width of one key or value row: one entry per KV head, each of
    /// `head_size` elements. This is what the KV cache is indexed by.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.head_size * self.n_kv_heads
    }

    /// How many query heads share one KV head.
    #[inline]
    pub fn kv_mul(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Total width of a query vector: one entry per query head.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.head_size * self.n_heads
    }
}

/// Single-token grouped-query attention against the KV cache.
///
/// * `q` is the query for this step: `n_heads * head_size`.
/// * `key_cache` / `value_cache` are `pos + 1` rows of `kv_dim` each, where
///   `kv_dim = head_size * n_kv_heads`. Only the first `pos + 1` rows are live.
/// * `scores` is caller-owned scratch of at least `pos + 1` floats, reused
///   across heads so the decode loop allocates nothing.
///
/// ### Why the head mapping is `h / kv_mul`
///
/// In grouped-query attention several *query* heads share one *key/value* head.
/// `kv_mul = n_heads / n_kv_heads` is the group size. Mapping query head `h` to
/// KV head `h / kv_mul` makes consecutive query heads land in the same group:
/// heads 0,1,2,3 share KV head 0 when `kv_mul == 4`. This matches the
/// reference, which achieves the same thing with
/// `torch.repeat_interleave(x, dim=2, repeats=n_rep)` — `repeat_interleave`
/// maps output index `i` to source index `i / n_rep`, which is the same
/// division. Note the distinction from `h % n_kv_heads`, which would *interleave*
/// the groups instead of blocking them, and which is mutant #3 in
/// `reference/mutation_check.py` precisely because it is a plausible-looking
/// mistake that still "works".
///
/// ### Why the attention window is `pos + 1`
///
/// Position `pos` is being decoded now and positions `0..pos` were decoded
/// before, so the causal mask allows attending to exactly `0..=pos` inclusive.
/// A KV cache works precisely because those earlier keys and values are already
/// computed and stored; re-reading the cache at width `pos` instead of `pos + 1`
/// would silently drop the current token from its own attention, which is
/// mutant #4.
pub fn attention(
    out: &mut [f32],
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    dims: &AttentionDims,
    pos: usize,
    scores: &mut [f32],
) {
    let AttentionDims {
        n_heads, head_size, ..
    } = *dims;
    let kv_dim = dims.kv_dim();
    let kv_mul = dims.kv_mul();

    debug_assert_eq!(out.len(), dims.q_dim());
    debug_assert_eq!(q.len(), dims.q_dim());

    // 1/sqrt(d) is the scaling that keeps dot products of unit-variance
    // activations from growing with head_size. Without it, deeper heads see
    // larger logits, softmax saturates, and gradients vanish. This is a
    // property of the architecture, not a hyperparameter.
    let scale = 1.0 / (head_size as f32).sqrt();

    // Inclusive upper bound: the current token is part of its own context.
    let limit = pos + 1;
    debug_assert!(scores.len() >= limit);
    debug_assert!(key_cache.len() >= limit * kv_dim);
    debug_assert!(value_cache.len() >= limit * kv_dim);

    for h in 0..n_heads {
        let kv = h / kv_mul;
        let q_head = &q[h * head_size..(h + 1) * head_size];
        let live = &mut scores[..limit];

        for t in 0..limit {
            let k_row = &key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size];
            let s = dot(q_head, k_row) * scale;
            live[t] = s;
        }

        softmax(live);

        let out_head = &mut out[h * head_size..(h + 1) * head_size];
        out_head.fill(0.0);
        for t in 0..limit {
            let v_row =
                &value_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size];
            let p = live[t];
            // Fused multiply-add would change rounding; the compiler is not
            // permitted to introduce it without fast-math, which is what keeps
            // this project's numerics reproducible.
            for d in 0..head_size {
                out_head[d] += p * v_row[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative tolerance helper. Used instead of absolute tolerances so that
    /// checks scale with the magnitude of the quantity being checked.
    #[track_caller]
    fn assert_close(actual: f32, expected: f32, tol: f32) {
        let diff = (actual - expected).abs();
        let allowed = tol * (1.0f32).max(expected.abs());
        assert!(
            diff <= allowed,
            "expected {expected}, got {actual} (|diff| = {diff}, allowed {allowed})"
        );
    }

    #[test]
    fn softmax_normalises_and_survives_extreme_logits() {
        // The scores include a value that would overflow `exp` outright
        // (exp(1000) = inf in f32) and one that underflows to zero. A softmax
        // that is correct here is provably using the max-subtraction trick: the
        // result is finite, non-negative, and sums to 1.
        let mut x = [1000.0f32, 1001.0, 999.0, -1e30];
        softmax(&mut x);

        assert!(x.iter().all(|v| v.is_finite()), "softmax produced {x:?}");
        assert!(x.iter().all(|v| *v >= 0.0), "softmax produced {x:?}");

        let sum: f32 = x.iter().sum();
        assert_close(sum, 1.0, 1e-6);

        // Argmax is preserved: shifting by a constant cannot change the
        // ordering, which is the whole reason the shift is legal.
        assert_eq!(
            x.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0,
            1
        );
    }

    #[test]
    fn softmax_of_a_single_element_is_one() {
        let mut x = [42.0f32];
        softmax(&mut x);
        assert_close(x[0], 1.0, 1e-6);
    }

    #[test]
    fn softmax_of_empty_slice_is_a_no_op() {
        let mut x: [f32; 0] = [];
        softmax(&mut x);
        assert!(x.is_empty());
    }

    #[test]
    fn rmsnorm_with_unit_weight_yields_unit_rms() {
        let x = [3.0f32, -1.0, 4.0, -5.0, 2.0, 7.0, -8.0, 0.5];
        let w = [1.0f32; 8];
        let mut out = [0.0f32; 8];
        rmsnorm(&mut out, &x, &w);

        let ss: f32 = out.iter().map(|v| v * v).sum();
        let rms = (ss / out.len() as f32).sqrt();
        // The residual error is RMS_NORM_EPS / mean(x^2): rms(out) is
        // rms(x) / sqrt(mean(x^2) + eps), which is strictly below 1 by that
        // factor. Here mean(x^2) is ~17, so the gap is ~3e-7, far under the
        // 1e-4 tolerance. If this ever fails, the eps handling changed.
        assert_close(rms, 1.0, 1e-4);
    }

    #[test]
    fn rmsnorm_applies_the_learned_scale() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 0.5, 2.0, 0.25];
        let mut a = [0.0f32; 4];
        let mut b = [0.0f32; 4];
        rmsnorm(&mut a, &x, &w);
        rmsnorm(&mut b, &x, &[1.0; 4]);
        for i in 0..4 {
            assert_close(a[i], b[i] * w[i], 1e-6);
        }
    }

    #[test]
    fn rmsnorm_of_all_zero_input_stays_finite() {
        // The eps is what stops this producing 0/0 = NaN. This is exactly the
        // case that makes a hand-rolled rmsnorm without eps wrong.
        let mut out = [0.0f32; 3];
        rmsnorm(&mut out, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert!(out.iter().all(|v| v.is_finite()), "got {out:?}");
    }

    #[test]
    fn rope_is_the_identity_at_position_zero() {
        let n_heads = 3;
        let head_size = 8;
        let original: Vec<f32> = (0..n_heads * head_size)
            .map(|i| i as f32 * 0.37 - 1.0)
            .collect();
        let mut v = original.clone();
        rope(&mut v, n_heads, head_size, 0);
        for i in 0..v.len() {
            assert_close(v[i], original[i], 1e-6);
        }
    }

    #[test]
    fn rope_preserves_pair_norms() {
        // Every rotation is orthonormal, so the length of each (2i, 2i+1) pair
        // must be invariant. This catches a wrong angle *and* a wrong pairing,
        // because misaligned pairs would change their combined length.
        let n_heads = 4;
        let head_size = 16;
        let original: Vec<f32> = (0..n_heads * head_size)
            .map(|i| ((i * 7) % 13) as f32 * 0.11 - 0.4)
            .collect();
        let mut v = original.clone();
        rope(&mut v, n_heads, head_size, 7);

        for h in 0..n_heads {
            for i in 0..head_size / 2 {
                let before = original[h * head_size + 2 * i].powi(2)
                    + original[h * head_size + 2 * i + 1].powi(2);
                let after = v[h * head_size + 2 * i].powi(2) + v[h * head_size + 2 * i + 1].powi(2);
                assert_close(after.sqrt(), before.sqrt(), 1e-5);
            }
        }
    }

    #[test]
    fn rope_never_mixes_neighbouring_heads() {
        // A head boundary at an odd offset would make `2i, 2i+1` straddle two
        // heads. With head_size 4 and a large position the last pair of head 0
        // and the first pair of head 1 would visibly interfere.
        let n_heads = 2;
        let head_size = 4;
        let mut v = [1.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        rope(&mut v, n_heads, head_size, 1);
        // Head 1 starts as (5, 0) and must be rotated by its own angle only.
        let f = 1.0 / ROPE_THETA.powf(0.0); // exponent 0 -> freq 1
        let (s, c) = f.sin_cos();
        assert_close(v[4], 5.0 * c, 1e-6);
        assert_close(v[5], 5.0 * s, 1e-6);
        // Head 0's second pair has exponent 2/4 = 0.5.
        let f2 = 1.0 / ROPE_THETA.powf(0.5);
        let (s2, c2) = f2.sin_cos();
        assert_close(v[2], 0.0 * c2 - 0.0 * s2, 1e-6);
    }

    #[test]
    fn rope_relative_property() {
        // dot(rope(x, p), rope(y, p)) must depend only on p and not on the
        // absolute offset, because the rotation is common to both operands and
        // R^T R = I. This is the property the architecture actually relies on.
        let head_size = 8;
        let x: Vec<f32> = (0..head_size).map(|i| i as f32 * 0.3).collect();
        let y: Vec<f32> = (0..head_size).map(|i| (i as f32 * 0.7).sin()).collect();

        let rot = |v: &[f32], p: usize| {
            let mut t = v.to_vec();
            rope(&mut t, 1, head_size, p);
            t
        };
        let at = |p: usize| dot(&rot(&x, p), &rot(&y, p));

        assert_close(at(5), at(1005), 1e-3);
    }

    #[test]
    fn matmul_matches_a_hand_computed_2x3_case() {
        // W is d x n = 3 x 2, row-major:
        //   [ 1  2  3 ]
        //   [ 4  5  6 ]
        // x has length d = 3.
        // Expected: [1+2*... ] -> [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 2];
        matmul(&mut out, &x, &w, 2, 3);
        assert_close(out[0], 14.0, 1e-6);
        assert_close(out[1], 32.0, 1e-6);
    }

    #[test]
    fn matmul_on_an_empty_output_is_a_no_op() {
        let mut out: [f32; 0] = [];
        matmul(&mut out, &[1.0, 2.0], &[], 0, 2);
    }

    #[test]
    fn dot_is_symmetric_in_the_multiplicands() {
        let a = [1.0f32, -2.5, 3.25, 0.0];
        let b = [0.5f32, 4.0, -1.0, 8.0];
        assert_close(dot(&a, &b), dot(&b, &a), 1e-6);
    }

    #[test]
    fn silu_is_zero_at_zero_and_linear_for_large_positive_input() {
        let mut x = 0.0f32;
        silu(&mut x);
        assert_close(x, 0.0, 1e-7);

        let mut x = 100.0f32;
        silu(&mut x);
        // For x >> 0, sigmoid(x) -> 1, so silu(x) -> x.
        assert_close(x, 100.0, 1e-3);

        let mut x = -100.0f32;
        silu(&mut x);
        // For x << 0 the result tends to 0 from below, and must not be NaN.
        assert!(x.is_finite(), "silu(-100) = {x}");
        assert!(x <= 0.0 && x > -1e-30, "silu(-100) = {x}");
    }

    /// A hand-rolled single-head, single-KV-head attention case where the
    /// answer can be computed on paper.
    ///
    /// q = (1, 0), k0 = (1, 0), k1 = (0, 1), v0 = (2, 0), v1 = (0, 4), pos = 1.
    /// head_size = 2 so the softmax scale is 1/sqrt(2).
    ///   score_0 = 1/sqrt(2), score_1 = 0
    ///   p_0 = e^a / (e^a + 1), p_1 = 1 / (e^a + 1)  with a = 1/sqrt(2)
    ///   out = p_0 * (2, 0) + p_1 * (0, 4)
    #[test]
    fn attention_matches_a_hand_computed_case() {
        let head_size = 2;
        let dims = AttentionDims {
            n_heads: 1,
            n_kv_heads: 1,
            head_size,
        };
        let q = [1.0f32, 0.0];
        let key_cache = [1.0f32, 0.0, 0.0, 1.0]; // t=0, t=1
        let value_cache = [2.0f32, 0.0, 0.0, 4.0];
        let mut out = [0.0f32; 2];
        let mut scores = [0.0f32; 2];

        attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &dims,
            1,
            &mut scores,
        );

        let a = 1.0 / (head_size as f32).sqrt();
        let denom = a.exp() + 1.0;
        let p0 = a.exp() / denom;
        let p1 = 1.0 / denom;
        assert_close(out[0], p0 * 2.0, 1e-6);
        assert_close(out[1], p1 * 4.0, 1e-6);
    }

    #[test]
    fn attention_weights_sum_to_one_over_the_visible_window() {
        // A weaker but broader check: for many random-ish caches, the output
        // must be a convex combination of exactly the rows in `0..=pos`, i.e.
        // it must lie inside the bounding box of those value rows. This catches
        // a softmax that fails to normalise and an off-by-one window that lets a
        // row from beyond `pos` leak in.
        let dims = AttentionDims {
            n_heads: 4,
            n_kv_heads: 2,
            head_size: 4,
        };
        let head_size = dims.head_size;
        let kv_dim = dims.kv_dim();
        let pos = 5;

        let q: Vec<f32> = (0..dims.q_dim()).map(|i| (i as f32 * 0.31).sin()).collect();
        let key_cache: Vec<f32> = (0..(pos + 1) * kv_dim)
            .map(|i| (i as f32 * 0.17).cos())
            .collect();
        let value_cache: Vec<f32> = (0..(pos + 1) * kv_dim)
            .map(|i| (i as f32 * 0.23).sin())
            .collect();

        let mut out = vec![0.0f32; dims.q_dim()];
        let mut scores = vec![0.0f32; pos + 1];
        attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &dims,
            pos,
            &mut scores,
        );

        for h in 0..dims.n_heads {
            let kv = h / dims.kv_mul();
            for d in 0..head_size {
                let vals: Vec<f32> = (0..=pos)
                    .map(|t| value_cache[t * kv_dim + kv * head_size + d])
                    .collect();
                let lo = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let got = out[h * head_size + d];
                assert!(
                    got >= lo - 1e-4 && got <= hi + 1e-4,
                    "head {h} dim {d}: {got} outside [{lo}, {hi}]"
                );
            }
        }
    }

    /// Grouped-query mapping check. With `n_heads = 4`, `n_kv_heads = 2`, heads
    /// 0 and 1 must read KV head 0 and heads 2 and 3 must read KV head 1.
    /// Detect which head each query head used by giving each KV head a
    /// distinctive key/value and checking the output block.
    #[test]
    fn gqa_maps_consecutive_query_heads_to_one_kv_head() {
        let dims = AttentionDims {
            n_heads: 4,
            n_kv_heads: 2,
            head_size: 2,
        };
        let pos = 0;

        // Cache layout is [t][kv][element], flattened. With pos = 0 there is
        // only row 0, and kv_dim = 4, so KV head 0 occupies offsets 0..2 and
        // KV head 1 occupies offsets 2..4. Give each KV head a key of
        // opposite-sign large magnitude and a distinct value: whichever KV head
        // a query head reads is then revealed by the value it produces.
        let key_cache = [10.0f32, 0.0, -10.0, 0.0];
        let value_cache = [1.0f32, 0.0, 2.0, 0.0];
        let q = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let mut out = vec![0.0f32; dims.q_dim()];
        let mut scores = vec![0.0f32; 1];
        attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &dims,
            pos,
            &mut scores,
        );

        // Heads 0,1 -> KV 0 -> value 1. Heads 2,3 -> KV 1 -> value 2.
        assert_close(out[0], 1.0, 1e-5);
        assert_close(out[2], 1.0, 1e-5);
        assert_close(out[4], 2.0, 1e-5);
        assert_close(out[6], 2.0, 1e-5);
    }

    /// Multiquery attention: with `n_kv_heads == 1` every query head must read
    /// the single KV head. This is the degenerate case of the same mapping, and
    /// it is the one real Llama 2 models ship with.
    #[test]
    fn mqa_gives_every_head_the_same_kv_head() {
        let dims = AttentionDims {
            n_heads: 4,
            n_kv_heads: 1,
            head_size: 2,
        };
        assert_eq!(dims.kv_mul(), 4);
        assert_eq!(dims.kv_dim(), 2);

        let key_cache = [1.0f32, 0.0];
        let value_cache = [7.0f32, 0.0];
        let q = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0f32; dims.q_dim()];
        let mut scores = vec![0.0f32; 1];
        attention(
            &mut out,
            &q,
            &key_cache,
            &value_cache,
            &dims,
            0,
            &mut scores,
        );
        for h in 0..4 {
            assert_close(out[h * 2], 7.0, 1e-5);
        }
    }

    #[test]
    fn attention_dims_reject_structurally_invalid_configs() {
        // dim must be divisible by n_heads
        assert!(AttentionDims::try_new(64, 5, 1).is_err());
        // n_heads must be divisible by n_kv_heads
        assert!(AttentionDims::try_new(64, 4, 3).is_err());
        // head_size must be even for RoPE. dim=12 with 4 heads gives head_size=3.
        assert!(AttentionDims::try_new(12, 4, 1).is_err());
        // zeros
        assert!(AttentionDims::try_new(0, 4, 1).is_err());
        assert!(AttentionDims::try_new(64, 0, 1).is_err());
        assert!(AttentionDims::try_new(64, 4, 0).is_err());

        // The legal configs used by the differential suite, with the shape
        // arithmetic spelled out. dim/n_heads is the head size; kv_dim counts
        // KV heads, q_dim counts query heads, and both multiply out to dim.
        let d = AttentionDims::try_new(64, 4, 4).unwrap(); // "tiny": MHA
        assert_eq!(d.head_size, 16);
        assert_eq!((d.kv_dim(), d.kv_mul(), d.q_dim()), (64, 1, 64));
        let d = AttentionDims::try_new(64, 8, 4).unwrap(); // "gqa-2x"
        assert_eq!(d.head_size, 8);
        assert_eq!((d.kv_dim(), d.kv_mul(), d.q_dim()), (32, 2, 64));
        let d = AttentionDims::try_new(96, 8, 2).unwrap(); // "gqa-4x"
        assert_eq!(d.head_size, 12);
        assert_eq!((d.kv_dim(), d.kv_mul(), d.q_dim()), (24, 4, 96));
        let d = AttentionDims::try_new(64, 4, 1).unwrap(); // "mqa"
        assert_eq!((d.kv_dim(), d.kv_mul(), d.q_dim()), (16, 4, 64));
    }
}
