//! Index safety for proofinfer's KV-cache and weight-layer slicing.
//!
//! A self-contained Verus development proving the property named in
//! CONTEXT.md section 10, question 7:
//!
//! > KV-cache and attention indexing never goes out of bounds for any
//! > `pos < seq_len`.
//!
//! Plus two extensions:
//! - `State::new`'s checked cache-size arithmetic and `isize::MAX` byte bound
//! - parametric `Weights::layer` slice bounds across all per-layer weight tensors
//!
//! This is a *standalone model* of the indexing arithmetic, not a
//! verification of the crate. `verus/README.md` gives the symbol-scoped
//! mapping, the scope of what is and is not covered, and the command that
//! checks this file.
//!
//! Layout:
//!   1. `Shapes`, a ghost model of the config and the derived dimensions
//!   2. the loader lemmas, deriving the shape invariants and `kv_dim <= dim`
//!      from the checks the loader actually performs
//!   3. shape lemmas, including the GQA head mapping
//!   4. the attention read lemma  (src/ops.rs)
//!   5. the per-layer write, slice, and weight-layer lemmas  (src/model.rs)
//!   6. the bridge from mathematical `int` back to `usize`
//!   7. the same index expressions transcribed as real Rust, including the
//!      composed `forward -> attention` call, `State::new`'s checked cache
//!      length, and `Weights::layer`, so Verus discharges the actual
//!      slice-indexing obligations
//!
//! Tactic discipline, which is what makes this file verify: in this Verus
//! release `by (nonlinear_arith)` decides polynomial *identities* only. It
//! cannot reason about the sign of a polynomial under hypotheses, so every
//! genuinely multiplicative step goes through a vstd order lemma
//! (`lemma_mul_inequality` and friends), `nonlinear_arith` is used only to
//! rewrite one side of an equation into the other, and plain `assert` (the
//! base linear theory) does the rest.

#![allow(unused)]

use vstd::prelude::*;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;
use vstd::arithmetic::mul::*;

verus! {

// ===========================================================================
// 1. The shapes, as a ghost struct
// ===========================================================================
//
// Mirrors `Config` in `src/model.rs` plus the quantities `src/ops.rs` derives:
// `kv_dim = head_size * n_kv_heads` (`ops.rs::AttentionDims::kv_dim`),
// `kv_mul = n_heads / n_kv_heads` (`ops.rs::AttentionDims::kv_mul`) and the
// per-layer cache stride `seq_len * kv_dim` (`model.rs::State::forward`).
//
// Fields are `int` rather than `nat` because this Verus version's `nat` is a
// distinct type from the `int` that vstd's arithmetic lemmas are stated over.
// `int` is the closer match for `usize` anyway, since both are signed at the
// type level and bounded only by the range facts stated explicitly below.

pub struct Shapes {
    pub n_layers: int,
    pub n_heads: int,
    pub n_kv_heads: int,
    pub head_size: int,
    pub seq_len: int,
}

impl Shapes {
    /// The structural rules, transcribed from the checks in
    /// `ops.rs::AttentionDims::try_new` and `model.rs::Config::validate`.
    /// `lemma_shapes_from_config` below shows that anything passing those
    /// checks really does satisfy this.
    pub open spec fn valid(&self) -> bool {
        &&& self.n_layers > 0
        &&& self.n_heads > 0
        &&& self.n_kv_heads > 0
        &&& self.head_size > 0
        &&& self.seq_len > 0
        &&& self.n_heads % self.n_kv_heads == 0
    }

    /// `ops.rs::AttentionDims::kv_dim`. Width of one key or value row.
    pub open spec fn kv_dim(&self) -> int {
        self.head_size * self.n_kv_heads
    }

    /// `ops.rs::AttentionDims::kv_mul`. How many query heads share one KV head.
    pub open spec fn kv_mul(&self) -> int {
        self.n_heads / self.n_kv_heads
    }

    /// `ops.rs::AttentionDims::q_dim`.
    pub open spec fn q_dim(&self) -> int {
        self.head_size * self.n_heads
    }

    /// `model.rs::State::forward`. Floats per layer of the cache.
    pub open spec fn kv_layer_stride(&self) -> int {
        self.seq_len * self.kv_dim()
    }

    /// `model.rs::State::new`. This is the length of `key_cache` and of
    /// `value_cache`.
    pub open spec fn cache_len(&self) -> int {
        self.n_layers * self.kv_layer_stride()
    }
}

// ===========================================================================
// 2. The loader lemmas
// ===========================================================================

/// A `Shapes` value produced by the loader really does satisfy `valid()`.
///
/// This is the honest link between the lemmas below and the code: the shape
/// invariants are *derived* from exactly the conditions
/// `model.rs::Config::validate` and `ops.rs::AttentionDims::try_new` check,
/// rather than assumed.
///
/// The one non-obvious step is `head_size > 0`. The loader never checks it,
/// because `ops.rs::AttentionDims::try_new` computes `head_size = dim / n_heads`
/// only after rejecting `dim == 0` and `dim % n_heads != 0`; a divisor of a
/// positive number is positive.
pub proof fn lemma_shapes_from_config(
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    seq_len: int,
) -> (s: Shapes)
    requires
        n_layers > 0,
        dim > 0,
        n_heads > 0,
        n_kv_heads > 0,
        seq_len > 0,
        dim % n_heads == 0,
        n_heads % n_kv_heads == 0,
    ensures
        s.n_layers == n_layers,
        s.n_heads == n_heads,
        s.n_kv_heads == n_kv_heads,
        s.head_size == dim / n_heads,
        s.seq_len == seq_len,
        s.valid(),
{
    let head_size = dim / n_heads;
    lemma_fundamental_div_mod(dim, n_heads);
    assert(dim == n_heads * head_size + dim % n_heads);
    assert(dim % n_heads == 0);
    if head_size == 0 {
        assert(dim == 0);
        assert(false);
    }
    let s = Shapes { n_layers, n_heads, n_kv_heads, head_size, seq_len };
    assert(s.valid());
    s
}

/// From `ops.rs::AttentionDims::try_new`'s checks alone, `kv_dim = head_size * n_kv_heads`
/// never exceeds `dim` (and therefore never overflows `usize`).
pub proof fn lemma_kv_dim_le_dim(dim: int, n_heads: int, n_kv_heads: int)
    requires
        dim > 0,
        dim <= usize::MAX,
        n_heads > 0,
        n_kv_heads > 0,
        dim % n_heads == 0,
        n_heads % n_kv_heads == 0,
    ensures
        (dim / n_heads) * n_heads == dim,
        (dim / n_heads) * n_kv_heads <= dim,
        (dim / n_heads) * n_kv_heads <= usize::MAX,
{
    let head_size = dim / n_heads;
    lemma_fundamental_div_mod(dim, n_heads);
    assert(dim == n_heads * head_size + dim % n_heads);
    lemma_mul_is_commutative(n_heads, head_size);
    assert(head_size * n_heads == dim);
    if head_size <= 0 {
        lemma_mul_inequality(head_size, 0, n_heads);
        assert(head_size * n_heads <= 0 * n_heads);
        assert(0 * n_heads == 0) by (nonlinear_arith);
        assert(dim <= 0);
        assert(false);
    }

    let kv_mul = n_heads / n_kv_heads;
    lemma_fundamental_div_mod(n_heads, n_kv_heads);
    assert(n_heads == n_kv_heads * kv_mul + n_heads % n_kv_heads);
    assert(n_heads == n_kv_heads * kv_mul);
    if kv_mul <= 0 {
        lemma_mul_inequality(kv_mul, 0, n_kv_heads);
        lemma_mul_is_commutative(kv_mul, n_kv_heads);
        assert(0 * n_kv_heads == 0) by (nonlinear_arith);
        assert(n_heads <= 0);
        assert(false);
    }
    assert(1 <= kv_mul);
    lemma_mul_inequality(1, kv_mul, n_kv_heads);
    lemma_mul_is_commutative(kv_mul, n_kv_heads);
    assert(1 * n_kv_heads == n_kv_heads) by (nonlinear_arith);
    assert(n_kv_heads <= n_heads);

    lemma_mul_inequality(n_kv_heads, n_heads, head_size);
    lemma_mul_is_commutative(n_kv_heads, head_size);
    lemma_mul_is_commutative(n_heads, head_size);
    assert(head_size * n_kv_heads <= head_size * n_heads);
    assert(head_size * n_kv_heads <= dim);
    assert(head_size * n_kv_heads <= usize::MAX);
}

// ===========================================================================
// 3. Shape lemmas
// ===========================================================================

/// `kv_mul * n_kv_heads == n_heads`: the group size really does tile the query
/// heads exactly. This is what `n_heads % n_kv_heads == 0` means.
pub proof fn lemma_kv_mul_tiles_heads(s: &Shapes)
    requires s.valid()
    ensures s.kv_mul() * s.n_kv_heads == s.n_heads
{
    lemma_fundamental_div_mod(s.n_heads, s.n_kv_heads);
    assert(s.n_heads == s.n_kv_heads * (s.n_heads / s.n_kv_heads) + s.n_heads % s.n_kv_heads);
    lemma_mul_is_commutative(s.n_kv_heads, s.n_heads / s.n_kv_heads);
    assert(s.n_heads == s.n_heads / s.n_kv_heads * s.n_kv_heads + s.n_heads % s.n_kv_heads);
    assert(s.n_heads % s.n_kv_heads == 0);
}

/// The GQA mapping `kv = h / kv_mul` (`ops.rs::attention`) lands inside `0..n_kv_heads`.
///
/// Worth stating even though it is not what the question is about, because it
/// also shows the limit of this style of proof: mutant #3 replaces
/// `h / kv_mul` with `h % n_kv_heads`, which is *also* in range. Bounds safety
/// cannot tell the two apart; the differential test is what catches that one.
pub proof fn lemma_gqa_head_in_range(s: &Shapes, h: int)
    requires
        s.valid(),
        0 <= h,
        h < s.n_heads,
    ensures h / s.kv_mul() < s.n_kv_heads
{
    lemma_kv_mul_tiles_heads(s);
    // kv_mul is positive: n_kv_heads is positive and divides the positive
    // n_heads, so it is at most n_heads and the quotient is at least 1.
    assert(0 < s.kv_mul()) by {
        if s.kv_mul() == 0 {
            assert(s.n_heads == 0);
            assert(false);
        }
    };
    // h == kv_mul * (h / kv_mul) + h % kv_mul, with 0 <= h % kv_mul < kv_mul
    lemma_fundamental_div_mod(h, s.kv_mul());
    assert(h == s.kv_mul() * (h / s.kv_mul()) + h % s.kv_mul());
    assert(0 <= h % s.kv_mul() && h % s.kv_mul() < s.kv_mul());
    if h / s.kv_mul() >= s.n_kv_heads {
        lemma_mul_inequality(s.n_kv_heads, h / s.kv_mul(), s.kv_mul());
        assert(s.n_kv_heads * s.kv_mul() <= (h / s.kv_mul()) * s.kv_mul());
        lemma_mul_is_commutative(s.n_kv_heads, s.kv_mul());
        assert(s.kv_mul() * s.n_kv_heads <= s.kv_mul() * (h / s.kv_mul()));
        assert(h >= s.kv_mul() * (h / s.kv_mul()));
        assert(h >= s.kv_mul() * s.n_kv_heads);
        lemma_mul_is_commutative(s.kv_mul(), s.n_heads);
        assert(s.kv_mul() * s.n_kv_heads == s.n_heads);
        assert(h >= s.n_heads);
        assert(false);
    }
}

// ===========================================================================
// 4. The attention read  (src/ops.rs)
// ===========================================================================
//
// `ops.rs::attention`:
//
//   for t in 0..limit {
//       let k_row = &key_cache[t * kv_dim + kv * head_size
//                            ..t * kv_dim + (kv + 1) * head_size];
//
// Claim: given `pos < seq_len`, `0 <= t <= pos` and `kv < n_kv_heads`, the
// range `[t*kv_dim + kv*head_size, t*kv_dim + (kv+1)*head_size)` is inside the
// `pos + 1` live rows, hence inside the layer.

/// The `pos + 1` live rows fit in one layer because `pos < seq_len`.
///
/// This is the step that turns the caller's guard in `model.rs::State::forward`
/// into attention's precondition in `ops.rs::attention`
/// (`limit * kv_dim <= key_cache.len()`).
pub proof fn lemma_limit_rows_fit_in_layer(s: &Shapes, pos: int)
    requires
        s.valid(),
        0 <= pos,
        pos < s.seq_len,
    ensures (pos + 1) * s.kv_dim() <= s.kv_layer_stride()
{
    assert(pos + 1 <= s.seq_len);
    lemma_mul_inequality(pos + 1, s.seq_len, s.kv_dim());
    assert((pos + 1) * s.kv_dim() <= s.seq_len * s.kv_dim());
    assert(s.kv_layer_stride() == s.seq_len * s.kv_dim());
}

/// Every index expression in the attention inner loop is within the buffer.
pub proof fn lemma_attention_row_in_layer(s: &Shapes, pos: int, t: int, kv: int)
    requires
        s.valid(),
        0 <= pos,
        pos < s.seq_len,
        0 <= t,
        t <= pos,
        0 <= kv,
        kv < s.n_kv_heads,
    ensures
        // the slice is non-empty: start < end
        t * s.kv_dim() + kv * s.head_size
            < t * s.kv_dim() + (kv + 1) * s.head_size,
        // the end index is within the pos + 1 live rows
        t * s.kv_dim() + (kv + 1) * s.head_size <= (pos + 1) * s.kv_dim(),
        // and therefore within the whole layer, which is the buffer the
        // caller actually handed to `attention`
        t * s.kv_dim() + (kv + 1) * s.head_size <= s.kv_layer_stride(),
{
    // (a) the row is non-empty because head_size > 0
    lemma_mul_strict_inequality(kv, kv + 1, s.head_size);
    assert(kv * s.head_size < (kv + 1) * s.head_size);
    assert(t * s.kv_dim() + kv * s.head_size < t * s.kv_dim() + (kv + 1) * s.head_size);

    // (b) the row ends inside its own slot, because kv < n_kv_heads
    assert(kv + 1 <= s.n_kv_heads);
    lemma_mul_inequality(kv + 1, s.n_kv_heads, s.head_size);
    assert((kv + 1) * s.head_size <= s.n_kv_heads * s.head_size);
    lemma_mul_is_commutative(s.n_kv_heads, s.head_size);
    assert(s.n_kv_heads * s.head_size == s.kv_dim());
    assert((kv + 1) * s.head_size <= s.kv_dim());
    assert(t * s.kv_dim() + (kv + 1) * s.head_size <= t * s.kv_dim() + s.kv_dim());

    // (c) row t ends at the end of row t, and t <= pos puts that inside the
    //     live window
    assert(t * s.kv_dim() + s.kv_dim() == (t + 1) * s.kv_dim()) by (nonlinear_arith);
    assert(t + 1 <= pos + 1);
    lemma_mul_inequality(t + 1, pos + 1, s.kv_dim());
    assert((t + 1) * s.kv_dim() <= (pos + 1) * s.kv_dim());
    assert(t * s.kv_dim() + (kv + 1) * s.head_size <= (pos + 1) * s.kv_dim());

    // (d) and the live window is inside the layer
    lemma_limit_rows_fit_in_layer(s, pos);
    assert((pos + 1) * s.kv_dim() <= s.kv_layer_stride());
}

// ===========================================================================
// 5. The per-layer write, cache slice, and weight-layer lemmas  (src/model.rs)
// ===========================================================================

/// The layer slice `[l*stride, l*stride + stride)` is inside the whole cache.
pub proof fn lemma_layer_slice_in_cache(s: &Shapes, l: int)
    requires
        s.valid(),
        0 <= l,
        l < s.n_layers,
    ensures l * s.kv_layer_stride() + s.kv_layer_stride() <= s.cache_len()
{
    assert(l + 1 <= s.n_layers);
    lemma_mul_inequality(l + 1, s.n_layers, s.kv_layer_stride());
    assert((l + 1) * s.kv_layer_stride() <= s.n_layers * s.kv_layer_stride());
    assert(l * s.kv_layer_stride() + s.kv_layer_stride() == (l + 1) * s.kv_layer_stride())
        by (nonlinear_arith);
    assert(s.cache_len() == s.n_layers * s.kv_layer_stride());
}

/// Parametric layer slice bound for `model.rs::Weights::layer`, covering all
/// nine per-layer weight accessors (`rms_att`, `rms_ffn`, `wq`, `wk`, `wv`,
/// `wo`, `w1`, `w2`, `w3`).
pub proof fn lemma_layer_tensor_slice(n_layers: int, per_layer: int, l: int)
    requires
        n_layers > 0,
        per_layer > 0,
        0 <= l,
        l < n_layers,
    ensures
        0 <= l * per_layer,
        l * per_layer < l * per_layer + per_layer,
        l * per_layer + per_layer <= n_layers * per_layer,
{
    lemma_mul_inequality(0, l, per_layer);
    assert(0 * per_layer == 0) by (nonlinear_arith);
    assert(0 <= l * per_layer);
    assert(l + 1 <= n_layers);
    lemma_mul_inequality(l + 1, n_layers, per_layer);
    assert((l + 1) * per_layer <= n_layers * per_layer);
    assert(l * per_layer + per_layer == (l + 1) * per_layer) by (nonlinear_arith);
    assert(l * per_layer + per_layer <= n_layers * per_layer);
}

/// The per-layer cache write is in bounds (`model.rs::State::forward`):
///
/// ```text
/// let base = l * kv_layer_stride + pos * kv_dim;
/// self.key_cache[base..base + kv_dim].copy_from_slice(&self.k);
/// self.value_cache[base..base + kv_dim].copy_from_slice(&self.v);
/// ```
///
/// The written range is exactly row `pos` of layer `l`.
pub proof fn lemma_cache_write_in_cache(s: &Shapes, l: int, pos: int)
    requires
        s.valid(),
        0 <= l,
        l < s.n_layers,
        0 <= pos,
        pos < s.seq_len,
    ensures
        l * s.kv_layer_stride() + pos * s.kv_dim() + s.kv_dim() <= s.cache_len(),
        // ... and inside this layer, which is what makes the attention read
        // immediately afterwards observe the value just written.
        l * s.kv_layer_stride() + pos * s.kv_dim() + s.kv_dim()
            <= l * s.kv_layer_stride() + s.kv_layer_stride(),
{
    // pos + 1 rows fit in the layer
    assert(pos + 1 <= s.seq_len);
    lemma_mul_inequality(pos + 1, s.seq_len, s.kv_dim());
    assert((pos + 1) * s.kv_dim() <= s.seq_len * s.kv_dim());
    assert(s.seq_len * s.kv_dim() == s.kv_layer_stride());
    assert((pos + 1) * s.kv_dim() == pos * s.kv_dim() + s.kv_dim()) by (nonlinear_arith);
    assert(l * s.kv_layer_stride() + pos * s.kv_dim() + s.kv_dim()
        <= l * s.kv_layer_stride() + (pos + 1) * s.kv_dim());
    // and the layer is inside the cache
    lemma_layer_slice_in_cache(s, l);
    assert(l * s.kv_layer_stride() + s.kv_layer_stride() <= s.cache_len());
}

/// The end-to-end statement, with the layer offset folded in, which is the
/// form the CONTEXT.md question is phrased in.
pub proof fn lemma_attention_row_in_cache(s: &Shapes, l: int, pos: int, t: int, kv: int)
    requires
        s.valid(),
        0 <= l,
        l < s.n_layers,
        0 <= pos,
        pos < s.seq_len,
        0 <= t,
        t <= pos,
        0 <= kv,
        kv < s.n_kv_heads,
    ensures
        l * s.kv_layer_stride() + t * s.kv_dim() + kv * s.head_size
            < l * s.kv_layer_stride() + t * s.kv_dim() + (kv + 1) * s.head_size,
        l * s.kv_layer_stride() + t * s.kv_dim() + (kv + 1) * s.head_size
            <= s.cache_len(),
{
    lemma_attention_row_in_layer(s, pos, t, kv);
    lemma_layer_slice_in_cache(s, l);
    assert(l * s.kv_layer_stride() + t * s.kv_dim() + (kv + 1) * s.head_size
        <= l * s.kv_layer_stride() + s.kv_layer_stride());
}

/// The score scratch window `&mut scores[..limit]` (`ops.rs::attention`)
/// against `scores`, allocated with `seq_len` floats in `model.rs::State::new`.
pub proof fn lemma_scores_window_in_scratch(s: &Shapes, pos: int)
    requires
        s.valid(),
        0 <= pos,
        pos < s.seq_len,
    ensures pos + 1 <= s.seq_len
{
    assert(pos + 1 <= s.seq_len);
}

/// The query head slice `&q[h * head_size..(h + 1) * head_size]`
/// (`ops.rs::attention`) against `q`, allocated with `n_heads * head_size`
/// floats (`model.rs::State::new`).
pub proof fn lemma_query_head_in_q(s: &Shapes, h: int)
    requires
        s.valid(),
        0 <= h,
        h < s.n_heads,
    ensures
        h * s.head_size < (h + 1) * s.head_size,
        (h + 1) * s.head_size <= s.q_dim(),
{
    lemma_mul_strict_inequality(h, h + 1, s.head_size);
    assert(h * s.head_size < (h + 1) * s.head_size);
    assert(h + 1 <= s.n_heads);
    lemma_mul_inequality(h + 1, s.n_heads, s.head_size);
    assert((h + 1) * s.head_size <= s.n_heads * s.head_size);
    lemma_mul_is_commutative(s.n_heads, s.head_size);
    assert(s.n_heads * s.head_size == s.q_dim());
}

// ===========================================================================
// 6. Bridging mathematical `int` back to machine `usize`
// ===========================================================================
//
// Everything above is in Verus's mathematical `int`, which cannot overflow.
// The real code computes `t * kv_dim + (kv + 1) * head_size` in `usize`, where
// the same expression could in principle wrap. It cannot, for the boring
// reason below: `usize` is bounded by `usize::MAX`, the real buffer lengths
// are `usize`s, and "the index is at most the buffer length" already implies
// "the index fits in a usize", so no intermediate step wrapped.
//
// In practice this obligation is discharged inside section 7 by Verus's own
// no-overflow check on the slice index, not by calling this lemma; it is
// stated separately so the claim is visible rather than implicit.

pub proof fn lemma_index_fits_in_usize(index: int, buf_len: int)
    requires
        buf_len <= usize::MAX,
        index <= buf_len,
    ensures index <= usize::MAX
{
}

// ===========================================================================
// 7. The same index expressions, as real Rust
// ===========================================================================
//
// The lemmas above are arithmetic over a model. What ties them to *this
// program* is what follows: the loop bodies of `ops::attention`,
// `model::State::forward`, `model::State::new`'s checked cache sizing, and
// `model::Weights::layer` with the index expressions copied across verbatim,
// so that Verus discharges the actual slice-indexing obligations, including
// its implicit no-overflow check on the index arithmetic.
//
// `exec_forward_cache_write_and_layer_slice` now calls `exec_attention_read`
// directly on `&key_cache[layer_start..layer_end]` and
// `&value_cache[layer_start..layer_end]`, machine-checking the composition
// across the `forward` -> `attention` boundary.

/// `model.rs::State::new`, the `checked_mul` cache-length arithmetic plus the
/// `isize::MAX` byte-capacity bound enforced by `Vec::try_reserve_exact`.
pub fn exec_checked_cache_len(
    n_layers: usize,
    seq_len: usize,
    kv_dim: usize,
) -> (res: Option<usize>)
    requires
        n_layers > 0,
        seq_len > 0,
        kv_dim > 0,
    ensures
        match res {
            Some(cache) => {
                &&& cache as int == (n_layers as int * seq_len as int) * kv_dim as int
                &&& cache as int == n_layers as int * (seq_len as int * kv_dim as int)
                &&& seq_len as int * kv_dim as int <= usize::MAX
                &&& n_layers as int * (seq_len as int * kv_dim as int) <= usize::MAX
                &&& cache as int * 4 <= isize::MAX
            }
            None => true,
        },
{
    let layer_seq = match n_layers.checked_mul(seq_len) {
        Some(v) => v,
        None => return None,
    };
    let cache = match layer_seq.checked_mul(kv_dim) {
        Some(v) => v,
        None => return None,
    };
    let bytes = match cache.checked_mul(4usize) {
        Some(b) => b,
        None => return None,
    };
    if bytes > isize::MAX as usize {
        return None;
    }
    proof {
        assert(layer_seq as int == n_layers as int * seq_len as int);
        assert(cache as int == layer_seq as int * kv_dim as int);
        assert((n_layers as int * seq_len as int) * kv_dim as int
            == n_layers as int * (seq_len as int * kv_dim as int)) by (nonlinear_arith);
        assert(cache as int == n_layers as int * (seq_len as int * kv_dim as int));
        assert(cache as int <= usize::MAX);
        // Since n_layers >= 1 and seq_len * kv_dim >= 0, seq_len * kv_dim <= cache <= usize::MAX.
        lemma_mul_inequality(0, seq_len as int, kv_dim as int);
        assert(0 * kv_dim as int == 0) by (nonlinear_arith);
        assert(0 <= seq_len as int * kv_dim as int);
        assert(1 <= n_layers as int);
        lemma_mul_inequality(1, n_layers as int, seq_len as int * kv_dim as int);
        assert(1 * (seq_len as int * kv_dim as int) == seq_len as int * kv_dim as int)
            by (nonlinear_arith);
        assert(seq_len as int * kv_dim as int <= cache as int);
        assert(seq_len as int * kv_dim as int <= usize::MAX);
        assert(bytes as int == cache as int * 4);
        assert(bytes as int <= isize::MAX as int);
    }
    Some(cache)
}

/// `model.rs::Weights::layer`, parametric slice bounds across all per-layer
/// weight tensors (`rms_att`, `rms_ffn`, `wq`, `wk`, `wv`, `wo`, `w1`, `w2`, `w3`).
pub fn exec_weight_layer_slice<'a>(
    buf: &'a [f32],
    n_layers: usize,
    l: usize,
    per_layer: usize,
) -> (res: &'a [f32])
    requires
        n_layers > 0,
        per_layer > 0,
        l < n_layers,
        n_layers as int * per_layer as int <= usize::MAX,
        buf.len() == n_layers * per_layer,
    ensures
        res.len() == per_layer,
{
    proof {
        lemma_layer_tensor_slice(n_layers as int, per_layer as int, l as int);
        assert(l as int * per_layer as int + per_layer as int <= buf.len() as int);
        assert(buf.len() as int <= usize::MAX);
    }
    let start = l * per_layer;
    &buf[start..start + per_layer]
}

/// `ops.rs::attention`, the cache reads on `&[f32]` slices.
///
/// `out` and `scores` are modelled as shared slices (`&[f32]`) rather than
/// `&mut [f32]`. Writing to them cannot affect whether an index is in bounds,
/// and Verus (correctly) refuses to carry `s.len()` across a loop iteration
/// for a `&mut`, so mutability is dropped here while slice bounds and
/// overflow checks remain machine-checked.
#[verifier::exec_allows_no_decreases_clause]
pub fn exec_attention_read(
    key_cache: &[f32],
    value_cache: &[f32],
    q: &[f32],
    out: &[f32],
    scores: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_size: usize,
    pos: usize,
)
    requires
        // `ops.rs::AttentionDims::try_new`
        n_heads > 0,
        n_kv_heads > 0,
        n_heads % n_kv_heads == 0,
        head_size > 0,
        // `model.rs::State::new`'s `checked_mul` + `lemma_kv_dim_le_dim`
        head_size * n_kv_heads <= usize::MAX,
        n_heads * head_size <= usize::MAX,
        pos < usize::MAX,
        // the `debug_assert!`s in `ops.rs::attention`
        out.len() == n_heads * head_size,
        q.len() == n_heads * head_size,
        scores.len() >= pos + 1,
        key_cache.len() >= (pos + 1) * (head_size * n_kv_heads),
        value_cache.len() >= (pos + 1) * (head_size * n_kv_heads),
{
    let kv_dim = head_size * n_kv_heads;
    let kv_mul = n_heads / n_kv_heads;
    let limit = pos + 1;
    proof {
        // The cast facts for the three `let` bindings above, established once
        // so the loop bodies do not have to re-derive them.
        assert((head_size * n_kv_heads) as int == head_size as int * n_kv_heads as int);
        assert(kv_dim as int == head_size as int * n_kv_heads as int);
        lemma_mul_is_commutative(n_kv_heads as int, head_size as int);
        assert(n_kv_heads as int * head_size as int == kv_dim as int);
        assert((n_heads * head_size) as int == n_heads as int * head_size as int);
        assert(n_heads as int * head_size as int == q.len() as int);
        assert(n_heads as int * head_size as int == out.len() as int);
        assert((pos + 1) as int == (pos as int) + 1);
        assert(limit as int == (pos as int) + 1);
    }

    let mut h: usize = 0;
    while h < n_heads
        invariant
            h <= n_heads,
            n_heads > 0,
            n_kv_heads > 0,
            head_size > 0,
            n_heads % n_kv_heads == 0,
            kv_dim == head_size * n_kv_heads,
            kv_mul == n_heads / n_kv_heads,
            limit == pos + 1,
            q.len() == n_heads * head_size,
            out.len() == n_heads * head_size,
            scores.len() >= limit,
            key_cache.len() >= limit * kv_dim,
            value_cache.len() >= limit * kv_dim,
        decreases n_heads - h,
    {
        proof {
            // `kv = h / kv_mul` is in range, by exactly the argument of
            // `lemma_gqa_head_in_range` above, restated over `usize`.
            assert((n_heads / n_kv_heads) as int == (n_heads as int) / (n_kv_heads as int));
            lemma_fundamental_div_mod(n_heads as int, n_kv_heads as int);
            assert(n_heads as int == n_kv_heads as int * (n_heads as int / n_kv_heads as int)
                + n_heads as int % n_kv_heads as int);
            assert(n_heads as int % n_kv_heads as int == 0);
            lemma_mul_is_commutative(n_heads as int / n_kv_heads as int, n_kv_heads as int);
            assert(n_heads as int == (kv_mul as int) * (n_kv_heads as int));
            assert(0 < kv_mul as int) by {
                if kv_mul as int == 0 {
                    assert(0 * (n_kv_heads as int) == 0) by (nonlinear_arith);
                    assert(n_heads as int == 0);
                    assert(false);
                }
            };
            assert((h / kv_mul) as int == (h as int) / (kv_mul as int));
            lemma_fundamental_div_mod(h as int, kv_mul as int);
            assert(h as int == (kv_mul as int) * ((h as int) / (kv_mul as int))
                + (h as int) % (kv_mul as int));
            assert(0 <= (h as int) % (kv_mul as int)
                && (h as int) % (kv_mul as int) < kv_mul as int);
            if (h as int) / (kv_mul as int) >= n_kv_heads as int {
                lemma_mul_inequality(n_kv_heads as int, (h as int) / (kv_mul as int), kv_mul as int);
                assert(n_kv_heads as int * kv_mul as int
                    <= ((h as int) / (kv_mul as int)) * kv_mul as int);
                assert((h as int) / (kv_mul as int) * kv_mul as int <= h as int);
                assert(h as int >= n_heads as int);
                assert(false);
            }
        }
        assert(kv_mul > 0);
        let kv = h / kv_mul;
        assert(kv < n_kv_heads);

        // `ops.rs::attention`: the query head, the score window and the
        // output head
        proof {
            assert(n_heads as int * head_size as int == q.len() as int);
            assert(n_heads as int * head_size as int == out.len() as int);
            assert((h as int) + 1 <= n_heads as int);
            lemma_mul_inequality(h as int + 1, n_heads as int, head_size as int);
            assert((h as int + 1) * head_size as int <= n_heads as int * head_size as int);
            assert((h as int + 1) * head_size as int <= q.len() as int);
            assert((h as int + 1) * head_size as int <= out.len() as int);
            assert(limit as int <= scores.len() as int);
        }
        // Explicit no-overflow bounds for the `usize` head slices below.
        proof {
            assert(q.len() as int <= usize::MAX);
            assert(out.len() as int <= usize::MAX);
            assert(h as int * head_size as int + head_size as int
                == (h as int + 1) * head_size as int) by (nonlinear_arith);
            assert(h as int * head_size as int + head_size as int <= q.len() as int);
            assert(h as int * head_size as int + head_size as int <= out.len() as int);
            assert(h as int * head_size as int + head_size as int <= usize::MAX);
            assert(h as int * head_size as int <= usize::MAX);
        }

        let _q_head = &q[h * head_size..(h + 1) * head_size];
        let _out_head = &out[h * head_size..(h + 1) * head_size];
        let _live = &scores[..limit];

        let mut t: usize = 0;
        while t < limit
            invariant
                h <= n_heads,
                t <= limit,
                n_kv_heads > 0,
                head_size > 0,
                n_heads % n_kv_heads == 0,
                kv_dim == head_size * n_kv_heads,
                q.len() == n_heads * head_size,
                scores.len() >= limit,
                key_cache.len() >= limit * kv_dim,
                value_cache.len() >= limit * kv_dim,
                kv < n_kv_heads,
            decreases limit - t,
        {
            // Step (b) of `lemma_attention_row_in_layer`: the row ends inside
            // its own slot, because kv < n_kv_heads.
            proof {
                assert(kv as int + 1 <= n_kv_heads as int);
                lemma_mul_inequality(kv as int + 1, n_kv_heads as int, head_size as int);
                assert((kv as int + 1) * head_size as int <= n_kv_heads as int * head_size as int);
                lemma_mul_is_commutative(n_kv_heads as int, head_size as int);
                assert(n_kv_heads as int * head_size as int == kv_dim as int);
                assert((kv as int + 1) * head_size as int <= kv_dim as int);
            }
            // Step (a): the row is non-empty, and both index terms are
            // non-negative, which the no-overflow checks below need.
            proof {
                lemma_mul_inequality(kv as int, kv as int + 1, head_size as int);
                assert(kv as int * head_size as int <= (kv as int + 1) * head_size as int);
                assert(0 <= kv as int * head_size as int);
                assert(0 <= t as int * kv_dim as int);
                assert(0 <= t as int * kv_dim as int + kv as int * head_size as int);
            }
            // Step (c): row t ends at the end of row t, and t <= limit puts
            // that inside the live window.
            proof {
                assert(t as int <= limit as int);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= t as int * kv_dim as int + kv_dim as int);
                assert(t as int * kv_dim as int + kv_dim as int == (t as int + 1) * kv_dim as int)
                    by (nonlinear_arith);
                assert(t as int + 1 <= limit as int);
                lemma_mul_inequality(t as int + 1, limit as int, kv_dim as int);
                assert((t as int + 1) * kv_dim as int <= limit as int * kv_dim as int);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= limit as int * kv_dim as int);
            }
            // The buffer is at least that long, and the `usize` index
            // arithmetic cannot overflow.
            proof {
                assert(limit as int * kv_dim as int <= key_cache.len() as int);
                assert(limit as int * kv_dim as int <= value_cache.len() as int);
                assert(key_cache.len() as int <= usize::MAX);
                assert(t as int * kv_dim as int <= key_cache.len() as int);
                assert(t as int * kv_dim as int + kv as int * head_size as int
                    <= t as int * kv_dim as int + (kv as int + 1) * head_size as int);
                assert(t as int * kv_dim as int + kv as int * head_size as int
                    <= key_cache.len() as int);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= key_cache.len() as int);
                assert(t as int * kv_dim as int + kv as int * head_size as int
                    <= value_cache.len() as int);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= value_cache.len() as int);
            }

            let _k_row =
                &key_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size];
            let _v_row =
                &value_cache[t * kv_dim + kv * head_size..t * kv_dim + (kv + 1) * head_size];
            t = t + 1;
        }
        h = h + 1;
    }
}

/// `model.rs::State::forward`, the per-layer cache write, layer slice, and
/// composed call into `exec_attention_read` (`ops.rs::attention`).
#[verifier::exec_allows_no_decreases_clause]
pub fn exec_forward_cache_write_and_layer_slice(
    key_cache: &Vec<f32>,
    value_cache: &Vec<f32>,
    q: &[f32],
    out: &[f32],
    scores: &[f32],
    n_layers: usize,
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_size: usize,
    pos: usize,
)
    requires
        // `model.rs::Config::validate` and `ops.rs::AttentionDims::try_new`
        n_layers > 0,
        n_heads > 0,
        head_size > 0,
        n_kv_heads > 0,
        n_heads % n_kv_heads == 0,
        // `model.rs::State::forward`'s guard
        pos < seq_len,
        // `model.rs::State::new`'s `checked_mul` + `lemma_kv_dim_le_dim`
        head_size * n_kv_heads <= usize::MAX,
        n_heads * head_size <= usize::MAX,
        (n_layers * seq_len) * (head_size * n_kv_heads) <= usize::MAX,
        seq_len * (head_size * n_kv_heads) <= usize::MAX,
        // the resulting allocations in `model.rs::State::new`
        q.len() == n_heads * head_size,
        out.len() == n_heads * head_size,
        scores.len() >= seq_len,
        key_cache.len() == n_layers * (seq_len * (head_size * n_kv_heads)),
        value_cache.len() == n_layers * (seq_len * (head_size * n_kv_heads)),
{
    let kv_dim = head_size * n_kv_heads;
    let kv_layer_stride = seq_len * kv_dim;
    proof {
        assert(kv_dim as int == head_size as int * n_kv_heads as int);
        assert((seq_len * (head_size * n_kv_heads)) as int == seq_len as int * kv_dim as int);
        assert(seq_len as int * kv_dim as int == kv_layer_stride as int);
        assert((n_layers * (seq_len * (head_size * n_kv_heads))) as int
            == n_layers as int * kv_layer_stride as int);
        assert((n_layers as int) * (seq_len as int) * kv_dim as int <= usize::MAX);
        assert(n_layers as int * kv_layer_stride as int == key_cache.len() as int);
        assert(n_layers as int * kv_layer_stride as int == value_cache.len() as int);
    }

    let mut l: usize = 0;
    while l < n_layers
        invariant
            l <= n_layers,
            n_layers > 0,
            n_heads > 0,
            head_size > 0,
            n_kv_heads > 0,
            n_heads % n_kv_heads == 0,
            pos < seq_len,
            head_size * n_kv_heads <= usize::MAX,
            n_heads * head_size <= usize::MAX,
            kv_dim == head_size * n_kv_heads,
            kv_layer_stride == seq_len * kv_dim,
            q.len() == n_heads * head_size,
            out.len() == n_heads * head_size,
            scores.len() >= seq_len,
            key_cache.len() == n_layers * kv_layer_stride,
            value_cache.len() == n_layers * kv_layer_stride,
        decreases n_layers - l,
    {
        // `model.rs::State::forward` cache write
        proof {
            assert((pos as int) < (seq_len as int));
            assert((pos as int) + 1 <= seq_len as int);
            lemma_mul_inequality((pos as int) + 1, seq_len as int, kv_dim as int);
            assert((pos as int + 1) * kv_dim as int <= seq_len as int * kv_dim as int);
            assert(seq_len as int * kv_dim as int == kv_layer_stride as int);
            assert((pos as int + 1) * kv_dim as int
                == pos as int * kv_dim as int + kv_dim as int) by (nonlinear_arith);
            assert(l as int * kv_layer_stride as int + pos as int * kv_dim as int + kv_dim as int
                <= l as int * kv_layer_stride as int + kv_layer_stride as int);
            assert(l as int + 1 <= n_layers as int);
            lemma_mul_inequality(l as int + 1, n_layers as int, kv_layer_stride as int);
            assert((l as int + 1) * kv_layer_stride as int
                <= n_layers as int * kv_layer_stride as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                == (l as int + 1) * kv_layer_stride as int) by (nonlinear_arith);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= n_layers as int * kv_layer_stride as int);
            assert(n_layers as int * kv_layer_stride as int == key_cache.len() as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= key_cache.len() as int);
            assert(n_layers as int * kv_layer_stride as int == value_cache.len() as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= value_cache.len() as int);
        }

        let base = l * kv_layer_stride + pos * kv_dim;
        let _k_slot = &key_cache[base..base + kv_dim];
        let _v_slot = &value_cache[base..base + kv_dim];

        // `model.rs::State::forward` layer slice -> composed `ops.rs::attention` read
        let layer_start = l * kv_layer_stride;
        let layer_end = layer_start + kv_layer_stride;
        let layer_k = &key_cache[layer_start..layer_end];
        let layer_v = &value_cache[layer_start..layer_end];

        proof {
            assert(layer_k.len() == kv_layer_stride);
            assert(layer_v.len() == kv_layer_stride);
            assert((pos as int + 1) * (head_size as int * n_kv_heads as int)
                <= kv_layer_stride as int);
            assert(pos < usize::MAX);
            assert(scores.len() >= pos + 1);
        }

        exec_attention_read(
            layer_k,
            layer_v,
            q,
            out,
            scores,
            n_heads,
            n_kv_heads,
            head_size,
            pos,
        );

        l = l + 1;
    }
}

// ===========================================================================
// 8. Q8_0 quantized block and scale indexing bounds  (src/ops.rs)
// ===========================================================================
//
// Proves that every `i8` weight block slice and `f32` per-group scale slice in
// `ops.rs::dot_q80` and `ops.rs::matmul_q80` is non-empty, in bounds, and
// overflow-free for any `d % group_size == 0`.

/// `d / group_size` tiles `d` exactly when `d % group_size == 0`.
pub proof fn lemma_q80_groups_tile_dim(d: int, group_size: int)
    requires
        d > 0,
        group_size > 0,
        d % group_size == 0,
    ensures
        (d / group_size) * group_size == d,
        d / group_size > 0,
{
    lemma_fundamental_div_mod(d, group_size);
    assert(d == group_size * (d / group_size) + d % group_size);
    lemma_mul_is_commutative(group_size, d / group_size);
    assert((d / group_size) * group_size == d);
    if d / group_size <= 0 {
        lemma_mul_inequality(d / group_size, 0, group_size);
        assert((d / group_size) * group_size <= 0 * group_size);
        assert(0 * group_size == 0) by (nonlinear_arith);
        assert(d <= 0);
        assert(false);
    }
}

/// Every group slice `[g * group_size .. (g + 1) * group_size)` fits in `0..d`.
pub proof fn lemma_q80_group_slice_in_row(d: int, group_size: int, g: int)
    requires
        d > 0,
        group_size > 0,
        d % group_size == 0,
        0 <= g,
        g < d / group_size,
    ensures
        0 <= g * group_size,
        g * group_size < g * group_size + group_size,
        g * group_size + group_size <= d,
{
    lemma_q80_groups_tile_dim(d, group_size);
    lemma_mul_inequality(0, g, group_size);
    assert(0 * group_size == 0) by (nonlinear_arith);
    assert(0 <= g * group_size);
    assert(g + 1 <= d / group_size);
    lemma_mul_inequality(g + 1, d / group_size, group_size);
    assert((g + 1) * group_size <= (d / group_size) * group_size);
    assert(g * group_size + group_size == (g + 1) * group_size) by (nonlinear_arith);
    assert(g * group_size + group_size <= d);
}

/// Every row `i < n` of a Q8_0 matrix has both its `i8` slice `[i*d..i*d+d)`
/// and its scale slice `[i*row_groups..i*row_groups+row_groups)` in bounds.
pub proof fn lemma_q80_row_and_scale_in_matrix(n: int, d: int, group_size: int, i: int)
    requires
        n > 0,
        d > 0,
        group_size > 0,
        d % group_size == 0,
        0 <= i,
        i < n,
    ensures
        0 <= i * d,
        i * d < i * d + d,
        i * d + d <= n * d,
        0 <= i * (d / group_size),
        i * (d / group_size) < i * (d / group_size) + (d / group_size),
        i * (d / group_size) + (d / group_size) <= n * (d / group_size),
{
    lemma_q80_groups_tile_dim(d, group_size);
    lemma_layer_tensor_slice(n, d, i);
    lemma_layer_tensor_slice(n, d / group_size, i);
}

/// `ops.rs::dot_q80`, executable slice and scale indexing over real `&[i8]`
/// and `&[f32]` slices.
pub fn exec_q80_dot_bounds(
    wq: &[i8],
    ws: &[f32],
    xq: &[i8],
    xs: &[f32],
    d: usize,
    group_size: usize,
)
    requires
        d > 0,
        group_size > 0,
        d % group_size == 0,
        wq.len() == d,
        xq.len() == d,
        ws.len() == d / group_size,
        xs.len() == d / group_size,
{
    let n_groups = d / group_size;
    let mut g: usize = 0;
    while g < n_groups
        invariant
            g <= n_groups,
            d > 0,
            group_size > 0,
            d % group_size == 0,
            n_groups == d / group_size,
            wq.len() == d,
            xq.len() == d,
            ws.len() == n_groups,
            xs.len() == n_groups,
        decreases n_groups - g,
    {
        proof {
            lemma_q80_group_slice_in_row(d as int, group_size as int, g as int);
            assert(d as int <= usize::MAX);
            assert(g as int * group_size as int + group_size as int <= d as int);
            assert(g as int * group_size as int <= usize::MAX);
            assert(g as int * group_size as int + group_size as int <= usize::MAX);
        }
        let start = g * group_size;
        let end = start + group_size;
        let _wg = &wq[start..end];
        let _xg = &xq[start..end];
        let _ws = ws[g];
        let _xs = xs[g];
        g = g + 1;
    }
}

/// `ops.rs::matmul_q80`, executable row/scale slicing composed directly with
/// `exec_q80_dot_bounds` (`ops.rs::dot_q80`).
pub fn exec_q80_matmul_bounds(
    wq: &[i8],
    ws: &[f32],
    xq: &[i8],
    xs: &[f32],
    n: usize,
    d: usize,
    group_size: usize,
)
    requires
        n > 0,
        d > 0,
        group_size > 0,
        d % group_size == 0,
        n as int * d as int <= usize::MAX,
        n as int * (d as int / group_size as int) <= usize::MAX,
        wq.len() == n * d,
        ws.len() == n * (d / group_size),
        xq.len() == d,
        xs.len() == d / group_size,
{
    let row_groups = d / group_size;
    let mut i: usize = 0;
    while i < n
        invariant
            i <= n,
            n > 0,
            d > 0,
            group_size > 0,
            d % group_size == 0,
            row_groups == d / group_size,
            n as int * d as int <= usize::MAX,
            n as int * row_groups as int <= usize::MAX,
            wq.len() == n * d,
            ws.len() == n * row_groups,
            xq.len() == d,
            xs.len() == row_groups,
        decreases n - i,
    {
        proof {
            lemma_q80_row_and_scale_in_matrix(n as int, d as int, group_size as int, i as int);
            assert(i as int * d as int + d as int <= wq.len() as int);
            assert(wq.len() as int <= usize::MAX);
            assert(i as int * row_groups as int + row_groups as int <= ws.len() as int);
            assert(ws.len() as int <= usize::MAX);
        }
        let q_row = &wq[i * d..i * d + d];
        let s_row = &ws[i * row_groups..i * row_groups + row_groups];
        exec_q80_dot_bounds(q_row, s_row, xq, xs, d, group_size);
        i = i + 1;
    }
}

} // verus!
