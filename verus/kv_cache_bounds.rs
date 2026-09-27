//! Index safety for tinyinfer's KV-cache slicing.
//!
//! A self-contained Verus development proving the property named in
//! CONTEXT.md section 10, question 7:
//!
//! > KV-cache and attention indexing never goes out of bounds for any
//! > `pos < seq_len`.
//!
//! This is a *standalone model* of the indexing arithmetic, not a
//! verification of the crate. See `verus/README.md` for the line-by-line
//! mapping, the scope of what is and is not covered, and the command that
//! checks this file.
//!
//! Layout:
//!   1. `Shapes`, a ghost model of the config and the derived dimensions
//!   2. arithmetic helpers (Verus's `int` is mathematical; vstd supplies the
//!      bilinear order facts that `nonlinear_arith` deliberately does not)
//!   3. the shape lemmas, including the GQA head mapping
//!   4. the attention read lemma  (src/ops.rs)
//!   5. the per-layer write and slice lemmas  (src/model.rs)
//!   6. the bridge from mathematical `int` back to `usize`
//!   7. the same index expressions, transcribed as real Rust, so that Verus
//!      discharges the actual slice-indexing obligations

#![allow(unused)]

use vstd::prelude::*;
use vstd::arithmetic::div_mod::lemma_fundamental_div_mod;
use vstd::arithmetic::mul::*;

verus! {

// ===========================================================================
// 1. The shapes, as a ghost struct
// ===========================================================================
//
// Mirrors `src/model.rs::Config` plus the quantities `src/ops.rs` derives:
// `kv_dim = head_size * n_kv_heads` (ops.rs:358-360), `kv_mul = n_heads /
// n_kv_heads` (ops.rs:364-366) and the per-layer cache stride
// `seq_len * kv_dim` (model.rs:682).
//
// Fields are `int` rather than `nat` because this Verus version's `nat` is a
// distinct type from the `int` that vstd's arithmetic lemmas are stated over.
// `int` is the closer match for `usize` anyway, since both are signed at the
// type level and bounded only by the range facts we state explicitly.

pub struct Shapes {
    pub n_layers: int,
    pub n_heads: int,
    pub n_kv_heads: int,
    pub head_size: int,
    pub seq_len: int,
}

impl Shapes {
    /// The structural rules, transcribed from the checks in
    /// `AttentionDims::try_new` (ops.rs:323-353) and `Config::validate`
    /// (model.rs:123-144). `lemma_shapes_from_config` below shows that
    /// anything passing those checks really does satisfy this.
    pub open spec fn valid(&self) -> bool {
        &&& self.n_layers > 0
        &&& self.n_heads > 0
        &&& self.n_kv_heads > 0
        &&& self.head_size > 0
        &&& self.seq_len > 0
        &&& self.n_heads % self.n_kv_heads == 0
    }

    /// ops.rs:358-360. Width of one key or value row.
    pub open spec fn kv_dim(&self) -> int {
        self.head_size * self.n_kv_heads
    }

    /// ops.rs:364-366. How many query heads share one KV head.
    pub open spec fn kv_mul(&self) -> int {
        self.n_heads / self.n_kv_heads
    }

    /// ops.rs:370-372.
    pub open spec fn q_dim(&self) -> int {
        self.head_size * self.n_heads
    }

    /// model.rs:682. Floats per layer of the cache.
    pub open spec fn kv_layer_stride(&self) -> int {
        self.seq_len * self.kv_dim()
    }

    /// model.rs:603-607 together with the allocation at model.rs:620-621.
    /// This is the length of `key_cache` and of `value_cache`.
    pub open spec fn cache_len(&self) -> int {
        self.n_layers * self.kv_layer_stride()
    }
}

// ===========================================================================
// 2. Arithmetic helpers
// ===========================================================================
//
// Note on tactic discipline, which is the thing that makes this file verify:
// `by (nonlinear_arith)` in this Verus release decides *polynomial identities*
// only. It cannot reason about the sign of a polynomial under hypotheses, so
// every genuinely multiplicative step below goes through a vstd order lemma
// (`lemma_mul_inequality` and friends), and `nonlinear_arith` is used only to
// rewrite one side of an equation into the other.

/// A `Shapes` value produced by the loader really does satisfy `valid()`.
///
/// This is the honest link between the lemma below and the code: rather than
/// assuming the shape invariants, we derive them from exactly the conditions
/// `Config::validate` (model.rs:123-144) and `AttentionDims::try_new`
/// (ops.rs:323-353) check.
///
/// The one non-obvious step is `head_size > 0`. The loader never checks it,
/// because it computes `head_size = dim / n_heads` (ops.rs:342) after having
/// rejected `dim == 0` (ops.rs:329-331) and `dim % n_heads != 0`
/// (ops.rs:332-335); a divisor of a positive number is positive.
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
    // dim == n_heads * head_size + dim % n_heads
    assert(dim == n_heads * head_size + dim % n_heads) by {
        lemma_fundamental_div_mod(dim, n_heads);
    };
    assert(dim % n_heads == 0);
    if head_size == 0 {
        assert(dim == 0);
        assert(false);
    }
    let s = Shapes { n_layers, n_heads, n_kv_heads, head_size, seq_len };
    assert(s.valid());
    s
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
    assert(s.n_heads == s.n_kv_heads * (s.n_heads / s.n_kv_heads) + s.n_heads % s.n_kv_heads)
        by {
        lemma_fundamental_div_mod(s.n_heads, s.n_kv_heads);
    };
    lemma_mul_is_commutative(s.n_kv_heads, s.n_heads / s.n_kv_heads);
    assert(s.n_heads == s.n_heads / s.n_kv_heads * s.n_kv_heads + s.n_heads % s.n_kv_heads);
    assert(s.n_heads % s.n_kv_heads == 0);
}

/// The GQA mapping `kv = h / kv_mul` (ops.rs:436) lands inside `0..n_kv_heads`.
///
/// Worth stating even though it is not what the question is about, because it
/// shows the limit of this style of proof: mutant #3 replaces `h / kv_mul`
/// with `h % n_kv_heads`, which is *also* in range. Bounds safety cannot tell
/// the two apart; the differential test is what catches that one.
pub proof fn lemma_gqa_head_in_range(s: &Shapes, h: int)
    requires
        s.valid(),
        0 <= h,
        h < s.n_heads,
    ensures h / s.kv_mul() < s.n_kv_heads
{
    lemma_kv_mul_tiles_heads(s);
    // kv_mul is positive because n_kv_heads <= n_heads (a positive divisor of
    // a positive number is at most it) and n_kv_heads divides n_heads.
    lemma_fundamental_div_mod(s.n_heads, s.n_kv_heads);
    assert(0 < s.kv_mul()) by {
        if s.kv_mul() == 0 {
            assert(s.n_heads == 0);
            assert(false);
        }
    };
    // h == kv_mul * (h / kv_mul) + h % kv_mul, with 0 <= h % kv_mul < kv_mul
    assert(h == s.kv_mul() * (h / s.kv_mul()) + h % s.kv_mul()) by {
        lemma_fundamental_div_mod(h, s.kv_mul());
    };
    assert(0 <= h % s.kv_mul() && h % s.kv_mul() < s.kv_mul()) by {
        lemma_fundamental_div_mod(h, s.kv_mul());
    };
    if h / s.kv_mul() >= s.n_kv_heads {
        lemma_mul_inequality(s.n_kv_heads, h / s.kv_mul(), s.kv_mul());
        assert(s.n_kv_heads * s.kv_mul() <= (h / s.kv_mul()) * s.kv_mul());
        lemma_mul_is_commutative(s.n_kv_heads, s.kv_mul());
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
// src/ops.rs:440-444 and 450-452:
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
/// This is the step that turns the caller's guard at model.rs:671-676 into
/// attention's precondition at ops.rs:430-433
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
        // and therefore within the whole layer, which is what the caller
        // actually handed to `attention`
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
// 5. The per-layer write and slice  (src/model.rs)
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

/// The per-layer cache write is in bounds.  src/model.rs:710-712:
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

/// The score scratch window `&mut scores[..limit]` (ops.rs:438) against
/// `scores`, allocated with `seq_len` floats at model.rs:619.
pub proof fn lemma_scores_window_in_scratch(s: &Shapes, pos: int)
    requires
        s.valid(),
        0 <= pos,
        pos < s.seq_len,
    ensures pos + 1 <= s.seq_len
{
    assert(pos + 1 <= s.seq_len);
}

/// The query head slice `&q[h * head_size..(h + 1) * head_size]` (ops.rs:437)
/// against `q`, allocated with `n_heads * head_size` floats (model.rs:610).
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
// program* is section 7: the loop bodies of `ops::attention` and
// `model::State::forward` with the index expressions copied across verbatim,
// so that Verus discharges the actual `Vec` slice-indexing obligations,
// including its implicit no-overflow check on the index arithmetic.
//
// These are proof artifacts, never called. Their value is that a future edit
// to one of these index expressions makes Verus say so.

/// src/ops.rs:435-444, the key-side read.
#[verifier::exec_allows_no_decreases_clause]
pub fn exec_attention_read(
    key_cache: &Vec<f32>,
    value_cache: &Vec<f32>,
    q: &Vec<f32>,
    out: &mut Vec<f32>,
    scores: &mut Vec<f32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_size: usize,
    pos: usize,
) {
    let kv_dim = head_size * n_kv_heads;
    let kv_mul = n_heads / n_kv_heads;
    let limit = pos + 1;

    let mut h: usize = 0;
    while h < n_heads
        invariant
            h <= n_heads,
            n_kv_heads > 0,
            head_size > 0,
            n_heads % n_kv_heads == 0,
            kv_dim == head_size * n_kv_heads,
            limit == pos + 1,
            q.len() == n_heads * head_size,
            out.len() == n_heads * head_size,
            key_cache.len() == limit * kv_dim,
            value_cache.len() == limit * kv_dim,
            scores.len() >= limit,
        decreases n_heads - h,
    {
        // `kv = h / kv_mul` is in range, using exactly the argument of
        // `lemma_gqa_head_in_range` above, restated over `usize`.
        proof {
            assert(n_heads as int == kv_mul as int * n_kv_heads as int) by {
                lemma_fundamental_div_mod(n_heads as int, n_kv_heads as int);
                assert(n_heads as int % n_kv_heads as int == 0);
            };
            assert(0 < kv_mul as int) by {
                if kv_mul as int == 0 {
                    assert(n_heads as int == 0);
                    assert(false);
                }
            };
            assert(h as int == kv_mul as int * (h as int / kv_mul as int)
                    + (h as int % kv_mul as int)) by {
                lemma_fundamental_div_mod(h as int, kv_mul as int);
            };
            assert(0 <= h as int % kv_mul as int
                && (h as int % kv_mul as int) < kv_mul as int) by {
                lemma_fundamental_div_mod(h as int, kv_mul as int);
            };
            if (h as int / kv_mul as int) >= n_kv_heads as int {
                lemma_mul_inequality(n_kv_heads as int, h as int / kv_mul as int, kv_mul as int);
                assert((h as int / kv_mul as int) * kv_mul as int <= h as int);
                assert(h as int >= n_heads as int);
                assert(false);
            }
        }
        assert(kv_mul > 0);
        let kv = h / kv_mul;
        assert(kv < n_kv_heads);

        let _q_head = &q[h * head_size..(h + 1) * head_size];
        let _out_head = &mut out[h * head_size..(h + 1) * head_size];
        let _live = &mut scores[..limit];

        let mut t: usize = 0;
        while t < limit
            invariant
                h <= n_heads,
                t <= limit,
                n_kv_heads > 0,
                head_size > 0,
                kv_dim == head_size * n_kv_heads,
                q.len() == n_heads * head_size,
                out.len() == n_heads * head_size,
                key_cache.len() == limit * kv_dim,
                value_cache.len() == limit * kv_dim,
                scores.len() >= limit,
                kv < n_kv_heads,
            decreases limit - t,
        {
            // The argument of `lemma_attention_row_in_layer` (a), (b) and (c).
            proof {
                assert(t as int <= limit as int);
                assert((kv as int) + 1 <= n_kv_heads as int);
                lemma_mul_inequality((kv as int) + 1, n_kv_heads as int, head_size as int);
                assert((kv as int + 1) * head_size as int
                    <= n_kv_heads as int * head_size as int);
                assert(n_kv_heads as int * head_size as int == kv_dim as int);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= t as int * kv_dim as int + kv_dim as int) by (nonlinear_arith);
                assert(t as int * kv_dim as int + kv_dim as int
                    == (t as int + 1) * kv_dim as int) by (nonlinear_arith);
                assert(t as int + 1 <= limit as int);
                lemma_mul_inequality((t as int) + 1, limit as int, kv_dim as int);
                assert((t as int + 1) * kv_dim as int
                    <= limit as int * kv_dim as int) by (nonlinear_arith);
                assert(t as int * kv_dim as int + (kv as int + 1) * head_size as int
                    <= key_cache.len() as int);
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

/// src/model.rs:710-723, the per-layer cache write and the layer slice handed
/// to `ops::attention`.
#[verifier::exec_allows_no_decreases_clause]
pub fn exec_forward_cache_write_and_layer_slice(
    key_cache: &Vec<f32>,
    value_cache: &Vec<f32>,
    n_layers: usize,
    seq_len: usize,
    head_size: usize,
    n_kv_heads: usize,
    pos: usize,
) {
    let kv_dim = head_size * n_kv_heads;
    let kv_layer_stride = seq_len * kv_dim;

    let mut l: usize = 0;
    while l < n_layers
        invariant
            l <= n_layers,
            n_layers > 0,
            head_size > 0,
            n_kv_heads > 0,
            pos < seq_len,
            kv_dim == head_size * n_kv_heads,
            kv_layer_stride == seq_len * kv_dim,
            key_cache.len() == n_layers * kv_layer_stride,
            value_cache.len() == n_layers * kv_layer_stride,
        decreases n_layers - l,
    {
        // model.rs:710-712
        proof {
            assert((pos as int) < (seq_len as int));
            assert((pos as int) + 1 <= seq_len as int);
            lemma_mul_inequality((pos as int) + 1, seq_len as int, kv_dim as int);
            assert((pos as int + 1) * kv_dim as int <= seq_len as int * kv_dim as int) by (nonlinear_arith);
            assert(seq_len as int * kv_dim as int == kv_layer_stride as int);
            assert(l as int * kv_layer_stride as int + pos as int * kv_dim as int + kv_dim as int
                <= l as int * kv_layer_stride as int + kv_layer_stride as int) by (nonlinear_arith);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= key_cache.len() as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= value_cache.len() as int);
        }

        let base = l * kv_layer_stride + pos * kv_dim;
        let _k_slot = &key_cache[base..base + kv_dim];
        let _v_slot = &value_cache[base..base + kv_dim];

        // model.rs:717-718 and 722-723
        proof {
            assert(l as int + 1 <= n_layers as int);
            lemma_mul_inequality((l as int) + 1, n_layers as int, kv_layer_stride as int);
            assert((l as int + 1) * kv_layer_stride as int
                <= n_layers as int * kv_layer_stride as int) by (nonlinear_arith);
            assert(n_layers as int * kv_layer_stride as int == key_cache.len() as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= key_cache.len() as int);
            assert(n_layers as int * kv_layer_stride as int == value_cache.len() as int);
            assert(l as int * kv_layer_stride as int + kv_layer_stride as int
                <= value_cache.len() as int);
        }

        let layer_start = l * kv_layer_stride;
        let layer_end = layer_start + kv_layer_stride;
        let _layer = &key_cache[layer_start..layer_end];
        let _layer_v = &value_cache[layer_start..layer_end];

        l = l + 1;
    }
}

} // verus!
