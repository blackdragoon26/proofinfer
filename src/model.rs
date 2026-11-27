//! Model configuration, checkpoint loading, KV-cached decoding.

use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::ops::{self, AttentionDims};

/// Everything that goes wrong while reading a checkpoint.
///
/// The loader treats the file as untrusted: the header dictates allocation
/// sizes, so a malformed or hostile file must produce an `Err` and never a
/// panic, a wrapped-around allocation size, or a short buffer that later gets
/// indexed out of bounds. Every variant therefore carries enough context to
/// say *which* invariant was violated and by how much.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// The file could not be opened or read.
    Io { path: String, msg: String },
    /// The file ended before a required field or tensor was fully read.
    Truncated {
        what: &'static str,
        need: usize,
        have: usize,
    },
    /// A size product exceeded the address space. The declared tensor is
    /// larger than could ever be allocated, so the header is nonsense.
    Overflow { what: &'static str },
    /// A header field was zero, negative where it cannot be, or failed a
    /// structural divisibility rule.
    InvalidHeader(String),
    /// The file had bytes left over after the last expected tensor. This is the
    /// signal that a tensor is missing, misordered, or the wrong size, which is
    /// exactly the class of bug a format this ad-hoc is prone to.
    TrailingBytes(usize),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io { path, msg } => write!(f, "cannot read {path}: {msg}"),
            LoadError::Truncated { what, need, have } => write!(
                f,
                "file truncated while reading {what}: needed {need} bytes, {have} available"
            ),
            LoadError::Overflow { what } => {
                write!(
                    f,
                    "size overflow computing {what}: header is not representable"
                )
            }
            LoadError::InvalidHeader(m) => write!(f, "invalid header: {m}"),
            LoadError::TrailingBytes(n) => write!(
                f,
                "{n} unexpected trailing bytes: the tensor list does not match the header"
            ),
        }
    }
}

impl Error for LoadError {}

/// Errors from running the model.
///
/// Separate from [`LoadError`] because these are not about a bad file: they
/// are about the *caller* asking for something the model cannot do, such as a
/// token outside the vocabulary or a position beyond the KV cache. Both are
/// recoverable and both are the caller's mistake, so they get their own type
/// rather than being folded into the loader's error space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    TokenOutOfRange { token: u32, vocab_size: usize },
    PositionOutOfRange { pos: usize, seq_len: usize },
    TooManyTokens { requested: usize, seq_len: usize },
    EmptyPrompt,
    StateMismatch { detail: String },
    NonFiniteLogit { pos: usize },
    InvalidSampleConfig { detail: String },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::TokenOutOfRange { token, vocab_size } => write!(
                f,
                "token {token} is outside the vocabulary (0..{vocab_size})"
            ),
            RunError::PositionOutOfRange { pos, seq_len } => write!(
                f,
                "position {pos} is past the end of the KV cache (seq_len = {seq_len})"
            ),
            RunError::TooManyTokens { requested, seq_len } => write!(
                f,
                "{requested} tokens will not fit in a context of {seq_len}"
            ),
            RunError::EmptyPrompt => write!(f, "the prompt contains no tokens"),
            RunError::StateMismatch { detail } => write!(f, "state mismatch: {detail}"),
            RunError::NonFiniteLogit { pos } => {
                write!(
                    f,
                    "non-finite (NaN or inf) logit produced at position {pos}"
                )
            }
            RunError::InvalidSampleConfig { detail } => {
                write!(f, "invalid sampling configuration: {detail}")
            }
        }
    }
}

impl Error for RunError {}

/// Model architecture, as declared by the checkpoint header.
///
/// This is a plain data struct with no hidden derived state. The derived
/// quantities (`kv_dim`, `head_size`, element counts) live in
/// [`AttentionDims`] and in the size computations in [`Weights::from_bytes`],
/// so there is exactly one place that knows each answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Config {
    /// Structural validation shared by the loader and the CLI.
    ///
    /// Note that `AttentionDims::try_new` is reused for the head geometry rules
    /// rather than reimplemented, so the loader and the attention kernel cannot
    /// drift apart on what a legal model is.
    pub(crate) fn validate(&self) -> Result<AttentionDims, LoadError> {
        if self.dim == 0 {
            return Err(LoadError::InvalidHeader("dim must be positive".into()));
        }
        if self.hidden_dim == 0 {
            return Err(LoadError::InvalidHeader(
                "hidden_dim must be positive".into(),
            ));
        }
        if self.n_layers == 0 {
            return Err(LoadError::InvalidHeader("n_layers must be positive".into()));
        }
        if self.vocab_size == 0 {
            return Err(LoadError::InvalidHeader(
                "vocab_size must be positive".into(),
            ));
        }
        if self.seq_len == 0 {
            return Err(LoadError::InvalidHeader("seq_len must be positive".into()));
        }
        self.attention_dims()
    }

    pub fn attention_dims(&self) -> Result<AttentionDims, LoadError> {
        AttentionDims::try_new(self.dim, self.n_heads, self.n_kv_heads)
            .map_err(LoadError::InvalidHeader)
    }

    /// Whether the file carried a separate `wcls` tensor.
    ///
    /// The legacy format encodes this in the sign of `vocab_size`, which is a
    /// genuinely unfortunate design but one we have to match. A negative value
    /// means the classifier is *not* tied to the token embedding.
    pub fn is_tied(&self, raw_vocab: i32) -> bool {
        raw_vocab > 0
    }
}

/// A bounds-checked cursor over the checkpoint bytes.
///
/// Every read goes through this type, which is the mechanism that makes the
/// loader total: there is no way to advance the cursor past the end of the
/// buffer, so there is no path from a malformed file to an out-of-bounds slice.
///
/// The important secondary property is that **allocation size is bounded by
/// file size**. Before any `Vec` is created the code asks "are there really
/// `n` floats left in the file?", and only then allocates. A header claiming a
/// 10^18-element tensor does not cause a 4-exabyte allocation; it fails
/// immediately. Without that ordering, validating the header alone is not enough,
/// because `Vec::with_capacity` on an attacker-chosen length is itself a
/// denial-of-service primitive.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// Require `n` bytes to be available, or report what was short.
    fn require(&self, n: usize, what: &'static str) -> Result<(), LoadError> {
        if self.remaining() < n {
            Err(LoadError::Truncated {
                what,
                need: n,
                have: self.remaining(),
            })
        } else {
            Ok(())
        }
    }

    /// Read one little-endian `i32`.
    ///
    /// Little-endian is the format's convention (it is what every platform that
    /// actually runs this writes), so the bytes are assembled explicitly rather
    /// than transmuted. A 4-byte transmute would work on x86 and arm but would
    /// be wrong on a big-endian target, and this project is meant to be read as
    /// code that does not contain clever tricks.
    fn read_i32(&mut self, what: &'static str) -> Result<i32, LoadError> {
        self.require(4, what)?;
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read `n` little-endian `f32` values.
    ///
    /// The byte count is computed with `checked_mul` so that a header claiming
    /// an absurd element count reports an overflow instead of wrapping to a
    /// small number. A wrapped count would pass the `require` check and then
    /// produce a `Vec` far shorter than the code believes it has, which is the
    /// classic out-of-bounds-read bug in a file parser.
    fn read_f32s(&mut self, n: usize, what: &'static str) -> Result<Vec<f32>, LoadError> {
        let nbytes = n.checked_mul(4).ok_or(LoadError::Overflow { what })?;
        self.require(nbytes, what)?;
        let bytes = &self.data[self.pos..self.pos + nbytes];
        self.pos += nbytes;
        // `chunks_exact` + `map` + `collect` beats an index loop with a bounds
        // check per element, and the slice length is provably a multiple of 4.
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Skip `n` floats without materialising them.
    fn skip_f32s(&mut self, n: usize, what: &'static str) -> Result<(), LoadError> {
        let nbytes = n.checked_mul(4).ok_or(LoadError::Overflow { what })?;
        self.require(nbytes, what)?;
        self.pos += nbytes;
        Ok(())
    }
}

/// Multiply a run of dimensions, treating any overflow as a load error.
///
/// This is the only place tensor element counts are computed, so the overflow
/// rule is stated once. The `what` label names the tensor for the error message
/// because "size overflow" without a tensor name is not actionable.
pub(crate) fn checked_tensor_len(what: &'static str, dims: &[usize]) -> Result<usize, LoadError> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(LoadError::Overflow { what })
}

/// All model parameters, in the layout the file specifies.
///
/// These are stored as flat `Vec<f32>` rather than nested `Vec<Vec<f32>>`.
/// The per-layer tensors are concatenated across layers in the file, so a flat
/// buffer is both a faithful representation of the file and one fewer
/// indirection in the hot loop; [`Weights::layer`] hands out the per-layer
/// slice.
///
/// Fields are public for test and tool inspection. Callers that construct or
/// mutate `Weights` fields directly rather than using [`Weights::load`] or
/// [`Weights::from_bytes`] are responsible for preserving buffer-length
/// invariants relative to `config`.
#[derive(Debug, Clone)]
pub struct Weights {
    pub config: Config,
    /// `vocab x dim`, row-major. Also serves as the classifier when tied.
    pub tok_embedding: Vec<f32>,
    pub rms_att: Vec<f32>,
    /// `n_layers x dim x dim`
    pub wq: Vec<f32>,
    /// `n_layers x kv_dim x dim`
    pub wk: Vec<f32>,
    /// `n_layers x kv_dim x dim`
    pub wv: Vec<f32>,
    /// `n_layers x dim x dim`
    pub wo: Vec<f32>,
    pub rms_ffn: Vec<f32>,
    /// `n_layers x hidden_dim x dim`
    pub w1: Vec<f32>,
    /// `n_layers x dim x hidden_dim`
    pub w2: Vec<f32>,
    /// `n_layers x hidden_dim x dim`
    pub w3: Vec<f32>,
    pub rms_final: Vec<f32>,
    /// `vocab x dim`, present only when the classifier is untied. `None` is
    /// not the same as "all zeros": the tied case genuinely has no such tensor
    /// and must reuse `tok_embedding`.
    pub wcls: Option<Vec<f32>>,
}

/// Element counts for every tensor in the file, in file order.
///
/// Computed up front and in full, before a single byte of weight data is read.
/// The ordering is deliberate:
///
/// * **Overflow is reported first, and by name.** If a header makes any single
///   tensor unrepresentable, we say "size overflow computing wq" instead of
///   first tripping over whichever earlier tensor merely happened to be large
///   and reporting a confusing "needed 8589934584 bytes, have 65536".
/// * **Nothing is ever allocated from an unvalidated number.** Every `Vec`
///   below is created with a length that has already been through
///   `checked_mul`, so the loader's memory use is bounded by the *file* size
///   rather than by the header.
#[derive(Debug, Clone, Copy)]
struct TensorSizes {
    tok_embedding: usize,
    rms_att: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    rms_ffn: usize,
    w1: usize,
    w2: usize,
    w3: usize,
    rms_final: usize,
    freq_cis: usize,
    wcls: Option<usize>,
}

impl TensorSizes {
    fn compute(config: &Config, dims: &AttentionDims, tied: bool) -> Result<Self, LoadError> {
        let (n_layers, d, kv, hid) = (
            config.n_layers,
            config.dim,
            dims.kv_dim(),
            config.hidden_dim,
        );
        let v = config.vocab_size;
        Ok(Self {
            tok_embedding: checked_tensor_len("token_embedding", &[v, d])?,
            rms_att: checked_tensor_len("rms_att", &[n_layers, d])?,
            wq: checked_tensor_len("wq", &[n_layers, d, d])?,
            wk: checked_tensor_len("wk", &[n_layers, kv, d])?,
            wv: checked_tensor_len("wv", &[n_layers, kv, d])?,
            wo: checked_tensor_len("wo", &[n_layers, d, d])?,
            rms_ffn: checked_tensor_len("rms_ffn", &[n_layers, d])?,
            w1: checked_tensor_len("w1", &[n_layers, hid, d])?,
            w2: checked_tensor_len("w2", &[n_layers, d, hid])?,
            w3: checked_tensor_len("w3", &[n_layers, hid, d])?,
            rms_final: checked_tensor_len("rms_final", &[d])?,
            freq_cis: checked_tensor_len("freq_cis", &[config.seq_len, dims.head_size / 2])?,
            wcls: if tied {
                None
            } else {
                Some(checked_tensor_len("wcls", &[v, d])?)
            },
        })
    }
}

impl Weights {
    /// Read a checkpoint from disk.
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path).map_err(|e| LoadError::Io {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Self::from_bytes(&data)
    }

    /// Parse a checkpoint from an in-memory buffer.
    ///
    /// Takes bytes rather than a path so that the entire parser, including the
    /// error paths, is reachable from tests without touching the filesystem.
    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        let mut r = Reader::new(data);

        // --- header -------------------------------------------------------
        // Read as i32 because the sign of vocab_size carries the weight-tying
        // flag. Everything else must end up positive.
        let h_dim = r.read_i32("header: dim")?;
        let h_hidden = r.read_i32("header: hidden_dim")?;
        let h_layers = r.read_i32("header: n_layers")?;
        let h_heads = r.read_i32("header: n_heads")?;
        let h_kv_heads = r.read_i32("header: n_kv_heads")?;
        let h_vocab = r.read_i32("header: vocab_size")?;
        let h_seq = r.read_i32("header: seq_len")?;

        // `unsigned_abs` rather than `abs` because i32::MIN has no positive
        // counterpart, and `.abs()` on it would itself panic in debug builds.
        // i32::MIN as a vocab size is nonsense, but it is exactly the sort of
        // value a fuzzer produces, and the right answer is "error", not a
        // panic inside the panic handler.
        let vocab = h_vocab.unsigned_abs() as usize;
        if h_vocab == 0 {
            return Err(LoadError::InvalidHeader(
                "vocab_size must be non-zero".into(),
            ));
        }
        if h_dim <= 0
            || h_hidden <= 0
            || h_layers <= 0
            || h_heads <= 0
            || h_kv_heads <= 0
            || h_seq <= 0
        {
            return Err(LoadError::InvalidHeader(format!(
                "all header fields must be positive, got dim={h_dim} hidden_dim={h_hidden} \
                 n_layers={h_layers} n_heads={h_heads} n_kv_heads={h_kv_heads} seq_len={h_seq}"
            )));
        }

        let config = Config {
            dim: h_dim as usize,
            hidden_dim: h_hidden as usize,
            n_layers: h_layers as usize,
            n_heads: h_heads as usize,
            n_kv_heads: h_kv_heads as usize,
            vocab_size: vocab,
            seq_len: h_seq as usize,
        };
        // Raises dim % n_heads, n_heads % n_kv_heads, and even head_size.
        let dims = config.validate()?;

        // --- weights ------------------------------------------------------
        // The per-layer tensors are stored transposed relative to how the
        // forward pass wants them: `wq` is `n_layers x dim x dim` with the
        // *input* dimension contiguous, because that is the layout the
        // reference's `nn.Linear` produces and therefore the layout in the
        // file. `matmul` contracts a row of `W` with the activation, so a row
        // is one output neuron over all inputs.
        let sizes = TensorSizes::compute(&config, &dims, config.is_tied(h_vocab))?;

        let tok_embedding = r.read_f32s(sizes.tok_embedding, "tensor: token_embedding")?;
        let rms_att = r.read_f32s(sizes.rms_att, "tensor: rms_att")?;
        let wq = r.read_f32s(sizes.wq, "tensor: wq")?;
        let wk = r.read_f32s(sizes.wk, "tensor: wk")?;
        let wv = r.read_f32s(sizes.wv, "tensor: wv")?;
        let wo = r.read_f32s(sizes.wo, "tensor: wo")?;
        let rms_ffn = r.read_f32s(sizes.rms_ffn, "tensor: rms_ffn")?;
        let w1 = r.read_f32s(sizes.w1, "tensor: w1")?;
        let w2 = r.read_f32s(sizes.w2, "tensor: w2")?;
        let w3 = r.read_f32s(sizes.w3, "tensor: w3")?;
        let rms_final = r.read_f32s(sizes.rms_final, "tensor: rms_final")?;

        // --- RoPE tables, deliberately skipped -----------------------------
        // The file stores `freq_cis_real` and `freq_cis_imag`, each
        // `seq_len x head_size/2`. We advance past them without reading, and
        // compute RoPE ourselves in `ops::rope`.
        //
        // The reason is testability, not elegance. If we consumed the table
        // from the file, the differential test would compare two programs
        // that share a precomputed constant, and a bug in *our* RoPE would be
        // masked by the file agreeing with the reference's. Computing it means
        // the differential test genuinely exercises our angle arithmetic. It
        // also means we do not have to trust that the table in the file was
        // written for this model's head size and theta.
        r.skip_f32s(sizes.freq_cis, "tensor: freq_cis_real")?;
        r.skip_f32s(sizes.freq_cis, "tensor: freq_cis_imag")?;

        // --- optional classifier -------------------------------------------
        let wcls = match sizes.wcls {
            None => None,
            Some(n) => Some(r.read_f32s(n, "tensor: wcls")?),
        };

        // Trailing bytes are an error, not something to skip. If a tensor is
        // missing or in the wrong order, the cursor lands early and this fires.
        // It is the cheapest available check against a silently wrong model.
        if r.remaining() != 0 {
            return Err(LoadError::TrailingBytes(r.remaining()));
        }

        Ok(Weights {
            config,
            tok_embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            wcls,
        })
    }

    /// Slice out layer `l` of a per-layer tensor.
    ///
    /// `per_layer` is the number of floats each layer contributes. Panics if
    /// `l` is out of range, which is a programming error in the forward loop
    /// rather than untrusted input: by the time this is called, `l` comes from
    /// `0..config.n_layers`.
    fn layer<'a>(&self, buf: &'a [f32], l: usize, per_layer: usize) -> &'a [f32] {
        let start = l * per_layer;
        &buf[start..start + per_layer]
    }

    pub fn rms_att_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.rms_att, l, self.config.dim)
    }
    pub fn rms_ffn_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.rms_ffn, l, self.config.dim)
    }
    pub fn wq_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.wq, l, self.config.dim * self.config.dim)
    }
    pub fn wk_layer(&self, l: usize) -> &[f32] {
        let kv = self.attention_dims().kv_dim();
        self.layer(&self.wk, l, kv * self.config.dim)
    }
    pub fn wv_layer(&self, l: usize) -> &[f32] {
        let kv = self.attention_dims().kv_dim();
        self.layer(&self.wv, l, kv * self.config.dim)
    }
    pub fn wo_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.wo, l, self.config.dim * self.config.dim)
    }
    pub fn w1_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.w1, l, self.config.hidden_dim * self.config.dim)
    }
    pub fn w2_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.w2, l, self.config.dim * self.config.hidden_dim)
    }
    pub fn w3_layer(&self, l: usize) -> &[f32] {
        self.layer(&self.w3, l, self.config.hidden_dim * self.config.dim)
    }

    pub fn attention_dims(&self) -> AttentionDims {
        self.config
            .attention_dims()
            .expect("weights only exist if the config validated at load time")
    }

    /// The classifier matrix: the untied `wcls` if present, otherwise the
    /// token embedding reused as the unembedding.
    ///
    /// This is weight tying, the trick of sharing one matrix between the input
    /// embedding and the output projection. It halves the parameter count and
    /// is what makes small models trainable, and it means the logits are just
    /// the dot product of the final hidden state with the token embeddings.
    pub fn classifier(&self) -> &[f32] {
        self.wcls.as_deref().unwrap_or(&self.tok_embedding)
    }
}

/// All mutable buffers a decode step needs, allocated once.
///
/// The design rule is that `forward` performs **no allocation**. Every buffer
/// here is sized from the config in [`State::new`] and then written in place for
/// the life of the object. That matters for two reasons: an allocator call in
/// the inner loop is measurable at these timescales, and a `forward` that could
/// allocate is a `forward` whose memory behaviour is not obvious from reading
/// it. Keeping the whole working set visible in one struct makes the memory
/// footprint of decoding a sum you can write down:
///
/// ```text
/// scratch   ~ 2*dim + hidden + 2*kv_dim + 2*q_dim + vocab
/// kv cache  ~ 2 * n_layers * seq_len * kv_dim      <- the dominant term
/// ```
///
/// The cache dominates, and it dominates because it is the whole point: it is
/// what makes decoding O(n) per token rather than O(n^2). See
/// [`State::forward`].
/// Allocate a zero-initialised `Vec<f32>` fallibly so both `isize::MAX` byte
/// capacity overflows and host out-of-memory errors become `LoadError::Overflow`
/// instead of aborting the process.
pub(crate) fn try_zeroed_f32_vec(what: &'static str, len: usize) -> Result<Vec<f32>, LoadError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| LoadError::Overflow { what })?;
    v.resize(len, 0.0);
    Ok(v)
}

#[derive(Debug, Clone)]
pub struct State {
    /// Model configuration this state was allocated for. Checked at the entry
    /// of [`State::forward`] so mismatched `(State, Weights)` pairs return an
    /// error instead of panicking inside a layer.
    pub(crate) config: Config,
    /// The residual stream. This is the vector that flows through the layers.
    pub(crate) x: Vec<f32>,
    /// Generic scratch for a normalised or projected activation.
    pub(crate) xb: Vec<f32>,
    /// SwiGLU gate output (`w1 @ xb`).
    pub(crate) hb: Vec<f32>,
    /// SwiGLU up output (`w3 @ xb`). Separate from `hb` because both are
    /// needed simultaneously to form `silu(gate) * up`.
    pub(crate) hb2: Vec<f32>,
    /// Query for this step, `n_heads * head_size`.
    pub(crate) q: Vec<f32>,
    /// Key for this step, `n_kv_heads * head_size`.
    pub(crate) k: Vec<f32>,
    /// Value for this step, `n_kv_heads * head_size`.
    pub(crate) v: Vec<f32>,
    /// Attention output for this step, `n_heads * head_size`, before `wo`.
    pub(crate) att: Vec<f32>,
    /// Final logits, `vocab_size`.
    pub(crate) logits: Vec<f32>,
    /// Attention score scratch, `seq_len`, reused across every head.
    pub(crate) scores: Vec<f32>,

    /// `n_layers * seq_len * kv_dim`, row `l * seq_len + t` holding the keys
    /// for layer `l` at position `t`.
    pub(crate) key_cache: Vec<f32>,
    /// Same layout, for values.
    pub(crate) value_cache: Vec<f32>,
}

impl State {
    /// Allocate every buffer for a given model.
    ///
    /// The cache allocation is a product of three header-derived numbers, so it
    /// uses the same checked arithmetic as the loader, followed by fallible
    /// allocation via `Vec::try_reserve_exact`. A structurally valid header can
    /// still describe a cache larger than address space or available RAM, and
    /// that has to be an `Err` at construction rather than an abort.
    pub fn new(config: &Config) -> Result<Self, LoadError> {
        let dims = config.validate()?;
        let d = config.dim;
        let q_dim = dims.q_dim();
        let kv_dim = dims.kv_dim();
        let cache = config
            .n_layers
            .checked_mul(config.seq_len)
            .and_then(|n| n.checked_mul(kv_dim))
            .ok_or(LoadError::Overflow { what: "kv cache" })?;

        Ok(State {
            config: *config,
            x: try_zeroed_f32_vec("x", d)?,
            xb: try_zeroed_f32_vec("xb", d)?,
            hb: try_zeroed_f32_vec("hb", config.hidden_dim)?,
            hb2: try_zeroed_f32_vec("hb2", config.hidden_dim)?,
            q: try_zeroed_f32_vec("q", q_dim)?,
            k: try_zeroed_f32_vec("k", kv_dim)?,
            v: try_zeroed_f32_vec("v", kv_dim)?,
            att: try_zeroed_f32_vec("att", q_dim)?,
            logits: try_zeroed_f32_vec("logits", config.vocab_size)?,
            scores: try_zeroed_f32_vec("scores", config.seq_len)?,
            key_cache: try_zeroed_f32_vec("kv cache", cache)?,
            value_cache: try_zeroed_f32_vec("kv cache", cache)?,
        })
    }

    /// Current residual-stream activation (`dim` floats).
    pub fn x(&self) -> &[f32] {
        &self.x
    }

    /// Most recently computed vocabulary logits (`vocab_size` floats).
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Forget the KV cache, so the same `State` can decode a new sequence.
    ///
    /// Zeroing rather than tracking a high-water mark: the cache is written
    /// before it is read at every position, so stale entries are never
    /// observable, and this keeps `forward` free of any "is this position
    /// fresh" bookkeeping.
    pub fn reset(&mut self) {
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);
    }

    /// Number of floats held by each KV cache. Exposed so the CLI can report
    /// the memory the cache is actually using.
    pub fn cache_len(&self) -> usize {
        self.key_cache.len()
    }

    /// Run one token at `pos` and return the resulting logits.
    ///
    /// # Why this is O(n) per token rather than O(n^2)
    ///
    /// A transformer block needs, for the current token, the keys and values of
    /// *every* previous token. Without a cache you would recompute all of them
    /// from scratch, which for token `t` means re-running the whole prefix:
    /// total work O(t^2) over a sequence. But the keys and values of position
    /// `s` depend only on the token at `s` and on the layer-`(l-1)` activations
    /// at `s` - never on any later token. So once computed they are valid
    /// forever, and the only thing that changes is that another one becomes
    /// available. Caching them turns each step into "compute this token's own
    /// k and v, append, attend over the `pos + 1` entries that now exist":
    /// O(t) work per step, O(n^2) total but with a tiny constant, and O(1) extra
    /// work compared to a cached implementation.
    ///
    /// The price is that the cache is the memory cost listed on [`State`], and
    /// that positions must be filled in order. Feeding position 7 before
    /// position 3 would leave a hole that attention would read as zeros, which
    /// is why `pos` is an explicit argument rather than an internal counter: the
    /// ordering is the caller's responsibility and should be visible.
    pub fn forward(&mut self, w: &Weights, token: u32, pos: usize) -> Result<&[f32], RunError> {
        let cfg = &w.config;
        if self.config != *cfg {
            return Err(RunError::StateMismatch {
                detail: format!(
                    "state config {:?} does not match weights config {:?}",
                    self.config, cfg
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

        // --- token embedding ----------------------------------------------
        // The lookup is a copy rather than a borrowed view. `x` is the residual
        // stream and it is accumulated into (`x[i] += ...`) at two points in
        // every layer, so it has to be owned scratch; it cannot alias
        // `w.tok_embedding` for the duration of the loop. The copy is one
        // memcpy of `dim` floats per token, which is noise next to the matmuls.
        let row = token as usize * d;
        self.x.copy_from_slice(&w.tok_embedding[row..row + d]);

        for l in 0..cfg.n_layers {
            // --- attention --------------------------------------------------
            // Normalise, then project. rmsnorm before the QKV matmuls is what
            // keeps the dot products in a sane numeric range; the learned scale
            // is per-channel and is part of the architecture, not an
            // afterthought.
            ops::rmsnorm(&mut self.xb, &self.x, w.rms_att_layer(l));
            ops::matmul(&mut self.q, &self.xb, w.wq_layer(l), q_dim, d);
            ops::matmul(&mut self.k, &self.xb, w.wk_layer(l), kv_dim, d);
            ops::matmul(&mut self.v, &self.xb, w.wv_layer(l), kv_dim, d);

            // RoPE is applied to q and k but never to v. Values are content,
            // not position: the position information is already carried by the
            // keys, and rotating the values would double-count it.
            ops::rope(&mut self.q, dims.n_heads, dims.head_size, pos);
            ops::rope(&mut self.k, dims.n_kv_heads, dims.head_size, pos);

            // Append this position's k and v to the cache for this layer.
            let base = l * kv_layer_stride + pos * kv_dim;
            self.key_cache[base..base + kv_dim].copy_from_slice(&self.k);
            self.value_cache[base..base + kv_dim].copy_from_slice(&self.v);

            // Attend over exactly the rows written so far. The cache slice is
            // the whole layer; `attention` only reads the first `pos + 1` rows,
            // so the rest being stale is harmless.
            let layer_start = l * kv_layer_stride;
            let layer_end = layer_start + kv_layer_stride;
            ops::attention(
                &mut self.att,
                &self.q,
                &self.key_cache[layer_start..layer_end],
                &self.value_cache[layer_start..layer_end],
                &dims,
                pos,
                &mut self.scores,
            );

            // Residual 1: the attention block's output joins the residual
            // stream. This is an *addition*; replacing x with the attention
            // output would discard everything the earlier layers computed and is
            // mutant 9 in the mutation check.
            ops::matmul(&mut self.xb, &self.att, w.wo_layer(l), d, q_dim);
            for i in 0..d {
                self.x[i] += self.xb[i];
            }

            // --- feed forward ----------------------------------------------
            ops::rmsnorm(&mut self.xb, &self.x, w.rms_ffn_layer(l));

            // SwiGLU: silu(w1 x) * (w3 x). `w1` is the "gate" and `w3` is the
            // "up" projection; which one gets the nonlinearity is not
            // symmetric, and swapping them is mutant 7.
            ops::matmul(&mut self.hb, &self.xb, w.w1_layer(l), cfg.hidden_dim, d);
            ops::matmul(&mut self.hb2, &self.xb, w.w3_layer(l), cfg.hidden_dim, d);
            for i in 0..cfg.hidden_dim {
                self.hb[i] = ops::silu_value(self.hb[i]) * self.hb2[i];
            }

            // Residual 2: the feed-forward block's output joins the stream.
            ops::matmul(&mut self.xb, &self.hb, w.w2_layer(l), d, cfg.hidden_dim);
            for i in 0..d {
                self.x[i] += self.xb[i];
            }
        }

        // --- head ----------------------------------------------------------
        // Final norm, then project to vocabulary logits. The untied case uses
        // wcls; the tied case reuses the token embedding, which is why a tied
        // model has no separate output matrix.
        ops::rmsnorm(&mut self.xb, &self.x, &w.rms_final);
        ops::matmul(
            &mut self.logits,
            &self.xb,
            w.classifier(),
            cfg.vocab_size,
            d,
        );

        Ok(&self.logits)
    }
}

/// Index and value of the largest logit, or `None` if `logits` is empty.
///
/// Returning `Option` enforces non-empty results in the type system so `argmax`
/// never panics and never returns an out-of-bounds index.
///
/// If any element is `NaN`, the **first** `NaN` encountered is returned
/// immediately rather than skipped: returning a finite token when part of the
/// logit vector has already diverged would hide model corruption, whereas
/// returning the first `NaN` is deterministic and surfaces the fault. Ties
/// among finite values resolve to the lowest index.
pub fn argmax(logits: &[f32]) -> Option<(usize, f32)> {
    let (&first, rest) = logits.split_first()?;
    if first.is_nan() {
        return Some((0, first));
    }
    let mut best_idx = 0usize;
    let mut best_val = first;
    for (offset, &v) in rest.iter().enumerate() {
        if v.is_nan() {
            return Some((offset + 1, v));
        }
        if v > best_val {
            best_idx = offset + 1;
            best_val = v;
        }
    }
    Some((best_idx, best_val))
}

/// Sampling hyperparameters for [`generate_sampled_stream`].
///
/// The default (`temperature = 0.0`) performs strictly greedy argmax decoding
/// and ignores `top_p` and `seed`, preserving byte-identical output against
/// `llama2.c`'s `run.c`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleConfig {
    /// Softmax temperature. `0.0` selects greedy decoding; positive finite
    /// values scale logits by `1 / temperature` before sampling.
    pub temperature: f32,
    /// Nucleus (top-p) cumulative probability threshold in `(0.0, 1.0]`.
    /// `1.0` samples from the full distribution without truncation.
    pub top_p: f32,
    /// Seed for the deterministic `xorshift64*` PRNG (matching `llama2.c`'s
    /// `random_u32`). `0` is mapped to `1` so the generator never stalls at zero.
    pub seed: u64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.9,
            seed: 0,
        }
    }
}

impl SampleConfig {
    /// Validate sampling hyperparameters.
    pub fn validate(&self) -> Result<(), RunError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RunError::InvalidSampleConfig {
                detail: format!(
                    "temperature must be finite and non-negative, got {}",
                    self.temperature
                ),
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(RunError::InvalidSampleConfig {
                detail: format!("top_p must be in (0.0, 1.0], got {}", self.top_p),
            });
        }
        Ok(())
    }
}

/// Deterministic `xorshift64*` PRNG matching `llama2.c`'s `random_u32`.
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        let mut s = self.state;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.state = s;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

pub(crate) fn select_next_token(
    logits: &[f32],
    pos: usize,
    sample: SampleConfig,
    rng: &mut Rng,
    probs: &mut Vec<f32>,
    candidates: &mut Vec<(usize, f32)>,
) -> Result<u32, RunError> {
    if logits.is_empty() {
        return Err(RunError::StateMismatch {
            detail: "forward returned empty logits".into(),
        });
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(RunError::NonFiniteLogit { pos });
    }
    if sample.temperature == 0.0 {
        let (idx, _) = argmax(logits).ok_or_else(|| RunError::StateMismatch {
            detail: "forward returned empty logits".into(),
        })?;
        return Ok(idx as u32);
    }

    // Shift by max before dividing by temperature so very small positive
    // temperatures cannot overflow `(logit - max) / temperature` to +inf.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    probs.clear();
    probs.extend(logits.iter().map(|&v| (v - max_logit) / sample.temperature));
    ops::softmax(probs);

    let coin = rng.next_f32();
    if sample.top_p >= 1.0 {
        let mut cdf = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cdf += p;
            if coin < cdf {
                return Ok(i as u32);
            }
        }
        return Ok((probs.len() - 1) as u32);
    }

    candidates.clear();
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            candidates.push((i, p));
        }
    }
    if candidates.is_empty() {
        return Ok(0);
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut cumulative = 0.0f32;
    let mut last_idx = candidates.len() - 1;
    for (idx, &(_, p)) in candidates.iter().enumerate() {
        cumulative += p;
        if cumulative > sample.top_p {
            last_idx = idx;
            break;
        }
    }

    let r = coin * cumulative;
    let mut cdf = 0.0f32;
    for &(token_idx, p) in &candidates[..=last_idx] {
        cdf += p;
        if r < cdf {
            return Ok(token_idx as u32);
        }
    }
    Ok(candidates[last_idx].0 as u32)
}

/// Generate up to `n` tokens after `prompt`, calling `on_token` for each
/// emitted token as soon as it is produced.
///
/// Returning `false` from `on_token` stops generation early without error.
/// Returns the number of tokens delivered to `on_token`.
pub fn generate_sampled_stream<F>(
    w: &Weights,
    state: &mut State,
    prompt: &[u32],
    n: usize,
    sample: SampleConfig,
    mut on_token: F,
) -> Result<usize, RunError>
where
    F: FnMut(u32) -> bool,
{
    sample.validate()?;
    if prompt.is_empty() {
        // An empty prompt has no last token to condition on, so there is
        // nothing to continue from. Silently starting from BOS instead would
        // hide a caller bug behind a plausible-looking result.
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

    let mut rng = Rng::new(sample.seed);
    let mut probs: Vec<f32> = Vec::new();
    let mut candidates: Vec<(usize, f32)> = Vec::new();

    // Consume the prompt. Only the final iteration's logits are used, and the
    // next token is picked immediately so the borrow on `state` ends before the
    // generation loop begins.
    let mut next = 0u32;
    for (pos, &t) in prompt.iter().enumerate() {
        let logits = state.forward(w, t, pos)?;
        if pos + 1 == prompt.len() {
            next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
        }
    }

    let mut emitted = 0usize;
    // The next token to be emitted has not been forwarded yet; it will occupy
    // position `prompt.len()`.
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
            // Emitting the last requested token; a further forward pass would
            // only compute logits nobody reads.
            break;
        }
        let logits = state.forward(w, next, pos)?;
        next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
        pos += 1;
    }

    Ok(emitted)
}

/// Greedy-decode up to `n` tokens after `prompt`, streaming each token to
/// `on_token` as it is produced.
pub fn generate_stream<F>(
    w: &Weights,
    state: &mut State,
    prompt: &[u32],
    n: usize,
    on_token: F,
) -> Result<usize, RunError>
where
    F: FnMut(u32) -> bool,
{
    generate_sampled_stream(w, state, prompt, n, SampleConfig::default(), on_token)
}

/// Greedy-decode up to `n` tokens after `prompt`.
///
/// Greedy means always taking the argmax: no temperature, no top-k, no
/// sampling. That is the default mode because greedy decoding is a pure
/// function of the weights and the prompt, so two implementations that agree on
/// the maths produce *byte-identical* output - a far stronger claim than "the
/// text looks about right". For optional seeded temperature/top-p sampling, use
/// [`generate_sampled_stream`].
///
/// # Forward-pass accounting
///
/// The prompt fills positions `0..prompt.len()`. The logits produced by the
/// *last* prompt token are the ones that predict the first generated token, so
/// that pass is not wasted. Each subsequent generated token needs exactly one
/// more pass, and the loop stops before issuing a pass whose output it would
/// not use. Total passes: `prompt.len() + n.saturating_sub(1)`.
///
/// Returns the generated tokens, excluding the prompt.
///
/// # Termination
///
/// Generation stops when the model emits **BOS**, not EOS. That is
/// llama2.c's rule (`if (next == 1) break;`) and it is the rule the models were
/// trained with: BOS is the document delimiter, so a model that has finished a
/// document emits BOS to start the next one, and that is the natural place to
/// stop. A model trained to emit EOS instead will simply run to `n`, which is a
/// harmless loss of throughput rather than a wrong answer.
pub fn generate(
    w: &Weights,
    state: &mut State,
    prompt: &[u32],
    n: usize,
) -> Result<Vec<u32>, RunError> {
    let mut generated: Vec<u32> = Vec::with_capacity(n.min(w.config.seq_len));
    generate_stream(w, state, prompt, n, |token| {
        generated.push(token);
        true
    })?;
    Ok(generated)
}
