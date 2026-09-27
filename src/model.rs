//! Model configuration, checkpoint loading, KV-cached decoding.

use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::ops::AttentionDims;

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
    fn validate(&self) -> Result<AttentionDims, LoadError> {
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
fn checked_tensor_len(what: &'static str, dims: &[usize]) -> Result<usize, LoadError> {
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
