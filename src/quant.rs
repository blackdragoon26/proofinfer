//! Q8_0 symmetric group-quantized checkpoint loader (`llama2.c` version 2)
//! and allocation-free quantized forward pass.
//!
//! Binary layout exported by `reference/llama2c/export.py::version2_export`:
//!
//! ```text
//! [0..256)   256-byte header:
//!              u32  magic = 0x616b3432 ("ak42")
//!              i32  version = 2
//!              i32  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//!              u8   shared_classifier (0 or 1)
//!              i32  group_size
//!              zero padding to 256 bytes
//! [256..)    fp32 norms: rms_att, rms_ffn, rms_final
//!            Q8_0 tensors (each stored as `numel` i8 followed by `numel/group_size` f32):
//!              tok_embeddings
//!              for l in 0..n_layers: wq[l]
//!              for l in 0..n_layers: wk[l]
//!              for l in 0..n_layers: wv[l]
//!              for l in 0..n_layers: wo[l]
//!              for l in 0..n_layers: w1[l]
//!              for l in 0..n_layers: w2[l]
//!              for l in 0..n_layers: w3[l]
//!              if shared_classifier == 0: wcls
//! ```

use std::path::Path;

use crate::model::{
    checked_tensor_len, select_next_token, try_zeroed_f32_vec, Config, LoadError, Rng, RunError,
    SampleConfig, State, Weights,
};
use crate::ops::{self, AttentionDims, QSlice, Q80_MAX_GROUP_SIZE};

/// Magic number written by `llama2.c` `export.py` v1/v2 (`"ak42"`).
pub const V2_MAGIC: u32 = 0x616b_3432;
/// Fixed header size in bytes for versioned `llama2.c` checkpoints.
pub const V2_HEADER_BYTES: usize = 256;

/// Owned Q8_0 quantized tensor (`i8` weights + per-group `f32` scales).
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub q: Vec<i8>,
    pub s: Vec<f32>,
}

impl QuantizedTensor {
    pub fn as_slice(&self) -> QSlice<'_> {
        QSlice {
            q: &self.q,
            s: &self.s,
        }
    }

    /// Borrow layer `l` of a stacked per-layer quantized tensor.
    pub fn layer(&self, l: usize, per_layer_q: usize, per_layer_s: usize) -> QSlice<'_> {
        let q_start = l * per_layer_q;
        let s_start = l * per_layer_s;
        QSlice {
            q: &self.q[q_start..q_start + per_layer_q],
            s: &self.s[s_start..s_start + per_layer_s],
        }
    }
}

/// All parameters for a `llama2.c` version 2 Q8_0 quantized checkpoint.
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    pub config: Config,
    pub group_size: usize,
    /// Dequantized `vocab x dim` embedding table computed once at load time.
    pub tok_embedding: Vec<f32>,
    /// Quantized `vocab x dim` embeddings (also reused as classifier when tied).
    pub q_tokens: QuantizedTensor,
    pub rms_att: Vec<f32>,
    pub rms_ffn: Vec<f32>,
    pub rms_final: Vec<f32>,
    pub wq: QuantizedTensor,
    pub wk: QuantizedTensor,
    pub wv: QuantizedTensor,
    pub wo: QuantizedTensor,
    pub w1: QuantizedTensor,
    pub w2: QuantizedTensor,
    pub w3: QuantizedTensor,
    pub wcls: Option<QuantizedTensor>,
}

/// Auto-detected checkpoint (either legacy FP32 or version 2 Q8_0).
#[derive(Debug, Clone)]
pub enum Checkpoint {
    Fp32(Box<Weights>),
    Q80(Box<QuantizedWeights>),
}

impl Checkpoint {
    /// Load either a legacy FP32 checkpoint or a version 2 Q8_0 checkpoint.
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path).map_err(|e| LoadError::Io {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Self::from_bytes(&data)
    }

    /// Parse bytes, dispatching on the versioned `"ak42"` magic header.
    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        if data.len() >= 4 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == V2_MAGIC {
                return Ok(Checkpoint::Q80(Box::new(QuantizedWeights::from_bytes(
                    data,
                )?)));
            }
        }
        Ok(Checkpoint::Fp32(Box::new(Weights::from_bytes(data)?)))
    }

    pub fn config(&self) -> Config {
        match self {
            Checkpoint::Fp32(w) => w.config,
            Checkpoint::Q80(w) => w.config,
        }
    }

    pub fn is_untied(&self) -> bool {
        match self {
            Checkpoint::Fp32(w) => w.wcls.is_some(),
            Checkpoint::Q80(w) => w.wcls.is_some(),
        }
    }
}

struct QReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> QReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

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

    fn read_u8(&mut self, what: &'static str) -> Result<u8, LoadError> {
        self.require(1, what)?;
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u32(&mut self, what: &'static str) -> Result<u32, LoadError> {
        self.require(4, what)?;
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self, what: &'static str) -> Result<i32, LoadError> {
        self.require(4, what)?;
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32s(&mut self, n: usize, what: &'static str) -> Result<Vec<f32>, LoadError> {
        let nbytes = n.checked_mul(4).ok_or(LoadError::Overflow { what })?;
        self.require(nbytes, what)?;
        let bytes = &self.data[self.pos..self.pos + nbytes];
        self.pos += nbytes;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn read_q80_tensor(
        &mut self,
        n_layers: usize,
        per_layer_elems: usize,
        group_size: usize,
        what: &'static str,
    ) -> Result<QuantizedTensor, LoadError> {
        if per_layer_elems % group_size != 0 {
            return Err(LoadError::InvalidHeader(format!(
                "{what} layer length ({per_layer_elems}) is not divisible by group_size ({group_size})"
            )));
        }
        let per_layer_scales = per_layer_elems / group_size;
        let total_q = checked_tensor_len(what, &[n_layers, per_layer_elems])?;
        let total_s = checked_tensor_len(what, &[n_layers, per_layer_scales])?;
        let scale_bytes = total_s.checked_mul(4).ok_or(LoadError::Overflow { what })?;
        let total_bytes = total_q
            .checked_add(scale_bytes)
            .ok_or(LoadError::Overflow { what })?;
        self.require(total_bytes, what)?;

        let mut q = Vec::new();
        q.try_reserve_exact(total_q)
            .map_err(|_| LoadError::Overflow { what })?;
        let mut s = Vec::new();
        s.try_reserve_exact(total_s)
            .map_err(|_| LoadError::Overflow { what })?;

        let layer_s_bytes = per_layer_scales * 4;
        for _ in 0..n_layers {
            let q_bytes = &self.data[self.pos..self.pos + per_layer_elems];
            self.pos += per_layer_elems;
            q.extend(q_bytes.iter().map(|&b| b as i8));

            let s_bytes = &self.data[self.pos..self.pos + layer_s_bytes];
            self.pos += layer_s_bytes;
            s.extend(
                s_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }

        Ok(QuantizedTensor { q, s })
    }
}

impl QuantizedWeights {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let data = std::fs::read(path).map_err(|e| LoadError::Io {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Self::from_bytes(&data)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, LoadError> {
        let mut r = QReader::new(data);
        r.require(V2_HEADER_BYTES, "v2 header")?;

        let magic = r.read_u32("v2 header: magic")?;
        if magic != V2_MAGIC {
            return Err(LoadError::InvalidHeader(format!(
                "unexpected checkpoint magic 0x{magic:08x}, expected 0x{V2_MAGIC:08x}"
            )));
        }
        let version = r.read_i32("v2 header: version")?;
        if version != 2 {
            return Err(LoadError::InvalidHeader(format!(
                "unsupported versioned checkpoint format v{version} (only v2 Q8_0 is supported)"
            )));
        }

        let h_dim = r.read_i32("v2 header: dim")?;
        let h_hidden = r.read_i32("v2 header: hidden_dim")?;
        let h_layers = r.read_i32("v2 header: n_layers")?;
        let h_heads = r.read_i32("v2 header: n_heads")?;
        let h_kv_heads = r.read_i32("v2 header: n_kv_heads")?;
        let h_vocab = r.read_i32("v2 header: vocab_size")?;
        let h_seq = r.read_i32("v2 header: seq_len")?;
        let shared_flag = r.read_u8("v2 header: shared_classifier")?;
        let h_group_size = r.read_i32("v2 header: group_size")?;

        if h_dim <= 0
            || h_hidden <= 0
            || h_layers <= 0
            || h_heads <= 0
            || h_kv_heads <= 0
            || h_vocab <= 0
            || h_seq <= 0
        {
            return Err(LoadError::InvalidHeader(format!(
                "all v2 header dimensions must be positive, got dim={h_dim} hidden_dim={h_hidden} \
                 n_layers={h_layers} n_heads={h_heads} n_kv_heads={h_kv_heads} \
                 vocab_size={h_vocab} seq_len={h_seq}"
            )));
        }
        let tied = match shared_flag {
            0 => false,
            1 => true,
            other => {
                return Err(LoadError::InvalidHeader(format!(
                    "shared_classifier flag must be 0 or 1, got {other}"
                )));
            }
        };
        if h_group_size <= 0 || (h_group_size as usize) > Q80_MAX_GROUP_SIZE {
            return Err(LoadError::InvalidHeader(format!(
                "group_size must be in 1..={Q80_MAX_GROUP_SIZE}, got {h_group_size}"
            )));
        }
        let group_size = h_group_size as usize;

        let config = Config {
            dim: h_dim as usize,
            hidden_dim: h_hidden as usize,
            n_layers: h_layers as usize,
            n_heads: h_heads as usize,
            n_kv_heads: h_kv_heads as usize,
            vocab_size: h_vocab as usize,
            seq_len: h_seq as usize,
        };
        let dims = config.validate()?;
        if config.dim % group_size != 0 {
            return Err(LoadError::InvalidHeader(format!(
                "dim ({}) must be divisible by group_size ({group_size})",
                config.dim
            )));
        }
        if config.hidden_dim % group_size != 0 {
            return Err(LoadError::InvalidHeader(format!(
                "hidden_dim ({}) must be divisible by group_size ({group_size})",
                config.hidden_dim
            )));
        }

        // Advance to end of the fixed 256-byte header.
        r.pos = V2_HEADER_BYTES;

        let (n_layers, d, kv, hid, v) = (
            config.n_layers,
            config.dim,
            dims.kv_dim(),
            config.hidden_dim,
            config.vocab_size,
        );

        // Pre-validate every tensor element count up front so overflows report
        // by tensor name before reading.
        let rms_att_len = checked_tensor_len("rms_att", &[n_layers, d])?;
        let rms_ffn_len = checked_tensor_len("rms_ffn", &[n_layers, d])?;
        let rms_final_len = checked_tensor_len("rms_final", &[d])?;
        let tok_len = checked_tensor_len("token_embedding", &[v, d])?;
        let wq_layer_len = checked_tensor_len("wq", &[d, d])?;
        let _ = checked_tensor_len("wq", &[n_layers, wq_layer_len])?;
        let wk_layer_len = checked_tensor_len("wk", &[kv, d])?;
        let _ = checked_tensor_len("wk", &[n_layers, wk_layer_len])?;
        let wv_layer_len = checked_tensor_len("wv", &[kv, d])?;
        let _ = checked_tensor_len("wv", &[n_layers, wv_layer_len])?;
        let wo_layer_len = checked_tensor_len("wo", &[d, d])?;
        let _ = checked_tensor_len("wo", &[n_layers, wo_layer_len])?;
        let w1_layer_len = checked_tensor_len("w1", &[hid, d])?;
        let _ = checked_tensor_len("w1", &[n_layers, w1_layer_len])?;
        let w2_layer_len = checked_tensor_len("w2", &[d, hid])?;
        let _ = checked_tensor_len("w2", &[n_layers, w2_layer_len])?;
        let w3_layer_len = checked_tensor_len("w3", &[hid, d])?;
        let _ = checked_tensor_len("w3", &[n_layers, w3_layer_len])?;

        // FP32 norms
        let rms_att = r.read_f32s(rms_att_len, "tensor: rms_att")?;
        let rms_ffn = r.read_f32s(rms_ffn_len, "tensor: rms_ffn")?;
        let rms_final = r.read_f32s(rms_final_len, "tensor: rms_final")?;

        // Q8_0 quantized tensors in exact `version2_export` order
        let q_tokens = r.read_q80_tensor(1, tok_len, group_size, "tensor: token_embedding")?;
        let wq = r.read_q80_tensor(n_layers, wq_layer_len, group_size, "tensor: wq")?;
        let wk = r.read_q80_tensor(n_layers, wk_layer_len, group_size, "tensor: wk")?;
        let wv = r.read_q80_tensor(n_layers, wv_layer_len, group_size, "tensor: wv")?;
        let wo = r.read_q80_tensor(n_layers, wo_layer_len, group_size, "tensor: wo")?;
        let w1 = r.read_q80_tensor(n_layers, w1_layer_len, group_size, "tensor: w1")?;
        let w2 = r.read_q80_tensor(n_layers, w2_layer_len, group_size, "tensor: w2")?;
        let w3 = r.read_q80_tensor(n_layers, w3_layer_len, group_size, "tensor: w3")?;
        let wcls = if tied {
            None
        } else {
            Some(r.read_q80_tensor(1, tok_len, group_size, "tensor: wcls")?)
        };

        let trailing = r.remaining();
        if trailing != 0 {
            return Err(LoadError::TrailingBytes(trailing));
        }

        let mut tok_embedding = try_zeroed_f32_vec("token_embedding", tok_len)?;
        ops::dequantize_q80(&mut tok_embedding, q_tokens.as_slice(), group_size);

        Ok(QuantizedWeights {
            config,
            group_size,
            tok_embedding,
            q_tokens,
            rms_att,
            rms_ffn,
            rms_final,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            wcls,
        })
    }

    pub fn attention_dims(&self) -> AttentionDims {
        self.config
            .attention_dims()
            .expect("quantized weights validated at load time")
    }

    pub fn rms_att_layer(&self, l: usize) -> &[f32] {
        let start = l * self.config.dim;
        &self.rms_att[start..start + self.config.dim]
    }

    pub fn rms_ffn_layer(&self, l: usize) -> &[f32] {
        let start = l * self.config.dim;
        &self.rms_ffn[start..start + self.config.dim]
    }

    pub fn wq_layer(&self, l: usize) -> QSlice<'_> {
        let q_len = self.config.dim * self.config.dim;
        self.wq.layer(l, q_len, q_len / self.group_size)
    }

    pub fn wk_layer(&self, l: usize) -> QSlice<'_> {
        let kv = self.attention_dims().kv_dim();
        let q_len = kv * self.config.dim;
        self.wk.layer(l, q_len, q_len / self.group_size)
    }

    pub fn wv_layer(&self, l: usize) -> QSlice<'_> {
        let kv = self.attention_dims().kv_dim();
        let q_len = kv * self.config.dim;
        self.wv.layer(l, q_len, q_len / self.group_size)
    }

    pub fn wo_layer(&self, l: usize) -> QSlice<'_> {
        let q_len = self.config.dim * self.config.dim;
        self.wo.layer(l, q_len, q_len / self.group_size)
    }

    pub fn w1_layer(&self, l: usize) -> QSlice<'_> {
        let q_len = self.config.hidden_dim * self.config.dim;
        self.w1.layer(l, q_len, q_len / self.group_size)
    }

    pub fn w2_layer(&self, l: usize) -> QSlice<'_> {
        let q_len = self.config.dim * self.config.hidden_dim;
        self.w2.layer(l, q_len, q_len / self.group_size)
    }

    pub fn w3_layer(&self, l: usize) -> QSlice<'_> {
        let q_len = self.config.hidden_dim * self.config.dim;
        self.w3.layer(l, q_len, q_len / self.group_size)
    }

    pub fn classifier(&self) -> QSlice<'_> {
        self.wcls
            .as_ref()
            .map(QuantizedTensor::as_slice)
            .unwrap_or_else(|| self.q_tokens.as_slice())
    }
}

/// Reusable decode state for Q8_0 inference, extending [`State`] with
/// pre-allocated dynamic activation quantization buffers so `forward` performs
/// zero heap allocations per token.
#[derive(Debug, Clone)]
pub struct QuantizedState {
    pub(crate) base: State,
    pub(crate) group_size: usize,
    pub(crate) xq: Vec<i8>,
    pub(crate) xs: Vec<f32>,
}

impl QuantizedState {
    pub fn new(config: &Config, group_size: usize) -> Result<Self, LoadError> {
        if group_size == 0 || group_size > Q80_MAX_GROUP_SIZE {
            return Err(LoadError::InvalidHeader(format!(
                "group_size must be in 1..={Q80_MAX_GROUP_SIZE}, got {group_size}"
            )));
        }
        let base = State::new(config)?;
        if config.dim % group_size != 0 || config.hidden_dim % group_size != 0 {
            return Err(LoadError::InvalidHeader(format!(
                "dim ({}) and hidden_dim ({}) must be divisible by group_size ({group_size})",
                config.dim, config.hidden_dim
            )));
        }
        let max_d = config.dim.max(config.hidden_dim);
        let mut xq = Vec::new();
        xq.try_reserve_exact(max_d)
            .map_err(|_| LoadError::Overflow { what: "xq scratch" })?;
        xq.resize(max_d, 0);
        let xs = try_zeroed_f32_vec("xs scratch", max_d / group_size)?;
        Ok(Self {
            base,
            group_size,
            xq,
            xs,
        })
    }

    pub fn logits(&self) -> &[f32] {
        self.base.logits()
    }

    pub fn reset(&mut self) {
        self.base.reset();
    }

    pub fn forward(
        &mut self,
        w: &QuantizedWeights,
        token: u32,
        pos: usize,
    ) -> Result<&[f32], RunError> {
        let cfg = &w.config;
        if self.base.config != *cfg || self.group_size != w.group_size {
            return Err(RunError::StateMismatch {
                detail: format!(
                    "quantized state config {:?} (gs={}) does not match weights {:?} (gs={})",
                    self.base.config, self.group_size, cfg, w.group_size
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
        self.base.x.copy_from_slice(&w.tok_embedding[row..row + d]);

        for l in 0..cfg.n_layers {
            ops::rmsnorm(&mut self.base.xb, &self.base.x, w.rms_att_layer(l));
            ops::quantize_q80(
                &mut self.xq[..d],
                &mut self.xs[..d_groups],
                &self.base.xb,
                gs,
            );
            let x_q = QSlice {
                q: &self.xq[..d],
                s: &self.xs[..d_groups],
            };
            ops::matmul_q80(&mut self.base.q, x_q, w.wq_layer(l), q_dim, d, gs);
            ops::matmul_q80(&mut self.base.k, x_q, w.wk_layer(l), kv_dim, d, gs);
            ops::matmul_q80(&mut self.base.v, x_q, w.wv_layer(l), kv_dim, d, gs);

            ops::rope(&mut self.base.q, dims.n_heads, dims.head_size, pos);
            ops::rope(&mut self.base.k, dims.n_kv_heads, dims.head_size, pos);

            let base = l * kv_layer_stride + pos * kv_dim;
            self.base.key_cache[base..base + kv_dim].copy_from_slice(&self.base.k);
            self.base.value_cache[base..base + kv_dim].copy_from_slice(&self.base.v);

            let layer_start = l * kv_layer_stride;
            let layer_end = layer_start + kv_layer_stride;
            ops::attention(
                &mut self.base.att,
                &self.base.q,
                &self.base.key_cache[layer_start..layer_end],
                &self.base.value_cache[layer_start..layer_end],
                &dims,
                pos,
                &mut self.base.scores,
            );

            ops::quantize_q80(
                &mut self.xq[..q_dim],
                &mut self.xs[..q_dim / gs],
                &self.base.att,
                gs,
            );
            let att_q = QSlice {
                q: &self.xq[..q_dim],
                s: &self.xs[..q_dim / gs],
            };
            ops::matmul_q80(&mut self.base.xb, att_q, w.wo_layer(l), d, q_dim, gs);
            for i in 0..d {
                self.base.x[i] += self.base.xb[i];
            }

            ops::rmsnorm(&mut self.base.xb, &self.base.x, w.rms_ffn_layer(l));
            ops::quantize_q80(
                &mut self.xq[..d],
                &mut self.xs[..d_groups],
                &self.base.xb,
                gs,
            );
            let ffn_q = QSlice {
                q: &self.xq[..d],
                s: &self.xs[..d_groups],
            };
            ops::matmul_q80(&mut self.base.hb, ffn_q, w.w1_layer(l), hid, d, gs);
            ops::matmul_q80(&mut self.base.hb2, ffn_q, w.w3_layer(l), hid, d, gs);
            for i in 0..hid {
                self.base.hb[i] = ops::silu_value(self.base.hb[i]) * self.base.hb2[i];
            }

            ops::quantize_q80(
                &mut self.xq[..hid],
                &mut self.xs[..hid_groups],
                &self.base.hb,
                gs,
            );
            let hb_q = QSlice {
                q: &self.xq[..hid],
                s: &self.xs[..hid_groups],
            };
            ops::matmul_q80(&mut self.base.xb, hb_q, w.w2_layer(l), d, hid, gs);
            for i in 0..d {
                self.base.x[i] += self.base.xb[i];
            }
        }

        ops::rmsnorm(&mut self.base.xb, &self.base.x, &w.rms_final);
        ops::quantize_q80(
            &mut self.xq[..d],
            &mut self.xs[..d_groups],
            &self.base.xb,
            gs,
        );
        let head_q = QSlice {
            q: &self.xq[..d],
            s: &self.xs[..d_groups],
        };
        ops::matmul_q80(
            &mut self.base.logits,
            head_q,
            w.classifier(),
            cfg.vocab_size,
            d,
            gs,
        );

        Ok(&self.base.logits)
    }
}

/// Stream up to `n` tokens from a Q8_0 model (single-threaded path).
pub fn generate_q80_sampled_stream<F>(
    w: &QuantizedWeights,
    state: &mut QuantizedState,
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

    let mut next = 0u32;
    for (pos, &t) in prompt.iter().enumerate() {
        let logits = state.forward(w, t, pos)?;
        if pos + 1 == prompt.len() {
            next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
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
        let logits = state.forward(w, next, pos)?;
        next = select_next_token(logits, pos, sample, &mut rng, &mut probs, &mut candidates)?;
        pos += 1;
    }

    Ok(emitted)
}
