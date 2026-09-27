//! The Llama 2 BPE tokenizer: vocabulary file, encoder, decoder.
//!
//! The format is the one llama2.c's `tokenizer.py` exports. It is a flat
//! little-endian dump with no structure beyond what the reader infers:
//!
//! ```text
//! i32  max_token_length
//! repeat 32000 times:
//!     f32 score
//!     i32 length
//!     length raw bytes
//! ```
//!
//! Special ids are fixed by the SentencePiece convention this vocabulary was
//! built with: 0 is `<unk>`, 1 is BOS, 2 is EOS, and 3..=258 are the 256 byte
//! fallbacks `<0x00>`..`<0xFF>` in order, so byte `b` maps to id `b + 3`.
//!
//! # Conformance
//!
//! `tests/tokenizer.rs` asserts that `encode` reproduces, exactly, the token id
//! sequences that Meta published in their own example scripts and that
//! llama2.c's `test.c` checks against. That matters more than it might look:
//! the tokenizer is the one component whose output is consumed by something
//! outside this project (a downloaded checkpoint, a pre-tokenised dataset), so
//! "close enough" is not a useful notion of correct here.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

pub const UNK_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
/// First byte-fallback token id. Byte `b` is token `b + BYTE_ID_OFFSET`.
pub const BYTE_ID_OFFSET: u32 = 3;

/// The number of tokens in a Llama 2 vocabulary.
///
/// `tokenizer.bin` does not record this, which is a long-standing wart in the
/// format (llama2.c's own comment says "i should have written the vocab_size
/// into the tokenizer file... sigh"). We take it as a parameter for the same
/// reason the C does: guessing it would mean guessing wrong silently.
pub const LLAMA2_VOCAB_SIZE: usize = 32000;

/// Errors from reading a tokenizer vocabulary.
///
/// This mirrors the shape of [`crate::model::LoadError`] rather than sharing
/// it: the two formats have genuinely different failure modes (one is a
/// seven-field header describing 13 tensors, the other is a 32000-iteration
/// loop of variable-length records), and a shared error type would end up with
/// a variant per format. The shared invariant is the same in both: the file is
/// untrusted, sizes are checked, and the result is never a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    Io {
        path: String,
        msg: String,
    },
    Truncated {
        what: &'static str,
        need: usize,
        have: usize,
    },
    Overflow {
        what: &'static str,
    },
    TrailingBytes(usize),
    Missing(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io { path, msg } => write!(f, "cannot read {path}: {msg}"),
            TokenizerError::Truncated { what, need, have } => write!(
                f,
                "file truncated while reading {what}: needed {need} bytes, {have} available"
            ),
            TokenizerError::Overflow { what } => {
                write!(f, "size overflow computing {what}")
            }
            TokenizerError::TrailingBytes(n) => write!(f, "{n} unexpected trailing bytes"),
            TokenizerError::Missing(what) => write!(f, "vocabulary is missing {what}"),
        }
    }
}

impl Error for TokenizerError {}

/// A loaded BPE vocabulary.
pub struct Tokenizer {
    /// Token id to raw bytes. Index 0 is `<unk>`.
    vocab: Vec<Vec<u8>>,
    /// BPE merge score per token. Higher wins when two adjacent tokens can be
    /// merged into a candidate.
    scores: Vec<f32>,
    /// Reverse lookup from raw bytes to token id.
    ///
    /// llama2.c sorts the vocabulary once and binary-searches it. A hash map is
    /// the same data structure with the log factor removed, and the encode path
    /// does a lookup per codepoint *and* per merge candidate, so this is the
    /// inner loop of the hottest part of tokenisation.
    ///
    /// The Llama 2 vocabulary is verified to contain no duplicate byte strings,
    /// so the mapping is unambiguous. If a vocabulary ever did contain one, the
    /// lowest id wins, which matches what a stable `qsort` plus `bsearch` would
    /// land on most of the time.
    lookup: HashMap<Vec<u8>, u32>,
    max_token_length: usize,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The vocab itself is 32000 byte strings; do not dump it.
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab.len())
            .field("max_token_length", &self.max_token_length)
            .finish()
    }
}

impl Tokenizer {
    /// Load a vocabulary from disk.
    pub fn load(path: &Path) -> Result<Self, TokenizerError> {
        let data = std::fs::read(path).map_err(|e| TokenizerError::Io {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Self::from_bytes(&data, LLAMA2_VOCAB_SIZE)
    }

    /// Parse a vocabulary from an in-memory buffer.
    ///
    /// `vocab_size` is the number of token records to read; see
    /// [`LLAMA2_VOCAB_SIZE`] for why the format does not tell us.
    pub fn from_bytes(data: &[u8], vocab_size: usize) -> Result<Self, TokenizerError> {
        let mut pos = 0usize;
        // A closure taking the cursor as an argument rather than capturing it
        // mutably. The cursor is an ordinary local, so threading it through
        // explicitly keeps the 32000-iteration record loop readable.
        let take =
            |pos: &mut usize, n: usize, what: &'static str| -> Result<&[u8], TokenizerError> {
                let remaining = data.len() - *pos;
                if remaining < n {
                    return Err(TokenizerError::Truncated {
                        what,
                        need: n,
                        have: remaining,
                    });
                }
                let s = &data[*pos..*pos + n];
                *pos += n;
                Ok(s)
            };

        let max_token_length =
            i32::from_le_bytes(take(&mut pos, 4, "max_token_length")?.try_into().unwrap()) as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);

        for id in 0..vocab_size {
            let score =
                f32::from_le_bytes(take(&mut pos, 4, "vocab score")?.try_into().map_err(|_| {
                    TokenizerError::Truncated {
                        what: "vocab score",
                        need: 4,
                        have: 0,
                    }
                })?);
            let len = i32::from_le_bytes(take(&mut pos, 4, "vocab length")?.try_into().unwrap());
            // A negative length is a malformed record, not a request to read
            // backwards. Cast through i64 so the check is exact on every
            // platform rather than relying on a wrapping cast.
            if len < 0 {
                return Err(TokenizerError::Truncated {
                    what: "vocab length",
                    need: 0,
                    have: usize::MAX,
                });
            }
            let len = len as usize;
            // `max_token_length` comes from the same untrusted file, so it is
            // a claim to be checked, not a bound to be trusted.
            if len > max_token_length {
                return Err(TokenizerError::Overflow {
                    what: "vocab entry length",
                });
            }
            let bytes = take(&mut pos, len, "vocab bytes")?.to_vec();
            vocab.push(bytes);
            scores.push(score);
            let _ = id;
        }

        if pos != data.len() {
            return Err(TokenizerError::TrailingBytes(data.len() - pos));
        }

        let mut lookup = HashMap::with_capacity(vocab.len());
        for (id, piece) in vocab.iter().enumerate() {
            lookup.entry(piece.clone()).or_insert(id as u32);
        }

        let t = Tokenizer {
            vocab,
            scores,
            lookup,
            max_token_length,
        };

        // The dummy-prefix space is not optional: `encode` pushes it before
        // every non-empty input, and a vocabulary without it means the file is
        // not a Llama 2 vocabulary. Check it here rather than panicking on a
        // lookup miss 400 lines into an encode.
        if !t.lookup.contains_key(b" ".as_slice()) {
            return Err(TokenizerError::Missing(
                "the dummy-prefix space token".into(),
            ));
        }

        Ok(t)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }

    /// Raw bytes of a single token.
    pub fn piece(&self, token: u32) -> Option<&[u8]> {
        self.vocab.get(token as usize).map(|v| v.as_slice())
    }

    /// Score of a single token.
    pub fn score(&self, token: u32) -> Option<f32> {
        self.scores.get(token as usize).copied()
    }

    /// Look up an exact byte string.
    pub fn id_of(&self, piece: &[u8]) -> Option<u32> {
        self.lookup.get(piece).copied()
    }

    /// Encode `text` into token ids, matching llama2.c's `encode` exactly.
    ///
    /// The algorithm, in the order llama2.c performs it:
    ///
    /// 1. Optionally prepend BOS.
    /// 2. If the text is non-empty, prepend the SentencePiece "dummy prefix"
    ///    token, which is a single space. SentencePiece treats a leading space
    ///    as a word boundary, so without this "hello" and " hello" would encode
    ///    identically and the model could not tell them apart. The token only
    ///    marks a boundary; [`decode`] strips it again.
    /// 3. Walk the text one *byte* at a time, accumulating UTF-8 codepoints.
    ///    For each, look the whole codepoint up in the vocabulary. If it is
    ///    absent, fall back to one byte-fallback token per byte.
    /// 4. Repeatedly merge the highest-scoring adjacent pair that exists in
    ///    the vocabulary, until no adjacent pair merges.
    /// 5. Optionally append EOS.
    ///
    /// Step 3 is byte-driven rather than codepoint-driven on purpose: that is
    /// how the reference does it, and the two differ on malformed UTF-8. The
    /// reference resets its buffer on any byte that is not a continuation byte,
    /// so a stray continuation byte starts a new piece rather than extending
    /// the previous one. Replicating that exactly is what makes the
    /// conformance tests meaningful.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
        if bos {
            tokens.push(BOS_ID);
        }

        let bytes = text.as_bytes();
        if !bytes.is_empty() {
            // Verified present by `from_bytes`.
            tokens.push(self.id_of(b" ").expect("dummy prefix checked at load time"));
        }

        // --- step 3: codepoints, or byte fallback ------------------------
        let mut piece: Vec<u8> = Vec::with_capacity(4);
        for (i, &b) in bytes.iter().enumerate() {
            // 0b10xxxxxx marks a UTF-8 continuation byte. Anything else starts
            // a new codepoint, so drop whatever was accumulating.
            if (b & 0xC0) != 0x80 {
                piece.clear();
            }
            piece.push(b);

            // Keep accumulating while the *next* byte continues this codepoint,
            // but stop at 4 bytes: that is the longest UTF-8 encoding, and the
            // cap stops a long run of continuation bytes from growing the
            // buffer without bound.
            let next_is_continuation = bytes.get(i + 1).is_some_and(|&nb| (nb & 0xC0) == 0x80);
            if next_is_continuation && piece.len() < 4 {
                continue;
            }

            match self.id_of(&piece) {
                Some(id) => tokens.push(id),
                None => {
                    for &fallback in &piece {
                        tokens.push(fallback as u32 + BYTE_ID_OFFSET);
                    }
                }
            }
            piece.clear();
        }

        // --- step 4: BPE merges -------------------------------------------
        // Reused across iterations so the loop allocates nothing. Sized from
        // max_token_length because a candidate is at most two tokens long.
        let mut candidate: Vec<u8> = Vec::with_capacity(2 * self.max_token_length);

        loop {
            // Sentinel for "no merge found". The reference initialises
            // best_score to -1e10 rather than -inf, which means a vocabulary
            // entry scoring exactly -1e10 would never be selected. Matching
            // that exactly costs nothing and removes a class of "works on my
            // machine" difference.
            let mut best_score = -1e10f32;
            let mut best_id = 0u32;
            let mut best_idx = usize::MAX;

            // `saturating_sub` so an empty or single-token sequence gives an
            // empty range instead of wrapping.
            for i in 0..tokens.len().saturating_sub(1) {
                candidate.clear();
                candidate.extend_from_slice(&self.vocab[tokens[i] as usize]);
                candidate.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);

                if let Some(&id) = self.lookup.get(candidate.as_slice()) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_id = id;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }

        if eos {
            tokens.push(EOS_ID);
        }
        tokens
    }

    /// Decode token ids back to text, exactly as llama2.c's `decode` does.
    ///
    /// Three pieces of post-processing, all inherited from the reference:
    ///
    /// * **Dummy prefix removal.** A token that directly follows BOS and begins
    ///   with a space has that space dropped. SentencePiece's decoder does the
    ///   same (PR #89 in llama2.c's comment refers to it). Without this, every
    ///   decode would begin with a stray space.
    /// * **Byte tokens.** `<0xNN>` is not text, it is how SentencePiece escapes
    ///   a raw byte that has no printable representation. It is turned back
    ///   into that byte, which is how the model can emit arbitrary UTF-8
    ///   including multi-byte characters split across several tokens.
    /// * **Nothing is special-cased.** BOS decodes to the literal seven
    ///   characters `\n<s>\n` that `tokenizer.py` stored for it, because that
    ///   genuinely is what is in the vocabulary.
    ///
    /// That last point is a wart of the format rather than a choice, and it is
    /// why [`decode_text`](Self::decode_text) exists. Use this one when you
    /// want reference-exact behaviour; use that one when you want the prompt.
    ///
    /// The output is not guaranteed to be valid UTF-8 - a byte-fallback
    /// sequence can name any byte at all - so the conversion is lossy rather
    /// than fallible.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.decode_inner(tokens, false)
    }

    /// Decode token ids to prompt text, treating BOS and EOS as structural
    /// markers that contribute no characters.
    ///
    /// This is what makes `decode_text(encode(s)) == s` hold, and it is the
    /// form a caller actually wants: nobody wants the string `"\n<s>\n"` in the
    /// middle of their output.
    ///
    /// Note that BOS is still *visited*, not skipped, because the dummy-prefix
    /// rule depends on the previous token being BOS. Filtering the specials out
    /// of the token list first would silently disable that rule and reintroduce
    /// a leading space.
    pub fn decode_text(&self, tokens: &[u32]) -> String {
        self.decode_inner(tokens, true)
    }

    fn decode_inner(&self, tokens: &[u32], skip_special: bool) -> String {
        let mut out: Vec<u8> = Vec::new();
        let mut prev: Option<u32> = None;

        for &token in tokens {
            let piece = match self.vocab.get(token as usize) {
                Some(p) => p,
                // An out-of-range id cannot come from our own `encode`. It can
                // come from a caller reading a token list off disk, so skip it
                // rather than panic.
                None => {
                    prev = Some(token);
                    continue;
                }
            };

            if skip_special && (token == BOS_ID || token == EOS_ID) {
                // Still recorded as `prev` below: the dummy-prefix rule keys
                // off the previous token being BOS.
            } else if prev == Some(BOS_ID) && piece.first() == Some(&b' ') {
                out.extend_from_slice(&piece[1..]);
            } else if let Some(byte) = parse_byte_token(piece) {
                out.push(byte);
            } else {
                out.extend_from_slice(piece);
            }
            prev = Some(token);
        }

        String::from_utf8_lossy(&out).into_owned()
    }

    /// Decode one token given the token before it.
    ///
    /// Generation needs this form, because the dummy-prefix rule depends on
    /// the previous token and streaming decoding does not have the whole
    /// sequence in hand.
    pub fn decode_one(&self, prev_token: Option<u32>, token: u32) -> String {
        if token == BOS_ID || token == EOS_ID {
            return String::new();
        }
        if prev_token == Some(BOS_ID) {
            if let Some(piece) = self.vocab.get(token as usize) {
                if piece.first() == Some(&b' ') {
                    return String::from_utf8_lossy(&piece[1..]).into_owned();
                }
            }
        }
        self.decode_inner(&[token], true)
    }
}

/// If `piece` is a `<0xNN>` byte-escape token, return the byte it names.
///
/// The grammar is fixed at exactly six bytes: `<`, `0`, `x`, two hex digits,
/// `>`. Matching the exact width matters. A looser parse that accepted any
/// prefix would also swallow ordinary text that happens to start with `<0x`,
/// and would mis-handle the uppercase/lowercase mix that SentencePiece uses.
fn parse_byte_token(piece: &[u8]) -> Option<u8> {
    if piece.len() != 6 || &piece[..3] != b"<0x" || piece[5] != b'>' {
        return None;
    }
    let hi = (piece[3] as char).to_digit(16)?;
    let lo = (piece[4] as char).to_digit(16)?;
    Some((hi * 16 + lo) as u8)
}
