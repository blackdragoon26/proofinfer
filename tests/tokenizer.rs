//! Tokenizer conformance against Meta's published Llama 2 token ids.
//!
//! The expected sequences below are lifted verbatim from
//! `reference/llama2c/test.c`, which in turn took them from
//! facebookresearch/llama's `example_text_completion.py` and read the expected
//! values out of a Python debugger. They are not values I derived from my own
//! implementation, which is the entire point: a test that compares my encoder
//! against my encoder proves nothing.
//!
//! Conformance here means *exact* equality of the id sequence. A tokenizer
//! that is 99% right is a tokenizer that silently corrupts a pre-tokenised
//! dataset or misaligns with a checkpoint's training data, and there is no
//! tolerance to fall back on.

use std::path::PathBuf;

use proofinfer::tokenizer::{Tokenizer, BOS_ID, EOS_ID, LLAMA2_VOCAB_SIZE, UNK_ID};

fn tokenizer() -> Tokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/llama2c/tokenizer.bin");
    Tokenizer::load(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
}

/// The five cases llama2.c's `test.c` checks. Index and prompt are paired with
/// the expected ids; keeping them in one table makes it obvious at a glance if
/// one is edited without the others.
const CASES: &[(&str, &[u32])] = &[
    // The empty string encodes to just BOS. No dummy prefix is added, because
    // the dummy prefix exists to mark a leading space and there is no text.
    ("", &[1]),
    // "I believe the meaning of life is"
    (
        "I believe the meaning of life is",
        &[1, 306, 4658, 278, 6593, 310, 2834, 338],
    ),
    // "Simply put, the theory of relativity states that "
    // Note the trailing space: it is the prompt that feeds the next token, and
    // the trailing 29871 is the " " token that the space produces. If the
    // encoder were trimming whitespace, this case would fail.
    (
        "Simply put, the theory of relativity states that ",
        &[
            1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871,
        ],
    ),
    // "A brief message congratulating the team on the launch:\n\n ... I just "
    // Contains newlines and runs of spaces, which exercise the byte-oriented
    // codepoint walk: "\n" is a single codepoint but the 8-space run only
    // becomes one token after BPE merging collapses it.
    (
        "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ",
        &[
            1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13,
            13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871,
        ],
    ),
    // "Translate English to French: ... menthe poivrée ..."
    // The accented "é" in "poivrée" is the interesting part. That codepoint is
    // not in the vocabulary on its own, so it goes through byte fallback as its
    // two UTF-8 bytes (0xC3 0xA9), and BPE then merges that pair back into a
    // real token. If the byte-fallback path were wrong, this case diverges.
    // Copy the prompt exactly, accent included.
    (
        "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivr\u{e9}e\n        plush girafe => girafe peluche\n        cheese =>",
        &[
            1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449,
            276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318,
            13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706,
            923, 968, 1149,
        ],
    ),
];

#[test]
fn encode_matches_the_reference_vectors() {
    let t = tokenizer();
    for (prompt, expected) in CASES {
        let got = t.encode(prompt, true, false);
        assert_eq!(
            &got[..],
            *expected,
            "\nprompt: {prompt:?}\n  expected: {expected:?}\n  got:      {got:?}"
        );
    }
}

#[test]
fn all_reference_cases_pass_at_once() {
    // A separate single-case test so a failure names the count as well as the
    // diff, which is the number that matters for a conformance claim.
    let t = tokenizer();
    let failures: Vec<&str> = CASES
        .iter()
        .filter(|(prompt, expected)| t.encode(prompt, true, false) != *expected)
        .map(|(prompt, _)| *prompt)
        .collect();
    assert!(
        failures.is_empty(),
        "{}/{} reference encodings failed: {failures:?}",
        failures.len(),
        CASES.len()
    );
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

/// The well-defined round trip: `decode_text(encode(s)) == s`.
///
/// `decode_text` is used rather than `decode` for a reason that is a property
/// of the *format*, not of this code. `tokenizer.py` stored the literal seven
/// characters `\n<s>\n` as BOS's piece, so the reference-faithful `decode`
/// emits them. BOS is still present in the token list handed to `decode_text`
/// (it is what triggers the dummy-prefix rule), it just contributes no
/// characters.
fn content_tokens(t: &Tokenizer, text: &str) -> String {
    t.decode_text(&t.encode(text, true, false))
}

#[test]
fn encode_then_decode_recovers_the_original_text() {
    let t = tokenizer();
    for (prompt, _) in CASES {
        if prompt.is_empty() {
            continue;
        }
        assert_eq!(
            content_tokens(&t, prompt),
            *prompt,
            "round trip changed the text for {prompt:?}"
        );
    }
}

#[test]
fn round_trip_survives_text_the_vocabulary_does_not_cover() {
    // Byte fallback is only exercised by text outside the vocabulary's
    // codepoints. Emoji, CJK, combining marks and a lone control character all
    // force the fallback path, and all of them must survive the trip.
    let t = tokenizer();
    let samples = [
        "hello world",
        "caf\u{00e9} na\u{00ef}ve",         // Latin-1 supplement
        "\u{4f60}\u{597d}\u{4e16}\u{754c}", // CJK
        "emoji: \u{1f600}\u{1f680}",       // astral plane, 4-byte UTF-8
        "e\u{0301}gal",                     // combining acute accent
        "tab\tand\nnewline",
        "emoji zwj: \u{1f469}\u{200d}\u{1f4bb}",
        "symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        "digits 0123456789",
        "a very long sentence that will be split into many tokens by the merge loop and then reassembled exactly as it was",
    ];
    for s in samples {
        assert_eq!(content_tokens(&t, s), s, "round trip failed for {s:?}");
    }
}

#[test]
fn a_lone_control_byte_round_trips_via_byte_fallback() {
    // 0x07 (BEL) has no printable form and is not in the vocabulary, so it must
    // be encoded as the byte-fallback token 0x07 + 3 = 10 and decoded back.
    let t = tokenizer();
    let s = "a\u{0007}b";
    let tokens = t.encode(s, true, false);
    assert!(
        tokens.contains(&(0x07u32 + 3)),
        "expected the BEL byte-fallback token in {tokens:?}"
    );
    assert_eq!(content_tokens(&t, s), s);
}

#[test]
fn bos_decodes_to_its_literal_piece() {
    // Pin the behaviour that separates `decode` from `decode_text`, so a
    // future change to how special tokens are handled cannot slip through
    // unnoticed. If someone "fixed" this by making the reference-faithful
    // `decode` skip BOS, the generation path would silently lose characters
    // and this test is what catches it.
    let t = tokenizer();
    assert_eq!(t.decode(&[BOS_ID]), "\n<s>\n");
    assert_eq!(t.decode(&[EOS_ID]), "\n</s>\n");
    // decode_text contributes nothing for the specials, but BOS must still be
    // present in the sequence so the dummy-prefix rule fires. Token 29871 is
    // the bare space and 278 is " the"; because 278 follows 29871 rather than
    // BOS, it keeps its own leading space.
    assert_eq!(t.decode_text(&[BOS_ID]), "");
    assert_eq!(t.decode_text(&[BOS_ID, 29871, 278]), " the");
    assert_eq!(t.decode_text(&[BOS_ID, 278]), "the");
    // decode_one applies the same rule from an explicit previous token.
    assert_eq!(t.decode_one(Some(BOS_ID), 29871), "");
    assert_eq!(t.decode_one(Some(BOS_ID), 278), "the");
    assert_eq!(t.decode_one(Some(29871), 278), " the");
}

// ---------------------------------------------------------------------------
// BOS / EOS handling
// ---------------------------------------------------------------------------

#[test]
fn bos_and_eos_are_added_on_request() {
    let t = tokenizer();
    let text = "hello";

    let with_bos = t.encode(text, true, false);
    assert_eq!(with_bos[0], BOS_ID);
    assert_eq!(with_bos.len(), t.encode(text, false, false).len() + 1);

    let with_eos = t.encode(text, false, true);
    assert_eq!(*with_eos.last().unwrap(), EOS_ID);

    let with_both = t.encode(text, true, true);
    assert_eq!(with_both[0], BOS_ID);
    assert_eq!(*with_both.last().unwrap(), EOS_ID);
    assert_eq!(with_both.len(), with_bos.len() + 1);
}

#[test]
fn the_empty_string_is_bos_and_nothing_else() {
    // The dummy prefix is only added for non-empty input, so the empty string
    // is the one case where encode does not emit the space token.
    let t = tokenizer();
    assert_eq!(t.encode("", true, false), vec![BOS_ID]);
    assert_eq!(t.encode("", true, true), vec![BOS_ID, EOS_ID]);
    assert!(t.encode("", false, false).is_empty());
}

// ---------------------------------------------------------------------------
// The dummy prefix
// ---------------------------------------------------------------------------

#[test]
fn a_leading_space_changes_the_encoding() {
    // This is the whole reason the dummy prefix exists. Without it,
    // "hello" and " hello" would be the same token sequence and the model
    // could not distinguish them.
    let t = tokenizer();
    let bare = t.encode("hello", true, false);
    let spaced = t.encode(" hello", true, false);
    assert_ne!(
        bare, spaced,
        "leading whitespace must change the token sequence"
    );
}

#[test]
fn the_dummy_prefix_is_not_repeated_at_decode_time() {
    // decode strips exactly one leading space from the first piece *after* BOS.
    // If it stripped more, " hello" would come back as "hello". This case has
    // no leading space of its own to lose, which is the point.
    let t = tokenizer();
    let s = " hello world";
    let tokens = t.encode(s, true, false);
    assert_eq!(t.decode_text(&tokens), s);
}

// ---------------------------------------------------------------------------
// Byte fallback arithmetic
// ---------------------------------------------------------------------------

#[test]
fn byte_fallback_ids_are_byte_plus_three() {
    // The contract that makes byte fallback work: the 256 byte tokens sit at
    // ids 3..=258 in order, so id = byte + 3 in both directions. The
    // differential harness and any external consumer depend on this mapping.
    let t = tokenizer();
    for b in 0u8..=255 {
        let id = u32::from(b) + 3;
        assert_eq!(
            t.piece(id),
            Some(format!("<0x{b:02X}>").as_bytes()),
            "byte {b:#04x} should be token {id}"
        );
    }
    assert_eq!(t.vocab_size(), LLAMA2_VOCAB_SIZE);
}

#[test]
fn special_ids_are_where_the_specification_says() {
    let t = tokenizer();
    assert_eq!(t.piece(BOS_ID), Some(&b"\n<s>\n"[..]));
    assert_eq!(t.piece(EOS_ID), Some(&b"\n</s>\n"[..]));
    assert_eq!(t.piece(UNK_ID), Some(&b"<unk>"[..]));
    // The dummy prefix is a plain single-space token.
    assert_eq!(t.id_of(b" "), Some(29871));
}

// ---------------------------------------------------------------------------
// Tokenizer file robustness
// ---------------------------------------------------------------------------

#[test]
fn a_truncated_vocabulary_is_an_error_not_a_panic() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/llama2c/tokenizer.bin");
    let data = std::fs::read(&path).unwrap();

    // Every prefix of the file must be rejected cleanly.
    for frac in [1usize, 25, 50, 90, 99] {
        let n = data.len() * frac / 100;
        let r = Tokenizer::from_bytes(&data[..n], LLAMA2_VOCAB_SIZE);
        assert!(r.is_err(), "a {frac}% file should not load");
    }
    assert!(Tokenizer::from_bytes(&[], LLAMA2_VOCAB_SIZE).is_err());
}

#[test]
fn trailing_bytes_in_the_vocabulary_are_an_error() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/llama2c/tokenizer.bin");
    let mut data = std::fs::read(&path).unwrap();
    data.push(0);
    let err = Tokenizer::from_bytes(&data, LLAMA2_VOCAB_SIZE).unwrap_err();
    assert!(
        matches!(err, proofinfer::tokenizer::TokenizerError::TrailingBytes(1)),
        "got {err:?}"
    );
}

#[test]
fn a_wrong_vocab_size_is_caught_either_way() {
    // The format does not record its own vocabulary size, so asking for the
    // wrong count has to fail in one of the two directions rather than
    // silently producing a misaligned vocabulary.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("reference/llama2c/tokenizer.bin");
    let data = std::fs::read(&path).unwrap();

    assert!(Tokenizer::from_bytes(&data, 100).is_err(), "too few tokens");
    assert!(
        Tokenizer::from_bytes(&data, LLAMA2_VOCAB_SIZE + 1).is_err(),
        "too many tokens"
    );
    assert!(
        Tokenizer::from_bytes(&data, LLAMA2_VOCAB_SIZE).is_ok(),
        "the exact count must load"
    );
}

#[test]
fn random_bytes_never_panic_the_vocabulary_parser() {
    // Same totality contract as the checkpoint loader.
    let mut state = 0x2468_1357u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for i in 0..500usize {
        let len = (i * 13) % 2048;
        let buf: Vec<u8> = (0..len).map(|_| (next() >> 24) as u8).collect();
        let vocab_size = 1 + (i % 64);
        let _ = Tokenizer::from_bytes(&buf, vocab_size);
    }
}
