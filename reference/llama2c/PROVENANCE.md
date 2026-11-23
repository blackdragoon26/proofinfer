# Provenance: vendored `llama2.c` reference

These files are vendored **verbatim** from upstream to serve as the oracle for
differential testing. `model.py` and `export.py` produce the reference logits
that `reference/diff_test.py` compares against, `test.c` pins the expected
token ids that `tests/tokenizer.rs` asserts, and `run.c` is the independent C
engine whose greedy output must match ours byte-for-byte. The value of this
directory is entirely proportional to its being an untouched copy of upstream.

| | |
|---|---|
| Upstream repository | https://github.com/karpathy/llama2.c |
| Branch | `master` (GitHub source tarball, top-level dir `llama2.c-master/`) |
| Commit | Not pinned — see note below |
| Retrieval date | 2026-09-27 (UTC) |
| Upstream file mtime | 2024-05-29 22:31:04 UTC (as shipped in the tarball) |
| License | MIT License, Copyright (c) 2023 Andrej Karpathy (see `LICENSE`) |

**On the commit:** the vendored source is the GitHub-generated tarball
`llama2.c.tar.gz`, which ships no `.git` directory, so no commit SHA is
recoverable from the artifact itself. The tarball is pinned by content instead:
the sha256 values in the table below fully identify this vendored snapshot. A
reviewer who needs a commit-level pin should re-fetch `master` and confirm these
hashes still match, or resolve them against upstream history.

## Vendored files

`filename`, `bytes`, and the full 64-character sha256 digest:

| filename | bytes | sha256 |
|---|---:|---|
| `model.py` | 15289 | `54f35dbd347100c0066da93dfdf9f19c2ec1e8758bfffcedbdfdb09d5c343e1f` |
| `export.py` | 24513 | `aba8d36ee881478bf59e13f63224e37a00b51a18cacbc38d0def52cc3c0cb475` |
| `tokenizer.py` | 2866 | `585d1c34be6f387d3143297c7fbbce49ad362dc9cecd9f68037bbed8126bcf33` |
| `tokenizer.bin` | 433869 | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |
| `LICENSE` | 1063 | `e87b912002f04cdfa837eaf7f70548ab79b9cb62a285495edf575f3da875aec4` |
| `test.c` | 3574 | `ea6d3b4368147d5aea36024e2294f4b69709fa44bff2c7b70aed86682de4f76a` |
| `run.c` | 38545 | `9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd` |

`test.c` is vendored for the token vectors it pins; `run.c` is the C reference
implementation used for the byte-identical greedy comparison benchmark.

## Integrity statement

Every file above is **byte-for-byte unmodified** relative to upstream. No
reformatting, no trailing-whitespace stripping, no re-save, no line-ending
normalization, no encoding change. The files were transferred with `cp -p`
straight from the extracted tarball and never opened for writing afterward.

To verify:

```sh
python3 reference/check_provenance.py
# or directly:
cd reference/llama2c
sha256sum model.py export.py tokenizer.py tokenizer.bin LICENSE test.c run.c
```

The full digests must match the table above. To verify against a
freshly downloaded upstream tarball, extract it and `diff -r` this directory
against the upstream copies of the same seven files; the result must be empty.

Verification recorded at vendoring time: `diff -r` between a pristine extraction
of `llama2c.tar.gz` and this directory produced no output (exit status 0), and
`cmp` reported every file identical.

Note on line endings: the text files are LF-only, as shipped. No CRLF
conversion was needed or performed.
