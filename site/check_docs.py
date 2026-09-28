"""Does the rendered page actually contain the markdown's content?

Structural counts (how many tables, how many <li>) only prove the converter ran,
and a substring check on raw HTML is worse: the page has tags inside sentences
and wraps lines differently from the source, so it reports divergence that is
not there. The claim being tested is simply that a faithful rendering contains
the same tokens, in the same order, as the source it came from.

So: reduce the source markdown and the rendered page to a token sequence each,
and require them to be equal. Any dropped, duplicated, reordered or invented
token shows up as a divergence, reported with its position.
"""

import html
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
REPO = SITE.parent.parent / "tinyinfer"
PAIRS = [
    ("testing.md", "testing.html"),
    ("performance.md", "performance.html"),
    ("design.md", "design.html"),
]

# Code-ish tokens: identifiers, numbers with exponents and signs, paths,
# operators. "|" is dropped on both sides because it is table punctuation in the
# source and a literal character in the cell text.
TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/<>=*:-]*")

# Tags that imply a word boundary, and tags that sit inside a word or sentence.
# Replacing every tag with a space turns "</a>." into ". " and loses a token;
# replacing every tag with nothing runs "one</p><p>two" together.
BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th",
         "pre", "div", "br", "table", "ul", "ol", "section", "main"}


def strip_tags(html_text: str) -> str:
    def sub(m: re.Match) -> str:
        name = re.match(r"</?([a-zA-Z0-9]+)", m.group(0))
        return " " if (name and name.group(1).lower() in BLOCK) else ""
    return re.sub(r"<[^>]+>", sub, html_text)


def tokens(text: str) -> list[str]:
    text = html.unescape(text).replace("|", " ")
    return TOKEN.findall(text)


def md_tokens(path: Path) -> list[str]:
    raw = path.read_text()
    raw = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)  # comments are not prose
    out: list[str] = []
    for line in raw.splitlines():
        if line.lstrip().startswith("```"):
            # The fence's language is metadata, and the page carries it in a
            # class attribute, so neither side contributes a token. An earlier
            # version added a hardcoded "bash" here, which the page then lacked.
            continue
        s = re.sub(r"^#{1,6}\s+", "", line.strip())
        # A list marker is structure, not prose: <ol> renders "1." through CSS
        # and the marker never appears in the page's text.
        s = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
        s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # link -> its label
        s = s.replace("**", "").replace("`", "")
        s = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"\1", s)  # *emphasis*
        out += tokens(s)
    return out


def html_tokens(path: Path) -> list[str]:
    body = path.read_text().split('<main class="doc">', 1)[1].split("</main>", 1)[0]
    # The "Source:" line is the generator's own footer, not the document's text.
    body = re.sub(r'<p class="source">.*?</p>', " ", body, flags=re.S)
    return tokens(strip_tags(body))


def first_diff(a: list[str], b: list[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            lo = max(0, i - 6)
            return f"at token {i}\n              source: ...{' '.join(a[lo:i+6])}\n              page:   ...{' '.join(b[lo:i+6])}"
    if len(a) != len(b):
        longer, name = (a, "source") if len(a) > len(b) else (b, "page")
        i = min(len(a), len(b))
        tail = " ".join(longer[i:i + 14])
        return f"{name} has {abs(len(a) - len(b))} extra token(s) from {i}: {tail!r}"
    return ""


def main() -> int:
    failures = 0
    for md_name, html_name in PAIRS:
        a = md_tokens(REPO / "docs" / md_name)
        b = html_tokens(SITE / html_name)
        if a == b:
            print(f"  ok    {html_name:<18} {len(a):5d} tokens, identical to the source")
        else:
            print(f"  FAIL  {html_name:<18} {len(a)} source vs {len(b)} page tokens")
            print("            " + first_diff(a, b))
            failures += 1

    print("\nevery doc page renders the repository's markdown exactly"
          if not failures else f"\n{failures} doc page(s) diverge")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
