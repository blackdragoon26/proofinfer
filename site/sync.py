#!/usr/bin/env python3
"""Render the project's docs and proof README as HTML pages for the landing site.

The site's generated HTML pages come directly from the markdown in the
repository (`verus/README.md` and `docs/*.md`), and the navbar inside
`site/index.html` is kept in sync via `<!-- NAV:START -->` / `<!-- NAV:END -->`
markers so the site and the repository cannot quietly say different things.

    python3 sync.py            # regenerate HTML pages and index.html nav
    python3 sync.py --check    # exit 1 if any committed HTML is stale

Deliberately not a general Markdown implementation. These docs use headings,
fenced code, pipe tables, bullet and numbered lists, bold, inline code and
inline links - and nothing else. A converter that only handles what is actually
present is easier to trust than one that half-handles everything.

One rule matters for correctness: code spans and code blocks are lifted out
*before* any inline markup is applied and put back afterwards, so a `**` or a
`[` inside a code span is left alone. Getting that wrong is the classic way a
Markdown renderer quietly corrupts a code sample.
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
REPO = SITE.parent
GITHUB = "https://github.com/blackdragoon26/proofinfer"
BLOB = f"{GITHUB}/blob/main"

# Repo-relative markdown source path, output filename in site/, page title.
PAGES = [
    ("verus/README.md", "proof.html", "Proof"),
    ("docs/testing.md", "testing.html", "Testing"),
    ("docs/performance.md", "performance.html", "Performance"),
    ("docs/design.md", "design.html", "Design"),
]

NAV = [
    ("Overview", "index.html"),
    ("Proof", "proof.html"),
    ("Testing", "testing.html"),
    ("Performance", "performance.html"),
    ("Design", "design.html"),
]

NAV_START = "<!-- NAV:START -->"
NAV_END = "<!-- NAV:END -->"

# Guards: the converter must never emit a construct it failed to understand,
# because a literal "**" in the output is a visible bug.
UNCONVERTED = re.compile(r"(?<![\w`])\*\*|^\s{0,3}\||^\s*```", re.M)


ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);")


def esc(s: str) -> str:
    """Escape for HTML, but leave an entity that is already an entity.

    The tables carry `max&nbsp;|err|`, so escaping every `&` would print the
    literal text "&nbsp;" instead of a space. Existing entities are held aside
    while the rest is escaped, then put back.
    """
    held: list[str] = []

    def hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00E{len(held) - 1}\x00"

    s = ENTITY.sub(hold, s)
    s = html.escape(s, quote=False)
    for i, ent in enumerate(held):
        s = s.replace(f"\x00E{i}\x00", ent)
    return s


def split_row(line: str) -> list[str]:
    """Split a table row into cells on unescaped pipes only.

    A header like `max&nbsp;|err|` is one cell, not three. Splitting on every
    `|` shattered it into fragments and shifted every column after it.
    """
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.replace(r"\|", "|").strip() for c in re.split(r"(?<!\\)\|", body)]


def inline(text: str, base_dir: str = "docs") -> str:
    """Apply inline markup to one run of text.

    Code spans are lifted out first, so a `**`, a `[` or a `<` inside one is
    left alone. An earlier version returned the placeholder's *index* instead
    of the code it stood for, which silently replaced every identifier in the
    docs with the character "0" - so the stash now keeps the text itself.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00S{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)

    # Escape before applying markup, so a bare < in prose is not read as HTML.
    text = esc(text)

    def link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if url.startswith(("http", "#")):
            href = esc(url)
        else:
            # A link in a doc is relative to that doc's own folder, so
            # "performance.md" in docs/ is docs/performance.md, "kv_cache_bounds.rs"
            # in verus/ is verus/kv_cache_bounds.rs, and "../README.md" is README.md.
            path = posixpath.normpath(posixpath.join(base_dir, url))
            href = f"{GITHUB}/blob/main/{esc(path)}"
        return f'<a href="{href}">{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", text)

    for i, code in enumerate(spans):
        text = text.replace(f"\x00S{i}\x00", f"<code>{esc(code)}</code>")
    return text


def convert(md: str, base_dir: str = "docs") -> str:
    # Drop HTML comments from the source. Each doc opens with one explaining how
    # it was split out of the README; rendered as text it reads as a stray line
    # above the title.
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Fenced code block.
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            lang = m.group(1)
            i += 1
            body = []
            while i < n and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence
            cls = f' class="lang-{esc(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{esc(chr(10).join(body))}</code></pre>")
            out.append("")
            continue

        # Pipe table: a header row followed by a |---|---| separator.
        if line.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            head = split_row(line)
            i += 2
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            out.append("<table>")
            out.append(
                "<thead><tr>"
                + "".join(f"<th>{inline(c, base_dir)}</th>" for c in head)
                + "</tr></thead>"
            )
            out.append("<tbody>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c, base_dir)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>")
            out.append("")
            continue

        # Heading.
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2), base_dir)}</h{level}>")
            i += 1
            continue

        # Bullet or numbered list. One flat level is all these docs use.
        #
        # The item test must be independent of `ordered`. An earlier version
        # gated the numbered case on `not ordered`, so for an ordered list the
        # body never matched "1. ", i was never incremented, and convert()
        # looped forever appending an empty <ol> on each pass. Always consume
        # at least the current line.
        if re.match(r"^\s*[-*]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>")
            while i < n and (
                re.match(r"^\s*[-*]\s+", lines[i])
                or re.match(r"^\s*\d+\.\s+", lines[i])
            ):
                item = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i])
                i += 1
                # A wrapped line is indented and does not start a new item.
                # Without this each item became its own <ol> and the rest of
                # the sentence fell out as a stray paragraph.
                while (
                    i < n
                    and lines[i].strip()
                    and not re.match(r"^\s*(?:[-*]|\d+\.)\s+", lines[i])
                    and lines[i][:1].isspace()
                ):
                    item += " " + lines[i].strip()
                    i += 1
                out.append(f"<li>{inline(item, base_dir)}</li>")
            out.append(f"</{tag}>")
            out.append("")
            continue

        # Blank.
        if not line.strip():
            i += 1
            continue

        # Paragraph: consume until a blank line or the start of another block.
        buf = [line.strip()]
        i += 1
        while i < n and lines[i].strip():
            nxt = lines[i]
            if (
                nxt.startswith(("#", "|", "```"))
                or re.match(r"^\s*[-*]\s+", nxt)
                or re.match(r"^\s*\d+\.\s+", nxt)
            ):
                break
            buf.append(nxt.strip())
            i += 1
        out.append(f"<p>{inline(' '.join(buf), base_dir)}</p>")
        out.append("")

    return "\n".join(out).strip()


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} &middot; proofinfer</title>
<meta name="description" content="proofinfer — {title}: {desc}">
<link rel="stylesheet" href="style.css">
<script src="motion.js" defer></script>
</head>
<body>
<canvas id="bg" aria-hidden="true"></canvas>

<nav>
{nav}
</nav>

<main class="doc">
{body}

<p class="source">Source: <a href="{src}">{src}</a> in the repository.</p>
</main>

</body>
</html>
"""


def nav_html(current: str) -> str:
    items = []
    for label, href in NAV:
        cls = ' class="here"' if href == current else ""
        items.append(f'<a href="{href}"{cls}>{label}</a>')
    return (
        '<div class="navlinks">'
        + "".join(items)
        + "</div>"
        + '<div class="navbtns">'
        + f'<a class="btn" href="{GITHUB}">Repository</a>'
        + '<a class="btn" href="https://sankalpjha.dev/">Sankalp</a>'
        + "</div>"
    )


def sync_index_nav(index_text: str) -> str:
    """Replace the marked `<nav>` block inside `site/index.html`."""
    pattern = re.compile(
        rf"{re.escape(NAV_START)}.*?{re.escape(NAV_END)}",
        flags=re.S,
    )
    if not pattern.search(index_text):
        raise ValueError(
            f"site/index.html is missing {NAV_START} ... {NAV_END} markers"
        )
    replacement = f"{NAV_START}\n<nav>\n{nav_html('index.html')}\n</nav>\n{NAV_END}"
    return pattern.sub(replacement, index_text, count=1)


DESCRIPTIONS = {
    "proof.html": "machine-checked Verus proof of KV-cache and weight-slice index safety, and its explicit limits",
    "testing.html": "how correctness is established, and how every check was shown to fail",
    "performance.html": "throughput, the 8-lane dot product, and the gap left by -Ofast",
    "design.html": "architecture, decisions, trusted computing base, and what is not verified",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the HTML is stale")
    args = ap.parse_args()

    stale = []
    for rel_md, html_name, title in PAGES:
        src = REPO / rel_md
        if not src.exists():
            print(f"error: {src} not found", file=sys.stderr)
            return 2
        raw = src.read_text()
        base_dir = posixpath.dirname(rel_md)
        body = convert(raw, base_dir=base_dir)

        left = UNCONVERTED.search(body)
        if left:
            print(
                f"error: {rel_md} produced unconverted markup near "
                f"{body[left.start():left.start()+40]!r}",
                file=sys.stderr,
            )
            return 2

        page = PAGE.format(
            title=html.escape(title),
            desc=DESCRIPTIONS[html_name],
            nav=nav_html(html_name),
            body=body,
            src=f"{BLOB}/{rel_md}",
        )
        out = SITE / html_name
        if args.check:
            if not out.exists() or out.read_text() != page:
                stale.append(html_name)
        else:
            out.write_text(page)
            print(f"  wrote {html_name}  ({len(page.splitlines())} lines)")

    index_path = SITE / "index.html"
    if not index_path.exists():
        print(f"error: {index_path} not found", file=sys.stderr)
        return 2
    old_index = index_path.read_text()
    try:
        new_index = sync_index_nav(old_index)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.check:
        if old_index != new_index:
            stale.append("index.html")
    else:
        if old_index != new_index:
            index_path.write_text(new_index)
            print("  updated nav in index.html")

    if args.check:
        if stale:
            print("stale (run: python3 site/sync.py): " + ", ".join(stale), file=sys.stderr)
            return 1
        print("  site HTML pages and index.html nav are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
