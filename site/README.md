# proofinfer — landing page

The site for this project, living in the repository at `site/`. Four pages, one
stylesheet, one small script. No framework, no build step, no dependencies. Open
it directly or serve the folder; it works either way.

```sh
open site/index.html
# or, from this folder
python3 -m http.server 8000   # then http://localhost:8000
```

## What is here

| file | what it is |
|---|---|
| `index.html` | the landing page — the only hand-written page |
| `testing.html`, `performance.html`, `design.html` | **generated** from `../docs/` |
| `style.css` | one stylesheet for all four pages |
| `motion.js` | the drifting background |
| `sync.py` | the generator for the three doc pages |
| `check_docs.py`, `check_light.py`, `motion_test.js` | the checks below |

## The doc pages are generated, not written

`testing.html`, `performance.html` and `design.html` are rendered from the
markdown in `../docs/` by `sync.py`. Each page links back to the markdown it came
from, so the site and the docs cannot quietly disagree about what the project
claims.

After changing anything in `../docs/`:

```sh
python3 sync.py           # regenerate the three pages
python3 sync.py --check   # exit 1 if the committed HTML is stale
```

`sync.py` is not a general Markdown implementation, and does not need to be.
These docs use headings, fenced code, pipe tables, bullet and numbered lists,
bold, inline code and inline links, and nothing else. It refuses to emit a page
it does not understand: the `UNCONVERTED` guard fails the run if stray `**`,
`|` or ` ``` ` survives into the output, which is the failure mode that makes a
hand-rolled converter quietly wrong.

To add a doc, add it to `PAGES` in `sync.py` and to `NAV`.

## Publishing

The site is static, so anything that serves files will do. The docs are in this
repository, so `site/` is the publish directory.

**GitHub Pages** — Settings -> Pages -> Deploy from a branch, `main` / `/site`.
No build step; Jekyll will pass the files through, and none of them begin with
an underscore so none will be excluded.

**Netlify / Vercel / Cloudflare Pages** — point them at the repository and set
the publish directory to `site`. No build command.

**Plain host** — copy `site/` to the web root. It is self-contained.

`sync.py` reads the docs from `..`, so it works from a plain checkout with no
arguments. That path matters only when regenerating; the committed HTML does not
depend on it at runtime.

## Editing

- **A result** — add one `<div>` inside `<div class="rows">`, label left and
  figure right. Keep the figures honest: every number on the page is produced
  by a command in the project's `docs/`.
- **A work item** — add an `<li>` to the list under "The work".
- **The description** — the `.lede` paragraph, plus the `<title>` and the
  `<meta name="description">`.
- **The navbar** — the page links live in `index.html` and, for the three
  generated pages, in `NAV` in `sync.py`. Keep them in step.
- **The background** — `COUNT`, `LINK`, `SPEED` and the two alphas at the top
  of `motion.js`. It is meant to be hard to notice; if you find yourself
  looking at it, turn something down.
- **Colours** — there are none. The palette is `#000` on `#fff`, inverted
  automatically if the reader's system is in dark mode.

## Tests

```sh
python3 sync.py --check   # are the committed doc pages current?
python3 check_docs.py     # do they render the markdown exactly?
python3 check_light.py    # is the light palette legible?
node motion_test.js       # does the background actually move?
```

**`sync.py --check` is not enough.** It compares the committed HTML against a
fresh render, so it is happy as long as the converter is deterministic — it
would confirm a page in which every code span had been replaced by the character
`0`, as long as the converter still did that. `check_docs.py` is the check that
catches that: it reduces the source markdown and the rendered page to token
sequences and requires them to be equal, so a dropped, duplicated, reordered or
substituted token is reported with its position.

```sh
# prove the check can go red, by reintroducing the bug it exists to catch
python3 - <<'EOF'
p='design.html'; s=open(p).read()
open(p,'w').write(s.replace('<code>from_bytes</code>','0'))
EOF
python3 check_docs.py     # FAIL at token 264: from_bytes -> 0
```

`check_light.py` exists because this project's light mode has never been
rendered — the browser it is developed against reports `prefers-color-scheme:
dark`, and flipping the system appearance to check would be rude. So it resolves
the cascade statically instead: it reports the contrast of each light-mode
foreground and background pair, and asserts that every dark-mode override of a
non-inherited property has a base counterpart at no greater specificity. That
second assertion is the one worth having — it is the check that would have
caught the bare `th` inside a media query losing to `.doc th` declared below it,
which is how the table headers once rendered light-on-light in dark mode.

`motion_test.js` exists because a backgrounded browser tab never fires
`requestAnimationFrame`, so the animation cannot be observed there however it
behaves. The harness stubs the few APIs `motion.js` uses, captures the frame
callback and steps it by hand. It is a supplement, not a substitute: with a
visible window, the live canvas can be sampled directly and compared over time.

## Keep it minimal

It works because there is nothing to ignore.
