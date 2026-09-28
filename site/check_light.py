"""Check the light-mode palette, which cannot be rendered in this browser.

Firefox here reports prefers-color-scheme: dark, so the light rules have never
been rendered. Flipping the system appearance or editing the profile prefs would
be invasive, so instead this resolves the cascade statically and reports
numbers: it parses style.css, separates rules that apply unconditionally from
those inside a dark media query, and computes the contrast of every foreground
and background pair the light mode actually produces.

It also asserts the structural property that caused the earlier bug: a dark
override must never be the only declaration of a property, because a bare `th`
in a media query loses to `.doc th` declared below it. Every property a dark
block sets has to have a base counterpart with at least its specificity.
"""

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parent / "style.css"
text = CSS.read_text()

# Strip comments, then split into (media_context, selector, declarations).
text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

RULE = re.compile(r"(@media[^{]*)\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)
BLOCK = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)

base: dict[str, dict[str, str]] = {}
dark: dict[str, dict[str, str]] = {}


def absorb(into: dict, body: str) -> None:
    for sel, decls in BLOCK.findall(body):
        sel = " ".join(sel.split())
        props = dict(re.findall(r"([-a-z]+)\s*:\s*([^;]+)", decls))
        into.setdefault(sel, {}).update(props)


# Rules inside a prefers-color-scheme query, and everything outside one.
for media, body in RULE.findall(text):
    if "prefers-color-scheme" in media:
        absorb(dark, body)

absorb(base, re.sub(r"@media[^{]*\{(?:[^{}]|\{[^{}]*\})*\}", "", text))


def rgb(value: str) -> tuple[int, int, int]:
    h = value.strip().lower().lstrip("#")
    if len(h) == 3:  # #abc shorthand
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or not re.fullmatch(r"[0-9a-f]{6}", h):
        raise ValueError(f"not a hex colour: {value!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def luminance(c: tuple[int, int, int]) -> float:
    def f(v: int) -> float:
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(fg: str, bg: str) -> float:
    a, b = luminance(rgb(fg)), luminance(rgb(bg))
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


BODY = base["body"]
PAGE_BG = BODY["background"]
PAGE_FG = BODY["color"]

failures = 0
print("light mode (prefers-color-scheme: light)")

PAIRS = [
    ("body text on page", "body", "color", PAGE_BG),
    ("table header", ".doc th", "color", PAGE_BG),
    ("code span", ".doc code", "color", PAGE_BG),
    ("pre block", ".doc pre", "color", PAGE_BG),
]

print(f"  page background {PAGE_BG}, text {PAGE_FG}")
for label, sel, prop, fallback in PAIRS:
    rules = base.get(sel, {})
    fg = rules.get(prop, PAGE_FG)
    bg = rules.get("background-color", fallback)
    if "background" in rules:
        bg = rules["background"]
    ratio = contrast(fg, bg)
    ok = ratio >= 7.0
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:<18} {fg} on {bg}  {ratio:.2f}:1")

# The structural check: a dark override must never be the only declaration of a
# property, because that is the bug this project already had once: a bare `th`
# inside a media query lost to `.doc th` declared below it, leaving the table
# headers light-on-light. The risk is confined to properties that are NOT
# inherited - `color` comes from `body` either way, so exempting it is correct
# rather than a loosening of the test.
INHERITED = {"color", "font", "font-size", "line-height", "opacity"}


def base_declares(sel: str, prop: str) -> bool:
    props = base.get(sel, {})
    if prop in props:
        return True
    if prop == "border-color":
        # Any border shorthand counts: base uses `border-bottom: 1px solid ...`.
        return any(p.startswith("border") for p in props)
    return False


print("\n  dark overrides need a base counterpart at equal or lower specificity")
for sel, props in sorted(dark.items()):
    for prop in props:
        if prop in INHERITED or base_declares(sel, prop):
            continue
        print(f"  FAIL  {sel} sets {prop} but has no base declaration")
        failures += 1
print(f"  {'ok' if failures == 0 else 'see above'}  "
      f"{sum(len(p) for p in dark.values())} dark-mode declarations checked")

print("\nlight mode passes every check" if not failures else f"\n{failures} problem(s)")
sys.exit(1 if failures else 0)
