"""The gate on a rendered handbook.

WHAT IT CHECKS IS THE RENDERED HTML, not a JSON artifact — which is unusual here
and is the point. `manuals/p/` has never had a gate: `make manuals` renders the
magazine and has never called the compact-page renderer, so CI's
`git diff --exit-code -- manuals/` could not fail on it because nothing
regenerated it. And no test imported `build_page` at all, while `build_page.py`
claimed in a comment that "a test asserts exactly that" about a rule nothing
asserted.

So the checks here are the ones that are decidable from the output and that
would have caught something real:

  A CONTENTS ENTRY THAT GOES NOWHERE. Every `xref` must resolve to an id on the
  page. A handbook cross-references by NUMBER precisely because it is read out
  of order, and a number pointing at a page that did not render is worse than
  no reference at all.

  MORE THAN TWO CALLOUTS ON A PAGE. Aviation's three levels only work while they
  are rare. A page with four warnings has none, and the reader learns the colour
  means nothing — the same failure as a validator that fires on correct data.

  A `<script>` TAG. Inherited rule, and the one that keeps the file printable
  and standalone.

  A BUILD DATE. Any `datetime.now()` reaching the page would make every rebuild
  a diff and destroy the determinism claim the CI gate rests on. Checked by
  looking for today's date in the output, which is crude and catches the actual
  mistake.

WHAT IT DELIBERATELY DOES NOT CHECK: whether the prose is any good, whether a
section is complete, or whether a figure is the right figure. None of that is
mechanically decidable, and a validator that fires on correct data is worse than
none.
"""

import datetime
import re
import sys

from manamap.config import MANUALS_DIR
from manamap.pilot import poh_spec as spec

_ID_RE = re.compile(r'id="(s[0-9-]+)"')
_XREF_RE = re.compile(r'href="#(s[0-9-]+)"')
_CALLOUT_RE = re.compile(r'class="poh-call ([a-z]+)"')
_SECTION_SPLIT = re.compile(r'<section class="poh-sec')


def validate(html, slug):
    errors, notes = [], []

    if "<script" in html.lower():
        errors.append(
            "the page carries a <script> tag — a handbook is a standalone file "
            "that prints; everything that folds is <details>")

    # A BUILD DATE IS A DIFF ON EVERY REBUILD.
    today = datetime.date.today().isoformat()
    if today in html:
        errors.append(
            f"the page contains today's date ({today}) — a build date breaks the "
            f"byte-identical rebuild the CI gate depends on. Dates come from "
            f"versions.json, never from render time")

    ids = set(_ID_RE.findall(html))
    dangling = sorted({x for x in _XREF_RE.findall(html) if x not in ids})
    if dangling:
        errors.append(
            f"cross-reference(s) to a section that did not render: "
            f"{', '.join(dangling)} — a numbered reference is the one thing a "
            f"reader cannot resolve by scrolling")

    # THE CALLOUT CAP, AND THE UNIT IS A PAGE.
    #
    # Measured against the whole fleet before being kept, which is what this
    # repo requires of a new check — and the first cut FAILED that measurement.
    # It counted per SECTION and fired on four decks, every one of them for the
    # same correct reason: Systems has one subsection per engine stage, each
    # carrying a CAUTION when that stage has a single point of failure, and a
    # deck with four fragile stages is telling the truth four times.
    #
    # A validator that fires on correct data is worse than none. So the unit is
    # the SUBSECTION where there are subsections — which is what a page is in
    # print, since `.poh-sub` avoids breaking inside — and the section itself
    # only where there are none.
    for chunk in _SECTION_SPLIT.split(html)[1:]:
        subs = re.split(r'<div class="poh-sub"', chunk)
        pages = subs[1:] if len(subs) > 1 else [chunk]
        for page in pages:
            found = _CALLOUT_RE.findall(page)
            if len(found) > spec.MAX_CALLOUTS_PER_PAGE:
                head = re.search(r'<h3>.*?</h3>|<h2>.*?</h2>', page) \
                    or re.search(r'<h2>.*?</h2>', chunk)
                where = (re.sub("<[^>]+>", " ", head.group(0)).strip()
                         if head else "?")
                errors.append(
                    f"{len(found)} callouts on one page ({where}) — the cap is "
                    f"{spec.MAX_CALLOUTS_PER_PAGE}. A page with four warnings "
                    f"has none")
        for level in _CALLOUT_RE.findall(chunk):
            if level not in spec.CALLOUTS:
                errors.append(f"{level!r} is not a callout level — "
                              f"one of {sorted(spec.CALLOUTS)}")

    # REPORTED, NEVER FAILED. A handbook missing its authored half is a normal
    # intermediate state; incompleteness belongs to whoever writes section 3,
    # not to a gate that reddens the board while they do.
    rendered = {m for m in ids if "-" not in m}
    missing = [s[0] for s in spec.SECTIONS if f"s{s[0]}" not in rendered]
    if missing:
        notes.append(f"section(s) not rendered: {', '.join(missing)}")
    if not any(x.startswith("s") for x in ids):
        errors.append("no numbered sections rendered at all")
    return errors, notes


def main(args):
    slug = args.slug
    path = MANUALS_DIR / "p" / f"{slug}.html"
    if not path.exists():
        print(f"OK   {slug} — no handbook rendered yet "
              f"(`manamap pilot build-poh {slug}`)")
        return 0
    html = path.read_text(encoding="utf-8")
    errors, notes = validate(html, slug)
    for n in notes:
        print(f"NOTE {n}")
    if errors:
        print(f"FAIL {slug} handbook ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    n_sec = len({m for m in _ID_RE.findall(html) if "-" not in m})
    print(f"OK   {slug} — {n_sec} section(s), {len(html):,} bytes, no script, "
          f"no build date" + (f"; {len(notes)} note(s)" if notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(type("Args", (), {"slug": sys.argv[1]})()))
