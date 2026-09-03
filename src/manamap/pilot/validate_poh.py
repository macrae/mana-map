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

    # A BUILD DATE IS A DIFF ON EVERY REBUILD — but only in the FURNITURE.
    #
    # The first cut looked for today's date anywhere on the page and fired on
    # correct data the first time real content arrived: an emergency page cited
    # "the 2026-09-02 Forge run" as its grounding, which is exactly the sort of
    # dated evidence the handbook SHOULD carry. A validator that fires on
    # correct data is worse than none.
    #
    # A build date lands in the title block, because that is where a renderer
    # stamps one. Authored prose citing a date is content. So the check is
    # scoped to the furniture, and the byte-identical rebuild test is what
    # actually proves the whole page is a pure function of its inputs.
    today = datetime.date.today().isoformat()
    title = re.search(r'<div class="poh-title">.*?</div>', html, re.S)
    if title and today in title.group(0):
        errors.append(
            f"the title block contains today's date ({today}) — a build date "
            f"breaks the byte-identical rebuild the CI gate depends on. Dates "
            f"come from versions.json, never from render time")

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

    # ── the authored half ───────────────────────────────────────────────
    #
    # Only what is decidable ABOUT A CHECKLIST, which is not much: whether the
    # conditions are in the closed set, whether the steps are numbered where
    # order matters, and whether a page is so long nobody finishes it. Whether a
    # step is FOLLOWABLE is the whole content of the section and no regex knows.
    # SPLIT ON THE MARKER, do not try to match a balanced <div>. The first cut
    # used a lookahead for "another procedure or end of string" and matched
    # nothing at all, because in real markup a procedure is followed by
    # `</section>` — a regex that silently matches zero blocks is a check that
    # silently does not run, which is the failure this whole file is about.
    # AND EACH CHUNK IS BOUNDED. Splitting alone leaves the LAST procedure
    # running to the end of the document, so it absorbs every list item in every
    # later section — it reported "53 steps" for a five-step page. Second time
    # this shape of mistake landed in this file: a split or a lookahead that
    # looks right and silently measures the wrong span.
    for raw in re.split(r'<div class="poh-procedure">', html)[1:]:
        page = re.split(r'</section>|<div class="poh-procedure">', raw)[0]
        head = re.search(r'<h3>.*?</h3>', page)
        where = re.sub("<[^>]+>", " ", head.group(0)).strip() if head else "?"
        # IMMEDIATE ACTION IS ORDERED. Step three before step one loses the game,
        # so the page must render <ol> and not <ul> for it.
        if "Immediate action" in page:
            after = page.split("Immediate action", 1)[1][:400]
            if "<ol>" not in after:
                errors.append(
                    f"{where}: immediate action is not a numbered list — order is "
                    f"the whole content of an emergency checklist")
        steps = re.findall(r"<li>", page)
        if len(steps) > 20:
            notes.append(f"{where}: {len(steps)} steps on one page — a checklist "
                         f"nobody finishes under pressure")

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
