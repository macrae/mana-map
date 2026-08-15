"""Pilot: how long is this issue, and where did the length go?

The magazine got good and got long. Vol. 009 measured 43,494 words and 74.5
screens of scroll — about 62 A4 pages, where a real issue is 30–50 including
full-page art. Nothing in the repo could say that, so length drifted the way any
unmeasured quantity drifts: one department at a time, each addition defensible.

Two numbers per section, and the gap between them is the point:

- **words** — everything the section contains.
- **visible words** — everything a reader scrolls past, which excludes content
  inside a collapsed `<details>`.

They disagree sharply and the disagreement is load-bearing. Judge's Desk is 21% of
the issue's words and 2.4% of its scroll, because all seven case files are folded.
Reading only the first number sends you to cut the appendix, which is the one
department that costs the reader nothing and holds the proof; reading only the
second hides that the same 120 citations are also printed in The Kill. A length
report that gives one number gives the wrong instruction half the time.

`--rendered` adds true pixel heights via playwright when it is installed. It is
opt-in because the base command must stay a plain, dependency-free read of a
tracked file — the thing you can run in any clone, before and after a cut.

Deterministic, no LLM, no network. Report-only: it never edits an artifact.
"""

import html
import json
import re

from manamap.config import MANUALS_DIR

# A screen at the 1100x900 the issue was measured on. Only used to turn a pixel
# height into a unit anyone can picture; the pixel number is the real one.
SCREEN_PX = 900

# Where the issue should land. Not enforced here — `validate-issue --strict` owns
# enforcement, and this command's job is to say where you are, not to fail you.
TARGET_SCREENS = 40

_SECTION_RE = re.compile(r'<section class="dept" id="([a-z0-9-]+)"')
# Everything inside a closed `<details>` EXCEPT its `<summary>`. The summary is
# the case row — it is on screen, it is what the reader scans, and folding it in
# with the body undercounts the appendix to almost nothing.
_DETAILS_RE = re.compile(
    r"(<details\b(?![^>]*\bopen\b)[^>]*>)(.*?)(</details>)", re.S)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>.*?</summary>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def words(fragment):
    """Word count of the text a fragment renders. Tags and entities resolved."""
    return len(html.unescape(_TAG_RE.sub(" ", fragment)).split())


def visible_words(fragment):
    """Words a reader scrolls past — collapsed `<details>` content excluded.

    `<details>` without `open` is closed on load, so its body occupies no height
    until someone asks for it. That is the entire difference between Judge's Desk
    being the issue's second-largest section and its second-smallest.

    The `<summary>` stays counted: it is the row on the page, and dropping it
    would report the appendix at nearly zero when a reader does scroll past one
    line per case.
    """
    return words(_DETAILS_RE.sub(
        lambda m: " ".join(_SUMMARY_RE.findall(m.group(2))), fragment))


def sections(markup):
    """`[(id, fragment)]` in document order."""
    parts = _SECTION_RE.split(markup)
    return [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


def measure(slug, rendered=False):
    path = MANUALS_DIR / f"{slug}.html"
    if not path.exists():
        raise SystemExit(f"{path} not found — run `manamap pilot build-manual {slug}`")
    markup = path.read_text(encoding="utf-8")

    rows = [{"id": sid, "words": words(frag), "visible_words": visible_words(frag)}
            for sid, frag in sections(markup)]
    if rendered:
        for row, px in zip(rows, _rendered_heights(path)):
            row["px"] = px
            row["screens"] = round(px / SCREEN_PX, 1)

    total_words = sum(r["words"] for r in rows)
    total_visible = sum(r["visible_words"] for r in rows)
    doc = {
        "slug": slug,
        "sections": rows,
        "totals": {
            "words": total_words,
            "visible_words": total_visible,
            "folded_words": total_words - total_visible,
            "bytes": len(markup),
        },
    }
    if rendered:
        px = sum(r.get("px", 0) for r in rows)
        doc["totals"]["px"] = px
        doc["totals"]["screens"] = round(px / SCREEN_PX, 1)
        doc["totals"]["target_screens"] = TARGET_SCREENS
    return doc


def _rendered_heights(path):
    """True section heights, in document order. Opt-in; needs playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("--rendered needs playwright: pip install -e '.[dev]' "
                         "and `playwright install chromium`")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": SCREEN_PX})
        page.goto(path.resolve().as_uri())
        # The page loads card art from Scryfall; heights are wrong until the
        # images that reserve space have settled.
        page.wait_for_load_state("networkidle")
        heights = page.evaluate(
            "() => [...document.querySelectorAll('section.dept')]"
            ".map(s => Math.round(s.getBoundingClientRect().height))")
        browser.close()
    return heights


def main(args):
    doc = measure(args.slug, rendered=getattr(args, "rendered", False))
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2))
        return 0

    totals = doc["totals"]
    rendered = "px" in totals
    rows = sorted(doc["sections"],
                  key=lambda r: -(r.get("px") or r["visible_words"]))

    head = f"{'section':22}{'words':>8}{'visible':>9}"
    if rendered:
        head += f"{'screens':>9}{'% scroll':>10}"
    print(f"{doc['slug']}\n{head}")
    for r in rows:
        line = f"{r['id']:22}{r['words']:8,}{r['visible_words']:9,}"
        if rendered:
            share = 100 * r["px"] / totals["px"] if totals["px"] else 0
            line += f"{r['screens']:9.1f}{share:9.1f}%"
        print(line)

    print(f"\n{'TOTAL':22}{totals['words']:8,}{totals['visible_words']:9,}")
    if totals["folded_words"]:
        print(f"  {totals['folded_words']:,} words are folded into collapsed case "
              f"files — they cost the reader no scroll until asked for.")
    if rendered:
        over = totals["screens"] - TARGET_SCREENS
        verdict = (f"{over:.1f} screens OVER" if over > 0
                   else f"{-over:.1f} screens under")
        print(f"  {totals['screens']:.1f} screens against a {TARGET_SCREENS}-screen "
              f"target — {verdict}.")
    else:
        print("  Run with --rendered for true scroll height (needs playwright).")
    return 0


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot issue-length <slug>`.")
