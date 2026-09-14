"""The handbook's stylesheet. Print-first, because a POH is a binder.

WHAT MAKES THIS DIFFERENT FROM THE PAGE IT REPLACES. `page_design.PAGE_CSS` is a
web page that prints acceptably: it opens the folds and hides the nav, and that
is the whole of its `@media print`. There is no `@page` rule anywhere in the
repo, no page size, no margins, no running heads, no page numbers, and no
control over where a procedure breaks.

A handbook is read under pressure by somebody who needs one page, so the page is
the unit. That means real pagination — `@page` with a size and margins, a
procedure that starts on its own sheet and never splits across two, a running
head carrying the section number and the revision so a loose page can be put
back, and printed page numbers.

THREE CALLOUT LEVELS AND NO MORE, in aviation's order: WARNING (you lose the
game), CAUTION (you lose tempo or a card), NOTE (context). Capped at two per
page by `validate_poh`, because a page with four warnings has none — the reader
learns the colour means nothing, which is the same failure as a validator that
fires on correct data.

REVISION BARS are a margin rule on any block the current revision changed. They
are driven by `manual_revisions.json`, an authored file, rather than by a diff
computed at render — a diff would change the bytes whenever git history moved
and break the byte-identical rebuild this file's whole family depends on.

COLOUR CARRIES MEANING ONLY. The callout levels and the engine stages; nothing
decorative. Black on off-white, one humanist sans for text, one mono for card
data and tables.
"""

import hashlib

POH_CSS = """
/* ── tokens ─────────────────────────────────────────────────────────── */
:root {
  --poh-ink: #16161a;
  --poh-paper: #fbfaf7;
  --poh-rule: #d6d2c8;
  --poh-soft: #6a665e;
  --poh-warning: #a11b1b;
  --poh-caution: #9a6a06;
  --poh-note: #55606e;
  --poh-sans: "Inter", "Helvetica Neue", -apple-system, system-ui, sans-serif;
  --poh-mono: "IBM Plex Mono", "SF Mono", ui-monospace, Menlo, monospace;
}

/* SINGLE THEME, DELIBERATELY. A handbook is printed on white paper; a page that
   inverts on a dark OS and then prints as ink-heavy grey is worse at the one job
   it has. Painted explicitly so it does not borrow a host background. */
html { background: var(--poh-paper); }
body.poh {
  background: var(--poh-paper); color: var(--poh-ink);
  font-family: var(--poh-sans); font-size: 15px; line-height: 1.55;
  margin: 0; padding: 0;
}
.poh-trim { max-width: 46rem; margin: 0 auto; padding: 2rem 1.5rem 5rem; }

/* ── front matter ───────────────────────────────────────────────────── */
.poh-title {
  border: 2px solid var(--poh-ink); padding: 1.4rem 1.6rem; margin-bottom: 2rem;
}
.poh-title h1 { font-size: 1.5rem; margin: 0 0 .2rem; letter-spacing: -.01em; }
.poh-title .sub { color: var(--poh-soft); margin: 0; }
.poh-title .rev {
  font-family: var(--poh-mono); font-size: .8rem; margin-top: .9rem;
  padding-top: .7rem; border-top: 1px solid var(--poh-rule);
}

/* ── numbered sections ──────────────────────────────────────────────── */
.poh-sec { margin: 3rem 0 0; }
.poh-sec > h2 {
  font-size: 1.05rem; letter-spacing: .02em; margin: 0 0 .2rem;
  padding-bottom: .4rem; border-bottom: 2px solid var(--poh-ink);
}
.poh-sec > h2 .n {
  font-family: var(--poh-mono); margin-right: .8rem; color: var(--poh-soft);
}
.poh-sec > .promise { color: var(--poh-soft); margin: .4rem 0 1.4rem; }
.poh-sub { margin: 1.8rem 0; }
.poh-sub > h3 { font-size: .95rem; margin: 0 0 .5rem; }
.poh-sub > h3 .n { font-family: var(--poh-mono); margin-right: .7rem; color: var(--poh-soft); }
a.xref { font-family: var(--poh-mono); color: inherit; }

/* ── callouts: three levels, aviation order ─────────────────────────── */
.poh-call {
  margin: 1.1rem 0; padding: .7rem 1rem .7rem 1.1rem;
  border-left: 4px solid var(--poh-note); background: rgba(0,0,0,.025);
}
.poh-call .lbl {
  display: block; font-family: var(--poh-mono); font-size: .72rem;
  letter-spacing: .12em; margin-bottom: .25rem;
}
.poh-call.warning { border-left-color: var(--poh-warning); }
.poh-call.warning .lbl { color: var(--poh-warning); }
.poh-call.caution { border-left-color: var(--poh-caution); }
.poh-call.caution .lbl { color: var(--poh-caution); }
.poh-call.note .lbl { color: var(--poh-note); }

/* ── revision bars ──────────────────────────────────────────────────── */
.rev-bar { position: relative; }
.rev-bar::before {
  content: ""; position: absolute; left: -1rem; top: 0; bottom: 0;
  border-left: 3px solid var(--poh-ink);
}

/* ── data ───────────────────────────────────────────────────────────── */
table.poh { border-collapse: collapse; width: 100%; margin: .9rem 0;
            font-family: var(--poh-mono); font-size: .82rem; }
table.poh th, table.poh td {
  text-align: left; padding: .3rem .6rem .3rem 0;
  border-bottom: 1px solid var(--poh-rule); vertical-align: top;
}
table.poh th { color: var(--poh-soft); font-weight: 600; }
table.poh td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* WIDE CONTENT SCROLLS INSIDE ITSELF; the page body never scrolls sideways. */
.poh-scroll { overflow-x: auto; }
.ev { color: var(--poh-soft); font-size: .85rem; }
.card { font-variant: small-caps; letter-spacing: .02em; }

/* ── card preview on hover ──────────────────────────────────────────── */
/* NO IMAGE ELEMENT. The page this handbook replaced put 176-308 hidden
   full-card images in the document, one per card mention, and revealed them on
   hover; they were most of a 275 KB file and were hidden on mobile AND on
   paper. `poh.card_attr` instead hands each name the committed Scryfall URL as
   `--poh-card-img`, and the box below is the only thing that ever asks for it.
   A pseudo-element whose rule does not match generates no box, so the fetch
   happens on hover and never before: the document grows by 108 bytes a card
   and the reader pays for exactly the cards they point at.

   SCREEN, POINTER, AND WIDE. `@media screen` keeps it off paper, which is the
   one job this file has. `hover: hover` keeps it off touch, the surface the
   old one "did nothing" on. And the width: the trim is 46rem centred, so a
   246px panel inset 1.5rem clears the text only from 1276px up — below that it
   would sit on the last few characters of every line the reader is reading,
   and with no way to dismiss it.

   THE NAME IS A LINK, into the atlas's Discover mode seeded with that card.
   It carries no underline because 100 underlined names in a table is a page of
   rules rather than a page of cards; the dotted rule and the `:focus` arm are
   what say it is reachable, and `:focus` is also what gives the preview a
   keyboard path the old one never had.

   DO NOT set `transform`, `filter`, `contain`, `will-change`, `perspective`,
   `backdrop-filter` or `content-visibility` on `.poh-trim`, `.poh-sec`,
   `.poh-sub`, `.poh-scroll` or `table.poh`. Any one of them makes that
   ancestor the containing block for fixed-position descendants, and this panel
   would be clipped into a sliver inside `.poh-scroll`'s `overflow-x: auto`
   with nothing to connect the two changes. */
a.cardref {
  color: inherit; text-decoration: none;
  border-bottom: 1px dotted var(--poh-rule);
}
a.cardref:hover, a.cardref:focus { border-bottom-color: var(--poh-ink); }
@media screen and (hover: hover) and (min-width: 80rem) {
  a.cardref.pop:hover::after,
  a.cardref.pop:focus::after {
    content: "";
    position: fixed; top: 1.5rem; right: 1.5rem; z-index: 60;
    /* STATED, not inherited from the absence of a reset: 244x340 is 488x680
       halved, and a `border-box` reset added later would start cropping the
       card frame. */
    box-sizing: content-box;
    width: 244px; height: 340px;
    /* LONGHANDS. An unset custom property inside a shorthand is invalid at
       computed-value time and takes the WHOLE declaration with it, paper
       colour included — a card with no image would pin a transparent ghost
       rather than a blank card back. */
    background-color: var(--poh-paper);
    background-image: var(--poh-card-img, none);
    background-position: center;
    background-size: contain;
    background-repeat: no-repeat;
    border: 1px solid var(--poh-rule);
    box-shadow: 0 6px 24px rgba(0, 0, 0, .22);
    /* Never swallow the cursor, and never animate: a fade only makes the
       first-hover empty box linger. */
    pointer-events: none;
  }
}

/* ── margin figures ─────────────────────────────────────────────────── */
/* FIRST MENTION ONLY, and the name is always text. The page this replaces
   carried 176-308 hidden full-card images that were hidden on mobile AND on
   paper — dead weight on both surfaces a handbook is read on. */
.poh-fig { float: left; width: 7.5rem; margin: .2rem 1.1rem .6rem -9rem; }
.poh-fig img { width: 100%; display: block; border: 1px solid var(--poh-rule); }
.poh-fig figcaption { font-size: .7rem; color: var(--poh-soft); margin-top: .25rem; }
@media (max-width: 60rem) {
  .poh-fig { float: none; margin: .8rem 0; width: 9rem; }
}

/* ── the schematic ──────────────────────────────────────────────────── */
.poh-schematic { margin: 1.4rem 0; }
.poh-schematic svg { width: 100%; height: auto; }
.poh-legend { font-size: .78rem; color: var(--poh-soft); margin-top: .5rem; }

/* ── charts ─────────────────────────────────────────────────────────── */
.poh-chart { margin: 1.4rem 0 1.8rem; }
.poh-chart figcaption { font-size: .8rem; margin-bottom: .5rem; }
.poh-chart .take { font-size: .85rem; color: var(--poh-soft); margin-top: .4rem; }

/* ══ PRINT — the whole point ═══════════════════════════════════════════ */
@page {
  size: A4;
  margin: 18mm 16mm 20mm;
}
@media print {
  html, body.poh { background: #fff; }
  .poh-trim { max-width: none; padding: 0; }
  /* A PROCEDURE OWNS ITS SHEET. */
  .poh-procedure { break-before: page; break-inside: avoid; }
  .poh-sec { break-before: page; }
  .poh-sec:first-of-type { break-before: auto; }
  .poh-sub, .poh-call, table.poh, .poh-chart, .poh-schematic { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
  /* Orphans and widows: a checklist step alone at the foot of a page is a step
     somebody will miss. */
  p, li { orphans: 3; widows: 3; }
  .poh-fig { float: none; margin: .6rem 0; width: 6rem; }
  a.xref { text-decoration: none; }
  /* A URL printed after every card name would be most of the ink. */
  a[href]::after { content: none; }
}
"""


def stylesheet_version():
    """Content hash, so a CSS edit busts the cache without a hand-bumped number."""
    return hashlib.sha256(POH_CSS.encode()).hexdigest()[:8]


def write_stylesheet(directory):
    path = directory / "poh.css"
    path.write_text(POH_CSS, encoding="utf-8")
    return path
