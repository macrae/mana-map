"""The compact page's own stylesheet — the tightening `magazine.css` cannot do.

WHY BOTH SHEETS. `magazine.css` carries the components this page reuses and is
64 KB of measured work: the stack theatre's 285 lines of `:checked ~` rules, the
constellation figure, the dossier folds, the card previews, the tier badges, the
data tables. Re-deriving that from scratch is a week and would produce a second
implementation of the most intricate CSS in the repo. So the page links it, then
overrides on top.

WHAT THE OVERRIDE IS FOR, MEASURED. Rendered against `magazine.css` alone the
page came to 22.0 screens on radagast — already a third of the magazine's 71.3,
because the folds do most of the work. But a magazine is set to be read at
leisure: generous leading, display type at feature size, air between departments.
A technical page is consulted, and consulting wants density. Everything below
buys screens back by tightening type and spacing, and nothing below changes what
is on the page.

NO `<script>`. Same rule as the magazine: a page is a standalone file that
rebuilds byte-identically and prints.
"""

import hashlib

from manamap.config import MANUALS_DIR

PAGE_CSS = """
/* ── The compact deck page. Loaded AFTER magazine.css, overrides only. ── */

body.page { font-size: 15px; line-height: 1.5; }
.page-trim { max-width: 940px; margin: 0 auto; padding: 0 20px 48px; }

/* Header: one line, not a cover. */
.page-head { padding: 24px 0 10px; border-bottom: 3px solid var(--rule, #2a2a2a); }
.page-head h1 {
  font-family: var(--display, sans-serif); text-transform: uppercase;
  font-size: clamp(22px, 4vw, 34px); line-height: 1.02; margin: 0 0 4px;
}
.page-sub { margin: 0 0 6px; font-family: var(--condensed, sans-serif);
            text-transform: uppercase; letter-spacing: .04em; font-size: 13px; }
.page-legend { margin: 0; display: flex; gap: 6px; flex-wrap: wrap; }

/* The nav rail replaces a contents page: one line, sticky, no prose. */
.page-nav {
  position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap; gap: 2px 14px;
  padding: 8px 0; margin-bottom: 4px;
  background: var(--paper, #fff); border-bottom: 1px solid var(--rule, #2a2a2a);
  font-family: var(--condensed, sans-serif); text-transform: uppercase;
  font-size: 11.5px; letter-spacing: .05em;
}
.page-nav a { text-decoration: none; }

/* Sections: a rule and a line of type, not a department opener. */
.page .dept { margin: 0; padding: 20px 0 4px; border: 0; }
.page .dept + .dept { border-top: 1px solid var(--rule, #2a2a2a); }
.page .dept-title {
  font-family: var(--display, sans-serif); text-transform: uppercase;
  font-size: 19px; line-height: 1; margin: 0 0 3px;
}
/* `magazine.css` already owns `.dept-promise` and sets `margin-left:auto;
   text-align:right; max-width:34ch` — a department opener whose promise hangs off
   the right edge. Reusing the class name inherits that, so the override has to
   name every property it is undoing, not just the ones it wants. This is the
   cost of linking the magazine's sheet, and it is still cheaper than
   re-deriving 64 KB of measured CSS. */
.page .dept-promise {
  margin: 0 0 12px; margin-left: 0; max-width: none; text-align: left;
  font-size: 12.5px; letter-spacing: .04em; opacity: .78;
}
.page .dept-promise .badge { margin-left: 5px; vertical-align: middle; }

/* Prose is the only thing that gets a reading measure. Tables and figures use
   the full trim, because a table narrowed to 62ch just gets taller. */
.page .dept > p, .page .dept > .ev, .page .line > p { max-width: 68ch; }
.page p { margin: 0 0 9px; }

/* Data tables carry most of this page. Dense, and scrollable rather than
   reflowed — a roster that wraps is a roster nobody can scan. */
.page table.data { font-size: 12.5px; margin: 8px 0 14px; width: 100%; }
.page table.data th, .page table.data td { padding: 2px 8px 2px 0; }
.page table.data caption {
  font-family: var(--condensed, sans-serif); text-transform: uppercase;
  font-size: 11px; letter-spacing: .05em; text-align: left; padding-bottom: 4px;
}
.page .roster td:first-child { width: 46%; }
.page .roster td:nth-child(2) { width: 12%; white-space: nowrap; }
.page .short { color: var(--stamp-red, #b00); }

/* City heads: a coloured rule and a name, not a spread. */
.page .city-head { margin: 14px 0 2px; padding: 0 0 2px; font-size: 14px; }
.page .city-head .gloss { font-size: 12px; opacity: .75; }

/* The lines. The board block is already at reference weight; this stops the
   article furniture doubling its height. */
.page .line { margin: 0 0 18px; }
.page .line h3 { font-size: 15.5px; margin: 0 0 2px; }
.page .line-dek { font-size: 13px; opacity: .8; margin: 0 0 6px; max-width: 68ch; }
.page .theatre-fold { margin: 8px 0; }
.page .line-result {
  border-left: 3px solid var(--tier-verified, #4CAF50); padding-left: 10px;
  margin: 8px 0 0; font-size: 13.5px; max-width: 68ch;
}
.page .theatre-fold > summary,
.page .spread > summary,
.page .assumptions > summary {
  cursor: pointer; font-family: var(--condensed, sans-serif);
  text-transform: uppercase; font-size: 11.5px; letter-spacing: .05em;
  padding: 4px 0;
}

/* Opening-hand histogram — the one figure nothing rendered before. */
.page .hist { margin: 10px 0 14px; }
.page .hist figcaption {
  font-family: var(--condensed, sans-serif); text-transform: uppercase;
  font-size: 11px; letter-spacing: .05em; margin-bottom: 4px;
}
.page .hist-row { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.page .hist-k { width: 1.2em; text-align: right; opacity: .7; }
.page .hist-bar {
  flex: 1; height: 9px; background: rgba(128,128,128,.16); border-radius: 2px;
}
.page .hist-bar i { display: block; height: 100%; background: var(--tier-data, #4A7BFF);
                    border-radius: 2px; }
.page .hist-v { width: 3.4em; text-align: right; font-variant-numeric: tabular-nums; }

.page .takeaways { margin: 6px 0 12px; padding-left: 18px; max-width: 68ch; }
.page .takeaways li { margin-bottom: 4px; }

.page .page-foot {
  margin-top: 26px; padding-top: 12px; border-top: 1px solid var(--rule, #2a2a2a);
  font-size: 11.5px; opacity: .72;
}
.page .page-fan { font-size: 10.5px; }

/* PRINT. Everything folded opens, because paper has no click. The magazine's
   sheet does this for `.dossier` only, so without these three the theatre prints
   one step and hides the rest. */
@media print {
  .page-nav { display: none; }
  .page details > *:not(summary) { display: block !important; }
  .page details { break-inside: avoid; }
}
"""


def stylesheet_bytes():
    return PAGE_CSS.encode("utf-8")


def stylesheet_version():
    """Content-addressed, exactly like the magazine's — so a CSS edit obliges a
    rebuild and a rebuild with no CSS edit is a no-op."""
    return hashlib.sha256(stylesheet_bytes()).hexdigest()[:8]


def write_stylesheet():
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANUALS_DIR / "page.css"
    path.write_bytes(stylesheet_bytes())
    return path


def stylesheet_link():
    return f'<link rel="stylesheet" href="page.css?v={stylesheet_version()}">'
