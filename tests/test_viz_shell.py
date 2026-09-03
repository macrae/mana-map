"""The shell strip and the library drawer — source-level invariants.

The strip and the drawer are the only chrome shared by all five viz pages, and
they are shared the hard way: **the CSS block that styles them is duplicated into
both stylesheets.** `index.html` loads only `mana-map.css`; `deck.html`,
`branch.html`, `workbench.html` and `spaces.html` load only `tokens.css`. So a
rule written into one file governs some surfaces and not others, and the failure
is silent — the drawer simply looks different on four pages out of five.

That is what these tests are for. There is no JS runtime here; the browser suite
asserts the drawer's actual behaviour (`test_viz_behaviour.py`). What is decidable
from the source is whether the two copies still say the same thing.

MEASURED BEFORE BEING KEPT, as this repo requires of a new check: the extraction
below finds **37 library rules in each stylesheet, with identical selectors and
identical bodies**. It does not fire on correct data.
"""

import re
from pathlib import Path

VIZ = Path(__file__).resolve().parents[1] / "viz"
SHEETS = (VIZ / "css" / "mana-map.css", VIZ / "css" / "tokens.css")


def _library_rules(path: Path) -> dict[str, str]:
    """Every rule in `path` whose selector names the library chrome.

    Comments are stripped first — they differ between the copies legitimately and
    are not what is being compared. Bodies are whitespace-normalised, because a
    line break moving is not a divergence.
    """
    src = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    out: dict[str, str] = {}
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", src):
        sel = " ".join(m.group(1).split())
        if ".lib-" in sel or ".shell-lib" in sel or ".shell-library" in sel:
            out[sel] = " ".join(m.group(2).split())
    return out


def test_the_library_block_agrees_between_the_two_stylesheets():
    """One drawer, two stylesheets, and nothing but discipline holding them equal.

    `index.html` loads `mana-map.css` alone and the other four pages load
    `tokens.css` alone, so a fix applied to one copy ships to one surface. The
    block even says so in its own comment — which is a note, not a check.
    """
    a, b = (_library_rules(p) for p in SHEETS)
    assert a, "no library rules found in mana-map.css — the extraction is broken"
    assert len(a) >= 30, f"only {len(a)} library rules matched; expected ~37"

    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    assert not only_a, f"only in mana-map.css, so four pages never get it: {only_a}"
    assert not only_b, f"only in tokens.css, so the atlas never gets it: {only_b}"

    differing = sorted(k for k in a if a[k] != b[k])
    assert not differing, (
        "the two copies of the library block disagree on: "
        + "; ".join(f"{k!r} -> {a[k]!r} vs {b[k]!r}" for k in differing))


def test_the_drawer_declares_its_own_scroll():
    """A library longer than the window must scroll, not be clipped.

    On `index.html` the body is a 100vh flex column with `overflow: hidden`, so a
    drawer with no cap and no overflow runs off the bottom with nothing able to
    scroll it — measured at 60 cards in a 900px window: 1683px tall, 783px below
    the fold, and `.main-area` and `.status` pushed off screen with it.

    `min-height: 0` is the non-obvious one. A flex item's automatic minimum size
    is its content's, so without it the drawer cannot be compressed at all and no
    amount of shrinking would have saved it.
    """
    for sheet in SHEETS:
        rules = _library_rules(sheet)
        drawer = rules.get(".lib-drawer")
        assert drawer, f"{sheet.name} has no .lib-drawer rule at all"
        for decl in ("max-height:", "overflow-y: auto", "min-height: 0"):
            assert decl in drawer, (
                f"{sheet.name}: .lib-drawer is missing {decl!r} — {drawer}")
        top = rules.get(".lib-top")
        assert top and "position: sticky" in top, (
            f"{sheet.name}: the drawer's head and pile tabs are not pinned, so its "
            f"own navigation scrolls away with the third row of cards")


def test_the_drawer_head_and_piles_render_inside_the_pinned_block():
    """The sticky wrapper is emitted by `shell.js`, and the CSS is inert without it."""
    src = (VIZ / "js" / "shell.js").read_text(encoding="utf-8")
    assert '<div class="lib-top">' in src, (
        "shell.js does not emit .lib-top, so the sticky rule styles nothing")
    # The head opens the wrapper and `zoneBar()` closes it: both must be inside,
    # or the pile tabs scroll away while the title stays.
    body = src[src.index('<div class="lib-top">'):]
    assert body.index("zoneBar()") < body.index("lib-grid"), (
        "the pile tabs are emitted outside the pinned block")
