"""One cache-bust value per page, and one owner for the manifest's.

`CLAUDE.md` has said "cache-bust `?v=N` on every script and CSS tag" since the
frontend was split, and the busts were checked one page at a time: `index.html`
by `test_viz_drill.py:168`, `deck.html` only for the PRESENCE of a bust, and
`workbench.html`, `branch.html` and `spaces.html` not at all. On 2026-09-12
`branch.html` sat at 216 while the other four were at 215 — harmless in itself,
and exactly the drift that makes a reader stop trusting the number.

The deeper one was `data/decks/index.json`, fetched FOUR different ways:

    workbench.js   '?v=3'
    build.js       '?v=2'
    discovery.js   '?v=' + MM.DATA_VERSION        (9)
    branch-view.js  no bust, `cache: 'no-cache'`

Two of them were already wrong, so a manifest shape change reached some readers
and not others. `api.js` — the one module every page loads — now owns it.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "viz"
PAGES = sorted(VIZ.glob("*.html"))


def _busts(page):
    return [int(n) for n in re.findall(r"\?v=(\d+)", page.read_text(encoding="utf-8"))]


def test_every_page_busts_all_of_its_assets_at_one_version():
    """Within a page, a mismatched pair is how `build.js` ends up talking to a
    stale `mana-map.js`."""
    checked = 0
    for page in PAGES:
        seen = set(_busts(page))
        assert seen, f"{page.name} busts nothing"
        assert len(seen) == 1, f"{page.name} mixes bust versions: {sorted(seen)}"
        checked += 1
    assert checked >= 5, f"only {checked} pages found"


def test_the_pages_agree_with_each_other():
    """They share `api.js`, `session.js`, `shell.js` and `tokens.css`, so a page
    left behind serves a stale copy of a file another page just updated."""
    versions = {page.name: set(_busts(page)).pop() for page in PAGES}
    assert len(set(versions.values())) == 1, (
        "pages disagree about the asset version — bump them together: "
        + ", ".join(f"{k}={v}" for k, v in sorted(versions.items())))


def test_the_deck_manifest_has_exactly_one_cache_bust_owner():
    """`api.js` declares it; nobody else may hand-write one."""
    api = (VIZ / "js" / "api.js").read_text(encoding="utf-8")
    assert re.search(r"window\.MANIFEST_VERSION\s*=\s*\d+", api), (
        "api.js no longer declares MANIFEST_VERSION")

    offenders, checked = [], 0
    for js in sorted((VIZ / "js").rglob("*.js")):
        if js.name == "api.js":
            continue
        checked += 1
        text = js.read_text(encoding="utf-8")
        for match in re.finditer(r"index\.json\?v=(\d+)", text):
            offenders.append(f"{js.name}: hand-written index.json{match.group(0)[10:]}")
        if re.search(r"deckIndex\s*\+\s*'\?v='\s*\+\s*\(\(?window\.MM", text):
            offenders.append(f"{js.name}: busts the manifest with the CARD MAP's "
                             f"DATA_VERSION — different lifecycles")
    assert checked >= 8, f"only {checked} scripts scanned"
    assert not offenders, (
        "the manifest's cache bust belongs to api.js alone:\n  "
        + "\n  ".join(offenders))


def test_the_manifest_version_is_not_the_card_map_version():
    """They move for different reasons and sharing one would bust 60 MB of
    embeddings because a deck gained a key."""
    api = (VIZ / "js" / "api.js").read_text(encoding="utf-8")
    mm = (VIZ / "js" / "mana-map.js").read_text(encoding="utf-8")
    # CODE, NOT PROSE. The first draft of this asserted the string was absent
    # and fired on api.js's own comment explaining why the two are separate —
    # a check that forbids documenting the rule it enforces. Comment lines are
    # dropped before looking.
    code = "\n".join(line for line in api.splitlines()
                     if not re.match(r"\s*(//|/\*|\*)", line))
    assert "DATA_VERSION" not in code, (
        "api.js must not reach for the card map's version in code")
    assert re.search(r"const DATA_VERSION\s*=\s*\d+", mm), (
        "mana-map.js no longer declares DATA_VERSION")
