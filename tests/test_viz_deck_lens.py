"""Build — the map's deck mode, and its three load-bearing assumptions.

`viz/js/build.js` overlays a loaded deck's 99 onto the card map. It was Deck Lens until
Deck Lens and Build Deck merged: they were halves of one activity, so a published deck
could be inspected but not edited, and a deck under construction had no roles, no curve and
no verified lines. The lens half is what this file guards. It does no
computation beyond a name -> row-index lookup and a role histogram, which is the point:
every figure it shows comes from the same tracked artifacts the deck page and the dossier
read. But that cheapness rests on three assumptions the browser cannot check for itself,
because a lookup miss there is silent — a card simply fails to light up.

1. Every deck card name resolves in `projection_2d.json`. Today all seven decks match
   exactly. A Scryfall rename or a hand-edited decklist would break that quietly.
2. Every role family in `card_roles.json` has a colour in the lens. A new family added to
   `ROLE_PATTERNS` would otherwise paint silently grey and vanish into `unclassified`.
3. `viz/index.html` loads the script, and its cache-bust matches the other viz scripts —
   a stale `?v=` is how a shipped fix fails to reach anyone.
"""

import json
import re

import pytest
from conftest import requires_data, requires_deck, requires_roles

from manamap.config import CARD_ROLES_PATH, DATA_DIR, DECKS_DIR, VIZ_DIR

LENS_JS = VIZ_DIR / "js" / "build.js"
INDEX_HTML = VIZ_DIR / "index.html"
PROJECTION = DATA_DIR / "projection_2d.json"


def _lens_source() -> str:
    return LENS_JS.read_text(encoding="utf-8")


def _map_source() -> str:
    return (VIZ_DIR / "js" / "mana-map.js").read_text(encoding="utf-8")


def _lens_families() -> set[str]:
    """The families the role palette paints.

    Parsed from mana-map.js, not build.js: the role taxonomy moved into the grouping
    registry (`MM.GROUPINGS.role`) so that a role is the same colour on the atlas, in the
    role-budget bars and in the segmented mana curve. Build now reads it rather than
    keeping a second table — which is exactly what these tests exist to protect, since two
    tables are how a family ends up coloured in one place and grey in another.
    """
    src = _map_source()
    block = re.search(r"const ROLE_PALETTE = \{(.*?)\n  \};", src, re.DOTALL)
    assert block, "ROLE_PALETTE block not found in mana-map.js"
    return set(re.findall(r"'?([a-z-]+)'?\s*:", block.group(1)))


def test_lens_exposes_the_overlay_contract():
    """mana-map.js calls exactly these two methods on whichever mode owns the panel."""
    src = _lens_source()
    for member in ("getOverlayTraces", "getDimmedIndices", "enter", "exit"):
        assert f"    {member},\n" in src or f"  {member}," in src, f"Build.{member} not exported"


def test_index_html_loads_the_lens_at_the_shared_cache_bust():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'value="build"' in html, "Build missing from the mode selector"

    busts = dict(re.findall(r'src="js/([\w-]+)\.js\?v=(\d+)"', html))
    assert "build" in busts, "build.js not loaded by viz/index.html"
    # One bump moves all three or the browser serves a mismatched pair.
    assert len(set(busts.values())) == 1, f"viz script cache-busts disagree: {busts}"


@requires_roles
def test_every_role_family_has_a_colour_in_the_lens():
    roles = json.loads(CARD_ROLES_PATH.read_text(encoding="utf-8"))["roles"]
    families = {r.split(":")[0] for card_roles in roles.values() for r in card_roles}
    missing = families - _lens_families()
    assert not missing, (
        f"role families with no colour in MM.GROUPINGS.role: {sorted(missing)} — "
        "they would render grey and be indistinguishable from unclassified"
    )


@requires_roles
def test_family_priority_covers_every_coloured_family():
    """A family with a colour but no priority entry can never win a tie."""
    src = _map_source()
    block = re.search(r"const ROLE_ORDER = \[(.*?)\];", src, re.DOTALL)
    assert block, "ROLE_ORDER block not found in mana-map.js"
    order = set(re.findall(r"'([a-z-]+)'", block.group(1)))
    # 'unclassified' is in the order (it is a real bucket) but is the fallback family, so
    # Build filters it out of its priority list rather than letting it win a tie.
    assert _lens_families() == order
    assert "unclassified" in order
    assert "familyPriority()" in _lens_source(), (
        "build.js must derive its priority from the registry, not keep its own list"
    )


@requires_data
@requires_deck
def test_every_deck_card_resolves_on_the_map():
    """The lens matches by name with no fuzzy fallback; a miss is an unlit card."""
    projection = json.loads(PROJECTION.read_text(encoding="utf-8"))
    known = {row["n"] for row in projection}

    manifest_path = DECKS_DIR / "index.json"
    if not manifest_path.exists():
        pytest.skip("no deck manifest (run `manamap pilot build-index`)")
    slugs = [d["slug"] for d in json.loads(manifest_path.read_text(encoding="utf-8"))["decks"]]

    unmatched: dict[str, list[str]] = {}
    for slug in slugs:
        cards_path = DECKS_DIR / slug / "cards.json"
        if not cards_path.exists():
            continue
        names = [c["name"] for c in json.loads(cards_path.read_text(encoding="utf-8"))["cards"]]
        missing = sorted(n for n in names if n not in known)
        if missing:
            unmatched[slug] = missing

    assert not unmatched, f"deck cards absent from projection_2d.json: {unmatched}"


@requires_data
@requires_deck
def test_every_deck_names_a_commander_the_map_knows():
    """The lens draws the commander as its one large star; a miss loses the anchor."""
    projection = json.loads(PROJECTION.read_text(encoding="utf-8"))
    known = {row["n"] for row in projection}

    manifest_path = DECKS_DIR / "index.json"
    if not manifest_path.exists():
        pytest.skip("no deck manifest (run `manamap pilot build-index`)")

    for deck in json.loads(manifest_path.read_text(encoding="utf-8"))["decks"]:
        assert deck["commander"] in known, (
            f"{deck['slug']}: commander {deck['commander']!r} is not on the map"
        )


def test_build_picks_the_cheap_dimming_path_when_it_can():
    """Per-point opacity means a 34,000-entry array per colour group, every render.

    Two dimming strategies, one module, and the choice matters. Dimming EVERYTHING and
    redrawing the 99 on top is one scalar and ~free — that is the normal case, and it was
    the Lens's. Dimming a GENUINE SUBSET — what you may not legally play, by format or by
    the commander's colour identity — has nothing drawn over it, so it needs the array;
    that was the builder's, and it is opt-in here because it is the expensive one.

    `dimsAll()` is how the mode declares which it is, and it must yield to the per-point
    path rather than both being true at once.
    """
    src = _lens_source()
    assert "function dimsAll()" in src
    assert re.search(r"^\s+dimsAll,$", src, re.M), "dimsAll not exported"
    assert "!showIllegal" in src, (
        "dimsAll must stand down when the per-point legality path is on, or the scalar "
        "wins and nothing is singled out"
    )
    assert "function getDimmedIndices()" in src
    assert "isColorIdentitySubset" in src and "isLegalInFormat" in src, (
        "the per-point path is what colour identity and format legality are FOR"
    )

    map_src = (VIZ_DIR / "js" / "mana-map.js").read_text(encoding="utf-8")
    assert "overlay.dimsAll && overlay.dimsAll()" in map_src
    # The SHAPE, not the brightness: this test is about the per-point path existing, and
    # pinning the lit alpha made a visual-tuning change look like the path had been deleted.
    assert re.search(r"dimmedIndices\.has\(idx\) \? 0\.08 : 0\.\d+", map_src), (
        "the per-point path must survive — it is the whole point of getDimmedIndices"
    )


def test_deck_html_busts_its_own_assets():
    """`deck.html` had NO test on its two `?v=` bumps — grep for `deck-view.js`
    across `tests/` returned zero before this.

    `index.html` has had one since the day a mismatched pair let `build.js` call a
    stale `mana-map.js`. The dossier is the page that just grew from nine artifact
    fetches to fifteen, including `info.json`, which is composed from every other
    artifact — the surface where a stale script serves a confidently wrong deck is
    now larger here than there.
    """
    import re
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "viz" / "deck.html").read_text()
    busts = dict(re.findall(r'(?:src|href)="(?:js|css)/([\w-]+)\.(?:js|css)\?v=(\d+)"', html))
    # `shell` joined when the navigation strip became shared across all three
    # surfaces. `session` joined when the library drawer did: the strip can COUNT
    # the library out of localStorage, but taking a card back out is a write, and
    # only Session may write. Both are dependency-free and carry no data.
    # `api` joined when the dossier gained the Measure buttons: a local server
    # can RUN the deterministic measurements, and a static host cannot — the
    # page has to know which one it is being.
    assert set(busts) == {"deck-view", "tokens", "shell", "session", "api"}, busts
    assert all(int(v) > 0 for v in busts.values())


def test_deck_html_loads_the_dossier_script():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "viz" / "deck.html").read_text()
    assert 'src="js/deck-view.js?v=' in html
    assert 'href="css/tokens.css?v=' in html


# ── The dossier's nine sections ──────────────────────────────────────────


def _dossier_array():
    """The `DOSSIER` literal out of `deck-view.js`, parsed as data."""
    import ast
    import re
    from pathlib import Path

    js = (Path(__file__).parent.parent / "viz/js/deck-view.js").read_text()
    m = re.search(r"var DOSSIER = (\[.*?\n  \]);", js, re.S)
    assert m, "the DOSSIER literal moved or changed shape"
    # The literal is deliberately plain data — arrays of strings, no
    # expressions, no concatenation — which is what makes it parseable at all.
    # `ast.literal_eval` reads JS single-quoted strings as Python ones.
    raw = re.sub(r"//[^\n]*", "", m.group(1))
    return [list(row) for row in ast.literal_eval(raw)]


def test_the_dossier_sections_match_the_python_registry():
    """PYTHON OWNS THE ORDER, and the browser transcribes it.

    The section order used to be an anonymous array literal inside `render()`
    with no statement of it anywhere — so "what sections does the deck page
    have, and why that order" was answerable only by reading the assembly line.
    It now lives in `page_spec.DOSSIER_SECTIONS` beside the manual's own
    registry, and this is what stops the transcription drifting: the same
    contract `decklist.js` lives under against the Python parser.
    """
    from manamap.pilot.page_spec import DOSSIER_SECTIONS

    js = _dossier_array()
    assert len(js) == len(DOSSIER_SECTIONS), (
        f"the browser has {len(js)} sections, Python has "
        f"{len(DOSSIER_SECTIONS)}")
    for got, want in zip(js, DOSSIER_SECTIONS):
        assert got[0] == want[0], f"order differs: {got[0]} vs {want[0]}"
        assert got[1] == want[1], f"{got[0]}: title differs"
        assert got[2] == want[2], f"{got[0]}: promise differs"
        assert list(got[3]) == list(want[3]), f"{got[0]}: tiers differ"


def test_the_cover_sheet_is_first_and_the_assessment_is_last():
    """The order is the argument, so it is asserted rather than assumed.

    A cover sheet you absorb in thirty seconds is the only thing that makes the
    depth behind it usable, and the assessment is last BECAUSE it is the
    analyst's opinion — a file where that sits inside the record loses trust,
    which is what the old page did by rendering the diagnosis verdict as one
    inline sentence in the middle of the audit panel.
    """
    from manamap.pilot.page_spec import DOSSIER_IDS

    assert DOSSIER_IDS[0] == "cover"
    assert DOSSIER_IDS[-1] == "assessment"
    assert DOSSIER_IDS.index("priors") < DOSSIER_IDS.index("logs"), (
        "the coded table reads before the prose it summarises")


def test_every_dossier_section_grants_a_tier_it_can_actually_mint():
    """`tiers` is what a section GRANTS, not what it mentions — `page_spec`'s
    own rule. A section that grants `verified` may mint a check-mark claim; one
    that does not may still discuss a verified line."""
    from manamap.pilot.page_spec import DOSSIER_SECTIONS

    known = {"verified", "data", "coach"}
    for sid, _title, promise, tiers in DOSSIER_SECTIONS:
        assert tiers, f"{sid} grants nothing — every section declares a tier"
        assert set(tiers) <= known, f"{sid}: {tiers}"
        assert len(promise) > 25, f"{sid} has no usable promise"
    grants = {s[0]: set(s[3]) for s in DOSSIER_SECTIONS}
    # The captain's log is the pilot speaking. It cannot mint a measurement.
    assert grants["logs"] == {"coach"}
    assert grants["assessment"] == {"coach"}
    # The cover sheet's three numbers are all seeded measurements.
    assert grants["cover"] == {"data"}
