"""The Deck Lens — the map's third mode, and its three load-bearing assumptions.

`viz/js/deck-map.js` overlays a published deck's 99 onto the card map. It does no
computation beyond a name -> row-index lookup and a role histogram, which is the point:
every figure it shows comes from the same tracked artifacts the magazine and the dossier
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

LENS_JS = VIZ_DIR / "js" / "deck-map.js"
INDEX_HTML = VIZ_DIR / "index.html"
PROJECTION = DATA_DIR / "projection_2d.json"


def _lens_source() -> str:
    return LENS_JS.read_text(encoding="utf-8")


def _lens_families() -> set[str]:
    """The families FAMILY_COLOR paints, parsed from the source."""
    src = _lens_source()
    block = re.search(r"const FAMILY_COLOR = \{(.*?)\n  \};", src, re.DOTALL)
    assert block, "FAMILY_COLOR block not found in deck-map.js"
    return set(re.findall(r"'?([a-z-]+)'?\s*:", block.group(1)))


def test_lens_exposes_the_overlay_contract():
    """mana-map.js calls exactly these two methods on whichever mode owns the panel."""
    src = _lens_source()
    for member in ("getOverlayTraces", "getDimmedIndices", "enter", "exit"):
        assert f"    {member},\n" in src or f"  {member}," in src, f"DeckMap.{member} not exported"


def test_index_html_loads_the_lens_at_the_shared_cache_bust():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'value="deck"' in html, "Deck Lens missing from the mode selector"

    busts = dict(re.findall(r'src="js/([\w-]+)\.js\?v=(\d+)"', html))
    assert "deck-map" in busts, "deck-map.js not loaded by viz/index.html"
    # One bump moves all three or the browser serves a mismatched pair.
    assert len(set(busts.values())) == 1, f"viz script cache-busts disagree: {busts}"


@requires_roles
def test_every_role_family_has_a_colour_in_the_lens():
    roles = json.loads(CARD_ROLES_PATH.read_text(encoding="utf-8"))["roles"]
    families = {r.split(":")[0] for card_roles in roles.values() for r in card_roles}
    missing = families - _lens_families()
    assert not missing, (
        f"role families with no colour in deck-map.js: {sorted(missing)} — "
        "they would render grey and be indistinguishable from unclassified"
    )


@requires_roles
def test_family_priority_covers_every_coloured_family():
    """A family with a colour but no priority entry can never win a tie."""
    src = _lens_source()
    block = re.search(r"const FAMILY_PRIORITY = \[(.*?)\];", src, re.DOTALL)
    assert block, "FAMILY_PRIORITY block not found"
    priority = set(re.findall(r"'([a-z-]+)'", block.group(1)))
    # 'unclassified' is the fallback, never a priority entry.
    assert _lens_families() - {"unclassified"} == priority


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


def test_lens_dims_with_a_scalar_not_a_per_point_array():
    """Per-point opacity means a 34,000-entry array per colour group, every render.

    The Lens dims *everything* and redraws its 99 as overlay traces on top, so one scalar
    is equivalent. The deck builder dims a genuine subset (format-illegal cards, colour
    identity violations) with nothing drawn over them, so it still needs the array —
    `dimsAll()` is how a mode declares which it is.
    """
    src = _lens_source()
    assert "function dimsAll()" in src
    assert re.search(r"^\s+dimsAll,$", src, re.M), "dimsAll not exported"
    assert "function getDimmedIndices() { return null; }" in src, (
        "the Lens should no longer build a 34K index Set"
    )

    map_src = (VIZ_DIR / "js" / "mana-map.js").read_text(encoding="utf-8")
    assert "overlay.dimsAll && overlay.dimsAll()" in map_src
    # The builder's per-point path must survive.
    assert "dimmedIndices.has(idx) ? 0.08 : 0.7" in map_src, (
        "the deck builder still needs per-point dimming"
    )
    builder = (VIZ_DIR / "js" / "deck-builder.js").read_text(encoding="utf-8")
    assert "dimsAll" not in builder, "the builder must NOT declare dimsAll"
