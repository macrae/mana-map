"""Archetypes and role templates: PRD §7.2-7.3.

The network is faked throughout. What is pinned here is the reasoning — that
styles are not ranked, that a display flag cannot change a lookup, and that the
delta exists because the raw histograms were measured and found too similar to
act on.
"""

import json

import pytest

from manamap.pilot import archetypes


@pytest.fixture
def fake(monkeypatch):
    """Two styles over one commander, sharing most of their cards."""
    from manamap.ingest import edhrec

    themes = [{"slug": "voltron", "name": "Voltron", "decks": 361},
              {"slug": "stax", "name": "Stax", "decks": 542},
              {"slug": "niche", "name": "Niche", "decks": 11}]
    shared = [("Sol Ring", 1), ("Arcane Signet", 1), ("Command Tower", 1)]
    decks = {
        None: shared + [("Ethereal Armor", 1), ("Winter Orb", 1)],
        "voltron": shared + [("Ethereal Armor", 1), ("All That Glitters", 1)],
        "stax": shared + [("Winter Orb", 1), ("Static Orb", 1)],
        "niche": shared + [("Winter Orb", 1)],
    }
    monkeypatch.setattr(edhrec, "themes", lambda c: themes)
    monkeypatch.setattr(edhrec, "average_deck",
                        lambda c, theme=None: {"slug": "c", "theme": theme,
                                               "commander": c,
                                               "cards": decks[theme]})
    monkeypatch.setattr(archetypes, "_roles", lambda: {
        "Sol Ring": ["ramp:rock"], "Arcane Signet": ["ramp:rock"],
        "Command Tower": ["land:fixing"],
        "Ethereal Armor": ["buff:attached"], "All That Glitters": ["buff:attached"],
        "Winter Orb": ["stax"], "Static Orb": ["stax"],
    })
    return themes


def test_styles_are_not_ranked(fake):
    """§7.2 is explicit: play rates may be shown as data; the platform does not
    tell the pilot which deck to want.

    EDHREC's own order is preserved — re-sorting by count would be inventing a
    ranking — and nothing in the output recommends one.
    """
    doc = archetypes.report("Zur the Enchanter")
    assert [t["slug"] for t in doc["themes"]] == ["voltron", "stax", "niche"], (
        "the theme order was changed — that is a ranking the PRD forbids")
    # Scanned over the DATA, not the whole document. The first version scanned
    # everything for "recommend" and fired on the module's own disclaimer —
    # "nothing here recommends a style" — which is a check failing on correct
    # output, the exact shape this repo keeps rejecting.
    payload = json.dumps({k: v for k, v in doc.items() if k != "note"}).lower()
    for word in ("recommend", "best", "should play", "strongest", "top pick"):
        assert word not in payload, f"the data recommends a style ({word!r})"
    assert "not a ranking" in doc["note"]
    # No per-theme flag that would amount to a ranking by another name.
    for t in doc["themes"]:
        assert set(t) == {"slug", "name", "decks"}, t


def test_a_display_limit_cannot_decide_what_exists(fake):
    """`--theme voltron --limit 1` reported that Zur has no voltron decks —
    false, and confidently phrased. `limit` truncated the list the theme was
    resolved against, so a display flag silently changed a lookup's answer."""
    doc = archetypes.report("Zur the Enchanter", theme="stax", limit=1)
    assert len(doc["themes"]) == 1, "limit must still truncate the DISPLAY"
    assert doc["template"]["theme"] == "stax", "the lookup saw the truncated list"


def test_an_unknown_theme_names_the_ones_that_exist(fake):
    with pytest.raises(SystemExit) as e:
        archetypes.report("Zur the Enchanter", theme="landfall")
    assert "voltron" in str(e.value)


def test_a_thin_style_is_reported_but_flagged(fake):
    """A role histogram over eleven decks is a description of eleven decks."""
    doc = archetypes.report("Zur the Enchanter", theme="niche")
    assert "warning" in doc and "11 decks" in doc["warning"]
    assert doc.get("template"), "it should still be computed, just qualified"


def test_the_template_counts_copies_not_entries(fake, monkeypatch):
    """Thirty basics are thirty cards. Counting entries once published '18 lands'
    for a 33-land deck."""
    from manamap.ingest import edhrec

    monkeypatch.setattr(edhrec, "average_deck",
                        lambda c, theme=None: {"slug": "c", "theme": theme,
                                               "commander": c,
                                               "cards": [("Sol Ring", 1),
                                                         ("Arcane Signet", 7)]})
    tpl = archetypes.role_template("X", "voltron")
    assert tpl["roles"]["ramp:rock"] == 8
    assert tpl["deck_size"] == 8


def test_the_delta_is_what_separates_the_styles(fake):
    """THE REASON THE DELTA EXISTS, and it was measured rather than assumed.

    Zur's raw style histograms are 0.955-0.978 cosine similar: every build runs
    the same signets, lands and removal, and that shared bulk swamps the part
    that differs. The delta against the commander's own baseline is the half
    worth acting on — measured on the real data, voltron comes back +7
    `buff:attached` / +3 `protection:self` / +2 `wincon:combat`, which is
    voltron described exactly.
    """
    volt = archetypes.report("Zur the Enchanter", theme="voltron")["distinguishing"]
    stax = archetypes.report("Zur the Enchanter", theme="stax")["distinguishing"]
    assert volt.get("buff:attached", 0) > 0, "voltron did not want more auras"
    assert stax.get("stax", 0) > 0, "stax did not want more stax pieces"
    assert volt.get("stax", 0) <= 0 and stax.get("buff:attached", 0) <= 0, (
        "the two styles came back wanting the same things — the delta is not "
        "discriminating and the whole approach needs re-arguing")


def test_the_delta_drops_roles_that_do_not_move(fake):
    """A zero in a difference list is noise; the reader is looking for what
    changed."""
    d = archetypes.report("Zur the Enchanter", theme="voltron")["distinguishing"]
    assert all(v != 0 for v in d.values())
    assert "ramp:rock" not in d, "a role both decks run identically was printed"
