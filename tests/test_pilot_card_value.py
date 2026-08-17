"""`card-value`: what each card is worth, and the two ways that measurement lies.

The command answers a question nothing else in the repo does — not "is this deck
good" but "what is THIS CARD carrying" — by replacing one card with a blank and
re-running the goldfish. On ur-dragon it ranked The Misty Mountains Cold, Smaug
and Scourge of the Throne as the three load-bearing cards, each of which had been
proposed for the cut list by someone reasoning from card text.

The tests below exist because a naive version of this is worse than not having it:

* **Replace, never remove.** A 99-card deck draws its remaining cards more often,
  so every removal reads as an improvement — measured, dropping a card the
  simulator never casts still "gained" ~2 points. Holding the deck at 100 removes
  the confound instead of correcting for it, and a card that does nothing then
  scores ~0 on its own.
* **Invisible is not the same as worthless.** With no opponents, removal,
  counterspells and protection score zero BY CONSTRUCTION — the bottom of the raw
  ranking was Swords to Plowshares, Counterspell, Teferi's Protection and
  `Blasphemous Act`, which is half of ur-dragon's only verified kill. Those cards
  are excluded from the ranking rather than ranked last, so the report cannot be
  read as a cut list.
"""

import json

import pytest

from manamap.pilot import card_value, goldfish


def _spell(name, oracle="", cmc=3, type_line="Creature — Dragon", power="4"):
    return {"name": name, "type_line": type_line, "cmc": cmc,
            "oracle_text": oracle, "quantity": 1, "is_commander": False,
            "power": power, "toughness": "4"}


# ── The visibility predicate ──────────────────────────────────────────────

def test_a_land_is_visible():
    """`classify` zeroes `produces` for lands, so a predicate built from the
    spell fields alone files every land in the deck as invisible — which is the
    exact bug this test was written after finding."""
    land = goldfish.classify(_spell("Mountain", type_line="Basic Land — Mountain",
                                    cmc=0, power=None))
    assert card_value._is_visible(land)


def test_a_creature_is_visible_and_a_counterspell_is_not():
    assert card_value._is_visible(goldfish.classify(_spell("Dragon")))
    counter = goldfish.classify(_spell(
        "Counterspell", "Counter target spell.", cmc=2,
        type_line="Instant", power=None))
    assert not card_value._is_visible(counter)


def test_an_extra_combat_permanent_is_visible():
    """Aggravated Assault is neither a body, a rock nor a tutor. Without combat
    it is invisible; with it, it is the deck's win condition."""
    assault = goldfish.classify(_spell(
        "Aggravated Assault",
        "{3}{R}{R}: Untap all creatures you control. After this main phase, "
        "there is an additional combat phase followed by an additional main phase.",
        cmc=3, type_line="Enchantment", power=None))
    assert card_value._is_visible(assault)


# ── The report ────────────────────────────────────────────────────────────

def _deck():
    """A deck the model can actually play: lands, a rock, beaters, and blanks."""
    cards = [
        {"name": "Cmd", "type_line": "Legendary Creature — Dragon", "cmc": 3,
         "oracle_text": "", "quantity": 1, "is_commander": True,
         "power": "4", "toughness": "4"},
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
         "oracle_text": "", "quantity": 40, "power": None, "toughness": None},
        dict(_spell("Sol Ring", "{T}: Add {C}{C}.", cmc=1,
                    type_line="Artifact", power=None), quantity=1),
        dict(_spell("Beater", "", cmc=3), quantity=40),
        dict(_spell("Do Nothing", "Counter target spell.", cmc=2,
                    type_line="Instant", power=None), quantity=18),
    ]
    return {"cards": cards}


@pytest.fixture
def report(tmp_path, monkeypatch):
    deck = tmp_path / "decks" / "fake"
    deck.mkdir(parents=True)
    (deck / "cards.json").write_text(json.dumps(_deck()))
    (deck / "goldfish_targets.json").write_text(
        json.dumps({"targets": [], "model_treasures": True, "model_combat": True}))
    monkeypatch.setattr(card_value, "deck_dir", lambda slug: deck)
    monkeypatch.setattr(card_value, "load_deck_cards", lambda slug: _deck())
    return card_value.build("fake", iterations=150)


def test_it_refuses_without_the_combat_model(tmp_path, monkeypatch):
    """Without combat every attack trigger and extra combat lands in the
    invisible bucket, and the report is noise about a model that is not looking."""
    deck = tmp_path / "decks" / "fake"
    deck.mkdir(parents=True)
    (deck / "goldfish_targets.json").write_text(json.dumps({"targets": []}))
    monkeypatch.setattr(card_value, "deck_dir", lambda slug: deck)
    monkeypatch.setattr(card_value, "load_deck_cards", lambda slug: _deck())
    with pytest.raises(SystemExit, match="model_combat"):
        card_value.build("fake", iterations=20)


def test_an_invisible_card_is_excluded_from_the_ranking_not_ranked_last(report):
    """The whole safety property. A reader must not be able to sort by value and
    find the interaction suite waiting at the bottom."""
    assert "Do Nothing" in report["invisible_to_this_model"]
    assert "Do Nothing" not in {row["card"] for row in report["cards"]}


def test_visible_cards_are_ranked_and_the_ranking_is_sorted(report):
    values = [row["value"] for row in report["cards"]]
    assert values == sorted(values, reverse=True)
    assert {"Mountain", "Beater", "Sol Ring"} <= {r["card"] for r in report["cards"]}


def test_the_noise_floor_is_reported_and_flags_each_row(report):
    """A ranking without a resolution is a horoscope."""
    assert report["noise_floor"] >= 0
    for row in report["cards"]:
        assert row["above_noise"] == (abs(row["value"]) > report["noise_floor"])


def test_the_report_says_it_is_not_a_cut_list(report):
    """Load-bearing prose: the one thing that stops this being misused."""
    assert any("not a cut list" in note.lower() for note in report["notes"])
    assert any("replaced by a blank" in note.lower() for note in report["notes"])


def test_every_variant_keeps_the_deck_at_full_size(monkeypatch):
    """Replace-not-remove, asserted on the population the simulator actually
    sees rather than on the report. If a variant ever drops a card instead of
    swapping it, every value silently gains the thinning bonus."""
    sizes = []
    real = card_value._measure

    def spy(cards, *a, **kw):
        sizes.append(sum(c.get("quantity", 1) for c in cards))
        return real(cards, *a, **kw)

    monkeypatch.setattr(card_value, "_measure", spy)
    monkeypatch.setattr(card_value, "load_deck_cards", lambda slug: _deck())

    class _Dir:
        def __truediv__(self, other):
            class _P:
                def exists(self_inner): return True
                def __str__(self_inner): return "targets"
                def open(self_inner): raise AssertionError
            return _P()
    monkeypatch.setattr(card_value, "deck_dir", lambda slug: _Dir())
    monkeypatch.setattr(card_value.json, "load",
                        lambda f: {"targets": [], "model_combat": True})
    monkeypatch.setattr("builtins.open", lambda *a, **k: __import__("io").StringIO("{}"))
    card_value.build("fake", iterations=20)
    assert len(set(sizes)) == 1, f"deck size changed across variants: {sorted(set(sizes))}"


def test_a_land_is_blanked_by_a_land(report):
    """Swapping a land for a spell-shaped blank measures 'one fewer land', not
    'this land does nothing', and would report the mana base as worth everything."""
    mountain = next(r for r in report["cards"] if r["card"] == "Mountain")
    # 40 Mountains in a 100-card deck: losing one to an inert LAND is worth
    # near-nothing, and would be worth a great deal if the blank were a spell.
    assert abs(mountain["value"]) <= max(report["noise_floor"] * 3, 0.05)
