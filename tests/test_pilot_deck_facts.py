"""Tests for deck-facts — the deterministic brief agents read instead of re-deriving.

The traps this module reports are the ones that actually cost tokens on real decks,
so the tests assert on the traps, not just on the plumbing.
"""

import json


from conftest import requires_deck
from manamap.pilot import deck_facts as df


def card(name, **overrides):
    base = {
        "name": name, "quantity": 1, "is_commander": False,
        "mana_cost": "{1}{G}", "cmc": 2.0, "type_line": "Creature — Elf",
        "oracle_text": "", "colors": ["G"], "color_identity": ["G"], "layout": "normal",
    }
    base.update(overrides)
    return base


# ── Mana restriction classification ──────────────────────────────────────
#
# Three cards carry "spend this mana only" and mean three different things. Getting
# this wrong is worse than saying nothing: the strategic frame for sisay asserted
# Secluded Courtyard could not pay an activated ability, and it can.

UNCLAIMED_TERRITORY = card(
    "Unclaimed Territory", type_line="Land", mana_cost="", cmc=0.0,
    oracle_text=("As this land enters, choose a creature type.\n"
                 "{T}: Add {C}.\n"
                 "{T}: Add one mana of any color. Spend this mana only to cast a "
                 "creature spell of the chosen type."),
)

SECLUDED_COURTYARD = card(
    "Secluded Courtyard", type_line="Land", mana_cost="", cmc=0.0,
    oracle_text=("As this land enters, choose a creature type.\n"
                 "{T}: Add {C}.\n"
                 "{T}: Add one mana of any color. Spend this mana only to cast a "
                 "creature spell of the chosen type or activate an ability of a "
                 "creature source of the chosen type."),
)

PLAZA_OF_HEROES = card(
    "Plaza of Heroes", type_line="Land", mana_cost="", cmc=0.0,
    oracle_text=("{T}: Add {C}.\n"
                 "{T}: Add one mana of any color. Spend this mana only to cast a "
                 "legendary spell.\n"
                 "{T}: Add one mana of any color among legendary permanents you control."),
)

DELIGHTED_HALFLING = card(
    "Delighted Halfling", type_line="Creature — Halfling Citizen", mana_cost="{G}",
    cmc=1.0,
    oracle_text=("{T}: Add {C}.\n"
                 "{T}: Add one mana of any color. Spend this mana only to cast a "
                 "legendary spell, and that spell can't be countered."),
)


def test_spells_only_mana_cannot_pay_an_ability():
    """An activated ability is not a spell — this is the whole trap."""
    assert df.classify_mana_restriction(UNCLAIMED_TERRITORY)["verdict"] == "spells_only"
    assert df.classify_mana_restriction(DELIGHTED_HALFLING)["verdict"] == "spells_only"


def test_an_activate_an_ability_clause_lifts_the_restriction():
    """Secluded Courtyard differs from Unclaimed Territory by exactly this clause."""
    assert df.classify_mana_restriction(SECLUDED_COURTYARD)["verdict"] == "pays_abilities"


def test_an_unrestricted_mode_only_counts_if_it_makes_colour():
    """Every one of these has '{T}: Add {C}'. Colourless pays no coloured pip.

    Plaza has a genuinely unrestricted *coloured* mode; the other two do not, which
    is why they classify differently despite the same escape-hatch shape.
    """
    assert df.classify_mana_restriction(PLAZA_OF_HEROES)["verdict"] == \
        "has_unrestricted_coloured_mode"
    for colourless_hatch in (UNCLAIMED_TERRITORY, DELIGHTED_HALFLING):
        assert df.classify_mana_restriction(colourless_hatch)["verdict"] == "spells_only"


def test_unrestricted_cards_classify_as_nothing():
    assert df.classify_mana_restriction(card("Llanowar Elves", oracle_text="{T}: Add {G}.")) is None


# ── Multi-face colours ───────────────────────────────────────────────────

ESIKA = card(
    "Esika, God of the Tree // The Prismatic Bridge",
    layout="modal_dfc", mana_cost="", cmc=3.0, colors=[],
    type_line="Legendary Creature — God // Legendary Enchantment",
    color_identity=["W", "U", "B", "R", "G"],
    card_faces=[
        {"name": "Esika, God of the Tree", "mana_cost": "{1}{G}{G}", "colors": ["G"],
         "type_line": "Legendary Creature — God", "oracle_text": "front"},
        {"name": "The Prismatic Bridge", "mana_cost": "{W}{U}{B}{R}{G}",
         "colors": ["W", "U", "B", "R", "G"],
         "type_line": "Legendary Enchantment", "oracle_text": "back"},
    ],
)


def test_multiface_reports_both_the_card_and_the_face_up_colours():
    """A permanent contributes its FACE's colours, not the card's union.

    Sisay counts "colours among other legendary permanents"; Esika on the
    battlefield is mono-green even though the card is five colours.
    """
    got = df.colours([ESIKA])["Esika, God of the Tree // The Prismatic Bridge"]
    assert got["card"] == ["W", "U", "B", "R", "G"]
    assert got["front_face"] == ["G"]


def test_single_faced_cards_have_no_front_face_key():
    got = df.colours([card("Llanowar Elves")])["Llanowar Elves"]
    assert got["card"] == ["G"] and "front_face" not in got


# ── Counts and curve ─────────────────────────────────────────────────────

def test_counts_are_entries_not_copies(tmp_path, monkeypatch):
    """"2 Plains" is one entry and two copies. Both are reported so neither misleads."""
    from manamap.pilot import common

    deck = tmp_path / "toy"
    deck.mkdir()
    cards = [
        card("Plains", quantity=2, type_line="Basic Land — Plains", mana_cost="", cmc=0.0),
        card("Llanowar Elves"),
    ]
    (deck / "cards.json").write_text(json.dumps({"cards": cards}))
    monkeypatch.setattr(common, "DECKS_DIR", tmp_path)
    monkeypatch.setattr(df, "load_deck_cards", lambda slug: {"cards": cards})

    counts = df.analyze("toy")["counts"]
    assert counts["entries"] == 2 and counts["copies"] == 3
    assert counts["lands"] == 1 and counts["nonland"] == 1


def test_curve_skips_lands():
    cards = [card("A", cmc=2.0), card("B", cmc=2.0), card("C", cmc=5.0),
             card("Forest", type_line="Basic Land — Forest", cmc=0.0)]
    assert df.curve(cards) == {"2": 2, "5": 1}


def test_legendary_detection_reads_the_front_face():
    assert df._is_legendary(ESIKA)
    assert not df._is_legendary(card("Llanowar Elves"))


# ── The notes block ──────────────────────────────────────────────────────

def test_notes_name_the_spells_only_cards_explicitly():
    facts = {
        "colours": {}, "combos": {"available": False},
        "roles": {"available": True, "no_role": []},
        "synergy": {"available": False},
        "mana": {"restricted_mana": [
            {"name": "Unclaimed Territory", "verdict": "spells_only",
             "unrestricted_coloured_modes": 0, "restricted_text": "..."},
        ]},
    }
    notes = " ".join(df.build_notes(facts))
    assert "Unclaimed Territory" in notes
    assert "not a spell" in notes


def test_notes_report_the_synergy_edge_count_rather_than_a_verdict():
    """Sisay has 0 intra-deck edges; edgar-vampires has 213. Report, don't assert."""
    facts = {
        "colours": {}, "combos": {"available": False},
        "roles": {"available": True, "no_role": []},
        "mana": {"restricted_mana": []},
        "synergy": {"available": True, "intra_deck_edges": 0,
                    "cards_in_graph": 86, "cards_absent": []},
    }
    assert "0 edge(s)" in " ".join(df.build_notes(facts))


# ── Real decks ───────────────────────────────────────────────────────────

@requires_deck
def test_real_deck_is_deterministic():
    """Same input, same output — this is a view, so it must never drift."""
    assert df.analyze("goblin-storm") == df.analyze("goblin-storm")


@requires_deck
def test_real_deck_counts_reconcile_to_the_decklist():
    facts = df.analyze("goblin-storm")
    counts = facts["counts"]
    assert counts["copies"] == 100
    assert counts["lands"] + counts["nonland"] == counts["entries"]
    assert counts["legendary"] + counts["nonlegendary"] == counts["entries"]


@requires_deck
def test_sisay_reproduces_the_figures_derived_by_hand():
    """Vol. 003 was written off these numbers; the command must agree with it."""
    facts = df.analyze("sisay")
    counts = facts["counts"]
    assert counts["legendary_permanents"] == 34
    assert counts["legendary_lands"] == 0          # the ladder's empty bottom rung
    assert counts["nonlegendary"] - counts["lands"] == 26   # outside the tutor's reach
    # Was 0, now 2 — and the change is the synergy graph getting better, not this
    # deck changing. Partners used to be tie-broken by embedding similarity, which
    # filled the top ten with obscure lookalikes; ranking by playability instead
    # surfaces cards people actually run, and two of Sisay's own cards are now
    # reachable that way. Both are score-1 `Flying + Damage Trigger` matches —
    # Kutzil, Malamet Exemplar with Raff Capashen and with Swan Song. Weak, but real
    # rule matches rather than noise, which is why the number moved up rather than
    # being suppressed.
    assert facts["synergy"]["intra_deck_edges"] == 2
    # Was six. Dispel, Fierce Guardianship, Negate and Swan Song all say
    # "counter target NONCREATURE spell" and the `counterspell` pattern once
    # demanded the literal "counter target spell", so half of Sisay's
    # interaction read as blank. What remains is genuinely unnamed: Silence is
    # a one-turn Time Stop and Teferi is a static rule change.
    assert facts["roles"]["no_role"] == ["Silence", "Teferi, Time Raveler"]
    verdicts = {r["name"]: r["verdict"] for r in facts["mana"]["restricted_mana"]}
    assert verdicts["Delighted Halfling"] == "spells_only"
    assert verdicts["Unclaimed Territory"] == "spells_only"
    assert verdicts["Secluded Courtyard"] == "pays_abilities"


def test_mana_facts_counts_copies_not_entries():
    """A mana base is about physical cards: 11 Islands are 11 blue sources.

    The `counts` block stays per-entry by convention (and reports both); the
    mana block must not, or every source figure on the deck page is halved.
    """
    cards = [
        {"name": "Island", "type_line": "Basic Land — Island", "quantity": 11,
         "oracle_text": "({T}: Add {U}.)", "color_identity": []},
        {"name": "Opt", "type_line": "Instant", "quantity": 1,
         "mana_cost": "{U}", "cmc": 1.0, "oracle_text": "Scry 1."},
    ]
    mana = df.mana_facts(cards)
    assert mana["lands"] == 11          # copies
    assert mana["land_entries"] == 1    # distinct cards
    assert mana["land_sources"]["U"] == 11
