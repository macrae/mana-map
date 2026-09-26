"""card-search: deterministic card mining over the corpus.

The bench could measure a deck from nine directions and could not answer the
question every measurement ends in — WHICH CARDS would fix it. These pin the
three rules that keep the answer honest: identity is derived and not authored,
a candidate is a card you do not already have, and a truncated list says so.
"""

import re

import pytest

from manamap.pilot import card_search

from conftest import requires_data, requires_deck


class Args:
    """argparse stand-in; `analyze` reads everything with getattr defaults."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@requires_data
def test_identity_is_a_subset_filter_not_an_equality_one():
    """A GU deck may play a mono-blue or colourless card. Filtering on equality
    would return only gold Simic cards, which is a tiny and wrong slice."""
    rows, _ = card_search.search(identity={"G", "U"}, oracle=[r"^Draw a card\."], limit=200)
    idents = {tuple(r["color_identity"]) for r in rows}
    assert idents, "expected some matches"
    assert all(set(i) <= {"G", "U"} for i in idents)
    assert any(i == () for i in idents) or any(len(i) == 1 for i in idents), \
        "a subset filter must admit colourless and mono-coloured cards"


@requires_data
def test_commander_illegal_cards_are_dropped_and_counted():
    """Legality is a filter, not a note — but the count is reported, so a caller
    can tell 'nothing matched' from 'everything that matched was illegal'."""
    rows, meta = card_search.search(oracle=[r"\bproliferate\b"], limit=500)
    assert rows and meta["matched"] == len(rows) or meta.get("truncated")
    assert meta["commander_illegal_skipped"] > 0
    assert all("Conspiracy" not in r["type_line"] for r in rows)


@requires_data
def test_truncation_is_stated_never_silent():
    """A silently cut list reads as 'that is all of them', which is the one claim
    this tool must never make by accident."""
    rows, meta = card_search.search(oracle=[r"\bcreature\b"], limit=5)
    assert len(rows) == 5 and meta["returned"] == 5
    assert meta["truncated"] == meta["matched"] - 5 > 0


@requires_data
def test_any_versus_all_across_several_oracle_patterns():
    """ANY is the default because several phrasings of one question are
    alternative answers, not a conjunction."""
    pats = [r"additional combat phase", r"\bproliferate\b"]
    any_rows, _ = card_search.search(oracle=pats, limit=500)
    all_rows, _ = card_search.search(oracle=pats, require_all=True, limit=500)
    assert len(all_rows) < len(any_rows)
    assert {r["name"] for r in all_rows} <= {r["name"] for r in any_rows}
    for r in all_rows:
        assert len(r["matched"]) == 2


@requires_data
def test_unranked_cards_sort_last_but_are_not_dropped():
    """The ordering, and the premise MEASURED rather than assumed.

    The docstring used to say "a card with no EDHREC rank is usually just new".
    Measured on this corpus that is false: **2,259 names are unranked and 19 of
    them are commander-legal (0.8%)** — unranked overwhelmingly means acorn,
    Alchemy or otherwise not legal, which `search` correctly drops and counts in
    `meta["commander_illegal_skipped"]`.

    So the testable property is the ORDERING (unranked sorts last, never
    dropped from a legal result set), and it needs a probe that actually has a
    legal unranked member. Basic lands are the reliable one.

    The loop also had no guard: an empty result set exercised nothing, and
    `seen_unranked` was assigned and then never asserted — so the half of the
    title that says "are not dropped" was checked nowhere.
    """
    rows, _ = card_search.search(oracle=[r"\bproliferate\b"], limit=500)
    ranks = [r["edhrec_rank"] for r in rows]
    assert len(ranks) > 20, f"only {len(ranks)} rows — this proves nothing"
    seen_unranked = False
    for r in ranks:
        if r is None:
            seen_unranked = True
        else:
            assert not seen_unranked, "a ranked card sorted after an unranked one"

    # NOT DROPPED — the other half of the title, on a probe that has a legal
    # unranked member. `Wastes` and the snow basics carry no EDHREC rank.
    lands, _ = card_search.search(names=[r"^(Snow-Covered )?Wastes$"], limit=50)
    assert lands, "the basics probe matched nothing"
    assert any(r["edhrec_rank"] is None for r in lands), (
        "a legal card with no EDHREC rank was dropped from the result set")


@requires_data
@requires_deck
def test_deck_scoping_derives_identity_and_excludes_what_you_already_have():
    doc = card_search.analyze(Args(deck="zur-enchantress", oracle=[r"\benchantment\b"],
                                   limit=200))
    assert doc["identity"] == ["B", "U", "W"], "derived from the commander, not authored"
    assert doc["identity_derived_from"] == "zur-enchantress"
    # TIED TO THE DECK, NOT TO A NUMBER. This asserted `== 93` and then `== 95`,
    # and broke on both of the swaps that moved zur-enchantress — a hard-coded
    # count turns a test of "deck scoping excludes what you hold" into a change
    # detector for one deck's list. What is actually being claimed is that
    # `card_search` and the shared face helper agree, and that a DFC is excluded
    # under EVERY name it has, so the exclusion set is strictly larger than the
    # decklist. Both survive a swap; neither survives the exclusion breaking.
    from manamap.pilot.common import expand_faces, load_deck_cards

    entries = [c["name"] for c in load_deck_cards("zur-enchantress")["cards"]]
    faces = set()
    for name in entries:
        faces |= expand_faces(name)
    assert doc["excluded_deck_cards"] == len(faces), (
        "card_search and expand_faces disagree about what the deck holds")
    dfcs = [n for n in entries if " // " in n]
    assert dfcs, "the fixture deck must hold at least one DFC for the next claim to bite"
    assert doc["excluded_deck_cards"] > len(entries), (
        "a DFC is excluded under every name it has, so faces outnumber entries")
    names = {r["name"] for r in doc["results"]}
    assert "Rhystic Study" not in names, "the deck already runs it — not a candidate"
    assert "Mystic Remora" not in names


@requires_data
@requires_deck
def test_include_owned_turns_the_exclusion_off():
    """Searched by NAME, not oracle: `--oracle "Hardened Scales"` matches cards whose
    RULES TEXT says that, which is nothing — the trap `--name` exists to avoid."""
    q = dict(deck="zur-enchantress", name=[r"^(Rhystic Study|Mystic Remora)$"],
             limit=50)
    assert card_search.analyze(Args(**q))["results"] == [], "both are in the deck"
    owned = card_search.analyze(Args(include_owned=True, **q))
    assert {r["name"] for r in owned["results"]} == {"Rhystic Study", "Mystic Remora"}


@requires_data
@requires_deck
def test_identity_may_not_override_a_deck_because_it_is_derived():
    """`build_deck.load_brief` derives identity from the commander and refuses an
    authored one; two derivations of one fact are two chances to disagree."""
    with pytest.raises(SystemExit, match="DERIVED"):
        card_search.analyze(Args(deck="zur-enchantress", identity="WUBRG"))


@requires_data
def test_game_changers_can_be_excluded_because_they_force_bracket_4():
    with_gc, _ = card_search.search(identity={"U"}, oracle=[r"\bdraw\b"], limit=500)
    without, _ = card_search.search(identity={"U"}, oracle=[r"\bdraw\b"], limit=500,
                                    allow_game_changers=False)
    assert any(r["game_changer"] for r in with_gc)
    assert not any(r["game_changer"] for r in without)


@requires_data
def test_the_search_that_motivated_this_command_still_answers():
    """Simic has exactly two extra-combat effects and it is why kianne's win
    condition is under-built — one creature, one attack, one opponent per turn.
    If this ever returns nothing the corpus or the filter has broken, and the
    finding it supports would silently stop being reproducible."""
    rows, _ = card_search.search(identity={"G", "U"}, oracle=[r"additional combat phase"],
                                 limit=50)
    assert {"Genji Glove", "Illusionist's Gambit"} <= {r["name"] for r in rows}
    glove = next(r for r in rows if r["name"] == "Genji Glove")
    assert re.search(r"double strike", glove["oracle_text"], re.I)


def test_the_compact_identity_form_is_parsed_as_LETTERS_not_one_token():
    """`--identity GU` silently returned only COLOURLESS cards.

    `analysis.common.parse_color_identity` splits on commas, because that is how
    cards.csv stores the column ("G, U"). Handed the compact form a human types it
    returned `{"GU"}` — one two-character token — and `{"U"} <= {"GU"}` is False for
    every coloured card. The header still read "identity GU", so the search reported
    a filter it was not applying. Same shape as the bug `card_pool._build_pool`
    records, and the registry help advertises exactly this spelling.
    """
    assert card_search.parse_identity_arg("GU") == {"G", "U"}
    assert card_search.parse_identity_arg("gu") == {"G", "U"}
    assert card_search.parse_identity_arg("G, U") == {"G", "U"}
    assert card_search.parse_identity_arg("G,U") == {"G", "U"}
    assert card_search.parse_identity_arg("WUBRG") == {"W", "U", "B", "R", "G"}
    # Colourless as an IDENTITY is the empty set, not a sixth colour.
    assert card_search.parse_identity_arg("C") == set()
    with pytest.raises(SystemExit, match="not a colour"):
        card_search.parse_identity_arg("GX")


@requires_data
def test_the_regression_the_identity_bug_actually_caused():
    """Simic has exactly two 'additional combat phase' cards. Before the fix this
    returned one — the colourless Equipment — and dropped the mono-blue instant."""
    rows, _ = card_search.search(identity=card_search.parse_identity_arg("GU"),
                                 oracle=[r"additional combat phase"], limit=50)
    assert {"Genji Glove", "Illusionist's Gambit"} <= {r["name"] for r in rows}


@requires_data
def test_owned_and_unowned_partition_the_same_search():
    """`pool-facts` knew the box but could not filter by oracle text; `card-search`
    filtered by oracle text but could not see the box. Every 'what could I add that I
    already have' question needed both. The two filters must be exact complements —
    if they are not, one of them is quietly dropping cards."""
    q = dict(identity={"G", "U"}, oracle=[r"\bproliferate\b"], limit=500)
    everything = {r["name"] for r in card_search.search(**q)[0]}
    owned = {r["name"] for r in card_search.search(owned=True, **q)[0]}
    unowned = {r["name"] for r in card_search.search(owned=False, **q)[0]}
    assert owned | unowned == everything
    assert not (owned & unowned)
    for r in card_search.search(owned=True, **q)[0]:
        assert r["owned"] is True
    # Absent the filter the field is None, not False — "not asked" is not "no".
    assert all(r["owned"] is None for r in card_search.search(**q)[0])


# ── the goldfish channels a card feeds ──
#
# THIS EXISTS BECAUSE OF A MEASURED FAILURE. Mining goblin-storm for cards that
# would convert card advantage into damage returned Guttersnipe, Firebrand Archer
# and Kessig Flamebreather; all three were staged, and all three read as VANILLA
# BODIES, because the model had no spell count and "whenever you cast" fired on
# nothing. The branch measured WORSE and the decline was an artifact of the
# instrument, not a verdict on the cards.


def test_channels_for_names_what_the_model_can_price():
    from manamap.pilot.card_search import channels_for
    fodder = channels_for({"name": "Expedite", "type_line": "Instant",
                           "oracle_text": "Target creature gains haste until end "
                                          "of turn. Draw a card."})
    assert "fodder" in fodder and "draw" in fodder
    storm = channels_for({"name": "Grapeshot", "type_line": "Instant",
                          "oracle_text": "Grapeshot deals 1 damage to any target. "
                                         "Storm (When you cast this spell, copy it "
                                         "for each spell cast before it this turn.)"})
    assert "storm" in storm
    pump = channels_for({"name": "Brute Force", "type_line": "Instant",
                         "oracle_text": "Target creature gets +3/+3 until end of turn."})
    assert "pump" in pump and "fodder" in pump
    # THE IMPORTANT CASE: a card the model reads as a body and a mana cost.
    # An empty list is the answer to "why did this measure as nothing".
    assert channels_for({"name": "Purphoros, God of the Forge",
                         "type_line": "Legendary Enchantment Creature — God",
                         "oracle_text": "Indestructible. Whenever another creature "
                                        "you control enters, Purphoros deals 2 damage "
                                        "to each opponent."}) == []


def test_per_cast_damage_and_magecraft_are_distinct_channels():
    """The rules distinction, surfaced in the search. "Whenever you cast" does not
    fire on a copy; "cast OR COPY" does — so a Zada deck should be able to tell
    them apart before it buys either."""
    from manamap.pilot.card_search import channels_for
    g = channels_for({"name": "Guttersnipe", "type_line": "Creature — Goblin Shaman",
                      "oracle_text": "Whenever you cast an instant or sorcery spell, "
                                     "this creature deals 2 damage to each opponent."})
    assert g == ["per-cast-damage"]
    s = channels_for({"name": "Storm-Kiln Artist", "type_line": "Creature — Dwarf Shaman",
                      "oracle_text": "Magecraft — Whenever you cast or copy an instant "
                                     "or sorcery spell, create a Treasure token."})
    assert s == ["magecraft"]


@requires_data
def test_the_channel_filter_and_the_unmodelled_filter_are_complements():
    from manamap.pilot.card_search import search
    fodder, _ = search(identity=set("R"), channels=["fodder"], limit=40)
    assert len(fodder) >= 20, f"only {len(fodder)} mono-red fodder cards"
    assert all("fodder" in r["channels"] for r in fodder)
    blind, _ = search(identity=set("R"), unmodelled=True, limit=40)
    assert blind, "no unmodelled cards at all — the filter is inverted"
    assert all(r["channels"] == [] for r in blind)
    # And every row carries the annotation whether or not it was filtered on.
    plain, _ = search(identity=set("R"), cmc_max=1, limit=10)
    assert all("channels" in r for r in plain)
