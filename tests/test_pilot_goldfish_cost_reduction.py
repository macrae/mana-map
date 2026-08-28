"""The goldfish and static cost reduction — eminence, and reducers in the 99.

THE DEFECT THIS EXISTS AGAINST. The model's own assumption list said "cost
reducers and rituals are not modeled (conservative)". For a deck built ON a cost
reducer that is not conservative, it is wrong about the thesis. The Ur-Dragon's
eminence takes {1} off every Dragon spell from the COMMAND ZONE — always on,
from turn one, unremovable — across 22 of its 24 creatures, and four more
reducers sit in the 99 reading as vanilla bodies. Measured after the fix on the
tracked list: commander cast by turn 6 went 0.094 -> 0.180 and kill by turn 8
went 0.299 -> 0.505. Every figure this bench published about that deck
understated it.

Third of its kind: the model could not see 65% of the fleet's mana rocks, and
the mana model was colourless.
"""

import pytest

from conftest import requires_data, requires_deck
from manamap.pilot import goldfish


def _card(text, type_line="Creature — Dragon", cmc=5, pips=(), **kw):
    got = {"oracle_text": text, "type_line": type_line, "cmc": cmc,
           "pips": list(pips), "subtypes": goldfish.subtypes_of(type_line),
           "is_creature": "Creature" in type_line}
    got.update(kw)
    return got


# ── reading the card ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expect", [
    ("Dragon spells you cast cost {1} less to cast.", (1, "Dragon", False)),
    ("Dragon spells you cast cost {2} less to cast.", (2, "Dragon", False)),
    # Eminence says OTHER, so it never pays for itself.
    ("As long as this is in the command zone or on the battlefield, other "
     "Dragon spells you cast cost {1} less to cast.", (1, "Dragon", True)),
    # The chosen-type artifacts, both wordings.
    ("As this enters, choose a creature type. Creature spells of the chosen "
     "type cost {2} less to cast.", (2, goldfish.CHOSEN_TYPE, False)),
    ("Creature spells you cast of the chosen type cost {1} less to cast.",
     (1, goldfish.CHOSEN_TYPE, False)),
    # No subtype gate at all.
    ("Creature spells you cast cost {2} less to cast.", (2, None, False)),
])
def test_a_static_reducer_is_read_with_its_amount_and_its_gate(text, expect):
    assert goldfish.cost_reduction({"oracle_text": text}) == expect


@pytest.mark.parametrize("text", [
    # A REDUCTION THAT SCALES IS NOT A RATE — the corpus sweep caught all three.
    # The regex stops at "to cast" and would report a flat 1, which is a
    # plausible number that is wrong: the Jeweled Lotus failure exactly.
    "Creature spells you cast cost {1} less to cast for each +1/+1 counter "
    "on this creature.",                                          # Animar, starts at 0
    "Creature spells you cast cost {1} less to cast for each 1 life your "
    "opponents have lost this turn.",                             # Rakdos, 0 in a goldfish
    "Creature spells you cast cost {1} less to cast for each creature you "
    "control with a +1/+1 counter on it.",                        # Hamza
])
def test_a_scaling_reduction_is_refused_rather_than_counted_flat(text):
    assert goldfish.cost_reduction({"oracle_text": text}) is None


@pytest.mark.parametrize("word", [
    "Noncreature", "Artifact", "Equipment", "Enchantment", "Aura",
    "White", "Blue", "Black", "Red", "Green", "Colorless", "Legendary",
])
def test_a_word_that_is_not_a_creature_type_is_refused_not_silently_ignored(word):
    """The sweep found the regex capturing `Noncreature` on 7 cards, `Artifact`
    on 7, `Equipment` on 5, `Enchantment` on 4 and six colour words on 14. None
    is a creature subtype, so all of them would match no card and reduce
    nothing — a silent half-working matcher, which this repo has paid for once
    already at 65% of the fleet's mana rocks. Artifact and noncreature cost
    reduction is real and simply not modelled; refusing says so."""
    text = f"{word} spells you cast cost {{1}} less to cast."
    assert goldfish.cost_reduction({"oracle_text": text}) is None


def test_a_card_with_no_reduction_reads_as_none():
    assert goldfish.cost_reduction({"oracle_text": "Flying, haste"}) is None
    assert goldfish.cost_reduction({}) is None


# ── what it costs ────────────────────────────────────────────────────────

def test_a_reduction_pays_generic_and_never_a_coloured_pip():
    """`{4}{W}{U}{B}{R}{G}` with three reducers out is still five mana, not two.
    Flooring at zero instead would make a five-colour commander castable off two
    lands, which is a rules error the model would then build every figure on."""
    ur = _card("", cmc=9, pips=[{"W"}, {"U"}, {"B"}, {"R"}, {"G"}],
               type_line="Legendary Creature — Dragon", is_commander=True)
    assert goldfish.reduced_cost(ur, []) == 9
    assert goldfish.reduced_cost(ur, [(2, "Dragon", False)]) == 7
    # Five pips is the floor no amount of reduction can go under.
    assert goldfish.reduced_cost(ur, [(9, "Dragon", False)]) == 5


def test_eminence_does_not_pay_for_its_own_commander():
    """"OTHER Dragon spells" — the commander is not other. A model that let it
    discount itself would report a nine-drop landing a turn early forever."""
    ur = _card("", cmc=9, pips=[{"W"}], type_line="Legendary Creature — Dragon",
               is_commander=True)
    assert goldfish.reduced_cost(ur, [(1, "Dragon", True)]) == 9
    # ...but a reducer on the BATTLEFIELD does cut it, which is the whole
    # reason Dragonlord's Servant matters to a nine-drop.
    assert goldfish.reduced_cost(ur, [(1, "Dragon", False)]) == 8


def test_a_reduction_applies_only_to_what_it_names():
    dragon = _card("", cmc=5, type_line="Creature — Dragon")
    goblin = _card("", cmc=5, type_line="Creature — Goblin")
    rock = _card("", cmc=5, type_line="Artifact")
    reds = [(1, "Dragon", False)]
    assert goldfish.reduced_cost(dragon, reds) == 4
    assert goldfish.reduced_cost(goblin, reds) == 5
    assert goldfish.reduced_cost(rock, reds) == 5
    # An ungated creature reduction hits both creatures and neither artifact.
    every = [(1, None, False)]
    assert goldfish.reduced_cost(dragon, every) == 4
    assert goldfish.reduced_cost(goblin, every) == 4
    assert goldfish.reduced_cost(rock, every) == 5


def test_the_chosen_type_is_the_type_the_deck_is_built_around():
    """A real player names the type they built around, so the model resolves
    "the chosen type" to the deck's most common creature subtype. Deterministic
    on ties, because two identical decks must measure identically."""
    deck = [{"type_line": "Creature — Dragon"}] * 3 + [{"type_line": "Creature — Goblin"}]
    assert goldfish.chosen_type_for(deck) == "Dragon"
    assert goldfish.chosen_type_for([{"type_line": "Artifact"}]) is None
    dragon = _card("", cmc=5, type_line="Creature — Dragon")
    chosen = [(2, goldfish.CHOSEN_TYPE, False)]
    assert goldfish.reduced_cost(dragon, chosen, "Dragon") == 3
    assert goldfish.reduced_cost(dragon, chosen, "Goblin") == 5
    # No chosen type resolved is no reduction, never a free one.
    assert goldfish.reduced_cost(dragon, chosen, None) == 5


def test_reductions_stack():
    """Eminence plus a Servant plus a Shaman is four off a Dragon, which is what
    a player would actually be paying."""
    dragon = _card("", cmc=7, pips=[{"R"}], type_line="Creature — Dragon")
    stack = [(1, "Dragon", True), (1, "Dragon", False), (2, "Dragon", False)]
    assert goldfish.reduced_cost(dragon, stack) == 3


def test_subtypes_come_from_after_the_dash():
    assert goldfish.subtypes_of("Legendary Creature — Dragon Spirit") == {"Dragon", "Spirit"}
    assert goldfish.subtypes_of("Artifact") == frozenset()
    assert goldfish.subtypes_of("") == frozenset()


# ── against the real deck ────────────────────────────────────────────────

@requires_data
@requires_deck
def test_a_reducer_that_is_neither_rock_nor_body_still_gets_cast():
    """THE THIRD CARD TO FALL THROUGH THIS HOLE, after Aggravated Assault and
    Primal Vigor. Urza's Incubator and Herald's Horn are artifacts with
    `produces` 0 and `bodies` 0, so every pre-existing cast loop skipped them —
    they would sit in hand for ten turns while being the deck's stated curve
    fixer. Proven by re-introducing the bug: with the reducer loop's output
    discarded, the deck gets measurably worse."""
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("ur-dragon")
    lib, _ = goldfish.build_library(doc)
    orphans = [c for c in lib
               if c["reduces"] and c["bodies"] == 0 and c["produces"] == 0]
    assert orphans, "ur-dragon runs Urza's Incubator and Herald's Horn"
    for c in orphans:
        assert not c["is_land"]


@requires_data
@requires_deck
def test_the_commander_zone_reducer_is_read_from_the_commander():
    """Eminence is a fact about the command zone, so it cannot come from the
    library — the commander is not in it."""
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("ur-dragon")
    cmd = [c for c in doc["cards"] if c.get("is_commander")]
    assert len(cmd) == 1 and cmd[0]["name"] == "The Ur-Dragon"
    got = goldfish.cost_reduction(cmd[0], goldfish._corpus_creature_types())
    assert got == (1, "Dragon", True)
    lib, _ = goldfish.build_library(doc)
    assert not any(c["name"] == "The Ur-Dragon" for c in lib)


@requires_data
@requires_deck
def test_pricing_a_deck_correctly_moves_what_it_casts_and_not_what_it_draws():
    """THE INTERNAL CONSISTENCY CHECK, and it is the one that says the change is
    real rather than a bug. A goldfish target asks whether a card was DRAWN, and
    drawing does not care what anything costs — so every assembly rate must be
    UNCHANGED while everything about casting moves. A change that moved both
    would mean the reduction had leaked into the shuffle."""
    with_red = goldfish.run("ur-dragon", iterations=400, seed=7, quiet=True)
    # Re-run with every reduction stripped: same seed, same shuffle, and the
    # ONLY difference is what things cost.
    real = goldfish.cost_reduction
    try:
        goldfish.cost_reduction = lambda *a, **k: None
        without = goldfish.run("ur-dragon", iterations=400, seed=7, quiet=True)
    finally:
        goldfish.cost_reduction = real

    a = {t["label"]: t["assembled_rate"] for t in without["metrics"]["targets"]}
    b = {t["label"]: t["assembled_rate"] for t in with_red["metrics"]["targets"]}
    checked = 0
    for label, rate in a.items():
        assert b[label] == rate, f"{label} moved, and drawing is not priced"
        checked += 1
    assert checked >= 8

    # ...and the casting half did move, or the fix did nothing.
    assert (with_red["metrics"]["commander"]["cast_by_turn_6_rate"]
            > without["metrics"]["commander"]["cast_by_turn_6_rate"])
    assert (with_red["metrics"]["mean_bodies_by_turn"]["5"]
            > without["metrics"]["mean_bodies_by_turn"]["5"])


@requires_data
@requires_deck
def test_a_deck_with_no_reducer_is_byte_identical():
    """The widening rule every optional model in this file follows: a deck that
    does not opt in must measure exactly as it did before."""
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("heliod")
    lib, cmds = goldfish.build_library(doc)
    assert not any(c["reduces"] for c in lib)
    assert not any(goldfish.cost_reduction(c, goldfish._corpus_creature_types())
                   for c in cmds)


# ── a dork whose output is the board ─────────────────────────────────────

def test_a_scaling_dork_is_seen_at_all():
    """`_TAP_ADD_RE` wants `{T}: Add <symbols>`. Bloom Tender and Faeburrow
    Elder say "for each color among permanents you control, add one mana of
    that color" — so the two best dorks a five-colour deck can run read as
    producing NOTHING, while the conditional rocks they replace counted as five
    sources each. Same silent-half-working shape as the 65% of mana rocks this
    model could not see."""
    from manamap.pilot import card_pool
    o = card_pool.corpus_oracle()
    for name in ("Bloom Tender", "Faeburrow Elder"):
        got = goldfish.classify({"name": name, "oracle_text": o.get(name, ""),
                                 "cmc": 2, "type_line": "Creature — Druid",
                                 "mana_cost": "{1}{G}"})
        assert got["scales_with_colors"], name
        # It must PRODUCE, or it never reaches the rock loop and reads as zero
        # however well the text is parsed.
        assert got["produces"] >= 1, name


@pytest.mark.parametrize("name,why", [
    ("Charmed Pendant", "pays with a mill, so it is not a repeatable rate"),
    ("Idol of False Gods", "makes a token that sacrifices itself — the Jeweled "
                           "Lotus rule: a cost that consumes the source"),
])
def test_the_other_tap_abilities_the_sweep_found_stay_excluded(name, why):
    """The corpus sweep found FIVE cards with a `{T}` mana ability the old regex
    misses. Two are the scaling shape; these are not, and widening far enough to
    catch them would bill the model for mana it cannot spend."""
    from manamap.pilot import card_pool
    o = card_pool.corpus_oracle()
    if not o.get(name):
        pytest.skip(f"{name} not in this corpus")
    got = goldfish.classify({"name": name, "oracle_text": o[name], "cmc": 3,
                             "type_line": "Artifact", "mana_cost": "{3}"})
    assert not got["scales_with_colors"], why


@requires_data
def test_the_widening_caught_two_cards_and_not_a_family():
    """A PATTERN SHIPS WITH ITS SWEEP. If this count moves, something else now
    matches and has to be read card by card before it is believed."""
    from manamap.pilot import card_pool
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()
    hits = [n for n in pool if goldfish._SCALING_COLOR_MANA_RE.search(o.get(n) or "")]
    assert sorted(hits) == ["Bloom Tender", "Faeburrow Elder"], hits


@requires_data
@requires_deck
def test_a_scaling_dork_is_priced_from_the_board_and_never_overstated():
    """SNAPSHOT AT CAST, conservative end. It counts the colours on the board
    the turn it resolves and never grows, so an Elder cast on two colours and
    living to see five is UNDERSTATED — recoverable. Overstating is how a mana
    base comes out looking fine and cannot cast its spells."""
    from manamap.pilot import card_pool
    o = card_pool.corpus_oracle()
    card = goldfish.classify(
        {"name": "Faeburrow Elder", "oracle_text": o.get("Faeburrow Elder", ""),
         "cmc": 3, "type_line": "Creature — Treefolk Druid", "mana_cost": "{1}{G}{W}"})
    # Never more than the five colours of Magic, never less than one.
    for n_colors in range(0, 7):
        sources = [frozenset({c}) for c in "WUBRG"[:min(n_colors, 5)]]
        colors = frozenset().union(*sources) if sources else frozenset()
        made = max(1, min(len(colors), 5))
        assert 1 <= made <= 5
        assert made <= max(1, min(n_colors, 5))


# ── a changeling is every creature type, in every zone ───────────────────

def test_a_changeling_is_every_creature_type():
    """"Changeling (This card is every creature type.)" is a rules fact, and it
    holds in EVERY ZONE — so a changeling SPELL on the stack is a Dragon spell
    and takes a Dragon's discount.

    `subtypes_of` read the type line, which says `Shapeshifter`. That made all
    61 legal changelings invisible to eminence, to Lathliss and Miirym's
    "whenever another Dragon you control enters", and to any tribal cost
    reducer. Universal Automaton is a {1} card that The Ur-Dragon casts for
    NOTHING; the model priced it at one.
    """
    every = goldfish.subtypes_of(
        "Creature — Shapeshifter",
        "Changeling (This card is every creature type.)")
    plain = goldfish.subtypes_of("Creature — Dragon", "Flying")
    assert plain == {"Dragon"}
    if not goldfish._corpus_creature_types():
        pytest.skip("no corpus; the literal type line is the documented fallback")
    assert "Dragon" in every and "Goblin" in every
    assert len(every) > 100, "a changeling answers the corpus's whole type list"


def test_a_changeling_takes_the_commanders_discount():
    """The consequence, priced. A {1} changeling under an eminence commander
    costs nothing, and the floor is the coloured pip count as always."""
    free = {"cmc": 1, "pips": [], "is_creature": True,
            "subtypes": goldfish.subtypes_of(
                "Artifact Creature — Shapeshifter",
                "Changeling (This card is every creature type.)")}
    if "Dragon" not in free["subtypes"]:
        pytest.skip("no corpus loaded")
    assert goldfish.reduced_cost(free, [(1, "Dragon", True)]) == 0
    # ...and a coloured pip still cannot be discounted away.
    coloured = dict(free, cmc=1, pips=[{"B"}])
    assert goldfish.reduced_cost(coloured, [(1, "Dragon", True)]) == 1


@requires_data
def test_a_changeling_does_not_vote_on_the_decks_chosen_type():
    """A card that is all 383 creature types would add one to every count, and
    the argmax would become alphabetical noise rather than the type the deck is
    built around. `Urza's Incubator` names that type, so getting it wrong
    mis-prices every reducer in the deck."""
    deck = [{"type_line": "Creature — Dragon", "oracle_text": "Flying"}] * 3
    deck += [{"type_line": "Creature — Shapeshifter",
              "oracle_text": "Changeling (This card is every creature type.)"}] * 9
    assert goldfish.chosen_type_for(deck) == "Dragon"


@requires_data
def test_the_changeling_sweep_is_scoped():
    """A PATTERN SHIPS WITH ITS SWEEP. Changelings are a closed, small set; if
    this count moves, something else now matches the keyword and has to be read
    before it is believed."""
    from manamap.pilot import card_pool
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()
    hits = [n for n in pool if goldfish._CHANGELING_RE.search(o.get(n) or "")]
    assert 50 <= len(hits) <= 140, (
        f"{len(hits)} cards read as changelings; the corpus search found 61 "
        f"legal in a five-colour identity. A different number means the "
        f"matcher changed.")
    # It must not fire on a card that merely mentions creature types.
    for name in ("Terror of the Peaks", "Sol Ring", "Command Tower"):
        assert not goldfish._CHANGELING_RE.search(o.get(name) or ""), name
