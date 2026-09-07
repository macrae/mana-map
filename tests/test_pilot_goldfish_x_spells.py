"""X-spell draw: the channel a card-advantage deck was measured without.

Found on heliod, whose entire plan is drawing cards. Of its 28 instants and
sorceries the model had a reason to cast FOUR, and the 24 it could not see
included Braingeyser, Stroke of Genius, Prosperity and Skyscribing — every
X-cost draw spell in the list. `mean_extra_cards_drawn_by_turn` read 0.428 by
turn eight on a deck built to draw, and the branch measurement that depended on
it could not tell a card-advantage change from noise.

The class was NAMED but not measured: `net_change`'s caveats already said
"activated, X-based, sacrifice-gated and death-triggered draw are unmodelled".
This closes the X-based quarter of it.

TWO THINGS MAKE IT WORK, and each has a test here because each was a bug first:

1. `draw_profile` returned early. Its `_DRAW_RE` guard wants a WRITTEN-OUT
   quantity ("draw two cards") and does not recognise "draws X cards", so the
   whole family returned an all-zero profile with `unmodelled` still None —
   the one value that means "there is nothing here to model".
2. The spell has to be cast LAST. Scryfall counts {X} as zero, so an X spell's
   `cmc` is only its fixed part and every cheapest-first loop in the module
   would have fired Stroke of Genius on turn three for X=0.
"""

import pytest

from manamap.pilot import goldfish

from conftest import requires_data, requires_deck


def _spell(name, text, mana_cost, cmc, type_line="Sorcery"):
    return {"name": name, "type_line": type_line, "cmc": cmc,
            "mana_cost": mana_cost, "oracle_text": text, "quantity": 1,
            "is_commander": False, "power": None, "toughness": None}


# ── 1. The profile, and the boundaries the sweep bought ───────────────────

def test_the_three_wordings_all_credit_the_caster():
    """The sweep found four shapes and they all give the CASTER the cards,
    which is why one rule covers them. `x_draw_multiplier` is the number of {X}
    symbols, not a boolean: {X}{X} buys one card per TWO mana."""
    for name, text, mc, cmc in [
            ("Braingeyser", "Target player draws X cards.", "{X}{U}{U}", 2),
            ("Mind Spring", "Draw X cards.", "{X}{U}{U}", 2),
            ("Prosperity", "Each player draws X cards.", "{X}{U}", 1),
    ]:
        p = goldfish.draw_profile(_spell(name, text, mc, cmc))
        assert p["x_draw_multiplier"] == 1, name
        assert p["x_draw_discard"] == 0, name
        assert p["unmodelled"] is None, name


def test_the_fixed_part_is_cmc_because_scryfall_counts_x_as_zero():
    """The whole rule rests on this. If it ever stopped being true the model
    would subtract the wrong number and credit X spells with mana they cannot
    spend, silently."""
    assert goldfish.draw_profile(
        _spell("Stroke of Genius", "Target player draws X cards.",
               "{X}{2}{U}", 3))["x_draw_multiplier"] == 1
    # {X}{2}{U} -> the three fixed mana are the cmc; {X}{U}{U} -> two.
    assert goldfish.draw_profile(
        _spell("Braingeyser", "Target player draws X cards.",
               "{X}{U}{U}", 2))["x_draw_multiplier"] == 1


def test_a_discard_rider_is_netted_and_a_full_discard_is_refused():
    """NET, NOT GROSS. Read the Runes draws X and discards up to X — a filter,
    not card advantage, and at {X}{U} it has the cheapest fixed cost in the
    whole family, so crediting it X would have made the worst card in the class
    read as the best. Pull from Tomorrow discards exactly one."""
    pull = goldfish.draw_profile(_spell(
        "Pull from Tomorrow", "Draw X cards, then discard a card.",
        "{X}{U}{U}", 2, "Instant"))
    assert pull["x_draw_multiplier"] == 1 and pull["x_draw_discard"] == 1

    runes = goldfish.draw_profile(_spell(
        "Read the Runes",
        "Draw X cards. For each card drawn this way, discard a card unless you "
        "sacrifice a permanent.", "{X}{U}", 1))
    assert runes["x_draw_multiplier"] == 0, "a filter is not card advantage"

    occult = goldfish.draw_profile(_spell(
        "Occult Epiphany", "Draw X cards, then discard X cards. Create a 1/1 "
        "white Spirit creature token with flying for each card type among "
        "cards discarded this way.", "{X}{U}", 1, "Instant"))
    assert occult["x_draw_multiplier"] == 0


def test_x_bought_with_something_other_than_mana_is_refused():
    """The rule divides the REMAINING MANA POOL, so it is only correct where X
    is bought with mana. Skeletal Scrying exiles X cards from a graveyard this
    model does not have; Ingenious Mastery has an alternative cost under which
    X is zero and the model has no way to choose. Both would otherwise have
    been credited the entire pool."""
    scrying = goldfish.draw_profile(_spell(
        "Skeletal Scrying",
        "As an additional cost to cast this spell, exile X cards from your "
        "graveyard. You draw X cards and you lose X life.", "{X}{B}", 1, "Instant"))
    assert scrying["x_draw_multiplier"] == 0

    mastery = goldfish.draw_profile(_spell(
        "Ingenious Mastery",
        "You may pay {2}{U} rather than pay this spell's mana cost. If the "
        "{2}{U} cost was paid, an opponent chooses X. Draw X cards.",
        "{X}{2}{U}", 3))
    assert mastery["x_draw_multiplier"] == 0


def test_a_split_card_is_refused_because_its_cmc_is_both_halves():
    """CR 202.3d: a split card's mana value is its halves combined. Expansion //
    Explosion is the only member of the family and the fixed part this rule
    subtracts would be wrong for it by the other half's cost."""
    p = goldfish.draw_profile(_spell(
        "Expansion // Explosion",
        "Copy target instant or sorcery spell with mana value 4 or less. // "
        "Destroy target creature. Target player draws X cards.",
        "{U/R}{U/R} // {X}{U}{U}{R}{R}", 6, "Instant // Instant"))
    assert p["x_draw_multiplier"] == 0


@requires_data
def test_the_whole_family_is_read_and_every_skip_is_deliberate():
    """THE SWEEP, held as a test. 33 X-cost instants and sorceries in the corpus
    that draw X; 28 credited and exactly 5 refused, each for a reason with its
    own test above. A widened pattern that starts crediting one of the five will
    fail here rather than quietly changing every deck's card-advantage figure.
    """
    import pandas as pd
    from manamap.config import OUTPUT_CSV_PATH
    df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    t, mc, tl = (df["oracle_text"].fillna(""), df["mana_cost"].fillna(""),
                 df["type_line"].fillna(""))
    fam = df[mc.str.contains(r"\{X\}", regex=True)
             & tl.str.contains("Instant|Sorcery", regex=True)
             & t.str.contains(r"draws? X cards?", case=False, regex=True)]
    assert len(fam) >= 30, "the family shrank — re-run the sweep before trusting this"
    skipped = set()
    for _, r in fam.iterrows():
        card = {k: (None if pd.isna(r.get(k)) else r.get(k))
                for k in ("name", "mana_cost", "type_line", "oracle_text", "cmc")}
        if not goldfish.draw_profile(card)["x_draw_multiplier"]:
            skipped.add(r["name"])
    assert skipped == {
        "Expansion // Explosion",   # cmc is both halves
        "Ingenious Mastery",        # alternative cost, X = 0
        "Occult Epiphany",          # draws X, discards X
        "Read the Runes",           # draws X, discards up to X
        "Skeletal Scrying",         # X is exiled cards, not mana
    }, f"the refusal set moved: {sorted(skipped)}"


# ── 2. The channel, driven through the simulation ─────────────────────────

@requires_data
@requires_deck
def test_the_channel_carries_a_deck_that_is_built_out_of_it():
    """Driven by BLINDING THE PROFILE rather than by re-deriving the rule: zero
    the multiplier and heliod must report far fewer cards. Measured at the time:
    3.44 extra cards by turn ten against 0.72 blind — a 4.8x figure on a deck
    whose whole plan is drawing, which is the size of what was missing."""
    real = goldfish.draw_profile

    def blind(card):
        p = real(card)
        p["x_draw_multiplier"] = 0
        return p

    on = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    goldfish.draw_profile = blind
    try:
        off = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    finally:
        goldfish.draw_profile = real
    a = on["mean_extra_cards_drawn_by_turn"]["10"]
    b = off["mean_extra_cards_drawn_by_turn"]["10"]
    assert a > 3 * b, (
        f"the X channel reads {a} against {b} blind — if these are close it is "
        f"not firing and every other test in this file proves only the parser")


@requires_data
@requires_deck
def test_casting_last_does_not_starve_the_board():
    """THE CONSERVATIVE CLAIM, and the reason the loop sits after every other
    one. An X spell here can only ever spend mana nothing else wanted, so
    switching the channel on must not cost the deck its board — a loop placed
    earlier would have eaten the whole pool and the figure would have been paid
    for out of creatures. Measured: board power at turn six went UP, 2.44 blind
    to 2.55, because the extra cards find more to cast."""
    real = goldfish.draw_profile

    def blind(card):
        p = real(card)
        p["x_draw_multiplier"] = 0
        return p

    on = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    goldfish.draw_profile = blind
    try:
        off = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    finally:
        goldfish.draw_profile = real
    assert (on["combat"]["mean_board_power_by_turn"]["6"]
            >= off["combat"]["mean_board_power_by_turn"]["6"])


@requires_data
@requires_deck
def test_the_floor_stops_the_model_burning_the_card_for_nothing():
    """`X_DRAW_MIN` is the one authored number in this channel, so it gets the
    test that shows it earns its place. Drop it to zero and the model casts
    Stroke of Genius the turn its three fixed mana are affordable, draws
    nothing, and no longer has the card for the turn it could have drawn five.
    Measured: 3.44 extra cards by turn ten at the floor against 3.03 without it.
    """
    old = goldfish.X_DRAW_MIN
    at_floor = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    goldfish.X_DRAW_MIN = 0
    try:
        no_floor = goldfish.run("heliod", iterations=2000, quiet=True)["metrics"]
    finally:
        goldfish.X_DRAW_MIN = old
    assert (at_floor["mean_extra_cards_drawn_by_turn"]["10"]
            > no_floor["mean_extra_cards_drawn_by_turn"]["10"]), (
        "casting X spells for nothing should draw FEWER cards over the game — "
        "if it does not, the floor is not doing anything and should go")
