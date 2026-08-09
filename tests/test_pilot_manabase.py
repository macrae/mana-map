"""Tests for the mana base builder (pilot/manabase.py)."""


from manamap.pilot.manabase import (
    build,
    cards_seen,
    count_pips,
    count_sources,
    effective_pips,
    enters_tapped,
    hypergeometric_at_least,
    land_colors,
    pip_requirements,
    select_lands,
    source_targets,
    sources_needed,
)


def _spell(name, mana_cost, cmc, type_line="Instant"):
    return {"name": name, "mana_cost": mana_cost, "cmc": cmc, "type_line": type_line}


def _land(name, text, type_line="Land", ci=""):
    return {"name": name, "oracle_text": text, "type_line": type_line, "color_identity": ci}


def _basic(colour, name):
    return _land(name, f"{{T}}: Add {{{colour}}}.", f"Basic Land — {name}", colour)


# ── hypergeometric core ──


def test_cards_seen_on_the_play():
    assert cards_seen(1) == 7
    assert cards_seen(3) == 9


def test_cards_seen_on_the_draw():
    assert cards_seen(1, on_the_play=False) == 8


def test_hypergeometric_certain_when_nothing_wanted():
    assert hypergeometric_at_least(0, 7, 0) == 1.0


def test_hypergeometric_impossible_when_too_few_sources():
    assert hypergeometric_at_least(1, 7, 2) == 0.0


def test_hypergeometric_increases_with_sources():
    a = hypergeometric_at_least(20, 9, 1)
    b = hypergeometric_at_least(30, 9, 1)
    assert b > a


def test_sources_needed_grows_with_pips():
    assert sources_needed(2, 3) > sources_needed(1, 3)


def test_sources_needed_shrinks_with_time():
    assert sources_needed(1, 5) < sources_needed(1, 3)


def test_sources_needed_zero_pips():
    assert sources_needed(0, 3) == 0


# ── pip counting ──


def test_count_pips_single_colour():
    assert count_pips("{3}{R}")["R"] == 1.0


def test_count_pips_double_pip_is_not_two_singles():
    """{G}{G} must weigh more than one {G} — v1 counted them flat."""
    assert count_pips("{G}{G}")["G"] == 2.0


def test_count_pips_hybrid_splits_evenly():
    pips = count_pips("{W/U}")
    assert pips["W"] == 0.5
    assert pips["U"] == 0.5


def test_count_pips_generic_only():
    assert all(v == 0 for v in count_pips("{4}").values())


def test_count_pips_empty():
    assert all(v == 0 for v in count_pips("").values())


# ── requirements ──


def test_pip_requirements_ignores_lands():
    cards = [_spell("Mountain", "", 0, "Basic Land — Mountain"), _spell("Bolt", "{R}", 1)]
    reqs = pip_requirements(cards)
    assert reqs["R"]["cards"] == 1


def test_pip_requirements_tracks_heaviest_and_earliest():
    cards = [_spell("A", "{G}", 1), _spell("B", "{2}{G}{G}", 4)]
    reqs = pip_requirements(cards)
    assert reqs["G"]["max_pips"] == 2
    assert reqs["G"]["earliest_turn"] == 1


def test_pip_requirements_drops_unused_colours():
    assert set(pip_requirements([_spell("A", "{R}", 1)])) == {"R"}


def test_source_targets_respects_the_planning_horizon():
    """A turn-1 single pip must not demand a turn-1-probability source count."""
    early = source_targets({"R": {"max_pips": 1, "effective_pips": 1, "earliest_turn": 1,
                                  "cards": 1, "total_pips": 1.0}})
    at_three = sources_needed(1, 3)
    assert early["R"] == at_three


# ── effective pips: one bomb must not distort the base ──


def test_effective_pips_ignores_a_lone_outlier():
    """A single {B}{B}{B} in a deck of single pips would demand 48 sources."""
    weights = [1] * 20 + [3]
    assert effective_pips(weights) == 1


def test_effective_pips_respects_a_real_quorum():
    weights = [2] * 10 + [1] * 10
    assert effective_pips(weights) == 2


def test_effective_pips_empty():
    assert effective_pips([]) == 0


def test_requirements_report_both_effective_and_max():
    cards = [_spell("A", "{B}", 1)] * 10 + [_spell("B", "{B}{B}{B}", 3)]
    reqs = pip_requirements(cards)
    assert reqs["B"]["max_pips"] == 3
    assert reqs["B"]["effective_pips"] == 1


# ── land colour detection ──


def test_land_colors_from_basic_subtype():
    assert land_colors(_land("Temple Garden", "", "Land — Forest Plains")) == {"G", "W"}


def test_land_colors_from_add_symbol():
    assert land_colors(_land("Bad River", "{T}: Add {U}.")) == {"U"}


def test_land_colors_any_color():
    assert land_colors(_land("City of Brass", "{T}: Add one mana of any color.")) == set("WUBRG")


def test_restricted_any_color_is_not_a_five_colour_source():
    """Haven of the Spirit Dragon taps for {C} in a deck with no Dragons.

    A greedy selector reaches for these precisely because they look like they
    cover everything, which is how a Vampire deck ended up running two
    Dragon-restricted lands.
    """
    haven = _land(
        "Haven of the Spirit Dragon",
        "{T}: Add {C}. {T}: Add one mana of any color. Spend this mana only to "
        "cast a Dragon creature spell.")
    assert land_colors(haven) == set()


def test_restricted_source_does_not_fall_back_to_color_identity():
    """The fallback would smuggle the restriction back in through the side door."""
    card = _land("Restricted Thing",
                 "{T}: Add one mana of any color. Spend this mana only to cast a Dragon.",
                 ci="W")
    assert land_colors(card) == set()


def test_tribal_restriction_is_counted_conservatively():
    """Cavern of Souls is nearly free in a tribal deck, but understating a
    source is recoverable and overstating one is not."""
    cavern = _land(
        "Cavern of Souls",
        "{T}: Add {C}. {T}: Add one mana of any color. Spend this mana only to "
        "cast a creature spell of the chosen type.")
    assert land_colors(cavern) == set()


def test_unrestricted_any_color_still_counts_fully():
    grotto = _land("Hidden Grotto", "{T}: Add {C}. {1}, {T}: Add one mana of any color.")
    assert land_colors(grotto) == set("WUBRG")


def test_enters_tapped_detection():
    assert enters_tapped(_land("Guildgate", "This land enters tapped."))
    assert not enters_tapped(_land("Plains", "{T}: Add {W}."))


# ── selection ──


def test_select_lands_prefers_lands_that_close_a_real_gap():
    """A five-colour land in a two-colour deck is worth its two colours, no more."""
    pool = [
        _land("City of Brass", "{T}: Add one mana of any color."),
        _land("Temple Garden", "", "Land — Forest Plains"),
    ]
    chosen, _, _ = select_lands(pool, {"G": 1, "W": 1}, slots=1)
    # Both cover both needed colours; the tie breaks deterministically by name
    assert len(chosen) == 1


def test_select_lands_skips_lands_that_help_nothing():
    pool = [_land("Island", "{T}: Add {U}.", "Basic Land — Island", "U")]
    chosen, remaining, _ = select_lands(pool, {"R": 2}, slots=5)
    assert chosen == []
    assert remaining == {"R": 2}


def test_select_lands_respects_the_tapped_budget():
    pool = [_land(f"Gate{i}", "This land enters tapped. {T}: Add {R}.") for i in range(10)]
    chosen, _, tapped = select_lands(pool, {"R": 10}, slots=10, tapped_budget=2)
    assert tapped <= 2
    assert len(chosen) <= 2


def test_select_lands_is_deterministic():
    pool = [_land("B", "{T}: Add {R}."), _land("A", "{T}: Add {R}.")]
    first, _, _ = select_lands(pool, {"R": 1}, slots=1)
    second, _, _ = select_lands(list(reversed(pool)), {"R": 1}, slots=1)
    assert [c["name"] for c in first] == [c["name"] for c in second]


# ── end to end ──


def test_build_fills_every_slot():
    spells = [_spell(f"S{i}", "{R}", 1) for i in range(30)]
    lands, _ = build(spells, [], 36, {"R": _basic("R", "Mountain")})
    assert len(lands) == 36


def test_build_mono_colour_meets_its_target():
    spells = [_spell(f"S{i}", "{1}{R}{R}", 3) for i in range(20)]
    lands, diag = build(spells, [], 36, {"R": _basic("R", "Mountain")})
    assert diag["shortfalls"] == {}
    assert diag["on_curve_probability"]["R"] >= 0.90


def test_build_reports_a_shortfall_rather_than_hiding_it():
    """Not enough slots for the requirement — say so, don't silently succeed."""
    spells = [_spell(f"S{i}", "{G}{G}{W}{W}", 4) for i in range(20)]
    _, diag = build(spells, [], 10, {"G": _basic("G", "Forest"), "W": _basic("W", "Plains")})
    assert diag["shortfalls"]


def test_build_splits_basics_between_colours():
    spells = [_spell("A", "{G}", 1), _spell("B", "{W}", 1)]
    lands, _ = build(spells, [], 20, {"G": _basic("G", "Forest"), "W": _basic("W", "Plains")})
    names = {land["name"] for land in lands}
    assert names == {"Forest", "Plains"}


def test_build_diagnostics_state_their_method():
    spells = [_spell("A", "{R}", 1)]
    _, diag = build(spells, [], 20, {"R": _basic("R", "Mountain")})
    assert "hypergeometric" in diag["method"]


def test_build_is_deterministic():
    spells = [_spell(f"S{i}", "{R}", 1) for i in range(10)]
    basics = {"R": _basic("R", "Mountain")}
    a, da = build(spells, [], 30, basics)
    b, db = build(spells, [], 30, basics)
    assert [c["name"] for c in a] == [c["name"] for c in b]
    assert da == db


def test_count_sources_counts_each_colour_a_land_makes():
    lands = [_land("Temple Garden", "", "Land — Forest Plains")]
    assert count_sources(lands, {"G", "W"}) == {"G": 1, "W": 1}
