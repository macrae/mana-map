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


# ── fetchlands: colours the DECK supplies, not the card ──


_FOOTHILLS = _land(
    "Wooded Foothills",
    "{T}, Pay 1 life, Sacrifice this land: Search your library for a Mountain "
    "or Forest card, put it onto the battlefield, then shuffle.")
_PANORAMA = _land(
    "Bant Panorama",
    "{T}: Add {C}. {1}, {T}, Sacrifice this land: Search your library for a "
    "basic Forest, Plains, or Island card, put it onto the battlefield tapped, "
    "then shuffle.")
_CRYPT = _land("Blood Crypt", "", "Land — Swamp Mountain")
_POOL = _land("Breeding Pool", "", "Land — Forest Island")


def test_a_fetchland_with_no_pool_produces_nothing():
    """THE BYTE-IDENTICAL GUARANTEE. `goldfish` calls this with a pool now, but
    anything that cannot supply one must get exactly the old answer."""
    assert land_colors(_FOOTHILLS) == set()


def test_a_fetchland_produces_what_it_can_actually_find():
    """Foothills reaches Blood Crypt on the Mountain type, so it is a black
    source in a deck holding one — the claim `mana-fit` could not make."""
    assert land_colors(_FOOTHILLS, pool=[_CRYPT]) == {"B", "R"}
    assert land_colors(_FOOTHILLS, pool=[_CRYPT, _POOL]) == {"B", "R", "G", "U"}


def test_a_fetchland_is_blind_to_lands_the_deck_does_not_hold():
    """The control the deck-blind version could not have: same card, empty deck."""
    assert land_colors(_FOOTHILLS, pool=[]) == set()
    assert land_colors(_FOOTHILLS, pool=[_land("Island", "", "Basic Land — Island")]) == set()


def test_a_basic_of_type_fetch_cannot_find_a_dual():
    """RE-INTRODUCING THE BUG THIS DISTINCTION EXISTS FOR. Bant Panorama and
    Wooded Foothills read almost identically — `a basic Forest, Plains, or
    Island card` against `a Mountain or Forest card`. Twenty corpus lands are
    the Panorama shape. Dropping the word `basic` makes the Panorama a
    five-colour source off duals it can never legally fetch."""
    forest = _land("Forest", "", "Basic Land — Forest")
    assert land_colors(_PANORAMA, pool=[_POOL]) == set(), "found a nonbasic dual"
    assert land_colors(_PANORAMA, pool=[forest]) == {"G"}
    # …while the true fetch DOES reach the dual, on the same pool.
    assert land_colors(_FOOTHILLS, pool=[_POOL]) == {"G", "U"}


_CAVE_TEXT = ("{T}: Add {C}. {3}, {T}, Sacrifice this land: Search your library "
              "for a land card, put it onto the battlefield tapped, then shuffle.")


def test_a_fetch_cannot_fetch_itself():
    """DEFENSIVE, and the assertion has to drive `fetch_targets` to see it.

    Going through `land_colors` proves nothing here: a self-fetch would be
    priced with a bare `land_colors`, which for a fetch is the empty set, so the
    union is unchanged either way and the test passes with the guard removed.
    No printed card is both a fetch and a legal target for itself — this holds
    the line for one that is.
    """
    from manamap.pilot.manabase import fetch_targets
    self_typed = _land(
        "Selffinder",
        "{T}, Sacrifice this land: Search your library for a Mountain card, "
        "put it onto the battlefield, then shuffle.",
        "Land — Mountain")
    assert fetch_targets(self_typed, [self_typed]) == []
    assert [c["name"] for c in fetch_targets(self_typed, [self_typed, _CRYPT])] \
        == ["Blood Crypt"]


def test_fetch_resolution_terminates_on_mutually_fetching_lands():
    """THE REASON TARGETS ARE PRICED WITH A BARE `land_colors`.

    Colours cannot show this — union is idempotent, so one hop and two hops
    agree on every real board. The failure mode is not a wrong colour, it is no
    answer at all: two `a land card` fetches each find the other and recurse
    until the stack ends. Urza's Cave is the printed one; a second is enough to
    close the loop.
    """
    cave = _land("Urza's Cave", _CAVE_TEXT)
    twin = _land("Urza's Other Cave", _CAVE_TEXT)
    # THE CARD IS IN ITS OWN POOL, which is how every caller passes it —
    # `mana_analysis` hands over the deck's whole land list. Leave it out and
    # the loop cannot close: cave finds twin, twin self-excludes, done.
    assert land_colors(cave, pool=[cave, twin, _CRYPT]) == {"B", "R"}
    assert land_colors(cave, pool=[cave, _FOOTHILLS]) == set()


def test_a_pool_does_not_move_a_land_that_does_not_fetch():
    """The control: `pool` must be inert for every land that is not a fetch."""
    for card in (_CRYPT, _POOL,
                 _land("City of Brass", "{T}: Add one mana of any color.")):
        assert land_colors(card) == land_colors(card, pool=[_CRYPT, _POOL])


def test_searches_that_are_not_at_will_battlefield_fetches_are_excluded():
    """Each of these fails the gate for a DIFFERENT reason, and each was read
    card by card in the corpus sweep."""
    from manamap.pilot.manabase import fetch_profile
    hand = _land("Thaumatic Compass",
                 "{3}, {T}: Search your library for a basic land card, reveal "
                 "it, put it into your hand, then shuffle.")
    death = _land("Flagstones of Trokair",
                  "{T}: Add {W}. When Flagstones of Trokair is put into a "
                  "graveyard from the battlefield, you may search your library "
                  "for a Plains card, put it onto the battlefield tapped.")
    cycle = _land("Ash Barrens",
                  "{T}: Add {C}. Basic landcycling {1} ({1}, Discard this card: "
                  "Search your library for a basic land card, reveal it, put it "
                  "into your hand, then shuffle.)")
    assert fetch_profile(hand) is None, "into your HAND is not a source"
    assert fetch_profile(death) is None, "a death trigger is not at-will"
    assert fetch_profile(cycle) is None, "landcycling is a cost paid from hand"
    # A search for a NON-LAND noun is not a fetch however it is worded.
    dragon = _land("Maelstrom of the Spirit Dragon",
                   "{T}: Add {C}. {4}, {T}, Sacrifice this land: Search your "
                   "library for a Dragon card, put it onto the battlefield.")
    assert fetch_profile(dragon) is None, "a Dragon is not a land"
    assert land_colors(death, pool=[_CRYPT]) == {"W"}, "keeps its own {W}, gains nothing"


def test_no_corpus_fetch_produces_a_colour_without_a_pool():
    """THE BYTE-IDENTICAL GUARANTEE, over the real corpus.

    `land_colors(card) == land_colors(card, pool=None)` would be the obvious
    assertion and it is worthless — both arguments take the same branch, so it
    holds however the fetch layer is wired. This asserts the property that
    branch exists to protect instead: a fetch handed no deck produces NOTHING,
    which is what keeps `goldfish` reproducible for every caller that has no
    pool to give. A default pool, or a fetch layer that ran unconditionally,
    fails here.

    Measured at the time of the change: 1266 corpus lands, 16 true fetches and
    20 basic-of-type, all of them empty without a pool.
    """
    import pytest
    from manamap.pilot.manabase import fetch_profile
    try:
        from manamap.pilot import card_pool
        pool = card_pool.load_pool()
        oracle = card_pool.corpus_oracle()
    except Exception:  # pragma: no cover - corpus absent
        pytest.skip("corpus not built")
    lands = fetches = 0
    for name, info in pool.items():
        if "Land" not in (info.get("type_line") or ""):
            continue
        card = dict(info, name=name, oracle_text=oracle.get(name, ""))
        lands += 1
        if fetch_profile(card) is None:
            continue
        fetches += 1
        # Its OWN text may still make mana (the Panoramas tap for {C}); what it
        # may never do is claim a colour only a target could supply.
        assert land_colors(card) <= land_colors(dict(card, oracle_text=""))\
            | {c for c in "WUBRG" if "{%s}" % c in (card["oracle_text"] or "")}, name
    assert lands >= 1000, f"only {lands} lands swept"
    assert fetches >= 30, f"only {fetches} fetches found"
