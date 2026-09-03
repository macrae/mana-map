"""Tests for `manamap pilot build` (pilot/autobuild.py).

Every one drives the production function. Where a test exists because a defect
shipped, the docstring names the defect — and each was proved by putting the bug
back and watching the test go red, which is the only thing that separates a
guard from a claim.
"""

import json

import pytest

from conftest import requires_deck

from manamap.pilot import autobuild
from manamap.pilot.autobuild import (
    STAGES,
    BuildError,
    _BRACKET_RE,
    _read_brief,
    composition,
    flagged,
    match_theme,
)


def _themes(*pairs):
    """EDHREC's shape: slug, display name, deck count, in EDHREC's own order."""
    return [{"slug": s, "name": n, "decks": d} for s, n, d in pairs]


# ── the two things a description actually drives ────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("esper enchantment tempo, bracket 3", 3),
    ("a bracket-4 pile", 4),
    ("b5 cEDH", 5),
    ("BRACKET 1, precon", 1),
    ("board by t4-5, threat of lethal t6-7", None),   # turns are not brackets
    ("36 lands and 10 rocks", None),
    ("", None),
])
def test_only_a_bracket_is_read_out_of_free_text(text, expected):
    hit = _BRACKET_RE.search(text)
    assert (int(hit.group(1)) if hit else None) == expected


def test_a_style_is_matched_only_against_the_commanders_real_archetypes():
    """A description resolves to a style EDHREC returned, or to nothing.

    The failure this forbids is inventing one. `role_budget_for` will happily
    shape a budget from any slug it is handed and fall back with a reason if the
    fetch fails, so a fabricated slug would read as "measured" in
    `role_budget_grounding` while describing nothing.
    """
    themes = _themes(("enchantress", "Enchantress", 1201),
                     ("voltron", "Voltron", 361),
                     ("tempo", "Tempo", 29))

    assert match_theme("esper enchantress tempo", themes)["slug"] == "enchantress"
    assert match_theme("suit up the commander, voltron", themes)["slug"] == "voltron"
    assert match_theme("I want it to go wide with tokens", themes) is None
    assert match_theme("", themes) is None
    assert match_theme("anything at all", []) is None


def test_a_bigger_overlap_wins_and_edhrec_order_breaks_the_tie():
    """Ties must NOT be broken by deck count — that would be a ranking.

    §7.2 permits play rates as data and forbids the tool choosing a style. So
    the tiebreak is EDHREC's own order, which `list_themes` preserves, and the
    test asserts the low-count style wins when it comes first.
    """
    themes = _themes(("stax", "Stax", 542), ("control", "Control", 529))
    assert match_theme("a stax control deck", themes)["slug"] == "stax"

    # Same overlap size, and the FIRST one wins even though it has fewer decks.
    ordered = _themes(("mill", "Mill", 62), ("discard", "Discard", 611))
    assert match_theme("mill discard", ordered)["slug"] == "mill"

    # Two words beat one, whatever the order.
    two = _themes(("control", "Control", 529),
                  ("pillow-fort", "Pillow Fort", 200))
    assert match_theme("pillow fort control", two)["slug"] == "pillow-fort"


def test_common_words_do_not_match_a_style():
    """"an enchantment deck" must not match a style whose name contains "Deck"."""
    themes = _themes(("decks", "Decks", 900), ("enchantress", "Enchantress", 1201))
    assert match_theme("I want a deck that is fun", themes) is None


# ── the composition report ──────────────────────────────────────────────────

def _card(name, cmc=2.0, type_line="Creature — Human", quantity=1):
    return {"name": name, "cmc": cmc, "type_line": type_line,
            "quantity": quantity, "card_faces": []}


def test_the_curve_counts_copies_not_entries(monkeypatch):
    """Thirty basics are thirty lands, not one.

    `cards.json` stores basics as ONE entry with `quantity: N`. Counting entries
    published "18 lands" for a 33-land deck once already, which is why
    `expand_copies` exists and why nothing here may iterate the raw list.
    """
    # `Dragon's Approach` is the reason a NONLAND may carry a quantity: it says
    # "a deck can have any number", and so do Rat Colony, Persistent Petitioners
    # and Shadowborn Apostle. Without one of those in the fixture the only
    # multi-copy entry is a basic — which the curve excludes anyway — and the
    # test cannot tell copies from entries at all. It was written that way
    # first, and passed with the bug put back, which is what proved it vacuous.
    doc = {"cards": [
        _card("Swamp", 0.0, "Basic Land — Swamp", quantity=30),
        _card("Dragon's Approach", 3.0, "Sorcery", quantity=20),
        _card("Sol Ring", 1.0, "Artifact"),
        _card("Zur the Enchanter", 4.0, "Legendary Creature — Human Wizard"),
        _card("Doom Blade", 2.0, "Instant"),
    ]}
    monkeypatch.setattr("manamap.pilot.common.load_deck_cards", lambda *a, **k: doc)

    comp = composition("any", {"manabase": {}, "role_budget": {}})

    # Twenty Approaches, not one. The thirty Swamps are lands and are excluded.
    assert comp["nonlands"] == 23
    assert comp["curve"] == {1: 1, 2: 1, 3: 20, 4: 1}
    assert comp["mean_mana_value"] == pytest.approx((1 + 2 + 3 * 20 + 4) / 23, abs=0.01)


def test_composition_carries_the_pip_to_source_coverage(monkeypatch):
    """A-1 asks for pip-to-source per colour, and it is the manabase's own.

    Passed through rather than recomputed: `manabase` owns the Karsten maths,
    and a second implementation here would be free to disagree with the one the
    build actually used.
    """
    monkeypatch.setattr("manamap.pilot.common.load_deck_cards",
                        lambda *a, **k: {"cards": [_card("Doom Blade")]})
    plan = {"role_budget": {"lands": 36, "removal": 8},
            "manabase": {"sources": {"W": 27}, "source_targets": {"W": 22},
                         "requirements": {"W": {"total_pips": 10.0}},
                         "on_curve_probability": {"W": 0.951}}}

    comp = composition("any", plan)

    assert comp["sources"] == {"W": 27}
    assert comp["source_targets"] == {"W": 22}
    assert comp["requirements"] == {"W": 10.0}
    assert comp["on_curve_probability"] == {"W": 0.951}
    assert comp["depth"] == {"lands": 36, "removal": 8}


def test_the_headline_reads_the_benchmarks_real_key(monkeypatch, tmp_path):
    """`benchmark.json` nests under `metrics`, not `measures`.

    The first cut of `_headline` read `measures`, got `{}` back, and printed a
    build report with no mana figures at all — a silently-empty read that looks
    exactly like a deck with nothing to say. The fixture is the real record's
    shape, so the key cannot be changed on one side only.
    """
    monkeypatch.setattr(autobuild, "deck_dir", lambda slug, branch=None: tmp_path)
    record = {"metrics": {
        "mana_screw": {"missed_land_drop_rate": 0.224, "mulligan_rate": 0.201,
                       "mana_at_turn_five": 4.325},
        "response": {"answer_cards": 14, "classes_covered": 7,
                     "classes_possible": 15},
        "speed": {"kill_by_turn_8": 0.016},
    }}

    head = autobuild._headline("any", record)

    assert head["mana_screw"]["missed_land_drop_rate"] == 0.224
    assert head["response"]["answer_cards"] == 14
    # And no goldfish artifact in tmp_path means those keys are ABSENT rather
    # than zero — a figure nobody measured must not read as a measurement.
    assert "commander_by_t6" not in head


# ── the flagged-gaps report ─────────────────────────────────────────────────

def _plan(**over):
    plan = {"role_budget_deviation": {}, "must_include_illegal": [],
            "manabase": {"shortfalls": {}}, "cut_for_bracket": [],
            "bracket": {"target": 3}, "slots": [], "lands": []}
    plan.update(over)
    return plan


def test_a_role_the_builder_could_not_fill_is_flagged(monkeypatch):
    """A-1: an unfilled category is flagged explicitly, never silently padded."""
    monkeypatch.setattr(autobuild, "_unowned", lambda plan: set())
    plan = _plan(role_budget_deviation={
        "removal": {"target": 8, "actual": 6},
        "draw": {"target": 9, "actual": 9},
        "ramp": {"target": 10, "actual": 12},          # over is not a shortfall
    })
    flags = flagged(plan, {"theme": "enchantress"}, theme_decks=1201)
    assert any("removal at depth 6, budget wants 8" in f for f in flags)
    assert not any("draw" in f for f in flags)
    assert not any("ramp" in f for f in flags)


def test_bracket_cuts_and_illegal_includes_are_named(monkeypatch):
    monkeypatch.setattr(autobuild, "_unowned", lambda plan: set())
    plan = _plan(cut_for_bracket=[{"name": "Planar Bridge"}],
                 must_include_illegal=[{"name": "Sterling Grove"}],
                 manabase={"shortfalls": {"W": 3}})
    flags = flagged(plan, {"theme": "stax"}, theme_decks=542)
    assert any("Planar Bridge" in f and "bracket 3" in f for f in flags)
    assert any("Sterling Grove" in f and "colour identity" in f for f in flags)
    assert any("W by 3" in f for f in flags)


def test_a_thin_style_is_flagged_rather_than_refused(monkeypatch):
    """A 29-deck histogram describes 29 decks.

    `archetypes.MIN_DECKS_FOR_TEMPLATE` is read by that module's REPORT and by
    nothing in the build path, so `role_budget_for` shapes a budget from a
    29-deck style exactly as readily as from a 1201-deck one. The pilot wrote
    the word, so this is named rather than overridden.
    """
    from manamap.pilot.archetypes import MIN_DECKS_FOR_TEMPLATE

    monkeypatch.setattr(autobuild, "_unowned", lambda plan: set())
    thin = flagged(_plan(), {"theme": "tempo"}, theme_decks=29)
    assert any("only 29 decks" in f for f in thin)

    fat = flagged(_plan(), {"theme": "enchantress"},
                  theme_decks=MIN_DECKS_FOR_TEMPLATE)
    assert not any("decks behind it" in f for f in fat)

    none = flagged(_plan(), {}, theme_decks=None)
    assert any("no style resolved" in f for f in none)


def test_an_empty_collection_does_not_report_the_whole_99_as_unowned(monkeypatch):
    """Ownership means a BOX, and no boxes is not the same as owning nothing."""
    monkeypatch.setattr("manamap.pilot.collection.owned_names", lambda **k: set())
    assert autobuild._unowned(_plan(slots=[{"name": "Sol Ring"}])) == set()

    monkeypatch.setattr("manamap.pilot.collection.owned_names",
                        lambda **k: {"Sol Ring"})
    plan = _plan(slots=[{"name": "Sol Ring"}, {"name": "Mana Crypt"}])
    assert autobuild._unowned(plan) == {"Mana Crypt"}


# ── the brief is authored, and a build may not overwrite one ────────────────

@requires_deck
def test_a_build_refuses_to_reshape_a_brief_that_already_exists():
    """`--bracket 4` against an existing brief is a refusal, not a silent no-op.

    Three behaviours were available — overwrite the file, ignore the flag, or
    refuse — and two of them lose something the pilot typed.
    """
    from types import SimpleNamespace

    args = SimpleNamespace(slug="zur-enchantress", commander=None, theme=None,
                           bracket=4, library=[], from_file=None, brief=None)
    with pytest.raises(BuildError) as exc:
        _read_brief(args)
    assert "--bracket" in str(exc.value)
    assert "already exists" in str(exc.value)


@requires_deck
def test_an_existing_brief_is_read_unchanged_when_no_flag_would_reshape_it():
    from types import SimpleNamespace

    path = None
    args = SimpleNamespace(slug="zur-enchantress", commander=None, theme=None,
                           bracket=None, library=[], from_file=None, brief=None)
    path, brief, extra = _read_brief(args)
    assert brief["commander"] == "Zur the Enchanter"
    assert extra is None
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == brief


def test_no_commander_and_no_cards_says_what_to_pass():
    with pytest.raises(BuildError) as exc:
        autobuild.propose_commanders([])
    assert "--commander" in str(exc.value)


# ── the stage list is the one source of the stage count ─────────────────────

def test_the_stage_names_are_stated_once():
    """The bar's total, the report and the docstring all read `STAGES`."""
    assert len(STAGES) == 6
    assert STAGES[0] == "intent" and STAGES[-1] == "land"
    doc = autobuild.__doc__
    for i, stage in enumerate(STAGES, 1):
        assert f"[{i}/{len(STAGES)}] {stage}" in doc, stage
