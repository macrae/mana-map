"""deck-audit: the cited axis table and the engine-activation read.

The load-bearing test in this file is `test_every_axis_quote_is_verbatim`. The
whole premise of the module is that an agent can cite a target instead of
inventing one, and a quote that has drifted out of `strategy.md` is a citation
the validator will reject at the far end of a spawn. Failing here is cheap;
failing there costs an agent round.
"""

import re

import pytest

from manamap.config import (
    DECK_ARCHETYPE_BUDGETS,
    DECK_ARCHETYPE_BUDGET_CITATION,
    DECK_AXIS_TARGETS,
    ENGINE_REDUNDANCY_CITATION,
)
from manamap.pilot import deck_audit
from manamap.pilot.common import load_strategy_db
from manamap.pilot.manabase import DECK_SIZE_AFTER_COMMANDER, hypergeometric_at_least

from conftest import requires_deck, requires_roles, requires_strategy


def _ws(text):
    """The same whitespace normalization validate_stack.validate_citations uses."""
    return re.sub(r"\s+", " ", text).strip()


def _card(name, oracle="", type_line="Creature — Human", quantity=1, **kw):
    return {"name": name, "oracle_text": oracle, "type_line": type_line,
            "quantity": quantity, **kw}


# ── The citation contract ────────────────────────────────────────────────

@requires_strategy
def test_every_axis_quote_is_verbatim():
    sections, _, _ = load_strategy_db()
    for axis, spec in DECK_AXIS_TARGETS.items():
        assert spec["source"] in sections, f"{axis} cites a section that does not exist"
        body = _ws(sections[spec["source"]]["text"])
        assert _ws(spec["quote"]) in body, (
            f"{axis}: quote is not verbatim text of {spec['source']} — "
            f"strategy.md drifted, or the quote was retyped"
        )


@requires_strategy
def test_archetype_and_engine_citations_are_verbatim():
    sections, _, _ = load_strategy_db()
    for citation in (DECK_ARCHETYPE_BUDGET_CITATION, ENGINE_REDUNDANCY_CITATION):
        body = _ws(sections[citation["rule"]]["text"])
        assert _ws(citation["quote"]) in body


def test_every_archetype_override_names_a_real_axis():
    for archetype, overrides in DECK_ARCHETYPE_BUDGETS.items():
        for axis in overrides:
            assert axis in DECK_AXIS_TARGETS, (
                f"{archetype} overrides '{axis}', which is not an axis — an "
                f"override on a name nothing measures is silently inert"
            )


def test_computed_odds_reproduce_the_cited_figures():
    """The engine block prices redundancy itself; the quote is corroboration.

    If these ever disagree the module is quoting a number it does not produce,
    which is exactly the failure the citation contract exists to prevent.
    """
    quoted = {5: 0.31, 7: 0.41, 10: 0.54}
    for copies, cited in quoted.items():
        computed = hypergeometric_at_least(copies, 7, 1, DECK_SIZE_AFTER_COMMANDER)
        assert abs(computed - cited) < 0.01, f"{copies} copies: {computed} vs {cited}"


# ── Verdicts ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,low,high,expected", [
    (5, 8, None, "under"),
    (8, 8, None, "at"),
    (9, 8, None, "at"),
    (4, 1, 3, "over"),
    (2, 1, 3, "at"),
    (0, 1, 3, "under"),
    (4, None, None, "informational"),
])
def test_verdict(value, low, high, expected):
    assert deck_audit._verdict(value, low, high) == expected


# ── Counting ─────────────────────────────────────────────────────────────

def test_count_copies_counts_copies_not_entries():
    cards = [_card("Swamp", type_line="Basic Land — Swamp", quantity=11)]
    roles = {"Swamp": ["land:basic"]}
    copies, names = deck_audit._count_copies(cards, roles, ("land:basic",))
    assert copies == 11
    assert names == ["Swamp"]


def test_count_copies_counts_a_card_once_per_axis():
    """Two matching roles on one card is one card, not two.

    deck-facts' histogram sums role instances, which double-counts here: a card
    that is both draw:burst and draw:engine would read as two cards of draw.
    """
    cards = [_card("Two Jobs")]
    roles = {"Two Jobs": ["draw:burst", "draw:engine"]}
    copies, _ = deck_audit._count_copies(cards, roles, deck_audit.AXIS_ROLES["card-advantage"])
    assert copies == 1


# ── Interaction breadth ──────────────────────────────────────────────────

def test_catch_all_removal_answers_three_classes():
    """"Return target nonland permanent" names no type and answers three."""
    cards = [_card("Cyclonic Rift", "Return target nonland permanent you don't "
                                    "control to its owner's hand.",
                   type_line="Instant")]
    roles = {"Cyclonic Rift": ["removal:bounce"]}
    breadth = deck_audit._interaction_breadth(cards, roles)
    assert breadth["creature"] == ["Cyclonic Rift"]
    assert breadth["artifact"] == ["Cyclonic Rift"]
    assert breadth["enchantment"] == ["Cyclonic Rift"]
    assert breadth["land"] == [], "'nonland permanent' must not answer land"


def test_destroy_target_permanent_answers_land_too():
    cards = [_card("Beast Within", "Destroy target permanent. Its controller "
                                   "creates a 3/3 green Beast creature token.",
                   type_line="Instant")]
    roles = {"Beast Within": ["removal:spot"]}
    breadth = deck_audit._interaction_breadth(cards, roles)
    assert breadth["land"] == ["Beast Within"]


def test_cannot_be_regenerated_is_not_protection():
    """A clause that REMOVES protection must not read as granting it."""
    assert not deck_audit._AXIS_PROBES["protection"].search("It can't be regenerated.")


def test_toward_does_not_match_ward():
    assert not deck_audit._AXIS_PROBES["protection"].search(
        "Each {B} in the mana costs of permanents you control counts toward your "
        "devotion to black.")


def test_self_exile_is_not_graveyard_hate():
    """Necropotence exiles its own discards; delve, escape and flashback likewise."""
    cards = [_card("Necropotence", "Whenever you discard a card, exile that card "
                                   "from your graveyard.", type_line="Enchantment")]
    roles = {"Necropotence": ["stax"]}
    assert deck_audit._interaction_breadth(cards, roles)["graveyard"] == []


def test_x_sweeper_answers_creatures():
    """`-X/-X` is not `-\\d+/-\\d+`; Toxic Deluge read as answering nothing."""
    cards = [_card("Toxic Deluge", "All creatures get -X/-X until end of turn.",
                   type_line="Sorcery")]
    roles = {"Toxic Deluge": ["removal:sweeper"]}
    assert deck_audit._interaction_breadth(cards, roles)["creature"] == ["Toxic Deluge"]


def test_a_counterspell_is_not_counted_for_breadth():
    """The corpus warns a counter answers a permanent only on the way in.

    Under-counting breadth reports a gap that may not exist; over-counting hides
    one that does, so the conservative direction is the deliberate one.
    """
    cards = [_card("Counterspell", "Counter target spell.", type_line="Instant")]
    roles = {"Counterspell": ["counterspell"]}
    breadth = deck_audit._interaction_breadth(cards, roles)
    assert all(names == [] for names in breadth.values())


# ── Probes ───────────────────────────────────────────────────────────────

def test_probe_names_a_draw_engine_the_taxonomy_missed():
    """Yawgmoth draws a card per activation and is filed as removal:debuff."""
    cards = [
        _card("Yawgmoth, Thran Physician",
              "Pay 1 life, Sacrifice another creature: Put a -1/-1 counter on up "
              "to one target creature and draw a card."),
        _card("Skullclamp", "Whenever equipped creature dies, draw two cards.",
              type_line="Artifact — Equipment"),
    ]
    roles = {"Yawgmoth, Thran Physician": ["removal:debuff"],
             "Skullclamp": ["draw:engine"]}
    probed = deck_audit._probe_uncounted("card-advantage", cards, roles, ["Skullclamp"])
    assert probed == ["Yawgmoth, Thran Physician"], "counted cards must not repeat"


def test_ramp_probe_skips_lands():
    """A land that taps for mana is the mana base, not acceleration."""
    cards = [_card("Castle Locthwain", "{T}: Add {B}.",
                   type_line="Land"),
             _card("Sol Ring", "{T}: Add {C}{C}.", type_line="Artifact")]
    probed = deck_audit._probe_uncounted("ramp", cards, {}, [])
    assert probed == ["Sol Ring"]


def test_probe_is_silent_for_axes_without_one():
    assert deck_audit._probe_uncounted("power", [_card("X", "draw a card")], {}, []) == []


# ── Engine components ────────────────────────────────────────────────────

def test_component_size_ignores_cards_not_in_the_deck():
    """Redundancy is what the shuffler can find, not what the target file remembers."""
    comp = deck_audit._component(
        {"any_of": ["In Deck", "Ghost"]}, {"In Deck"}, {"In Deck": ["sac-outlet"]})
    assert comp["size"] == 1
    assert comp["cards"] == ["In Deck"]
    assert comp["not_in_deck"] == ["Ghost"]


def test_component_shared_role_needs_two_members():
    """One card's roles describe the card, not the group's job."""
    single = deck_audit._component({"any_of": ["A"]}, {"A"}, {"A": ["sac-outlet"]})
    assert single["shared_roles"] == []
    pair = deck_audit._component({"any_of": ["A", "B"]}, {"A", "B"},
                                 {"A": ["sac-outlet"], "B": ["sac-outlet"]})
    assert pair["shared_roles"] == ["sac-outlet"]


def test_component_falls_back_to_a_shared_axis():
    """Sol Ring is ramp:rock and Dark Ritual is ramp:ritual — one job, two names."""
    comp = deck_audit._component(
        {"any_of": ["Sol Ring", "Dark Ritual"]}, {"Sol Ring", "Dark Ritual"},
        {"Sol Ring": ["ramp:rock"], "Dark Ritual": ["ramp:ritual"]})
    assert comp["shared_roles"] == []
    assert comp["shared_axes"] == ["ramp"]


def test_component_odds_shrink_with_the_group():
    one = deck_audit._component({"any_of": ["A"]}, {"A"}, {})
    five = deck_audit._component({"any_of": list("ABCDE")}, set("ABCDE"), {})
    assert one["odds"]["opening_seven"] < five["odds"]["opening_seven"]
    assert one["odds"]["opening_seven"] == pytest.approx(0.071, abs=0.001)
    assert one["thin"] and not five["thin"]


def test_singleton_component_gets_no_role_route():
    """The bug this guards: a component holding only Blowfly Infestation returned
    Massacre Wurm and Dismember, because Blowfly is filed as removal:debuff."""
    pool = {"Massacre Wurm": {"color_identity": set("B"), "legal": True,
                              "edhrec_rank": 1, "game_changer": False,
                              "type_line": "Creature", "cmc": 6.0, "mana_cost": ""}}
    comp = deck_audit._component({"any_of": ["Blowfly Infestation"]},
                                 {"Blowfly Infestation"},
                                 {"Blowfly Infestation": ["removal:debuff"]})
    closers = deck_audit._closers(comp, [], {"Blowfly Infestation"}, set("B"), pool,
                                  {"Massacre Wurm": ["removal:debuff"]}, None)
    assert closers["by_role"] == []
    assert closers["role_signature"] is None
    assert "combo route" in closers["note"]


def test_closers_report_unavailable_without_a_pool():
    comp = deck_audit._component({"any_of": ["A", "B"]}, {"A", "B"},
                                 {"A": ["sac-outlet"], "B": ["sac-outlet"]})
    closers = deck_audit._closers(comp, [], {"A", "B"}, set("B"), None, {}, None)
    assert closers["available"] is False
    assert "pipeline run" in closers["reason"]


def test_closers_exclude_cards_already_in_the_deck():
    pool = {name: {"color_identity": set("B"), "legal": True, "edhrec_rank": i,
                   "game_changer": False, "type_line": "Creature", "cmc": 1.0,
                   "mana_cost": ""}
            for i, name in enumerate(("Already In", "Available"))}
    roles = {"A": ["sac-outlet"], "B": ["sac-outlet"],
             "Already In": ["sac-outlet"], "Available": ["sac-outlet"]}
    comp = deck_audit._component({"any_of": ["A", "B"]}, {"A", "B"}, roles)
    closers = deck_audit._closers(comp, [], {"A", "B", "Already In"}, set("B"),
                                  pool, roles, None)
    assert [c["name"] for c in closers["by_role"]] == ["Available"]


def test_closers_respect_colour_identity():
    pool = {"Off Colour": {"color_identity": set("G"), "legal": True,
                           "edhrec_rank": 1, "game_changer": False,
                           "type_line": "Creature", "cmc": 1.0, "mana_cost": ""}}
    roles = {"A": ["sac-outlet"], "B": ["sac-outlet"], "Off Colour": ["sac-outlet"]}
    comp = deck_audit._component({"any_of": ["A", "B"]}, {"A", "B"}, roles)
    closers = deck_audit._closers(comp, [], {"A", "B"}, set("B"), pool, roles, None)
    assert closers["by_role"] == []


# ── Archetype ────────────────────────────────────────────────────────────

def test_archetype_override_wins_over_the_frame():
    detected = deck_audit.detect_archetype("yawgmoth-swarm", override="aggro")
    assert detected == {"archetype": "aggro", "source": "--archetype"}


def test_unknown_archetype_yields_no_overrides():
    overrides, citation = deck_audit.archetype_overrides(None)
    assert overrides == {} and citation is None
    overrides, citation = deck_audit.archetype_overrides("tribal-goats")
    assert overrides == {} and citation is None


def test_known_archetype_carries_its_citation():
    overrides, citation = deck_audit.archetype_overrides("control")
    assert overrides["sweepers"] == (5, 7)
    assert citation["rule"] == "strategy:deckbuilding.archetype-selection"


# ── Integration ──────────────────────────────────────────────────────────

@requires_deck
@requires_roles
def test_audit_is_deterministic():
    import json
    first = json.dumps(deck_audit.analyze("goblin-storm"), sort_keys=True)
    second = json.dumps(deck_audit.analyze("goblin-storm"), sort_keys=True)
    assert first == second


@requires_deck
@requires_roles
def test_audit_emits_every_axis_with_a_source():
    audit = deck_audit.analyze("goblin-storm")
    names = [a["axis"] for a in audit["axes"]]
    assert len(names) == len(set(names)), "an axis emitted twice"
    for axis in audit["axes"]:
        assert axis["verdict"] in {"under", "at", "over", "informational"}
        if axis["target"] is not None:
            assert axis["target"]["source"].startswith("strategy:")
            assert axis["target"]["quote"]
        assert axis["measured"]["how"], f"{axis['axis']} does not say how it measured"


@requires_deck
@requires_roles
def test_audit_reports_freshness_against_the_current_decklist():
    audit = deck_audit.analyze("goblin-storm")
    fresh = audit["freshness"]
    assert fresh["decklist_sha256"]
    assert "goldfish_metrics.json" in fresh["artifacts"]
    # Whatever the answer, it must be a real comparison rather than an assumption.
    for name, state in fresh["artifacts"].items():
        assert set(state) >= {"present"}, name
