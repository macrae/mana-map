"""scenario-facts: the deterministic brief that makes recalled figures unnecessary.

Every assertion here corresponds to an error that reached an agent brief during
the Vol. 008 session and was refused by the agent rather than written down.
"""

import json

import pytest

from conftest import requires_deck
from manamap.pilot import scenario_facts as sf


# ── Board parsing ────────────────────────────────────────────────────────


def test_tokens_are_bodies_not_furniture():
    """In a sacrifice deck the tokens ARE the bodies.

    The first cut of this module filtered tokens out as "not cards", which made
    yawgmoth's 002 and 003 look identical and erased the extra Human Soldier —
    the whole reason their matching totals are not comparable.
    """
    split = sf.board_bodies([
        "Yawgmoth, Thran Physician (2/4, untapped, no counters)",
        "Nest of Scarabs (enchantment)",
        "Insect token B (1/1 black Insect, no counters)",
        "Human Soldier token (1/1 white, no counters)",
        "Swamp (untapped)",
    ])
    assert split["creature_bodies"] == [
        "Yawgmoth, Thran Physician", "Insect token B", "Human Soldier token"]
    assert split["other_permanents"] == ["Nest of Scarabs"]
    assert split["lands"] == ["Swamp"]


def test_an_already_paid_cost_is_not_on_the_battlefield():
    """The annotated body is LISTED and NOT available — it changes every bound."""
    split = sf.board_bodies([
        "Insect token A (1/1 black Insect) — already sacrificed to pay the cost of "
        "the ability now on the stack",
        "Insect token B (1/1 black Insect, no counters)",
    ])
    assert split["spent_paying_a_cost"] == ["Insect token A"]
    assert split["creature_bodies"] == ["Insect token B"]


# ── The arithmetic that reached a brief wrong ────────────────────────────


def test_per_opponent_and_pod_total_are_stated_separately():
    """"28 from each opponent" was 7 per opponent and 28 across four seats."""
    opps = [{"seat": f"opponent_{c}", "life": 40} for c in "abcd"]
    facts = sf.drain_arithmetic(opps)
    assert facts["opponents"] == 4
    assert facts["opposing_life_total"] == 160
    assert "per seat" in facts["note"] and "across the pod" in facts["note"]


def test_both_board_shapes_are_read():
    """Seven decks use `opponents: [...]`; yawgmoth uses `opponent_a..d`."""
    listed = sf.opponents_of({"board": {"opponents": [{"name": "P2", "life": 33}]}})
    assert listed == [{"seat": "P2", "life": 33}]

    keyed = sf.opponents_of({
        "board": {"you": [], "opponent_a": ["no permanents"], "opponent_b": ["x"]},
        "extras": {"life_totals": {"you": 39, "opponent_a": 40, "opponent_b": 33}},
    })
    assert [o["seat"] for o in keyed] == ["opponent_a", "opponent_b"]
    assert [o["life"] for o in keyed] == [40, 33]


# ── Sibling comparability ────────────────────────────────────────────────


def test_same_body_count_still_reports_what_differs_both_ways():
    """Two boards can match on count and still answer different questions.

    A one-directional diff hid exactly this: 002 reaches three bodies with a Human
    Soldier where 003 uses Zulaport, because Bastion is an enchantment and cannot
    be a body itself. Reconciling that by hand cost two rounds of stack 008.
    """
    scenarios = {
        "002": {"board": {"you": ["Yawgmoth (2/4)", "Insect token B (1/1)",
                                  "Human Soldier token (1/1 white)",
                                  "Bastion of Remembrance (enchantment)"]}},
        "003": {"board": {"you": ["Yawgmoth (2/4)", "Insect token B (1/1)",
                                  "Zulaport Cutthroat (1/1)"]}},
    }
    [sib] = sf.comparable_siblings("002", scenarios)
    assert sib["stack"] == "003"
    assert sib["same_body_count"] is True
    assert sib["only_on_that_board"] == ["Zulaport Cutthroat"]
    assert sib["only_on_this_board"] == ["Human Soldier token"]


# ── Membership ───────────────────────────────────────────────────────────


def test_membership_names_what_left_the_deck():
    got = sf.membership(["Nest of Scarabs", "Ad Nauseam"], {"Nest of Scarabs"})
    assert got["in_the_deck"] == ["Nest of Scarabs"]
    assert got["NOT_IN_THE_DECK"] == ["Ad Nauseam"]


def test_tokens_are_not_reported_as_missing_cards():
    """A warning that fires on every scenario is one an agent learns to skip.

    Tokens are never in a decklist, so membership-checking them made the
    "not in the deck" note fire with nothing wrong — burying the one case that
    matters, a real card of yours that has left the 99.
    """
    got = sf.membership(["Insect token A", "Human Soldier token", "Ad Nauseam"],
                        {"Nest of Scarabs"})
    assert got["NOT_IN_THE_DECK"] == ["Ad Nauseam"]
    assert got["tokens_not_checked"] == ["Human Soldier token", "Insect token A"]


# ── Against the real committed decks ─────────────────────────────────────


@requires_deck
def test_runs_on_every_committed_deck_with_stacks():
    from manamap.config import DECKS_DIR
    ran = 0
    for deck in sorted(DECKS_DIR.iterdir()):
        if not (deck / "stacks").is_dir() or not (deck / "cards.json").exists():
            continue
        facts = sf.analyze(deck.name)
        assert facts["slug"] == deck.name
        for sid, s in facts["stacks"].items():
            assert "drain_arithmetic" in s and "card_membership" in s
        ran += 1
    assert ran >= 1
