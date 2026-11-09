"""The deck's declared engine resource, and the blind gate it feeds (2026-11-08).

The search for Meren's experience was steered by what the goldfish could read — deaths —
so Toph, Earthbending Master and the proliferate family never came up. These tests hold
the three things that fix it: the scan reads CARD TEXT, a card the bodies channel reads is
still BLIND when its engine text is not, and `try` withholds rows it cannot stand behind.
"""

import re

from conftest import simulator_source

from manamap.pilot import model_coverage as mc
from manamap.pilot import resources, try_swap
from manamap.pilot.validate_brief import validate as validate_brief

EXPERIENCE = {"name": "experience counters", "kind": "player_counter",
              "makes": [r"\byou get an? (?:\w+ )?experience counters?\b"],
              "spends": "Meren's end step"}

TOPH = {"name": "Toph, Earthbending Master", "type_line": "Legendary Creature — Human Warrior Ally",
        "oracle_text": "Landfall — Whenever a land you control enters, you get an experience "
                       "counter. Whenever you attack, earthbend X, where X is the number of "
                       "experience counters you have."}
SAGE = {"name": "Evolution Sage", "type_line": "Creature — Elf Druid",
        "oracle_text": "Landfall — Whenever a land you control enters, proliferate. (Choose any "
                       "number of permanents and/or players, then give each another counter of "
                       "each kind already there.)"}
SCALES = {"name": "Hardened Scales", "type_line": "Enchantment",
          "oracle_text": "If one or more +1/+1 counters would be put on a creature you control, "
                         "that many plus one +1/+1 counters are put on it instead."}
MEREN = {"name": "Meren of Clan Nel Toth", "type_line": "Legendary Creature — Human Shaman",
         "oracle_text": "Whenever another creature you control dies, you get an experience counter."}


# ── the declaration ─────────────────────────────────────────────────────

def test_a_good_declaration_validates():
    assert resources.validate([EXPERIENCE]) == []


def test_a_bad_declaration_is_refused_entry_by_entry():
    assert resources.validate([])                                   # empty list
    assert any("kind" in e for e in resources.validate([{**EXPERIENCE, "kind": "mana"}]))
    assert any("makes" in e for e in resources.validate([{**EXPERIENCE, "makes": []}]))
    assert any("compile" in e for e in resources.validate([{**EXPERIENCE, "makes": ["(unclosed"]}]))
    assert any("name" in e for e in resources.validate([{**EXPERIENCE, "name": " "}]))


def test_the_brief_gate_refuses_a_bad_resources_block():
    """The brief validator calls the same check, so a broken declaration fails
    `validate-brief` rather than surfacing as an empty scan."""
    doc = {"slug": "x", "commander": None, "format": "modern",
           "resources": [{**EXPERIENCE, "makes": ["(unclosed"]}]}
    errors, _ = validate_brief(doc, "x", rows={}, names=set())
    assert any("does not compile" in e for e in errors)


# ── what the scan matches ───────────────────────────────────────────────

def test_proliferate_multiplies_a_player_counter_and_scales_does_not():
    """The kind remembers the multipliers: proliferate doubles experience. Hardened
    Scales puts counters on CREATURES, and the first sweep's permanent vocabulary listed
    136 such cards for Meren — almost none of which touch experience."""
    assert resources.match(EXPERIENCE, SAGE["oracle_text"])[1]
    assert not any(resources.match(EXPERIENCE, SCALES["oracle_text"]))
    # …while for a +1/+1 deck the same card IS a multiplier
    plus = {**EXPERIENCE, "kind": "permanent_counter", "makes": [r"\+1/\+1 counter"]}
    assert resources.match(plus, SCALES["oracle_text"])[1]


def test_reminder_text_never_makes_a_hit():
    reminder_only = "Toxic 1 (Players dealt combat damage by this creature also get an " \
                    "experience counter.)"
    assert resources.match(EXPERIENCE, reminder_only) == ([], [])


# ── read or blind ───────────────────────────────────────────────────────

def test_toph_is_seen_by_coverage_and_blind_to_her_engine():
    """THE HOLE THIS CLOSES. `card_state` calls Toph seen — she is a 2/4 and `bodies`
    is always on — while the line that is the reason to play her is read by nothing."""
    assert mc.card_state(TOPH, flags={})["state"] == "seen"
    assert resources.engine_unread("x", TOPH, [EXPERIENCE], readers=set()) == ["experience counters"]


def test_the_commander_is_read_when_her_channel_is_declared():
    """The control: without it the test above passes on a gate that calls everything blind."""
    assert resources.engine_unread("x", MEREN, [EXPERIENCE], readers={MEREN["name"]}) == []
    assert resources.engine_unread("x", MEREN, [EXPERIENCE], readers=set()) == ["experience counters"]


def test_read_multipliers_matches_the_simulator():
    """`READ_MULTIPLIERS` is a claim about the turn loop. Empty while nothing in the
    simulator proliferates; the day a reader lands, this fails until the gate is told."""
    has_proliferate = bool(re.search(r"proliferat", simulator_source(), re.I))
    assert has_proliferate == bool(resources.READ_MULTIPLIERS)


# ── the gate ────────────────────────────────────────────────────────────

def _card(name, state="seen", unread=()):
    return {"side": "in", "name": name, "roles": [], "in_target": False, "state": state,
            "dark": [], "engine_unread": list(unread), "forge": None, "proof": None}


def test_blind_cards_counts_engine_text_as_well_as_state():
    cards = [_card("Toph", unread=["experience counters"]), _card("Toxic Deluge", "invisible"),
             _card("Grave Titan")]
    assert try_swap.blind_cards(cards) == {"Toph", "Toxic Deluge"}


def test_a_withheld_read_prints_no_rows():
    r = {"slug": "x", "branch": None, "swaps": [("A", "Toph")], "cards": [_card("Toph", unread=["experience counters"])],
         "forge_run": None, "warnings": [], "mana": {}, "table": [], "per_swap": [], "withheld": True,
         "verdict": {"call": "blind", "better": [], "worse": [], "trust": "none", "raise": []},
         "harness": {"iterations": 10, "seed": 1, "champion_cached": False}}
    text = try_swap.render(r)
    assert "WITHHELD — BLIND" in text and "==> BLIND" in text
    assert "interval on the difference" not in text


# ── coverage reads Meren's graveyard readers ────────────────────────────

def test_reanimation_is_a_channel_under_merens_declaration():
    """Persist read INVISIBLE for two days while the turn loop cast it every game: the
    graveyard readers act under `model_commander_recursion`, which sat in NOT_A_CHANNEL."""
    persist = {"name": "Persist", "type_line": "Sorcery",
               "oracle_text": "Return target nonlegendary creature card from your graveyard to "
                              "the battlefield with a -1/-1 counter on it."}
    assert mc.card_state(persist, {"model_commander_recursion": True})["state"] == "seen"
    assert mc.card_state(persist, {})["state"] == "dark"
    assert "model_commander_recursion" not in mc.NOT_A_CHANNEL
