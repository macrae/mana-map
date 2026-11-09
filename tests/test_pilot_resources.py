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

def test_a_seen_body_can_be_blind_to_its_engine():
    """THE HOLE THIS CLOSES. `card_state` called Toph seen for being a 2/4 while her
    landfall experience was read by nothing (fixed 2026-11-08). Azula is that case today:
    a body `bodies` reads, and a discard-for-experience line the turn loop never applies."""
    azula = {"name": "Azula, Ruthless Firebender", "type_line": "Legendary Creature — Human Noble",
             "oracle_text": "Whenever Azula attacks, you may discard a card. Then you get an "
                            "experience counter for each player who discarded a card this turn."}
    assert mc.card_state(azula, flags={"model_commander_recursion": True})["state"] == "seen"
    per_player = {**EXPERIENCE, "makes": EXPERIENCE["makes"] + [r"\bexperience counters? for each\b"]}
    assert resources.engine_unread("x", azula, [per_player], readers={"Meren"}) == ["experience counters"]
    # …and Toph, now read, is not
    assert resources.engine_unread("x", TOPH, [EXPERIENCE], readers={"Meren"}) == []


def test_the_commander_is_read_when_her_channel_is_declared():
    """The control: without it the test above passes on a gate that calls everything blind."""
    assert resources.engine_unread("x", MEREN, [EXPERIENCE], readers={MEREN["name"]}) == []
    assert resources.engine_unread("x", MEREN, [EXPERIENCE], readers=set()) == ["experience counters"]


def test_a_read_experience_card_is_read_under_the_commander_channel():
    """Constrictor's replacement and Sage's landfall proliferate are applied by the turn loop
    (2026-11-08), so under Meren's channel they read READ; without it, BLIND."""
    con = {"name": "Winding Constrictor", "type_line": "Creature — Snake",
           "oracle_text": "If you would get one or more counters, you get that many plus one of "
                          "each of those kinds of counters instead."}
    assert resources.engine_unread("x", SAGE, [EXPERIENCE], readers={"Meren"}) == []
    assert resources.engine_unread("x", SAGE, [EXPERIENCE], readers=set()) == ["experience counters"]
    assert resources.match(EXPERIENCE, con["oracle_text"])[1]
    assert resources.engine_unread("x", con, [EXPERIENCE], readers={"Meren"}) == []


def test_an_activated_proliferate_stays_blind():
    yawg = {"name": "Yawgmoth, Thran Physician", "type_line": "Legendary Creature — Human Cleric",
            "oracle_text": "{B}{B}, Discard a card: Proliferate."}
    assert resources.engine_unread("x", yawg, [EXPERIENCE], readers={"Meren"}) == ["experience counters"]


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
