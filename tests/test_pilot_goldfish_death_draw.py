"""Death-triggered draw and drain, read where the text puts them (2026-10-19).

THE GAP AS FOUND. Edgar's mean-v1 branch added Morbid Opportunist ("Whenever
one or more other creatures DIE, draw a card. This ability triggers only once
each turn") and Vraan, Executioner Thane, and the paired net-change read the
package as a LOSS of half a card by turn eight: `_DEATH_TRIGGER_RE` wanted the
singular "dies", so both read as nothing, while two draw engines the model DID
read (Cosmos Elixir, Crossway Troublemakers) left. The pilot's call — "measure
twice, cut once" — was to teach the instrument before cutting a card.

Four things changed, each proven here by re-introducing the bug:
  1. the plural "die" is a death trigger (19 corpus cards);
  2. "triggers only once each turn" caps the trigger, across every death path;
  3. a payload behind a PRICE or a CONDITION ("you may pay 2 life. If you do,
     draw") is NAMED, never credited — a bare "you may" is a free choice;
  4. "with mana value 3 or less" is a readable arrival qualifier (Tocasia's
     Welcome), honoured at the one door with a token's mana value of 0.
And the draw series is appended at the END of the turn, like the sacrifice
series since #34, so a draw made in the sacrifice sweep lands on its own turn.
"""

import pytest

from manamap.pilot import card_pool, goldfish, try_swap
from manamap.pilot.goldfish_profiles import _DEATH_TRIGGER_RE, death_profile, draw_profile

from conftest import assert_corpus_count, requires_data, requires_deck, simulator_source


def _card(name, text, type_line="Creature — Vampire", cmc=3):
    return {"name": name, "oracle_text": text, "type_line": type_line, "cmc": cmc}


MORBID = ("Whenever one or more other creatures die, draw a card. This ability "
          "triggers only once each turn.")
VRAAN = ("Whenever one or more other creatures you control die, each opponent loses "
         "2 life and you gain 2 life. This ability triggers only once each turn.")
VEIN_RIPPER = ("Flying\nWard—Sacrifice a creature.\nWhenever a creature dies, target "
               "opponent loses 2 life and you gain 2 life.")
EREBOS = ("Whenever another creature you control dies, you may pay 2 life. If you do, "
          "draw a card.")
SHENANIGANS = ("Whenever another Goblin you control is put into a graveyard from the "
               "battlefield, you may have this enchantment deal 1 damage to target player "
               "or planeswalker.")
HARVESTER = "Whenever another nontoken creature dies, you may draw a card."
TOCASIA = ("Whenever one or more creatures you control with mana value 3 or less enter, "
           "draw a card. This ability triggers only once each turn.")


# ── 1. The plural ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,text,field,value", [
    ("Morbid Opportunist", MORBID, "death_draw", 1),
    ("Vraan, Executioner Thane", VRAAN, "death_drain", 2),
    ("Vein Ripper", VEIN_RIPPER, "death_drain", 2),
])
def test_the_plural_and_the_target_opponent_idiom_read(name, text, field, value):
    """Re-introduce the bug (drop `die\\b` from the trigger, or "target opponent"
    from the drain pattern) and every row here reads 0 and `unreadable`."""
    p = death_profile(_card(name, text))
    assert p[field] == value, p
    assert goldfish.is_death_engine(p) and p["unreadable"] is None


def test_the_plural_is_a_creature_dying_not_a_self_death():
    """"When this creature dies" is a self-death the sweep never sees; the
    plural must not drag it in."""
    p = death_profile(_card("Solemn", "When this creature dies, draw a card.", "Artifact Creature — Golem"))
    assert not goldfish.is_death_engine(p)


# ── 2. Once each turn ──────────────────────────────────────────────────────

def test_once_each_turn_is_read_on_the_trigger():
    assert death_profile(_card("Morbid Opportunist", MORBID))["death_once"] is True
    assert death_profile(_card("Vraan", VRAAN))["death_once"] is True
    assert death_profile(_card("Vein Ripper", VEIN_RIPPER))["death_once"] is False


def test_the_once_cap_is_consulted_on_every_death_path():
    """A flag the model sets is a claim the model must ACT ON. The cap is
    `death_once_used`, reset each turn, consulted in the sacrifice sweep and in
    `_deaths_after_combat`, which the one-turn copies and the death rate call."""
    src = simulator_source()
    assert src.count("death_once_used") >= 5
    assert "_deaths_after_combat(_n_died, True)" in src, "the one-turn copies"
    assert "_deaths_after_combat(1, bool(_died[4]))" in src, "the death rate, with the body's token-ness"


# ── 3. A price or a condition is a gate; a bare "you may" is not ───────────

def test_a_priced_payload_is_named_not_credited():
    """Erebos was credited a free draw per death for a month; 15 corpus cards
    carried the same over-credit. Re-introduce the bug (drop the conditional
    guard) and `death_draw` reads 1 here."""
    p = death_profile(_card("Erebos, Bleak-Hearted", EREBOS))
    assert p["death_draw"] == 0 and p["unreadable"] == "Erebos, Bleak-Hearted"


def test_a_free_choice_still_reads():
    assert death_profile(_card("Boggart Shenanigans", SHENANIGANS, "Enchantment"))["death_damage"] == 1
    hv = death_profile(_card("Harvester of Souls", HARVESTER))
    assert hv["death_draw"] == 1 and hv["nontoken_only"] is True


# ── 4. The mana-value qualifier ────────────────────────────────────────────

def test_tocasias_welcome_reads_with_its_gate():
    d = draw_profile(_card("Tocasia's Welcome", TOCASIA, "Enchantment"))
    assert d["arrival_draw"] == 1 and d["arrival_draw_once"] and d["arrival_mv_max"] == 3
    assert "arrival_mv_max" in simulator_source(), "the gate is acted on at the door"
    assert "mv=card[\"cmc\"]" in simulator_source(), "a cast body passes its cost"


# ── 5. The series lands on its own turn ────────────────────────────────────

def test_the_draw_series_is_appended_at_the_end_of_the_turn():
    """The sacrifice sweep runs inside combat; the draw series was appended
    before it, so a death draw landed a turn late and turn ten's was lost.
    Re-introduce the bug (move the append above the combat step) and the
    order below flips."""
    src = simulator_source()
    assert src.index("sacrifices_by_turn.append(sacrifices)") < src.index("drawn_extra_by_turn.append(drawn_extra)")
    assert src.index("# ── Combat step") < src.index("drawn_extra_by_turn.append(drawn_extra)")


# ── 6. The corpus, locked ──────────────────────────────────────────────────

@requires_data
def test_the_sweeps_are_locked():
    """Widening a pattern needs a corpus sweep in the same commit: the death
    trigger, its plural share, what it credits, what it caps, and the one
    arrival engine with a mana-value gate. Bands, not pins (`assert_corpus_count`)."""
    import re
    frame = card_pool.load_frame()
    triggers = plural = draws = drains = once = mv = 0
    for _, r in frame.iterrows():
        text = r.get("oracle_text") or ""
        if not isinstance(text, str) or not text:
            continue
        card = {"name": r["name"], "oracle_text": text, "type_line": r.get("type_line") or ""}
        m = _DEATH_TRIGGER_RE.search(text)
        if m:
            triggers += 1
            plural += bool(re.search(r"\bdie\b", m.group(0), re.I))
            p = death_profile(card)
            draws += bool(p["death_draw"])
            drains += bool(p["death_drain"])
            once += p["death_once"]
        d = draw_profile(card)
        mv += d["arrival_mv_max"] is not None and bool(d["arrival_draw"])
    assert_corpus_count(triggers, 368, "death triggers")
    assert_corpus_count(plural, 19, "plural 'die' death triggers")
    assert_corpus_count(draws, 33, "death triggers credited a draw")
    assert_corpus_count(drains, 40, "death triggers credited a drain")
    assert_corpus_count(once, 13, "death triggers capped once each turn")
    assert mv == 1, "Tocasia's Welcome is the one mana-value-gated arrival engine"


# ── 7. On the deck it was found in ─────────────────────────────────────────

@pytest.mark.regression
@requires_data
@requires_deck
def test_morbid_opportunist_now_draws_on_edgars_sacrifices():
    """Before the plural read, this swap measured as nothing for draw. The
    sweep converts about 1.5 tokens a game by turn ten on edgar, and the cap
    holds the engine to one draw a turn, so the lift is small and POSITIVE —
    the claim is the interval's bottom, not a size."""
    r = try_swap.run("edgar-vampires", [("Swamp", "Morbid Opportunist")], iterations=3000)
    row = next(t for t in r["table"] if t["measure"] == "extra cards by T8")
    assert row["ci95_diff"][0] > 0.0, row
