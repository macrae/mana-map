"""A death payoff scoped to NONTOKEN creatures never fires on a sacrificed token.

The goldfish's sacrifice sweep converts tokens only, and it fired every death
engine on each one — so Midnight Reaper ("Whenever a nontoken creature you
control dies, ... you draw a card") read +0.14 extra cards by T8 in Edgar's
deck, on deaths its own text rules out. Found by the edgar skeptic on
2026-10-05; 13 corpus cards carry the scope (Midnight Reaper, High-Society
Hunter, Grim Haruspex, Judith, Life Insurance, ...). The payoff still fires on
the measured own-death rate, where the creature that died is a real one.
"""
import pytest

from manamap.pilot import try_swap
from manamap.pilot.goldfish_profiles import death_profile, is_death_engine

from conftest import requires_data, requires_deck


def _card(name, text):
    return {"name": name, "oracle_text": text, "type_line": "Creature — Zombie Knight"}


@pytest.mark.parametrize("name,text,scoped", [
    ("Midnight Reaper", "Whenever a nontoken creature you control dies, Midnight Reaper "
     "deals 1 damage to you and you draw a card.", True),
    ("High-Society Hunter", "Flying\nWhenever High-Society Hunter attacks, you may sacrifice "
     "another creature. If you do, put a +1/+1 counter on High-Society Hunter.\nWhenever "
     "another nontoken creature dies, draw a card.", True),
    ("Blood Artist", "Whenever Blood Artist or another creature dies, target player loses "
     "1 life and you gain 1 life.", False),
    # "nontoken" ELSEWHERE in the text is not the trigger's scope.
    ("Odd Lord", "Other nontoken Vampires you control get +1/+1.\nWhenever another "
     "creature you control dies, draw a card.", False),
])
def test_the_scope_is_read_off_the_trigger_and_only_the_trigger(name, text, scoped):
    prof = death_profile(_card(name, text))
    assert is_death_engine(prof)
    assert prof["nontoken_only"] is scoped


@pytest.mark.regression
@requires_data
@requires_deck
def test_midnight_reaper_draws_nothing_off_a_sacrificed_token():
    """The bug as found, on the deck it was found in. Reaper replaces a card the
    sweep does not need (a Plains): its modelled draw must be ~0, never the
    phantom +0.14. Re-introduce the bug (drop the `nontoken_only` skip in the
    sweep) and this reads ~+0.14."""
    r = try_swap.run("edgar-vampires", [("Plains", "Midnight Reaper")], iterations=3000)
    row = next(t for t in r["table"] if t["measure"] == "extra cards by T8")
    # The bug read +0.14 [+0.12, +0.16]; a Plains-for-a-creature swap moves the
    # mana a little, so the claim is about the interval's TOP, not a zero point.
    assert row["ci95_diff"][1] < 0.07, row
