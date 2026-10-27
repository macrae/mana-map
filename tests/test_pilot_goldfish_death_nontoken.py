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
def test_midnight_reaper_draws_nothing_off_a_sacrificed_token(monkeypatch):
    """The bug as found, on the deck it was found in. Reaper replaces a card the
    sweep does not need (a Swamp — the Plains went in v2.0.0): its modelled draw
    must be ~0, never the phantom +0.14. Re-introduce the bug (drop the
    `nontoken_only` skip in the sweep) and this reads ~+0.14."""
    # Since 2026-10-19 Edgar declares `model_deaths`: the table kills his real
    # creatures at a measured rate, and Reaper rightly draws off THOSE (+0.25).
    # The claim here is the sweep's TOKENS, so the channel is switched off.
    from manamap.pilot import goldfish
    from manamap.pilot.common import deck_file, load_json
    tdoc = load_json(deck_file("edgar-vampires", "goldfish_targets.json"))
    assert tdoc.pop("model_deaths", None), "Edgar no longer declares model_deaths"
    run = goldfish.run
    monkeypatch.setattr(goldfish, "run", lambda *a, **k: run(
        *a, **{**k, "_targets_doc": k.get("_targets_doc") or tdoc}))
    # `try_swap.run` reads the champion from its cache, keyed on the files, so
    # the two arms are read here directly — the same paired comparison.
    from manamap.pilot import diagnostic, net_change
    from manamap.pilot.common import load_deck_cards
    base = load_deck_cards("edgar-vampires")
    swapped, _, _, _ = try_swap.apply_swaps("edgar-vampires", None, [("Swamp", "Midnight Reaper")])
    a = diagnostic.run_on(base, "edgar-vampires", iterations=3000, seed=11, quiet=True, keep_games=True)
    b = diagnostic.run_on(swapped, "edgar-vampires", iterations=3000, seed=11, quiet=True, keep_games=True)
    row = next(t for t in net_change.compare_readings(a, b) if t["measure"] == "extra cards by T8")
    # The bug read +0.14 [+0.12, +0.16]; a land-for-a-creature swap moves the
    # mana a little, so the claim is about the interval's TOP, not a zero point.
    assert row["ci95_diff"][1] < 0.07, row
