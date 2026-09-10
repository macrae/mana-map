"""An activated ability is bought, not triggered, and this model prices it at
zero on purpose — the Forge AI mostly never buys one either (Ashnod's Altar,
0 for 59 castings; Viscera Seer, never cast), so a credit here would be a claim
the table cannot cash.

Two parsers were crediting activations anyway, found 2026-09-10 on Ingris
Stingerquill and swept:

  - the cast-token loop read the WHOLE text, so "{T}: Create a 2/2 Vampire"
    (Bloodline Keeper) and "−7: Create four 4/4 Dragons" (Sarkhan, Fireblood)
    were free bodies the turn the card was cast — 399 corpus cards, twelve in
    the fleet, four of ur-dragon's Dragons among them;
  - the attack-trigger window ran 220 characters past the trigger into the
    next ability, so Ignoble Hierarch's exalted read "{T}: Add {B}" as ATTACK
    MANA and Ingris's "{4}: Create a 2/2 … Cadet" as a token per attack.
"""

import csv

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot import goldfish
from conftest import requires_data


def _p(text, type_line="Creature — Vampire", power="3", name="a card"):
    return goldfish.combat_profile({"name": name, "oracle_text": text, "type_line": type_line, "power": power})


def test_a_token_behind_a_cost_is_not_a_cast_token():
    assert _p("Flying\n{T}: Create a 2/2 black Vampire creature token with flying.")["token_bodies"] == 0
    assert _p("+1: Sarkhan gets +1/+1.\n−7: Create four 4/4 red Dragon creature tokens with flying.",
              "Legendary Planeswalker — Sarkhan", None)["token_bodies"] == 0
    assert _p("{1}{R}, {T}: Create a 0/1 red Kobold creature token.", "Land", None)["token_bodies"] == 0
    # The triggered and the bought token on ONE card: only the trigger counts.
    assert _p("When this enters, create a 1/1 white Vampire creature token with lifelink.\n"
              "{3}{W}: Create a 1/1 white Soldier creature token.", "Enchantment", None)["token_bodies"] == 1
    # Positive controls — a trigger, a cast effect, a saga chapter.
    assert _p("When Grave Titan enters or attacks, create two 2/2 black Zombie creature tokens.")["token_bodies"] == 2
    assert _p("At the beginning of your upkeep, you lose 1 life and create a 1/1 black Faerie Rogue creature token with flying.",
              "Enchantment", None)["token_bodies"] == 1
    assert _p("III — Create a 3/3 green Dinosaur creature token.", "Enchantment — Saga", None)["token_bodies"] == 1


def test_the_attack_window_stops_at_the_next_ability():
    hierarch = _p("Exalted (Whenever a creature you control attacks alone, that creature gets +1/+1 until end of turn.)\n"
                  "{T}: Add {B}, {R}, or {G}.", "Creature — Goblin Shaman", "1", name="Ignoble Hierarch")
    assert hierarch["attack_mana"] == 0, "the {T}: Add on the next line read as attack mana"
    assert hierarch["unreadable"] == "Ignoble Hierarch", "exalted is not priced, and that must be said"
    # A trigger INSIDE an activation's granted text is still a trigger (Den of the Bugbear).
    den = _p("{T}: Add {R}.\n{3}{R}: Until end of turn, this land becomes a 3/2 red Goblin creature with "
             "\"Whenever this creature attacks, create a 1/1 red Goblin creature token that's tapped and attacking.\"",
             "Land", None)
    assert den["attack_token_bodies"] == 1


@requires_data
def test_the_corpus_sweep_is_locked():
    """1,994 corpus cards keep a cast/ETB token credit after the bound; 399
    lost one. A change to either parser moves this on purpose."""
    credited = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            credited += bool(goldfish.combat_profile({"oracle_text": row["oracle_text"],
                                                      "type_line": row["type_line"],
                                                      "power": row.get("power")})["token_bodies"])
    assert credited == 1994, credited
