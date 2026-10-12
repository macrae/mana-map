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

import pytest
import csv

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot import goldfish
from conftest import assert_corpus_count, requires_data


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


@pytest.mark.slow
@requires_data
def test_the_corpus_sweep_is_locked():
    """1,994 corpus cards keep a cast/ETB token credit after the bound; 399
    lost one.

    A change to either parser moves this on purpose — but so does a SET
    RELEASE, which the original note did not say, so the failure message points
    at code that did not change. Corpus at 34,890 rows (2026-08-12); if the
    count moved and the corpus did not, a parser did.
    """
    credited = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            credited += bool(goldfish.combat_profile({"oracle_text": row["oracle_text"],
                                                      "type_line": row["type_line"],
                                                      "power": row.get("power")})["token_bodies"])
    # 2026-10-02 Reality Fracture refresh, 34,955 rows: 1994 + 19 new − 21 Alchemy
    # `A-` cards Scryfall dropped = 1992; then the bracketed-loyalty fix took
    # four phantom bodies off (Kiora of Salt and Sand, Way of the Wildspeaker,
    # Way of the Healer, Elspeth's Talent) = 1988. Way of the Deathbringer is the
    # one bracketed false positive left, named in `_ACTIVATION_COST_RE`'s comment.
    # Then +7 when `prepare` joined MULTI_FACE_LAYOUTS and 70 cards gained their
    # spell face: Fatehold Chronologist, Semester Foreseer, Prudent Fateseer, Campus
    # Composer, Lluwen, Eccentric Pestfinder and Strife Scholar each make a token
    # there = 1995. Pestfinder's copy is gated on lifegain, which the credit ignores.
    # Then −35 (2026-10-11) when a NON-MANA cost joined `_ACTIVATION_COST_RE`:
    # every Thallid's "Remove three spore counters:", Ominous Seas' foreshadow
    # counters, Voice of the Woods' "Tap five untapped Elves", Zektar Shrine's
    # "Sacrifice:", Grove of the Guardian's cost too long for the {…} bound. The
    # sweep moved 36 oracle cards, every one a body REMOVED, none gained = 1960.
    assert_corpus_count(credited, 1960, "cast/ETB token credits")


def test_a_granted_bracketed_loyalty_ability_is_not_a_free_body():
    """Kiora of Salt and Sand GIVES planeswalkers `"[−8]: Create an 8/8 …"`. The
    token is bought with loyalty, so it is not a cast/ETB body — and before
    2026-10-02 the bracket hid the cost and credited an 8/8 for free."""
    kiora = {"oracle_text": 'Planeswalkers you control have "[−8]: Create an 8/8 '
                            'blue Leviathan creature token."',
             "type_line": "Legendary Creature — Merfolk Noble", "power": "3"}
    assert goldfish.combat_profile(kiora)["token_bodies"] == 0
    # The unbracketed printed form was already handled, and still is.
    printed = dict(kiora, oracle_text="−8: Create an 8/8 blue Leviathan creature token.")
    assert goldfish.combat_profile(printed)["token_bodies"] == 0


def test_a_non_mana_cost_is_an_activation_too():
    """Ominous Seas: "Remove eight foreshadow counters from this enchantment:
    Create an 8/8 blue Kraken". The cost is a verb, not a symbol, and before
    2026-10-11 the Kraken was credited the turn the two-mana enchantment was
    cast. Re-introduce it by dropping the third alternative of
    `_ACTIVATION_COST_RE` and the first assert fails."""
    seas = {"oracle_text": "Whenever you draw a card, put a foreshadow counter on this "
                           "enchantment.\nRemove eight foreshadow counters from this "
                           "enchantment: Create an 8/8 blue Kraken creature token.\n"
                           "Cycling {2} ({2}, Discard this card: Draw a card.)",
            "type_line": "Enchantment", "power": None}
    assert goldfish.combat_profile(seas)["token_bodies"] == 0
    for cost in ("Sacrifice this enchantment", "Tap five untapped Elves you control",
                 "Discard two cards", "Pay 3 life"):
        card = dict(seas, oracle_text=f"{cost}: Create a 2/2 green Bear creature token.")
        assert goldfish.combat_profile(card)["token_bodies"] == 0, cost
    # A triggered token is still a body; the widening must not eat it.
    etb = dict(seas, oracle_text="When this enchantment enters, create two 1/1 colorless "
                                 "Thopter artifact creature tokens with flying.\nMax speed — "
                                 "Sacrifice this enchantment: Create two 1/1 colorless "
                                 "Thopter artifact creature tokens with flying.")
    assert goldfish.combat_profile(etb)["token_bodies"] == 2
