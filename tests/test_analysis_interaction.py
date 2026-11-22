"""The interaction space (2026-11-22): signatures from the engine graph's parser,
`alike` as a 128-d space, `feeds` as a directed function. Inline scripts; no Forge."""
import numpy as np

from manamap import spaces
from manamap.analysis import interaction as ix
from manamap.pilot import ability_graph as ag

ARTIST = ("Name:Blood Artist\nManaCost:1 B\nTypes:Creature Vampire\nPT:0/1\n"
          "T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | Execute$ T\n"
          "SVar:T:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ G\n"
          "SVar:G:DB$ GainLife | Defined$ You | LifeAmount$ 1\n")
CELEBRANT = ARTIST.replace("Blood Artist", "Cruel Celebrant")
ALTAR = ("Name:Ashnod's Altar\nManaCost:3\nTypes:Artifact\n"
         "A:AB$ Mana | Cost$ Sac<1/Creature> | Produced$ C | Amount$ 2\n")
SOL = "Name:Sol Ring\nManaCost:1\nTypes:Artifact\nA:AB$ Mana | Cost$ T | Produced$ C | Amount$ 2\n"


def _sig(name, text):
    return ix.signature(ag.card_abilities(name, text))


def test_a_signature_names_what_a_card_consumes_produces_and_converts():
    s = _sig("Blood Artist", ARTIST)
    assert "C:DIES" in s and "P:LIFE_LOSS" in s and "P:LIFE_GAIN" in s and "L:DIES>LIFE_LOSS" in s
    altar = _sig("Ashnod's Altar", ALTAR)
    assert "K:BODIES" in altar and "P:DIES" in altar and "P:MANA" in altar
    assert not any(t.startswith("P:CAST") or t.startswith("P:ETB") for t in s), "implicit productions stay out"


def test_feeds_is_directed():
    altar, artist, sol = _sig("A", ALTAR), _sig("B", ARTIST), _sig("S", SOL)
    assert ix.feeds(altar, artist) > 0.5
    assert ix.feeds(artist, altar) == 0.0
    assert ix.feeds(sol, artist) == 0.0


def test_two_payoffs_are_alike_and_a_payoff_and_an_outlet_are_not():
    events = {"cards": {"Blood Artist": {"sig": _sig("x", ARTIST)}, "Cruel Celebrant": {"sig": _sig("y", CELEBRANT)},
                        "Ashnod's Altar": {"sig": _sig("z", ALTAR)}, "Sol Ring": {"sig": _sig("s", SOL)},
                        "Vanilla": {"sig": []}}}
    names = ["Blood Artist", "Cruel Celebrant", "Ashnod's Altar", "Sol Ring", "Vanilla", "Blood Artist"]
    # MIN_DF is for the corpus; on five cards every token is rare.
    z, info = ix.alike_matrix(events=events, names=names, min_df=1)
    assert z.shape == (6, ix.DIM) and z.dtype == np.float32
    assert np.allclose(z[0], z[5]), "a second printing shares its name's vector"
    cos = lambda a, b: float(z[a] @ z[b])
    assert cos(0, 1) > 0.99 and cos(0, 1) > cos(0, 2)
    assert info["empty"] == ["Vanilla"] and not z[4].any()


def test_the_space_is_registered_by_name_and_is_not_the_default():
    sp = spaces.get(spaces.INTERACTION)
    assert sp.npy.name == "embeddings_interaction.npy"
    assert spaces.DEFAULT != spaces.INTERACTION and spaces.INTERACTION not in spaces.BROWSABLE


import pytest  # noqa: E402


@pytest.mark.regression
@pytest.mark.skipif(not __import__("manamap.sim.forge_cards", fromlist=["x"]).installed(),
                    reason="card_events.json is built from Forge's card scripts")
def test_the_committed_card_events_match_a_fresh_build():
    """The gate a new tracked artifact needs (CLAUDE.md): `data/card_events.json`
    equals a fresh parse of the corpus — a parser change without `manamap
    interaction` is caught here, not on the next deck that reads it."""
    committed = ix.load_card_events()
    if committed is None:
        pytest.skip("card_events.json not on disk")
    fresh = ix.build_card_events()
    assert committed["meta"] == fresh["meta"]
    diff = [n for n in fresh["cards"] if committed["cards"].get(n) != fresh["cards"][n]]
    assert not diff, f"{len(diff)} card(s) differ, e.g. {diff[:5]} — `manamap interaction`"
