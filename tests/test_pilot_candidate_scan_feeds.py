"""`scan-candidates --dimension feeds[:EVENT]` (2026-11-22): pool cards admitted by
what their Forge script PRODUCES, read by the engine graph's own parser, ranked by
whether they also consume an event the deck is rich in. The helpers on inline
parses; the resolver on the synthetic deck; the live scan on the corpus (regression).
"""
import json

import pytest

from manamap.pilot import ability_graph as ag
from manamap.pilot import candidate_scan as cs
from test_pilot_engine_graph import SLUG, synthetic_deck  # noqa: F401 - the fixture

SEER = ("Name:Viscera Seer\nManaCost:B\nTypes:Creature Vampire Wizard\nPT:1/1\n"
        "A:AB$ Scry | Cost$ Sac<1/Creature> | ScryNum$ 1\n")
ARTIST = ("Name:Blood Artist\nManaCost:1 B\nTypes:Creature Vampire\nPT:0/1\n"
          "T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Graveyard | ValidCard$ Card.Self,Creature.Other | Execute$ T\n"
          "SVar:T:DB$ LoseLife | ValidTgts$ Player | LifeAmount$ 1 | SubAbility$ G\n"
          "SVar:G:DB$ GainLife | Defined$ You | LifeAmount$ 1\n")


def test_card_events_reads_a_cost_as_a_production_and_a_trigger_as_a_consumption():
    produces, consumes, how = cs.card_events(ag.card_abilities("Viscera Seer", SEER))
    assert "DIES" in produces and "SELECTION" in produces
    assert "BODIES" in consumes
    assert "cost" in how["DIES"]
    produces, consumes, how = cs.card_events(ag.card_abilities("Blood Artist", ARTIST))
    assert consumes == {"DIES"} and {"LIFE_LOSS", "LIFE_GAIN"} <= produces
    assert how["LIFE_LOSS"].startswith("trigger on DIES")


def test_the_dimension_name_is_feeds_or_feeds_colon_event():
    assert cs.is_feeds_dim("feeds") and cs.is_feeds_dim("feeds:DIES")
    assert not cs.is_feeds_dim("drain") and not cs.is_feeds_dim("feedstock")


def test_feeds_resolves_the_thinnest_component_and_refuses_without_a_record(synthetic_deck):  # noqa: F811
    from manamap.pilot import engine_graph as eg
    with pytest.raises(SystemExit, match="engine-graph"):
        cs.feeds_events(SLUG, "feeds")
    eg.write(SLUG)
    ev, why, engine = cs.feeds_events(SLUG, "feeds")
    assert ev in {c["event"] for c in engine["components"]} and "thinnest" in why
    ev2, why2, _ = cs.feeds_events(SLUG, "feeds:dies")
    assert ev2 == "DIES" and "command line" in why2
    with pytest.raises(SystemExit, match="not an event kind"):
        cs.feeds_events(SLUG, "feeds:FROBNICATE")
    assert "DIES" in cs.deck_produced_events(engine) and "LIFE_LOSS" in cs.deck_produced_events(engine)


@pytest.mark.skipif(not __import__("manamap.sim.forge_cards", fromlist=["x"]).installed(),
                    reason="the scan reads Forge's card scripts")
def test_the_live_scan_admits_only_producers_of_the_event():
    from conftest import requires_corpus, requires_deck  # noqa: F401
    from manamap.config import DECKS_DIR, OUTPUT_CSV_PATH
    if not OUTPUT_CSV_PATH.exists() or not (DECKS_DIR / "sharknado" / "engine.json").exists():
        pytest.skip("corpus or sharknado's engine record not on disk")
    doc = cs.scan("sharknado", ("feeds:DISCARD",), limit=12)
    block = doc["dimensions"]["feeds:DISCARD"]
    assert block["admits"]["produces"] == ["DISCARD"] and block["matched"] >= 12
    rows = block["candidates"]
    assert rows and all(r["matched"]["produces"][0].startswith("DISCARD") for r in rows)
    # Two-way fits sort first; nothing in the 99 appears.
    fits = [len(r["fits_back"]) for r in rows if not r["combos"]["infinite_with"]]
    assert fits == sorted(fits, reverse=True)
    names = {c["name"] for c in json.loads((DECKS_DIR / "sharknado" / "cards.json").read_text())["cards"]}
    assert not any(r["name"] in names for r in rows)
    assert not any(set(r["matched"]["consumes"]) & {"MANA", "HAND"} for r in rows), "substrate is not a fit"
