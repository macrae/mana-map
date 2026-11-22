"""`similar-cards --space` and `--feeds` (2026-11-22). The feeds lists are driven
through `feeds_rows` over a stubbed card_events.json; the live space on the corpus."""
import json

import pytest

from manamap.analysis import interaction as ix
from manamap.pilot import similar_cards as sc

EVENTS = {"meta": {}, "cards": {
    "Ashnod's Altar": {"sig": ["K:BODIES", "P:DIES", "P:MANA", "L:BODIES>DIES"]},
    "Blood Artist": {"sig": ["C:DIES", "C:DIES:any", "P:LIFE_LOSS", "P:LIFE_GAIN", "L:DIES>LIFE_LOSS"]},
    "Vito": {"sig": ["C:LIFE_GAIN", "P:LIFE_LOSS", "L:LIFE_GAIN>LIFE_LOSS"]},
    "Sol Ring": {"sig": ["P:MANA"]},
    "Grizzly Bears": {"sig": []},
}}


def test_feeds_rows_are_directed_and_ranked_with_ties_by_edhrec(tmp_path, monkeypatch):
    path = tmp_path / "card_events.json"
    path.write_text(json.dumps(EVENTS))
    monkeypatch.setattr(ix, "CARD_EVENTS_PATH", path)
    pool = [n for n in EVENTS["cards"] if n != "Blood Artist"]
    feeds, fed_by = sc.feeds_rows("Blood Artist", pool, rank={"Vito": 5})
    assert [n for n, _ in feeds] == ["Vito"], "Blood Artist gains life, Vito reads it"
    assert [n for n, _ in fed_by] == ["Ashnod's Altar"], "the Altar's sacrifice is a death"
    assert sc.feeds_rows("Grizzly Bears", pool) == ([], []), "no signature, no relation"


def test_an_unknown_space_is_refused_by_name():
    with pytest.raises(KeyError):
        sc.space_matrix("no-such-space")


@pytest.mark.regression
def test_the_interaction_space_ranks_death_payoffs_beside_blood_artist():
    from manamap import spaces
    from manamap.config import OUTPUT_CSV_PATH
    if not OUTPUT_CSV_PATH.exists() or not spaces.get(spaces.INTERACTION).exists():
        pytest.skip("corpus or interaction space not on disk")
    rows, meta = sc.similar(["Blood Artist"], limit=10, space=spaces.INTERACTION, with_feeds=True)
    assert "interaction" in meta["space"]
    assert "Falkenrath Noble" in {r["name"] for r in rows}
    assert meta["feeds"] and meta["fed_by"]
    with pytest.raises(SystemExit, match="ONE seed"):
        sc.similar(["Blood Artist", "Zulaport Cutthroat"], space=spaces.INTERACTION, with_feeds=True)
