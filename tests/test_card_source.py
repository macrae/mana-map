"""What the CSV threw away, and the guard against forgetting to put it back."""

import pytest

from manamap.config import OUTPUT_CSV_PATH, RAW_JSON_PATH
from manamap.training import card_source as CS

pytestmark = pytest.mark.skipif(
    not RAW_JSON_PATH.exists(), reason="raw dump not downloaded")


@pytest.fixture(scope="module")
def dump():
    return CS.load_dump()


def test_produced_mana_is_always_a_list_even_when_the_card_makes_none(dump):
    """Scryfall OMITS the key for the 32,190 cards that make no mana; it never
    writes an empty list. `load_dump` normalises that, because "produces nothing"
    and "nobody enriched this record" must not encode identically."""
    assert all(isinstance(v["produced_mana"], list) for v in dump.values())
    empty = [v for v in dump.values() if v["produced_mana"] == []]
    assert len(empty) > 20_000, "the non-producers should dominate"


def test_a_known_five_colour_source_reports_five_colours(dump):
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    tower = frame[frame["name"] == "Command Tower"].iloc[0].to_dict()
    assert set(CS.enrich(tower, dump)["produced_mana"]) == set("WUBRG")


def test_enrich_returns_a_copy_and_does_not_mutate_the_row(dump):
    """Mutating the caller's row would make the enrichment order-dependent and
    invisible, which is how the flattened-oracle bug survived as long as it did."""
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    row = frame[frame["name"] == "Command Tower"].iloc[0].to_dict()
    before = dict(row)
    out = CS.enrich(row, dump)
    assert out is not row
    assert row == before, "enrich mutated its argument"
    assert "produced_mana" in out and "produced_mana" not in row


def test_enrich_restores_the_newlines_the_csv_flattened(dump):
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    cards = frame.to_dict("records")[:3000]
    flat = sum("\n" in str(c.get("oracle_text") or "") for c in cards)
    rich = sum("\n" in CS.enrich(c, dump)["oracle_text"] for c in cards)
    assert flat == 0, "the CSV should have no newlines left"
    assert rich > 1000, f"only {rich} enriched cards have a newline"


def test_a_corpus_the_dump_does_not_cover_is_refused(dump):
    with pytest.raises(SystemExit, match="absent from"):
        CS.enriched([{"oracle_id": "not-a-real-id", "name": "Ghost"}], dump)


def test_every_corpus_card_is_covered(dump):
    import pandas as pd

    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not built")
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    missing = [o for o in frame["oracle_id"] if o not in dump]
    assert missing == [], f"{len(missing)} cards absent from the dump"
