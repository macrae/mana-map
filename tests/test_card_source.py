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


def test_a_card_stops_saying_its_own_name():
    """12.6% of cards say their own name in their own rules text, so the `name`
    slot and an ability slot share a literal string and the model can learn the
    identity instead of the function."""
    assert CS.redact_name("Whenever Gishath deals combat damage", "Gishath, Sun's Avatar") \
        == "Whenever ~ deals combat damage"
    assert CS.redact_name("Shock deals 2 damage.", "Shock") == "~ deals 2 damage."


def test_a_possessive_keeps_its_s():
    """The first cut ended the match at a word boundary, so `Eluge's power` kept
    the name in full — the very leak this exists to close."""
    assert CS.redact_name("Eluge's power is 3.", "Eluge, the Shoreless Sea") \
        == "~'s power is 3."


def test_the_longest_name_part_goes_first():
    """Otherwise the full name is left as a half-redacted `~, Sweettooth Scourge`."""
    out = CS.redact_name("Greta, Sweettooth Scourge enters. Greta attacks.",
                         "Greta, Sweettooth Scourge")
    assert out == "~ enters. ~ attacks."


def test_redaction_splits_on_commas_and_never_on_spaces():
    """A card named `Food Fight` must not redact the word Food out of "create a
    Food token" — that is somebody else's game object."""
    assert CS.redact_name("Create a Food token.", "Food Fight") == "Create a Food token."
    assert CS.redact_name("Food Fight deals damage.", "Food Fight") == "~ deals damage."


def test_a_very_short_name_is_left_alone():
    """Below four characters the odds of colliding with ordinary rules
    vocabulary outrun the benefit."""
    assert CS.redact_name("Add one mana of any color.", "Ith") \
        == "Add one mana of any color."


def test_redaction_does_not_touch_other_cards_names():
    assert CS.redact_name("Whenever Sol Ring enters", "Arcane Signet") \
        == "Whenever Sol Ring enters"
