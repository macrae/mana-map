"""`first_released_at`: when a card FIRST existed, across every printing.

`released_at` in cards.csv is the date of whichever printing the oracle dump
chose, and it skews recent — Sol Ring's corpus printing is Reality Fracture
Commander (2026-10-02). A date filter built on it returns Sol Ring for "released
after 2026" and is confidently wrong. The column is reduced from Scryfall's
every-printing bulk in step 1 and joined by oracle id in step 2.
"""

import pandas as pd
import pytest
from conftest import requires_data

from manamap.config import FIRST_PRINTINGS_PATH, OUTPUT_CSV_PATH
from manamap.ingest.download import oracle_id_of, reduce_first_printings
from manamap.pilot.card_search import date_bound


# ── the reduction ───────────────────────────────────────────────────────


def test_earliest_paper_printing_wins():
    got = reduce_first_printings([
        {"oracle_id": "a", "released_at": "2026-10-02"},
        {"oracle_id": "a", "released_at": "1993-08-05"},
        {"oracle_id": "a", "released_at": "2003-07-28"},
    ])
    assert got == {"a": "1993-08-05"}


def test_an_earlier_digital_printing_does_not_win_over_paper():
    """An MTGO cube printing predating the cardboard is not when a pilot could
    first have sleeved the card."""
    got = reduce_first_printings([
        {"oracle_id": "a", "released_at": "2014-06-16", "digital": True},
        {"oracle_id": "a", "released_at": "2020-01-01"},
    ])
    assert got == {"a": "2020-01-01"}


def test_a_digital_only_card_still_gets_a_date():
    got = reduce_first_printings([{"oracle_id": "a", "released_at": "2022-03-17",
                                   "digital": True}])
    assert got == {"a": "2022-03-17"}


def test_reversible_cards_carry_the_oracle_id_on_their_faces():
    card = {"released_at": "2023-01-01",
            "card_faces": [{"oracle_id": "face"}, {"oracle_id": "face"}]}
    assert oracle_id_of(card) == "face"
    assert reduce_first_printings([card]) == {"face": "2023-01-01"}


# ── the bounds card-search applies ──────────────────────────────────────


@pytest.mark.parametrize("text,end,want", [
    ("2024", False, "2024-01-01"),
    ("2024", True, "2024-12-31"),
    ("2024-06", False, "2024-06-01"),
    ("2024-6", True, "2024-06-31"),
    ("2024-06-15", True, "2024-06-15"),
])
def test_partial_dates_pad_to_the_edge_they_face(text, end, want):
    """'2024-06-31' is not a calendar date and does not need to be: it is a
    string bound, and every June date compares <= it."""
    assert date_bound(text, end) == want


def test_a_non_date_is_refused():
    with pytest.raises(SystemExit):
        date_bound("last year", end=False)


# ── the corpus ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus():
    df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False,
                     usecols=["name", "released_at", "first_released_at"])
    return df


@requires_data
@pytest.mark.parametrize("name,year", [
    ("Sol Ring", "1993"), ("Lightning Bolt", "1993"), ("Counterspell", "1993"),
    ("Demonic Tutor", "1993"), ("Swords to Plowshares", "1993"),
    ("Edgar Markov", "2017"),
])
def test_old_cards_resolve_to_their_real_first_printing(corpus, name, year):
    """Every one of these has a corpus printing years later than its first."""
    row = corpus[corpus["name"] == name].iloc[0]
    assert str(row["first_released_at"]).startswith(year), (name, row.to_dict())


@requires_data
def test_no_card_first_existed_after_its_own_printing(corpus):
    """The invariant that catches the whole class rather than six examples: the
    corpus printing is ONE of the printings, so the earliest can never be later."""
    have = corpus.dropna(subset=["first_released_at"])
    bad = have[have["first_released_at"] > have["released_at"]]
    assert len(have) > 30000
    assert bad.empty, bad.head(10).to_dict("records")


@requires_data
def test_almost_every_card_resolves(corpus):
    """Absent is allowed (it is never guessed), but it must be rare — a broad gap
    means the join key or the reduction broke."""
    missing = corpus["first_released_at"].isna().sum()
    assert missing <= len(corpus) * 0.001, f"{missing} cards with no first date"


@requires_data
def test_the_reduction_is_on_disk_for_extract():
    assert FIRST_PRINTINGS_PATH.exists(), "run `manamap download`"
