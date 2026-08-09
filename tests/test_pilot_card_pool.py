"""`card_pool`: one parse of cards.csv, and the views taken from it.

Eight sites in `pilot/` read cards.csv independently — three pandas reads with
different `usecols`, two `csv`-module readers, one with no `usecols` at all.
Consolidating them meant re-expressing a `csv.DictReader` loop as a slice of a
pandas frame, and the two libraries disagree about empty cells: `csv` yields
`''`, pandas yields `NaN`. A straight port turns every blank `mana_cost` into
the string `'nan'` — truthy, and read downstream as a real cost.

So the port is CHECKED against the reader it replaced rather than assumed.
`card_pool.read_pool_with_csv_module` exists only for this test.
"""

import pytest

from manamap.pilot import card_pool

from conftest import requires_data


@requires_data
def test_the_pandas_pool_matches_the_csv_reader_it_replaced():
    """Field-by-field over all 33,540 cards — the port's real gate.

    An assertion on counts alone would pass with every `mana_cost` set to the
    string 'nan'.
    """
    new = card_pool.load_pool()
    old = card_pool.read_pool_with_csv_module()
    assert set(new) == set(old), "the two readers disagree about which cards exist"

    mismatches = []
    for name in old:
        for field, value in old[name].items():
            if new[name][field] != value:
                mismatches.append((name, field, value, new[name][field]))
    assert not mismatches, (
        f"{len(mismatches)} field(s) differ, e.g. {mismatches[:3]}")


@requires_data
def test_a_blank_cell_stays_empty_not_the_string_nan():
    """The specific hazard, pinned separately so a regression names itself."""
    pool = card_pool.load_pool()
    texts = [c["mana_cost"] for c in pool.values()] + \
            [c["type_line"] for c in pool.values()]
    assert "nan" not in texts, (
        "a pandas NaN reached a string field — blank cells must read as ''")


@requires_data
def test_the_union_covers_every_column_its_consumers_read():
    """`build_deck` reaches columns through `commander.to_dict()`, so the
    frame's SHAPE is part of its contract. `supertype` was needed by exactly
    one consumer and a scan of the other seven would not have found it."""
    frame = card_pool.load_frame()
    from manamap.pilot.pool_facts import _COLUMNS
    missing = set(_COLUMNS) - set(frame.columns)
    assert not missing, f"pool_facts reads {missing}, absent from CORPUS_COLUMNS"
    for column in ("supertype", "game_changer", "legal_commander", "oracle_text"):
        assert column in frame.columns, f"{column} dropped from the union"


@requires_data
def test_the_views_agree_with_the_frame():
    """Each derived view is built by zipping columns rather than `itertuples`,
    which is a rewrite worth checking against the frame it came from."""
    frame = card_pool.load_frame()
    flags = card_pool.card_flags()
    names = card_pool.corpus_names()
    oracle = card_pool.corpus_oracle()

    assert len(flags) == frame["name"].nunique()
    assert len(oracle) == frame["name"].nunique()
    # Every joined DFC name contributes its faces on top of the joined form.
    assert names >= set(frame["name"])
    dfc = [n for n in frame["name"] if " // " in n]
    if dfc:
        front = dfc[0].split(" // ")[0]
        assert front in names, "a DFC's front face must resolve"


@requires_data
def test_one_parse_serves_every_view(monkeypatch):
    """The property the consolidation exists for.

    Before it, `build-deck` parsed cards.csv three times in one process and
    `validate-build` twice.
    """
    import pandas as pd

    from manamap.pilot.common import clear_memo
    clear_memo()

    reads = []
    real = pd.read_csv
    monkeypatch.setattr(
        pd, "read_csv", lambda *a, **k: (reads.append(1), real(*a, **k))[1])

    card_pool.load_frame()
    card_pool.load_pool()
    card_pool.card_flags()
    card_pool.corpus_names()
    card_pool.corpus_oracle()
    assert len(reads) == 1, f"{len(reads)} parses for five views"
    clear_memo()


def test_load_frame_returns_none_without_a_corpus(monkeypatch, tmp_path):
    """A fresh clone degrades rather than raising — callers that genuinely
    cannot proceed say so themselves."""
    monkeypatch.setattr(card_pool, "OUTPUT_CSV_PATH", tmp_path / "absent.csv")
    assert card_pool.load_frame() is None
    assert card_pool.load_pool() is None
    with pytest.raises(FileNotFoundError):
        card_pool.card_flags()
