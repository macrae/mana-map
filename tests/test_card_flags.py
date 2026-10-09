"""`card_flags.json`, the browser-side legality flags, built from inline rows.

The unit half of the contract: given a tiny corpus, `build_card_flags` must
produce exactly the dict the page will read — names in the full `"A // B"` form,
lists sorted, a format with nothing banned ABSENT rather than empty, counts that
agree with the lists, and the same bytes twice. The regression half, which holds
the tracked file to `card_pool`'s own reading of `cards.csv`, lives beside the
other viz-index artifact tests in `tests/test_viz_index.py`; it cannot live here
because this file must stay runnable against an empty `data/`.
"""

import json

import pandas as pd

from manamap.config import LEGALITY_FORMATS
from manamap.export import viz_index as vi


def _frame(game_changer_cells):
    """Four cards: a Game Changer, a Commander ban, a Legacy ban, a plain card.

    `game_changer_cells` is the column as given, because pandas reads it as bool
    on a clean CSV and as object (strings) when a blank slips in, and the two
    must read the same way.
    """
    rows = [
        {"name": "Tergrid, God of Fright // Tergrid's Lantern", "game_changer": game_changer_cells[0],
         "legal_commander": "legal", "legal_legacy": "legal"},
        {"name": "Erayo, Soratami Ascendant // Erayo's Essence", "game_changer": game_changer_cells[1],
         "legal_commander": "banned", "legal_legacy": "legal"},
        {"name": "Zirda, the Dawnwaker", "game_changer": game_changer_cells[2],
         "legal_commander": "legal", "legal_legacy": "banned"},
        {"name": "Grizzly Bears", "game_changer": game_changer_cells[3],
         "legal_commander": "legal", "legal_legacy": "legal"},
    ]
    frame = pd.DataFrame(rows)
    for fmt in LEGALITY_FORMATS:
        if f"legal_{fmt}" not in frame.columns:
            frame[f"legal_{fmt}"] = "not_legal"
    return frame


EXPECTED = {
    "as_of": "2026-10-02",
    "game_changers": ["Tergrid, God of Fright // Tergrid's Lantern"],
    "banned": {
        "commander": ["Erayo, Soratami Ascendant // Erayo's Essence"],
        "legacy": ["Zirda, the Dawnwaker"],
    },
    "counts": {"cards": 4, "game_changers": 1, "banned": {"commander": 1, "legacy": 1}},
}


def test_the_flags_are_exactly_what_the_page_reads():
    frame = _frame([True, False, False, False])
    assert vi.build_card_flags(frame, as_of="2026-10-02") == EXPECTED


def test_string_cells_read_like_bool_cells():
    """`pilot/card_pool` reads the cell as `str(cell).lower() == "true"`; a bare
    `bool("False")` would make every plain card a Game Changer."""
    frame = _frame(["True", "False", "false", float("nan")])
    assert vi.build_card_flags(frame, as_of="2026-10-02") == EXPECTED


def test_formats_with_nothing_banned_are_absent_not_empty():
    flags = vi.build_card_flags(_frame([False] * 4), as_of="2026-10-02")
    assert flags["banned"] == {"commander": [EXPECTED["banned"]["commander"][0]],
                               "legacy": [EXPECTED["banned"]["legacy"][0]]}
    assert "standard" not in flags["banned"] and "standard" not in flags["counts"]["banned"]
    assert flags["game_changers"] == [] and flags["counts"]["game_changers"] == 0


def test_lists_are_sorted_and_deduplicated_across_printings():
    """51 corpus names have two printings; a banned card is banned once."""
    frame = _frame([True, True, False, False])
    twice = pd.concat([frame, frame.iloc[::-1]], ignore_index=True)
    flags = vi.build_card_flags(twice, as_of="2026-10-02")
    assert flags["game_changers"] == sorted(EXPECTED["game_changers"]
                                            + EXPECTED["banned"]["commander"])
    assert flags["banned"]["commander"] == EXPECTED["banned"]["commander"]
    assert flags["counts"] == {"cards": 8, "game_changers": 2,
                               "banned": {"commander": 1, "legacy": 1}}


def test_the_written_file_is_deterministic(tmp_path):
    frame = _frame([True, False, False, False])
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for seed, path in enumerate(paths):           # two row orders, one file
        vi.write_card_flags(frame.sample(frac=1, random_state=seed),
                            path=path, as_of="2026-10-02")
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert json.loads(paths[0].read_text(encoding="utf-8")) == EXPECTED
    assert "\n" not in paths[0].read_text(encoding="utf-8"), "compact JSON"


def test_as_of_prefers_the_dump_date_and_falls_back_to_the_csv(tmp_path):
    csv = tmp_path / "cards.csv"
    csv.write_text("name\n", encoding="utf-8")
    meta = tmp_path / ".download-meta.json"
    meta.write_text(json.dumps({"updated_at": "2026-10-02T21:02:01.893+00:00"}))
    assert vi.flags_as_of(meta_path=meta, csv_path=csv) == "2026-10-02"
    meta.unlink()
    fallback = vi.flags_as_of(meta_path=meta, csv_path=csv)
    assert len(fallback) == 10 and fallback[4] == "-" and fallback[7] == "-"
