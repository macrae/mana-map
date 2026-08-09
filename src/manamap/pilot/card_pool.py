"""The corpus: the ONLY reader of cards.csv, and every view taken from it.

`cards.csv` is 24.7 MB across 35 columns. This module parses it once per process
and derives every shape the pilot layer needs — a frame, a name-keyed pool, the
bracket flags, the name set, the oracle map — each memoized on the same
`(mtime_ns, size)` signature, so asking twice is free. Nothing else in `pilot/`
may open the file; `tests/test_pilot_memo.py` fails if something does.

**One shared parse rather than a per-caller `usecols` memo**, because two callers
wanting different columns would miss each other's cache and the duplicate parses
would survive. The trade is measured and worth stating plainly: `build-deck`
(which needs three of these views) runs 1.78 -> 1.55s, while `bracket-check`
goes 0.93 -> 1.00s and `validate-build` 0.64 -> 0.70s, because each needs one
view and the 13-column union costs ~0.06s more than a 3-column read. Everything
else is flat. The cost is I/O and 34,322 rows, not column count or type
inference — `dtype=str`, `low_memory` and `na_filter` all measure inside noise.

So this is a correctness-and-maintenance win with a small, known latency cost:
one place to fix a bug rather than eight, and one `card_flags` dict rather than
three copies of the same comprehension.

Everything here is READ-ONLY and shared — `load_frame()` hands back one
DataFrame that callers must not mutate.

Naming: "pool" here means the *format-wide* corpus. In `pool_facts` it means the
cards you physically own. Different question, deliberately different module.
"""

import csv

from manamap.analysis.common import parse_color_identity
from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot.common import expand_faces, mtime_memo

# Un-sets are not Commander-legal and their cards are joke designs; Stickers are
# not cards at all. Both would otherwise rank as pool candidates.
UNSET_CODES = {"unf", "sunf", "ust", "ugl", "unh", "und", "ulst"}

# The union of every column the pilot layer reads. Deliberately explicit rather
# than "all 35": the unread 22 are ~30% of the parse time and nothing wants them.
#
# `supertype` is here because `build_deck` reads it and a static scan of the
# other seven readers would not have found it — it reaches columns through
# `commander.to_dict()`, so the frame's shape is part of its contract. Adding a
# column here is cheap; discovering a missing one costs a wrong deck.
CORPUS_COLUMNS = [
    "name", "supertype", "layout", "type_line", "oracle_text",
    "color_identity", "cmc", "mana_cost", "edhrec_rank",
    "game_changer", "legal_commander", "mechanical_tags", "set_code",
]


def _read_frame():
    import pandas as pd

    return pd.read_csv(OUTPUT_CSV_PATH, usecols=CORPUS_COLUMNS)


def load_frame():
    """The corpus as a DataFrame over `CORPUS_COLUMNS`, parsed once per process.

    READ-ONLY — every caller shares this object. Returns None when cards.csv is
    absent (fresh clone, no pipeline run) so callers can degrade rather than
    fail; the ones that genuinely cannot proceed raise for themselves.
    """
    if not OUTPUT_CSV_PATH.exists():
        return None
    return mtime_memo(OUTPUT_CSV_PATH, "corpus:frame", _read_frame)


def _build_pool():
    frame = load_frame()
    pool = {}
    # `zip` over the seven columns this needs, not `itertuples` over all 13:
    # itertuples builds a 13-field namedtuple per row and there are 34,322 of
    # them, which cost more than the wider parse it was meant to amortise.
    for row in _rows(frame, "name", "set_code", "type_line", "color_identity",
                     "legal_commander", "edhrec_rank", "game_changer", "cmc",
                     "mana_cost"):
        if str(row["set_code"] or "").lower() in UNSET_CODES:
            continue
        if "Stickers" in str(row["type_line"] or ""):
            continue
        rank = row["edhrec_rank"]
        pool[row["name"]] = {
            # cards.csv stores this as "B, R, W" — comma AND space separated.
            # `set(...)` on the raw string yields {' ', ',', 'B', 'R', 'W'},
            # which can never be a subset of a commander's identity, so an
            # eligibility check silently rejected EVERY multicoloured card in
            # the format. Mono-coloured and colourless cards have no separator
            # and worked, which is exactly why it survived: the mono-black
            # decks this was built on could not see the difference.
            "color_identity": parse_color_identity(_text(row["color_identity"])),
            "legal": _text(row["legal_commander"]) == "legal",
            "edhrec_rank": None if _missing(rank) else int(float(rank)),
            "game_changer": _text(row["game_changer"]).lower() == "true",
            "type_line": _text(row["type_line"]),
            "cmc": 0.0 if _missing(row["cmc"]) else float(row["cmc"]),
            "mana_cost": _text(row["mana_cost"]),
        }
    return pool


def _rows(frame, *columns):
    """Iterate `columns` as dicts — much cheaper than `itertuples` on a wide frame.

    `itertuples` materialises a namedtuple carrying every column in the frame, so
    a view needing three fields pays for thirteen, once per row, 34,322 times.
    """
    keys = list(columns)
    for values in zip(*(frame[c] for c in keys)):
        yield dict(zip(keys, values))


def _missing(value):
    return value is None or value != value          # NaN is the pandas empty cell


def _text(value):
    """A cell as a string, with pandas' NaN read as empty.

    `csv.DictReader` yields '' for a blank cell and pandas yields NaN, so any
    code moving between them must convert: an unconverted NaN becomes the string
    'nan', which is truthy rather than empty and reads as a real value.
    """
    return "" if _missing(value) else str(value)


def load_pool():
    """name -> {color_identity, legal, edhrec_rank, game_changer, type_line, cmc, mana_cost}.

    Returns None when cards.csv is absent (fresh clone, no pipeline run), so every
    caller degrades to "no pool" rather than failing.
    """
    if not OUTPUT_CSV_PATH.exists():
        return None
    return mtime_memo(OUTPUT_CSV_PATH, "corpus:pool", _build_pool)


def _build_flags():
    frame = load_frame()
    return {
        name: {"game_changer": bool(gc), "legal_commander": legal}
        for name, gc, legal in zip(frame["name"], frame["game_changer"],
                                   frame["legal_commander"])
    }


def card_flags():
    """{name: {game_changer, legal_commander}} — the bracket engine's signal.

    Built once here for the three modules that need it (`bracket`, `build_deck`,
    `validate_build`), so a change to what "legal" means has one home.
    """
    if not OUTPUT_CSV_PATH.exists():
        raise FileNotFoundError(OUTPUT_CSV_PATH)
    return mtime_memo(OUTPUT_CSV_PATH, "corpus:flags", _build_flags)


def _build_names():
    names = set()
    for name in load_frame()["name"]:
        names |= expand_faces(name)
    return names


def corpus_names():
    """Every card name, plus each face of a DFC. None if the corpus is absent.

    Both faces are included because a scenario or decklist may legitimately name
    either one; `cards.csv` keys only the joined `"A // B"` form.
    """
    return mtime_memo(OUTPUT_CSV_PATH, "corpus:names", _build_names)


def _build_oracle():
    frame = load_frame()
    return {n: (t if isinstance(t, str) else "")
            for n, t in zip(frame["name"], frame["oracle_text"])}


def corpus_oracle():
    """{name: oracle_text}. {} if the corpus is absent."""
    return mtime_memo(OUTPUT_CSV_PATH, "corpus:oracle", _build_oracle, absent={}) or {}


def read_pool_with_csv_module():
    """The pre-consolidation `csv`-module reader, kept ONLY as a test oracle.

`load_pool` slices the shared pandas frame; this reads the CSV directly with
    the stdlib. `csv` and pandas disagree about empty cells ('' versus NaN) and
    about numeric coercion, so `tests/test_pilot_card_pool.py` asserts the two
    agree field-by-field across all 33,540 cards rather than assuming it. Not
    called in production; delete it only together with that test.
    """
    if not OUTPUT_CSV_PATH.exists():
        return None
    pool = {}
    with open(OUTPUT_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("set_code", "").lower() in UNSET_CODES:
                continue
            if "Stickers" in (row.get("type_line") or ""):
                continue
            pool[row["name"]] = {
                "color_identity": parse_color_identity(row.get("color_identity")),
                "legal": row.get("legal_commander") == "legal",
                "edhrec_rank": int(float(row["edhrec_rank"])) if row.get("edhrec_rank") else None,
                "game_changer": (row.get("game_changer") or "").lower() == "true",
                "type_line": row.get("type_line", ""),
                "cmc": float(row["cmc"]) if row.get("cmc") else 0.0,
                "mana_cost": row.get("mana_cost", ""),
            }
    return pool
