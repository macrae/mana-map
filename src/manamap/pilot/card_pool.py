"""The card pool: every legal Commander card, name-keyed.

This is the corpus half of "what else could this deck play" — nothing to do with
any one deck's contents. It lived in `upgrade_facts.py`, which was named for the
retired bench brief and deleted with it; `deck_audit.engine_activation` imports
this to answer "which pool cards would join the engine's thinnest component", and
that question has no bench in it.

Read with `csv` rather than pandas: this is a name-keyed lookup, not a frame, and
the pilot subsystem otherwise avoids the pandas import.
"""

import csv

from manamap.analysis.common import parse_color_identity
from manamap.config import OUTPUT_CSV_PATH

# Un-sets are not Commander-legal and their cards are joke designs; Stickers are
# not cards at all. Both would otherwise rank as pool candidates.
UNSET_CODES = {"unf", "sunf", "ust", "ugl", "unh", "und", "ulst"}


def load_pool():
    """name -> {color_identity, legal, edhrec_rank, game_changer, type_line, cmc, mana_cost}.

    Returns None when cards.csv is absent (fresh clone, no pipeline run), so every
    caller degrades to "no pool" rather than failing.
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
                # cards.csv stores this as "B, R, W" — comma AND space separated.
                # `set(...)` on the raw string yields {' ', ',', 'B', 'R', 'W'},
                # which can never be a subset of a commander's identity, so an
                # eligibility check silently rejected EVERY multicoloured card in
                # the format. Mono-coloured and colourless cards have no separator
                # and worked, which is exactly why it survived: the mono-black
                # decks this was built on could not see the difference.
                "color_identity": parse_color_identity(row.get("color_identity")),
                "legal": row.get("legal_commander") == "legal",
                "edhrec_rank": int(float(row["edhrec_rank"])) if row.get("edhrec_rank") else None,
                "game_changer": (row.get("game_changer") or "").lower() == "true",
                "type_line": row.get("type_line", ""),
                "cmc": float(row["cmc"]) if row.get("cmc") else 0.0,
                "mana_cost": row.get("mana_cost", ""),
            }
    return pool
