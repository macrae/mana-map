"""Pilot: readers for a POOL FILE — the card names a brief may constrain a build to.

A pool is a constraint, never an inventory. `build_deck.resolve_pool` narrows the
candidate frame to the names a brief's `pool_files` list, `build_deck.deck_printings`
takes the printing each line pins, and `validate_build._validate_pool` re-derives
the same set to prove the plan stayed inside it. All three read through here, so a
pool built from and a pool validated against can never disagree about what it holds.

Until 2026-11-22 this module was also a command — `pool-facts`, "what deck can I
build from a box of cards" — and the box was the pilot's physical collection. The
bench no longer has a notion of what anyone holds (no boxes, no buy list, no bill),
so the analysis went with it: depth, castability, role coverage and contained
combos over a pile of cardboard were answers to a question the bench no longer asks.
The readers stayed because a brief can still say "build from these names", and that
is a statement about the build, not about a shelf.

Note the word: "pool" everywhere else in this repo means the *format-wide* corpus
(`card_pool`). Here it means one brief's list of names.

    parsing     pilot/fetch_deck.py   parse_mainboard   (quantities, printings, *F*)
    cards       pilot/card_pool.py    load_frame        (one parse per process)
"""

from collections import Counter, defaultdict
from pathlib import Path

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot.card_pool import load_frame
from manamap.pilot.common import mtime_memo
from manamap.pilot.fetch_deck import parse_mainboard

# Columns a pool card row carries. The readers below key on `name` alone, but the
# rows are handed around as "a card row" and widening or narrowing them would
# quietly change what a caller sees in a `.keys()` or a dump — so the slice is
# fixed here and `tests/test_pilot_card_pool.py` holds it to the corpus union.
_COLUMNS = [
    "name", "layout", "type_line", "oracle_text", "color_identity", "cmc",
    "edhrec_rank", "game_changer", "legal_commander", "mechanical_tags",
]


def _read_cards():
    # Slice to this module's columns rather than handing out the full union:
    # these records are passed around as "a card row" and widening them would
    # quietly change what downstream code sees in a `.keys()` or a dump.
    return {r["name"]: r
            for r in load_frame()[_COLUMNS].to_dict("records")}


def load_cards():
    """The cards.csv slice this module reads, as {name: row-dict}.

    Built from `card_pool.load_frame()`, so a build that names a pool parses the
    24 MB CSV once for this, the bracket engine and everything else together —
    it used to be parsed here and again inside `bracket.load_reference`, with
    `build_deck` calling in through `resolve_pool` for a third. Treat the result
    as read-only; callers share one parse.
    """
    if not OUTPUT_CSV_PATH.exists():
        raise SystemExit(
            f"{OUTPUT_CSV_PATH} not found — run `manamap extract` first. "
            f"A pool cannot be read without the card database."
        )
    return mtime_memo(OUTPUT_CSV_PATH, "pool_facts:cards", _read_cards)


def front_face_map(cards):
    """{front face → joined name} for multi-face cards.

    cards.csv and every global graph key on the joined `" // "` form; decklists
    carry the front face alone. Skipping this translation silently drops ~2.5%
    of the corpus from every join, and drops it *quietly* — the card is simply
    absent from the results rather than reported as unresolved.
    """
    return {name.split(" // ")[0]: name for name in cards if " // " in name}


def read_sources(paths, cards):
    """Parse decklist files into {file: {name: copies}} plus the unresolved names."""
    front = front_face_map(cards)
    per_file = {}
    unresolved = defaultdict(list)
    for path in paths:
        counts = Counter()
        for entry in parse_mainboard(Path(path).read_text()):
            name = entry["name"]
            resolved = name if name in cards else front.get(name)
            if resolved is None:
                unresolved[name].append(Path(path).name)
                continue
            counts[resolved] += int(entry.get("quantity") or 1)
        per_file[Path(path).name] = dict(counts)
    return per_file, {n: sorted(set(f)) for n, f in unresolved.items()}


def pool_printings(paths, cards):
    """{name: {set, collector_number, foil}} — the PRINTING the pool file pins.

    `parse_mainboard` already captures set, collector number and foil from a
    decklist's `(SET) 123 *F*` suffix; `read_sources` deliberately drops all of it
    because counting copies does not care. A build from a pool that pins printings
    does care, and dropping it here is how a 100-card list of pinned cards became
    a Scryfall default for every one of them — the legacy page credited Marvel
    Super Heroes Commander art for a Sol Ring that is nothing of the sort, and
    Featured Artist counts artists per printing.

    First occurrence wins. A card in two pool files is one card; which line it
    came from is not a decision this function should invent.
    """
    front = front_face_map(cards)
    out = {}
    for path in paths:
        for entry in parse_mainboard(Path(path).read_text()):
            name = entry["name"]
            resolved = name if name in cards else front.get(name)
            if resolved is None or resolved in out:
                continue
            if not entry.get("set"):
                continue
            out[resolved] = {
                "set": entry["set"],
                "collector_number": entry.get("collector_number"),
                "foil": bool(entry.get("foil")),
            }
    return out


def collect_paths(targets, exclude):
    """Expand files and directories into a sorted list of decklists to read."""
    excluded = {Path(e).resolve() for e in exclude or []}
    found = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            found.extend(sorted(path.glob("*.txt")))
        elif path.exists():
            found.append(path)
        else:
            raise SystemExit(f"No such file or directory: {target}")
    kept = [p for p in found if p.resolve() not in excluded]
    if not kept:
        raise SystemExit(f"No decklist files found in {', '.join(map(str, targets))}")
    return kept
