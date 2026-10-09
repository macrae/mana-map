"""Pilot: a deck's known combo lines and its one-card near misses, from Spellbook.

Two questions, both answered by set intersection over `combo_details.json` and
nothing else:

- **Included** — every Commander Spellbook line whose cards are ALL in the 99
  plus the command zone. Each row carries the line's bracket letter, whether it
  is infinite, whether it is banned, and whether it quietly assumes one of its
  own pieces is your commander (`assumes_other_commander` — see the note on
  `COMMANDER_TELL`). `bracket.assess` reads the same rows and keeps its own
  floor logic on top of them, so the bracket report and this one cannot
  disagree about which lines a deck contains.
- **Near misses** — every line with exactly ONE card absent, where that card is
  legal in the DECK'S FORMAT (`formats.for_doc` → the `legal_<format>` column;
  Commander for the fleet), inside the deck's colour identity, and the line is
  not banned. A near miss is a question for the pilot ("would this card close a
  line?"), never a recommendation: the goldfish cannot see most of them and
  Spellbook's popularity is a deck count, not a verdict.

Fully deterministic — no LLM calls, no randomness, no network. The same list
against the same combo file always writes the same bytes (◆ evidence).
`validate_deck_combos` is the gate; `tests/test_pilot_artifact_freshness.py`
recomputes every tracked copy.

Why this lives in its own module rather than in `bracket.py`: the bracket is a
FLOOR driven by the lines a deck contains, and it has no business knowing about
lines the deck does not contain. The near-miss scan reads the corpus for
legality and identity, which the bracket engine never needed.
"""

import json

from manamap.config import COMBO_DETAILS_PATH, DECK_COMBOS_NEAR_LIMIT, OUTPUT_CSV_PATH
from manamap.pilot import formats
from manamap.pilot.card_pool import legality, load_pool
from manamap.pilot.card_search import deck_identity
from manamap.pilot.common import (
    deck_dir,
    decklist_sha256,
    load_combo_details,
    load_deck_cards,
)

ARTIFACT = "combos.json"

INFINITE_PREFIX = "infinite"

# Commander Spellbook is format-agnostic about *which* card sits in the command
# zone: a line can quietly assume one of its pieces is your commander, and
# "commander" in `produces` is the tell. goblin-storm stack 004 is the cautionary
# tale — the combo graph promised goblin-storm an infinite off Krenko, but CR
# 903.9a scopes the graveyard-to-command-zone action to *a commander*, and Zada
# holds that seat. Counting those lines toward a bracket floor would inflate
# every deck that happens to run a popular legendary creature in the 99.
COMMANDER_TELL = "commander"


def combos_in_deck(names, details):
    """Indices of every combo whose cards are all present in `names`.

    Uses the by_card index rather than scanning 114K combos — the candidate set
    is the union of combos touching any deck card, which is a few thousand.
    """
    present = set(names)
    candidates = set()
    for name in present:
        candidates.update(details["by_card"].get(name, []))
    combos = details["combos"]
    return sorted(i for i in candidates if all(c in present for c in combos[i]["cards"]))


def is_infinite(combo):
    """Does this combo produce an unbounded loop?"""
    return any(str(p).lower().startswith(INFINITE_PREFIX) for p in combo.get("produces", []))


def assumes_other_commander(combo, commanders):
    """Does this line only work if one of its pieces is your commander?

    True when `produces` mentions the command zone but none of the combo's
    cards actually holds that seat in this deck. Such a line is not available
    to the pilot and must not raise their bracket floor.
    """
    if not any(COMMANDER_TELL in str(p).lower() for p in combo.get("produces", [])):
        return False
    return not any(card in commanders for card in combo["cards"])


def _row(combo, commanders):
    """One included line, every field the report and the bracket engine read."""
    return {
        "id": combo.get("id"),
        "cards": list(combo["cards"]),
        "produces": list(combo.get("produces", [])),
        "infinite": is_infinite(combo),
        "bracket": combo.get("bracket"),
        "banned": bool(combo.get("banned")),
        "mana_value_needed": combo.get("mana_value_needed"),
        "popularity": combo.get("popularity"),
        "assumes_other_commander": assumes_other_commander(combo, commanders),
    }


def _rank(row):
    """Bracket desc, popularity desc, id — the order a reader wants.

    A None bracket (a banned line) sorts below bracket 1, and a None popularity
    (Spellbook did not count it) sorts below 0: absent is not zero, and it
    must not outrank a measured zero either.
    """
    bracket = row.get("bracket")
    popularity = row.get("popularity")
    return (-(bracket if bracket is not None else -1),
            -(popularity if popularity is not None else -1),
            str(row.get("id")))


def included_combos(names, commanders, details, ranked=True):
    """Every combo fully contained in `names ∪ commanders`, as rows.

    `ranked=True` (the report) orders by `_rank`. `ranked=False` keeps
    Spellbook's file order — the order `combos_in_deck` has always returned and
    the order every tracked `bracket_report.json` was written in: `assess`
    takes the FIRST highest-bracket line as its named driver and lists the
    two-card infinites in file order, so handing it ranked rows would move
    bytes in a report whose content had not changed.
    """
    present = set(names) | set(commanders)
    combos = details["combos"]
    rows = [_row(combos[i], set(commanders)) for i in combos_in_deck(present, details)]
    if ranked:
        rows.sort(key=_rank)
    return rows


def near_misses(names, commanders, identity, details, pool, limit=DECK_COMBOS_NEAR_LIMIT,
                legal=None):
    """Lines one card short, where that card is legal and in identity.

    Returns `(rows, total)` — the rows capped at `limit` after sorting
    `(popularity desc, id)`, and the UNCAPPED count, so a reader can tell "50
    shown" from "50 exist".

    `legal` is `{name: "legal" | "banned" | "not_legal"}` for the deck's format
    column (`card_pool.legality`). None — the fleet's path — reads the pool's
    own `legal` flag, which IS the Commander column: `build_report` passes a
    map only for a non-default format, so the fourteen tracked reports did not
    move when the parameter arrived (2026-10-09).

    A line whose `produces` assumes its own commander and whose pieces do not
    include this deck's is left out too: adding the missing card to the 99
    would not make it work, so it is not one card short — it is a different
    deck's line.
    """
    present = set(names) | set(commanders)
    identity = {c.upper() for c in identity}
    combos = details["combos"]
    candidates = set()
    for name in present:
        candidates.update(details["by_card"].get(name, []))
    rows = []
    for i in sorted(candidates):
        combo = combos[i]
        if combo.get("banned"):
            continue
        missing = [c for c in combo["cards"] if c not in present]
        if len(missing) != 1:
            continue
        rec = pool.get(missing[0])
        if rec is None:
            continue
        if (legal.get(missing[0]) != "legal") if legal is not None else not rec.get("legal"):
            continue
        if not set(rec.get("color_identity") or ()) <= identity:
            continue
        if assumes_other_commander(combo, commanders):
            continue
        rows.append({
            "id": combo.get("id"),
            "cards": list(combo["cards"]),
            "missing": missing[0],
            "infinite": is_infinite(combo),
            "bracket": combo.get("bracket"),
            "mana_value_needed": combo.get("mana_value_needed"),
            "popularity": combo.get("popularity"),
        })
    rows.sort(key=lambda r: (-(r["popularity"] if r["popularity"] is not None else -1),
                             str(r["id"])))
    return rows[:limit], len(rows)


def summarize(included, near, near_total):
    """The counts a page shows, from the rows — the validator recomputes this
    and compares, so a stored summary can never drift from its own rows."""
    counted = [c for c in included if not c["assumes_other_commander"]]
    infinites = [c for c in counted if c["infinite"]]
    brackets = [c["bracket"] for c in counted if c.get("bracket") is not None]
    return {
        "included": len(included),
        "infinite": len(infinites),
        "two_card_infinite": sum(1 for c in infinites if len(c["cards"]) == 2),
        "excluded_commander_assumption": len(included) - len(counted),
        "near": len(near),
        "near_total": near_total,
        # None when no counted line carries a bracket: absent, never 1.
        "highest_bracket": max(brackets) if brackets else None,
    }


def build_report(slug, branch=None):
    """The artifact dict for one deck or branch. Reads, never writes."""
    if not COMBO_DETAILS_PATH.exists():
        raise SystemExit(f"Missing {COMBO_DETAILS_PATH} — run `manamap process-combos`.")
    pool = load_pool()
    if pool is None:
        raise SystemExit(f"Missing {OUTPUT_CSV_PATH} — run `manamap extract` "
                         f"(near misses need legality and identity).")
    doc = load_deck_cards(slug, branch)
    cards = doc.get("cards", [])
    names = [c["name"] for c in cards]
    commanders = [c["name"] for c in cards if c.get("is_commander")]
    details = load_combo_details()
    meta = details.get("meta") or {}

    included = included_combos(names, commanders, details)
    # The deck's format decides which card is a legal add. Spellbook's lines are
    # Commander's and stay so (`included` is a pure intersection); only the
    # near-miss filter — "could this card go in THIS deck" — reads the column.
    spec = formats.for_doc(doc)
    legal = None if spec is formats.DEFAULT else legality(spec.legality_column)
    near, near_total = near_misses(names, commanders, deck_identity(doc), details, pool,
                                   legal=legal)
    return {
        "slug": slug,
        "branch": branch,
        "decklist_sha256": decklist_sha256(slug, branch),
        "combo_data": {
            "source_timestamp": (meta.get("source") or {}).get("timestamp"),
            "combo_count": meta.get("combo_count", len(details["combos"])),
        },
        "summary": summarize(included, near, near_total),
        "included": included,
        "near": near,
    }


def format_report(report):
    """The short table: included lines, then the near misses, then the total."""
    s = report["summary"]
    where = report["slug"] + (f"@{report['branch']}" if report.get("branch") else "")
    lines = [f"{where}: {s['included']} known line(s) in the list, "
             f"{s['infinite']} infinite ({s['two_card_infinite']} two-card)"
             + (f", highest bracket {s['highest_bracket']}" if s["highest_bracket"] else "")
             + (f", {s['excluded_commander_assumption']} assume another commander"
                if s["excluded_commander_assumption"] else "")]
    for c in report["included"]:
        tag = "∞ " if c["infinite"] else "  "
        flags = []
        if c["bracket"] is not None:
            flags.append(f"bracket {c['bracket']}")
        if c["banned"]:
            flags.append("BANNED")
        if c["assumes_other_commander"]:
            flags.append("assumes own commander")
        lines.append(f"  {tag}{' + '.join(c['cards'])}"
                     + (f"  [{', '.join(flags)}]" if flags else ""))
    lines.append(f"  near misses: {s['near']} shown of {s['near_total']} one card short "
                 f"(legal, in identity, not banned)")
    for c in report["near"]:
        tag = "∞ " if c["infinite"] else "  "
        have = [x for x in c["cards"] if x != c["missing"]]
        pop = f"  ({c['popularity']} decks)" if c["popularity"] else ""
        lines.append(f"  {tag}+ {c['missing']}  with {' + '.join(have)}"
                     + (f"  [bracket {c['bracket']}]" if c["bracket"] is not None else "")
                     + pop)
    return "\n".join(lines)


def main(args):
    branch = getattr(args, "branch", None)
    report = build_report(args.slug, branch)
    if getattr(args, "as_json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    if getattr(args, "write", False):
        # Same shape as `bracket.main`: a ◆ artifact beside the deck's own
        # decklist, or beside the branch's — `deck_dir` makes the scoping
        # structural, so a branch run cannot touch the tracked report.
        out = deck_dir(args.slug, branch) / ARTIFACT
        with open(out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"  Wrote {out}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-combos <slug>`.")
