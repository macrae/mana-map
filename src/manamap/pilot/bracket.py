"""Pilot: compute a deck's Commander bracket floor and the evidence for it.

The bracket system is defined by deck *contents*, which means most of it is
mechanically checkable. Three signals do the work:

- **Game Changers** — a flag Scryfall carries on WotC's published list.
- **Combo content** — Commander Spellbook tags every variant with a bracket
  letter. Intersect a deck's names against the combo index, take the highest
  bracket among the combos it *fully contains*, and you can name the exact line
  that pushes a deck out of Bracket 2.
- **Mass land denial** — a curated list, and the honest weak spot.

Two things this module refuses to do. It does not tell you what bracket your
deck is: WotC is explicit that brackets are a conversation, not a calculator,
so we compute a **floor** — the lowest bracket the contents are consistent with
— and report what drove it. And it does not raise that floor on tutor density:
WotC removed the tutor guardrail entirely on 2025-10-21 ("not all Tutors are
created equal") and now catches the efficient ones through Game Changers
membership, which this module already counts. Tutors are reported, never scored.

Fully deterministic: set intersections over committed artifacts, no LLM calls,
no randomness. The same deck always produces the same report (◆ evidence).
"""

import json
import sys


from manamap.config import (
    BRACKET_EARLY_COMBO_MANA,
    BRACKETS,
    CARD_ROLES_PATH,
    COMBO_DETAILS_PATH,
    MASS_LAND_DENIAL,
    OUTPUT_CSV_PATH,
)
from manamap.pilot.card_pool import card_flags
from manamap.pilot.common import (
    deck_dir,
    load_card_roles,
    load_combo_details,
    load_deck_cards,
)

INFINITE_PREFIX = "infinite"

# Commander Spellbook is format-agnostic about *which* card sits in the command
# zone: a line can quietly assume one of its pieces is your commander, and
# "commander" in `produces` is the tell. goblin-storm stack 004 is the cautionary
# tale — the combo graph promised goblin-storm an infinite off Krenko, but CR
# 903.9a scopes the graveyard-to-command-zone action to *a commander*, and Zada
# holds that seat. Counting those lines toward a bracket floor would inflate
# every deck that happens to run a popular legendary creature in the 99.
COMMANDER_TELL = "commander"


def load_reference():
    """Load the three global artifacts an assessment needs.

    Returns (card_flags, roles, details) where card_flags is
    {name: {"game_changer": bool, "legal_commander": str}}.
    """
    missing = [p for p in (OUTPUT_CSV_PATH, CARD_ROLES_PATH, COMBO_DETAILS_PATH) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing reference artifacts: "
            + ", ".join(str(p) for p in missing)
            + " — run `manamap extract`, `manamap card-roles`, `manamap process-combos`."
        )

    return _card_flags(), load_card_roles(), load_combo_details()


def _card_flags():
    """{name: {game_changer, legal_commander}} from cards.csv, once per process.

    Three modules built this identical dict from three separate parses; it now
    lives once in `card_pool`.
    """
    return card_flags()


def combos_in_deck(names, details):
    """Indices of every combo whose cards are all present in `names`.

    Uses the by_card index rather than scanning 83K combos — the candidate set
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


def assess(names, card_flags, roles, details, commanders=()):
    """Compute a bracket floor and the evidence for it. Returns a report dict."""
    names = sorted(set(names))
    commanders = set(commanders)
    combos = details["combos"]
    indices = combos_in_deck(names, details)
    all_contained = [combos[i] for i in indices]

    excluded = [c for c in all_contained if assumes_other_commander(c, commanders)]
    contained = [c for c in all_contained if c not in excluded]

    game_changers = sorted(n for n in names if card_flags.get(n, {}).get("game_changer"))
    banned = sorted(n for n in names if card_flags.get(n, {}).get("legal_commander") == "banned")
    denial = sorted(n for n in names if n in MASS_LAND_DENIAL)
    tutors = sorted(n for n in names if "tutor:unrestricted" in roles.get(n, []))

    infinites = [c for c in contained if is_infinite(c)]
    two_card = [c for c in infinites if len(c["cards"]) == 2]
    early = [
        c for c in two_card
        if c.get("mana_value_needed") is not None
        and c["mana_value_needed"] <= BRACKET_EARLY_COMBO_MANA
    ]

    brackets = [c["bracket"] for c in contained if c.get("bracket") is not None]
    combo_floor = max(brackets) if brackets else 1

    # Each driver states the floor it forces and why, so a disagreement with a
    # declared bracket is always traceable to a named card or line.
    drivers = []
    floor = 1

    if game_changers:
        gc_floor = 3 if len(game_changers) <= BRACKETS[3]["game_changers"] else 4
        floor = max(floor, gc_floor)
        drivers.append({
            "signal": "game_changers",
            "forces": gc_floor,
            "detail": f"{len(game_changers)} Game Changer(s): {', '.join(game_changers)}",
        })

    if combo_floor > 1:
        floor = max(floor, combo_floor)
        worst = max(contained, key=lambda c: (c.get("bracket") or 0))
        drivers.append({
            "signal": "combo_content",
            "forces": combo_floor,
            "detail": f"highest-bracket line present: {' + '.join(worst['cards'])}",
        })

    if two_card:
        tc_floor = 4 if early else 3
        floor = max(floor, tc_floor)
        example = (early or two_card)[0]
        drivers.append({
            "signal": "two_card_infinite",
            "forces": tc_floor,
            "detail": (
                f"{len(two_card)} two-card infinite(s); "
                f"{len(early)} assemble at mana value <= {BRACKET_EARLY_COMBO_MANA}. "
                f"e.g. {' + '.join(example['cards'])}"
            ),
        })

    if denial:
        floor = max(floor, 4)
        drivers.append({
            "signal": "mass_land_denial",
            "forces": 4,
            "detail": ", ".join(denial),
        })

    notes = [
        "Brackets are a conversation, not a calculator — this is the lowest "
        "bracket the deck's contents are consistent with, not a verdict.",
        f"Tutor density ({len(tutors)}) is reported, never scored: WotC removed "
        "the tutor restriction on 2025-10-21 and now relies on Game Changers "
        "membership, which is already counted above.",
        "Mass land denial is checked against a curated list and is incomplete "
        "by construction.",
    ]
    if banned:
        notes.append(f"Deck contains Commander-banned cards: {', '.join(banned)}.")
    if excluded:
        notes.append(
            f"{len(excluded)} known line(s) excluded: they assume one of their own "
            f"pieces is your commander, and it isn't. See CR 903.9a."
        )

    return {
        "floor": floor,
        "floor_name": BRACKETS[floor]["name"],
        "drivers": drivers,
        "game_changers": game_changers,
        "banned": banned,
        "mass_land_denial": denial,
        "tutors": tutors,
        "combo_count": len(contained),
        "combo_bracket_floor": combo_floor,
        "two_card_infinites": [
            {"cards": c["cards"], "produces": c["produces"],
             "mana_value_needed": c.get("mana_value_needed"), "bracket": c.get("bracket")}
            for c in two_card
        ],
        "excluded_commander_assumption": [
            {"cards": c["cards"], "bracket": c.get("bracket")} for c in excluded
        ],
        "notes": notes,
    }


def offending_cards(report, target):
    """Cards to consider cutting to reach `target`, most-implicated first.

    Returns [{"name", "reason", "forces"}]. Empty when the floor already fits.
    """
    if report["floor"] <= target:
        return []

    limit = BRACKETS[target]["game_changers"]
    counts = {}

    def bump(name, reason, forces):
        entry = counts.setdefault(name, {"name": name, "reasons": set(), "forces": forces})
        entry["reasons"].add(reason)
        entry["forces"] = max(entry["forces"], forces)

    if limit is not None and len(report["game_changers"]) > limit:
        # Everything over the limit is a candidate; the caller ranks by slot value.
        for name in report["game_changers"]:
            bump(name, "game changer", 3 if limit else 3)

    if BRACKETS[target]["two_card_infinites"] == 0:
        for combo in report["two_card_infinites"]:
            for name in combo["cards"]:
                bump(name, f"two-card infinite with {' + '.join(combo['cards'])}", 4)

    if not BRACKETS[target]["mass_land_denial"]:
        for name in report["mass_land_denial"]:
            bump(name, "mass land denial", 4)

    ranked = sorted(
        counts.values(),
        key=lambda e: (-e["forces"], -len(e["reasons"]), e["name"]),
    )
    return [{"name": e["name"], "reasons": sorted(e["reasons"]), "forces": e["forces"]}
            for e in ranked]


def format_report(slug, report, target=None):
    """Human-readable assessment block."""
    lines = [f"{slug}: bracket floor {report['floor']} ({report['floor_name']})"]
    if target is not None:
        verdict = "OK  " if report["floor"] <= target else "OVER"
        lines.append(f"{verdict} target bracket {target} ({BRACKETS[target]['name']})")
    for driver in report["drivers"]:
        lines.append(f"  → {driver['signal']} forces {driver['forces']}: {driver['detail']}")
    if not report["drivers"]:
        lines.append("  → nothing in the deck raises it above 1")
    lines.append(
        f"  {report['combo_count']} known combo(s) fully contained; "
        f"{len(report['two_card_infinites'])} two-card infinite(s); "
        f"{len(report['tutors'])} unrestricted tutor(s)"
    )
    for combo in report["excluded_commander_assumption"]:
        lines.append(f"  excluded (assumes its own commander): {' + '.join(combo['cards'])}")
    if target is not None and report["floor"] > target:
        lines.append("  Cut candidates:")
        for card in offending_cards(report, target)[:10]:
            lines.append(f"    - {card['name']} ({'; '.join(card['reasons'])})")
    for note in report["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(args):
    doc = load_deck_cards(args.slug)
    cards = doc.get("cards", [])
    names = [c["name"] for c in cards]
    commanders = [c["name"] for c in cards if c.get("is_commander")]
    card_flags, roles, details = load_reference()
    report = assess(names, card_flags, roles, details, commanders)
    target = getattr(args, "target", None)

    report["slug"] = args.slug
    out = deck_dir(args.slug) / "bracket_report.json"

    # A deck's target bracket is a property of the DECK, not of this invocation.
    # Without this, `bracket-check <slug>` with no --target silently rewrote the
    # tracked report and dropped `target`, `within_target` and `cut_candidates` —
    # the file went from answering "is this deck inside its bracket" to answering
    # only "what is its floor", and nothing said so. It happened twice to hapatra
    # in one session, each time as a side effect of an agent re-deriving a figure.
    if target is None and out.exists():
        try:
            with open(out) as f:
                target = json.load(f).get("target")
        except (OSError, ValueError):      # pragma: no cover — unreadable report
            target = None

    if target is not None:
        report["target"] = target
        report["within_target"] = report["floor"] <= target
        report["cut_candidates"] = offending_cards(report, target)

    # Write the assessment as a ◆ artifact, not just stdout. A floor that only
    # exists in a terminal cannot be cited by validate-build, cached as a
    # routine input, or rendered into a manual.
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    if getattr(args, "as_json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(args.slug, report, target))
        print(f"  Wrote {out}")

    if target is not None and report["floor"] > target:
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot bracket-check <slug>`.")
