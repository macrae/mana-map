"""Pilot: what each sideboard card would actually do if you ran it.

A sideboard renders today as a flat tile strip — no swap suggestion, no curve or
pip impact, no bracket delta, not even a synergy chip. Every question a pilot
actually asks about a sideboard ("is this better than what I'm running?", "does
it turn on a line?", "does it push me out of tier?") has been left to prose.

Most of those questions are computable, and computing them is what keeps the
answer honest: a swap that raises your bracket floor is a fact, not an opinion.
The division of labour is deliberate — this module says what each card *is* and
what it *would add*; the `sideboard-analyst` agent decides what to cut for it.

Two things make this cheap rather than a rewrite:

  * `bracket.assess()` takes a plain name list. The sideboard filter lives in
    bracket.main(), not in assess(), so a hypothetical "what if I ran this"
    floor costs one extra call with the card appended.
  * `combos_in_deck()` is likewise name-list based, so the lines a card
    *completes* are exactly the set difference against the deck without it.

Computed on demand, never committed — the same rule artist_credits states.
"""

import json
from collections import Counter

from manamap.analysis.common import parse_tag_set
from manamap.config import CARD_ROLES_PATH
from manamap.pilot.artist_credits import is_accessory
from manamap.pilot.bracket import assess, combos_in_deck, is_infinite, load_reference
from manamap.pilot.common import (
    deck_dir,
    is_land,
    load_card_roles,
    load_deck_cards,
    mainboard,
    sideboard,
)
from manamap.pilot.manabase import WUBRG, count_pips


def split_deck(doc):
    """(maindeck, real sideboard cards, accessories).

    Accessories are Secret Lair table aids — `Red Mana`, `Storm Counter` — with
    type_line "Card" and no rules text. Asking "what would this do if I ran it"
    of a counter token is meaningless, so they are listed and never analysed.
    Predicate reused from artist_credits rather than re-implemented.
    """
    cards = doc.get("cards", [])
    side = sideboard(cards)
    return (mainboard(cards),
            [c for c in side if not is_accessory(c)],
            [c for c in side if is_accessory(c)])


def _roles():
    if not CARD_ROLES_PATH.exists():
        return {}
    return load_card_roles()


def card_pips(card):
    """Non-zero coloured pips in this card's cost."""
    pips = count_pips(card.get("mana_cost", ""))
    return {c: pips[c] for c in WUBRG if pips.get(c)}


def lines_opened(main_names, card_name, details):
    """Combo lines that become fully contained once this card is in the deck.

    Set difference, not a fresh scan: a line already complete without the card
    is not something the sideboard 'opens'.
    """
    before = set(combos_in_deck(main_names, details))
    after = combos_in_deck(list(main_names) + [card_name], details)
    opened = []
    for idx in after:
        if idx in before:
            continue
        combo = details["combos"][idx]
        opened.append({
            "cards": combo["cards"],
            "produces": combo.get("produces", []),
            "bracket": combo.get("bracket"),
            "popularity": combo.get("popularity"),
            "infinite": is_infinite(combo),
            # Never a fact until a stack artifact says so. Same wording the
            # architect and the strategic frame use, so one grep finds them all.
            "status": "needs a stack scenario",
        })
    opened.sort(key=lambda line: (-(line["popularity"] or 0), line["cards"]))
    return opened


def analyze(slug):
    doc = load_deck_cards(slug)
    main, side, accessories = split_deck(doc)

    if not side:
        return {
            "slug": slug,
            "available": False,
            "reason": ("no analysable sideboard — "
                       f"{len(accessories)} table accessor{'y' if len(accessories) == 1 else 'ies'}, "
                       "0 real cards" if accessories else "no sideboard"),
            "accessories": sorted(c["name"] for c in accessories),
        }

    main_names = [c["name"] for c in main]
    commanders = [c["name"] for c in main if c.get("is_commander")]
    roles = _roles()

    try:
        card_flags, role_map, details = load_reference()
        base = assess(main_names, card_flags, role_map, details, commanders)
        base_floor = base["floor"]
        have_reference = True
    except SystemExit:
        card_flags = role_map = details = None
        base_floor = None
        have_reference = False

    identity = set()
    for c in main:
        if c.get("is_commander"):
            identity.update(c.get("color_identity") or [])

    entries = []
    for card in side:
        name = card["name"]
        entry = {
            "name": name,
            "mana_cost": card.get("mana_cost", ""),
            "cmc": card.get("cmc"),
            "type_line": card.get("type_line", ""),
            "oracle_text": card.get("oracle_text", ""),
            "colors": card.get("colors") or [],
            "color_identity": card.get("color_identity") or [],
            "roles": roles.get(name, []),
            "tags": sorted(parse_tag_set(card.get("mechanical_tags"))),
            "pips": card_pips(card),
            "is_land": is_land(card),
            "edhrec_rank": card.get("edhrec_rank"),
            # A sideboard card outside the commander's identity is unplayable here.
            # validate_deck exempts the sideboard, so nothing else catches this.
            "in_color_identity": set(card.get("color_identity") or []) <= identity
            if identity else True,
        }
        if have_reference:
            hypothetical = assess(main_names + [name], card_flags, role_map,
                                  details, commanders)
            entry["bracket_if_added"] = {
                "before": base_floor,
                "after": hypothetical["floor"],
                "delta": hypothetical["floor"] - base_floor,
                "drivers_added": [
                    d for d in hypothetical.get("drivers", []) if d not in base.get("drivers", [])
                ],
            }
            entry["opens_lines"] = lines_opened(main_names, name, details)
        entries.append(entry)

    facts = {
        "slug": slug,
        "available": True,
        "commander": commanders,
        "maindeck_entries": len(main),
        "sideboard": entries,
        "accessories": sorted(c["name"] for c in accessories),
        "current_bracket_floor": base_floor,
        "role_gaps": sorted(_role_gaps(main, roles)),
    }
    facts["notes"] = build_notes(facts)
    return facts


def _role_gaps(main, roles):
    """Roles the maindeck is thin on — context for whether a swap fills a hole."""
    hist = Counter()
    for card in main:
        for role in roles.get(card["name"], []):
            hist[role] += 1
    return {role for role, n in hist.items() if n <= 1}


def build_notes(facts):
    notes = []
    if facts["accessories"]:
        notes.append(
            f"{len(facts['accessories'])} table accessor"
            f"{'y' if len(facts['accessories']) == 1 else 'ies'} "
            f"({', '.join(facts['accessories'])}) are listed but not analysed — "
            f"they have no rules text and cannot be swapped in."
        )
    off_identity = [e["name"] for e in facts["sideboard"] if not e["in_color_identity"]]
    if off_identity:
        notes.append(
            "OUTSIDE THE COMMANDER'S COLOUR IDENTITY and therefore unplayable in this "
            f"deck: {', '.join(off_identity)}. validate-deck exempts the sideboard, so "
            "nothing else reports this."
        )
    raised = [e for e in facts["sideboard"]
              if e.get("bracket_if_added", {}).get("delta", 0) > 0]
    if raised:
        notes.append(
            "Raises the bracket floor if run: "
            + "; ".join(f"{e['name']} {e['bracket_if_added']['before']}"
                        f"->{e['bracket_if_added']['after']}" for e in raised)
            + ". That is a computed floor, not a judgment — a swap that changes it "
              "changes what table the deck belongs at."
        )
    openers = [e for e in facts["sideboard"] if e.get("opens_lines")]
    if openers:
        notes.append(
            "Completes combo line(s) the deck cannot currently assemble: "
            + "; ".join(f"{e['name']} ({len(e['opens_lines'])})" for e in openers)
            + ". Every one is a candidate — a line is fact only once a stack "
              "artifact carries checker.verdict == 'pass'."
        )
    unranked = [e["name"] for e in facts["sideboard"] if e.get("edhrec_rank") is None]
    if unranked:
        notes.append(
            f"No EDHREC rank for {', '.join(unranked)} — the popularity signal is "
            "absent, so a 'stronger long-term default' claim about these has to rest "
            "on the deck's own needs rather than on how often others run them."
        )
    return notes


def format_report(facts):
    if not facts.get("available"):
        return f"{facts['slug']}: {facts['reason']}"
    lines = [f"{facts['slug']}: {len(facts['sideboard'])} analysable sideboard card(s), "
             f"{len(facts['accessories'])} accessor{'y' if len(facts['accessories']) == 1 else 'ies'}"]
    for e in facts["sideboard"]:
        bracket = e.get("bracket_if_added")
        delta = ""
        if bracket:
            delta = (f", bracket {bracket['before']}->{bracket['after']}"
                     if bracket["delta"] else f", bracket {bracket['after']} (no change)")
        lines.append(f"  {e['name']} — {e['mana_cost']} {e['type_line']}{delta}")
        if e.get("roles"):
            lines.append(f"      roles: {', '.join(e['roles'])}")
        if e.get("opens_lines"):
            lines.append(f"      opens {len(e['opens_lines'])} combo line(s) — all unverified")
        if not e["in_color_identity"]:
            lines.append("      OUTSIDE COLOUR IDENTITY — unplayable in this deck")
    for note in facts.get("notes", []):
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(args):
    facts = analyze(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(facts))
    out = getattr(args, "out", None)
    if out:
        path = deck_dir(args.slug) / out if "/" not in str(out) else out
        with open(path, "w") as f:
            json.dump(facts, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path}")



if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot sideboard-facts <slug>`.")
