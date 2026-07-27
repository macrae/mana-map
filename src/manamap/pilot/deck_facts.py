"""Pilot: the deterministic facts every agent would otherwise re-derive by hand.

Nine agents work on a deck and none of them shares a brief. Each one opens
`cards.json`, counts colours, works out which permanents are legendary, builds a
curve, tallies pips and looks up roles — and on sisay five separate agents derived
the same tutor ladder independently, each paying tokens for arithmetic Python does
in milliseconds. Worse, each rediscovers the same traps: that `synergy_graph.json`
is a *global* top-10 shortlist with zero intra-deck edges, that a card missing from
`card_roles.json` has no role rather than no function, that some mana cannot pay an
activated ability at all.

This command answers all of it once, deterministically, and says the traps out loud.

**Computed on demand, never committed.** `artist_credits.py` states the rule this
follows: a second copy of facts already in `cards.json` could only ever desync. The
output is a view, not an artifact — `--out` exists for piping into a scratchpad, not
for tracking in git.

Everything here composes primitives that already exist. No new analysis lives in
this module:

    colours          ingest/extract.py       get_colors      (DFC-correct)
    pips, sources    pilot/manabase.py       count_pips, pip_requirements,
                                             source_targets, land_colors, enters_tapped
    combos           pilot/bracket.py        load_reference, combos_in_deck, is_infinite
    tags             analysis/common.py      parse_tag_set
"""

import json
from collections import Counter

from manamap.analysis.common import parse_tag_set
from manamap.config import CARD_ROLES_PATH, SYNERGY_GRAPH_PATH
from manamap.ingest.extract import get_colors
from manamap.pilot.bracket import combos_in_deck, is_infinite
from manamap.pilot.common import (
    deck_dir,
    is_land,
    load_card_roles,
    load_combo_details,
    load_deck_cards,
    load_synergy_graph,
    mainboard,
)
from manamap.pilot.manabase import (
    RESTRICTED_MANA as RESTRICTED_PHRASE,
    WUBRG,
    count_pips,
    enters_tapped,
    land_colors,
    pip_requirements,
    source_targets,
)

# A permanent type line, for "which of these can a Sisay-style tutor actually find".
PERMANENT_TYPES = ("Creature", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle")


def _front(card, key, default=""):
    """A field from the front face, falling back to the whole card.

    The face that is up on the battlefield is the one that contributes colours and
    type, and it is not always what the top-level field says: Esika's card colours
    are the union of both faces, but the permanent you control is mono-green.
    """
    faces = card.get("card_faces") or []
    if faces and faces[0].get(key):
        return faces[0][key]
    return card.get(key, default)


_is_land = is_land  # canonical predicate lives in common; alias kept for callers


def _is_legendary(card):
    return "Legendary" in str(_front(card, "type_line"))


def _is_permanent(card):
    line = str(_front(card, "type_line"))
    return any(t in line for t in PERMANENT_TYPES)


def colours(cards):
    """Per-card colours, resolved correctly for multi-face cards.

    Two answers, because two questions get asked. `card` is the union across faces
    (what cards.csv reports, and what "is this a red card" means for deckbuilding).
    `front` is the face-up permanent's colours, which is what an ability counting
    "colours among permanents you control" actually sees.
    """
    out = {}
    for card in cards:
        union = get_colors(card)
        front = [c for c in WUBRG if c in set(count_pips(_front(card, "mana_cost")))
                 and count_pips(_front(card, "mana_cost"))[c] > 0]
        entry = {"card": union}
        if front != union:
            entry["front_face"] = front
        out[card["name"]] = entry
    return out


def curve(cards):
    """Mana-value histogram of the nonland cards, as string keys for stable JSON."""
    hist = Counter(int(c.get("cmc") or 0) for c in cards if not _is_land(c))
    return {str(k): hist[k] for k in sorted(hist)}


def _makes_colour(line):
    """Does this mana line produce anything other than colourless?"""
    lowered = line.lower()
    if "any color" in lowered or "any of this creature's colors" in lowered:
        return True
    return any(f"{{{c.lower()}}}" in lowered for c in WUBRG)


def classify_mana_restriction(card):
    """How a 'spend this mana only' clause constrains what the mana can pay for.

    Blunt detection is worse than none here, because the distinction is exactly
    what trips pilots up. Three cards can all carry the phrase and mean different
    things:

      Unclaimed Territory  "...only to cast a creature spell of the chosen type"
                           → spells only. Cannot pay an activated ability, ever.
      Secluded Courtyard   "...to cast a creature spell of the chosen type OR
                           ACTIVATE AN ABILITY of a creature source..."
                           → pays abilities fine. One clause apart from the above.
      Plaza of Heroes      one restricted mode, plus other modes with no clause
                           → unrestricted mana is available; pick the right mode.

    Returns None when nothing is restricted, else a classification dict.
    """
    text = str(card.get("oracle_text", "") or "")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    restricted_lines = [ln for ln in lines if RESTRICTED_PHRASE in ln.lower()]
    if not restricted_lines:
        return None
    # An unrestricted mode is only an escape hatch if it makes COLOUR. Delighted
    # Halfling and Unclaimed Territory both have one — "{T}: Add {C}" — and it pays
    # exactly none of a coloured cost, which is the whole trap.
    unconditional_coloured = [
        ln for ln in lines
        if "add " in ln.lower()
        and RESTRICTED_PHRASE not in ln.lower()
        and _makes_colour(ln)
    ]
    pays_abilities = any("activate an ability" in ln.lower() for ln in restricted_lines)
    if pays_abilities:
        verdict = "pays_abilities"
    elif unconditional_coloured:
        verdict = "has_unrestricted_coloured_mode"
    else:
        verdict = "spells_only"
    return {
        "name": card["name"],
        "verdict": verdict,
        "restricted_text": restricted_lines[0],
        "unrestricted_coloured_modes": len(unconditional_coloured),
    }


def mana_facts(cards):
    """What the mana base can actually produce, and what the spells actually owe."""
    lands = [c for c in cards if _is_land(c)]
    spells = [c for c in cards if not _is_land(c)]
    reqs = pip_requirements(spells)
    sources = {c: 0 for c in WUBRG}
    for land in lands:
        for colour in land_colors(land):
            sources[colour] += 1
    # Any permanent can carry the trap, not just lands — Delighted Halfling is a dork.
    restricted = [r for r in (classify_mana_restriction(c) for c in cards) if r]
    return {
        "lands": len(lands),
        "spells": len(spells),
        "enters_tapped": sum(1 for land in lands if enters_tapped(land)),
        "pip_requirements": reqs,
        "source_targets": source_targets(reqs),
        "land_sources": sources,
        "restricted_mana": sorted(restricted, key=lambda r: r["name"]),
    }


def role_facts(names):
    """Role coverage for this deck, and the cards the taxonomy has nothing to say about.

    `card_roles.json` omits unclassified cards entirely, so a `.get(name, [])` miss is
    silent. Per-deck coverage has never been reported anywhere; on sisay the six
    uncovered cards were the entire counterspell suite.
    """
    if not CARD_ROLES_PATH.exists():
        return {"available": False}
    roles = load_card_roles()
    covered, missing, hist = {}, [], Counter()
    for name in names:
        got = roles.get(name, [])
        if got:
            covered[name] = got
            hist.update(got)
        else:
            missing.append(name)
    return {
        "available": True,
        "coverage": round(len(covered) / len(names), 4) if names else 0.0,
        "no_role": sorted(missing),
        "histogram": dict(sorted(hist.items())),
    }


def synergy_facts(names):
    """How many synergy edges actually fall inside this deck. Usually very few.

    The graph is a top-10 shortlist selected over all ~26K tagged cards, so for a
    fixed 100-card list the intersection is close to empty by construction —
    0 on sisay. Agents keep querying it hoping for a per-deck fit score; report the
    number so they stop.
    """
    if not SYNERGY_GRAPH_PATH.exists():
        return {"available": False}
    graph = load_synergy_graph()
    present = set(names)
    edges = 0
    absent = []
    for name in names:
        entry = graph.get(name)
        if entry is None:
            absent.append(name)
            continue
        edges += sum(1 for e in entry if e.get("partner") in present)
    return {
        "available": True,
        "cards_in_graph": len(names) - len(absent),
        "cards_absent": sorted(absent),
        "intra_deck_edges": edges,
    }


def combo_facts(names, commanders):
    """Combos fully contained in the deck, from combo_details (never the graph)."""
    try:
        details = load_combo_details()
    except FileNotFoundError:
        return {"available": False}
    lines = []
    for idx in combos_in_deck(names, details):
        combo = details["combos"][idx]
        lines.append({
            "cards": combo["cards"],
            "produces": combo.get("produces", []),
            "bracket": combo.get("bracket"),
            "mana_value_needed": combo.get("mana_value_needed"),
            "popularity": combo.get("popularity"),
            "infinite": is_infinite(combo),
            "includes_commander": any(c in commanders for c in combo["cards"]),
        })
    lines.sort(key=lambda line: (-(line["popularity"] or 0), line["cards"]))
    return {
        "available": True,
        "contained": len(lines),
        "two_card_infinites": sum(1 for line in lines
                                  if line["infinite"] and len(line["cards"]) == 2),
        "lines": lines,
    }


def build_notes(facts):
    """The traps, said out loud, so nobody pays tokens to rediscover them."""
    notes = []
    syn = facts["synergy"]
    if syn.get("available"):
        notes.append(
            f"synergy_graph.json has {syn['intra_deck_edges']} edge(s) falling inside this "
            f"deck. It is a top-10 shortlist selected over every tagged card in the format, "
            f"not a per-deck fit score — it cannot tell you how well a card fits this list."
        )
    roles = facts["roles"]
    if roles.get("available") and roles["no_role"]:
        notes.append(
            f"{len(roles['no_role'])} card(s) have no entry in card_roles.json: "
            f"{', '.join(roles['no_role'][:8])}"
            f"{' …' if len(roles['no_role']) > 8 else ''}. Absence of a role is not "
            f"absence of the function — the taxonomy simply has no pattern for them."
        )
    restricted = facts["mana"]["restricted_mana"]
    spells_only = [r["name"] for r in restricted if r["verdict"] == "spells_only"]
    if spells_only:
        notes.append(
            "Cannot pay an activated ability at all — the mana is restricted to casting "
            "SPELLS, and an activated ability is not a spell: "
            + ", ".join(spells_only)
            + ". These accelerate toward a commander without accelerating anything the "
              "commander then does."
        )
    nuanced = [r for r in restricted if r["verdict"] != "spells_only"]
    if nuanced:
        notes.append(
            "Restricted-looking but usable, one clause at a time: "
            + "; ".join(
                f"{r['name']} "
                + ("says 'or activate an ability', so it does pay ability costs"
                   if r["verdict"] == "pays_abilities"
                   else f"has {r['unrestricted_coloured_modes']} unrestricted COLOURED "
                        f"mode(s) alongside the restricted one")
                for r in nuanced
            )
            + ". Read the clause, not the phrase."
        )
    multiface = [n for n, c in facts["colours"].items() if "front_face" in c]
    if multiface:
        notes.append(
            "Multi-face card(s) whose face-up colours differ from the card's: "
            + ", ".join(multiface)
            + ". `card` is the union across faces; `front_face` is what a permanent "
              "on the battlefield actually contributes."
        )
    combos = facts["combos"]
    if combos.get("available") and combos["contained"]:
        notes.append(
            f"{combos['contained']} combo line(s) are fully contained in this deck and "
            f"{combos['two_card_infinites']} are two-card infinites. None is verified: a "
            f"line is fact only once a stack artifact carries checker.verdict == 'pass'."
        )
    return notes


def analyze(slug):
    doc = load_deck_cards(slug)
    cards = mainboard(doc.get("cards", []))
    names = [c["name"] for c in cards]
    commanders = [c["name"] for c in cards if c.get("is_commander")]
    legendary = [c for c in cards if _is_legendary(c)]

    facts = {
        "slug": slug,
        "commander": commanders,
        "counts": {
            # Entries, not copies — one "2 Plains" line is a single entry. Every count
            # in this block is per entry, the same convention Featured Artist uses, so
            # `entries` will not equal 100 for a deck with duplicate basics.
            "entries": len(cards),
            "copies": sum(int(c.get("quantity") or 1) for c in cards),
            "lands": sum(1 for c in cards if _is_land(c)),
            "nonland": sum(1 for c in cards if not _is_land(c)),
            "legendary": len(legendary),
            "legendary_permanents": sum(1 for c in legendary if _is_permanent(c)),
            "legendary_lands": sum(1 for c in legendary if _is_land(c)),
            "nonlegendary": len(cards) - len(legendary),
        },
        "curve": curve(cards),
        "colours": colours(cards),
        "mana": mana_facts(cards),
        "roles": role_facts(names),
        "synergy": synergy_facts(names),
        "combos": combo_facts(names, commanders),
        "tags": {
            c["name"]: sorted(parse_tag_set(c.get("mechanical_tags")))
            for c in cards if parse_tag_set(c.get("mechanical_tags"))
        },
    }
    facts["notes"] = build_notes(facts)
    return facts


def main(args):
    facts = analyze(args.slug)
    text = json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False)
    out = getattr(args, "out", None)
    if out:
        path = deck_dir(args.slug) / out if "/" not in str(out) else out
        with open(path, "w") as f:
            f.write(text + "\n")
        print(f"Wrote {path}")
    else:
        print(text)



if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-facts <slug>`.")
