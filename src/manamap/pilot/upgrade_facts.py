"""Pilot: what the card pool would offer a deck that has no sideboard.

`sideboard-facts` answers "what would each bench card do if run" — and stops
dead when the bench is empty. This module is the empty-bench counterpart: the
deterministic brief for the `upgrade-scout` agent, computed over the WHOLE
card pool instead of a sideboard. Three evidence channels, all from tracked
artifacts, none of them opinions:

  * **Obsolescence** — `obsolescence_index.json` maps an old card to strictly
    better replacements (same job, newer, at least one concrete advantage).
    Keyed on each maindeck card, it is the "this slot has a straight upgrade"
    channel. The index gates colour *requirement*, not colour *identity*, so
    identity is re-checked here against the commander.
  * **Combo openers** — pool cards that complete a `combo_details.json` line
    whose every other piece is already in the deck. Candidates, never facts:
    the same "needs a stack scenario" wording every other artifact uses.
  * **Synergy** — pool cards whose `synergy_graph.json` shortlist names cards
    this deck runs. The graph is a format-wide top-10, not a fit score; two or
    more in-deck partners is the reportable signal.

The division of labour matches sideboard-facts: this module says what the pool
*contains*; the `upgrade-scout` agent decides what is worth wanting and why.
Computed on demand, never committed — the same rule artist_credits states.
"""

import csv
import json

from manamap.config import (
    CARD_ROLES_PATH,
    DECK_ROLE_BUDGET,
    DECK_ROLE_GROUPS,
    OBSOLESCENCE_INDEX_PATH,
    OUTPUT_CSV_PATH,
)
from manamap.pilot.artist_credits import is_accessory
from manamap.pilot.bracket import is_infinite, load_reference
from manamap.pilot.common import (
    deck_dir,
    load_card_roles,
    load_combo_details,
    load_deck_cards,
    load_json,
    load_json_memo,
    load_synergy_graph,
    mainboard,
    sideboard,
)

# Un-set / sticker printings carry legal_commander == "legal" in Scryfall's
# data; build_deck.candidate_pool drops them and so does this brief.
UNSET_CODES = {"unf", "sunf", "ust", "ugl", "unh", "und", "ulst"}

# Keep each evidence channel readable for an agent brief. These are floors on
# nothing — the full artifacts stay one Read away.
MAX_OBSOLESCENCE = 25
MAX_COMBO_OPENERS = 25
MAX_SYNERGY = 25
MIN_SYNERGY_PARTNERS = 2


def load_pool():
    """name -> {color_identity, legal, edhrec_rank, game_changer, type_line, cmc, mana_cost}.

    Read with csv rather than pandas: this module needs a name-keyed lookup,
    not a frame, and the pilot subsystem otherwise avoids the pandas import.
    Returns None when cards.csv is absent (fresh clone, no pipeline run).
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
                "color_identity": set(row.get("color_identity") or ""),
                "legal": row.get("legal_commander") == "legal",
                "edhrec_rank": int(float(row["edhrec_rank"])) if row.get("edhrec_rank") else None,
                "game_changer": (row.get("game_changer") or "").lower() == "true",
                "type_line": row.get("type_line", ""),
                "cmc": float(row["cmc"]) if row.get("cmc") else 0.0,
                "mana_cost": row.get("mana_cost", ""),
            }
    return pool


def _eligible(name, pool, identity, deck_names):
    """Playable in this deck and not already in it."""
    info = pool.get(name)
    return (
        info is not None
        and name not in deck_names
        and info["legal"]
        and info["color_identity"] <= identity
    )


def _candidate(name, pool, roles):
    info = pool[name]
    return {
        "name": name,
        "mana_cost": info["mana_cost"],
        "cmc": info["cmc"],
        "type_line": info["type_line"],
        "roles": roles.get(name, []),
        "edhrec_rank": info["edhrec_rank"],
        "game_changer": info["game_changer"],
    }


def obsolescence_upgrades(main_names, pool, identity, roles):
    """Deck cards with a strictly-better replacement available in this identity."""
    index = load_json(OBSOLESCENCE_INDEX_PATH, default=None)
    if index is None:
        return None
    hits = []
    for deck_card in main_names:
        for rep in (index.get(deck_card) or {}).get("obsoleted_by", []):
            name = rep["name"]
            if not _eligible(name, pool, identity, set(main_names)):
                continue
            hits.append({
                **_candidate(name, pool, roles),
                "upgrades": deck_card,
                "advantages": rep.get("advantages", []),
                "similarity": rep.get("similarity"),
            })
    hits.sort(key=lambda h: (-(h["similarity"] or 0), h["name"]))
    return hits[:MAX_OBSOLESCENCE]


def combo_openers(main_names, pool, identity, roles):
    """Pool cards that complete a combo line the deck is one piece short of."""
    details = load_combo_details()
    deck = set(main_names)
    openers = {}
    for name, indices in details.get("by_card", {}).items():
        if not _eligible(name, pool, identity, deck):
            continue
        for idx in indices:
            combo = details["combos"][idx]
            others = set(combo["cards"]) - {name}
            if not others or not others <= deck:
                continue
            entry = openers.setdefault(name, {**_candidate(name, pool, roles),
                                              "lines": []})
            entry["lines"].append({
                "cards": combo["cards"],
                "produces": combo.get("produces", []),
                "bracket": combo.get("bracket"),
                "popularity": combo.get("popularity"),
                "infinite": is_infinite(combo),
                # Never a fact until a stack artifact says so.
                "status": "needs a stack scenario",
            })
    ranked = sorted(
        openers.values(),
        key=lambda e: (-max((line["popularity"] or 0) for line in e["lines"]),
                       e["name"]),
    )
    for entry in ranked:
        entry["lines"].sort(key=lambda line: (-(line["popularity"] or 0), line["cards"]))
    return ranked[:MAX_COMBO_OPENERS]


def synergy_candidates(main_names, pool, identity, roles):
    """Pool cards whose synergy shortlist names cards this deck runs."""
    graph = load_synergy_graph()
    deck = set(main_names)
    out = []
    for name, entries in graph.items():
        if not _eligible(name, pool, identity, deck):
            continue
        partners = [e for e in entries if e.get("partner") in deck]
        if len(partners) < MIN_SYNERGY_PARTNERS:
            continue
        out.append({
            **_candidate(name, pool, roles),
            "partners_in_deck": [
                {"partner": e["partner"], "score": e.get("score"),
                 "synergies": e.get("synergies", [])}
                for e in partners
            ],
        })
    out.sort(key=lambda e: (-len(e["partners_in_deck"]),
                            e["edhrec_rank"] or 10**9, e["name"]))
    return out[:MAX_SYNERGY]


def role_budget_diff(main, roles):
    """Deck role-group counts against the deterministic builder's budget.

    The budget is the PROVISIONAL, uncited one build_deck uses — reported as
    context, never as evidence. Rollup mirrors build_deck.role_group():
    most-specific mapping wins, everything unmapped lands in flex.
    """
    role_to_group = {role: group
                     for group, members in DECK_ROLE_GROUPS.items()
                     for role in members}
    have = {group: 0 for group in DECK_ROLE_BUDGET}
    for card in main:
        group = next((role_to_group[r] for r in roles.get(card["name"], [])
                      if r in role_to_group), None)
        if group is None:
            group = "lands" if "Land" in card.get("type_line", "") else "flex"
        have[group] = have.get(group, 0) + 1
    return {
        "budget": dict(DECK_ROLE_BUDGET),
        "have": have,
        "short": {g: DECK_ROLE_BUDGET[g] - have.get(g, 0)
                  for g in DECK_ROLE_BUDGET
                  if have.get(g, 0) < DECK_ROLE_BUDGET[g]},
        "note": ("DECK_ROLE_BUDGET is provisional and uncited; role coverage is a "
                 "floor (unclassified cards are invisible to it). Context, not evidence."),
    }


def analyze(slug):
    # The Short List serves every deck: benches under ten fill from the pool,
    # and even a full bench's picks get audited against pool alternatives —
    # so the old "has a sideboard -> not available" gate is gone.
    doc = load_deck_cards(slug)
    cards = doc.get("cards", [])

    pool = load_pool()
    if pool is None:
        return {
            "slug": slug,
            "available": False,
            "reason": "requires a pipeline run (data/cards.csv is absent on a fresh clone)",
        }

    main = mainboard(cards)
    main_names = [c["name"] for c in main]
    commanders = [c["name"] for c in main if c.get("is_commander")]
    identity = set()
    for c in main:
        if c.get("is_commander"):
            identity.update(c.get("color_identity") or [])
    roles = load_card_roles() if CARD_ROLES_PATH.exists() else {}

    floor = (load_json(deck_dir(slug) / "bracket_report.json", default={}) or {}).get("floor")

    facts = {
        "slug": slug,
        "available": True,
        "commander": commanders,
        "color_identity": sorted(identity),
        "current_bracket_floor": floor,
        "pool_size": sum(1 for n, i in pool.items()
                         if i["legal"] and i["color_identity"] <= identity
                         and n not in set(main_names)),
        "obsolescence_upgrades": obsolescence_upgrades(main_names, pool, identity, roles),
        "combo_openers": combo_openers(main_names, pool, identity, roles),
        "synergy_candidates": synergy_candidates(main_names, pool, identity, roles),
        "role_budget": role_budget_diff(main, roles),
    }
    facts["notes"] = build_notes(facts)
    return facts


def build_notes(facts):
    notes = []
    if facts["obsolescence_upgrades"] is None:
        notes.append("obsolescence_index.json is absent — the straight-upgrade "
                     "channel is empty for that reason, not because none exist.")
    infinite_openers = [e["name"] for e in facts["combo_openers"]
                        if any(line["infinite"] for line in e["lines"])]
    if infinite_openers:
        notes.append(
            "Completes an infinite-flagged line (all unverified candidates): "
            + ", ".join(infinite_openers)
            + ". A line is fact only once a stack artifact carries "
              "checker.verdict == 'pass'."
        )
    gc = [e["name"] for e in facts["combo_openers"] + (facts["obsolescence_upgrades"] or [])
          + facts["synergy_candidates"] if e.get("game_changer")]
    if gc:
        notes.append(
            "Game Changer(s) among the candidates: " + ", ".join(sorted(set(gc)))
            + ". Adding one moves the computed floor — a fact about which table "
              "the deck belongs at, reported here and decided by the pilot."
        )
    notes.append("The synergy graph is a format-wide top-10 shortlist, not a "
                 "per-deck fit score; partner counts locate candidates, they do "
                 "not rank them.")
    return notes


def format_report(facts):
    if not facts.get("available"):
        return f"{facts['slug']}: {facts['reason']}"
    lines = [f"{facts['slug']}: pool scout brief — {facts['pool_size']} eligible pool "
             f"card(s), floor {facts['current_bracket_floor']}"]
    obs = facts["obsolescence_upgrades"] or []
    lines.append(f"  straight upgrades (obsolescence): {len(obs)}")
    for e in obs[:8]:
        lines.append(f"    {e['name']} upgrades {e['upgrades']} "
                     f"({', '.join(e['advantages'])})")
    lines.append(f"  combo openers: {len(facts['combo_openers'])} — all lines unverified")
    for e in facts["combo_openers"][:8]:
        lines.append(f"    {e['name']} completes {len(e['lines'])} line(s)")
    lines.append(f"  synergy candidates (>= {MIN_SYNERGY_PARTNERS} in-deck partners): "
                 f"{len(facts['synergy_candidates'])}")
    for e in facts["synergy_candidates"][:8]:
        lines.append(f"    {e['name']} — {len(e['partners_in_deck'])} partner(s) in deck")
    short = facts["role_budget"]["short"]
    if short:
        lines.append("  role groups under the (provisional) budget: "
                     + ", ".join(f"{g} -{n}" for g, n in sorted(short.items())))
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
    raise SystemExit("Run via `manamap pilot upgrade-facts <slug>`.")
