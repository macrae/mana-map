"""Pilot: the deterministic brief a `deck-engineer` reads before reasoning.

WHY THIS EXISTS. Four artifacts already say something true about a deck's engine
and none of them says all of it, in the same shape:

  * `goldfish_targets.json` declares components as `any_of` groups, and
    `deck_audit.engine_activation` prices and measures them.
  * checker-passed `stacks/*.json` are the only FACT-tier statement that a
    specific multi-card pairing works.
  * `combo_details.json:by_card` indexes every line a card participates in.
  * `deck_map.json` clusters the deck by what its cards SAY.

The engine is defined by what cards do TO EACH OTHER, which is a different
relation from all four — so an agent has to reason across them. This module hands
it every joinable fact first, for the reason `deck-facts` exists: an agent that
re-derives a figure is both more expensive and, measured across one previous
session, wrong five times.

THE SCATTER TABLE IS THE POINT OF THE BRIEF. Every declared component is walked
against the deck map's cities, so the agent STARTS from the disagreement rather
than discovering it. On radagast the metronome class spans five cities and the
flash traps span five — the clusters are an input to engine analysis, not the
analysis.

Computed on demand, never committed — same rule as `deck-facts` and `deck-audit`,
because it embeds goldfish and audit figures that go stale the moment the
decklist moves.
"""

import json
from itertools import combinations

from manamap.pilot.build_index import line_cards
from manamap.pilot.common import (
    deck_dir,
    load_card_roles,
    load_combo_details,
    load_deck_cards,
    load_json,
    presentable,
    resolve_out_path,
)

# A card in this many checker-passed scenarios is load-bearing regardless of what
# any taxonomy calls it. Same quorum `validate_goldfish_targets` uses for its
# win-line coverage check — one number, one meaning.
LOAD_BEARING_QUORUM = 2


# ── Verified pairings: the only fact tier ───────────────────────────────


def verified_pairings(slug, deck_names):
    """Card pairs co-named in a checker-passed scenario, with the stacks proving it.

    `line_cards` is reused rather than re-implemented: it already encodes the rule
    for when a `board` entry is a line piece and when it is furniture, which got
    edgar and heliod wrong in opposite directions before it was settled. A second
    answer to that question here would be a second thing to keep correct.

    A pair is evidence that those two cards were on the table together in a line a
    rules checker passed. It is NOT evidence that either causes the other — that
    reading is the agent's job, and the note is where it argues for it.
    """
    pairs, per_card, per_stack = {}, {}, {}
    directory = deck_dir(slug) / "stacks"
    if not directory.is_dir():
        return {"pairs": [], "by_card": {}, "by_stack": {}}
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text())
        if not presentable(doc):
            continue
        sid = doc.get("id") or path.stem[:3]
        named = [c for c in line_cards(doc.get("scenario") or {}, deck_names)
                 if c in deck_names]
        per_stack[sid] = {"title": doc.get("title", ""), "cards": sorted(named)}
        for card in named:
            per_card.setdefault(card, []).append(sid)
        for a, b in combinations(sorted(set(named)), 2):
            pairs.setdefault((a, b), []).append(sid)
    return {
        "pairs": [{"cards": [a, b], "stacks": sorted(set(s))}
                  for (a, b), s in sorted(pairs.items())],
        "by_card": {k: sorted(set(v)) for k, v in sorted(per_card.items())},
        "by_stack": per_stack,
    }


# ── Contained combo lines: candidates, never facts ──────────────────────


def contained_lines(deck_names):
    """Combo lines every card of which is in the 99.

    Deduped on `frozenset(cards)`, because `combo_details.json` holds several
    RECORDS per interaction — a box that contains 31 distinct lines reports 33
    without this, which is a documented trap rather than a hypothetical one.

    Format-agnostic by construction: a Commander Spellbook line may assume a card
    is your commander ("Infinite commander casts" in `produces` is the tell), so
    every line here is a CANDIDATE that needs a stack scenario before anyone
    states it as fact. That flag rides along rather than being filtered out — the
    agent should see the line and the caveat together.
    """
    details = load_combo_details() or {}
    by_card = details.get("by_card") or {}
    combos = details.get("combos") or []
    seen, out = set(), []
    for card in deck_names:
        for idx in by_card.get(card, []):
            if idx >= len(combos):
                continue
            combo = combos[idx]
            cards = combo.get("cards") or []
            if not cards or not set(cards) <= deck_names:
                continue
            key = frozenset(cards)
            if key in seen:
                continue
            seen.add(key)
            produces = combo.get("produces") or []
            out.append({
                "cards": sorted(cards),
                "produces": produces,
                "bracket": combo.get("bracket"),
                "assumes_commander": any("commander" in str(p).lower()
                                         for p in produces),
                "status": "needs a stack scenario",
            })
    return sorted(out, key=lambda c: (-len(c["cards"]), c["cards"]))


# ── The scatter table: where the declaration meets the map ──────────────


def component_scatter(audit_engine, deck_map):
    """For each declared component, how many cities of the map it lands in.

    This is the finding the whole subsystem was built around, computed rather
    than asserted: a component that spans five cities is not a place, and naming
    it after one of them is wrong in a way that reads as correct.
    """
    if not deck_map:
        return []
    city_of = {c["name"]: c["city"] for c in deck_map.get("cards", [])}
    label = {}
    for region in deck_map.get("regions", []):
        if region.get("level") == 0:
            label[int(region["id"].rsplit("-", 1)[-1])] = (
                region.get("label") or region.get("fallback") or region["id"])

    rows = []
    for target in audit_engine.get("targets", []):
        for component in target.get("components", []):
            spread = {}
            for card in component.get("cards", []):
                city = city_of.get(card)
                if city is not None:
                    spread[label.get(city, str(city))] = \
                        spread.get(label.get(city, str(city)), 0) + 1
            rows.append({
                "target": target.get("label", ""),
                "size": component.get("size"),
                "thin": component.get("thin"),
                "placed": sum(spread.values()),
                "cities": len(spread),
                "spread": dict(sorted(spread.items(), key=lambda kv: -kv[1])),
                "coherent": len(spread) <= 1,
            })
    return rows


# ── Build ───────────────────────────────────────────────────────────────


def build(slug):
    from manamap.pilot.card_pool import load_pool
    from manamap.pilot.deck_audit import engine_activation

    deck = load_deck_cards(slug)
    cards = deck["cards"]
    deck_names = {c["name"] for c in cards}
    roles = load_card_roles()
    # A SET, not a sorted list: `deck_audit._closers` tests `info["color_identity"]
    # <= identity`, which is subset containment and raises on a list.
    identity = {c for card in cards for c in (card.get("color_identity") or [])}

    engine = engine_activation(slug, cards, roles, identity, load_pool)
    base = deck_dir(slug)
    deck_map = load_json(base / "deck_map.json")
    frame = load_json(base / "strategic_frame.json") or {}
    targets = load_json(base / "goldfish_targets.json") or {}

    pairings = verified_pairings(slug, deck_names)
    load_bearing = sorted(
        [card for card, stacks in pairings["by_card"].items()
         if len(stacks) >= LOAD_BEARING_QUORUM])

    cities = {}
    if deck_map:
        for region in deck_map.get("regions", []):
            if region.get("level") == 0:
                cities[region["id"]] = {
                    "label": region.get("label") or region.get("fallback"),
                    "gloss": region.get("gloss"),
                    "count": region.get("count"),
                    "verified_count": region.get("verified_count"),
                    "cards": region.get("cards", []),
                }

    return {
        "slug": slug,
        "decklist_sha256": deck.get("decklist_sha256"),
        "engine_declaration": {
            "targets": [t.get("label") for t in targets.get("targets", [])],
            "note": targets.get("_note"),
        },
        "audit_engine": engine,
        "verified": {
            "pairings": pairings["pairs"],
            "by_card": pairings["by_card"],
            "by_stack": pairings["by_stack"],
            "load_bearing_cards": load_bearing,
        },
        "candidate_lines": contained_lines(deck_names),
        "map": {"cities": cities,
                "scatter": component_scatter(engine, deck_map)},
        "declared_engines": frame.get("engines", []),
        "roles": {c["name"]: roles.get(c["name"], []) for c in cards},
        "notes": _notes(engine, pairings, deck_names),
    }


def _notes(engine, pairings, deck_names):
    """The traps, named — the same service `deck-facts.notes[]` performs."""
    notes = []
    n = len(pairings["pairs"])
    if not n:
        notes.append(
            "No checker-passed stack names two deck cards together, so this deck "
            "has NO fact-tier pairing evidence. Every line you assert is a claim "
            "and belongs in open_questions with settled_by: resolve-stack.")
    spofs = engine.get("single_points_of_failure") or []
    if spofs:
        notes.append(
            f"{len(spofs)} single point(s) of failure already priced by deck-audit: "
            # `component` is the component OBJECT, not its name — it carries the
            # cards, which is what a reader of this note actually wants.
            + ", ".join(sorted({
                ", ".join((s.get("component") or {}).get("cards") or ["?"])
                for s in spofs}))
            + ". A stage built on one of these is one removal spell from off.")
    notes.append(
        "candidate_lines are Commander Spellbook records, not verified lines. A "
        "line whose `assumes_commander` is true may be counting casts you only get "
        "as the commander. None is a fact until a stack passes.")
    notes.append(
        "The synergy graph is NOT in this brief on purpose: it is a format-wide "
        "retrieval shortlist, and its edges are not a statement that two cards work "
        "together in THIS deck. Query it yourself to find candidates; never cite it "
        "as evidence.")
    return notes


def main(args):
    doc = build(args.slug)
    requested = getattr(args, "out", None)
    if requested:
        path = resolve_out_path(requested, args.slug, "engine-facts")
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")
        return
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return

    engine = doc["audit_engine"]
    print(f"ENGINE FACTS — {doc['slug']}")
    print(f"  declared targets      {len(doc['engine_declaration']['targets'])}")
    print(f"  verified pairings     {len(doc['verified']['pairings'])} "
          f"across {len(doc['verified']['by_stack'])} passing stack(s)")
    print(f"  load-bearing cards    {len(doc['verified']['load_bearing_cards'])} "
          f"(in >= {LOAD_BEARING_QUORUM} passing stacks)")
    print(f"  candidate combo lines {len(doc['candidate_lines'])}")
    print(f"  single points of fail {len(engine.get('single_points_of_failure') or [])}")
    scatter = doc["map"]["scatter"]
    if scatter:
        coherent = sum(1 for r in scatter if r["coherent"])
        print(f"\n  COMPONENT vs THE MAP — {coherent}/{len(scatter)} components "
              f"sit in a single city:")
        for row in sorted(scatter, key=lambda r: -r["cities"]):
            mark = "OK " if row["coherent"] else "   "
            print(f"   {mark}{row['target'][:40]:42s} {row['placed']:3d} cards "
                  f"-> {row['cities']} cities")
    for note in doc["notes"]:
        print(f"\n  NOTE {note}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot engine-facts <slug>`.")
