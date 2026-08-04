"""Pilot: the deterministic mana/land analysis behind Sources Say (◆).

Everything here is a pure function of cards.json (+ goldfish_metrics.json for
the turn-by-turn tie-ins) — no LLM, no network, no randomness. The renderer
reads the tracked artifact `data/decks/<slug>/mana_analysis.json`; the writer's
`mana_base` prose key narrates it in Ledger's voice.

Reuses the deck-builder's hypergeometric kit (manabase.py) verbatim: the same
math that sizes a mana base during a build audits one here. Land classes come
from ROLE_LAND_PATTERNS + the derived classes card_roles.py computes globally,
applied deck-locally so a fresh clone with only cards.json can still run this.
"""

import json
import re

from manamap.config import (
    ROLE_COST_REDUCTION,
    ROLE_LAND_PATTERNS,
    ROLE_LAND_RAMP,
    ROLE_MANA_BY_SUPERTYPE,
    ROLE_MANA_SOURCE,
)
from manamap.pilot.common import (
    deck_dir,
    expand_copies,
    is_land,
    load_deck_cards,
    load_json,
    mainboard,
)
from manamap.pilot.manabase import (
    WUBRG,
    achieved_probability,
    enters_tapped,
    enters_tapped_unconditionally,
    land_colors,
    pip_requirements,
    source_targets,
)

_MANA_SOURCE_RE = re.compile(ROLE_MANA_SOURCE, re.IGNORECASE)
_LAND_RAMP_RE = re.compile(ROLE_LAND_RAMP, re.IGNORECASE)
_COST_REDUCTION_RE = re.compile(ROLE_COST_REDUCTION, re.IGNORECASE)
_LAND_RES = {name: re.compile(pattern, re.IGNORECASE)
             for name, pattern in ROLE_LAND_PATTERNS.items()}
_TWO_COLOURS_RE = re.compile(r"add (?:\{[wubrg]\}|one mana).{0,40}?or", re.IGNORECASE)


def land_classes(card):
    """Every class a land belongs to (a land can be several — a snow tapped
    dual is all three). Mirrors analysis/card_roles.py's taxonomy."""
    classes = set()
    type_line = str(card.get("type_line", "") or "")
    text = str(card.get("oracle_text", "") or "")
    front_type = type_line.split(" // ")[0]
    if "Basic" in front_type:
        classes.add("basic")
    if "Snow" in front_type:
        classes.add("snow")
    if " // " in type_line:
        classes.add("mdfc")
    for name, pattern in _LAND_RES.items():
        if pattern.search(text):
            classes.add(name.split(":", 1)[1])
    if _TWO_COLOURS_RE.search(text) and "tapped" not in classes:
        classes.add("untapped-dual")
    if len(land_colors(card)) >= 2 and "tapped" not in classes and "basic" not in classes:
        classes.add("untapped-dual")
    return classes


def nonland_producer_kind(card):
    """ramp:rock / ramp:dork / ramp:ritual for a nonland mana source, else None."""
    if is_land(card):
        return None
    if not _MANA_SOURCE_RE.search(str(card.get("oracle_text", "") or "")):
        return None
    front_type = str(card.get("type_line", "") or "").split(" // ")[0]
    for supertype, kind in ROLE_MANA_BY_SUPERTYPE.items():
        if supertype in front_type:
            return kind
    return "ramp:rock"


def analyze(slug):
    deck_doc = load_deck_cards(slug)
    entries = mainboard(deck_doc["cards"])
    # Every count below is about the library the shuffler sees, so it runs on
    # COPIES: eleven Islands are eleven blue sources, not one. Counting entries
    # here understates the mana base by every duplicated basic (STYLEv3 §7.6 —
    # the honest number is the whole point of this section).
    cards = expand_copies(entries)
    identity = {
        c
        for card in cards if card.get("is_commander")
        for c in card.get("color_identity", []) if c in WUBRG
    }

    lands = sorted((c for c in cards if is_land(c)), key=lambda c: c["name"])
    land_entries = sorted((c for c in entries if is_land(c)),
                          key=lambda c: c["name"])
    class_counts, land_sources = {}, {c: 0 for c in WUBRG}
    tapped = always_tapped = 0
    for card in lands:
        for cls in land_classes(card):
            class_counts[cls] = class_counts.get(cls, 0) + 1
        if enters_tapped(card):
            tapped += 1
        if enters_tapped_unconditionally(card):
            always_tapped += 1
        for colour in land_colors(card) & (identity or set(WUBRG)):
            land_sources[colour] += 1
    # The table lists one row per distinct land, with its copy count — a reader
    # wants "Island x11", not eleven identical rows.
    land_rows = [{"name": card["name"],
                  "copies": int(card.get("quantity") or 1),
                  "classes": sorted(land_classes(card)),
                  "produces": sorted(land_colors(card) & (identity or set(WUBRG)))}
                 for card in land_entries]

    producers, ramp_counts, nonland_sources = [], {}, {c: 0 for c in WUBRG}
    for card in sorted(entries, key=lambda c: c["name"]):
        copies = int(card.get("quantity") or 1)
        kind = nonland_producer_kind(card)
        if kind:
            produces = sorted(land_colors(card) & (identity or set(WUBRG)))
            ramp_counts[kind] = ramp_counts.get(kind, 0) + copies
            for colour in produces:
                nonland_sources[colour] += copies
            producers.append({"name": card["name"], "kind": kind,
                              "cmc": card.get("cmc"), "produces": produces})
        text = str(card.get("oracle_text", "") or "")
        if _LAND_RAMP_RE.search(text) and not is_land(card):
            ramp_counts["ramp:land"] = ramp_counts.get("ramp:land", 0) + copies
        if _COST_REDUCTION_RE.search(text) and not is_land(card):
            ramp_counts["ramp:cost-reduction"] = ramp_counts.get(
                "ramp:cost-reduction", 0) + copies

    requirements = pip_requirements(cards)
    targets = source_targets(requirements)
    total_sources = {c: land_sources[c] + nonland_sources[c] for c in WUBRG}
    p_lands = achieved_probability(requirements, land_sources)
    p_all = achieved_probability(requirements, total_sources)

    # Pip share vs source share — the intuitive check: a colour demanding 40%
    # of the pips wants roughly 40% of the sources.
    total_pips = sum(r["total_pips"] for r in requirements.values()) or 1.0
    total_land_sources = sum(land_sources.values()) or 1
    shares = {
        c: {
            "pip_share": round(requirements[c]["total_pips"] / total_pips, 3),
            "source_share": round(land_sources[c] / total_land_sources, 3),
        }
        for c in sorted(requirements)
    }

    notes = []
    for colour in sorted(requirements):
        short = targets.get(colour, 0) - land_sources.get(colour, 0)
        if short > 0:
            notes.append(
                f"{colour}: {land_sources.get(colour, 0)} land sources against "
                f"the {targets[colour]} a 90% on-curve rate would take — the "
                f"target is a yardstick, not a possibility, in a "
                f"{len(lands)}-land deck; rocks and dorks lift the count to "
                f"{total_sources.get(colour, 0)}.")
    if lands and tapped / len(lands) > 1 / 3:
        notes.append(
            f"{tapped} of {len(lands)} lands enter tapped "
            f"({tapped / len(lands):.0%}) — over the one-in-three budget the "
            f"deck builder allows itself.")

    goldfish = load_json(deck_dir(slug) / "goldfish_metrics.json") or {}
    metrics = goldfish.get("metrics", {})

    return {
        "slug": slug,
        "decklist_sha256": deck_doc.get("decklist_sha256"),
        "lands": {
            # `total` is copies — the answer to "how many lands does this deck
            # run". `entries` is distinct cards, kept beside it so the two can
            # never be confused again.
            "total": len(lands),
            "entries": len(land_entries),
            # Two numbers because they answer two questions, and reporting
            # only the first made a reader say "3 taplands" about a deck with
            # two. `enters_tapped` is a substring superset — it counts
            # shocklands (a choice at two life) and every "enters tapped
            # unless…" land, including the "unless you have two or more
            # opponents" cycle, which in Commander is always true. That is the
            # right budget for the land SELECTOR, which should be conservative.
            # It is the wrong number for tempo, and deck-audit uses the second.
            "enters_tapped": tapped,
            "enters_tapped_always": always_tapped,
            "classes": dict(sorted(class_counts.items())),
            "list": land_rows,
        },
        "sources": {
            "lands": land_sources,
            "nonland": nonland_sources,
            "total": total_sources,
            "producers": producers,
        },
        "pips": {c: requirements[c] for c in sorted(requirements)},
        "shares": shares,
        "source_targets": dict(sorted(targets.items())),
        "on_curve_probability": {
            "lands_only": p_lands,
            "with_rocks_and_dorks": p_all,
        },
        "ramp": dict(sorted(ramp_counts.items())),
        "goldfish": {
            "land_drop_hit_rate_by_turn": metrics.get(
                "land_drop_hit_rate_by_turn", {}),
            "mean_available_mana_by_turn": metrics.get(
                "mean_available_mana_by_turn", {}),
        },
        "assumptions": [
            "Colour probabilities are hypergeometric draws from a 99-card "
            "library — no mulligans, no card selection, no fetching decisions.",
            "A source counts only for mana it produces unconditionally; "
            "restricted mana (“spend this mana only…”) counts "
            "for nothing (the deck-builder's rule).",
            "Hybrid pips charge half a pip to each side.",
            "Rocks and dorks are counted as sources from the turn they can "
            "tap, with no survival discount on creatures.",
        ],
        "notes": notes,
    }


def main(args):
    result = analyze(args.slug)
    out = deck_dir(args.slug) / "mana_analysis.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    lands = result["lands"]
    probs = result["on_curve_probability"]["lands_only"]
    print(
        f"Wrote {out}: {lands['total']} lands "
        f"({lands['enters_tapped_always']} always tapped, "
        f"{lands['enters_tapped']} incl. conditional), "
        f"sources {'/'.join(f'{c}{n}' for c, n in result['sources']['lands'].items() if n)}, "
        f"on-curve (lands only) "
        f"{', '.join(f'{c} {p:.0%}' for c, p in sorted(probs.items()))}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot mana-analysis <slug>`.")
