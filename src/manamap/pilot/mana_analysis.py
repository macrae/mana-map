"""Pilot: the deterministic mana/land analysis — pips vs sources, castability, tapped lands (◆).

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
    front_face,
    deck_dir,
    expand_copies,
    is_land,
    deck_file,
    load_deck_cards,
    load_json,
)
from manamap.pilot import manabase
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
    front_type = front_face(type_line)
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


#: WHAT THE BASE CHARGES IN LIFE, which no source count can see.
#:
#: A source count says a land makes the colour; it says nothing about the price.
#: Measured on ur-dragon: `eminence-v3` and `landbase-v1` differ by 6 lands and
#: sit within noise on every sampled axis, because the goldfish models no life at
#: all. The whole case for the change is here — Tarnished Citadel charges 3 life
#: EVERY TAP and a fetch charges 1 ONCE, so the two costs are different kinds of
#: number and adding them together would hide exactly that.
#:
#: THE SWEEP, over 1266 corpus lands:
#:
#:   52  RECURRING   paid again on every activation, forever. The painland
#:                   cycle, City of Brass, Mana Confluence, Ancient Tomb,
#:                   Tarnished Citadel at 3.
#:   38  ONE-TIME    paid once. Ten shocklands at 2, twelve fetches at 1, and
#:                   sixteen modern spell//land MDFCs at 3.
#:    0  BOTH        the classes do not overlap.
#:
#: TWO GATES, and each was wrong in the first pass:
#:
#:   * RECURRING requires an `add` clause. Without it `Sorrow's Path` — which
#:     has NO mana ability and merely hurts you when it taps — reads as a
#:     2-life mana source. A life cost that buys no mana is not a mana cost.
#:   * ONE-TIME must NOT require one. A FETCHLAND MAKES NO MANA ITSELF, so the
#:     same gate zeroed all four of ur-dragon's fetches. And shocklands name
#:     themselves — "As Blood Crypt enters, you may pay 2 life" — where the
#:     MDFCs say "this land", so the subject has to be read as either. With the
#:     first pass's gate the ledger read ZERO for a list holding six shocklands
#:     and four fetches, and looked entirely plausible doing it.
_LIFE_RECURRING_RES = (
    re.compile(r"\{t\}[^:.]{0,40}?,\s*pay (\d+) life\s*:\s*add", re.IGNORECASE),
    re.compile(r"add[^.]{0,80}?\.\s*(?:this land|it|[A-Z][\w', ]{2,28}) "
               r"deals (\d+) damage to you", re.IGNORECASE),
    re.compile(r"becomes tapped, (?:it|this land|[A-Z][\w', ]{2,28}) "
               r"deals (\d+) damage to you", re.IGNORECASE),
)
_LIFE_ONE_TIME_RES = (
    re.compile(r"as [^,.]{0,40} enters, you may pay (\d+) life", re.IGNORECASE),
    re.compile(r"pay (\d+) life,\s*sacrifice this land", re.IGNORECASE),
)
_REMINDER_RE = re.compile(r"\([^)]*\)")
_ADD_RE = re.compile(r"\badd\b", re.IGNORECASE)


def life_cost(card):
    """`{recurring, one_time}` in life, for one land. Never summed together."""
    text = _REMINDER_RE.sub(" ", str(card.get("oracle_text", "") or ""))

    def most(patterns):
        return max([int(m.group(1)) for p in patterns for m in p.finditer(text)]
                   or [0])

    return {"recurring": most(_LIFE_RECURRING_RES) if _ADD_RE.search(text) else 0,
            "one_time": most(_LIFE_ONE_TIME_RES)}


def nonland_producer_kind(card):
    """ramp:rock / ramp:dork / ramp:ritual for a nonland mana source, else None."""
    if is_land(card):
        return None
    # REMINDER TEXT IS NOT THIS CARD'S ABILITY — the same fix `land_colors` got,
    # and this is the GATE, so without it a Treasure-maker still counts as a
    # dork producing nothing. Prosperous Innkeeper creates a Treasure and has no
    # mana ability of its own; its reminder text describes the token's.
    if not _MANA_SOURCE_RE.search(
            manabase._REMINDER_RE.sub(" ", str(card.get("oracle_text", "") or ""))):
        return None
    front_type = front_face(card.get("type_line", ""))
    for supertype, kind in ROLE_MANA_BY_SUPERTYPE.items():
        if supertype in front_type:
            return kind
    return "ramp:rock"


def analyze(slug, branch=None):
    deck_doc = load_deck_cards(slug, branch)
    entries = deck_doc["cards"]
    # Every count below is about the library the shuffler sees, so it runs on
    # COPIES: eleven Islands are eleven blue sources, not one. Counting entries
    # here understates the mana base by every duplicated basic — the honest
    # number is the whole point of this analysis.
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
    # THE SAME COUNT WITH THE GATED LANDS LEFT OUT — a LOWER BOUND on what can
    # pay a coloured cost on curve. Both arms are reported and neither is
    # collapsed into the other: the truth is between them, and choosing a
    # fraction to place it at would be an authored number driving the headline.
    ungated_sources = {c: 0 for c in WUBRG}
    tapped = always_tapped = 0
    for card in lands:
        for cls in land_classes(card):
            class_counts[cls] = class_counts.get(cls, 0) + 1
        if enters_tapped(card):
            tapped += 1
        if enters_tapped_unconditionally(card):
            always_tapped += 1
        for colour in land_colors(card, pool=lands) & (identity or set(WUBRG)):
            land_sources[colour] += 1
            if not manabase.gated_colour_source(card):
                ungated_sources[colour] += 1
    # The table lists one row per distinct land, with its copy count — a reader
    # wants "Island x11", not eleven identical rows.
    land_rows = [{"name": card["name"],
                  "copies": int(card.get("quantity") or 1),
                  "classes": sorted(land_classes(card)),
                  "produces": sorted(land_colors(card, pool=lands)
                                     & (identity or set(WUBRG)))}
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
    p_ungated = achieved_probability(requirements, ungated_sources)

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

    goldfish = load_json(deck_file(slug, "goldfish_metrics.json", branch)) or {}
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
            # THE PRICE OF THE MANA. Two figures because they are two kinds of
            # number: a painland charges every tap, a fetch or a shock charges
            # once. Summing them would hide the only thing that distinguishes
            # them. See `life_cost`.
            "life": {
                "recurring_per_tap_cycle":
                    sum(life_cost(c)["recurring"] for c in lands),
                "one_time_on_entry":
                    sum(life_cost(c)["one_time"] for c in lands),
            },
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
            # HOW MUCH OF `total` CANNOT PAY ON CURVE. Counted, because they do
            # make the colour, and NAMED, because a reader comparing `have`
            # against a Karsten target deserves to know that some of it arrives
            # a turn late and a mana short.
            "gated": {
                "count": sum(1 for c in lands if manabase.gated_colour_source(c)),
                "names": sorted(c["name"] for c in lands
                                if manabase.gated_colour_source(c)),
                "why": ("counted in `total` and in every colour they can make, "
                        "but their coloured mode costs extra mana on top of the "
                        "tap, so they cannot pay a coloured cost on curve"),
            },
            "producers": producers,
        },
        "pips": {c: requirements[c] for c in sorted(requirements)},
        "shares": shares,
        "source_targets": dict(sorted(targets.items())),
        "on_curve_probability": {
            "lands_only": p_lands,
            "with_rocks_and_dorks": p_all,
            # THE LOWER BOUND. `lands_only` counts a land whose coloured mode
            # costs extra mana as though it were a Swamp, which flatters every
            # deck running them — and made a change that RAISED real sources by
            # 2/2/3 read as a drop of 2-3 points, because it had cut four such
            # lands. A figure and its floor, so a reader can see the spread.
            "lands_only_ungated": p_ungated,
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
    branch = getattr(args, "branch", None)
    result = analyze(args.slug, branch)
    out = deck_dir(args.slug, branch) / "mana_analysis.json"
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
