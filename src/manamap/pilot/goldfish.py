"""Pilot: goldfish simulation — seeded Monte Carlo resource-development metrics.

Tier-2 (data-derived) evidence for the pilot's manual. The model simulates
resource development, NOT full games; its assumptions are stated in the output
artifact and rendered in the manual. Deterministic: same seed and deck produce
byte-identical metrics.

Model assumptions (v1):
- Multiplayer Commander: every player draws on each of their turns, turn 1 included.
- Mulligan rule: keep a 7 with 2-5 lands; otherwise redraw a fresh 7 (up to 2
  redraws), keeping the last hand regardless. No bottoming.
- One land played per turn when available.
- Persistent mana producers ("{T}: Add ...") are cast greedily after the
  commander and contribute their mana starting the following turn.
- The commander is cast on the first turn it is affordable (highest priority).
- Bodies-by-turn casts creature/token cards greedily by cost with leftover
  mana, counting the card itself (if a creature) plus tokens parsed from
  "create ... token" text. Crude by design: no interactions, no haste math.
- Cost reducers, rituals, and card draw beyond one per turn are NOT modeled;
  estimates are therefore conservative for decks that use them.
"""

import json
import random
import re

from manamap.config import (
    GOLDFISH_ITERATIONS,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_MAX_TURN,
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
    GOLDFISH_SEED,
)
from manamap.pilot.common import deck_dir, load_deck_cards

MODEL_ASSUMPTIONS = [
    "Simulates resource development, not full games (no interaction, no removal).",
    "Draw every turn including turn 1 (multiplayer Commander).",
    "Mulligan: keep 7-card hands with 2-5 lands; up to 2 fresh redraws, keep the last.",
    "One land drop per turn when available.",
    "Mana rocks ('{T}: Add') contribute from the turn after they are cast.",
    "Commander cast on first affordable turn (highest spending priority).",
    "Bodies count = creatures cast + tokens parsed from 'create ... token' text.",
    "Target assembly counts cards DRAWN by a turn (cast cards still count).",
    "Cost reducers, rituals, and extra card draw are not modeled (conservative).",
]

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "x": 0,
}

_TOKEN_RE = re.compile(r"create (\w+)(?: [\w/+-]+)* tokens?", re.IGNORECASE)
_TAP_ADD_RE = re.compile(r"\{T\}: Add ((?:\{[WUBRGC0-9]\})+)")


def produced_mana(oracle_text):
    """Mana a persistent '{T}: Add ...' producer yields per turn (0 if none)."""
    match = _TAP_ADD_RE.search(oracle_text or "")
    if not match:
        return 0
    return len(re.findall(r"\{[WUBRGC0-9]\}", match.group(1)))


def body_count(card):
    """Bodies this card contributes when cast: itself (if creature) + tokens."""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for word in _TOKEN_RE.findall(card.get("oracle_text", "") or ""):
        bodies += _NUMBER_WORDS.get(word.lower(), 1 if not word.isdigit() else int(word))
    return bodies


def classify(card):
    """Return a compact sim-card dict for one physical copy."""
    type_line = card.get("type_line", "")
    return {
        "name": card["name"],
        "is_land": "Land" in type_line and "Creature" not in type_line.split("//")[0],
        "cmc": int(card.get("cmc") or 0),
        "produces": 0 if "Land" in type_line else produced_mana(card.get("oracle_text")),
        "bodies": 0 if "Land" in type_line else body_count(card),
    }


def build_library(doc):
    """Expand the main deck (minus commanders) into per-copy sim cards."""
    library = []
    commanders = []
    for card in doc["cards"]:
        if card.get("is_sideboard"):
            continue
        if card.get("is_commander"):
            commanders.append(card)
            continue
        library.extend([classify(card)] * card.get("quantity", 1))
    return library, commanders


def keepable(hand):
    lands = sum(1 for c in hand if c["is_land"])
    return GOLDFISH_MULLIGAN_MIN_LANDS <= lands <= GOLDFISH_MULLIGAN_MAX_LANDS


def _target_met(target, names_in_hand, commander_cast):
    if target.get("commander") and not commander_cast:
        return False
    for need in target["need"]:
        if not any(name in names_in_hand for name in need["any_of"]):
            return False
    return True


def simulate_once(rng, library, commander_cmc, targets, max_turn):
    """One goldfish iteration. Returns a per-iteration result dict."""
    deck = library[:]
    rng.shuffle(deck)

    hand = deck[:7]
    deck = deck[7:]
    mulligans = 0
    while not keepable(hand) and mulligans < GOLDFISH_MAX_MULLIGANS:
        mulligans += 1
        deck = library[:]
        rng.shuffle(deck)
        hand = deck[:7]
        deck = deck[7:]

    opening_lands = sum(1 for c in hand if c["is_land"])
    seen = {c["name"] for c in hand}

    lands_in_play = 0
    rock_production = 0
    commander_turn = None
    land_hits = []
    mana_by_turn = []
    bodies_cum = 0
    bodies_by_turn = []
    target_turns = [None] * len(targets)

    for turn in range(1, max_turn + 1):
        if deck:
            drawn = deck.pop(0)
            hand.append(drawn)
            seen.add(drawn["name"])

        land_index = next((i for i, c in enumerate(hand) if c["is_land"]), None)
        if land_index is not None:
            hand.pop(land_index)
            lands_in_play += 1
            land_hits.append(True)
        else:
            land_hits.append(False)

        pool = lands_in_play + rock_production
        mana_by_turn.append(pool)

        if commander_turn is None and pool >= commander_cmc:
            commander_turn = turn
            pool -= commander_cmc

        # Cast rocks cheapest-first; they produce starting next turn.
        for card in sorted((c for c in hand if c["produces"] > 0), key=lambda c: c["cmc"]):
            if card["cmc"] <= pool:
                pool -= card["cmc"]
                rock_production += card["produces"]
                hand.remove(card)

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0), key=lambda c: c["cmc"]):
            if card["cmc"] <= pool:
                pool -= card["cmc"]
                bodies_cum += card["bodies"]
                hand.remove(card)
        bodies_by_turn.append(bodies_cum)

        for i, target in enumerate(targets):
            if target_turns[i] is None and _target_met(target, seen, commander_turn is not None):
                target_turns[i] = turn

    return {
        "opening_lands": opening_lands,
        "mulligans": mulligans,
        "land_hits": land_hits,
        "mana_by_turn": mana_by_turn,
        "commander_turn": commander_turn,
        "bodies_by_turn": bodies_by_turn,
        "target_turns": target_turns,
    }


def _round(x):
    return round(x, 3)


def aggregate(results, targets, max_turn):
    n = len(results)
    turns = list(range(1, max_turn + 1))

    commander_turns = [r["commander_turn"] for r in results]
    cast_counts = {}
    for t in commander_turns:
        key = str(t) if t is not None else "not_by_max_turn"
        cast_counts[key] = cast_counts.get(key, 0) + 1
    cast_values = sorted(t for t in commander_turns if t is not None)

    target_stats = []
    for i, target in enumerate(targets):
        assembled = sorted(r["target_turns"][i] for r in results if r["target_turns"][i] is not None)
        target_stats.append({
            "label": target["label"],
            "assembled_rate": _round(len(assembled) / n),
            "mean_turn": _round(sum(assembled) / len(assembled)) if assembled else None,
            "by_turn_6_rate": _round(sum(1 for t in assembled if t <= 6) / n),
        })

    opening_histogram = {}
    for r in results:
        key = str(r["opening_lands"])
        opening_histogram[key] = opening_histogram.get(key, 0) + 1

    return {
        "iterations": n,
        "opening_hand": {
            "land_histogram": opening_histogram,
            "keep_first_seven_rate": _round(sum(1 for r in results if r["mulligans"] == 0) / n),
            "mean_mulligans": _round(sum(r["mulligans"] for r in results) / n),
        },
        "land_drop_hit_rate_by_turn": {
            str(t): _round(sum(1 for r in results if r["land_hits"][t - 1]) / n) for t in turns
        },
        "mean_available_mana_by_turn": {
            str(t): _round(sum(r["mana_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        "commander": {
            "cast_turn_histogram": cast_counts,
            "mean_cast_turn": _round(sum(cast_values) / len(cast_values)) if cast_values else None,
            "median_cast_turn": cast_values[len(cast_values) // 2] if cast_values else None,
            "cast_by_turn_6_rate": _round(sum(1 for t in cast_values if t <= 6) / n),
        },
        "mean_bodies_by_turn": {
            str(t): _round(sum(r["bodies_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        "targets": target_stats,
    }


def run(slug, iterations=None, seed=None, max_turn=None):
    """Run the goldfish simulation for a deck. Returns the metrics document."""
    iterations = iterations or GOLDFISH_ITERATIONS
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN

    doc = load_deck_cards(slug)
    library, commanders = build_library(doc)
    if not commanders:
        raise SystemExit(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    targets_path = deck_dir(slug) / "goldfish_targets.json"
    targets = []
    if targets_path.exists():
        with open(targets_path) as f:
            targets = json.load(f)["targets"]

    rng = random.Random(seed)
    results = [
        simulate_once(rng, library, commander_cmc, targets, max_turn)
        for _ in range(iterations)
    ]

    return {
        "meta": {
            "deck": slug,
            "decklist_sha256": doc.get("decklist_sha256"),
            "seed": seed,
            "iterations": iterations,
            "max_turn": max_turn,
            "commander": commanders[0]["name"],
            "commander_cmc": commander_cmc,
            "model_assumptions": MODEL_ASSUMPTIONS,
        },
        "metrics": aggregate(results, targets, max_turn),
    }


def main(args):
    doc = run(args.slug)
    out = deck_dir(args.slug) / "goldfish_metrics.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    commander = doc["meta"]["commander"]
    stats = doc["metrics"]["commander"]
    print(
        f"Wrote {out}\n  {commander}: mean cast turn {stats['mean_cast_turn']}, "
        f"cast by turn 6 in {stats['cast_by_turn_6_rate']:.0%} of games"
    )
    for target in doc["metrics"]["targets"]:
        print(f"  {target['label']}: by turn 6 in {target['by_turn_6_rate']:.0%} of games")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot goldfish <slug>`.")
