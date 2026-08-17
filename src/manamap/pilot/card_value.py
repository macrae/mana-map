"""What is each card in this 99 actually WORTH — measured, one card at a time.

Every other measurement in this repo describes a deck. This one describes its
cards: re-run the goldfish with a card removed and see what the deck loses. On
ur-dragon it ranked `The Misty Mountains Cold` (+5.2 points of kill-by-turn-8),
`Smaug the Magnificent` (+4.1) and `Scourge of the Throne` (+2.8) as the three
load-bearing cards, and each of those had been proposed for the cut list by
somebody reasoning from card text.

TWO THINGS MAKE A NAIVE VERSION OF THIS ACTIVELY DANGEROUS, and both are the
reason this module is longer than the sweep it replaces.

**1. Removing a card is not a fair test, because it also thins the deck.** A
99-card deck draws its remaining cards more often than a 100-card one, so EVERY
removal looks like an improvement — measured on ur-dragon, dropping a card the
simulator never even casts still "gained" ~2.0 points. Read naively that says two
thirds of the deck is dead weight, which is an artifact of the population change
and nothing else. So this does not remove: it **replaces each card with an inert
placeholder**, holding the deck at exactly 100. Draw odds are untouched, the
confound is gone rather than corrected for, and a card that does nothing scores
zero automatically instead of scoring whatever the thinning was worth.

**2. It reads as a cut list, and it must never be one.** The bottom of the raw
ranking on ur-dragon was Swords to Plowshares, Counterspell, Swan Song, Teferi's
Protection, Deflecting Swat, Smuggler's Share — and `Blasphemous Act`, which is
half of that deck's only checker-verified kill. Every one of them scores zero *by
construction*: this simulation has no opponents to interact with, no spells to
counter and nothing to protect against, and `MODEL_ASSUMPTIONS` states outright
that cost reducers are not modelled either. A tool that prints "least valuable
cards" over a model that cannot see interaction is a machine for generating
exactly that mistake.

So invisible cards are **excluded from the ranking entirely**, not ranked last.
They are derived structurally rather than by their score — a card the simulator
never casts and whose presence changes nothing is invisible whatever number came
out — and printed in their own section so the blindness is legible without being
an invitation.

Requires `model_combat`. Without it every attack trigger, additional combat phase
and combat-gated Treasure source lands in the invisible bucket and the report is
mostly noise about a model that is not looking.
"""

import json
import random

from manamap.config import GOLDFISH_MAX_TURN, GOLDFISH_SEED
from manamap.pilot import goldfish
from manamap.pilot.common import deck_dir, load_deck_cards, resolve_out_path

DEFAULT_ITERATIONS = 3000

# A card with none of these does nothing the simulator can observe: it is never
# cast (the casting loop only reaches rocks, tutors, bodies and extra-combat
# permanents) or it is cast and changes no series. Structural, not score-based —
# a card can score badly for real reasons, and that is a finding; a card the
# model cannot see is not a finding at all.
def _is_visible(card):
    combat = card.get("combat") or {}
    return bool(
        # A land is the most consequential card in the deck to this simulation —
        # it decides the land drop, which decides everything downstream. It has
        # `produces == 0` because `classify` zeroes that for lands, so a
        # predicate built from the spell fields alone files all 36 as invisible.
        card["is_land"]
        or card["produces"] or card["bodies"] or card["tutor"]
        or card["treasure_trigger"] or card["treasure_bonus"]
        or combat.get("is_creature") or combat.get("attack_mana")
        or combat.get("attack_treasure") or combat.get("attack_draw")
        or combat.get("attack_damage") or combat.get("attack_token_bodies")
        or combat.get("extra_combat_free")
        or combat.get("extra_combat_cost") is not None
    )


def _swap_one_copy(cards, name, blank):
    """One COPY out, one blank in — never the whole entry.

    `cards.json` stores basics as a single entry with `quantity: N`, so dropping
    the entry for "Mountain" drops all 40 of them and the deck falls to 61 cards.
    Every value in the report then carries a 39-card thinning, which is the exact
    confound replace-not-remove exists to eliminate. Caught by the deck-size
    invariant test, which is why that test asserts on the population the
    simulator sees rather than on the report.
    """
    out = []
    swapped = False
    for card in cards:
        if card["name"] == name and not swapped:
            swapped = True
            remaining = card.get("quantity", 1) - 1
            if remaining > 0:
                out.append(dict(card, quantity=remaining))
            continue
        out.append(card)
    return out + [blank]


METRICS = {
    "kill-by-8": ("kill by turn 8",
                  lambda res, n: sum(1 for r in res
                                     if r["kill_turn"] is not None and r["kill_turn"] <= 8) / n),
    "kill-by-10": ("kill by turn 10",
                   lambda res, n: sum(1 for r in res
                                      if r["kill_turn"] is not None and r["kill_turn"] <= 10) / n),
    "board-power": ("mean board power, turn 6",
                    lambda res, n: sum(r["board_power_by_turn"][5] for r in res) / n),
    "hoard": ("mean Treasure hoard, turn 6",
              lambda res, n: sum(r["treasures_by_turn"][5] for r in res) / n),
}


def _measure(cards, commander_cmc, metric, iterations, seed, max_turn,
             model_treasures, model_combat):
    """One deck configuration, one number. Seeded per call so it is comparable."""
    library, _ = goldfish.build_library({"cards": cards})
    rng = random.Random(seed)
    results = [
        goldfish.simulate_once(rng, library, commander_cmc, [], max_turn,
                               model_treasures=model_treasures,
                               model_combat=model_combat)
        for _ in range(iterations)
    ]
    return METRICS[metric][1](results, iterations)


def build(slug, metric="kill-by-8", iterations=DEFAULT_ITERATIONS, seed=None,
          max_turn=None):
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN
    if metric not in METRICS:
        raise SystemExit(f"unknown metric {metric!r}; choose from {', '.join(METRICS)}")

    doc = load_deck_cards(slug)
    targets_path = deck_dir(slug) / "goldfish_targets.json"
    model_treasures = model_combat = False
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        model_treasures = bool(targets_doc.get("model_treasures"))
        model_combat = bool(targets_doc.get("model_combat"))
    if not model_combat:
        raise SystemExit(
            f"{slug} has not opted into the combat model, so every attack trigger, "
            "additional combat phase and combat-gated Treasure source would score "
            "zero and land in the invisible bucket. Set \"model_combat\": true in "
            f"{targets_path} and re-run `manamap pilot goldfish {slug}` first.")

    cards = doc["cards"]
    commanders = [c for c in cards if c.get("is_commander")]
    if not commanders:
        raise SystemExit(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    def run(subset):
        return _measure(subset, commander_cmc, metric, iterations, seed,
                        max_turn, model_treasures, model_combat)

    baseline = run(cards)

    # A vanilla blank that costs what an average card costs. Swapped IN wherever
    # a card is swapped out, so every variant is a 100-card deck and the draw
    # odds never move. `classify` gives it no bodies, no mana, no triggers, so
    # the simulation will never cast it and it can contribute nothing.
    inert = {"name": "__inert__", "type_line": "Enchantment", "cmc": 3,
             "oracle_text": "", "quantity": 1, "is_commander": False,
             "power": None, "toughness": None}
    # Swapping a LAND for a spell-shaped blank would measure "one fewer land"
    # rather than "this land does nothing", and the deck would look like its
    # mana base is worth everything. A land is replaced by an inert LAND.
    inert_land = {"name": "__inert_land__", "type_line": "Land", "cmc": 0,
                  "oracle_text": "", "quantity": 1, "is_commander": False,
                  "power": None, "toughness": None}

    # The noise floor. Same deck, a different seed: whatever this moves by is
    # what the sampling alone is worth, and a `value` smaller than it means
    # nothing. Stating a resolution the sample cannot support is how a ranking
    # becomes a horoscope.
    noise = abs(baseline - _measure(cards, commander_cmc, metric, iterations,
                                    seed + 1, max_turn, model_treasures,
                                    model_combat))

    classified = {c["name"]: goldfish.classify(c) for c in cards
                  if not c.get("is_commander")}

    ranked, invisible = [], []
    for card in cards:
        if card.get("is_commander"):
            continue
        name = card["name"]
        if not _is_visible(classified[name]):
            invisible.append(name)
            continue
        blank = inert_land if classified[name]["is_land"] else inert
        swapped = run(_swap_one_copy(cards, name, blank))
        # Positive = the deck is WORSE with this card replaced by a blank, i.e.
        # the card was carrying that much.
        ranked.append({
            "card": name,
            "metric_without": round(swapped, 4),
            "value": round(baseline - swapped, 4),
            "above_noise": bool(abs(baseline - swapped) > noise),
        })
    ranked.sort(key=lambda r: -r["value"])

    return {
        "slug": slug,
        "metric": metric,
        "metric_label": METRICS[metric][0],
        "iterations": iterations,
        "seed": seed,
        "baseline": round(baseline, 4),
        "noise_floor": round(noise, 4),
        "cards": ranked,
        "invisible_to_this_model": sorted(invisible),
        "notes": [
            "`value` is what the deck loses when this card is REPLACED BY A BLANK, "
            "never when it is removed: removing also thins the deck, which makes "
            "every card look good to cut. The deck is 100 cards in every variant.",
            f"The noise floor is {noise:.3f} — the same deck on a different seed. "
            "A `value` smaller than that is sampling, not signal (`above_noise`).",
            "Cards the simulation cannot observe are EXCLUDED from the ranking, "
            "not ranked last. This model has no opponents, so removal, "
            "counterspells and protection score zero by construction, and cost "
            "reducers are excluded by MODEL_ASSUMPTIONS. This is not a cut list.",
            f"Sampled at {iterations:,} games per card, not the full "
            f"{goldfish.GOLDFISH_ITERATIONS:,} the tracked metrics use — "
            "differences smaller than about a point are noise.",
        ],
    }


def main(args):
    doc = build(args.slug,
                metric=getattr(args, "metric", "kill-by-8"),
                iterations=getattr(args, "iterations", None) or DEFAULT_ITERATIONS)

    out = getattr(args, "out", None)
    if out:
        path = resolve_out_path(out, args.slug, "card-value")
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"Wrote {path}")
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2))
        return 0

    print(f"{doc['slug']} — card value by {doc['metric_label']}"
          f"  ({doc['iterations']:,} games/card, seed {doc['seed']})")
    print(f"  baseline {doc['baseline']:.3f}   noise floor {doc['noise_floor']:.3f} "
          "(same deck, different seed — below this is not signal)\n")

    ranked = doc["cards"]
    print(f"  {'card':34}{'value':>9}{'as a blank':>12}")
    for row in ranked:
        mark = "" if row["above_noise"] else "   ·"
        print(f"  {row['card']:34}{row['value']:>+9.3f}{row['metric_without']:>12.3f}{mark}")
    print("   · = inside the noise floor")

    if doc["invisible_to_this_model"]:
        print(f"\n  NOT RANKED — invisible to this model ({len(doc['invisible_to_this_model'])}). "
              "The simulation has no\n  opponents, so these score zero by construction. "
              "This is a fact about the\n  model, not about the cards, and it is NOT a cut list.")
        for name in doc["invisible_to_this_model"]:
            print(f"    {name}")
    for note in doc["notes"]:
        print(f"\n  note: {note}")
    return 0
