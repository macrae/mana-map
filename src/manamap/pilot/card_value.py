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

from manamap.config import GOLDFISH_MAX_TURN, GOLDFISH_SEED
from manamap.pilot import goldfish, model_coverage
from manamap.pilot.common import deck_dir, load_deck_cards, resolve_out_path

DEFAULT_ITERATIONS = 3000

#: ONE PREDICATE, ONE HOME. "Would any casting loop ever select this card, under
#: THIS deck's flags" is `model_coverage.never_cast`, it is fleet-tested, and it
#: is the mirror the goldfish's own loops are checked against. This module used
#: to carry a second, hand-rolled answer to the same question — lands, mana,
#: bodies, tutors, Treasure and combat — WITH NO DRAW CHANNEL IN IT AT ALL, and
#: no drain, discard, sacrifice or death channel either.
#:
#: On a deck built on any of those it filed the engine under "invisible to this
#: model" and printed, as its reason, "the simulation has no opponents, so these
#: score zero by construction. This is a fact about the model, not about the
#: cards." That sentence was false about every one of them.
#:
#: FLEET SWEEP 2026-09-13, before the change: 286 cards invisible across twelve
#: decks, 250 after — 39 newly ranked on seven decks, and they are the decks'
#: engines. edgar-vampires gets Ashnod's Altar and Altar of Dementia (its
#: sacrifice outlets) and Night's Whisper (its only unconditional draw);
#: zur-enchantress gets the three Shrines that `CLAUDE.md` already records as
#: having measured "exactly nothing"; ur-dragon and gishath get Dragon Tempest,
#: Temur Ascendancy and Garruk's Uprising, the per-dragon payoffs the deck is
#: built to trigger; sharknado gets eleven, every wheel spell among them.
#:
#: THREE CARDS MOVE THE OTHER WAY and both directions are corrections. "An
#: Offer You Can't Refuse" (sisay, heliod) was ranked because it sets
#: `treasure_trigger` — it is a counterspell that hands the OPPONENT two
#: Treasure, and there are no opponents. Waterbender Ascension (sharknado) is
#: combat-gated draw this model does not read, and it was being ranked at
#: exactly +0.000.
def _is_visible(card, flags):
    return not model_coverage.never_cast(card, flags)


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
            # PER ENTRY, on purpose: the list with ONE copy removed.
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


def _measure(slug, doc, targets_doc, cards, metric, iterations, seed, max_turn):
    """One deck configuration, one number. Seeded per call so it is comparable.

    THROUGH `goldfish.run`, AND THAT IS THE WHOLE POINT. This used to call
    `simulate_once` itself, forwarding exactly two flags — `model_treasures` and
    `model_combat` — and no commander profile at all. Everything else the
    declaration says was silently off, so on a deck whose plan is a channel this
    function did not forward, it ranked the cards of a deck that does not exist.

    MEASURED on sharknado, 2026-09-13: the tracked goldfish kills by turn eight
    in 87 games per 100 and this function's baseline read **0.228**, because
    `model_draw` and `model_discard` were off and NEITHER COMMANDER WAS CAST —
    so Brallin's damage-per-discard, Shabraz's counter-per-draw and every wheel
    in the deck contributed nothing. The wheels then landed in the invisible
    bucket, which was the honest report of a dishonest run: given how it was
    being simulated they really were doing nothing.

    `goldfish.run` already assembles the flags, both commanders, the partner,
    the attack tutor, the reveal and the interaction names. One door. It takes
    `doc=` for exactly this kind of caller and `with_results=True` hands back the
    per-iteration records the metric reductions below are written against, so
    the reductions are untouched.

    `_targets_doc` HANDS THE DECLARATION OVER rather than letting `run` read the
    same file a second time, so the flags this function's visibility predicate
    was asked about and the flags the simulation ran under cannot drift apart.
    Its `targets` are emptied on the way through: a card ranking does not need
    assembly rates, and computing twelve of them per card is the only thing this
    indirection would otherwise add.
    """
    out = goldfish.run(slug, doc=dict(doc, cards=cards), iterations=iterations,
                       seed=seed, max_turn=max_turn, quiet=True, _band=False,
                       with_results=True,
                       _targets_doc=dict(targets_doc, targets=[]))
    return METRICS[metric][1](out["_results"], iterations)


def build(slug, metric="kill-by-8", iterations=DEFAULT_ITERATIONS, seed=None,
          max_turn=None):
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN
    if metric not in METRICS:
        raise SystemExit(f"unknown metric {metric!r}; choose from {', '.join(METRICS)}")

    doc = load_deck_cards(slug)
    targets_path = deck_dir(slug) / "goldfish_targets.json"
    # EVERY FLAG THE DECLARATION SETS, because the visibility predicate is
    # asked "under THIS deck's flags" and `goldfish.run` reads the same file to
    # decide what it simulates. Reading two of them here and all of them there
    # is how the report came to describe a different deck than it measured.
    targets_doc, flags = {}, {}
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        flags = {k: bool(v) for k, v in targets_doc.items()
                 if k.startswith("model_") and isinstance(v, bool)}
    model_combat = bool(flags.get("model_combat"))
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

    def run(subset):
        return _measure(slug, doc, targets_doc, subset, metric, iterations,
                        seed, max_turn)

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
    noise = abs(baseline - _measure(slug, doc, targets_doc, cards, metric,
                                    iterations, seed + 1, max_turn))

    # `pool=` OR EVERY FETCHLAND IS INVISIBLE. `classify`'s own docstring says
    # the pool exists for exactly one card class and that "without it every
    # fetch is a colourless land that never produces anything" — and
    # `build_library` always passes it. Omitting it here made the VISIBILITY
    # test, which decides whether a card is rankable at all, read Wooded
    # Foothills as contributing nothing: a card silently dropped from the
    # ranking rather than ranked low.
    land_pool = [c for c in cards if "Land" in str(c.get("type_line") or "")]
    classified = {c["name"]: goldfish.classify(c, pool=land_pool) for c in cards
                  if not c.get("is_commander")}

    ranked, invisible = [], []
    for card in cards:
        if card.get("is_commander"):
            continue
        name = card["name"]
        if not _is_visible(classified[name], flags):
            invisible.append(name)
            continue
        blank = inert_land if classified[name]["is_land"] else inert
        swapped = run(_swap_one_copy(cards, name, blank))
        # Positive = the deck is WORSE with this card replaced by a blank, i.e.
        # the card was carrying that much.
        # ROUNDED ON BOTH SIDES, so the flag is one a reader can reproduce from
        # the two numbers printed beside it. Computing it from the full-precision
        # difference against the full-precision floor let a row print
        # `value 0.0133`, `noise floor 0.0133` and `above_noise true`, which is a
        # report contradicting itself in the space of one line.
        value = round(baseline - swapped, 4)
        ranked.append({
            "card": name,
            "metric_without": round(swapped, 4),
            "value": value,
            "above_noise": bool(abs(value) > round(noise, 4)),
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
            "Cards no casting loop would ever select under this deck's declared "
            "flags are EXCLUDED from the ranking, not ranked last — the "
            "predicate is `model_coverage.never_cast`, the same mirror the "
            "goldfish's own loops are checked against. A card lands here for "
            "one of three reasons and they are NOT the same reason: the model "
            "has no opponents (removal, counterspells, protection, an "
            "opponent-draw tax); the effect is real but unparsed, and "
            "`meta.draw_not_modelled` and `model-coverage` name those; or the "
            "channel it feeds is switched OFF in goldfish_targets.json, which "
            "is a decision, not a property of the card. Check which before "
            "reading anything into a name here. This is not a cut list.",
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
        print(f"\n  NOT RANKED — no casting loop selects these under this deck's "
              f"flags ({len(doc['invisible_to_this_model'])}).\n  Three different "
              "reasons land a card here: no opponents to aim at, an effect the "
              "parser\n  does not read, or a channel switched off in the "
              "declaration. A fact about the\n  model, not about the cards, and "
              "it is NOT a cut list.")
        for name in doc["invisible_to_this_model"]:
            print(f"    {name}")
    for note in doc["notes"]:
        print(f"\n  note: {note}")
    return 0
