"""Pilot: the standard benchmark — four measures, one frozen configuration.

    manamap pilot benchmark <slug>
    manamap pilot benchmark --all

PRD §9. Four per-deck measures — mana screw, speed, consistency, response —
computed under conditions that are IDENTICAL for every deck, because §9.2 is the
whole of the requirement: "the aggregate is only meaningful if the simulations
are controlled … uncontrolled sim output cannot be aggregated into a ranking."

WHY THIS DOES NOT READ A DECK'S OWN `goldfish_metrics.json`, which is the first
thing anybody would try. Those files are produced under each deck's OWN
declaration, and the declarations disagree: of twelve decks with a 99, exactly
ONE opts into `model_combat` and `model_treasures`. Ranking the fleet off them
would compare a deck measured with a kill clock against eleven measured without,
and `mean_bodies_by_turn` does not even mean the same thing across the two
(a Treasure is not a body). Two decks have no declaration at all and a third's
has never been edited.

So the benchmark runs its own goldfish at a fixed seed, a fixed iteration count
and uniform model flags, and writes its own record. It never touches the deck's
tracked metrics, and a deck's tracked metrics never move because a benchmark
ran. Measured cost: ~2.3s per deck, so the whole fleet is well under a minute —
which is what makes re-running it after every swap realistic rather than a
ceremony.

**IT READS THE 99, NOT THE DECLARATION.** Every metric below is a property of
the cards. That is not a convenience: a declaration is authored, so a benchmark
that read one would rank decks partly on how well their pilot writes JSON, and
would refuse to score the two decks that have none. `blar` and `kinnan` are
scoreable here on the day they are built.

WHAT THIS IS NOT. There is no pod, no opponent deck and no interaction, so this
is not a win rate and must never be presented beside one. `simulate` measures a
table; this measures a deck. The two answer different questions and the record
says so in `limits`.

THE AGGREGATE IS NOT COMPUTED HERE, AND THAT IS DELIBERATE. §14.1 leaves the
formula open — weighted sum, or something that penalises a floor — and it cannot
be chosen honestly from a desk. Extract first, run the fleet, look at the real
spread, then decide. A formula picked before the distribution is a guess wearing
a number.
"""

import json
import statistics

from manamap.pilot.common import deck_dir, load_card_roles, load_deck_cards

BENCHMARK_FILE = "benchmark.json"

#: THE FROZEN HARNESS. §9.2 asks for pod definition, iteration count, seeding
#: and metric extraction to be fixed before any score is published. There is no
#: pod — a goldfish has no opponents — so what is frozen is everything else.
#:
#: Bump `version` when any of this changes, because a score computed under a
#: different configuration is not comparable to one that is not, and the record
#: carries the version so nobody has to remember which.
HARNESS = {
    "version": 1,
    "iterations": 10000,
    "seed": 42,
    "max_turn": 10,
    # UNIFORM, and overriding each deck's own opt-in on purpose — see above.
    "model_treasures": True,
    "model_combat": True,
}

#: The turns a land drop is still the thing that decides the game. Chosen to
#: match the window `deck_audit` and every diagnosis already argue over rather
#: than invented here: turn two is the first drop that can be missed after a
#: keep, and by seven the deck is either operating or has lost for other reasons.
SCREW_TURNS = ("2", "3", "4", "5", "6")

#: Roles that answer what the table is doing. `stax` carries no colon.
ANSWER_AXES = ("removal", "protection", "counterspell", "hate", "stax")

#: The classes an answer suite is measured for BREADTH across. A deck holding
#: nine spot-removal spells and nothing else answers one kind of problem nine
#: times, which is not the same as answering nine kinds — the distinction
#: `deck_audit`'s `interaction-breadth` axis already makes.
ANSWER_CLASSES = ("removal:spot", "removal:sweeper", "removal:bounce",
                  "removal:edict", "removal:damage", "removal:fight",
                  "removal:debuff", "removal:tax", "counterspell",
                  "protection:self", "protection:granted", "protection:fog",
                  "protection:redirect", "hate:graveyard", "stax")


def _axis(role):
    return role.split(":", 1)[0]


def mana_screw(metrics):
    """How often, and how badly, the deck fails to have mana.

    Two independent failures, kept separate rather than blended: hands you
    cannot keep, and drops you miss once you have kept. A deck can be fine on
    one and terrible on the other, and a single number would hide which.
    """
    hits = metrics["land_drop_hit_rate_by_turn"]
    missed = [1.0 - hits[t] for t in SCREW_TURNS if t in hits]
    hand = metrics["opening_hand"]
    return {
        "missed_land_drop_rate": round(statistics.mean(missed), 4) if missed else None,
        "worst_turn": (max(
            ((t, round(1.0 - hits[t], 4)) for t in SCREW_TURNS if t in hits),
            key=lambda kv: kv[1])[0] if missed else None),
        "mulligan_rate": round(1.0 - hand["keep_first_seven_rate"], 4),
        "mean_mulligans": round(hand["mean_mulligans"], 4),
        "mana_at_turn_five": round(metrics["mean_available_mana_by_turn"]["5"], 4),
    }


def speed(metrics):
    """How fast the deck executes its plan, against nobody.

    `mean_kill_turn` is CENSORED — a deck that never kills inside `max_turn`
    contributes the cap, so the mean of a slow deck is a statement about the cap
    rather than about the deck. The by-turn rates are not censored, so they are
    what the measure reports, with the mean beside them as description.
    """
    combat = metrics.get("combat")
    if not combat:
        return None
    rates = combat["kill_by_turn_rate"]
    return {
        "kill_by_turn_6": round(rates.get("6", 0.0), 4),
        "kill_by_turn_8": round(rates.get("8", 0.0), 4),
        "kill_by_turn_10": round(rates.get("10", 0.0), 4),
        "never_kills_rate": round(combat["no_kill_by_max_turn_rate"], 4),
        "median_kill_turn": combat.get("median_kill_turn"),
    }


def consistency(metrics, results):
    """Does it do its thing reliably — the SPREAD of a thing every deck does.

    MEASURED AND REPLACED. The first version took the spread of the kill-turn
    histogram, and the fleet run showed it correlating with speed at **r = 0.78**
    — it was speed wearing another name. The reason is an artifact of counting:
    the spread was computed over the games that KILLED, so a deck killing in
    0.1% of games contributed ten tightly-clustered late kills and scored as
    supremely consistent, while a deck killing in 40% spread its kills across
    turns four to ten and scored as erratic. Exactly backwards.

    So it measures RESOURCE DEVELOPMENT instead: the standard deviation of
    available mana at turns four through six, across all ten thousand games.
    Every deck develops mana, every game contributes, and nothing is censored.
    A low number is a deck that plays the same game every time.

    It correlates with the mana LEVEL at r = 0.78, and that was checked rather
    than assumed: a coefficient of variation (stdev / mean) moves it only to
    0.78 from 0.90, so most of the relationship survives removing the scale. The
    reading is substantive — a ramp-heavy deck is genuinely more variable,
    because ramp drawn and ramp not drawn are different games — and it is
    therefore left as a plain standard deviation rather than dressed up. It is
    also a second, independent argument against a naive weighted sum: two
    measures at 0.78 in one total is that quantity counted nearly twice.
    """
    spreads = {}
    for turn in (4, 5, 6):
        values = [r["mana_by_turn"][turn - 1] for r in results]
        spreads[str(turn)] = round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
    lands = [sum(1 for hit in r["land_hits"][:6] if hit) for r in results]
    return {
        "mana_stdev_by_turn": spreads,
        "mana_stdev_turn_five": spreads["5"],
        "land_drops_by_six_stdev": round(statistics.pstdev(lands), 4),
        "basis": "spread across all games of a quantity every deck has — not of "
                 "kill turn, which is censored by how often a deck kills at all",
    }


def response(slug):
    """How much answer the deck CARRIES.

    THE ONE MEASURE THAT IS NOT A SIMULATION, and the record says so. A goldfish
    has no opponents, so nothing in it can observe a deck answering anything —
    "ability to answer what the table does" is unmeasurable in a model with no
    table. What is measurable from the 99 is capacity: how many answers, and
    across how many kinds of problem.

    Reported as a count and a breadth, never blended, because they fail
    differently: nine spot-removal spells answer one kind of problem nine times.
    """
    roles = load_card_roles()
    doc = load_deck_cards(slug)
    names = [c["name"] for c in doc.get("cards", []) if not c.get("is_commander")]

    answers, classes = 0, set()
    for name in names:
        got = [r for r in (roles.get(name) or []) if _axis(r) in ANSWER_AXES]
        if not got:
            continue
        answers += 1
        for r in got:
            if r in ANSWER_CLASSES:
                classes.add(r)
    return {
        "answer_cards": answers,
        "answer_share": round(answers / len(names), 4) if names else 0.0,
        "classes_covered": len(classes),
        "classes_possible": len(ANSWER_CLASSES),
        "classes": sorted(classes),
        "basis": "card_roles.json over the 99 — CAPACITY, not performance",
    }


def measure(slug):
    """Run the frozen harness against one deck and return its record."""
    from manamap.pilot import goldfish

    doc = goldfish.run(
        slug,
        iterations=HARNESS["iterations"],
        seed=HARNESS["seed"],
        max_turn=HARNESS["max_turn"],
        model_treasures=HARNESS["model_treasures"],
        model_combat=HARNESS["model_combat"],
        with_results=True,
    )
    metrics = doc["metrics"]
    cards = load_deck_cards(slug)
    return {
        "slug": slug,
        "harness": dict(HARNESS),
        "decklist_sha256": cards.get("decklist_sha256"),
        "metrics": {
            "mana_screw": mana_screw(metrics),
            "speed": speed(metrics),
            "consistency": consistency(metrics, doc["_results"]),
            "response": response(slug),
        },
        # NO AGGREGATE. §14.1 is open and the formula is chosen against the real
        # spread, not before it.
        "score": None,
        "limits": [
            "No pod, no opponent and no interaction: this measures a DECK, not a "
            "table. It is not a win rate and must not be shown beside one.",
            "Nothing blocks and nothing removes, so every speed figure is a "
            "ceiling — the fastest this deck goes when unopposed.",
            "`response` is CAPACITY from card_roles.json, not a measured "
            "response: a model with no table cannot observe one.",
            "Uniform model flags override each deck's own declaration, which is "
            "what makes decks comparable and also what makes these figures "
            "differ from the deck's own tracked goldfish_metrics.json.",
        ],
    }


def write(slug):
    record = measure(slug)
    path = deck_dir(slug) / BENCHMARK_FILE
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path, record


def main(args):
    import contextlib
    import io as _io

    from manamap.config import DECKS_DIR

    slugs = ([args.slug] if getattr(args, "slug", None) else
             sorted(d.name for d in DECKS_DIR.iterdir()
                    if (d / "cards.json").exists()))

    records = []
    for slug in slugs:
        # The goldfish narrates to stdout; the benchmark's answer is the table.
        with contextlib.redirect_stdout(_io.StringIO()):
            path, record = write(slug)
        records.append(record)
        if not getattr(args, "as_json", False):
            print(f"  measured {slug}")

    if getattr(args, "as_json", False):
        print(json.dumps(records if len(records) > 1 else records[0], indent=2))
        return

    print(f"\nBENCHMARK v{HARNESS['version']} — {HARNESS['iterations']} sims, "
          f"seed {HARNESS['seed']}, uniform flags\n")
    print(f"  {'deck':18} {'missed':>7} {'mull':>6} {'t5 mana':>8} "
          f"{'kill t8':>8} {'never':>7} {'spread':>7} {'answers':>8} {'kinds':>6}")
    print("  " + "-" * 82)
    for r in records:
        m, s = r["metrics"]["mana_screw"], r["metrics"]["speed"]
        c, a = r["metrics"]["consistency"], r["metrics"]["response"]
        print(f"  {r['slug']:18} {m['missed_land_drop_rate']:>7.3f} "
              f"{m['mulligan_rate']:>6.3f} {m['mana_at_turn_five']:>8.2f} "
              f"{(s['kill_by_turn_8'] if s else 0):>8.3f} "
              f"{(s['never_kills_rate'] if s else 0):>7.3f} "
              f"{c['mana_stdev_turn_five']:>7.2f} "
              f"{a['answer_cards']:>8} {a['classes_covered']:>6}")
    print("\n  No aggregate score: the formula is chosen against this spread "
          "(PRD §14.1), not before it.")
