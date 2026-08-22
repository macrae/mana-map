"""Who does the table attack, and why — measured, not assumed.

WHAT THIS IS, AND WHAT IT IS NOT. Every other measurement in this repo is about
OUR deck: how fast it develops, what it kills, whether a change helped. This one
is about the OPPONENTS' choices. In a four-player game the central strategic
tension is that presenting the biggest board makes the table point at you, and
until now nothing here could say whether that is true, or by how much.

It is empirical opponent modelling: the input half of a game-theoretic argument.
Measure the pod's target-selection policy, and you can best-respond to it. It is
NOT an equilibrium, a solution concept, or a model of human politics — there are
no deals here, no grudges, no table talk, and no player who remembers what you
did last turn. Do not let it be described as "game theory" without the words
"Forge's AI" attached, which is why the artifact key is
`forge_ai_targeting_policy` and not something that would survive a slide deck
with the caveat trimmed off.

THE UNIT IS A DECISION, NOT A GAME. One declare-attackers step by one seat with
at least TWO living opponents — because a forced choice is not a choice, and a
one-on-one run contributes nothing by construction rather than by being excluded
by hand. That reframing is what makes the question answerable: twelve games is
nothing, but twelve games hold hundreds of targeting decisions.

Not per attacking creature: five creatures almost always go at the same player,
so counting them separately would multiply the sample without adding a single
independent choice.

WHY STRENGTH IS REVEALED DAMAGE AND NOT BOARD POWER. Forge prints a creature's
PRINTED power on resolution, and `bridge.py` already parses it — but counters,
anthems, auras, equipment and token counts are all invisible in the log. A
board-power ranking would therefore be biased against exactly the archetypes at
this pod: token decks, counter decks, anthem decks. Cumulative combat damage
already dealt to players is exact, is in the log, and is what an opponent can
actually observe. It is a revealed measure rather than a stated one.
"""

import json
import random
from datetime import date

from manamap.pilot.common import deck_dir
from manamap.sim.forge import ASSUMPTIONS, FORGE_AI_CAVEAT, SIM_DIR
from manamap.sim.parse import mean_ci, parse_games, wilson

STARTING_LIFE = 40
THREAT_DIR = "threat"
DEFAULT_ITERATIONS = 10000


# Each hypothesis ranks the seats a seat could attack. A "hit" is the chosen
# defender landing in the top set — TIES INCLUDED, and the null expectation for
# that decision is the size of the tied set over the size of the choice set.
# Early on every seat is at forty life, and a rule that broke ties arbitrarily
# would manufacture signal out of a three-way tie.
HYPOTHESES = {
    "most_damage_dealt": ("attacks the seat that has dealt the most combat damage",
                          lambda st, s: st["dealt"].get(s, 0), max),
    "lowest_life": ("attacks the lowest-life seat",
                    lambda st, s: st["life"].get(s, STARTING_LIFE), min),
    "highest_life": ("attacks the highest-life seat",
                     lambda st, s: st["life"].get(s, STARTING_LIFE), max),
}


def decisions(game):
    """Every targeting decision in one game, with the state as it stood.

    Walks the event stream in order, so the state attached to a decision is the
    state BEFORE that attack — which is the only state the attacker could have
    been reacting to.
    """
    seats = list(game["seats"])
    life = {s: STARTING_LIFE for s in seats}
    dealt = {s: 0 for s in seats}
    owner = game.get("owner") or {}
    out = []
    for ev in game["events"]:
        kind = ev["kind"]
        if kind == "life":
            life[ev["seat"]] = ev["to"]
        elif kind == "damage" and ev.get("combat") and ev.get("to_player"):
            src = owner.get((ev.get("source") or (None, None))[1])
            if src:
                dealt[src] = dealt.get(src, 0) + int(ev.get("amount") or 0)
        elif kind == "attack":
            attacker, defender = ev["seat"], ev.get("defender")
            if not defender:
                continue
            choices = [s for s in seats
                       if s != attacker and life.get(s, STARTING_LIFE) > 0]
            # A forced choice is not a choice. This is also what makes a 1v1 run
            # contribute zero decisions rather than needing to be filtered out.
            if len(choices) < 2 or defender not in choices:
                continue
            out.append({"turn": ev.get("turn"), "attacker": attacker,
                        "defender": defender, "choices": list(choices),
                        "life": {s: life.get(s, STARTING_LIFE) for s in choices},
                        "dealt": {s: dealt.get(s, 0) for s in choices}})
    return out


def _score(decision, key):
    """(hit, expected-under-uniform) for one decision under one hypothesis."""
    _, value_of, pick = HYPOTHESES[key]
    choices = decision["choices"]
    values = [value_of(decision, s) for s in choices]
    best = pick(values)
    tied = [s for s, v in zip(choices, values) if v == best]
    return decision["defender"] in tied, len(tied) / len(choices)


def _permutation_p(hits, expected, seed, iterations):
    """Is the observed hit count beyond what uniform targeting would give?

    Each decision has its OWN choice-set size, so the null is a sum of Bernoullis
    with different probabilities rather than one binomial — which is exactly why
    this is simulated rather than looked up. Seeded, so it replays.
    """
    rng = random.Random(seed)
    n = len(expected)
    if not n:
        return None
    extreme = 0
    for _ in range(iterations):
        total = 0
        for p in expected:
            if rng.random() < p:
                total += 1
        if total >= hits:
            extreme += 1
    return round((extreme + 1) / (iterations + 1), 4)


def analyse(all_decisions, per_game, seed=0, iterations=DEFAULT_ITERATIONS):
    """Rates, intervals and a permutation p for each hypothesis."""
    n = len(all_decisions)
    out = {}
    for key, (label, _, _) in HYPOTHESES.items():
        scored = [_score(d, key) for d in all_decisions]
        hits = sum(1 for h, _ in scored if h)
        expected = [e for _, e in scored]
        lo, hi = wilson(hits, n)
        # Decisions cluster inside games, so the pooled interval is optimistic.
        # The game-clustered mean is reported beside it and they are allowed to
        # disagree; a single number here would be hiding the dependence.
        by_game = [sum(1 for h, _ in (_score(d, key) for d in ds) if h) / len(ds)
                   for ds in per_game if ds]
        out[key] = {
            "hypothesis": label,
            "decisions": n, "hits": hits,
            "rate": round(hits / n, 4) if n else None,
            "ci95": [lo, hi],
            "uniform_expected_rate": round(sum(expected) / n, 4) if n else None,
            "permutation_p": _permutation_p(hits, expected, seed, iterations),
            "per_game_rate": mean_ci(by_game),
        }
    return out


def _best_set(decision, key):
    _, value_of, pick = HYPOTHESES[key]
    values = [value_of(decision, s) for s in decision["choices"]]
    best = pick(values)
    return {s for s, v in zip(decision["choices"], values) if v == best}


def contested(all_decisions, a="most_damage_dealt", b="lowest_life"):
    """The decisions where the two leading hypotheses point at DIFFERENT seats.

    THIS IS THE HONEST HALF OF THE METRIC. "Attacks the biggest threat" and
    "attacks the easiest kill" agree most of the time — a seat that has been
    hitting people is usually also the one that has been hit back — so a headline
    rate over ALL decisions cannot separate them, and quoting one of them as the
    pod's policy would be reading a correlation as a mechanism.

    Restricting to the disagreements is the only way to ask which the AI is
    actually following, and on the current sample the answer is that we cannot
    tell. That belongs in the artifact next to the result that IS significant.

    The three outcomes are mutually exclusive, so these are shares of one
    multinomial, not three independent proportions: comparing `a` to `b` here
    needs an interval on the difference of CORRELATED proportions, which is why
    none is offered and the note says so.
    """
    split = [d for d in all_decisions if not (_best_set(d, a) & _best_set(d, b))]
    n = len(split)
    if not n:
        return None
    hit_a = sum(1 for d in split if d["defender"] in _best_set(d, a))
    hit_b = sum(1 for d in split if d["defender"] in _best_set(d, b))
    neither = n - hit_a - hit_b
    def share(k):
        lo, hi = wilson(k, n)
        return {"hits": k, "rate": round(k / n, 4), "ci95": [lo, hi]}
    return {
        "decisions": n,
        "note": ("only the decisions where the two hypotheses name different seats. "
                 "The three outcomes are mutually exclusive shares of one multinomial, "
                 "so the intervals below are marginal and comparing two of them needs "
                 "an interval on the difference of CORRELATED proportions — not "
                 "offered here, and not to be eyeballed from the overlap."),
        a: share(hit_a), b: share(hit_b), "neither": share(neither),
    }


def _limits(runs, games, n):
    return [
        "THIS MEASURES FORGE'S AI TARGET SELECTION. It is not human politics: "
        "there are no deals, no grudges, no table talk, and no player who "
        "remembers what you did last turn.",
        FORGE_AI_CAVEAT,
        "Four fixed decks in four fixed seats. Any policy measured here is "
        "confounded with deck identity and with turn order — it is a statement "
        "about this pod, not about Commander.",
        "Strength is REVEALED cumulative combat damage dealt to players. Printed "
        "board power is only partly recoverable from a Forge log — counters, "
        "anthems, auras, equipment and token counts are invisible — so a "
        "board-power ranking would be biased against token, counter and anthem "
        "decks, which is most of this pod.",
        "The unit is one declare-attackers step with at least two living "
        "opponents, not one attacker: five creatures almost always go at the "
        "same player. A one-on-one run contributes zero decisions by "
        "construction.",
        "Decisions cluster within games, so the pooled Wilson interval is "
        "optimistic; the game-clustered mean is reported beside it.",
        f"Pooled over {runs} run(s), {games} game(s), {n} decision(s).",
        *ASSUMPTIONS[:1],
    ]


def build(slug, run_ids=None, seed=0, iterations=DEFAULT_ITERATIONS):
    """Pool every available run's logs into one policy measurement.

    Pooling is the point: this is a question about the POD, not about one run, and
    one run of a dozen games holds a couple of hundred decisions where the whole
    fleet holds several times that.
    """
    # Runs AND experiment arms. An experiment is the same pod playing the same
    # game, and the question here is about the pod's policy, not about our
    # decklist — so excluding them would throw away roughly half the sample to
    # protect against a confound (`our` seat's list varying) that is already
    # named in `limits` and applies to the pod's four fixed decks anyway.
    sources = []
    for sub in (SIM_DIR, "experiments"):
        base = deck_dir(slug) / sub
        if base.is_dir():
            sources.extend((p, base / "logs" / p.stem) for p in sorted(base.glob("*.json")))
    if run_ids:
        wanted = {r if r.endswith(".json") else f"{r}.json" for r in run_ids}
        sources = [(p, d) for p, d in sources if p.name in wanted]
    if not sources:
        raise SystemExit(f"{slug}: no sim runs or experiments — run `simulate` first")

    all_dec, per_game, used, games = [], [], [], 0
    for rec_path, log_dir in sources:
        logs = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
        if not logs:
            continue
        used.append(rec_path.stem)
        for log in logs:
            for g in parse_games(log.read_text(encoding="utf-8", errors="replace")):
                games += 1
                ds = decisions(g)
                if ds:
                    per_game.append(ds)
                    all_dec.extend(ds)
    if not all_dec:
        raise SystemExit(
            f"{slug}: no targeting decisions found. Logs are gitignored and only "
            f"exist where the run happened; a one-on-one run has none by design.")

    return {
        "slug": slug, "at": date.today().isoformat(),
        "runs": used, "games": games, "decisions": len(all_dec),
        "seed": seed, "permutation_iterations": iterations,
        "forge_ai_targeting_policy": analyse(all_dec, per_game, seed, iterations),
        "when_the_hypotheses_disagree": contested(all_dec),
        "limits": _limits(len(used), games, len(all_dec)),
    }


def main(args):
    doc = build(args.slug, run_ids=getattr(args, "run", None) or None,
                seed=getattr(args, "seed", None) or 0,
                iterations=getattr(args, "iterations", None) or DEFAULT_ITERATIONS)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
        return
    out_dir = deck_dir(args.slug) / THREAT_DIR
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "targeting.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"TARGETING POLICY — {args.slug}  "
          f"({doc['decisions']} decisions, {doc['games']} games, "
          f"{len(doc['runs'])} run(s))\n")
    for h in doc["forge_ai_targeting_policy"].values():
        ci = h["ci95"]
        print(f"  {h['hypothesis']:<52} {h['rate']:.3f}  "
              f"ci95 [{ci[0]}, {ci[1]}]  uniform {h['uniform_expected_rate']:.3f}  "
              f"p {h['permutation_p']}")
    c = doc.get("when_the_hypotheses_disagree")
    if c:
        print(f"\n  where they DISAGREE ({c['decisions']} decisions):")
        for k in ("most_damage_dealt", "lowest_life", "neither"):
            s = c[k]
            print(f"    {k:<20} {s['rate']:.3f}  ci95 [{s['ci95'][0]}, {s['ci95'][1]}]")
    print(f"\n  {doc['limits'][0]}")
    print(f"  -> {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot targeting <slug>`.")
