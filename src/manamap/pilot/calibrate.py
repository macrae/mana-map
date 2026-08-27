"""Does the model predict anything? The question nothing here could answer.

EVERY FIGURE IN THIS SUITE IS INTERNALLY CONSISTENT AND NONE IS VALIDATED. The
goldfish has intervals, MDEs, placebo controls and honest refusals; what it does
not have is any demonstrated relationship to an outcome. Forge is the only
ground truth in the repo, and there is almost none of it — 5 decks with any
games, 4 with n >= 20, and Spearman(Forge win rate, kill-by-8) = -0.10 over all
five. That is not a validation. It is the ABSENCE of one, and until this command
existed nothing said so out loud.

SO THIS REFUSES A VERDICT RATHER THAN PRODUCING A WEAK ONE. Below `MIN_DECKS`
with `MIN_GAMES` each it reports what is missing and what it would take. A rank
correlation on five points has a confidence interval spanning almost the whole
range, and printing `rho = -0.10` without that is how a null gets read as a
finding — the failure this repo has rejected in three other places.

It is also the argument for where compute goes. The whole deterministic layer is
~53s per deck against a 5-10 minute budget; the only thing that blows that
budget is Forge, and the only thing with external validity is Forge.
"""

import glob
import json

from manamap.pilot.common import deck_dir

#: Below this the command reports the gap instead of a coefficient. At n=5 a
#: Spearman needs |rho| > 0.9 to clear p<0.05; at n=8 it needs 0.74, at n=10
#: 0.65. Ten decks is the first point where an ordinary effect is detectable.
MIN_DECKS = 10
#: A deck's Forge win rate is a proportion; at n=20 its 95% interval is about
#: +/-0.18 wide, which is most of the range this fleet occupies (0.00-0.21).
MIN_GAMES = 20

#: The model figures worth correlating, at the benchmark's frozen harness.
#: `damage_8` stands in for the whole combat family — see the note in
#: `candidates.AXES`; correlating all three would be one test counted thrice.
MEASURES = {
    "kill_by_8": ("combat", "kill_by_turn_rate", "8"),
    "damage_8": ("combat", "mean_damage_by_turn", "8"),
    "mana_7": (None, "mean_available_mana_by_turn", "7"),
    "cmdr_turn": ("commander", "mean_cast_turn", None),
}

#: WHICH WAY EACH MEASURE SHOULD POINT IF THE MODEL IS TRACKING ANYTHING.
#: Without this the command reports its strongest correlation as its best
#: result, and on the first real run that was **`cmdr_turn` at +0.760** — i.e.
#: decks whose commander casts LATER win MORE. Read as a validation that is
#: "cast your commander later"; read honestly it is an expensive commander
#: standing in for an expensive DECK (gishath 7.8, ur-dragon 8.0 and edgar 6.5
#: are three of the four best win rates; hapatra 2.1, sisay 3.2 and radagast 3.9
#: are three of the four worst). A correlation with the WRONG SIGN is evidence
#: of a confound, never of the model working, and the two look identical in a
#: table of coefficients.
EXPECTED = {"kill_by_8": +1, "damage_8": +1, "mana_7": +1, "cmdr_turn": -1}

#: Two-tailed p<0.05 critical values for Spearman's rho. A coefficient below the
#: line is not a weak finding, it is no finding — and printing it bare is how a
#: null gets read as one.
RHO_CRITICAL = {8: 0.738, 9: 0.700, 10: 0.648, 11: 0.618, 12: 0.587,
                13: 0.560, 14: 0.538, 15: 0.521, 16: 0.503, 18: 0.475,
                20: 0.450, 25: 0.400, 30: 0.364}


def critical_rho(n):
    """The |rho| this many decks needs to clear p<0.05, two-tailed."""
    if n < min(RHO_CRITICAL):
        return 1.0
    return RHO_CRITICAL[min(RHO_CRITICAL, key=lambda k: abs(k - n))]


def forge_record(pod=None):
    """Wins and games per OUR seat, POOLED ONLY WITHIN ONE POD.

    A WIN RATE IS AGAINST SOMEBODY, and the first cut of this function summed
    every tracked run regardless. Measured on what exists: kianne's 24 games are
    12 against the standard pod and **12 in a 1v1 against giada alone** — a
    different game, with no politics and no second threat — while radagast's 28
    are 20 against the standard pod and **8 against a pod of our OWN decks**.
    Pooling those produces a number that is not a win rate against anything.

    So runs are grouped by their opponent set and only the MODAL pod is kept.
    Modal rather than hardcoded, because the canonical table is whatever the
    pilot has actually been playing and a constant here would go stale silently.
    Everything dropped is reported: a smaller sample the caller knows about
    beats a larger one it does not.
    """
    by_pod = {}
    for seat, opponents, wins, games in _seat_rows():
        by_pod.setdefault(opponents, []).append((seat, wins, games))
    if not by_pod:
        return {}, None, []
    if pod is None:
        # Most GAMES, not most runs: one 100-game run is better evidence than
        # three 8-game ones, and counting runs would prefer the noise.
        pod = max(by_pod, key=lambda k: sum(g for _, _, g in by_pod[k]))
    got = {}
    for seat, wins, games in by_pod.get(pod, []):
        a, b = got.get(seat, (0, 0))
        got[seat] = (a + wins, b + games)
    dropped = [{"pod": sorted(k), "games": sum(g for _, _, g in v),
                "seats": sorted({s for s, _, _ in v})}
               for k, v in by_pod.items() if k != pod]
    return got, sorted(pod), dropped


def _seat_rows():
    """(our seat, frozenset of opponents, wins, games) per tracked run."""
    out = []
    for path in sorted(glob.glob("data/decks/**/sim/*.json", recursive=True)):
        if "logs" in path:
            continue
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        analysis = doc.get("analysis") or {}
        n = (analysis.get("games") or (doc.get("summary") or {}).get("games")
             or len(doc.get("games") or []) or 0)
        seats = analysis.get("seats") or {}
        for seat, row in seats.items():
            if row.get("wins") is None:
                continue
            # Only OUR decks: an opponent seat is a fetched EDHREC average list
            # with no goldfish figures of its own to correlate against.
            try:
                deck_dir(seat.split("@")[0])
            except Exception:
                continue
            out.append((seat, frozenset(seats) - {seat}, row["wins"], n))
    return out


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    a, b = rank(xs), rank(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def calibrate(iterations=3000):
    from manamap.pilot import goldfish
    record, pod, dropped = forge_record()
    usable = {s: v for s, v in record.items() if v[1] >= MIN_GAMES}
    rows = []
    for seat, (wins, games) in sorted(usable.items()):
        slug, _, branch = seat.partition("@")
        try:
            got = goldfish.run(slug, branch=branch or None, iterations=iterations,
                               seed=20260826, max_turn=10, model_treasures=True,
                               model_combat=True, quiet=True)
        except Exception:
            continue
        m = got["metrics"]
        vals = {}
        for name, (block, key, sub) in MEASURES.items():
            cell = (m[block][key] if block else m[key])
            vals[name] = cell[sub] if sub else cell
        rows.append({"seat": seat, "wins": wins, "games": games,
                     "win_rate": round(wins / games, 4), **vals})

    doc = {"decks": rows, "min_decks": MIN_DECKS, "min_games": MIN_GAMES,
           "eligible": len(rows), "with_any_forge_data": len(record),
           "pod": pod, "dropped_other_pods": dropped}
    if len(rows) < MIN_DECKS:
        short = MIN_DECKS - len(rows)
        doc["verdict"] = "NOT ANSWERABLE"
        doc["why"] = (
            f"{len(rows)} deck(s) have {MIN_GAMES}+ Forge games; {MIN_DECKS} are "
            f"needed before a rank correlation can distinguish an ordinary "
            f"effect from noise (at n=5 a Spearman needs |rho|>0.9 for p<0.05; "
            f"at n=10, 0.65). Reporting a coefficient here would publish a null "
            f"as a finding.")
        doc["what_it_would_take"] = (
            f"{short} more deck(s) with {MIN_GAMES}+ games each against THIS "
            f"pod — `manamap pilot simulate <slug> --vs "
            f"{' '.join(pod or ['<pod>'])} --games {MIN_GAMES}`.")
        return doc
    doc["verdict"] = "measured"
    n = len(rows)
    doc["critical_rho"] = critical_rho(n)
    out = {}
    for name in MEASURES:
        rho = round(_spearman([r["win_rate"] for r in rows],
                              [r[name] for r in rows]), 3)
        agrees = (rho >= 0) == (EXPECTED[name] > 0)
        out[name] = {
            "rho": rho,
            "expected_direction": "higher is better" if EXPECTED[name] > 0
                                  else "lower is better",
            "sign_agrees": agrees,
            "significant": abs(rho) >= doc["critical_rho"],
            "reading": (
                "tracks the outcome" if agrees and abs(rho) >= doc["critical_rho"]
                else "points the right way but does not clear significance at "
                     f"n={n}" if agrees
                else "WRONG SIGN — this is evidence of a confound, not of the "
                     "model working"),
        }
    doc["spearman"] = out
    # A win rate at n=20 has a 95% interval about +/-0.18 wide against a fleet
    # spanning 0.00-0.30, and two seats carry 100+ games while nine carry ~20.
    # A rank correlation weights them equally; saying so is cheaper than
    # pretending otherwise.
    doc["limits"] = [
        f"n={n} decks. |rho| must reach {doc['critical_rho']} for p<0.05.",
        "Unequal samples: two seats carry 100+ games and the rest ~20, and a "
        "rank correlation weights them equally.",
        "Forge's AI is a weak pilot (see sim/pilot_quality); this measures "
        "whether the model tracks THAT table, not a human one.",
    ]
    return doc


def main(args):
    doc = calibrate(iterations=getattr(args, "iterations", None) or 3000)
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1)); return
    print(f"\nCALIBRATION — does the model track real outcomes?\n")
    print(f"  {'seat':30} {'forge win':>10} {'games':>6} " +
          "".join(f"{k:>11}" for k in MEASURES))
    for r in doc["decks"]:
        print(f"  {r['seat']:30} {r['win_rate']:>10.3f} {r['games']:>6} " +
              "".join(f"{r[k]:>11.3f}" for k in MEASURES))
    print(f"\n  pod: {', '.join(doc['pod'] or ['—'])}")
    for d in doc["dropped_other_pods"]:
        print(f"  dropped {d['games']:>4} game(s) vs a DIFFERENT pod "
              f"({', '.join(d['pod'])}) — a win rate is against somebody")
    print(f"  {doc['with_any_forge_data']} seat(s) have games vs this pod; "
          f"{doc['eligible']} have {doc['min_games']}+.")
    if doc["verdict"] == "NOT ANSWERABLE":
        print(f"\n  VERDICT: NOT ANSWERABLE")
        print(f"    {doc['why']}")
        print(f"    {doc['what_it_would_take']}")
        return
    print(f"\n  Spearman(Forge win rate, model figure), n={doc['eligible']}"
          f"   (|rho| >= {doc['critical_rho']} for p<0.05)")
    for k, v in doc["spearman"].items():
        mark = "  " if v["sign_agrees"] else "!!"
        print(f"  {mark}{k:14} {v['rho']:+.3f}  ({v['expected_direction']})"
              f"  {v['reading']}")
    for line in doc["limits"]:
        print(f"\n  · {line}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot calibrate`.")
