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


def forge_record():
    """Wins and games per OUR seat, across every tracked sim run."""
    got = {}
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
        for seat, row in (analysis.get("seats") or {}).items():
            if row.get("wins") is None:
                continue
            # Only OUR decks: an opponent seat is a fetched EDHREC average list
            # with no goldfish figures of its own to correlate against.
            try:
                deck_dir(seat.split("@")[0])
            except Exception:
                continue
            a, b = got.get(seat, (0, 0))
            got[seat] = (a + row["wins"], b + n)
    return got


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
    record = forge_record()
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
           "eligible": len(rows), "with_any_forge_data": len(record)}
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
            f"{short} more deck(s) with {MIN_GAMES}+ games each — roughly "
            f"{short} runs of `manamap pilot simulate <slug> --vs <pod> "
            f"--games {MIN_GAMES}`.")
        return doc
    doc["verdict"] = "measured"
    doc["spearman"] = {
        name: round(_spearman([r["win_rate"] for r in rows],
                              [r[name] for r in rows]), 3)
        for name in MEASURES}
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
    print(f"\n  {doc['with_any_forge_data']} seat(s) have any Forge games; "
          f"{doc['eligible']} have {doc['min_games']}+.")
    if doc["verdict"] == "NOT ANSWERABLE":
        print(f"\n  VERDICT: NOT ANSWERABLE")
        print(f"    {doc['why']}")
        print(f"    {doc['what_it_would_take']}")
        return
    print(f"\n  Spearman(Forge win rate, model figure), n={doc['eligible']}:")
    for k, v in doc["spearman"].items():
        print(f"    {k:14} {v:+.3f}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot calibrate`.")
