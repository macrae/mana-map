"""The net change: what a branch would cost, what it would buy, and whether it met
what it set out to do.

THIS IS THE DOCUMENT A SPENDING DECISION RESTS ON. It was assembled by hand once —
eight commands and a page of HTML — to decide whether to buy 21 cards for the
Ur-Dragon treasure refactor. The answer was no, and the report is why the money
stayed in the bank. Doing that by hand again is how the next one gets skipped.

IT COMPOSES; IT COMPUTES ONE THING. Every figure here comes from a command that
already owns it — `diagnostic.compare` for the delta table with its intervals and
per-row MDE, `deck_branch.source` for the bill, the tracked `sim/*.json` for the
real table. The single new measurement is the ENGINE LIFT, and it earns its place:

    kill rate in games where the declared engine came online by T3
      MINUS
    kill rate in games where it did not

It is the only test of whether a deck's stated engine does anything. On ur-dragon
the champion reads **+0.085** (0.638 -> 0.722, interval excluding zero) and the
treasure branch reads **-0.029** — assembling what the branch claims to need makes
it win LESS, which no other figure in the suite would have said out loud.

WHAT IT REFUSES TO DO. It does not rank the two lists, score them, or recommend.
It states the objective the branch was opened with, grades it, and shows the
trade. Which side of a trade is worth taking is the pilot's, and this repo has
deleted the last thing that tried to have an opinion about that.
"""

import glob
import json

from manamap.pilot import deck_branch
from manamap.pilot.common import deck_dir, load_deck_cards, load_json
from manamap.sim import stats

ARTIFACT = "net_change.json"

#: What the report shows, and which direction is an improvement. Direction is
#: load-bearing: without it a lower stall reads as a loss. Same table
#: `diagnostic.interpret` keeps, and for the same reason.
ROWS = (
    ("hoard @T10", "output", "hoard_by_turn", "10", +1),
    ("hoard @T6", "output", "hoard_by_turn", "6", +1),
    ("damage @T10", "output", "damage_by_turn", "10", +1),
    ("board power @T6", "output", "board_power_by_turn", "6", +1),
    ("killed by T6", "output", "kill_by_turn", "6", +1),
    ("killed by T10", "output", "kill_by_turn", "10", +1),
    ("stall, 2 in a row", "stall", "two_in_a_row", None, -1),
    ("missed drop by T5", "mana", "missed_land_drop_by_five", None, -1),
    ("mulliganed", "mana", "mulliganed", None, -1),
)


def _cell(doc, block, key, turn=None):
    got = (doc.get(block) or {}).get(key)
    if turn and isinstance(got, dict):
        got = got.get(turn)
    return got if isinstance(got, dict) and "rate" in got else None


def engine_lift(slug, branch, iterations, seed):
    """Does assembling the declared engine make this list win?

    Split every iteration by whether every `required` target was assembled by
    turn three, and compare the kill rates. A deck whose engine does nothing —
    or whose engine competes with its own kill for cards and mana — says so
    here and nowhere else.
    """
    from manamap.pilot import diagnostic, goldfish
    targets = (load_json(deck_file_or_none(slug, branch)) or {}).get("targets") or []
    req = [i for i, t in enumerate(targets) if t.get("required")]
    if not req:
        return {"available": False,
                "why": "no `required` component is declared, so there is nothing "
                       "to call the engine and no lift to measure"}
    rows = goldfish.run(slug, branch=branch, with_results=True, quiet=True,
                        iterations=iterations, seed=seed,
                        max_turn=diagnostic.HARNESS["max_turn"])["_results"]
    if not rows or "kill_turn" not in rows[0]:
        return {"available": False,
                "why": "this list does not opt into `model_combat`, so there is "
                       "no kill to correlate the engine against"}

    def online(r):
        return all((r["target_turns"][i] is not None and r["target_turns"][i] <= 3)
                   for i in req)

    def killed(r):
        return r["kill_turn"] is not None and r["kill_turn"] <= 10

    on = [r for r in rows if online(r)]
    off = [r for r in rows if not online(r)]
    if not on or not off:
        return {"available": False,
                "why": (f"the engine was online in {len(on)} of {len(rows)} games — "
                        f"with one side empty there is nothing to compare")}
    k_on, k_off = sum(1 for r in on if killed(r)), sum(1 for r in off if killed(r))
    d = stats.diff_proportions(k_off, len(off), k_on, len(on))
    return {"available": True,
            "online": {"games": len(on), "kill_rate": round(k_on / len(on), 4)},
            "offline": {"games": len(off), "kill_rate": round(k_off / len(off), 4)},
            "lift": round(k_on / len(on) - k_off / len(off), 4),
            "ci95": d["ci95"], "excludes_zero": d.get("excludes_zero"),
            "reading": ("assembling the engine makes this list win more"
                        if k_on / len(on) > k_off / len(off) else
                        "assembling the engine makes this list win LESS — the "
                        "declared engine competes with the kill for cards and mana")}


def deck_file_or_none(slug, branch):
    from manamap.pilot.common import deck_file
    return deck_file(slug, "goldfish_targets.json", branch)


def forge(slug, branch):
    """The real table, if it has been played. Pooled within one pod only."""
    def rows_for(pattern, want):
        wins = games = 0
        by_route = {}
        for path in sorted(glob.glob(pattern)):
            if "logs" in path:
                continue
            doc = json.load(open(path))
            a = doc.get("analysis") or {}
            n = a.get("games") or 0
            for seat, v in (a.get("seats") or {}).items():
                if seat != want or v.get("wins") is None:
                    continue
                wins += v["wins"]
                games += n
            for g in (doc.get("games") or []):
                if g.get("winner") and want.split("@")[0] in str(g["winner"]):
                    by_route[g.get("won_by") or "unstated"] = \
                        by_route.get(g.get("won_by") or "unstated", 0) + 1
        return wins, games, by_route

    a_w, a_n, a_r = rows_for(f"data/decks/{slug}/sim/*.json", slug)
    b_w, b_n, b_r = rows_for(
        f"data/decks/{slug}/branches/{branch}/sim/*.json",
        deck_branch_seat(slug, branch))
    if not (a_n and b_n):
        return {"available": False,
                "why": ("no Forge run on " +
                        ("the branch" if a_n else "the deck") +
                        " — `manamap pilot simulate` puts both at a table")}
    d = stats.diff_proportions(a_w, a_n, b_w, b_n)
    m = stats.mde_proportion(a_w / a_n, a_n, b_n) or {}
    return {"available": True,
            "champion": {"wins": a_w, "games": a_n, "rate": round(a_w / a_n, 4),
                         "won_by": a_r},
            "branch": {"wins": b_w, "games": b_n, "rate": round(b_w / b_n, 4),
                       "won_by": b_r},
            "delta": round(b_w / b_n - a_w / a_n, 4),
            "ci95": d["ci95"], "excludes_zero": d.get("excludes_zero"),
            "mde": m.get("minimum_detectable_difference"),
            "caveat": ("Forge's AI is a weak pilot; the comparison is fair only "
                       "because both seats were played at a comparable rate — see "
                       "`sim/pilot_quality`.")}


def deck_branch_seat(slug, branch):
    from manamap.sim.forge import deck_meta_name
    return deck_meta_name(f"{slug}@{branch}")


def build(slug, branch, iterations=None, seed=None):
    from manamap.pilot import candidates, diagnostic
    it = iterations or diagnostic.HARNESS["iterations"]
    sd = seed if seed is not None else diagnostic.HARNESS["seed"]
    a = diagnostic.run(slug, iterations=it, seed=sd, quiet=True)
    b = diagnostic.run(slug, branch=branch, iterations=it, seed=sd, quiet=True)

    table = []
    for label, blk, key, turn, want in ROWS:
        ca, cb = _cell(a, blk, key, turn), _cell(b, blk, key, turn)
        if not (ca and cb):
            continue
        delta = round(cb["rate"] - ca["rate"], 4)
        mde = max(diagnostic.mde(ca), diagnostic.mde(cb))
        good = (delta > 0) == (want > 0)
        table.append({
            "measure": label, "champion": ca["rate"], "branch": cb["rate"],
            "delta": delta, "mde": round(mde, 4),
            "verdict": (("better" if good else "worse") if abs(delta) > mde
                        else "noise")})

    doc_meta = deck_branch.meta(slug, branch) or {}
    objective = doc_meta.get("objective")
    staged = len(doc_meta.get("staged") or [])
    grade = None
    if objective:
        block, key, turn = candidates.OBJECTIVE_AXES.get(
            objective["axis"], (None, None, None))
        cell = _cell(b, block, key, turn) if block else None
        grade = deck_branch.grade_objective(
            objective, cell["rate"] if cell else None,
            mde=diagnostic.mde(cell) if cell else None)

    doc = {
        "slug": slug, "branch": branch,
        "harness": {"iterations": it, "seed": sd},
        "decklist_sha256": (b.get("decklist_sha256")),
        "objective": objective,
        "objective_grade": grade,
        "staged": staged,
        "table": table,
        "engine_lift": {"champion": engine_lift(slug, None, it, sd),
                        "branch": engine_lift(slug, branch, it, sd)},
        "forge": forge(slug, branch),
        "bill": deck_branch.source(slug, branch),
        "limits": [
            "The goldfish has no opponents and nothing blocks: its kill turn is a "
            "CLOCK, not a win rate, and it cannot see interaction, removal or any "
            "alternate win.",
            "Each list's engine figure is measured against ITS OWN declaration. A "
            "larger number may be a smaller claim.",
            "Card advantage is measured nowhere in this suite.",
        ],
    }
    # Derived from the finished document, so it can never disagree with the rows
    # it summarises — the same reason `deck_info` composes and computes nothing.
    doc["recommendation"] = recommend(doc)
    return doc


def main(args):
    branch = getattr(args, "branch", None)
    if not branch:
        raise SystemExit(
            f"net-change compares a BRANCH against the deck. "
            f"`--branch <name>`; `manamap pilot deck-branch {args.slug} list` "
            f"shows what there is.")
    doc = build(args.slug, branch,
                iterations=getattr(args, "iterations", None),
                seed=getattr(args, "seed", None))
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(json.dumps(doc, indent=1))
    else:
        _print(doc)
    if getattr(args, "write", False):
        out = deck_dir(args.slug, branch) / ARTIFACT
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"\n  Wrote {out}")


#: The states a recommendation may be in. A merge decision is not a scalar, and
#: five words is the whole vocabulary.
STATES = ("merge", "a trade", "do not merge", "inconclusive", "no objective")


def recommend(doc):
    """Sort the table into what rose, what fell, and what the run cannot tell.

    THE AXES MOVE IN BOTH DIRECTIONS ON PURPOSE, so a single number would have to
    weight them and every weight here would be invented — this repo deleted a
    six-factor card scorer for exactly that. What a pilot needs instead is the
    LEDGER plus a rule stated plainly enough to argue with:

        objective met, nothing fell      -> merge
        objective met, something fell    -> a trade; name both sides and the bill
        objective not met                -> do not merge
        objective stated but unreadable  -> inconclusive, and say WHY it is
        no objective at all              -> no objective; the ledger still stands

    THE LAST TWO ARE DIFFERENT AND WERE ONE STATE IN THE FIRST DRAFT. A branch
    that stated a goal the run could not read has been falsifiable all along and
    simply was not measured; a branch that stated none never could be. Collapsing
    them would let the second borrow the credibility of the first, which is the
    Ur-Dragon treasure branch's exact failure — it hit "treasure is the engine"
    4.4x over and missed the purpose nobody wrote down.

    NOTHING HERE RE-MEASURES. Every row's `verdict` was set against its own MDE in
    `build`; this reads them.
    """
    table = doc.get("table") or []
    rose = [r["measure"] for r in table if r["verdict"] == "better"]
    fell = [r["measure"] for r in table if r["verdict"] == "worse"]
    no_call = [r["measure"] for r in table if r["verdict"] == "noise"]

    objective, grade = doc.get("objective"), doc.get("objective_grade") or {}
    state = grade.get("state")
    if not objective:
        out = ("no objective",
               "This branch never stated what it was for, so nothing here can "
               "say whether it worked — only what changed.")
    elif state == "met":
        if fell:
            out = ("a trade",
                   f"You buy {_and(rose)} and pay {_and(fell)}.")
        else:
            out = ("merge",
                   f"The objective is met and nothing measured here got worse"
                   + (f"; {_and(rose)} improved." if rose else "."))
    elif state == "not met":
        out = ("do not merge",
               f"The objective is not met"
               + (f" — {grade.get('why')}" if grade.get("why") else ".")
               + (f" {_and(rose).capitalize()} improved anyway, which is a "
                  f"different branch's case." if rose else ""))
    elif state == "not resolvable":
        out = ("inconclusive",
               f"The miss is smaller than this run can see. "
               f"{grade.get('why', '')} A larger N is the only thing that "
               f"settles it.")
    else:                                            # "not measured", or absent
        out = ("inconclusive",
               f"The objective names {objective.get('axis')}, and this list has "
               f"no reading for it. That is a missing measurement, not a failure "
               f"— the axis may need a model flag set in goldfish_targets.json.")

    got = {"state": out[0], "because": out[1],
           "rose": rose, "fell": fell, "no_call": no_call,
           "bill": (doc.get("bill") or {}).get("counts") or {}}

    notes = []
    # THE MEASUREMENT IS DECK-LEVEL: swap a handful of cards, measure the lift.
    # One card USUALLY will not register — a 100-card singleton dilutes it below
    # what the run can resolve — but that is a statement about the typical card,
    # not a law. A Game Changer or a table-warper moves a number on its own, and
    # some cards are. So a blank table on a barely-changed branch is arithmetic
    # rather than a verdict on the swaps, and reading it as "these did nothing"
    # is the wrong lesson from a correct measurement. Measured: a one-swap
    # branch of ur-dragon returned noise on all nine rows.
    staged = doc.get("staged")
    if table and not rose and not fell:
        head = (f"Nothing moved: all {len(no_call)} measures came back inside "
                f"this run's minimum detectable difference.")
        if staged is not None and 0 < staged <= 3:
            notes.append(
                f"{head} With {staged} swap(s) staged this branch is nearly the "
                f"deck. A 100-card singleton dilutes one card below what this "
                f"run can resolve unless it is a Game Changer or a table-warper "
                f"— so this is not a verdict on the swap(s). Stage the rest of "
                f"the treatment and measure the lift on the whole thing.")
        else:
            notes.append(
                f"{head} The change is real and smaller than this run can "
                f"resolve — an answer about its SIZE, not a failure to measure.")

    # THE REAL TABLE IS EVIDENCE THE RULE DOES NOT USE, and hiding it because the
    # rule ignores it would be the worse error. Named beside the verdict, never
    # folded into it.
    f = doc.get("forge") or {}
    if f.get("available") and f.get("ci95"):
        lo, hi = f["ci95"]
        notes.append(
            f"Forge, against a real pod: {f['delta']:+.3f} win rate, "
            f"CI [{lo:+.3f}, {hi:+.3f}] — "
            + ("this run cannot separate the two lists."
               if (lo <= 0 <= hi) else "the difference excludes zero."))
    lift = (doc.get("engine_lift") or {}).get("branch") or {}
    if lift.get("available") and lift.get("excludes_zero") is False:
        notes.append(
            "The branch's own engine does not measurably change whether it "
            "wins — its lift spans zero, so a bigger engine is not yet a "
            "better deck.")
    got["notes"] = notes
    return got


def _and(names):
    if not names:
        return "nothing"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _print(doc):
    h = doc["harness"]
    print(f"\nNET CHANGE — {doc['slug']} vs branch {doc['branch']}"
          f"   ({h['iterations']:,} games each, seed {h['seed']})")

    rec = doc.get("recommendation")
    if rec:
        print(f"\n  ==> {rec['state'].upper()}")
        print(f"      {rec['because']}")
        for n in rec.get("notes") or []:
            print(f"      {n}")

    o, g = doc.get("objective"), doc.get("objective_grade")
    print("\n  OBJECTIVE")
    if not o:
        print("    NONE — this branch predates the requirement and cannot be graded.")
    else:
        print(f"    {o['axis']} {o['op']} {o['value']}"
              + (f"   — {o['why']}" if o.get("why") else ""))
        state = (g or {}).get("state", "?").upper()
        print(f"    RESULT   {(g or {}).get('reading', '—')}   ->   {state}")
        if (g or {}).get("why"):
            print(f"             {g['why']}")

    print("\n  MEASURED")
    print(f"    {'measure':20} {'champion':>10} {'branch':>10} {'delta':>9}  verdict")
    for r in doc["table"]:
        print(f"    {r['measure']:20} {r['champion']:>10.3f} {r['branch']:>10.3f} "
              f"{r['delta']:>+9.3f}  {r['verdict']}"
              + (f" (MDE {r['mde']:.3f})" if r["verdict"] == "noise" else ""))

    print("\n  DOES THE ENGINE MAKE IT WIN?")
    for who in ("champion", "branch"):
        e = doc["engine_lift"][who]
        if not e.get("available"):
            print(f"    {who:10} not measured — {e['why'][:64]}")
            continue
        print(f"    {who:10} {e['lift']:+.3f}   "
              f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
              f"   {'real' if e['excludes_zero'] else 'spans zero'}")
        print(f"    {'':10} {e['reading']}")

    f = doc["forge"]
    print("\n  THE REAL TABLE")
    if not f.get("available"):
        print(f"    {f['why']}")
    else:
        print(f"    champion {f['champion']['wins']}/{f['champion']['games']} "
              f"({f['champion']['rate']:.3f})   "
              f"branch {f['branch']['wins']}/{f['branch']['games']} "
              f"({f['branch']['rate']:.3f})")
        print(f"    delta {f['delta']:+.3f}  CI [{f['ci95'][0]:+.3f}, "
              f"{f['ci95'][1]:+.3f}]  MDE {f['mde']}")
        if f["mde"] and abs(f["delta"]) < f["mde"]:
            print(f"    UNDERPOWERED — this run could only resolve a difference of "
                  f"{f['mde']}; it rules out a large effect and cannot say which "
                  f"list is better.")

    c = doc["bill"]["counts"]
    print(f"\n  THE BILL   " + "   ".join(f"{k}={v}" for k, v in c.items()))
    owned = c.get("in_deck", 0) + c.get("box", 0) + c.get("elsewhere", 0)
    print(f"    you already own {owned} of {sum(c.values())}")

    for line in doc["limits"]:
        print(f"\n  · {line}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot net-change <slug> --branch <name>`.")
