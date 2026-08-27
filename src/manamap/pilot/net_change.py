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
    grade = None
    if objective:
        block, key, turn = candidates.OBJECTIVE_AXES.get(
            objective["axis"], (None, None, None))
        cell = _cell(b, block, key, turn) if block else None
        grade = deck_branch.grade_objective(
            objective, cell["rate"] if cell else None,
            mde=diagnostic.mde(cell) if cell else None)

    return {
        "slug": slug, "branch": branch,
        "harness": {"iterations": it, "seed": sd},
        "decklist_sha256": (b.get("decklist_sha256")),
        "objective": objective,
        "objective_grade": grade,
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


def _print(doc):
    h = doc["harness"]
    print(f"\nNET CHANGE — {doc['slug']} vs branch {doc['branch']}"
          f"   ({h['iterations']:,} games each, seed {h['seed']})")

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
