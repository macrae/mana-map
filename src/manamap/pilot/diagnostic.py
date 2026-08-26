"""Pilot: one diagnostic run — the deck's vitals, measured, with intervals.

    manamap pilot diagnose <slug> [--branch NAME] [--vs main] [--json]

THE LOOP THIS CLOSES is measure -> change -> re-measure. Everything here already
had a home; what was missing was a single reading that can be taken before and
after a change and compared honestly. The whole existing stack costs 12.6s on one
deck, so re-measuring is cheap enough to do on every swap rather than as a
ceremony.

WHAT IT ADDS OVER `benchmark`, and why both exist. `benchmark` freezes a harness
so twelve decks can be put beside each other; it reads the 99 and ignores the
declaration on purpose, because a benchmark that read one would rank decks partly
on how well their pilot writes JSON. This reads THE DECLARATION, and asks a
different question: is this deck doing what IT says it does. The two must not be
merged — one is cross-deck and strategy-blind, the other is strategy-relative and
never ranked against another deck.

NO AGGREGATE SCORE, AND THAT REFUSAL IS UNCHANGED. `benchmark.py` records why:
`speed` spans 400x across the fleet and ranks a combo deck last for not
attacking, `consistency` was `speed` under another name at r=0.78, and two of the
four inputs correlate at 0.97. Grading here is PER AXIS and always against this
deck's own declared target, which dissolves the archetype problem rather than
hiding it — heliod killing slowly is a fault only if heliod claims to kill fast.

EVERY RATE CARRIES AN INTERVAL. `sim/stats.wilson_bounds` already exists and a
rate published without one is what this repo refuses everywhere else.
"""

import json
import statistics

from manamap.pilot.common import deck_dir, deck_file, load_json
from manamap.sim import stats as st

ARTIFACT = "diagnostic.json"

#: The frozen configuration for a diagnostic reading. Separate from
#: `benchmark.HARNESS` because they answer different questions and must be
#: versioned independently — a benchmark comparable across decks, a diagnostic
#: comparable across VERSIONS of one deck.
HARNESS = {
    "version": 1,
    "iterations": 10000,
    "seed": 20260826,
    "max_turn": 10,
}

#: Turn windows the diagnostic reports against. Turn 1 is excluded from the
#: headline stall figure on purpose and the reason is measured: on ur-dragon
#: P(stall) is 0.629 on turn one and 0.040 by turn three, because a Commander
#: deck has one mana on turn one and almost no one-drops. Including it would
#: make the metric mostly a restatement of that structural fact.
STALL_FROM_TURN = 2


#: WHERE THE FLEET SITS, so a reading can be placed without being GRADED against
#: other decks. Measured across all 13 tracked decks at 4,000 games each on
#: 2026-08-26. It is context, never a verdict: the grade is strategy-relative and
#: comes from the deck's own declaration, because a combo deck is not unhealthy
#: for scoring badly on a brawler's axis.
#:
#: `tests/test_pilot_diagnostic.py` re-derives these from the fleet, so they
#: cannot outlive their evidence — the discipline `scaffold_targets.BROAD_GROUP`
#: already uses.
FLEET = {
    "stall_two_in_a_row": {"min": 0.008, "median": 0.036, "max": 0.079},
    "missed_land_drop_by_five": {"min": 0.342, "median": 0.452, "max": 0.546},
    "mulliganed": {"min": 0.163, "median": 0.213, "max": 0.261},
}

#: THE PRD'S THRESHOLD IS INERT AND THAT IS WHY IT IS NOT HERE. It asks for
#: `P(stall by turn 4) > 0.15 -> red`. Across the whole fleet the highest reading
#: is **0.079**, so it fires on ZERO of 13 decks — a red line that can never go
#: red, which is as useless as one that always does and would have shipped
#: looking rigorous. This repo has rejected three checks for the opposite failure
#: (one fired on 27% of correct data); this is the same mistake pointing the
#: other way, and the only way to see either is to run the fleet first.
PRD_STALL_THRESHOLD_REJECTED = 0.15

#: THREE OF THE MANA READINGS ARE ONE MEASUREMENT. Measured across the fleet:
#: missed-drop-by-five vs the all-turn drop rate r = +0.994, vs mulligan rate
#: r = +0.968, and the latter two r = +0.958. They are all driven by land count.
#: `benchmark.py` already recorded the two-way version of this (r = 0.97) and
#: refused to sum them; the third member is new. They are reported together with
#: the correlation stated, so three confirmations of one fact cannot read as
#: three findings — and nothing here adds them.
MANA_READINGS_ARE_CORRELATED = (
    "missed-drop-by-five, the all-turn drop rate and the mulligan rate move "
    "together across the fleet (r = +0.96 to +0.99): they are one measurement "
    "seen three ways, all driven by land count. Read them as one signal."
)

#: Stall is the one reading INDEPENDENT of that family — r = -0.26 to -0.41
#: against all three — so it is a genuine second dimension rather than the same
#: fact restated, which is what `consistency` turned out to be in the benchmark.


def _place(value, key):
    """Where a reading sits against the fleet — context, never a grade."""
    band = FLEET.get(key)
    if band is None or value is None:
        return None
    if value < band["min"]:
        return "better than any tracked deck"
    if value > band["max"]:
        return "worse than any tracked deck"
    return ("above the fleet median" if value > band["median"]
            else "below the fleet median")


def _rate(k, n):
    """A rate with its 95% interval, or an absent reading rather than a zero."""
    if not n:
        return None
    lo, hi = st.wilson_bounds(k, n)
    return {"rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n}


def stall(rows):
    """P(no legal play), per turn and as a run of two.

    A STALL IS A TURN WITH NOTHING CASTABLE, not a turn with nothing cast. The
    goldfish is a resource model — it never casts a wipe or a counterspell — so
    "nothing was cast" measures what the model declines to represent. Scored that
    way ur-dragon reads 6.4 dead turns in ten while its hand grows to eleven
    cards. `goldfish` records castability instead, which needs only mana value
    and available mana and is therefore true of cards the model would never pick
    up.
    """
    if not rows or "stall_by_turn" not in rows[0]:
        return None
    n = len(rows)
    turns = len(rows[0]["stall_by_turn"])
    by_turn = {}
    for t in range(turns):
        by_turn[str(t + 1)] = _rate(sum(r["stall_by_turn"][t] for r in rows), n)
    i0 = STALL_FROM_TURN - 1
    consecutive = sum(
        1 for r in rows
        if any(r["stall_by_turn"][i] and r["stall_by_turn"][i + 1]
               for i in range(i0, turns - 1)))
    # WHICH PROBLEM IT IS. A stall holding cards is a mana problem; a stall with
    # an empty hand is a draw problem, and they want opposite fixes. Measured on
    # ur-dragon: 906 stall turns, none of them an empty hand.
    stalled = [(r, i) for r in rows for i in range(i0, turns) if r["stall_by_turn"][i]]
    empty = sum(1 for r, i in stalled if r["hand_size_by_turn"][i] == 0)
    two = _rate(consecutive, n)
    return {
        "by_turn": by_turn,
        "two_in_a_row": two,
        "fleet": _place(two["rate"] if two else None, "stall_two_in_a_row"),
        "from_turn": STALL_FROM_TURN,
        "cause": {"stall_turns": len(stalled), "hand_empty": empty,
                  "mana_short": len(stalled) - empty},
        "basis": "a turn on which no card in hand was castable with the mana "
                 "that turn produced — not a turn on which nothing was cast",
    }


def declaration_fits(targets, names):
    """Does this declaration describe THE LIST BEING MEASURED?

    A branch inherits the deck's authored `goldfish_targets.json`, which is right
    for a swap and wrong for a rebuild. Measured on ur-dragon's treasure branch:
    every one of the ten targets names cards the branch does not run, because the
    branch cut the whole Dragon-typal package the declaration was written about.

    A number computed from it would be a real measurement OF A DIFFERENT DECK,
    and it would look completely ordinary. So the mismatch is detected and the
    engine reading is withheld, which is the same contract as an absent
    declaration: absent, never zero, and always with the reason.
    """
    held = set(names)
    missing = {}
    for t in targets:
        if not (t.get("required") or t.get("route")):
            continue
        gone = sorted({c for g in (t.get("need") or [])
                       for c in (g.get("any_of") or []) if c not in held})
        if gone:
            missing[t["label"]] = gone
    return missing


def engine(rows, targets, missing=None):
    if missing:
        worst = max(missing.items(), key=lambda kv: len(kv[1]))
        return {"available": False,
                "why": (f"the declaration does not describe this list — "
                        f"{len(missing)} load-bearing target(s) name cards it does "
                        f"not run (e.g. '{worst[0][:48]}' wants "
                        f"{', '.join(worst[1][:3])}). A branch inherits the deck's "
                        f"declaration, which is right for a swap and wrong for a "
                        f"rebuild: write the branch its own "
                        f"`goldfish_targets.json`."),
                "declaration_mismatch": {k: v for k, v in list(missing.items())[:6]},
                "declared_targets": len(targets)}
    return _engine(rows, targets)


def _engine(rows, targets):
    """Is the engine online, and by when — measured, never multiplied.

    THE REFUSAL THIS RESOLVES is already on the record: "a deck with four kills
    has no single assembled_rate, and averaging them would be inventing a number
    the simulation never measured." The fix is not a cleverer average. It is for
    the declaration to say which components the deck cannot do without
    (`required: true`) and which are alternative routes to the same end
    (`route: "a"`). Then:

        engine_online = P(every REQUIRED target assembled by turn N)
        any_route     = P(at least one ROUTE assembled by turn N)

    BOTH ARE COUNTED OVER PER-ITERATION ROWS, not composed from marginals.
    Components share cards, so P(A and B) is not P(A)P(B) and a product would be
    wrong in a direction nobody could see.

    ABSENT DECLARATION MEANS AN ABSENT FIGURE, never a zero. A deck whose targets
    say nothing about required-ness gets no engine reading and a note saying so —
    the same contract `model_treasures` keeps, because "0.0" is a measurement
    nobody made.
    """
    req = [i for i, t in enumerate(targets) if t.get("required")]
    routes = [i for i, t in enumerate(targets) if t.get("route")]
    if not req:
        return {"available": False,
                "why": "no target in goldfish_targets.json is marked "
                       "`required: true`, so there is nothing to call the "
                       "engine. Mark the components this deck cannot do "
                       "without — the alternatives get `route`.",
                "declared_targets": len(targets)}
    n = len(rows)
    turns = len(rows[0]["stall_by_turn"]) if rows and "stall_by_turn" in rows[0] else 10

    def met(row, i, turn):
        got = row["target_turns"][i]
        return got is not None and got <= turn

    online, any_route = {}, {}
    for turn in range(1, turns + 1):
        online[str(turn)] = _rate(
            sum(1 for r in rows if all(met(r, i, turn) for i in req)), n)
        if routes:
            any_route[str(turn)] = _rate(
                sum(1 for r in rows if any(met(r, i, turn) for i in routes)), n)
    # The component that fails first is the one to fix, and it is the one a
    # marginal rate can name while a joint cannot.
    bottleneck = None
    if req:
        worst = min(req, key=lambda i: sum(
            1 for r in rows if met(r, i, 3)))
        bottleneck = {"label": targets[worst]["label"],
                      "by_turn_three": _rate(
                          sum(1 for r in rows if met(r, worst, 3)), n)}
    return {
        "available": True,
        "required": [targets[i]["label"] for i in req],
        "routes": [targets[i]["label"] for i in routes],
        "online_by_turn": online,
        "any_route_by_turn": any_route or None,
        "bottleneck": bottleneck,
        "basis": "counted over per-iteration assembly, so the correlations "
                 "between components that share cards are preserved",
    }


def mana(rows):
    """Land drops and available mana — the substrate everything else needs."""
    if not rows:
        return None
    n = len(rows)
    turns = len(rows[0]["mana_by_turn"])
    # THE WINDOW MATTERS AND THE OBVIOUS PHRASING IS USELESS. "Missed any land
    # drop" over ten turns is true in 95.8% of ur-dragon's games and tells you
    # nothing — every deck runs out of lands eventually. What a pilot needs is
    # whether the EARLY drops land, because that is the window a curve is built
    # against.
    early = 5
    return {
        "missed_land_drop_by_five": _rate(
            sum(1 for r in rows if not all(r["land_hits"][:early])), n),
        "missed_land_drop_rate": round(statistics.mean(
            1.0 - (sum(r["land_hits"]) / len(r["land_hits"])) for r in rows), 4),
        "mulliganed": _rate(sum(1 for r in rows if r["mulligans"]), n),
        "mean_mana_by_turn": {
            str(t + 1): round(statistics.mean(r["mana_by_turn"][t] for r in rows), 3)
            for t in range(turns)},
        "fleet": _place(
            (_rate(sum(1 for r in rows if not all(r["land_hits"][:early])), n) or {})
            .get("rate"), "missed_land_drop_by_five"),
        "correlated": MANA_READINGS_ARE_CORRELATED,
        "mana_stdev_turn_five": (
            round(statistics.pstdev([r["mana_by_turn"][4] for r in rows]), 4)
            if turns >= 5 else None),
    }


def run(slug, branch=None, iterations=None, seed=None, quiet=False):
    from manamap.pilot.common import load_deck_cards
    return run_on(load_deck_cards(slug, branch), slug, branch=branch,
                  iterations=iterations, seed=seed, quiet=quiet)


def run_on(doc, slug, branch=None, iterations=None, seed=None, quiet=False,
           targets=None):
    """The same reading, taken on a list that need not be on disk.

    This is what lets a candidate be judged by SUBSTITUTION — put the card in,
    measure, take it out — rather than by a score over its properties. The model
    is identical; only the source of the list changes.
    """
    from manamap.pilot import goldfish
    # `targets` overrides the authored declaration FOR THIS RUN ONLY — the
    # hypothetical a candidate sweep asks: if this card counted toward that
    # component, how far would the engine move? Never written back.
    if targets is None:
        targets_doc = load_json(deck_file(slug, "goldfish_targets.json", branch)) or {}
        targets = targets_doc.get("targets") or []
    got = goldfish.run(slug, branch=branch, with_results=True, doc=doc, quiet=quiet,
                       targets_override=targets,
                       iterations=iterations or HARNESS["iterations"],
                       seed=seed if seed is not None else HARNESS["seed"],
                       max_turn=HARNESS["max_turn"])
    rows = got.get("_results") or []
    names = {c["name"] for c in doc.get("cards", [])}
    missing = declaration_fits(targets, names)
    return {
        "slug": slug,
        "branch": branch,
        "harness": dict(HARNESS, iterations=iterations or HARNESS["iterations"],
                        seed=seed if seed is not None else HARNESS["seed"]),
        "decklist_sha256": (got.get("meta") or {}).get("decklist_sha256"),
        "stall": stall(rows),
        "engine": engine(rows, targets, missing),
        "mana": mana(rows),
        "limits": [
            "No pod, no opponent and no interaction: this measures a DECK, not "
            "a table. It is not a win rate.",
            "Strategy-relative by construction — every figure is read against "
            "this deck's own declaration and never ranked against another deck.",
            "A stall is a turn with nothing CASTABLE. The model does not cast "
            "wipes, counterspells or targeted removal, so 'nothing was cast' "
            "would measure the model rather than the deck.",
        ],
    }


# ── Comparison ───────────────────────────────────────────────────────────

def compare(a, b):
    """Two readings, and the difference with an interval on the DIFFERENCE.

    UNPAIRED, and that is not a shortcut. A changed decklist changes every
    shuffle, so a shared seed buys replayability and never pairing — the fact
    `experiment`'s assumptions block already states. Newcombe for rates.

    `intervals_overlap` is deliberately not reported. Non-overlap implies a
    difference; overlap implies nothing at all, because two marginal intervals
    can overlap while the interval on their difference excludes zero.
    """
    out = {}
    for path, label in (("stall.two_in_a_row", "stall (2 in a row)"),
                        ("mana.missed_land_drop_by_five", "missed a land drop by T5"),
                        ("mana.mulliganed", "mulliganed")):
        ra, rb = _dig(a, path), _dig(b, path)
        if not (ra and rb):
            continue
        out[path] = _diff_rate(label, ra, rb)
    ea, eb = (a.get("engine") or {}), (b.get("engine") or {})
    # THE TWO ENGINE FIGURES MAY NOT BE ABOUT THE SAME THING, and the delta
    # between them looks identical either way. Each is measured against ITS OWN
    # declaration — that is the whole strategy-relative design — so when the
    # declarations differ, "+0.36 engine online" means "the branch assembles
    # what IT claims more often than the champion assembles what IT claims".
    # That is a real and useful comparison. It is not "the same measure, 36
    # points better", and nothing on the line would tell you which.
    same_claim = (ea.get("required") or []) == (eb.get("required") or [])
    if not same_claim and ea.get("available") and eb.get("available"):
        out["_engine_declarations_differ"] = {
            "a": ea.get("required"), "b": eb.get("required"),
            "note": "each engine figure is measured against its own declaration, "
                    "so the delta compares each list against ITS OWN intent — "
                    "not the same quantity twice."}
    if ea.get("available") and eb.get("available"):
        for turn in ("3", "5", "8"):
            ra = (ea.get("online_by_turn") or {}).get(turn)
            rb = (eb.get("online_by_turn") or {}).get(turn)
            if ra and rb:
                out[f"engine_online_turn_{turn}"] = _diff_rate(
                    f"engine online by turn {turn}", ra, rb)
    return out


def _dig(doc, path):
    cur = doc
    for part in path.split("."):
        cur = (cur or {}).get(part)
    return cur


#: Above this many games per arm the EXACT power calculation is not usable and
#: does not need to be. `stats.mde_proportion` walks a binomial grid, which is
#: the right method for `experiment` — twenty games an arm, where the normal
#: approximation is poor and the boundary matters — and it overflows a float at
#: 4,000 (`math.comb(4000, k)` exceeds the double range). At these sample sizes
#: the normal approximation is not a compromise but the regime it is valid in.
EXACT_MDE_MAX_N = 400


def _mde(p_a, n_a, n_b):
    """Smallest difference this many games could reliably detect (power 0.8)."""
    if n_a <= EXACT_MDE_MAX_N and n_b <= EXACT_MDE_MAX_N:
        got = st.mde_proportion(p_a, n_a, n_b)
        return got.get("minimum_detectable_difference") if got else None
    # z(0.975) + z(0.80) = 1.9600 + 0.8416
    import math
    return round(2.8016 * math.sqrt(p_a * (1 - p_a) * (1 / n_a + 1 / n_b)), 4)


def _diff_rate(label, ra, rb):
    """One measure, both readings, and the interval on the DIFFERENCE.

    `stats.diff_proportions` already returns diff / ci95 / excludes_zero / method
    — it is Newcombe and it names itself, so this carries its answer through
    rather than recomputing a worse one.
    """
    ka, na = round(ra["rate"] * ra["n"]), ra["n"]
    kb, nb = round(rb["rate"] * rb["n"]), rb["n"]
    d = st.diff_proportions(ka, na, kb, nb)
    # WHAT A DIFFERENCE WOULD HAVE TO BE TO BE VISIBLE AT THIS N. Below it,
    # "no change" and "not enough games" are the same reading, and a rank
    # ordering built on differences under the MDE is a ranking of noise.
    mde = _mde(ra["rate"], na, nb)
    return {"label": label, "a": ra["rate"], "b": rb["rate"],
            "delta": d["diff"], "ci95_diff": d["ci95"],
            "excludes_zero": d["excludes_zero"], "method": d["method"],
            "mde": mde}


# ── CLI ──────────────────────────────────────────────────────────────────

def _fmt(r):
    if not r:
        return "     —"
    return f"{r['rate']:>6.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def _print(doc):
    where = f"{doc['slug']}" + (f"/{doc['branch']}" if doc.get("branch") else "")
    h = doc["harness"]
    print(f"DIAGNOSTIC — {where}   ({h['iterations']} games, seed {h['seed']})")
    e = doc.get("engine") or {}
    print("\n  ENGINE")
    if not e.get("available"):
        print(f"    not measured — {e.get('why', 'no declaration')}")
    else:
        for turn in ("3", "5", "8"):
            r = (e.get("online_by_turn") or {}).get(turn)
            print(f"    online by turn {turn}   {_fmt(r)}")
        if e.get("any_route_by_turn"):
            print(f"    any kill route T8   {_fmt(e['any_route_by_turn'].get('8'))}")
        b = e.get("bottleneck")
        if b:
            print(f"    bottleneck: {b['label'][:58]}")
            print(f"                by turn 3   {_fmt(b['by_turn_three'])}")
    s = doc.get("stall") or {}
    if s:
        print("\n  STALL  (a turn with nothing castable)")
        print(f"    two in a row (from T{s['from_turn']})   {_fmt(s['two_in_a_row'])}")
        c = s["cause"]
        print(f"    cause: {c['mana_short']} mana-short, {c['hand_empty']} hand-empty")
        if s.get("fleet"):
            print(f"    ({s['fleet']})")
    m = doc.get("mana") or {}
    if m:
        print("\n  MANA")
        print(f"    missed a drop by T5  {_fmt(m['missed_land_drop_by_five'])}")
        print(f"    missed-drop rate     {m['missed_land_drop_rate']:>6.3f}  (all turns)")
        print(f"    mulliganed           {_fmt(m['mulliganed'])}")
        if m.get("fleet"):
            print(f"    ({m['fleet']})")
        print(f"    ! {MANA_READINGS_ARE_CORRELATED}")


def _print_compare(a, b, deltas):
    an = a["slug"] + (f"/{a['branch']}" if a.get("branch") else " (champion)")
    bn = b["slug"] + (f"/{b['branch']}" if b.get("branch") else " (champion)")
    print(f"\nCOMPARISON — {an}  vs  {bn}\n")
    print(f"  {'measure':30} {'A':>7} {'B':>7} {'delta':>8}  {'ci95 on the difference':>24}")
    for key, d in deltas.items():
        if key.startswith("_"):
            continue
        mark = "  *" if d["excludes_zero"] else "   "
        print(f"  {d['label'][:30]:30} {d['a']:>7.3f} {d['b']:>7.3f} "
              f"{d['delta']:>+8.3f}  [{d['ci95_diff'][0]:>+7.3f}, {d['ci95_diff'][1]:>+7.3f}]{mark}")
    print("\n  * the interval on the DIFFERENCE excludes zero.")
    warn = deltas.get("_engine_declarations_differ")
    if warn:
        print("\n  ! THE TWO ENGINE FIGURES ANSWER DIFFERENT QUESTIONS.")
        print("    A declares: " + "; ".join(x[:44] for x in (warn["a"] or [])))
        print("    B declares: " + "; ".join(x[:44] for x in (warn["b"] or [])))
        print("    Each is measured against its own declaration, so the delta says")
        print("    each list meets ITS OWN intent more or less often — not that one")
        print("    is better at the same thing.")
    any_mde = next((d["mde"] for d in deltas.values()), None)
    if any_mde:
        print(f"  Smallest difference this many games could detect: ~{any_mde:.3f}.")
        print("  Below that, 'no change' and 'not enough games' are the same reading.")


def main(args):
    branch = getattr(args, "branch", None)
    doc = run(args.slug, branch=branch,
              iterations=getattr(args, "iterations", None),
              seed=getattr(args, "seed", None))
    other = getattr(args, "vs", None)
    if other:
        vs_branch = None if other == "main" else other
        alt = run(args.slug, branch=vs_branch,
                  iterations=getattr(args, "iterations", None),
                  seed=getattr(args, "seed", None))
        deltas = compare(alt, doc)
        if getattr(args, "json", False):
            print(json.dumps({"a": alt, "b": doc, "deltas": deltas}, indent=1)); return
        _print(doc); _print_compare(alt, doc, deltas); return
    if getattr(args, "json", False):
        print(json.dumps(doc, indent=1)); return
    _print(doc)
    if getattr(args, "write", False):
        out = deck_dir(args.slug, branch) / ARTIFACT
        out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"\n  Wrote {out}")
