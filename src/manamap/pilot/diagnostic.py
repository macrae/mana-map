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
import math
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
#: MEASURED across 13 decks at a uniform harness with both models on:
#: board power@6 vs damage@8 r=+0.97, vs kill@10 r=+0.98; damage@8 vs kill@8
#: r=+0.92. One dimension, three views. The hoard is NOT part of it (r=0.08 to
#: 0.25 against all three), which is what makes it a real second axis rather
#: than `consistency`'s mistake again.
COMBAT_READINGS_ARE_CORRELATED = (
    "board power, damage and kill rate move together across the fleet "
    "(r = +0.86 to +0.98): they are one dimension seen three ways. Read them "
    "as one signal, and rank on only one of them — `damage_8` is the axis, "
    "because a kill rate floors out on decks that rarely kill.")

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


def _mean_cell(xs):
    """A MEAN with its interval and its spread. The spread rides along because
    an MDE for a mean needs it — `p(1-p)` is a proportion's variance."""
    if not xs:
        return None
    mean, sd = st._mean_sd(xs)
    half = (st.t_crit(len(xs) - 1) * sd / (len(xs) ** 0.5)
            if len(xs) > 1 and sd else 0.0)
    return {"rate": round(mean, 4),
            "ci95": [round(mean - half, 4), round(mean + half, 4)],
            "sd": round(sd, 4), "n": len(xs)}


def steam(rows, got):
    """DOES THE DECK KEEP GOING, AND CAN IT AFFORD TO ANSWER ANYTHING.

    The three questions edgar-vampires' captain's log kept asking that nothing
    in this module could answer: "I ran out of steam", "the interaction never
    got cast", "a two-land keep going fifth".

    `extra_cards` is ABSENT rather than zero when the deck has not opted into
    `model_draw` — the series is a flat zero on the unopted path and a reader
    cannot tell that from a deck with no draw in it, which is the one thing
    this block exists to distinguish.
    """
    if not rows:
        return None
    n = len(rows)
    turns = len(rows[0]["mana_by_turn"])
    opted = bool((got.get("metrics") or {}).get("mean_extra_cards_drawn_by_turn"))
    T = range(turns)

    in_hand = {str(t + 1): _rate(
        sum(1 for r in rows if r["interaction_in_hand_by_turn"][t]), n) for t in T}
    castable = {str(t + 1): _rate(
        sum(1 for r in rows if r["interaction_castable_by_turn"][t]), n) for t in T}
    # CONDITIONAL, AND IT IS THE INTERPRETABLE ONE. Cutting three interaction
    # spells lowers `castable` without anything about the mana changing, so the
    # raw series answers "how much interaction do you run" and this one answers
    # the question actually asked: WHEN you hold one, can you afford it? Its
    # denominator is the games where you held one, which is why it carries its
    # own interval rather than being a division a reader does by hand.
    conditional = {}
    for t in T:
        held = sum(1 for r in rows if r["interaction_in_hand_by_turn"][t])
        ok = sum(1 for r in rows if r["interaction_castable_by_turn"][t])
        conditional[str(t + 1)] = _rate(ok, held)
    return {
        "extra_cards_by_turn": (
            {str(t + 1): _mean_cell([r["drawn_extra_by_turn"][t] for r in rows])
             for t in T} if opted else None),
        "extra_cards_unavailable": None if opted else (
            "this deck does not opt into `model_draw` in goldfish_targets.json, "
            "so every draw beyond the one-a-turn draw step is unmodelled. That "
            "is an absent measurement, not a zero."),
        "interaction_in_hand_by_turn": in_hand,
        "interaction_castable_by_turn": castable,
        "castable_given_in_hand_by_turn": conditional,
        "keep_can_act_by_t3": _rate(
            sum(1 for r in rows if r["keep_can_act_by_t3"]), n),
        "basis": (
            "`castable` is the mana left at the END OF THE MAIN PHASE — the "
            "moment the decision is made, before combat. Extra combat phases "
            "are paid for after it and attack triggers add mana after it, so on "
            "a deck with Aggravated Assault this float is larger than what "
            "survives the turn. It is also a FLOOR against the spending policy: "
            "the model casts everything it can afford every turn, so a pilot "
            "choosing to hold up two mana would score higher. A low figure is a "
            "real finding; a high one is unambiguous good news."),
    }


def output(got):
    """The MAGNITUDE series, carried through from the goldfish that owns them.

    WHY THIS BLOCK HAD TO EXIST. Every other axis here is a MEMBERSHIP axis: a
    goldfish target asks whether a card was DRAWN, so the ninth member of a
    component raises assembly by the same amount whichever card it is. Measured
    on ur-dragon's treasure branch, all eight declared multipliers returned the
    identical +0.039 — a true answer to "what is one more member worth" and no
    answer at all to "which member", which is the question a pilot actually
    asks. Magnitude is the axis that can separate them, and the goldfish has
    been emitting it all along with nothing exposing it.

    ABSENT ⇒ ABSENT, NEVER ZERO. A deck without `model_treasures` has no hoard
    and one without `model_combat` has no clock; the key is missing rather than
    0.0, which is a measurement nobody made. Same contract as the flags.
    """
    m = got.get("metrics") or {}
    rows = got.get("_results") or []
    out = {}
    # EVERY FIGURE HERE CARRIES ITS INTERVAL, in the same {rate, ci95, n} shape
    # every other block uses — so `candidates._read`, `compare` and the report
    # need no special case for a mean, and a magnitude axis cannot quietly
    # become the one number in this document published bare.
    #
    # Derived from the per-iteration rows rather than restated from `metrics`,
    # because a mean with no dispersion beside it cannot produce one. A test
    # asserts these means agree with the goldfish's own to 3dp, so the goldfish
    # stays the owner of the figure in fact as well as in principle.
    def series(field, present):
        if not present or not rows:
            return None
        turns = sorted({t for r in rows for t in range(len(r.get(field) or []))})
        got_series = {}
        for t in turns:
            xs = [(r.get(field) or [0] * (t + 1))[t] for r in rows
                  if len(r.get(field) or []) > t]
            if not xs:
                continue
            mean, sd = st._mean_sd(xs)
            half = (st.t_crit(len(xs) - 1) * sd / (len(xs) ** 0.5)
                    if len(xs) > 1 and sd else 0.0)
            got_series[str(t + 1)] = {
                "rate": round(mean, 4),
                "ci95": [round(mean - half, 4), round(mean + half, 4)],
                # The spread rides along because an MDE for a MEAN needs it —
                # p(1-p) is a proportion's variance and a hoard of 6.5 has none.
                "sd": round(sd, 4),
                "n": len(xs)}
        return got_series or None

    tre = m.get("treasure") or {}
    hoard = series("treasures_by_turn", tre.get("mean_treasures_in_hoard_by_turn"))
    if hoard:
        out["hoard_by_turn"] = hoard
    com = m.get("combat") or {}
    for field, name, present in (
            ("damage_by_turn", "damage_by_turn", com.get("mean_damage_by_turn")),
            ("board_power_by_turn", "board_power_by_turn",
             com.get("mean_board_power_by_turn"))):
        got_series = series(field, present)
        if got_series:
            out[name] = got_series
    # A kill is a RATE, not a mean — it is the share of games closed by turn N,
    # so it takes a Wilson bound like every other proportion here.
    if com.get("kill_by_turn_rate") and rows:
        kills = {}
        for t in range(1, (HARNESS["max_turn"]) + 1):
            k = sum(1 for r in rows
                    if r.get("kill_turn") is not None and r["kill_turn"] <= t)
            kills[str(t)] = _rate(k, len(rows))
        out["kill_by_turn"] = kills
    if not out:
        return {"available": False,
                "why": "no magnitude series — this deck opts into neither "
                       "`model_treasures` nor `model_combat` in "
                       "goldfish_targets.json, so there is no hoard and no "
                       "clock to read. That is an absent measurement, not a zero."}
    out["available"] = True
    out["basis"] = ("means over the same iterations as every other figure here; "
                    "a mean over a skewed sample describes no single game, so "
                    "read a difference rather than a level")
    return out


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
        "steam": steam(rows, got),
        "output": output(got),
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

#: How many placebo removals an ablation draws. Four is enough to bracket the
#: band on the fleet's spreads and cheap enough that nobody skips the control.
PLACEBO_DRAWS = 4


def ablate(doc, slug, names, axis_block, axis_key, axis_turn, branch=None,
           iterations=None, seed=None, targets=None, placebos=PLACEBO_DRAWS):
    """Remove a set of cards, measure, AND MEASURE A PLACEBO OF THE SAME SIZE.

    WHY THE PLACEBO IS NOT OPTIONAL. Taking N cards out shrinks the library, so
    every card left is drawn more often — the deck gets FASTER by construction.
    Measured on ur-dragon's treasure branch, removing any 8 non-declared cards
    raised turn-8 damage by **+0.92 on average**, and removing the 8 declared
    multipliers raised it by **+1.26**: a gap of 0.34 against an MDE of 0.56.
    Read without the control that is "cutting your multipliers kills faster",
    which is a plausible, well-shaped, interval-backed and entirely wrong
    finding — it is deck size wearing a card effect's clothes.

    The same run on the HOARD axis is the opposite: the placebos all move UP
    (+0.67 to +1.06) and the multipliers move DOWN (-0.77), so that effect is
    real and the direction alone says so.

    A verdict is therefore relative to the placebo BAND, never to the baseline.
    `candidates` is unaffected — it substitutes one-for-one and holds the size
    fixed, which is exactly why it never needed this.
    """
    import copy
    import random

    def read(d):
        cell = ((d.get(axis_block) or {}).get(axis_key) or {})
        if axis_turn:
            cell = cell.get(axis_turn) or {}
        return cell

    def run_without(drop):
        d2 = copy.deepcopy(doc)
        d2["cards"] = [c for c in d2["cards"] if c["name"] not in drop]
        return read(run_on(d2, slug, branch=branch, iterations=iterations,
                           seed=seed, quiet=True, targets=targets))

    base = read(run_on(doc, slug, branch=branch, iterations=iterations,
                       seed=seed, quiet=True, targets=targets))
    if not base:
        return {"available": False, "why": "the baseline has no reading on that axis"}
    got = run_without(set(names))
    delta = round(got["rate"] - base["rate"], 4)

    declared = set(names)
    pool = [c["name"] for c in doc["cards"]
            if not c.get("is_commander")
            and "Land" not in (c.get("type_line") or "")
            and c["name"] not in declared and (c.get("quantity") or 1) == 1]
    band = []
    if len(pool) >= len(names):
        rng = random.Random(seed if seed is not None else HARNESS["seed"])
        for _ in range(placebos):
            r = run_without(set(rng.sample(pool, len(names))))
            band.append(round(r["rate"] - base["rate"], 4))
    if not band:
        return {"available": True, "baseline": base["rate"], "delta": delta,
                "placebo": None,
                "verdict": "no placebo was possible — too few comparable cards "
                           "to remove, so this delta cannot be separated from "
                           "the effect of a smaller library"}
    lo, hi = min(band), max(band)
    inside = lo <= delta <= hi
    return {
        "available": True, "baseline": base["rate"], "delta": delta,
        "n_removed": len(names),
        "placebo": {"draws": len(band), "band": [lo, hi],
                    "mean": round(sum(band) / len(band), 4)},
        "mde": mde(base),
        "verdict": ("inside the placebo band — this is the LIBRARY GETTING "
                    "SMALLER, not these cards" if inside else
                    "outside the placebo band — a real effect of these cards"),
        "real": not inside,
    }


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


def mde(cell, n_a=None, n_b=None):
    """The smallest difference this design could see — for a rate OR a mean.

    A PROPORTION and a MEAN do not share a formula, and using the proportion's
    on a mean is not a bad approximation but a domain error: `p(1-p)` under a
    hoard of 6.5 is negative. Whichever a cell is, it carries what its own MDE
    needs — `sd` on a mean, nothing extra on a rate.
    """
    n_a = n_a or cell.get("n")
    n_b = n_b or n_a
    if cell.get("sd") is not None:
        # z(0.975) + z(0.80), the same constant the proportion branch uses.
        return round(2.8016 * cell["sd"] * math.sqrt(1 / n_a + 1 / n_b), 4)
    return _mde(cell["rate"], n_a, n_b)


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


# ── Reading the numbers ──────────────────────────────────────────────────
#
# AN INTERPRETATION IS WHERE A TOOL STARTS INVENTING THINGS, so every line here
# is derived from a comparison by a stated rule and carries the measure it came
# from. Nothing weighs one axis against another and nothing calls a trade good:
# the trade is shown, and ruling on it is the pilot's.
#
# The distinction that does the most work is between "did not change" and
# "could not be seen". They look identical on the page and they are opposite
# findings — one is evidence of no effect, the other is evidence of nothing.

#: One in N. A rate is hard to feel and a frequency is not: 0.593 is "three games
#: in five", which is the sentence a pilot can hold at the table.
def as_frequency(rate):
    if rate is None:
        return None
    if rate <= 0.0:
        return "never"
    if rate >= 0.995:
        return "every game"
    # A RELATIVE TOLERANCE, because an absolute one lies at small rates. At 0.035
    # absolute, both 0.039 and 0.053 round to "1 game in 20" — and the interval
    # on that difference EXCLUDES ZERO, so the phrasing erased a real cost.
    for denom in (2, 3, 4, 5, 6, 8, 10, 20, 50, 100):
        num = round(rate * denom)
        if num >= 1 and abs(num / denom - rate) <= max(0.008, rate * 0.10):
            g = math.gcd(num, denom)          # "2 games in 50" is "1 game in 25"
            num, denom = num // g, denom // g
            return f"{num} game{'s' if num != 1 else ''} in {denom}"
    return f"{rate:.0%} of games"


def _pair(a, b):
    """Both endpoints, phrased so they cannot collapse into each other.

    A frequency is easier to hold than a rate, and it is a PRESENTATION AID: the
    moment it would print the same words for two numbers a confidence interval
    says are different, it has to give way to the numbers.
    """
    fa, fb = as_frequency(a), as_frequency(b)
    if fa == fb:
        return f"{a:.1%} -> {b:.1%}"
    return f"{fa} -> {fb}"


GAIN, COST, FLAT, UNSEEN, LIMIT, CAVEAT = (
    "gain", "cost", "flat", "unseen", "limit", "caveat")

#: Which direction is an improvement, per measure. Without this a lower stall
#: reads as a loss.
LOWER_IS_BETTER = ("stall", "missed", "mulligan")


def _better_when_lower(label):
    return any(w in label.lower() for w in LOWER_IS_BETTER)


def interpret(a, b, deltas):
    """Read a comparison: what moved, what it cost, what still limits it."""
    out = []
    for key, d in deltas.items():
        if key.startswith("_"):
            continue
        improved = (d["delta"] < 0) if _better_when_lower(d["label"]) else (d["delta"] > 0)
        if d["excludes_zero"]:
            out.append({
                "kind": GAIN if improved else COST,
                "measure": d["label"],
                "says": f"{d['label']}: {_pair(d['a'], d['b'])}",
                "detail": (f"{d['delta']:+.3f}, and the interval on the difference "
                           f"{d['ci95_diff']} excludes zero"),
            })
            continue
        # NOT THE SAME FINDING, and they look identical on the page.
        mde = d.get("mde")
        if mde is not None and abs(d["delta"]) < mde:
            out.append({
                "kind": UNSEEN, "measure": d["label"],
                "says": f"{d['label']}: no reading either way",
                "detail": (f"the difference is {d['delta']:+.3f} and this many "
                           f"games can only see {mde:.3f}. That is evidence of "
                           f"NOTHING, not evidence of no change — run more games "
                           f"if it matters."),
            })
        else:
            out.append({
                "kind": FLAT, "measure": d["label"],
                "says": f"{d['label']}: unchanged",
                "detail": (f"the difference is {d['delta']:+.3f} and the interval "
                           f"{d['ci95_diff']} spans zero, which this many games "
                           f"COULD have resolved — so it is flat, not unmeasured."),
            })
    e = (b.get("engine") or {})
    if e.get("available") and e.get("bottleneck"):
        bn = e["bottleneck"]
        r = (bn.get("by_turn_three") or {}).get("rate")
        out.append({
            "kind": LIMIT, "measure": "engine bottleneck",
            "says": f"what limits it now: {bn['label']}",
            "detail": (f"assembled {as_frequency(r)} by turn three. Widening this "
                       f"component is the change the engine figure is most "
                       f"sensitive to — `candidates --as` prices it."),
        })
    if deltas.get("_engine_declarations_differ"):
        out.append({
            "kind": CAVEAT, "measure": "engine",
            "says": "the two engine figures answer different questions",
            "detail": ("each is measured against its own declaration, so the delta "
                       "says each list meets ITS OWN intent more or less often — "
                       "not that one is better at the same thing."),
        })
    out.append({
        "kind": CAVEAT, "measure": "the model",
        "says": "no pod, no opponent, no interaction",
        "detail": ("this measures a DECK, not a table: nothing blocks, nothing "
                   "removes and nobody is racing you. It cannot tell you whether "
                   "the deck wins, only whether it does what it says."),
    })
    return out


ICON = {GAIN: "+", COST: "-", FLAT: "=", UNSEEN: "?", LIMIT: ">", CAVEAT: "!"}


def _print_reading(reading):
    print("\n  THE READING")
    for r in reading:
        print(f"    {ICON[r['kind']]} {r['says']}")
        print(f"        {r['detail']}")


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
    o = doc.get("output") or {}
    if o:
        print("\n  OUTPUT  (magnitude — what the deck produced, not what it drew)")
        if not o.get("available"):
            print(f"    not measured — {o.get('why', '')[:88]}")
        else:
            for key, label in (("hoard_by_turn", "treasures in hoard"),
                               ("board_power_by_turn", "board power"),
                               ("damage_by_turn", "damage dealt"),
                               ("kill_by_turn", "killed by")):
                series = o.get(key)
                if not series:
                    continue
                cells = "   ".join(f"T{t} {series[t]['rate']:>6.2f}"
                                   for t in ("6", "8", "10") if t in series)
                print(f"    {label:20} {cells}")
            if any(k in o for k in ("board_power_by_turn", "kill_by_turn")):
                print(f"    ! {COMBAT_READINGS_ARE_CORRELATED}")
            # THE ONLY AXIS THAT CAN RANK WITHIN A COMPONENT, and the reason it
            # exists is worth carrying next to the numbers.
            print("    ! a membership axis asks whether a card was DRAWN, so it "
                  "reads alike for every")
            print("      member of one component. These read what the deck "
                  "PRODUCED, which is what")
            print("      separates them. Means over a skewed sample: compare "
                  "differences, not levels.")
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
            print(json.dumps({"a": alt, "b": doc, "deltas": deltas,
                              "reading": interpret(alt, doc, deltas)}, indent=1)); return
        _print(doc); _print_compare(alt, doc, deltas)
        if not getattr(args, "no_read", False):
            _print_reading(interpret(alt, doc, deltas))
        return
    if getattr(args, "json", False):
        print(json.dumps(doc, indent=1)); return
    _print(doc)
    if getattr(args, "write", False):
        out = deck_dir(args.slug, branch) / ARTIFACT
        out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"\n  Wrote {out}")
