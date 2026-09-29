"""Did the AI actually PLAY this deck, or just hold the cards?

A Forge result is AI-vs-AI, and Forge says of its own AI that it "is not
trained" and is "poor to ok in control decks, pretty bad for most combo decks".
That caveat travels with every run record — but a caveat is a warning, not a
measurement, and it cannot tell you whether THIS run was piloted well enough for
its win rate to mean anything.

WHAT THIS MEASURES, AND WHY IT IS A RATIO RATHER THAN A THRESHOLD. Absolute
piloting quality would need a calibrated "good" and there is nothing to calibrate
against — no human plays inside Forge. But every run already contains its own
control: THE OTHER SEATS, played by the same AI, in the same games, under the
same engine. So the question becomes answerable without a constant:

    is our seat played about as well as the pod?

Measured on ur-dragon's treasure branch, 100 games: our seat 0.67 land drops per
own turn against a pod mean of 0.72, and 1.04 casts per turn against 1.11. Every
seat misses roughly a third of its land drops — the AI is UNIFORMLY weak, which
is a very different finding from being weak at our archetype, and it is the
difference between "this comparison is noise" and "this comparison is fair but
played badly by both sides".

WHAT IT LICENSES AND WHAT IT DOES NOT. A uniform weakness leaves an A/B between
two of YOUR OWN lists against the same pod substantially intact: both are played
equally badly. It does not rescue an absolute win rate, and it never makes a
Forge result a claim about how the deck plays in your hands.

THE VERDICT RESTS ON LAND DROPS ALONE, AND CASTS ARE CONTEXT. Both are reported,
but only one is a piloting measure. **Casts per turn is confounded by the deck's
own curve** — measured across every tracked run, `corr(mean mana value, casts
ratio) = -0.50`: an expensive deck casts fewer spells while being played
perfectly well. Scoring on it flagged radagast NOT COMPARABLE at 0.84 against a
0.85 line, on a deck whose only fault is a mean mana value of 2.97 and a run of
twenty games. That is a check firing on correct data, which this repo has
rejected three times before.

A land drop is not confounded that way: every deck wants its land every turn
whatever it costs, so a seat that is not making them is not being piloted.
"""

#: Below this share of the pod's own rate, our seat was handled worse than the
#: table it is being compared against — and a comparison drawn from it is
#: measuring the AI's preferences rather than the decks.
#:
#: 0.85 is deliberately generous and is NOT calibrated from a fleet, because
#: there is no fleet of runs to calibrate from; it is the point past which a
#: gap stops being sampling noise on ~900 turns and starts being a pattern.
#: It is a stated judgement, and the ratio it guards is reported either way so a
#: reader never has to take the verdict's word for it.
COMPARABLE = 0.85

#: Half the median WITHIN-deck run-to-run spread in the land ratio, measured over the 47
#: tracked runs that carry a verdict (median spread 0.037). A ratio this close to the line
#: cannot be told from one on the other side of it, so the verdict is withheld rather than
#: decided — see the long comment in `from_record`.
BAND = 0.019

LANDS, CASTS = "lands_per_turn", "casts_per_turn"

#: Below this many games the rates are one table's variance. The n=1 smoke run
#: reads 0.60 on land drops, which is a shuffle rather than a finding.
MIN_GAMES = 8


def from_record(rec):
    """Per-seat piloting rates, and whether our seat was handled like the rest.

    Reads the RECORD, never the logs: the logs are gitignored and only exist
    where the run was made, and this has to be readable from a checkout.
    """
    games = rec.get("games") or []
    if not games:
        return None
    seats = [s["slug"] for s in (rec.get("seats") or [])]
    if not seats:
        return None
    from manamap.sim.forge import deck_meta_name
    ours = seats[0]
    per = {}
    for slug in seats:
        key = deck_meta_name(slug)
        rows = [g["per_seat"][key] for g in games
                if key in (g.get("per_seat") or {})]
        # `round` is the game's round count, which is each seat's own turn count
        # in a Commander game — every seat takes one turn per round until it is
        # eliminated, so this understates a seat that died early. Reported, not
        # corrected: an eliminated seat's piloting is exactly what we want to see.
        turns = sum((g.get("round") or 0) for g in games
                    if key in (g.get("per_seat") or {}))
        if not rows or not turns:
            continue
        per[slug] = {
            LANDS: round(sum(r.get("lands") or 0 for r in rows) / turns, 3),
            CASTS: round(sum(r.get("casts") or 0 for r in rows) / turns, 3),
            "games": len(rows), "turns": turns,
        }
    if ours not in per or len(per) < 2:
        return None
    pod = [v for k, v in per.items() if k != ours]
    out = {"seat": ours, "per_seat": per, "comparable_at": COMPARABLE}
    for metric in (LANDS, CASTS):
        mean_pod = sum(p[metric] for p in pod) / len(pod)
        ratio = (per[ours][metric] / mean_pod) if mean_pod else None
        out[metric] = {"ours": per[ours][metric], "pod_mean": round(mean_pod, 3),
                       "ratio": round(ratio, 3) if ratio else None}
    # A SINGLE GAME CANNOT SUPPORT A VERDICT. The n=1 smoke run reads 0.60 on
    # land drops, which is one game's variance and not a finding.
    turns = per[ours]["turns"]
    if per[ours]["games"] < MIN_GAMES:
        out["comparable"] = None
        out["reading"] = (f"only {per[ours]['games']} game(s) — too few to say "
                          f"whether the AI played this seat like the rest. The "
                          f"rates are reported; the verdict is withheld.")
        return out
    ratio = out[LANDS]["ratio"]
    out["verdict_from"] = LANDS
    # A HARD CUT INSIDE THE RUN-TO-RUN NOISE PRODUCES A VERDICT THAT FLIPS ON THE SAME
    # DECK, and it did. Measured over the 47 tracked runs that carry a verdict (median
    # land ratio 0.946, sd 0.090), the WITHIN-deck spread across runs of one list at one
    # table has a median of 0.037 — and the 0.85 line falls inside that spread for two
    # decks:
    #
    #   edgar-vampires at standard-v3      0.826 flagged, 0.933 comparable
    #   goblin-storm@zada-v1 at standard-v3  0.840 flagged, 0.851 and 0.855 comparable
    #
    # goblin-storm@zada-v1's three runs span 0.015 and the threshold is inside it, so the
    # verdict there is not a fact about the deck. It is a coin flip reported as a finding.
    #
    # So a ratio within half the median spread of the line gets the verdict this module
    # already gives a run with too few games: WITHHELD, with the rates reported. `None`
    # is not a new state — `MIN_GAMES` has meant exactly this since the module shipped,
    # and extending it to a second cause is cheaper than a threshold nobody can defend.
    #
    # The 0.85 itself is left where it is. It was set by judgement with no fleet to
    # calibrate against, and the fleet now says the distribution is centred at 0.95 with
    # 6 of 47 runs below the line — a defensible place for it. What the fleet also says is
    # that a line cannot be read to three decimals.
    #
    # AND THE BAND IS A MAGNITUDE HEURISTIC ON PURPOSE, which closes a note the
    # `KNOWN_FLAGGED` list has carried since 2026-09-12: this gate "compares two per-turn
    # rates with NO INTERVAL ON THE DIFFERENCE, which this repo's own doctrine forbids
    # everywhere else… That is a defect in the gate, filed separately."
    #
    # Right diagnosis. Measured, the fix is not available at these sample sizes. The data
    # is there — `games[].per_seat[seat].lands` over `games[].round` gives a per-game
    # series for our seat and for the pod — so `stats.diff_means` yields a Welch interval
    # on the difference. Both readings of it were computed over all 47 runs:
    #
    #   interval EXCLUDES ZERO  ->  flags 16 of 47, including ur-dragon at ratio 0.931 and
    #                              zur-enchantress at 0.921, for shortfalls near 3%. It
    #                              answers "is there ANY difference", not "is it big
    #                              enough to matter" — significance read as magnitude.
    #   interval EXCLUDES THE   ->  flags 0 of 47. The half-width is ~0.03 and the 0.85
    #   LINE                       line as a difference is ~0.06, so nothing can be told
    #                              from the line with confidence — not even goblin-storm
    #                              at 0.776, whose [-0.127, -0.057] straddles -0.062.
    #
    # So a properly powered version of this gate CANNOT FIRE at 20-120 games, and the
    # honest reading is that n is the limit rather than the threshold. The band is the
    # cheap approximation: it withholds where the fleet's own run-to-run noise says the
    # line is unreadable, and leaves the magnitude judgement where a human can see it.
    if ratio and abs(ratio - COMPARABLE) < BAND:
        out["comparable"] = None
        out["reading"] = (
            f"land drops {ratio:.3f} against a {COMPARABLE} line — INSIDE the run-to-run "
            f"noise (the median within-deck spread across the fleet is {2 * BAND:.3f}), "
            f"so the verdict is withheld rather than decided on a coin flip. Two decks' "
            f"verdicts flip between runs at this threshold. The rates are reported; read "
            f"them, and prefer an A/B where both arms sit on the same side.")
        out["casts_note"] = (
            "reported, not scored: casts per turn is confounded by the deck's own "
            "curve (corr with mean mana value = -0.50 across tracked runs), so an "
            "expensive deck casts fewer spells while being piloted fine.")
        return out
    out["comparable"] = bool(ratio) and ratio >= COMPARABLE
    out["casts_note"] = (
        "reported, not scored: casts per turn is confounded by the deck's own "
        "curve (corr with mean mana value = -0.50 across tracked runs), so an "
        "expensive deck casts fewer spells while being piloted fine.")
    out["reading"] = (
        "our seat was handled about as well as the pod, so an A/B between two of "
        "your own lists against this table is played equally badly on both sides "
        "and remains informative. It is still not a claim about how the deck "
        "plays in your hands."
        if out["comparable"] else
        "OUR SEAT WAS HANDLED WORSE THAN THE POD. A win rate from this run is "
        "measuring the AI's preferences as much as the decks; treat the outcome "
        "as uninformative and read the observations instead."
    )
    return out


def render(q):
    if not q:
        return []
    lines = ["  PILOTING (was the AI playing this deck, or holding it?)"]
    for metric, label in ((LANDS, "land drops per own turn"),
                          (CASTS, "spells cast per own turn")):
        m = q[metric]
        lines.append(f"    {label:26} ours {m['ours']:.2f}  pod {m['pod_mean']:.2f}"
                     f"  ({m['ratio']:.0%} of the pod's rate)")
    lines.append("    " + ("COMPARABLE" if q["comparable"]
                           else "NOT ENOUGH GAMES" if q["comparable"] is None
                           else "NOT COMPARABLE"))
    lines.append("    " + q["reading"])
    return lines
