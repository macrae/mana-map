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

Lands and casts per own turn are the two signals, chosen because they are the
ones that separated: they need no card knowledge, they are recorded for every
seat, and a deck that is not making land drops is not being piloted whatever else
is true of it.
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

LANDS, CASTS = "lands_per_turn", "casts_per_turn"


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
    ratios = [out[m]["ratio"] for m in (LANDS, CASTS) if out[m]["ratio"]]
    out["comparable"] = bool(ratios) and min(ratios) >= COMPARABLE
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
    lines.append("    " + ("COMPARABLE" if q["comparable"] else "NOT COMPARABLE"))
    lines.append("    " + q["reading"])
    return lines
