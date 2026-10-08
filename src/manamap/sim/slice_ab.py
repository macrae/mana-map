"""A/B on one board: the same scenario with option A and option B, paired seed by seed.

The scenario-slice counterpart of `try`. Each seed deals the hidden cards ONCE, by the
keyed shuffle in `slice_state` (a card's place depends on the seed and the card, not on
what else is in the pile), so arm A and arm B on seed k face the same draws apart from
the card being tested. The difference is read seed by seed — a paired interval, the
noise the two arms share cancels — on ONE primary measure named before the run, with
the rest exploratory and Holm-corrected (the `net-change` contract).

WHAT IT ANSWERS AND WHAT IT DOES NOT. Every seat is the Forge AI. A slice is evidence
about what the AI does with the board, which is why each report carries a MISPLAY flag:
when a card the arms differ on starts in your hand and the AI never casts it in most
replicates, the arm measured a hand holding a dead card, not the card. Sean discounts
that arm; the report says so in its answer line.

`compare(base, seat_slugs, arms, seeds, primary)` -> report dict;
`format_report(report)` -> the plain answer plus the table.
"""

import math
import re

from manamap.pilot import game_state as gs
from manamap.sim import slice as sl
from manamap.sim import slice_state as ss
from manamap.sim import stats

#: name -> (definition, function(record, ctx) -> number). `ctx` carries the seat
#: index of `you` (always 0), your seat's Forge name and your commanders.
def _life(seat):
    return seat["life"]


def _opp_life_lost(r, ctx):
    return sum(s0["life"] - s1["life"] for s0, s1 in zip(r["start"][1:], r["end"][1:]))


def _your_life_change(r, ctx):
    return r["end"][0]["life"] - r["start"][0]["life"]


def _board_change(r, ctx):
    return len(r["end"][0]["battlefield"]) - len(r["start"][0]["battlefield"])


def _hand_end(r, ctx):
    return len(r["end"][0]["hand"])


def _commander_out(r, ctx):
    bf = {n.split(" // ")[0] for n in r["end"][0]["battlefield"]}
    return 1.0 if any(c.split(" // ")[0] in bf for c in ctx["commanders"]) else 0.0


def _you_lost(r, ctx):
    return 1.0 if r["end"][0]["lost"] else 0.0


_DRAW = re.compile(r"^Zone Change: .+ \(\d+\) was put into Hand from Library\.? owner (\S+)")


def _cards_drawn(r, ctx):
    return float(sum(1 for line in r["log"]
                     if (m := _DRAW.match(line)) and m.group(1) == ctx["name"]))


MEASURES = {
    "opp_life_lost": ("life the opponents lost over the slice, summed across them", _opp_life_lost),
    "your_life_change": ("your life at the end minus at the start", _your_life_change),
    "your_board_change": ("your permanents at the end minus at the start", _board_change),
    "your_hand_end": ("cards in your hand when the slice stopped", _hand_end),
    "commander_out": ("1 if a commander of yours is on the battlefield at the end", _commander_out),
    "you_lost": ("1 if you lost the game inside the slice", _you_lost),
    "cards_drawn": ("cards you put into hand from your library during the slice "
                    "(the telemetry log's zone lines)", _cards_drawn),
}
#: A card an arm adds that the AI left uncast in at least this share of replicates.
HELD_SHARE = 0.5


def paired(xs, ys):
    """The mean of (y - x) seed by seed with a 95% t interval, or None under 2 pairs."""
    d = [y - x for x, y in zip(xs, ys)]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1))
    se = sd / math.sqrt(n)
    half = stats.t_crit(n - 1) * se if se else 0.0
    lo, hi = round(mean - half, 3), round(mean + half, 3)
    z = (mean / se) if se else (0.0 if mean == 0 else float("inf"))
    return {"diff": round(mean, 3), "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "n": n, "z": z}


def _your_cards(scenario):
    me = gs.our_seat(scenario) or {}
    hand = me.get("hand")
    names = list(hand) if isinstance(hand, list) else list((hand or {}).get("known") or [])
    return names + [gs.entry_name(e) for e in me.get("board") or []]


def key_cards(arms):
    """Per arm, the cards of yours it holds that the other arm does not."""
    a, b = list(arms)
    ca, cb = _your_cards(arms[a]), _your_cards(arms[b])
    only = lambda xs, ys: sorted(set(xs) - set(ys))
    return {a: only(ca, cb), b: only(cb, ca)}


def _cast(r, ctx, card):
    front = card.split(" // ")[0]
    for line in r["log"]:
        if line.startswith(f"Add To Stack: {ctx['name']} cast {front}") or \
           line.startswith(f"Land: {ctx['name']} played {front} ("):
            return True
    return False


def compare(base, seat_slugs, arms, seeds, primary, rounds=1, measures=None, run=None):
    """Play both arms on every seed and read the difference seed by seed.

    `arms` is `{label: scenario}`, exactly two, A first. `seat_slugs` maps seat ids to
    the decks that play them. `run` is injectable for tests (defaults to `slice.run`)."""
    if len(arms) != 2:
        raise ValueError("an A/B has exactly two arms")
    names = list(measures or MEASURES)
    if primary not in names:
        raise ValueError(f"primary {primary!r} is not a measure: {', '.join(names)}")
    run = run or sl.run
    seats = [gs.our_seat(base)] + gs.opponent_seats(base)
    order = [seat_slugs[s["seat"]] for s in seats]
    cases, notes = [], []
    for label, sc in arms.items():
        for seed in seeds:
            text, n = ss.to_forge_state(sc, seat_slugs, seed)
            cases.append((label, seed, text))
            if seed == seeds[0]:
                notes += [f"{label}: {x}" for x in n]
    recs = run(order, cases, rounds=rounds)

    from manamap.sim import forge
    you = seat_slugs["you"]
    commanders = ss._deck_copies(you)[1]
    ctx = {"name": f"Ai(1)-{forge.SIM_DECK_PREFIX}{forge.deck_meta_name(you)}", "commanders": commanders}

    a, b = list(arms)
    by = {(r["label"], r["seed"]): r for r in recs}
    good = [s for s in seeds if "error" not in by.get((a, s), {"error": 1})
            and "error" not in by.get((b, s), {"error": 1})]
    errors = [f"{r['label']} seed {r['seed']}: {r['error']}" for r in recs if "error" in r]

    rows = []
    for m in names:
        defn, fn = MEASURES[m]
        xs = [fn(by[(a, s)], ctx) for s in good]
        ys = [fn(by[(b, s)], ctx) for s in good]
        pr = paired(xs, ys)
        rows.append({"measure": m, "definition": defn, "primary": m == primary,
                     "a_mean": round(sum(xs) / len(xs), 3) if xs else None,
                     "b_mean": round(sum(ys) / len(ys), 3) if ys else None, "paired": pr})
    explo = [r for r in rows if not r["primary"] and r["paired"]]
    for r, h in zip(explo, stats.holm([r["paired"]["z"] for r in explo]) if explo else []):
        r["holm"] = h

    held = []
    for label, cards in key_cards(arms).items():
        for card in cards:
            starts_in_hand = [s for s in good if card.split(" // ")[0] in
                              {h.split(" // ")[0] for h in by[(label, s)]["start"][0]["hand"]}]
            if not starts_in_hand:
                continue
            k = sum(1 for s in starts_in_hand if not _cast(by[(label, s)], ctx, card))
            if k / len(starts_in_hand) >= HELD_SHARE:
                held.append({"arm": label, "card": card, "held": k, "of": len(starts_in_hand)})

    report = {"setup": {"seats": dict(zip([s["seat"] for s in seats], order)),
                        "arms": {lab: key_cards(arms)[lab] for lab in arms},
                        "seeds": len(good), "rounds": rounds, "primary": primary,
                        "turn": base.get("turn"), "phase": base.get("phase")},
              "rows": rows, "misplays": held, "notes": notes, "errors": errors}
    report["answer"] = answer(report)
    return report


def answer(report):
    """One plain sentence on the primary measure, and the misplay caveat if any."""
    a, b = list(report["setup"]["arms"])
    row = next(r for r in report["rows"] if r["primary"])
    pr, n = row["paired"], report["setup"]["seeds"]
    if not pr:
        s = f"Too few paired seeds ({n}) to read {row['measure']}."
    elif not pr["excludes_zero"]:
        s = (f"No difference on {row['measure']} the run could see: {b} minus {a} = "
             f"{pr['diff']:+} [{pr['ci95'][0]:+}, {pr['ci95'][1]:+}] over {n} paired seeds.")
    else:
        word = "more" if pr["diff"] > 0 else "less"
        s = (f"{b} gives {word} {row['measure']} than {a}: {pr['diff']:+} "
             f"[{pr['ci95'][0]:+}, {pr['ci95'][1]:+}] over {n} paired seeds.")
    for h in report["misplays"]:
        s += (f" DISCOUNT {h['arm']}: the AI held {h['card']} uncast in {h['held']} of "
              f"{h['of']} replicates.")
    return s


def format_report(report):
    st = report["setup"]
    out = [report["answer"], "",
           f"  setup: {', '.join(f'{k}={v}' for k, v in st['seats'].items())} · turn {st['turn']} "
           f"{st['phase']} · {st['rounds']} round(s) after this turn · {st['seeds']} paired seeds",
           "  arms: " + " | ".join(f"{k}: {', '.join(v) or '(same cards)'}" for k, v in st["arms"].items())]
    a, b = list(st["arms"])
    out.append(f"\n  {'measure':22} {a:>8} {b:>8}   {b}-{a} [95%]")
    for r in report["rows"]:
        pr = r["paired"]
        tag = "PRIMARY" if r["primary"] else ("holm ✓" if (r.get("holm") or {}).get("significant") else "")
        cell = f"{pr['diff']:+.2f} [{pr['ci95'][0]:+.2f}, {pr['ci95'][1]:+.2f}]" if pr else "—"
        out.append(f"  {r['measure']:22} {r['a_mean'] if r['a_mean'] is not None else '—':>8} "
                   f"{r['b_mean'] if r['b_mean'] is not None else '—':>8}   {cell}  {tag}")
    for n in report["notes"]:
        out.append(f"  note: {n}")
    for e in report["errors"]:
        out.append(f"  error: {e}")
    out.append("\n  every seat is the Forge AI: this is what the AI does with the board, not proof")
    return "\n".join(out)
