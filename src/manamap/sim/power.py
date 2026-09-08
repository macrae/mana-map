"""What a planned experiment could actually see, BEFORE it is launched.

`stats` has carried `power_for`, `mde_proportion` and `games_for_difference`
since the statistics went in, and nothing ever called them before a run. So a
100-game-per-arm A/B was launched against a 0.244 baseline, took four hours,
and had a **0.34 probability** of detecting a real ten-point improvement. It was
always more likely to miss than to find, and that was knowable in a millisecond.

  games/arm at 80% power, baseline 0.244
    +0.05   >1000/arm       —
    +0.10     324/arm    15.4 h
    +0.15     149/arm     7.1 h

THE WIN RATE IS A LOW-POWER ENDPOINT and no amount of care fixes that: it is
binary, it is rare, and a clocked-out game has no winner so it is DISCARDED —
17% of one run's games were paid for and then thrown out of the denominator.
This does not stop anybody running anything. It prints the arithmetic at the one
moment it can still change the decision.
"""

import json

from manamap.config import DECKS_DIR
from manamap.sim import stats

#: Observed on this machine at 4 jobs, across the heliod runs: ~0.7 games/min.
#: Used only to turn a game count into an hour count, so a stale value makes the
#: hours wrong and never the statistics.
GAMES_PER_MINUTE = 0.7


def baseline_rate(slug, opponents):
    """Our seat's most recent measured win rate against THIS table.

    Absent rather than guessed: a preflight computed from a default rate would
    be a number about nothing. Returns (rate, run_id) or (None, None).
    """
    d = DECKS_DIR / slug / "sim"
    if not d.is_dir():
        return None, None
    want = sorted(opponents)
    best = None
    for p in sorted(d.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                             # pragma: no cover
            continue
        if sorted(s["slug"] for s in rec.get("seats", [])[1:]) != want:
            continue
        seat = (rec.get("analysis", {}).get("seats") or {}).get(slug) or {}
        if seat.get("win_rate") is None:
            continue
        n = rec.get("games_completed") or 0
        if best is None or n > best[2]:
            best = (seat["win_rate"], rec.get("run_id", p.stem), n)
    return (best[0], best[1]) if best else (None, None)


def preflight(p_a, games, detect=None, per_minute=GAMES_PER_MINUTE):
    """Lines describing what `games` per arm can resolve against `p_a`."""
    if p_a is None:
        return ["  POWER: no measured baseline for this table — run `simulate` "
                "once first, or accept that nothing here can say what this "
                "experiment could see."]
    out = []
    m = stats.mde_proportion(p_a, games)
    hours = 2 * games / per_minute / 60
    out.append(f"  POWER PREFLIGHT   baseline {p_a:.3f} · {games} games/arm · "
               f"about {hours:.1f} h")
    out.append(f"    this run resolves a change of {m['minimum_detectable_difference']:+.3f} "
               f"or larger (a rise to {m['minimum_detectable_rate_b']:.3f})")
    out.append("    smaller than that comes back INCONCLUSIVE — which is not "
               "evidence of no effect")
    out.append("")
    out.append(f"    {'you want to detect':<22}{'power here':>12}{'games/arm needed':>19}{'hours':>8}")
    wanted = [0.05, 0.10, 0.15, 0.20]
    if detect is not None and detect not in wanted:
        wanted = sorted(wanted + [detect])
    for d in wanted:
        pw = stats.power_for(p_a, min(0.999, p_a + d), games, games)
        need = stats.games_for_difference(p_a, d)
        need_s = ">1000" if need is None else str(need)
        hrs = "—" if need is None else f"{2*need/per_minute/60:.1f}"
        star = "  <-- asked" if detect is not None and abs(d - detect) < 1e-9 else ""
        out.append(f"    {('+%.2f' % d):<22}{pw:>12.2f}{need_s:>19}{hrs:>8}{star}")
    if detect is not None:
        pw = stats.power_for(p_a, min(0.999, p_a + detect), games, games)
        out.append("")
        if pw < 0.5:
            out.append(f"    UNDERPOWERED: {pw:.0%} chance of seeing a {detect:+.2f} change. "
                       f"This run is more likely to MISS a real effect than to find it.")
        elif pw < 0.8:
            out.append(f"    THIN: {pw:.0%} chance of seeing a {detect:+.2f} change; "
                       f"0.80 is the usual floor.")
        else:
            out.append(f"    adequate: {pw:.0%} chance of seeing a {detect:+.2f} change.")
    out.append("")
    out.append("    The win rate also DISCARDS clocked-out games — they have no")
    out.append("    winner — so the effective sample is below the games played.")
    out.append("    A mechanism endpoint (does the swap do its job?) is usually a")
    out.append("    cheaper question than the outcome (does the deck win more?).")
    return out
