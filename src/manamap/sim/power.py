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
This does not stop anybody running anything — unless the pilot ASKED a question
the run cannot answer. `--detect X` is that question, and a run that cannot see
X at 80% power is refused with the arithmetic, and `--anyway` runs it regardless
(a noise floor, a smoke test, the first half of a bigger sample are all
legitimate; they are just not "can this see X"). Without `--detect` it prints
and proceeds, as it always has.
"""

import json


from manamap import config
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
    d = config.DECKS_DIR / slug / "sim"
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


def null_rate(pod_name):
    """The table's subject null and the games behind it, or (None, None).

    ABSENT MEANS ABSENT: a table nothing has been measured against has no null,
    and a default here would be an invented figure standing exactly where a
    measured one belongs. Every failure mode — no pod name, no calibration
    file, an untracked table, a zero rate — returns (None, None). Moved here
    from `net_change._pod_null` so `simulate`, `experiment` and `net-change`
    read the one figure the same way.
    """
    if not pod_name:
        return None, None
    try:
        from manamap.sim import pods
        cal = pods.calibration(pod_name) or {}
        row = cal.get("subject_null") or {}
        rate = row.get("rate")
    except Exception:                                 # noqa: BLE001
        return None, None
    if not isinstance(rate, (int, float)) or rate <= 0:
        return None, None
    return rate, row.get("games") or cal.get("games")


def refuse_if_underpowered(p_a, games, detect, anyway=False, n_a=None,
                           target_power=0.8):
    """Refuse a run that cannot see the effect the pilot asked for.

    Only when `detect` was given: without it the run made no claim and nothing
    is refused. With it, power below `target_power` is a SystemExit naming the
    power and the games that would do it — unless `anyway`, which records that
    the pilot chose to run a screen rather than a test.
    """
    if detect is None or anyway or p_a is None:
        return None
    n_a = n_a or games
    pw = stats.power_for(p_a, min(0.999, p_a + detect), n_a, games) \
        if max(n_a, games) <= stats.EXACT_MDE_MAX_N \
        else stats._power_normal(p_a, min(0.999, p_a + detect), n_a, games)
    if pw >= target_power:
        return pw
    need = stats.games_for_difference(p_a, detect)
    raise SystemExit(
        f"UNDERPOWERED — {games} games can see a {detect:+.2f} change with "
        f"{pw:.0%} power against a baseline of {p_a:.3f}; 80% needs "
        f"{'more than 5000' if need is None else need} per arm. Run it anyway as a "
        f"smoke test with --anyway, or ask a question this many games can answer.")


def preflight(p_a, games, detect=None, per_minute=GAMES_PER_MINUTE, arms=2,
              n_a=None):
    """Lines describing what `games` per arm can resolve against `p_a`.

    `arms` is how many arms the hours are for — two for an experiment, one for
    `simulate`, whose comparison arm is the pod's null and costs nothing to
    play again. `n_a` is that comparison arm's size when it is not `games`
    (the null's game count), so the power is the test that will actually be
    run rather than an equal-arms stand-in.
    """
    if p_a is None:
        return ["  POWER: no measured baseline for this table — run `simulate` "
                "once first, or accept that nothing here can say what this "
                "experiment could see."]
    out = []
    n_a = n_a or games
    m = stats.mde_proportion(p_a, n_a, games)
    hours = arms * games / per_minute / 60
    out.append(f"  POWER PREFLIGHT   baseline {p_a:.3f} · {games} games"
               f"{'/arm' if arms > 1 else ''}"
               f"{'' if n_a == games else f' against {n_a} on the other side'}"
               f" · about {hours:.1f} h")
    out.append(f"    this run resolves a change of {m['minimum_detectable_difference']:+.3f} "
               f"or larger (a rise to {m['minimum_detectable_rate_b']:.3f})")
    out.append("    smaller than that comes back INCONCLUSIVE — which is not "
               "evidence of no effect")
    out.append("")
    out.append(f"    {'you want to detect':<22}{'power here':>12}{'games/arm needed':>19}{'hours':>8}")
    wanted = [0.05, 0.10, 0.15, 0.20]
    if detect is not None and detect not in wanted:
        wanted = sorted(wanted + [detect])
    def _power(d, n):
        pb = min(0.999, p_a + d)
        if max(n_a, n) <= stats.EXACT_MDE_MAX_N:
            return stats.power_for(p_a, pb, n_a, n)
        return stats._power_normal(p_a, pb, n_a, n)

    for d in wanted:
        pw = _power(d, games)
        need = stats.games_for_difference(p_a, d)
        need_s = ">5000" if need is None else str(need)
        hrs = "—" if need is None else f"{arms*need/per_minute/60:.1f}"
        star = "  <-- asked" if detect is not None and abs(d - detect) < 1e-9 else ""
        out.append(f"    {('+%.2f' % d):<22}{pw:>12.2f}{need_s:>19}{hrs:>8}{star}")
    if detect is not None:
        pw = _power(detect, games)
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
