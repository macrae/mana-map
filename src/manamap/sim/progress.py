"""What a running simulation has done so far — and how far from settled it is.

WRITTEN AFTER AN EIGHT-HOUR JOB WITH NO WINDOW INTO IT. `simulate` and
`experiment` print once, at the end. A 120-game pod run takes two and a half
hours and a two-arm experiment twice that, and for the whole of it the only way
to know anything was to grep the logs by hand — which is exactly what happened,
four times in one session, each time with a slightly different ad-hoc parser.

THIS IS A PROGRESS VIEW, NOT A RESULT, and the distinction is the whole design.
Two things make a partial read worse than no read:

  OPTIONAL STOPPING. Watch a rate wander and stop when it looks good and you
  have manufactured a result. The run's target N is fixed when it is launched
  and this command never suggests ending early.

  GAMES INSIDE A JOB ARE NOT INDEPENDENT. Forge's `Match` carries `lastOutcome`
  and gives the first turn to the previous game's LOSER, so a job's early games
  are systematically different from its late ones. A partial slice is biased in
  a direction that depends on who has been losing, which is why the seats are
  rotated per job in the first place (see `forge.run`).

So what is reported is the running estimate WITH its interval and, beside it,
how much the interval still has to shrink. `converged` is not a verdict about
the deck; it is arithmetic about the sample.
"""

import glob
import math
import os
import re
import time
from pathlib import Path

from manamap.config import DECKS_DIR
from manamap.sim import parse as sim_parse

#: Forge writes one of these per finished game, whatever the ending.
_DONE_RE = re.compile(r"^Game Result: Game (\d+) ended in ", re.M)
_START_RE = re.compile(r"^Turn: Turn 1 \(", re.M)
#: `...-n120-...` / `...-n100-...` — the target the run was launched with.
_TARGET_RE = re.compile(r"-n(\d+)-")


def _bar(done, total, width=32):
    if not total:
        return "-" * width
    filled = min(width, int(round(width * done / total)))
    return "█" * filled + "·" * (width - filled)


def _seat_map(texts):
    """Forge seat token -> slug, ours first. The tokens rotate per job and an
    experiment suffixes the arm, so this is derived from the logs rather than
    assumed from the record — which may not exist yet."""
    toks = set()
    for t in texts:
        toks |= set(re.findall(r"Ai\(\d\)-[\w-]+", t))

    def slug(tok):
        n = tok.split("-", 1)[1]
        n = n[3:] if n.startswith("mm-") else n
        if n.startswith("x-"):                      # experiment arm: mm-x-<slug>-a
            n = re.sub(r"^x-(.+?)-[ab]$", r"\1", n)
        return n

    ours = None
    for t in sorted(toks):
        if "-x-" in t or t.endswith("-a") or t.endswith("-b"):
            ours = slug(t)
    lab = {t: slug(t) for t in sorted(toks)}
    if ours is None and lab:
        # a plain `simulate` run: ours is the deck the log dir belongs to, and
        # the caller passes it in through `_report`.
        pass
    return lab


def _order_round_robin(per_job):
    """Approximate ARRIVAL order by interleaving the jobs.

    Jobs run in parallel at similar rates, so job 0's k-th game finished at
    roughly the same time as job 1's. Interleaving is closer to the order the
    games actually completed than concatenating whole jobs, which would make the
    convergence trace read as "job 0's whole sample, then job 1's".

    It is still an APPROXIMATION and is labelled as one wherever it is printed.
    """
    out = []
    for i in range(max((len(j) for j in per_job), default=0)):
        for j in per_job:
            if i < len(j):
                out.append(j[i])
    return out


def _wilson_half(k, n):
    if not n:
        return None
    lo, hi = sim_parse.wilson(k, n)
    return (hi - lo) / 2


def _convergence(flags, target):
    """The running rate at checkpoints, and how much the interval must still
    shrink. `flags` is arrival-ordered 1/0 for "our seat won this decided game".
    """
    n = len(flags)
    steps = [s for s in (10, 20, 30, 40, 50, 75, 100, 150, 200) if s <= n]
    if n and (not steps or steps[-1] != n):
        steps.append(n)
    rows = []
    for s in steps:
        k = sum(flags[:s])
        lo, hi = sim_parse.wilson(k, s)
        rows.append((s, k / s, lo, hi, (hi - lo) / 2))
    # Where the half-width lands if the rate holds to the target N.
    projected = None
    if n and target and target > n:
        p = sum(flags) / n
        k_t = int(round(p * target))
        lo, hi = sim_parse.wilson(k_t, target)
        projected = (target, p, lo, hi, (hi - lo) / 2)
    return rows, projected


def _arm_started(paths):
    """When THIS arm began, from its own log files.

    The run directory is created when the whole job starts, so an experiment's
    arm B was being credited with arm A's two and a half hours: seven games over
    145 minutes read as 0.05 games/min and "about 1921 min left". The arms run in
    SEQUENCE, so only the arm's own files can say when it started.
    """
    times = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        times.append(getattr(st, "st_birthtime", st.st_ctime))
    return min(times) if times else None


def _report(paths, label, target, started, ours_hint=None):
    texts = [Path(p).read_text(errors="replace") for p in paths]
    started = _arm_started(paths) or started
    done = sum(len(_DONE_RE.findall(t)) for t in texts)
    started_n = sum(len(_START_RE.findall(t)) for t in texts)
    elapsed = time.time() - started if started else 0
    rate = done / (elapsed / 60) if elapsed > 0 else 0
    eta = ((target - done) / rate) if (rate > 0 and target and target > done) else None

    print(f"  {label}")
    print(f"    [{_bar(done, target)}] {done}/{target or '?'} finished"
          f"{f', {started_n - done} in flight' if started_n > done else ''}")
    if elapsed:
        line = f"    {rate:.2f} games/min · {elapsed/60:.0f} min elapsed"
        if eta is not None:
            line += f" · about {eta:.0f} min left"
        print(line)
    if not done:
        print("    nothing finished yet — no estimate\n")
        return

    lab = _seat_map(texts)
    if ours_hint:
        lab = {**{k: v for k, v in lab.items() if v == ours_hint},
               **{k: v for k, v in lab.items() if v != ours_hint}}
    facts, agg = sim_parse.analyze_logs(texts, lab, None)
    ours = ours_hint or next(iter(lab.values()), None)
    clocked = agg["games"] - agg["decided"]
    print(f"    {agg['decided']} decided of {agg['games']} parsed"
          + (f" ({clocked} clocked out)" if clocked else ""))
    for name, v in sorted(agg["seats"].items(), key=lambda kv: -(kv[1]["win_rate"] or 0)):
        ci = v["win_rate_ci95"]
        mark = " <-" if name == ours else ""
        print(f"      {name:<18}{v['wins']:>4}  {v['win_rate']:>6.3f}  "
              f"[{ci[0]:.2f}, {ci[1]:.2f}]{mark}")

    # CONVERGENCE, over our seat only — the figure a branch is judged on.
    per_job = []
    for t in texts:
        job = []
        for f in sim_parse.parse_games(t):
            fact = sim_parse.game_facts(f)
            if not fact["winner"] or fact.get("truncated"):
                continue
            job.append(1 if lab.get(fact["winner"]) == ours else 0)
        per_job.append(job)
    flags = _order_round_robin(per_job)
    rows, projected = _convergence(flags, target)
    if rows:
        print(f"\n    CONVERGENCE — {ours}'s rate as the sample grew"
              f"  (jobs interleaved to approximate arrival order)")
        print(f"      {'n':>5}{'rate':>8}   {'ci95':<16}{'+/-':>7}")
        for s, p, lo, hi, half in rows:
            print(f"      {s:>5}{p:>8.3f}   [{lo:.2f}, {hi:.2f}]  {half:>6.3f}")
        if projected:
            s, p, lo, hi, half = projected
            print(f"      {s:>5}{p:>8.3f}   [{lo:.2f}, {hi:.2f}]  {half:>6.3f}"
                  f"   <- where it lands at the target N IF the rate holds")
    print()


def main(args):
    slug = args.slug
    roots = [DECKS_DIR / slug / "sim" / "logs",
             DECKS_DIR / slug / "experiments" / "logs"]
    dirs = [d for r in roots if r.is_dir() for d in sorted(r.iterdir()) if d.is_dir()]
    if getattr(args, "run", None):
        dirs = [d for d in dirs if args.run in d.name]
    if not dirs:
        raise SystemExit(f"{slug}: no simulation logs under "
                         f"{'/'.join(str(r) for r in roots)} — logs are gitignored "
                         f"and only exist where the run was made")
    # Newest first; a running job is almost always the one you meant.
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    if not getattr(args, "all", False):
        dirs = dirs[:1]

    for d in dirs:
        m = _TARGET_RE.search(d.name)
        target = int(m.group(1)) if m else None
        st = d.stat()
        started = getattr(st, "st_birthtime", st.st_ctime)
        arms = {"a": sorted(glob.glob(str(d / "a-part-*.log"))),
                "b": sorted(glob.glob(str(d / "b-part-*.log")))}
        plain = sorted(glob.glob(str(d / "part-*.log")))
        live = any(time.time() - os.path.getmtime(f) < 300
                   for f in (plain + arms["a"] + arms["b"]) if os.path.exists(f))
        print(f"\nSIM PROGRESS — {slug}   {'RUNNING' if live else 'idle'}")
        print(f"  {d.name}\n")
        if plain:
            _report(plain, "run", target, started, ours_hint=slug)
        else:
            for arm in ("a", "b"):
                if arms[arm]:
                    _report(arms[arm], f"arm {arm.upper()}", target, started,
                            ours_hint=slug)
                else:
                    print(f"  arm {arm.upper()}\n    [{_bar(0, target)}] 0/{target}"
                          f" — not started; the arms run in sequence\n")

    print("  A PROGRESS VIEW, NOT A RESULT.")
    print("  · Games inside a job are NOT independent: Forge gives the first turn to")
    print("    the previous game's loser, so a job's early games differ systematically")
    print("    from its late ones and a partial slice is biased.")
    print("  · Stopping when a number looks good manufactures a result. The target N")
    print("    is fixed at launch; nothing here is a reason to end early.")
    print("  · The convergence trace interleaves jobs to APPROXIMATE arrival order.")
    print("    The games did not really finish in that order.")
