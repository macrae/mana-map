"""Simulation S1: the Forge harness — run N Commander games of one deck against
opponents, headless, across JVMs, and record the run.

Forge is the rules engine (docs/simulation.md records the spike and the verdict). This
module owns everything around it: converting `decklist.txt` to Forge's `.dck` through
the repo's own parser (so a seat can never disagree with `fetch-deck`), placing decks
where Forge's sim mode looks, splitting N games across J JVMs, capturing every game log,
and writing ONE tracked run record beside them.

WHAT A RUN RECORD IS. `data/decks/<slug>/sim/<run-id>.json` — tier ◆ **seeded**: Forge's
sim mode takes `-s <seed>` (undocumented in its wiki, present in its source and in this
release), and with it identical inputs reproduce the logs BYTE FOR BYTE — measured twice
on this machine, including a two-game sequence under one seed. The first runs in this
repo were made before that was known and are recorded as SAMPLED; they stay valid as
history. A seeded run is replayable game by game (`-n g -s <seed+job>`), which is what
the S4 bridge stands on. The record states N, the seeds, what it measured per game, and
the assumptions that bound it — including Forge's own verdict on its AI, verbatim. The
raw logs sit in `sim/logs/<run-id>/` and are gitignored: large and regenerable, now
exactly.

The run id is `<opponents>-n<N>-<sha8 over every seat's decklist>-s<seed>`. The default
seed is derived from that digest, so the same configuration replays the same games; a
NEW sample is a new `--seed`, and a swap is a new digest. Re-running an existing id is a
replay and is refused without `--force` (it would write the same bytes).

S1 records per-game OUTCOME only (winner, turn, wall ms, how each seat lost) — those lines
are unambiguous in the log. The event parser that turns the rest of the log into damage,
tokens, blocks and life is S2, and reads the same logs.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from manamap.config import (DECKS_DIR, FORGE_DECKS_DIR, FORGE_HOME, FORGE_JVM_ARGS,
                            SIM_DECK_PREFIX, SIM_DEFAULT_GAMES, SIM_DIR,
                            SIM_GAME_CLOCK_SECONDS)
from manamap.pilot.common import deck_dir, load_json
from manamap.sim import parse as sim_parse

# Forge's own verdict on its AI, from its docs/AI.md. Quoted into every run record so a
# number never travels without the limit that bounds it.
FORGE_AI_CAVEAT = ('Forge\'s AI "is not trained"; it is "best with aggro and midrange '
                   'decks, poor to ok in control decks, pretty bad for most combo decks" '
                   '(Forge docs/AI.md). A control deck\'s win rate here is a lower bound '
                   'on a competent pilot\'s; a combo deck\'s is not a measurement.')
SEEDED_NOTE = ("SEEDED: every JVM job ran `-s <seed>` (seeds recorded per job), and with this "
               "Forge build identical inputs reproduce the logs byte for byte — game g of job j "
               "replays as `-n g -s <seed_j>`. Still a sample of N games: the interval is the "
               "claim, the seed is the receipt.")
SAMPLED_NOTE = ("SAMPLED, NOT SEEDED: this run was made before `-s` was known. Identical runs "
                "diverge; every figure is a sample of N games and no single game is replayable.")
ASSUMPTIONS = [
    SEEDED_NOTE,
    FORGE_AI_CAVEAT,
    "Seats are Forge AIs on both sides — including YOUR deck. A result is AI-vs-AI, "
    "never pilot-vs-pod.",
    "Two turn counts: `round` is the winner's own turn count (Forge's `Game Outcome: "
    "Turn N`), `global_turn` is the game's last `Turn:` line — in a 4-seat game round 8 "
    "is global turn ~32. Measured on a 2-seat log: Outcome Turn 8, last Turn: line 16.",
    "A game past the clock is recorded as a draw, not dropped.",
]

_OPPONENT_ROOTS = ("opponents", "decks")     # data/opponents/<slug> first (S3), then a deck


def seat_dir(slug):
    """A seat is a deck under data/opponents/ (the pod) or data/decks/ (your own)."""
    for root in _OPPONENT_ROOTS:
        p = DECKS_DIR.parent / root / slug
        if (p / "decklist.txt").exists():
            return p
    raise SystemExit(f"no decklist.txt for seat {slug!r} under data/opponents/ or data/decks/")


def to_dck(slug):
    """Forge .dck text for a seat, through the repo's own decklist parser."""
    from manamap.pilot.fetch_deck import parse_decklist
    text = (seat_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
    entries = parse_decklist(text)
    cmd = [e for e in entries if e.get("is_commander")]
    main = [e for e in entries if not e.get("is_commander")]
    if not cmd:
        raise SystemExit(f"{slug}: decklist names no commander (*CMDR* or a Commander: section)")
    lines = ["[metadata]", f"Name={SIM_DECK_PREFIX}{slug}", "[Commander]"]
    lines += [f"{int(e.get('quantity') or 1)} {e['name']}" for e in cmd]
    lines += ["[Main]"] + [f"{int(e.get('quantity') or 1)} {e['name']}" for e in main]
    return "\n".join(lines) + "\n"


def install_deck(slug, decks_dir=None):
    """Write the seat's .dck where Forge's sim mode looks. Returns the meta name."""
    decks_dir = decks_dir or FORGE_DECKS_DIR
    decks_dir.mkdir(parents=True, exist_ok=True)
    name = f"{SIM_DECK_PREFIX}{slug}"
    (decks_dir / f"{name}.dck").write_text(to_dck(slug), encoding="utf-8")
    return name


def forge_jar(home=None):
    home = home or FORGE_HOME
    jars = sorted(home.glob("forge-gui-desktop-*-jar-with-dependencies.jar"))
    if not jars:
        raise SystemExit(f"no Forge desktop jar under {home} — install Forge there or set "
                         f"MANAMAP_FORGE_HOME (docs/simulation.md)")
    return jars[-1]


def forge_version(home=None):
    jar = forge_jar(home)
    m = re.search(r"forge-gui-desktop-([\d.]+)-jar", jar.name)
    build = (home or FORGE_HOME) / "build.txt"
    return {"version": m.group(1) if m else jar.name,
            "build": build.read_text().strip().splitlines()[0] if build.exists() else None}


def seat_sha(slug):
    base = seat_dir(slug)
    text_path = base / "decklist.txt"
    return hashlib.sha256(text_path.read_bytes()).hexdigest()


def config_digest(slug, opponents):
    return hashlib.sha256(
        "\n".join(f"{s}:{seat_sha(s)}" for s in [slug, *opponents]).encode()).hexdigest()[:8]


def default_seed(slug, opponents):
    """Derived from the configuration, so the default replays; pass --seed for a new sample."""
    return int(config_digest(slug, opponents), 16) % 2_000_000_000


def run_id(slug, opponents, games, seed=None):
    seed = default_seed(slug, opponents) if seed is None else int(seed)
    return f"{'-vs-'.join(opponents)}-n{games}-{config_digest(slug, opponents)}-s{seed}"


def split_games(games, jobs):
    """N games across J processes, as evenly as integers allow, no empty job."""
    jobs = max(1, min(int(jobs), int(games)))
    base, extra = divmod(int(games), jobs)
    return [base + (1 if i < extra else 0) for i in range(jobs)]


def command(seat_names, games, clock, jar=None, seed=None):
    """The exact argv one JVM runs. A pure function so a test can read it."""
    jar = jar or forge_jar()
    argv = ["java", *FORGE_JVM_ARGS, "-jar", str(jar), "sim",
            "-d", *seat_names, "-f", "commander", "-n", str(games), "-c", str(clock)]
    if seed is not None:
        argv += ["-s", str(int(seed))]
    return argv


# ── Outcome parsing (the unambiguous lines; the event parser is S2) ─────────

_OUTCOME_TURN = re.compile(r"^Game Outcome: Turn (\d+)")
_TURN = re.compile(r"^Turn: Turn (\d+)")
# "has won because all opponents have lost" is the common line; an alternate win
# condition prints "has won due to effect of 'Approach of the Second Sun'". Measured on
# the first tracked run: two heliod wins read as draws until the second form was matched.
_OUTCOME_WON = re.compile(r"^Game Outcome: (Ai\(\d+\)-\S+) has won (?:because|due to) (.*)$")
_OUTCOME_LOST = re.compile(r"^Game Outcome: (Ai\(\d+\)-\S+) has lost because (.*)$")
_OUTCOME_DRAW = re.compile(r"^Game Outcome: .*draw", re.I)
_RESULT = re.compile(r"^Game Result: Game (\d+) ended in (\d+) ms")


def parse_outcomes(text):
    """Per-game outcomes from one JVM's log: winner, turn, ms, how each seat lost."""
    games, cur, last_turn = [], None, None
    for line in text.splitlines():
        m = _TURN.match(line)
        if m:
            last_turn = int(m.group(1)); continue
        if cur is None and line.startswith("Game Outcome:"):
            cur = {"winner": None, "won_by": None, "round": None, "global_turn": last_turn,
                   "draw": False, "lost": {}, "ms": None}
            last_turn = None
        if cur is None:
            continue
        m = _OUTCOME_TURN.match(line)
        if m:
            cur["round"] = int(m.group(1)); continue
        m = _OUTCOME_WON.match(line)
        if m:
            cur["winner"] = m.group(1); cur["won_by"] = m.group(2).rstrip("."); continue
        m = _OUTCOME_LOST.match(line)
        if m:
            cur["lost"][m.group(1)] = m.group(2).rstrip("."); continue
        if _OUTCOME_DRAW.match(line):
            cur["draw"] = True; continue
        m = _RESULT.match(line)
        if m:
            cur["ms"] = int(m.group(2))
            games.append(cur); cur = None
    if cur is not None and (cur["winner"] or cur["draw"]):   # a final game with no Result line
        games.append(cur)
    return games


def _seat_label(seat_names):
    """Forge labels seats `Ai(k)-<meta name>` in -d order; map back to slugs."""
    return {f"Ai({i + 1})-{name}": name[len(SIM_DECK_PREFIX):] if name.startswith(SIM_DECK_PREFIX) else name
            for i, name in enumerate(seat_names)}


def run(slug, opponents, games=SIM_DEFAULT_GAMES, jobs=None, clock=SIM_GAME_CLOCK_SECONDS,
        seed=None, force=False, dry_run=False, home=None, decks_dir=None):
    """Run the games and write the run record. Returns (record_path, record)."""
    if not opponents:
        raise SystemExit("simulate needs at least one opponent: --vs <slug> (repeatable)")
    seats = [slug, *opponents]
    names = [install_deck(s, decks_dir) for s in seats]
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    parts = split_games(games, jobs)
    seed_base = default_seed(slug, opponents) if seed is None else int(seed)
    seeds = [seed_base + i for i in range(len(parts))]
    rid = run_id(slug, opponents, games, seed_base)
    out_dir = deck_dir(slug) / SIM_DIR
    log_dir = out_dir / "logs" / rid
    record_path = out_dir / f"{rid}.json"
    if record_path.exists() and not force and not dry_run:
        raise SystemExit(f"{slug}: {record_path.name} exists — the same configuration and seed "
                         f"replays the same games. Pass --seed N for a new sample, or --force "
                         f"to replay it.")
    jar = forge_jar(home)
    cmds = [command(names, g, clock, jar, seed=seeds[i]) for i, g in enumerate(parts)]
    if dry_run:
        return record_path, {"run_id": rid, "seats": seats, "jobs": len(parts),
                             "games_per_job": parts, "seeds": seeds, "commands": cmds}

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(i_cmd):
        i, cmd = i_cmd
        log = log_dir / f"part-{i:02d}.log"
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  cwd=str(jar.parent), text=True)
        return log, proc.returncode

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        results = list(ex.map(one, enumerate(cmds)))
    wall = round(time.time() - t0, 1)

    label = _seat_label(names)
    outcomes = []
    for log, rc in results:
        for gi, g in enumerate(parse_outcomes(log.read_text(encoding="utf-8", errors="replace"))):
            outcomes.append({
                "winner": label.get(g["winner"], g["winner"]) if g["winner"] else None,
                "won_by": g["won_by"], "draw": g["draw"], "round": g["round"],
                "global_turn": g["global_turn"], "ms": g["ms"],
                "lost": {label.get(k, k): v for k, v in g["lost"].items()},
                "log": log.name, "seed": seeds[int(log.stem.split("-")[1])],
                "game_in_job": gi + 1})
    texts = [log.read_text(encoding="utf-8", errors="replace") for log, _ in results]
    facts, analysis = sim_parse.analyze_logs(texts, label)
    wins = {s: sum(1 for o in outcomes if o["winner"] == s) for s in seats}
    draws = sum(1 for o in outcomes if o["draw"] or not o["winner"])
    frame = load_json(deck_dir(slug) / "strategic_frame.json") or {}
    record = {
        "run_id": record_path.stem, "slug": slug, "at": date.today().isoformat(),
        "engine": {"forge": forge_version(home), "java": _java_version()},
        "seats": [{"slug": s, "forge_name": names[i], "decklist_sha256": seat_sha(s)}
                  for i, s in enumerate(seats)],
        "games_requested": int(games), "games_completed": len(outcomes),
        "jobs": len(cmds), "games_per_job": parts, "seed_base": seed_base, "seeds": seeds,
        "clock_seconds": clock,
        "wall_seconds": wall, "nonzero_exit_jobs": sum(1 for _, rc in results if rc),
        "summary": {"wins": wins, "draws": draws,
                    "win_rate": (round(wins[slug] / len(outcomes), 3) if outcomes else None),
                    "mean_round": (round(sum(o["round"] or 0 for o in outcomes) / len(outcomes), 1)
                                   if outcomes else None),
                    "mean_global_turn": (round(sum(o["global_turn"] or 0 for o in outcomes) / len(outcomes), 1)
                                         if outcomes and all(o["global_turn"] for o in outcomes) else None)},
        "outcomes": outcomes,
        "analysis": analysis,
        "games": [sim_parse.compact(f, label) for f in facts],
        "assumptions": ASSUMPTIONS + ([f"{slug}'s strategic frame calls it "
                                       f"{frame.get('archetype')!r} — read the AI caveat "
                                       f"against that."] if frame.get("archetype") else []),
        "logs": f"{SIM_DIR}/logs/{log_dir.name}/ (gitignored; regenerate by re-running)",
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record_path, record


def analyze(slug, run_id_or_path):
    """Re-derive a run's analysis from its kept logs and rewrite the record. The logs
    are gitignored, so this works only where the run was made; `validate-sim` uses the
    same path to prove the tracked analysis is what the logs say."""
    base = deck_dir(slug) / SIM_DIR
    path = base / (run_id_or_path if str(run_id_or_path).endswith(".json") else f"{run_id_or_path}.json")
    if not path.exists():
        raise SystemExit(f"{slug}: no run record {path.name} under {SIM_DIR}/")
    rec = load_json(path)
    log_dir = base / "logs" / path.stem
    logs = sorted(log_dir.glob("part-*.log"))
    if not logs:
        raise SystemExit(f"{slug}: no logs under {log_dir} — the raw games are gitignored and "
                         f"only exist where the run was made")
    label = _seat_label([s["forge_name"] for s in rec["seats"]])
    facts, analysis = sim_parse.analyze_logs([l.read_text(encoding="utf-8", errors="replace") for l in logs], label)
    rec["analysis"] = analysis
    rec["games"] = [sim_parse.compact(f, label) for f in facts]
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return path, rec


def _java_version():
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return (out.stderr or out.stdout).strip().splitlines()[0]
    except (OSError, IndexError):
        return None


def list_runs(slug):
    base = deck_dir(slug) / SIM_DIR
    return [load_json(p) for p in sorted(base.glob("*.json"))] if base.is_dir() else []


def main(args):
    slug = args.slug
    if getattr(args, "analyze", None):
        path, rec = analyze(slug, args.analyze)
        a = rec["analysis"]
        me = a["seats"].get(slug, {})
        print(f"{slug}: re-derived {a['games']} game(s) from logs → {path.name}")
        print(f"  win rate {me.get('win_rate')} ci95 {me.get('win_rate_ci95')} · "
              f"token damage share {me.get('tokens', {}).get('token_damage_share')}")
        return
    if getattr(args, "list", False) or not getattr(args, "vs", None):
        runs = list_runs(slug)
        if not runs:
            print(f"{slug}: no simulation runs — "
                  f"`manamap pilot simulate {slug} --vs <opponent> [--vs …] --games N`")
            return
        print(f"SIMULATION RUNS — {slug} ({len(runs)})\n")
        for r in runs:
            s = r["summary"]
            me = (r.get("analysis") or {}).get("seats", {}).get(slug, {})
            ci = me.get("win_rate_ci95")
            print(f"{r['run_id']}  {r['at']}  {r['games_completed']}/{r['games_requested']} games  "
                  f"win {s['win_rate']}{' ci95 ' + str(ci) if ci else ''}  mean round {s['mean_round']}  "
                  f"{r['wall_seconds']}s on {r['jobs']} JVM(s)")
            print(f"      vs {', '.join(x['slug'] for x in r['seats'][1:])}  ·  wins {s['wins']}")
        return
    path, rec = run(slug, args.vs, games=args.games or SIM_DEFAULT_GAMES, jobs=args.jobs,
                    clock=args.clock or SIM_GAME_CLOCK_SECONDS, seed=getattr(args, "seed", None),
                    force=getattr(args, "force", False), dry_run=getattr(args, "dry_run", False))
    if getattr(args, "dry_run", False):
        print(f"would run {rec['games_per_job']} games across {rec['jobs']} JVM(s), seeds "
              f"{rec['seeds']} → {path}")
        for c in rec["commands"]:
            print("  " + " ".join(c))
        return
    s = rec["summary"]
    print(f"{slug}: {rec['games_completed']}/{rec['games_requested']} games vs "
          f"{', '.join(args.vs)} in {rec['wall_seconds']}s on {rec['jobs']} JVM(s)")
    print(f"  wins {s['wins']}  draws {s['draws']}  win rate {s['win_rate']}  "
          f"mean round {s['mean_round']} (global turn {s['mean_global_turn']})")
    if rec["nonzero_exit_jobs"]:
        print(f"  WARNING {rec['nonzero_exit_jobs']} JVM(s) exited non-zero — read the logs")
    print(f"  → {path.relative_to(deck_dir(slug))}  (logs: {rec['logs']})")
    me = rec["analysis"]["seats"].get(slug, {})
    print(f"  ci95 {me.get('win_rate_ci95')} · eliminated by {me.get('eliminated_by')} · "
          f"token damage share {(me.get('tokens') or {}).get('token_damage_share', {}).get('mean')}")
    print(f"  ◆ seeded ({rec['seed_base']}; per job {rec['seeds']}) — {len(rec['assumptions'])} "
          f"assumptions in the record; game g of job j replays as `-n g -s <seed_j>`")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot simulate <slug> --vs <opponent> [--games N]`.")
