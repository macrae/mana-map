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
    # MEASURED AND WRONG, so it says what actually happens. `summary.draws` is 0
    # on every tracked run INCLUDING the two with three clock-hit games each, and
    # edgar-vampires n=400 carries 75 clock-hit games (19%) all with winners.
    "A game past Forge's `-c` clock is NOT recorded as a draw: Forge reports a "
    "WINNER for it and the parse takes that at face value, so a deck that trips "
    "the clock has its rate scored off truncated games with nothing marking "
    "them. Measured: edgar-vampires n=400 had 75 clock-hit games (19%) and zero "
    "recorded draws.",
    "A JOB is capped at clock x games x 1.5 + 120s of wall time. Forge's `-c` "
    "only ends the game's accounting, not its AI thread — two tracked 20-game "
    "runs took 3.7 and 4.2 HOURS with 95% of the wall claimed by no game. A "
    "killed job's finished games are kept; `truncated_jobs` names the rest.",
]

_OPPONENT_ROOTS = ("opponents", "decks")     # data/opponents/<slug> first (S3), then a deck


#: A BRANCH IS A SEAT. `ur-dragon@treasure-v2` sits down at the table exactly as
#: `ur-dragon` does — it is a list, and Forge does not care where the list came
#: from. That also makes the most interesting run expressible in the existing
#: grammar: a branch against its own champion, same pod, same seed.
BRANCH_SEP = "@"


def split_seat(slug):
    """`ur-dragon@treasure-v2` -> ('ur-dragon', 'treasure-v2')."""
    if BRANCH_SEP in slug:
        base, branch = slug.split(BRANCH_SEP, 1)
        return base, branch or None
    return slug, None


def seat_dir(slug):
    """A seat is a deck under data/opponents/ (the pod), data/decks/ (your own),
    or a BRANCH of one of your own (`<slug>@<branch>`)."""
    base, branch = split_seat(slug)
    if branch:
        from manamap.pilot.common import deck_dir as _dd
        p = _dd(base, branch)
        if (p / "decklist.txt").exists():
            return p
        raise SystemExit(f"no decklist.txt for branch {branch!r} of {base!r}")
    for root in _OPPONENT_ROOTS:
        p = DECKS_DIR.parent / root / slug
        if (p / "decklist.txt").exists():
            return p
    raise SystemExit(f"no decklist.txt for seat {slug!r} under data/opponents/ or data/decks/")


def _out_dir(slug):
    """Where a run record goes — BESIDE THE LIST IT MEASURED.

    A branch run must never land in the deck's own `sim/`: the record carries the
    figures, and a branch's win rate filed under the champion's name is the
    silent-overwrite class this repo keeps finding. `deck_dir(base, branch)`
    resolves it structurally, the same way every other `--branch` command scopes.
    """
    from manamap.pilot.common import deck_dir as _dd
    base, branch = split_seat(slug)
    return _dd(base, branch) / SIM_DIR


def commanders_from_text(decklist_text):
    """The commander name(s) named by a decklist TEXT.

    An experiment arm's list rides IN its artifact, so an arm's commander comes from
    that arm's own text rather than from disk — two arms may legitimately be two
    versions with different commanders, and each must be scored against its own.
    """
    from manamap.pilot.fetch_deck import parse_decklist
    return {e["name"] for e in parse_decklist(decklist_text) if e.get("is_commander")}


def _commanders_by_slug(seats):
    """{slug -> set of commander names} read from each seat's decklist.

    The Forge log never says which permanent is a commander, so the decklist is the only
    honest source — but it is read ONCE, when the run is made or migrated, and then
    written into the record (see `record_commanders`). A seat whose decklist cannot be
    read is omitted, which makes its commander-damage block absent rather than zero.
    """
    from manamap.pilot.fetch_deck import parse_decklist
    out = {}
    for slug in seats:
        try:
            text = (seat_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
            cmd = {e["name"] for e in parse_decklist(text) if e.get("is_commander")}
        except (SystemExit, OSError):
            continue
        if cmd:
            out[slug] = cmd
    return out


def record_commanders(rec):
    """{Forge seat label -> commander name(s)} taken from the RUN RECORD, never from disk.

    Re-derivation must depend only on the record and its logs. Looking the commander up
    from `data/decks/<slug>/decklist.txt` at validate time would mean a later commander
    swap on any seat reads as parser drift on a run that was correct when it was made —
    and a record written before this field existed would suddenly gain a block its
    stored analysis does not have, turning every old run red.

    A record whose seats carry no `commander` therefore yields nothing, and re-derives
    exactly as it always did. `simulate <slug> --analyze <run>` is the migration: it
    reads the decklists once, writes the field, and rewrites the analysis with it.
    """
    out = {}
    for i, seat in enumerate(rec.get("seats") or []):
        names = seat.get("commander")
        if not names:
            continue
        if isinstance(names, str):
            names = [names]
        out[f"Ai({i + 1})-{seat['forge_name']}"] = set(names)
    return out


def dck_from_text(meta_name, decklist_text, who="the list"):
    """Forge .dck text from a decklist TEXT, through the repo's own parser — so an
    experiment arm's historical list converts exactly the way a live seat's does."""
    from manamap.pilot.fetch_deck import parse_decklist
    entries = parse_decklist(decklist_text)
    cmd = [e for e in entries if e.get("is_commander")]
    main = [e for e in entries if not e.get("is_commander")]
    if not cmd:
        raise SystemExit(f"{who}: decklist names no commander (*CMDR* or a Commander: section)")
    lines = ["[metadata]", f"Name={meta_name}", "[Commander]"]
    lines += [f"{int(e.get('quantity') or 1)} {e['name']}" for e in cmd]
    lines += ["[Main]"] + [f"{int(e.get('quantity') or 1)} {e['name']}" for e in main]
    return "\n".join(lines) + "\n"


def to_dck(slug):
    """Forge .dck text for a seat directory."""
    text = (seat_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
    # `@` is legal in a filename on this platform and is asking for trouble in
    # Forge's own deck registry, so the seat's meta name flattens it. The RECORD
    # still carries the seat as written, because that is what identifies the list.
    return dck_from_text(f"{SIM_DECK_PREFIX}{deck_meta_name(slug)}", text, who=slug)


def deck_meta_name(slug):
    """A Forge-safe name for a seat, branch included."""
    return slug.replace(BRANCH_SEP, "-")



def tally_wins(outcomes, seats):
    """Wins per SEAT SLUG, matching on the Forge name.

    EXTRACTED SO A TEST CAN DRIVE IT. This expression lived inline, and the test
    guarding it built its own correct copy — so a regression to
    `o["winner"] == s` left the test green while the bug it was written for
    reappeared. The bug: a branch seat is written to Forge as
    `ur-dragon-treasure-v2` because `@` has no business in a deck registry, so
    the outcome names the flattened form while `seats` holds the slug. A bare
    `==` matched every OTHER seat and silently scored ours zero — the run
    reported "wins 0" for a list that had won ELEVEN of a hundred, with the
    other three seats correct beside it, which is exactly the shape that gets
    believed.
    """
    return {s: sum(1 for o in outcomes if o["winner"] == deck_meta_name(s))
            for s in seats}

def install_named(meta_name, decklist_text, decks_dir=None):
    """Install an arbitrary list under an explicit Forge meta name (experiments)."""
    decks_dir = decks_dir or FORGE_DECKS_DIR
    decks_dir.mkdir(parents=True, exist_ok=True)
    (decks_dir / f"{meta_name}.dck").write_text(
        dck_from_text(meta_name, decklist_text, who=meta_name), encoding="utf-8")
    return meta_name


def install_deck(slug, decks_dir=None):
    """Write the seat's .dck where Forge's sim mode looks. Returns the meta name."""
    decks_dir = decks_dir or FORGE_DECKS_DIR
    decks_dir.mkdir(parents=True, exist_ok=True)
    name = f"{SIM_DECK_PREFIX}{deck_meta_name(slug)}"
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


#: The AI personality every seat gets unless something says otherwise. Named
#: because "Default" appears in three places and a typo in any of them would
#: silently produce a different pilot.
#: How much longer than its own clock a job may take before it is killed, and a
#: floor for JVM start-up plus deck loading. Generous on purpose: the point is to
#: bound a RUNAWAY, not to police a slow game.
TIMEOUT_SLACK = 1.5
TIMEOUT_FLOOR = 120

DEFAULT_PROFILE = "Default"

#: THE POD'S STANDARD PILOT, changed from Default on 2026-08-30 after measuring
#: it. The Default AI misplays a token deck badly: over 100 games on one pod,
#: switching the OPPONENTS to Experimental moved baylen-tokens from 0.130 to
#: 0.190 and vito from 0.422 to 0.330 — so the table was never uniformly weak,
#: it was weak UNEVENLY, which quietly flattered whichever seat the Default AI
#: happened to pilot competently.
#:
#: Our own seat is unaffected: switching it changed the win rate by -0.0125
#: with an interval spanning zero, so this is about the table being honest and
#: not about the result moving.
#:
#: THE TAG RULE IS DELIBERATELY UNCHANGED. `profile_tag` still omits a suffix
#: only for "Default", so every run made before this date keeps its id and
#: still means what it said, and every run made after carries `-podExperimental`
#: visibly. A standard that silently reinterpreted existing records would be
#: worse than a longer id. EVERY FORGE RECORD PREDATING THIS LINE WAS MEASURED
#: AGAINST THE DEFAULT POD and is not directly comparable to one made after it.
STANDARD_POD_PROFILE = "Experimental"


def profile_tag(profile=None, vs_profile=None):
    """The run-id suffix for a non-default pilot, or "" for the usual case.

    THE RUN ID DID NOT CARRY THE PROFILE AND THAT IS A SILENT OVERWRITE. The id
    is built from the opponents, the game count, a digest over every seat's
    DECKLIST and the seed — none of which move when the AI does. So
    `simulate <deck> --profile Experimental` wrote to exactly the path the
    Default run had already written, and the second result replaced the first
    with no warning and no way to tell them apart afterwards. Same class as
    `goldfish.main` filing a branch measurement under the champion's name.

    Absent means the default, so every existing run id is unchanged and no
    record on disk has to be renamed.
    """
    # `me` AND `pod`, NOT `ai` AND `vsai`. The first cut used those, and
    # `-aiExperimental` is a SUBSTRING of `-vsaiExperimental` — so any glob or
    # grep for the one matches the other, and the first comparison written
    # against these ids read the same directory twice and reported two
    # configurations as byte-identical. Neither of these is a substring of the
    # other, in either order.
    bits = []
    if profile and profile != DEFAULT_PROFILE:
        bits.append(f"me{profile}")
    if vs_profile and vs_profile != DEFAULT_PROFILE:
        bits.append(f"pod{vs_profile}")
    return ("-" + "-".join(bits)) if bits else ""


def run_id(slug, opponents, games, seed=None, profile=None, vs_profile=None):
    seed = default_seed(slug, opponents) if seed is None else int(seed)
    return (f"{'-vs-'.join(opponents)}-n{games}-{config_digest(slug, opponents)}"
            f"-s{seed}{profile_tag(profile, vs_profile)}")


def split_games(games, jobs):
    """N games across J processes, as evenly as integers allow, no empty job."""
    jobs = max(1, min(int(jobs), int(games)))
    base, extra = divmod(int(games), jobs)
    return [base + (1 if i < extra else 0) for i in range(jobs)]


def command(seat_names, games, clock, jar=None, seed=None, profiles=None):
    """The exact argv one JVM runs. A pure function so a test can read it.

    `profiles` is per-seat AI profiles in `-d` order (Forge's `-a`), from
    res/ai/: Default, Cautious, Reckless, Experimental. Measured 2026-08-19 on
    radagast vs a Default edgar, 6 seeded games per profile on OUR seat:
    Default 3/6, Experimental 2/6, Reckless 2/6 — the aggro profiles make a
    hold-up deck worse, so Default stays the default and the AI caveat stands.
    """
    jar = jar or forge_jar()
    argv = ["java", *FORGE_JVM_ARGS, "-jar", str(jar), "sim",
            "-d", *seat_names, "-f", "commander", "-n", str(games), "-c", str(clock)]
    if profiles:
        argv += ["-a", *profiles]
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
        seed=None, force=False, dry_run=False, home=None, decks_dir=None,
        profile=None, vs_profile=None):
    """Run the games and write the run record. Returns (record_path, record)."""
    if not opponents:
        raise SystemExit("simulate needs at least one opponent: --vs <slug> (repeatable)")
    seats = [slug, *opponents]
    names = [install_deck(s, decks_dir) for s in seats]
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    parts = split_games(games, jobs)
    seed_base = default_seed(slug, opponents) if seed is None else int(seed)
    seeds = [seed_base + i for i in range(len(parts))]
    pod = vs_profile or STANDARD_POD_PROFILE
    rid = run_id(slug, opponents, games, seed_base, profile, pod)
    out_dir = _out_dir(slug)
    log_dir = out_dir / "logs" / rid
    record_path = out_dir / f"{rid}.json"
    if record_path.exists() and not force and not dry_run:
        raise SystemExit(f"{slug}: {record_path.name} exists — the same configuration and seed "
                         f"replays the same games. Pass --seed N for a new sample, or --force "
                         f"to replay it.")
    jar = forge_jar(home)
    # THE POD IS PART OF THE INSTRUMENT. `--profile` set only OUR seat and left
    # every opponent on Default, so the table could never be made to play
    # differently — and a win rate is relative to the pod's competence as much
    # as to the deck. `--vs-profile` sets every opponent seat.
    profiles = None
    if profile or pod != DEFAULT_PROFILE:
        profiles = [profile or DEFAULT_PROFILE] + [pod] * len(opponents)
    cmds = [command(names, g, clock, jar, seed=seeds[i], profiles=profiles)
            for i, g in enumerate(parts)]
    if dry_run:
        return record_path, {"run_id": rid, "seats": seats, "jobs": len(parts),
                             "games_per_job": parts, "seeds": seeds, "commands": cmds}

    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # A HUNG JVM HAD NO BOUND AT ALL, and that is where the hours went.
    # Forge's `-c` is a FutureTask timeout: it REPORTS the game as ended and the
    # AI thread carries on. Measured on the tracked runs, per-game `ms` against
    # wall time:
    #
    #   edgar-vampires  n=20   509s wall,  95% accounted by its own games
    #   edgar-vampires  n=400 11243s wall, 90% accounted
    #   yawgmoth-swarm  n=20  13372s wall,  3% accounted   <- 3.7 HOURS
    #   zur-enchantress n=20  15220s wall,  5% accounted   <- 4.2 HOURS
    #
    # Eleven of thirteen runs are healthy; two burned four hours on twenty games
    # and 95% of that time is claimed by no game at all. This caps a job at what
    # its own clock says it could possibly need, with generous headroom, and
    # records the truncation rather than hiding it.
    per_job_cap = int(clock * max(parts) * TIMEOUT_SLACK) + TIMEOUT_FLOOR

    def one(i_cmd):
        i, cmd = i_cmd
        log = log_dir / f"part-{i:02d}.log"
        with open(log, "w", encoding="utf-8") as f:
            try:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=str(jar.parent), text=True,
                                      timeout=per_job_cap)
                return log, proc.returncode, False
            except subprocess.TimeoutExpired:
                # The games it DID finish are already in the log and are parsed
                # normally; what is lost is the tail of this job.
                f.write(f"\n[manamap] job killed after {per_job_cap}s "
                        f"(clock {clock}s x {max(parts)} games x {TIMEOUT_SLACK} "
                        f"+ {TIMEOUT_FLOOR}s)\n")
                return log, None, True

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        raw = list(ex.map(one, enumerate(cmds)))
    timed_out = [i for i, (_log, _rc, killed) in enumerate(raw) if killed]
    results = [(log, rc) for log, rc, _killed in raw]
    wall = round(time.time() - t0, 1)
    if timed_out:
        print(f"  WARNING {len(timed_out)} of {len(cmds)} job(s) hit the "
              f"{per_job_cap}s cap and were killed; their unfinished games are "
              f"absent from this run. `truncated_jobs` records which.")

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
    cmd_by_slug = {s: sorted(c) for s, c in _commanders_by_slug(seats).items()}
    commanders = {f"Ai({i + 1})-{names[i]}": set(cmd_by_slug[s])
                  for i, s in enumerate(seats) if cmd_by_slug.get(s)}
    facts, analysis = sim_parse.analyze_logs(texts, label, commanders)
    # MATCH ON THE FORGE NAME, NOT THE SEAT SLUG. A branch seat is written to
    # Forge as `ur-dragon-treasure-v2` because `@` has no business in a deck
    # registry — so the outcome names the flattened form while `seats` holds the
    # slug, and a bare `==` matched every OTHER seat and silently scored ours
    # zero. The run reported "wins 0" for a list that had won ELEVEN of a
    # hundred, with the other three seats' counts all correct beside it, which is
    # exactly the shape that gets believed.
    wins = tally_wins(outcomes, seats)
    draws = sum(1 for o in outcomes if o["draw"] or not o["winner"])
    frame = load_json(deck_dir(split_seat(slug)[0]) / "strategic_frame.json") or {}
    record = {
        # ABSENT MEANS ABSENT: a run with no truncation carries an empty list,
        # never a missing key, so a reader can tell "none" from "not checked".
        "truncated_jobs": timed_out,
        "run_id": record_path.stem, "slug": slug, "at": date.today().isoformat(),
        "engine": {"forge": forge_version(home), "java": _java_version()},
        "seats": [{"slug": s, "forge_name": names[i], "decklist_sha256": seat_sha(s),
                   "commander": sorted(cmd_by_slug.get(s, []))}
                  for i, s in enumerate(seats)],
        "games_requested": int(games), "games_completed": len(outcomes),
        "jobs": len(cmds), "games_per_job": parts, "seed_base": seed_base, "seeds": seeds,
        "profiles": profiles,
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
    base = _out_dir(slug)
    path = base / (run_id_or_path if str(run_id_or_path).endswith(".json") else f"{run_id_or_path}.json")
    if not path.exists():
        raise SystemExit(f"{slug}: no run record {path.name} under {SIM_DIR}/")
    rec = load_json(path)
    log_dir = base / "logs" / path.stem
    logs = sorted(log_dir.glob("part-*.log"))
    if not logs:
        raise SystemExit(f"{slug}: no logs under {log_dir} — the raw games are gitignored and "
                         f"only exist where the run was made")
    names = [s["forge_name"] for s in rec["seats"]]
    label = _seat_label(names)
    # The one place a decklist is consulted: --analyze backfills the commander names on a
    # record written before the field existed, so the block can appear at all. After this
    # the record is self-describing and nothing reads disk again.
    by_slug = _commanders_by_slug([s["slug"] for s in rec["seats"]])
    for seat in rec["seats"]:
        if not seat.get("commander") and by_slug.get(seat["slug"]):
            seat["commander"] = sorted(by_slug[seat["slug"]])
    facts, analysis = sim_parse.analyze_logs(
        [l.read_text(encoding="utf-8", errors="replace") for l in logs], label,
        record_commanders(rec))
    rec["analysis"] = analysis
    rec["games"] = [sim_parse.compact(f, label) for f in facts]
    # SUMMARY IS RE-DERIVED TOO, or `--analyze` cannot repair a wrong headline.
    # It did not, and the run that scored a branch seat zero kept saying zero
    # through a re-derivation that had the right numbers in `analysis` all along.
    seats = [x["slug"] for x in rec["seats"]]
    outcomes = [g for g in rec["games"] if g.get("winner") or g.get("draw")]
    wins = {x: sum(1 for g in outcomes if g.get("winner") == deck_meta_name(x))
            for x in seats}
    n = len(outcomes) or None
    rec["summary"] = dict(rec.get("summary") or {}, wins=wins,
                          win_rate=(round(wins[slug] / n, 3) if n else None))
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    return path, rec


def _java_version():
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return (out.stderr or out.stdout).strip().splitlines()[0]
    except (OSError, IndexError):
        return None


def list_runs(slug):
    base = _out_dir(slug)
    return [load_json(p) for p in sorted(base.glob("*.json"))] if base.is_dir() else []


def main(args):
    slug = args.slug
    if getattr(args, "analyze", None):
        path, rec = analyze(slug, args.analyze)
        a = rec["analysis"]
        me = a["seats"].get(deck_meta_name(slug), {})
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
                    force=getattr(args, "force", False), dry_run=getattr(args, "dry_run", False),
                    profile=getattr(args, "profile", None),
                    vs_profile=getattr(args, "vs_profile", None))
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
    # COMPUTED AT PRINT TIME, NEVER WRITTEN INTO THE RECORD — the same rule
    # `targeting` follows, so adding this moved no tracked run. It is derived
    # from the record, so it reads the same from a fresh checkout.
    from manamap.sim import pilot_quality
    for line in pilot_quality.render(pilot_quality.from_record(rec)):
        print(line)
    if rec["nonzero_exit_jobs"]:
        print(f"  WARNING {rec['nonzero_exit_jobs']} JVM(s) exited non-zero — read the logs")
    print(f"  → {path}  (logs: {rec['logs']})")
    me = rec["analysis"]["seats"].get(slug, {})
    print(f"  ci95 {me.get('win_rate_ci95')} · eliminated by {me.get('eliminated_by')} · "
          f"token damage share {(me.get('tokens') or {}).get('token_damage_share', {}).get('mean')}")
    print(f"  ◆ seeded ({rec['seed_base']}; per job {rec['seeds']}) — {len(rec['assumptions'])} "
          f"assumptions in the record; game g of job j replays as `-n g -s <seed_j>`")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot simulate <slug> --vs <opponent> [--games N]`.")
