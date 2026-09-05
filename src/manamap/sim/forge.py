"""Simulation S1: the Forge harness — run N Commander games of one deck against
opponents, headless, across JVMs, and record the run.

Forge is the rules engine (docs/simulation.md records the spike and the verdict). This
module owns everything around it: converting `decklist.txt` to Forge's `.dck` through
the repo's own parser (so a seat can never disagree with `fetch-deck`), placing decks
where Forge's sim mode looks, splitting N games across J JVMs, capturing every game log,
and writing ONE tracked run record beside them.

WHAT A RUN RECORD IS. `data/decks/<slug>/sim/<run-id>.json` — tier ◆ **seeded**: Forge's
sim mode takes `-s <seed>` (undocumented in its wiki, present in its source and in this
release), and it fixes the SHUFFLE.

A SEED IS NOT A RECEIPT, AND THIS FILE CLAIMED OTHERWISE FOR MONTHS — "identical inputs
reproduce the logs BYTE FOR BYTE, measured twice". Measured again properly on 2026-09-02:
goblin-storm, same pod, same seed (424242), same clock, same job count, run twice. ALL
FOUR GAMES DIFFERED — turns 33/41/32/33 against 31/43/42/41 — and game 3 had a DIFFERENT
WINNER. One game took 23 s in one pass and 398 s in the other. The mechanism is in the
logs: `AI eval thread at timeout` fired 12 times across those 4 games. Forge budgets its
AI's evaluation in WALL TIME and dumps the thread when it overruns, after which the AI
plays whatever it had — so how well a seat is piloted depends on machine load, and no seed
controls that. The original "measured twice" was almost certainly two short games under
light load, where the abort never fired.

WHAT THAT COSTS. The game-by-game replay claim: `-n g -s <seed_j>` reproduces the deal,
not the game, so the S4 bridge lifts a board FROM THE STORED LOG rather than by re-running
it. The stored log is the receipt — which is why `validate-sim` re-derives every figure
from the logs instead of trusting the record. It costs nothing at the level a run record is
actually read: N games against a pod is a SAMPLE either way, and the interval was always
the claim. The record states N, the seeds, what it measured per game, and the assumptions
that bound it, including Forge's own verdict on its AI. The raw logs sit in
`sim/logs/<run-id>/` and are gitignored: large, and regenerable only as a fresh sample.

The run id is `<opponents>-n<N>-<sha8 over every seat's decklist>-s<seed>`, plus a tag for
a non-default AI profile or clock. The default seed is derived from that digest; a NEW
sample is a new `--seed`, and a swap is a new digest. Re-running an existing id is refused
without `--force` — not because it would write the same bytes, which it would not, but
because it would OVERWRITE one measurement with a different one under the same name.

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
                            SIM_GAME_CLOCK_SECONDS, SIM_CLOCK_ID_BASELINE)
from manamap.pilot.common import deck_dir, load_json
from manamap.sim import parse as sim_parse
from manamap.sim import pods as _pods

# Forge's own verdict on its AI, from its docs/AI.md. Quoted into every run record so a
# number never travels without the limit that bounds it.
FORGE_AI_CAVEAT = ('Forge\'s AI "is not trained"; it is "best with aggro and midrange '
                   'decks, poor to ok in control decks, pretty bad for most combo decks" '
                   '(Forge docs/AI.md). A control deck\'s win rate here is a lower bound '
                   'on a competent pilot\'s; a combo deck\'s is not a measurement.')
SEEDED_NOTE = ("SEEDED FIXES THE SHUFFLE, NOT THE GAMES. Every JVM job ran `-s <seed>` "
               "(seeds recorded per job) and the same seed deals the same cards — but one "
               "configuration replayed twice on 2026-09-02 gave four different games and a "
               "different winner in one of them. Forge budgets its AI's evaluation in WALL "
               "TIME and abandons the search when it overruns (`AI eval thread at timeout`, "
               "12 times in those 4 games), so how well a seat is piloted depends on machine "
               "load. The STORED LOGS are the receipt, not the seed — `validate-sim` "
               "re-derives every figure from them. N games against a pod is a sample either "
               "way, and the interval is the claim.")
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
    # CORRECTED 2026-09-02. The previous text observed the symptom — clock-hit
    # games carrying winners and `summary.draws` reading 0 — and concluded that
    # Forge reports a winner for them. It does not. Decompiled, Forge catches its
    # own timeout, prints "Stopping slow match as draw" and calls
    # `setGameOver(GameEndReason.Draw)`; it then prints `has won because all
    # opponents have lost` for EVERY SEAT STILL ALIVE. Our parser assigned on
    # each such line, so the LAST one won — the highest-numbered survivor.
    #
    # Our deck is always `Ai(1)`. Across 121 truncated games it was credited
    # with ZERO while surviving to the clock in 93 of them, and `baylen-tokens`
    # (always the last seat) took 73 of its 85 recorded wins that way. It is also
    # the whole of the "win rate falls as N grows" signature: the clock-hit share
    # runs 0% at n=20, 9% at n=100, 18% at n=400.
    "A game past Forge's `-c` clock IS a draw — Forge calls `setGameOver(Draw)` "
    "— but it prints a `has won` line for every surviving seat, so a naive parse "
    "credits one of them. Such games are now recorded `truncated: true` with no "
    "winner and are EXCLUDED from the win rate; `summary.truncated` and "
    "`summary.decided` report how many. A truncated game is also not "
    "byte-reproducible: Forge interrupts the accounting but not the AI thread, "
    "so the outcome lines are a snapshot of a game still being mutated.",
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
    seats = rec.get("seats") or []
    for seat in seats:
        names = seat.get("commander")
        if not names:
            continue
        if isinstance(names, str):
            names = [names]
        # EVERY SEAT INDEX, not just the one this deck was passed in as. Seats
        # rotate per job, so a commander keyed to one position would credit
        # commander damage to whichever deck later sat there. The label already
        # carries the deck name; position was never needed. See `_seat_label`.
        for k in range(1, len(seats) + 1):
            out[f"Ai({k})-{seat['forge_name']}"] = set(names)
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


def pod_tag_name(pod, opponents):
    """What `profile_tag` should call this table's pilot.

    A scalar passes straight through, so nothing that ran before pods existed
    changes name. A MAP collapses to the shared profile when every seat agrees —
    so a pod file setting one profile is byte-identical to `--vs-profile <that>`
    — and to `Mixed<8hex>` when they do not, because two tables that play
    differently must not write the same path.
    """
    if not isinstance(pod, dict):
        return pod
    from manamap.sim import pods as _pods

    return _pods.mixed_tag(pod, list(opponents), STANDARD_POD_PROFILE)


def clock_tag(clock=None):
    """The run-id suffix for a clock that is not the historical one, or "".

    THE CLOCK IS PART OF THE CONFIGURATION AND THE ID DID NOT CARRY IT — the
    same omission as the profile, one field over. A run's id is built from the
    seats, the game count, a decklist digest and the seed, so
    `simulate <deck> --clock 600` wrote to the exact path the 300 s run had
    already written and replaced it with no warning.

    It is not merely a filing problem: the clock decides which games are
    TRUNCATED, and a truncated game has no winner. Two records at different
    clocks are two different measurements of the deck, and pooling them under
    one name would mix populations.

    Compared against SIM_CLOCK_ID_BASELINE and never against the current
    default, so raising the default does not rename a single record on disk.
    """
    return f"-c{int(clock)}" if clock and int(clock) != SIM_CLOCK_ID_BASELINE else ""


def run_id(slug, opponents, games, seed=None, profile=None, vs_profile=None, clock=None):
    seed = default_seed(slug, opponents) if seed is None else int(seed)
    return (f"{'-vs-'.join(opponents)}-n{games}-{config_digest(slug, opponents)}"
            f"-s{seed}{profile_tag(profile, vs_profile)}{clock_tag(clock)}")


def run_id_for(slug, opponents, games, seed, profile, vs_profile, clock):
    """The run id a given configuration WILL write, before anything runs.

    A function because two callers need the answer and neither should re-derive
    it: `run` names the record, and the pod tests assert that `--pod standard`
    and the three `--vs` flags it replaces produce the SAME id. A test that
    recomputed the rule would be testing itself, and this rule is the entire
    safety argument for pod files.
    """
    pod = vs_profile or STANDARD_POD_PROFILE
    return run_id(slug, opponents, games, seed, profile,
                  pod_tag_name(pod, opponents), clock)


def default_jobs():
    """How many JVMs to run at once — PERFORMANCE CORES, not logical CPUs.

    Was `cpu_count() - 1`, which is 7 on this machine. It has 4 performance and
    4 efficiency cores, and a JVM scheduled onto an E-core runs the same game at
    roughly half the speed. Forge's `-c` is WALL time, so those slow seats hit
    the clock and their games were recorded as truncated — a measurement
    artifact of the scheduler, indistinguishable in the record from a genuinely
    stalled game.

    That is what the censoring numbers say, re-derived across every tracked run
    on 2026-09-04 rather than quoted from the two that existed when this was
    written:

        4 jobs    7 runs    408 games     14 truncated    3.4%
        7 jobs   13 runs    880 games    120 truncated   13.6%

    A FOUR-FOLD DIFFERENCE, and the decks did not change between them. The
    original wording here said the 4-JVM runs truncated 0%, which was true of
    the two runs available at the time and is no longer true — five more 4-job
    runs since have truncated between 0% and 5%. Four jobs is much better, not
    perfect, and the difference between those two claims is exactly the kind
    that goes stale silently.

    Falls back to half the logical CPUs where the split is not reported, which
    is the right shape on a homogeneous machine too: Forge is not the only thing
    running, and an oversubscribed core costs wall time on every job at once.
    """
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                             capture_output=True, text=True, check=False)
        perf = int(out.stdout.strip())
        if perf >= 1:
            return perf
    except (OSError, ValueError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


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
#: Forge's own draw verdict. THE NEGATIVE LOOKAHEAD IS THE WHOLE PATTERN: the
#: first version was `^Game Outcome: .*draw`, and Forge writes
#:
#:     Game Outcome: <SEAT> has lost trying to draw cards from empty library
#:
#: for a player who DECKS OUT — which contains "draw" and is not a draw at all.
#: Measured 2026-09-04: every one of the 14 lines matching the loose pattern
#: across every log on disk was that decking line, and ZERO were a genuine draw.
#: The consequence was not cosmetic: 12 games in one 60-game run were recorded
#: as draws while still carrying a winner, `tally_wins` counted those winners
#: anyway, and the run's own accounting came to 72 of 60 games — which is what
#: `test_every_tracked_run_accounts_for_all_its_games` fired on.
_OUTCOME_DRAW = re.compile(
    r"^Game Outcome: (?!.*\btrying to draw cards\b).*\bdraw\b", re.I)
# TWO ENDINGS, TWO FORMAT STRINGS. Forge prints
#   "Game Result: Game %d ended in %d ms."          -- a decided game
#   "Game Result: Game %d ended in a Draw! Took %d ms."  -- GameOutcome.isDraw()
# and this pattern only matched the first. `parse_outcomes` closes a game
# ONLY on this line, so a real draw left the record open, the next game's
# `Game Outcome:` lines fell through the `cur is None` guard, and TWO GAMES
# MERGED INTO ONE credited to the second one's winner — with
# `games_completed` silently one short.
# It has never fired because `isDraw()` has been false in all 901 games: the
# clock-out games Forge calls draws were being handed to a survivor instead.
# Fixing the truncation bug is exactly what would have armed this one.
_RESULT = re.compile(
    r"^Game Result: Game (\d+) ended in (?:a Draw! Took )?(\d+) ms")


def parse_outcomes(text):
    """Per-game outcomes from one JVM's log: winner, turn, ms, how each seat lost.

    A GAME THAT RAN OUT THE CLOCK HAS NO WINNER, and Forge says so in a way that
    reads like the opposite. When `-c <clock>` fires, Forge stops the game where
    it stands — mid-combat, on turn 25 — and prints a `has won because all
    opponents have lost` line for EVERY SURVIVING SEAT:

        Game Outcome: Ai(1)-mm-goblin-storm has won because all opponents have lost
        Game Outcome: Ai(2)-mm-giada-angels has won because all opponents have lost
        Game Outcome: Ai(3)-mm-vito has won because all opponents have lost
        Game Outcome: Ai(4)-mm-baylen-tokens has won because all opponents have lost

    This loop assigned `cur["winner"]` on each match, so the LAST line won —
    always the highest-numbered surviving seat. Measured on the 100-game
    goblin-storm run: 12 games hit the clock, all 12 declared more than one
    winner, and **8 of 100 winners were recorded wrong**. `baylen-tokens` was
    credited 11 wins where Forge's own `Game Result` line gives it 3; `vito`
    lost 7 it had won; and one of OUR wins was reassigned away.

    A truncated game is now `winner=None, truncated=True` and is excluded from
    the win rate rather than awarded to whoever sat last. That is the
    absent-means-absent rule: a game nobody won is not a game somebody won, and
    inventing a winner from seat order is the most misleading option available.
    """
    games, cur, last_turn = [], None, None
    for line in text.splitlines():
        m = _TURN.match(line)
        if m:
            last_turn = int(m.group(1)); continue
        if cur is None and line.startswith("Game Outcome:"):
            cur = {"winner": None, "won_by": None, "round": None, "global_turn": last_turn,
                   "draw": False, "lost": {}, "ms": None,
                   # Every seat Forge declared a winner. One is a result; more
                   # than one means the clock stopped the game.
                   "_won": [], "truncated": False}
            last_turn = None
        if cur is None:
            continue
        m = _OUTCOME_TURN.match(line)
        if m:
            cur["round"] = int(m.group(1)); continue
        m = _OUTCOME_WON.match(line)
        if m:
            cur["_won"].append((m.group(1), m.group(2).rstrip(".")))
            continue
        m = _OUTCOME_LOST.match(line)
        if m:
            cur["lost"][m.group(1)] = m.group(2).rstrip("."); continue
        if _OUTCOME_DRAW.match(line):
            cur["draw"] = True; continue
        m = _RESULT.match(line)
        if m:
            cur["ms"] = int(m.group(2))
            games.append(_settle(cur)); cur = None
    if cur is not None and (cur["_won"] or cur["draw"]):   # a final game with no Result line
        games.append(_settle(cur))
    return games


def _settle(game):
    """Turn the collected win lines into one winner, or into no winner at all."""
    won = game.pop("_won", [])
    seats = {name for name, _why in won}
    if len(seats) > 1:
        # THE CLOCK STOPPED IT. Not a draw — a draw is a result both players
        # reached — and not a win for anyone. It is a game that did not finish,
        # and the record says so rather than picking a survivor.
        game["truncated"] = True
        game["winner"] = None
        game["won_by"] = None
    elif won:
        game["winner"], game["won_by"] = won[0]
    return game


def _seat_label(seat_names):
    """Forge labels seats `Ai(k)-<meta name>`; map every label back to its slug.

    POSITION-INDEPENDENT ON PURPOSE. This used to key on `Ai({i+1})-{name}` for
    the one order the decks were passed in, which was fine while the `-d` order
    never changed. Seats now ROTATE per job — so `Ai(2)` is a different deck in
    job 1 than in job 0, and a map built from one order would attribute wins to
    whichever deck happened to sit there first. That is a far worse error than
    the seat bias the rotation exists to remove.

    The label already carries the deck name, so position never had to be part of
    the key. Every seat index is mapped for every name, and the result is correct
    under any ordering.
    """
    strip = lambda n: n[len(SIM_DECK_PREFIX):] if n.startswith(SIM_DECK_PREFIX) else n
    return {f"Ai({k})-{name}": strip(name)
            for name in seat_names
            for k in range(1, len(seat_names) + 1)}


def pod_profile(pod, slug):
    """One opponent seat's pilot. `pod` is a name for all of them, or a map.

    A pod file may give a seat its own profile — B-2's "each opponent deck
    carries an AI strategy profile" — and the map form is how that reaches here.
    A seat the map does not mention falls back to the standard, so a pod that
    sets one seat does not silently re-pilot the rest.
    """
    if isinstance(pod, dict):
        return pod.get(slug) or STANDARD_POD_PROFILE
    return pod


def _profiles_for(rotation, subject, profile, pod):
    """AI profiles in the ROTATED seat order.

    `profiles` is index-aligned with `-d`, so rotating the decks without
    rotating the profiles would hand our seat's pilot to whichever deck landed
    in slot 0 — silently swapping which deck is played by which AI, which is a
    worse bug than the one the rotation fixes.
    """
    return [(profile or DEFAULT_PROFILE) if s == subject else pod_profile(pod, s)
            for s in rotation]


def run(slug, opponents, games=SIM_DEFAULT_GAMES, jobs=None, clock=SIM_GAME_CLOCK_SECONDS,
        seed=None, force=False, dry_run=False, home=None, decks_dir=None,
        profile=None, vs_profile=None, pod_name=None):
    """Run the games and write the run record. Returns (record_path, record)."""
    if not opponents:
        raise SystemExit("simulate needs at least one opponent: --vs <slug> (repeatable)")
    seats = [slug, *opponents]
    names = [install_deck(s, decks_dir) for s in seats]
    jobs = jobs or default_jobs()
    parts = split_games(games, jobs)
    seed_base = default_seed(slug, opponents) if seed is None else int(seed)
    seeds = [seed_base + i for i in range(len(parts))]
    pod = vs_profile or STANDARD_POD_PROFILE
    rid = run_id_for(slug, opponents, games, seed_base, profile, vs_profile, clock)
    out_dir = _out_dir(slug)
    log_dir = out_dir / "logs" / rid
    record_path = out_dir / f"{rid}.json"
    if record_path.exists() and not force and not dry_run:
        raise SystemExit(f"{slug}: {record_path.name} exists — re-running it would "
                         f"OVERWRITE that measurement with a different one under the "
                         f"same name (the seed fixes the shuffle, not the games; see "
                         f"SEEDED_NOTE). Pass --seed N for a separate sample, or "
                         f"--force to replace it.")
    jar = forge_jar(home)
    # THE POD IS PART OF THE INSTRUMENT. `--profile` set only OUR seat and left
    # every opponent on Default, so the table could never be made to play
    # differently — and a win rate is relative to the pod's competence as much
    # as to the deck. `--vs-profile` sets every opponent seat.
    profiles = None
    if profile or pod_tag_name(pod, opponents) != DEFAULT_PROFILE:
        profiles = [profile or DEFAULT_PROFILE] + [pod] * len(opponents)
    # SEATS ROTATE PER JOB, and the reason is that Forge's turn order is not a
    # coin flip. `GameAction.determineFirstTurnPlayer` picks, from game 2 on, the
    # LOWEST-INDEXED SEAT THAT DID NOT WIN THE PREVIOUS GAME — and all N games of
    # a job run inside one `Match` object that carries `lastOutcome` forward.
    #
    # Two consequences, both bad. Our deck is always index 0, so it started
    # 323 of 400 games in one tracked run; and the games are a MARKOV CHAIN
    # rather than independent draws, which is what a Wilson interval assumes.
    # Every `win_rate_ci95` written before this line was not the interval it
    # claimed to be.
    #
    # Rotating the `-d` order per job spreads each deck across seat positions,
    # so position is averaged out instead of confounded with the deck. Games
    # within one job are still dependent; across jobs they are not, and with
    # seven jobs no seat is pinned. `tally_wins` already matches on the Forge
    # meta name rather than on position, so nothing downstream needs to know.
    rotations = [seats[i % len(seats):] + seats[:i % len(seats)]
                 for i in range(len(parts))]
    job_names = [[install_deck(s, decks_dir) for s in rot] for rot in rotations]
    cmds = [command(job_names[i], g, clock, jar, seed=seeds[i],
                    profiles=_profiles_for(rotations[i], slug, profile, pod)
                    if profiles else None)
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

    # ONE LABEL MAP PER JOB, because the seats rotate. `Ai(2)` is a different
    # deck in job 1 than in job 0, and a single map built from the unrotated
    # order would attribute every win to whoever happened to sit there first —
    # a far worse error than the seat bias the rotation fixes.
    label = _seat_label(names)      # position-independent; safe under rotation
    outcomes = []
    for log, rc in results:
        j = int(log.stem.split("-")[1])
        for gi, g in enumerate(parse_outcomes(log.read_text(encoding="utf-8", errors="replace"))):
            outcomes.append({
                "winner": label.get(g["winner"], g["winner"]) if g["winner"] else None,
                "won_by": g["won_by"], "draw": g["draw"], "round": g["round"],
                "truncated": g.get("truncated", False),
                "global_turn": g["global_turn"], "ms": g["ms"],
                "lost": {label.get(k, k): v for k, v in g["lost"].items()},
                "log": log.name, "seed": seeds[j], "seat_order": rotations[j],
                "game_in_job": gi + 1})
    texts = [log.read_text(encoding="utf-8", errors="replace") for log, _ in results]
    cmd_by_slug = {s: sorted(c) for s, c in _commanders_by_slug(seats).items()}
    # Every seat index for every deck, same reason as `_seat_label`: the decks
    # rotate, so a commander keyed to one position would credit commander damage
    # to whichever deck later sat there.
    commanders = {f"Ai({k})-{names[i]}": set(cmd_by_slug[s])
                  for i, s in enumerate(seats) if cmd_by_slug.get(s)
                  for k in range(1, len(seats) + 1)}
    facts, analysis = sim_parse.analyze_logs(texts, label, commanders)
    # MATCH ON THE FORGE NAME, NOT THE SEAT SLUG. A branch seat is written to
    # Forge as `ur-dragon-treasure-v2` because `@` has no business in a deck
    # registry — so the outcome names the flattened form while `seats` holds the
    # slug, and a bare `==` matched every OTHER seat and silently scored ours
    # zero. The run reported "wins 0" for a list that had won ELEVEN of a
    # hundred, with the other three seats' counts all correct beside it, which is
    # exactly the shape that gets believed.
    # TALLIED OVER DECIDED GAMES ONLY. `outcomes` still carries every game,
    # truncated ones included, so nothing is hidden from the record.
    # A TRUNCATED GAME IS NOT A DRAW AND NOT A WIN — it is a game that did not
    # finish, and it is counted separately so the win rate has an honest
    # denominator. Forge calls it a draw internally; from the bench's side the
    # useful fact is "the clock stopped it", because that is a property of the
    # HARNESS (the `-c` setting) rather than of the game.
    truncated = sum(1 for o in outcomes if o.get("truncated"))
    decided = [o for o in outcomes if not o.get("truncated")]
    draws = sum(1 for o in outcomes if o["draw"] and not o.get("truncated"))
    wins = tally_wins(decided, seats)
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
        # THE DENOMINATOR IS DECIDED GAMES, and `truncated` is beside it so the
        # reader can see how many were thrown out. Dividing by every game played
        # counts an unfinished one as a loss for whoever was still alive in it —
        # which for our seat, always `Ai(1)`, was all 93 of them.
        "summary": {"wins": wins, "draws": draws, "truncated": truncated,
                    "decided": len(decided),
                    "win_rate": (round(wins[slug] / len(decided), 3) if decided else None),
                    "mean_round": (round(sum(o["round"] or 0 for o in outcomes) / len(outcomes), 1)
                                   if outcomes else None),
                    "mean_global_turn": (round(sum(o["global_turn"] or 0 for o in outcomes) / len(outcomes), 1)
                                         if outcomes and all(o["global_turn"] for o in outcomes) else None)},
        # THE TABLE, NAMED. A run faces one pod, and until pods existed the only
        # record of which was the opponent slugs in the run id — so "report by
        # pod composition" (PRD §6 B-2) had nothing to group on. `named` says
        # whether the pilot passed `--pod` (a fact about the run) or the pod was
        # inferred from the opponents against today's files (a reading).
        "pod": _pods.record_for(pod_name, opponents),
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
    # THE POD, BACKFILLED — and marked `named: false`, because a record written
    # before pods existed faced a table nobody named and this is a reading of
    # today's files rather than a fact about the run. A stamp made by `--pod`
    # already carries `named: true` and is left alone.
    if not (rec.get("pod") or {}).get("named"):
        pod = _pods.record_for(None, [s["slug"] for s in rec["seats"][1:]])
        if pod:
            rec["pod"] = pod
    # THE ASSUMPTIONS ARE A PROPERTY OF THE HARNESS, NOT OF THE RUN, so a
    # re-derive refreshes them. They were not, and that left a claim the harness
    # had been PROVEN NOT TO MEET sitting in a freshly written record: every run
    # made before 2026-09-02 stored "identical inputs reproduce the logs byte
    # for byte", which a same-seed double run falsified. `--analyze` updated
    # `analysis.limits` beside it and the two then disagreed inside one file.
    #
    # SEEDED vs SAMPLED is preserved, because THAT is a property of the run:
    # a record made before `-s` existed is not seeded and never becomes so.
    was_sampled = any(a.startswith("SAMPLED") for a in (rec.get("assumptions") or []))
    rec["assumptions"] = [SAMPLED_NOTE if (was_sampled and a is SEEDED_NOTE) else a
                          for a in ASSUMPTIONS]
    # SUMMARY IS RE-DERIVED TOO, or `--analyze` cannot repair a wrong headline.
    # It did not, and the run that scored a branch seat zero kept saying zero
    # through a re-derivation that had the right numbers in `analysis` all along.
    seats = [x["slug"] for x in rec["seats"]]
    # `tally_wins`, NOT A THIRD INLINE COPY. Its own docstring records that this
    # expression "lived inline, and the test guarding it built its own correct
    # copy — so a regression left the test green while the bug reappeared". It
    # had grown back here, and that is exactly why `--analyze` could not repair
    # the truncation bug: `run()` and `analyze()` tallied by different code.
    outcomes = [g for g in rec["games"] if g.get("winner") or g.get("draw")
                or g.get("truncated")]
    truncated = sum(1 for g in outcomes if g.get("truncated"))
    decided = [g for g in outcomes if not g.get("truncated")]
    wins = tally_wins(decided, seats)
    n = len(decided) or None
    rec["summary"] = dict(rec.get("summary") or {}, wins=wins,
                          truncated=truncated, decided=len(decided),
                          draws=sum(1 for g in outcomes
                                    if g.get("draw") and not g.get("truncated")),
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


def resolve_table(args):
    """The opponents and their pilots, from `--vs` flags or a named `--pod`.

    ONE PLACE, because `simulate` and `experiment` must not disagree about what
    a table is — they already disagreed about the pod's AI profile, and that made
    a controlled A/B controlled against the wrong thing for as long as both
    commands existed.

    A pod expands to the SAME ordered slugs the flags would have given, so the
    run id is unchanged and `--pod standard` is a spelling rather than a new
    measurement. Mixing the two is refused: two sources for one list is how a
    seat goes missing silently.
    """
    from manamap.sim import pods

    vs = list(getattr(args, "vs", None) or [])
    name = getattr(args, "pod", None)
    if vs and name:
        raise SystemExit(
            f"--pod {name} and --vs together — pick one. A pod IS the list of "
            f"--vs flags; `manamap pilot pods {name}` prints the ones it expands "
            f"to if you want to start from them.")
    if not name:
        return vs, None
    try:
        return pods.seats(name), pods.profiles(name)
    except pods.PodError as exc:
        raise SystemExit(str(exc))


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
    # THE TABLE IS RESOLVED BEFORE THE LIST BRANCH, not after. This read
    # `args.vs` directly, so `--pod standard` looked like "no opponents given"
    # and silently printed the run list instead of running anything — a flag
    # that appears to do nothing is worse than one that errors.
    opponents, seat_profiles = resolve_table(args)
    if getattr(args, "list", False) or not opponents:
        runs = list_runs(slug)
        if not runs:
            print(f"{slug}: no simulation runs — `manamap pilot simulate {slug} "
                  f"--pod standard --games N` (or --vs <opponent>, repeatable)")
            return
        print(f"SIMULATION RUNS — {slug} ({len(runs)})\n")
        for r in runs:
            s = r["summary"]
            me = (r.get("analysis") or {}).get("seats", {}).get(slug, {})
            ci = me.get("win_rate_ci95")
            print(f"{r['run_id']}  {r['at']}  {r['games_completed']}/{r['games_requested']} games  "
                  f"win {s['win_rate']}{' ci95 ' + str(ci) if ci else ''}  mean round {s['mean_round']}  "
                  f"{r['wall_seconds']}s on {r['jobs']} JVM(s)")
            pod = r.get("pod")
            table = ", ".join(x["slug"] for x in r["seats"][1:])
            if pod:
                # THE NULL, BESIDE THE RATE. A four-player win rate reads
                # against 0.25 unless something says otherwise, and nothing did:
                # measured over the tracked runs, `standard` gives one seat 0.572
                # and another 0.052, and the subject chair 0.159. A deck scoring
                # 0.16 there is AT the table's typical subject rate, not two
                # thirds below a quarter.
                cal = _pods.calibration(pod["name"])
                null = (cal.get("subject_null") or {}).get("rate")
                if null is not None:
                    print(f"      null {null} — what our decks score in seat 0 "
                          f"on this table ({cal['runs']} runs, {cal['games']} "
                          f"games); `pods {pod['name']} --calibration`")
                # `~` for an inferred pod: the record predates pods and this is
                # today's files read against it, not what was configured.
                mark = "" if pod.get("named") else "~"
                table = f"{mark}{pod['name']} ({pod['players']}p) — {table}"
            print(f"      vs {table}  ·  wins {s['wins']}")
        return
    path, rec = run(slug, opponents, games=args.games or SIM_DEFAULT_GAMES, jobs=args.jobs,
                    clock=args.clock or SIM_GAME_CLOCK_SECONDS, seed=getattr(args, "seed", None),
                    force=getattr(args, "force", False), dry_run=getattr(args, "dry_run", False),
                    profile=getattr(args, "profile", None),
                    pod_name=getattr(args, "pod", None),
                    vs_profile=getattr(args, "vs_profile", None) or seat_profiles)
    if getattr(args, "dry_run", False):
        print(f"would run {rec['games_per_job']} games across {rec['jobs']} JVM(s), seeds "
              f"{rec['seeds']} → {path}")
        for c in rec["commands"]:
            print("  " + " ".join(c))
        return
    s = rec["summary"]
    print(f"{slug}: {rec['games_completed']}/{rec['games_requested']} games vs "
          f"{', '.join(opponents)} in {rec['wall_seconds']}s on {rec['jobs']} JVM(s)")
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
