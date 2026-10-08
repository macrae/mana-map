"""Forge scenario slices (PRD v2 Step 5): one board, a few turns, many seeds, one JVM.

A whole Forge game asks "who wins this pod", which is a 400-game question (an A/A at
200/arm reads ±0.09). A SLICE asks "on THIS board, what happens next": the board is
applied at the first priority, every seat is the Forge AI, and play stops when the turn
after the current turn plus `rounds` full rounds begins. The answer is the board it left
behind, per seed.

THE RUNNER is `data/forge_driver/ScenarioSlice.java`, built into its OWN small jar
(`mm-scenario-slice.jar`, beside Forge's) and put on the classpath next to whichever
Forge jar a pod run would use (`telemetry.jar_for_run`: our log formatter and AI patches
included). It is deliberately NOT one of `data/forge_patches/`: that set is the
`-tl<sha8>` fingerprint every pod run records, and a class that never executes in a
`sim` game must not split every future run from its history.

The jar carries the sha of the source it was built from (`mm/source.sha`); `ensure_built`
rebuilds when the tracked source moved, so a stale runner cannot answer.

THE SPIKE'S TRAP, kept here because it is the reason the runner exists rather than a
`sim` flag: `GameState.applyToGame` queues onto Forge's game-thread executor when called
from a sim thread and the state HALF-APPLIES (zones cleared, never filled). The runner
calls the protected `applyGameOnThread` inside `Match.startGame`'s hook instead.

The state text is Forge's puzzle `[state]` block — `p0life=40`, `p0hand=A;B`,
`p1battlefield=…`, `activeplayer=p0`, `activephase=MAIN1`, `turn=6` — one `pN` per seat,
in the same order as `decks`. Starting mid-combat is 1-vs-1 only (Forge's limit).
"""

import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from manamap import config

SOURCE = config.DATA_DIR / "forge_driver" / "ScenarioSlice.java"
JAR_NAME = "mm-scenario-slice.jar"
MAIN = "mm.ScenarioSlice"
SHA_ENTRY = "mm/source.sha"
MARK = "MMSLICE "
DEFAULT_TIMEOUT_S = 120


def source_sha():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]


def jar_path(home=None):
    return Path(home or config.FORGE_HOME) / JAR_NAME


def built_sha(home=None):
    """The source sha the installed runner was built from, or None."""
    p = jar_path(home)
    if not p.is_file():
        return None
    try:
        with zipfile.ZipFile(p) as zf:
            return zf.read(SHA_ENTRY).decode().strip()
    except (KeyError, zipfile.BadZipFile):
        return None


def build(home=None, javac="javac"):
    """Compile the runner against the pristine Forge jar into its own small jar."""
    from manamap.sim import forge

    pristine = forge.forge_jar(home)
    out = jar_path(home)
    with tempfile.TemporaryDirectory() as work:
        subprocess.run([javac, "--release", "21", "-cp", str(pristine), "-d", work, str(SOURCE)],
                       check=True, capture_output=True, text=True)
        (Path(work) / SHA_ENTRY).write_text(source_sha() + "\n")
        tmp = out.with_suffix(".jar.building")
        if tmp.exists():
            tmp.unlink()
        subprocess.run(["jar", "cf", str(tmp), "-C", work, "."],
                       check=True, capture_output=True, text=True)
        tmp.replace(out)
    return out


def ensure_built(home=None):
    """The runner jar, rebuilt when it is missing or built from another source."""
    if built_sha(home) != source_sha():
        build(home)
    return jar_path(home)


def command(decks, cases, rounds=1, timeout=DEFAULT_TIMEOUT_S, jar=None, home=None):
    """The exact argv. `decks` are Forge deck FILE names (`<meta>.dck`), `cases` are
    `(label, seed, path)` triples — ONE BOARD PER (ARM, SEED), because the hidden cards
    are dealt per seed. A pure function given its jars, so a test can read it."""
    from manamap.sim import telemetry

    base = jar or telemetry.jar_for_run(home)[0]
    argv = ["java", *config.FORGE_JVM_ARGS, "-cp", f"{base}:{jar_path(home)}", MAIN,
            "--decks", ",".join(decks),
            "--rounds", str(int(rounds)), "--timeout", str(int(timeout))]
    for label, seed, path in cases:
        argv += ["--case", f"{label}:{int(seed)}={path}"]
    return argv


def parse_output(text):
    """The replicate records out of everything the JVM printed."""
    out = []
    for line in text.splitlines():
        if line.startswith(MARK):
            out.append(json.loads(line[len(MARK):]))
    return out


def run(seats, cases, rounds=1, timeout=DEFAULT_TIMEOUT_S, home=None):
    """Play every case. `seats` are bench slugs (decks or opponents) in seat order p0, p1,
    …; `cases` are `(label, seed, state_text)` — one board per arm and seed, as
    `slice_state.to_forge_state` deals it. Returns the replicate records in case order,
    each `{label, seed, start, end, stopped, winner, start_turn, stop_turn,
    ended_at_turn, elapsed_ms, log}` or `{label, seed, error}`."""
    from manamap.sim import forge

    cases = list(cases)
    if not cases:
        raise ValueError("no case to play")
    ensure_built(home)
    decks = [forge.install_deck(s) + ".dck" for s in seats]
    home_dir = Path(home or config.FORGE_HOME)
    with tempfile.TemporaryDirectory() as tmp:
        triples = []
        for i, (label, seed, text) in enumerate(cases):
            p = Path(tmp) / f"{i:04d}.state"
            p.write_text(text, encoding="utf-8")
            triples.append((label, seed, p))
        argv = command(decks, triples, rounds=rounds, timeout=timeout, home=home)
        budget = timeout * len(triples) + 120                 # boot plus every replicate
        got = subprocess.run(argv, cwd=home_dir, capture_output=True, text=True, timeout=budget)
    records = parse_output(got.stdout)
    fatal = [r for r in records if "label" not in r and r.get("error")]
    if fatal or not records:
        why = fatal[0]["error"] if fatal else (got.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"the slice runner failed: {why}")
    return records
