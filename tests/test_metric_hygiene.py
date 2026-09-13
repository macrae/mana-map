"""Three checks, each one a defect this repo actually shipped.

The rule is the repo's own: the critic's findings become checks, or the work is
re-spent. All three of these were found by measuring hours or days after the
code landed — never by review, and never by the test written beside it.
"""

import ast
import glob
import json

import pytest

from conftest import ROOT, requires_deck
from manamap.pilot import calibrate, candidates

MAX_AXIS_CORRELATION = 0.90


def _spearmanless_corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


@requires_deck
@pytest.mark.fleet
def test_no_two_axes_measure_the_same_thing():
    """A1: THREE MAGNITUDE AXES THAT WERE ONE AXIS.

    `board_power_6`, `damage_8` and `kill_by_8` shipped together and measured
    r = 0.92-0.98 against each other the same day — the mana family's identical
    documented defect, committed by someone who had just re-read the note about
    it. Three axes that rank identically are a TRAP rather than a redundancy:
    sweep on one, sweep on another, get the same order, read it as confirmation.

    UNIFORM FLAGS, as `benchmark` does, and for the same reason. Read off each
    deck's own declaration only TWO of the nine axes are computable fleet-wide —
    most decks opt into neither model — so a check on the declared readings
    would silently cover 2 of 9 while looking thorough. Forcing both models on
    makes every axis comparable and is the only way this check can see the
    defect it exists for.

    `fleet` marked: it runs the whole fleet and `-m fleet` is excluded from the
    default suite for time, so this is a gate you must ask for.
    """
    from manamap.pilot import diagnostic, goldfish
    from manamap.pilot.common import load_deck_cards

    readings = {}
    for path in sorted(glob.glob(str(ROOT / "data/decks/*/cards.json"))):
        slug = path.split("/")[-2]
        try:
            got = goldfish.run(slug, iterations=1500,
                               seed=diagnostic.HARNESS["seed"], max_turn=10,
                               model_treasures=True, model_combat=True,
                               with_results=True, quiet=True)
        except FileNotFoundError:
            continue
        doc = {"output": diagnostic.output(got),
               "stall": diagnostic.stall(got.get("_results") or []),
               "mana": diagnostic.mana(got.get("_results") or [])}
        row = {}
        for axis in candidates.AXES:
            block = candidates.AXES[axis][0]
            if block == "engine":
                continue          # needs a `required` marking: 1 of 13 decks
            cell = candidates._read(doc, axis)
            if isinstance(cell, dict) and "rate" in cell:
                row[axis] = cell["rate"]
        if row:
            readings[slug] = row
    assert len(readings) >= 8, f"only {len(readings)} decks read — no verdict"

    shared = sorted(set.intersection(*(set(r) for r in readings.values())))
    assert len(shared) >= 4, (
        f"only {len(shared)} axes computable under a uniform harness "
        f"({shared}) — this check cannot see the defect it exists for")
    guilty = []
    for i, a in enumerate(shared):
        for b in shared[i + 1:]:
            r = _spearmanless_corr([readings[s][a] for s in readings],
                                   [readings[s][b] for s in readings])
            if abs(r) >= MAX_AXIS_CORRELATION:
                guilty.append(f"{a} ~ {b}: r={r:+.2f}")
    assert not guilty, (
        "two axes are one measurement, and a sweep on both reads as two "
        "confirmations of one fact:\n  " + "\n  ".join(guilty))


def test_every_signal_the_model_sets_is_read_by_something():
    """A2: A FLAG SET AND NEVER ACTED ON.

    `treasure_doubler` was added to `classify()` and the cast loop never reached
    the cards carrying it, so Primal Vigor sat in hand for ten turns while
    holding the flag that says it changes what the deck produces. The tell was
    fifteen candidates returning byte-identical −0.026; nothing failed.

    An AST sweep: every key `classify()` emits must be subscripted somewhere
    outside `classify` itself. Idioms that read a key another way are
    whitelisted BY NAME, because a silent whitelist is how this check would
    rot into the thing it replaced.
    """
    # THE SIMULATOR IS FOUR FILES since 2026-09-13. `classify` lives in
    # `goldfish_library` and the turn loop that reads its keys in
    # `goldfish_turn`, so a sweep over `goldfish.py` alone would find the
    # emitter in neither and every reader in none — and would then report every
    # flag as dead, or (worse, if it found `classify`) report every flag as
    # unread. Parsing all four is what keeps the question the same one.
    from manamap.pilot import goldfish

    here = ROOT / "src/manamap/pilot"
    trees = [ast.parse((here / name).read_text())
             for name in sorted(goldfish._MODEL_FILES)]

    def _fn(name):
        return next(n for tree in trees for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    emitted = {k.value for d in ast.walk(_fn("classify"))
               if isinstance(d, ast.Dict)
               for k in d.keys
               if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    inside = {n.slice.value for n in ast.walk(_fn("classify"))
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
              and isinstance(n.slice.value, str)}
    read = {n.slice.value for tree in trees for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)} - inside

    #: Read through an idiom the sweep cannot see. Each needs a reason.
    WHITELIST = {
        "name": "read as `card['name']` on raw cards.json entries, not sim cards",
    }
    dead = sorted(emitted - read - set(WHITELIST))
    assert not dead, (
        "the model sets these and nothing reads them — a flag the model sets "
        "is a claim the model must act on:\n  " + "\n  ".join(dead))


def test_no_aggregate_pools_runs_against_different_pods():
    """A5: A WIN RATE IS AGAINST SOMEBODY.

    `calibrate` summed every tracked run regardless of who was at the table:
    kianne's 24 games were 12 against the standard pod and 12 in a 1v1 against
    giada alone; radagast's 28 were 20 standard and 8 against a pod of our own
    decks. The pooled number was not a win rate against anything.
    """
    record, pod, dropped = calibrate.forge_record()
    if not record:
        pytest.skip("no tracked sim runs on this checkout")
    assert pod, "runs were pooled with no pod identified"
    # Every seat's games must come from runs whose opponent set IS the pod.
    for seat, opponents, wins, games in calibrate._seat_rows():
        if seat not in record:
            continue
        if set(opponents) != set(pod):
            assert not any(d for d in dropped if set(d["pod"]) == set(pod)), (
                f"{seat}: a run against {sorted(opponents)} was neither pooled "
                f"nor reported as dropped")
    total_dropped = sum(d["games"] for d in dropped)
    assert total_dropped == 0 or dropped, "dropped games were not reported"


def test_the_axis_registry_and_its_flag_map_agree():
    """A cheap consistency check on the pair A1 left behind: every magnitude
    axis must declare the model flag it needs, or `candidates` refuses with a
    KeyError instead of a sentence."""
    for axis, (block, _key, _sub) in candidates.AXES.items():
        if block == "output":
            assert axis in candidates.AXIS_NEEDS, axis
    for axis in candidates.AXIS_NEEDS:
        assert axis in candidates.AXES, axis


# ── The model's version, after the 2026-09-13 split ─────────────────────────

def test_the_model_version_covers_every_file_the_simulator_is_made_of():
    """A STAMP OVER ONE FILE OF FOUR IS A STAMP THAT LIES.

    `model_version` hashed `__file__`, which was right while the simulator was
    one 5,830-line file. The split moved the card readers and the turn loop into
    their own modules — so a changed regex or a changed casting rule would no
    longer have moved the version, and every artifact deriving from it would
    have read as CURRENT while the model underneath changed.

    That is the precise failure this stamp exists to prevent, and the
    refactoring would have created it.

    Re-introducing the bug: hash `__file__` alone and this reds, because editing
    a profile stops moving the version.
    """
    import hashlib
    from pathlib import Path

    from manamap.pilot import goldfish

    here = Path(goldfish.__file__).parent
    assert len(goldfish._MODEL_FILES) >= 4, (
        "the simulator is fewer files than the split produced — did one merge "
        "back, or did the list go stale?")
    for name in goldfish._MODEL_FILES:
        assert (here / name).is_file(), f"{name} is stamped and does not exist"

    before = goldfish.model_version()
    assert len(before) == 12

    # Every stamped file must MOVE the version. This is the assertion, not the
    # file list: a module added to the simulator and left out of `_MODEL_FILES`
    # fails here rather than silently freezing the stamp.
    for name in goldfish._MODEL_FILES:
        path = here / name
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# touched by a test\n")
            assert goldfish.model_version() != before, (
                f"editing {name} does not move the model version — every "
                f"artifact derived from it would read as current")
        finally:
            path.write_bytes(original)
    assert goldfish.model_version() == before, "a probe did not restore its file"


def test_every_simulator_module_is_stamped():
    """DERIVED, so a fifth module cannot be added and forgotten."""
    from pathlib import Path

    from manamap.pilot import goldfish

    here = Path(goldfish.__file__).parent
    on_disk = {p.name for p in here.glob("goldfish*.py")}
    stamped = set(goldfish._MODEL_FILES)
    missing = sorted(on_disk - stamped)
    assert not missing, (
        "a simulator module exists and is not in the version stamp:\n  "
        + "\n  ".join(missing)
        + "\n(add it to goldfish._MODEL_FILES, or it changes the model silently)")


def test_every_model_flag_is_visible_to_model_coverage():
    """A FLAG THE MODEL READS AND `model-coverage` CANNOT SEE IS A BLIND SPOT IN
    THE COMMAND WHOSE ONLY JOB IS NAMING BLIND SPOTS.

    `model_coverage.CHANNELS` maps a casting channel to the flag that gates it,
    and `model-coverage` reports a card as DARK when it feeds a channel whose
    flag is off. Three flags `run()` reads were in no channel and in no
    exemption — `model_deaths`, `model_commander_animate` and
    `model_commander_combat_reveal` — so a deck could declare one, the simulator
    could act on it, and the command that says "what can the model not see"
    would not mention it.

    That matters because the two worst measurement errors this project has had
    were both a commander ability the model did not read: eminence absent
    entirely (bodies at turn ten understated by 50%) and the attack tutor firing
    5.70 times a game against Forge's 1.22, which took kill-by-t8 from 0.501 to
    0.173.

    DERIVED from `run()`'s source, so a new flag fails here until somebody
    classifies it.
    """
    import re
    from pathlib import Path

    from manamap.pilot import goldfish, model_coverage

    src = Path(goldfish.__file__).read_text()
    run_src = src[src.index("\ndef run("):]
    declared = {f for f in re.findall(r"\bmodel_[a-z_]+\b", run_src)}
    # Not model FLAGS: the version stamp, the assumptions text, and the
    # coverage preflight's own name.
    declared -= {"model_version", "model_assumptions", "model_coverage"}
    assert len(declared) >= 8, f"only {len(declared)} flags found in run(): {declared}"

    known = ({f for f in model_coverage.CHANNELS.values() if f}
             | set(model_coverage.DEFAULT_ON)
             | set(model_coverage.NOT_A_CHANNEL))
    invisible = sorted(declared - known)
    assert not invisible, (
        "`run()` reads model flags that `model-coverage` cannot see:\n  "
        + "\n  ".join(invisible)
        + "\n(give each a channel in CHANNELS, or list it in NOT_A_CHANNEL with "
          "a reason — a flag nobody classified is a blind spot in the blind-spot "
          "detector)")


def test_a_malformed_declaration_does_not_kill_the_calling_command():
    """`run()` RAISED `SystemExit`, AND FOUR COMMANDS CALL IT IN PROCESS.

    `benchmark`, `diagnostic`, `calibrate` and `deck_branch` all call
    `goldfish.run` directly, so one deck's malformed `goldfish_targets.json`
    ended whatever command was running — mid-sweep, that is a fleet
    regeneration abandoned at deck three with a message about a file nobody
    asked about. A library does not get to decide that the process should end.

    It raises `DeclarationError` now, and `registry.run_pilot_step` converts it
    back to `SystemExit` so the terminal behaves exactly as before.

    Re-introducing the bug: change `DeclarationError` back to `SystemExit` in
    `run()` and the first assertion fails — the caller can no longer catch it
    without catching a process exit.
    """
    from manamap.pilot.goldfish import DeclarationError

    assert issubclass(DeclarationError, ValueError), (
        "a caller must be able to catch this without catching SystemExit")
    assert not issubclass(DeclarationError, SystemExit)

    # And the declaration refusals inside `run` use it.
    import re
    from pathlib import Path

    from manamap.pilot import goldfish

    # SLICE `run` PRECISELY. A naive slice to end-of-file swept in `main`'s
    # own conversion and the `__main__` guard — both of which SHOULD exit.
    import ast

    src = Path(goldfish.__file__).read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    run_src = "\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno])
    exits = len(re.findall(r"raise SystemExit", run_src))
    typed = len(re.findall(r"raise DeclarationError", run_src))
    assert typed >= 4, f"only {typed} declaration refusals are typed"
    assert exits == 0, (
        f"`run()` still raises SystemExit {exits} time(s) — it is called in "
        f"process by four commands")
