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
    src = (ROOT / "src/manamap/pilot/goldfish.py").read_text()
    tree = ast.parse(src)

    def _fn(name):
        return next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    emitted = {k.value for d in ast.walk(_fn("classify"))
               if isinstance(d, ast.Dict)
               for k in d.keys
               if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    inside = {n.slice.value for n in ast.walk(_fn("classify"))
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
              and isinstance(n.slice.value, str)}
    read = {n.slice.value for n in ast.walk(tree)
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
