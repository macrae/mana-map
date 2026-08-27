"""Every deterministic deck artifact must equal a fresh recomputation.

`test_pilot_manual_freshness` covers the rendered deck page. This covers the layer
underneath it — the artifacts the page quotes figures from.

The gap this closes is specific. `goldfish_metrics.json` and
`mana_analysis.json` stamp the decklist they were built from, so a decklist edit
is detectable and there are already staleness tests for it. **`bracket_report.json`
stamps nothing at all** — `deck_audit.freshness` reports `current: null` for it
for exactly that reason — so a bracket floor could silently outlive its deck.
And no stamp of any kind catches the other direction: a change to `bracket.py`,
`goldfish.py` or `mana_analysis.py` leaves every stamp valid while changing what
the artifact should say. Four published manuals were stale for days on precisely
that failure mode, and the stamps were all green throughout.

Recomputation is the only check that catches both. It is deterministic and free.
"""

import json
import shutil

import pytest

from manamap.config import (CARD_ROLES_PATH, COMBO_DETAILS_PATH, DECKS_DIR,
                            OUTPUT_CSV_PATH)
from manamap.pilot import (
    bracket, deck_info, diagnostic, goldfish, mana_analysis, net_change)

from conftest import SRC, requires_deck, requires_data

# What each producer can possibly read: the whole pilot source tree, `config.py`,
# and the deck's own directory. Naming a producer's exact imports by hand would
# invalidate less often and could be WRONG — a missed transitive edge does not
# fail here, it serves a stale pass. See `conftest._digest`.
#
# This is the WHOLE package, and it is deliberately not a hand-traced list. The
# previous version named `pilot/` + `config.py` and asserted in this comment that
# they were "a COMPLETE closure … checked rather than assumed: no module under
# `src/manamap/pilot/` imports from anywhere in `manamap` except `manamap.pilot`
# and `manamap.config`". That was false when written or became false after, in
# NINE modules across three subpackages: `deck_info` imports `manamap.sim`,
# `manabase`/`card_pool`/`deck_facts`/`pool_facts`/`build_deck`/`validate_build`
# import `manamap.analysis.common`, and `fetch_deck`/`deck_facts` plus the three
# DB builders import `manamap.ingest`. So a change to `sim/`, `analysis/` or
# `ingest/` left these tests served from a stale cache — silently passing, which
# is the exact failure the paragraph above warns about.
#
# Naming the package costs some cache hits and cannot be wrong. A missed edge does
# not fail here, it serves a stale pass, so conservative is the only safe side.
CODE = (SRC,)


def _slugs(artifact):
    """Every place this artifact is tracked — DECKS AND THEIR BRANCHES.

    THE BRANCH TREE WAS INVISIBLE TO EVERY GATE. Nine tests globbed
    `DECKS_DIR.iterdir()` at top level and none reached `branches/`, so the ten
    tracked files under `ur-dragon/branches/treasure-v2/` were validated by
    nothing and freshness-gated by nothing — including the one artifact this
    repo has already caught being measured against the WRONG DECKLIST
    (`goldfish.main` read the champion and wrote to the branch, understating the
    turn-10 hoard by a factor of four).

    Returns `(slug, branch)` pairs; `branch` is None for the deck itself.
    """
    if not DECKS_DIR.is_dir():
        return []
    out = []
    for d in sorted(DECKS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if (d / artifact).exists():
            out.append((d.name, None))
        for b in sorted((d / "branches").glob("*")):
            if b.is_dir() and (b / artifact).exists():
                out.append((d.name, b.name))
    return out


def _id(target):
    slug, branch = target
    return f"{slug}@{branch}" if branch else slug


def _roundtrip(target, artifact, rerun, tmp_path):
    """Recompute in place, compare to the tracked copy, restore either way."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    path = root / artifact
    backup = tmp_path / artifact
    shutil.copy2(path, backup)
    try:
        rerun()
        fresh = json.loads(path.read_text())
    finally:
        shutil.copy2(backup, path)
    return fresh, json.loads(backup.read_text())


@requires_deck
@requires_data
@pytest.mark.parametrize("target", _slugs("bracket_report.json"), ids=_id)
def test_bracket_report_matches_a_fresh_run(target, tmp_path, unchanged):
    """The one artifact with no stamp of its own.

    `--target` adds `target`/`within_target`/`cut_candidates`, so the rerun has
    to pass back whatever the tracked copy recorded — otherwise a report built
    with a target looks stale against a rerun without one, which is a false
    alarm rather than a finding.
    """
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    # Bracket also reads three global artifacts outside the deck directory.
    unchanged(*CODE, root, OUTPUT_CSV_PATH, CARD_ROLES_PATH,
              COMBO_DETAILS_PATH)
    tracked = json.loads((root / "bracket_report.json").read_text())
    # NOT `target` — that is this test's parametrize argument now, and shadowing
    # it handed `_roundtrip` an int. Two meanings of one word in one scope.
    bracket_target = tracked.get("target")

    def rerun():
        bracket.main(type("Args", (), {"slug": slug, "branch": branch,
                                       "target": bracket_target,
                                       "as_json": False})())

    fresh, old = _roundtrip(target, "bracket_report.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/bracket_report.json is stale — rerun "
        f"`manamap pilot bracket-check {slug}"
        f"{f' --target {bracket_target}' if bracket_target else ''}` and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("goldfish_metrics.json"), ids=_id)
def test_goldfish_metrics_match_a_fresh_run(target, tmp_path, unchanged):
    """Seeded and deterministic, so a difference is a real change in the model
    or in the deck — never noise.

    Ten thousand games per deck, ninety thousand across the fleet, 20.5 s every
    run to re-derive nine files that move only when a decklist or the simulator
    does. That determinism is exactly what makes it safe to cache.
    """
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root)

    def rerun():
        goldfish.main(type("Args", (), {"slug": slug, "branch": branch})())

    fresh, old = _roundtrip(target, "goldfish_metrics.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/goldfish_metrics.json is stale — rerun "
        f"`manamap pilot goldfish {slug}` and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("mana_analysis.json"), ids=_id)
def test_mana_analysis_matches_a_fresh_run(target, tmp_path, unchanged):
    """Run AFTER goldfish in the real pipeline — it embeds goldfish figures —
    but the tracked copies are consistent, so order does not matter here."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root)

    def rerun():
        mana_analysis.main(type("Args", (), {"slug": slug, "branch": branch})())

    fresh, old = _roundtrip(target, "mana_analysis.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/mana_analysis.json is stale — rerun "
        f"`manamap pilot mana-analysis {slug}` and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("net_change.json"), ids=_id)
def test_net_change_matches_a_fresh_run(target, tmp_path, unchanged):
    """The document a purchase rests on. Deterministic under a fixed seed, so a
    difference is a real change in the model or in either list — and a report
    that no longer describes the lists it compares is worse than none, because
    it was already acted on."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root, DECKS_DIR / slug)

    def rerun():
        net_change.main(type("Args", (), {
            "slug": slug, "branch": branch, "write": True, "as_json": False,
            "json": False, "iterations": None, "seed": None})())

    fresh, old = _roundtrip(target, "net_change.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/net_change.json is stale — rerun `manamap pilot "
        f"net-change {slug} --branch {branch} --write` and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("diagnostic.json"), ids=_id)
def test_diagnostic_matches_a_fresh_run(target, tmp_path, unchanged):
    """The vitals. Seeded and deterministic like the goldfish it composes, so a
    difference is a real change in the model or the deck — never noise.

    It was TRACKED and gated by nothing: no validator, no freshness test, no
    `deck_status` row. Composed from the goldfish, so it goes stale on every
    model change — the artifact whose staleness was least visible.
    """
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root)

    def rerun():
        diagnostic.main(type("Args", (), {
            "slug": slug, "branch": branch, "write": True, "as_json": False,
            "iterations": None, "seed": None, "vs": None})())

    fresh, old = _roundtrip(target, "diagnostic.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/diagnostic.json is stale — rerun `manamap pilot "
        f"diagnose {slug}" + (f" --branch {branch}" if branch else "")
        + " --write` and commit it.")


@requires_data
@requires_deck
@pytest.mark.parametrize("target", _slugs("info.json"), ids=_id)
def test_info_json_matches_a_fresh_run(target, tmp_path, unchanged):
    """`info.json` is what the deck page fetches, and it is the only COMMITTED
    artifact composed from every other one — status, bracket, goldfish, engine,
    audit, diagnosis, sim, experiments, prescriptions and the derived `next`.

    That breadth is exactly why it needs this gate: it goes stale when ANY of its
    inputs move, and it stamps nothing. `deck-info` was "never committed" precisely
    to avoid this problem; committing it is what makes the deck page possible, and
    recomputation is the price.

    The version block is absent by construction (`deck_info.fetchable`), so this test
    cannot fail on a git walk that a committed copy could never keep up with.
    """
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root, OUTPUT_CSV_PATH, CARD_ROLES_PATH,
              COMBO_DETAILS_PATH)

    def rerun():
        deck_info.main(type("Args", (), {"slug": slug, "branch": branch, "write": True})())

    fresh, old = _roundtrip(target, "info.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/info.json is stale — rerun `manamap pilot deck-info {slug} --write` "
        f"and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("info.json"), ids=_id)
def test_info_json_never_carries_a_version_block(target):
    """A committed version number is one commit behind FOREVER — the commit that
    changes `decklist.txt` gets its sha after anything written in the same commit.
    A wrong version is worse than an absent one, because the captain's log stamps
    games against it. The page reads a deploy-time `versions.json` instead."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    path = DECKS_DIR / slug / "info.json"
    if not path.exists():
        pytest.skip(f"{slug} has no info.json yet")
    doc = json.loads(path.read_text())
    assert "version" not in doc, "versions cannot be committed accurately"
    assert "_note" in doc and "one commit behind" in doc["_note"]


@requires_deck
@pytest.mark.parametrize("target", _slugs("benchmark.json"), ids=_id)
def test_benchmark_matches_a_fresh_run(target):
    """`benchmark.json` is tracked, so the workbench can read it on a static
    host — and it is deterministic (fixed seed, fixed iterations, uniform
    flags), so it must equal what a fresh run produces. A stale benchmark is a
    ranking computed against a deck that no longer exists.
    """
    # `benchmark` has no branch concept — it freezes ONE harness so decks are
    # comparable, and a branch is not a deck. `_slugs` still yields the tuple.
    slug, _branch = target
    import io as _io
    import contextlib

    from manamap.config import DECKS_DIR
    from manamap.pilot import benchmark

    path = DECKS_DIR / slug / "benchmark.json"
    if not path.exists():
        pytest.skip(f"{slug} has no benchmark record")
    with contextlib.redirect_stdout(_io.StringIO()):
        fresh = benchmark.measure(slug)
    stored = json.loads(path.read_text())
    assert stored == fresh, (
        f"{_id(target)}/benchmark.json is stale — `manamap pilot benchmark {slug}`")
