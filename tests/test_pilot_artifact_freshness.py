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


def _is_retired(deck_dir):
    """Broken-down, superseded or retired — `deck_info.STATE_RETIRED`'s bucket."""
    import json as _json
    info = deck_dir / "info.json"
    if not info.exists():
        return False
    try:
        return bool((_json.loads(info.read_text()) or {}).get("lifecycle"))
    except Exception:                            # pragma: no cover - defensive
        return False


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
        if _is_retired(d):
            # A RETIRED DECK'S ARTIFACTS ARE HISTORY, NOT CLAIMS. Nothing plays
            # the list, so a model correction leaves them "stale" forever and
            # the only way to green the gate is regenerating a document about a
            # deck nobody will shuffle. The pilot's rule, 2026-08-27.
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


def test_the_net_change_gate_says_so_when_it_has_nothing_to_gate():
    """AN EMPTY PARAMETRIZE IS A GATE THAT EVAPORATED, and pytest reports it as
    one grey "got empty parameter set" line nobody reads.

    `net_change.json` exists ONLY on branches, so a bench with no open branches
    takes `test_net_change_matches_a_fresh_run` from ten cases to zero — which
    is correct, and is exactly the state after the pinned decks were cleared.
    The gate going quiet is fine; going quiet *silently* is not. This says it
    out loud so "no coverage" and "coverage passing" cannot look alike.
    """
    targets = _slugs("net_change.json")
    if not targets:
        pytest.skip("no branches on this bench — net-change has nothing to "
                    "compare, which is the state of a fleet with no open "
                    "candidate lists, not a broken gate")
    assert all(b for _s, b in targets), (
        "net-change is branch-only by definition; a deck-level target means "
        "something wrote one where it does not belong")


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
def test_benchmark_matches_a_fresh_run(target, unchanged):
    """`benchmark.json` is tracked, so the workbench can read it on a static
    host — and it is deterministic (fixed seed, fixed iterations, uniform
    flags), so it must equal what a fresh run produces. A stale benchmark is a
    ranking computed against a deck that no longer exists.

    THE `unchanged` GATE WAS MISSING, and it is the only freshness test in this
    file that ever lacked it. `benchmark.measure` is a full 10,000-iteration
    goldfish with treasures and combat forced on, so all ten targets ran on
    EVERY `make test`, cache warm or cold: 40.6s of CPU that no edit had asked
    for, and the floor under the whole suite. Its seven siblings all take the
    fixture; this one takes it now.
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
    unchanged(*CODE, DECKS_DIR / slug)
    with contextlib.redirect_stdout(_io.StringIO()):
        fresh = benchmark.measure(slug)
    stored = json.loads(path.read_text())
    assert stored == fresh, (
        f"{_id(target)}/benchmark.json is stale — `manamap pilot benchmark {slug}`")


# ── versions.json — the rap sheet, and the one artifact that reads git ───


def _head():
    """HEAD's sha, as a cache key. `unchanged` digests FILE BYTES and cannot see
    git, so without this the cache serves a stale pass the moment a decklist is
    committed — the artifact would move underneath a test that never re-ran."""
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=DECKS_DIR.parent.parent,
                              check=False).stdout.strip() or "no-git"
    except OSError:
        return "no-git"


@requires_deck
@pytest.mark.parametrize("target", _slugs("versions.json"), ids=_id)
def test_versions_json_matches_a_fresh_run(target, tmp_path, unchanged, monkeypatch):
    """`versions.json` IS THE RAP SHEET, and it was gitignored until 2026-09-02.

    The old argument: a version row carries the sha and date of the commit that
    created it, which are unknowable inside that commit — so a copy written in
    the SAME commit as a decklist change is one version behind. True, and it
    misses two things. Nothing reads `sha`, `first_sha` or `subject` (the rap
    sheet reads version, date, in[], out[], record, tags), and the tracked
    `deck_versions.json` already stores commit shas written in a LATER commit
    than the one they name.

    WHY THIS GATE IS STABLE. `deck_history.revisions()` runs `git log --follow --
    decklist.txt`, so commits that do not touch a decklist are INVISIBLE to it.
    A `versions.json` written in such a commit is therefore a FIXED POINT —
    regenerating it at any later HEAD is byte-identical until the next decklist
    change. That is what makes a byte-comparison gate satisfiable here at all,
    and it is why the two-commit rule in `test_pilot_commit_protocol.py` is the
    other half of this.
    """
    from manamap.pilot import deck_versions

    slug, _branch = target
    unchanged(*CODE, DECKS_DIR / slug, _head())

    def rerun():
        deck_versions.main(type("Args", (), {
            "slug": slug, "action": "list", "ref": None, "write": True,
            "as_json": False, "full": False, "at": None, "note": None,
            "clear": False, "force": False})())

    fresh, old = _roundtrip(target, "versions.json", rerun, tmp_path)
    assert fresh == old, (
        f"{slug}/versions.json is stale — run `make manuals` and commit it. "
        f"If you just changed {slug}'s decklist, that is a SEPARATE commit: a "
        f"version's sha is not knowable inside the commit that creates it.")


@requires_deck
def test_every_deck_with_a_decklist_has_a_tracked_version_list():
    """THE RAP SHEET RENDERED ITS EMPTY STATE IN PRODUCTION.

    `versions.json` was gitignored and deferred to a "deploy-time step with git
    available" that was never built — so the deployed site fetched a 404, the
    dossier's rap sheet showed "No committed versions yet", and it said that
    about a deck with three versions and a v1.0.2 release. Five of ten decks did
    not even have one locally.

    This is the check that the artifact exists everywhere it should, which is a
    different question from whether it is fresh.
    """
    import subprocess

    root = DECKS_DIR.parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "data/decks/*/versions.json"],
        capture_output=True, text=True, cwd=root, check=False).stdout.split()
    have = {p.split("/")[2] for p in tracked}
    want = {d.name for d in DECKS_DIR.iterdir()
            if d.is_dir() and (d / "decklist.txt").exists()}
    missing = sorted(want - have)
    assert not missing, (
        f"no tracked versions.json for {missing} — run `make manuals` and "
        f"commit. The deck page fetches this file and renders an empty rap "
        f"sheet without it.")
    assert len(want) >= 5
