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
    bracket, deck_combos, deck_info, diagnostic, goldfish, mana_analysis,
    net_change)

from conftest import is_retired, module_closure, requires_deck, requires_data

# ── The code half of the key: DERIVED, never hand-traced ────────────────────
#
# This was `(SRC,)` — the whole source tree — so an edit anywhere under
# `src/manamap/` re-ran 279 freshness cases, most of them 10,000-game goldfish
# runs. That was the right answer to the wrong question, and the comment it
# replaced is the reason it was chosen: the version BEFORE it named the inputs
# by hand and asserted the closure was "checked rather than assumed". It was
# wrong in NINE modules across three subpackages (`deck_info` imports
# `manamap.sim`; six modules import `manamap.analysis.common`; five import
# `manamap.ingest`), and a missed edge here does not go red — it serves a
# stale PASS.
#
# So: neither the whole tree nor a list somebody types. `module_closure` walks
# the SYNTAX TREE of each producer, transitively, counting imports at any depth
# because this package imports lazily almost everywhere. Two controls in
# `tests/test_conftest_cache.py` hold it to reality — one asserts it covers
# every `manamap.*` module a real producer run actually imports, one
# re-introduces the bug by touching a transitive dependency.
#
# Measured 2026-09-12: 60 of 182 files, so `sven/`, `training/`, `export/`,
# `cli.py`, the eval harnesses and the frozen magazine renderer no longer
# invalidate a goldfish run.
CODE = module_closure(bracket, deck_combos, deck_info, diagnostic, goldfish,
                      mana_analysis, net_change)

# ── The data half: WIDER for the two artifacts that read OTHER DECKS ─────────
#
# #49. `info.json` and `net_change.json` both embed, per card, which other decks
# hold it and whether each of THOSE is locked (`deck_branch.source()`). So a
# paper lock on ur-dragon rewrites a field inside gishath's dossier and inside
# four branch bills — and every one of those cases named only its own deck's
# directory, so the key did not move and the cache served a passing result for
# inputs that had changed. Found on 2026-09-12 when an unrelated source edit
# forced a real run; five artifacts were stale, one of them for a day.
#
# Narrowing the code half without widening this one would have made it worse:
# over-invalidation on source was the only thing that ever flushed these.
CROSS_DECK = (DECKS_DIR,)


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
        if is_retired(d):
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
@requires_data
@pytest.mark.parametrize("target", _slugs("combos.json"), ids=_id)
def test_combos_matches_a_fresh_run(target, tmp_path, unchanged):
    """The combo report stamps its list, but a stamp cannot see a change to
    `deck_combos.py` or a Spellbook refresh — only recomputation can. It reads
    the combo file AND the corpus (legality and identity for the near misses),
    so both are in the key."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    unchanged(*CODE, root, COMBO_DETAILS_PATH, OUTPUT_CSV_PATH)

    def rerun():
        deck_combos.main(type("Args", (), {"slug": slug, "branch": branch,
                                           "write": True, "as_json": False})())

    fresh, old = _roundtrip(target, "combos.json", rerun, tmp_path)
    assert fresh == old, (
        f"{_id(target)}/combos.json is stale — rerun `manamap pilot deck-combos {slug}"
        f"{f' --branch {branch}' if branch else ''} --write` and commit it.")


@requires_deck
@pytest.mark.parametrize("target", _slugs("combos.json"), ids=_id)
def test_the_bracket_report_counts_the_lines_the_combo_report_lists(target):
    """`bracket.assess` and `deck_combos` read the same rows: `combo_count` is
    the included lines minus the ones that assume another commander, which is
    exactly the set `assess` keeps as `contained`. Two files, one answer."""
    slug, branch = target
    root = DECKS_DIR / slug / ("branches/" + branch if branch else "")
    if not (root / "bracket_report.json").exists():
        pytest.skip(f"{_id(target)} has no bracket_report.json")
    report = json.loads((root / "bracket_report.json").read_text())
    combos = json.loads((root / "combos.json").read_text())
    counted = [c for c in combos["included"] if not c["assumes_other_commander"]]
    assert report["combo_count"] == len(counted), (
        f"{_id(target)}: bracket_report.combo_count {report['combo_count']} vs "
        f"{len(counted)} counted line(s) in combos.json — one of them is stale")
    assert len(report["excluded_commander_assumption"]) == \
        combos["summary"]["excluded_commander_assumption"]


@pytest.mark.slow
@pytest.mark.regen
@requires_data
@requires_deck
def test_the_fleet_regenerates_byte_identically(unchanged):
    """goldfish_metrics, net_change, diagnostic and benchmark — every deck and
    branch — must equal what `manamap pilot regen` writes today. Seeded and
    deterministic, so a difference is a real change in the model or a list,
    never noise; and these are the documents a purchase rests on, so one that
    no longer describes its lists is worse than none.

    ONE REGEN, NOT 85 RE-DERIVATIONS (2026-10-05). This was four parametrized
    tests, one case per artifact per deck and branch: 85 cases, 74% of the
    regression tier's CPU, and under xdist's scheduler two workers ground
    through them for ~15 minutes while six sat idle. `regen --jobs` does the
    same work in dependency order across every core in ~6 minutes, and covers
    EXACTLY the same targets — checked case for case before the swap
    (goldfish 38, net-change 33, diagnose 6, benchmark 8; retired decks skipped
    by both). It also compares bytes, which is stricter than the JSON equality
    the four used.

    It writes IN PLACE, so it runs alone: marked `regen`, excluded from the
    parallel run and run serially after it (`make regression`). Every tracked
    file under data/decks/ is snapshotted first and restored after, pass or
    fail. The key is the whole decks tree because net_change and info embed
    other decks' state (CROSS_DECK, above).
    """
    import subprocess

    from manamap.pilot import benchmark, regen

    unchanged(*module_closure(regen, benchmark), *CODE, *CROSS_DECK)
    root = DECKS_DIR.parent.parent
    listed = subprocess.run(["git", "ls-files", "-z", "--", str(DECKS_DIR)],
                            cwd=root, capture_output=True, check=True).stdout
    tracked = [root / f for f in listed.decode().split("\0") if f]
    assert len(tracked) >= 100, f"only {len(tracked)} tracked deck files to compare"
    before = {f: f.read_bytes() for f in tracked if f.exists()}
    try:
        result = regen.run(echo=lambda *a, **k: None)
        stale = sorted(str(f.relative_to(root)) for f, b in before.items()
                       if f.read_bytes() != b)
    finally:
        for f, b in before.items():
            if not f.exists() or f.read_bytes() != b:
                f.write_bytes(b)
    assert not result["failures"], f"regen failed: {result['failures']}"
    assert not stale, (
        f"{len(stale)} tracked artifact(s) no longer match a fresh run — "
        f"`manamap pilot regen --jobs 8 && make manuals`, then commit:\n  "
        + "\n  ".join(stale))


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


@pytest.mark.regression
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
    unchanged(*CODE, *CROSS_DECK, root, OUTPUT_CSV_PATH, CARD_ROLES_PATH,
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
