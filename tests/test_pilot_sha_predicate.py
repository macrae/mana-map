"""One decklist sha, two named meanings, one comparison.

"The current decklist sha" meant two different things and had SIX definitions
(audited 2026-09-12):

    the hash of `decklist.txt`   deck_notes, deck_versions, deck_branch,
                                 check_in, fetch_deck
    what cards.json was built from   install_agent, deck_info, validate_diagnosis

They are equal exactly when the deck is in sync, which is when the difference
stops mattering — so every caller that conflated them was right until it was
not. And the first group disagreed about HOW: two hashed `read_bytes`, three
hashed `read_text().encode("utf-8")`, which differ on a file with Windows line
endings.
"""

import hashlib
import json

import pytest

from manamap import config
from manamap.pilot import common

from conftest import requires_deck


def _deck(tmp_path, text, monkeypatch):
    root = tmp_path / "decks"
    (root / "slug").mkdir(parents=True)
    (root / "slug" / "decklist.txt").write_text(text, encoding="utf-8",
                                                newline="")
    monkeypatch.setattr(config, "DECKS_DIR", root)
    return root / "slug"


def test_a_crlf_decklist_is_the_same_deck_through_every_caller(tmp_path,
                                                               monkeypatch):
    """THE RE-INTRODUCTION TEST, and the reason `read_text` won.

    `deck_notes` and `check_in` hashed RAW BYTES; `deck_versions`,
    `deck_branch` and `fetch_deck` hashed decoded text. On an LF file those are
    identical, which is why nothing ever caught it — all 40 tracked decklists
    are LF, checked before the consolidation landed.

    On a CRLF file they are not. A pilot whose editor saves Windows line
    endings would have had the same 99 read as a DIFFERENT deck in two modules
    and the same deck in three: a phantom version, a captain's log entry joined
    to nothing, and a diagnosis called stale while it was current.

    `read_text` translates newlines, so the sha is a property of the LIST rather
    than of the file's encoding. Putting `read_bytes` back in `deck_notes` reds
    this.
    """
    from manamap.pilot import deck_notes, deck_versions

    body = "1 Sol Ring\n1 Mox Diamond\n"
    _deck(tmp_path, body.replace("\n", "\r\n"), monkeypatch)

    shas = {
        "common": common.decklist_sha256("slug"),
        "deck_notes": deck_notes.decklist_sha256("slug"),
        "deck_versions": deck_versions.working_sha("slug"),
    }
    assert len(set(shas.values())) == 1, (
        f"a CRLF decklist hashes differently depending on who asks: {shas}")
    # And it is the SAME list as the LF form — that is what "a property of the
    # list" means.
    assert shas["common"] == common.list_sha256(body), (
        "a CRLF re-save reads as a different deck")


def test_the_two_meanings_have_two_names(tmp_path, monkeypatch):
    """`decklist_sha256` is the list; `measured_sha` is what was measured.

    When they differ the deck has moved and nothing has re-fetched — the state
    #45 is about. From the code alone, six call sites gave no way to tell which
    question was being asked.
    """
    deck = _deck(tmp_path, "1 Sol Ring\n", monkeypatch)
    (deck / "cards.json").write_text(json.dumps({"decklist_sha256": "a" * 64}),
                                     encoding="utf-8")
    common.clear_memo()

    assert common.decklist_sha256("slug") != common.measured_sha("slug")
    assert common.measured_sha("slug") == "a" * 64

    (deck / "cards.json").write_text(
        json.dumps({"decklist_sha256": common.decklist_sha256("slug")}),
        encoding="utf-8")
    common.clear_memo()
    assert common.decklist_sha256("slug") == common.measured_sha("slug")


def test_a_truncated_sha_still_matches():
    """THREE DECKS STORE TWELVE CHARACTERS, which is why the comparison had to
    be prefix-tolerant — and why having it in two places was a bug.

    `deck_status._stamp_is_stale` tolerated a prefix; `validate_diagnosis`
    compared with `!=`, so it called a current deck stale. The second was the
    exact defect the first's docstring describes. Both call `sha_matches` now.
    """
    full = "214af3084f809e6691c0492fbca4014063c50e75d34dcee46d165c71c4b43175"
    assert common.sha_matches(full, full[:12])
    assert common.sha_matches(full[:12], full)
    assert not common.sha_matches(full, "f" * 12)
    # Absent means ABSENT, never a match.
    assert not common.sha_matches(full, None)
    assert not common.sha_matches(None, None)
    # And a prefix too short to mean anything is not a match either.
    assert not common.sha_matches(full, full[:4])


@requires_deck
def test_every_caller_agrees_on_the_real_fleet():
    """The consolidation is a no-op on tracked data, asserted rather than
    assumed."""
    from manamap.pilot import deck_branch, deck_notes, deck_versions

    checked = 0
    for path in sorted(config.DECKS_DIR.glob("*/decklist.txt")):
        slug = path.parent.name
        shas = {common.decklist_sha256(slug), deck_notes.decklist_sha256(slug),
                deck_versions.working_sha(slug)}
        assert len(shas) == 1, f"{slug}: callers disagree — {shas}"
        checked += 1
    for path in sorted(config.DECKS_DIR.glob("*/branches/*/decklist.txt")):
        slug = path.relative_to(config.DECKS_DIR).parts[0]
        branch = path.parent.name
        assert common.decklist_sha256(slug, branch) == \
            deck_branch._sha_of_list(slug, branch), f"{slug}@{branch}"
        checked += 1
    assert checked >= 20, f"only {checked} lists checked"


# ── Copies: three questions, three names, one default ───────────────────────

def test_count_copies_and_expand_copies_answer_the_same_question():
    """They are the SUM and the LIST of the same thing, so they cannot disagree."""
    cards = [{"name": "Island", "quantity": 11}, {"name": "Sol Ring", "quantity": 1},
             {"name": "No Key Card"}]
    assert common.count_copies(cards) == len(common.expand_copies(cards)) == 13


def test_an_entry_without_a_quantity_counts_as_one():
    """THE DEFAULT WAS THE WHOLE DEFECT.

    `validate_deck` summed `c.get("quantity", 0)` and thirteen other sites used
    1. An entry written without the key would make a 100-card deck read as 99
    and FAIL THE SIZE INVARIANT — the one check that module exists for, firing
    on a correct deck.

    Latent today: all 1,066 tracked entries carry the key, and `fetch_deck` is
    the only writer. Latent is not fixed; a second writer is a normal thing to
    add. Re-introducing the bug: put `0` back as the default and this reds.
    """
    from manamap.pilot import formats, validate_deck

    cards = [{"name": f"Card {i}", "quantity": 1} for i in range(99)]
    cards.append({"name": "The Hundredth", "is_commander": True})   # no quantity
    assert common.count_copies(cards) == 100

    errors = validate_deck.validate({"cards": cards}, spec=formats.DEFAULT)
    assert not any("100" in e or "size" in e.lower() for e in errors), (
        f"a 100-card deck with one quantity-less entry failed the size check: "
        f"{errors}")


@requires_deck
def test_the_sum_sites_agree_with_the_expansion_on_the_real_fleet():
    """Swept before landing: routing the sum sites through one function is a
    no-op on every tracked `cards.json`."""
    checked = 0
    for path in sorted(config.DECKS_DIR.glob("*/cards.json")):
        cards = json.loads(path.read_text()).get("cards") or []
        if not cards:
            continue
        assert common.count_copies(cards) == len(common.expand_copies(cards)), (
            f"{path.parent.name}: the sum and the expansion disagree")
        checked += 1
    assert checked >= 10, f"only {checked} decks checked"


# ── The deck root: one home, and no module may take a copy ──────────────────

def test_no_module_holds_a_module_level_copy_of_the_deck_root():
    """A `from`-IMPORT IS A COPY, AND A COPY OUTLIVES A PATCH.

    `from manamap.config import DECKS_DIR` at module scope binds the value once,
    at import. Patching `config.DECKS_DIR` — the documented home, the thing
    `MANAMAP_DATA_DIR` moves — reaches none of those copies, so a test had to
    patch every module that had taken one, and WHICH modules those are depends
    on import order. That is #31: quiet, total, and order-dependent, so the same
    suite passes under `-n0` and fails under `-n auto`.

    SCOPED TO THE DECK ROOT, DELIBERATELY. The general rule was measured first
    and rejected: 195 by-value path imports across 77 modules, of which exactly
    ONE constant is ever moved — `DECKS_DIR`, patched at 85 test sites.
    `ABILITY_EMBEDDINGS_PATH` is patched once. The pipeline modules read their
    paths at import and run as one-shot steps; converting 195 sites to prevent
    a class that has only ever bitten one of them is churn, and the plan said so
    in advance: an allowlist that big means the rule is wrong.

    FUNCTION-LEVEL IMPORTS ARE FINE and are not flagged: an import inside a
    function re-reads `config` on every call, which is the behaviour this wants.
    """
    import ast
    import pathlib

    from conftest import SRC

    offenders, checked = [], 0
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                      # pragma: no cover - defensive
            continue
        checked += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (node.module or "").endswith("config"):
                continue
            if node.col_offset != 0:
                continue                         # inside a function: re-read per call
            if any(alias.name == "DECKS_DIR" for alias in node.names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")

    assert checked >= 100, f"only {checked} modules parsed — has src/ moved?"
    assert not offenders, (
        "a module took a copy of the deck root at import; read "
        "`config.DECKS_DIR` at call time (or `common.decks_root()` inside "
        "pilot):\n  " + "\n  ".join(offenders))


def test_patching_the_configured_root_reaches_every_reader(tmp_path, monkeypatch):
    """ONE PATCH POINT — the property the check above protects."""
    from manamap.pilot import common, deck_branch, regen
    from manamap.sim import opponents

    (tmp_path / "decks" / "probe").mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path / "decks")

    assert common.decks_root() == tmp_path / "decks"
    assert common.deck_dir("probe") == tmp_path / "decks" / "probe"
    # Derived roots follow too — the pod directory hangs off the deck root.
    assert opponents.opponents_dir() == tmp_path / "opponents"
    # And a fleet walk sees the patched tree, not the real one.
    assert deck_branch._deck_holders("Sol Ring", skip=None) == []
    assert [t for t in regen.targets("info.json")] == []


# ── The artifact registries: four views, and what they owe each other ───────

#: Gate entries that are NOT a per-deck file, and so cannot be a `deck-status`
#: stage. Each is exempt for a stated reason rather than by being forgotten.
_NOT_A_DECK_FILE = {
    # A computed predicate over the collection and the other decks, not a file.
    "@ownership",
    # Rendered OUTSIDE the deck directory, into `manuals/p/`.
    "manuals/p/<slug>.html",
}


def test_every_artifact_the_gate_refuses_on_is_one_the_status_board_shows():
    """A DASHBOARD THAT IS GREEN WHILE THE GATE IS RED IS WORSE THAN NO
    DASHBOARD, because people stop checking the gate.

    `deck_status.VALIDATED`'s own docstring records that lesson, measured once
    on ur-dragon mid-swap: `deck-status` FAIL=0 and `validate-issue` FAIL=1 in
    the same second. It was fixed for that gate and the rule was never made a
    check, so it recurred — `promote.GATES[SLEEVED]` required `benchmark.json`,
    `info.json` and a Forge run, and `deck-status` had never mentioned any of
    the three. A pilot saw a complete board and a refused promotion, with no
    file named in common.

    Found by comparing the registries against each other, not by anything going
    red. That is the point of having the comparison.

    THESE FOUR REGISTRIES ARE NOT FOUR COPIES and are not merged: `STAGES` is a
    sequence with staleness keys, `VALIDATED` maps artifact to validator,
    `regen.STAGES` is what rebuilds automatically in dependency order, and
    `promote.GATES` is what a rung requires. Each answers a different question
    and a single table would need a column per consumer. What they owe each
    other is CONSISTENCY, which is what this asserts.
    """
    from manamap.pilot import deck_status, promote

    known = {row[1] for row in deck_status.STAGES} | set(deck_status.VALIDATED)
    gated = {artifact
             for rows in promote.GATES.values()
             for (_label, artifact, _why) in rows}
    assert len(gated) >= 8, f"only {len(gated)} gate entries — has the ladder moved?"

    invisible = sorted(gated - known - _NOT_A_DECK_FILE)
    assert not invisible, (
        "`promote` refuses a rung on artifacts `deck-status` never shows:\n  "
        + "\n  ".join(invisible)
        + "\n(add a STAGES row, or a VALIDATED entry, or exempt it in "
          "_NOT_A_DECK_FILE with a reason)")


def test_everything_regen_rebuilds_is_on_the_status_board():
    """`regen` rebuilds a pinned deck's chain without being asked, so an
    artifact it writes and the board does not show is one nobody can see go
    stale — and staleness is the whole reason the board exists."""
    from manamap.pilot import deck_status, regen

    known = {row[1] for row in deck_status.STAGES} | set(deck_status.VALIDATED)
    rebuilt = {row[1] for row in regen.STAGES}
    assert len(rebuilt) >= 5, f"only {len(rebuilt)} regen stages"
    invisible = sorted(rebuilt - known)
    assert not invisible, (
        "`regen` rebuilds artifacts `deck-status` never shows:\n  "
        + "\n  ".join(invisible))


def test_the_registries_name_artifacts_that_exist_on_a_real_deck():
    """A registry entry for a file nothing writes is a permanent GATE row that
    can never clear — the shape `deck_status.ADDED_2026_08` exists to make
    visible rather than to hide."""
    from manamap.pilot import deck_status, promote, regen

    named = ({row[1] for row in deck_status.STAGES}
             | set(deck_status.VALIDATED)
             | {row[1] for row in regen.STAGES}
             | {a for rows in promote.GATES.values() for (_l, a, _w) in rows})
    named -= _NOT_A_DECK_FILE

    # BRANCHES COUNT. `branch.json` and `net_change.json` live only under
    # `branches/<name>/`, never at the deck root — the first draft of this
    # looked at deck roots alone and reported both as written by nothing.
    roots = []
    for deck in sorted(config.DECKS_DIR.iterdir()):
        if not deck.is_dir():
            continue
        roots.append(deck)
        branches = deck / "branches"
        if branches.is_dir():
            roots.extend(b for b in sorted(branches.iterdir()) if b.is_dir())

    seen, checked = set(), 0
    for root in roots:
        checked += 1
        for artifact in named:
            if (root / artifact.rstrip("/")).exists():
                seen.add(artifact)
    assert checked >= 30, f"only {checked} deck and branch roots scanned"
    never = sorted(named - seen)
    assert not never, (
        "a registry names an artifact that exists on NO deck in the fleet — "
        "either nothing writes it, or the name is wrong:\n  " + "\n  ".join(never))
