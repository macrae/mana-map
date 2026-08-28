"""Every tracked agent artifact must pass its own validator.

Nothing asserted this, and the gap was not theoretical. `considering.json` on
edgar-vampires and hapatra sat in the tree FAILING `validate-considering` — each
claiming an obsolescence the rebuilt `obsolescence_index.json` no longer
supports — while `cache-status` reported `the-ten` as **HIT, inputs unchanged**.

That is worse than a stale artifact. A HIT actively says "do not regenerate", so
the cache was defending a broken file. The routine had been re-blessed rather
than re-spawned after the 2026-07-31 index rebuild, on the reasoning that
nothing in it quoted a synergy rank numerically — true, and it missed that this
artifact makes obsolescence claims the validator re-checks. CLAUDE.md's own rule
covers the case ("if a future MISS touches a routine whose output cites the
changed artifact, re-spawn it"); the judgment call simply went the wrong way.

A validator is only a gate if something runs it. This runs all of them, over
everything tracked, every time.
"""

import importlib

import pytest

from manamap.config import (CARD_ROLES_PATH, COLLECTION_DIR, COMBO_DETAILS_PATH, DECKS_DIR,
                            OBSOLESCENCE_INDEX_PATH, OUTPUT_CSV_PATH,
                            STRATEGY_DOC_PATH, STRATEGY_INDEX_PATH,
                            SYNERGY_GRAPH_PATH)
from manamap.pilot.deck_status import VALIDATED
from manamap.pilot import (
    validate_build,
    validate_considering,
    validate_deck,
    validate_deck_map,
    validate_diagnosis,
    validate_engine,
    validate_goldfish_targets,
    validate_issue,
    validate_prescription,
    validate_stack,
    validate_strategic_frame,
    validate_tutor_guide,
)

from conftest import SRC, requires_branch, requires_deck

# Everything a validator in GATED can read. The pilot tree is a verified closure
# of the code (nothing under it imports outside `manamap.pilot` / `manamap.config`);
# the rest is the tracked global artifacts the deck validators reach for — role
# tags, the two graphs, the combo records, the corpus and the strategy doc. Listed
# rather than digesting `data/`, which is 326 MB and mostly irrelevant.
INPUTS = (SRC, CARD_ROLES_PATH, COMBO_DETAILS_PATH,
          OBSOLESCENCE_INDEX_PATH, OUTPUT_CSV_PATH, STRATEGY_DOC_PATH,
          SYNERGY_GRAPH_PATH,
          # `validate_recon` falsifies a recon's `ownership` claims against the
          # pilot's boxes, so a box edit changes a verdict. Without this the cache
          # keys on bytes that did not move and serves a stale pass.
          COLLECTION_DIR)

# artifact filename -> module exposing main(args) taking a slug.
#
# The map now lives in `deck_status.VALIDATED` and is imported rather than
# restated. It was duplicated here for one cycle, which meant the TEST knew
# which artifacts had gates while `deck-status` — the command the runbook says
# to run first, every time — did not, and reported a deck green while
# `validate-issue` failed on it in the same second. Two maps that can disagree
# about what is gated is the same class of defect as two records of what is
# applied.
#
# `build_plan.json` is gated here only: it is not a lifecycle stage, so it has
# no `STAGES` row for `deck-status` to hang a verdict on.
GATED = {name: importlib.import_module(dotted)
         for name, dotted in VALIDATED.items()}


def test_the_test_does_not_know_about_a_gate_the_status_command_lacks():
    """THE HAND-PATCH IS GONE, AND THIS KEEPS IT GONE.

    `GATED["build_plan.json"] = validate_build` used to sit here, so the TEST
    knew about a gate `deck-status` did not — the exact two-maps divergence the
    `VALIDATED` map was extracted to end, left open for one artifact. Any future
    addition goes in `deck_status.VALIDATED` where both readers see it.
    """
    assert "build_plan.json" in VALIDATED
    assert set(GATED) == set(VALIDATED)

# Both read the corpus through `card_pool`, the only reader of the gitignored
# `cards.csv` — `validate_build` for the declared pool, `validate_recon` to prove
# every named card is real, legal and in identity.
NEEDS_CORPUS = {"build_plan.json", "deck_recon.json"}


#: Validators that can be pointed at a branch. The rest take a slug only, so a
#: branch copy of their artifact is genuinely ungatable until they learn one —
#: named here rather than skipped silently, because "no case" and "no gate" look
#: identical from the outside.
BRANCH_AWARE = {"cards.json", "deck_map.json", "goldfish_targets.json",
                "net_change.json"}


def _is_retired(deck_dir):
    """Broken-down, superseded or retired — one bucket, `deck_info.STATE_RETIRED`.

    A RETIRED DECK'S ARTIFACTS ARE HISTORY, NOT CLAIMS. Nothing plays it and
    nothing derives from it, so holding its documents to today's model is the
    "gate that reddens history" that `validate_prescription` already refused to
    be. Measured the day it bit: a correctness fix to `manabase.land_colors`
    moved the colour-source count on six decks, and three of them — hapatra,
    radagast, sisay — were broken down or belong to someone else. Regenerating
    those meant re-running an agent over a deck nobody will play.

    The pilot's rule, 2026-08-27: "if a deck is deprecated, broken down, exclude
    it from these downstream tasks."
    """
    import json as _json
    info = deck_dir / "info.json"
    if not info.exists():
        return False
    try:
        return bool((_json.loads(info.read_text()) or {}).get("lifecycle"))
    except Exception:                            # pragma: no cover - defensive
        return False


def _cases():
    """(slug, branch, artifact) for every tracked copy — DECKS AND BRANCHES.

    THE BRANCH TREE WAS GATED BY NOTHING. Nine tests globbed
    `DECKS_DIR.iterdir()` at top level and none reached `branches/`, so ten
    tracked files under `ur-dragon/branches/treasure-v2/` had no validator and no
    freshness test — in a subsystem where this repo has already caught an
    artifact being measured against the wrong decklist.
    """
    if not DECKS_DIR.is_dir():
        return []
    out = []
    for d in sorted(DECKS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if _is_retired(d):
            continue
        for art in sorted(GATED):
            if (d / art).exists():
                out.append((d.name, None, art))
        for b in sorted((d / "branches").glob("*")):
            if not b.is_dir():
                continue
            for art in sorted(set(GATED) & BRANCH_AWARE):
                if (b / art).exists():
                    out.append((d.name, b.name, art))
    return out


def _case_id(case):
    slug, branch, art = case
    return f"{slug}@{branch}/{art}" if branch else f"{slug}/{art}"


# Two validators reach through `validate_stack.load_strategy_sections` and report
# every `strategy:<id>` citation as an error when the DB is absent. The DB is
# gitignored and built locally, so on a FRESH CLONE these sixteen cases failed —
# a newcomer's first `make test` was red for a missing artifact rather than a
# defect. Found by cloning into an empty directory and running `make setup &&
# make test`, which is the only check that finds this class of thing.
NEEDS_STRATEGY = {"tutor_guide.json", "diagnosis.json"}


@requires_deck
@pytest.mark.parametrize("case", _cases(), ids=_case_id)
def test_tracked_artifact_passes_its_validator(case, capsys, unchanged, request):
    """A tracked artifact that fails its own gate is a published error."""
    slug, branch, artifact = case
    if artifact in NEEDS_STRATEGY and not STRATEGY_INDEX_PATH.exists():
        pytest.skip("requires the strategy DB (run `manamap pilot build-strategy-db`)")
    if artifact in NEEDS_CORPUS and not OUTPUT_CSV_PATH.exists():
        pytest.skip("requires the card corpus (run `manamap extract`)")
    unchanged(*INPUTS, DECKS_DIR / slug)
    try:
        GATED[artifact].main(
            type("Args", (), {"slug": slug, "branch": branch})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{_case_id(case)} fails its validator:\n"
                        f"{capsys.readouterr().out}")


@requires_branch
def test_every_branch_artifact_a_validator_can_reach_is_gated():
    """NON-EMPTY GUARD, and a real one — the branch cases exist to catch a class
    this repo has already been bitten by, and a parametrize that silently yields
    zero of them would pass forever while gating nothing."""
    branch_cases = [c for c in _cases() if c[1]]
    if not any((DECKS_DIR / d.name / "branches").is_dir()
               for d in DECKS_DIR.iterdir() if d.is_dir()):
        pytest.skip("no branches on this checkout")
    assert branch_cases, (
        "a branch exists and no branch artifact is gated — the recursion in "
        "`_cases` has regressed to the top-level glob it replaced")


def _deck_slugs():
    if not DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in DECKS_DIR.iterdir()
                  if d.is_dir() and (d / "stacks").is_dir())


@requires_deck
@pytest.mark.parametrize("slug", _deck_slugs())
def test_every_tracked_stack_and_decision_passes_the_citation_contract(slug, capsys,
                                                                       unchanged):
    """`validate-stack` over a whole deck — 56 stacks and 18 decisions, ungated.

    The map above is keyed by artifact FILENAME, which is why this one is
    separate: `validate_stack` takes a slug and walks two directories. That
    shape is the reason it was never added, and the reason sisay's decision 002
    was failing its own validator in a tracked artifact while every test passed.

    The citation contract is the load-bearing promise of this whole project — a
    ✓ badge means every step quotes a real rule verbatim — so an unrun validator
    here is worse than an unrun validator anywhere else.
    """
    if not STRATEGY_INDEX_PATH.exists():
        pytest.skip("requires the strategy DB (run `manamap pilot build-strategy-db`)")
    unchanged(*INPUTS, DECKS_DIR / slug)
    try:
        validate_stack.main(type("Args", (), {"slug": slug, "stack": None,
                                              "scenario_only": False})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug} stacks/decisions fail the citation contract:\n"
                        f"{capsys.readouterr().out}")


def _prescription_slugs():
    if not DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in DECKS_DIR.iterdir()
                  if d.is_dir() and (d / "prescriptions").is_dir())


@requires_deck
@pytest.mark.parametrize("slug", _prescription_slugs())
def test_every_tracked_prescription_passes_its_validator(slug, capsys, unchanged):
    """`validate-prescription` over a whole deck. Keyed by directory like stacks, so
    it cannot ride the filename map; a stale one (older decklist) is form-checked
    only by design — prescriptions accumulate and history is not an error."""
    unchanged(*INPUTS, DECKS_DIR / slug)
    try:
        validate_prescription.main(type("Args", (), {"slug": slug, "id": None})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug} prescriptions fail their validator:\n"
                        f"{capsys.readouterr().out}")


def _sim_slugs():
    if not DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in DECKS_DIR.iterdir()
                  if d.is_dir() and (d / "sim").is_dir() and list((d / "sim").glob("*.json")))


@requires_deck
@pytest.mark.parametrize("slug", _sim_slugs())
def test_every_tracked_simulation_run_passes_its_validator(slug, capsys, unchanged):
    """`validate-sim` over a deck's runs: form always; analysis re-derived from the logs
    where they exist (gitignored, so only where the run was made). A sampled artifact
    cannot be replayed, but its parser can be re-run — that is the whole check."""
    from manamap.sim import validate_sim
    unchanged(*INPUTS, DECKS_DIR / slug / "sim")
    try:
        validate_sim.main(type("Args", (), {"slug": slug})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug} simulation runs fail their validator:\n{capsys.readouterr().out}")


# ── The two validators that were wired into nothing ──────────────────────
#
# THIRD INSTANCE of a defect class this file's own header documents twice.
# `validate_issue` gates 9 `issue.json` + 9 `issue_plan.json` and reaches into
# `manual_prose.json` and `tutor_guide.json`; `validate_strategy` gates
# `strategy.md` and `CHANGELOG.md`, both tracked. Neither was ever run by a
# test. `deck_status.py` even carries a comment recording `validate-issue`
# failing live on ur-dragon while `deck-status` reported the deck green.
#
# Like `validate_stack`, `validate_issue` takes a SLUG and walks several files,
# so it cannot ride the filename-keyed map — which is exactly why it was skipped.

#: Decks whose issue fails today, with the reason. STRICT xfail: if one starts
#: passing, this list is wrong and the test says so rather than going quiet.
#:
#: NOT hand-patched. Both are agent-authored prose, and `magazine-editor` was
#: retired in the 2026-08 pivot — there is no agent left to re-spawn, so the
#: honest options are to mark them or to delete the artifacts. Marking keeps the
#: gate live for the other seven and blocks a tenth from joining them.
ISSUE_XFAIL = {
    "edgar-vampires":
        "prose predates THE LOCK's 12 swaps — captions and the roster name "
        "Sacred Foundry, Diabolic Intent, Demonic Tutor and Cavern of Souls, "
        "all cut. Fixing it means re-writing copy a retired agent authored.",
    "ur-dragon":
        "quotes '31 lands' in three places — the DISTINCT-CARD count, where "
        "the deck runs 36 copies. The copies-vs-entries defect this repo "
        "documents, live in tracked prose.",
}


def _issue_slugs():
    if not DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in DECKS_DIR.iterdir()
                  if d.is_dir() and (d / "issue.json").exists())


@requires_deck
@pytest.mark.parametrize("slug", _issue_slugs())
def test_every_tracked_issue_passes_validate_issue(slug, capsys, unchanged, request):
    if slug in ISSUE_XFAIL:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason=ISSUE_XFAIL[slug]))
    unchanged(*INPUTS, DECKS_DIR / slug)
    from manamap.pilot import validate_issue
    try:
        validate_issue.main(type("Args", (), {"slug": slug, "strict": False})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug} fails validate-issue:\n{capsys.readouterr().out}")


def test_the_issue_gate_actually_has_cases():
    """NON-EMPTY GUARD. Nine issues are tracked; a parametrize that yields zero
    would pass forever while gating nothing — which is the state this test was
    added to end."""
    assert len(_issue_slugs()) >= 5, _issue_slugs()


def test_the_strategy_doc_passes_its_validator(capsys):
    """`strategy.md` and `CHANGELOG.md` are tracked, and every `strategy:<id>`
    citation in the fleet resolves against them. Ungated until now."""
    from manamap.pilot import validate_strategy
    try:
        validate_strategy.main(type("Args", (), {})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"strategy.md fails its validator:\n"
                        f"{capsys.readouterr().out}")
