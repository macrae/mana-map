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

import pytest

from manamap.config import (CARD_ROLES_PATH, COMBO_DETAILS_PATH, DECKS_DIR,
                            OBSOLESCENCE_INDEX_PATH, OUTPUT_CSV_PATH,
                            STRATEGY_DOC_PATH, STRATEGY_INDEX_PATH,
                            SYNERGY_GRAPH_PATH)
from manamap.pilot import (
    validate_build,
    validate_considering,
    validate_deck,
    validate_deck_map,
    validate_diagnosis,
    validate_engine,
    validate_goldfish_targets,
    validate_issue,
    validate_stack,
    validate_strategic_frame,
    validate_tutor_guide,
)

from conftest import SRC, requires_deck

# Everything a validator in GATED can read. The pilot tree is a verified closure
# of the code (nothing under it imports outside `manamap.pilot` / `manamap.config`);
# the rest is the tracked global artifacts the deck validators reach for — role
# tags, the two graphs, the combo records, the corpus and the strategy doc. Listed
# rather than digesting `data/`, which is 326 MB and mostly irrelevant.
INPUTS = (SRC / "pilot", SRC / "config.py", CARD_ROLES_PATH, COMBO_DETAILS_PATH,
          OBSOLESCENCE_INDEX_PATH, OUTPUT_CSV_PATH, STRATEGY_DOC_PATH,
          SYNERGY_GRAPH_PATH)

# artifact filename -> module exposing main(args) taking a slug
GATED = {
    "considering.json": validate_considering,
    "tutor_guide.json": validate_tutor_guide,
    "strategic_frame.json": validate_strategic_frame,
    "issue_plan.json": validate_issue,
    "diagnosis.json": validate_diagnosis,
    # Added 2026-08-13 with the subsystems that write them. Both were shipped a
    # cycle earlier than this line and neither was gated, which is precisely the
    # shape of the failure this file's docstring describes — a validator nothing
    # runs is not a gate. `deck_map.json` carries agent-supplied names over a
    # measured membership; `engine.json` carries `verified_by` claims against
    # checker-passed stacks. Both are exactly the kind of artifact that rots
    # silently, because both look complete when they are wrong.
    "deck_map.json": validate_deck_map,
    "engine.json": validate_engine,
    # Added 2026-08-15. Three more validators that existed and were wired into
    # nothing — the same gap, found the same way, one cycle later. Two tracked
    # artifacts were failing their own validator at the time and nobody could
    # have known: yawgmoth's `build_plan.json` named a pool directory that a
    # rename had moved out from under it, and it stayed broken through a full
    # test run because no test ran `validate-build` on a tracked plan.
    "build_plan.json": validate_build,
    "goldfish_targets.json": validate_goldfish_targets,
    "cards.json": validate_deck,
}

# `validate_build` reads the declared pool through `card_pool`, the only reader
# of the gitignored `cards.csv`.
NEEDS_CORPUS = {"build_plan.json"}


def _cases():
    if not DECKS_DIR.is_dir():
        return []
    return [(d.name, art) for d in sorted(DECKS_DIR.iterdir()) if d.is_dir()
            for art in sorted(GATED) if (d / art).exists()]


# Two validators reach through `validate_stack.load_strategy_sections` and report
# every `strategy:<id>` citation as an error when the DB is absent. The DB is
# gitignored and built locally, so on a FRESH CLONE these sixteen cases failed —
# a newcomer's first `make test` was red for a missing artifact rather than a
# defect. Found by cloning into an empty directory and running `make setup &&
# make test`, which is the only check that finds this class of thing.
NEEDS_STRATEGY = {"tutor_guide.json", "diagnosis.json"}


@requires_deck
@pytest.mark.parametrize("slug,artifact", _cases())
def test_tracked_artifact_passes_its_validator(slug, artifact, capsys, unchanged,
                                               request):
    """A tracked artifact that fails its own gate is a published error."""
    if artifact in NEEDS_STRATEGY and not STRATEGY_INDEX_PATH.exists():
        pytest.skip("requires the strategy DB (run `manamap pilot build-strategy-db`)")
    if artifact in NEEDS_CORPUS and not OUTPUT_CSV_PATH.exists():
        pytest.skip("requires the card corpus (run `manamap extract`)")
    unchanged(*INPUTS, DECKS_DIR / slug)
    try:
        GATED[artifact].main(type("Args", (), {"slug": slug})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug}/{artifact} fails its validator:\n"
                        f"{capsys.readouterr().out}")


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
