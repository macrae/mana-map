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
                            STRATEGY_DOC_PATH, SYNERGY_GRAPH_PATH)
from manamap.pilot import (
    validate_considering,
    validate_deck_map,
    validate_diagnosis,
    validate_engine,
    validate_issue,
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
}


def _cases():
    if not DECKS_DIR.is_dir():
        return []
    return [(d.name, art) for d in sorted(DECKS_DIR.iterdir()) if d.is_dir()
            for art in sorted(GATED) if (d / art).exists()]


@requires_deck
@pytest.mark.parametrize("slug,artifact", _cases())
def test_tracked_artifact_passes_its_validator(slug, artifact, capsys, unchanged):
    """A tracked artifact that fails its own gate is a published error."""
    unchanged(*INPUTS, DECKS_DIR / slug)
    try:
        GATED[artifact].main(type("Args", (), {"slug": slug})())
    except SystemExit as exit_:
        if exit_.code:
            pytest.fail(f"{slug}/{artifact} fails its validator:\n"
                        f"{capsys.readouterr().out}")
