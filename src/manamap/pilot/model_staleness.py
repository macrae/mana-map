"""Was this prose written against the model that is running now?

THE GAP THIS CLOSES. Regenerating the fleet after the mana-rock and colour fixes
moved `mean_cast_turn` on seven decks, and four of them quote the old scalar in
prose an agent wrote — gishath's tutor guide three times, heliod's strategic
frame three times and a decision spread twice, sisay across four artifacts,
radagast's diagnosis. **39 stale figures, and `validate-diagnosis`,
`validate-strategic-frame` and `validate-tutor-guide` all passed**, because the
decklist sha had not moved and nothing recorded which MODEL produced a number.

REPORTED, NEVER FAILED. Prose written against an older model was correct when it
was written; a gate that reddens it would demand either a ~2.5M-token re-spawn
or a hand-patch, and hand-patching agent prose to green a gate puts a fresh claim
under an old byline. So this says so on every run — the `scaffold` note's
reasoning exactly — and the state cannot go quiet.

WHAT IT CANNOT DO. It cannot tell whether a stale-stamped document actually
quotes a figure that moved; only that it was written under a different model. A
document with no stamp at all is the common case today and reports as unknown
rather than as stale, because "written before stamping existed" and "written
against an older model" are different claims and only one of them is evidence.
"""

import json

from manamap.pilot.common import deck_dir


def stamp_of(slug, branch=None):
    """The model version the deck's goldfish figures were produced under."""
    try:
        path = deck_dir(slug, branch) / "goldfish_metrics.json"
    except FileNotFoundError:
        # An unknown deck or branch is not a staleness verdict. This runs inside
        # three validators; raising here would turn a missing directory into a
        # crash in a gate that is about something else entirely.
        return None
    if not path.exists():
        return None
    try:
        return (json.loads(path.read_text()).get("meta") or {}).get("model_version")
    except Exception:                              # pragma: no cover - defensive
        return None


def note(slug, artifact_doc, branch=None):
    """A one-line staleness note, or "" when there is nothing to say."""
    from manamap.pilot import goldfish
    current = goldfish.model_version()
    figures = stamp_of(slug, branch)
    if figures and figures != current:
        return (f"\n     THE DECK'S FIGURES ARE STALE — goldfish_metrics.json was "
                f"produced by model {figures}, the simulator is now {current}. "
                f"Re-run `manamap pilot goldfish {slug}` before trusting any "
                f"number quoted from it.")
    written = (artifact_doc or {}).get("model_version")
    if written is None:
        return ""            # predates stamping: unknown, not stale
    if written != current:
        return (f"\n     WRITTEN AGAINST AN OLDER MODEL — this prose quotes "
                f"figures from model {written} and the simulator is now "
                f"{current}. It was correct when written; re-spawn the agent if "
                f"a quoted number matters. Do NOT hand-edit the prose.")
    return ""
