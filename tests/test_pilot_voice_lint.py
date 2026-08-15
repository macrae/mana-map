"""The voice layer: who is supposed to be speaking, and whether they all sound alike.

This module is referenced by name in CLAUDE.md as the thing that keeps
`issue_spec.PROSE_KEY_DEPARTMENT` honest — and it had never been written. The
first test below is that promise, paid.

The checks here are deliberately CROSS-DECK, which is why they are not in
`validate_issue`: that command takes one slug and can never see a pattern that
only exists across the fleet. A formula is invisible in a single issue and
obvious in three.
"""

import collections
import json
import re

from manamap.pilot.issue_spec import PROSE_KEY_DEPARTMENT, voices_for
from manamap.config import DECKS_DIR as DECKS


def test_every_voice_checked_prose_key_is_one_a_renderer_actually_reads():
    """`PROSE_KEY_DEPARTMENT` is how `validate-issue` asks "who is speaking here".

    A key that drifts out of that map stops being voice-checked **silently** —
    the lint simply finds nothing and passes, which is the worst shape a check
    can fail in. So the map's keys must be keys the renderer really reads, and
    each must resolve to a department with a byline.
    """
    from manamap.pilot import build_manual

    source = build_manual.__loader__.get_source("manamap.pilot.build_manual")
    for key, dept in sorted(PROSE_KEY_DEPARTMENT.items()):
        assert f'"{key}"' in source, (
            f"PROSE_KEY_DEPARTMENT names {key!r} but no renderer mentions it — "
            f"either the renderer dropped the key or the map is stale, and the "
            f"voice lint has been silently skipping it either way")
        assert voices_for(key), (
            f"{key!r} maps to {dept!r}, which has no byline — nothing can be "
            f"voice-checked against an unsigned department")


def test_the_panel_is_the_one_prose_key_that_carries_its_own_voices():
    """`pilots_log` is deliberately ABSENT from the map, and that is the design.

    Its turns each carry a `voice`, which is a better answer than a department
    byline and the reason that shape was chosen. If it ever appears in the map,
    the per-turn check and the per-key check would both run and disagree.
    """
    assert "pilots_log" not in PROSE_KEY_DEPARTMENT


def _hot_take_openers():
    opens = {}
    for path in sorted(DECKS.glob("*/manual_prose.json")):
        turns = json.loads(path.read_text()).get("pilots_log")
        if not isinstance(turns, list) or not turns:
            continue
        text = (turns[0] or {}).get("text", "").strip()
        if text:
            first = re.split(r"(?<=[.!?])\s+", text)[0]
            opens[path.parent.name] = " ".join(first.split()[:4]).lower().rstrip(".,:;")
    return opens


def test_no_two_decks_open_their_hot_take_the_same_way():
    """A hot take is fine once and a formula by the second issue.

    Eight decks wrote one and **five opened "Here is the thing…"** — radagast
    shipped first and the rest converged on its shape. That is the defect STYLEv3
    §7.2 already legislates against for deks ("six in one issue is a formula"),
    and it lands harder here: the hot take is the first sentence of the loudest
    department in the book, so a reader who meets the construction twice learns
    to skip the opener.

    The probe is the first FOUR words, which is where a formula lives. Two takes
    may legitimately share a word or two; "Here is the thing" is four.
    """
    opens = _hot_take_openers()
    if len(opens) < 2:
        return                                  # nothing to compare yet
    shared = collections.defaultdict(list)
    for slug, stem in opens.items():
        shared[stem].append(slug)
    clashes = {stem: sorted(s) for stem, s in shared.items() if len(s) > 1}
    assert not clashes, (
        "hot takes sharing an opening construction:\n  "
        + "\n  ".join(f"{stem!r}: {slugs}" for stem, slugs in clashes.items())
        + "\nOpen on the board, the number, the card or the instruction — never "
          "on the throat-clearing before it (pilot-panel charter)."
    )
