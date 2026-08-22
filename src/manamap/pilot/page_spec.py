"""The compact deck page: nine sections, and nothing else.

WHY THIS IS A SEPARATE MODULE. `issue_spec.py` is the magazine's spec — seventeen
departments, five acts, rhythm tags, bylines, columnists, packaging copy — and it
is frozen, with a deletion scheduled (`docs/manual-v5-spec.md` phase 4). A live
registry inside a file waiting to be deleted is a merge conflict with a date on
it. This module survives that deletion; that one does not.

IT IS DELIBERATELY SMALL. The point of the replacement is that the single source
of truth shrinks: no ACTS, no INTENSITY, no MODE, no ACCENT, no
OPTIONAL_DEPARTMENTS, no bylines, no promises the renderer has to keep. Position
in the list IS the order. If something here starts growing a second axis, that is
the magazine coming back.

THE ORDER IS THE SPEC'S AND THAT IS DELIBERATE. The obvious move, given the page
exists to ramp a new pilot, is to hoist KEEP OR SHIP and AT THE TABLE above the
roster — get them to the mulligan and the threat read as fast as possible. The
magazine already tried the other arrangement and wrote down why it lost: you
cannot evaluate a keep before you have seen the deck, so THE 99 comes first and
"once somebody has actually had the opportunity to look through the 99, those
mulligans make a lot more sense". That reasoning is about a reader, not about a
magazine, so it survives the format change. Build order is a different question
and is not this file's business.
"""

# (id, title, promise, tiers)
#
# `tiers` is what the section GRANTS, not what it mentions — the same rule the
# departments lived under. A section whose tier list includes "verified" is one
# where a ✓ claim may be made; one that does not may still discuss a verified
# line, it just cannot mint a new one.
SECTIONS = [
    ("plan", "The Plan",
     "What this deck is trying to do, and what it is assuming away.",
     ("data", "coach")),
    ("the-99", "The 99",
     "Every card, grouped by what that part of the deck does.",
     ("data",)),
    ("keep-or-ship", "Keep or Ship",
     "What a keepable hand looks like, and how often you get one.",
     ("data", "coach")),
    ("the-lines", "The Lines",
     "How it actually wins, resolved and checked against the rules.",
     ("verified",)),
    ("at-the-table", "At the Table",
     "Who turns on you, when, and how the matchups go.",
     ("coach",)),
    ("play", "Play",
     "Real spots, and what you would do. Answers on click.",
     ("coach",)),
    # NINTH SECTION, and the one thing the v5 spec does not list. The spec says
    # the manual "does not grow a log or a version panel" — that forbids a FEED,
    # not synthesis, and the distinction is mechanical rather than rhetorical:
    # this section may print nothing keyed by a log entry id or a timestamp. A
    # test asserts exactly that, which is what stops it drifting back into a feed.
    ("debrief", "What the Games Taught",
     "Lessons from games actually played, with the sample size stated.",
     ("coach",)),
    ("the-numbers", "The Numbers",
     "The seeded measurements, and what they do not model.",
     ("data",)),
    ("the-record", "The Record",
     "The proof, one row per case. Unabridged on click.",
     ("verified",)),
]

SECTION_IDS = [s[0] for s in SECTIONS]
SECTION_BY_ID = {s[0]: {"title": s[1], "promise": s[2], "tiers": s[3]}
                 for s in SECTIONS}

# Which legacy department each section's content came from, recorded mechanically
# so nobody re-derives it by eye when comparing `issue-length` before and after.
# A section with several sources merged them; a department absent from the values
# was DROPPED, not moved.
FROM_DEPARTMENT = {
    "plan": ("first-turns", "command-zone"),
    "the-99": ("the-99",),
    "keep-or-ship": ("keep-or-ship",),
    "the-lines": ("the-kill",),
    "at-the-table": ("at-the-table",),
    "play": ("whats-your-play",),
    "debrief": (),                       # new; no legacy counterpart
    "the-numbers": ("by-the-numbers", "sources-say"),
    "the-record": ("judges-desk",),
}
