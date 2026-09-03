"""The Pilot's Operating Handbook — the section registry, and nothing else.

The closest real object is the binder in an aircraft's side pocket. Its ordering
is what is being stolen, because it is built for somebody who needs the right
page under pressure: EMERGENCIES COME BEFORE NORMAL OPERATION, and a procedure is
a numbered checklist rather than prose.

The dossier (`viz/deck.html`) is what sits on the desk. This is what is in the
cockpit.

THIS FILE IS DELIBERATELY SMALL, for the same reason `page_spec.py` is: position
in the list is the order, the number is the identity, and if something here grows
a second axis that is the magazine coming back. It holds vocabularies and no
logic.

WHY NUMBERS RATHER THAN SLUGS. `page_spec` identifies a section by slug and the
manual cross-references by name ("see above"). A POH cross-references by NUMBER —
3.2.1 — and that only works if the number is a fact about the section rather than
a thing a writer typed. `number_of()` is the single source, `xref()` is the only
way to render one, and a validator checks that every reference resolves.
"""

#: (number, id, title, promise, tiers, source)
#:
#: `tiers` is what the section GRANTS, carried over from `page_spec` verbatim
#: because the rule is the same: a section whose list includes "verified" may
#: MINT a ✓ claim; one that does not may discuss a verified line without
#: promoting it.
#:
#: `source` says who writes it, and it is load-bearing rather than documentary:
#:   "data"     regenerates from tracked artifacts; a version bump rewrites it
#:   "authored" drafted by an agent, edited by the pilot, revision-barred
#: A section is one or the other and never both — the thing that made the
#: magazine unmaintainable was prose and figures sharing a key.
SECTIONS = [
    ("0", "front-matter", "Front matter",
     "What this book is, which deck it applies to, and what changed.",
     (), "data"),
    ("1", "general", "General",
     "What the machine is and what it is for.",
     ("data",), "data"),
    ("2", "limitations", "Limitations",
     "What this deck cannot do, stated flatly.",
     ("data",), "data"),
    # THIRD, AND THAT IS THE WHOLE POINT OF THE FORM. A handbook that puts
    # normal operation first is a manual; one that puts emergencies first is a
    # handbook you can use while something is going wrong.
    ("3", "emergency", "Emergency procedures",
     "One condition per page. What is happening, how you know, what to do.",
     ("coach",), "authored"),
    ("4", "normal", "Normal procedures",
     "Pre-flight, startup, assembly, cruise, closing.",
     ("data", "coach"), "authored"),
    ("5", "performance", "Performance",
     "The measured numbers, charted, each with its interval.",
     ("data",), "data"),
    ("6", "systems", "Systems description",
     "One subsection per component of the schematic.",
     ("data", "coach"), "data"),
    ("7", "handling", "Handling and rules of engagement",
     "Threat optics, alliances, and who to hit first.",
     ("coach",), "authored"),
    ("8", "matchups", "Matchups",
     "One page per archetype the pod actually fields.",
     ("coach",), "authored"),
    ("9", "appendices", "Appendices",
     "Card reference, proven lines, revision log, index.",
     ("verified", "data"), "data"),
]

SECTION_IDS = [s[1] for s in SECTIONS]
SECTION_BY_ID = {s[1]: {"number": s[0], "title": s[2], "promise": s[3],
                        "tiers": s[4], "source": s[5]} for s in SECTIONS}
SECTION_BY_NUMBER = {s[0]: s[1] for s in SECTIONS}

#: Sections that regenerate from data on a version bump, versus those a person
#: edits. `manual_revisions.json` records a revision against the second set;
#: the first set is never revision-barred because a bar on machine churn tells
#: the reader nothing.
DATA_SECTIONS = tuple(s[1] for s in SECTIONS if s[5] == "data")
AUTHORED_SECTIONS = tuple(s[1] for s in SECTIONS if s[5] == "authored")

#: THE ENGINE STAGES, in schematic order — mirrored from `validate_engine.STAGES`
#: and asserted equal by a test rather than retyped as a second vocabulary.
#:
#: The brief for this handbook named four stages: fuel, ignition, payoff,
#: conversion, with protection and answers "drawn as guards". Three of those are
#: real; `payoff` is called `output` here, there is no `answers` stage at all,
#: and `mana`, `fodder` and `wincon` have no counterpart in the brief. Renaming
#: them for the manual would put a second vocabulary beside a closed one that
#: four validators already police, so the schematic uses the real eight.
#:
#: GUARDS ARE NOT A SEPARATE THING. `protection -> conversion` is an ordinary
#: line in `engine.json` on a real deck, so protection is drawn as a stage with
#: edges like any other rather than as a box around the diagram.
STAGE_ORDER = ("mana", "ignition", "fuel", "fodder", "conversion",
               "output", "protection", "wincon")

#: Aviation's three levels, and the ONLY three. Ordered by severity so a page's
#: worst callout can be found by index.
#:
#: THE CAP IS TWO PER PAGE and `validate_poh` enforces it. A page with four
#: warnings has no warnings — the reader learns the colour means nothing, which
#: is the same failure a validator that fires on correct data causes.
CALLOUTS = {
    "warning": "you lose the game",
    "caution": "you lose tempo or a card",
    "note": "useful context",
}
CALLOUT_ORDER = ("warning", "caution", "note")
MAX_CALLOUTS_PER_PAGE = 2

#: The fixed template for an emergency page. Every one of them has these five
#: fields in this order, because a reader under pressure should not have to
#: work out where the actions are on this particular page.
EMERGENCY_FIELDS = ("condition", "indications", "immediate", "subsequent", "notes")

#: And for a systems subsection.
SYSTEM_FIELDS = ("purpose", "components", "normal_operation", "failure_modes",
                 "assessment", "open_items")

#: The conditions an emergency page may cover. Mirrored from
#: `deck_notes.CAUSES` — the closed vocabulary the pilot already files games
#: under — plus the two conditions that are about the table rather than about
#: how a game ended.
#:
#: THE JOIN IS THE POINT: a game logged `--cause wipe` and the emergency page
#: for a wipe are keyed the same, so the handbook's procedures can be read
#: against the games that went wrong that way.
EMERGENCY_CONDITIONS = {
    "wipe": "a board wipe is coming, or has landed",
    "removal": "the commander or a key permanent is targeted",
    "combo": "somebody at the table is going off",
    "mana-drought": "the mana has stopped",
    "stalled": "the engine has not assembled and the window is closing",
    "politics": "you have become the table's problem",
    "raced": "somebody is faster and you cannot answer on board",
}


#: The authored artifact. One file, three sections, because they are written in
#: one pass by one hand and a version bump revises them together.
PROCEDURES_ARTIFACT = "poh_procedures.json"

#: `normal` is a fixed sequence and the order is the procedure. A checklist whose
#: steps can be reordered is a list.
NORMAL_PHASES = (
    ("preflight", "Pre-flight", "the mulligan: what a keepable hand looks like"),
    ("startup", "Startup", "turns one to three, as a decision"),
    ("assembly", "Engine assembly", "the order components come online, and why the order"),
    ("cruise", "Cruise", "the per-turn cycle"),
    ("closing", "Closing", "recognising lethal, and when to wait a turn"),
)


def number_of(sid):
    """The section number, from the registry. Never typed by hand."""
    return SECTION_BY_ID[sid]["number"]


def xref(number, label=None):
    """A cross-reference, rendered as a NUMBER and never as "see above".

    A handbook read out of order has no "above". Every reference resolves to a
    section number, and `validate_poh` checks that the number exists — a
    reference to a section that was renumbered or dropped is a reader sent to a
    page that is not there.
    """
    text = f"{number}" + (f" {label}" if label else "")
    return f'<a class="xref" href="#s{number.replace(".", "-")}">{text}</a>'
