"""Pilot: the canonical department system for a Pilot's Manual issue.

Single source of truth shared by the issue-plan validator, the renderer, and the
`magazine-editor` agent's contract. The department list, its order, and each
department's evidence tier come from STYLEv3.md §5 — the fixed reading
experience every issue delivers. Changing anything here changes every issue, so
treat it the way config.py is treated: deliberately, never in a refactor.

Tier semantics are inherited from the three-tier evidence contract
(docs/pilot.md): "verified" = citation contract + adversarial checker,
"data" = seeded reproducible artifact, "coach" = labeled judgment. A department
may not override its tier — costume never earns the badge (STYLEv3 §10).
"""

# (id, title, promise, tiers, needs_copy, byline)
# `tiers`: badge(s) the section renders; () = structural, no evidence claim.
# `needs_copy`: section requires kicker/headline/dek from the issue plan.
# `byline`: the signing columnist(s), reader-facing; None = unsigned furniture.
#
# Order is the STYLEv3 §5 five-act flight plan (v3.4 amendment): meet the deck,
# fly it, work the table, show the work, and leave the proof in the appendix.
# Position in this literal IS the canonical order — nothing else encodes it.
# Promises are written in the signing columnist's voice; the renderer prints
# them verbatim in the Flight Plan.
#
# v3.4 replaced a monotonic depth ramp with an identity-first one, on the
# founder's reading of the shipped issues (docs/magazine-feedback-2026-08.md).
# The argument, in his words: "all commanders are built around a commander —
# when you ask somebody what deck are you playing, they lead with who the
# commander is." The book now opens the way a player hands you their deck. You
# read the commander, you hear the plan, you flip through the 99 — and only THEN
# are you asked to mulligan, because "once somebody has actually had the
# opportunity to look through the 99, those mulligans make a lot more sense."
# The Kill follows the decision spread for the same reason: here is how you fly
# it, and here is what it kills with. Judge's Desk does NOT move — the proof
# still lives at the back, which is the one part of the old depth ramp the
# resequence preserves rather than inverts.
DEPARTMENTS = [
    ("cover", "The Cover",
     "Why should I care about this deck?", (), False, None),
    ("contents", "The Flight Plan",
     "You are here. Everything else is one tap away.", (), False, None),
    # Act I — Meet the Deck: who is in charge, what they want, what they brought.
    ("command-zone", "The Command Zone",
     "Why this commander is exactly where you want to be — on the record.",
     ("verified", "coach"), True,
     "Counselor Vera Dictum with Coach Sunny Brightside"),
    ("first-turns", "The Game Plan",
     "What this deck wants to do — and why it's going to work.",
     ("coach",), True, "Coach Sunny Brightside"),
    ("the-99", "The 99",
     "Roll call. Every card earns its seat — or hears about it.",
     ("coach",), True, "Coach Sunny Brightside"),
    # Act II — Fly It: the hand, the hard call, the kill.
    ("keep-or-ship", "Keep or Ship",
     "Seven cards, one call. The Coach trusts your gut; Ledger brought receipts.",
     ("coach", "data"), True, 'Coach Sunny Brightside with "Ledger" Lin Marginal'),
    ("whats-your-play", "What's Your Play?",
     "Real board, real stakes. Commit before the Coach shows his hand.",
     ("coach",), True, "Coach Sunny Brightside"),
    ("the-kill", "The Kill",
     "The winning lines, argued and affirmed. Every step on the record.",
     ("verified",), True, "Counselor Vera Dictum"),
    # Act III — At the Table: tactics against three live opponents. Pure Coach.
    ("politics-table", "Table Manners",
     "Three opponents, one you. How to win friends and eliminate people.",
     ("coach",), True, "Coach Sunny Brightside"),
    ("know-your-enemy", "Know Your Enemy",
     "The decks that want you dead, and how to disappoint them.",
     ("coach",), True, "Coach Sunny Brightside"),
    ("fetch-quests", "Fetch Quests",
     "You get one wish per tutor. Here's how not to waste it.",
     ("coach",), True, "Coach Sunny Brightside"),
    # Act IV — Show Your Work: the mana, the stats, the future. Pure Ledger.
    ("sources-say", "Sources Say",
     "Pips versus sources — does this mana base keep its promises?",
     ("data",), True, '"Ledger" Lin Marginal'),
    ("by-the-numbers", "By the Numbers",
     "Ten thousand opening hands don't lie.",
     ("data",), True, '"Ledger" Lin Marginal'),
    # The id stays `upgrade-watch`: it is the stable key `validate_issue`, the act
    # table and every rendered manual are pinned to. The section is The Short List.
    ("upgrade-watch", "The Short List",
     "Ten cards worth knowing about. Whether you own them is your business.",
     ("data",), True, '"Ledger" Lin Marginal'),
    # Act V — The Appendix: the proof, the paint, the door out.
    ("judges-desk", "Judge's Desk",
     "The full case files. The Counselor read them twice.",
     ("verified",), True, "Counselor Vera Dictum"),
    ("featured-artist", "Featured Artist",
     "The hands that painted your deck — counted and credited.",
     ("data", "coach"), True, '"Ledger" Lin Marginal'),
    ("back-page", "The Back Page",
     "The next flight leaves soon.", (), False, None),
]

# The five acts (STYLEv3 §5): the Flight Plan groups its rows under these
# headers, in this order. Every section after cover/contents belongs to
# exactly one act — a test asserts the flattened acts equal DEPARTMENT_IDS.
#
# Acts III and IV are single-voice by construction, which the old grouping never
# managed: three consecutive Coach sections, then three consecutive Ledger ones.
ACTS = [
    ("Meet the Deck", ("command-zone", "first-turns", "the-99")),
    ("Fly It", ("keep-or-ship", "whats-your-play", "the-kill")),
    ("At the Table", ("politics-table", "know-your-enemy", "fetch-quests")),
    ("Show Your Work", ("sources-say", "by-the-numbers", "upgrade-watch")),
    ("The Appendix", ("judges-desk", "featured-artist", "back-page")),
]

# The masthead trio (STYLEv3 §7.7). Reprinted in In This Issue every issue;
# the renderer and the magazine-editor both read this — never restate it.
MASTHEAD_COLUMNISTS = [
    {"tier": "data", "glyph": "◆", "name": '"Ledger" Lin Marginal',
     "bio": "Staff quant. Ran it 10,000 times so you don't have to."},
    {"tier": "verified", "glyph": "✓", "name": "Counselor Vera Dictum",
     "bio": "Rules attorney. Reads the Comprehensive Rules for pleasure. Twice."},
    {"tier": "coach", "glyph": "★", "name": "Coach Sunny Brightside",
     "bio": "The corner office. Has never once believed you're going to lose."},
]

# Sections followed by a renderer-emitted full-bleed breather spread (STYLEv3
# §6, v3.3): the dense-adjacency check skips the pair on either side of a
# declared breather, because the reader gets an art break between them.
BREATHER_AFTER = frozenset({"sources-say"})

DEPARTMENT_IDS = [d[0] for d in DEPARTMENTS]
DEPARTMENT_BY_ID = {d[0]: {"title": d[1], "promise": d[2], "tiers": d[3],
                           "needs_copy": d[4], "byline": d[5]}
                    for d in DEPARTMENTS}

# Departments the magazine-editor agent must supply packaging copy for.
COPY_DEPARTMENTS = [d[0] for d in DEPARTMENTS if d[4]]

# Rhythm tags (STYLEv3 §6). Used to check that dense departments alternate.
INTENSITY = {
    "cover": "peak", "contents": "low", "first-turns": "high",
    "command-zone": "medium", "by-the-numbers": "medium", "the-kill": "peak",
    "politics-table": "medium", "whats-your-play": "high",
    "know-your-enemy": "medium", "fetch-quests": "medium",
    "sources-say": "medium", "the-99": "low", "featured-artist": "low",
    "keep-or-ship": "medium",
    "upgrade-watch": "low", "judges-desk": "low", "back-page": "low",
}

# Cognitive mode — two "dense" departments must never sit adjacent.
DENSE_MODES = {"analysis", "reference", "technical"}
MODE = {
    "cover": "anticipation", "contents": "orientation", "first-turns": "narrative",
    "command-zone": "instruction", "by-the-numbers": "analysis",
    # The Kill carries technical content but *reads* as narrative — that is
    # precisely what makes it the breather after By the Numbers (STYLEv3 §6).
    "the-kill": "narrative", "politics-table": "reflection",
    "whats-your-play": "participation", "know-your-enemy": "reference",
    "fetch-quests": "instruction", "sources-say": "analysis",
    "the-99": "browsing", "featured-artist": "appreciation", "keep-or-ship": "practice",
    "upgrade-watch": "imagination", "judges-desk": "reference",
    "back-page": "closure",
}

# The component library (STYLEv3 §8.4). The agent composes from this fixed set
# and may not invent furniture.
COMPONENTS = {
    "violator", "pilot-tip", "fast-facts", "power-meter", "callout-step",
    "threat-box", "scenario-box", "dossier-file", "pull-quote",
    "folio", "tax-ladder", "artist-gallery",
}

# Issue identity fields required in data/decks/<slug>/issue.json.
# Authored, never generated — a generated date breaks byte-identical rebuilds
# (STYLEv3 §4.2). decklist_sha256 pins the issue to the decklist it was
# published from: the volume number is presentation, the hash is identity, and
# validate-issue asserts it still matches cards.json — the "manual rebuilt
# against a changed decklist without a new issue" failure is caught in form.
REQUIRED_ISSUE_KEYS = {
    "volume", "issue_date", "cover_price", "deck_name", "commander",
    "cover_tagline", "next_issue", "decklist_sha256",
}

# Departments with bespoke layouts that take no per-department furniture.
# The cover's bursts live in the plan's top-level `cover` block; the contents
# page is a generated table. Every other department renders whatever furniture
# the plan gives it, so the validator rejects furniture here rather than
# letting the renderer drop it silently.
NO_FURNITURE_DEPARTMENTS = frozenset({"cover", "contents"})
FURNITURE_KEYS = ("pilot_tips", "captions", "callouts", "pull_quote")

MASTHEAD = "MANA MAP"
SERIES_SLUG = "PILOT'S MANUAL"
STANDING_TAGLINE = "THE INSIDE SOURCE FOR YOUR COMMAND ZONE"
