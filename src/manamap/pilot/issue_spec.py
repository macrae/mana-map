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

# (id, title, promise, tiers, needs_copy)
# `tiers`: badge(s) the department renders; () = structural, no evidence claim.
# `needs_copy`: department requires kicker/headline/dek from the issue plan.
#
# Order is the STYLEv3 §5 three-act arc (v3.1 amendment): the Coach opens with
# the thesis, the Counselor makes the case, the Quant runs the numbers, the
# Coach takes you back to the table, the appendix holds the proof. Position in
# this literal IS the canonical order — nothing else encodes it.
DEPARTMENTS = [
    ("cover", "The Cover",
     "Why should I care about this deck?", (), False),
    ("contents", "In This Issue",
     "Where am I, and how do I read this?", (), False),
    ("first-turns", "First Turns",
     "What is this deck actually trying to do?", ("coach",), True),
    ("command-zone", "The Command Zone",
     "Why this commander — and what does the format change?", ("verified", "coach"), True),
    ("the-kill", "The Kill",
     "How does this deck actually win?", ("verified",), True),
    ("by-the-numbers", "By the Numbers",
     "What can I expect, turn by turn?", ("data",), True),
    ("keep-or-ship", "Keep or Ship",
     "Should I keep this hand?", ("coach", "data"), True),
    ("upgrade-watch", "Upgrade Watch",
     "What's next for this deck?", ("data",), True),
    ("featured-artist", "Featured Artist",
     "Who painted your deck?", ("data", "coach"), True),
    ("politics-table", "The Politics Table",
     "How do I survive three opponents?", ("coach",), True),
    ("whats-your-play", "What's Your Play?",
     "What would you do here?", ("coach",), True),
    ("know-your-enemy", "Know Your Enemy",
     "Who beats me, and why?", ("coach",), True),
    ("the-99", "The 99",
     "Why is each card in here?", ("coach",), True),
    ("judges-desk", "Judge's Desk",
     "Prove it.", ("verified",), True),
    ("back-page", "The Back Page",
     "What's in the next issue?", (), False),
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

DEPARTMENT_IDS = [d[0] for d in DEPARTMENTS]
DEPARTMENT_BY_ID = {d[0]: {"title": d[1], "promise": d[2], "tiers": d[3], "needs_copy": d[4]}
                    for d in DEPARTMENTS}

# Departments the magazine-editor agent must supply packaging copy for.
COPY_DEPARTMENTS = [d[0] for d in DEPARTMENTS if d[4]]

# Rhythm tags (STYLEv3 §6). Used to check that dense departments alternate.
INTENSITY = {
    "cover": "peak", "contents": "low", "first-turns": "high",
    "command-zone": "medium", "by-the-numbers": "medium", "the-kill": "peak",
    "politics-table": "medium", "whats-your-play": "high",
    "know-your-enemy": "medium", "the-99": "low", "featured-artist": "low",
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
    "the-99": "browsing", "featured-artist": "appreciation", "keep-or-ship": "practice",
    "upgrade-watch": "imagination", "judges-desk": "reference",
    "back-page": "closure",
}

# The component library (STYLEv3 §8.4). The agent composes from this fixed set
# and may not invent furniture.
COMPONENTS = {
    "violator", "pilot-tip", "fast-facts", "power-meter", "callout-step",
    "threat-box", "scenario-box", "dossier-file", "map-key", "pull-quote",
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
