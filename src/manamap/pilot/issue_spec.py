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
    #
    # The editor's letter opens the book — it is a welcome and a preview, and it
    # asks the reader to know nothing yet. Everything after it teaches the deck,
    # and only THEN do the three pilots argue about flying it.
    #
    # The Pilot's Log used to sit second, ahead of The Command Zone, and that was
    # wrong for a reason worth keeping written down: a three-way argument is the
    # densest thing in the issue and it is *about* material the reader has not met.
    # Sunny's hot take lands as a contrarian claim about a deck you have not seen,
    # Vera's "that is not established" refers to a line you have not read, and
    # Ledger prices an engine you could not name. Behind The 99 all three land,
    # because by then the reader has met the commander, heard the plan and flipped
    # through the roster — which is exactly the order in which someone hands you
    # their deck and then starts arguing with their friends about it.
    #
    # Both front-of-book pieces are OPTIONAL_DEPARTMENTS while they are piloted on
    # one deck — see that constant for why, and remove them once every issue
    # carries them.
    ("editors-letter", "The Editor's Letter",
     "What this deck is, and whether it is for you.",
     (), True, "Editor-in-Chief Margot Stet"),
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
    # Tier is ("coach",) and NOT all three, which is the load-bearing decision. A
    # department's tier is what the department GRANTS, not what its speakers
    # mention. Vera will cite a ruling and Ledger a rate, but both were granted
    # their badges in The Kill and By the Numbers; the panel points at them. Give
    # this ("verified", "data", "coach") and a conversation becomes a place where
    # a new verified claim can be smuggled in wearing three voices at once.
    ("pilots-log", "The Pilot's Log",
     "Three pilots, one deck, and an argument about how to fly it.",
     ("coach",), True,
     'Coach Sunny Brightside, Counselor Vera Dictum and "Ledger" Lin Marginal'),
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
    #
    # ONE department where there were three, because three consecutive Coach
    # openers were three answers to the same question — who wants you dead, how
    # they come at you, and what you go and get about it — and the reader met the
    # same byline, the same colour and the same page furniture three times in a
    # row. The act header already said "At the Table"; the departments under it
    # were subheads pretending to be destinations. Merged, the tactics read as one
    # argument and the issue loses two full openers.
    #
    # The three originals stay in the spec and are OPTIONAL, because eight issues
    # are built against them and a department cannot be changed on one deck any
    # other way. When every deck carries `at-the-table`, delete them — and the
    # ones that outlive their migration are the ones nobody finished.
    ("at-the-table", "At the Table",
     "Three opponents, one you. Who wants you dead, and what you go get about it.",
     ("coach",), True, "Coach Sunny Brightside"),
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
    ("Meet the Deck", ("editors-letter", "command-zone", "first-turns",
                       "the-99", "pilots-log")),
    ("Fly It", ("keep-or-ship", "whats-your-play", "the-kill")),
    ("At the Table", ("at-the-table", "politics-table",
                      "know-your-enemy", "fetch-quests")),
    ("Show Your Work", ("sources-say", "by-the-numbers", "upgrade-watch")),
    ("The Appendix", ("judges-desk", "featured-artist", "back-page")),
]

# The masthead trio (STYLEv3 §7.7). Reprinted in In This Issue every issue;
# the renderer and the magazine-editor both read this — never restate it.
# The editor-in-chief carries **no tier and no glyph**, and that is the whole
# design rather than an omission. Each of the three columnists owns exactly one
# evidence tier, and the badge means what it means because a voice cannot grant
# itself one (§10). A fourth signer holding a badge would make four tiers out of
# three; a fourth holding one of the existing three would put two names on it. So
# the editor introduces and the other three testify — which is also what an
# editor-in-chief actually does.
#
# `badge()` raises on an unknown tier, deliberately: nothing should ever ask this
# entry for one, and a KeyError is a better answer than a blank span.
MASTHEAD_COLUMNISTS = [
    {"tier": None, "glyph": None, "name": "Editor-in-Chief Margot Stet",
     "bio": "Decides what runs. Holds no badge, which is how you know the other "
            "three earned theirs."},
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

# Departments an issue MAY carry but is not required to.
#
# The spec has had no such concept until now, and adding one is a real cost — a
# reader's "fixed reading experience every issue delivers" (§5) is weakened by
# every optional slot. It exists for exactly one reason: a new department is
# otherwise a fleet-wide event. The moment an id lands in DEPARTMENTS, every
# existing issue plan is missing it and fails `validate-issue`, so a department
# cannot be piloted on one deck — it arrives on nine at once or not at all.
#
# So: optional means "an older plan without it stays valid", NOT "the editor may
# skip it". Once a deck's plan carries the department it is validated exactly like
# any other, and an id should be REMOVED from this set as soon as every deck has
# it. A permanently optional department is a department nobody committed to.
OPTIONAL_DEPARTMENTS = frozenset({
    "editors-letter", "pilots-log",
    # The Act III merge, mid-migration in both directions: radagast carries
    # `at-the-table`, the other eight still carry the three it replaces, and
    # neither set may be required while both exist.
    "at-the-table", "politics-table", "know-your-enemy", "fetch-quests",
})

DEPARTMENT_IDS = [d[0] for d in DEPARTMENTS]
DEPARTMENT_BY_ID = {d[0]: {"title": d[1], "promise": d[2], "tiers": d[3],
                           "needs_copy": d[4], "byline": d[5]}
                    for d in DEPARTMENTS}

# Departments the magazine-editor agent must supply packaging copy for.
COPY_DEPARTMENTS = [d[0] for d in DEPARTMENTS if d[4]]

# Rhythm tags (STYLEv3 §6). Used to check that dense departments alternate.
INTENSITY = {
    "cover": "peak", "contents": "low", "first-turns": "high",
    "editors-letter": "low", "pilots-log": "high",
    "command-zone": "medium", "by-the-numbers": "medium", "the-kill": "peak",
    "at-the-table": "medium",
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
    # A letter is someone talking to you; a panel is people talking to each other.
    # Neither is dense, which matters: they sit ahead of The Command Zone's
    # instruction and give the reader two soft entries before the first hard one.
    "editors-letter": "welcome", "pilots-log": "conversation",
    "command-zone": "instruction", "by-the-numbers": "analysis",
    # The Kill carries technical content but *reads* as narrative — that is
    # precisely what makes it the breather after By the Numbers (STYLEv3 §6).
    "the-kill": "narrative", "at-the-table": "reflection",
    "politics-table": "reflection",
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
    # `coach-gauge` is `power-meter`'s ★ counterpart: the same shape of claim on a
    # five-point scale with no printed percentage, because a percentage asserts a
    # measurement. `stat-slab` runs the issue's signature number once, full width.
    "coach-gauge", "stat-slab",
    # The two pictures: how the deck RUNS, and what shape it is.
    "engine-flow", "constellation",
    # `letterhead` is the Editor's Letter's own furniture and belongs to no other
    # department — a second page dressed as a letter is a second editor.
    # `stack-theatre` is The Kill's: a stack you move through rather than read.
    "letterhead", "stack-theatre",
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

# Which department renders which `manual_prose.json` key. The other half of this
# correspondence lives in `build_manual`'s renderers, which name the key inline;
# `tests/test_pilot_voice_lint.py` asserts the two agree, because a key that
# drifts out of this map stops being voice-checked SILENTLY — the lint would
# simply find nothing and pass.
#
# It exists so `validate-issue` can ask "who is supposed to be speaking here".
# `pilots_log` is absent on purpose: its turns carry their own `voice`, which is a
# better answer than a department byline and the reason that shape was chosen.
PROSE_KEY_DEPARTMENT = {
    "how_it_wins": "first-turns",
    "combo_lines": "the-kill",
    "card_roles": "the-99",
    "mulligan": "keep-or-ship",
    "threat_assessment": "politics-table",
    "matchups": "know-your-enemy",
    "mana_base": "sources-say",
    "upgrades": "upgrade-watch",
    "editors_letter": "editors-letter",
}


def voices_for(prose_key):
    """The masthead names a prose key is written under, from its department byline.

    DERIVED, not listed: a byline edit or a fourth columnist updates this with no
    second place to remember. A shared department ("Coach … with 'Ledger' …")
    returns both, and the lint then flags only what BOTH voices are barred from —
    a word banned for Sunny is not an error in a department where Ledger also
    speaks, because Ledger may have written that sentence.
    """
    dept = PROSE_KEY_DEPARTMENT.get(prose_key)
    if dept is None:
        return ()
    byline = (DEPARTMENT_BY_ID[dept]["byline"] or "")
    return tuple(c["name"] for c in MASTHEAD_COLUMNISTS if c["name"] in byline)


MASTHEAD = "MANA MAP"
SERIES_SLUG = "PILOT'S MANUAL"
# The same words set in running case, for the places that are not a cover: the
# <title> tag, og:title, the newsstand. Authored rather than derived because
# `"PILOT'S MANUAL".title()` is `"Pilot'S Manual"` — Python capitalises after an
# apostrophe, and a title-cased constant is cheaper than remembering that.
SERIES_TITLE = "Pilot's Manual"
STANDING_TAGLINE = "THE INSIDE SOURCE FOR YOUR COMMAND ZONE"
