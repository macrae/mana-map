"""Pilot: mechanically enforce form on an issue plan (STYLEv3 §11).

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

The `magazine-editor` agent writes packaging decisions and copy as structured
data; this module is the gate that runs before the renderer. Same philosophy as
validate_stack.py: code enforces *form*, humans judge *substance*.

Checks:
- issue.json carries the full authored identity block (no generated dates)
- every department in the canonical list is present, in canonical order
- copy departments carry kicker + headline + dek
- the cover promises something specific (coverline + >=1 tease)
- components come from the fixed library
- tier costume is never overridden (a department can't claim a badge the
  department system doesn't grant it)
- pilot tips and captions name cards that actually exist in the deck
- a featured artist actually painted a card in the deck
"""

import json

from manamap.pilot.common import (deck_dir, load_deck_cards, presentable,
                                   report_errors)
from manamap.pilot.issue_spec import (
    ISSUE_STATUSES,
    MASTHEAD_COLUMNISTS,
    voices_for,
    OPTIONAL_DEPARTMENTS,
    BREATHER_AFTER,
    COMPONENTS,
    DENSE_MODES,
    FURNITURE_KEYS,
    NO_FURNITURE_DEPARTMENTS,
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    MODE,
    REQUIRED_ISSUE_KEYS,
    PROSE_BUDGET,
    ENTRY_BUDGET,
    BRANCH_BUDGET,
    MAX_DEK_SENTENCES,
    MAX_CALLOUT_SENTENCES,
    MAX_PILOT_TIP_SENTENCES,
)

REQUIRED_PLAN_KEYS = {"slug", "angle", "cover", "departments"}
REQUIRED_COPY_KEYS = {"kicker", "headline", "dek"}
MAX_VIOLATORS_PER_SPREAD = 2


def validate_identity(issue, deck_sha256=None):
    """Check data/decks/<slug>/issue.json. Returns error strings.

    `deck_sha256` is cards.json's decklist hash when available: the issue's
    stamped hash must match it, or the manual is being rebuilt against a
    decklist this issue never described.
    """
    errors = []
    missing = REQUIRED_ISSUE_KEYS - set(issue)
    if missing:
        errors.append(f"issue.json missing keys: {sorted(missing)}")
    volume = issue.get("volume")
    if not isinstance(volume, int) or volume < 1:
        errors.append(f"issue.json volume must be a positive integer, got {volume!r}")
    # `status` is optional, but a value the renderer does not know silently
    # renders NOTHING — the deck reads as live when someone meant to retire it.
    # The renderer tolerates it (a typo must not take a magazine offline); this
    # is where it gets reported.
    status = issue.get("status")
    if status is not None and status not in ISSUE_STATUSES:
        errors.append(
            f"issue.json status {status!r} is not one of "
            f"{sorted(ISSUE_STATUSES)} — the banner will not render"
        )
    stamped = issue.get("decklist_sha256")
    if deck_sha256 and stamped and stamped != deck_sha256:
        errors.append(
            f"issue.json decklist_sha256 {stamped[:12]} does not match cards.json "
            f"{deck_sha256[:12]} — the decklist changed after this issue was "
            f"published; stamp the new hash (and version the deck) deliberately"
        )
    return errors


def validate_plan(plan, card_names=None, artists=None):
    """Check an issue plan. Returns error strings (empty = form holds).

    `card_names` / `artists` of None mean *skip that class of check* — not
    "the deck has no cards". An empty set is the opposite: it makes every
    caption and every featured artist an error. main() passes None only when
    cards.json is unreadable.
    """
    errors = []

    missing = REQUIRED_PLAN_KEYS - set(plan)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors  # too broken to keep going

    if not plan.get("angle"):
        errors.append("angle is required — every issue is about one idea (STYLEv3 §11)")

    # Cover must promise something specific.
    cover = plan.get("cover") or {}
    if not cover.get("dominant_coverline"):
        errors.append("cover.dominant_coverline is required")
    teases = cover.get("teases") or []
    if not teases:
        errors.append("cover.teases must name at least one specific thing in the issue")
    violators = cover.get("violators") or []
    if len(violators) > MAX_VIOLATORS_PER_SPREAD:
        errors.append(
            f"cover has {len(violators)} violators — max {MAX_VIOLATORS_PER_SPREAD} (STYLEv3 §8.4)"
        )

    # Departments: complete and in canonical order.
    departments = plan.get("departments") or []
    seen = [d.get("id") for d in departments]
    unknown = [i for i in seen if i not in DEPARTMENT_BY_ID]
    if unknown:
        errors.append(f"unknown department id(s): {unknown}")
    # An optional department may be absent from an older plan; everything else is
    # required. See `issue_spec.OPTIONAL_DEPARTMENTS` for why the concept exists
    # and why an id should not stay in it.
    required = [i for i in DEPARTMENT_IDS if i not in OPTIONAL_DEPARTMENTS]
    absent = [i for i in required if i not in seen]
    if absent:
        errors.append(
            f"missing department(s): {absent} — all {len(required)} "
            f"render every issue")
    ordered = [i for i in seen if i in DEPARTMENT_BY_ID]
    if ordered != [i for i in DEPARTMENT_IDS if i in ordered]:
        errors.append("departments are out of canonical order (STYLEv3 §5)")

    for dept in departments:
        dept_id = dept.get("id")
        spec = DEPARTMENT_BY_ID.get(dept_id)
        if spec is None:
            continue
        where = f"department {dept_id}"

        if spec["needs_copy"]:
            missing_copy = REQUIRED_COPY_KEYS - {k for k in REQUIRED_COPY_KEYS if dept.get(k)}
            if missing_copy:
                errors.append(f"{where}: missing {sorted(missing_copy)}")

        for component in dept.get("components", []):
            if component not in COMPONENTS:
                errors.append(f"{where}: unknown component {component!r}")

        # Furniture the renderer would drop must be rejected, not accepted.
        if dept_id in NO_FURNITURE_DEPARTMENTS:
            carried = [k for k in FURNITURE_KEYS if dept.get(k)]
            if carried:
                errors.append(
                    f"{where}: has a bespoke layout and renders no department "
                    f"furniture — move {sorted(carried)} elsewhere "
                    f"(cover bursts belong in the plan's top-level cover block)"
                )

        # A dek sells the department; it does not interview the reader (STYLEv3 §7.2).
        dek = dept.get("dek")
        if dek and _QUESTION_OPENER_RE.match(dek):
            errors.append(
                f"{where}: dek opens by asking the reader a question — "
                f"{dek.split('?')[0][:60]!r}?. Open on a moment instead; six of "
                f"these in one issue reads as a formula, not as six ideas."
            )

        # Costume never earns the badge (STYLEv3 §10).
        claimed = dept.get("tiers")
        if claimed is not None and tuple(claimed) != tuple(spec["tiers"]):
            errors.append(
                f"{where}: claims tiers {tuple(claimed)} but the department system "
                f"grants {spec['tiers']} — a department may not restyle its evidence tier"
            )

        if card_names is not None:
            for tip in dept.get("pilot_tips", []):
                card = tip.get("card")
                if card and card not in card_names:
                    errors.append(f"{where}: PILOT TIP names {card!r}, not in the deck")
                if not tip.get("text"):
                    errors.append(f"{where}: PILOT TIP for {card!r} has no text")
            for card in (dept.get("captions") or {}):
                if card not in card_names:
                    errors.append(f"{where}: caption names {card!r}, not in the deck")
            for group in dept.get("roster", []):
                for card in group.get("cards", []):
                    if card not in card_names:
                        errors.append(
                            f"{where}: roster group {group.get('role', '?')!r} names "
                            f"{card!r}, not in the deck"
                        )

        if artists is not None:
            named = [(dept.get("featured") or {}).get("artist")]
            named += [o.get("artist") for o in dept.get("also_worth_noting", [])]
            for artist in [a for a in named if a]:
                if artist not in artists:
                    errors.append(
                        f"{where}: names artist {artist!r}, who painted no card in "
                        f"this deck"
                    )

    # Rhythm: no two dense sections adjacent (STYLEv3 §6) — unless the
    # renderer emits a declared full-bleed breather spread between them.
    for a, b in zip(ordered, ordered[1:]):
        if a in BREATHER_AFTER:
            continue
        if MODE.get(a) in DENSE_MODES and MODE.get(b) in DENSE_MODES:
            errors.append(
                f"rhythm: {a} ({MODE[a]}) and {b} ({MODE[b]}) are both dense and adjacent — "
                f"insert a breather (STYLEv3 §6)"
            )

    return errors


# ── Self-containment (STYLEv3 L10): every issue is the reader's first ────

# Patterns that mark changelog voice. Deliberately narrow: "swap"/"wave"/
# "benched" are legitimate Commander vocabulary and are handled editorially,
# not mechanically. These four have no innocent reading in reader-facing copy.
import re

_CONTINUITY_RE = re.compile(
    r"\bv[1-9]\b"                       # "v2's answer", "V3 added"
    r"|HISTORY\.md"
    r"|\bprevious (?:version|build|list)\b"
    r"|\bearlier build\b"
    r"|\bsuperseded\b",
    re.IGNORECASE,
)

# Internal taxonomy ids, in copy a reader sees. `strategy:multiplayer.pod-management`
# is how an agent addresses the strategy DB; it is not English and it is not a
# citation the reader can follow — the manual has no strategy bibliography, so the
# tag resolves to nothing on the page. Every prose agent is told to GROUND claims in
# strategy sections, and the tag is what grounding looks like in the agent's own
# reasoning, so it leaks by a very natural mistake and nothing caught it: 68
# occurrences reached the rendered HTML of all eight published issues before this
# existed (docs/history/magazine-feedback-2026-08-13.md §2).
#
# Matched anywhere, not just in parentheses. Every live occurrence happened to be a
# trailing parenthetical, which made the cleanup mechanical — but a rule written to
# the shape of the instances found once would miss the next one that arrives inline.
_TAXONOMY_RE = re.compile(r"\bstrategy:[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)*")

# A dek that opens by asking the reader a question. Six of Vol. 009's departments
# did, which reads as a formula rather than as six separate ideas — and a magazine
# that opens every section by posing a question is teaching the reader that the
# answer is always three sentences away. Open on a moment instead.
#
# Anchored to the START of the dek only: a question later in the copy is rhetoric,
# and a question in a HEADLINE is a different device this does not govern.
_QUESTION_OPENER_RE = re.compile(
    r"^\s*(?:What|When|Which|Why|How|Who|Where|Is|Are|Does|Do|Can|Should)\b[^?]{0,200}\?"
)

# Plan fields the reader never sees; everything else in a department is copy.
#
# `gaps` and `rhythm_notes` are the editor talking to the next editor — what the
# strategy DB was missing, why a spread sits where it does — and `build_manual`
# renders neither (grep it: there is no reader-facing use of either key). They are
# exactly where naming a `strategy:` id is CORRECT, so linting them as reader copy
# flags the one honest note in the file.
_EDITOR_ONLY_PLAN_KEYS = {"note", "components", "id", "tiers", "gaps", "rhythm_notes"}


def _walk_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


# Constructions a voice may not use, per STYLEv3 §7.7. Narrow on purpose: each
# entry is a word that has no innocent reading in that columnist's register, not a
# word that is merely uncommon there.
#
# The evidence this exists on: an editor read Vol. 009 and found Coach Sunny
# Brightside — whose bio is "has never once believed you're going to lose" —
# writing "the deflection posture the strategic frame prescribes". The verdict was
# "that's not a coach, that's a McKinsey deck", and the diagnosis underneath it was
# that with the bylines covered no reader could attribute a paragraph. A lint
# cannot check whether prose sounds like a person; it can check the specific words
# that made it sound like nobody.
_VOICE_BANS = {
    "Coach Sunny Brightside": (
        # Consulting register. Sunny talks about what you DO.
        "posture", "prescribes", "prescribed", "framework", "optimise", "optimize",
        "suboptimal", "methodology", "in terms of", "strategic frame",
    ),
    '"Ledger" Lin Marginal': (
        # Intensifiers and evaluative adjectives. For Ledger a number is the
        # adjective; "a huge 40.2%" says less than "40.2%".
        # EVALUATIVE ADJECTIVES ONLY. The first version also banned the
        # intensifiers "very", "really" and "extremely", and measuring it against
        # the fleet killed them: hapatra's Ledger writes "a number the deck does
        # not really have", which is a HEDGE and correct — the ban's rationale is
        # that a number is the adjective, and a hedge is not an adjective. An
        # intensifier and a hedge are the same word, so no regex separates them.
        # These six have no hedging reading.
        "huge", "incredible", "terrible", "amazing", "massive", "insane",
    ),
}

# Matched on WORD BOUNDARIES, and the first version was not. `"very "` as a
# substring matches "e-very " — so the lint's first run reported 13 violations
# across the fleet of which every `very` hit was the word "every", in sentences
# that were correct. That is the failure this repo has now written down five
# times: a validator that fires on accurate data teaches its reader to ignore it,
# and I shipped one anyway within an hour of documenting the rule. Multi-word
# entries ("in terms of") still work — `\b` binds to the outer words.
_VOICE_BAN_RE = {
    voice: {b: __import__("re").compile(r"\b" + __import__("re").escape(b) + r"\b",
                                        __import__("re").IGNORECASE)
            for b in bans}
    for voice, bans in _VOICE_BANS.items()
}


def _voice_violations(voice, text):
    for banned, pattern in _VOICE_BAN_RE.get(voice, {}).items():
        if pattern.search(text):
            yield banned


# A citation's `rule` field is where a taxonomy id BELONGS: it is the structured
# half of the citation contract, the renderer never prints it raw for a strategy
# citation (only `CR <n>` for rules citations), and a decision branch citing a
# strategy section is the contract working exactly as designed.
#
# This skip is the difference between a validator and a nuisance. Without it the
# taxonomy rule fires on 51 correct decision citations across four decks, and a
# check that fails on accurate data teaches everyone to ignore red — which is the
# same lesson three rejected `validate-diagnosis` proposals were killed over.
def _is_citation_id(path):
    return path.endswith(".rule") and ".citations[" in path


# A "the Command Zone must not teach the format" lint was prototyped here and
# REJECTED on measurement, which is the rule this repo keeps breaking and then
# re-learning. Eight patterns were run against all nine decks' `command-zone.body`:
#
#   the command zone is where…        0 hits      commander tax is/means…   0 hits
#   your commander is a legendary…    0 hits      singleton means…          0 hits
#   in commander you start with…      0 hits      when your commander dies… 0 hits
#   every/any commander deck…         0 hits      in this/the format        2 hits
#
# The seven targeted patterns find nothing because the defect is not phrased that
# way anywhere in the corpus, and the one that hits twice (gishath, ur-dragon)
# matches a clause that is fine in isolation. Meanwhile the defect is real and sits
# in copy that scores clean: radagast opens "Your commander begins the game in the
# command zone and is the only card you always have (CR 903.4)" — true, cited,
# well-written, and a lesson for a reader who has played this format for years.
#
# No regex separates "explaining the format" from "citing a rule about this
# commander", because they are the same sentence with a different subject. The rule
# is editorial and lives in STYLEv3 §3.3 and the magazine-editor's charter. A
# validator that fires on correct data teaches everyone to ignore red.

# The Coach's byline, matched loosely on the surname so a masthead rename does not
# silently disable the check. Derived from MASTHEAD_COLUMNISTS rather than typed:
# the ★ tier is what makes a voice the Coach, and there is exactly one.
_COACH = next((c["name"] for c in MASTHEAD_COLUMNISTS if c.get("tier") == "coach"),
              "")


def _hot_take_errors(turns):
    """The panel opens on a hot take, and somebody answers it.

    Three mechanical checks and no semantic ones. Whether a take is genuinely
    counter-intuitive, correct and insightful is a judgment — it belongs in the
    charter and in an editor's read, not in a regex, and the last three checks
    this repo tried to write about *meaning* were all rejected on measurement.

    What IS checkable is the structure the department depends on: the opener is
    marked, it is the Coach's, and at least one later turn is explicitly a reply
    to it. That third one is the load-bearing check — a hot take nobody answers is
    not a conversation, it is an epigraph, and the whole reason this department
    exists is that a disagreement makes three voices argue instead of alternate.
    """
    errors = []
    first = turns[0] or {}
    if first.get("kind") != "hot-take":
        errors.append(
            "pilots_log[0]: the panel must open on the hot take — set "
            '"kind": "hot-take" (STYLEv3 §5, department 7)')
    elif _COACH and _COACH not in str(first.get("voice") or ""):
        errors.append(
            f"pilots_log[0]: the hot take is {_COACH}'s, not "
            f"{first.get('voice')!r} — it is a ★ judgment, and the other two "
            f"answer it")
    if not any((t or {}).get("responds_to") == "hot-take" for t in turns[1:]):
        errors.append(
            'pilots_log: no later turn carries "responds_to": "hot-take" — a take '
            "nobody answers is an epigraph, not a conversation")
    for i, turn in enumerate(turns[1:], start=1):
        if (turn or {}).get("kind") == "hot-take":
            errors.append(f"pilots_log[{i}]: a second hot take — the department "
                          f"opens on one and argues with it")
    return errors


def _lint_strings(doc, label, skip_key=None):
    errors = []
    for path, text in _walk_strings(doc):
        if skip_key and skip_key(path):
            continue
        match = _CONTINUITY_RE.search(text)
        if match:
            errors.append(
                f"{label} [{path}]: changelog voice — {match.group()!r} "
                f"(STYLEv3 L10: every issue is the reader's first)"
            )
        tag = _TAXONOMY_RE.search(text)
        if tag and not _is_citation_id(path):
            errors.append(
                f"{label} [{path}]: internal taxonomy id in reader copy — "
                f"{tag.group()!r}. Ground the claim, then say it in English; the "
                f"issue has no strategy bibliography for the tag to point at."
            )
    return errors


# -- Land-count truth (the entries-vs-copies trap) -----------------------

# `mana_analysis.lands.entries` counts distinct land CARDS; `lands.total`
# counts physical copies. Eleven Islands are one entry and eleven lands. An
# issue once shipped claiming "18 lands" for a 33-land deck because prose read
# the wrong field, so the number that is never a land count gets linted.
_LAND_COUNT_RE = re.compile(r"\b(\d{1,3})[\s-]lands?\b", re.IGNORECASE)


def validate_land_counts(base, plan):
    """Reader-facing copy may not state the entry count as a land count."""
    path = base / "mana_analysis.json"
    if not path.exists():
        return []
    lands = json.loads(path.read_text()).get("lands", {})
    entries, total = lands.get("entries"), lands.get("total")
    if not entries or entries == total:
        return []  # nothing to confuse

    errors = []
    docs = [(plan, "issue_plan.json")]
    for fname in ("manual_prose.json", "considering.json", "tutor_guide.json"):
        extra = base / fname
        if extra.exists():
            docs.append((json.loads(extra.read_text()), fname))
    for doc, label in docs:
        for where, text in _walk_strings(doc):
            for match in _LAND_COUNT_RE.finditer(text):
                if int(match.group(1)) == entries:
                    errors.append(
                        f"{label} [{where}]: says {match.group()!r}, but "
                        f"{entries} is the count of distinct land CARDS - this "
                        f"deck runs {total} lands. Quote lands.total."
                    )
    return errors


def _editor_only(path):
    """Is this field the editor talking to the next editor, rather than to a reader?

    Shared by every artifact linted below, not just the plan: `gaps` means the same
    thing in `issue_plan.json` and in `considering.json` — an unrendered note about
    what the evidence could not settle — and a rule that knew about only one of them
    fired on the other the moment the lint widened.
    """
    last = path.rsplit(".", 1)[-1].split("[")[0]
    # departments[].note is editor-facing except under featured/also_worth_noting,
    # which build_manual renders. Keep the mechanical rule simple: skip bare
    # `note` only when it is a department-level key.
    return last in _EDITOR_ONLY_PLAN_KEYS and ".featured" not in path \
        and "also_worth_noting" not in path


def validate_features(base, plan):
    """`the-kill.features` must name presentable stacks, and only those.

    The key decides which verified lines get a feature spread and which get an
    index row, so a typo does not fail loudly — it quietly demotes the issue's
    best line to a one-liner and nothing else changes. That is why this is a
    validator rather than a renderer exception: `render_the_kill` skips an unknown
    id on purpose, because a crash here costs the whole magazine.

    A non-presentable id is the sharper error of the two. It means the plan is
    trying to feature a line the publication gate already refused, which is the
    one mistake this department must never make.
    """
    errors = []
    dept = next((d for d in plan.get("departments") or []
                 if d.get("id") == "the-kill"), None)
    if not dept or "features" not in dept:
        return errors

    features = dept["features"]
    if not isinstance(features, list) or not features:
        return ["the-kill.features must be a non-empty list of stack ids — omit "
                "the key entirely to feature every verified line"]

    ids = [str(f) for f in features]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        errors.append(f"the-kill.features repeats {duplicates} — a line gets one "
                      f"spread or one row, never both")

    stacks_dir = base / "stacks"
    available = set()
    for path in sorted(stacks_dir.glob("*.json")) if stacks_dir.is_dir() else []:
        with open(path) as f:
            doc = json.load(f)
        if presentable(doc):
            available.add(str(doc.get("id")))
    if not available:
        return errors                       # no stacks on disk; nothing to check

    for sid in ids:
        if sid not in available:
            errors.append(
                f"the-kill.features names {sid!r}, which is not a presentable "
                f"stack — presentable ids are {sorted(available)}")
    if set(ids) == available and len(available) > 1:
        errors.append(
            "the-kill.features names every presentable stack, which is what "
            "omitting the key already does — drop it rather than restating the "
            "default, or the index below the spreads renders empty")
    return errors


def validate_self_containment(base, plan):
    """Reader-facing text must carry no memory of previous deck versions."""
    errors = []

    errors += _lint_strings(plan, "issue_plan.json", skip_key=_editor_only)

    path = base / "manual_prose.json"
    if path.exists():
        with open(path) as f:
            errors += _lint_strings(json.load(f), "manual_prose.json")

    decisions = base / "decisions"
    if decisions.exists():
        for dec in sorted(decisions.glob("*.json")):
            with open(dec) as f:
                errors += _lint_strings(json.load(f), f"decisions/{dec.name}")

    # The Short List and Fetch Quests are rendered departments whose copy lives
    # outside issue_plan.json, written by their own agents — so they were invisible
    # to this check while being just as reader-facing as everything above. Nine
    # taxonomy ids survived the first cleanup pass in exactly these two files, and
    # the L10 changelog rule had never been applied to them at all.
    # The panel: each turn is attributable, and a voice keeps to its register.
    prose_path = base / "manual_prose.json"
    if prose_path.exists():
        with open(prose_path) as f:
            prose_doc = json.load(f)
        turns = prose_doc.get("pilots_log")
        if isinstance(turns, list) and turns:
            errors += _hot_take_errors(turns)
        if isinstance(turns, list):
            known = {c["name"] for c in MASTHEAD_COLUMNISTS}
            for i, turn in enumerate(turns):
                voice = (turn or {}).get("voice", "")
                text = (turn or {}).get("text", "")
                if voice not in known:
                    errors.append(
                        f"pilots_log[{i}]: voice {voice!r} is not on the masthead — "
                        f"the renderer keys each turn's colour off this name, so a "
                        f"misspelling renders an unowned grey rail")
                for banned in _voice_violations(voice, text):
                    errors.append(
                        f"pilots_log[{i}] ({voice}): uses {banned!r}, which this "
                        f"voice does not say (STYLEv3 §7.7). If the bylines were "
                        f"covered, could a reader still tell who is speaking?")
        # The other departments. One agent wrote six keys under three bylines in
        # ONE pass, which the 2026-08 record named as the structural cause of the
        # magazine reading monovocal — so these are the keys where the check bites.
        # Since 2026-08-19 there is one writer (pilot-notes) in one technical voice;
        # its charter carries these bans, so the legacy lint stays satisfiable.
        for key, text in sorted(prose_doc.items()):
            if key in ("pilots_log", "editors_letter") or not isinstance(text, str):
                continue
            voices = voices_for(key)
            if not voices:
                continue
            # A word is an error only if EVERY named voice is barred from it. In a
            # shared department both columnists speak, so a Sunny-banned word may
            # simply be Ledger's sentence — flagging it would fire on correct copy.
            common = set.intersection(*[set(_VOICE_BANS.get(v, ())) for v in voices]) \
                if voices else set()
            for banned in sorted(common):
                if any(_VOICE_BAN_RE[v][banned].search(text) for v in voices
                       if banned in _VOICE_BAN_RE.get(v, {})):
                    errors.append(
                        f"{key} ({' + '.join(voices)}): uses {banned!r}, which this "
                        f"voice does not say (STYLEv3 §7.7)")

        letter = prose_doc.get("editors_letter")
        if isinstance(letter, str) and letter.strip():
            # Margot Stet holds no badge and may not make a claim that needs one.
            # A bare percentage is the commonest such claim and the only shape a
            # validator can see; a ruling reads like prose and cannot be caught
            # here, which §7.7 says out loud rather than pretending otherwise.
            import re as _re
            for hit in _re.findall(r"\b\d+(?:\.\d+)?\s?%", letter):
                errors.append(
                    f"editors_letter: states {hit!r}. The editor-in-chief carries "
                    f"no tier and may not assert a measured figure — name the "
                    f"columnist who established it instead (STYLEv3 §7.7)")

    for name in ("considering.json", "tutor_guide.json"):
        path = base / name
        if path.exists():
            with open(path) as f:
                errors += _lint_strings(json.load(f), name, skip_key=_editor_only)
    return errors


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if s]


def validate_budget(base, plan):
    """The length budget (STYLEv3 §7.1, `issue_spec.PROSE_BUDGET`).

    Returned SEPARATELY from the form errors and reported rather than failed
    unless `--strict`. The budget arrived after eight issues were written against
    no budget at all, and failing them all on the day it lands would turn eight
    tracked artifacts red for copy that was correct when it shipped — which is how
    a team learns to ignore red. `--strict` is the gate for new work; the plain
    run tells you where you stand.
    """
    notes = []
    for dept in plan.get("departments") or []:
        did = dept.get("id", "?")
        if dept.get("dek"):
            n = len(_sentences(dept["dek"]))
            if n > MAX_DEK_SENTENCES:
                notes.append(f"{did}.dek: {n} sentences (max {MAX_DEK_SENTENCES})")
        for i, c in enumerate(dept.get("callouts") or []):
            n = len(_sentences(c.get("text")))
            if n > MAX_CALLOUT_SENTENCES:
                notes.append(f"{did}.callouts[{i}]: {n} sentences "
                             f"(max {MAX_CALLOUT_SENTENCES})")
        for i, t in enumerate(dept.get("pilot_tips") or []):
            n = len(_sentences(t.get("text")))
            if n > MAX_PILOT_TIP_SENTENCES:
                notes.append(f"{did}.pilot_tips[{i}]: {n} sentences — a PILOT TIP is "
                             f"one imperative sentence (STYLEv3 §7.5)")

    prose_path = base / "manual_prose.json"
    if prose_path.exists():
        with open(prose_path) as f:
            prose_doc = json.load(f)
        for key, cap in sorted(PROSE_BUDGET.items()):
            text = prose_doc.get(key)
            if isinstance(text, str) and len(text) > cap:
                notes.append(f"{key}: {len(text):,} chars (budget {cap:,}) — "
                             f"{len(text) - cap:,} over")
        for key, cap in sorted(ENTRY_BUDGET.items()):
            for sub, text in sorted((prose_doc.get(key) or {}).items()):
                if isinstance(text, str) and len(text) > cap:
                    notes.append(f"{key}[{sub}]: {len(text):,} chars (budget {cap:,})")

    decisions = base / "decisions"
    if decisions.is_dir():
        for path in sorted(decisions.glob("*.json")):
            with open(path) as f:
                doc = json.load(f)
            for i, branch in enumerate(doc.get("branches") or []):
                for key, cap in sorted(BRANCH_BUDGET.items()):
                    text = branch.get(key)
                    if isinstance(text, str) and len(text) > cap:
                        notes.append(f"decisions/{path.name} branch[{i}].{key}: "
                                     f"{len(text):,} chars (budget {cap:,})")
    return notes


def main(args):
    base = deck_dir(args.slug)
    errors = []

    issue_path = base / "issue.json"
    if not issue_path.exists():
        raise SystemExit(
            f"{issue_path} not found — author the issue identity block first "
            f"(volume, issue_date, cover_price, deck_name, commander, "
            f"cover_tagline, next_issue). See STYLEv3 §4.1."
        )
    deck_sha256 = None
    cards_path = base / "cards.json"
    if cards_path.exists():
        with open(cards_path) as f:
            deck_sha256 = json.load(f).get("decklist_sha256")
    with open(issue_path) as f:
        errors += validate_identity(json.load(f), deck_sha256)

    plan_path = base / "issue_plan.json"
    if not plan_path.exists():
        raise SystemExit(
            f"{plan_path} not found — the magazine-editor is retired; this "
            f"validator gates the LEGACY plans on already-published decks only."
        )
    with open(plan_path) as f:
        plan = json.load(f)

    try:
        deck_cards = load_deck_cards(args.slug)["cards"]
        card_names = {c["name"] for c in deck_cards}
        artists = {c["artist"] for c in deck_cards if c.get("artist")}
    except FileNotFoundError:
        card_names = artists = None
        print("WARN cards.json absent — skipping card-name checks")

    errors += validate_plan(plan, card_names, artists)
    errors += validate_features(base, plan)
    errors += validate_self_containment(base, plan)
    errors += validate_land_counts(base, plan)

    budget = validate_budget(base, plan)
    if budget and getattr(args, "strict", False):
        errors += [f"OVER BUDGET — {n}" for n in budget]

    report_errors(f"issue plan for {args.slug}", errors)
    print(
        f"OK   issue plan for {args.slug} — {len(plan['departments'])} departments, "
        f"form holds; angle: {plan['angle'][:60]}"
    )
    if budget:
        print(f"\nBUDGET  {len(budget)} field(s) over the length budget "
              f"(STYLEv3 §7.1) — reported, not failed. `--strict` fails on these.")
        for note in budget:
            print(f"  - {note}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-issue <slug>`.")
