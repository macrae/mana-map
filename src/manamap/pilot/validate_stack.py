"""Pilot: mechanically enforce the citation contract on scenario artifacts.

Two artifact kinds share this gate (`kind` field; missing = "stack"):
- "stack" (stacks/): rules resolutions. A resolution cannot pass with an
  uncited effect, a citation of a nonexistent rule, or a non-verbatim quote.
  The rules-checker agent judges *meaning*; this module enforces *form*.
- "decision" (decisions/): coaching decision trees (tier-3 evidence). Form
  checks only — branches well-shaped, recommendation matches a branch — plus
  the same citation contract for any branch that does cite rules.

Citations dispatch on ID shape: CR rule numbers / glossary terms check against
the rules DB; `strategy:<id>` IDs (tier-* grounding from data/strategy/) check
against the strategy DB under the same verbatim-quote contract. The strategy DB
loads best-effort in main() — its absence only errors if a strategy citation
actually appears.
"""

import json
import re
import sys

from manamap.config import RESOLVE_SCOPE_BUDGET
from manamap.pilot import game_state
from manamap.pilot.common import (
    RULE_ID_RE, STRATEGY_ID_RE, deck_dir, expand_faces, load_rules_db)

REQUIRED_TOP_KEYS = {"id", "slug", "deck", "title", "scenario", "resolution"}
# A scenario can be checked before it has an answer — that is the whole point of the
# preflight, which costs milliseconds where a resolver spawn costs ~35k tokens.
REQUIRED_PREFLIGHT_KEYS = REQUIRED_TOP_KEYS - {"resolution"}
REQUIRED_SCENARIO_KEYS = {"stack", "question"}
# Lettered sub-questions: "(a) ... (b) ..." — the strongest cheap predictor that a
# scenario spans several rules domains and will fail atomically on the weakest one.
_SUBQUESTION_RE = re.compile(r"\(([a-h])\)")
CHECKER_STATUSES = {"supported", "unsupported", "irrelevant", "misquoted"}

REQUIRED_DECISION_KEYS = {"id", "slug", "deck", "title", "scenario", "branches", "recommendation"}
REQUIRED_BRANCH_KEYS = {"choice", "line", "signals", "coalition_risk", "coaching"}


# An inline cross-reference in `strategy.md` — "…the deflection budget
# (strategy:multiplayer.threat-deflection) lasts until…". There are 43 of them.
_STRATEGY_XREF_RE = re.compile(r"\s*\(strategy:[a-z0-9-]+\.[a-z0-9-]+\)")


def _normalize_ws(text):
    """Whitespace-flatten, and drop `strategy.md`'s inline cross-references.

    **Two validators used to make this quote impossible to write.**
    `validate_stack` requires a `strategy:` citation to be VERBATIM; `validate_issue`
    (the legacy page gate) forbids an internal taxonomy id in reader copy. Where a quotable sentence has a
    `(strategy:…)` cross-reference in the middle of it, keeping the tag fails the
    second check and dropping it failed the first — and sisay's decision 002 sat
    broken between them, quoting the sentence without the tag, invisible because
    neither validator was wired into a test.

    The tag is APPARATUS, not prose: it points a reader of the strategy doc at a
    sibling section, and it means nothing on a deck page that prints no strategy
    bibliography. So it is stripped from both sides here, which makes a quotation
    that elides it verbatim in every sense that matters and leaves the reader-copy
    lint free to keep banning the id. Both rules stay whole; neither bends.
    """
    return re.sub(r"\s+", " ", _STRATEGY_XREF_RE.sub("", text)).strip()


def validate_citations(citations, rules, where, errors, strategy_sections=None):
    """Shared citation contract: valid IDs, existing rules/sections, verbatim quotes.

    `strategy:` IDs (tier-★ grounding in decision branches) dispatch against the
    strategy DB with the same verbatim-quote check; None means the DB is
    unavailable, which only errors if a strategy citation actually appears.
    """
    for cite in citations:
        rule_id = cite.get("rule", "")
        quote = cite.get("quote", "")
        if STRATEGY_ID_RE.match(rule_id):
            if strategy_sections is None:
                errors.append(
                    f"{where}: cites {rule_id} but the strategy DB is unavailable — "
                    f"run `manamap pilot build-strategy-db`"
                )
                continue
            if rule_id not in strategy_sections:
                errors.append(f"{where}: cites nonexistent strategy section {rule_id}")
                continue
            if not quote:
                errors.append(f"{where}: citation of {rule_id} has no quote")
                continue
            if _normalize_ws(quote) not in _normalize_ws(strategy_sections[rule_id]["text"]):
                errors.append(
                    f"{where}: quote is not verbatim text of {rule_id}: {quote[:60]!r}..."
                )
            continue
        if not RULE_ID_RE.match(rule_id):
            errors.append(f"{where}: malformed rule id {rule_id!r}")
            continue
        if rule_id not in rules:
            errors.append(f"{where}: cites nonexistent rule {rule_id}")
            continue
        if not quote:
            errors.append(f"{where}: citation of {rule_id} has no quote")
            continue
        if _normalize_ws(quote) not in _normalize_ws(rules[rule_id]["text"]):
            errors.append(f"{where}: quote is not verbatim text of {rule_id}: {quote[:60]!r}...")


def validate_preflight(doc):
    """Form-check a scenario BEFORE it has a resolution. Returns (errors, warnings).

    Everything here is free and runs in milliseconds; the resolver spawn it guards
    costs ~35k tokens. An empty `scenario.stack` once aborted three resolutions
    *after* they had all run, because nothing checked the scenario until it had an
    answer attached.
    """
    errors, warnings = [], []
    missing = REQUIRED_PREFLIGHT_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors, warnings

    scenario = doc["scenario"]
    missing = REQUIRED_SCENARIO_KEYS - set(scenario)
    if missing:
        errors.append(f"scenario missing keys: {sorted(missing)}")
    if not str(scenario.get("question", "")).strip():
        errors.append("scenario.question is empty — there is nothing to resolve")
    # A v2 game state (docs/pilot.md → Game state v2) has seats that can act and may
    # resolve a list of ACTIONS instead of a stack; its form is checked by the shared
    # vocabulary module, and the v1 hand/board conventions below do not apply to it.
    if game_state.is_v2(scenario):
        errors += game_state.validate_v2(scenario)
        return errors, warnings
    stack = scenario.get("stack")
    if not isinstance(stack, list) or not stack:
        errors.append("scenario.stack must be a non-empty ordered list (pos 0 = bottom)")

    parts = {m.group(1) for m in _SUBQUESTION_RE.finditer(str(scenario.get("question", "")))}
    limit = RESOLVE_SCOPE_BUDGET["max_subquestions"]
    if len(parts) > limit:
        warnings.append(
            f"scenario.question has {len(parts)} lettered sub-questions (budget {limit}). "
            f"Broad scenarios fail atomically: the checker's verdict covers the whole "
            f"artifact, so one weak sub-answer discards the rest. Prefer one rules domain "
            f"per scenario and split the others into their own files."
        )

    # Format conventions (see .claude/skills/resolve-stack/SKILL.md step 1).
    #
    # WARN on an artifact that already carries a resolution, ERROR on one that does
    # not. The gate exists to stop a ~35k spawn being wasted, and it does that job
    # fully in warn-mode against work already done. Erroring on the committed corpus
    # would force normalising 42 scenarios — and `scenario:self` is a fingerprint
    # input, so tidying them costs 42 respawns to fix formatting nobody misread.
    authored = "resolution" not in doc
    def flag(msg):
        (errors if authored else warnings).append(msg)

    hand = scenario.get("hand")
    if hand is not None and not isinstance(hand, list):
        flag(f"scenario.hand must be a list ([] when empty), got {type(hand).__name__}. "
             f"Prose here has been read as a card name and shipped into the manifest.")
    elif isinstance(hand, list):
        for entry in hand:
            if isinstance(entry, str) and (len(entry) > 60 or entry.rstrip().endswith(".")):
                flag(f"scenario.hand entry looks like prose, not a card: {entry[:60]!r}. "
                     f"Use [] for an empty hand.")

    mana = scenario.get("mana_available")
    if isinstance(mana, str) and mana.strip() == "":
        flag('scenario.mana_available is "" — use "{0}" for none. Empty string and '
             '"{0}" have meant opposite things on different boards.')

    board = scenario.get("board")
    if isinstance(board, dict) and any(k.startswith("opponent_") for k in board):
        warnings.append(
            "board uses the `opponent_a..d` shape; the documented shape is "
            "`opponents: [{life, board}]`. Both are read, but every consumer needs "
            "a compatibility branch for the second one.")

    return errors, warnings


def unknown_cards(doc, slug):
    """Cards named on YOUR board or in hand that this deck does not have.

    `validate_stack` never opened `cards.json` — card-name verification was left
    entirely to the LLM (`rules-checker.md`: "verify card names ... match reality").
    That is a ~30k-token spawn doing a set membership test, and it only happens
    after the resolver has already spent ~35k answering a question that may be
    unanswerable. A scenario naming a card of yours that is not in the 99 describes
    a line nobody can play.

    Deliberately narrow, because a false error here blocks legitimate work:
    only `board.you` and `hand`, never the opponents' boards (their permanents are
    not in your deck by definition) and never tokens (never in a decklist).
    """
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.scenario_facts import (board_bodies, membership,
                                              unconditional_creatures)

    scenario = doc.get("scenario") or {}
    try:
        cards = load_deck_cards(slug)["cards"]
    except Exception:
        return [], []                   # no cards.json yet — not this check's job
    # Index each DFC face as well as the full name on BOTH sides. `cards.json`
    # stores "Esika, God of the Tree // The Prismatic Bridge"; a scenario naming
    # the front face is naming a card the deck runs, and matching full names only
    # reported sisay/003 as exploring a card outside the 99 when it does not.
    main_names = set()
    for card in cards:
        name = card["name"]
        main_names.update(expand_faces(name))

    if game_state.is_v2(scenario):
        named = game_state.our_named_cards(scenario)
    else:
        you = board_bodies((scenario.get("board") or {}).get("you"),
                           unconditional_creatures(slug))
        named = you["creature_bodies"] + you["other_permanents"] + you["spent_paying_a_cost"]
        named += [h for h in (scenario.get("hand") or []) if isinstance(h, str)]

    # Three outcomes, not two — but the third is now "a real card this deck does not
    # run", not "a benched card". With no sideboard the zone cannot grade anything,
    # so the question becomes whether the name is a Magic card at all: a scenario may
    # legitimately name a card outside the 99 (an opponent's permanent, a card under
    # consideration), while a name the corpus has never heard of is a typo and the
    # line as written cannot be played.
    #
    # Faces are indexed as well as full names. `cards.csv` stores a DFC as
    # "A // B", so an exact-match-only check calls the front face unknown —
    # sisay/003 names "Esika, God of the Tree" and was reported as a card the deck
    # does not have, when the deck runs the card it is the front of.
    outside = membership(named, main_names)["NOT_IN_THE_DECK"]
    corpus = _corpus_names()
    if corpus is None:                  # no cards.csv (fresh clone): never error
        unknown, elsewhere = [], outside
    else:
        unknown = [c for c in outside if c not in corpus]
        elsewhere = [c for c in outside if c in corpus]
    errors = [
        f"scenario names {c!r} on your board or in hand, and no such Magic card "
        f"exists — the line as written cannot be played. "
        f"Check `manamap pilot scenario-facts {slug}`."
        for c in unknown
    ]
    warnings = [
        f"scenario names {c!r}, which this deck does not run — the line explores a "
        f"card outside the 99. Fine to author; do not present it as a line the "
        f"current deck can run."
        for c in elsewhere
    ]
    return errors, warnings


def _corpus_names():
    """Every card name in the corpus, plus each face of a DFC. None if absent."""
    from manamap.pilot.card_pool import corpus_names
    try:
        return corpus_names()
    except Exception:                   # pragma: no cover — unreadable corpus
        return None


def scope_warnings(doc):
    """Advisory size checks on a finished resolution. Never errors — see config."""
    warnings = []
    steps = (doc.get("resolution") or {}).get("steps", [])
    citations = sum(len(s.get("citations", [])) for s in steps)
    if len(steps) > RESOLVE_SCOPE_BUDGET["max_steps"]:
        warnings.append(
            f"over scope budget: {len(steps)} steps "
            f"(budget {RESOLVE_SCOPE_BUDGET['max_steps']})"
        )
    if citations > RESOLVE_SCOPE_BUDGET["max_citations"]:
        warnings.append(
            f"over scope budget: {citations} citations "
            f"(budget {RESOLVE_SCOPE_BUDGET['max_citations']}) — "
            f"every artifact at <=32 citations passed in 1-2 rounds; every one at >=59 "
            f"needed 4 rounds or failed"
        )
    warnings.extend(stale_cross_references(doc))
    return warnings


#: A step reference, deliberately NOT a general "step <number>" match.
#:
#: `(?!\.\d)` is the whole check. A resolution routinely writes a step reference
#: and a CR citation in one breath — "(step 1, 107.4a)" is real, committed text —
#: and a pattern that takes the number after a comma reads `107` as a step. The
#: first cut of this check did exactly that: swept over 66 resolutions and 327
#: references it returned two hits, and ONE OF THEM WAS ITS OWN FALSE POSITIVE.
#: That is the measurement that decided the shape, and it is the reason six
#: earlier proposed validators were rejected — a check that fires on correct data
#: is worse than no check, because it teaches its reader to skip the output.
#:
#: It rejects a dot followed by a DIGIT, not any dot: the first attempt used
#: `(?!\d*\.)` and silently missed every reference that ends a sentence, since
#: `steps 3 and 9.` puts a period right where a rule number's decimal would be.
#:
#: `(?!\d)` is the second guard and it is not redundant. Without it the engine
#: BACKTRACKS into the rule number: `703.4h` fails the decimal lookahead at
#: `703`, retries at `70`, finds `3` rather than a dot, and reports a
#: cross-reference to step 70. Both guards were added because a test failed,
#: not because they were foreseen.
_STEP_REF = re.compile(r"[Ss]teps?\s+(\d+)(?!\d)(?!\.\d)"
    r"(?:\s*(?:and|through|-|–)\s*(\d+)(?!\d)(?!\.\d))?")


def stale_cross_references(doc):
    """A reference to a step that does not exist, as an ADVISORY.

    A resolution cross-references its own steps by number, and an inserted or
    merged step renumbers every one of them. This shipped twice in one session:
    iteration 2 of goblin-storm/012 left "steps 4-5" pointing at what had become
    5 and 6 after a step was added at position 1, and iteration 3 left a `gaps`
    entry pointing at "Step 13" in a twelve-step artifact.

    Both were caught by a human reading, and one of my own audits missed the
    second because it grepped case-sensitively for `step 13` while the text said
    `Step 13`. That is what makes this mechanical rather than a review habit.

    ADVISORY, not an error, for two reasons. It cannot distinguish a genuinely
    stale pointer from prose that means something else by "step", and — the
    deciding one — a resolution is allowed only three iterations, so a cosmetic
    pointer must not be able to condemn an artifact whose rules and arithmetic
    the checker has confirmed. `012` ships `pass` with this warning against it.
    """
    res = doc.get("resolution") or {}
    steps = res.get("steps") or []
    if not steps:
        return []
    n = len(steps)
    seen = {}
    for m in _STEP_REF.finditer(json.dumps(res)):
        for g in m.groups():
            if g and not 1 <= int(g) <= n:
                seen.setdefault(int(g), m.group(0))
    if not seen:
        return []
    return [f"cross-reference to step {t} in a {n}-step resolution "
            f"(as {ref!r}) — a renumbering left it behind"
            for t, ref in sorted(seen.items())]


def validate_scenario(doc, rules, strategy_sections=None):
    """Return a list of error strings (empty = the contract holds)."""
    errors = []

    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors  # structure too broken to continue

    missing = REQUIRED_SCENARIO_KEYS - set(doc["scenario"])
    if missing:
        errors.append(f"scenario missing keys: {sorted(missing)}")
    if game_state.is_v2(doc["scenario"]):
        errors += game_state.validate_v2(doc["scenario"])
    else:
        stack = doc["scenario"].get("stack")
        if not isinstance(stack, list) or not stack:
            errors.append("scenario.stack must be a non-empty ordered list (pos 0 = bottom)")

    steps = doc["resolution"].get("steps", [])
    if not steps:
        errors.append("resolution.steps is empty")

    for step in steps:
        n = step.get("n", "?")
        if not step.get("action"):
            errors.append(f"step {n}: missing action")
        citations = step.get("citations", [])
        if not citations:
            errors.append(f"step {n}: NO CITATIONS — every effect must cite a rule")
            continue
        validate_citations(citations, rules, f"step {n}", errors, strategy_sections)

    checker = doc.get("checker")
    if checker is not None:
        if checker.get("verdict") not in {"pass", "fail"}:
            errors.append(f"checker.verdict must be pass|fail, got {checker.get('verdict')!r}")
        for finding in checker.get("findings", []):
            if finding.get("status") not in CHECKER_STATUSES:
                errors.append(
                    f"checker finding for step {finding.get('step')}: "
                    f"invalid status {finding.get('status')!r}"
                )
        if checker.get("verdict") == "pass":
            bad = [f for f in checker.get("findings", []) if f.get("status") != "supported"]
            if bad:
                errors.append(
                    f"checker verdict is 'pass' but {len(bad)} finding(s) are not 'supported'"
                )
    return errors


def validate_decision(doc, rules, strategy_sections=None):
    """Form checks for a coaching decision tree (tier-3). Returns error strings."""
    errors = []

    missing = REQUIRED_DECISION_KEYS - set(doc)
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors

    if not doc["scenario"].get("question"):
        errors.append("scenario.question is required")
    if not doc["scenario"].get("board"):
        errors.append("scenario.board is required (include table context)")

    branches = doc.get("branches", [])
    if len(branches) < 2:
        errors.append(f"decision needs >=2 branches, found {len(branches)}")
    choices = []
    for i, branch in enumerate(branches):
        missing = REQUIRED_BRANCH_KEYS - set(branch)
        if missing:
            errors.append(f"branch {i}: missing keys {sorted(missing)}")
        if branch.get("choice"):
            choices.append(branch["choice"])
        validate_citations(
            branch.get("citations", []), rules, f"branch {i}", errors, strategy_sections
        )

    rec = doc.get("recommendation") or {}
    if not rec.get("rationale"):
        errors.append("recommendation.rationale is required")
    if rec.get("choice") not in choices:
        errors.append(
            f"recommendation.choice {rec.get('choice')!r} does not match any branch choice"
        )
    return errors


def validate_any(doc, rules, strategy_sections=None):
    """Dispatch on kind (missing kind = stack)."""
    if doc.get("kind", "stack") == "decision":
        return validate_decision(doc, rules, strategy_sections)
    return validate_scenario(doc, rules, strategy_sections)


def load_strategy_sections():
    """Best-effort strategy DB load; None = unavailable (only errors if cited)."""
    from manamap.pilot.common import load_strategy_db

    try:
        sections, _, _ = load_strategy_db()
        return sections
    except FileNotFoundError:
        return None
    except ValueError as e:
        print(f"WARN strategy DB unusable ({e}) — strategy citations will fail")
        return None


def main(args):
    rules, _, _ = load_rules_db()
    strategy_sections = load_strategy_sections()
    base = deck_dir(args.slug)
    if args.stack:
        paths = sorted((base / "stacks").glob(f"{args.stack}-*.json"))
        if not paths:
            raise SystemExit(f"No scenario {args.stack} under {base / 'stacks'}")
    else:
        paths = sorted((base / "stacks").glob("*.json")) + sorted(
            (base / "decisions").glob("*.json")
        )
        if not paths:
            raise SystemExit(f"No scenarios found under {base}/stacks or {base}/decisions")

    failed = False
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        kind = doc.get("kind", "stack")

        if getattr(args, "scenario_only", False):
            if kind == "decision":
                continue
            errors, warnings = validate_preflight(doc)
            card_errors, card_warnings = unknown_cards(doc, args.slug)
            # Same warn-for-published rule as the format checks: an artifact that
            # already carries a resolution is finished work, and blocking it costs
            # a respawn to fix a scenario nobody misread.
            (errors if "resolution" not in doc else warnings).extend(card_errors)
            warnings += card_warnings
            if errors:
                failed = True
                print(f"FAIL {path.name} (scenario preflight):")
                for e in errors:
                    print(f"  - {e}")
            else:
                print(f"OK   {path.name} (scenario form holds — safe to spawn)")
            for w in warnings:
                print(f"  ! {w}")
            continue

        errors = validate_any(doc, rules, strategy_sections)
        if errors:
            failed = True
            print(f"FAIL {path.name} ({kind}):")
            for e in errors:
                print(f"  - {e}")
        elif kind == "decision":
            print(f"OK   {path.name} (decision form holds; coaching tier)")
        else:
            verdict = (doc.get("checker") or {}).get("verdict", "unchecked")
            print(f"OK   {path.name} (contract holds; checker: {verdict})")
            # THE BOARD ROTS EVEN THOUGH THE CITATIONS DO NOT.
            #
            # `unknown_cards` was reachable ONLY through `--scenario-only`, the
            # preflight that runs BEFORE a resolver is spawned. So it checked
            # every scenario on the way in and no scenario ever again — and a
            # deck's list moves underneath its stacks constantly.
            #
            # Measured the day this was wired: 6 of edgar-vampires' 11
            # checker-passed stacks name a card the deck no longer runs
            # (Exquisite Blood, Purphoros, Warleader's Call…), and 3 of
            # ur-dragon's. Every one printed OK, because the CITATIONS are still
            # correct — what rotted is the board they are cited against, and
            # nothing looked at that.
            #
            # A WARNING, never an error, for the reason the preflight path
            # already gives: a finished artifact is finished work and failing it
            # costs a respawn to fix something nobody misread. What it must not
            # do is stay silent, because these render as ✓ — the only fact tier
            # — under a heading that says this is how the deck wins.
            # ONLY A PRESENTABLE STACK WARNS. `presentable: false` IS the
            # answer to this warning — the pilot has already read it, decided
            # the line describes a board the deck can no longer make, and
            # withheld it with a note saying which card. Warning again on a
            # decision already taken is how a reader learns the mark means
            # nothing, which is the failure this repo names about validators
            # that fire on correct data.
            if doc.get("presentable") is not False:
                _, stale = unknown_cards(doc, args.slug)
                for w in stale:
                    print(f"  ! {w}")
            # THE LABEL TRAVELS WITH THE WARNING. This printed "over scope
            # budget: " in front of whatever it was handed, which was fine while
            # the function returned only size warnings and became a lie the hour
            # `stale_cross_references` was added to it — a dangling step
            # reference is not a budget problem, and a reader who is told it is
            # looks in the wrong place.
            for w in scope_warnings(doc):
                print(f"  ! {w}")
    # Deliberately NOT `common.report_errors`, though every other validator uses
    # it: this command validates N artifacts per invocation and reports each by
    # name. `report_errors` exits on the first error list it is handed, so
    # calling it inside this loop would hide every failure after the first, and
    # calling it after would lose the per-file attribution that makes the output
    # actionable. Different shape, not an oversight.
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-stack <slug>`.")
