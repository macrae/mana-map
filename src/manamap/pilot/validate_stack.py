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

from manamap.pilot.common import RULE_ID_RE, STRATEGY_ID_RE, deck_dir, load_rules_db

REQUIRED_TOP_KEYS = {"id", "slug", "deck", "title", "scenario", "resolution"}
REQUIRED_SCENARIO_KEYS = {"stack", "question"}
CHECKER_STATUSES = {"supported", "unsupported", "irrelevant", "misquoted"}

REQUIRED_DECISION_KEYS = {"id", "slug", "deck", "title", "scenario", "branches", "recommendation"}
REQUIRED_BRANCH_KEYS = {"choice", "line", "signals", "coalition_risk", "coaching"}


def _normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def _validate_citations(citations, rules, where, errors, strategy_sections=None):
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
        _validate_citations(citations, rules, f"step {n}", errors, strategy_sections)

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
        _validate_citations(
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


def _load_strategy_sections():
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
    strategy_sections = _load_strategy_sections()
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
        errors = validate_any(doc, rules, strategy_sections)
        kind = doc.get("kind", "stack")
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
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-stack <slug>`.")
