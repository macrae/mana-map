"""Pilot: mechanically enforce the citation contract on stack scenarios.

This is the honesty gate for the LLM stack-resolver: a resolution cannot pass
with an uncited effect, a citation of a nonexistent rule, or a quote that is
not verbatim rule text. The rules-checker agent judges *meaning*; this module
enforces *form* — and a resolution must clear this gate before the checker
even looks at it.
"""

import json
import re
import sys

from manamap.pilot.common import RULE_ID_RE, deck_dir, load_rules_db

REQUIRED_TOP_KEYS = {"id", "slug", "deck", "title", "scenario", "resolution"}
REQUIRED_SCENARIO_KEYS = {"stack", "question"}
CHECKER_STATUSES = {"supported", "unsupported", "irrelevant", "misquoted"}


def _normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def validate_scenario(doc, rules):
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
        for cite in citations:
            rule_id = cite.get("rule", "")
            quote = cite.get("quote", "")
            if not RULE_ID_RE.match(rule_id):
                errors.append(f"step {n}: malformed rule id {rule_id!r}")
                continue
            if rule_id not in rules:
                errors.append(f"step {n}: cites nonexistent rule {rule_id}")
                continue
            if not quote:
                errors.append(f"step {n}: citation of {rule_id} has no quote")
                continue
            if _normalize_ws(quote) not in _normalize_ws(rules[rule_id]["text"]):
                errors.append(
                    f"step {n}: quote is not verbatim text of {rule_id}: {quote[:60]!r}..."
                )

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


def main(args):
    rules, _, _ = load_rules_db()
    stacks_dir = deck_dir(args.slug) / "stacks"
    if args.stack:
        paths = sorted(stacks_dir.glob(f"{args.stack}-*.json"))
        if not paths:
            raise SystemExit(f"No scenario {args.stack} under {stacks_dir}")
    else:
        paths = sorted(stacks_dir.glob("*.json"))
        if not paths:
            raise SystemExit(f"No scenarios found under {stacks_dir}")

    failed = False
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        errors = validate_scenario(doc, rules)
        if errors:
            failed = True
            print(f"FAIL {path.name}:")
            for e in errors:
                print(f"  - {e}")
        else:
            verdict = (doc.get("checker") or {}).get("verdict", "unchecked")
            print(f"OK   {path.name} (contract holds; checker: {verdict})")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-stack <slug>`.")
