"""Pilot: query the rules DB — semantic top-k search and exact rule lookup.

The `--json` CLI output is the agent interface: the stack-resolver discovers
rules with `query-rules`, and the rules-checker verifies citations with
`lookup-rule` (exact fetch only — never semantic).
"""

import json

from manamap.pilot import retrieve
from manamap.pilot.common import load_rules_db


def query(text, k=None):
    """Semantic search. Returns [(rule_id, rule_text, score)] best-first.

    THE SHAPE IS THE CONTRACT and it has not moved — the stack-resolver reads
    this tuple. Only the maths moved, into `retrieve.search`, so that four
    corpora share one ranking rather than four copies of it.
    `tests/test_pilot_retrieve.py` pins the results against a baseline captured
    before the delegation, because "I refactored the ranking function and it
    looks fine" is not evidence.
    """
    return [(cid, rec["text"], score)
            for cid, rec, score in retrieve.search("rules", text, k=k)]


def lookup(rule_id):
    """Exact fetch by rule ID. Raises KeyError with suggestions on a miss."""
    try:
        return retrieve.fetch("rules", rule_id)
    except KeyError:
        # The wording is preserved verbatim: the rules-checker charter quotes
        # this sentence, and an agent that greps for it would stop finding it.
        rules, _, _ = load_rules_db()
        near = sorted(r for r in rules if r.startswith(rule_id))[:8]
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"Rule {rule_id!r} not found in the rules index.{hint}")


def main(args):
    if args.pilot_command == "lookup-rule":
        try:
            result = lookup(args.rule_id)
        except KeyError as e:
            raise SystemExit(str(e.args[0]))
        if args.as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"[{result['id']}] ({result['section']})\n{result['text']}")
        return

    results = query(args.query, k=args.k)
    if args.as_json:
        print(json.dumps(
            [{"rule": r, "score": round(s, 4), "text": t} for r, t, s in results],
            indent=2,
            ensure_ascii=False,
        ))
    else:
        for rule_id, text, score in results:
            first_line = text.split("\n")[0]
            print(f"{score:.3f}  [{rule_id}] {first_line[:100]}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot query-rules` / `manamap pilot lookup-rule`.")
