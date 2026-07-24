"""Pilot: query the rules DB — semantic top-k search and exact rule lookup.

The `--json` CLI output is the agent interface: the stack-resolver discovers
rules with `query-rules`, and the rules-checker verifies citations with
`lookup-rule` (exact fetch only — never semantic).
"""

import json

import numpy as np

from manamap.config import RULES_QUERY_TOP_K
from manamap.ingest.preprocess import compute_text_embeddings
from manamap.pilot.common import load_rules_db


def query(text, k=None):
    """Semantic search. Returns [(rule_id, rule_text, score)] best-first."""
    if k is None:
        k = RULES_QUERY_TOP_K
    rules, order, embeddings = load_rules_db()
    q = compute_text_embeddings([text])[0]
    q = q / max(np.linalg.norm(q), 1e-8)
    scores = embeddings @ q  # rows pre-normalized -> cosine
    top = np.argsort(-scores)[:k]
    return [(order[i], rules[order[i]]["text"], float(scores[i])) for i in top]


def lookup(rule_id):
    """Exact fetch by rule ID. Raises KeyError with suggestions on a miss."""
    rules, _, _ = load_rules_db()
    if rule_id in rules:
        return {"id": rule_id, **rules[rule_id]}
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
