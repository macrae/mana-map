"""Pilot: query the strategy DB — semantic top-k search and exact section lookup.

The `--json` CLI output is the agent interface: the strategy-researcher (and
pilot-notes, the doctor) discover relevant theory with `query-strategy`,
and ground claims with `lookup-strategy` (exact fetch only — never semantic).
"""

import json

from manamap.pilot import retrieve
from manamap.pilot.common import load_strategy_db


def query(text, k=None):
    """Semantic search. Returns [(section_id, title, text, score)] best-first.

    A four-tuple where the rules query returns a three-tuple, and that stays —
    the shape is what callers read. Only the ranking moved into
    `retrieve.search`, shared with three other corpora.
    """
    return [(cid, rec["title"], rec["text"], score)
            for cid, rec, score in retrieve.search("strategy", text, k=k)]


def lookup(section_id):
    """Exact fetch by section ID. Raises KeyError with suggestions on a miss."""
    try:
        return retrieve.fetch("strategy", section_id)
    except KeyError:
        sections, _, _ = load_strategy_db()
        near = sorted(s for s in sections if s.startswith(section_id))[:8]
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"Strategy section {section_id!r} not found in the index.{hint}")


def main(args):
    if args.pilot_command == "lookup-strategy":
        try:
            result = lookup(args.section_id)
        except KeyError as e:
            raise SystemExit(str(e.args[0]))
        if args.as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            sources = "\n".join(f"  - {s}" for s in result["sources"])
            print(f"[{result['id']}] {result['title']}\n{result['text']}\nSources:\n{sources}")
        return

    results = query(args.query, k=args.k)
    if args.as_json:
        print(json.dumps(
            [
                {"section": sid, "score": round(score, 4), "title": title, "text": text}
                for sid, title, text, score in results
            ],
            indent=2,
            ensure_ascii=False,
        ))
    else:
        for sid, title, text, score in results:
            first_line = text.split("\n")[0]
            print(f"{score:.3f}  [{sid}] {title} — {first_line[:80]}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot query-strategy` / `manamap pilot lookup-strategy`.")
