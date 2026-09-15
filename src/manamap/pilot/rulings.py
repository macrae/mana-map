"""Pilot: official card rulings, from the local Scryfall dump — INPUT, never a citation.

The resolve-stack loop derived every outcome from the Comprehensive Rules alone
and rediscovered, through `query-rules`, mechanics WotC had already ruled on
card by card. A ruling states the outcome and names the mechanic; the agent
then finds and cites the CR rule that establishes it. So a ruling is read
FIRST and cited NEVER: `citations[].rule` stays a CR id or `glossary:` term,
and `validate_stack` is untouched by this module.

Three shapes, because ABSENT MEANS ABSENT and an empty list is a measurement:

  * the dump is not on disk           -> `section()` returns `{"absent": ..., "run": ...}`
  * the name is not in the corpus     -> the card reads `{"absent": "not-in-corpus"}`
  * the card has no WotC rulings      -> `{"rulings": [], "note": "no official rulings"}`

WotC-only by default: Scryfall's own 69 lines are editorial notes, reported as
a count so a reader can tell "none" from "none from WotC".

Reads `config.RULINGS_*` at call time so tests can patch `manamap.config`.
"""

import gzip
import json
import sys

from manamap import config
from manamap.pilot.card_pool import corpus_oracle_ids
from manamap.pilot.common import expand_faces, load_json_memo, mtime_memo

ABSENT_REASON = ("data/rulings/rulings.jsonl.gz is not on disk — run "
                 "`manamap pilot download-rulings` (5 MB, seconds)")
RUN = "manamap pilot download-rulings"
NOT_IN_CORPUS = "not-in-corpus"
NO_RULINGS = "no official rulings"
POLICY = ("official WotC rulings only; INPUT, never a citation — cite the CR rule "
          "the ruling states")


def _parse(path):
    """{oracle_id: [{"date", "text", "source"}]}, each list sorted by (date, text)."""
    by_id = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_id.setdefault(r["oracle_id"], []).append(
                {"date": r.get("published_at") or "",
                 "text": r.get("comment") or "",
                 "source": r.get("source") or ""})
    for rows in by_id.values():
        rows.sort(key=lambda r: (r["date"], r["text"]))
    return by_id


def load_rulings_db():
    """The parsed dump, memoised on the file's (mtime, size). None when absent."""
    return mtime_memo(config.RULINGS_PATH, "rulings:db",
                      lambda: _parse(config.RULINGS_PATH), absent=None)


def rulings_meta():
    """The download sidecar, or None."""
    if not config.RULINGS_META_PATH.exists():
        return None
    return load_json_memo(config.RULINGS_META_PATH)


def resolve_oracle_id(name, ids=None):
    """The corpus oracle_id for a card name or any of its faces, else None."""
    ids = corpus_oracle_ids() if ids is None else ids
    if name in ids:
        return ids[name]
    for face in expand_faces(name):
        if face in ids:
            return ids[face]
    return None


def rulings_for(name, db, *, wotc_only=True, ids=None):
    """One card's rulings block — one of the three shapes in the module docstring."""
    oid = resolve_oracle_id(name, ids)
    if oid is None:
        return {"name": name, "absent": NOT_IN_CORPUS}
    rows = db.get(oid, [])
    kept = [r for r in rows if r["source"] == "wotc"] if wotc_only else rows
    out = {"name": name, "oracle_id": oid,
           "rulings": [{"date": r["date"], "text": r["text"]} if wotc_only
                       else dict(r) for r in kept]}
    if wotc_only:
        out["scryfall_notes_omitted"] = len(rows) - len(kept)
    if not kept:
        out["note"] = NO_RULINGS
    return out


def cards_block(names, *, wotc_only=True):
    """{name: rulings_for(...)} for every name, or None when the dump is absent.

    This is the `rulings:scenario` cache digest's input: exactly what the agents
    are shown, and nothing that moves when an unrelated card's ruling changes.
    """
    db = load_rulings_db()
    if db is None:
        return None
    ids = corpus_oracle_ids()
    return {n: rulings_for(n, db, wotc_only=wotc_only, ids=ids) for n in names}


def section(names, *, wotc_only=True):
    """What `scenario-facts --stack` embeds under `rulings`."""
    block = cards_block(names, wotc_only=wotc_only)
    if block is None:
        return {"absent": ABSENT_REASON, "run": RUN}
    meta = rulings_meta() or {}
    return {
        "source": "data/rulings/rulings.jsonl.gz",
        "updated_at": meta.get("updated_at"),
        "content_sha256": (meta.get("content_sha256") or "")[:12] or None,
        "policy": POLICY,
        "cards": {n: b for n, b in block.items() if "absent" not in b},
        "not_in_corpus": [n for n, b in block.items() if "absent" in b],
    }


def _print_text(block):
    for name, b in block.items():
        if "absent" in b:
            print(f"{name}: {b['absent']}")
            continue
        n = len(b["rulings"])
        omitted = b.get("scryfall_notes_omitted", 0)
        tail = f" (+{omitted} Scryfall note(s) omitted; --all-sources)" if omitted else ""
        print(f"{name} — {n} official ruling(s){tail}" if n else f"{name} — {NO_RULINGS}{tail}")
        for r in b["rulings"]:
            src = f" [{r['source']}]" if "source" in r else ""
            print(f"  {r['date']}{src}: {r['text']}")
        print()


def main(args):
    """`card-rulings <name> [<name>...] [--json] [--all-sources]`."""
    block = cards_block(args.names, wotc_only=not getattr(args, "all_sources", False))
    if block is None:
        raise SystemExit(ABSENT_REASON)
    if getattr(args, "as_json", False):
        print(json.dumps(list(block.values()), indent=2, ensure_ascii=False))
        return
    _print_text(block)
    print(f"# {POLICY}", file=sys.stderr)
