"""Pilot: chunk the strategy doc and embed it into the strategy DB.

One chunk per `##`/`###` section of data/strategy/strategy.md (chunk ID =
section ID = citation ID, e.g. "strategy:whos-the-beatdown"). Mirrors
build_rules_db.py: stored text is verbatim doc prose (quote-substring checks
run against it); only the *embedded* string is prefixed with id + title. The
index records the doc's sha256 so load_strategy_db can refuse a stale DB.
"""

import json

import numpy as np

from manamap.config import (
    STRATEGY_DB_META_PATH,
    STRATEGY_DOC_PATH,
    STRATEGY_EMBEDDINGS_PATH,
    STRATEGY_INDEX_PATH,
    TEXT_MODEL_NAME,
)
from manamap.ingest.preprocess import compute_text_embeddings
from manamap.pilot.common import strategy_doc_sha256
from manamap.pilot.validate_strategy import oversize_warnings, parse_strategy


def build_embed_text(chunk):
    """The string that gets embedded (id + title context + text)."""
    return f"{chunk['id']} {chunk['title']}: {chunk['text']}"


def main(args=None):
    if not STRATEGY_DOC_PATH.exists():
        raise SystemExit(
            f"{STRATEGY_DOC_PATH} not found — author the strategy doc first "
            f"(see docs/pilot.md, Strategy DB)."
        )
    text = STRATEGY_DOC_PATH.read_text(encoding="utf-8")

    print("Chunking strategy doc...")
    chunks, errors = parse_strategy(text)
    if errors:
        listing = "\n".join(f"  - {e}" for e in errors)
        raise SystemExit(
            f"strategy.md fails form validation ({len(errors)} error(s)) — "
            f"fix before building:\n{listing}"
        )
    for warning in oversize_warnings(chunks):
        print(f"  WARN {warning}")
    n_pillars = sum(1 for c in chunks if c["parent"] is None)
    print(f"  {len(chunks)} sections ({n_pillars} pillars, {len(chunks) - n_pillars} subsections)")

    print(f"Embedding sections ({TEXT_MODEL_NAME})...")
    embeddings = compute_text_embeddings([build_embed_text(c) for c in chunks])
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    embeddings = (embeddings / norms).astype(np.float32)

    index = {
        "meta": {
            "doc_sha256": strategy_doc_sha256(),
            "model": TEXT_MODEL_NAME,
            "chunk_count": len(chunks),
        },
        "order": [c["id"] for c in chunks],
        "sections": {
            c["id"]: {
                "title": c["title"],
                "text": c["text"],
                "section": c["section"],
                "parent": c["parent"],
                "sources": c["sources"],
            }
            for c in chunks
        },
    }

    np.save(STRATEGY_EMBEDDINGS_PATH, embeddings)
    with open(STRATEGY_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    with open(STRATEGY_DB_META_PATH, "w") as f:
        json.dump(index["meta"], f, indent=2)

    print(f"  Wrote {STRATEGY_INDEX_PATH} and {STRATEGY_EMBEDDINGS_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
