"""Pilot: chunk the Comprehensive Rules and embed them into the rules DB.

One chunk per numbered rule (chunk ID = rule number = citation ID), plus one
chunk per glossary term. Continuation and "Example:" lines are appended to the
owning rule so quotes from examples still satisfy the citation contract.

The text stored in the index is verbatim CR text (the quote-substring check
runs against it); only the *embedded* string is prefixed with the rule ID and
section title, which helps retrieval for sub-rules whose text never names
their mechanic (702.40a describes storm without saying "storm").
"""

import json
import re

import numpy as np

from manamap.config import (
    CR_RULES_META_PATH,
    CR_RULES_PATH,
    RULES_DB_META_PATH,
    RULES_EMBEDDINGS_PATH,
    RULES_INDEX_PATH,
    TEXT_MODEL_NAME,
)
from manamap.ingest.preprocess import compute_text_embeddings
from manamap.pilot.common import RULE_ID_RE

RULE_LINE_RE = re.compile(r"^(\d{3}\.\d+)([a-z])?\.?\s+(.*)$")
SECTION_LINE_RE = re.compile(r"^(\d{3})\.\s+(\S.*)$")


def kebab_term(term):
    """Glossary term -> id-safe kebab: lowercase, [a-z0-9'-] only."""
    slug = re.sub(r"[^a-z0-9']+", "-", term.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def parse_rules(text):
    """Parse CR text into chunks: [{"id", "text", "section", "parent"}].

    The table of contents lists section headers but never numbered rules, so
    starting chunks only on rule lines is TOC-safe. Glossary parsing begins at
    the first "Glossary" line after the last numbered rule; credits are dropped.
    """
    lines = text.split("\n")

    # Locate the glossary: first standalone "Glossary" line after the last rule line.
    last_rule_i = max((i for i, l in enumerate(lines) if RULE_LINE_RE.match(l)), default=-1)
    glossary_i = next(
        (i for i in range(last_rule_i + 1, len(lines)) if lines[i].strip() == "Glossary"),
        len(lines),
    )
    credits_i = next(
        (i for i in range(glossary_i + 1, len(lines)) if lines[i].strip() == "Credits"),
        len(lines),
    )

    chunks = []
    section = ""
    current = None

    for line in lines[:glossary_i]:
        stripped = line.strip()
        rule_m = RULE_LINE_RE.match(stripped)
        if rule_m:
            if current:
                chunks.append(current)
            number = rule_m.group(1) + (rule_m.group(2) or "")
            parent = rule_m.group(1) if rule_m.group(2) else number.rsplit(".", 1)[0]
            current = {"id": number, "text": rule_m.group(3), "section": section, "parent": parent}
            continue
        section_m = SECTION_LINE_RE.match(stripped)
        if section_m:
            if current:
                chunks.append(current)
                current = None
            section = stripped
            continue
        if current and stripped:
            # Continuation or "Example:" line — belongs to the owning rule.
            current["text"] += "\n" + stripped
        elif current and not stripped:
            chunks.append(current)
            current = None
    if current:
        chunks.append(current)

    # Glossary: term line, definition lines, blank separator.
    term, definition = None, []
    for line in lines[glossary_i + 1 : credits_i]:
        stripped = line.strip()
        if not stripped:
            if term and definition:
                chunks.append({
                    "id": f"glossary:{kebab_term(term)}",
                    "text": term + "\n" + "\n".join(definition),
                    "section": "Glossary",
                    "parent": None,
                })
            term, definition = None, []
        elif term is None:
            term = stripped
        else:
            definition.append(stripped)
    if term and definition:
        chunks.append({
            "id": f"glossary:{kebab_term(term)}",
            "text": term + "\n" + "\n".join(definition),
            "section": "Glossary",
            "parent": None,
        })

    return chunks


def build_embed_text(chunk):
    """The string that gets embedded (id + section context + text)."""
    return f"{chunk['id']} {chunk['section']}: {chunk['text']}"


def main(args=None):
    if not CR_RULES_PATH.exists():
        raise SystemExit(
            f"{CR_RULES_PATH} not found — run `manamap pilot download-rules` first."
        )
    text = CR_RULES_PATH.read_text(encoding="utf-8")

    print("Chunking comprehensive rules...")
    chunks = parse_rules(text)
    bad = [c["id"] for c in chunks if not RULE_ID_RE.match(c["id"])]
    if bad:
        raise SystemExit(f"Chunker produced {len(bad)} invalid rule IDs, e.g. {bad[:5]}")
    n_glossary = sum(1 for c in chunks if c["id"].startswith("glossary:"))
    print(f"  {len(chunks):,} chunks ({len(chunks) - n_glossary:,} rules, {n_glossary:,} glossary)")

    print("Embedding chunks (all-MiniLM-L6-v2)...")
    embeddings = compute_text_embeddings([build_embed_text(c) for c in chunks])
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    embeddings = (embeddings / norms).astype(np.float32)

    effective_date = None
    if CR_RULES_META_PATH.exists():
        with open(CR_RULES_META_PATH) as f:
            effective_date = json.load(f).get("effective_date")

    index = {
        "meta": {
            "effective_date": effective_date,
            "model": TEXT_MODEL_NAME,
            "chunk_count": len(chunks),
        },
        "order": [c["id"] for c in chunks],
        "rules": {
            c["id"]: {"text": c["text"], "section": c["section"], "parent": c["parent"]}
            for c in chunks
        },
    }

    np.save(RULES_EMBEDDINGS_PATH, embeddings)
    with open(RULES_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
    with open(RULES_DB_META_PATH, "w") as f:
        json.dump(index["meta"], f, indent=2)

    print(f"  Wrote {RULES_INDEX_PATH} and {RULES_EMBEDDINGS_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
