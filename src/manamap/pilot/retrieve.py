"""One retrieval core, four corpora.

`query_rules.py` and `query_strategy.py` were 65 and 72 lines of the same thing:
load a DB, embed the query, matmul against pre-normalized rows, take the top k,
plus an exact `lookup` with prefix suggestions. Adding `docs` and `code` as two
more copies would have made four places to fix a ranking bug.

So the maths lives here once, and the corpora differ only in DATA:

    rules      the Comprehensive Rules, chunked by rule number
    strategy   the strategy KB, chunked by section
    docs       this repo's own `docs/` — the "why did we do it this way" corpus
    code       this repo's own `src/` — which module owns a predicate

Every DB is the same triple, which is what made the collapse possible at all:
`(records, order, embeddings)`, where `records[order[i]]` is embedded by
`embeddings[i]` and rows are L2-normalized at build time so a query is a bare
matmul.

THE OLD MODULES STILL EXIST AND STILL EXPORT `query`/`lookup`. They delegate
here rather than being deleted, because they are the agent interface — the
stack-resolver discovers rules with `query-rules` and the rules-checker verifies
citations with `lookup-rule`. `tests/test_pilot_retrieve.py` asserts the
delegation returns results identical to the implementation it replaced, which is
the only evidence that a refactor of a ranking function is safe.
"""

import json

from manamap import config


def _rules():
    from manamap.pilot.common import load_rules_db
    return load_rules_db()


def _strategy():
    from manamap.pilot.common import load_strategy_db
    return load_strategy_db()


def _docs():
    return _load_simple(config.DOCS_INDEX_PATH, config.DOCS_EMBEDDINGS_PATH,
                        "docs", "build-docs-db")


def _code():
    return _load_simple(config.CODE_INDEX_PATH, config.CODE_EMBEDDINGS_PATH,
                        "code", "build-code-db")


#: name -> (loader, default k, the field to show in a one-line result)
CORPORA = {
    "rules": (_rules, config.RULES_QUERY_TOP_K, "text"),
    "strategy": (_strategy, config.STRATEGY_QUERY_TOP_K, "title"),
    "docs": (_docs, config.DOCS_QUERY_TOP_K, "title"),
    "code": (_code, config.CODE_QUERY_TOP_K, "title"),
}


def _load_simple(index_path, embeddings_path, name, build_cmd):
    """Load a `{records, order}` index plus its embeddings, with the same
    consistency check the two older loaders make.

    That check is not ceremony: a half-rebuilt DB — index written, embeddings
    not — is silently answerable and silently wrong, returning the text of one
    chunk under the id of another.
    """
    import numpy as np

    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found — run `manamap pilot {build_cmd}` first.")
    with open(index_path) as f:
        index = json.load(f)
    embeddings = np.load(embeddings_path)
    order = index["order"]
    if len(order) != embeddings.shape[0]:
        raise ValueError(
            f"{name} DB inconsistent: index has {len(order)} chunks but "
            f"embeddings has {embeddings.shape[0]} rows. Rebuild with "
            f"`manamap pilot {build_cmd}`.")
    return index["records"], order, embeddings


def search(corpus, text, k=None):
    """Semantic top-k. Returns `[(id, record, score)]`, best first.

    The ranking is one matmul because rows are unit-normalized at build time —
    a decision the builders own and this reads. `compute_text_embeddings` does
    NOT normalize its output, so the query is normalized here; forgetting that
    turns cosine into a dot product weighted by query length, which reorders
    results subtly enough to look like a model problem.
    """
    import numpy as np

    if corpus not in CORPORA:
        raise ValueError(f"{corpus!r} is not a corpus. Have: {', '.join(sorted(CORPORA))}")
    from manamap.ingest.preprocess import compute_text_embeddings

    load, default_k, _field = CORPORA[corpus]
    records, order, embeddings = load()
    q = compute_text_embeddings([text])[0]
    q = q / max(np.linalg.norm(q), 1e-8)
    scores = embeddings @ q
    top = np.argsort(-scores)[:(k or default_k)]
    return [(order[i], records[order[i]], float(scores[i])) for i in top]


def fetch(corpus, key):
    """Exact lookup, with prefix suggestions on a miss.

    Suggestions matter more than they look: a checker verifying a citation must
    never fall back to semantic search — an approximate rule is a wrong rule —
    so a miss has to be a hard error that still helps.
    """
    load, _k, _field = CORPORA[corpus]
    records, _order, _emb = load()
    if key in records:
        return {"id": key, **records[key]}
    near = sorted(r for r in records if r.startswith(key))[:8]
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    raise KeyError(f"{key!r} not found in the {corpus} index.{hint}")


def summarize(corpus, record):
    """A one-line description of a hit, for the terminal."""
    _load, _k, field = CORPORA[corpus]
    line = (record.get(field) or record.get("text") or "").strip()
    return line.split("\n")[0][:100]


def main(args):
    cmd = args.pilot_command
    if cmd == "lookup-doc":
        try:
            record = fetch("docs", args.chunk_id)
        except KeyError as exc:
            raise SystemExit(str(exc.args[0]))
        if args.as_json:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print(f"[{record['id']}]  {record.get('title', '')}\n\n{record['text']}")
        return

    corpus = "docs" if cmd == "query-docs" else "code"
    hits = search(corpus, args.query, k=args.k)
    if args.as_json:
        print(json.dumps(
            [{"id": i, "score": round(sc, 4), "title": r.get("title"),
              "source": r.get("source"), "text": r["text"]} for i, r, sc in hits],
            indent=2, ensure_ascii=False))
        return
    for chunk_id, record, score in hits:
        print(f"{score:.3f}  [{chunk_id}]")
        print(f"       {summarize(corpus, record)}")
        if getattr(args, "full", False):
            body = "\n".join("       " + line for line in record["text"].splitlines())
            print(f"{body}\n")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot query-docs` / `query-code`.")
