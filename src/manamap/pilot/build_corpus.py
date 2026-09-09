"""Build the `docs` and `code` indexes from the working tree.

Same shape as `build_rules_db` and `build_strategy_db`, and deliberately so:
chunk, embed with the frozen MiniLM, L2-normalize the rows, write
`{records, order}` plus a `.npy`. A fourth chunking strategy would have been a
fourth thing to debug; what differs between corpora is only how text is cut.

BOTH ARE GITIGNORED, like `data/rules/`. They are derived from files already in
the repo, so committing them would add a large binary that goes stale the moment
someone edits a doc — and a stale code index answers questions about code that
no longer exists, confidently.

WHAT GETS CHUNKED, and why the cuts are where they are:

  docs   markdown, split on `##`/`###` headings. A heading is what a human
         wrote to mean "this is one idea", so it is a better boundary than any
         fixed window. The heading text rides along in the embedded string,
         because "Gotchas" and "The rules that bite" are the words a question
         will use.

  code   Python, split on top-level `def`/`class`. The DOCSTRING AND SIGNATURE
         carry almost all the retrievable meaning in this repo — the bodies are
         numpy and argparse, which embed to noise — so the chunk is signature
         plus docstring plus a bounded head of the body, not the whole function.
         This repo's docstrings are unusually load-bearing, which is what makes
         a code index worth having at all here and not in most repos.
"""

import json
import re

from manamap import config

_MD_HEADING = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.M)
_PY_DEF = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)", re.M)

#: Directories never worth embedding. `history/` is superseded by construction —
#: its whole purpose is to hold what is no longer true — and surfacing it beside
#: current docs is how a reader ends up citing a design that was abandoned.
_SKIP_DOC_DIRS = frozenset({"history"})
_SKIP_CODE_DIRS = frozenset({"__pycache__", "egg-info"})


def _split_markdown(text, path):
    """`[(anchor, title, body)]` — one entry per heading, plus any preamble."""
    marks = list(_MD_HEADING.finditer(text))
    out = []
    if marks and marks[0].start() > 0:
        head = text[:marks[0].start()].strip()
        if head:
            out.append(("", _first_title(head, path), head))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if body:
            out.append((_anchor(m.group(2)), m.group(2), body))
    return out


def _first_title(head, path):
    m = re.search(r"^#\s+(.+)$", head, re.M)
    return m.group(1).strip() if m else path.stem


def _anchor(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _split_python(text, path):
    """`[(name, title, body)]` — signature + docstring + a bounded body head."""
    marks = list(_PY_DEF.finditer(text))
    out = []
    if marks and marks[0].start() > 0:
        head = text[:marks[0].start()].strip()
        if len(head) >= config.CORPUS_MIN_CHARS:
            out.append(("module", f"{path.stem} (module docstring)", head))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.start():end].rstrip()
        out.append((m.group(1), f"{path.stem}.{m.group(1)}",
                    body[:config.CORPUS_MAX_CHARS]))
    return out


def _paragraphs(body):
    """Paragraphs, with anything still oversized cut on line boundaries.

    A BLANK LINE IS NOT A RELIABLE BOUNDARY IN THIS REPO'S DOCS. Measured on the
    first build: `docs/gotchas-bench.md` produced a single 107,197-character
    "paragraph", because its longest section is one continuous markdown table
    with no blank lines in it — and `docs/pilot.md#commands` produced a 25,355
    one for the same reason, a fenced command block.

    That is not a cosmetic overflow. A 107 KB chunk embeds to a vector that
    means nothing in particular, so it never ranks for anything, and the passage
    a question wanted is inside it and unreachable. The whole gotchas table —
    the most expensively-earned content in the repo — was one dead chunk.

    So paragraphs first, and a hard line-level cut for whatever is still over.
    Lines, not characters, so a table row and a code line survive intact.
    """
    for para in body.split("\n\n"):
        if len(para) <= config.CORPUS_MAX_CHARS:
            yield para
            continue
        buf, size = [], 0
        for line in para.split("\n"):
            if size + len(line) > config.CORPUS_MAX_CHARS and buf:
                yield "\n".join(buf)
                buf, size = [], 0
            buf.append(line)
            size += len(line) + 1
        if buf:
            yield "\n".join(buf)


def _pack(sections, source):
    """Fold the tiny, split the huge. Yields `(suffix, title, text)`.

    A section under `CORPUS_MIN_CHARS` is merged forward: a lone heading embeds
    to noise and then crowds out the passage that actually answers the question,
    which is worse than not indexing it.
    """
    pending = None
    for anchor, title, body in sections:
        if pending:
            anchor, title, body = pending[0], pending[1], pending[2] + "\n\n" + body
            pending = None
        if len(body) < config.CORPUS_MIN_CHARS:
            pending = (anchor, title, body)
            continue
        if len(body) <= config.CORPUS_MAX_CHARS:
            yield anchor, title, body
            continue
        # Split on paragraph boundaries so a chunk is never cut mid-sentence.
        part, size, n = [], 0, 0
        for para in _paragraphs(body):
            if size + len(para) > config.CORPUS_MAX_CHARS and part:
                yield f"{anchor}~{n}", title, "\n\n".join(part)
                part, size, n = [], 0, n + 1
            part.append(para)
            size += len(para)
        if part:
            yield (f"{anchor}~{n}" if n else anchor), title, "\n\n".join(part)
    if pending:
        yield pending


def collect(kind):
    """`[{id, title, source, text}]` for one corpus, from the working tree."""
    # `config._REPO_ROOT` rather than CWD: these corpora index the SOURCE
    # tree, which is where the module lives, not wherever the command ran.
    root = config._REPO_ROOT
    if kind == "docs":
        paths = [p for p in sorted((root / "docs").rglob("*.md"))
                 if not _SKIP_DOC_DIRS & set(p.relative_to(root / "docs").parts)]
        paths += [root / "CLAUDE.md", root / "README.md", root / "PLAN.md"]
        split = _split_markdown
    elif kind == "code":
        paths = [p for p in sorted((root / "src").rglob("*.py"))
                 if not _SKIP_CODE_DIRS & set(p.parts)]
        split = _split_python
    else:
        raise ValueError(f"{kind!r} is not a buildable corpus (docs, code)")

    chunks = []
    for path in paths:
        if not path.exists():
            continue
        rel = str(path.relative_to(root))
        for suffix, title, body in _pack(split(path.read_text(errors="replace"), path), rel):
            chunks.append({
                "id": f"{rel}#{suffix}" if suffix else rel,
                "title": title,
                "source": rel,
                "text": body,
            })
    return chunks


def embed_text(chunk):
    """What actually gets embedded: the title, then the body.

    The title is repeated into the embedded string because it carries the words
    a question will use — "the rules that bite", "gotchas" — and a body often
    never restates them.
    """
    return f"{chunk['title']}\n\n{chunk['text']}"


def build(kind, quiet=False):
    import numpy as np

    from manamap import console
    from manamap.ingest.preprocess import compute_text_embeddings

    paths = {"docs": (config.DOCS_DIR, config.DOCS_INDEX_PATH,
                      config.DOCS_EMBEDDINGS_PATH),
             "code": (config.CODE_DIR, config.CODE_INDEX_PATH,
                      config.CODE_EMBEDDINGS_PATH)}[kind]
    directory, index_path, embeddings_path = paths

    chunks = collect(kind)
    if not chunks:
        raise SystemExit(f"{kind}: nothing to index")
    with console.task(f"Embedding {kind}", total=len(chunks), unit="chunks") as bar:
        vectors = compute_text_embeddings([embed_text(c) for c in chunks])
        bar.advance(len(chunks))
    # Normalized HERE, at build time, so every query is a bare matmul. The two
    # older builders make the same promise and `common.load_rules_db` documents
    # it on the read side.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = (vectors / np.maximum(norms, 1e-8)).astype("float32")

    directory.mkdir(parents=True, exist_ok=True)
    records = {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in chunks}
    with open(index_path, "w") as f:
        json.dump({"records": records, "order": [c["id"] for c in chunks]},
                  f, indent=2, ensure_ascii=False)
    np.save(embeddings_path, vectors)
    if not quiet:
        sources = len({c["source"] for c in chunks})
        print(f"Wrote {index_path}\n  {len(chunks)} chunk(s) from {sources} file(s)")
    return len(chunks)


def main(args):
    kind = "docs" if args.pilot_command == "build-docs-db" else "code"
    build(kind)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot build-docs-db` / `build-code-db`.")
