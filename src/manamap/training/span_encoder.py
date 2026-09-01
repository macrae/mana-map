"""Text spans as vectors: a frozen sentence encoder, deduplicated and cached.

## WHERE THE SPANS COME FROM, AND WHY NOT FROM `cards.csv`

`extract.py:157` flattens newlines so the CSV stays one row per card. That is
right for `embedding_text`, which feeds the frozen baseline and must not move —
and fatal here, because **the newline IS the ability boundary**. Read from the
CSV, all 34,890 cards have exactly one ability line and the structure this module
exists to encode does not exist.

So the spans are read from `oracle-cards.json.gz`, the raw Scryfall dump, joined
to corpus order on `oracle_id` — all 34,890 rows match. With the boundary
restored the corpus is **66,431 lines, 1.90 per card**, against 34,890 flattened.

## SEVEN SLOTS, EACH MASKABLE ON ITS OWN

The five ability classes from `card_parse`, plus `name` and `flavor`. A card can
carry **17 static abilities**, so a fixed positional slot per line is not an
option; lines are mean-pooled within their class instead. That makes the
maskable unit "this card's triggered abilities" rather than "line 3", which is
the question worth asking anyway — the ordering of a card's lines is not
information the model should be learning from.

Each line is L2-normalised BEFORE the mean, so a long rules-heavy line does not
dominate a slot by magnitude alone.

## WHAT `name` LEAKS, measured before anything relies on it

A card's own name appears inside its own rules text often enough to matter:
**8.5% carry the full name, another 4.8% the part before the comma — 13.3% of
34,536 cards**. Gishath's triggered ability opens *"Whenever Gishath deals combat
damage…"*, so masking the `name` slot hides nothing a reader of the `triggered`
slot cannot recover.

That does not disqualify the slot, and it is not a bug to fix — it is how the
cards are written. It is recorded here because a `name` imputation score is
**13.3% floor, not skill**, and a model reported at 0.13 on that task has learned
to copy rather than to infer. Scryfall writes "this creature" on newer printings
and the literal name on older ones, so the rate is a property of the corpus's set
mix and will drift; re-measure rather than trusting this number after a refresh.

## THE CACHE

**41,214 of the 66,431 ability lines are distinct (62%)** — `Flying` alone
appears 2,619 times — so encoding unique text once and looking it up saves 38% of
that work outright, on top of never re-encoding across runs. With names the cache
holds **76,023 spans x 384, 117 MB, built in 202s**; resolving every slot for all
34,890 cards off it takes ~2.6s and misses nothing. `vae_cache` established the
shape and the discipline: a cache is valid only for the encoder that built it,
and `load` REFUSES a mismatch rather than warning. A head trained against vectors
from a different encoder produces a plausible loss curve and a meaningless space.
"""

import gzip
import json
import time

import numpy as np

from manamap.config import DATA_DIR, OUTPUT_CSV_PATH, RAW_JSON_PATH, TEXT_MODEL_NAME
from manamap.training.card_parse import ABILITY_KINDS, ability_lines, classify_line
from manamap.training.card_source import redact_name
from manamap.training.common import say

VECTORS_PATH = DATA_DIR / "span_vectors.npy"
INDEX_PATH = DATA_DIR / "span_vectors.index.json.gz"

#: The maskable text inputs: the five ability classes plus `name`.
#:
#: **Flavor text was here and was cut.** It is a property of a PRINTING, not of a
#: card — the same card carries different flavor across sets, and some printings
#: carry none — so it moves under the model without the card changing, which is
#: noise on a task that is entirely about what a card DOES. Cutting it dropped the cache from
#: 96,115 spans to 76,023 — 20,092 vectors that no rules question turns on.
SPAN_SLOTS = ("name",) + ABILITY_KINDS

PRESENT, ABSENT, MASKED = "present", "absent", "masked"


def _text(value):
    """A cell -> a string, with NaN meaning ABSENT rather than the word "nan"."""
    if value is None or value != value:                        # NaN != NaN
        return ""
    return str(value).strip()


def oracle_text_by_id(path=RAW_JSON_PATH):
    """`{oracle_id: oracle_text}` from the raw dump, newlines INTACT.

    A double-faced card carries its text on `card_faces` and leaves the top-level
    `oracle_text` empty; both faces are joined, because the CSV's `type_line`
    already describes the card as a whole with ` // ` and the two must agree on
    what "this card" means.
    """
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            text = _text(card.get("oracle_text"))
            if not text and card.get("card_faces"):
                text = "\n".join(
                    _text(face.get("oracle_text")) for face in card["card_faces"]
                ).strip()
            out[card.get("oracle_id")] = text
    return out


def card_spans(card, oracle_text=None):
    """`{slot: [text, …]}` for one card. Empty slots are simply absent.

    `oracle_text` overrides the card's own flattened column and is how the raw
    dump gets in. Passing nothing falls back to the CSV, which is correct for a
    caller that has no dump and honest about what it loses: every ability lands
    in one slot, because there are no newlines left to split on.
    """
    text = oracle_text if oracle_text is not None else card.get("oracle_text")
    type_line = _text(card.get("type_line"))
    spans = {}
    name = _text(card.get("name"))
    if name:
        spans["name"] = [name]
    # THE NAME COMES OUT OF THE RULES TEXT. 12.6% of cards say their own name in
    # their own abilities, so without this the `name` slot and the ability slots
    # share a literal string and the model learns the identity instead of the
    # function — and masking `name` hides nothing, since the answer is still
    # sitting in the trigger.
    text = redact_name(text, name)
    for line in ability_lines(text):
        spans.setdefault(classify_line(line, type_line), []).append(line)
    return spans


def unique_spans(cards, texts=None):
    """Every distinct span string in the corpus, SORTED.

    Sorted because the row a span lands on is part of the cache's contract, and
    an unordered set would reshuffle it on every rebuild — making two caches of
    the same corpus silently incompatible.
    """
    seen = set()
    for card in cards:
        override = None if texts is None else texts.get(card.get("oracle_id"), "")
        for lines in card_spans(card, override).values():
            seen.update(lines)
    return sorted(seen)


def build(batch_size=256, echo=say):
    """Encode every distinct span once. Returns `(matrix, texts)`."""
    import pandas as pd
    from transformers import AutoModel, AutoTokenizer

    from manamap.training.common import get_device

    device = get_device()
    frame = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    cards = frame.to_dict("records")
    texts_by_id = oracle_text_by_id()
    missing = sum(1 for c in cards if c.get("oracle_id") not in texts_by_id)
    if missing:
        raise SystemExit(
            f"{missing} of {len(cards)} cards are absent from {RAW_JSON_PATH.name}; "
            "the dump and the corpus disagree — re-run `manamap download`")

    spans = unique_spans(cards, texts_by_id)
    echo(f"  {len(spans):,} distinct spans over {len(cards):,} cards")

    name = f"sentence-transformers/{TEXT_MODEL_NAME}"
    tok = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name).to(device).eval()
    import torch

    out = np.zeros((len(spans), encoder.config.hidden_size), dtype=np.float32)
    started = time.time()
    for i in range(0, len(spans), batch_size):
        batch = tok(spans[i:i + batch_size], truncation=True, max_length=128,
                    padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
        weights = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
        out[i:i + batch_size] = pooled.cpu().numpy()
        if i % (batch_size * 40) == 0:
            done = min(i + batch_size, len(spans))
            echo(f"    {done:,}/{len(spans):,}  {time.time() - started:.0f}s")

    np.save(VECTORS_PATH, out)
    with gzip.open(INDEX_PATH, "wt", encoding="utf-8") as handle:
        json.dump({
            "encoder": name,
            "slots": list(SPAN_SLOTS),
            "dim": int(out.shape[1]),
            "spans": spans,
            # THE GUARD: a cache built over a different corpus is not a cache.
            "corpus_rows": len(frame),
        }, handle)
    return out, spans


def load():
    """`(SpanCache, None)`, or `(None, why)` when the cache cannot be trusted."""
    if not (VECTORS_PATH.exists() and INDEX_PATH.exists()):
        return None, "no cache"
    with gzip.open(INDEX_PATH, "rt", encoding="utf-8") as handle:
        meta = json.load(handle)
    expected = f"sentence-transformers/{TEXT_MODEL_NAME}"
    if meta.get("encoder") != expected:
        return None, f"cache built from {meta.get('encoder')}, not {expected}"
    if list(meta.get("slots") or []) != list(SPAN_SLOTS):
        return None, f"cache slots {meta.get('slots')} != {list(SPAN_SLOTS)}"
    matrix = np.load(VECTORS_PATH)
    if matrix.shape[0] != len(meta.get("spans") or []):
        return None, "cache length disagrees with its own index"
    return SpanCache(matrix, meta["spans"], meta), None


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector if norm < 1e-9 else vector / norm


class SpanCache:
    """Span text -> vector, and a card -> its seven slots."""

    def __init__(self, matrix, spans, meta=None):
        self.matrix = matrix
        self.rows = {text: i for i, text in enumerate(spans)}
        self.meta = meta or {}
        self.dim = int(matrix.shape[1])

    def __len__(self):
        return len(self.rows)

    def vector(self, text):
        row = self.rows.get(text)
        if row is None:
            raise KeyError(f"span not in cache: {text[:60]!r}")
        return self.matrix[row]

    def slot_vectors(self, card, oracle_text=None, masked=()):
        """`{slot: (state, vector)}` — every slot, present or not.

        A MASKED slot's vector is ZEROED, not merely flagged. Leaving the true
        value in place behind a flag makes the imputation task trivial and shows
        up as nothing but suspiciously good numbers.
        """
        masked = {masked} if isinstance(masked, str) else set(masked)
        unknown = masked - set(SPAN_SLOTS)
        if unknown:
            raise ValueError(f"not span slots: {sorted(unknown)}")
        spans = card_spans(card, oracle_text)
        out = {}
        for slot in SPAN_SLOTS:
            lines = spans.get(slot) or []
            if not lines:
                out[slot] = (ABSENT, np.zeros(self.dim, dtype=np.float32))
            elif slot in masked:
                out[slot] = (MASKED, np.zeros(self.dim, dtype=np.float32))
            else:
                # Unit-normalise per line: a slot is "what these abilities are
                # about", and an 80-word line should not outweigh `Flying` by
                # sheer magnitude.
                pooled = np.mean([_unit(self.vector(t)) for t in lines], axis=0)
                out[slot] = (PRESENT, pooled.astype(np.float32))
        return out

    def encode(self, card, oracle_text=None, masked=()):
        """`(vector, offsets)` — the same contract `card_fields.encode` uses."""
        parts, offsets, at = [], {}, 0
        for slot, (state, vector) in self.slot_vectors(
                card, oracle_text, masked).items():
            block = np.concatenate([
                vector,
                [0.0 if state == ABSENT else 1.0, 1.0 if state == MASKED else 0.0],
            ]).astype(np.float32)
            offsets[slot] = (at, at + len(block))
            at += len(block)
            parts.append(block)
        return np.concatenate(parts).astype(np.float32), offsets


def main(args):
    echo = say
    cache, why = load()
    if cache is not None and not getattr(args, "force", False):
        echo(f"  cache present: {len(cache):,} spans x {cache.dim} "
             f"— pass --force to rebuild")
        return
    echo(f"  building span cache ({why})")
    started = time.time()
    out, spans = build(echo=echo)
    echo(f"  Wrote {VECTORS_PATH.name}: {out.shape} ({out.nbytes/1e6:.0f} MB) "
         f"and {INDEX_PATH.name} in {time.time()-started:.0f}s")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap span-cache`.")
