"""The corpus as the MODEL sees it: `cards.csv` plus what the CSV threw away.

Two facts the model needs live only in the raw Scryfall dump:

* **`oracle_text` with its newlines.** `extract.py:157` flattens them so the CSV
  stays one row per card, which is right for `embedding_text` and fatal for
  ability structure — read from the CSV, every card has exactly ONE ability line.
* **`produced_mana`.** Nothing in `cards.csv` says Command Tower taps for five
  colours. Its type line is `Land`, it has no mana cost, no power, no subtypes —
  before this, the second-most-played card in Commander encoded as almost
  entirely ABSENT while an unplayable vanilla creature encoded richly.

`enrich` merges both into a CSV record. It ALWAYS sets `produced_mana`, even to
an empty list, because Scryfall omits the key for the 32,190 cards that produce
nothing — and "produces nothing" and "nobody enriched this record" must not
encode identically. A field that reads a missing key as `False` would report
every card as producing no mana the moment a caller forgot this step, silently
and plausibly.
"""

import gzip
import json

from manamap.config import RAW_JSON_PATH


def _text(value):
    """A cell -> a string, with NaN meaning empty rather than the word "nan"."""
    if value is None or value != value:                        # NaN != NaN
        return ""
    return str(value).strip()


def load_dump(path=RAW_JSON_PATH):
    """`{oracle_id: {oracle_text, produced_mana}}` from the raw dump.

    A double-faced card carries its text on `card_faces` and leaves the top-level
    `oracle_text` empty; both faces are joined, because the CSV's `type_line`
    already describes the card as a whole with ` // `.
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
            out[card.get("oracle_id")] = {
                "oracle_text": text,
                # ALWAYS a list. Scryfall omits the key entirely for the 32,190
                # cards that make no mana; an absent key here would mean
                # "unknown", which is a different thing.
                "produced_mana": list(card.get("produced_mana") or []),
            }
    return out


def enrich(card, dump):
    """A CSV record + the dump -> the record the encoders expect.

    Returns a COPY. Mutating the caller's row would make the enrichment
    order-dependent and invisible, which is how the flattened-oracle bug survived
    as long as it did.
    """
    extra = dump.get(card.get("oracle_id"))
    out = dict(card)
    if extra is None:
        return out
    out["oracle_text"] = extra["oracle_text"]
    out["produced_mana"] = list(extra["produced_mana"])
    return out


def enriched(cards, dump=None):
    """Every record, enriched. Raises if the dump and the corpus disagree."""
    dump = load_dump() if dump is None else dump
    missing = [c.get("name") for c in cards if c.get("oracle_id") not in dump]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(cards)} cards are absent from "
            f"{RAW_JSON_PATH.name} (e.g. {missing[:3]}) — re-run `manamap download`")
    return [enrich(card, dump) for card in cards]
