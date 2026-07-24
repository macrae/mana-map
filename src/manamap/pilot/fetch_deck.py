"""Pilot: decklist.txt -> Scryfall /cards/collection -> cards.json.

Idempotent: re-running against unchanged Scryfall data produces byte-identical
output (decklist order preserved, dict keys sorted, volatile fields stripped).
Misspelled card names fail loudly, naming every missing card.
"""

import hashlib
import json
import re
import time

import requests

from manamap.config import (
    SCRYFALL_BATCH_SIZE,
    SCRYFALL_COLLECTION_URL,
    SCRYFALL_REQUEST_DELAY_S,
    USER_AGENT,
)
from manamap.pilot.common import deck_dir

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


# Moxfield-style printing annotation: "Card Name (SET) COLLECTOR [*F*]"
_PRINTING_RE = re.compile(r"\s+\(([A-Z0-9]{2,6})\)\s+([\w-]+)$")


def parse_decklist(text):
    """Parse a decklist into entries, order preserved.

    Entry: {"name", "quantity", "is_commander", "is_sideboard"} plus optional
    "set"/"collector_number" when a Moxfield printing annotation is present.

    Supports: `1 Card Name`, `1x Card Name`, bare `Card Name` (quantity 1),
    Moxfield `(SET) COLLECTOR` printing suffixes and `*F*` foil markers,
    `Commander:`/`Commanders:` and `Sideboard:` section headers, a trailing
    `*CMDR*` marker, `#` and `//` comment lines, blank lines.
    """
    entries = []
    section = "deck"
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        lowered = line.lower().rstrip(":")
        if lowered in ("commander", "commanders"):
            section = "commander"
            continue
        if lowered in ("deck", "mainboard", "main"):
            section = "deck"
            continue
        if lowered in ("sideboard", "side", "maybeboard", "considering"):
            section = "sideboard"
            continue

        is_commander = section == "commander"
        if line.upper().endswith("*CMDR*"):
            is_commander = True
            line = line[: line.upper().rfind("*CMDR*")].strip()
        if line.upper().endswith("*F*"):
            line = line[: line.upper().rfind("*F*")].strip()

        quantity = 1
        parts = line.split(None, 1)
        head = parts[0].lower().rstrip("x")
        if head.isdigit() and len(parts) == 2:
            quantity = int(head)
            name = parts[1].strip()
        else:
            name = line

        entry = {
            "name": name,
            "quantity": quantity,
            "is_commander": is_commander,
            "is_sideboard": section == "sideboard",
        }
        printing = _PRINTING_RE.search(name)
        if printing:
            entry["name"] = name[: printing.start()].strip()
            entry["set"] = printing.group(1).lower()
            entry["collector_number"] = printing.group(2)
        entries.append(entry)
    return entries


def _post_collection(identifiers):
    """POST identifier batches to /cards/collection. Returns (cards, not_found_ids)."""
    cards, not_found = [], []
    for start in range(0, len(identifiers), SCRYFALL_BATCH_SIZE):
        batch = identifiers[start : start + SCRYFALL_BATCH_SIZE]
        if start > 0:
            time.sleep(SCRYFALL_REQUEST_DELAY_S)
        payload = {"identifiers": batch}
        resp = SESSION.post(SCRYFALL_COLLECTION_URL, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(1.0)
            resp = SESSION.post(SCRYFALL_COLLECTION_URL, json=payload, timeout=60)
        resp.raise_for_status()
        doc = resp.json()
        cards.extend(doc.get("data", []))
        not_found.extend(doc.get("not_found", []))
    return cards, not_found


def fetch_collection(names):
    """Fetch by name. Returns (by_name_lower, not_found_names)."""
    cards, not_found = _post_collection([{"name": n} for n in names])
    by_name = {card["name"].lower(): card for card in cards}
    return by_name, [nf.get("name", "?") for nf in not_found]


def fetch_printings(printings):
    """Fetch by (set, collector_number). Returns {(set, cn): card}.

    Fallback for names Scryfall can't resolve (tokens, art cards in
    sideboards). Unresolvable printings are simply absent from the result.
    """
    identifiers = [{"set": s, "collector_number": cn} for s, cn in printings]
    cards, _ = _post_collection(identifiers)
    return {(card.get("set", ""), card.get("collector_number", "")): card for card in cards}


def _shape_face(face):
    image_uris = face.get("image_uris") or {}
    return {
        "name": face.get("name"),
        "mana_cost": face.get("mana_cost", ""),
        "type_line": face.get("type_line", ""),
        "oracle_text": face.get("oracle_text", ""),
        "power": face.get("power"),
        "toughness": face.get("toughness"),
        "image": image_uris.get("normal"),
    }


def shape_card(sc, quantity, is_commander, is_sideboard=False):
    """Project a Scryfall card object onto the cards.json schema."""
    image_uris = sc.get("image_uris") or {}
    faces = sc.get("card_faces") or []
    image = image_uris.get("normal")
    if image is None and faces:
        image = (faces[0].get("image_uris") or {}).get("normal")
    return {
        "name": sc["name"],
        "quantity": quantity,
        "is_commander": is_commander,
        "is_sideboard": is_sideboard,
        "mana_cost": sc.get("mana_cost", ""),
        "cmc": sc.get("cmc", 0.0),
        "type_line": sc.get("type_line", ""),
        "oracle_text": sc.get("oracle_text")
        or " // ".join(f.get("oracle_text", "") for f in faces),
        "colors": sc.get("colors", []),
        "color_identity": sc.get("color_identity", []),
        "keywords": sc.get("keywords", []),
        "power": sc.get("power"),
        "toughness": sc.get("toughness"),
        "loyalty": sc.get("loyalty"),
        "layout": sc.get("layout", "normal"),
        "image": image,
        "scryfall_uri": sc.get("scryfall_uri"),
        "card_faces": [_shape_face(f) for f in faces],
    }


def resolve_entries(entries, by_name, by_printing=None):
    """Match decklist entries to fetched cards. Entry names may be a single face
    of a multi-face card (Scryfall resolves them; response name is the full
    ' // ' name); entries with a printing annotation fall back to (set, cn)
    lookup when the name misses. Duplicate names (e.g. several basic-land
    printings) merge into one entry, quantities summed, first position kept.
    Returns (cards, unmatched_names)."""
    by_printing = by_printing or {}
    # Secondary index: front-face name -> full card.
    by_face = {}
    for card in by_name.values():
        for face in card.get("card_faces") or []:
            by_face.setdefault(face["name"].lower(), card)

    cards, unmatched = [], []
    merged = {}  # (name_lower, is_sideboard) -> shaped card
    for entry in entries:
        key = entry["name"].lower()
        sc = by_name.get(key) or by_face.get(key)
        if sc is None and "set" in entry:
            sc = by_printing.get((entry["set"], entry["collector_number"]))
        if sc is None:
            unmatched.append(entry["name"])
            continue
        merge_key = (sc["name"].lower(), entry["is_sideboard"])
        if merge_key in merged:
            merged[merge_key]["quantity"] += entry["quantity"]
            merged[merge_key]["is_commander"] |= entry["is_commander"]
            continue
        shaped = shape_card(sc, entry["quantity"], entry["is_commander"], entry["is_sideboard"])
        merged[merge_key] = shaped
        cards.append(shaped)
    return cards, unmatched


def main(args):
    path = deck_dir(args.slug) / "decklist.txt"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — paste the decklist there (one card per line, "
            f"commander under a 'Commander:' header or marked *CMDR*)."
        )
    text = path.read_text()
    entries = parse_decklist(text)
    if not entries:
        raise SystemExit(f"{path} parsed to zero cards — is it empty?")
    print(f"Parsed {len(entries)} decklist entries ({sum(e['quantity'] for e in entries)} cards)")

    unique_names = sorted({e["name"] for e in entries})
    by_name, not_found = fetch_collection(unique_names)

    # Second pass: entries whose names missed but that carry a printing
    # annotation (sideboard tokens/art cards resolve by set + collector number).
    unresolved_printings = sorted({
        (e["set"], e["collector_number"])
        for e in entries
        if "set" in e and e["name"].lower() not in by_name
    })
    by_printing = fetch_printings(unresolved_printings) if unresolved_printings else {}
    if by_printing:
        print(f"  Resolved {len(by_printing)} entr(ies) by set/collector number")

    cards, unmatched = resolve_entries(entries, by_name, by_printing)
    if unmatched:
        raise SystemExit(
            "Scryfall could not resolve these card names (fix the decklist):\n  - "
            + "\n  - ".join(sorted(set(unmatched)))
        )

    doc = {
        "deck": args.slug,
        "decklist_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "cards": cards,
    }
    out = deck_dir(args.slug) / "cards.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    main_total = sum(c["quantity"] for c in cards if not c["is_sideboard"])
    side_total = sum(c["quantity"] for c in cards if c["is_sideboard"])
    commanders = [c["name"] for c in cards if c["is_commander"]]
    print(
        f"Wrote {out}: {main_total} main + {side_total} sideboard, "
        f"commander: {', '.join(commanders) or 'NONE FLAGGED'}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fetch-deck <slug>`.")
