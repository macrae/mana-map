"""Pilot: decklist.txt -> Scryfall /cards/collection -> cards.json.

Idempotent: re-running against unchanged Scryfall data produces byte-identical
output (decklist order preserved, dict keys sorted, volatile fields stripped).
Misspelled card names fail loudly, naming every missing card.
"""

import hashlib
import json
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


def parse_decklist(text):
    """Parse a decklist into [{"name", "quantity", "is_commander"}], order preserved.

    Supports: `1 Card Name`, `1x Card Name`, bare `Card Name` (quantity 1),
    a `Commander:`/`Commanders:` section header, a trailing `*CMDR*` marker,
    `#` and `//` comment lines, blank lines.
    """
    entries = []
    in_commander_section = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        lowered = line.lower().rstrip(":")
        if lowered in ("commander", "commanders"):
            in_commander_section = True
            continue
        if lowered in ("deck", "mainboard", "main"):
            in_commander_section = False
            continue

        is_commander = in_commander_section
        if line.upper().endswith("*CMDR*"):
            is_commander = True
            line = line[: line.upper().rfind("*CMDR*")].strip()

        quantity = 1
        parts = line.split(None, 1)
        head = parts[0].lower().rstrip("x")
        if head.isdigit() and len(parts) == 2:
            quantity = int(head)
            name = parts[1].strip()
        else:
            name = line
        entries.append({"name": name, "quantity": quantity, "is_commander": is_commander})
    return entries


def fetch_collection(names):
    """POST names to /cards/collection in batches. Returns (by_name_lower, not_found)."""
    by_name = {}
    not_found = []
    for start in range(0, len(names), SCRYFALL_BATCH_SIZE):
        batch = names[start : start + SCRYFALL_BATCH_SIZE]
        if start > 0:
            time.sleep(SCRYFALL_REQUEST_DELAY_S)
        payload = {"identifiers": [{"name": n} for n in batch]}
        resp = SESSION.post(SCRYFALL_COLLECTION_URL, json=payload, timeout=60)
        if resp.status_code == 429:
            time.sleep(1.0)
            resp = SESSION.post(SCRYFALL_COLLECTION_URL, json=payload, timeout=60)
        resp.raise_for_status()
        doc = resp.json()
        for card in doc.get("data", []):
            by_name[card["name"].lower()] = card
        not_found.extend(nf.get("name", "?") for nf in doc.get("not_found", []))
    return by_name, not_found


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


def shape_card(sc, quantity, is_commander):
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


def resolve_entries(entries, by_name):
    """Match decklist entries to fetched cards. Entry names may be a single face
    of a multi-face card (Scryfall resolves them; response name is the full
    ' // ' name). Returns (cards, unmatched_names)."""
    # Secondary index: front-face name -> full card.
    by_face = {}
    for card in by_name.values():
        for face in card.get("card_faces") or []:
            by_face.setdefault(face["name"].lower(), card)

    cards, unmatched = [], []
    for entry in entries:
        key = entry["name"].lower()
        sc = by_name.get(key) or by_face.get(key)
        if sc is None:
            unmatched.append(entry["name"])
            continue
        cards.append(shape_card(sc, entry["quantity"], entry["is_commander"]))
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

    by_name, not_found = fetch_collection([e["name"] for e in entries])
    cards, unmatched = resolve_entries(entries, by_name)
    missing = sorted(set(not_found) | set(unmatched))
    if missing:
        raise SystemExit(
            "Scryfall could not resolve these card names (fix the decklist):\n  - "
            + "\n  - ".join(missing)
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
    total = sum(c["quantity"] for c in cards)
    commanders = [c["name"] for c in cards if c["is_commander"]]
    print(f"Wrote {out}: {total} cards, commander: {', '.join(commanders) or 'NONE FLAGGED'}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fetch-deck <slug>`.")
