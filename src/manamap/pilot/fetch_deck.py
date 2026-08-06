"""Pilot: decklist.txt -> Scryfall /cards/collection -> cards.json.

Two-phase resolution: a Moxfield export's `(SET) COLLECTOR [*F*]` annotations
are looked up **first** by set + collector number, so cards.json carries the
physical card the pilot owns — artist, border, finish, art crop — rather than
Scryfall's default printing. Names are the fallback for unannotated lines.

Idempotent twice over: the run short-circuits entirely when the decklist hash
matches what produced the existing cards.json (`--force` to override after
oracle errata), and the output itself is byte-identical for unchanged Scryfall
data (decklist order preserved, dict keys sorted, volatile fields and image
cache-busters stripped). Misspelled card names fail loudly, naming every miss.
"""

import hashlib
import json
import re
import time

import requests

from manamap.config import (
    SCRYFALL_BATCH_SIZE,
    SCRYFALL_COLLECTION_URL,
    SCRYFALL_MAX_RETRIES,
    SCRYFALL_REQUEST_DELAY_S,
    SCRYFALL_RETRY_BACKOFF_S,
    USER_AGENT,
)
from manamap.ingest.extract import get_colors
from manamap.pilot.common import (
    COMMANDER_SECTION_MARKERS,
    MAIN_SECTION_MARKERS,
    SIDEBOARD_SECTION_MARKERS,
    deck_dir,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


# Moxfield-style printing annotation: "Card Name (SET) COLLECTOR [*F*]"
_PRINTING_RE = re.compile(r"\s+\(([A-Z0-9]{2,6})\)\s+([\w-]+)$")


def parse_decklist(text):
    """Parse a decklist into entries, order preserved.

    Entry: {"name", "quantity", "is_commander", "foil"} plus
    optional "set"/"collector_number" when a Moxfield printing annotation is
    present.

    Marker-stripping order is load-bearing: `*CMDR*` and `*F*` come off the end
    of the line *before* `_PRINTING_RE` runs, because that pattern is anchored
    to `$`. Reorder these and every foil line silently loses its printing.

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
        if lowered in COMMANDER_SECTION_MARKERS:
            section = "commander"
            continue
        if lowered in MAIN_SECTION_MARKERS:
            section = "deck"
            continue
        if lowered in SIDEBOARD_SECTION_MARKERS:
            # There is no sideboard any more, but the MARKER still has to be
            # consumed: a pasted list that carries one would otherwise file every
            # card after it as maindeck and break the 100-card invariant. Stop
            # reading — everything below the line is out of the deck.
            break

        is_commander = section == "commander"
        if line.upper().endswith("*CMDR*"):
            is_commander = True
            line = line[: line.upper().rfind("*CMDR*")].strip()
        foil = False
        if line.upper().endswith("*F*"):
            foil = True
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
            "foil": foil,
        }
        printing = _PRINTING_RE.search(name)
        if printing:
            entry["name"] = name[: printing.start()].strip()
            entry["set"] = printing.group(1).lower()
            entry["collector_number"] = printing.group(2)
        entries.append(entry)
    return entries


def is_up_to_date(cards_path, decklist_sha256):
    """True if cards.json was already built from this exact decklist.

    Mirrors download_rules.is_up_to_date. This matters beyond the saved
    Scryfall round trips: a no-op re-fetch that rewrites cards.json would
    invalidate every downstream agent routine that reads it (see
    docs/agent-cost.md), silently costing a full regeneration.
    """
    if not cards_path.exists():
        return False
    with open(cards_path) as f:
        return json.load(f).get("decklist_sha256") == decklist_sha256


def _post_collection(identifiers):
    """POST identifier batches to /cards/collection. Returns (cards, not_found_ids)."""
    cards, not_found = [], []
    for start in range(0, len(identifiers), SCRYFALL_BATCH_SIZE):
        batch = identifiers[start : start + SCRYFALL_BATCH_SIZE]
        if start > 0:
            time.sleep(SCRYFALL_REQUEST_DELAY_S)
        payload = {"identifiers": batch}
        # Retry 429 (rate limit) and 5xx (transient outage) with linear backoff.
        # A single 503 used to abort the whole fetch mid-batch, losing every batch
        # already retrieved; observed live against /cards/collection.
        for attempt in range(SCRYFALL_MAX_RETRIES):
            resp = SESSION.post(SCRYFALL_COLLECTION_URL, json=payload, timeout=60)
            if resp.status_code != 429 and resp.status_code < 500:
                break
            if attempt < SCRYFALL_MAX_RETRIES - 1:
                time.sleep(SCRYFALL_RETRY_BACKOFF_S * (attempt + 1))
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

    The **primary** resolution path: a Moxfield export names the exact card the
    pilot owns, and resolving by name instead would silently substitute
    Scryfall's default printing. `resolve_entries` prefers these over name
    matches; name lookup is the fallback for unannotated lines. Unresolvable
    printings are simply absent from the result.
    """
    identifiers = [{"set": s, "collector_number": cn} for s, cn in printings]
    cards, _ = _post_collection(identifiers)
    return {(card.get("set", ""), card.get("collector_number", "")): card for card in cards}


def stable_image_url(url):
    """Drop Scryfall's cache-busting query string.

    The `?1783903430` suffix changes whenever Scryfall re-renders an image,
    which would churn cards.json on every re-fetch for no visual difference.
    """
    if not url:
        return url
    return url.split("?", 1)[0]


def _shape_face(face):
    image_uris = face.get("image_uris") or {}
    return {
        "name": face.get("name"),
        "mana_cost": face.get("mana_cost", ""),
        "type_line": face.get("type_line", ""),
        "oracle_text": face.get("oracle_text", ""),
        # Face colours are the ONLY place a transform/modal-DFC's colours live —
        # Scryfall omits the top-level `colors` for those layouts entirely. Dropping
        # this field left cards.json with no way to recover them. See shape_card.
        "colors": face.get("colors", []),
        "power": face.get("power"),
        "toughness": face.get("toughness"),
        "image": stable_image_url(image_uris.get("normal")),
        "art_crop": stable_image_url(image_uris.get("art_crop")),
    }


def shape_card(sc, quantity, is_commander, foil=False):
    """Project a Scryfall card object onto the cards.json schema.

    Printing metadata (set, collector number, artist, finishes, borderless/frame
    effects, art_crop) is what lets the manual show the *physical* cards in the
    deck — the borderless Secret Lair, not a default reprint. `art_crop` in
    particular is full-bleed art with no frame, which is what reads as magazine
    photography rather than a card scan.

    Two fields look alike and mean different things: `finishes` is Scryfall's
    list of finishes this printing was *released in*, while `foil` comes from
    the decklist's `*F*` marker and says whether *this pilot's copy* is foil.
    The renderer's holographic treatment keys off `foil`.
    """
    image_uris = sc.get("image_uris") or {}
    faces = sc.get("card_faces") or []
    image = image_uris.get("normal")
    art_crop = image_uris.get("art_crop")
    if image is None and faces:
        front = faces[0].get("image_uris") or {}
        image = front.get("normal")
        art_crop = front.get("art_crop")
    return {
        "name": sc["name"],
        "quantity": quantity,
        "is_commander": is_commander,
        "mana_cost": sc.get("mana_cost", ""),
        "cmc": sc.get("cmc", 0.0),
        "type_line": sc.get("type_line", ""),
        "oracle_text": sc.get("oracle_text")
        or " // ".join(f.get("oracle_text", "") for f in faces),
        # NOT sc.get("colors"): transform and modal-DFC cards carry no top-level
        # `colors` — it lives on card_faces. get_colors() unions the faces and falls
        # back to the top level for split/adventure/flip, which is the same helper
        # the card pipeline uses for cards.csv, so the two agree by construction.
        "colors": get_colors(sc),
        "color_identity": sc.get("color_identity", []),
        "keywords": sc.get("keywords", []),
        "power": sc.get("power"),
        "toughness": sc.get("toughness"),
        "loyalty": sc.get("loyalty"),
        "layout": sc.get("layout", "normal"),
        "image": stable_image_url(image),
        "art_crop": stable_image_url(art_crop),
        "scryfall_uri": sc.get("scryfall_uri"),
        "card_faces": [_shape_face(f) for f in faces],
        # Printing identity — which physical card this is.
        "set": sc.get("set"),
        "set_name": sc.get("set_name"),
        "collector_number": sc.get("collector_number"),
        "artist": sc.get("artist"),
        "border_color": sc.get("border_color"),
        "frame_effects": sc.get("frame_effects") or [],
        "finishes": sc.get("finishes") or [],
        "foil": bool(foil),
    }


def resolve_entries(entries, by_name, by_printing=None):
    """Match decklist entries to fetched cards. Entry names may be a single face
    of a multi-face card (Scryfall resolves them; response name is the full
    ' // ' name); entries with a printing annotation fall back to (set, cn)
    lookup when the name misses. Duplicate names (e.g. several basic-land
    printings) merge into one entry, quantities summed, first position kept —
    note that means the *first* printing's artist, set and art represent every
    copy, so a deck with four different basic-land arts credits only one.
    Returns (cards, unmatched_names)."""
    by_printing = by_printing or {}
    # Secondary index: front-face name -> full card.
    by_face = {}
    for card in by_name.values():
        for face in card.get("card_faces") or []:
            by_face.setdefault(face["name"].lower(), card)

    cards, unmatched = [], []
    merged = {}  # name_lower -> shaped card
    for entry in entries:
        key = entry["name"].lower()
        # The decklist's own printing annotation wins: it names the physical
        # card the pilot owns. Name lookup is the fallback for unannotated
        # lines (and for printings Scryfall can't resolve).
        sc = None
        if "set" in entry:
            sc = by_printing.get((entry["set"], entry["collector_number"]))
        if sc is None:
            sc = by_name.get(key) or by_face.get(key)
        if sc is None:
            unmatched.append(entry["name"])
            continue
        merge_key = sc["name"].lower()
        if merge_key in merged:
            merged[merge_key]["quantity"] += entry["quantity"]
            merged[merge_key]["is_commander"] |= entry["is_commander"]
            continue
        shaped = shape_card(sc, entry["quantity"], entry["is_commander"],
                            entry.get("foil", False))
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
    decklist_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = deck_dir(args.slug) / "cards.json"
    if not getattr(args, "force", False) and is_up_to_date(out, decklist_sha256):
        print(f"  Already up to date — skipping Scryfall fetch ({out}).")
        print("  (The hash covers the decklist, not Scryfall's data — use "
              "--force after an oracle update.)")
        return

    entries = parse_decklist(text)
    if not entries:
        raise SystemExit(f"{path} parsed to zero cards — is it empty?")
    print(f"Parsed {len(entries)} decklist entries ({sum(e['quantity'] for e in entries)} cards)")

    # Printings first — a Moxfield export names the exact card the pilot owns,
    # and resolving by name alone would silently substitute a default reprint.
    wanted_printings = sorted({
        (e["set"], e["collector_number"]) for e in entries if "set" in e
    })
    by_printing = fetch_printings(wanted_printings) if wanted_printings else {}
    if wanted_printings:
        print(f"  Resolved {len(by_printing)}/{len(wanted_printings)} exact printing(s)")

    # Names still needing resolution: unannotated lines, plus any printing
    # Scryfall couldn't find.
    unique_names = sorted({
        e["name"] for e in entries
        if "set" not in e or (e["set"], e["collector_number"]) not in by_printing
    })
    by_name, not_found = fetch_collection(unique_names) if unique_names else ({}, [])

    cards, unmatched = resolve_entries(entries, by_name, by_printing)
    if unmatched:
        raise SystemExit(
            "Scryfall could not resolve these card names (fix the decklist):\n  - "
            + "\n  - ".join(sorted(set(unmatched)))
        )

    doc = {
        "deck": args.slug,
        "decklist_sha256": decklist_sha256,
        "cards": cards,
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    total = sum(c["quantity"] for c in cards)
    commanders = [c["name"] for c in cards if c["is_commander"]]
    print(
        f"Wrote {out}: {total} cards, "
        f"commander: {', '.join(commanders) or 'NONE FLAGGED'}"
    )


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fetch-deck <slug>`.")
