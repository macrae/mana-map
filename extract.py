"""Step 2: Parse Scryfall JSON into a clean CSV with derived columns."""

import json

import pandas as pd

from config import (
    EXCLUDED_LAYOUTS,
    LEGALITY_FORMATS,
    MULTI_FACE_LAYOUTS,
    OUTPUT_CSV_PATH,
    RAW_JSON_PATH,
    SUPERTYPE_PRIORITY,
)
from mechanical_tags import tag_oracle_text


def combine_faces(card, field):
    """Combine a field across card_faces with ' // ' separator."""
    faces = card.get("card_faces", [])
    values = [f.get(field, "") or "" for f in faces]
    return " // ".join(v for v in values if v)


def get_colors(card):
    """Get colors for a card, handling multi-face differences.

    Split/adventure/flip: colors live on the top-level card.
    Transform/modal_dfc/reversible: colors live on individual faces.
    Falls back to top-level if faces lack colors.
    """
    wubrg = list("WUBRG")
    faces = card.get("card_faces", [])

    # Try union of face colors first
    all_colors = set()
    for face in faces:
        face_colors = face.get("colors")
        if face_colors:
            all_colors.update(face_colors)

    # Fall back to top-level colors (split, adventure, flip)
    if not all_colors:
        all_colors.update(card.get("colors") or [])

    return [c for c in wubrg if c in all_colors]


def first_face_value(card, field):
    """Get a field from whichever face has it (first wins)."""
    for face in card.get("card_faces", []):
        val = face.get(field)
        if val is not None:
            return val
    return None


def derive_primary_color(colors, color_identity):
    """Derive a single primary color category.

    Uses color_identity as fallback for lands (which often have empty colors).
    """
    effective = colors if colors else color_identity
    if not effective:
        return "Colorless"
    if len(effective) > 1:
        return "Multicolor"
    return effective[0]


def derive_supertype(type_line):
    """Pick the highest-priority supertype from the front face type line."""
    if not type_line:
        return "Unknown"
    # For multi-face, only look at the front face (before ' // ')
    front = type_line.split(" // ")[0]
    for st in SUPERTYPE_PRIORITY:
        if st in front:
            return st
    # Old-style "Summon X" type lines are creatures
    if front.lower().startswith("summon"):
        return "Creature"
    # Handle lowercase Un-card types (e.g. "instant")
    front_lower = front.lower()
    for st in SUPERTYPE_PRIORITY:
        if st.lower() in front_lower:
            return st
    return "Unknown"


def build_embedding_text(name, type_line, oracle_text, keywords):
    """Build a combined text string for future embedding work."""
    parts = [name]
    if type_line:
        parts.append(type_line)
    if oracle_text:
        parts.append(oracle_text)
    if keywords:
        parts.append(f"Keywords: {keywords}")
    return ". ".join(parts)


def process_card(card):
    """Transform a single Scryfall card object into a flat row dict."""
    layout = card.get("layout", "")
    is_multi = layout in MULTI_FACE_LAYOUTS

    # Core fields — multi-face cards combine across faces
    if is_multi and "card_faces" in card:
        name = card.get("name", "")
        oracle_text = combine_faces(card, "oracle_text")
        mana_cost = combine_faces(card, "mana_cost")
        type_line = card.get("type_line", "")
        colors = get_colors(card)
        power = first_face_value(card, "power")
        toughness = first_face_value(card, "toughness")
        loyalty = first_face_value(card, "loyalty")
        defense = first_face_value(card, "defense")
    else:
        name = card.get("name", "")
        oracle_text = card.get("oracle_text", "") or ""
        mana_cost = card.get("mana_cost", "") or ""
        type_line = card.get("type_line", "") or ""
        colors = card.get("colors", [])
        power = card.get("power")
        toughness = card.get("toughness")
        loyalty = card.get("loyalty")
        defense = card.get("defense")

    # Flatten newlines in text fields so CSV stays one-row-per-card
    oracle_text = oracle_text.replace("\n", " ")

    color_identity = card.get("color_identity", [])
    keywords = card.get("keywords", [])
    keywords_str = ", ".join(keywords)

    # Derived columns
    primary_color = derive_primary_color(colors, color_identity)
    supertype = derive_supertype(type_line)
    embedding_text = build_embedding_text(name, type_line, oracle_text, keywords_str)

    # Mechanical tags from oracle text + keywords + type line
    combined_text = oracle_text
    if keywords_str:
        combined_text += " " + keywords_str
    if type_line:
        combined_text += " " + type_line
    tags = tag_oracle_text(combined_text)

    # Legalities
    legalities = card.get("legalities", {})
    legal_cols = {
        f"legal_{fmt}": legalities.get(fmt, "not_legal")
        for fmt in LEGALITY_FORMATS
    }

    row = {
        "oracle_id": card.get("oracle_id", ""),
        "name": name,
        "layout": layout,
        "type_line": type_line,
        "supertype": supertype,
        "mana_cost": mana_cost,
        "cmc": card.get("cmc", 0.0),
        "colors": ", ".join(colors),
        "color_identity": ", ".join(color_identity),
        "primary_color": primary_color,
        "oracle_text": oracle_text,
        "keywords": keywords_str,
        "power": power,
        "toughness": toughness,
        "loyalty": loyalty,
        "defense": defense,
        "rarity": card.get("rarity", ""),
        "set_code": card.get("set", ""),
        "set_name": card.get("set_name", ""),
        "released_at": card.get("released_at", ""),
        "artist": card.get("artist", ""),
        "flavor_text": (card.get("flavor_text") or "").replace("\n", " "),
        "edhrec_rank": card.get("edhrec_rank"),
        "reserved": card.get("reserved", False),
        "mechanical_tags": ", ".join(tags),
        "embedding_text": embedding_text,
        **legal_cols,
    }
    return row


def main():
    print("Loading JSON...")
    with open(RAW_JSON_PATH, "r") as f:
        cards = json.load(f)
    print(f"  Loaded {len(cards):,} total entries.")

    # Filter excluded layouts
    filtered = [c for c in cards if c.get("layout") not in EXCLUDED_LAYOUTS]
    print(f"  After filtering excluded layouts: {len(filtered):,} cards.")

    # Process each card
    print("Processing cards...")
    rows = [process_card(c) for c in filtered]

    # Build DataFrame and write CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  Wrote {len(df):,} rows to {OUTPUT_CSV_PATH}")

    # Summary stats
    print(f"\nSupertype distribution:")
    print(df["supertype"].value_counts().to_string())
    print(f"\nPrimary color distribution:")
    print(df["primary_color"].value_counts().to_string())


if __name__ == "__main__":
    main()
