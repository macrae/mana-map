"""Step 6: Reduce 128-dim embeddings to 2D with PaCMAP and export JSON for visualization."""

import json

import numpy as np
import pacmap
import pandas as pd

from config import (
    ABILITY_EMBEDDINGS_PATH,
    ABILITY_PROJECTION_PATH,
    CARD_METADATA_PATH,
    EMBEDDINGS_PATH,
    LEGALITY_FORMATS,
    OUTPUT_CSV_PATH,
    PROJECTION_PATH,
)


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D using PaCMAP."""
    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)


def _safe_str(val) -> str | None:
    """Return a non-empty string or None."""
    if pd.isna(val) or val == "":
        return None
    return str(val)


def _legal_formats(cards_row: pd.Series) -> str | None:
    """Build comma-joined string of formats where the card is legal."""
    fmts = [fmt for fmt in LEGALITY_FORMATS if cards_row.get(f"legal_{fmt}") == "legal"]
    return ",".join(fmts) if fmts else None


def build_viz_records(
    projection: np.ndarray,
    metadata: pd.DataFrame,
    cards: pd.DataFrame,
) -> list[dict]:
    """Merge 2D coordinates with card metadata and full card data into compact records."""
    records = []
    for i in range(len(metadata)):
        meta = metadata.iloc[i]
        card = cards.iloc[i]
        rec = {
            "x": round(float(projection[i, 0]), 4),
            "y": round(float(projection[i, 1]), 4),
            "n": str(meta["name"]),
            "s": str(meta["supertype"]),
            "c": str(meta["primary_color"]),
            "m": float(meta["cmc"]) if pd.notna(meta["cmc"]) else 0.0,
            "r": str(meta["rarity"]),
            "t": _safe_str(card["type_line"]),
            "mc": _safe_str(card["mana_cost"]),
            "o": _safe_str(card["oracle_text"]),
            "k": _safe_str(card["keywords"]),
            "p": _safe_str(card["power"]),
            "th": _safe_str(card["toughness"]),
            "l": _safe_str(card["loyalty"]),
            "d": _safe_str(card["defense"]),
            "ci": _safe_str(card["color_identity"]),
            "er": int(card["edhrec_rank"]) if pd.notna(card["edhrec_rank"]) else None,
            "f": _legal_formats(card),
        }
        # Strip None values to save space
        records.append({k: v for k, v in rec.items() if v is not None})
    return records


def export_json(records: list[dict], output_path) -> None:
    """Write records as compact JSON (no whitespace)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, separators=(",", ":"))


def run_reduce(emb_path, output_path):
    """Run PaCMAP reduction and export JSON for a given embedding file.

    Returns the list of records.
    """
    print(f"  Loading {emb_path}...")
    embeddings = np.load(emb_path)
    print(f"    Shape: {embeddings.shape}")

    metadata = pd.read_csv(CARD_METADATA_PATH)
    cards = pd.read_csv(OUTPUT_CSV_PATH)

    assert len(embeddings) == len(metadata), (
        f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata rows"
    )

    print(f"  Reducing to 2D with PaCMAP...")
    projection = reduce_to_2d(embeddings)

    print(f"  Building visualization records...")
    records = build_viz_records(projection, metadata, cards)

    print(f"  Exporting {len(records):,} records to {output_path}...")
    export_json(records, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Wrote {size_mb:.1f} MB")
    return records


def main():
    print("Loading metadata...")
    metadata = pd.read_csv(CARD_METADATA_PATH)
    print(f"  {len(metadata):,} cards")

    # ── Default projection ────────────────────────────────────────────────
    print("\n[Default Map] Color + Type projection")
    run_reduce(EMBEDDINGS_PATH, PROJECTION_PATH)

    # ── Ability projection ────────────────────────────────────────────────
    if ABILITY_EMBEDDINGS_PATH.exists():
        print("\n[Ability Map] Abilities projection")
        run_reduce(ABILITY_EMBEDDINGS_PATH, ABILITY_PROJECTION_PATH)
    else:
        print(f"\n  Skipping ability projection ({ABILITY_EMBEDDINGS_PATH} not found)")

    print(f"\nTo view: python -m http.server 8000")
    print(f"  Then open http://localhost:8000/viz/index.html")


if __name__ == "__main__":
    main()
