"""Step 5: Generate final 128-dim embeddings and card metadata CSV."""

import numpy as np
import pandas as pd
import torch

from config import (
    CARD_FEATURES_PATH,
    CARD_METADATA_PATH,
    EMBEDDINGS_PATH,
    MODEL_PATH,
    OUTPUT_CSV_PATH,
    TEXT_EMBEDDINGS_PATH,
)
from model import CardEmbeddingModel
from train import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("Loading trained model...")
    model = CardEmbeddingModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Load features
    print("Loading features...")
    text_embs = np.load(TEXT_EMBEDDINGS_PATH)
    features = dict(np.load(CARD_FEATURES_PATH))
    n = len(text_embs)
    print(f"  {n:,} cards")

    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 512
    all_embs = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            text = torch.tensor(text_embs[start:end], dtype=torch.float32).to(device)
            supertype = torch.tensor(features["supertype"][start:end], dtype=torch.long).to(device)
            rarity = torch.tensor(features["rarity"][start:end], dtype=torch.long).to(device)
            ci = torch.tensor(features["color_identity"][start:end], dtype=torch.long).to(device)
            layout = torch.tensor(features["layout"][start:end], dtype=torch.long).to(device)
            continuous = torch.tensor(features["continuous"][start:end], dtype=torch.float32).to(device)
            keywords = torch.tensor(features["keywords"][start:end], dtype=torch.float32).to(device)

            emb = model(text, supertype, rarity, ci, layout, continuous, keywords)
            all_embs.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    print(f"  Embeddings shape: {embeddings.shape}")

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"  Saved {EMBEDDINGS_PATH}")

    # Build metadata CSV
    print("Building card metadata...")
    df = pd.read_csv(OUTPUT_CSV_PATH)

    # Parse WUBRG from color_identity
    wubrg = list("WUBRG")
    for c in wubrg:
        df[f"color_{c}"] = df["color_identity"].fillna("").apply(
            lambda ci, color=c: int(color in [x.strip() for x in ci.split(",")]) if ci else 0
        )

    metadata = df[[
        "oracle_id", "name", "supertype", "primary_color",
        "color_identity", "cmc", "rarity",
        "color_W", "color_U", "color_B", "color_R", "color_G",
    ]].copy()

    metadata.to_csv(CARD_METADATA_PATH, index=False)
    print(f"  Saved {CARD_METADATA_PATH} — {len(metadata):,} rows")
    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()
