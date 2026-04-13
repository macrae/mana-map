"""Step 5: Generate final 128-dim embeddings and card metadata CSV."""

import numpy as np
import pandas as pd
import torch

from config import (
    ABILITY_CI_EMBEDDING_DIM,
    ABILITY_EMBEDDINGS_PATH,
    ABILITY_KEYWORD_EMBEDDING_DIM,
    ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
    ABILITY_MODEL_PATH,
    CARD_FEATURES_PATH,
    CARD_METADATA_PATH,
    EMBEDDINGS_PATH,
    MECHANICAL_TAG_DIM,
    MECHANICAL_TAGS_PATH,
    MODEL_PATH,
    OUTPUT_CSV_PATH,
    TEXT_EMBEDDINGS_PATH,
)
from model import CardEmbeddingModel
from train import get_device


def run_embed(model_path, output_path, model_kwargs=None, use_mechanical_tags=False):
    """Generate embeddings for all cards using a given model.

    Args:
        model_path: Path to model checkpoint.
        output_path: Path to save embeddings .npy.
        model_kwargs: Dict of kwargs for CardEmbeddingModel constructor.
        use_mechanical_tags: Whether to pass mechanical tags to the model.
    """
    device = get_device()

    model = CardEmbeddingModel(**(model_kwargs or {})).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    text_embs = np.load(TEXT_EMBEDDINGS_PATH)
    features = dict(np.load(CARD_FEATURES_PATH))
    mechanical_tags = None
    if use_mechanical_tags:
        mechanical_tags = np.load(MECHANICAL_TAGS_PATH)

    n = len(text_embs)
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

            kwargs = {}
            if use_mechanical_tags and mechanical_tags is not None:
                kwargs["mechanical_tags"] = torch.tensor(
                    mechanical_tags[start:end], dtype=torch.float32
                ).to(device)

            emb = model(text, supertype, rarity, ci, layout, continuous, keywords, **kwargs)
            all_embs.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    np.save(output_path, embeddings)
    return embeddings


def main():
    device = get_device()
    print(f"Using device: {device}")

    # ── Default model ─────────────────────────────────────────────────────
    print("\nGenerating default embeddings...")
    embeddings = run_embed(MODEL_PATH, EMBEDDINGS_PATH)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved {EMBEDDINGS_PATH}")

    # ── Ability model ─────────────────────────────────────────────────────
    if ABILITY_MODEL_PATH.exists():
        print("\nGenerating ability embeddings...")
        ability_embeddings = run_embed(
            ABILITY_MODEL_PATH,
            ABILITY_EMBEDDINGS_PATH,
            model_kwargs={
                "ci_emb_dim": ABILITY_CI_EMBEDDING_DIM,
                "keyword_emb_dim": ABILITY_KEYWORD_EMBEDDING_DIM,
                "mechanical_tag_dim": MECHANICAL_TAG_DIM,
                "mechanical_tag_emb_dim": ABILITY_MECHANICAL_TAG_EMBEDDING_DIM,
            },
            use_mechanical_tags=True,
        )
        print(f"  Ability embeddings shape: {ability_embeddings.shape}")
        print(f"  Saved {ABILITY_EMBEDDINGS_PATH}")
    else:
        print(f"\n  Skipping ability embeddings ({ABILITY_MODEL_PATH} not found)")

    # ── Build metadata CSV ────────────────────────────────────────────────
    print("\nBuilding card metadata...")
    df = pd.read_csv(OUTPUT_CSV_PATH)

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
