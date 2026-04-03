"""Step 3: Pre-process cards.csv into feature arrays for the embedding model."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    CARD_FEATURES_PATH,
    COLOR_VECTORS_PATH,
    KEYWORD_DIM,
    OUTPUT_CSV_PATH,
    TEXT_EMBEDDING_DIM,
    TEXT_EMBEDDINGS_PATH,
    TEXT_MODEL_NAME,
)


def compute_text_embeddings(texts, model_name=TEXT_MODEL_NAME, batch_size=512):
    """Encode texts with a frozen sentence-transformer model.

    Returns (N, 384) float32 array.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def build_vocab_index(vocab_list):
    """Build {value: index} mapping from a list of known values.

    Index 0 is the first element; max+1 is reserved for unknowns.
    """
    return {v: i for i, v in enumerate(vocab_list)}


def encode_categorical(series, vocab_index):
    """Map a pandas Series to int64 indices using vocab_index.

    Unknown values (not in vocab_index) map to len(vocab_index).
    """
    unknown_idx = len(vocab_index)
    return np.array(
        [vocab_index.get(v, unknown_idx) for v in series], dtype=np.int64
    )


def build_color_identity_vocab(df):
    """Build a sorted vocab for color_identity strings.

    Colorless ("") sorts to index 0.
    """
    unique = sorted(df["color_identity"].fillna("").unique())
    # Ensure "" is at index 0
    if "" in unique:
        unique.remove("")
    unique = [""] + unique
    return unique


def build_top_keywords(df, top_n=KEYWORD_DIM):
    """Return the top_n most frequent keywords across all cards."""
    all_kw = []
    for kw_str in df["keywords"].fillna(""):
        if kw_str:
            all_kw.extend([k.strip() for k in kw_str.split(",")])
    from collections import Counter

    counts = Counter(all_kw)
    return [kw for kw, _ in counts.most_common(top_n)]


def encode_keywords_multihot(df, top_keywords):
    """Encode keywords as (N, len(top_keywords)) float32 multi-hot vectors."""
    kw_to_idx = {kw: i for i, kw in enumerate(top_keywords)}
    n = len(df)
    dim = len(top_keywords)
    result = np.zeros((n, dim), dtype=np.float32)
    for i, kw_str in enumerate(df["keywords"].fillna("")):
        if kw_str:
            for kw in kw_str.split(","):
                kw = kw.strip()
                if kw in kw_to_idx:
                    result[i, kw_to_idx[kw]] = 1.0
    return result


def normalize_cmc(series):
    """Normalize CMC: clip to [0, 16], divide by 16."""
    arr = series.fillna(0).values.astype(np.float32)
    arr = np.clip(arr, 0, 16)
    return arr / 16.0


def normalize_edhrec_rank(series):
    """Normalize EDHREC rank: fill NaN with median, log1p, scale to [0, 1]."""
    arr = series.copy()
    median_val = arr.median()
    arr = arr.fillna(median_val).values.astype(np.float32)
    arr = np.log1p(arr)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr.astype(np.float32)


def parse_color_vectors(df):
    """Parse color_identity into (N, 5) WUBRG binary vectors.

    This is visualization metadata, NOT fed into the model.
    """
    wubrg = list("WUBRG")
    n = len(df)
    result = np.zeros((n, 5), dtype=np.float32)
    for i, ci_str in enumerate(df["color_identity"].fillna("")):
        if ci_str:
            colors = [c.strip() for c in ci_str.split(",")]
            for c in colors:
                if c in wubrg:
                    result[i, wubrg.index(c)] = 1.0
    return result


def main():
    print("Loading cards.csv...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards loaded.")

    # ── Text embeddings ───────────────────────────────────────────────────
    print("\nComputing text embeddings...")
    texts = df["embedding_text"].fillna("").tolist()
    text_embs = compute_text_embeddings(texts)
    assert text_embs.shape == (len(df), TEXT_EMBEDDING_DIM), (
        f"Expected ({len(df)}, {TEXT_EMBEDDING_DIM}), got {text_embs.shape}"
    )
    np.save(TEXT_EMBEDDINGS_PATH, text_embs)
    print(f"  Saved {TEXT_EMBEDDINGS_PATH} — shape {text_embs.shape}")

    # ── Categorical features ──────────────────────────────────────────────
    print("\nEncoding categorical features...")

    # Supertype
    supertype_vocab = sorted(df["supertype"].dropna().unique())
    supertype_index = build_vocab_index(supertype_vocab)
    supertype_encoded = encode_categorical(df["supertype"].fillna("Unknown"), supertype_index)
    print(f"  supertype vocab: {len(supertype_vocab)} + 1 unknown")

    # Rarity
    rarity_vocab = sorted(df["rarity"].dropna().unique())
    rarity_index = build_vocab_index(rarity_vocab)
    rarity_encoded = encode_categorical(df["rarity"].fillna(""), rarity_index)
    print(f"  rarity vocab: {len(rarity_vocab)} + 1 unknown")

    # Color identity (as string, e.g. "W, U" or "")
    ci_vocab = build_color_identity_vocab(df)
    ci_index = build_vocab_index(ci_vocab)
    ci_encoded = encode_categorical(df["color_identity"].fillna(""), ci_index)
    print(f"  color_identity vocab: {len(ci_vocab)} + 1 unknown")

    # Layout
    layout_vocab = sorted(df["layout"].dropna().unique())
    layout_index = build_vocab_index(layout_vocab)
    layout_encoded = encode_categorical(df["layout"].fillna(""), layout_index)
    print(f"  layout vocab: {len(layout_vocab)} + 1 unknown")

    # ── Continuous features ───────────────────────────────────────────────
    print("\nNormalizing continuous features...")
    cmc_norm = normalize_cmc(df["cmc"])
    edhrec_norm = normalize_edhrec_rank(df["edhrec_rank"])
    continuous = np.stack([cmc_norm, edhrec_norm], axis=1)
    print(f"  continuous shape: {continuous.shape}")

    # ── Keywords ──────────────────────────────────────────────────────────
    print("\nEncoding keywords...")
    top_keywords = build_top_keywords(df)
    keywords_multihot = encode_keywords_multihot(df, top_keywords)
    print(f"  top {len(top_keywords)} keywords, multi-hot shape: {keywords_multihot.shape}")

    # ── Save all features ─────────────────────────────────────────────────
    np.savez(
        CARD_FEATURES_PATH,
        supertype=supertype_encoded,
        rarity=rarity_encoded,
        color_identity=ci_encoded,
        layout=layout_encoded,
        continuous=continuous,
        keywords=keywords_multihot,
        supertype_vocab=np.array(supertype_vocab),
        rarity_vocab=np.array(rarity_vocab),
        ci_vocab=np.array(ci_vocab),
        layout_vocab=np.array(layout_vocab),
        top_keywords=np.array(top_keywords),
    )
    print(f"  Saved {CARD_FEATURES_PATH}")

    # ── Color vectors (metadata) ──────────────────────────────────────────
    print("\nParsing WUBRG color vectors...")
    color_vecs = parse_color_vectors(df)
    np.save(COLOR_VECTORS_PATH, color_vecs)
    print(f"  Saved {COLOR_VECTORS_PATH} — shape {color_vecs.shape}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
