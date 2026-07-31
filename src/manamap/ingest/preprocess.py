"""Step 3: Pre-process cards.csv into feature arrays.

**The row-index invariant starts here.** Every array this step writes is
positional: row i of `text_embeddings.npy`, of each array in
`card_features.npz`, of `mechanical_tags.npy` and `color_vectors.npy` is row i
of `cards.csv`. Everything downstream inherits it — embeddings, the 2D
projection, `embeddings.bin`, the region files. Nothing keys by card name
(names duplicate across printings), so a regenerated `cards.csv` with a
different row count or order silently pairs each card's coordinates with
another card's text. Never regenerate one stage without the ones after it.

Second coupling worth knowing: the categorical vocabularies here are derived
from the *current* CSV, but `training/model.py` sizes its embedding tables from
the fixed `*_VOCAB_SIZE` constants in config.py. If Scryfall introduces a new
layout or rarity, those constants must grow too or training raises an opaque
index error far from this file.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from manamap.config import (
    CARD_FEATURES_PATH,
    COLOR_IDENTITY_VOCAB_SIZE,
    COLOR_VECTORS_PATH,
    EDHREC_RANK_SCALE,
    KEYWORD_DIM,
    LAYOUT_VOCAB_SIZE,
    MECHANICAL_TAG_NAMES,
    MECHANICAL_TAGS_PATH,
    OUTPUT_CSV_PATH,
    PIP_COUNT_SCALE,
    POWER_TOUGHNESS_SCALE,
    RARITY_VOCAB_SIZE,
    SUPERTYPE_VOCAB_SIZE,
    TEXT_EMBEDDING_DIM,
    TEXT_EMBEDDINGS_PATH,
    TEXT_MODEL_NAME,
)
from manamap.mechanical_tags import encode_tags_multihot


_MODEL_CACHE = {}


def compute_text_embeddings(texts, model_name=TEXT_MODEL_NAME, batch_size=512):
    """Encode texts with a frozen sentence-transformer model.

    Returns a (N, D) float32 array, where D follows the model — 384 for the
    default all-MiniLM-L6-v2, which both RAG indexes assume.

    Output is **not** L2-normalized. Callers that need unit vectors
    (build_rules_db, build_strategy_db) normalize at build time so queries can
    use a bare matmul for cosine; `common.load_rules_db` documents that
    guarantee on the read side.

    The model is cached per name: loading MiniLM costs ~2 s, and the RAG query
    paths (query-rules, query-strategy) call this once per query. The weights
    are frozen, so a cached instance cannot change any output.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _MODEL_CACHE[model_name]
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
    """Return the top_n most frequent keywords across all cards.

    The `if k.strip()` matters more than it looks. 17,117 of 34,322 cards have no
    keywords at all, and `"".split(",")` yields `[""]` — so without the guard the
    empty string is the single most common "keyword" by a factor of five, takes
    index 0 of the vocabulary, and `encode_keywords_multihot` then never sets that
    column because it skips falsy strings. The result is a permanently dead feature
    slot and only 49 real keywords out of a nominal 50.
    """
    all_kw = []
    for kw_str in df["keywords"].fillna(""):
        all_kw.extend([k.strip() for k in kw_str.split(",") if k.strip()])
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


def normalize_edhrec_rank(series, max_rank=EDHREC_RANK_SCALE):
    """Normalize EDHREC rank: fill NaN with median, log1p, scale by a FIXED bound.

    The scale is a constant, not the observed min/max, because this used to be a
    per-run min-max: the same card got a different feature value in every pipeline
    run as the corpus grew, so two runs' embeddings were not comparable and neither
    were their input features. Ranks currently top out near 31,800;
    `EDHREC_RANK_SCALE` leaves headroom so a few new sets do not silently start
    clipping.
    """
    arr = series.copy()
    arr = arr.fillna(arr.median()).values.astype(np.float32)
    return np.clip(np.log1p(arr) / np.log1p(max_rank), 0.0, 1.0).astype(np.float32)


def normalize_power_toughness(df):
    """(N, 3) of [power, toughness, has_stats].

    Power and toughness reach either model for the first time here — they were in
    `cards.csv` and used by power_creep's stat comparison, but never fed to
    training, so nothing distinguished a 1/1 from a 7/7 with the same text.

    250 cards carry non-numeric stats (`*`, `1+*`). Those get 0.0 with `has_stats`
    still set: "a creature whose power is variable" is a real and different thing
    from "not a creature", and collapsing them would teach the model the latter.
    """
    power = pd.to_numeric(df["power"], errors="coerce")
    toughness = pd.to_numeric(df["toughness"], errors="coerce")
    has_stats = (df["power"].notna() | df["toughness"].notna()).values.astype(np.float32)
    scale = float(POWER_TOUGHNESS_SCALE)
    return np.stack([
        np.clip(power.fillna(0).values.astype(np.float32), 0, scale) / scale,
        np.clip(toughness.fillna(0).values.astype(np.float32), 0, scale) / scale,
        has_stats,
    ], axis=1)


def parse_mana_pips(df):
    """(N, 6) of [W, U, B, R, G pip counts, generic cost], each scaled.

    Only scalar CMC reached the model before, so `{U}{U}` and `{2}` were identical
    inputs — which erases the difference between a double-blue spell and a colourless
    one of the same cost, and that difference is most of what makes a card castable.
    Hybrid and Phyrexian symbols count toward every colour they can be paid with,
    which is the honest reading for a castability feature.
    """
    wubrg = "WUBRG"
    symbols = re.compile(r"\{([^}]+)\}")
    result = np.zeros((len(df), 6), dtype=np.float32)
    for i, cost in enumerate(df["mana_cost"].fillna("")):
        for symbol in symbols.findall(str(cost)):
            if symbol.isdigit():
                result[i, 5] += float(symbol)
                continue
            for colour in symbol.split("/"):
                if colour in wubrg:
                    result[i, wubrg.index(colour)] += 1.0
    scale = float(PIP_COUNT_SCALE)
    return np.clip(result, 0, scale) / scale


def build_color_identity_features(df):
    """(N, 6) of [W, U, B, R, G, colour count].

    Replaces the string categorical for model purposes. `"B, G"` and `"B, G, R"`
    were unrelated one-hot indices, so nothing could express that Golgari is a
    subset of Jund — and that vocabulary sat at exactly its table capacity (32 + 1
    of 33), one new Scryfall value away from silently overflowing into the unknown
    bucket. Multi-hot has neither problem and cannot outgrow itself.
    """
    wubrg = parse_color_vectors(df)
    count = wubrg.sum(axis=1, keepdims=True) / 5.0
    return np.concatenate([wubrg, count], axis=1).astype(np.float32)


def assert_vocab_fits(name, vocab, capacity):
    """Fail loudly here rather than opaquely inside an embedding table.

    `preprocess` derives vocabularies from the current CSV while `model.py` sizes
    its tables from fixed constants. When Scryfall adds a layout or a rarity, the
    mismatch used to surface as an index error deep in training, hours later and
    far from the cause.
    """
    if len(vocab) + 1 > capacity:
        raise ValueError(
            f"{name} vocabulary has {len(vocab)} values (+1 unknown) but "
            f"config caps it at {capacity}. Raise the constant in config.py and "
            f"retrain — the embedding table shape is baked into the checkpoint."
        )


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
    assert_vocab_fits("supertype", supertype_vocab, SUPERTYPE_VOCAB_SIZE)
    supertype_index = build_vocab_index(supertype_vocab)
    supertype_encoded = encode_categorical(df["supertype"].fillna("Unknown"), supertype_index)
    print(f"  supertype vocab: {len(supertype_vocab)} + 1 unknown")

    # Rarity
    rarity_vocab = sorted(df["rarity"].dropna().unique())
    assert_vocab_fits("rarity", rarity_vocab, RARITY_VOCAB_SIZE)
    rarity_index = build_vocab_index(rarity_vocab)
    rarity_encoded = encode_categorical(df["rarity"].fillna(""), rarity_index)
    print(f"  rarity vocab: {len(rarity_vocab)} + 1 unknown")

    # Color identity (as string, e.g. "W, U" or "")
    ci_vocab = build_color_identity_vocab(df)
    assert_vocab_fits("color_identity", ci_vocab, COLOR_IDENTITY_VOCAB_SIZE)
    ci_index = build_vocab_index(ci_vocab)
    ci_encoded = encode_categorical(df["color_identity"].fillna(""), ci_index)
    print(f"  color_identity vocab: {len(ci_vocab)} + 1 unknown")

    # Layout
    layout_vocab = sorted(df["layout"].dropna().unique())
    assert_vocab_fits("layout", layout_vocab, LAYOUT_VOCAB_SIZE)
    layout_index = build_vocab_index(layout_vocab)
    layout_encoded = encode_categorical(df["layout"].fillna(""), layout_index)
    print(f"  layout vocab: {len(layout_vocab)} + 1 unknown")

    # ── Continuous features ───────────────────────────────────────────────
    print("\nNormalizing continuous features...")
    cmc_norm = normalize_cmc(df["cmc"])
    edhrec_norm = normalize_edhrec_rank(df["edhrec_rank"])
    continuous = np.stack([cmc_norm, edhrec_norm], axis=1)
    print(f"  continuous shape: {continuous.shape}")

    # Structured numeric features. Saved now, consumed when the model's input
    # tuple is rebuilt — see config.py's Feature Dims block for why they are
    # staged rather than wired in here.
    power_toughness = normalize_power_toughness(df)
    mana_pips = parse_mana_pips(df)
    color_features = build_color_identity_features(df)
    print(f"  power/toughness: {power_toughness.shape} "
          f"({int(power_toughness[:, 2].sum()):,} cards with stats)")
    print(f"  mana pips: {mana_pips.shape} · colour features: {color_features.shape}")

    # ── Keywords ──────────────────────────────────────────────────────────
    print("\nEncoding keywords...")
    top_keywords = build_top_keywords(df)
    keywords_multihot = encode_keywords_multihot(df, top_keywords)
    print(f"  top {len(top_keywords)} keywords, multi-hot shape: {keywords_multihot.shape}")

    # (features saved below after mechanical tags are computed)

    # ── Mechanical tags ────────────────────────────────────────────────────
    print("\nEncoding mechanical tags...")
    mech_tags = encode_tags_multihot(df, MECHANICAL_TAG_NAMES)
    np.save(MECHANICAL_TAGS_PATH, mech_tags)
    print(f"  Saved {MECHANICAL_TAGS_PATH} — shape {mech_tags.shape}")

    # Also add to card_features.npz for the ability model
    np.savez(
        CARD_FEATURES_PATH,
        supertype=supertype_encoded,
        rarity=rarity_encoded,
        color_identity=ci_encoded,
        layout=layout_encoded,
        continuous=continuous,
        power_toughness=power_toughness,
        mana_pips=mana_pips,
        color_features=color_features,
        keywords=keywords_multihot,
        mechanical_tags=mech_tags,
        supertype_vocab=np.array(supertype_vocab),
        rarity_vocab=np.array(rarity_vocab),
        ci_vocab=np.array(ci_vocab),
        layout_vocab=np.array(layout_vocab),
        top_keywords=np.array(top_keywords),
        mechanical_tag_names=np.array(MECHANICAL_TAG_NAMES),
    )
    print(f"  Updated {CARD_FEATURES_PATH} with mechanical_tags")

    # Tag coverage stats
    tagged_count = (mech_tags.sum(axis=1) > 0).sum()
    non_land_count = (df["supertype"] != "Land").sum()
    non_land_tagged = ((mech_tags.sum(axis=1) > 0) & (df["supertype"] != "Land").values).sum()
    print(f"  Tagged cards: {tagged_count:,}/{len(df):,} ({tagged_count/len(df)*100:.1f}%)")
    print(f"  Non-land tagged: {non_land_tagged:,}/{non_land_count:,} ({non_land_tagged/non_land_count*100:.1f}%)")

    # ── Color vectors (metadata) ──────────────────────────────────────────
    print("\nParsing WUBRG color vectors...")
    color_vecs = parse_color_vectors(df)
    np.save(COLOR_VECTORS_PATH, color_vecs)
    print(f"  Saved {COLOR_VECTORS_PATH} — shape {color_vecs.shape}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
