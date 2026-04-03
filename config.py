from pathlib import Path

DATA_DIR = Path("data")
RAW_JSON_PATH = DATA_DIR / "oracle-cards.json"
DOWNLOAD_META_PATH = DATA_DIR / ".download-meta.json"
OUTPUT_CSV_PATH = DATA_DIR / "cards.csv"

BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
BULK_DATA_TYPE = "oracle_cards"

EXCLUDED_LAYOUTS = {
    "token",
    "double_faced_token",
    "emblem",
    "planar",
    "scheme",
    "vanguard",
    "art_series",
}

MULTI_FACE_LAYOUTS = {
    "split",
    "flip",
    "transform",
    "modal_dfc",
    "adventure",
    "reversible_card",
}

SUPERTYPE_PRIORITY = [
    "Creature",
    "Planeswalker",
    "Battle",
    "Instant",
    "Sorcery",
    "Enchantment",
    "Artifact",
    "Land",
]

LEGALITY_FORMATS = [
    "standard",
    "modern",
    "legacy",
    "vintage",
    "commander",
    "pioneer",
    "pauper",
    "historic",
]

USER_AGENT = "mana-map/1.0"

# ── Embedding Pipeline Paths ──────────────────────────────────────────────
TEXT_EMBEDDINGS_PATH = DATA_DIR / "text_embeddings.npy"
CARD_FEATURES_PATH = DATA_DIR / "card_features.npz"
COLOR_VECTORS_PATH = DATA_DIR / "color_vectors.npy"
MODEL_PATH = DATA_DIR / "model.pt"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CARD_METADATA_PATH = DATA_DIR / "card_metadata.csv"
PROJECTION_PATH = DATA_DIR / "projection_2d.json"
VIZ_DIR = Path("viz")

# ── Text Encoder (frozen) ─────────────────────────────────────────────────
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384

# ── Categorical Vocab Sizes (each includes +1 unknown bucket) ─────────────
SUPERTYPE_VOCAB_SIZE = 10
RARITY_VOCAB_SIZE = 7
COLOR_IDENTITY_VOCAB_SIZE = 33
LAYOUT_VOCAB_SIZE = 17

# ── Categorical Embedding Dims ────────────────────────────────────────────
SUPERTYPE_EMBEDDING_DIM = 16
RARITY_EMBEDDING_DIM = 8
COLOR_IDENTITY_EMBEDDING_DIM = 32
LAYOUT_EMBEDDING_DIM = 16

# ── Feature Dims ──────────────────────────────────────────────────────────
CONTINUOUS_DIM = 2
KEYWORD_DIM = 50
FINAL_EMBEDDING_DIM = 128

# ── Fusion MLP ────────────────────────────────────────────────────────────
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.1

# ── Training Hyperparameters ──────────────────────────────────────────────
BATCH_SIZE = 256
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
TRIPLET_MARGIN = 0.3
VAL_SPLIT = 0.1
