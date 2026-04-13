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
    "Planeswalker",
    "Battle",
    "Land",           # Before Creature so "Land Creature" -> Land
    "Creature",
    "Instant",
    "Sorcery",
    "Enchantment",
    "Artifact",
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

# ── Combo / Deck Builder Data ────────────────────────────────────────────
COMBOS_API_URL = "https://backend.commanderspellbook.com/variants/"
COMBOS_RAW_PATH = DATA_DIR / "combos_raw.json"
COMBOS_META_PATH = DATA_DIR / ".combos-meta.json"
COMBO_GRAPH_PATH = DATA_DIR / "combo_graph.json"
EMBEDDINGS_BIN_PATH = DATA_DIR / "embeddings.bin"

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
EARLY_STOPPING_PATIENCE = 5
ABILITY_NUM_EPOCHS = 100

# ── Mechanical Tags ─────────────────────────────────────────────────────
MECHANICAL_TAGS_PATH = DATA_DIR / "mechanical_tags.npy"

# Tag name → regex pattern (applied to oracle text, case-insensitive)
# Each pattern is compiled with re.IGNORECASE
MECHANICAL_TAGS = {
    # Triggers
    "etb": r"enters the battlefield|enters under your control|when .* enters",
    "death_trigger": r"when .* dies|whenever .* dies|whenever .* is put into a graveyard from the battlefield",
    "attack_trigger": r"whenever .* attacks",
    "damage_trigger": r"whenever .* deals (?:combat )?damage",
    "upkeep_trigger": r"at the beginning of (?:your |each )?upkeep",
    # Effects
    "sacrifice": r"sacrifice (?:a |an |another )",
    "draw": r"draw (?:a |two |three |\d+ )?cards?",
    "removal": r"destroy (?:target |all |each )|(?:target |each ).*gets? [+-]\d+/-\d+|exile (?:target |all |each )",
    "bounce": r"return (?:target |a ).*to (?:its |their )?owner'?s hand",
    "counterspell": r"counter target spell",
    "blink": r"exile .*(?:then |, )return (?:it|that card|them) to the battlefield|flicker",
    "reanimate": r"return .*from (?:a |your )?graveyard to the battlefield|put .*from (?:a |your )?graveyard onto the battlefield",
    "tutor": r"search your library for",
    "discard": r"discard (?:a |two |\d+ )?cards?",
    # Generators
    "tokens": r"create (?:a |an |two |three |four |five |ten |\d+ )?(?:\d+/\d+ )?\w+ \w+ (?:creature |artifact |enchantment )?tokens?|create (?:a |an |two |three |\d+ )?tokens?",
    "counters_plus": r"put (?:a |two |three |\d+ )?\+1/\+1 counters? on",
    "counters_minus": r"put (?:a |two |three |\d+ )?-1/-1 counters? on",
    "ramp": r"search your library for .*(?:land|forest|plains|island|swamp|mountain).*(?:put|onto)|add \{[WUBRGC]\}|adds? (?:\w+ )?mana",
    "lifegain": r"you gain (?:\d+ )?life|lifelink",
    "mill": r"mills? (?:\w+ )?cards?|put the top .* cards? of .* library into .* graveyard",
    # Modifiers
    "anthem": r"(?:other )?creatures you control get \+|creatures you control have",
    "cost_reduction": r"(?:spells?|abilities) .*costs? \{?\d\}? less|reduce the cost",
    "copy": r"copy (?:target |that |a |it|the )?(?:spell|instant|sorcery|creature|permanent|artifact|enchantment)|copies of",
    "protection": r"hexproof|shroud|indestructible|protection from",
    "evasion_flying": r"flying",
    "evasion_trample": r"trample",
    "evasion_menace": r"menace",
    "evasion_unblockable": r"unblockable|can't be blocked",
    # Permanents
    "equipment": r"equip \{|equip—|equipped creature",
    "aura": r"enchant (?:creature|permanent|player|land|artifact)",
    "tap_ability": r"\{T\}:",
    # Graveyard
    "graveyard_matters": r"(?:cards? in|from) (?:your |a )?graveyard|flashback|dredge|unearth|escape",
    # Storm
    "storm": r"\bstorm\b",
}

MECHANICAL_TAG_NAMES = sorted(MECHANICAL_TAGS.keys())
MECHANICAL_TAG_DIM = len(MECHANICAL_TAGS)

# ── Ability Model ────────────────────────────────────────────────────────
ABILITY_MODEL_PATH = DATA_DIR / "model_ability.pt"
ABILITY_EMBEDDINGS_PATH = DATA_DIR / "embeddings_ability.npy"
ABILITY_EMBEDDINGS_BIN_PATH = DATA_DIR / "embeddings_ability.bin"
ABILITY_PROJECTION_PATH = DATA_DIR / "projection_2d_ability.json"

ABILITY_CI_EMBEDDING_DIM = 8
ABILITY_KEYWORD_EMBEDDING_DIM = 32
ABILITY_MECHANICAL_TAG_EMBEDDING_DIM = 32
MIN_SHARED_TAGS_POSITIVE = 2

# ── Synergy Rules ────────────────────────────────────────────────────────
SYNERGY_GRAPH_PATH = DATA_DIR / "synergy_graph.json"
SYNERGY_MAX_PARTNERS = 10

# Each rule: (tag_A, tag_B, label) — card with tag_A synergizes with card having tag_B.
# Rules are applied bidirectionally. Do NOT add reverse duplicates.
SYNERGY_RULES = [
    ("blink", "etb", "Blink + ETB"),
    ("sacrifice", "death_trigger", "Sac + Death Trigger"),
    ("tokens", "anthem", "Tokens + Anthem"),
    ("reanimate", "death_trigger", "Reanimate + Death Trigger"),
    ("tokens", "sacrifice", "Tokens + Sacrifice"),
    ("draw", "discard", "Draw + Discard"),
    ("mill", "graveyard_matters", "Mill + Graveyard"),
    ("counters_plus", "tokens", "Counters + Tokens"),
    ("copy", "etb", "Copy + ETB"),
    ("storm", "cost_reduction", "Storm + Cost Reduction"),
    ("sacrifice", "etb", "Sac + ETB"),
    ("reanimate", "etb", "Reanimate + ETB"),
    ("lifegain", "death_trigger", "Lifegain + Death Trigger"),
    ("bounce", "etb", "Bounce + ETB"),
    ("removal", "death_trigger", "Removal + Death Trigger"),
    ("counters_minus", "death_trigger", "-1/-1 + Death Trigger"),
    ("evasion_flying", "damage_trigger", "Flying + Damage Trigger"),
    ("evasion_unblockable", "damage_trigger", "Unblockable + Damage Trigger"),
    ("evasion_trample", "damage_trigger", "Trample + Damage Trigger"),
    ("attack_trigger", "tokens", "Attack Trigger + Tokens"),
    ("equipment", "attack_trigger", "Equipment + Attack Trigger"),
    ("aura", "protection", "Aura + Protection"),
    ("ramp", "cost_reduction", "Ramp + Cost Reduction"),
    ("counterspell", "draw", "Counterspell + Draw"),
]

# ── Power Creep / Obsolescence ───────────────────────────────────────────
OBSOLESCENCE_INDEX_PATH = DATA_DIR / "obsolescence_index.json"
OBSOLESCENCE_SIMILARITY_THRESHOLD = 0.75
OBSOLESCENCE_SINGLE_TAG_THRESHOLD = 0.98
OBSOLESCENCE_MIN_TAGS = 1
OBSOLESCENCE_MAX_REPLACEMENTS = 5
