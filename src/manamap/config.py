import os
from pathlib import Path

# Anchor all paths to the repo root (src/manamap/config.py → two levels up)
# so modules work regardless of CWD. Overridable for sandboxed runs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MANAMAP_DATA_DIR", _REPO_ROOT / "data"))
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
VIZ_DIR = Path(os.environ.get("MANAMAP_VIZ_DIR", _REPO_ROOT / "viz"))

# ── Combo / Deck Builder Data ────────────────────────────────────────────
COMBOS_API_URL = "https://backend.commanderspellbook.com/variants/"
COMBOS_RAW_PATH = DATA_DIR / "combos_raw.json"
COMBOS_META_PATH = DATA_DIR / ".combos-meta.json"
COMBO_GRAPH_PATH = DATA_DIR / "combo_graph.json"
COMBO_DETAILS_PATH = DATA_DIR / "combo_details.json"
EMBEDDINGS_BIN_PATH = DATA_DIR / "embeddings.bin"

# Commander Spellbook classifies every combo variant by power, stored as a
# single letter (Variant.BracketTag in their backend). The letters map onto
# WotC's bracket ladder, which is what makes a computed bracket floor possible:
# a deck's floor is the highest-bracket combo it fully contains.
# "B" (Banned) is deliberately absent — those variants use Commander-banned
# cards and carry their own flag rather than a bracket.
COMBO_BRACKET_TAGS = {
    "E": 1,   # Exhibition
    "C": 2,   # Core
    "O": 2,   # Oddball
    "P": 3,   # Powerful
    "S": 3,   # Spicy
    "R": 4,   # Ruthless
}
COMBO_BANNED_TAG = "B"

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

# ── Deckbuilding Roles ───────────────────────────────────────────────────
# MECHANICAL_TAGS is a retrieval vocabulary — it answers "what is this card
# like". Deckbuilding needs a different question: "what job does this card do
# in a 99". The two overlap but are not interchangeable: `ramp` there is one
# regex covering rocks, dorks, land ramp and rituals, which is fatal for curve
# modeling. This dict is deliberately SEPARATE — editing MECHANICAL_TAGS
# invalidates model_ability.pt and forces a retrain, and roles change often.
CARD_ROLES_PATH = DATA_DIR / "card_roles.json"

# Roles that need the type line to disambiguate are resolved in card_roles.py;
# these patterns run against oracle text + keywords.
ROLE_PATTERNS = {
    # Card advantage
    "draw:engine": r"(?:whenever|at the beginning of).{0,80}?draws? (?:a |two |three |\d+ )?cards?",
    "draw:burst": r"draw (?:two|three|four|five|six|seven|\d+) cards",
    "draw:impulse": r"exile the top (?:\w+ )?cards?.{0,60}?(?:you may (?:play|cast)|until)",
    "draw:wheel": r"discards? (?:their|your) hand.{0,60}?draws?|each player draws",
    # Interaction
    "removal:sweeper": r"destroy all|exile all|destroy each|all creatures get -|each creature gets -",
    "removal:spot": r"destroy target|exile target (?:creature|permanent|artifact|enchantment|planeswalker|battle)",
    "removal:damage": r"deals? (?:\d+|x) damage to (?:target|any target|each)",
    "removal:edict": r"sacrifices? a (?:creature|permanent)",
    "removal:tax": r"(?:spells?|abilities).{0,40}?costs? \{?\d+\}? more",
    "removal:fight": r"\bfights? (?:target|another)",
    "counterspell": r"counter target (?:spell|ability)",
    # Consistency
    "tutor:unrestricted": r"search your library for a card",
    "tutor:narrow": r"search your library for (?:a|an|up to \w+) (?!card)[\w\s]{0,30}?cards?",
    "recursion": r"return (?:target |a |another )?[\w\s]{0,30}?(?:card )?from (?:your|a) graveyard to (?:the battlefield|your hand)",
    # Resilience
    "protection:self": r"\b(?:hexproof|shroud|indestructible|ward)\b",
    "protection:granted": r"(?:target |another target )?creatures? you control (?:gains?|have|has) (?:hexproof|indestructible|protection|shroud)",
    # Engines
    "sac-outlet": r"sacrifice (?:a|an|another) (?:creature|permanent|artifact|token)[^.]{0,20}?:",
    "stax": r"players? can't|can't be (?:activated|cast)|skip (?:your|their) |don't untap|enters? tapped and",
    # Finishers
    "wincon:alt": r"wins? the game|loses? the game",
    "wincon:drain": r"each opponent loses (?:\d+|x) life|each opponent (?:loses|sacrifices)",
    "wincon:combat": r"\b(?:infect|double strike)\b|deals? double|can't be blocked",
    # Broad jobs. Most cards in a deck are not a tutor or a sweeper — they are
    # a body, an enters-trigger, or an activated ability, and a builder counts
    # those slots too.
    "value:etb": r"when(?:ever)? (?:this|[\w\s,']{0,30}?) enters",
    "utility:activated": r"\{T\}[,:]|\{\d+\}[,:].{0,40}?:",
    "removal:debuff": r"gets? -\d+/-\d+|gets? -\d+/-0|put a -1/-1 counter",
    # Voltron and combat-trick slots. An Aura that pumps and an Equipment that
    # pumps are the same job wearing different card types.
    "buff:attached": r"(?:enchanted|equipped) creature (?:gets?|has|have)|\bequip[\s—]|\benchant creature\b",
    "buff:pump": r"gets? \+\d+/\+\d+ until end of turn|creatures you control get \+",
    "buff:counters": r"put (?:a|two|three|\d+|x) \+1/\+1 counters?",
    "hate:graveyard": r"exile[\w\s]{0,30}?from (?:a|target opponent's|each) (?:player's )?graveyard|graveyards? instead",
}

# A creature with no other listed job is still doing one: attacking and
# blocking. `threat:body` is that fallback, applied by type line. It is
# reported separately from the specific roles so it can never flatter the
# coverage number — see ROLE_COVERAGE_TARGET.
ROLE_BODY_FALLBACK = "threat:body"

# Mana production is one regex plus the type line: an artifact that taps for
# mana is a rock, a creature is a dork, an instant is a ritual. v1's single
# `ramp` tag scored a Signet and a Dark Ritual identically.
ROLE_MANA_SOURCE = r"add \{|adds? (?:one|two|three|\d+) mana|add one mana"
ROLE_MANA_BY_SUPERTYPE = {
    "Artifact": "ramp:rock",
    "Creature": "ramp:dork",
    "Enchantment": "ramp:rock",
    "Instant": "ramp:ritual",
    "Sorcery": "ramp:ritual",
}
ROLE_LAND_RAMP = r"search your library for (?:a|up to \w+)[\w\s]{0,30}?land"
ROLE_COST_REDUCTION = r"costs? \{?\d+\}? less|spells? you cast cost"

# Land quality — lands never carry spell roles, so these are evaluated alone.
ROLE_LAND_PATTERNS = {
    # Fetchlands name basic land *types*, not "land card" — Windswept Heath
    # says "a Forest or Plains card" and would otherwise read as plain utility.
    "land:fetch": r"search your library for a[\w\s]{0,40}?(?:land card|Plains|Island|Swamp|Mountain|Forest)",
    "land:tapped": r"enters tapped|enters the battlefield tapped",
    "land:utility": r"\{T\},|\{T\}: (?!add)|draw a card|deals? \d+ damage",
}

ROLE_NAMES = sorted(
    set(ROLE_PATTERNS) | set(ROLE_LAND_PATTERNS) | set(ROLE_MANA_BY_SUPERTYPE.values())
    | {"ramp:land", "ramp:cost-reduction", "land:untapped-dual", "land:mdfc",
       "land:basic", ROLE_BODY_FALLBACK}
)

# Coverage floors, asserted by the test suite. `COVERAGE` counts any role at
# all; `SPECIFIC` excludes cards carrying only the body fallback, because a
# classifier that labels every creature "threat:body" and stops has told the
# slot filler nothing.
#
# These are regression floors set just under what the classifier measured
# (86.3% / 67.6%), not aspirations. 100% is explicitly not the goal: the ~4.3K
# cards that stay unclassified are genuinely miscellaneous one-off effects
# (Aura Graft, Saheeli's Artistry, Theft of Dreams), and forcing them into a
# bucket with looser regexes would buy coverage with false positives — which
# is strictly worse for a slot filler than an honest null.
ROLE_COVERAGE_TARGET = 0.85
ROLE_SPECIFIC_COVERAGE_TARGET = 0.65

# ── Power Creep / Obsolescence ───────────────────────────────────────────
OBSOLESCENCE_INDEX_PATH = DATA_DIR / "obsolescence_index.json"
OBSOLESCENCE_SIMILARITY_THRESHOLD = 0.75
OBSOLESCENCE_SINGLE_TAG_THRESHOLD = 0.98
OBSOLESCENCE_MIN_TAGS = 1
OBSOLESCENCE_MAX_REPLACEMENTS = 5

# ── Region Clustering ──────────────────────────────────────────────────
REGIONS_DEFAULT_PATH = DATA_DIR / "regions_default.json"
REGIONS_ABILITY_PATH = DATA_DIR / "regions_ability.json"

REGION_L0_MIN_CLUSTER_SIZE = 800
REGION_L0_MIN_SAMPLES = 50
REGION_L1_MIN_CLUSTER_SIZE = 100
REGION_L1_MIN_SAMPLES = 15

REGION_COLOR_DOMINANCE = 0.40
REGION_TYPE_DOMINANCE = 0.30
REGION_MIN_TAG_PRESENCE = 0.10  # Tag must be in >=10% of cluster to be used in naming

REGION_TAG_DISPLAY_NAMES = {
    "etb": "ETB",
    "death_trigger": "Death Triggers",
    "tokens": "Token Makers",
    "sacrifice": "Sacrifice",
    "draw": "Card Draw",
    "removal": "Removal",
    "blink": "Blink",
    "reanimate": "Reanimation",
    "bounce": "Bounce",
    "counterspell": "Counterspells",
    "tutor": "Tutors",
    "discard": "Discard",
    "ramp": "Mana Ramp",
    "lifegain": "Lifegain",
    "mill": "Mill",
    "anthem": "Anthems",
    "cost_reduction": "Cost Reduction",
    "copy": "Copy Effects",
    "protection": "Protection",
    "equipment": "Equipment",
    "aura": "Auras",
    "tap_ability": "Tap Abilities",
    "graveyard_matters": "Graveyard",
    "storm": "Storm",
    "counters_plus": "+1/+1 Counters",
    "counters_minus": "-1/-1 Counters",
    "attack_trigger": "Attack Triggers",
    "damage_trigger": "Damage Triggers",
    "upkeep_trigger": "Upkeep",
    "evasion_flying": "Flyers",
    "evasion_trample": "Tramplers",
    "evasion_menace": "Menace",
    "evasion_unblockable": "Unblockable",
}

REGION_COLOR_DISPLAY_NAMES = {
    "W": "White",
    "U": "Blue",
    "B": "Black",
    "R": "Red",
    "G": "Green",
    "Colorless": "Colorless",
    "Multicolor": "Gold",
}

REGION_GUILD_NAMES = {
    frozenset(["W", "U"]): "Azorius",
    frozenset(["U", "B"]): "Dimir",
    frozenset(["B", "R"]): "Rakdos",
    frozenset(["R", "G"]): "Gruul",
    frozenset(["G", "W"]): "Selesnya",
    frozenset(["W", "B"]): "Orzhov",
    frozenset(["U", "R"]): "Izzet",
    frozenset(["B", "G"]): "Golgari",
    frozenset(["R", "W"]): "Boros",
    frozenset(["G", "U"]): "Simic",
}

# ── Pilot: Comprehensive Rules ───────────────────────────────────────────
RULES_DIR = DATA_DIR / "rules"
# Update per set release — current link lives at https://magic.wizards.com/en/rules
CR_RULES_URL = "https://media.wizards.com/2026/downloads/MagicCompRules%2020260619.txt"
CR_RULES_PATH = RULES_DIR / "comprehensive_rules.txt"
CR_RULES_META_PATH = RULES_DIR / ".rules-meta.json"
RULES_INDEX_PATH = RULES_DIR / "rules_index.json"
RULES_EMBEDDINGS_PATH = RULES_DIR / "rules_embeddings.npy"
RULES_DB_META_PATH = RULES_DIR / ".rules-db-meta.json"
RULES_QUERY_TOP_K = 8

# ── Pilot: Decks & Manuals ───────────────────────────────────────────────
DECKS_DIR = DATA_DIR / "decks"
MANUALS_DIR = _REPO_ROOT / "manuals"
SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection"
SCRYFALL_BATCH_SIZE = 75
SCRYFALL_REQUEST_DELAY_S = 0.1
RESOLVE_MAX_ITERATIONS = 3

# ── Pilot: Commander Brackets ────────────────────────────────────────────
# WotC's bracket ladder. The restrictions here are the ones we can check
# mechanically; the system is explicitly NOT a calculator, so bracket.py
# reports what a deck *contains* and the floor that implies. It never tells a
# player what bracket their deck is.
#
# `game_changers`/`two_card_infinites` are counts (None = unlimited).
# `tutor_budget` is advisory only — WotC says "few tutors", never a number, so
# it is reported and never raises a floor. Inventing a threshold and refusing
# on it would be exactly the kind of unverifiable claim this repo avoids.
BRACKETS = {
    1: {"name": "Exhibition", "game_changers": 0, "two_card_infinites": 0,
        "mass_land_denial": False, "tutor_budget": 1, "expected_turns": "9+"},
    2: {"name": "Core", "game_changers": 0, "two_card_infinites": 0,
        "mass_land_denial": False, "tutor_budget": 3, "expected_turns": "8+"},
    3: {"name": "Upgraded", "game_changers": 3, "two_card_infinites": 0,
        "mass_land_denial": False, "tutor_budget": 6, "expected_turns": "6+"},
    4: {"name": "Optimized", "game_changers": None, "two_card_infinites": None,
        "mass_land_denial": True, "tutor_budget": None, "expected_turns": "4+"},
    5: {"name": "cEDH", "game_changers": None, "two_card_infinites": None,
        "mass_land_denial": True, "tutor_budget": None, "expected_turns": "any"},
}
BRACKET_DEFAULT = 3
BRACKET_MAX = 5

# A two-card infinite assembling at or below this mana value is "early game",
# which is what separates a bracket-3 deck from a bracket-4 one.
BRACKET_EARLY_COMBO_MANA = 6

# Curated and deliberately conservative — this is the one bracket signal that
# isn't derived from data, so it is incomplete by construction and bracket.py
# says so in its report rather than implying the check is exhaustive.
MASS_LAND_DENIAL = frozenset({
    "Armageddon", "Ravages of War", "Catastrophe", "Decree of Annihilation",
    "Jokulhaups", "Obliterate", "Wildfire", "Burning of Xinye", "Sunder",
    "Cataclysm", "Global Ruin", "Ruination", "Mana Vortex", "Fall of the Thran",
    "Death Cloud", "Realm Razer", "Worldfire", "Devastation", "Epicenter",
    "Keldon Firebombers", "Impending Disaster", "Boom // Bust", "Tectonic Break",
    "Winter Moon", "Bend or Break", "Numot, the Devastator",
})

# ── Pilot: Goldfish Simulation ───────────────────────────────────────────
GOLDFISH_SEED = 42
GOLDFISH_ITERATIONS = 10000
GOLDFISH_MAX_TURN = 10
GOLDFISH_MULLIGAN_MIN_LANDS = 2
GOLDFISH_MULLIGAN_MAX_LANDS = 5
GOLDFISH_MAX_MULLIGANS = 2

# ── Pilot: Strategy Knowledge Base ───────────────────────────────────────
STRATEGY_DIR = DATA_DIR / "strategy"
STRATEGY_DOC_PATH = STRATEGY_DIR / "strategy.md"
STRATEGY_CHANGELOG_PATH = STRATEGY_DIR / "CHANGELOG.md"
STRATEGY_INDEX_PATH = STRATEGY_DIR / "strategy_index.json"
STRATEGY_EMBEDDINGS_PATH = STRATEGY_DIR / "strategy_embeddings.npy"
STRATEGY_DB_META_PATH = STRATEGY_DIR / ".strategy-db-meta.json"
STRATEGY_QUERY_TOP_K = 6
STRATEGY_RESEARCH_MAX_ITERATIONS = 3
STRATEGY_SECTION_WARN_CHARS = 1200

# ── Pilot: Agent Invocation Cache ────────────────────────────────────────
# Subagent spawns are the only real cost in this subsystem (the renderer is
# free and deterministic). A routine is one agent invocation whose output is a
# tracked artifact; if none of its declared inputs changed, the skill reuses
# the artifact instead of paying for the spawn. See docs/agent-cost.md.
AGENT_CACHE_VERSION = 1        # bump to invalidate every routine everywhere
AGENT_CACHE_FILENAME = ".agent-cache.json"
AGENT_PROMPTS_DIR = _REPO_ROOT / ".claude" / "agents"
STYLE_DOC_PATH = _REPO_ROOT / "STYLEv3.md"
ISSUE_SPEC_PATH = _REPO_ROOT / "src" / "manamap" / "pilot" / "issue_spec.py"

# Input tokens resolved by agent_cache.resolve_inputs():
#   deck:<name>[?]     file under data/decks/<slug>/ ('?' = optional; absence
#                      is still hashed, so an input appearing invalidates)
#   stacks:passing     every stacks/*.json with checker.verdict == "pass"
#   decisions:all      every decisions/*.json
#   global:<CONST>     repo-level artifact named by a config constant
#   repo:<CONST>       tracked source file named by a config constant
#   scenario:self      the {title, scenario} slice of the routine's own
#                      artifact — so the resolution/checker blocks the loop
#                      writes back never self-invalidate
#   strategy:doc       strategy.md bytes via common.strategy_doc_sha256();
#                      never the derived index, so build-strategy-db is free
#   prose:shape        manual_prose.json key skeleton only — rewording is free,
#                      adding or removing a section is not
#   rules:version      effective_date from data/rules/.rules-meta.json
AGENT_ROUTINES = {
    "strategic-frame": {
        "agent": "strategy-researcher",
        "artifact": "strategic_frame.json",
        "inputs": ["cards:semantic", "deck:goldfish_metrics.json",
                   "stacks:passing", "strategy:doc"],
    },
    "coach-prose": {
        "agent": "pilot-coach",
        "artifact": "manual_prose.json",
        "artifact_keys": ["threat_assessment", "matchups"],
        "inputs": ["cards:semantic", "deck:goldfish_metrics.json", "stacks:passing",
                   "deck:strategic_frame.json?", "strategy:doc"],
    },
    "writer-prose": {
        "agent": "manual-writer",
        "artifact": "manual_prose.json",
        "artifact_keys": ["cover", "how_it_wins", "combo_lines", "card_roles",
                          "mulligan", "upgrades"],
        "inputs": ["cards:semantic", "stacks:passing", "deck:strategic_frame.json?",
                   "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
    "issue-plan": {
        "agent": "magazine-editor",
        "artifact": "issue_plan.json",
        "inputs": ["repo:STYLE_DOC_PATH", "repo:ISSUE_SPEC_PATH",
                   "deck:issue.json", "cards:semantic", "stacks:passing",
                   "decisions:all", "deck:goldfish_metrics.json",
                   "deck:strategic_frame.json?", "prose:shape", "cards:printing",
                   "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
}

# Dynamic routines (stack:NNN / decision:NNN) — resolved per artifact.
AGENT_ROUTINE_STACK_AGENT = "stack-resolver+rules-checker"
AGENT_ROUTINE_STACK_INPUTS = ["scenario:self", "cards:semantic", "rules:version"]
AGENT_ROUTINE_DECISION_AGENT = "pilot-coach"
AGENT_ROUTINE_DECISION_INPUTS = ["scenario:self", "cards:semantic",
                                 "deck:goldfish_metrics.json",
                                 "deck:strategic_frame.json?", "strategy:doc"]
