"""Every constant in the project, in one file on purpose.

Two things make a single module the right shape here, and both would be weakened
by splitting it behind a façade:

**The frozen/mutable boundary is a rule, not a filing system.** Changing
`MECHANICAL_TAGS` or any model-facing dimension invalidates `model_ability.pt`
and forces a retrain of steps 3-5. Those constants sit together under one loud
warning; spread across modules, the rule becomes something you have to remember
rather than something you read. `ROLE_PATTERNS` is a SEPARATE dict for exactly
this reason — roles change often, tags must not.

**Seventy modules import from here.** A façade would mean two files to keep in
sync forever, for no behaviour change.

Sections, in file order:

    paths            data artifacts, discovery bundle, eval, combo/deck data
    FROZEN — model   text encoder, vocab sizes, embedding dims, feature dims,
                     the function model's output split, fusion MLP, training
                     hyperparameters, MECHANICAL_TAGS, the InfoNCE objective
    rulebooks        SYNERGY_RULES (24), ROLE_PATTERNS and friends (53 roles in
                     19 families), power-creep criteria, region clustering
    pilot            rules DB, decks and manuals, Commander brackets, deck
                     construction, DECK_AXIS_TARGETS (the diagnosis substrate),
                     goldfish, strategy KB, AGENT_ROUTINES (the cache registry)

Three source files and every agent charter are declared cache inputs
(`repo:STYLE_DOC_PATH`, `repo:ISSUE_SPEC_PATH`, `repo:DECK_AUDIT_PATH`). This
file is NOT one of them — editing it invalidates nothing by itself, but changing
a VALUE an artifact was derived from does.
"""

import os
from pathlib import Path

# Anchor all paths to the repo root (src/manamap/config.py → two levels up)
# so modules work regardless of CWD. Overridable for sandboxed runs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MANAMAP_DATA_DIR", _REPO_ROOT / "data"))
# Raw dumps are stored gzipped: JSON compresses ~8.5x and these two are 600 MB
# uncompressed. `ingest/common.open_dump` reads either form, so an existing
# uncompressed file keeps working until the next download replaces it.
RAW_JSON_PATH = DATA_DIR / "oracle-cards.json.gz"
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

# ── Discovery artifacts (the front door) ──────────────────────────────────
# Two small files that let the browser land on a card and branch from it without
# fetching the 12.9 MB projection or the 16.8 MB embedding matrix. Both are
# positionally aligned with cards.csv and belong to the same index invariant:
# regenerate them with the pipeline, never on their own.
VIZ_INDEX_PATH = DATA_DIR / "viz_index.json"
NEIGHBOURS_BIN_PATH = DATA_DIR / "neighbours.bin"

NEIGHBOURS_MAGIC = b"MMNB"
# v2 adds the synergy reason block and an appended vocabulary. The 64-byte header
# was already padded, so the vocabulary length fits in the spare 4 bytes and the
# file stays a single self-describing fetch.
NEIGHBOURS_FORMAT_VERSION = 2
NEIGHBOURS_HEADER_BYTES = 64
NEIGHBOURS_NONE = 0xFFFF  # slot sentinel; safe because 34,322 < 65,535
# Reason slots are uint8, so this is the "no reason" sentinel. 24 rules today, and
# the guard in build_tables fails loudly rather than wrapping if that ever passes 255.
NEIGHBOURS_NO_REASON = 0xFF

# How many of each relation to carry. Similar is the default click, so it gets the
# most room. Synergy is 10 because that is exactly what the graph holds for every
# card that has any (min = median = max = 10) — truncating would drop real entries
# for no saving. Obsolescence tops out at 5.
NEIGHBOURS_K_SIMILAR = 12
NEIGHBOURS_K_SYNERGY = 10
NEIGHBOURS_K_OBSOLETE = 5

# ── Embedding Quality Evaluation ──────────────────────────────────────────
# The golden set is hand-authored and must stay independent of every training
# signal — see the _comment block inside the file itself.
EVAL_DIR = DATA_DIR / "eval"
SIMILARITY_GOLDEN_PATH = EVAL_DIR / "similarity_golden.json"
# Sample size for the geometry statistics (effective dim, neighbour spread).
# The full 34K x 34K similarity matrix is 4.7 GB; a sample answers the same
# question for free. Seeded, so the numbers are comparable across runs.
EVAL_GEOMETRY_SAMPLE = 4000
EVAL_SPREAD_PROBES = 300
EVAL_SEED = 42
VIZ_DIR = Path(os.environ.get("MANAMAP_VIZ_DIR", _REPO_ROOT / "viz"))

# ── Combo / Deck Builder Data ────────────────────────────────────────────
COMBOS_API_URL = "https://backend.commanderspellbook.com/variants/"
COMBOS_RAW_PATH = DATA_DIR / "combos_raw.json.gz"
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
LAYOUT_VOCAB_SIZE = 18  # 17 layouts + 1 unknown as of the 2026-08-12 dump ('front_card' and 'prepare' arrived); the preprocess guard fires exactly when this needs raising, and raising it means retrain

# ── Categorical Embedding Dims ────────────────────────────────────────────
SUPERTYPE_EMBEDDING_DIM = 16
RARITY_EMBEDDING_DIM = 8
COLOR_IDENTITY_EMBEDDING_DIM = 32
LAYOUT_EMBEDDING_DIM = 16

# ── Feature Dims ──────────────────────────────────────────────────────────
CONTINUOUS_DIM = 2
KEYWORD_DIM = 50
FINAL_EMBEDDING_DIM = 128

# Structured numeric features. Written by preprocess as separate arrays in
# card_features.npz; wired into the model when the training objective is
# replaced, so that model.py's input tuple changes exactly once.
POWER_TOUGHNESS_DIM = 3  # [power, toughness, has_stats]
MANA_PIPS_DIM = 6        # [W, U, B, R, G pip counts, generic]
COLOR_FEATURE_DIM = 6    # [W, U, B, R, G, colour count]
STRUCTURED_FEATURE_DIM = POWER_TOUGHNESS_DIM + MANA_PIPS_DIM + COLOR_FEATURE_DIM

# ── The function model's output split ─────────────────────────────────────
# The final 128 dims are two independently L2-normalized halves, weighted so
# their squared weights sum to 1. That makes the resulting cosine an exact
# convex combination:
#
#     sim(a, b) = (1 - W) * cos_learned(a, b) + W * cos_text(a, b)
#
# which is the point: the frozen text scores 0.244 recall@10 and the trained
# model scored 0.093, so a model free to discard the text did exactly that.
# This makes discarding it structurally impossible and the floor arithmetic
# rather than hope.
FUNCTION_TEXT_DIM = 32                   # dims carrying the projected frozen text
FUNCTION_LEARNED_DIM = FINAL_EMBEDDING_DIM - FUNCTION_TEXT_DIM
TEXT_PASSTHROUGH_WEIGHT = 0.3            # W above: text's share of the similarity

# Fixed scales, deliberately not derived from the data. A per-run min-max makes
# the same card's features differ between pipeline runs, so two runs' embeddings
# stop being comparable — which is what EDHREC rank used to do.
EDHREC_RANK_SCALE = 50_000   # ranks currently reach ~31,800
POWER_TOUGHNESS_SCALE = 15   # above this is a rounding error's worth of cards
PIP_COUNT_SCALE = 4          # {W}{W}{W}{W} is already extreme

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

# ── The function model's objective ────────────────────────────────────────
# In-batch InfoNCE replaces TripletMarginLoss. The margin loss stopped producing
# gradient the moment it was satisfied — which for a task as easy as the old one
# was epoch 3 — so nothing pressured the model to preserve structure within a
# class. InfoNCE keeps ranking every anchor against all B-1 other positives in
# the batch, so it stays informative long after the easy cases are solved.
#
# Temperature on L2-normalized vectors: 0.05 is the sentence-transformers MNRL
# default (scale=20). Too high and everything blurs together (watch neighbour
# spread collapse); too low and it overfits the hardest pairs (watch val loss
# diverge while recall stalls).
INFONCE_TEMPERATURE = 0.05

# Positive mining. Roles beat mechanical tags on coverage — 72.6% of cards carry
# a *specific* role at 1.62 each, against 46.9% with the two tags the old rule
# demanded — so for most of the corpus the old positive was a fallback or a
# random card. ROLE_BODY_FALLBACK is excluded deliberately: it labels all 19,050
# creatures, so "shares threat:body" is barely narrower than "is a creature" and
# would rebuild the trivial task this work exists to escape.
#
# Rarest-role-first: two cards sharing `doubler:tokens` (11 cards) say far more
# about each other than two sharing `value:etb` (5,580). Searching the anchor's
# rarest role first spends the positive on its most specific claim.
#
# Deliberately NOT gated on text similarity. The output's text half already
# guarantees the frozen-text floor, so the learned half should capture what the
# text misses; selecting positives the text already scores highly would make the
# two halves redundant.
ROLE_POSITIVE_CANDIDATES = 50

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
    "draw:burst": r"draws? (?:two|three|four|five|six|seven|\d+) cards",
    "draw:impulse": r"exile the top (?:\w+ )?cards?.{0,60}?(?:you may (?:play|cast)|until)",
    # Selection plus a replacement — Ponder, Opt, Preordain. Not card advantage
    # and not an engine: a cantrip smooths a draw step, which is a different job
    # from drawing two, and a curve model that files them together is wrong.
    "draw:cantrip": (
        r"(?:scry \d+|surveil \d+|look at the top [\w\s]{0,30}? of your library)"
        r"[\w\s,.—]{0,80}?draws? a card"
    ),
    "draw:wheel": r"discards? (?:their|your) hand.{0,60}?draws?|each player draws",
    # Interaction
    # "exile all" must be scoped to permanents: Demonic Consultation says
    # "exile all cards from your library named that card" and is a tutor, not a
    # wrath. The -1/-1 clause catches Black Sun's Zenith, which sweeps without
    # ever using the word "destroy".
    "removal:sweeper": (
        r"destroy all|destroy each"
        r"|exile all (?:other )?(?:creature|permanent|nonland|artifact|enchantment)"
        # "Each NON-VAMPIRE creature gets -X/-X" — one-sided sweepers are the
        # typal deck's wrath, and requiring "each creature" adjacent missed
        # every one of them.
        r"|all [\w-]{0,20} ?creatures get -|each [\w-]{0,20} ?creature gets -"
        r"|put (?:x|\d+|a|two|three) -1/-1 counters? on each"
    ),
    # "nonland permanent" is the modern templating for unconditional exile
    # (Anguished Unmaking, Rite of Oblivion) and the bare alternation missed it,
    # because the type word no longer follows "target" directly.
    "removal:spot": (
        r"destroy target"
        r"|exile target (?:nonland )?(?:creature|permanent|artifact|enchantment|planeswalker|battle)"
    ),
    # Bounce is interaction, not recursion: battlefield -> hand, and the tell is
    # "owner's hand" where recursion says "your hand". Cyclonic Rift heads a
    # population of ~110 cards that carried no role at all.
    "removal:bounce": r"return (?:target|all|each)[\w\s,'-]{0,50}? to (?:its|their) owner'?s? hand",
    "removal:damage": r"deals? (?:\d+|x) damage to (?:target|any target|each)",
    # Opponent-facing only. The bare "sacrifices a creature" pattern fired on
    # your own activated sacrifice *costs* — Viscera Seer, Ashnod's Altar,
    # Carrion Feeder — which made two thirds of the removal:edict population
    # sacrifice outlets rather than interaction, and inflated every deck's
    # apparent removal count.
    # The optional "other" is load-bearing: Grave Pact and Syphon Flesh say
    # "each OTHER player sacrifices", and requiring opponent/player to follow
    # each/target directly missed the format's canonical edict by two words.
    "removal:edict": r"(?:each|target) (?:other )?(?:opponent|player)[\w\s,']{0,40}? sacrifices?",
    "sac-cost": r"sacrifice (?:a|an|another)[\w\s]{0,20}?[:,]",
    "removal:tax": r"(?:spells?|abilities).{0,40}?costs? \{?\d+\}? more",
    "removal:fight": r"\bfights? (?:target|another)",
    # Half of Commander's counterspells name what they answer — "counter target
    # NONCREATURE spell", "counter target enchantment, instant, or sorcery
    # spell". Negate, Swan Song and Fierce Guardianship all missed the literal.
    "counterspell": r"counter target [\w\s,]{0,40}?(?:spell|ability)",
    # Consistency
    "tutor:unrestricted": r"search your library for a card",
    "tutor:narrow": r"search your library for (?:a|an|up to \w+) (?!card)[\w\s]{0,30}?cards?",
    # Reanimation has two templates and this pattern only knew one. "Put target
    # creature card from a graveyard ONTO THE BATTLEFIELD" (Reanimate) and
    # "return the chosen cards to the battlefield" (Victimize) both missed.
    "recursion": (
        r"return (?:target |a |another )?[\w\s]{0,30}?(?:card )?from (?:your|a) graveyard to (?:the battlefield|your hand)"
        r"|put target [\w\s]{0,40}?from (?:a|your|target opponent's) graveyard onto the battlefield"
        r"|return (?:the chosen|those) cards? to the battlefield"
    ),
    # Resilience
    # Phasing and blanket protection are the same job as a hexproof grant and
    # were invisible to the keyword alternation — Teferi's Protection, the
    # 107th most played card in the format, carried no role at all.
    "protection:self": (
        r"\b(?:hexproof|shroud|indestructible|ward)\b"
        r"|protection from everything|phases? out"
    ),
    # A fog is not protection granted to a permanent — it answers the whole
    # combat step. ~128 cards, headed by Fog, Darkness and Arachnogenesis.
    "protection:fog": r"prevents? all (?:combat )?damage|damage that would be dealt this turn is prevented",
    # The Swat/Bend cycle: it does not counter the spell, it points it back.
    # Functionally how a Commander deck saves its commander from a removal
    # spell, and there was no role that described it.
    "protection:redirect": r"choose new targets for target|changes? the target of",
    # The keyword rarely comes first. Akroma's Memorial reads "creatures you
    # control have flying, first strike, vigilance, trample, haste, and
    # protection from black" — requiring adjacency missed the whole anthem.
    "protection:granted": (
        r"(?:target |another target )?creatures? you control (?:gains?|have|has) "
        r"[\w\s,]{0,60}?(?:hexproof|indestructible|protection|shroud)"
    ),
    # Engines
    "sac-outlet": r"sacrifice (?:a|an|another) (?:creature|permanent|artifact|token)[^.]{0,20}?:",
    "stax": r"players? can't|can't be (?:activated|cast)|skip (?:your|their) |don't untap|enters? tapped and",
    # Finishers
    "wincon:alt": r"wins? the game|loses? the game",
    # The old pattern demanded a literal numeral and the word "each", so the two
    # halves of the format's most famous loop both missed: Sanguine Bond says
    # "target opponent loses THAT MUCH life" and Exquisite Blood says "whenever
    # AN opponent loses life". Blood Artist and Vito missed for the same reason.
    "wincon:drain": (
        r"each opponent loses (?:\d+|x) life"
        r"|each opponent (?:loses|sacrifices)"
        r"|(?:target |an? )?(?:opponent|player)s? loses? (?:that much|\d+|x) life"
        r"|whenever (?:an?|target) opponent loses life"
    ),
    "wincon:combat": r"\b(?:infect|double strike)\b|deals? double|can't be blocked",
    # Broad jobs. Most cards in a deck are not a tutor or a sweeper — they are
    # a body, an enters-trigger, or an activated ability, and a builder counts
    # those slots too.
    # The single largest hole the taxonomy had: 532 Commander-legal cards create
    # creature tokens and none of them carried a role for it. A token maker is a
    # threat generator — same family as a body, different mechanism — so this
    # needs no new family and no colour.
    "threat:tokens": r"creates? [\w\s,/\d'\-]{0,60}?creature tokens?",
    # Treasure is ramp, not a threat, and the distinction matters to a curve
    # model: a Treasure is a one-shot rock the deck can cash for colour.
    "ramp:treasure": r"creates? (?:a|an|two|three|four|x|\d+|that many)[\w\s]{0,20}?treasure tokens?",
    "value:etb": r"when(?:ever)? (?:this|[\w\s,']{0,30}?) enters",
    "utility:activated": r"\{T\}[,:]|\{\d+\}[,:].{0,40}?:",
    "removal:debuff": r"gets? -\d+/-\d+|gets? -\d+/-0|put (?:a|x|\d+|two|three) -1/-1 counters?",
    # Cards that trigger off counters rather than placing them. Whole archetypes
    # (Hapatra's -1/-1 payoffs: Nest of Scarabs, Flourishing Defenses) carried
    # no role at all before this, and absence of a role reads to a slot filler
    # as absence of a function.
    "payoff:counters": (
        r"(?:whenever|when).{0,40}?(?:you )?put (?:one or more|a|x|\d+|two|three)"
        r"(?: \+1/\+1| -1/-1)? counters?"
        r"|(?:\+1/\+1|-1/-1) counters? (?:is|are) put"
    ),
    # Voltron and combat-trick slots. An Aura that pumps and an Equipment that
    # pumps are the same job wearing different card types.
    "buff:attached": r"(?:enchanted|equipped) creature (?:gets?|has|have)|\bequip[\s—]|\benchant creature\b",
    # Modern typal templating drops the word "creatures" — "Other Vampires you
    # control get +1/+1". Requiring it made Legion Lieutenant, the best two-drop
    # in a Vampire deck, invisible to every automated signal in the repo.
    "buff:pump": (
        r"gets? \+\d+/\+\d+ until end of turn"
        r"|(?:other )?[\w\s]{0,20}?you control get \+\d+/\+\d+"
    ),
    "buff:counters": r"put (?:a|two|three|\d+|x) \+1/\+1 counters?",
    "hate:graveyard": r"exile[\w\s]{0,30}?from (?:a|target opponent's|each) (?:player's )?graveyard|graveyards? instead",
    # Typal payoffs almost never name the tribe: "as this enters, choose a
    # creature type". Herald's Horn, Vanquisher's Banner, Door of Destinies and
    # friends carried no role at all, so no text search for a tribe and no role
    # query could find them — a structural blind spot for every typal commander.
    "payoff:typal": r"choose a creature type|of the chosen type|creature type of your choice",
    # Multiplication is its own job and the taxonomy had no name for it. A
    # doubler is not a payoff and not a threat — it is a multiplier on whatever
    # the deck already does, which is why Doubling Season sits in a thousand
    # lists it has no other business in. Panharmonicon and Anointed Procession
    # carried no role at all; Mondrak was classified on its ward clause, i.e.
    # on the least interesting sentence on the card.
    # Two templates for the same effect. Parallel Lives says "would CREATE one
    # or more tokens"; Mondrak reverses it to "one or more TOKENS WOULD BE
    # created", which is why it was classified on its ward clause instead.
    "doubler:tokens": (
        r"would create (?:one or more )?tokens?[\w\s,'\-]{0,60}?(?:twice|that many plus)"
        r"|tokens? would be created[\w\s,'\-]{0,40}?(?:twice|that many plus)"
    ),
    "doubler:counters": (
        r"would put (?:one or more )?[\w\s+/\-]{0,20}?counters?"
        r"[\w\s,'\-]{0,70}?(?:twice|that many plus)"
    ),
    "doubler:triggers": r"(?:ability|abilities) triggers? an additional time",
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
# Farseek and Nature's Lore name basic land *types*, never the word "land" —
# the same miss `land:fetch` already works around, applied to the spell side.
ROLE_LAND_RAMP = (
    r"search your library for (?:a|up to \w+)[\w\s]{0,30}?land"
    r"|search your library for a[\w\s,]{0,40}?(?:Plains|Island|Swamp|Mountain|Forest) card"
)
ROLE_COST_REDUCTION = r"costs? \{?\d+\}? less|spells? you cast cost"

# Land quality — lands never carry spell roles, so these are evaluated alone.
ROLE_LAND_PATTERNS = {
    # Fetchlands name basic land *types*, not "land card" — Windswept Heath
    # says "a Forest or Plains card" and would otherwise read as plain utility.
    "land:fetch": r"search your library for a[\w\s]{0,40}?(?:land card|Plains|Island|Swamp|Mountain|Forest)",
    "land:tapped": r"enters tapped|enters the battlefield tapped",
    # Channel lands are a spell stapled to a land, and lands never carry spell
    # roles by design — so Takenuma and Boseiju read as blank without this.
    "land:utility": r"\{T\},|\{T\}: (?!add)|draw a card|deals? \d+ damage|\bchannel\b",
    # Fixing is the job a three-colour deck actually buys a land for, and it was
    # unnamed: Reflecting Pool, Command Tower and Mana Confluence all produce
    # "any colour", while Urborg and Yavimaya rewrite every other land's types.
    "land:fixing": (
        r"mana of any (?:type|color|colour)"
        r"|each land is a|lands? you control are"
    ),
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
# L2 — neighbourhoods. Clustered WITHIN each L1 rather than globally, so nesting is a
# property of the construction instead of something to check afterwards. The two global
# passes above nest by luck: measured on the 2026-07-31 map, zero of 106 L1 clusters
# straddled an L0 boundary, but nothing prevents it.
REGION_L2_MIN_CLUSTER_SIZE = 25
REGION_L2_MIN_SAMPLES = 5
# Below this an L1 region is already a neighbourhood and is not subdivided.
REGION_L2_MIN_PARENT_SIZE = 120
# Hand-authored region names, keyed by content signature. Tracked, human-editable.
REGION_NAMES_PATH = DATA_DIR / "region_names.json"

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

# A PHYSICAL card collection: `*.txt` decklists of what someone actually owns.
# `deck_history.pending()` reads it to decide whether a proposed swap reads as
# "buy" or "own", and that is the only ownership question left in the repo.
#
# Overridable, and that matters more than it looks. These files lived in a
# top-level `share/` with no config entry, absent from the layout diagram and
# documented in one buried sentence — so every clone of this repo silently
# answered "do you own this card?" from ONE person's boxes. Point
# `MANAMAP_COLLECTION_DIR` at your own, or leave it and get the worked example.
# An absent directory is not an error: no collection means no ownership claim.
COLLECTION_DIR = Path(os.environ.get("MANAMAP_COLLECTION_DIR",
                                     DATA_DIR / "collection"))
SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection"
SCRYFALL_BATCH_SIZE = 75
SCRYFALL_REQUEST_DELAY_S = 0.1
# Transient 429/5xx retries. A fetch is all-or-nothing across batches, so one 503
# on batch 2 of 3 used to discard the two that succeeded.
SCRYFALL_MAX_RETRIES = 4
SCRYFALL_RETRY_BACKOFF_S = 1.0
RESOLVE_MAX_ITERATIONS = 3

# Scope budget for one stack artifact — advisory, and unlike the bound above it is
# actually imported (validate_stack.py).
#
# Citation count predicts iteration count, measured across the first three published
# decks. Every artifact at <= 32 citations passed in one or two rounds (goblin-storm,
# 5 of 5); every artifact at >= 59 needed four rounds or failed outright (hapatra 59
# @4, sisay 84 fail, 82 fail, 116 @3). The checker's verdict is atomic over the whole
# artifact, so each citation is an independent chance to be judged unsupported and the
# probability of a clean pass falls off a cliff as the artifact grows.
#
# The fix is authoring discipline, not a bigger bound: one rules domain per scenario.
# Sisay 003's answers (a)-(d) were verified correct in all three passes and were
# discarded because sub-question (e) failed in the same file.
RESOLVE_SCOPE_BUDGET = {
    "max_steps": 12,
    "max_citations": 40,
    "max_subquestions": 3,
}

# ── Pilot: Commander Brackets ────────────────────────────────────────────
# WotC's bracket ladder. The restrictions here are the ones we can check
# mechanically; the system is explicitly NOT a calculator, so bracket.py
# reports what a deck *contains* and the floor that implies. It never tells a
# player what bracket their deck is.
#
# `game_changers`/`two_card_infinites` are counts (None = unlimited).
# `expected_turns` matches WotC's current published copy verbatim.
#
# There is deliberately no tutor field. An earlier version of this table
# carried a `tutor_budget` of 1/3/6, which had no source — and the
# strategy:deckbuilding.power-level research established something stronger
# than "unnumbered": WotC *removed* the tutor restriction outright on
# 2025-10-21, because "not all Tutors are created equal", and now relies on
# Game Changers membership to catch the efficient ones. Since the code already
# counts Game Changers, a tutor budget would encode a rule that no longer
# exists. bracket.py still reports tutor density as information.
BRACKETS = {
    1: {"name": "Exhibition", "game_changers": 0, "two_card_infinites": 0,
        "mass_land_denial": False, "expected_turns": "9+"},
    2: {"name": "Core", "game_changers": 0, "two_card_infinites": 0,
        "mass_land_denial": False, "expected_turns": "8+"},
    3: {"name": "Upgraded", "game_changers": 3, "two_card_infinites": 0,
        "mass_land_denial": False, "expected_turns": "6+"},
    4: {"name": "Optimized", "game_changers": None, "two_card_infinites": None,
        "mass_land_denial": True, "expected_turns": "4+"},
    5: {"name": "cEDH", "game_changers": None, "two_card_infinites": None,
        "mass_land_denial": True, "expected_turns": "any"},
}
BRACKET_DEFAULT = 3
BRACKET_MAX = 5

# WotC states the Bracket 3 combo guardrail as a *turn* ("you don't expect to
# win or lose before turn six"), not a mana value. This is our proxy for that,
# and it is a proxy — a cheap combo that needs a long setup will read as early
# when it isn't. Note WotC also says holding a combo back doesn't exempt it:
# "if a combo could frequently come up, it's not the best fit for that bracket."
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

# ── Pilot: Deck Construction ─────────────────────────────────────────────
# The deterministic builder. Every constant here has a home so it can be
# tested and tuned — the browser prototype's equivalents were JS literals with
# no derivation, no tests, and no way to reason about them.
DECK_SIZE = 100
DECK_BUILD_ALTERNATES = 3
DECK_BUILD_MAX_BRACKET_PASSES = 10
DECK_BUILD_MAX_ITERATIONS = 3      # architect ⇄ critic, mirrors RESOLVE_MAX_ITERATIONS

# Slot budget for the 99. PROVISIONAL: these are conventional Commander ratios
# and they are the one part of the builder not yet grounded in a citable
# source. The strategy:deckbuilding.ratios research pass (M0) replaces them
# with numbers the architect can cite; until then the plan reports them as
# uncited and `validate_build` does not treat them as evidence.
DECK_ROLE_BUDGET = {
    "lands": 36,
    "ramp": 10,
    "draw": 10,
    "removal": 8,
    "sweeper": 3,
    "protection": 3,
    "recursion": 2,
    "tutor": 2,
    "wincon": 3,
    "flex": 22,
}

# Which classifier roles satisfy each budget line. `flex` takes anything and is
# what makes the deck feel like the commander rather than a checklist.
DECK_ROLE_GROUPS = {
    # ramp:treasure was missing until deck-audit counted a Treasure maker as
    # `flex`. It is a real classifier role on 364 cards (4 of them in
    # yawgmoth-swarm) and it accelerates mana, which is the only question this
    # group asks.
    "ramp": ("ramp:rock", "ramp:dork", "ramp:ritual", "ramp:land",
             "ramp:cost-reduction", "ramp:treasure"),
    "draw": ("draw:engine", "draw:burst", "draw:impulse", "draw:wheel", "draw:cantrip"),
    "removal": ("removal:spot", "removal:damage", "removal:edict", "removal:fight",
                "removal:debuff", "removal:bounce", "counterspell"),
    "sweeper": ("removal:sweeper",),
    # protection:redirect likewise — a card that redirects a removal spell is
    # protection by any reading, and it was falling to flex.
    "protection": ("protection:self", "protection:granted", "protection:fog",
                   "protection:redirect"),
    "recursion": ("recursion",),
    "tutor": ("tutor:unrestricted", "tutor:narrow"),
    "wincon": ("wincon:alt", "wincon:drain", "wincon:combat"),
    # Engine payoffs and multipliers are the deck, not filler. A doubler files
    # here rather than under wincon: it multiplies whatever the deck already
    # does instead of being a way to win on its own.
    "flex": ("payoff:counters", "payoff:typal", "sac-cost",
             "doubler:tokens", "doubler:counters", "doubler:triggers"),
}

# ── Pilot: Deck Audit (the diagnosis substrate) ──────────────────────────
# `deck-audit` joins deck-facts, mana_analysis, goldfish, bracket_report and
# card_roles into one axis table. Every axis carries a target the corpus can
# actually support, so an agent reading the audit can cite the number instead
# of inventing it — the failure mode DECK_ROLE_BUDGET was built to have.
#
# `quote` is a VERBATIM span of the named section (whitespace-normalized at
# comparison, the same gate validate_stack.validate_citations applies).
# tests/test_pilot_deck_audit.py fails if the doc drifts away from any of
# them — which is the point: a target nobody can quote is not a target.
#
# `low`/`high` are inclusive; `None` means the corpus states no bound in that
# direction. Draw has no ceiling in the literature and this does not invent one.
DECK_AXIS_TARGETS = {
    "mana-base": {
        # LAND copies against the format's conventional band. Burgess's formula
        # lives on `mana-sources` below, not here: it budgets sources and counts
        # 0-mana rocks as lands, so applying it to the land count alone asks a
        # 5-colour deck with a 9-mana commander for 45 lands.
        "low": 36, "high": 38,
        "source": "strategy:deckbuilding.ratios",
        "quote": ("36–38 lands, 10–12 ramp, 10 card draw, 10–12 targeted "
                  "removal, 3–4 board wipes"),
    },
    "mana-sources": {
        # "Budget mana *sources*, not lands" is the section's first sentence, and
        # this is that instruction made literal: lands plus the PERSISTENT
        # producers (rocks, dorks, land ramp). Rituals and Treasures are one-shot
        # and are deliberately not sources.
        "low": None, "high": None, "formula": "burgess",
        "source": "strategy:deckbuilding.mana-base",
        "quote": ("lands = 31 + colours in the commander's identity + the "
                  "commander's mana value, counting 0-mana rocks as lands"),
    },
    "creatures": {
        # No base target — the corpus sets a creature count only per archetype
        # (aggro 26-32), so unarchetyped decks get the number reported, not judged.
        "low": None, "high": None,
        "source": "strategy:deckbuilding.archetype-selection",
        "quote": ("aggro 26-32 creatures / 5-6 removal / 34-36 lands, control "
                  "12-15 removal / 5-7 wipes / 37-39 lands, combo 4-8 tutors and "
                  "4-6 protection, Voltron 12-16 equipment and auras"),
    },
    "taplands": {
        "low": None, "high": 8,
        "source": "strategy:deckbuilding.mana-base",
        "quote": ("around eight taplands is comfortable in a slow deck, near "
                  "zero in an aggressive one"),
    },
    "colour-sources": {
        # Target is per colour, computed by manabase.source_targets from the
        # deck's own pip load. The quote is the yardstick that produces it.
        "low": None, "high": None, "formula": "karsten",
        "source": "strategy:deckbuilding.mana-base.color-sources",
        "quote": ('his 99-card column assumes 40 lands, on the play, casting '
                  'on curve, "consistently" ≈ 90%'),
    },
    "ramp": {
        "low": 10, "high": 12,
        "source": "strategy:deckbuilding.ratios",
        "quote": ("36–38 lands, 10–12 ramp, 10 card draw, 10–12 targeted "
                  "removal, 3–4 board wipes"),
    },
    "card-advantage": {
        "low": 10, "high": None,
        "source": "strategy:deckbuilding.ratios",
        "quote": ("36–38 lands, 10–12 ramp, 10 card draw, 10–12 targeted "
                  "removal, 3–4 board wipes"),
    },
    "interaction": {
        "low": 8, "high": None,
        "source": "strategy:deckbuilding.interaction-suite",
        "quote": ('Walser\'s baseline is "at least 8-10 removal spells" inside '
                  'a "15- to 20-card interactive suite"'),
    },
    "interaction-breadth": {
        # Five permanent classes; the measure is how many are answered at all.
        "low": 5, "high": None,
        "source": "strategy:deckbuilding.interaction-suite",
        "quote": ("cover the classes — creature, artifact, enchantment, "
                  "graveyard, land — before doubling any"),
    },
    "sweepers": {
        "low": 1, "high": 3,
        "source": "strategy:deckbuilding.interaction-suite",
        "quote": 'wipes "one or two max" (Zupke caps at 3)',
    },
    "protection": {
        "low": 5, "high": 7,
        "source": "strategy:deckbuilding.threat-density",
        "quote": ('3-5 finishers, "at least 3 finishers in your deck", 5-7 '
                  "protection slots, 20-25 flexible strategy cards"),
    },
    "threat-density": {
        "low": 3, "high": None,
        "source": "strategy:deckbuilding.threat-density",
        "quote": ('Countable floor, Zupke\'s structure: 3-5 finishers, "at '
                  'least 3 finishers in your deck"'),
    },
    "consistency": {
        # Roach goldfished 100 real decks; the complement of his 26% miss rate
        # is the only turn-3 bound the corpus actually states. Engine redundancy
        # is reported separately, per component, under ENGINE_REDUNDANCY_CITATION
        # — one number for "does the deck function" and one for "does the engine".
        "low": 0.74, "high": None, "unit": "rate",
        "source": "strategy:deckbuilding.mana-base",
        "quote": ("had 26% miss the turn-3 land drop with no mana source at all"),
    },
    "curve": {
        "low": None, "high": None, "formula": "modal-two",
        "source": "strategy:deckbuilding.curve",
        "quote": ("puts the modal mana value at 2, with 15.7 two-drops and "
                  "15.4 three-drops in the average deck and only ~1.5 cards at "
                  "mana value 8+"),
    },
    "tutors": {
        # No base bound, deliberately. WotC removed the tutor restriction on
        # 2025-10-21 and bracket.py reports tutor density without scoring it; the
        # corpus gives a count only inside the combo archetype's spread. An axis
        # with no number to cite gets none rather than an invented one.
        "low": None, "high": None,
        "source": "strategy:deckbuilding.archetype-selection",
        "quote": ("aggro 26-32 creatures / 5-6 removal / 34-36 lands, control "
                  "12-15 removal / 5-7 wipes / 37-39 lands, combo 4-8 tutors and "
                  "4-6 protection, Voltron 12-16 equipment and auras"),
    },
    "power": {
        # Reported, never scored: the bracket engine computes a floor and WotC
        # is explicit that contents are not a verdict.
        "low": None, "high": None,
        "source": "strategy:deckbuilding.power-level",
        "quote": "Contents give a floor, never a verdict.",
    },
}

# Per-archetype overrides on the axis targets above. `strategy:deckbuilding.ratios`
# ends by saying templates are "counts of *functions*, not cards" and that the
# right move is to "derive the counts from the deck's actual failure modes";
# `.archetype-selection` is where the corpus actually varies them, so these are
# its numbers and nothing else. An archetype with no entry keeps the base targets
# and the audit says so rather than guessing.
# NOTE aggro's 26-32 lands on `creatures`, not on `threat-density`: it is a
# creature count, and threat-density counts wincon:* roles. Comparing the two
# told edgar-vampires it was thirteen finishers short of an aggro deck.
DECK_ARCHETYPE_BUDGETS = {
    "aggro":   {"mana-base": (34, 36), "interaction": (5, 6), "creatures": (26, 32)},
    "control": {"mana-base": (37, 39), "interaction": (12, 15), "sweepers": (5, 7)},
    "combo":   {"tutors": (4, 8), "protection": (4, 6)},
    "voltron": {"protection": (6, 8)},
}
DECK_ARCHETYPE_BUDGET_CITATION = {
    "rule": "strategy:deckbuilding.archetype-selection",
    "quote": ("aggro 26-32 creatures / 5-6 removal / 34-36 lands, control 12-15 "
              "removal / 5-7 wipes / 37-39 lands, combo 4-8 tutors and 4-6 "
              "protection, Voltron 12-16 equipment and auras"),
}

# The engine block reads each goldfish target's `any_of` groups as the engine's
# components and prices their size through the hypergeometric. The computed odds
# REPRODUCE this quote (5 -> 0.312, 7 -> 0.412, 10 -> 0.537 in an opening seven),
# which is why the corpus line is carried as corroboration rather than as the
# source of the numbers — a test asserts they still agree.
ENGINE_REDUNDANCY_CITATION = {
    "rule": "strategy:deckbuilding.redundancy-vs-tutors",
    "quote": ("5 copies is 31% of openers, 7 is 41% (WitchPHD's 7-of, the "
              "singleton 4-of, 41.1%), 10 is 54%"),
}

# An `any_of` group this small is a single point of failure worth naming. Not a
# probability threshold on purpose: the odds are reported per group and the
# reader can see them, but "one card" and "two cards" are the cases where a
# closer is worth looking for at all.
ENGINE_THIN_GROUP = 3
# Closers are ranked lists, not exhaustive ones — the whole pool is one
# the pool brief away.
ENGINE_MAX_CLOSERS = 8

# How stale a `deck_recon.json` may be before the diagnose loop re-runs it.
# Enforced by the SKILL, not by deck_audit: the audit is deterministic and
# never reads the clock, so it reports `as_of` verbatim and the caller judges.
RECON_MAX_AGE_DAYS = 120

# The diagnose loop's bound — doctor ⇄ skeptic, mirroring RESOLVE_MAX_ITERATIONS
# and DECK_BUILD_MAX_ITERATIONS. A diagnosis that cannot satisfy the skeptic in
# three passes is a finding about the deck's artifacts, not a reason to keep paying.
ENGINE_MAX_ITERATIONS = 3
DIAGNOSE_MAX_ITERATIONS = 3

# Scoring weights. Sum to 1.0. Three deliberate departures from the prototype:
# EDHREC rank is log-scaled and bracket-damped (a uniform global popularity
# rank pushes every deck toward the same staples, which is wrong for a synergy
# deck and wrong for a low bracket); curve is scored, so a 13-drop no longer
# competes with a two-drop for a ramp slot; and castability measures pip
# intensity rather than colour legality, which the pool filter already
# guarantees.
DECK_BUILD_WEIGHTS = {
    "similarity": 0.30,
    "synergy": 0.22,
    "combo": 0.12,
    "curve": 0.16,
    "edhrec": 0.10,
    "castability": 0.10,
}

# Curve scoring. Commander is not a format where a 13-mana card is a ramp
# payoff, and the prototype's hardcoded 7-bucket target had no derivation.
DECK_CURVE_SWEET_SPOT = 3
DECK_CURVE_TOLERANCE = 5.0

# EDHREC's contribution is scaled by bracket: at Exhibition the format's
# most-played cards are the least appropriate answer.
DECK_BUILD_EDHREC_BY_BRACKET = {1: 0.2, 2: 0.5, 3: 1.0, 4: 1.0, 5: 1.0}

# ── Pilot: Goldfish Simulation ───────────────────────────────────────────
GOLDFISH_SEED = 42
GOLDFISH_ITERATIONS = 10000
GOLDFISH_MAX_TURN = 10
GOLDFISH_MULLIGAN_MIN_LANDS = 2
GOLDFISH_MULLIGAN_MAX_LANDS = 5
GOLDFISH_MAX_MULLIGANS = 2
# Combat model (opt-in per deck via `model_combat` in goldfish_targets.json).
# ONE opponent, because a goldfish kill clock answers "how fast could this board
# finish a seat", not "how fast does it win a four-player game" — summing three
# opponents' life would invent a number the model cannot support.
GOLDFISH_OPPONENT_LIFE = 40

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
# ...and be sure you mean it: a bump produces no *input* changes, so it can never
# be STALE_OK and `rebless` cannot rescue it. Every routine on every deck becomes
# a real spawn. There is no partial setting. Use CARD_REFS_VERSION below instead
# whenever the question is "which cards does this artifact reference", which is
# the only thing most changes to this machinery actually alter.
AGENT_CACHE_FILENAME = ".agent-cache.json"

# Bumped when the card-reference EXTRACTION changes (what counts as a mention),
# not when a deck changes. Refs ride outside the fingerprint — they refine
# invalidation without defining a HIT — so bumping this invalidates NOTHING. It
# only tells `rebless` that existing records carry refs computed under older
# rules and should be re-seeded, which is a pure re-fingerprint with no spawn.
# Without it, a fix to the extractor is inert on every already-recorded routine:
# rebless skips any HIT that merely *has* refs, however stale their derivation.
CARD_REFS_VERSION = 2
AGENT_PROMPTS_DIR = _REPO_ROOT / ".claude" / "agents"
STYLE_DOC_PATH = _REPO_ROOT / "STYLEv3.md"
ISSUE_SPEC_PATH = _REPO_ROOT / "src" / "manamap" / "pilot" / "issue_spec.py"
# The audit's CODE, for the same reason deck-diagnosis declares bracket_report and
# mana_analysis: a diagnosis must carry deck-audit's figures, so a change to the
# regexes that compute them can flip a recorded diagnosis from true to false. Fixing
# the land-class false positives moved edgar's interaction-breadth 5 -> 4 while the
# cache still read HIT, which is the "green board over a stale document" this
# registry exists to prevent — the data inputs were declared and the measurer was not.
DECK_AUDIT_PATH = _REPO_ROOT / "src" / "manamap" / "pilot" / "deck_audit.py"

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
        "inputs": ["cards:semantic", "deck:goldfish_metrics.json!meta.decklist_sha256",
                   "stacks:passing", "strategy:doc"],
    },
    "coach-prose": {
        "agent": "pilot-coach",
        "artifact": "manual_prose.json",
        "artifact_keys": ["threat_assessment", "matchups"],
        "inputs": ["cards:semantic", "deck:goldfish_metrics.json!meta.decklist_sha256", "stacks:passing",
                   "deck:strategic_frame.json?", "strategy:doc"],
    },
    # deck:goldfish_metrics.json was added 2026-07-28: the writer quotes
    # goldfish figures in how_it_wins/mulligan, and the cache not knowing
    # that is why figure staleness used to need hand-grepped re-spawns.
    "writer-prose": {
        "agent": "manual-writer",
        "artifact": "manual_prose.json",
        # No "cover" key: issue_plan.json owns the cover (build_manual.render_cover
        # reads plan["cover"] and issue["cover_tagline"], never manual_prose).
        "artifact_keys": ["how_it_wins", "combo_lines", "card_roles",
                          "mulligan", "upgrades", "mana_base"],
        # COMBO_GRAPH_PATH stands in for COMBO_DETAILS_PATH here on purpose: agents
        # read the details file, but process_combos writes both in one step, so the
        # 4.5 MB graph is a faithful invalidation proxy for the 25.7 MB details and
        # costs far less to hash.
        "inputs": ["cards:semantic", "stacks:passing", "deck:strategic_frame.json?",
                   "deck:goldfish_metrics.json!meta.decklist_sha256", "deck:mana_analysis.json?",
                   "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
    # The Short List (v3.3): one artifact replaces the sideboard-analysis /
    # upgrade-watch pair — the ten cards most worth the pilot's sleeves,
    # bench-first, pool-filled. Applicable to every deck. Deliberately NOT an
    # input to writer-prose or issue-plan: a bench edit should cost one
    # analysis, not a full manual regeneration — the renderer reads the
    # artifact directly, so the coupling stays one-way.
    "deck-engine": {
        "agent": "deck-engineer+engine-critic",
        "artifact": "engine.json",
        # `deck:deck_map.json?` is optional and `stacks:passing` is not: the map is
        # an input the model may contradict, but the verified pairings are its only
        # fact tier, so a newly passing stack must MISS this routine — it may be the
        # thing that turns a dashed line solid.
        "inputs": ["cards:semantic", "stacks:passing",
                   "deck:goldfish_targets.json", "deck:deck_map.json?",
                   "deck:strategic_frame.json?", "global:COMBO_DETAILS_PATH"],
    },
    "deck-map-names": {
        "agent": "deck-cartographer",
        "artifact": "deck_map.json",
        # Names only. Membership is a MEASUREMENT this routine may not touch, so the
        # merge writes `label`/`gloss` and nothing else (see pilot/merge_deck_map.py).
        #
        # `deck:deck_map.json` is the input rather than `cards:semantic`: a decklist
        # edit changes the clusters, but so does a retrain, and the map is the thing
        # that actually moved. Hashing the map catches both and nothing else — a
        # re-render or a prose pass leaves these names alone, which is the point of
        # naming places rather than describing contents.
        "inputs": ["deck:deck_map.json", "deck:strategic_frame.json?"],
    },
    "panel-prose": {
        "agent": "pilot-panel",
        "artifact": "manual_prose.json",
        "artifact_keys": ["editors_letter", "pilots_log"],
        # `deck:engine.json` is the load-bearing input: the panel argues in the
        # engine's stage names and may not assert a line the model draws dashed,
        # so a re-run of the engine loop MUST re-open the conversation.
        "inputs": ["cards:semantic", "deck:engine.json",
                   "deck:strategic_frame.json?", "stacks:passing"],
    },
    "the-ten": {
        "agent": "short-list-analyst",
        "artifact": "considering.json",
        # deck:diagnosis.json? was added 2026-08-03: when a diagnosis exists the
        # ten answer its NAMED deficits instead of index hits, so a re-diagnosis
        # must MISS this routine. Optional, because most decks have none yet and
        # the bench-first contract stands on its own without one.
        "inputs": ["cards:semantic", "stacks:passing", "deck:strategic_frame.json?",
                   "deck:bracket_report.json?", "deck:pilot_feedback.md?",
                   "deck:diagnosis.json?",
                   "global:COMBO_DETAILS_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
    # Fetch Quests (v3.3): the coach's tutor guide — one wish per tutor.
    # N/A for decks with zero library-search tutors in the 99 (the renderer
    # prints standing copy instead; see agent_cache applicability).
    "tutor-guide": {
        "agent": "pilot-coach",
        "artifact": "tutor_guide.json",
        "inputs": ["cards:semantic", "stacks:passing", "deck:strategic_frame.json?",
                   "deck:goldfish_metrics.json!meta.decklist_sha256", "strategy:doc"],
    },
    "issue-plan": {
        "agent": "magazine-editor",
        "artifact": "issue_plan.json",
        "inputs": ["repo:STYLE_DOC_PATH", "repo:ISSUE_SPEC_PATH",
                   "deck:issue.json", "cards:semantic", "stacks:passing",
                   "decisions:all", "deck:goldfish_metrics.json!meta.decklist_sha256",
                   "deck:strategic_frame.json?", "prose:shape", "cards:printing",
                   "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
    # The diagnosis loop. `deck-recon` is the one routine in this registry whose
    # staleness is TIME rather than inputs: a decklist edit does not change what
    # strong lists for this commander run, so hashing cards.json here would MISS a
    # web pass on every swap and cost a research round for nothing. Its declared
    # input is the brief (commander + target bracket — the only things that change
    # what recon should look for); age is judged by the SKILL against
    # RECON_MAX_AGE_DAYS, because deck_audit is deterministic and never reads the
    # clock. Deliberately NOT an input to deck-diagnosis's siblings: a recon
    # refresh should cost one diagnosis, not a manual regeneration.
    "deck-recon": {
        "agent": "deck-doctor",
        "artifact": "deck_recon.json",
        "inputs": ["deck:brief.json?"],
    },
    "deck-diagnosis": {
        "agent": "deck-doctor+deck-skeptic",
        "artifact": "diagnosis.json",
        # bracket_report and mana_analysis are declared because the audit reads
        # them for the power and mana axes — re-running either must MISS this
        # routine rather than flip a recorded diagnosis from true to false in
        # silence. Same reasoning validate-build's bracket_report input carries.
        "inputs": ["cards:semantic", "stacks:passing",
                   "deck:goldfish_metrics.json!meta.decklist_sha256",
                   "deck:goldfish_targets.json?", "deck:bracket_report.json?",
                   "deck:mana_analysis.json?", "deck:strategic_frame.json?",
                   "deck:deck_recon.json?", "deck:pilot_feedback.md?",
                   "repo:DECK_AUDIT_PATH",
                   "global:CARD_ROLES_PATH", "global:COMBO_DETAILS_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH", "strategy:doc"],
    },
    # Deck construction. Note none of these take `cards:semantic` — it digests a
    # cards.json that by definition does not exist before a build, and would
    # raise MissingInput -> exit 2 -> "stop, don't spawn". The authored brief is
    # the root input instead.
    "candidate-pool": {
        "agent": "deck-analyst",
        "artifact": "candidate_pool.json",
        # COMBO_GRAPH_PATH as the invalidation proxy for combo_details — same
        # reasoning as writer-prose above: process_combos writes both in one
        # step, and the 4.5 MB graph hashes for a tenth of the 25.7 MB details.
        "inputs": ["deck:brief.json", "global:CARD_ROLES_PATH",
                   "global:COMBO_GRAPH_PATH", "global:SYNERGY_GRAPH_PATH",
                   "global:OBSOLESCENCE_INDEX_PATH"],
    },
    "deck-build": {
        "agent": "deck-architect+deck-critic",
        "artifact": "build_plan.json",
        "artifact_keys": ["archetype", "gameplan", "role_budget",
                          "role_budget_citations", "swaps", "engines", "keep",
                          "gaps", "critic"],
        # bracket_report.json is declared because validate-build cross-checks
        # the plan's floor against it — re-running bracket-check must MISS this
        # routine, not flip a recorded plan from valid to invalid in silence.
        "inputs": ["deck:brief.json", "deck:candidate_pool.json",
                   "deck:bracket_report.json?",
                   "global:CARD_ROLES_PATH", "global:COMBO_GRAPH_PATH",
                   "global:SYNERGY_GRAPH_PATH", "strategy:doc"],
    },
}

# Dynamic routines (stack:NNN / decision:NNN) — resolved per artifact.
AGENT_ROUTINE_STACK_AGENT = "stack-resolver+rules-checker"
AGENT_ROUTINE_STACK_INPUTS = ["scenario:self", "cards:semantic", "rules:version"]
AGENT_ROUTINE_DECISION_AGENT = "pilot-coach"
AGENT_ROUTINE_DECISION_INPUTS = ["scenario:self", "cards:semantic",
                                 "deck:goldfish_metrics.json!meta.decklist_sha256",
                                 "deck:strategic_frame.json?", "strategy:doc"]

# What each manual_prose key ACTUALLY depends on — the per-key refinement of
# the owning routine's input list. A routine-level MISS consults this to name
# which keys are stale, so a scoped re-spawn regenerates only those; the
# routine-level fingerprint is untouched (these ride outside it). A key not
# listed here falls back to whole-routine staleness.
PROSE_KEY_INPUTS = {
    # writer-prose
    "card_roles": ["cards:semantic"],
    "combo_lines": ["stacks:passing"],
    "how_it_wins": ["cards:semantic", "deck:strategic_frame.json?",
                    "deck:goldfish_metrics.json!meta.decklist_sha256", "stacks:passing"],
    "mulligan": ["cards:semantic", "deck:goldfish_metrics.json!meta.decklist_sha256"],
    "upgrades": ["cards:semantic", "deck:strategic_frame.json?", "stacks:passing"],
    "mana_base": ["cards:semantic", "deck:mana_analysis.json?",
                  "deck:goldfish_metrics.json!meta.decklist_sha256"],
    # coach-prose
    "threat_assessment": ["cards:semantic", "deck:goldfish_metrics.json!meta.decklist_sha256",
                          "stacks:passing", "deck:strategic_frame.json?",
                          "strategy:doc"],
    "matchups": ["cards:semantic", "deck:goldfish_metrics.json!meta.decklist_sha256",
                 "stacks:passing", "deck:strategic_frame.json?", "strategy:doc"],
}
