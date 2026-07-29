# Architecture

## Overview

Mana Map embeds every Magic: The Gathering oracle card (~34,300 and growing) into a 128-dim vector space using two lightweight fusion MLPs, then projects to 2D with PaCMAP for the interactive map. On top of the embeddings sit four analysis layers: synergy detection, power-creep detection, region clustering, and the deckbuilding role taxonomy.

## Models (`src/manamap/training/model.py`)

**CardEmbeddingModel**: parameterized fusion MLP, 128-dim L2-normalized output.

Constructor parameters customize the model for different training objectives:
- `ci_emb_dim` — color identity embedding dim (default 32)
- `keyword_emb_dim` — if > 0, keywords pass through a learned Linear+ReLU projection; if 0/None, raw multi-hot passthrough
- `mechanical_tag_dim` — number of mechanical tag inputs (0 = no tag input)
- `mechanical_tag_emb_dim` — if > 0 with `mechanical_tag_dim` > 0, learns tag embeddings

Forward pass: categorical embeddings + text embedding + continuous features + keywords (+ optional tags) → concat → 3-layer MLP (hidden→ReLU→dropout→128→ReLU→dropout→128) → L2 normalize.

**Color+Type model** (`model.pt`):
- Inputs: text(384) + supertype(16) + rarity(8) + ci(32) + layout(16) + continuous(2) + keywords(50) = 508-dim
- ~181K trainable params, no mechanical tag input

**Ability model** (`model_ability.pt`):
- Inputs: text(384) + supertype(16) + rarity(8) + ci(**8**, shrunk) + layout(16) + continuous(2) + keywords(**50→32** learned) + tags(**33→32** learned) = 498-dim
- ~180K trainable params
- Color identity deliberately shrunk (32→8) so the model *can't* lean on color
- Result: cards cluster by function — all blink cards together regardless of color

The text encoder (all-MiniLM-L6-v2) is **frozen** — only the MLP trains.

## Training (`src/manamap/training/`)

Shared utilities (`get_device`, `collate_triplets`, `run_epoch`) live in `training/common.py`.

**`train.py`** (Color+Type):
- Positives = same (supertype, primary_color) group
- Negatives = different supertype AND different color (rejection sampled, 50 attempts)
- Fallback chain for positives: same group → same supertype → same color → random

**`train_ability.py`** (Abilities):
- Positives = share >= 2 mechanical tags (`MIN_SHARED_TAGS_POSITIVE`)
- Negatives = share 0 mechanical tags
- Fallback chain: >= 2 shared → >= 1 shared → random

**Both:** triplet margin loss (margin 0.3), batch 256, Adam lr 1e-3, 90/10 split, early stopping patience 5. Color+Type converges near-zero loss by ~epoch 7 (expected — color/type groups easily satisfy the margin); the ability model typically stops ~epoch 16 (tag groups are fuzzier, best val_loss ~0.05).

## Supertype classification (`src/manamap/ingest/extract.py`)

`SUPERTYPE_PRIORITY` in `config.py` — first match wins:

```
Planeswalker > Battle > Land > Creature > Instant > Sorcery > Enchantment > Artifact
```

**Land is before Creature** so "Land Creature" cards (Dryad Arbor, Jasconian Isle) classify as Land, preventing power-creep false positives against real creatures.

## Mechanical tags (`src/manamap/mechanical_tags.py`)

33 regex-based tags extracted from oracle text, defined in `config.py` `MECHANICAL_TAGS`. Case-insensitive matching. Coverage ~80% of non-land cards.

- **Triggers (5):** `etb`, `death_trigger`, `attack_trigger`, `damage_trigger`, `upkeep_trigger`
- **Effects (9):** `sacrifice`, `draw`, `removal`, `bounce`, `counterspell`, `blink`, `reanimate`, `tutor`, `discard`
- **Generators (6):** `tokens`, `counters_plus`, `counters_minus`, `ramp`, `lifegain`, `mill`
- **Modifiers (8):** `anthem`, `cost_reduction`, `copy`, `protection`, `evasion_flying`, `evasion_trample`, `evasion_menace`, `evasion_unblockable`
- **Permanents (3):** `equipment`, `aura`, `tap_ability`
- **Graveyard (1):** `graveyard_matters` · **Storm (1):** `storm`

Evasion is split into 4 granular tags so power creep doesn't treat flying as equivalent to trample. `MECHANICAL_TAG_NAMES = sorted(MECHANICAL_TAGS.keys())` — always sorted for consistent multi-hot encoding.

**Changing `MECHANICAL_TAGS` changes `MECHANICAL_TAG_DIM`, invalidating `model_ability.pt`. Retrain steps 3–5 after any tag change.**

## Deckbuilding roles (`src/manamap/analysis/card_roles.py`)

**Roles ≠ mechanical tags**, and the distinction is as load-bearing as "Synergy ≠ Similar" below. `MECHANICAL_TAGS` is a *retrieval* vocabulary — it answers "what is this card like" and it clusters well. Roles answer "what job does this card do in a 99", and the answers have to be **countable**.

The canonical difference: `MECHANICAL_TAGS` has one `ramp` tag matching everything that adds mana or fetches a land, so a Sol Ring, a Llanowar Elves, a Rampant Growth and a Dark Ritual are indistinguishable. `ROLE_PATTERNS` splits that into `ramp:rock` / `ramp:dork` / `ramp:land` / `ramp:ritual` / `ramp:cost-reduction`, **disambiguated by the type line** — an artifact that taps for mana is a rock, a creature is a dork, an instant is a ritual. They cost differently, they die differently, and a curve model that conflates them is wrong.

31 regex patterns in `config.py` `ROLE_PATTERNS`, plus type-derived `land:*` (6), `ramp:*` (5) and the `threat:body` fallback — **43 emitted roles**. Families: `ramp:*`, `draw:*`, `removal:*`, `tutor:*`, `buff:*`, `wincon:*`, `protection:*`, `land:*`, plus `recursion`, `counterspell`, `sac-outlet`, `stax`, `hate:graveyard`, `payoff:counters`, `value:etb`, `utility:activated`, `threat:body`.

Two rules keep it honest:

- **Lands never carry spell roles.** A land is evaluated only for land quality (fetch / tapped / untapped-dual / utility / basic / mdfc).
- **Coverage is published, not assumed.** The `meta` block reports `coverage` (86.5% of Commander-legal cards) *and* `specific_coverage` (68.1%, excluding cards carrying only the `threat:body` fallback), because labelling every creature a body and stopping would tell a slot filler nothing. `ROLE_COVERAGE_TARGET` / `ROLE_SPECIFIC_COVERAGE_TARGET` are regression floors asserted by the test suite. 100% is explicitly not the goal — the ~4.3K stragglers are genuinely miscellaneous one-off effects, and looser regexes would buy coverage with false positives, which is worse than an honest null.

`ROLE_PATTERNS` is a **separate dict from `MECHANICAL_TAGS` on purpose**: roles are not model-facing, so editing them costs only `manamap card-roles` (step 13) rather than a retrain.

## Commander brackets (`src/manamap/pilot/bracket.py`)

Computes a bracket **floor** — the lowest WotC bracket a deck's contents are consistent with — from three mechanically checkable signals, each of which names the specific card or line responsible:

- **Game Changers**, from the `game_changer` column in `cards.csv` (Scryfall carries WotC's list).
- **Combo content.** Commander Spellbook tags every variant, and `COMBO_BRACKET_TAGS` maps its letters onto the ladder (`E`→1, `C`/`O`→2, `P`/`S`→3, `R`→4; `B` is banned and excluded). A separate **two-card-infinite test** runs independently of that tag — which matters, because Spellbook tags a real Hapatra two-card infinite as bracket 1 and the engine returns 4 anyway.
- **Mass land denial**, a curated `MASS_LAND_DENIAL` name list — the one signal not derived from data, and the report says so.

Two deliberate refusals. It never scores **tutor density**: WotC removed that guardrail on 2025-10-21 and now relies on Game Changers membership, so a tutor budget would encode a rule that no longer exists. And it never returns a verdict — brackets are a conversation, so the report says what a deck *contains* and what that is consistent with.

It also excludes lines that assume one of their own pieces is your commander (`"Infinite commander casts"` in `produces` is the tell), encoding Judge's Desk A-004 where CR 903.9a refuted a promised infinite.

## Mana base math (`src/manamap/pilot/manabase.py`)

Colour-source counts are **computed hypergeometrically** (callers must pass `common.expand_copies()` output — `manabase.py` counts whatever list it is handed, and handing it decklist *entries* silently halves every basic-land colour), not quoted from a table: the fewest sources k such that P(at least `pips` among the cards seen by turn T) ≥ 90%, over a 99-card library. Three calibrations keep the output usable. **Conditional sources count only at their unconditional value**: "add one mana of any color" is a five-colour source only when unrestricted, so Haven of the Spirit Dragon ("spend this mana only to cast a Dragon creature spell") taps for `{C}` in a Vampire deck. Understating a source is recoverable; overstating one produces a deck that cannot cast its spells, and a greedy selector reaches for those lands precisely because they *look* like they cover everything. Then — requirements are sized against the pip weight a 20% quorum of a colour's cards actually demand (`PIP_WEIGHT_QUORUM`), so one `{B}{B}{B}` bomb doesn't demand 48 sources; and planning never targets earlier than turn 3 (`MIN_PLANNING_TURN`), since sizing to turn 1 asks for more sources than a 36-land base can hold. Both the target and the *achieved* on-curve probability are reported, so a shortfall is information rather than a failure.

## Mana audit (`src/manamap/pilot/mana_analysis.py`)

The same hypergeometric kit, pointed at a *finished* deck instead of a pool, and the one
magazine section with **no agent at all**: `manamap pilot mana-analysis <slug>` writes
`mana_analysis.json` deterministically. It classifies each land (basic / snow / mdfc /
fetch / tapped / untapped-dual / utility, reusing `ROLE_LAND_PATTERNS` deck-locally),
counts land and nonland sources per colour, computes pip share against source share, and
reports on-curve probability both from lands alone and with rocks and dorks.

**It counts copies, not decklist entries.** Basics live in `cards.json` as one entry with
`quantity: N`, so the artifact reports `lands.total` (copies — the real answer) beside
`lands.entries` (distinct cards). Counting entries once published "18 lands" for a 33-land
deck and understated every colour's sources across the whole fleet; `validate-issue` now
lints reader-facing copy that quotes the entry count as a land count.

## Synergy detection (`src/manamap/analysis/synergy.py`)

Synergies are **complementary** — cards that *complete* each other (blink finds ETB), NOT cards that do the same thing (that's "Find Similar", which uses embedding neighbors).

24 rules in `config.py` `SYNERGY_RULES`, each `(tag_A, tag_B, label)`, covering 29/33 tags. Examples: blink+etb, sacrifice+death_trigger, tokens+anthem, mill+graveyard_matters, storm+cost_reduction, aura+protection, counterspell+draw. Rules apply bidirectionally. Known combo partners (from `combo_graph.json`, the partners map) are excluded to surface NEW synergies. Top `SYNERGY_MAX_PARTNERS` (10) per card, ranked by rule count with ability-embedding cosine similarity as tiebreaker (falls back to color+type embeddings).

The graph is **format-agnostic by design** — filtering happens at consumption time (deck builder filters by legality + color identity; explore mode shows everything).

Shared helpers (`parse_tag_set`, `cosine_similarity`, `load_first_embeddings`) live in `analysis/common.py`.

## Power creep (`src/manamap/analysis/power_creep.py`)

Card B obsoletes Card A only if **all** hold:
1. Same supertype (Land and Unknown never compared)
2. Cosine similarity gate in ability embedding space — **tiered**: 1-tag cards need >= 0.98 (`OBSOLESCENCE_SINGLE_TAG_THRESHOLD`), 2+-tag cards >= 0.75 (`OBSOLESCENCE_SIMILARITY_THRESHOLD`). Keeps iconic single-tag upgrades (Doom Blade → Fatal Push, sim 0.999) while filtering false positives.
3. B.cmc <= A.cmc
4. B's color requirement same or easier (pip comparison)
5. B has all of A's mechanical tags (superset)
6. B has same or better power/toughness (creatures)
7. B has at least one concrete advantage
8. B printed later than A

Exclusions: cards with < `OBSOLESCENCE_MIN_TAGS` (1) tags, empty/NaN mana cost (augments, tokens), modifier stats (`+2`) parse to None. Up to `OBSOLESCENCE_MAX_REPLACEMENTS` (5) per card, sorted by similarity then advantage count. Batch matrix multiply per supertype group for performance.

## Region clustering (`src/manamap/analysis/cluster_regions.py`)

HDBSCAN names geographic-style regions on both 2D maps at two zoom levels:
- **L0 mega-regions**: `min_cluster_size=800`, `min_samples=50` → ~10–15 regions (zoomed out)
- **L1 sub-regions**: `min_cluster_size=100`, `min_samples=15` → ~40–120 regions (zoomed in)

Naming — Color+Type map: dominant color (>= 40%) + type (>= 30%), guild names for 2-color pairs (>= 50%), fallback to top tag. Abilities map: TF-IDF-like scoring (cluster tag freq / global freq), top 1–2 overrepresented tags, minimum presence threshold `REGION_MIN_TAG_PRESENCE=0.10`. Label dedup: tag suffixes (max 2 for L0, 3 for L1), then spatial direction (N/S/E/W).

**Index-alignment invariant**: `projection[i]` corresponds exactly to `cards.csv[i]` (maintained through embed → reduce). Tags are looked up by direct index, never by name (duplicate card names exist).
