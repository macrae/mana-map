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

`train.py` uses a **triplet margin loss** (margin 0.3): the layout task is "same colour,
same type", the supervision is exact, and a margin is the right shape for it.

**`train_ability.py`** (Function) uses a different objective, because the function task is
harder and a margin loss is the wrong instrument for it:

- **In-batch InfoNCE** (`INFONCE_TEMPERATURE = 0.05`), symmetric, over L2-normalised
  embeddings. A margin loss stops teaching the moment it is satisfied; a contrastive
  softmax keeps ranking the whole batch. There is no explicit negative — every other item
  in the batch is one.
- **Positives mined from `card_roles.json`, rarest shared role first**, so the pairs that
  teach most are the ones a common role would have drowned out. `ROLE_BODY_FALLBACK` is
  excluded, because it labels all 19,050 creatures and would make "creature" the lesson.
- **A fixed-weight text passthrough** makes similarity exactly
  `0.7·cos_learned + 0.3·cos_text`. The weight is structural, not learned, so the model
  *cannot* discard the frozen text it was built from — which is what the previous
  architecture allowed and what made it lose to its own input.

**Both:** batch 256, Adam lr 1e-3, 90/10 split, early stopping patience 5.

### The two spaces do different jobs

**Layout** (`train.py` → `embeddings.npy`) organises the map by colour and type. That is what
a *picture* wants, and it is all this model is asked for — it feeds `projection_2d.json` and
nothing else. It still converges to near-zero loss in a few epochs, and that is fine for a
task whose whole content is "same colour, same type".

**Function** (`train_ability.py` → `embeddings_ability.npy`) answers whether two cards do the
same job. Every "find similar" answer in the product comes from here, on either map.

### The function model, measured (`manamap eval-embeddings`, step 15)

Held-out `test` split of `data/eval/similarity_golden.json`:

| space | dim | effective dim | 1st→50th gap | r@10 | r@50 | median rank |
|---|---|---|---|---|---|---|
| layout | 128 | 3.20 | 0.0041 | 0.090 | 0.142 | 1651 |
| frozen MiniLM text (the input) | 384 | 50.41 | 0.1411 | 0.244 | 0.414 | 124 |
| **function** | 128 | **27.87** | 0.0315 | **0.245** | **0.455** | **78** |

Recall@10 ties the frozen text it is built from (0.245 vs 0.244 — noise across ~160
queries); the function space earns its place on depth instead, recall@50 +0.041 and median
rank 124 → 78.

**The two halves are complementary, and that is the load-bearing result:**

| | r@10 | r@50 | median rank |
|---|---|---|---|
| learned half alone (96d) | 0.136 | 0.341 | 172 |
| text half alone (32d) | 0.219 | 0.369 | 152 |
| combined (128d) | **0.245** | **0.455** | **78** |

The learned half is *worse alone* than the text half, yet the combination beats both — and
beats the full 384-dim frozen text at depth. This is why positives are deliberately **not**
gated on text similarity: selecting pairs the text already scores highly would make the
learned half a copy of the text half instead of a complement to it.

**Do not tune the mixing weight on this golden set.** Everything in W ∈ [0.15, 0.6] falls
inside noise at ~50 dev and ~160 test queries, and the two splits disagree about the
optimum — selecting on `test` picks 0.45, selecting on `dev` picks 0.15. The shipped 0.3
was chosen a priori and fitted to neither.

**Open:** neighbour spread is 0.0315 against a 0.05 target — the top-50 sit tighter than a
well-separated space would put them. `tests/test_embedding_quality.py` keeps that gate
failing as `xfail(strict=True)` rather than lowering the threshold to match the result.
Hard-negative mining is the obvious next lever, and needs a similarity ceiling: 39% of
cards have a text neighbour above 0.75, so unfiltered mining would label true synonyms as
negatives.

### The input text (`ingest/extract.py:build_embedding_text`)

`"{type}. Cost {mana_cost}. {P}/{T}. {oracle}. Keywords: …"` — **no card name.**

**The name is excluded on purpose.** Including it buys similarity off shared *words*
rather than shared function — *Rhystic Study* → *White Rhystic Study* scores 0.951 that
way, and *Sol Ring* → *Sisay's Ring*. It is also a large fraction of a short card: `Sol
Ring`'s entire text is eleven words, three of them its name. Measured on the held-out
split:

| text | r@10 | r@50 | median rank |
|---|---|---|---|
| name-led | 0.187 | 0.362 | 159 |
| no name | **0.248** | 0.407 | 129 |
| no name + cost + P/T (shipped) | 0.244 | **0.414** | **124** |

Dropping the name is the win. Cost and P/T are a wash in the *text* (the r@10 difference is
inside noise) and earn their place as structured features instead — they are in the string
because it costs nothing and helps the frozen-text fallback, not because they moved this table.

Note that effective dimensionality *fell* here, 81.0 → 50.4, while quality rose. Names added
variance, and variance is not quality: the participation ratio is a collapse **detector**, good
for telling 3 from 50, not a score to maximise. The evaluation is deliberately independent of every training signal (see
`data/eval/similarity_golden.json`), and its `test` split was written after the diagnosis, so
these numbers are not self-graded.

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

40 regex patterns in `config.py` `ROLE_PATTERNS`, plus type-derived `land:*`, `ramp:*` and the `threat:body` fallback — **53 emitted roles in 19 families**. Families: `ramp:*`, `draw:*`, `removal:*`, `tutor:*`, `buff:*`, `wincon:*`, `protection:*`, `land:*`, `doubler:*`, `payoff:*`, `threat:*`, plus `recursion`, `counterspell`, `sac-outlet`, `sac-cost`, `stax`, `hate:graveyard`, `value:etb`, `utility:activated`.

**Patterns match templating, not meaning, and that is where they fail.** Every hole found so far has been one card saying the same thing a different way. `removal:edict` demanded `each|target` adjacent to `opponent|player`, so Grave Pact's "each **other** player sacrifices" missed by two words. `wincon:drain` demanded a literal numeral, so Sanguine Bond's "loses **that much** life" and Exquisite Blood's "whenever **an** opponent loses life" both missed — the two halves of the format's most famous loop. `counterspell` demanded the literal "counter target spell", which blanked half of Sisay's interaction. Mondrak reverses Parallel Lives' word order ("tokens **would be created**") and was classified on its ward clause instead. When adding a pattern, test it against the *other* templates for the same effect.

Two rules keep it honest:

- **Lands never carry spell roles.** A land is evaluated only for land quality (fetch / tapped / untapped-dual / utility / basic / mdfc).
- **Coverage is published, not assumed.** The `meta` block reports `coverage` (89.5% of Commander-legal cards) *and* `specific_coverage` (73.2%, excluding cards carrying only the `threat:body` fallback), because labelling every creature a body and stopping would tell a slot filler nothing. `ROLE_COVERAGE_TARGET` / `ROLE_SPECIFIC_COVERAGE_TARGET` are regression floors asserted by the test suite. 100% is explicitly not the goal — the stragglers are genuinely miscellaneous one-off effects (Silence, Teferi Time Raveler, Chaos Warp), and looser regexes would buy coverage with false positives, which is worse than an honest null.

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

Shared helpers (`parse_tag_set`, `top_k_similar`, `load_first_embeddings`) live in `analysis/common.py`.

### Partners are ranked by playability, not similarity

Ranking is `(-score, -playability)`. Breaking the tie by **embedding similarity would be
backwards**: synergy is a *complementary* relation, so a similarity tiebreak surfaces cards
that resemble the anchor rather than cards that play with it.

The tiebreak decides almost everything, because a score tier is usually large — median 70
cards, p90 1,529. Measured over 4,000 cards:

| | median partner EDHREC rank | in the top 2,000 most-played |
|---|---|---|
| similarity tiebreak | 10,713 | 7.0% |
| **playability tiebreak** | **1,472** | **60.2%** |

Skullclamp went from recommending *Playable Delusionary Hydra* to *Yawgmoth, Thran Physician*.
`tests/test_synergy.py:test_synergy_partners_are_playable` is the gate.

Playability is `1 - log1p(rank)/log1p(EDHREC_RANK_SCALE)`, clipped strictly below 1 so it can
never outrank a full score step — a 2-rule match always beats a 1-rule match. Unranked cards
score 0. It is popularity, not quality, and will bias toward staples.

**Known limit:** a card whose top tier is genuinely small cannot be rescued by re-ranking.
Skullclamp's holds 3 cards, so its answer barely moves — that is coarseness in the 24 rules,
not in the ordering.

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

**L0 and L1 are two independent flat clusterings, not a tree.** L1 is a separate HDBSCAN run over the same 2D coordinates, not a subdivision of L0, and `parent` is assigned by nearest L0 centroid with **no containment test** — an L1 region can be parented to an L0 region it does not overlap. Any UI that presents them as a hierarchy is claiming more than the data supports.

**Membership is stored, and noise is a real answer.** `regions_*.json` carries `membership.l0` / `membership.l1`: positional arrays over `cards.csv` row order, `-1` for noise. 29% of cards on the default map are L0 noise and belong to no region at all — they stay `-1` rather than being snapped to a nearest centroid they were never clustered into. Regions also record `w`/`h` beside `span`, because `span = max(w, h)` alone cannot distinguish a filament from a blob.

**Index-alignment invariant**: `projection[i]` corresponds exactly to `cards.csv[i]` (maintained through embed → reduce), and `membership.l0[i]` describes the same card. Tags are looked up by direct index, never by name (duplicate card names exist).
