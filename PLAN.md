# Mana Map — Deck Builder Feature Plan

## Goal

Add a human-in-the-loop deck builder to Mana Map. Users provide 1+ seed cards, select a format, and the algorithm recommends cards using 128-dim embedding similarity + Commander Spellbook combo data + EDHREC rank. Users accept/reject recommendations iteratively, then auto-generate an optimized mana base and export the final deck list.

## Decisions Made

- **128-dim embeddings**: Lazy-load `embeddings.bin` (17.2 MB Float32Array) on entering Build mode for accurate cosine similarity
- **Combo data**: Download Commander Spellbook combos → process into `combo_graph.json` (~2-4 MB)
- **File structure**: Split `viz/index.html` into `index.html` + `css/mana-map.css` + `js/mana-map.js` + `js/deck-builder.js`
- **Delivery**: 4 incremental PRs

---

## PR 1: Pipeline — Combo Data + Embeddings Binary

**Branch**: `feat/deck-builder-data`

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `download_combos.py` | Step 7: Paginate Commander Spellbook API → `data/combos_raw.json` | ~80 |
| `process_combos.py` | Step 8: Build `data/combo_graph.json` from raw combos | ~120 |
| `export_embeddings.py` | Step 9: Convert `embeddings.npy` → `data/embeddings.bin` (raw Float32) | ~30 |
| `test_combos.py` | Tests for combo processing | ~100 |

### Modified Files

**`config.py`** — Add constants:
```python
COMBOS_API_URL = "https://backend.commanderspellbook.com/variants/"
COMBOS_RAW_PATH = DATA_DIR / "combos_raw.json"
COMBOS_META_PATH = DATA_DIR / ".combos-meta.json"
COMBO_GRAPH_PATH = DATA_DIR / "combo_graph.json"
EMBEDDINGS_BIN_PATH = DATA_DIR / "embeddings.bin"
```

**`pipeline.py`** — Add Steps 7-9 after reduce.

**`.gitignore`** — Add `data/combos_raw.json` to ignored, add exceptions for `data/combo_graph.json` and `data/embeddings.bin`.

### `combo_graph.json` Format

```json
{
  "partners": {
    "Sol Ring": ["Dramatic Reversal", "Isochron Scepter", ...],
    "Exquisite Blood": ["Sanguine Bond", "Vito, Thorn of the Dusk Rose", ...],
    ...
  },
  "combos": [
    {
      "cards": ["Dramatic Reversal", "Isochron Scepter", "Sol Ring"],
      "produces": ["Infinite colorless mana", "Infinite storm count"],
      "ci": "U"
    },
    ...
  ]
}
```

- `partners`: Card name → list of combo partner names. O(1) lookup for "does card X combo with anything in my deck?"
- `combos`: Full combo details for tooltip display. Only combos where ALL cards exist in our 33,504-card dataset.

### `embeddings.bin` Format

Raw little-endian Float32, 33,504 × 128 = 4,288,512 floats = 17,154,048 bytes. No header. JS loads as:
```js
const buf = await fetch('data/embeddings.bin').then(r => r.arrayBuffer());
const embeddings = new Float32Array(buf); // index: card_i * 128 + dim_j
```

### `download_combos.py` Design

Follow `download.py` patterns:
- Use `requests.Session()` with `USER_AGENT`
- Paginate with `?format=json&limit=100&offset=N`
- Sidecar metadata (`.combos-meta.json`) for idempotent re-runs
- Rate limit: 200ms between requests (~2.5 min for ~750 pages)
- Save raw response list to `combos_raw.json`

### `process_combos.py` Design

1. Load `combos_raw.json` and `cards.csv`
2. Build a set of known card names from `cards.csv`
3. For each combo variant: extract card names from `uses` array, check all exist in our dataset
4. Build `partners` adjacency map (deduplicated) and `combos` detail list
5. Write `combo_graph.json` with `json.dump(separators=(',', ':'))`

### Verification

```bash
python download_combos.py    # Downloads ~750 pages, creates combos_raw.json
python process_combos.py     # Creates combo_graph.json (~2-4 MB)
python export_embeddings.py  # Creates embeddings.bin (17.2 MB)
python -c "import json; d=json.load(open('data/combo_graph.json')); print(len(d['partners']), 'cards with combos,', len(d['combos']), 'total combos')"
pytest test_combos.py -v
```

---

## PR 2: UI Shell + File Split + Seed Card Management

**Branch**: `feat/deck-builder-ui`

### File Split

Extract from the current monolithic `viz/index.html`:

| File | Contents |
|------|----------|
| `viz/index.html` | HTML structure only, `<link>` and `<script>` tags |
| `viz/css/mana-map.css` | All existing + new CSS |
| `viz/js/mana-map.js` | Existing explore mode JS (palettes, render, search, toggles, detail panel, pinch zoom) |
| `viz/js/deck-builder.js` | New deck builder JS (state, UI, algorithm — grows in PR 3-4) |

### Mode Toggle

Add to toolbar:
```html
<select id="modeSelect">
  <option value="explore">Explore</option>
  <option value="build">Build Deck</option>
</select>
```

Entering Build mode: shows deck builder panel, lazy-loads `embeddings.bin` + `combo_graph.json` (with loading spinner), dims format-illegal cards on the plot. Exiting: restores Explore mode, hides panel, removes overlays.

### Deck Builder Panel (Right Sidebar, 420px)

Replaces detail panel when in Build mode. Sections:

```
┌─────────────────────────────────────────┐
│ DECK BUILDER                      [×]   │
├─────────────────────────────────────────┤
│ Format: [Commander ▼]                   │
│ Commander: [____________] [Set]         │  ← only shown for Commander format
├─────────────────────────────────────────┤
│ SEED CARDS (3)                [Clear]   │
│  Lightning Bolt       {R}        [×]    │
│  Counterspell         {U}{U}     [×]    │
│  Sol Ring             {1}        [×]    │
│                                         │
│  Click cards on the map to add seeds    │
├─────────────────────────────────────────┤
│ TYPE DISTRIBUTION          [Defaults]   │
│  Creature    [24] ████████████          │
│  Instant     [10] ████                  │
│  Sorcery     [ 6] ███                   │
│  Enchantment [ 8] ████                  │
│  Artifact    [ 6] ███                   │
│  Planeswalker[ 1] █                     │
│  Land        [24] ████████████          │
│  Total: 79  Target: 99  Remaining: 20  │
├─────────────────────────────────────────┤
│ [▶ Generate Recommendations]            │
├─────────────────────────────────────────┤
│ (Recommendations appear here — PR 3)    │
├─────────────────────────────────────────┤
│ (Deck list appears here — PR 4)         │
└─────────────────────────────────────────┘
```

### Deck State (`js/deck-builder.js`)

```js
let deckState = null; // null when in Explore mode

function initDeckState() {
  return {
    format: 'commander',
    commander: null,           // allData index or null
    colorIdentity: new Set(),  // derived from commander or union of seeds
    seeds: [],                 // allData indices
    accepted: [],              // allData indices
    rejected: new Set(),       // allData indices
    recommendations: [],       // [{index, score, comboPartners}]
    distribution: { Creature: 30, Instant: 10, Sorcery: 7, Enchantment: 8, Artifact: 10, Planeswalker: 1, Land: 33, Battle: 0 },
    landSlots: [],             // allData indices
    embeddings: null,          // Float32Array (lazy-loaded)
    comboGraph: null,          // {partners, combos} (lazy-loaded)
    nameToIndex: null,         // Map<name, index>
  };
}
```

### Format Rules (hardcoded in `deck-builder.js`)

```js
const FORMAT_RULES = {
  standard:  { deckSize: 60, maxCopies: 4, sideboard: 15 },
  modern:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
  legacy:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
  vintage:   { deckSize: 60, maxCopies: 4, sideboard: 15 },
  commander: { deckSize: 100, maxCopies: 1, sideboard: 0 },
  pioneer:   { deckSize: 60, maxCopies: 4, sideboard: 15 },
  pauper:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
  historic:  { deckSize: 60, maxCopies: 4, sideboard: 15 },
};
```

Default type distributions per format stored similarly.

### Seed Card Interactions

- **In Build mode**: clicking a card on the plot adds it to seeds (instead of opening detail panel). Shift+click opens detail panel.
- **Seed overlay**: Gold star markers on the plot for seed cards (reuse `_isReference` trace pattern).
- **Remove**: Click × on a seed card in the panel list.
- **Format validation**: Warn (red border) if a seed card isn't legal in the selected format.
- **Commander search**: Autocomplete text input filtered to legendary creatures legal in Commander.

### Plot Changes in Build Mode

- Format-illegal cards dimmed to opacity 0.15
- Seed cards shown as gold stars (overlay trace)
- Color identity violations dimmed (for Commander — cards outside commander's color identity)

### Verification

1. `python -m http.server 8000` → open viz
2. Toggle to "Build Deck" mode — panel appears, loading spinner while fetching embeddings + combos
3. Select format "Commander" — commander input appears
4. Click cards on map — they appear in seed list with mana symbols
5. Remove a seed — card disappears from list, star removed from plot
6. Toggle back to "Explore" — panel hides, stars removed, detail panel works normally
7. Check file split: all existing Explore features still work identically

---

## PR 3: Recommendation Algorithm

**Branch**: `feat/deck-builder-recs`

### Scoring Function

For each candidate card `c` given the current deck (seeds + accepted):

```
score(c) = 0.50 × embedding_sim(c)
         + 0.25 × combo_bonus(c)
         + 0.15 × edhrec_score(c)
         + 0.10 × keyword_overlap(c)
```

**Components:**

1. **`embedding_sim(c)`** [0, 1]: Compute centroid of all deck card embeddings (average, L2-normalize). Score = `(dot(emb[c], centroid) + 1) / 2`. Captures thematic/mechanical similarity.

2. **`combo_bonus(c)`** {0, 1}: Binary — 1 if `comboGraph.partners[c.name]` contains any card in the deck, else 0. High weight (0.25) to strongly prefer combo-enabling cards.

3. **`edhrec_score(c)`** [0, 1]: `1 - (rank / maxRank)`. Cards without EDHREC rank get 0.3 (neutral). Biases toward proven/popular cards.

4. **`keyword_overlap(c)`** [0, 1]: Jaccard similarity between card's keywords and deck's keyword set.

### Candidate Filtering (before scoring)

1. Format legality (card's `f` field contains selected format)
2. Color identity subset of deck's color identity
3. Not already in deck (seeds + accepted + lands)
4. Not in rejected set
5. Supertype has unfilled slots in distribution
6. Copy limit not exceeded (basic lands exempt)

### Centroid Math

```js
function computeCentroid(indices, embeddings, dim) {
  const c = new Float32Array(dim);
  for (const idx of indices) {
    const off = idx * dim;
    for (let j = 0; j < dim; j++) c[j] += embeddings[off + j];
  }
  let norm = 0;
  for (let j = 0; j < dim; j++) norm += c[j] * c[j];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let j = 0; j < dim; j++) c[j] /= norm;
  return c;
}
```

Performance: 33,504 × 128 dot products ≈ 15ms. Total recommendation generation ≈ 35ms. No Web Worker needed.

### Recommendation UI

Below the "Generate Recommendations" button:

```
RECOMMENDATIONS (20)            [Accept All]
┌────────────────────────────────────────┐
│ ★ Guttersnipe         0.87   [✓] [×]  │
│   ⚡ Combos with: Dualcaster Mage     │
│ ★ Young Pyromancer    0.85   [✓] [×]  │
│ ★ Brainstorm          0.82   [✓] [×]  │
│ ...                                     │
└────────────────────────────────────────┘
```

- Score displayed as 0-1 value
- Combo indicator line if card combos with any deck card (shows partner name)
- [✓] Accept → moves card to accepted list, removes from recommendations
- [×] Reject → adds to rejected set, removes from recommendations
- [Accept All] → accepts all 20
- After accepting/rejecting, user clicks "Generate Recommendations" again for next batch
- Accepted cards shown as green circle overlay traces on plot
- Recommendation cards shown as orange diamond overlay traces on plot

### Iteration Loop

1. User adds seed cards
2. Clicks "Generate Recommendations" → 20 cards appear
3. Accepts some, rejects others
4. Accepted cards join the deck → centroid shifts
5. Clicks "Generate" again → new 20 based on updated deck
6. Repeat until type distribution is satisfied
7. Then proceed to mana base (PR 4)

### Verification

1. Add 3 seed cards (e.g., Lightning Bolt, Counterspell, Sol Ring)
2. Click "Generate Recommendations" — 20 cards appear with scores
3. Verify combo indicators appear for cards that combo with seeds
4. Accept 5 cards → they move to deck list area, green markers on plot
5. Reject 3 cards → they disappear, won't come back
6. Re-generate → new set of 20, different from before, centroid has shifted
7. Verify only format-legal, color-identity-valid cards appear
8. Verify type distribution filtering works (if creature slots full, no more creatures recommended)

---

## PR 4: Mana Base + Export + Polish

**Branch**: `feat/deck-builder-complete`

### Mana Base Algorithm

**Step 1 — Count pips** in all non-land deck cards:
```js
function countPips(deckCards) {
  const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  for (const card of deckCards) {
    if (card.s === 'Land') continue;
    const tokens = (card.mc || '').match(/\{([^}]+)\}/g) || [];
    for (const tok of tokens) {
      const inner = tok.slice(1, -1);
      if ('WUBRG'.includes(inner)) pips[inner]++;
      if (inner.includes('/')) {
        inner.split('/').forEach(p => { if ('WUBRG'.includes(p)) pips[p] += 0.5; });
      }
    }
  }
  return pips;
}
```

**Step 2 — Allocate land slots** proportionally to pip counts:
```
totalLandSlots = distribution.Land (e.g., 24 or 37)
utilitySlots = floor(totalLandSlots * 0.10)   // 2-4 utility lands
coloredSlots = totalLandSlots - utilitySlots
For each color C: slots[C] = round(coloredSlots * pips[C] / totalPips)
```

**Step 3 — Select lands** via greedy scoring:

For each legal land in the format + color identity:
```
score = colorsProvided.length × 10           // multi-color coverage
      + (hasBasicSubtype ? 5 : 0)            // fetchable (shock lands, triomes)
      + edhrecPopularity × 3                 // proven quality
      - (entersTapped ? 4 : 0)               // ETB tapped penalty
```

Greedy set cover: pick highest-scored land that covers the most-needed color, decrement that color's remaining slots, repeat.

**Step 4 — Fill remaining** with basic lands (Plains/Island/Swamp/Mountain/Forest).

**Step 5 — Commander extras**: Auto-include Command Tower (2+ colors), prioritize Arcane Signet.

### Land Detection Heuristics

Identify what colors a land produces by checking:
- Type line contains basic land subtypes (Plains→W, Island→U, Swamp→B, Mountain→R, Forest→G)
- Oracle text contains "Add {W}" / "Add {U}" / etc.
- Color identity field (fallback)

### Deck List Display

Below recommendations in the panel:

```
DECK LIST (58/60)              [Export] [Reset]
─────────────────────────────────────────
Creatures (22/24):
  Guttersnipe, Young Pyromancer, ...
Instants (10/10): ✓
  Lightning Bolt, Counterspell, ...
Lands (22/24):
  4× Mountain, 4× Island, 2× Steam Vents...
  [▶ Auto-Generate Mana Base]
─────────────────────────────────────────
  Mana Curve: 1●●● 2●●●●● 3●●●● 4●● 5● 6+●
  Colors: R(45%) U(35%) Colorless(20%)
```

- Type groups with progress bars (filled/target)
- Inline mana curve visualization (dot histogram by CMC)
- Color distribution percentage
- Individual card removal with × button

### Export

"Export" button copies to clipboard in standard text format:
```
1 Lightning Bolt
1 Counterspell
1 Sol Ring
4 Mountain
4 Island
2 Steam Vents
...
```

Compatible with MTGO, Arena, Moxfield, Archidekt imports.

### Polish

- Status bar updates: "42/60 cards · 18 remaining · Build mode"
- Keyboard: ESC exits Build mode (with confirmation if deck non-empty)
- Deck persistence: Save/restore `deckState` to `localStorage` so refreshing doesn't lose work
- Smooth transitions between Explore ↔ Build mode
- Mobile: Deck builder panel becomes full-width overlay on small screens (<768px)

### Verification

1. Build a complete 60-card Modern deck from 3 seed cards
2. Click "Auto-Generate Mana Base" → lands fill in proportionally to pip distribution
3. Verify dual lands are preferred over basics when available
4. Verify lands that enter tapped are deprioritized
5. Click "Export" → paste into Moxfield → all 60 cards recognized
6. Refresh page → deck persists via localStorage
7. Verify mana curve chart updates as cards are added/removed
8. Test Commander: 100 cards, singleton, color identity enforced by commander
9. Test Pauper: only commons appear in recommendations
10. Mobile: panel displays as full-width overlay, scrollable

---

## Files Changed (All PRs Combined)

### New Files (6)
- `download_combos.py` — Step 7 pipeline
- `process_combos.py` — Step 8 pipeline
- `export_embeddings.py` — Step 9 pipeline
- `test_combos.py` — Combo processing tests
- `viz/css/mana-map.css` — Extracted + new CSS (~350 lines)
- `viz/js/deck-builder.js` — Deck builder JS (~600-800 lines)

### Modified Files (5)
- `config.py` — 5 new path constants + combo API URL
- `pipeline.py` — Steps 7-9
- `.gitignore` — Track `combo_graph.json` + `embeddings.bin`, ignore `combos_raw.json`
- `viz/index.html` — Slimmed to HTML structure + `<link>`/`<script>` tags + mode toggle
- `viz/js/mana-map.js` — Extracted existing explore JS (refactored to export shared state/functions for deck-builder.js)

### New Data Files (tracked in git)
- `data/combo_graph.json` (~2-4 MB)
- `data/embeddings.bin` (17.2 MB)

---

## Performance Budget

| Operation | Time | When |
|-----------|------|------|
| Load `combo_graph.json` | ~500ms | Enter Build mode (once) |
| Load `embeddings.bin` | ~1-2s | Enter Build mode (once) |
| Build `nameToIndex` map | <5ms | Enter Build mode (once) |
| Compute centroid | <0.1ms | Each recommendation run |
| Score all 33,504 candidates | ~15ms | Each recommendation run |
| Filter + sort | ~15ms | Each recommendation run |
| **Total recommendation** | **~35ms** | Each "Generate" click |
| Generate mana base | ~10ms | Each "Auto-Generate" click |

Memory: ~24 MB additional (embeddings 17MB + combo graph 5MB + name index 2MB). Total page ~65 MB.
