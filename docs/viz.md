# Visualization

Static frontend in `viz/` — no build tooling. Plotly.js 2.35.2 from CDN (`scattergl` WebGL). Dark theme (#1a1a2e background, #c4a747 gold accents).

## Serving

```bash
python -m http.server 8000
# open http://localhost:8000/viz/index.html
```

**Must serve from the repo root**: the JS fetches `../data/<file>` relative to `viz/index.html`. This mirrors the GitHub Pages deployment, which serves the repo as-is — `viz/` and `data/` must stay top-level siblings, and all fetch URLs must remain `../data/<name>`.

## Files

| File | Role |
|------|------|
| `viz/index.html` | Static shell: toolbar, plot div, detail panel, deck panel, script tags |
| `viz/css/mana-map.css` | All styles: explore + deck builder + synergy + obsolescence + responsive (<768px) |
| `viz/js/mana-map.js` | Explore mode (~1,300 lines). IIFE; exposes shared state as `window.MM` |
| `viz/js/deck-builder.js` | Deck builder (~1,400 lines). IIFE; exposes `window.DeckBuilder`; depends on `MM` |

**Script order matters**: `mana-map.js` must load before `deck-builder.js` (deck-builder reads `MM.*` at load time). mana-map degrades gracefully if deck-builder is absent (all calls guarded).

## Cache busting

Manual `?v=N` query strings in `index.html` on both JS files and the CSS link. **Bump the version whenever you change JS/CSS** before pushing — Pages/browser caches are aggressive.

## Data paths

All data URLs are centralized in the `DATA` map at the top of `mana-map.js` (built on `DATA_BASE = '../data/'`). `MAP_CONFIGS` (per-map projection/embeddings/regions) and every fetch reference it; deck-builder consumes `MM.DATA.*`. Add new data files there, never as inline literals.

## window.MM API surface

Every member has a live caller (deck-builder.js, generated onclick handlers, or index.html) — exports without callers were trimmed 2026-07; don't re-add one without a consumer.

Getters: `allData`, `currentMap`, `obsolescence`.
Helpers: `escHtml`, `buildHoverTextMinimal`, `renderManaSymbols`, `closeDetail`, `removeFromSelection`, `bringToTop`, `selectByName`, `findSimilar`, `findSynergies`, `render`, `setStatus`, `setMode`.
Constants: `MAP_CONFIGS`, `DATA`, `EMBED_DIM`.
Async data loaders: `getEmbeddings()`, `getSynergyGraph()` — the deck builder awaits these instead of downloading its own copies of the two largest payloads (17.5 MB + 27.8 MB); both resolve to the shared cached instance.

## Explore mode highlights

- Two maps (Color+Type / Abilities), projections + embeddings cached for instant switching
- Color by primary color / supertype / rarity; supertype filter toggles
- 4-tier search (exact → starts-with → includes → oracle text, capped 200)
- Multi-select up to 8 (Shift+click / Shift+drag box select); keyboard nav (arrows, 1–8, Delete, Escape, `/`)
- Find Similar: 20 nearest in 128D cosine. Find Synergies: complementary tag matches (magenta)
- Region labels as Plotly annotations with zoom-dependent L0/L1 crossfade; optional density contours ("Topo")
- Custom 2-finger pinch zoom on mobile (Plotly scattergl lacks it natively; `touch-action: none`)

## Deck builder highlights

- 8 formats; commander support (100-card singleton, autocomplete with 200ms debounce)
- 6-factor recommendation scoring: 35% embedding similarity (`Math.max(0, dot)`), 20% combo (proportional `min(count/3,1)`), 20% synergy (`min(matches/SYNERGY_CAP,1)`), 10% EDHREC, 5% curve fit, 10% keyword Jaccard
- Precomputes `deckNames`/`deckKw` Sets once per generate
- Mana base generator: greedy set cover (colors covered ×10 + basic-subtype bonus + EDHREC ×3 − ETB-tapped penalty); Command Tower auto-add
- Obsolescence warnings (amber) in recommendations + deck list via `MM.obsolescence`
- LocalStorage persistence (`manamap-deck` key); text export, commander first

## Known Plotly gotcha

`Plotly.relayout` triggers `plotly_relayout` events — use a guard flag to avoid infinite loops (see `_labelUpdateInFlight` in mana-map.js).

## Future options (deliberately not done)

ES-module migration / splitting the IIFEs, moving the ~17 inline styles in generated HTML into CSS, content-hash cache busting. Lint/format/CI intentionally not set up.
