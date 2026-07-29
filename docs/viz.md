# Visualization

Static frontend in `viz/` — no build tooling. **Two independent pages** that share a
directory and nothing else:

- **`index.html` — the card map.** Plotly.js 2.35.2 from CDN (`scattergl` WebGL), dark
  theme (#1a1a2e background, #c4a747 gold accents), styles in `css/mana-map.css`.
- **`deck.html` — the deck dossier.** No Plotly, no `mana-map.js`; the magazine's design
  tokens in `css/tokens.css` (ported from `pilot/design.py`) plus Google Fonts.

## Serving

```bash
python -m http.server 8000
# http://localhost:8000/viz/index.html               the card map
# http://localhost:8000/viz/deck.html?deck=heliod    a deck's dossier
```

**Must serve from the repo root**: the JS fetches `../data/<file>` relative to `viz/`. This mirrors the GitHub Pages deployment, which serves the repo as-is — `viz/` and `data/` must stay top-level siblings, and all fetch URLs must remain `../data/<name>`.

## Files

| File | Role |
|------|------|
| `viz/index.html` | Map shell: toolbar, plot div, detail panel, deck panel, script tags |
| `viz/css/mana-map.css` | Map + deck-builder styles, flat hex, no custom properties (~310 lines) |
| `viz/js/mana-map.js` | Explore mode (~1,330 lines). IIFE; exposes shared state as `window.MM` |
| `viz/js/deck-builder.js` | Deck builder (~1,370 lines). IIFE; exposes `window.DeckBuilder`; depends on `MM` |
| `viz/deck.html` | Dossier shell: masthead, deck picker, panel grid |
| `viz/css/tokens.css` | The magazine's design tokens in a dark register (~170 lines) |
| `viz/js/deck-view.js` | The dossier (~340 lines). IIFE; no globals exported, no `MM` dependency |

**Script order matters on the map page**: `mana-map.js` must load before `deck-builder.js` (deck-builder reads `MM.*` at load time). mana-map degrades gracefully if deck-builder is absent (all calls guarded). `deck.html` loads only `deck-view.js` and shares no code with the map.

## Cache busting

Manual `?v=N` query strings, per page: `index.html` on both JS files and `mana-map.css`; `deck.html` on `deck-view.js` and `tokens.css`. **Bump the version on the page you touched** before pushing — Pages/browser caches are aggressive.

For contrast, `manuals/magazine.css` is **content-addressed** (`?v=<sha8>` from the CSS text, in `pilot/design.py`), so a stylesheet change there obligates rebuilding every manual page but can never go stale. That is the pattern to copy if `viz/` ever outgrows manual bumps.

## Data paths

**Two registries, one per page** — the map's and the dossier's, deliberately disjoint:

- **Map** (`mana-map.js`): the `DATA` map at the top (built on `DATA_BASE = '../data/'`) holds all nine card-map artifacts. `MAP_CONFIGS` (per-map projection/embeddings/regions) and every fetch reference it; deck-builder consumes `MM.DATA.*`. Add new card-map files there, never as inline literals.
- **Dossier** (`deck-view.js`): `BASE = '../data/decks/'` plus a `FILES` map of per-deck artifact names. It fetches `data/decks/index.json` first — the manifest written by `manamap pilot build-index`, carrying the deck list and each deck's **passing** stack filenames, because a browser can list neither the deck directory nor `stacks/`. Never hardcode a deck list; add a deck and re-run `build-index`.

## window.MM API surface

Every member has a live caller (deck-builder.js, generated onclick handlers, or index.html) — exports without callers were trimmed 2026-07; don't re-add one without a consumer.

Getters: `allData`, `currentMap`, `obsolescence`.
Helpers: `escHtml`, `buildHoverTextMinimal`, `renderManaSymbols`, `closeDetail`, `removeFromSelection`, `bringToTop`, `selectByName`, `findSimilar`, `findSynergies`, `render`, `setStatus`, `setMode`.
Constants: `MAP_CONFIGS`, `DATA`, `EMBED_DIM`.
Async data loaders: `getEmbeddings()`, `getSynergyGraph()` — the deck builder awaits these instead of downloading its own copies of the two largest payloads (~17 MB + ~27 MB); both resolve to the shared cached instance.

## The deck dossier (`deck.html`)

Renders a deck's **committed pilot artifacts** and nothing else. Slug comes from
`?deck=<slug>` — the only URL state in the frontend.

| Panel | Artifact | Tier |
|---|---|---|
| Bracket Floor + its named driver | `bracket_report.json` | ◆ |
| Sources Say (pips vs sources, land classes, on-curve) | `mana_analysis.json` | ◆ |
| By the Numbers (meters, turn table, assumptions) | `goldfish_metrics.json` | ◆ |
| The Short List (ten, with source chips) | `considering.json` | ◆★ |
| Fetch Quests (collapsible per tutor) | `tutor_guide.json` | ★ |
| The Kill (case files, citations verbatim) | `stacks/*.json`, passing only | ✓ |
| The Builder's Record (slots, scores, runners-up) | `build_plan.json` | ◆ |

**Nothing is recomputed in the browser and nothing is hardcoded.** The manual renders these
same artifacts as ◆ reproducible evidence, so a second implementation that drifted would
quietly break the tier contract. A missing artifact means an absent panel, not an error —
only `hapatra` has a `build_plan.json` today, so six of seven dossiers show no builder
panel. `tests/test_pilot_deck_manifest.py` asserts the manifest matches the artifacts and
that every stack it lists is checker-passed.

Each issue's Back Page links to `../viz/deck.html?deck=<slug>`, and the dossier links back
to the issue, the map, and the newsstand. Before this the two products shared exactly one
link, one-way.

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
