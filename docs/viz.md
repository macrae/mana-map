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
| `viz/js/deck-map.js` | Deck Lens (~480 lines). IIFE; exposes `window.DeckMap`; depends on `MM` |
| `viz/js/deck-builder.js` | Deck builder (~1,370 lines). IIFE; exposes `window.DeckBuilder`; depends on `MM` |
| `viz/deck.html` | Dossier shell: masthead, deck picker, panel grid |
| `viz/css/tokens.css` | The magazine's design tokens in a dark register (~170 lines) |
| `viz/js/deck-view.js` | The dossier (~340 lines). IIFE; no globals exported, no `MM` dependency |

**Script order matters on the map page**: `mana-map.js` must load before `deck-map.js` and `deck-builder.js` (both read `MM.*` at load time). mana-map degrades gracefully if either is absent — every call is guarded. `deck.html` loads only `deck-view.js` and shares no code with the map.

## The three map modes

`#modeSelect` switches between them and `MM.setMode` owns the transition. Build and Deck
Lens share one side panel (`#deckPanel`), so entering either exits the other.

| Mode | Panel | Overlay source |
|---|---|---|
| Explore | detail panel | — |
| Deck Lens | `#deckPanel` + detail panel | `window.DeckMap` |
| Build Deck | `#deckPanel` (detail hidden) | `window.DeckBuilder` |

**The overlay contract.** Any mode that paints over the base scatter implements exactly
two methods, and `render()` calls whichever mode is current:

- `getOverlayTraces()` → an array of Plotly traces drawn above the base scatter. Mark them
  `_isDeckOverlay: true`.
- `getDimmedIndices()` → a `Set` of row indices to render at 0.08 opacity, or `null` for
  no dimming.

Row indices are indices into `MM.allData`, which is `projection_2d.json`, which is
`cards.csv` row order. Both modes also expose `enter()` / `exit()`.

### Deck Lens

Overlays a published deck's 99 on the map: the deck lights up, the other ~34,200 cards
dim, and the deck's footprint in card space becomes visible — a storm deck is a tight
blob, a goodstuff pile is scattered. It reads the same tracked artifacts the magazine and
the dossier read, and computes nothing beyond a name→index lookup and a role histogram.

| Layer | Artifact | Rendering |
|---|---|---|
| The 99, one trace per role family | `cards.json` + `card_roles.json` | filled dots, legend doubles as role budget |
| Commander | `index.json` `commander` | large gold star |
| Verified lines | `stacks/*.json` (manifest-listed, passing only) | green edges between the cards each scenario names |
| The Short List | `considering.json` | open blue rings |
| Sideboard (off by default) | `cards.json` `is_sideboard` | open gold rings |

Three things worth knowing. **A card carries several roles**, so the lens paints it with
one — `FAMILY_PRIORITY` decides, and `threat` loses every tie because it sits on 19,032 of
34,322 cards. Cards with no role fall back to the map's supertype for lands only.
**Bars count copies, dots count distinct cards** — the panel says so out loud rather than
letting the two numbers disagree in silence. **A verified line naming fewer than two deck
cards draws no edge** but stays in the list, so the panel's count always agrees with the
manifest's `verified`.

`tests/test_viz_deck_lens.py` guards the three assumptions the browser cannot check for
itself: every deck card name resolves in `projection_2d.json`, every role family has a
colour, and `index.html` loads the script at a cache-bust matching its siblings.

## Cache busting

Manual `?v=N` query strings, per page: `index.html` on all three JS files and `mana-map.css`; `deck.html` on `deck-view.js` and `tokens.css`. **Bump the version on the page you touched** before pushing — Pages/browser caches are aggressive. On `index.html` the three script busts must move together; a test asserts it, because a mismatched pair is how `deck-map.js` ends up talking to a stale `mana-map.js`.

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
`?deck=<slug>`, the frontend's only URL state — now honoured by **both** pages:
`index.html?deck=<slug>` enters the Deck Lens with that deck loaded rather than dropping
the reader on an unfiltered map with a query string they cannot see.

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

The three surfaces now form a cycle. Each issue's Back Page links to
`../viz/deck.html?deck=<slug>`; the dossier links to the issue, the newsstand, and
`index.html?deck=<slug>`; the Lens links back to both the issue and the dossier. Before
the dossier shipped, the two products shared exactly one link, one-way.

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
