# Visualization

Static frontend in `viz/` — no build tooling. **Two independent pages** that share a
directory and nothing else:

- **`index.html` — the card map.** Plotly.js 2.35.2 from CDN (`scattergl` WebGL) for three
  of the four modes, plus d3 v7 for The Walk, which uses canvas instead. Dark theme
  (#1a1a2e background, #c4a747 gold accents), styles in `css/mana-map.css`.
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
| `viz/js/drill.js` | Drill mode (~410 lines). IIFE; exposes `window.Drill`; depends on `MM` |
| `viz/js/force.js` | The Walk (~470 lines). Canvas + d3, **no Plotly**; exposes `window.Force` |
| `viz/js/deck-map.js` | Deck Lens (~490 lines). IIFE; exposes `window.DeckMap`; depends on `MM` |
| `viz/js/deck-builder.js` | Deck builder (~1,370 lines). IIFE; exposes `window.DeckBuilder`; depends on `MM` |
| `viz/deck.html` | Dossier shell: masthead, deck picker, panel grid |
| `viz/css/tokens.css` | The magazine's design tokens in a dark register (~170 lines) |
| `viz/js/deck-view.js` | The dossier (~340 lines). IIFE; no globals exported, no `MM` dependency |

**Script order matters on the map page**: `mana-map.js` must load before `deck-map.js` and `deck-builder.js` (both read `MM.*` at load time). mana-map degrades gracefully if either is absent — every call is guarded. `deck.html` loads only `deck-view.js` and shares no code with the map.

## The four map modes

`#modeSelect` switches between them and `MM.setMode` owns the transition. Build and Deck
Lens share one side panel (`#deckPanel`), so entering either exits the other.

| Mode | Panel | Overlay source |
|---|---|---|
| Explore | detail panel | — |
| Deck Lens | `#deckPanel` + detail panel | `window.DeckMap` |
| Build Deck | `#deckPanel` (detail hidden) | `window.DeckBuilder` |
| The Walk | `#deckPanel` (detail hidden) | **its own canvas** — Plotly is hidden entirely |

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

## Drill mode (`viz/js/drill.js`)

**Orthogonal to mode.** Explore / Deck Lens / Build decide what is *painted over* the map;
drill replaces the map's **coordinates**. It works from any mode and the base traces go
`visible: false` while it is active.

The world map is one PaCMAP layout of 34,322 cards at `n_neighbors=10` — the regime that
preserves global shape by compressing local shape. Drilling recomputes a layout for the
selected cards alone from the 128-d embeddings, so the structure the projection squashed
out becomes the whole view. Measured on a real region: 156 Aura cards occupy **0.3 × 0.7**
on the world map and **45.2 × 49.9** once re-mapped.

**Three entries**, all routed through `Drill.enter(indices, label)`:

| Trigger | Path |
|---|---|
| Box/lasso select over 8 cards | `plotly_selected` → `Drill.offer(...)` → a button in the bar |
| Region label click | raw click hit-tested against annotation anchors → `Drill.enterRegion(id)` → `regions_*.json` `membership` |
| Current filters | the `Drill ⤓` toolbar button → `Drill.enterFiltered()` |

Box-select **offers** rather than drills, because the same gesture already feeds the
8-card detail stack; hijacking it silently would be worse than a button. It is also the
only thing that reports how many cards the box actually caught — the handler used to
truncate to 8 and say nothing.

**The animation.** Points start at their world positions and relax toward the target
layout over 90 frames of stochastic stress majorization against 128-d chord distance
(`sqrt(2 - 2cos)`; embedding rows are L2-normalised, so the dot product *is* the cosine).
Seeding from world positions is what makes it read as a dive rather than a cut — you can
see which cards were already neighbours and which travel. `alpha` decays as `1 - t³`, and
the per-frame residual is the weight and bounce.

Frames are driven by `requestAnimationFrame` and pushed with **`Plotly.restyle`**, never
`react`: restyle preserves the axis range where react resets it (see
`tests/test_viz_camera.py`), and it is the only Plotly fast path in the codebase. The
whole subset is one trace with a per-point colour array so a frame is a *single* restyle —
splitting by category would multiply per-frame Plotly calls by the number of groups.

**`MAX_DRILL = 2000`**, and the cap is announced in the breadcrumb rather than applied
silently — *and sampled evenly rather than taken as a prefix*. `sampleEvenly(rows, cap)`
strides across the set, because `slice(0, N)` takes the first N rows in `cards.csv` order,
which is Scryfall's export order: a truncated drill of a 3,434-card region showed whichever
cards happened to be exported first. The breadcrumb was honest about the count and silent
about the bias. The Walk uses the same helper for its 500-node cap.

**The `Drill ⤓` button states its size and refuses over the cap.** It reads
`Drill 34,322 ⤓`, greyed, with no filters — because "re-map everything" is not a thing: a
2,000-card sample of the whole universe has no reason to cohere, and it flew in from all
over the map and settled into a multicoloured pile. Filter below the cap (or box-select, or
click a region label) and it goes live: `Drill 39 ⤓`. Clicking it while inert explains
itself in the status line rather than being a dead control. Measured: restyle on a 1,200-point `scattergl` trace runs ~32 ms median with the
full world still loaded.

**Contours and labels do not animate.** `histogram2dcontour` is main-thread SVG over the
whole subset, and region labels are annotations on a 150 ms debounce — both would stutter
the settle. They return once, together, at arrival. Contour levels are **not comparable
across drills**: the trace auto-bins to whatever extent it is handed.

**Hidden tabs.** `requestAnimationFrame` does not fire in a background tab, so switching
away mid-flight would freeze the points at meaningless intermediate positions forever —
the callback that would schedule the next frame never runs. `finishNow()` runs the
remaining relaxation without painting and lands on the settled layout; a `visibilitychange`
listener and a `document.hidden` check at entry both route to it.

### The honesty rule

**A drilled position is local.** The same card sits somewhere else on the world map, and
the two coordinate systems mean different things. There must be no state in which both are
on screen without the breadcrumb saying which you are looking at.

What that costs, concretely — everything anchored to world coordinates is suppressed while
drilling: region labels (`annotations: drilling ? [] : ...`), the search highlight, and the
status line's world count. The selection highlight is *not* suppressed but re-anchored:
`Drill.localPosition(idx)` returns the card's local position, or `null` for a card outside
the subset, and callers must drop it rather than defaulting to a world coordinate. That one
was found by eye in a screenshot after the source checks already passed — a gold ring, in
the one colour the map uses to mean *this is the card you are looking at*, pointing at
nothing.

`tests/test_viz_drill.py` covers the contract, the suppressions, the local-position
lookup, the announced truncation, and the hidden-tab fallback.

## Explore is an orientation lens, not a second workspace

Discovery is generative — you build something, and it responds. Explore was inspective: a
fixed picture behind filters, opening on 34,322 points and the words "34,322 cards shown",
where clicking added to an 8-card *list* rather than to a structure. Same app, two different
verbs, and the second one felt worse.

So Explore was given the job it is uniquely good at. `force.js` states in its own header
that the graph encodes **adjacency, not absolute position** — so "where does this sit in
card space" is precisely what it cannot answer. Entering Explore from a graph now calls
`MM.orientTo(rows, label, anchor)`: your cards light up gold, the card you were on gets a
white star, the other 34,000 dim to texture, and the status says what you are looking at.
Esc restores the whole atlas. It participates in the same
`getOverlayTraces` / `getDimmedIndices` / `dimsAll` contract as Deck Lens.

**Region labels now reject overlaps**, which `force.js` has done for node labels since the
graph shipped. The map emitted every label unconditionally, so the atlas was a pile of
colliding text while the graph never was — most of why one felt noisier than the other.

The collision test is evaluated **in pixels at position time**, not in `setAnnotations`.
Annotations carry world coordinates (`region.cx/cy`), and comparing those against label
widths in pixels is a units error that rejects almost everything — world coords span about
±40 while a label is 150 px wide. Doing it at position time also makes it zoom-responsive
for free: labels that collide zoomed out separate as you zoom in.

## Discovery — the front door (`viz/js/discovery.js`)

The map used to be where you arrived. Now the landing is **one card**: hover it, click a
relation, and the graph grows from there. `?card=<name>` and `?seed=<n>` make a landing
reproducible; `?mode=explore` goes straight to the atlas, which is what every existing
browser fixture now asks for.

**It is the same force engine as The Walk, with different chrome.**
`Force.enter(rows, label, {chrome: 'discovery'})` hands the side panel to Discovery and
otherwise reuses the physics, drag-and-fling, hover popup and card detail. A second
simulation for the landing would have been the duplicate-k-NN mistake this codebase has
already had to undo twice.

**Boot: 1.83 MB, against 18.4 MB to reach a first branch before.**

| artifact | gzipped | needed for |
|---|---|---|
| `viz_index.json` | 0.56 MB | pick a card, filter, resolve a name |
| `neighbours.bin` | 1.27 MB | branch — synchronously, with reasons |
| `projection_2d.json` | 2.90 MB | the atlas; upgrades card records behind the landing |
| `embeddings_ability.bin` | 15.54 MB | **not fetched on the discovery path at all** |

Nothing in discovery reads the embedding matrix, so it is never requested there — a
speculative prefetch was tried and removed, both because it was 16.8 MB spent on nothing
and because it showed up as contention that made two browser tests pass alone and fail in
the full run. A **seeded** walk (deck or region) still awaits it: `linkWithinFromTable`
only links cards whose precomputed top-12 are also in the set, which on a 97-card deck is
38 links instead of ~290 — a visibly sparser graph, caught by the browser suite.

**One relation mechanism, one behaviour.** The controls live in `buildCardDetailHtml`, so
Discover, The Walk, the explore accordion and the browse panel all offer the same thing, and
`MM.relate(row, relation)` always does the same thing with it: grow the graph from that card.
From Explore that means switching modes — the click carries you into the walk, seeded on the
card you clicked.

That dispatch used to fork. Explore opened a linear **browse set** instead, on the reasoning
that a scatter plot cannot grow. The reasoning was sound and the result was still wrong: one
control meant two things, so the same button rewarded you differently depending on where you
happened to be standing, and the atlas was the version that felt dead. The fix is not to
teach the scatter plot to grow — it is to treat Explore as a **launchpad**: you go there to
see where things sit, then click to start walking from one. `Discovery.show` seeds the graph,
`Discovery.focus` re-centres it if the card is already present, and `Force.branchByRow` opens
the relation. The **Keep** control moved into the same shared HTML for the same reason —
keeping a card you spotted in the atlas is the same act as keeping one you walked to.

Box-select still opens a browse set; that is a different question ("what did I just lasso?")
and keeps the arrow-key walk.

This replaced **Find Similar Cards** / **Find Synergies**, which were broken four ways at
once and untested: they took no card argument and read `selectedCards`, so they were silent
no-ops in Discover *and* the browse panel (both clear it), acted on the **wrong card** in The
Walk while drawing onto a hidden Plotly surface, and threw outright under `?renderer=canvas`
where `#plot` has no `.data`. Their highlight-trace machinery is gone with them.

**Synergy edges say why.** `neighbours.bin` v2 carries a uint8 reason code per synergy slot
plus the 24-entry vocabulary appended after the data, so branching stays synchronous and the
codebook never becomes a third fetch. Edges are inked by relation — deck gold, synergy violet,
obsolescence red, similarity cool blue — and synergy edges are labelled with their reason
("Sac + Death Trigger"), placed into the *same* collision set as node labels so a reason can
never sit on a card name, capped at 8 and dropped entirely past 60 nodes.

Those strings were already being computed and thrown away: the old Find Synergies wrote them
into a Plotly trace and then set `hoverinfo: 'none'`.

**When measuring label counts, wait for the layout to settle.** A cramped graph collides every
label, so `Force.edgeLabelCount` mid-settle reads ~0 and looks exactly like a broken feature.

**A click must survive a shaky hand.** `d3.drag`'s `clickDistance` defaults to **0**, so any
pointer movement between mousedown and mouseup makes d3 install a capture-phase suppressor that
eats the following `click` event. Measured on this page: 0px jitter delivered the click, **1px
and 3px swallowed it**. That is the whole of "some cards don't expand the first time, then work
if I click again" — the second click was just steadier. The drag now sets `clickDistance(6)`, a
tap tolerance: below it you meant to click, above it you meant to fling. The click handler also
falls back to the hovered node when the hit test misses, because the simulation keeps running
and a node can drift out from under the cursor between press and release.

**Names without hovering.** The graph places a bounded sample of card labels — priority to
hovered, pinned, the trail, then seeds — greedily, rejecting any that would collide. Labelling
all 500 nodes is a smear; labelling only the hovered one means the graph says nothing until
you touch it. The set thins when dense and fills in as you zoom, with no zoom logic of its own.
`Force.labelCount` exists because canvas text cannot be queried by a test.

**The hover card is bounded.** `positionPopup` clamps to the plot frame, but it used to
measure the popup the instant the `<img>` was inserted — before the network returned
anything — so the height read ~0, the bottom clamp had nothing to clamp, and a card hovered
near the foot of the page ran off it. The CSS now reserves the 488:680 card box so the height
is known before load, and the fallback is explicit.

**Three defects the single-seed landing exposed**, all fixed:

- **Every graph was a tree.** `branchFrom` skipped neighbours already present and only ever
  added parent→child edges. Invisible from a multi-seed start because `enter()` ran
  `linkWithin` first; from one seed it meant no cycles and no cross-links, so two
  near-duplicates reached down different branches sat far apart with nothing between them.
  Branching now links to cards already on the graph as well.
- **The panel followed the cursor.** `hovered || pinned` meant "click to open details"
  evaporated the moment the mouse moved. Now `pinned || hovered`.
- **A single seed was a 6 px dot.** The pinned card renders as real art — a DOM `<img>`
  over the canvas, *not* `ctx.drawImage`, because Scryfall's image endpoint redirects and
  the redirect chain refuses `crossOrigin="anonymous"` (verified: the load fails). Drawing
  it without that flag would taint the canvas and `getImageData` on `#forceCanvas` would
  start throwing, which a browser test depends on. Same DOM-over-canvas call as region
  labels, for the same class of reason.

**Relation counts are stated before the click**, because they are precomputed: "Similar 12
· Synergy 10 · Outclassed by 5". 23.6% of cards have nothing but similar, and a button that
turns out to do nothing reads as broken rather than as a fact about the card. Synergy is
exactly 10 partners for every card that has any, so the UI says it is a rule-based list
rather than a ranking.

### The tray, import, and the hand-off

**The camera belongs to the user, and the layout arrives finished.**

Two things made this feel clunky, and they had one shape: the graph kept re-framing itself.

- `enter` ran `fitToGraph` on a 550 ms timer, **fourteen times**, plus once more on
  `sim.on('end')`. Zooming while it settled was overwritten by the next fit — the reported
  "zooming zooms back out". Auto-fit is now a *suggestion*: `fitToGraph(animate, auto)`
  returns early once `userAdjusted` is set, which any real pan, zoom or drag does.
  `ev.sourceEvent` is what distinguishes a gesture from a programmatic transform.
  `Force.fit()` is an explicit request and always wins.
- The initial layout animated from scaled world coordinates, so a loaded deck appeared as a
  distorted smear and collapsed inward over several seconds with the user locked out.
  `enter` now **pre-settles** it: `sim.tick()` advances the simulation *without* dispatching
  tick events, so a few hundred synchronous ticks cost milliseconds and nothing draws until
  the layout is done. Measured: a 79-card deck arrives arranged in ~120 ms and does not move
  afterwards. The simulation is then left **stopped** — dragging and branching reheat it, so
  it is alive exactly when something is happening.

`alphaDecay` moved 0.015 → 0.08 as a consequence. The slow decay existed so the initial
layout could be watched arranging itself; once that stopped animating, all it bought was a
graph drifting under the cursor for eight seconds after every branch. Now ~1.3 s: new cards
fly out, find their place, and stop.

**Loading a checked-in deck.** The picker reads `data/decks/index.json`, and `loadDeck(slug)`
resolves that deck's `cards.json` against `viz_index` — so it needs neither the projection nor
Deck Lens. It differs from a pasted import in the way that matters: the manifest carries a
**known** commander, so it is ringed and centred rather than inferred from a `*CMDR*` marker.

**Brought versus found is the visual language.** Nodes carry `deck` and `commander` flags set
at `enter({deck: {rows, commander}})`; anything `branchFrom` adds is by construction neither.

| | radius | fill | ring | edges |
|---|---|---|---|---|
| commander | 9 | full | double gold | warm |
| deck card | 6 | full | thin white | **warm gold, 1.7px** |
| explored | 4.5 | 50% alpha | none | thin cool blue |

That is what lets a deck stay legible as a structure while you explore outward from it — and
labels follow the same priority, naming the commander and deck cards before anything you
wandered into. `Force.membership()` exposes the split for tests, since none of this is
queryable from canvas pixels.

**The graph drives the panel.** Clicking a card calls `Discovery.focus(row)`, so the panel
shows that card's art, its relation counts, and a Keep button that adds the card you actually
clicked. `focus` is deliberately not `show` — `show` reseeds the graph, and opening a card you
walked to must not discard the walk that reached it.

**The tray** is a deliberately light selected-set, separate from the graph: the graph is
where you are looking, the tray is what you are keeping. It is the fifth "set of cards"
idea in this codebase and the only one that exports.

**Import** parses a pasted Moxfield export with `viz/js/decklist.js`, resolves names
against `viz_index.json`, and seeds the graph with the whole list, commander pinned. It
deliberately does **not** touch Deck Lens: `deck-map.js` refuses any slug absent from the
CLI-built `data/decks/index.json`, and an imported deck has no slug and never will.
Measured on the tracked Edgar list — 136 entries, 129 unique cards, 0 unresolved, 26 ms.

Pinning uses `Force.pinCard`, not `focusCard`. `focusCard` *branches*, so importing a
129-card deck produced a 135-node graph — six cards the deck did not contain.

**Optimize is a brief, not a button.** There is no backend and this does not add one: the
pilot loop is 6–10 serially dependent LLM subagents costing ~330k–1.7M tokens, and a
static page cannot run Python. The tray emits a JSON brief (download + clipboard) naming
the cards and candidate commanders, which a human pastes into Claude Code where that loop
already works. The brief says so in its own `next_step` field.

## The canvas renderer (`?renderer=canvas`) — Phases 2–3 of the migration

`viz/js/render/canvas.js` draws the map instead of Plotly. Both renderers are live at once:
`?renderer=canvas` switches, so they can be compared on identical data. The Walk proved the
machinery on 500 nodes; this points it at 34,322.

**The layer format IS the trace format.** A layer is a Plotly-shaped
`{x, y, customdata, name, visible, mode, marker: {size, color, opacity, symbol, line}}`, so
`render()` builds one structure and hands it to whichever renderer is active. No adapter to
write now and delete later, and no second definition of what a layer is.

Measured on this data *before* it was written, because the decision rested on the numbers:

| | Plotly | canvas |
|---|---|---|
| `render()` | ~30 ms | **15 ms** |
| Box-select (22,161 caught) | **138 ms** per mousemove | **4.5 ms** |
| Hover pick | — | **~0 ms** |
| Draw 34,322 points | — | 7.8 ms batched · 16.9 ms per-point |
| Quadtree build | — | 23.5 ms, cached across renders |

Two decisions worth keeping:

- **Batch one path per colour.** 7.8 ms versus 16.9 ms issuing a fill per point — 128 fps
  against 59. The map already groups by colour/supertype/rarity, so the grouping is free.
- **`setLayers` draws synchronously; only pan/zoom coalesces through rAF.** A filter toggle
  is a discrete state change and the caller wants it now — and rAF does not fire in a hidden
  tab, so an rAF-only draw leaves the canvas blank until focus and leaves the browser tests
  unable to see anything.
- **The quadtree is cached** on a signature of the pickable layers. Rebuilding is 23.5 ms and
  `setLayers` runs on every filter and keystroke, most of which do not change the point set.
  Without the cache `render()` was 38 ms — slower than Plotly.

### Phase 3 — the four things Plotly still owned

Each was the last reason to keep a Plotly code path, and each came out better on the way
across rather than merely equal:

- **Region labels are now real DOM** (`.map-label` buttons in `.map-labels`). Plotly drew
  them as layout annotations, which meant a relayout to change one, a click hit-test written
  by hand against anchor positions, and no crossfade — the L0→L1 handoff *popped*, because
  the only way to fade was rebuilding an `rgba()` alpha on the 150 ms debounce. They are now
  a CSS `transition` and an ordinary `onclick`, positioned with a `translate()`.
- **Contours are `d3.contourDensity`**, computed in base-fit space and cached on the same
  quadtree signature. Plotly's `histogram2dcontour` auto-binned to whatever extent it was
  handed, so its levels were never comparable between two filter states; these are.
- **The legend is a positioned `<div>`** built from the same layer list that draws, so it
  cannot disagree with what is on screen.
- **Box-select is the quadtree** — shift-armed marquee, `pickRect`, 4.5 ms against Plotly's
  138 ms per mousemove.

Two things needed for correctness rather than parity:

- **Per-point opacity**, batched by `(colour, opacity)` bucket rather than `colour` alone.
  Deck Lens dims 34,000 points against ~100 with an opacity *array*; without this the canvas
  had no way to draw it and fell back to a scalar.
- **`updateLayerBy(flag, patch)`** — the `Plotly.restyle` fast path. Drill pushes ~90 frames
  of stress-majorization positions; rebuilding every layer per frame would have made the
  animation the slowest thing on the page.

**Camera moves apply instantly in a hidden tab.** `setCamera({animate: true})` runs a d3
transition, which is rAF-driven and therefore does not advance in a background tab — the
move silently never happened. That is the fourth rAF-throttling bug in this file's history
(`schedule()`, `ResizeObserver`, CSS transitions, and now transitions); a camera that
arrives without easing beats one that never arrives.

## Render cost — the rules that keep it snappy

The map draws 34,322 WebGL points. Four rules, each of which was violated and measured:

**Never build what nothing displays.** Every trace sets `hoverinfo: 'none'` and nothing
reads `trace.text`, but all of them were building hover strings anyway — ~34,000 `escHtml`
calls per render, four chained global regexes on three fields each. **37 ms of a 90 ms
render, for output that was discarded.** `buildHoverTextMinimal` is kept and exported for
when hover is turned on; call it from the hover callback for the point under the cursor,
never in bulk.

**One `Plotly.react` draws everything.** `react` replaces the trace list, so it dropped the
selection highlight and `updateSelectionHighlight()` added it straight back — an extra
`addTraces` of the entire selection per render, which with a 15,000-card browse is a full
rebuild on every pan, filter and panel open. `render()` now folds `buildSelectionTraces()`
into its own trace list.

**Scalar opacity, unless a mode genuinely needs per-point.** Per-point opacity means a
34,000-entry array per colour group plus Plotly's per-point WebGL path. The Deck Lens dims
*everything* and redraws its 99 on top, so one scalar is equivalent — it declares
`dimsAll()`. The deck builder dims a real subset (format-illegal, colour-identity
violations) with nothing over it and keeps the array.

**Do the work once per gesture.** Box-select used to build the 8-card stack, render the
panel and rebuild the highlight, then — for a big box — throw all of it away and do it
again as browse. One pass over the points, one destination.

Measured, same page, median of 7:

| | before | after |
|---|---|---|
| `render()`, nothing selected | 90 ms | **30 ms** |
| `render()`, 15,000-card browse | 128 ms | **36 ms** |
| Arrow press while browsing | 25 ms | **16 ms** |
| Plotly calls per render | react + add + delete | **react** |
| Plotly calls per arrow press | delete + add | **restyle** |

**What is still slow, and is not ours.** A shift-drag fires `plotly_selecting` on every
mousemove, and Plotly hit-tests all 34,322 scattergl points each time — **measured ~138 ms
per event** with only the seven base traces loaded. That is the dominant cost of box-select
and it is inside Plotly. Any large highlight trace left on the plot adds to it, which is
why the browse selection is one trace rather than one per colour. Deck Lens renders in the
low hundreds of ms because it draws 16 separate role traces; that is the price of the
legend doubling as the role budget, and it is a mode switch, not a per-frame cost.

## Data cache-busting

`MM.DATA` URLs carry `?v=DATA_VERSION`. **Bump it when a data artifact's schema changes** —
a new key, a renamed field, a changed shape — not for content refreshes, where serving a
slightly stale copy is harmless.

This exists because `membership` was added to `regions_*.json` and every browser that had
already loaded the map kept serving its cached copy, so drill-by-region found no membership
and disabled itself. It failed politely, which is what makes the class expensive: the code
was right and the bytes were old.

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

### Navigating with the arrows

Arrows mean "step", and what they step through depends on what is selected:

| Selection | `←` / `→` |
|---|---|
| One card | its **neighbourhood** — k nearest in 128-d, nearest-first |
| 2–8 cards | the accordion stack |
| A browse set | the ordered set |

**One card seeds a neighbourhood.** `enterNeighbourhood(row)` takes the k nearest by cosine
and reuses `browseSet` wholesale — counter, `‹ ›` buttons, `moveBrowseMarker`'s
single-restyle fast path, image preloading — adding one field, `anchor`. `Enter` re-anchors
to whatever you have walked to, so you can travel outward card by card. The anchor keeps its
own blue ring on the map, because otherwise nothing says whose neighbourhood you are in.

**The two orderings are opposites and the panel says which is showing.** A plain browse is
furthest-from-centroid ("least typical first"); a neighbourhood is nearest-first from its
anchor, and shows the cosine as you step.

**Similarity is not the displayed map.** `loadEmbeddings` used to read
`MAP_CONFIGS[currentMap].embeddings`, so on the colour/type map "similar" meant "same colour
and type" — a space measured at 3.05 of its 128 effective dimensions and 0.090 recall@10
against known functional equivalents, which is why *Doubling Season* returned arbitrary green
enchantments. Find Similar, the walk and drill now all read `SIMILARITY_EMBEDDINGS` (the
function space) whichever projection is on screen. The projection is a picture; similarity is
a question. Switching maps no longer drops the loaded array either — the old per-map keying
re-fetched 17 MB and gave the same card different answers depending on the view.

**One k-nearest, `MM.nearestTo(row, k, opts)`, and now genuinely one.** The header here used
to claim `cosineSimilarity` and the sort inside `findSimilarCards` had been consolidated into
it. They had not — that scan was still live, sorting all 34,322 rows to take 20, with
different filter semantics. Both are gone. `respectFilters` defaults to true so a
neighbourhood will not walk you into a supertype you have hidden; The Walk passes `false`,
because a graph you are branching through should not change shape when a toolbar toggle
flips. It also excludes by **name**, not just row: `cards.csv` carries 51 duplicate names, so
self-exclusion alone let a card return its own twin at cosine 1.0.

**The keyboard gate was broken.** Arrows sat behind `if (selectedCards.length === 0) return;`
and `enterBrowse` sets `selectedCards = []` — so the arrow *keys* were dead in browse mode
and only the on-screen buttons worked, while the panel's hint read "← → browse". Arrows are
now handled first and gated on their own terms.

### Hover: the card at the cursor

A floating image in `#plot`, 180 ms delay, flipping side near the edges.

**`plotly_hover` fires even though every trace sets `hoverinfo: 'none'`** — verified in a
browser before building on it: `'none'` suppresses the *label*, `'skip'` suppresses the
*event*. So the popup needs no `text` arrays and reintroduces none of the per-point work
that made Plotly's own hover cost 37 ms a render; it reads `MM.allData[i]` on demand.

`pointer-events: none` on the popup is essential — it sits under the cursor by construction,
so without it the card steals the hover from the point that summoned it and flickers.

The magazine has a card preview too (`design.py` `.card-pop`), but it is pure CSS anchored
to a static inline element. A point in a WebGL scatter is not an element, so only the look
transfers, not the mechanism.

**Aiming at a card's pixel does not guarantee that card.** `hovermode: 'closest'` over
34,322 points means a denser neighbour a pixel away wins — asking for Sol Ring's coordinates
returns Krark-Clan Ironworks. Tests assert the popup matches *what Plotly reported hovering*,
not what they aimed at.

### The card viewer

One card selected renders as a plain detail panel. **More than one and the list becomes
the panel**: an accordion where the open card's detail expands *inside the row you
clicked*.

The previous layout put the detail on top and the list underneath, so changing card meant
scrolling down past a whole card to reach the list, clicking, then scrolling back up to
look at what you picked — a round trip on every change, with up to eight cards in play.

Three things make it hold together, and each was a separate fix:

- **`scrollActiveRowIntoView()` lives in `updateViewerPanel`**, not in `bringToTop`, so
  every path reveals the open row — clicking, the header arrows, arrow keys, number keys,
  removing a card, and selecting a new one from the map. It was originally only on the
  click path, which left map-selection scrolling nowhere. The row lands 89px down, just
  under the sticky header. `.detail-inner` needs `position: relative` for the `offsetTop`
  arithmetic.
- **The header is sticky and bleeds across the panel padding** (`margin: -16px -16px`).
  `.detail-inner` has 16px padding, so a header sized to the content box leaves gutters
  either side where the scrolling list shows through beside it — measured at 16px left,
  22px right before the fix.
- **`cycleSelection(delta)` is shared** by the `‹ ›` header buttons and the arrow keys, so
  the two cannot drift into one wrapping and one clamping. It wraps in both directions:
  with at most eight cards, stopping at the end is more annoying than looping.

### Browse mode — selections too big for a list

Over `MAX_SELECTED` (8) cards, the panel switches to **arrows only** and holds the *whole*
selection with no cap. Only the card you are looking at is ever fetched, so the cost is one
Scryfall request per arrow press rather than one per card in the box.

It replaced a real bug. `plotly_selected` returns points **grouped by trace** — colour
groups in palette order (`G, R, Colorless, U, B, W, Multicolor`), then `cards.csv` row
order within each — and the handler took the first 8. Box a mixed cluster and you got eight
green cards in Scryfall dump order: not a sample of your selection, an artifact of how the
traces were built.

**Ordering** is descending distance from the selection's own centroid in the 128-d
embedding space, so you start on the least typical card in the box and walk inward to the
most representative. The centroid is **renormalised** before scoring — rows are
L2-normalised at export so a dot product is a cosine, but the *mean* of unit vectors is not
itself a unit vector, and skipping that step makes the ordering depend on how tightly
clustered the selection happens to be.

The order does real work. Browsing the `Blue Artifacts` L0 region (3,434 cards) opens on
Ob Nixilis, three Lilianas and Garruk — black planeswalkers sitting inside a region named
for blue artifacts — and ends on Mantle of Tides and Neurok Stealthsuit. That is a finding
about the clustering, surfaced by the first screen.

**On the map**: the whole selection lights gold at size 5, and the card you are on renders
at size 16 with a white ring. Static, not animated — one restyle per press, nothing to tear
down. `moveBrowseMarker()` restyles just the single-point trace (found by `_isBrowseCurrent`,
not by name) and falls back to a full rebuild only if it is missing; rebuilding the whole
highlight per press measured **197 ms** on 3,434 cards versus **23 ms** for the fast path.

Browse and the 8-card stack are **mutually exclusive** — clicking a single point clears
`browseSet`, because two "current card" markers on the plot would be two different claims
about what you are looking at.

Neighbouring cards' images are preloaded (`preloadNeighbourImages`), in both modes, because
each is a Scryfall round-trip and without it every arrow press showed a beat of empty grey —
most of what made the old panel feel slow to browse. Neighbours only: preloading all eight would
be eight requests for seven cards the reader may never open. The open card's image is
deliberately **not** `loading="lazy"` — it is the only card image ever rendered and it is
scrolled into view as it appears.
- Relations on every card: *similar* (12 nearest in 128D cosine), *synergy* (rule-based complements, carrying their reason), *outclassed by* — all precomputed in `neighbours.bin`, all growing the graph
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

## The Walk (`viz/js/force.js`) — the first thing here that is not Plotly

A fourth map mode, and the opening move of the renderer migration. Cards become nodes,
128-d cosine distance becomes link length, and a velocity-Verlet simulation gives the
graph weight: it settles, it wobbles, you can grab a card and fling it and the rest
follows. Click a card and its nearest neighbours in the full 34,322-card corpus are pulled
in, the simulation reheats, and the graph grows toward whatever you were curious about.
That is the walk. The path you took is drawn as a gold trail.

**Canvas, `d3-force`, `d3-zoom`, `d3-drag` — no Plotly at all.** It is built as a new mode
precisely because nothing can regress there: it proves the canvas renderer, the zoom
behaviour and the hit-test against real data before any of that goes under the map itself.

**Seeding** comes from `MM.selectedRows()` — browse set first, then the 8-card stack — so
every existing way of picking cards feeds it: box-select, a region, a deck. Relations no
longer seed a drill; they grow the graph instead, which is the same idea with physics.

With **nothing** selected it shows an empty state offering all seven decks and the largest
L1 regions, one click each. That routing lives inside `renderPanel` rather than at each
call site: an empty graph must never render as a `0 CARDS / 0 LINKS` scoreboard, and
putting the check at the one place that draws the panel means a new caller cannot
reintroduce the dead end.

### What the picture claims

Link length is the model's own 128-d cosine, not the PaCMAP projection, so two adjacent
cards really are alike *to the model* — a stronger claim than the world map makes. It is
still a 2-D embedding of a high-dimensional space: the layout satisfies link lengths
approximately and nothing more. **Read adjacency, not absolute position.** There are no
axes, deliberately.

### Five things that were not obvious

- **Seed jitter is load-bearing, not cosmetic.** d3 only assigns initial positions to nodes
  that lack `x`/`y`, so seeding every node at its world-map position seeds a degenerate
  cluster when the source region is degenerate — and some are. The White Sorceries filament
  is 187 cards spanning **0.1 × 0.0** on the world map, and without jitter the whole graph
  collapsed to a single point (bbox `1 × 0`, a blank canvas).
- **Fit on settle, not on a timer.** `alphaDecay: 0.015` gives an ~8 s settle. A fit at
  700 ms frames a graph that then grows straight out of the viewport. `sim.on('end')` does
  the authoritative fit; an interval keeps it framed on the way there.
- **One zoom behaviour.** Programmatic transforms must go through the same `d3.zoom`
  instance that is bound to the canvas, or its internal state desyncs and the next wheel
  event snaps the view back.
- **`ctx.arc` needs an explicit `moveTo`.** Without one the arc connects from whatever the
  current point was after the link strokes, and every node renders as a pac-man wedge.
- **Node radius divides by `transform.k`,** so a card is the same size on screen at any
  zoom. Otherwise fitting a tight cluster turns every node into a dinner plate.
- **The canvas must be sized in CSS, with percentages.** A `<canvas>` is a *replaced*
  element, so an absolutely-positioned one with `width: auto` uses its intrinsic size —
  the backing-store attribute — and `inset: 0` cannot stretch it. Setting
  `canvas.style.width` from JS instead decouples it from the parent: `enter()` resizes
  before the side panel opens, so the canvas kept the full-width size, overhung the 420px
  panel at `z-index: 10`, and silently swallowed every click on the deck menu. The panel
  looked perfect and was completely inert. A `ResizeObserver` is **not** a fix — RO
  callbacks are throttled in background tabs exactly like `rAF` and CSS transitions, so
  the overhang can outlive them. `width: 100%; height: 100%` follows the parent
  unconditionally; only the backing store is set from JS.
- **`New walk ↺` exists because restore made the walk a one-way door.** The deck menu only
  renders when the graph is empty, and since re-entry restores the graph, the first set you
  picked was the only set you could ever pick.

### Feel

`PHYSICS` at the top of the file, and the three sliders in the panel expose the ones worth
touching. `velocityDecay: 0.22` is friction — d3's default `0.4` settles fast and dead;
this keeps inertia so a flung node swings. `charge: -110` is repulsion, `linkScale: 190`
converts chord distance to pixels.

**The card renders in the walk's own panel.** force mode hides `#detailPanel`, so the old
"Open the card →" button pushed the card into an invisible element: nothing appeared, and
it then popped open when you left the Walk, which is worse than nothing. `force.js` now
calls `MM.buildCardDetailHtml` — the same builder Explore uses, so there is one card
renderer rather than two that drift.

**Leaving keeps the graph.** `exit()` stops the simulation and nothing else; re-entering
without a seed picks up where you left off, trail included. It deliberately does *not*
touch `canvas.style.display` — `#plot:not(.force-mode)` already hides the canvas, and an
inline hide set on exit survived re-entry, so the graph rebuilt correctly into a 0x0
hidden element: right node count, right status line, blank screen.

`MAX_NODES = 500` caps the live graph — the simulation and canvas both scale much further,
but past a few hundred nodes it is a hairball and the walk stops being legible. The cap is
announced in the panel, never silent.
