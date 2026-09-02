# Visualization

Static frontend in `viz/` — no build tooling. **Four pages in two families**, sharing a
directory and, between the families, nothing else:

- **`index.html` — the card map.** One renderer: `<canvas>` + d3 v7 throughout, the atlas
  via `js/render/canvas.js` and the graph modes via `js/force.js`. Plotly is gone. Dark theme
  (#1a1a2e background, #c4a747 gold accents), styles in `css/mana-map.css`.
- **`workbench.html` — the landing page.** Every deck, racked by whether it is sleeved,
  waiting on cardboard, on the bench or history; or one fleet table with five sorts. Reads
  every deck's `info.json`.
- **`deck.html` — the deck dossier.** One deck, one screen.
- **`branch.html` — the branch workbench.** One candidate 99: the proposal, the verdict,
  the measured table with each row's definition, reward/risk/cost, the bill.

The last three share `css/tokens.css` (ported from `pilot/design.py`, the legacy page's
stylesheet) plus Google Fonts, load no `mana-map.js`, and export no globals except test
hooks. **They compute nothing**: every figure is composed by the Python and read out of a
committed artifact, which is what lets them work on a static host.


## `spaces.html` — the embedding-space reference

The fifth page, and the only one that is a document rather than a tool. It
answers three questions the atlas raises and never explains: **where each
embedding space comes from**, **what its metrics mean**, and **which one to ask**.

It renders `data/eval/space_projections.json` (`manamap project-spaces`) as five
side-by-side scatters — identical PaCMAP settings, sample and seed, so any
difference is the space rather than the projection — colourable by colour
identity, card type or EDHREC tribe. Those three are deliberately facts **none of
the spaces optimised directly**, which makes the question "did this structure
emerge" rather than "was it supplied".

**Its numbers are a SNAPSHOT and the page says so.** There is no metrics artifact
to read — `eval-embeddings` prints a report and writes nothing — so the tables
live in `spaces-view.js` beside the command that reproduces them and the date
they were taken. When a metrics artifact exists, the page should read it instead;
a test asserts the page names the command.

No `window.MM`: no atlas, no card index, no deck file, the same way
`workbench.js` and `deck-view.js` stand alone.

## Serving

```bash
python -m http.server 8000
# http://localhost:8000/viz/workbench.html                       the landing page
# http://localhost:8000/viz/index.html                           the card map
# http://localhost:8000/viz/deck.html?deck=heliod                a deck's dossier
# http://localhost:8000/viz/branch.html?deck=X&branch=Y          a candidate 99
```

**Must serve from the repo root**: the JS fetches `../data/<file>` relative to `viz/`. This mirrors the GitHub Pages deployment, which serves the repo as-is — `viz/` and `data/` must stay top-level siblings, and all fetch URLs must remain `../data/<name>`.

## Files

| File | Role |
|------|------|
| `viz/index.html` | Map shell: toolbar, plot div, detail panel, deck panel, script tags |
| `viz/css/mana-map.css` | Map + panel styles, flat hex, no custom properties (~520 lines) |
| `viz/js/mana-map.js` | Explore mode (~3,200 lines). IIFE; exposes shared state as `window.MM` |
| `viz/js/drill.js` | Drill mode (~430 lines). IIFE; exposes `window.Drill`; depends on `MM` |
| `viz/js/stage.js` | Shared canvas primitives (~260 lines). Surface, camera, labels, typed edges |
| `viz/js/session.js` | Focus, **library**, commander (~560 lines). One answer each; force registers as its graph provider |
| `viz/js/force.js` | The graph engine (~1,430 lines). Canvas + d3-force; exposes `window.Force` |
| `viz/js/discovery.js` | Discover — the front door (~1,180 lines). Landing card, relations, library, import, seeding from named cards, `brief()` |
| `viz/js/render/canvas.js` | The map renderer (~1,150 lines). The ONLY renderer; owns the aura + ambient drift |
| `viz/js/decklist.js` | Moxfield paste parser (~90 lines). Fixture-locked to the Python parser |
| `viz/js/build.js` | Build (~1,780 lines). Deck Lens + Build Deck merged; exposes `window.Build` |
| `viz/deck.html` | Dossier shell: masthead, deck picker, panel grid |
| `viz/css/tokens.css` | The design tokens (from `pilot/design.py`) in a dark register (~810 lines). Shared by `deck.html`, `workbench.html` AND `branch.html` |
| `viz/js/deck-view.js` | The dossier (~1,580 lines). IIFE; exposes `window.Deck` for the server verbs, no `MM` dependency |
| `viz/workbench.html` | **The landing page**: racks + fleet table over every deck's `info.json` |
| `viz/js/workbench.js` | The landing page (~520 lines). IIFE; no globals, no `MM` dependency — same shape as `deck-view.js` |
| `viz/branch.html` | Branch shell: the objective mount and the panel grid |
| `viz/js/branch-view.js` | The branch workbench (~590 lines). IIFE; exposes `window.Branch` for the browser suite |
| `viz/js/shell.js` | The library drawer (~660 lines), mounted on every page; `Shell.cardImageUrl` is the name-only art helper |
| `viz/js/api.js` | The local-server probe (~105 lines). `Api.ready` is false on a static host and every verb degrades to a named command |

**Script order matters on the map page**: `stage.js` and `session.js` load first, then `mana-map.js` before `build.js` (which reads `MM.*` at load time). mana-map degrades gracefully if either is absent — every call is guarded. `deck.html`, `workbench.html` and `branch.html` share `shell.js`, `session.js` and
`api.js` with each other and no code at all with the map.

## The three map modes

`#modeSelect` switches between them and `MM.setMode` owns the transition. **Discover is
the front door** — `viz/index.html` opens on one random card, not the 34K scatter;
`?mode=explore` asks for the atlas and `?deck=<slug>` lands in Build.

| Mode | Panel | Surface |
|---|---|---|
| **Discover** | `Discovery.render` owns the panel | the force graph — one card, grow it by clicking relations |
| **Explore** | detail panel | the atlas (`render/canvas.js`), live-lit with whatever you hold |
| **Build** | `#deckPanel` + detail panel | `window.Build` — the force graph by default, the atlas overlay by toggle |

Drill is **orthogonal** to all three and is documented in its own section below.

Two mode transitions carry rules that were each learned by breaking them. **Leaving Build
calls `Force.newWalk(true)`** so Discover gets a clean landing card instead of someone
else's 97-card deck — but **Explore is exempt**, because it is a lens over whatever graph
is current and clearing on the way there would empty the thing it exists to show.
`currentMode` is assigned before the `exit()` calls in `setMode`, which is what lets
`Build.exit` tell those two destinations apart. And **the panel belongs to the mode, not
the engine**: `Force.renderPanel` asks `MM.mode`, because one force engine now has two
owners and "Discovery always owns the panel" repainted Build's roles and curve with
Discover's landing controls on every reheat.

Formerly four: **The Walk** was Discover with different chrome (four `chrome ===` reads =
two behaviours and a status string) and is deleted, its panel keepers folded into
`Discovery.render`; **Deck Lens** and Build Deck merged into `viz/js/build.js`. Deleting a
mode can strand a capability without deleting it — `Force.seedFrom` (box-select → graph)
was reachable only by entering The Walk with a selection live, and survived as a working,
callerless function. `MM.growFromBrowse` is its caller now. **When you delete an entry
point, grep for what only it could reach.**

**The overlay contract.** Any mode that paints over the base scatter implements exactly
two methods, and `render()` calls whichever mode is current:

- `getOverlayTraces()` → an array of layers drawn above the base scatter. Mark them
  `_isDeckOverlay: true`.
- `getDimmedIndices()` → a `Set` of row indices to render at 0.08 opacity, or `null` for
  no dimming.

Row indices are indices into `MM.allData`, which is `projection_2d.json`, which is
`cards.csv` row order. Implemented by `build.js` and `drill.js` (Drill's `getDimmedIndices`
returns `null` — it re-lays-out a subset rather than dimming one), with `mana-map.js`
supplying the no-overlay default. Both overlay modes also expose `enter()` / `exit()`.

### Build's map view (formerly Deck Lens)

Overlays a published deck's 99 on the map: the deck lights up, the other ~34,800 cards
dim, and the deck's footprint in card space becomes visible — a storm deck is a tight
blob, a goodstuff pile is scattered. It reads the same tracked artifacts the deck page and
the dossier read, and computes nothing beyond a name→index lookup and a role histogram.

| Layer | Artifact | Rendering |
|---|---|---|
| The 99, one trace per role family | `cards.json` + `card_roles.json` | filled dots, legend doubles as role budget |
| Commander | `index.json` `commander` | large gold star |
| Verified lines | `stacks/*.json` (manifest-listed, passing only) | green edges between the cards each scenario names |
| The Short List | `considering.json` | open blue rings |

Three things worth knowing. **A card carries several roles**, so the lens paints it with
one — `FAMILY_PRIORITY` decides, and `threat` loses every tie because it sits on 19,032 of
34,890 cards. Cards with no role fall back to the map's supertype for lands only.
**Bars count copies, dots count distinct cards** — the panel says so out loud rather than
letting the two numbers disagree in silence. **A verified line naming fewer than two deck
cards draws no edge** but stays in the list, so the panel's count always agrees with the
manifest's `verified`.

`tests/test_viz_deck_lens.py` guards the three assumptions the browser cannot check for
itself: every deck card name resolves in `projection_2d.json`, every role family has a
colour, and `index.html` loads the script at a cache-bust matching its siblings.

#### An open line explains itself

Clicking a verified line spotlights its cards **and prints its prose**. `buildEdges` used
to keep a stack's `title` and drop the document it came from — while `loadDeck` was
already fetching that whole document, resolution and all. The explanation was fetched,
parsed, held in memory and thrown away.

Three sources, each keeping its own slot and attribution rather than being blended:

| slot | source | fleet coverage |
|---|---|---|
| the intro | `manual_prose.json`'s `combo_lines[<stack id>]` — authored, gated on `stacks:passing` | 47 of 50 |
| **The answer** | `resolution.answer` — only where the scenario asks something sharp enough | 4 of 50 |
| **Where it ends** | `resolution.final_state.summary` | **50 of 50** |

Measured across the fleet before shipping: **zero published lines have no prose**, so the
panel never renders an empty block — a fact about the artifacts, not a hope about them.

**Nothing is derived, summarised or truncated.** A checker read these words; re-wording
them in the browser would put a ✓ over prose no checker saw, which is the same mistake as
editing a resolution's step text to fix a stale cross-reference.
`test_the_prose_is_the_artifact_verbatim` was proven to fire by truncating a summary to
120 characters — precisely the "helpful" regression it exists to catch.

Only the OPEN line renders prose: summaries run a median of 838 characters and up to
4,337, and printing all of them turns the list you are choosing from into a wall. The
block caps at 340px and fades at the bottom, and **the fade is a claim that must be true
in both directions** — it comes off at the end, and it comes off for a block shorter than
the cap, which never fires a scroll event and so cannot be settled by the scroll handler
(goblin-storm 002 is 877 characters). A fade over the end of a complete text is a lie
about there being more.

#### Verified edges point, and say what they carry

A verified line becomes a **clique** over the cards its stack names, and a clique has no
arrows: `{source, target}` is whichever order the pair was built in, and `findLink`
matches either way. Drawing an arrow on that would be array order wearing a claim.

`engine.json` is the one artifact that knows. Each line is `from → to` across two of
eight stages with a `carries` noun, written by an engineer and attacked by `engine-critic`.
Build fetches it (gated on the manifest's `has.engine`, because a browser cannot stat),
builds a card→stage lookup, and **orients** each pair: `a → b` when `a` sits in the `from`
stage and `b` in the `to` stage.

A pair that does not span the two stages gets **no arrowhead** — a clique includes pairs
sitting wholly inside one stage, and for those the direction genuinely is not known. On
ur-dragon that is **5 directed edges of 196**: the arrow is the exception, and earned.

**A stack can carry two lines.** ur-dragon's 002 is cited twice, for `bodies` and for
`triggers`, and `.find()` would have silently dropped half of what that board proves. All
matching lines are read, agreeing nouns are joined (`bodies · triggers`), and if two lines
citing one stack DISAGREED on direction no arrowhead is drawn at all — a pair pointing two
ways is a pair whose direction is not a fact. Arrowheads live in `Stage.drawEdges` behind
an `e.dir` flag, so the atlas can draw the same edge and mean the same thing by it.

Deliberately out of scope: parsing magnitudes (damage, mana, copies) out of
`final_state.summary` prose. That is the string-matching this repo keeps rejecting; a
magnitude should be an authored structural field on the engine line, which is a question
for the engineer's charter.

#### A bar is a control

Clicking a curve segment or a role row spotlights exactly those cards. The colours already
agreed — `renderManaCurve`, the role bars and the map scatter all read
`MM.GROUPINGS[MM.grouping]`'s `order` and `palette` — so what was missing was only
interaction. It routes exactly like `applyLine` and for the same reason: **Build defaults
to the GRAPH**, and a handler that only moved the map spends its time restyling a
`display:none` canvas, which reads as the click doing nothing. The two surfaces answer at
their own scale — the graph spotlights this deck's cards in the group, the map spotlights
the group across the atlas and composes with the deck lens through the legend's existing
`spotlightFor(g)` predicate.

A group and a line are two answers to "show me", so taking one puts the other down.

**The bug this exposed**: node fill was dimmed and then `globalAlpha` reset to 1 before the
rim was stroked, so a spotlight left 96 bright outlines and the picture never actually
dimmed. `Force.setLine` had shipped with the same defect. One alpha now covers both.

## Drill mode (`viz/js/drill.js`)

**Orthogonal to mode.** Explore and Build decide what is *painted over* the map;
drill replaces the map's **coordinates**. It works from any mode and the base traces go
`visible: false` while it is active.

The world map is one PaCMAP layout of 34,890 cards at `n_neighbors=10` — the regime that
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
only thing that reports how many cards the box actually caught. Truncating to 8 without
saying so is silent data loss.

**The animation.** Points start at their world positions and relax toward the target
layout over 90 frames of stochastic stress majorization against 128-d chord distance
(`sqrt(2 - 2cos)`; embedding rows are L2-normalised, so the dot product *is* the cosine).
Seeding from world positions is what makes it read as a dive rather than a cut — you can
see which cards were already neighbours and which travel. `alpha` decays as `1 - t³`, and
the per-frame residual is the weight and bounce.

Frames are driven by `requestAnimationFrame` and pushed with **`updateLayerBy('_isDrill',
…)`**, never `setLayers`: it moves one layer's points and leaves the other 34,890 alone.
(Under Plotly this was `restyle` rather than `react`, for the same reason plus one that no
longer applies — `react` also reset the axis range.) The whole subset is one layer with a
per-point colour array so a frame is a *single* update; splitting by category would multiply
per-frame calls by the number of groups.

**`MAX_DRILL = 2000`**, and the cap is announced in the breadcrumb rather than applied
silently — *and sampled evenly rather than taken as a prefix*. `sampleEvenly(rows, cap)`
strides across the set, because `slice(0, N)` takes the first N rows in `cards.csv` order,
which is Scryfall's export order: a truncated drill of a 3,434-card region showed whichever
cards happened to be exported first. The breadcrumb was honest about the count and silent
about the bias. The graph uses the same helper for its 500-node cap.

**The `Drill ⤓` button states its size and refuses over the cap.** It reads
`Drill 34,890 ⤓`, greyed, with no filters — because "re-map everything" is not a thing: a
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
fixed picture behind filters, opening on 34,890 points and the words "34,890 cards shown",
where clicking added to an 8-card *list* rather than to a structure. Same app, two different
verbs, and the second one felt worse.

So Explore was given the job it is uniquely good at. `force.js` states in its own header
that the graph encodes **adjacency, not absolute position** — so "where does this sit in
card space" is precisely what it cannot answer. `MM.orientTo(rows, label, anchor)` lights
your cards gold, gives the card you were on a white star, dims the other 34,000 to texture,
and says in the status what you are looking at. Esc restores the whole atlas. It
participates in the same `getOverlayTraces` / `getDimmedIndices` / `dimsAll` contract as
Build.

**Arriving is not asking for it.** `setMode('explore')` must NOT auto-orient on a non-empty
library. Doing so — `if (Session.size()) orientTo(null, 'your walk')` — means walking three
cards in Discover and switching opens the atlas with almost all of itself at 8% alpha and the camera
somewhere else. The lens is right; as an *entry* state it meant the atlas almost never got
to be the atlas, and the dimming read as a rendering fault rather than as a lens. Entry now
clears the lens, the region focus, the legend focus and the selection, and refits. Every
other `orientTo` caller stays: the lens re-engages the moment you ask from inside Explore,
which is what `MM.relate` does.

**Region labels now reject overlaps**, which `force.js` has done for node labels since the
graph shipped. The map emitted every label unconditionally, so the atlas was a pile of
colliding text while the graph never was — most of why one felt noisier than the other.

The collision test is evaluated **in pixels at position time**, not in `setAnnotations`.
Annotations carry world coordinates (`region.cx/cy`), and comparing those against label
widths in pixels is a units error that rejects almost everything — world coords span about
±40 while a label is 150 px wide. Doing it at position time also makes it zoom-responsive
for free: labels that collide zoomed out separate as you zoom in.

## Discovery — the front door (`viz/js/discovery.js`)

The landing is **one card**, not the atlas: hover it, click a
relation, and the graph grows from there. `?card=<name>` and `?seed=<n>` make a landing
reproducible; `?mode=explore` goes straight to the atlas, which is what every existing
browser fixture now asks for.

**It is the same force engine Build seeds.**
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

### Starting a walk from cards you name

`?cards=Sol+Ring%0ALightning+Bolt`, or a textarea on the panel. Two forms have to work —
`Zur the Enchanter`, and `1) zur, 2) sol ring` — and they need different splitting rules.

**A COMMA CANNOT BE A SEPARATOR.** 3,222 of 34,890 card names contain one (9.2%), and they
are overwhelmingly the legendary creatures somebody would actually seed a walk with:
*Miirym, Sentinel Wyrm*. Splitting on commas turns the commonest input into two cards that
do not exist, and it fails **silently** — the graph comes up short with nothing said.

So the **enumeration** is the separator. A `1)` / `2.` / `3:` marker splits an item only
where an item can BEGIN: at the start, after a newline, or after a comma or semicolon.
That last clause is the whole trick — it makes `1) Miirym, Sentinel Wyrm, 2) Sol Ring` two
cards while leaving `Miirym, Sentinel Wyrm` one. Requiring a boundary also protects the
**eight corpus names carrying a marker inside them** — `Vault 87: Forced Evolution` and its
five Fallout siblings, which a naive `/\d+[).:]/` cuts in half.

The normalised, one-per-line text then goes to **`Decklist.parse`, never a second name
reader**. That parser is fixture-locked in parity with `pilot/fetch_deck.py` and already
handles quantity prefixes, `*CMDR*` and printing suffixes. Normalising in FRONT of it costs
nothing; forking it would put a third decklist parser in a repo that has twice paid for
having two. Unresolved names are **reported**, never dropped — a typo that quietly yields a
one-card walk is indistinguishable from the feature not working.

**`Discovery.seedFromRows(rows, label, opts)`** is the shared path, and it exists because
writing a fifth caller by hand is exactly how the rule gets broken again:

> **Growing must never be able to delete.** `Force.enter([row])` REBUILDS — it replaces the
> graph with that one card. Two callers have shipped that bug, each destroying a walk
> silently.

So there are **two buttons and each says which it is**: *Start here* reseeds (the explicit
request that makes replacement legitimate) and *Add to walk* adopts, appearing only once
there is a walk to add to. A single button switching on graph state is the
control-means-two-things bug fixed in the atlas. `Force.adopt` is exported for this —
`branchByRow` was the only way in and it also pulls the row's relations, right for a
relation click and wrong for "add this card", where the graph grows by twelve when one was
asked for.

Three of the four seed sites route through the helper. **Build deliberately does not**:
`seedFromRows` ends by calling Discovery's `render()`, which would repaint Build's roles and
curve with Discover's landing controls — the defect `Force.renderPanel` was taught to avoid
by asking `MM.mode`. The panel belongs to the mode.

`?cards=` goes through the same reader as the textarea rather than splitting on commas
itself: query strings are the one context where a comma IS conventionally a separator, and
`URLSearchParams` has already decoded any `%2C`, so by read time the two are
indistinguishable. Ambiguous input reports itself unresolved instead of half-working. It
seeds **once** — `show()` then a reseed would draw a card, throw it away and draw the set on
the first frame anyone ever sees.

**One relation mechanism, one behaviour.** The controls live in `buildCardDetailHtml`, so
Discover, Build, the explore accordion and the browse panel all offer the same thing, and
`MM.relate(row, relation)` always does the same thing with it: grow the graph from that card.
From Explore that means switching modes — the click carries you into the walk, seeded on the
card you clicked.

**That dispatch must not fork.** Opening a linear **browse set** in Explore instead is
tempting — a scatter plot cannot grow — but sound reasoning gives the wrong result here: one
control would mean two things, so the same button would reward you differently depending on where you
happened to be standing, and the atlas was the version that felt dead. The fix is not to
teach the scatter plot to grow — it is to treat Explore as a **launchpad**: you go there to
see where things sit, then click to start walking from one. `Discovery.show` seeds the graph,
`Discovery.focus` re-centres it if the card is already present, and `Force.branchByRow` opens
the relation. The **Keep** control moved into the same shared HTML for the same reason —
keeping a card you spotted in the atlas is the same act as keeping one you walked to.

Box-select still opens a browse set; that is a different question ("what did I just lasso?")
and keeps the arrow-key walk.

**Seed only when there is nothing to lose.** Growing the graph must never be able to empty
it, and twice it could. `Force.enter` restores an existing graph *only when it is handed no
seed* — the `!rowIndices && nodes.length` branch — so an explicit `[row]` always takes the
rebuild path. `MM.relate` sent every card not already on the walk through `Discovery.show`,
which calls `newWalk(true)`; and `Discovery.enter` reseeded with `[current]`, which meant a
round trip to the atlas and back destroyed the walk on its own. The second one is why fixing
`relate` alone did not hold: the outer path had already wiped the graph before the inner
check ran.

The rule is now explicit in both places. Reseed only when `Force.nodeCount === 0`; otherwise
the card is **adopted** — `Force.adoptRow` adds it to the graph you already have, born at the
graph's centre of mass rather than at its world position (against an established cluster the
world position is usually off screen, which reads as a bug rather than as arrival) and linked
to whichever of its precomputed neighbours are already present. `branchByRow` then branches
from it normally. Two browser tests hold the line: one builds a walk, round-trips through
Explore, grows from a card that was never on the graph and asserts every original node
survives; the other asserts an *empty* graph still seeds, so the fix cannot become a no-op.

This replaced **Find Similar Cards** / **Find Synergies**, which were broken four ways at
once and untested: they took no card argument and read `selectedCards`, so they were silent
no-ops in Discover *and* the browse panel (both clear it), acted on the **wrong card** in The
Walk while drawing onto a hidden Plotly surface, and threw outright under `?renderer=canvas`
where `#plot` has no `.data`. Their highlight-trace machinery is gone with them.

### A cluster label is a camera move

Clicking a region label runs `focusRegion(id)`: frame the region from its members' **real**
extent (not the stored `w`/`h`, so the camera agrees with what is drawn after the supertype
filters) and draw only its members. Same points, same world coordinates, closer camera —
which is why picking keeps working.

It must not run **drill** — a different operation wearing the same gesture. Drill re-embeds
the subset from the 128-d vectors with stress majorization, so points fly out of their world
positions over 90 frames and land somewhere new. Right when you want local structure
revealed; disorienting when you clicked a name expecting to look closer. Drill stays on the
toolbar and box-select, where a re-layout is an explicit request.

Escape peels outermost-first: focused region → orientation lens → selection.

**Two bugs fell out of building it, both of the same shape — two places deciding one thing.**

`render()`'s group loop re-tested `activeSupertypes` against `allData`, while `filtered` (fed
to the contours and the status count) applied its own copy. Adding the region filter to
`filtered` therefore changed nothing on screen: 34,890 points still drawn for an 875-card
region. They now share one `visible(d, i)`.

And `updateLayerBy` moves points without telling the quadtree. The tree signature is layer
lengths plus endpoint ids — cheap on purpose, since a rebuild is 23.5 ms and `setLayers`
runs on every keystroke — and therefore blind to positions. A drill mutates coordinates for
90 frames with every one of those fields unchanged, so the tree went on answering with the
world-seeded positions it started from and hit-testing landed on nothing. That is exactly
"the points settle and then I can't interact with them". `mapRenderer.reindex()` is the
explicit "positions moved" signal, called once at settle and never per frame.

### The constellation: Explore grows in place

Clicking a relation in Explore adds the card and its relations to the graph **and stays
there**, drawing the edges at the cards' true atlas positions — so you see reach and
position at once, which is the one thing the force layout structurally cannot show.

This control has now been written three times and the history is the reasoning:

1. **Fork** — graph modes branched, Explore opened a linear browse set, because a scatter
   plot cannot grow. One control meaning two things is why the atlas felt dead.
2. **Carry you out** — a relation in Explore switched to Discover, seeded on that card.
   Better, but the map could still only hand you off, never respond.
3. **Grow in place** — the constellation appears on the map.

**Which relations earn an arc is measured, not chosen.** Median edge length as a multiple of
a random pair on the same map:

| relation | default (colour/type) | ability (function) |
|---|---|---|
| outclassed-by | 7.4u — **0.29×** | 0.82u — 0.04× |
| similar | 15.2u — **0.60×** | 0.27u — 0.01× |
| synergy | 24.0u — **0.95×** | 19.3u — **1.04×** |

`MAP_ARC_RELATIONS` encodes exactly that:

- **default map** — similar and outclassed-by are real structure: long enough to see, short
  enough to mean something. This is where the constellation earns its keep.
- **ability map** — those same relations are already stacked (0.27u apart on a 71u map, 97%
  inside 5% of the atlas). An arc is a pixel pretending to be information, so none is drawn
  and the status points at **drill**, which already exists and is the honest answer to
  "these are all on top of each other".
- **synergy, either map** — indistinguishable from random, and that is *correct*: synergy is
  complementary, so partners belong in different regions by construction (blink finds an ETB
  creature). It is orthogonal to every 2-D projection here, so it is **never** drawn as an
  atlas arc. The partners still join the graph and the status points at the force layout,
  where adjacency IS the geometry.

Drawing all three would have looked richer and rendered the most interesting relation as
spaghetti. Three browser tests hold the line, including one asserting every arc **terminates
at a real card's atlas position** — the actual claim the picture makes.

### session.js — one answer to "what am I holding"

Eight "set of cards" containers, seven answers to "which card is selected", nothing
reconciling them. That is not untidiness, it is the mechanism behind the complaint: a click
was a *read* in Explore (`clearSelection` + `addToSelection`, opens a panel), a *write to a
list* in Build, and a *structural mutation of a graph* in Discover — because each mode wrote
somewhere different.

`Session` owns the **focus** and the **library**, and is the interface everything asks.

**It does not own the graph's storage, deliberately.** `force.js` holds nodes as live d3
bodies whose x/y/vx/vy are mutated every tick and whose identity d3 owns; a second
membership array would be exactly the duplicate model this exists to delete. So force
registers itself via `Session.useGraph({rows, links, has, grow})` and Session delegates.
One interface, one storage, nothing to drift. `Session.links()` returns links as **rows**
rather than node objects, so a consumer with no simulation — the atlas — can read the same
relations and draw them at world positions.

Out of scope on purpose: `Build.active` and `Drill.indices` are different
concepts (a deck under construction, a published decklist, a re-layout subset). `browseSet`
stays too: box-select answers a different question and keeps its arrow-key walk.

### The orientation lens is live

`orientation` held `{rows: Set, label, anchor}` **copied out of `Force.rows()` at the moment
you entered Explore**, and never updated. Grow the graph and the atlas showed the old one
until you left and came back — a photograph of your walk. It now holds only `{label}`;
membership is `Session.rows()` read on every render and the anchor is `Session.focus`.

`test_the_orientation_lens_is_live_not_a_snapshot` grows the graph *while Explore is on
screen* and fails if the lit set does not move: 18 → 24 when measured. It engages the lens
explicitly first, because entry no longer does.

### One grouping registry — `MM.GROUPINGS`

A colour is a language, and this app was speaking two of them. The map coloured by
`COLOR_PALETTE` / `SUPERTYPE_PALETTE` / `RARITY_PALETTE` in `mana-map.js`; `build.js` kept
its own `FAMILY_COLOR` table for the role-budget bars. Same screen, same cards, two
unrelated palettes and two legends that could never agree.

`MM.GROUPINGS` is now the one definition: four groupings (`supertype`, `role`, `color`,
`rarity`), each `{label, keyOf(d), palette, order, ensure?}`.

- **`order` is authoritative.** `Object.values(groups)` is hash order, which made the legend
  shuffle between renders and match nothing else. Every surface that reports these groups
  sorts by it.
- **`ensure()` is for data outside the boot payload.** `role` lazy-loads `card_roles.json`
  (0.39 MB gz) when the grouping is *selected* — never inside the 1.83 MB discovery boot.
- **Build reads the registry through functions, not constants.** `familyColour()` /
  `familyPriority()` defer the `MM.*` read to first use; reading it at module scope would
  make `build.js` depend on script order in `index.html`, and touching `MM.*` before
  `mana-map.js` has exported it aborts the IIFE and takes every later file with it.
- **Changing the overlay calls `regroup()`**, which repaints the map *and* Build's panel.
  Repainting only the map left the mana curve answering in supertypes after the overlay had
  been switched to roles.

Build's mana curve is **segmented by the current grouping** with a key, instead of one flat
gold bar. The legend rows are controls: clicking one spotlights that group, composed with a
focused region through a single `spotlightFor(g)` predicate — a group focus is one scalar
per trace and never touches the 34K per-point array.

The default overlay is **supertype**, with a frequency-aware palette: Creature is 55.5% of
the corpus (19,050 of 34,890), Planeswalker 1.0%, Battle 39 cards, so saturation runs
*inverse* to frequency. The previous palette gave Creature `#22C55E` — the exact green
`COLOR_PALETTE` uses for G — on more than half the points, so the supertype map read as a
broken colour-identity map.

### Atmosphere at altitude, crisp up close

Two ramps in `render/canvas.js`, pulling opposite ways, because zooming changes two things.

`closeness()` (0 at the whole-map fit, 1 by neighbourhood scale) drives point size and
alpha: points draw at a constant *screen* size, so without it a dense field reads as grey
haze far out and as sparse grey dots up close — the same dimness twice.

`auraLevel()` is its inverse and drives the additive halo and the radial vignette. Zoomed
in, Explore has to converge on Discover and Build, which are the same force engine drawing
plain dots with **no halo at all** — there is no `shadowBlur` in `force.js`. A halo that
grew as you approached broke that parity and sat on top of the points you were reading.

Both ramps use `transform.k`, which is **relative to `baseFit`** — k=1 is the whole-map fit,
not an absolute data→pixel scale. Constants written in absolute scale leave the ramp flat at
0 everywhere, and on/off then measures pixel-identical, which reads as "the halo does
nothing" rather than "the halo never ran".

The halo's cap is measured, because "some aura" and "a wash" are a factor of two apart.
Alpha coverage of the canvas at the fitted view, against a 4.5% no-halo baseline:

| cap / radius | canvas coverage | full-strength ink |
|---|---|---|
| 0.55 / 3.2x | 68.4% | 4.3% |
| 0.35 / 2.4x | 50.9% | 4.2% |
| **0.25 / 1.9x** (shipped) | **35.9%** | 4.2% |
| no halo | 4.5% | 4.1% |

Full-strength ink is flat across all of them: this only ever moves the halo, never the
cards.

### Ambient motion — the galaxy layer

At altitude the atlas turns. The whole field carries a slow differential swirl about the
fit's centre plus a bounded Lissajous drift, so clusters read as islands of a moving
system rather than as a printed sheet.

**The motion is in the projection, not in the data.** `wx`/`wy` are still the static
base-fit mapping; `proj(x, y)` is those plus a time term, and since every drawn thing
already funnelled through one function, adding it there moved the whole surface at once.
Moving the points instead is the one implementation that cannot work here — `buildTree`
is 23.5 ms and its signature is deliberately blind to positions, so per-frame mutation
either rebuilds the quadtree 20 times a second or reproduces exactly the stale-hit-test
bug `reindex()` exists to document, and every world-anchored overlay detaches with it.
Two callers deliberately keep the static mapping: the cached contour field (carried by a
single rigid rotation at the bulk radius instead) and `setCamera`'s target, which must
not jitter.

**`unproj` is an exact inverse, and the motion was designed so it could be.** The swirl
is a rotation about a fixed centre, so it preserves radius — which means the angle a
point was turned by is recoverable from where it landed. `pick` un-drifts, reads the
radial bin off the rotated radius, and rotates back. Measured: five named cards, five
exact hits at a 3 px pick radius, sampled across different phases of the sway. Hit-testing
against stored positions while drawing elsewhere would be tens of pixels out at the fit,
which in this corpus is a different card, and would present as a flaky map.

**It is a sway, not a rotation, and that is a correctness decision.** A galaxy that
genuinely rotates *winds up*: shorter inner periods wrap the arms, and cards PaCMAP
placed side by side end up a quarter of the map apart — the picture keeps looking good
while it stops being true. A rigid rotation avoids the shear and costs spatial memory
instead (the region you learned was north ends up west). So the excursion is bounded and
always returns: peak **32 px on a 900 px viewport**. Kepler survives where it is legible
— period varies with radius (T ∝ a^1.5, so the rim is slower than the core) and phase
lags outward, so each instant is a shallow spiral that unwinds and rewinds.
`test_the_drift_is_bounded_and_returns` asserts both halves, because an accumulating
motion passes "does it move" and fails only that one.

**It rides the aura's ramp and stands down for anything precise.** `ambient()` is 1 at
the fit and 0 by the time a region fills the screen (`K1: 6`, the same altitude the halo
fades at), so points slide home as you approach and Explore still converges on Discover
and Build up close. It also stops when the tab is hidden, when `prefers-reduced-motion`
is set (the boot default — the **Motion** toolbar button reports the renderer's state
rather than the markup's), and when the canvas is off screen. That last check reads
`canvas.offsetParent`, **not** `host.offsetParent`: `#plot` is shared with the force
graph and stays visible in Discover and Build, so asking the host animated 34,890 points
into a surface nobody could see.

Cost control, since this is the first continuously-animating thing in the app:

| what | value | why |
|---|---|---|
| full draw | 9.9 ms | 34,890 points, unchanged from the static path |
| ambient cadence | ~20 fps | the sway travels 1–3 px/**second**; 60 fps buys nothing |
| hover/ripple cadence | ~30 fps | these do move fast enough to judder |
| per-point trig | none | 96 radial (cos, sin) bins, tabulated once a frame |
| label measurement | cached | `offsetWidth` forces layout; only position recomputes |

**Box select had to change with it.** A rotation maps a rectangle to something that is
not a rectangle, so the stored positions inside a screen marquee stop being an
axis-aligned range. `pickRect` pads its quadtree pruning by the largest displacement
currently in play and then decides membership by projecting each candidate *forward* —
exact, and it costs one projection per point the tree already visited.

### Touch: hover and click

The map answers back. Hovering a card draws a two-ring highlight that grows in over
~90 ms and then breathes; clicking one sends a ripple out from the point. Both are drawn
inside the world transform at a radius divided by `transform.k`, so they hold a constant
screen size at every zoom, and both are positioned through `proj` — a ring without the
ambient term sits beside the card it is pointing at. The cursor also becomes `pointer`
over a card and `grab` over empty space, which the surface previously never said: 34,890
points that all looked equally inert until you happened to click one.

These are the only self-animating elements in the renderer, so `wantsFrames()` counts
them — a hover or a click keeps the ticker alive for exactly as long as the animation
lasts. A hovered card that stops being drawn (a filter toggle rebuilds the pickable set)
clears the hover rather than just skipping the ring, or the ticker never stands down.

### Telescopic labels

Which names are on screen is a question about **depth relative to what is focused**, not
only about camera span. Answering from absolute span alone has two
consequences: L2 would need span < 6 while L2 spans are ~0.6, leaving 168 of the ability
map's 227 names unreachable; and focusing a region frames it without naming a single
thing inside it, so clicking into a country told you less than standing outside it did.

`regions_*.json` already carries `parent` on every entry, so the hierarchy needed no new
data. With nothing focused the span bands decide, L2 fading in far earlier than before. With
a region focused, its children are always named and its grandchildren appear once the camera
is close enough; everything outside keeps a faint L0 label, for the same reason focusing
mutes points instead of hiding them.

Two things fell out of it:

- **Labels below 0.09 alpha are dropped, not drawn.** Placement is greedy and sorted
  big-first, so a country name fading through 0.03 alpha still claimed the largest collision
  box on screen and suppressed the readable neighbourhood name underneath it.
- **The outline is written inline, scaled to each label's opacity.** Names sit at cluster
  centroids — on top of the densest colour — so they need a ring, not a drop shadow. But a
  fixed 0.95-alpha ring under a 0.28-alpha context label is more opaque than the text and
  renders as a dark smudge. CSS cannot see the per-label opacity, so `setAnnotations` emits
  the `text-shadow`. L2 labels also get `pointer-events: none`: every label is a DOM button
  over the canvas and therefore a hole in the map — a card underneath one cannot be hovered,
  because the pointer lands on the button and the canvas gets `mouseleave`. Countries and
  states keep their events because clicking them navigates.

### The boot map is `currentMap`, in both places

Explore opens on the **ability** map. Two literals had the boot map hardcoded —
`fetch(MAP_CONFIGS.default.projection)` and `loadRegionData('default')` — so changing the
default left `currentMap` saying one thing while the coordinates, and then the region names,
came from the other. Neither errors: the projection silently draws the wrong positions, and
the label pass silently finds no data and emits an empty list. Both read `currentMap` now,
and both `<select>` elements are pinned from the JS defaults at boot rather than from markup
order.

A map switch must also `reindex()` the quadtree, **after** the render that installs the new
layers. `buildTree` copies coordinates out of the *layer* arrays and `render()` rebuilds
those arrays, so reindexing first rebuilds from the outgoing layers and `setLayers` then
skips its own rebuild against an unchanged signature — the stale positions survive the very
call meant to remove them. It looks fixed and measures broken.

### stage.js — what the two canvas renderers stopped writing twice

There are two canvas renderers and there always will be: the atlas draws 34,890 cards at
fixed world positions (*where does this sit*), `force.js` draws a few hundred at simulated
positions (*what is this next to*). They shared **zero lines** while separately implementing
canvas creation and DPR resize (character-identical, both carrying the same "cost an
afternoon" comment), d3-zoom wiring, the draw prologue, world→screen, screen→world,
fit-to-extent, the `/transform.k` constant-screen-size trick, and greedy AABB label
collision — which `render/canvas.js` openly noted it had copied from the graph engine.

`Stage` owns the **surface**: pixels, gestures, geometry. `surface()` + `surface.open()`,
`camera()`, `placer()`/`placeLabels()`, `drawEdges()`/`edgeInk()`. −114 lines across the two
renderers for +235 of shared primitive.

**Stage never stores a coordinate, and that is the load-bearing decision.** `force.js`
mutates node x/y every tick; `canvas.js` never touches its points and moves only the
transform. An abstraction owning positions would have to serve both and become a union of
two designs. Callers paint inside a transform Stage sets up and hand it screen-space boxes.

Two places the API followed the caller rather than the reverse: label placement is
**incremental** (the graph draws each label on accept, caps on the number *accepted*, and
shares one collision set between edge and node labels so a reason can never sit on a card
name), and `drawEdges` takes a `relOf` callback because deck-ness is a property of the graph
— both endpoints from a loaded decklist — not of the link.

### Typed edges, and why `mode: 'lines'` was not enough

A `lines` layer is one flattened polyline with a single colour for the whole layer, excluded
from the quadtree so it can never be hovered. That draws Build's verified-line edges
and is structurally unable to say *this* edge is a synergy and *that* one an obsolescence.

`mode: 'edges'` carries `[{source: [x, y], target: [x, y], rel, reason, d}]` and hands the
inks to `Stage.INK`, so an edge means the same thing on the map as on the graph. Coordinates
are explicit rather than row indices — the renderer still knows nothing about cards, and the
producer already has the positions. Edge layers stay out of the quadtree (nothing to pick)
and out of the legend (a swatch would be a dot standing for a line). `curve` bows the line,
because a straight segment between two distant cards reads as an assertion about the space
between them.

`test_the_atlas_draws_typed_edges` asserts this by **reading pixels**, not by inspecting the
layer list: "the layer is present" is exactly the claim that passes while nothing is drawn.

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

**The hover card is bounded.** `positionPopup` clamps to the plot frame, and must measure
the popup only after the image has loaded. Measuring the instant the `<img>` is inserted —
before the network returns — reads height ~0, so the bottom clamp has nothing to clamp and a card hovered
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

### The library, import, and the hand-off

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
Build's deck picker. It differs from a pasted import in the way that matters: the manifest carries a
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

**The library** is a deliberately light selected-set, separate from the graph: the graph
is where you are LOOKING, the library is what you are KEEPING. It is called a library
rather than a basket or a tray because in Magic your library IS your deck — the cards you
gather while brewing are the deck you are gathering. (Distinct from **the bench**, which
is every deck you own.) It is the fifth "set of cards"
idea in this codebase and the only one that exports.

**Import** parses a pasted Moxfield export with `viz/js/decklist.js`, resolves names
against `viz_index.json`, and seeds the graph with the whole list, commander pinned. It
deliberately does **not** touch the deck picker: `build.js` refuses any slug absent from the
CLI-built `data/decks/index.json`, and an imported deck has no slug and never will.
Measured on the tracked Edgar list — 136 entries, 129 unique cards, 0 unresolved, 26 ms.

Pinning uses `Force.pinCard`, not `focusCard`. `focusCard` *branches*, so importing a
129-card deck produced a 135-node graph — six cards the deck did not contain.

**Optimize is a brief, not a button.** There is no backend and this does not add one: the
pilot loop is 6–10 serially dependent LLM subagents costing ~330k–1.7M tokens, and a
static page cannot run Python. The library emits a JSON brief (download + clipboard) naming
the cards and candidate commanders, which a human pastes into Claude Code where that loop
already works. The brief says so in its own `next_step` field.

## The canvas renderer — Phases 2–3 of the migration

`viz/js/render/canvas.js` draws the map, and is now the **only** renderer: Plotly is
deleted, there is no CDN tag, and **`?renderer=` no longer exists** — nothing in the JS
reads it. The section below is the record of how it got there, from when both were live at
once and that flag switched between them so they could be compared on identical data. The
graph engine proved the machinery on 500 nodes; this points it at 34,890.

*(A flag that outlives its migration reads as a supported option. Left as history in the
prose below, deliberately, because the comparisons are the reasoning — but the heading and
anything that looks like an instruction must not offer a switch that is gone.)*

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
| Draw 34,890 points | — | 7.8 ms batched · 16.9 ms per-point |
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
  Build's map view dims 34,000 points against ~100 with an opacity *array*; without this the canvas
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

The map draws 34,890 WebGL points. Four rules, each of which was violated and measured:

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
34,000-entry array per colour group plus Plotly's per-point WebGL path. Build's map view dims
*everything* and redraws its 99 on top, so one scalar is equivalent — it declares
`dimsAll()`. The deck builder dims a real subset (format-illegal, colour-identity
violations) with nothing over it and keeps the array.

**Do the work once per gesture.** Box-select decides its destination before building
anything: constructing the 8-card stack, rendering the panel and rebuilding the highlight,
only to discard all of it for a big box, is three wasted passes. One pass over the points,
one destination.

Measured, same page, median of 7:

| | before | after |
|---|---|---|
| `render()`, nothing selected | 90 ms | **30 ms** |
| `render()`, 15,000-card browse | 128 ms | **36 ms** |
| Arrow press while browsing | 25 ms | **16 ms** |
| Plotly calls per render | react + add + delete | **react** |
| Plotly calls per arrow press | delete + add | **restyle** |

**What is still slow, and is not ours.** A shift-drag fires `plotly_selecting` on every
mousemove, and Plotly hit-tests all 34,890 scattergl points each time — **measured ~138 ms
per event** with only the seven base traces loaded. That is the dominant cost of box-select
and it is inside Plotly. Any large highlight trace left on the plot adds to it, which is
why the browse selection is one trace rather than one per colour. Build's map view renders in the
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

Manual `?v=N` query strings, per page: `index.html` on all **nine** JS files and `mana-map.css`; `deck.html` on `deck-view.js` and `tokens.css`. **Bump the version on the page you touched** before pushing — Pages/browser caches are aggressive. On `index.html` all nine script busts must move together; a test asserts it, because a mismatched pair is how `build.js` ends up talking to a stale `mana-map.js`.

For contrast, `manuals/magazine.css` (the legacy page's stylesheet) is **content-addressed** (`?v=<sha8>` from the CSS text, in `pilot/design.py`), so a stylesheet change there obligates rebuilding every manual page but can never go stale. That is the pattern to copy if `viz/` ever outgrows manual bumps.

## Data paths

**Two registries, one per page** — the map's and the dossier's, deliberately disjoint:

- **Map** (`mana-map.js`): the `DATA` map at the top (built on `DATA_BASE = '../data/'`) holds all nine card-map artifacts. `MAP_CONFIGS` (per-map projection/embeddings/regions) and every fetch reference it; `build.js` and `discovery.js` consume `MM.DATA.*`. Add new card-map files there, never as inline literals.
- **Dossier** (`deck-view.js`): `BASE = '../data/decks/'` plus a `FILES` map of per-deck artifact names. It fetches `data/decks/index.json` first — the manifest written by `manamap pilot build-index`, carrying the deck list and each deck's **passing** stack filenames, because a browser can list neither the deck directory nor `stacks/`. Never hardcode a deck list; add a deck and re-run `build-index`.

## window.MM API surface

Every member has a live caller (build.js, generated onclick handlers, or index.html) — exports without callers were trimmed 2026-07; don't re-add one without a consumer.

Getters: `allData`, `currentMap`, `obsolescence`.
Helpers: `escHtml`, `buildHoverTextMinimal`, `renderManaSymbols`, `closeDetail`, `removeFromSelection`, `bringToTop`, `selectByName`, `findSimilar`, `findSynergies`, `render`, `setStatus`, `setMode`.
Constants: `MAP_CONFIGS`, `DATA`, `EMBED_DIM`.
Async data loaders: `getEmbeddings()`, `getSynergyGraph()` — the deck builder awaits these instead of downloading its own copies of the two largest payloads (~17 MB + ~27 MB); both resolve to the shared cached instance.

## The workbench (`workbench.html`) — the landing page

The front door for a **pilot**, as `index.html` is the front door for the corpus. It
fetches every deck's `info.json` from the manifest (N+1 requests, no new artifact) and
answers one question: *which deck should I spend tonight on?*

Two views over the same payload, toggled and carried in the URL
(`?view=table&sort=played`, written with `history.replaceState` so a sorted view is
linkable and the back button is not filled with noise):

- **Racks** group by lifecycle — LOCKED (built in paper, playable tonight), ON THE
  BENCH (lists and build plans, nothing sleeved), and the dead.
- **The fleet table** is one row per deck across record, stages, evidence, table and
  open work, sorted four ways: *recently played*, *needs game logs*, *needs analysis*,
  *optimisations identified*. Every sort maps to a predicate `deck_info._next` already
  computes — the page adds no judgement of its own.

**`info.next[0]` is the last column and gets its OWN full-width row.** As a cell it
broke the table: every other column is `nowrap`, so their widths summed past 100% and
the browser pushed the whole thing behind a horizontal scrollbar. It is a sentence, not
a cell, and the fix was to stop treating it as one.

**Three labelled links per card — MANUAL · DOSSIER · ON THE MAP — and no whole-card hit
target.** The card used to be one `<a>` to the dossier with a small separate manual
link, which is the interaction bug this repo already fixed once in the atlas: *a
control that opens two different things depending on where you click*. Every
destination is now named. `ON THE MAP` is `index.html?deck=<slug>`, the documented
inbound contract that lands in Build with the deck loaded.

**The bug worth remembering**: the games/record chip read `info.status.games`, and
`deck-info` writes the games under **`info.record`** — `info.status` is the stage-count
block. The chip had never rendered once, on any deck, so two decks with logged games
looked identical to nine without. It survived because a missing chip is invisible; a
wrong chip would have been reported in a day.

`workbench.js` shares `deck-view.js`'s shape deliberately: an IIFE that exports nothing,
depends on no `MM`, and reads only committed artifacts. Neither page loads the map.

## The deck dossier (`deck.html`)

### It is a DOSSIER, not a report — nine sections with a spine

A report has a conclusion; a dossier has a **latest entry**. The page renders
nine sections whose order is owned by **`pilot/page_spec.DOSSIER_SECTIONS`**
and transcribed into `deck-view.js`'s `DOSSIER` literal, with
`tests/test_viz_deck_lens.py` locking the two together — the same contract
`decklist.js` lives under. Before that the order was an anonymous array inside
`render()`, so "what sections does this page have, and why that order" was
answerable only by reading the assembly line.

| # | section | what it holds | from |
|---|---|---|---|
| 1 | **Cover sheet** | mugshot, stamp, MO, engine health, three numbers | `info.json`, the manifest |
| 2 | **Rap sheet** | one row per version: what changed, why, expected, observed | `versions.json` |
| 3 | **Known associates** | the 99 by the job each card does | `deck_map.json` |
| 4 | **Vitals** | the seeded diagnostic | `info.diagnostic` |
| 5 | **Priors** | every game, one row, with a coded cause | `log.jsonl`, `log_causes.json` |
| 6 | **Captain's logs** | the pilot's words, unedited | `log.jsonl` |
| 7 | **Exhibits** | the evidence, stamped with the list it describes | *(pending)* |
| 8 | **Open leads** | what is unresolved and which loop settles it | `info.open_questions` |
| 9 | **Analyst's assessment** | the current read, dated, filed apart | `info.diagnosis` |

Four rules the sections are built to:

- **ALL NINE ALWAYS RENDER.** A tab that vanishes when its drawer is empty is
  the "cleaned up into a narrative" failure — you can no longer tell an
  un-played deck from one whose log you have not opened. Three sections used to
  vanish and one (`vitals`) vanished on **nine of ten decks**, because
  `diagnostic.json` is a gated artifact with no `STAGES` row so `absent()` never
  found a todo for it.
- **THE ASSESSMENT IS LAST AND SEPARATE.** A file where the analyst's opinion
  sits inside the record loses trust, and the old page rendered the diagnosis
  verdict as one inline sentence in the middle of the audit panel.
- **THE COVER SHEET'S RECORD IS THIS VERSION'S**, never the lifetime total. A
  deck that went 0–3 on a list you have replaced is a fact about a deck that no
  longer exists.
- **THE ENGINE-HEALTH WORD TRAVELS WITH ITS MEASURE.** It is a verdict, which
  this repo normally refuses to publish; it ships only because
  `deck_info.ENGINE_HEALTH_BANDS` is a named constant the pilot can move and
  the rate and interval sit beside the word. Absent — never `WEAK` — when
  nothing was measured.

**Two numbers with a definition attached.** "Keepable sevens" is
`keep_can_act_by_t3_rate`, the STRICT rule, labelled *"can act by turn three"*.
`goldfish.py` reports a loose one too and warns in its own comment that it
"sits near 100% inside the keep window for every deck — informative about the
mulligan rule, useless as a fitness signal."

### The rest of the page

Renders a deck's **committed pilot artifacts** and nothing else. Slug comes from
`?deck=<slug>`, the frontend's only URL state — now honoured by **both** pages:
`index.html?deck=<slug>` enters **Build** with that deck loaded rather than dropping
the reader on an unfiltered map with a query string they cannot see.

**The workbench half comes first**, because the questions a pilot sits down with are
"where is this, and what do I do" — not "what shape is the mana curve". Those panels read
`info.json`, the shape `deck-info --write` composes; the page renders it rather than
re-deriving anything, so it cannot disagree with the command that owns each figure.

| Panel | Artifact | Tier |
|---|---|---|
| **What to do next** (the derived `next`) | `info.json` | ◆ |
| Where it stands (stages, gates, bracket, record) | `info.json` | ◆ |
| Every list this deck has been | `versions.json` (TRACKED; `make manuals`) | ◆ |
| What limits it (audit axes + diagnosis) | `info.json` | ◆★ |
| The engine (stages, lines solid where a stack proves them) | `engine.json` | ✓◆★ |
| At a table (sim runs + experiments) | `sim/*.json`, `experiments/*.json` | ◆ |
| Asked and answered | `prescriptions/*.json` | ◆★ |
| The captain's log (+ the debrief beside each entry) | `log.jsonl`, `log_annotations.json` | ★ |
| Open questions, and the loop that would settle each | `info.json` | ★ |
| **The Constellation** (below) | `deck_map.json` | ◆ |
| Bracket Floor + its named driver | `bracket_report.json` | ◆ |
| Sources Say (pips vs sources, land classes, on-curve) | `mana_analysis.json` | ◆ |
| By the Numbers (meters, turn table, assumptions) | `goldfish_metrics.json` | ◆ |
| The Short List (ten) | `considering.json` | ◆★ |
| The tutor guide (collapsible per tutor) | `tutor_guide.json` | ★ |
| The Kill (case files, citations verbatim) | `stacks/*.json`, passing only | ✓ |
| The Builder's Record (slots, scores, runners-up) | `build_plan.json` | ◆ |

**A browser cannot list a directory**, which is why `data/decks/index.json` names the
files in `sim/`, `experiments/`, `prescriptions/` and `decisions/` alongside the stack
filenames it always carried. Nothing else was blocking those panels — every one of those
artifacts is tracked and fetchable; the page simply had no way to learn their names.

**Every sim figure carries its median, its interval, its N and the AI caveat**, or it does
not appear. `figure()` builds them that way, and the panel closes with Forge's own rating
of its AI. A mean without its spread is a number that describes no game: kianne's
experiment arm B read mean 17.42 with a **median of 0**.

**Use `.wb-list`, never `.dc-legend`, for a generic list.** `.dc-legend i` is an 11px
colour swatch for the constellation key, so reusing that class turned every `<i>` in an
engine line into a little square sitting on top of the text.

**Nothing is recomputed in the browser and nothing is hardcoded.** The manual renders these
same artifacts as ◆ reproducible evidence, so a second implementation that drifted would
quietly break the tier contract. A missing artifact means an absent panel, not an error —
only `hapatra` and `yawgmoth-swarm` have a `build_plan.json`, so six of eight dossiers show no builder
panel. `tests/test_pilot_deck_manifest.py` asserts the manifest matches the artifacts and
that every stack it lists is checker-passed.

The three surfaces form a cycle. Each legacy page's back matter links to
`../viz/deck.html?deck=<slug>`; the dossier links to the page, the deck list, and
`index.html?deck=<slug>`; the Lens links back to both. Before the dossier shipped, the two
products shared exactly one link, one-way.

### The Constellation panel

The same `deck_map.json` the legacy page prints, drawn the same way, in the **same colours** —
and the one thing print cannot do: hover a point and it names the card and its city.
*"Ohran Frostfang — A CARD PER BODY."* That association is the whole reason the panel
exists, since the city names are the vocabulary the notes and the engine model speak in.

The hover is a plain SVG `<title>` on a **transparent 11px disc** over each 4px dot. No
tooltip layer, no JS, works with scripting off. A 4px hover target is not a target; the
invisible disc is what makes it one.

Density is one soft disc per neighbourhood rather than the printed page's convex hull. At this
size it reads the same, and a hull the page also has to hit-test through is code bought for
nothing.

**`CITY_INK` in `deck-view.js` is a TRANSCRIPTION of `pilot/design.py`'s list, and that is a
standing drift hazard.** There is no shared module: the printed page must render with no JS
and this page with no Python. If the two lists diverge, the printed map and the site disagree
about which territory is which **while both look perfectly correct**. Noted at both sites;
change them together.

A deck map position is **LOCAL** — the deck re-laid-out from its own cards — and is not an
atlas position. The panel says so in its own body copy, for the same reason drill does.

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

**Similarity is not the displayed map.** `loadEmbeddings` reads `SIMILARITY_EMBEDDINGS`,
never `MAP_CONFIGS[currentMap].embeddings`. Reading the displayed map makes "similar" mean
"same colour and type" on the default map — a space measured at 3.05 of its 128 effective
dimensions and 0.090 recall@10
against known functional equivalents, which is why *Doubling Season* returned arbitrary green
enchantments. Find Similar, the walk and drill now all read `SIMILARITY_EMBEDDINGS` (the
function space) whichever projection is on screen. The projection is a picture; similarity is
a question. Switching maps no longer drops the loaded array either — the old per-map keying
re-fetched 17 MB and gave the same card different answers depending on the view.

**One k-nearest, `MM.nearestTo(row, k, opts)`, and now genuinely one.** The header here used
to claim `cosineSimilarity` and the sort inside `findSimilarCards` had been consolidated into
it. They had not — that scan was still live, sorting all 34,890 rows to take 20, with
different filter semantics. Both are gone. `respectFilters` defaults to true so a
neighbourhood will not walk you into a supertype you have hidden; the graph passes `false`,
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

The legacy page has a card preview too (`design.py` `.card-pop`), but it is pure CSS anchored
to a static inline element. A point in a WebGL scatter is not an element, so only the look
transfers, not the mechanism.

**Aiming at a card's pixel does not guarantee that card.** `hovermode: 'closest'` over
34,890 points means a denser neighbour a pixel away wins — asking for Sol Ring's coordinates
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
- Region labels as real DOM buttons with a zoom-dependent L0/L1 CSS crossfade; optional density contours ("Topo")
- Pinch zoom on mobile comes from `d3.zoom` (the hand-rolled version existed only because `scattergl` has none)

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

ES-module migration / splitting the IIFEs, moving the ~17 inline styles in generated HTML into CSS, content-hash cache busting. Lint and format are deliberately absent; CI does exist (`.github/workflows/test.yml`) and deliberately does NOT run the browser suite.

## The graph engine (`viz/js/force.js`) — the first thing here that was not Plotly

A fourth map mode, and the opening move of the renderer migration. Cards become nodes,
128-d cosine distance becomes link length, and a velocity-Verlet simulation gives the
graph weight: it settles, it wobbles, you can grab a card and fling it and the rest
follows. Click a card and its nearest neighbours in the full 34,890-card corpus are pulled
in, the simulation reheats, and the graph grows toward whatever you were curious about.
That is the walk. The path you took is drawn as a gold trail.

**Canvas, `d3-force`, `d3-zoom`, `d3-drag` — no Plotly at all.** It is built as a new mode
precisely because nothing can regress there: it proves the canvas renderer, the zoom
behaviour and the hit-test against real data before any of that goes under the map itself.

**Seeding** comes from `MM.selectedRows()` — browse set first, then the 8-card stack — so
every existing way of picking cards feeds it: box-select, a region, a deck. Relations no
longer seed a drill; they grow the graph instead, which is the same idea with physics.

With **nothing** selected it shows an empty state offering every published deck and the largest
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
