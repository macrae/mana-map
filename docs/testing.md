# Testing

```bash
.venv/bin/python -m pytest              # everything (1,312, ~10 min)
.venv/bin/python -m pytest -m "not browser"   # fast suite (1,189, ~35 s)
.venv/bin/python -m pytest -m "not browser" -n auto   # same, ~19 s
.venv/bin/python -m pytest -m browser         # the 80 browser tests (~340 s)
```

1,312 tests in `tests/`: 438 card-pipeline + 751 pilot-subsystem + **123 browser**.
One is a still-unmet `xfail(strict=True)` ship gate in `test_embedding_quality.py` — see below.

## Source assertions do not catch regressions

The frontend has two kinds of test and only one of them is real.

`test_viz_{drill,deck_lens,viewer}.py` read JS as **text** and assert that certain strings
appear in certain files. They are cheap, they document intent well, and they are genuinely
useful for invariants a human keeps breaking (cache-bust parity, "this function must not be
called twice"). But they cannot see behaviour.

They also fail correct refactors. Deleting Plotly renamed the calls they grep for without
changing anything they were protecting: one asserted the box-select handler used an early
`return`, and failed because the canvas handler expresses the same exclusion with
`if/else`. `test_viz_camera.py` was **retired** in that pass rather than ported — all three
of its assertions were about a Plotly-only hazard (`react` replacing layout wholesale and
silently autoranging), and the invariant they stood for was already covered behaviourally
below, where it belonged all along.

On 2026-07-30 a perf commit deleted a variable declaration and left the property that
referenced it. `drill.js:getOverlayTraces()` threw `ReferenceError` on every render while
drilling; drill mode rendered nothing at all. **All 13 tests in `test_viz_drill.py`
passed** — every string they looked for was still in the file.

`test_viz_behaviour.py` exists because of that. It boots a real Chromium against a real
server and asserts on what rendered. Verified both ways: against the broken revision the
source tests pass and the behavioural tests fail with
`assert ['text is not defined'] == []`.

**When adding a frontend test, ask which kind you are writing.** If it would still pass
against a renderer that draws nothing, it is a source assertion — fine, but it is not
coverage.

**Drive the real input, not the function behind it.** `test_browse_cycling_moves_the_marker`
called `MM.cycleNext()` and passed for weeks while the arrow *keys* were dead in browse
mode — the handler bailed on `selectedCards.length === 0` and `enterBrowse` empties that
array, so only the on-screen buttons worked while the panel's own hint said "← → browse".
The navigation tests now dispatch `KeyboardEvent`s.

**Source assertions go stale on shape, not on meaning.** `test_arrows_and_arrow_keys_share_one_implementation`
matched literal indentation and broke the moment the key handler was rewritten to fix that
gate — while the invariant it cared about was untouched. It now asserts the delegation
(`cycleSelection` is called; the handler does not recompute an index) rather than the text.

### Browser tests (123) — `tests/test_viz_behaviour.py` + `test_decklist_parity.py`

The session fixtures `browser` and `viz_server` live in `tests/conftest.py` — see the
section below for why they cannot live anywhere else. `conftest_viz.py` still holds the
page-level helpers: an ephemeral `http.server` rooted at the repo — `viz/` and `data/` must
be siblings, the same constraint GitHub Pages imposes — plus a booted page that waits on
`MM.allData` rather than a timer, because the projection is 12.9 MB. Playwright is imported
lazily, so the other 1,189 tests never pay for it.

Every test asserts `page.js_errors == []`. That list collects `pageerror` and console
errors, and it is what catches the class of bug above.

**Transitions are disabled in the page fixture, and that is not a nicety.** Playwright
pages run backgrounded and Chrome throttles CSS transitions there, so the side panels'
`transition: width 0.25s` never advances and `.deck-panel.open` measures 1px forever.
That blinded the suite to a real bug — the walk's deck menu rendering into a collapsed,
unclickable panel — because the tests only ever asserted that buttons *existed in the DOM*
and clicked them programmatically. `test_the_walk_panel_is_actually_on_screen` now asserts
geometry and hit-testing instead: panel width, buttons inside the viewport, and
`elementFromPoint` at a button's centre actually landing on that button.

The same throttling applies to `requestAnimationFrame` and `ResizeObserver`. Anything that
depends on either to stay correct cannot be verified here — and, more importantly, cannot
be relied on in a background tab at all.

Covers: boot, plot geometry, drill render + return, the accordion, browse mode holding a
whole selection, browse cycling, camera preservation across filter and search, camera
*refit* on a map switch, Build's map view, mode exclusivity, two perf ceilings (render
budget, and that a render is exactly one `setLayers`), and **the graph** — that it
resolves rather than collapsing, that link lengths stay inside the chord range `[0, 2]`
(screen distance is not bounded, so this fails the moment 2-D positions leak in), that
branching grows the graph and records the trail, and that leaving restores the map.

Setup, one time: `.venv/bin/python -m playwright install chromium` (~94 MB). Without it the
whole file skips cleanly, so a fresh clone still runs the other 1,189.

## A passing check proves nothing until you have seen it fail

Three measurements in one session agreed with the code for the wrong reason. Each looked
like a green test.

**Synthetic mouse events do not carry `offsetX`/`offsetY`.** The canvas hover handler picks
on exactly those, and a hand-built `MouseEvent` leaves them at 0 — so every probe was
reporting the card at the canvas origin rather than the one under the cursor, and passed
against completely broken hit-testing. Use `page.mouse.move`. Anything driven through
`dispatchEvent` is testing a different code path from the one users take.

**`getImageData` returns colour un-premultiplied.** The map canvas has a transparent
background — the dark page shows through — so a point drawn at 0.09 alpha lands as *full
colour, alpha 23*. Read RGB and a fully dimmed map is indistinguishable from a lit one:
measured, luminance moved 6.88 → 6.85 while the composited image lost 63% of its bright
pixels. **Dimming lives in the alpha channel.** `_ink` counts alpha > 10 ("did anything
draw"); `_ink_strength` adds alpha > 150 ("is it drawn at full strength"), which is what
separates spotlit from muted.

**A total can be dominated by the term you are not testing.**
`test_focusing_a_region_dims_the_map_instead_of_erasing_it` measured ink over the whole
canvas and passed with the unlit alpha set to **0** — the exact regression it exists to
catch. The focused region's own points, at full strength and haloed, carry enough ink that
the total clears any threshold no matter what happens to everything else. It now picks the
densest 120px patch containing *no member* of the focused region, chosen from the data at
runtime so a re-cluster cannot point it at empty space. Erased: 0 of 5,442 px.

The corollary: **when an effect shares pixels with what you are measuring, move the camera
rather than lower the bar.** `test_canvas_draws_density_contours` asserts topo triples the
ink. At the fitted view the atmospheric halo already covers ~36% of the canvas and the
contours draw over the same clusters, so the toggle moved total ink by 2.8 points — not
because contours had stopped drawing but because the measure had saturated. Zoomed in, where
`auraLevel()` is 0 by design, the same toggle is 7.6x.

## A test that inherits a default is testing the default

Ten browser tests took their map from whatever the app booted on. Flipping the default to
the ability map silently repointed all of them at a map whose answers differ *by design* —
`MAP_ARC_RELATIONS` draws similarity arcs on the colour/type map and none on the ability map
— and they failed as though the renderer had broken. `canvas_page` now pins the map, and
tests that care which map they are on say so in their own body.

The same applies to constants borrowed from a palette: `test_the_atlas_draws_typed_edges`
asserted exactly `7` legend rows, which was the size of the colour palette rather than
anything about edges. It compares legend rows against the marker-layer count now — the
invariant it was actually trying to state.

## Wait for the condition, never for a timer

Two tests in this suite have failed intermittently for the same reason, and both looked
like flakes rather than what they were.

`test_the_hover_card_stays_inside_the_frame` waited a fixed 420 ms and then asserted the
popup was taller than 200 px. That is a test of whether Scryfall answered in 420 ms. It
failed in full runs and passed alone — the signature of timing an external fetch under
load. It now waits for the `<img>` `load` event.

Fixing it exposed the sharper version of the same mistake. Waiting for the popup *element*
is not enough: it is created once and reused, so from the second hover onward it is already
in the DOM while still hidden behind the 180 ms hover delay. Polling for existence returns
instantly and measures a `display:none` box as 0×0 — a green-looking wait that measures
nothing. The condition that actually means "ready" is `style.display === 'block'`, and then
the image.

The earlier instance was node labels in the graph, measured mid-settle when every label
legitimately collides; that one produced a confident and completely wrong conclusion that
labelling was broken. **If an assertion depends on layout, network, or a simulation
settling, wait for that thing — a `setTimeout` long enough to usually work is a flake with
a delay on it.**

## Do not assert on things outside the code's control

The same shape has now cost three debugging sessions, one layer apart each time.

1. A **fixed timer** waiting for a Scryfall image, then asserting the popup's height.
2. Waiting for the popup **element** rather than its visible state — it is created once and
   reused, so from the second hover it is already in the DOM while still hidden.
3. **Console noise from a third-party host.** `js_errors` captured every console error,
   including failed fetches of card art, so one `ERR_CONNECTION_RESET` failed
   `test_escape_returns_the_whole_atlas` — a test that loads no image on purpose and has
   nothing to do with Scryfall.

`conftest_viz._record` now filters `Failed to load resource`, and **narrowly**: a
`ReferenceError`, a `TypeError`, or a failed fetch of our OWN data still fails the test,
which is the entire point of capturing them.

## Session fixtures belong in `conftest.py`, not in an imported module

`browser` and `viz_server` live in `tests/conftest.py`. They used to live in
`conftest_viz.py` and be imported per test module — and a fixture imported into two
modules is **registered twice**, so two files importing `browser` opened two concurrent
`sync_playwright()` contexts and every browser test errored at setup.

It only appeared in a full run. Each file passed alone, which is the worst shape a
failure can have: the obvious debugging move (run the failing file) makes it disappear.

Playwright is still imported lazily inside the fixture body, so the 1,034 non-browser
tests pay nothing for its presence in `conftest.py`.

## Ship gates: `xfail(strict=True)` as a stated goal

`tests/test_embedding_quality.py` carries three tests that fail today, on purpose. They
encode a defect the project has committed to fixing — the trained embedding scores 0.093
recall@10 against the frozen text it is built from at 0.187 — and a plain failing test would
just leave the suite red for the duration, which teaches everyone to ignore red.

`xfail(strict=True)` is the right shape for this:

- today it reports XFAIL, so the suite is green and the goal is visible in the output;
- when the retrain succeeds it reports **XPASS, and strict turns that into a failure**, so the
  markers cannot be left on and the achievement cannot be silently pocketed.

Use it for "we know this is broken and intend to fix it". Do not use it for flakiness — an
`xfail` on a test that sometimes passes is how a real regression gets hidden.

Alongside them are ordinary **regression floors** set at 80% of measured values. Those catch a
change that makes things worse while the gates are still red, which is the window where a
well-meant refactor would otherwise be invisible.

## The rest of the suite

377 card-pipeline + 601 pilot-subsystem. Three categories:

**Card-pipeline unit tests (281) — no data files needed, run anywhere:**

| File | Tests | Covers |
|------|-------|--------|
| `test_extract.py` | 53 | Multi-face cards, derived columns, supertype classification |
| `test_preprocess.py` | 23 | Vocab building, encoding, normalization, multi-hot |
| `test_mechanical_tags.py` | 45 | All 33 tag regexes, removal edge cases, multi-hot |
| `test_synergy.py` | 19 | Rule matching, bidirectionality, combo exclusion, ranking |
| `test_power_creep.py` | 36 | Strictly-better detection, tiered similarity gate, stat parsing |
| `test_combos.py` | 30 | Combo extraction, graph building, dedup |
| `test_cluster_regions.py` | 31 | Region naming (color/type/guild/TF-IDF), geometry, dedup |
| `test_card_roles.py` | 27 | Role classification, type-line mana disambiguation, coverage floors |
| `test_analysis_common.py` | 17 | Colour-identity masks, name index, vectorized top-k |

**Card-pipeline data-dependent tests (42) — need artifacts from a pipeline run:**

| File | Tests | Covers |
|------|-------|--------|
| `test_pipeline_integration.py` | 30 | Cross-artifact count consistency, output quality checks |
| `test_find_similar.py` | 12 | Binary format fidelity, L2 normalization, 128D vs 2D ranking |

Both are skip-guarded: `test_pipeline_integration.py` skips per-file via `requires_file(...)`; `test_find_similar.py` uses the module-level `requires_data` marker from `tests/conftest.py` (gates on `embeddings.npy` existing).

**Pilot-subsystem tests (601) — mostly pure-function with inline fixtures; data-gated ones behind markers:**

| File | Tests | Covers | Data gate |
|------|-------|--------|-----------|
| `test_pilot_rules_db.py` | 12 | CR chunker edge cases (TOC, subrules, examples, glossary) | 2 behind `requires_rules` |
| `test_pilot_query_rules.py` | 5 | Semantic top-k, exact lookup, suggestions | all behind `requires_rules` |
| `test_pilot_fetch_deck.py` | 24 | Decklist parsing, mocked Scryfall, exact printings, decklist-hash short-circuit | 1 behind `requires_deck` |
| `test_pilot_validate_stack.py` | 18 | Citation contract, decision form, strategy-citation dispatch, golden artifacts | golden test behind `requires_deck` **and** `requires_rules` |
| `test_pilot_goldfish.py` | 16 | Seeded determinism, mulligan rule, target assembly | 1 behind `requires_deck` |
| `test_pilot_build_manual.py` | 42 | Department completeness, contract integrity, furniture rendering, determinism, escaping | — |
| `test_pilot_strategy_db.py` | 9 | Strategy chunker (IDs, sources, parents), real-DB alignment | 3 behind `requires_strategy` |
| `test_pilot_validate_strategy.py` | 18 | Doc form errors, changelog contract, strategy citations through `validate_citations` | — |
| `test_pilot_validate_issue.py` | 29 | Issue identity incl. the decklist_sha256 stamp, department completeness/order, tier-costume integrity, card-name accuracy | — |
| `test_pilot_artist_credits.py` | 24 | Standout detection, per-entry counting, drop runs, roster overlap | 1 behind `requires_deck` |
| `test_pilot_agent_cache.py` | 57 | Fingerprint stability/order-independence, prose-shape semantics, staleness diffs, record guards, N/A scan semantics (incl. sideboard gating) and exit codes, memoized loaders | 5 behind `requires_deck` |
| `test_pilot_build_deck.py` | 48 | Pool hard filters (bracket, identity, bans), scoring components, slot filling with alternates, emergent-combo pass, decklist naming | — |
| `test_pilot_manabase.py` | 40 | Hypergeometric source counts, pip counting incl. hybrid, effective-pip quorum, greedy land selection, land quality | — |
| `test_pilot_bracket.py` | 35 | Floor drivers, commander-assumption exclusion (A-004), two-card infinites, tutors-never-scored, goblin-storm golden checks | 3 behind `requires_deck` + `requires_roles` |
| `test_pilot_validate_build.py` | 37 | Card count, singleton, identity, per-role budget arithmetic, bracket cross-check, manabase staleness, critic verdict consistency | — |
| `test_pilot_deck_facts.py` | 14 | Deterministic deck brief: DFC colours, curve, restricted-mana classes, notes | 4 behind `requires_deck` |
| `test_pilot_sideboard_facts.py` | 14 | Board split, accessory exclusion, lines-opened set difference, bracket-if-added | 2 behind `requires_deck` |
| `test_pilot_validate_sideboard.py` | 22 | Swap form (in/out/why/when), recomputed bracket deltas, verdict closed set | — |
| `test_pilot_validate_strategic_frame.py` | 15 | Frame form, engine strategy_refs, candidate-line status, shared validator tail | — |

## conftest.py

- `requires_data` — skipif marker gating on `embeddings.npy` (card pipeline)
- `requires_rules` — gates on `rules_index.json` (build the rules DB first)
- `requires_deck` — gates on `data/decks/goblin-storm/cards.json`
- `requires_strategy` — gates on `strategy_index.json` (run `manamap pilot build-strategy-db`)
- `requires_roles` — gates on `card_roles.json` (run `manamap card-roles`)
- `data_dir` — session fixture returning the resolved `config.DATA_DIR`

Each pilot marker gates on the *last* artifact of its stage so a partially populated directory still skips cleanly.

Paths come from `manamap.config` (never hardcode `Path("data")`), so the suite runs from any CWD and honors `MANAMAP_DATA_DIR`:

```bash
MANAMAP_DATA_DIR=/nonexistent .venv/bin/python -m pytest   # data tests skip cleanly
```

## Notes for writing tests

- Unit tests build inline DataFrames/dicts — keep it that way (no fixture files)
- `test_synergy.py` patches `manamap.analysis.synergy.load_combo_partners` to stub combo I/O
- Integration count assertions enforce the index-alignment invariant — if you change the card count (new Scryfall data), re-run the full pipeline before expecting green
