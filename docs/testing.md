# Testing

```bash
make test            # THE DEFAULT: non-browser, non-forge, -n auto, cached  ~22 s
pytest -m forge      # ONE real Forge game (~10 s; needs ~/.mana-map/forge)
make test-fresh      # same, nothing served from the cache            ~29 s
make test-browser    # playwright, -n 4, plus the one serial_only test ~4 min
make test-all        # test-fresh + test-browser
pytest -n0 -k NAME   # a single test; worker startup outweighs the split
pytest -m ""         # literally everything, browser included         ~10 min
pytest --lf          # only what failed last time
```

**A bare `pytest` is `make test`** — `addopts` carries `-m 'not browser and not forge' -n auto`.
It used to be all 1,624 cases including the browser suite, because "browser is
excluded by default" was written in a comment and in no config anyone ran.

Three of those numbers were wrong in three different files before 2026-08-15
(~19 s, ~32 s and ~58 s for the same command). They are measured on an idle
8-core machine and they will drift again; the point of stating them here and
nowhere else is that there is one place to correct.

**This file is the only place that states test counts.** They move on almost every commit,
so restating them in `README.md` or `CLAUDE.md` guarantees drift; those files point here
instead. To print the current numbers rather than trust a snapshot:

```bash
.venv/bin/python -m pytest -m "not browser" --collect-only -q | tail -1
```

### Measured 2026-08-15, idle 8-core machine

| | |
|---|---:|
| `make test` — cold cache | **36 s** |
| `make test` — warm cache | **22 s** |
| `make test-fresh` | **29 s** |
| serial, before any of this work | 86.5 s |
| `make test-browser` (`-n 4`) | 234 s |
| **a fresh clone, cold** | **20 s** (1,360 passed, 129 gate-skipped) |

A fresh clone is FASTER than a developed checkout, which is not a paradox: 129
cases gate on gitignored artifacts a clone has not generated, and the expensive
`requires_data` ones are among them.

**That fresh-clone number is measured by actually doing it** — `git clone` into
an empty directory, `make setup`, `make test` — and the first time anyone did,
**23 tests failed**. Five read the pool through `card_pool` (the only reader of
the gitignored `cards.csv`) and eighteen validate `strategy:` citations against a
DB that is built locally. All 23 were correct tests with missing gates, and no
amount of running the suite on a developed machine could have found them: the
artifacts were always there. Re-clone and re-run whenever you add a test that
touches `data/`.

As of 2026-08-21: **1,843 tests** across 81 files — 1,706 fast, 136 browser and 1 `forge`
(a real Forge game, opt-in). One is a deliberately unmet `xfail(strict=True)` ship gate in
`test_embedding_quality.py` (see below); it is a target the code has not reached, not a
broken test.

Why the count cannot be checked mechanically: **433 of those cases do not exist in the
source** — there are 1,410 `def test_` functions and 1,843 collected cases, the difference
being parametrization over lists computed at collection time. The only way to count them is
to run pytest, and running pytest from inside pytest recurses. (That subtraction is the
cheap way to re-derive the figure: `grep -rhcE "^(async )?def test_" tests/*.py` against a
collection total.) `tests/test_docs_counts.py` guards every count
that *can* be derived cheaply (subcommands, agents, skills, decks, routines, pipeline
steps) and deliberately leaves this one to editorial discipline.

## Import `conftest`, never `tests.conftest`

Two files imported `from tests.conftest import …` and collected fine under `make test`,
which runs `python -m pytest` — that form puts the repo root on `sys.path`, so `tests` is
importable as a package. The **console script** does not, so every invocation CLAUDE.md
documents (`.venv/bin/pytest -n0 -k NAME`, `.venv/bin/pytest -m ""`) died at collection
with `ModuleNotFoundError: No module named 'tests'`, taking the whole run down rather than
those two files — a collection error is not a test failure and does not isolate. The other
78 files use bare `from conftest import …`, which works under both. Match them.

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
lazily, so every non-browser test avoids paying for it.

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
whole file skips cleanly, so a fresh clone still runs everything else.

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

The nastiest variant is a call that **returns early and silently**, because then there is
nothing slow to wait for and the test does not fail where the mistake is.
`mapRenderer.setCamera` no-ops when `baseFit` is null, and `baseFit` is built on the first
`setLayers` — so a test that moved the camera before the renderer was ready simply did not
move it, and went on to measure the FITTED view believing it had zoomed in.
`test_canvas_draws_density_contours` then read a halo-saturated baseline (38.7 ink instead
of 3.8) and reported it as "the contours stopped drawing", about one full-suite run in
three. In isolation with output attached it was byte-identical five times running, which is
exactly what kept it looking like a fixed-wait problem.

Two things worth copying from the fix. **The readiness probe was the renderer's own**:
`getCamera()` returns null under precisely the condition that makes `setCamera` a no-op, so
there was no need to invent a signal. And **it belongs in the fixture, not the test** —
`canvas_page` waited for the canvas element and the data but not for the fit, so every test
on that fixture had the same hole and one of them happened to be sensitive enough to show
it. Do not fix a shared-fixture race in the one test that caught it.

A failed attempt is worth recording too: moving the `setCamera` call *inside* a
`wait_for_function` poll made it strictly worse (~3 runs in 6), because the predicate runs
every animation frame and re-applying the zoom transform continuously left the camera back
at the fit. Retry-until-it-takes is not a substitute for waiting until it *can* take.

## Measure where the claim is, not across the whole surface

`test_the_spotlight_actually_dims_the_canvas` counted green pixels over the entire canvas
and asserted the count drops when you clear a spotlight. It failed consistently — and the
feature was working the whole time.

Clearing a spotlight does two things at once: it rests the line you were looking at, **and
it un-mutes every other verified line**. On goblin-storm those cancel almost exactly: 868
green px spotlit against 833 cleared, a ratio of 0.96 sitting right in the range the test
reserved for the bug it guards. The bounding boxes say what the totals hide — spotlit,
green occupies 47x294 px; cleared, it is spread across 591x687. Restricted to a box around
the spotlit line's own cards the same click reads 1024 -> 301.

**A global aggregate cannot distinguish two changes that move it in opposite directions.**
The test now reads `Force.spotlitRows` and measures only near those cards. When a pixel
assertion covers more surface than the claim does, a compensating change elsewhere will
either mask a real regression or manufacture a fake one, and you cannot tell which from the
number alone.

Its threshold is also measured on **both** sides rather than fitted to the healthy one:
healthy runs give 0.204 / 0.255 / 0.422 / 0.439, and with the resting ink disabled
(`Stage.INK.verifiedQuiet = Stage.INK.verified`) 0.589 / 0.601 / 0.617 / 0.695. 0.52 sits
between them. Deliberately noted in the test: that simulated bug is only PARTIAL, since the
resting state also halves the stroke weight, so a real regression in both would sit near
1.0. A threshold chosen only from passing runs tells you nothing about what it catches.

## The browser suite runs in parallel, and getting there found five latent flakes

**`pytest -m "browser and not serial_only" -n 4` is 232 s against ~600 s serial** — 2.6x,
and it works with no fixture surgery because `viz_server` already picks a free port per
session and `browser` is session-scoped, so xdist hands each worker its own of both.

The interesting part is what parallelism *exposed*. Five tests failed under load and every
one was a fixed `wait_for_timeout` measuring the machine rather than the behaviour — the
mistake this file has now documented four separate times:

| test | read | should have been |
|---|---|---|
| `test_the_drill_button_reports_what_it_would_do` | the previous status string | wait for the status to change |
| `test_canvas_redraws_when_the_filter_changes` | the boot status | wait for the status to change |
| `test_clicking_a_cluster_label_zooms_and_filters` | a camera mid-transition | poll until the span settles |
| `test_hover_shows_a_card_image_at_the_cursor` | an empty `src` | wait for the attribute our own code writes |
| `test_canvas_draws_density_contours` | the fitted view, again | verify the camera took; re-set if a late render refit it |

The last one is worth reading twice. The `canvas_page` fix — waiting for `getCamera()` to
answer, so `baseFit` exists — is **necessary but not sufficient**: selecting the map starts
a render that ends in a fit, and under `-n 4` that render can land *after* the test's camera
move and silently undo it. The fix is a bounded retry with a settle between attempts, which
is NOT the thing that failed before: putting `setCamera` inside a `wait_for_function`
predicate re-applies the zoom every animation frame and pins the camera at the fit.

**One test is marked `serial_only` and deselected from the parallel run.**
`test_canvas_render_beats_the_plotly_budget` asserts a 30 ms wall-clock budget; sharing a
machine with three other Chromiums it measured 41 ms while the renderer was unchanged. A
performance assertion under contention measures the contention. Run it with
`pytest -m "browser and serial_only"`.

Serial still passes — the fixes are condition waits, so they are strictly more robust on an
idle machine too, and none of them depends on being run in parallel.

## Inventories drift silently, so generate the comparison

Five documentation defects in one pass, none findable by reading the file that
contained them — each needed the doc compared against the code:

- `eval_embeddings.py` claimed **"Step 14"**, a number `viz_index.py` already owned, so the
  pipeline had two step 14s and no step 15 while every doc referred to "step 15" for exactly
  that module. `train.py` said "Step 4" where the registry says "4a".
- **Nine of the pilot commands** existed only in the CLI (`mana-analysis`, `scenario-facts`,
  `diagnosis-report`, `merge-prose`, `cache-snapshot`, `cache-rerecord`, three validators).
  A reader looking for "how do I check the goldfish declaration" found nothing and would
  reasonably conclude there was no way — which is how a validator gets written twice.
- **`data/decks/radagast/None`**: 25 KB of superseded deck map, written by an early run that
  handed `resolve_out_path` a literal `None` (every call site guards with `if out:` now).
  Tracked for weeks, read by nothing, city labels all null.
- `docs/viz.md` documented **`?renderer=canvas`** as a live switch in a heading; nothing in
  the JS has read it since Plotly was deleted. A flag that outlives its migration reads as a
  supported option.
- Its file table was out by up to **35%** on line counts (`force.js` ~980 vs 1,323) and was
  missing three files entirely.

All five are now `test_docs_section_count.py` guards that compare against the registry, the
CLI, `git ls-files` and the filesystem — never against a second hand-kept list, which is the
same argument `build-index` makes for the deck manifest. **The guard's own false-positive
rate was measured before it was kept**: the file-existence check first fired on
`docs/testing.md` naming `test_viz_camera.py` to say it was *retired* — a correct mention —
so it was narrowed to look at the prose around the reference, re-measured to zero false
positives across every live doc, and then re-checked by reintroducing the real defect to
confirm it still trips.

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

`browser` and `viz_server` live in `tests/conftest.py`, where pytest discovers them
without an import. They must **not** live in `conftest_viz.py` to be imported per module:
a fixture imported into two modules is **registered twice**, so two files importing
`browser` open two concurrent `sync_playwright()` contexts and every browser test errors
at setup.

That failure only appears in a full run. Each file passes alone, which is the worst shape a
failure can have: the obvious debugging move (run the failing file) makes it disappear.

Playwright is still imported lazily inside the fixture body, so the non-browser
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

## The whole suite, by file

Descriptions rather than counts — a count next to a filename is a promise to update it on
every commit, and that promise is not kept. Run `--collect-only -q` for live numbers.

**Card pipeline — pure functions, no artifacts needed:**

| File | Covers |
|---|---|
| `test_extract.py` | Multi-face cards, derived columns, supertype classification |
| `test_preprocess.py` | Vocab building, encoding, normalisation, multi-hot |
| `test_mechanical_tags.py` | Every tag regex, removal edge cases, multi-hot encoding |
| `test_synergy.py` | Rule matching, bidirectionality, combo exclusion, playability ranking |
| `test_power_creep.py` | Strictly-better detection, tiered similarity gate, stat parsing |
| `test_combos.py` | Combo extraction, graph building, dedup |
| `test_cluster_regions.py` | Region naming (colour/type/guild/TF-IDF), geometry, dedup |
| `test_card_roles.py` | Role classification, type-line mana disambiguation, coverage floors |
| `test_analysis_common.py` | Colour-identity masks, name index, vectorised top-k |
| `test_ingest_common.py` | Gzipped raw dumps, with an uncompressed fallback |

**Card pipeline — needs a pipeline run (skip-guarded):**

| File | Covers | Gate |
|---|---|---|
| `test_pipeline_integration.py` | Cross-artifact count consistency, output quality | per-file `requires_file(...)` |
| `test_find_similar.py` | Binary-format fidelity, L2 normalisation, 128D vs 2D ranking | module-level `requires_data` |
| `test_embedding_quality.py` | Recall/rank floors against the golden set, plus one `xfail(strict=True)` ship gate | `requires_data` |
| `test_viz_index.py` | `viz_index.json` + `neighbours.bin`: shape, ordering, embedding-sha agreement | `requires_data` |

**Pilot — build:**

| File | Covers |
|---|---|
| `test_pilot_build_deck.py` | Pool hard filters (bracket, identity, bans), scoring, slot filling with alternates, decklist naming |
| `test_pilot_manabase.py` | Hypergeometric source counts, hybrid pips, effective-pip quorum, greedy land selection |
| `test_pilot_bracket.py` | Floor drivers, commander-assumption exclusion, two-card infinites, tutors-never-scored |
| `test_pilot_validate_build.py` | Card count, singleton, identity, per-role budget arithmetic, bracket cross-check |
| `test_pilot_pool_facts.py` | Depth vs castability, restriction-aware sources, combo containment dedup |
| `test_pilot_card_pool.py` | The single corpus reader and its views, checked against the reader it replaced |

**Pilot — publish:**

| File | Covers |
|---|---|
| `test_pilot_fetch_deck.py` | Decklist parsing, mocked Scryfall, exact printings, decklist-hash short-circuit |
| `test_pilot_rules_db.py` | CR chunker edge cases (TOC, subrules, examples, glossary) |
| `test_pilot_query_rules.py` | Semantic top-k, exact lookup, suggestions |
| `test_pilot_strategy_db.py` | Strategy chunker (ids, sources, parents), real-DB alignment |
| `test_pilot_validate_strategy.py` | Doc form, changelog contract, strategy citations |
| `test_pilot_validate_stack.py` | The citation contract, decision form, strategy dispatch, golden artifacts |
| `test_pilot_goldfish.py` | Seeded determinism, mulligan rule, target assembly |
| `test_pilot_mana_analysis.py` | Land classes, sources, producer kinds |
| `test_pilot_build_manual.py` | LEGACY renderer: section completeness, contract integrity, furniture, determinism, escaping |
| `test_pilot_build_index.py` | The manifest the browser reads instead of listing a directory |
| `test_pilot_validate_issue.py` | LEGACY gate: issue identity, section order, tier-costume integrity, card-name accuracy |
| `test_pilot_artist_credits.py` | Standout detection, per-entry counting, drop runs |
| `test_pilot_merge_prose.py` | One agent (`pilot-notes`) writing a file that also holds frozen legacy keys; every legacy key survives a merge |
| `test_pilot_validate_considering.py` | LEGACY gate on the frozen `considering.json`: exactly ten, none in the deck, claims verified |
| `test_pilot_validate_tutor_guide.py` | One wish per tutor, real fetches, legal targets |
| `test_pilot_validate_strategic_frame.py` | Frame form, engine `strategy_refs`, candidate-line status |
| `test_pilot_deck_map.py` | The constellation's balance bound — and its refusal to assert the linkage |
| `test_pilot_issue_length.py` | Words vs visible words; `<summary>` counts, collapsed bodies do not |
| `test_pilot_voice_lint.py` | **Cross-deck.** Who is supposed to be speaking, and whether nine decks sound alike |

`test_pilot_voice_lint.py` is the only pilot test that reads the whole fleet, and that is
structural rather than stylistic: `validate-issue` takes one slug and can never see a
pattern that exists only across decks. A formula is invisible in a single issue and
obvious in three — eight decks wrote a hot take and five opened *"Here is the thing…"*,
which no per-issue check could have caught.

**Pilot — diagnose:**

| File | Covers |
|---|---|
| `test_pilot_deck_audit.py` | The cited axis table and the engine-activation read |
| `test_pilot_deck_facts.py` | DFC colours, curve, restricted-mana classes, the `notes` traps |
| `test_pilot_deck_history.py` | Applied swaps derived from git, plus pending ones |
| `test_pilot_scenario_facts.py` | The deterministic scenario brief |
| `test_pilot_diagnosis_report.py` | The diagnosis rendered readable, deterministically |
| `test_pilot_validate_diagnosis.py` | Axis re-derivation, marginal prescription frame, computed `orphans_stack` |
| `test_pilot_validate_goldfish_targets.py` | Declared cards still in the 99; undeclared win lines reported |

**Pilot — the bench (2026-08-19):**

| File | Covers |
|---|---|
| `test_pilot_deck_notes.py` | The captain's log appends and stamps the decklist as it stood; `merge-debrief` by id; `validate-debrief` — the annotation may name nothing the note and the 99 do not; the `log` row runs the debrief gate |
| `test_pilot_prescribe.py` | A question's id is its hash; the merge writes answer keys only; `prescription:<id>` digests only the prompt; `record` refuses without a passing skeptic; stale prescriptions are form-checked only |
| `test_pilot_deck_versions.py` | A version is a content change not a commit; the log joins by the stamped sha; an uncommitted working list is reported not guessed; tags resolve; `restore` is a dry run unless asked |
| `test_pilot_deck_info.py` | The workbench view composes a bare deck without crashing and derives `next` from what is true |
| `test_sim_forge.py` | The Forge harness: `.dck` from the repo parser, seats, run id with seed, game splitting, argv, outcome parsing (round vs global turn, alternate win lines), dry run; one `forge`-marked real game |
| `test_sim_parse.py` | Logs → events → facts → aggregates on a real game: seats learned from assignment lines, tokens counted two honest ways, drain kills attributed, intervals, `validate-sim` re-proof |
| `test_sim_bridge.py` | `game_state` v2 form check; `validate-stack`/`scenario-facts` on v2; a board lifted at a CR step — lands exact, Morph unmorphs, tokens from first use, commander exit as `command`, hand an estimate, question left empty |
| `test_sim_opponents.py` | `fetch-opponent`: the EDHREC slug, the repo-format decklist, the on-disk shape with provenance |

**Pilot — infrastructure:**

| File | Covers |
|---|---|
| `test_pilot_agent_cache.py` | Fingerprint stability, per-key staleness, the shared charter contract, staleness diffs, record guards, exit codes |
| `test_pilot_card_refs.py` | The card-reference matcher and its ambiguity handling |
| `test_pilot_impact.py` | Reference/figure/target/zone staleness reporting |
| `test_pilot_memo.py` | One memo discipline; a rewrite must be noticed |
| `test_pilot_out_path_guard.py` | `--out` is slug-scoped, and every per-deck command uses the guard |
| `test_pilot_imports.py` | Every pilot module can *run*, not merely import |
| `test_pilot_artifact_freshness.py` | Every deterministic artifact equals a fresh recomputation |
| `test_pilot_manual_freshness.py` | Every tracked manual equals a fresh render |
| `test_pilot_tracked_artifacts_validate.py` | Every tracked agent artifact passes its own validator |
| `test_pilot_deck_manifest.py` | The contract between pilot artifacts and `viz/deck.html` |

**Docs and contracts:**

| File | Covers |
|---|---|
| `test_docs_counts.py` | Prose counts match the repo; no doc names a deleted module |
| `test_docs_section_count.py` | **Documentation inventory guards** (7). No prose restates the legacy section count or enumerates its ids; every step module agrees with `pipeline.STEPS` about its number and no two share one; `docs/pilot.md` lists every subcommand in `PILOT_STEPS`; every tracked per-deck file is documented somewhere; no live doc names a source file that does not exist |
| `test_decklist_parity.py` | The Python and JS decklist parsers agree on hand-authored fixtures |

**Frontend:** `test_viz_behaviour.py` (playwright, the real gate) plus the source-assertion
files `test_viz_{drill,deck_lens,viewer}.py` — see the section above on why the latter
cannot catch a rendering regression.

## Fixed sleeps in the browser suite — the remaining debt

`test_viz_behaviour.py` holds **49 unconditional `wait_for_timeout` calls
totalling 53.6 s**, down from 53 and 67.2 s. Four were converted to condition
waits (4000, 3500 and two 3000 ms): a boot into Explore and a mode switch now
wait on `mapRenderer.getCamera() !== null` — the renderer's own readiness answer,
already documented in `canvas_page` — and the two arrow-key walks wait on
`MM.browseSet.pos > 0`, which is the step itself rather than time passing.

**The other 49 are left deliberately.** Every conversion is a chance to write a
condition that is subtly not the thing being waited for, and this suite has
already paid for that lesson twice: putting `setCamera` inside a
`wait_for_function` predicate re-applied the zoom every frame and pinned the
camera at the fit, which was *worse* than the sleep. Convert them when a test
next needs touching, one at a time, each verified by three consecutive `-n 4`
runs — not in a sweep.

The wall-clock payoff is also smaller than the numbers suggest: 13.5 s of sleeps
spread across four workers is about 3 s of the 234 s run. The reason to convert
them is flakiness, not speed.

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
