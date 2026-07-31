# Frontend v2 — a deck-building surface

*Status: proposed. Written 2026-07-26; **audited 2026-07-29; largely superseded 2026-07-31**.*

> **Superseded in the parts that describe the surface.** The front end was reframed around
> discovery — one card, typed relations, a graph you grow — and the map moved onto canvas.
> `viz_index.json` now exists (pipeline step 14), though carrying discovery's fields rather
> than the engine's. What still stands from this document is the *engine* analysis: the
> scorer diverges from `config.py`, saved decks persist raw row indices, and the port is
> still blocked on `data/cards.csv` being gitignored. Read `docs/viz.md` for what the front
> end actually is now.

> **Read this first.** The analysis below still holds — the engine is portable, the
> scorer diverges from `config.py`, decks persist raw row indices. Two things have
> changed since it was written:
>
> 1. **The sequencing is wrong.** It orders M1 (data) → M2 (engine) → M6 (`deck.html`),
>    but every deck artifact is already tracked and servable (~3.6 MB, seven decks,
>    uniform schema), so **`deck.html` has no prerequisites** — while the engine port is
>    blocked on `data/cards.csv` being gitignored. The dossier ships first. `PLAN.md`
>    carries the current order.
> 2. **M3's premise is stale.** It targets a 15-department layout and a component set
>    that predate the v3.2/v3.3 magazine work: there are now 17 sections in five acts,
>    with bylines, three columnists and L10. Port the *tokens* (they are unchanged);
>    re-read `issue_spec.py` and `STYLEv3.md` §5 before porting any layout.
>
> Also corrected by audit: embeddings are **not** loaded twice (fixed in `3d3edc6`); the
> int8 argument stands on its own. And M6 was re-scoped from a builder-record view to a
> **deck dossier**, because `build_plan.json` exists for exactly one deck (hapatra), so
> six of seven pages would have been empty.

## Context

The deck builder in `viz/js/deck-builder.js` is a prototype from before the Python builder
existed. It now has a better implementation sitting next to it in the same repo, and two
audits found the gap is wider than "the JS one is older":

- **The entire deterministic builder is portable to JS.** `build_deck.py`, `bracket.py`,
  `manabase.py`, `goldfish.py` and `validate_build.py` use numpy/pandas as convenience over
  arrays and dicts; every constant is a literal in `config.py`. Bracket floors would be
  byte-identical. Goldfish runs 10k iterations in ~200–500 ms in a Worker.
- **The blocker is 476 KB.** The browser is missing exactly four card fields —
  `game_changer`, `mechanical_tags`, `layout`, and `legal_commander` as a tri-state. That's
  a positional delta index measuring 476 KB gzipped, or 16% of one projection file.
- **~1 MB of finished JSON per deck is committed and unrendered.** `build_plan.json` carries
  every slot's six-component score and its runners-up with deltas; `bracket_report.json`
  carries display-ready driver sentences. `combo_details.json` (3.25 MB gz) and
  `card_roles.json` (376 KB gz) are tracked, served by Pages, and never fetched.
- **The magazine has the better design system.** `design.py` + STYLEv3 define tokens, a type
  scale and a validated component library — `power_meter`, `fast_facts`, `badge`, `callout`,
  `threat_box`, `card_figure`. The viz reinvented weaker versions in flat hex with zero CSS
  variables.

**Decisions taken** (asked and answered): built for one expert user — depth over onboarding,
no mobile or a11y push; a new builder surface with the map kept as-is; everything
deterministic runs client-side and hands off a brief to Claude Code for the judgment layer;
and the magazine's design system becomes the single design language.

## The problem worth solving

Building a curated 100-card deck today costs **~250 clicks**, each one triggering a full
`innerHTML` rebuild that resets scroll, collapses every expanded row and drops focus. The
alternative is ~12 clicks and no agency at all. **There is no middle gear.**

The middle gear is a change of primitive. Today you *accumulate* a deck by triaging
recommendations one at a time. Instead, **the deck starts complete** — the deterministic
builder already fills 63 slots by role budget and already keeps 2–3 scored alternates per
slot. So the interaction becomes *review and swap*, not *search and accept*. Most slots you
leave alone. Twenty deliberate decisions, not two hundred and fifty reflexive ones.

Two things make that loop feel like something, and both are newly possible:

- **Live consequence.** Swap a slot → goldfish re-runs → the curve moves. Nothing in the
  current tool tells you what a choice cost.
- **A live bracket floor that names its driver.** Swap in Mikaeus and the header changes to
  *"Bracket 4 — Mikaeus, the Unhallowed + Devoted Druid."* No deckbuilding site does this,
  because none of them has a rules-verified combo corpus behind it.

## Architecture

```
viz/
  index.html            explore (unchanged shell, fixes only)
  build.html            NEW — the builder surface
  deck.html             NEW — the dossier: render committed artifacts for a built deck
  css/
    tokens.css          NEW — ported from design.py
    components.css      NEW — the component library as real CSS
  js/
    engine/             NEW — ports of the Python deterministic core
      constants.js      GENERATED from config.py — single source of truth
      bracket.js  manabase.js  build.js  goldfish.js  validate.js
      worker.js         runs the above off the main thread
    ui/
      components.js     NEW — badge / power-meter / fast-facts / callout builders
      slots.js          the deck-as-slots surface
    mana-map.js         explore (targeted fixes)
    deck-builder.js     DELETED once build.html reaches parity
```

### The one rule that keeps this honest

**Two implementations of the same scorer is the bug we already have.** `deck-builder.js`
carries its own six-factor scorer with *different weights and a different sixth factor* than
`DECK_BUILD_WEIGHTS`, documented in two places with no cross-reference.

So: `viz/js/engine/constants.js` is **generated from `config.py`** by a new pipeline step,
never hand-edited. `DECK_BUILD_WEIGHTS`, `BRACKETS`, `COMBO_BRACKET_TAGS`,
`DECK_ROLE_BUDGET`, `DECK_ROLE_GROUPS`, `MASS_LAND_DENIAL`, `SYNERGY_RULES`,
`ROLE_PATTERNS` and the goldfish constants all cross the boundary from one home.

And a **parity test**: run the Python builder and the JS builder on the same brief in CI-less
pytest via a node subprocess, and assert identical `build_plan.json`. If they ever diverge,
that's a failing test, not a documentation problem.

## M1 — Data layer

**`src/manamap/export/viz_index.py`**, a new pipeline step after `card-roles`. Emits
`data/viz_index.json` in **cards.csv row order**, so `viz_index[i]` ≡ `projection[i]` ≡
`embeddings[i*128]` — the positional invariant already holds and is load-bearing.

Fields: `game_changer` (bool), `mechanical_tags` (the synergy term's fuel — weight 0.22, the
second largest, currently uncomputable in the browser), `layout` (split-card decklist
naming), `legal_commander` as a **tri-state** (today `reduce.py` collapses `banned` and
`not_legal` into "absent", so the browser cannot distinguish 83 banned cards from 2,617
not-legal ones, and `bracket.py`'s banned-cards note is unreproducible).

Measured: **3.03 MB raw, 476 KB gzipped.** Add to `.gitignore` negations and the `DATA` map.

Also start fetching the two tracked-but-unused files: `combo_details.json` and
`card_roles.json`. New-to-the-wire cost for a full builder session: **~3.86 MB gzipped.**

**And quantise the embeddings.** `embeddings.bin` is 17.6 MB raw and 16.3 MB gzipped —
float32 doesn't compress. Int8 with a per-vector scale is **4.3 MB** with negligible impact
on cosine ranking, and it's currently downloaded *twice* and held as two 17 MB
`Float32Array`s because `mana-map.js` and `deck-builder.js` each load it independently. Fix
both: one loader, int8. That's the single highest-leverage load change available.

## M2 — Port the deterministic core

Each module is a direct port, kept structurally parallel to its Python twin so the two can be
diffed by eye. All run in `worker.js`.

- **`bracket.js`** — the easiest and highest-value. `assess()` is set intersections and
  `max()`; `pandas` appears once, only to build a lookup dict. Output byte-identical to
  Python. Gives the UI a live floor with named drivers and cut candidates.
- **`manabase.js`** — `math` and `re` only. The binomial products stay under 2^53 for
  99-card draws so plain JS numbers are safe; use the log-gamma form anyway to remove the
  question. Port `manabase.py` *over* `generateManaBase`, don't extend the prototype — its
  own docstring exists to describe the two bugs that prototype has.
- **`build.js`** — pool filter, six-component scoring, role-budget slot fill with alternates,
  and the bounded `enforce_bracket` pass.
- **`goldfish.js`** — the best client-side opportunity in the system. **Determinism matters
  here**: the manual renders these as ◆ reproducible evidence, so a browser producing
  different numbers would quietly break the tier contract. Needs an MT19937 port matching
  `random.shuffle`'s reverse Fisher–Yates with `_randbelow` (~80 lines), and a test asserting
  the JS output equals the committed `goldfish_metrics.json` for goblin-storm and hapatra.
- **`validate.js`** — all of `validate_build.py` including the citation substring contract,
  since `strategy.md` is tracked at 73 KB and the check is a substring test, not a semantic
  one.

## M3 — The design system

Port `design.py`'s token block to `viz/css/tokens.css` and its component builders to
`viz/js/ui/components.js`. This is not new design work — the constitution exists, the tokens
are named, and the component set is validated against `issue_spec.COMPONENTS`.

The mapping is almost one-for-one with what a builder needs:

| Magazine component | Builder use |
|---|---|
| `power_meter` | bracket floor, on-curve probability per colour, goldfish assembly rates |
| `fast_facts` | the deck spec sheet — identity, curve, sources, floor, role budget |
| `badge` | tier badges (✓ ◆ ★) and role badges (`ramp:rock`, `removal:sweeper`) |
| `callout` | numbered explanations — why this slot, why this floor |
| `threat_box` | matchup panels when `strategic_frame.json` exists |
| `card_figure` | a card with a caption that teaches |
| `violator` | the bracket flag in the header |

The tool runs the magazine's **dark register** rather than cream paper — same tokens, same
type scale, same components, appropriate surface. Tier colours are fixed by STYLEv3 §8.3 and
must not be restyled.

## M4 — The builder surface (`build.html`)

Three regions. No modes, no hidden panels.

```
┌─ header ───────────────────────────────────────────────────────────┐
│ Hapatra, Vizier of Poisons   B/G      ▮▮▮▮ BRACKET 4  ← live       │
│ "Mikaeus, the Unhallowed + Devoted Franchise" ← the driver, named   │
├─ slots (main) ──────────────────┬─ inspector (right rail) ─────────┤
│ ▸ ramp        10/10             │  selected slot                   │
│   Sol Ring          .71  ⇄ 3    │  ├ card image + oracle           │
│   Arcane Signet     .68  ⇄ 2    │  ├ six-component score bars      │
│ ▸ removal      8/8              │  ├ roles, synergy labels          │
│   Assassin's Trophy .66  ⇄ 3    │  └ combo lines w/ produces        │
│ ▸ flex        21/21             │                                   │
│ ▸ lands       36/36             │  ── diagnostics (persistent) ──   │
│                                 │  mana: B 29/22 ▮▮▮ 96.2%          │
│                                 │  curve: ▁▃█▆▂▁                    │
│                                 │  goldfish: cmdr t2.0 · tutor 68%  │
└─────────────────────────────────┴───────────────────────────────────┘
```

**The slot is the primitive.** Each row is a filled slot showing card, score, and an
alternates count. Clicking `⇄ 3` opens the runners-up *inline* with their score deltas —
`build_plan.json` already carries them, and a delta of 0.01 is the signal that says "the
scorer was nearly indifferent here, so your judgment is cheap." That is the middle gear.

**Every mutation is incremental.** No `innerHTML` rebuild: swap one row, recompute in the
worker, patch the diagnostics. The current 14-call-site full-rebuild costs a scroll reset, a
collapse of every expanded row, a focus drop and a leaked document listener per render.

**Diagnostics are always visible and always live.** Swap a slot and the mana base, curve,
bracket floor and goldfish all update. This is the whole point: the tool should tell you what
a choice cost.

**The map becomes an input, not the only input.** From explore, a card gains "pin as
must-include" and "start a brief from this commander". From the builder, a slot can ask
"show alternates on the map." The map stops being the sole way to add a card — which is what
makes the current mobile build mode a dead end and forces the detail panel off-screen.

**Brief-first entry.** Commander, bracket target, must-include/exclude, playstyle note. It
serialises to exactly the `brief.json` the Python builder reads, so the two paths converge.

## M5 — The handoff

The browser does everything deterministic. The judgment layer — architect, critic, coach,
strategic frame, verified stacks, the magazine — needs Claude Code and cannot run here.

Make that a workflow, not a wall. A **Hand off to Claude Code** action that writes
`brief.json` + `decklist.txt` to a download and shows the exact command:

```
manamap pilot fetch-deck <slug> && /build-deck <slug>
```

The architecture already anticipates this: `build_deck.py`'s docstring says the deterministic
half "stands alone on purpose… a cache miss degrades to a worse deck rather than no deck."
The browser gets that baseline. The agent layer is the upgrade, and its outputs are already
committed as JSON — which is what M6 renders.

## M6 — The dossier (`deck.html`)

The cheapest substantial win in the whole plan: **fetch and render what is already
committed.** No new pipeline work, no recomputation.

- `build_plan.json` — slot table with score bars, per-card component breakdown, the swap
  history (56 entries for hapatra), `cut_for_bracket` with reasons, the alternates picker
- `bracket_report.json` — floor gauge with a full provenance trail; every driver is already
  a display-ready sentence
- `goldfish_metrics.json` — charts plus the nine `model_assumptions` verbatim, then a
  **"re-run with my changes"** button, because M2 makes that live
- `strategic_frame.json` — archetype, engines, matchup frames, and the `gaps` list
- `stacks/*.json` — verified lines with citations, each rule number **clickable**
- `candidate_pool.json` — the evidence pool, read-only (agent output, never regenerable here)

**Un-gitignore `rules_index.json` (1.34 MB) and `strategy_index.json` (81 KB).** Exact
citation lookup needs no model — it's a dict fetch — and it makes every `CR 704.5q` in every
manual clickable and verifiable. Semantic search stays Python-side; that needs MiniLM.

## M7 — Explore-mode fixes

Not a redesign, just the debt that makes the map worse than it is:

- **Turn on hover tooltips.** All 13 traces set `hoverinfo: 'none'` while building hover
  strings for all 34,322 points on every render — ~275,000 regex operations per render
  producing text that is then thrown away. Either show them or delete the work. Showing them
  is the single biggest UX improvement available anywhere in the app.
- **A search results list.** The 4-tier cascade is good ranking; it currently renders as
  indistinguishable white diamonds and a count in a 12px status line.
- **Keep the detail panel in build mode** — it is currently hidden exactly when you are
  deciding whether a card belongs in your deck.
- **`MM.render()` per-mutation cost**: ~34k regex ops and ~70k array allocations per deck
  change, growing with the deck. Cache the legality/identity masks; they don't change.
- **URL state.** There is none — no `location`, `history` or `URLSearchParams` anywhere. For
  a map whose value is "look at this region," being unable to link to a region is a
  first-order failure.
- **Migrate the localStorage schema.** Decks are stored as raw row indices into
  `projection_2d.json`; any pipeline re-run that changes card order silently reinterprets
  every saved deck. Store names + a schema version, and migrate or discard on read.
- **Link the two products.** `MM.selectByName()` already exists; deep-linking every card tile
  in a manual to its position on the map is a few lines in `design.py`'s `_card_tile`.
  Currently one hyperlink connects them, in one direction.

## Sequencing

M1 → M2 → M6 → M3 → M4 → M5, with M7 folded in opportunistically.

**M6 before M4 deliberately.** Rendering committed artifacts is low-risk, needs no new data,
and forces the component library into existence against real content before anything depends
on it. It also gives an immediate visible win on decks that already exist.

## Verification

- **Parity test**: Python and JS builders produce identical `build_plan.json` for the same
  brief. This is the test that keeps the two implementations from diverging the way the
  current scorer already has.
- **Goldfish determinism**: JS output equals the committed `goldfish_metrics.json` for
  goblin-storm and hapatra, byte for byte. If the MT19937 port is wrong this fails loudly.
- **Bracket parity**: JS `assess()` matches `bracket_report.json` for all three decks,
  including the commander-assumption exclusion that encodes Judge's Desk A-004.
- Index alignment: `viz_index[i].name` equals `projection[i]` name for all 34,322 rows.
- Payload budget: explore first paint stays under 3 MB gzipped; a full builder session under
  10 MB after int8 quantisation (down from ~19.6 MB today).
- A curated 100-card deck reachable in **under 40 deliberate interactions**, with no scroll
  reset and no lost expansion state.

## Risks

- **Two implementations of one algorithm.** Mitigated by generating `constants.js` from
  `config.py` and by the parity test — but it is the central risk and the reason both exist.
- **Goldfish determinism is fiddly.** Python's `random.shuffle` has specific semantics. If
  matching proves unreasonable, the honest fallback is to label browser-computed goldfish as
  an *estimate* and never let it overwrite a committed ◆ artifact.
- **No build step, by choice.** Hand-written browser JS, manual `?v=N` cache busting, and
  every new file needs a bump. More files means more discipline; a tiny generated manifest
  would help.
- **Scope.** M4 is a real application. M6 is not, and delivers most of the visible value —
  if the plan stalls, it should stall after M6, not before it.
