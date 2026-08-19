---
name: refresh-corpus
description: Pull a fresh Scryfall dump and regenerate the whole corpus — the full pipeline, every gate, the doc sweep, the cache re-bless pass, and the deploy checklist. Run before building any new deck, and whenever a new set or Secret Lair drop lands.
---

# Refresh the corpus (the full-pipeline runbook)

A fresh Scryfall dump changes the card count, and index alignment
(`projection[i] == cards.csv[i] == embeddings[i]`) makes a partial regeneration
incoherent — so a refresh is always the FULL `manamap run`, retrain included
(~1.5–2h wall, dominated by the ~1h ability-model retrain on MPS). This runbook
exists because the cost is not the pipeline: it is the dozen sharp edges around
it, each of which has cut someone once.

**Trigger policy:** run this before starting any new-deck build; after any set
release or Secret Lair drop you care about; never casually. **Preview-season
caveat:** Scryfall marks unreleased cards `not_legal` until release day, so a
refresh during spoiler season imports cards the builder and bracket engine will
refuse. If the point of the refresh is a new commander, wait for release or plan
a second run.

## 1. Preflight (all free)

- `git status` — clean tree. The run rewrites ~13 tracked files (~100 MB of
  diff); do not mix it with unrelated work.
- **Confirm the dump actually moved**: `download` is a NO-OP when
  `data/.download-meta.json` still matches Scryfall's `updated_at`
  (`src/manamap/ingest/download.py:33-38`). Check the catalog date against the
  sidecar; if you must force, delete the sidecar.
- Save the current eval baseline for before/after:
  `.venv/bin/manamap eval-embeddings` → keep the **test-split** numbers.
  (Never tune on them; a fresh dump also confounds before/after comparison —
  quote both facts when reporting.)
- `manamap pilot cache-status <slug>` on each deck — know the board's colour
  BEFORE the refresh so post-refresh MISSes are attributable.

## 2. The run

```bash
.venv/bin/manamap run          # all 15 steps; 1 (Scryfall) and 7 (Spellbook) need internet
```

Background it; it is safe to leave. If it dies mid-way, resume with
`manamap run --from STEP` — but never stop permanently between `extract` and
`viz-index`: a half-regenerated `data/` violates index alignment.

## 3. Gates (in this order, and read the failures — they are designed tripwires)

1. `.venv/bin/python -m pytest -m "not browser" -n auto` — the fast suite.
   - `test_embedding_quality.py::test_every_golden_card_still_exists` fails if
     the dump renamed any of the 163 hand-authored golden cards → hand-edit
     `data/eval/similarity_golden.json` deliberately (it must STAY hand-authored).
   - The **strict xfail** on neighbour spread (`> 0.05`) fails the suite if a
     retrain IMPROVES past the target. That is a decision, not a bug: remove the
     marker in its own commit with the measured number, never silently.
   - Regression floors (`MEASURED × 0.8`) catch a genuinely worse retrain — a
     failure here means investigate, not re-roll the seed.
2. `.venv/bin/python -m pytest -m browser` — REQUIRED, not optional: the corpus
   count assertions live here (derived from `cards.csv` via fixture, but the
   rendering checks only run in a real browser).
3. `tests/test_viz_index.py` sha gate — `neighbours.bin` must digest the live
   embeddings; if red, `manamap viz-index` did not run after `embed`.
4. `tests/test_pipeline_integration.py::test_game_changers_are_commander_legal`
   — zero flagged GCs means the oracle dump is stale.

## 4. The sweep nothing automates: corpus counts in prose

`tests/test_docs_counts.py` guards command/agent/skill counts but NOT card
counts. Update by hand (grep for the old total to find any new ones):

- `CLAUDE.md` — the header total, the supertype percentages (Creature share,
  Planeswalker, Battle counts), the discovery-boot notes if sizes moved.
- `docs/architecture.md`, `docs/data-artifacts.md` — totals, coverage ratios,
  file sizes. (`docs/history/deck-builder-v2.md` and `docs/history/frontend-v2.md` are frozen
  design records — leave them.)

## 4b. Region names: a re-cluster orphans authored names, and naming them is a STEP

`cluster-regions` matches hand-authored names in `data/region_names.json` by
`map|level|mechanical-label` signature. A fresh clustering changes signatures, so
dozens of L0/L1 regions come back unmatched — the step REPORTS them, and
`test_regions_are_named_three_levels_deep` fails until every L0/L1 is named.
Reuse direction-shifted families where the core matches (East Featherbourne →
North Featherbourne), author new names in the established register for the rest,
then re-run `manamap cluster-regions` (seconds). The 2026-08-12 refresh needed
96: 18 adapted, 78 authored.

## 5. DATA_VERSION (`viz/js/mana-map.js`)

Bump when a consumer would draw a **different conclusion** from the bytes. A
full run RETRAINS, so every embedding-derived value changes meaning → **always
bump on this runbook**. (A hypothetical frozen-model content refresh would not
need it — that path does not currently exist.) Script/CSS `?v=` tags: only if
JS/CSS changed, which this runbook does not do.

## 6. The cache pass (8 decks, and impact runs FIRST)

Regenerating `synergy_graph.json`, `obsolescence_index.json`, `card_roles.json`
and the combo files MISSes, per deck, `strategic-frame`, `deck-diagnosis`,
`candidate-pool`, `deck-build` and every `prescription:<id>` — read which tokens each
declares in `config.py` `AGENT_ROUTINES` rather than this sentence (`pilot-notes` no
longer hashes the graphs; `the-ten` is retired)
(`config.py` AGENT_ROUTINES). `cards.csv` itself is a declared input to none —
deck digests hash each deck's own `cards.json`, which a refresh never touches.

1. `manamap pilot impact <slug>` for every deck — BEFORE any `cache-record`;
   recording first destroys the diff baseline (learned twice).
2. For each MISSed routine: grep the artifact for numeric synergy/obsolescence
   ranks. Quotes none (the recorded 2026-08 precedent: nothing rules-verified
   depends on those graphs) → validate, then `cache-record` with the reasoning
   stated in the commit. Quotes a rank → **re-spawn, never re-bless**.
3. Never `cache-record` to make a board green. The record is a claim someone
   read the artifact and agreed it still holds.

## 7. Deploy checklist

1. Serve locally (`python -m http.server 8000` from repo root) and LOOK at it:
   the map renders, Find Similar returns sane neighbours for a known card (they
   SHOULD have moved — cached pre-retrain neighbours are the failure), drill
   works, obsolescence badges render. Badges and neighbours are where refresh
   regressions hide (memory-recorded lesson).
2. Rerun any live `pool-facts` analyses — new-set names should now resolve.
3. Commit: code/docs and the ~13 tracked `data/` artifacts (projections, .bin
   files, viz_index, regions, synergy/obsolescence, combo files). NO Git LFS —
   Pages serves pointers.
4. Push deploys. Confirm with the user first, per house rule.

## Failure modes worth naming

- "Download ran but nothing changed" → the sidecar matched; see Preflight.
- Suite green but viz shows old neighbours → browser cache; `DATA_VERSION`
  bump missing or not deployed.
- A new set's commander refuses to build → preview-season `not_legal`; this is
  Scryfall being correct, not a bug. Re-run after release day.
