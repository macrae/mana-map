---
name: write-manual
description: Generate a deck's pilot's manual — evidence gathering, prose writing, and deterministic zine HTML build. Use when the user wants a manual (re)generated for a deck that has cards.json and verified stack scenarios.
---

# Write a pilot's manual (v2 pipeline)

Pipeline for `data/decks/<slug>/` → `manuals/<slug>.html`. Evidence tiers: ✓ rules-verified, ◆ data-derived, ★ coaching (see `docs/pilot.md`).

0. **Cache gate** (do this first — these four agents cost ~330k tokens together):
   `.venv/bin/manamap pilot cache-status <slug>` prints one line per routine and exits
   0 only if every routine is current. Each agent step below is individually guarded;
   run its `cache-status --routine <R>` **before** spawning:
   - **exit 0** (`HIT` or `EDITED`) → **do not spawn**; the tracked artifact is current.
     `EDITED` means a human tuned it by hand and the inputs haven't changed — their
     edit wins.
   - **exit 1** (`MISS`) → spawn, write the artifact, validate it, and only then
     `cache-record`.
   - **exit 2** → a required input is missing. Stop and report; do not spawn.

   The order is always **check → (miss) spawn → write → validate → record**. Never
   record an artifact that failed validation — the next run must re-spawn it. Add
   `--force` to any check to rebuild deliberately.
1. **Preconditions**: `cards.json` exists (build-deck-db skill) and at least one stack in `stacks/` has `checker.verdict == "pass"` (resolve-stack skill — a manual with zero verified lines is thin).
2. **Goldfish** (◆): `.venv/bin/manamap pilot goldfish <slug>` — regenerate whenever the decklist changed; curate `goldfish_targets.json` for the deck's key piece sets.
3. **Evidence pull**: spawn `deck-analyst` for synergy clusters, curve analysis, upgrade shortlists.
4. **Strategic frame** (★): check `cache-status <slug> --routine strategic-frame`. On
   exit 0, reuse the existing `strategic_frame.json` and go to step 5. On exit 1, spawn
   `strategy-researcher` with `MODE: consult` and the slug, asking for the deck's
   strategic frame. Save its JSON to `data/decks/<slug>/strategic_frame.json` (tracked),
   then `cache-record <slug> --routine strategic-frame`. Queue its
   `candidate_missing_lines` for resolve-stack runs (ask the user which to resolve now);
   carry its `gaps` list as topics for the next research-strategy pass.
5. **Coaching** (★): check `cache-status <slug> --routine coach-prose`. On exit 0, keep
   the existing `threat_assessment` + `matchups` keys and go to step 6. On exit 1, spawn
   `pilot-coach` for that prose (grounded in the goldfish numbers, verified stacks, and
   the strategic frame) and decision scenarios → save decisions to `decisions/NNN-*.json`,
   validate with `manamap pilot validate-stack <slug>`, then
   `cache-record <slug> --routine coach-prose`. Decision spreads are cached separately as
   `decision:<NNN>` — see the author-decision skill.
6. **Prose**: check `cache-status <slug> --routine writer-prose`. On exit 0, keep the
   existing writer keys and go to step 7. On exit 1, spawn `manual-writer` for
   cover/how-it-wins/combo-line intros/card-roles/mulligan/upgrades (it also receives the
   strategic frame). Merge its keys into `manual_prose.json` **without touching the
   coach's `threat_assessment`/`matchups` keys**, then
   `cache-record <slug> --routine writer-prose`. Zero-guessing rule throughout; surface
   "needs a stack scenario" flags to the user.
7. **Build**: `.venv/bin/manamap pilot build-manual <slug>` then `.venv/bin/manamap pilot build-index` — deterministic; only verified stacks appear; missing prose renders [TODO].
8. **Review**: open `manuals/<slug>.html`; decisions, coaching sections, and the strategic frame are the founder-review surface (tracked JSON, red-linable).

`manual_prose.json` is tracked and human-editable — tune voice directly and rebuild without re-running agents. The cache reports a hand edit as `EDITED` and still says "don't spawn"; run `cache-record` to bless it. Rewording prose never invalidates the issue plan — the editor's cache hashes prose *structure*, not text.
