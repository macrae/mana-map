---
name: write-manual
description: Generate a deck's pilot's manual — evidence gathering, prose writing, and deterministic zine HTML build. Use when the user wants a manual (re)generated for a deck that has cards.json and verified stack scenarios.
---

# Write a pilot's manual (v2 pipeline)

Pipeline for `data/decks/<slug>/` → `manuals/<slug>.html`. Evidence tiers: ✓ rules-verified, ◆ data-derived, ★ coaching (see `docs/pilot.md`).

0. **Cache gate** (do this first — the agents below cost ~200k tokens together):
   `.venv/bin/manamap pilot cache-status <slug>` prints one line per routine and exits
   0 only if every routine is current. Each agent step below is individually guarded;
   run its `cache-status --routine <R>` **before** spawning:
   - **exit 0** (`HIT` or `EDITED`) → **do not spawn**; the tracked artifact is current.
     `EDITED` means a human tuned it by hand and the inputs haven't changed — their
     edit wins.
   - **exit 1** (`MISS`) → spawn, write the artifact, validate it, and only then
     `cache-record`.
   Canonical exit-code semantics: the table in `docs/agent-cost.md` — skills restate
   only what they need.
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
5. **The notes** (★/✓): check `cache-status <slug> --routine pilot-notes`. On exit 0,
   keep the existing five keys and go to step 6. On exit 1, spawn `pilot-notes` —
   it reads `engine.json`, the frame, the goldfish figures and the verified stacks
   and writes `how_it_wins`, `mulligan`, `combo_lines[<stack>]`, `threat_assessment`
   and `matchups` in one technical voice. On a keyed MISS, `cache-status` names the
   `stale keys:`; scope the spawn to exactly those. Then
   `manamap pilot merge-prose <slug> pilot-notes` — the merge touches ONLY the five
   owned keys, so the frozen legacy keys on a published deck (`card_roles`,
   `mana_base`, `upgrades`, `editors_letter`, `pilots_log`) survive — then
   `validate-issue <slug>` (budgets, taxonomy leaks, the voice bans) and
   `cache-record <slug> --routine pilot-notes`. Zero-guessing rule throughout;
   surface "needs a stack scenario" flags to the user. Decision spreads and the tutor
   guide are the same agent under their own routines — `decision:<NNN>` (the
   author-decision skill) and `tutor-guide`.
6. **Build**: `.venv/bin/manamap pilot build-manual <slug>` then `.venv/bin/manamap pilot build-index` — deterministic; only verified stacks appear; missing prose renders [TODO].
7. **Review**: open `manuals/<slug>.html`; decisions, coaching sections, and the strategic frame are the founder-review surface (tracked JSON, red-linable).

`manual_prose.json` is tracked and human-editable — tune the wording directly and rebuild without re-running agents. The cache reports a hand edit as `EDITED` and still says "don't spawn"; run `cache-record` to bless it.

## After a deck change: regenerate the diff, not the manual

The cache is card-scoped. When the decklist changes:

1. `.venv/bin/manamap pilot impact <slug>` — FIRST, before any rebless (a
   rebless advances the card baseline and blinds the deck-diff). The report:
   which artifacts/keys/departments reference the changed cards, stale prose
   figures, goldfish-target ghosts, zone-framing flags. Report-only.
2. `.venv/bin/manamap pilot cache-rebless <slug>` — clears every `STALE_OK`
   routine (deck changed, but nothing that artifact references) without a
   single spawn. What remains MISS is real work.
3. For a keyed routine's MISS, `cache-status` names `stale keys:` — spawn the
   agent **scoped to exactly those keys** (the charters' Partial revision
   mode); merge, validate, record as usual.
4. Figures flagged by the impact audit are revised by the owning agent in a
   scoped spawn — never hand-patched into prose.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with a short summary. Read the file, validate it, then merge — never ask for the JSON in the reply. A 133 KB `candidate_pool.json` returned inline costs ~35k tokens of context for nothing.
