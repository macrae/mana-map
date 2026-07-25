---
name: write-manual
description: Generate a deck's pilot's manual — evidence gathering, prose writing, and deterministic zine HTML build. Use when the user wants a manual (re)generated for a deck that has cards.json and verified stack scenarios.
---

# Write a pilot's manual (v2 pipeline)

Pipeline for `data/decks/<slug>/` → `manuals/<slug>.html`. Evidence tiers: ✓ rules-verified, ◆ data-derived, ★ coaching (see `docs/pilot.md`).

1. **Preconditions**: `cards.json` exists (build-deck-db skill) and at least one stack in `stacks/` has `checker.verdict == "pass"` (resolve-stack skill — a manual with zero verified lines is thin).
2. **Goldfish** (◆): `.venv/bin/manamap pilot goldfish <slug>` — regenerate whenever the decklist changed; curate `goldfish_targets.json` for the deck's key piece sets.
3. **Evidence pull**: spawn `deck-analyst` for synergy clusters, curve analysis, upgrade shortlists.
4. **Strategic frame** (★): spawn `strategy-researcher` with `MODE: consult` and the slug, asking for the deck's strategic frame. Save its JSON to `data/decks/<slug>/strategic_frame.json` (tracked). Queue its `candidate_missing_lines` for resolve-stack runs (ask the user which to resolve now); carry its `gaps` list as topics for the next research-strategy pass.
5. **Coaching** (★): spawn `pilot-coach` for `threat_assessment` + `matchups` prose (grounded in the goldfish numbers, verified stacks, and the strategic frame) and decision scenarios → save decisions to `decisions/NNN-*.json`, validate with `manamap pilot validate-stack <slug>`.
6. **Prose**: spawn `manual-writer` for cover/how-it-wins/combo-line intros/card-roles/mulligan/upgrades (it also receives the strategic frame). Merge the coach's `threat_assessment` and `matchups` keys into the same `manual_prose.json`. Zero-guessing rule throughout; surface "needs a stack scenario" flags to the user.
7. **Build**: `.venv/bin/manamap pilot build-manual <slug>` then `.venv/bin/manamap pilot build-index` — deterministic; only verified stacks appear; missing prose renders [TODO].
8. **Review**: open `manuals/<slug>.html`; decisions, coaching sections, and the strategic frame are the founder-review surface (tracked JSON, red-linable).

`manual_prose.json` is tracked and human-editable — tune voice directly and rebuild without re-running agents.
