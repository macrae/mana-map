---
name: write-manual
description: Generate a deck's pilot's manual — evidence gathering, prose writing, and deterministic zine HTML build. Use when the user wants a manual (re)generated for a deck that has cards.json and verified stack scenarios.
---

# Write a pilot's manual

Pipeline for `data/decks/<slug>/` → `manuals/<slug>.html`:

1. **Preconditions**: `cards.json` exists (else run the build-deck-db skill) and at least one stack in `stacks/` has `checker.verdict == "pass"` (else run resolve-stack — a manual with zero verified combo lines is thin).
2. **Evidence** (optional but recommended): spawn `deck-analyst` for synergy clusters, curve analysis, and upgrade shortlists scoped to this deck.
3. **Prose**: spawn the `manual-writer` agent with the slug + evidence. Write its JSON output to `data/decks/<slug>/manual_prose.json`. The zero-guessing rule applies: combo lines only from verified stacks; roles/upgrades trace to graph entries or oracle text. If it flags "needs a stack scenario" items, surface them to the user as candidate resolve-stack runs.
4. **Build**: `.venv/bin/manamap pilot build-manual <slug>` — deterministic renderer; only verified stacks appear; missing prose keys render as visible [TODO] markers rather than failing.
5. **Review**: open `manuals/<slug>.html` (serve from repo root or open the file directly — the manual is self-contained except hotlinked card images). Rebuild after prose edits is byte-identical for unchanged inputs.

`manual_prose.json` is tracked and human-editable — the user can tune voice directly and rebuild without re-running the agent.
