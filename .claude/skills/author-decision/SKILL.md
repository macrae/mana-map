---
name: author-decision
description: Author a table-politics decision scenario (tier-3 coaching artifact) for a deck's pilot's manual — archetypal board state, branches with signaling/coalition analysis, validated form. Use when the user wants a "what do I do here" spot turned into a reviewable decision spread.
---

# Author a decision scenario

Produces `data/decks/<slug>/decisions/NNN-<kebab>.json` (`kind: "decision"`). Schema: `docs/pilot.md`. Numbering continues from existing files in `decisions/`.

1. **Frame the spot**: board + **table** state (who's ahead, open mana, what's been signaled, life totals that matter) and one concrete decision question. Specific beats generic — a coachable spot names names.
2. **Coach it**: for an existing spread, first check
   `.venv/bin/manamap pilot cache-status <slug> --routine decision:<NNN>` — exit 0
   means the scenario framing and its evidence inputs are unchanged and the spread
   stands; do not spawn. Exit 2 means a required input is missing (the routine
   declares `goldfish_metrics.json`; `strategic_frame.json` is optional) — fix that
   first, don't spawn. Canonical exit-code table: `docs/agent-cost.md`. On exit 1
   (or for a brand-new spread), spawn the `pilot-notes`
   agent with the slug, the spot, and pointers to `goldfish_metrics.json` + verified
   stacks. It returns the decision JSON: 2-4 branches, each with `choice`, `line`,
   `signals`, `coalition_risk`, `coaching` (+ `citations` for any rules claim), and a
   `recommendation` matching a branch.
3. **Write + validate + record**: save the file, then
   `.venv/bin/manamap pilot validate-stack <slug>` — form errors go back to the coach.
   Once it passes: `.venv/bin/manamap pilot cache-record <slug> --routine decision:<NNN>`.
4. **Review note**: decisions are pure coaching (★). The tracked JSON is the review surface — flag new scenarios to the user for red-lining; founder review is the quality mechanism, not machine verification.
5. Rebuild when ready: `.venv/bin/manamap pilot build-manual <slug>`.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with a short summary. Read the file, validate it, then merge — never ask for the JSON in the reply. A 133 KB `candidate_pool.json` returned inline costs ~35k tokens of context for nothing.
