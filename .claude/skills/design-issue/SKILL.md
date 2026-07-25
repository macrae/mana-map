---
name: design-issue
description: Design and build a complete Pilot's Manual issue for a deck — the magazine-editor agent plans the issue (cover, departments, headlines, furniture), the plan is validated mechanically, then the deterministic renderer builds the HTML. Use when creating or regenerating a deck's magazine issue.
---

# Design an issue (the magazine loop)

Turns `data/decks/<slug>/` artifacts into `manuals/<slug>.html` — a complete issue of
*Pilot's Manual*. Design authority: `STYLEv3.md`. Evidence contract: `docs/pilot.md`.

## Preconditions

- `cards.json` exists (build-deck-db skill)
- At least one stack in `stacks/` has `checker.verdict == "pass"` (resolve-stack)
- `goldfish_metrics.json` exists (`manamap pilot goldfish <slug>`)
- `manual_prose.json` exists (write-manual skill supplies body prose)
- `issue.json` exists — **authored by a human, never generated**: volume, issue_date,
  cover_price, deck_name, commander, cover_tagline, next_issue (STYLEv3 §4.1).
  A generated date would break byte-identical rebuilds.

## Loop

1. **Gather.** Confirm the preconditions above. Note which departments have thin or
   missing artifacts — they render `[TODO]`, never vanish silently.
2. **Plan.** Spawn `magazine-editor` with the deck slug. It reads STYLEv3 and every
   artifact, then returns the issue plan as JSON. Write it to
   `data/decks/<slug>/issue_plan.json` (tracked, human-editable).
3. **Validate.** `.venv/bin/manamap pilot validate-issue <slug>` — checks identity
   block, all 14 departments in canonical order, copy completeness, component library,
   tier-costume integrity, card-name accuracy, and rhythm. On failure, re-spawn the
   editor with the errors; do not hand-fix content beyond mechanical formatting.
4. **Build.** `.venv/bin/manamap pilot build-manual <slug>` then
   `.venv/bin/manamap pilot build-index`. Deterministic — rerun must be byte-identical.
5. **Review.** Open `manuals/<slug>.html` and run the STYLEv3 §12 checklist. The
   Five Promises and the contract checks are the ship gate.
6. **Report.** Surface the editor's `gaps` list: thin departments, candidate lines for
   resolve-stack, strategy topics for research-strategy.

## Notes

- `issue_plan.json` is the packaging layer (cover, headlines, captions, furniture);
  `manual_prose.json` is the body-copy layer. Both are tracked and hand-editable —
  tune a headline directly and rebuild without re-running any agent.
- Only checker-passed stacks render. A refuted line is still a feature — the honest
  finding is the best story in the issue (STYLEv3 §7.6).
- The Command Zone department is mandatory and must be format-specific. An issue that
  could be about any format has failed the Commander Mandate (STYLEv3 §3).
