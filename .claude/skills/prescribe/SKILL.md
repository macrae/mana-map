---
name: prescribe
description: Ask the deck doctor one question about a deck — "I keep getting wrathed on five", "make it faster", "should I run Sol Ring" — and get a ranked, validated, skeptic-reviewed prescription (cuts priced, adds that close a named axis, up to ten). Reads the captain's log. Use when the pilot wants advice in response to a prompt or a pattern in their games, rather than the full diagnosis.
---

# Prescribe (one question to the doctor)

`data/decks/<slug>/prescriptions/<id>-<kebab>.json` — the question (authored) and
the answer (doctor ⇄ skeptic), one file, accumulating. Schema: the `deck-doctor`
charter, MODE prescribe. The diagnosis contract applies in full; this is the
diagnosis scoped to what was asked.

0. **Facts** — free: `deck-audit <slug>`, `deck-facts <slug>`, `deck-notes <slug>
   list`. If the log has un-debriefed games, run `/debrief` first — the doctor reads
   `log_annotations.json`, not raw notes.
1. **Freshness** — as in `/diagnose-deck`: if `goldfish_metrics.json` or
   `mana_analysis.json` is stamped with another decklist, re-run them first.
2. **Open the question.** `.venv/bin/manamap pilot prescribe <slug> "<the question>"`.
   Prints the id. The same question (whitespace/case-insensitive) maps to the same
   id; an existing file is never overwritten.
3. **Cache gate.** `.venv/bin/manamap pilot cache-status <slug> --routine
   prescription:<id>`. Exit 0 → already answered under this deck; report it (`prescribe
   <slug> --list`). Exit 1 → continue. Exit 2 → a required input is missing.
4. **Spawn `deck-doctor` with `MODE: prescribe`**, the slug, the id and the prompt
   verbatim. It returns a path.
5. **Merge, validate:** `prescribe <slug> --merge <id>` then
   `validate-prescription <slug> --id <id>`. FAIL → back to the doctor with the errors.
6. **Skeptic.** Spawn `deck-skeptic` on the prescription (it writes
   `deck-skeptic-prescribe-<id>.json`); `prescribe <slug> --merge <id>` again to fold
   the verdict in; `validate-prescription` again. `fail` → doctor revises; loop to
   `DIAGNOSE_MAX_ITERATIONS`. A `pass` is what makes it recordable.
7. **Record.** `cache-record <slug> --routine prescription:<id>` — refused without a
   passing skeptic block.
8. **Report** the reading, the ranked adds with their natural cuts, the cuts with
   their price, and the open questions with their routes. Applying a swap is the
   pilot's act, by hand, followed by `fetch-deck`.

**Agent output arrives as a path, not inline JSON.** Read the file, merge,
validate — never ask for the JSON in the reply.
