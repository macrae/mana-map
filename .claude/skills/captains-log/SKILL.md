---
name: captains-log
description: Render a deck's captain's-log notes as a ship's log in Picard's register — one entry per night flown, with the pilot's own note preserved verbatim behind it. Use after logging a game, or to render nights that have none.
---

# The captain's log (the rendering loop)

Turns `log.jsonl` — the pilot's own 300-600 word notes — into
`data/decks/<slug>/captains_log.json` (tracked): one entry per night, six
sections each, in the register of a ship's captain dictating after the fact.

**This is a rendering, not a replacement.** The note stays authored, tracked and
reachable behind every entry on the deck page, and the `debrief` annotation
remains the machine-readable reading that the doctor consults. If a night has no
debrief, run `/debrief` too — the two read the same source and answer different
questions.

## Loop

0. **What is outstanding** — free, and it is the whole brief:

   ```bash
   .venv/bin/manamap pilot captains-log <slug>          # nights, and which are rendered
   .venv/bin/manamap pilot captains-log <slug> --json   # the skeleton the agent quotes
   ```

   Every fact the agent needs is in that JSON: the stardate, the grouping, the
   version sleeved, where the night sat in an evening that spanned several decks.
   **The agent computes none of it.**

1. **Cache gate** — check before spawning:

   ```bash
   .venv/bin/manamap pilot cache-status <slug> --routine captains-log
   ```

   exit 0 = current, report and **do not spawn**. exit 1 = run it. exit 2 = the
   deck has no logged games; there is nothing to render and nothing to cache.

2. **Model**: spawn `captains-log` with the slug, scoped to the night keys that
   read `NOT YET RENDERED`. It writes `.agent-out/captains-log.json`.

3. **Merge, then validate** — in that order, because the validator reads the
   tracked file:

   ```bash
   .venv/bin/manamap pilot merge-captains-log <slug>
   .venv/bin/manamap pilot validate-captains-log <slug>
   ```

   The merge recomputes the skeleton and takes **only** the six prose sections,
   so an invented stardate never lands. A FAIL goes back to the agent with the
   errors; **do not hand-patch the prose** — editing prose to satisfy a check
   puts a fresh claim under an old byline.

4. **Record**, last, only once the validator passes:

   ```bash
   .venv/bin/manamap pilot cache-record <slug> --routine captains-log
   ```

5. **Read the `NOTE` lines.** The validator prints, and does not fail on, four
   checks that could not be proved harmless before there was any prose to
   measure: shouty capitals carried through from the source, orders not phrased
   as already issued, superlatives, and jargon. They are the abstraction layer's
   real failure modes. Read them against the prose; a note that is right is a
   revision, and a note that fires on correct prose is a check to delete.

## What the gate can and cannot see

`validate-captains-log` recomputes the grouping, the stardates and the evening
positions and compares them; it holds the header to quoting the stardate and the
version verbatim; it enforces `self -> ship -> circumstance` in the assessment,
the closed station vocabulary, all six sections present, and no exclamation
marks. Every one of those is possible **only because the skeleton is
deterministic** — the more Python computes, the more a validator can hold to
account without judging a word of prose.

**It cannot see whether the log is any good, and it cannot see a finding that
went missing.** A note that names a change in direction can be absorbed into a
graceful sentence and lost. That is the one failure worth reading for, and the
reason the raw note sits behind every entry rather than being replaced by one.

## Notes

- **A decklist edit does not stale this artifact.** A log records a night that
  happened; a swap on Tuesday does not make Saturday's log wrong. `cards:semantic`
  is deliberately not a cache input and there is no `STAGES` row.
- **A change to `captains_log.py` does stale it** — `repo:CAPTAINS_LOG_PATH` is
  declared, because editing `STATIONS` or `stardate()` changes what the agent may
  write. That is a re-spawn, not a re-bless.
- `captains_log.json` is **tracked**; the `--json` skeleton is a view and is
  never committed, same rule as `deck-facts`.

**Agent output arrives as a path, not inline JSON.** Read the file, merge it,
then validate.
