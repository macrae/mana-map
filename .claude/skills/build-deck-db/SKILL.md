---
name: build-deck-db
description: Ingest a Commander decklist into a deck database (cards.json) for the pilot subsystem — parse, fetch from Scryfall, validate the 100-card invariants. Use when setting up a new deck or after a decklist change.
---

# Build a deck database

```bash
.venv/bin/manamap pilot fetch-deck <slug>      # decklist.txt → cards.json
.venv/bin/manamap pilot validate-deck <slug>   # 100 cards, commander flagged, singleton, color identity
```

## New deck setup

1. `mkdir -p data/decks/<slug>/stacks` — slug is kebab-case (`goblin-storm`, `edgar-vampires`)
2. Write `data/decks/<slug>/decklist.txt`: one card per line (`1 Card Name`, `1x`, or bare name), commander under a `Commander:` header or with a trailing `*CMDR*`, `#`/`//` comments allowed
3. Run the two commands above. `fetch-deck` fails loudly naming every unresolvable card — fix spellings in the decklist and re-run (idempotent; byte-identical output when nothing changed)

Multi-face cards may be listed by a single face name ("Bonecrusher Giant"); the stored name is always the full `" // "` name, matching the combo/synergy/obsolescence graph key convention.

If the deck directory or decklist is missing, commands fail gracefully with the exact path to create — relay that to the user rather than inventing a decklist.
