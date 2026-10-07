---
name: incubate
description: Turn pilot feedback on a deck (new games, a question, the Context Keeper's open questions) into checked hypotheses in the fleet queue — the Incubation Pod proposes, the Challenger argues once, survivors are promoted — and work a promoted item with the lightest test that settles it, then hand Sean the call. Use when Sean says yes to "want me to look into these?", asks "what's in the queue / what did you find", or names a queue item to test. Never start it unasked.
---

# Incubate: feedback → hypotheses → the queue → a finding

The queue is `data/queue.jsonl` (`src/manamap/pilot/queue.py`, `docs/queue.md`). It is
append-only, and every state is derived from its lines. Agents never write it.
Everything goes in through `manamap pilot queue apply <draft>`, which checks each
transition and refuses an illegal one with nothing written.

## Incubate (Sean said yes)

1. **Propose.** Spawn `incubation-pod` with `MODE incubate`, the deck, and the source:
   the new log ids, Sean's question, or the Keeper's returned questions. Then:
   ```bash
   .venv/bin/manamap pilot queue apply data/decks/<slug>/.agent-out/incubation-pod.json
   ```
2. **Challenge, one round.** Spawn `challenger` with the new ids, then apply its draft.
   `apply` settles each item by itself: a `promote` verdict promotes it and a `drop`
   drops it with the reason.
3. **One rebuttal.** If any item came back `revise`, spawn `incubation-pod` with
   `MODE rebut` and those findings, then apply its draft. A revision promotes; a
   withdrawal drops. There is no second round, and the queue refuses one.
4. **Report** in a few lines: what was promoted (id and claim), what was dropped and
   why. Then ask Sean which, if any, to test.
5. **Refresh the deck's context block:** `manamap pilot context <slug> --refresh`.

## Work an item (Sean named it, or said "the top one")

`manamap pilot queue list` gives the order: Sean's ranking, then decks by how
recently they were played. Use the item's `test.method`, because it is the lightest
check that settles the claim:

| method | how |
|---|---|
| `context` | answer from `manamap pilot context <slug>` |
| `argument` | reason it out from How it plays and the cards; the reasoning *is* the answer |
| `data` | the `data-analyst` agent |
| `try` | `manamap pilot try <slug> --out "A" --in "B"`: read the paired row the claim names, with its interval |
| `rules` | the `rules-question` agent (until it lands: `/rules-lookup`) |
| `strategy` | the `strategist` agent (until it lands: `/strategy-lookup`) |

Write a result draft and apply it:

```json
{"kind": "result", "of": "Q007", "method": "try", "verdict": "supported|refuted|inconclusive",
 "answer": "one line, figures with their intervals", "evidence": ["the commands run", "stack ids"]}
```

Then tell Sean **what you found and what it means**, and ask for his call:

```bash
.venv/bin/manamap pilot queue decide Q007 stage|drop|watch|more --note "…"
```

`stage` means he wants the swap staged. That is a separate step, taken only with his
go-ahead. **Never move on to the next item unasked.**

## Sean's own verbs

- `queue rank Q004 Q002 …` puts his order first.
- `queue kill Q003 --reason "…"` removes an item.
- `queue list --all` includes the closed and expired items.

An item no line has touched in 14 days reads EXPIRED, and any new line revives it.
