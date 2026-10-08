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
3. **One rebuttal, then the second look.** If any item came back `revise`, spawn
   `incubation-pod` with `MODE rebut` and those findings, then apply its draft. A
   withdrawal drops. A revision waits: spawn `challenger` with `MODE recheck` and
   the revised ids, then apply its draft. `yes` promotes, `no` drops with its
   reason. There is no second round, and the queue refuses one.
4. **Report** in a few lines: what was promoted (id and claim), what was dropped and
   why. Then ask Sean which, if any, to test.
5. **Refresh the deck's context block:** `manamap pilot context <slug> --refresh`.

## Sean adds his own (he said "add this to the queue" or "look into X later")

Turn his words into the fields a hypothesis needs — confirm them with him in one line
if anything is a guess — and add it:

```bash
.venv/bin/manamap pilot queue add --deck <slug> --claim "…" --expect "…" \
    --method context|argument|data|try|rules|strategy --how "…" --why "Sean, <date>"
```

It is INCUBATING, not promoted: run step 2 above (the Challenger, one round) on the
new id, and step 3 (rebuttal, then the second look) if it comes back `revise`. His claim skips the pod, never the round.

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
| `rules` | the `rules-question` agent |
| `strategy` | the `strategist` agent |
| `scenario` | the `scenario-sim` agent writes the spec and returns the board; show it to Sean, and on his OK run `manamap pilot scenario-ab --spec <path>` (about 20 s for 10 seeds); the answer line is the result, `DISCOUNT` included |

Write a result draft and apply it:

```json
{"kind": "result", "of": "Q007", "method": "try", "verdict": "supported|refuted|inconclusive",
 "answer": "one line, figures with their intervals", "evidence": ["the commands run", "stack ids"]}
```

Then tell Sean **what you found and what it means**, and ask for his call:

```bash
.venv/bin/manamap pilot queue decide Q007 stage|drop|watch|more --note "…"
```

`watch` on a result that is a LIST OF CARDS means: write it as a set in the deck's
watch list, so he can review it in the Atlas's Build mode (pick the deck, then the set;
the review grid under the map has Watch / Pass / Note):

```bash
.venv/bin/python -c "from manamap.pilot import watchlist as wl; wl.add_set('<slug>', '<set-id>', '<title>', [{'name': …, 'pays': 'both|brallin|shabraz|none|n/a', 'axis': 'interaction|momentum|ramp|protection|other', 'why': '…'}, …], source='Q007', query='<the command>')"
```

`stage` means he wants the swap staged. That is a separate step, taken only with his
go-ahead. **Never move on to the next item unasked.**

## Sean's own verbs

- `queue rank Q004 Q002 …` puts his order first.
- `queue kill Q003 --reason "…"` removes an item.
- `queue list --all` includes the closed and expired items.

An item no line has touched in 14 days reads EXPIRED, and any new line revives it.
