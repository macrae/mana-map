# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what the bench is; `CLAUDE.md` carries the
rules; this says where things stand and what is open. The plan as it stood until
2026-10-05 — every dated block, and the pilot's PUNCHLIST of 2026-09-16 — is
[`docs/history/plan-to-2026-10-05.md`](docs/history/plan-to-2026-10-05.md).*

Last updated **2026-10-09**. Do not quote a figure from here without re-running the
command beside it.

## Where things stand

**PRD v2 is built except for the deletions.** One living Deck Context per deck
(`data/decks/<slug>/CONTEXT.md`, `docs/deck-context.md`), `/jarvis` as the one entry
point, the lean agent roster with response-time targets, the hypothesis queue
(`docs/queue.md`), Forge scenario slices (`manamap pilot scenario-ab`), the live job graph
in the job band and the page-state beacon all shipped 2026-10-07..08. Sharknado's context
is approved; the other seven are in review. The engine model, the handbook (`poh*.py`,
`manuals/p/`) and the stack-resolution chain are still in the repo, because nothing is
deleted until every context is approved.

**Card and game support shipped 2026-10-08..09.** Banned vs not-legal and Game Changers on
the card panel; known combos per card and per deck (`deck-combos` -> `combos.json`); prices
as dated evidence (`prices` -> `prices.json`, from Mana Pool's public feed — no token —
with Scryfall as the fallback; `docs/integrations.md`); a printings picker and a Mana Pool
buy list; 60-card formats (`pilot/formats.py`), with the first Modern deck, `elves`;
Moxfield export, link and import by paste (`deck-export`, `deck-link`).

**The fleet.** Six decks are sleeved — edgar-vampires, gishath, goblin-storm, heliod,
sharknado, ur-dragon (`deck-version <slug>`); emiel-blink, meren-recursion and elves are
on the bench; the rest are retired or broken down. Every live deck and branch is priced.

**The push gate** (`make prepush`, `docs/testing.md`) runs only the tiers a diff can break
and leaves the fleet regen to CI. CI's browser job runs two workers (a hosted runner has
four cores); the corpus job runs weekly or by hand.

## Next, in order

1. **Finish the context reviews** (the PRD's Step 4). Verdicts go through a review page; a
   context marked for changes goes back to the Context Keeper.
2. **The deletions** (Step 6): the engine model, the handbook and its agents, the
   stack-resolution chain that fed them. The carve-outs are mapped: `info.json` and
   `deck_info.compose` stay, `validate_citations` moves out of `validate_stack`,
   `sim/forge.py` stays.
3. **Response-time logging for every agent.** `manamap pilot sla-report` has one recorded
   agent run; the PRD's "90% within target" cannot be measured until every agent logs.
4. **A corpus refresh** (`/refresh-corpus`): Scryfall has moved past the tracked corpus
   (Reality Fracture released 2026-10-02), which is what the weekly corpus job reports.
   A model change follows it, so `regen` and `make manuals` too — on the pilot's word.
5. **The PRD's open questions**: the context length limit before pruning, whether a pilot
   log starts the incubation pod or Jarvis asks, how the queue is prioritised, and the
   final per-agent targets.

Still open from 2026-10-05 (`docs/history/plan-to-2026-10-05.md` has the detail):
edgar-vampires' diagnosis re-run after the static-lord fix, sharknado's Forge piloting,
the hoard_6/hoard_10 axis overlap (`docs/known-issues.md` 9b), a goldfish switch for
goblin-storm's ritual-into-commander, and edgar-vampires' Treasure flag.

## Decisions that bind

- **The frontend stays LLM-free**; the pipeline and pilot commands make zero LLM calls.
- **The goldfish decides, paired; Forge probes.** No Forge rate is graded without an A/A
  at the same N.
- **Never cut Vish Kal**, and nothing on a deck's keep list. Read every OUT list before any
  compute.
- **A flat +1/+1 anthem is too slow**; a card earns its slot by multiplying or scaling. No
  filler slots.
- **Versions come from git; tags are authored.** The paper lock is the one claim about
  cardboard.
- **Similarity comes from the function space; synergy is complementary, not similar.**
- **A retired deck is not a downstream target.**
