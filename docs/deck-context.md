# The Deck Context

**One living document per deck: `data/decks/<slug>/CONTEXT.md`.** Introduced by PRD v2
(2026-10-07). It is the first place `/jarvis` looks for anything about a deck, and
Sean reads it himself in a few minutes. It replaces the engine model, the Pilot's
Handbook and the dossier as the thing a person reads. Those stay on disk until every
deck's context is approved (Phase 4 deletes them).

## Two kinds of text in one file

| | Who writes it | Goes stale how |
|---|---|---|
| **Prose**: Summary, How it plays, Cards by role, Pilot notes, Open questions, notes under Change history | the `context-keeper` agent, installed through a code gate | stamped `<!-- ctx:written-for version=… sha=… at=… -->`; STALE once the 99 moves past that sha |
| **Generated blocks**: `<!-- ctx:gen summary\|numbers\|record\|history\|queue -->` … `<!-- /ctx:gen NAME -->` | `src/manamap/pilot/deck_context.py`, from `deck_info.compose`, `deck_versions.report` and `decisions.jsonl` | never: `regen`'s last stage (`context`) re-renders them, and a test compares them to a fresh render |

**A figure lives only in a generated block.** The Keeper's charter forbids it from
writing a number. Prose points at the Numbers section instead, so a model change or a
swap can never leave a stale figure in a sentence.

The template, in order:

1. the header and summary block;
2. `## Summary`;
3. `## How it plays`;
4. `## Cards by role`;
5. `## Numbers` (block);
6. `## Pilot notes` (block plus prose);
7. `## Change history` (block plus notes);
8. `## Open questions` (the `queue` block, then prose citing items by id);
9. `## Context changelog`.

`deck_context.SECTIONS` is the one statement of it.

## Cards by role

There is one `### <what these cards do in THIS deck>` heading per role, in plain
language and never a fixed vocabulary. A card may sit under several roles. **Every
card in the 99 is placed**, the commanders included. The install gate refuses a draft
that leaves one out. Cards are written `[[Card Name]]`, and the installer turns them
into ManaMap links (`https://manamap.seanmacrae.com/viz/index.html?cards=…`).

## The gate: why cut cards cannot hide

The engine model rendered cards the deck had stopped running (Kefnet, Treasure Map
and others on sharknado, 2026-10-07), and nothing said so. The gate reads every card
link back:

- **A card outside the 99 in Summary, How it plays or Cards by role is an ERROR**
  while the stamp is current, and a **named STALE warning** once the list has moved.
- Pilot notes, Change history and Open questions name cut cards and candidates on
  purpose, so they are not checked.
- **A generated block that differs from a fresh render is an ERROR.** Run
  `--refresh` to fix it.
- **Broken markers** (unclosed, nested, unknown, duplicated) refuse every rewrite.
- **Install** (`--install`) is strict. It requires all of the above, every card
  placed, no `_(to be written)_` and at most 400 lines. It keeps a `.prev` and
  appends a Context changelog line.

## Commands

```bash
manamap pilot context <slug>                         # print it
manamap pilot context <slug> --slice numbers plays   # only these sections: what an agent is sent
manamap pilot context <slug> --check                 # the gate (== validate-context)
manamap pilot context <slug> --scaffold              # a fresh file: blocks filled, prose placeholders
manamap pilot context --refresh --all                # re-render the blocks on every deck that has one
manamap pilot context <slug> --install data/decks/<slug>/.agent-out/context-keeper.md --note "…"
```

## The Keeper's three modes

The full charter is `.claude/agents/context-keeper.md`. Its target is under 2 minutes,
and it runs on Sonnet.

- **`seed`**: the first context. It folds the deck's older agent prose (engine,
  handbook prose, captain's log, recon, frame) into plain language. That prose is
  checked card by card against the 99, and **the pilot's own words outrank every
  agent's**.
- **`log`**: after `deck-notes add`. It folds the games into Pilot notes and returns
  1–3 open questions for Jarvis to *offer* Sean. Incubation asks first; it never
  starts unasked.
- **`deck-change`**: after a merge. It re-places cards and **prunes** the prose a swap
  made false, rather than appending to it.

## Response-time targets

Every PRD sub-agent declares `sla_s:` in its charter frontmatter (`context-keeper`
120, `data-analyst` 30). The job band shows each running agent's elapsed time against
its target, and turns yellow when it goes over. Each finished run is appended to
`.progress/sla-log.jsonl`, and `manamap pilot sla-report` summarises the log.
