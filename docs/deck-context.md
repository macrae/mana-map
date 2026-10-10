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

The `summary` block is the header: commander, version and status, colours, lands and the
bracket floor, then one **Combos** line from `info.combos` (`deck-combos --write`'s summary)
— `7 known lines (5 infinite, 1 two-card) · 312 one card short`, or `not computed` naming the
command, the same word the bracket floor uses when its artifact is missing. It is the line
`/jarvis` answers "what combos does it have" from; the lines themselves are
`deck-combos <slug> --json`.

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

## On the deck page

`viz/deck.html` renders the context as the dossier's **first panel** (`has.context` in
the manifest, `MANIFEST_VERSION` 5) through `viz/js/context-md.js`: a small, escaped
renderer for exactly the subset the Keeper writes. Card links become hoverable
`a.cardref`s that open the Atlas. **Cards by role** gets a filter bar (name, colour,
type, mana value) over the deck's `cards.json`, fetched only when a context exists; a
role written as a paragraph of names filters like a list.

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
- **`deck-change`**: after a merge, a check-in or a `save-version`. It re-places cards and **prunes** the prose a swap
  made false, rather than appending to it.

**The hook.** `deck-branch merge --write` and `check-in --write` call
`deck_context.list_change` the moment the list is written. It refreshes the generated
blocks, reads the copies that moved from the `.txt.bak` the command just left, and
prints `CONTEXT STALE` with the Keeper's `deck-change` spawn (ins and outs named) and the
`--install` line. A command cannot spawn an agent, so the PRD's "updates automatically"
means Jarvis runs that pass as its next step, and says so.
On a bench or brewing deck, `edit` refreshes the generated blocks after each edit and
says the prose is behind the list, offering the Keeper at save; `save-version` prints the line
(`next (Jarvis): …`), and Jarvis runs the pass only after a save.

## Response-time targets

Every PRD sub-agent declares `sla_s:` in its charter frontmatter (`context-keeper`
120, `data-analyst` 30). The job band shows each running agent's elapsed time against
its target, and turns yellow when it goes over. Each finished run is appended to
`.progress/sla-log.jsonl`, and `manamap pilot sla-report` summarises the log. Only a run
spawned BY ITS TYPE is logged: the band reads the target from `.claude/agents/<type>.md`
for the `subagent_type` the agent was spawned as. A charter pasted into a
`general-purpose` agent is untimed. That is why the log held one row on 2026-10-09: the
timed trials of card-scout, strategist, rules-question and scenario-sim ran that way,
because their charters were added mid-session and are not spawnable until a restart. The band also times every prompt from submit to the first piece of the response
(`.progress/latency-log.jsonl`; the PRD's "under 2 s"), and `sla-report` prints its median, p90
and misses. It warns when the band Claude Code is running is not the repo's (a plugin edit
needs a version bump and `claude plugin update`).
