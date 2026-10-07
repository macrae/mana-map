# The hypothesis queue

**`data/queue.jsonl`** (PRD v2 Phase 2, 2026-10-07) turns pilot feedback into checked,
testable claims, and records what testing them found. It is one fleet-wide,
append-only file. "What's in the queue?" is a single read, and the queue stays small.
The module is `src/manamap/pilot/queue.py` and the skill is `/incubate`.

## The loop

```
pilot feedback ──► incubation-pod ──► challenger (ONE round) ──► promote / drop
                   (≤3 hypotheses)     promote | revise | drop     │
                                         └─ revise ─► one rebuttal ┘
promoted ──► Jarvis tests it with the lightest method ──► result ──► Sean decides
```

- **The pod proposes** at most three claims per run. Each is specific enough to be
  wrong, carries its expected effect, and names the lightest test that settles it:
  `context`, `argument`, `data`, `try`, `rules` or `strategy`. `scenario` is refused
  until Phase 3 builds Forge scenario slices.
- **The Challenger argues once.** It asks four questions: is there a cheaper
  explanation, is it already answered, can the named test settle it, is it a
  duplicate. A `revise` gets one rebuttal from the pod. `queue apply` writes the
  promote or drop that follows from those lines itself, so the single round cannot be
  stretched.
- **Jarvis tests a promoted item** only when Sean asks, writes a `result` (with its
  verdict, a one-line answer and the evidence), and hands Sean the call: `stage`,
  `drop`, `watch` or `more`.

## State is derived, never stored

| State | From the lines |
|---|---|
| INCUBATING | a hypothesis and nothing else |
| CHALLENGED | challenged, not yet settled (waiting for the rebuttal) |
| PROMOTED | promoted, untested; or `decide more` sent it back |
| DROPPED | dropped, with a reason (the Challenger's, or the pod withdrew it) |
| TESTED | a result, waiting for Sean |
| DECIDED | Sean decided |
| KILLED | Sean removed it |
| EXPIRED | live, but no line in 14 days; computed from the dates, and any new line revives it |

`queue.check(line, lines)` is the one statement of the legal transitions. `apply`,
Sean's verbs and `validate-queue` all call it. A draft with an illegal line writes
nothing at all.

## Order

Sean's latest `rank` comes first. Then decks by how recently they were played (the
newest captain's-log entry), then age, oldest first. `list` shows the live items and
at most ten promoted ones.

## Commands

```bash
manamap pilot queue list [--all] [--deck S] [--json]
manamap pilot queue show Q007
manamap pilot queue apply data/decks/<slug>/.agent-out/<incubation-pod|challenger>.json
manamap pilot queue rank Q004 Q002
manamap pilot queue kill Q003 --reason "…"
manamap pilot queue decide Q007 stage|drop|watch|more --note "…"
manamap pilot validate-queue
```

Every write refreshes the affected deck's `CONTEXT.md`. Its `## Open questions`
section carries a generated `queue` block that lists the deck's live items. The block
does not apply expiry, so the rendering cannot change by itself overnight.
