# Agent Cost & the Invocation Cache

Where LLM spend actually lives in this project, how much each routine costs, and how
the cache stops us paying twice for identical work.

## The finding that shapes everything

**No Python module in this repo calls an LLM.** There is no `anthropic`/`openai` SDK,
nothing in `pyproject.toml`, and no subprocess spawning. The "agents" are Claude Code
subagent *definitions* in `.claude/agents/*.md`, spawned by the orchestrating session
when a skill says to. They call our CLI via Bash; the CLI never calls them back.

Two consequences:

- **There is nothing to mock in `conftest.py`** and no HTTP request/response pair to
  memoize. A Python-level API cache would intercept zero calls.
- **`manamap pilot build-manual` already costs $0.** It is pure deterministic
  rendering, and it is deliberately *not* cached — a cache there would buy nothing and
  risk stale HTML.

All the money is in subagent spawns, so the cache lives at the **skill-orchestration
layer**: before spawning, a skill asks the CLI whether the agent's declared inputs have
changed since its output artifact was recorded.

## Measured cost per routine

Real numbers from the session that built the magazine layer (2026-07-25):

| Routine | Agent | Tokens | Typical trigger |
|---|---|---|---|
| `issue-plan` | magazine-editor | **147,351** | any STYLEv3 edit, prose change, new stack |
| *(research pass)* | strategy-researcher (research) | 117,827 / 91,332 | explicit research request |
| `strategic-frame` | strategy-researcher (consult) | 80,948 | new decklist or newly verified line |
| `coach-prose` | pilot-coach | 54,515 | frame change, new stack |
| `writer-prose` | manual-writer | 47,188 | frame change, new stack, graph refresh |
| `stack:<NNN>` resolve | stack-resolver | 35,278 / 38,231 | new or re-run scenario |
| `stack:<NNN>` check | rules-checker | 29,625 / 28,097 | every resolver iteration |

**A full manual regeneration ≈ 330k tokens** across four serially-dependent agents.
`resolve-stack` is 2–6 spawns per scenario (resolver + checker, up to
`RESOLVE_MAX_ITERATIONS = 3`).

What this bought before the cache: re-running `write-manual` after a one-word prose
tweak paid all four agents again, and `design-issue` paid another 147k because
`manual_prose.json` was one of the editor's declared inputs.

## What the cache does

Every agent's output was already a tracked artifact; we simply never recorded which
inputs produced it. `data/decks/<slug>/.agent-cache.json` (tracked) now stores, per
routine: the input paths and their content hashes, the agent-prompt hash, the artifact
hash, and the resulting fingerprint.

```
check → (miss) spawn → write → validate → record
```

Record **last**, after validation. `record()` refuses an artifact that doesn't exist,
lacks the routine's owned keys, or (for a stack) has no `checker` block — so a failed
run can't poison the cache.

| Status | Exit | Meaning |
|---|---|---|
| `HIT` | 0 | inputs unchanged → do not spawn |
| `EDITED` | 0 | inputs unchanged but the artifact was hand-edited → the human wins, do not spawn |
| `MISS` | 1 | spawn, then record |
| error | 2 | required input missing → stop, don't spawn |

```bash
manamap pilot cache-status <slug> [--routine R] [--json] [--force]
manamap pilot cache-record <slug> --routine R
manamap pilot cache-clear  <slug> [--routine R]
```

A MISS always names what changed — a MISS you can't explain is a bug report, not a
cache.

## Design decisions worth knowing

**Prose structure, not prose text.** `issue-plan` hashes only `manual_prose.json`'s key
skeleton (which sections and which cards/stacks have copy), never the wording. The
editor *packages* prose; it doesn't rewrite it. So a typo fix is free, while adding a
combo line or dropping a section correctly forces a re-plan. If a rewrite is heavy
enough to change the issue's angle, use `--force`.

**Agent prompts are inputs.** Editing `.claude/agents/pilot-coach.md` changes what the
agent produces from identical artifacts, so it invalidates that agent's routines by
design.

**Full content hashes, never mtime.** The three global graphs total ~56MB and hash in
0.17s. Using size+mtime would be actively harmful: `regen-analysis` rewrites them
byte-identical on every run, which would false-invalidate ~200k tokens of agent work.

**The strategy-doc handshake.** `strategy:doc` reuses `common.strategy_doc_sha256()`,
which hashes `strategy.md` bytes — never the derived index. So
`manamap pilot build-strategy-db` cannot invalidate a single agent routine.

**Stacks fingerprint their own scenario slice.** `stack:<NNN>` hashes only
`{title, scenario}` of its artifact, so the `resolution` and `checker` blocks the loop
writes back never self-invalidate.

**Coach and writer share a file, not a fingerprint.** Both write `manual_prose.json`
but own disjoint keys; each digests only its own keys, so one running doesn't make the
other look hand-edited.

**Only passing stacks are inputs.** A failing stack can't be published, so editing one
doesn't invalidate downstream prose — but flipping it to `pass` does.

**`cards.json` is hashed semantically.** Agents read names, oracle text, costs and
types; they never read artist, set, collector number, finishes, or image URLs. So
`cards:semantic` digests only `CARD_SEMANTIC_FIELDS`. This was proven by the printing-
fidelity change: enriching all 82 cards with exact Secret Lair printings produced a
byte-identical semantic digest, so the prose stayed valid and the enrichment cost **0
agent tokens** instead of ~330k.

## The other cost: `fetch-deck`

`fetch_deck.py` computed `decklist_sha256` and wrote it into `cards.json` but never
read it back, so every run re-POSTed the whole decklist to Scryfall *and rewrote
`cards.json`*. The rewrite was the expensive part: `cards.json` is an input to all four
agent routines, so a habitual no-op re-fetch could silently cost a 330k-token
regeneration. It now short-circuits when the decklist is unchanged (`--force` to
override). The hash covers the decklist text, not Scryfall's data, so oracle errata
need `--force` — the right default for a locked-decklist subsystem.

## Not cached, deliberately

- **`build-manual` / `build-index`** — already free and deterministic.
- **Research passes** — web search is the point; new content is the goal, so hit rates
  would be near zero.
- **Ad-hoc `strategy-lookup` consults** — they produce no artifact, so there is nothing
  to key against. If consult volume ever matters, make consults write a keyed transcript
  rather than trying to cache them.

## Deferred

Memoizing the SentenceTransformer in `ingest/preprocess.py` (currently reconstructed on
every call, ~2s per RAG query, with `show_progress_bar=True` hardcoded so single-text
queries emit a one-item bar into the `--json` agent interface). Pure wall-clock and
noise win, no LLM cost impact.
