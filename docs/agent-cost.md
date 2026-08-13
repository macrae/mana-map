# Agent Cost & the Invocation Cache

Where LLM spend actually lives in this project, how much each routine costs, and how
the cache stops us paying twice for identical work.


> **Reading a MISS.** `cache-status` reports MISS both for a routine that has never run and
> for one whose declared inputs moved, and those need opposite responses. The `changed` list
> tells them apart: empty means never recorded, so there is nothing to re-bless and the only
> option is to run it. A charter edit (`.claude/agents/*.md`) MISSes every routine that agent
> owns and **disqualifies STALE_OK by construction**, which is why charter edits belong
> *before* a `cache-record` pass, never after. Never `cache-record` to clear the board: the
> record asserts that someone read the artifact and agreed it holds.

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
| `deck-map-names` | deck-cartographer | ~60,000–93,000 (9 spawns, the fleet batch) | `deck-map` re-run: a decklist edit or a retrain moves the clusters |
| `deck-engine` | deck-engineer ⇄ engine-critic | **~120k per engineer pass, ~140k per critic** — radagast took 4 spawns over 3 iterations | decklist edit, a newly passing stack (it may turn a dashed line solid) |
| `writer-prose` | manual-writer | 47,188 | frame change, new stack, graph refresh |
| `stack:<NNN>` resolve | stack-resolver | 35,278 / 38,231 | new or re-run scenario |
| `stack:<NNN>` check | rules-checker | 29,625 / 28,097 | every resolver iteration |

From the session that built the deck builder and built hapatra (2026-07-25):

| Routine | Agent | Tokens | Typical trigger |
|---|---|---|---|
| `candidate-pool` | deck-analyst | **235,579** | new brief, role or combo-data refresh |
| `deck-build` | deck-architect | 105,096 + 96,380 (revision) | new pool, critic findings |
| `deck-build` | deck-critic | 94,468 | every architect iteration |
| *(research pass)* | strategy-researcher (research) | 120,545 / 92,559 / 166,544 | the `strategy:deckbuilding` corpus, 3 passes |
| `the-ten` | short-list-analyst | 76,000–115,000 (7 spawns, the fleet batch) | decklist edit, new pilot feedback, pool refresh |
| `tutor-guide` | pilot-coach | 60,000–90,000 (7 spawns) | a tutor enters or leaves the 99 |

The diagnosis loop (2026-08-03), **estimates until the first run measures them** —
recorded here as estimates on purpose, so the gap between the guess and the real
number is visible rather than quietly overwritten:

| Routine | Agent | Tokens (est.) | Typical trigger |
|---|---|---|---|
| `deck-recon` | deck-doctor (MODE recon) | 60,000–90,000 | age, not inputs — see below |
| `deck-diagnosis` | deck-doctor ⇄ deck-skeptic | 200,000–300,000 | decklist edit, new verified stack, goldfish re-run |

`deck-recon` is the only routine in `AGENT_ROUTINES` whose staleness is **time**
rather than inputs. A decklist edit does not change what strong lists for that
commander run, so hashing `cards.json` here would buy a web pass on every swap;
its declared input is `deck:brief.json?` and `RECON_MAX_AGE_DAYS` is judged by the
skill. It is also deliberately not an input to any manual routine — a recon
refresh should cost one diagnosis, not a regeneration.

The deterministic half costs **zero**: `manamap pilot deck-audit` joins five
existing artifacts into sixteen cited axes plus the engine-activation read, and a
cache miss on both routines still leaves the whole measurement on the table.

`the-ten` is The Short List: ten cards worth knowing about that could play well with the
deck, scouted from the whole card pool (one routine, one artifact — `considering.json`).
It replaced the retired `sideboard-analysis`/`upgrade-watch` pair, and then the sideboard
itself was retired: ownership is no longer a criterion, so the pool is always the whole
card database rather than a bench topped up from it.

`tutor-guide` reports `N/A` for a deck with no library-search tutors, so it never becomes a
permanent MISS on a tutorless list.

**A full manual regeneration ≈ 330k tokens** across four serially-dependent agents.
`resolve-stack` is 2–6 spawns per scenario (resolver + checker, up to
`RESOLVE_MAX_ITERATIONS = 3`).

From the session that published Vol. 002 (2026-07-26):

| Routine | Agent | Tokens | Note |
|---|---|---|---|
| `candidate-pool` | deck-analyst | 235,579 / 130,161 | rebuilt on the bracket retarget |
| `deck-build` | deck-architect ⇄ deck-critic | ~430,000 | 2 architect passes + 1 critic pass |
| `strategic-frame` | strategy-researcher (consult) | 130,161 | |
| `coach-prose` | pilot-coach | 78,093 | |
| `writer-prose` | manual-writer | 68,798 | |
| `issue-plan` | magazine-editor | 152,697 | |
| `stack:001` | stack-resolver ⇄ rules-checker | **~600,000** | 4 resolver passes, 4 checks |

**The stack is the outlier and it is worth understanding before queueing more.** Vol. 001
resolved *five* lines for less than stack 001 cost alone. The spend was earned — the answer
overturned the deck's premise and the checker was right on every pass — but it is not a
repeatable rate, and `RESOLVE_MAX_ITERATIONS = 3` was deliberately overridden to reach a
verdict. See `PLAN.md` for the structural fix that implies.

**A full publish ≈ 1.7M tokens** for a deck built from scratch, dominated by the build loop
and one hard rules question. A deck with an existing decklist and no contested combo skips
both and lands nearer Vol. 001's ~330k.

**A full build ≈ 530k tokens** for a first pass with one critic iteration
(pool → architect → critic → architect), bounded by `DECK_BUILD_MAX_ITERATIONS = 3`.
The pool dominates, which is why `candidate-pool` is cached separately from
`deck-build`: revising a plan against critic findings reuses the pool for free.
And the deterministic builder underneath costs **zero** — a cache miss on both
routines still leaves you a legal, bracket-compliant 99.

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
| `STALE_OK` | 0 | inputs changed but no referenced card did — `cache-rebless` clears it, no spawn |
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

**Full content hashes, never mtime.** The manual routines hash ~38MB of global graphs
(`combo_graph.json` 4.5MB + `synergy_graph.json` 27.8MB + `obsolescence_index.json`
5.9MB); the build routines hash ~34MB (`combo_graph.json` 4.5MB as the documented
proxy for `combo_details.json` + `card_roles.json` 1.9MB + `synergy_graph.json`
27.8MB). Using size+mtime would be actively harmful: `regen-analysis`
rewrites them byte-identical on every run, which would false-invalidate ~200k tokens of
agent work. `cached_file_sha256` memoizes on `(path, mtime_ns, size)` so a single
`cache-status` scan hashes each file once.

Note the build routines deliberately hash `card_roles.json` by **content** rather than
taking a `roles:version` token — a role edit that doesn't change any card's
classification correctly costs nothing.

**The strategy-doc handshake.** `strategy:doc` reuses `common.strategy_doc_sha256()`,
which hashes `strategy.md` bytes — never the derived index. So
`manamap pilot build-strategy-db` cannot invalidate a single agent routine.

**Stacks fingerprint their own scenario slice.** `stack:<NNN>` hashes only
`{title, scenario}` of its artifact, so the `resolution` and `checker` blocks the loop
writes back never self-invalidate.

**Coach and writer share a file, not a fingerprint.** Both write `manual_prose.json`
but own disjoint keys; each digests only its own keys, so one running doesn't make the
other look hand-edited.

**Art is a separate token.** `cards:printing` (artist, set, collector number, border,
frame effects, finishes, foil) is an input to `issue-plan` only, because the
magazine-editor is the one agent that reads it — Featured Artist names an artist. Coach,
writer, stacks and decisions don't reason about art, so they never see it.

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

## Incremental regeneration (2026-07-28)

The cache is card-scoped. Every record stores the card names its artifact
references (`card_refs`, conservative matcher: full names + DFC faces +
distinctive name-tokens), the sidecar keeps a per-card digest map, and keyed
routines store per-key fingerprints over per-key inputs (`PROSE_KEY_INPUTS`).
Three consequences:

- A deck change that touches no referenced card reports `STALE_OK` (exit 0);
  `manamap pilot cache-rebless <slug>` re-records the lot without a spawn.
  A one-land swap costs zero spawns rather than a ~330k full sweep.
- A real MISS on a keyed routine names its `stale keys:`, and the charters'
  Partial revision mode scopes the spawn to exactly those keys — the writer
  costs what the stale keys cost, not the whole file.
- `manamap pilot impact <slug>` (run it BEFORE cache-rebless — a rebless
  advances the card baseline and blinds the deck diff) is the deterministic
  staleness report:
  per-artifact/key/department card references, a rounding-aware audit of
  goldfish/bracket figures quoted in prose, goldfish-target ghosts, and
  zone-framing flags. Report-only; regeneration always goes through an agent.

None of the new data enters the fingerprint — a HIT still means exactly what
it meant, and pre-existing records behave classically until their next
record seeds refs.

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
