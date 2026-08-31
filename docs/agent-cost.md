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

**No Python module in this repo calls an LLM directly.** There is no
`anthropic`/`openai` SDK and nothing in `pyproject.toml`. The "agents" are Claude Code
subagent *definitions* in `.claude/agents/*.md`, spawned by the orchestrating session
when a skill says to. They call our CLI via Bash; the CLI never calls them back.

**ONE EXCEPTION, added later and missed by this page until 2026-08-31.** `serve.py`'s
`ask` endpoint (`_spawn`, ~`serve.py:495`) runs `subprocess.run(["claude", "-p", …])`
with a 30-minute timeout, polled through the `job` verb. So the sentence that used to
stand here — "and no subprocess spawning" — was false for as long as that endpoint has
existed. It is still true that nothing in the *pipeline* or the *pilot* commands spawns
a model; the local bridge does, deliberately, because letting the Build page ask for an
agent is the point of the bridge.

Two consequences:

- **There is nothing to mock in `conftest.py`** and no HTTP request/response pair to
  memoize. A Python-level API cache would intercept zero calls.
- **Rendering already costs $0.** `build-manual` (the legacy page) is pure deterministic
  rendering, and it is deliberately *not* cached — a cache there would buy nothing and
  risk stale HTML. The same is true of every `deck-*`, `validate-*` and `sim` command.

All the money is in subagent spawns, so the cache lives at the **skill-orchestration
layer**: before spawning, a skill asks the CLI whether the agent's declared inputs have
changed since its output artifact was recorded.

## Measured cost per routine — the bench (current)

Every figure is a token count from a real spawn, with the date it was measured; an
estimate is marked as one. The cache routines are `config.AGENT_ROUTINES` plus the
per-artifact `stack:<NNN>`, `decision:<NNN>` and `prescription:<id>`.

| Routine | Agent(s) | Tokens | Typical trigger |
|---|---|---|---|
| `stack:<NNN>` resolve | stack-resolver | 35,278 / 38,231 (2026-07-25); **97k / 95k / 60k** on the v2 board of radagast 008 (2026-08-19) | new or re-run scenario |
| `stack:<NNN>` check | rules-checker | 29,625 / 28,097 (2026-07-25); **96k / 115k / 106k** on radagast 008 | every resolver iteration |
| `deck-engine` | deck-engineer ⇄ engine-critic | **~120k per engineer pass, ~140k per critic** — radagast took 4 spawns over 3 iterations | decklist edit; a newly passing stack (it may turn a dashed line solid) |
| `strategic-frame` | strategy-researcher (consult) | 80,948 / 130,161 | new decklist or newly verified line |
| `deck-diagnosis` | deck-doctor ⇄ deck-skeptic | 200,000–300,000 (est.) | decklist edit, new verified stack, goldfish re-run, a new sim run (`sim:runs`), a new debrief |
| `prescription:<id>` | deck-doctor ⇄ deck-skeptic | ≈ one diagnosis pass (est.; unmeasured — none run yet) | the question's own prompt (`prompt:self`); otherwise as `deck-diagnosis` |
| `deck-recon` | deck-doctor (MODE recon) | 60,000–90,000 (est.) | age, not inputs — see below |
| `debrief` | debrief | **est. 15,000–30,000** per batch of un-debriefed entries (unmeasured — nothing logged yet) | a new `log.jsonl` entry (N/A until one exists) |
| `pilot-notes` | pilot-notes | unmeasured; its two predecessors cost 54,515 + 47,188 for the same keys (2026-07-25) | frame change, new stack, engine change |
| `tutor-guide` | pilot-notes | 60,000–90,000 (7 spawns, 2026-07) | a tutor enters or leaves the 99 |
| `deck-map-names` | deck-cartographer | ~60,000–93,000 (9 spawns, 2026-08) — optional, no longer a lifecycle stage | `deck-map` re-run |
| `candidate-pool` | deck-analyst | **235,579** / 130,161 | new brief, role or combo-data refresh |
| `deck-build` | deck-architect ⇄ deck-critic | 105,096 + 96,380 (architect, revision) + 94,468 (critic); ~430,000 for a full loop | new pool, critic findings |
| *(research pass)* | strategy-researcher (research) | 91,332–166,544 per pass | an explicit research request |

**The resolve loop is the outlier, and it is measured twice.** hapatra's stack 001
(2026-07-26) reached **~600k** over 4 resolver + 4 checker passes; radagast's stack 008
(2026-08-19, the first board lifted from a Forge game) reached **~570k** over 3
iterations — 62 citations, against the scope budget of 40. Both confirm the rule in
`docs/pilot.md` → *Scenario scope*: an artifact past ~59 citations takes three or four
rounds however right the question is. `RESOLVE_MAX_ITERATIONS = 3` is the bound;
stack 001 overrode it deliberately to reach a verdict.

`deck-recon` is the only routine in `AGENT_ROUTINES` whose staleness is **time**
rather than inputs. A decklist edit does not change what strong lists for that
commander run, so hashing `cards.json` here would buy a web pass on every swap;
its declared input is `deck:brief.json?` and `RECON_MAX_AGE_DAYS` is judged by the
skill. It is also deliberately not an input to the notes — a recon refresh should
cost one diagnosis, not a regeneration.

The deterministic half costs **zero**: `deck-audit` joins five existing artifacts into
sixteen cited axes plus the engine-activation read; `simulate` runs N Forge games and
`sim/parse.py` turns them into intervals; `deck-info`, `deck-version`, `deck-notes` and
every validator are Python. A cache miss on an agent routine still leaves the whole
measurement on the table.

`tutor-guide` reports `N/A` for a deck with no library-search tutors and `debrief` for a
deck with nothing logged, so neither becomes a permanent MISS.

**A full build ≈ 530k tokens** for a first pass with one critic iteration
(pool → architect → critic → architect), bounded by `DECK_BUILD_MAX_ITERATIONS = 3`.
The pool dominates, which is why `candidate-pool` is cached separately from
`deck-build`: revising a plan against critic findings reuses the pool for free.
And the deterministic builder underneath costs **zero** — a cache miss on both
routines still leaves you a legal, bracket-compliant 99.

### Legacy measurements — the magazine (retired 2026-08-19)

Kept as the record of what the magazine cost; none of these routines or agents exist now
(`docs/agent-audit-2026-08-19.md`). `issue-plan` (magazine-editor) and `panel-prose`
(pilot-panel) were deleted; `coach-prose` (pilot-coach) and `writer-prose`
(manual-writer) folded into `pilot-notes`; `the-ten` (short-list-analyst) folded into
`prescribe`.

| Routine | Agent | Tokens | Measured |
|---|---|---|---|
| `issue-plan` | magazine-editor | 147,351 from scratch; 100,450–125,290 on a re-plan (mean ~113k) | 2026-07-25 / 2026-08-14 |
| `panel-prose` | pilot-panel | ~134,000 | 2026-08-14 |
| `coach-prose` | pilot-coach | 54,515 / 78,093 | 2026-07-25 / 07-26 |
| `writer-prose` | manual-writer | 47,188 / 68,798 | 2026-07-25 / 07-26 |
| `the-ten` | short-list-analyst | 76,000–115,000 (7 spawns) | 2026-08 |

A full manual regeneration was **≈ 690k tokens across six serially-dependent routines**
(`strategic-frame` ~81k · `deck-engine` ~260k · `coach-prose` ~55k · `writer-prose` ~47k ·
`panel-prose` ~134k · `issue-plan` ~113k), and a full publish of a deck built from
scratch **≈ 1.7M**, dominated by the build loop and one hard rules question. The pivot
deleted roughly 350k of that per deck and kept the evidence routines. Before the cache
existed, re-running `write-manual` after a one-word prose tweak paid all four agents
again, and `design-issue` paid another 147k because `manual_prose.json` was one of the
editor's declared inputs — which is why `prose:shape` (hash the key skeleton, not the
words) was invented, and why it was deleted with the editor.

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

**Agent prompts are inputs.** Editing `.claude/agents/pilot-notes.md` changes what the
agent produces from identical artifacts, so it invalidates that agent's routines by
design. **`.claude/agents-common.md` is hashed with every agent** (inside
`agent_prompt_sha256`, not per routine, so a new routine cannot forget it): it holds
the contract that used to be pasted into twelve charters, and editing it invalidates
the whole fleet — exactly as editing twelve pasted copies did, now visibly and once.

**Full content hashes, never mtime.** The diagnosis routines hash the combo records and
the obsolescence index, and the build routines hash ~34MB (`combo_graph.json` 4.5MB as the documented
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

**`pilot-notes` shares `manual_prose.json` with keys nobody owns.** The routine digests
only its five keys, so the frozen legacy keys beside them (`card_roles`, `mana_base`,
`upgrades`, the panel keys) can never make it read EDITED, and `merge-prose` never touches
them. (Until 2026-08-19 the same mechanism kept two agents, coach and writer, from
clobbering each other in the same file.)

**Art was a separate token, and it is gone.** `cards:printing` existed only because the
magazine-editor read printings for Featured Artist; it was deleted with the editor.

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
`cards.json`*. The rewrite was the expensive part: `cards.json` is an input to nearly every
agent routine, so a habitual no-op re-fetch could silently cost a 330k-token
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

## Resolved 2026-08-31 — and the triage was wrong

This page filed the SentenceTransformer reconstruction under *Deferred*, priced at "~2s
per RAG query" and dismissed as "pure wall-clock and noise win, no LLM cost impact".

**Measured, the figure was ~8s, and wall-clock on the question path IS the cost.**
`query-rules` ran **12.1s** end to end, of which ~5.5s is importing
`sentence_transformers` and ~2.5s is constructing a frozen MiniLM the previous
invocation had already built and thrown away. `_MODEL_CACHE` handles it *within* a
process, and the CLI is one process per question — while `.claude/skills/rules-lookup`
tells the agent to "try several phrasings", so three phrasings cost ~36 seconds of
reloading an identical model.

The fix was not memoization but a **warm worker**: `manamap serve` already held the
model, the corpus and every JSON memo, and its allow-list simply did not expose the
question commands. `/api/cli` runs read-only pilot commands in that process and the
terminal routes to it when one is listening.

    query-rules      6.93s -> 0.16s   43x, output byte-identical
    query-strategy   6.87s -> 0.16s
    deck-facts       1.44s -> 0.14s

The lesson for this page's triage, not just for that item: **"no LLM cost impact" is
not the same as "no cost".** An agent loop is 5-15 cold CLI calls per step before it
spends a single token, and those seconds are what make the whole bench feel heavy.

Still deferred: `preprocess.py:74` hardcodes `show_progress_bar=True`, so a single-text
query emits a one-item bar into the `--json` interface.
