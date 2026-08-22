# Common contract for every Mana Map agent

Every pilot charter in `.claude/agents/` opens by telling you to read this file
(`pipeline-runner` and `viz-dev` write no deck artifact and are exempt). It holds
the rules that used to be pasted into each one — ~1,000 lines of identical text across
twelve charters, every edit to which MISSed every routine. Now there is one copy, and
`agent_cache.agent_prompt_sha256` hashes this file with every agent, so editing it still
invalidates everything it governs — deliberately.

It lives beside `agents/`, not inside it: Claude Code loads `.claude/agents/*.md` as
agent definitions and `tests/test_docs_counts.py` counts them.

## 1. You are read-only with respect to tracked files

You write one JSON object to the deck's agent scratchpad and return its path. You never
edit `cards.json`, `decklist.txt`, `goldfish_targets.json`, anything under `src/`, or
any tracked artifact. The orchestrator validates your file and merges it. The single
exception is `strategy-researcher` in MODE research, whose write scope is stated in its
own charter.

## 2. Start here: `deck-facts`

Before deriving anything about a deck's composition, run:

```bash
.venv/bin/manamap pilot deck-facts <slug>
```

It returns, deterministically and in one shot, the facts agents used to recompute by
hand: entry/copy counts, the mana-value curve, per-card colours **resolved correctly for
multi-face cards** (both the card's union and the face-up permanent's), per-colour pip
load and source targets, role coverage plus the cards the taxonomy has no pattern for,
every combo line fully contained in the deck, and a `notes` block naming the traps — how
many synergy edges actually fall inside this deck, and which mana is restricted in a way
that cannot pay an activated ability.

Read it first and cite it. Re-deriving these by hand costs tokens and has produced wrong
answers before: `cards.json` colours read as empty for every double-faced card until it
was fixed, and "spend this mana only" was misread as blanket-restricted on a land whose
clause explicitly permits activating abilities. Five errors reached agent briefs in a
single session and every one was a correct-sounding figure recalled rather than looked
up.

Your charter names the *other* deterministic commands you run first (`deck-audit`,
`engine-facts`, `scenario-facts`, …). Same rule for all of them: read, cite, do not
re-derive, and **do not contradict** — if a command disagrees with a prose artifact, say
so in your summary rather than picking one. The disagreement is itself a finding.

## 3. Write per-deck views with `--out <dir>/`, never a shell redirect

You may run concurrently with agents working other decks, and you all share one
scratchpad directory. `deck-audit`, `deck-facts`, `deck-history`, `impact`,
`diagnosis-report` and `scenario-facts` take `--out`; hand it a DIRECTORY and it
auto-names `<command>-<slug>.json`, so a collision is impossible:

```bash
.venv/bin/manamap pilot deck-audit <slug> --out "$SCRATCH/"
```

A generic name (`audit.json`, `aud.json`) is how one deck's view silently replaces
another's — seven agents read the wrong deck's numbers under their own invocation before
this was found, and every catch was someone noticing an implausible figure. `--out`
REFUSES a path whose filename omits the slug. A shell redirect (`> audit.json`) is not
policed and must not be used for per-deck data.

## 4. The evidence ladder, in one paragraph

A checker-passed stack (`stacks/*.json` with `checker.verdict == "pass"`) is the only
fact about a line. Everything else is a candidate: a Commander Spellbook record, a
synergy-graph edge, an obsolescence-index hit, a role tag, a cosine score. **Never
assert that a combo works** — a line without a passing stack is `"status": "needs a
stack scenario"`, always. Spellbook lines can quietly assume a piece is your commander
(`"Infinite commander casts"` in `produces` is the tell; goblin-storm stack 004 refuted
one this way). Deterministic Python over committed artifacts is ◆ data-derived. A
strategy citation is ★ coaching and never upgrades a claim to rules-verified. Costume
never earns the badge.

Look things up the way the data is shaped: `combo_details.json` via `by_card`, never a
linear scan; `combo_graph.json` is adjacency only; `synergy_graph.json` is a format-wide
top-10 shortlist, not a per-deck fit score; embeddings are positional (`row i ==
cards.csv row i`, names duplicate). **Read the oracle text before you trust an index
hit** — the taxonomy is literal and the indexes are format-agnostic.

## 5. Before you write a superlative, enumerate the set

"The only", "no other", "every X is", "tops out at", "nothing else", "unlike every" — a
true observation about a subset, generalised into a false claim about the whole, and
then a prescription sized off the generalisation. Eight diagnoses failed the skeptic on
exactly this at least once. Mechanically: name the set, enumerate it from `cards.json` or
a named card list (never from counts, memory or reputation), read every member's oracle
text, and re-scope the claim to what survived. Then read your finished artifact once more
looking only for those words.

## 6. Partial revision mode

When the spawning prompt scopes you to named entries (or keys), that scope is a
contract:

- Revise ONLY the named pieces. Every other entry is copied **byte-identical** from the
  tracked artifact — copy programmatically (load the file and carry the values), never
  retype from memory. When editing a string in place, use a single-occurrence assert so
  a failed match aborts instead of silently mangling.
- Return the FULL artifact as usual; the orchestrator diffs and merges.
- State, one sentence per revised piece, what changed and why.
- If revising a scoped piece would make an UNSCOPED piece false, say so in your summary
  instead of silently editing it — the orchestrator widens the scope; you don't.

An unscoped spawn is the classic full rewrite. The scoped mode exists because
regeneration cost tracks the pieces that changed, not the file they live in.

## 7. Revision iterations

When your prompt includes a verifier's `findings`, address **every** non-`supported`
finding: fix a miscount from the artifact rather than by arithmetic, replace a mis-cited
quote with exact `lookup-*` text, and either ground an over-claim or downgrade it into
`open_questions`. Note what you changed per finding in your returned summary. A finding
you disagree with is answered with evidence **in the artifact**, not argued in the
summary — rebut rather than weaken.

## 8. Returning your output

Write your JSON to the deck's agent scratchpad and return **only the path plus a short
summary** — never the JSON itself:

```bash
mkdir -p data/decks/<slug>/.agent-out
cat > data/decks/<slug>/.agent-out/<agent>.json <<'JSON'
{ ...your JSON... }
JSON
```

Your charter names the exact filename (some carry a stack id or a mode suffix so two
runs never clobber each other). Then say, in at most ~200 words: the path you wrote, what
you concluded, and anything the orchestrator must decide. That is the whole final
message.

Why: an artifact can run to tens of thousands of tokens, and returning it inline costs
that much again in the orchestrating session's context — `candidate_pool.json` alone
reaches 133 KB. The directory is gitignored; the orchestrator validates your file and
merges it into the tracked artifact.

**Write it EARLY and extend it — a partial artifact beats a stalled one.** Create the
file as soon as you have the headline plus one finding, then rewrite it as you learn
more. Two `deck-doctor` recon runs were lost in a single session — one stalled on a
WebFetch that never returned, one to the machine sleeping mid-response — and because
both held everything until the end, each produced **nothing at all** and the whole spawn
was re-spent. The third run wrote early, survived, and produced the best artifact of the
session. Your last write wins, so there is no cost to writing sooner.

**Cap your retries on any single source.** If a fetch or a command fails twice, record
what you could not reach in `gaps` and move on. An agent that cannot finish is worth
less than one that finishes and says what it missed — and naming the hole is itself a
finding, since the reader needs to know which claims rest on nothing.

## 9. Things you cannot do, and what to do instead

- **You cannot spawn agents.** What you cannot settle goes in `open_questions` with
  `settled_by` ∈ `resolve-stack` | `research-strategy` | `goldfish` | `diagnose`, and
  the orchestrator dispatches it. A question worth asking beats a claim worth doubting.
- **Deterministic by default.** Same inputs → same output; no dates, no randomness.
  Exceptions are named per charter (`as_of` in recon mode; a prompted mode keyed by its
  prompt).
- **History is allowed.** The legacy magazine law "every issue is the reader's first" (L10)
  is repealed for the workbench: a deck's versions, what changed and why, and what
  happened in a logged game are *inputs*, and you may name them. The one thing that
  still may not be rewritten is a checker-passed artifact's text — regenerate, never
  patch.
