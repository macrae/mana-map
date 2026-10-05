# Vision — a workbench for Magic deck builds

*The one page every other document is written against. Last revised 2026-10-05. If a
doc, a docstring or a charter disagrees with this page, this page wins and the other
is stale.*

## What this is

A **workbench for crafting, experimenting, researching and analysing Commander decks**,
built around one idea: **a claim about a deck is worth what the experiment behind it is
worth.** Everything here exists to turn "I think this deck wants more lands" into a
measurement with a number, an interval and a stated limit — and then to keep that
measurement where you can find it again.

The centre of the workbench is **the swap loop**, and it runs on two engines with
different jobs:

- **A seeded, PAIRED Monte Carlo goldfish** — the decision instrument. `try` runs the
  current list and the candidate through the same games, seed for seed, and reports the
  difference with an interval on the difference, in about ten seconds. Every channel it
  reads is declared per deck and checkable with `model-coverage`, so a null can be told
  apart from a blind spot.
- **Forge** — the real rules engine, run headless and seeded, as a **targeted probe**
  (since 2026-10-04). It answers what the goldfish cannot: does the AI actually play this
  card (`forge-cast-check`), and what happens against blockers and removal. Overnight pod
  runs left the decision loop: a day per answer, an MDE of ~0.14 at 200 games, and an AI
  that mis-pilots some decks into floors.

Around that sit the things that make an experiment mean something: a deterministic
builder, a rules-citation loop for lines that must be *proven* rather than measured, web
reconnaissance for what the wider field actually plays, deterministic card mining over
34,955 cards, and a frontend that surfaces the results.

It is optimised for one player and open-sourced so anyone can stand up their own bench —
not so anyone else is supported.

## The hypothesis loop

```
  a question                 →  an experiment            →  a result you can cite
  "is this swap better?"        try --out A --in B           damage@T10 +0.83 [+0.68, +0.98], paired
  "does the AI play this?"      forge-cast-check --card X    drawn 28, cast 0 — held
  "is this line lethal?"        /resolve-stack               ✓ or refuted, with CR cites
  "how fast does it go off?"    goldfish                     mean t4.19, 89% by t6
  "what do strong lists run?"   /prescribe, deck-recon       ranked, cited, skeptic-checked
  "what would fix this axis?"   card-search + deck-audit     candidates that move the number
```

**`try <slug> --out A --in B` is the flagship.** About ten seconds, nothing written: every
card in and out with its roles and what the goldfish can see of it, the pilot's keep list,
colour sources before and after, and `net-change`'s rows with a PAIRED interval on the
difference — each goldfish game seeded on its own (`seed:i`) and the two lists aligned slot
for slot, so the noise of two independent samples cancels and a list against itself reads
exactly zero. A swap that survives becomes a branch (`try … --stage NAME`), and
`net-change` grades the branch on one pre-registered objective plus twelve Holm-corrected
rows. Never a comparison of two marginal intervals — that is the overlap fallacy, and it
was deleted from the code.

## What the workbench does, end to end

| you want to… | the bench gives you | tier |
|---|---|---|
| screen a swap | `try` — the paired goldfish delta in ~10 s, every card's visibility, the keep list | ◆ seeded, paired |
| grade a candidate 99 | `deck-branch` + `net-change` — one primary objective, twelve Holm-corrected rows, a Forge loss as a warning | ◆ seeded, paired |
| know whether the AI plays a card | `forge-cast-check` — a two-seat Forge shell: drawn / cast / activated / held while castable | ◆ seeded |
| measure a deck against a **table** (optional) | `simulate`, `experiment` — Forge games against a named pod, every rate with its interval; graded only beside an A/A at the same N | ◆ seeded |
| measure it against nobody | `goldfish` — Monte Carlo resource development, 10,000 seeded games. **Every channel is opt-in per deck** (draw, combat, Treasure, sacrifice, discard, the spell count, four commander abilities), so what it can see is declared and checkable with `model-coverage` | ◆ seeded |
| build a legal 99 from a brief | `build-deck` — role budget crossed with a cited curve target, combo lines completed, bracket-gated | ◆ |
| find the cards that would fix an axis | `card-search` + `deck-audit` — 16 cited axes, then deterministic mining over the corpus, filtered by what you own | ◆ |
| know what the field actually plays | `deck-recon` — dated web reconnaissance, every card verified in-identity and legal | ★ dated |
| know what is **true** about a line | `/resolve-stack` — a board (authored, or **lifted from a simulated game**) resolved with CR citations and adversarially checked | ✓ |
| understand the machine | `analyze-engine` — eight stages, solid where a stack proves a line, dashed where it is a reading | ✓◆★ |
| know where a deck stands | `deck-info <slug>` — the whole join, and a derived **next** | ◆ |
| keep the list honest across swaps | `deck-version` — every list from git, joined to the games played on it | ◆ |
| remember what happened at the table | `deck-notes add` → `/debrief` → `/prescribe` | authored → ★ → ◆★ |
| know whether a deck EXISTS as cardboard | `deck-version <slug> paper` — the one claim nothing can derive | authored |
| decide which deck to spend tonight on | the **workbench** — `viz/workbench.html`, every deck and its derived next | ◆ |
| read any of it in a browser | the **deck page** — `viz/deck.html?deck=<slug>` — and the **Pilot's Operating Handbook**, `manuals/p/<slug>.html` | all |

## The evidence contract — the part that never moves

| | tier | granted by |
|---|---|---|
| ✓ | rules-verified | a stack artifact whose every step cites a real CR rule verbatim, then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | deterministic Python over committed artifacts. **Seeded** where randomness is involved: same inputs, same bytes. **Sampled** is said out loud where it cannot be. |
| ★ | coaching | labelled judgment, and dated meta claims. Useful, never disguised as measurement. |

**A figure travels with its interval, its N and its limits — or it does not travel.**
That is the rule the simulation layer is built to keep, and it is enforced in code:
`mean_ci` cannot emit a mean without a median and a spread beside it, and a sim panel
cannot render a win rate without its interval and Forge's own caveat about its AI.

Every agent returns JSON a validator checks. No agent writes prose claiming a tier it was
not granted. **The pipeline and the pilot commands make zero LLM calls** — every figure
on this bench is arithmetic you can re-derive, and that is not negotiable. Two local,
opt-in surfaces do call a model: `serve.py`'s `ask` bridge, and `mm ask` (Sven Botstrom),
which routes questions to those same deterministic commands and is installed only by the
`ask` extra. Neither can compute a figure; both can only read one back. The deployed site
calls nothing, and your machine runs the same code.

## The frontend

Six pages over one data layer: the card atlas, the workbench landing page, the deck
page, the branch workbench, **Curate** (`viz/library.html`) and the embedding-space
appendix.

**The card atlas** (`viz/index.html`) — 34,890 oracle cards embedded by two small neural
nets. It opens on **one card**; click a relation and its neighbours join a graph you grow.
Three relations, each precomputed so a click is instant: **similar** (embedding
neighbours), **synergy** (rule-based complements), **outclassed by** (strictly-better
replacements). Boot costs 1.9 MB.

**The workbench** (`viz/workbench.html`) — the landing page, and the only screen that
answers *which deck should I spend tonight on*. Racks group by whether a deck is sleeved;
the fleet table is one row per deck sorted by recently played, needs game logs, needs
analysis, or optimisations identified. Every sort maps to a predicate `deck-info` already
computes, so the page adds no judgement of its own.

**The deck page** (`viz/deck.html?deck=<slug>`) — the workbench surface. What to do next,
where the deck stands, every list it has been, what limits it, the engine, **the
experiments and simulation runs with their intervals**, prescriptions, the captain's log,
open questions, and the constellation. It renders `info.json` — the shape `deck-info`
composes — rather than re-deriving anything, so it cannot disagree with the command that
owns each figure.

## What is live, what is legacy, what is honest (2026-10-05)

**Live** — the whole loop above: `try`, branches and `net-change`, the goldfish, the keep
list, Forge as a probe (`forge-cast-check`, and `simulate`, `experiment`, `sim-scenario`
when a question needs a table), the deterministic builder, `card-search`,
`deck-audit`, `deck-info`, `deck-version`, `deck-notes`/`/debrief`/`/prescribe`,
`/resolve-stack`, `analyze-engine`, `deck-recon`, the card atlas, the deck page, the
**workbench landing page** and the **Pilot's Operating Handbook**
(`build-poh` → `manuals/p/`).

**Deleted, not frozen** — the magazine renderer is gone. The **Pilot's Operating
Handbook** (`poh.py`) superseded it on 2026-09-02, rendering the same
`manuals/p/<slug>.html`; the renderer itself was deleted on 2026-09-13 (commit
`443cf6b7`) — eleven modules, seven subcommands, nine pages and nine test files, 14,064
lines. `build_manual`, `issue_spec`, `design`, `validate_issue` and `build_page` do not
exist, `manuals/index.html` does not exist, and STYLEv3.md went on 2026-08-25. What it
measured is kept in `docs/gotchas-magazine-legacy.md`; the code is in git.

**Honest about three things.**

*Forge's AI pilots the deck — including yours.* Forge rates its own AI "poor to ok in
control, pretty bad for combo", and that sentence is quoted verbatim in every run record.
A control deck's win rate is a **lower bound on the pilot**; a combo deck's is not a
measurement at all. What a run is genuinely good at: whether the AI plays a card at all, the clock the
table sets, who kills you and how, and whether the kill the goldfish measured actually
lands. That is why it is a probe and not the judge.

*The goldfish has no blockers.* It is the decision instrument because it is fast,
paired and declared — not because it sees everything. It cannot price a blocker's value
or removal, and its verdict on board quality is not evidence.
`model-coverage` and `try` say what it cannot see, card by card.

*Nineteen games. Not two hundred.* Five decks carry a captain's log (as of 2026-10-05) and
every entry has been debriefed, and two of them fed a prescription — enough to have proved the loop
works end to end, and nowhere near enough to conclude anything about any one deck. The gap
no amount of implementation closes is still open; it is just narrower than it was.

*Eight of fourteen decks are not marked as built in paper.* Whether a deck exists as
cardboard is an assertion only the pilot can make, and six have been asserted — Edgar,
Gishath, Goblin Storm, Heliod, Sharknado, Ur-Dragon. Of the other eight, five are broken
down for parts and one is retired. An unlocked deck SAYS it is unlocked rather than being
quietly assumed playable — the third state, after LOCKED and dead — and that one authored
flag is also what decides whether the automatic chain runs for a deck at all.

## Vocabulary

*deck* (a 99 + commander, `data/decks/<slug>/`) · *version* (a content-distinct
`decklist.txt` in git; `V1`…) · *the pod* (your opponents, `data/opponents/`) · *a screen* (one `try`: a paired goldfish delta) ·
*run* (N seeded Forge games, one record) · *probe* (a narrow Forge question, usually
`forge-cast-check`) · *the keep list* (`protected.json`, cards the pilot will not cut) · *experiment* (two versions, one table, one artifact) ·
*stack* (a scenario + its cited resolution + the checker's verdict) · *game state v2*
(seats that can act; CR step names; actions) · *the log* (authored), *the debrief*
(derived) · *prescription* (one question to the doctor) · *recon* (dated field
reconnaissance) · *the collection* (your physical boxes, `data/collection/`) · *branch* (a candidate
99 you cannot yet sleeve) · *proposal* (a branch the pilot has ACCEPTED and cannot
merge yet — the decision is frozen, the blocker is cardboard and is recomputed on
every read) · *the pull list* (what a proposal still needs: buy / unsleeve / proxy
/ free) ·
*the doctor / the skeptic / the resolver / the checker / the engineer / the critic*
(agents, always in pairs where a claim reaches a decklist or a ✓).

Not in the vocabulary any more: *issue, volume, department, columnist, byline, the Short
List, the magazine* — legacy words for the legacy renderer.
