# Vision — a workbench for Magic deck builds

*The one page every other document is written against. Last revised 2026-08-22. If a
doc, a docstring or a charter disagrees with this page, this page wins and the other
is stale.*

## What this is

A **workbench for crafting, experimenting, researching and analysing Commander decks**,
built around one idea: **a claim about a deck is worth what the experiment behind it is
worth.** Everything here exists to turn "I think this deck wants more lands" into a
measurement with a number, an interval and a stated limit — and then to keep that
measurement where you can find it again.

The centre of the workbench is **simulation**. Two engines, answering different questions:

- **Forge** — the real rules engine, run headless and **seeded**, playing your list
  against your pod's actual decks. Same inputs, same games, byte for byte. This is where
  a hypothesis about how a deck *plays* gets tested.
- **A seeded Monte Carlo goldfish** — 10,000 games of resource development against
  nobody, for the questions that are about the deck's own curve rather than about a
  table: when does the commander land, how often is the engine assembled by turn six.

Around that sit the things that make an experiment mean something: a deterministic
builder, a rules-citation loop for lines that must be *proven* rather than measured, web
reconnaissance for what the wider field actually plays, deterministic card mining over
34,890 cards, and a frontend that surfaces the results.

It is optimised for one player and open-sourced so anyone can stand up their own bench —
not so anyone else is supported.

## The hypothesis loop

```
  a question                 →  an experiment            →  a result you can cite
  "does it want more lands?"    experiment --a V1 --b V2     +0.27 mana on turn 5,
  "is this line lethal?"        /resolve-stack               ✓ or refuted, with CR cites
  "how fast does it go off?"    goldfish                     mean t4.19, 89% by t6
  "what do strong lists run?"   /prescribe, deck-recon       ranked, cited, skeptic-checked
  "what would fix this axis?"   card-search + deck-audit     candidates that move the number
```

**`experiment <slug> --a <ref> --b <ref> --vs <pod> --games N` is the flagship.** Two
versions of a deck, the same table, the same N, the same seeds, one artifact carrying both
arms, the delta, and the sentence people skip: whether the intervals overlap. Same seeds
are **not** paired games — a changed list changes every shuffle — so seeds buy per-arm
replayability and the control is N. An A/A is refused with the reason.

## What the workbench does, end to end

| you want to… | the bench gives you | tier |
|---|---|---|
| test one change against another | `experiment` — two arms, one table, the delta and the overlap sentence | ◆ seeded |
| measure a deck against a **table** | `simulate <slug> --vs <pod>` — N seeded Forge games: win rate with its interval, who kills you and how, the kill curve, **commander damage per defender**, token pay-off | ◆ seeded |
| measure it against nobody | `goldfish` — Monte Carlo resource development; Treasure and combat opt-in | ◆ seeded |
| build a legal 99 from a brief | `build-deck` — role budget crossed with a cited curve target, combo lines completed, bracket-gated | ◆ |
| find the cards that would fix an axis | `card-search` + `deck-audit` — 16 cited axes, then deterministic mining over the corpus, filtered by what you own | ◆ |
| know what the field actually plays | `deck-recon` — dated web reconnaissance, every card verified in-identity and legal | ★ dated |
| know what is **true** about a line | `/resolve-stack` — a board (authored, or **lifted from a simulated game**) resolved with CR citations and adversarially checked | ✓ |
| understand the machine | `analyze-engine` — eight stages, solid where a stack proves a line, dashed where it is a reading | ✓◆★ |
| know where a deck stands | `deck-info <slug>` — the whole join, and a derived **next** | ◆ |
| keep the list honest across swaps | `deck-version` — every list from git, joined to the games played on it | ◆ |
| remember what happened at the table | `deck-notes add` → `/debrief` → `/prescribe` | authored → ★ → ◆★ |
| read any of it in a browser | the **deck page** — `viz/deck.html?deck=<slug>` | all |

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
not granted. The Python makes **zero LLM calls**; the deployed site and your machine run
the same code.

## The frontend

Two surfaces over one data layer.

**The card atlas** (`viz/index.html`) — 34,890 oracle cards embedded by two small neural
nets. It opens on **one card**; click a relation and its neighbours join a graph you grow.
Three relations, each precomputed so a click is instant: **similar** (embedding
neighbours), **synergy** (rule-based complements), **outclassed by** (strictly-better
replacements). Boot costs 1.8 MB.

**The deck page** (`viz/deck.html?deck=<slug>`) — the workbench surface. What to do next,
where the deck stands, every list it has been, what limits it, the engine, **the
experiments and simulation runs with their intervals**, prescriptions, the captain's log,
open questions, and the constellation. It renders `info.json` — the shape `deck-info`
composes — rather than re-deriving anything, so it cannot disagree with the command that
owns each figure.

## What is live, what is legacy, what is honest (2026-08-22)

**Live** — the whole loop above. Simulation (`simulate`, `experiment`, `validate-sim`,
`sim-scenario`, `fetch-opponent`), the goldfish, the deterministic builder, `card-search`,
`deck-audit`, `deck-info`, `deck-version`, `deck-notes`/`/debrief`/`/prescribe`,
`/resolve-stack`, `analyze-engine`, `deck-recon`, the card atlas and the deck page.

**Legacy, frozen** — the magazine renderer (`build_manual`, `issue_spec`, `design`,
`validate_issue`, STYLEv3) and the artifacts only it reads. It still renders nine pages;
nothing regenerates its inputs; it is replaced by the compact deck page
(`docs/manual-v5-spec.md`). Marked LEGACY and left accurate rather than rewritten.

**Honest about two things.**

*Forge's AI pilots the deck — including yours.* Forge rates its own AI "poor to ok in
control, pretty bad for combo", and that sentence is quoted verbatim in every run record.
A control deck's win rate is a **lower bound on the pilot**; a combo deck's is not a
measurement at all. What a run is genuinely good at: the clock the table sets, who kills
you and how, and whether the kill the goldfish measured actually lands.

*Nothing has been logged at a real table yet.* The captain's log, the debrief and
prescriptions are built, tested and wired into the deck page against **zero real
entries**. That is the one gap no amount of implementation closes, and the first entry
will teach the agents more than another sprint would.

## Vocabulary

*deck* (a 99 + commander, `data/decks/<slug>/`) · *version* (a content-distinct
`decklist.txt` in git; `V1`…) · *the pod* (your opponents, `data/opponents/`) · *run* (N
seeded Forge games, one record) · *experiment* (two versions, one table, one artifact) ·
*stack* (a scenario + its cited resolution + the checker's verdict) · *game state v2*
(seats that can act; CR step names; actions) · *the log* (authored), *the debrief*
(derived) · *prescription* (one question to the doctor) · *recon* (dated field
reconnaissance) · *the collection* (your physical boxes, `data/collection/`) ·
*the doctor / the skeptic / the resolver / the checker / the engineer / the critic*
(agents, always in pairs where a claim reaches a decklist or a ✓).

Not in the vocabulary any more: *issue, volume, department, columnist, byline, the Short
List, the magazine* — legacy words for the legacy renderer.
