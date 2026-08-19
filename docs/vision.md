# Vision — a lab bench for one pilot's paper decks

*The one page every other document is written against. Last revised 2026-08-19. If a
doc, a docstring or a charter disagrees with this page, this page wins and the other
is stale.*

## Who this is for

One person: a Commander player in Orinda who builds paper decks, plays them at a table
of friends, and wants to get **measurably better** — as a builder, as a pilot, and as a
repeated winner. The software is a **lab bench**, not a publication: it exists to sharpen
that one player's judgment, and it is open-sourced so anyone can stand up their own
bench, not so anyone else is supported.

The user is a **deck scientist and a pilot**. A scientist runs controlled experiments,
keeps a log, and trusts a number only when it arrives with its method and its limits. A
pilot needs the same things at the table, faster: what the deck is for, what has to be
true for it to work, which of its lines are proven, and what to do on turn four.

## What the bench does, end to end

```
 decide a list ── version it ── (simulate it) ── play it ── log it ── debrief ── ask ── prescribe ── swap ── repeat
                 deck-version    simulate         paper      deck-notes  /debrief   /prescribe            deck-version
```

| you want to… | the bench gives you | tier |
|---|---|---|
| know where a deck stands, right now | `deck-info <slug>` — version, record, status, figures, and a derived **next** | ◆ |
| keep the list honest across swaps | `deck-version` — every list the deck has been, from git, joined to the games played on it; tags you name; `restore` | ◆ |
| measure the deck against nobody | `goldfish` — seeded Monte Carlo resource development, Treasure and combat opt-in | ◆ seeded |
| measure it against a **table** | `simulate <slug> --vs <pod>` — N seeded games in Forge, real rules, your pod's decks; win rate with an interval, who kills you and how, the kill curve, token pay-off | ◆ seeded |
| remember what happened at the table | `deck-notes add` — the captain's log, in your words, stamped with the list you held | authored |
| turn a note into work | `/debrief` — what the note says, in structure: seats, cards that over/under-performed, takeaways, questions routed to the loop that can settle them | ★ |
| ask the deck a question | `/prescribe <slug> "…"` — the doctor, scoped to your question, reading the log and the sim: ranked adds that close a named axis, cuts priced, skeptic-checked | ◆+★ |
| know what is **true** about a line | `/resolve-stack` — a board (authored, or **lifted from a simulated game**) resolved step by step with Comprehensive Rules citations, adversarially checked | ✓ |
| understand the machine | `analyze-engine` — the deck's engine as eight stages and the lines between them, solid where a stack proves it, dashed where it is a reading | ✓◆★ |
| read it before game one | the deck page (today the legacy magazine; next the compact manual) | all |

## The evidence contract — the part that never moves

| | tier | granted by |
|---|---|---|
| ✓ | rules-verified | a stack artifact whose every step cites a real CR rule verbatim, then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | deterministic Python over committed artifacts. **Seeded** where randomness is involved (goldfish, Forge runs): same inputs, same bytes. **Sampled** is said out loud where it cannot be (the earliest Forge runs). |
| ★ | coaching | labelled judgment. Useful, never disguised as measurement. |

Every agent returns JSON a validator checks; no agent writes prose that pretends to a
tier it was not granted; a number travels with its interval, its N and its limits. The
frontend never calls an LLM; the deployed site and your machine run the same code.

## What is live, what is legacy, what is next (2026-08-19)

**Live** — the whole loop above on the CLI: `deck-info`, `deck-version`, `deck-notes`,
`/debrief`, `/prescribe`, `simulate` (+ `validate-sim`, `sim-scenario`, `fetch-opponent`),
`goldfish`, `/resolve-stack` (v1 boards and v2 game states), `analyze-engine`,
`/diagnose-deck`, the build loop, the card atlas in `viz/`.

**Legacy, frozen** — the magazine renderer (`build_manual`, `issue_spec`, `design`,
`validate_issue`, STYLEv3) and the artifacts only it reads on the nine published decks
(`issue_plan.json`, the panel keys, `card_roles`/`mana_base`/`upgrades`,
`considering.json`). It still renders; nothing regenerates those inputs; it is replaced
by the compact deck page in `docs/manual-v5-spec.md`. Code and docs about it are marked
LEGACY and left accurate rather than rewritten.

**Next** — `experiment` (A/B two versions under the same seeds and table); the first real
entries in the captain's log; Forge AI profiles per seat; the deck page in the viz
(notes, versions, sim, prescriptions, manual as a tab); an agent in your seat for a
handful of games once the AI's play is the thing limiting the measurement.

## Vocabulary

*deck* (a 99 + commander, `data/decks/<slug>/`) · *version* (a content-distinct
`decklist.txt` in git; `V1`…) · *the pod* (your opponents, `data/opponents/`) · *run*
(N Forge games, one record) · *stack* (a scenario + its cited resolution + the checker's
verdict) · *game state v2* (seats that can act; CR step names; actions) · *the log*
(authored), *the debrief* (derived) · *prescription* (one question to the doctor) ·
*the doctor / the skeptic / the resolver / the checker / the engineer / the critic*
(agents, always in pairs where a claim reaches a decklist or a ✓).

Not in the vocabulary any more: *issue, volume, department, columnist, byline, the Short
List, the magazine* — legacy words for the legacy renderer.
