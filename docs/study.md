# The Study — the artifact a decision rests on (2026-11-21)

A **study** is one question Jarvis investigated about one deck, and everything the call
rested on: the facts from every source with their tier and stamp, each agent's answer kept
verbatim, the call, who decided, and the pilot's words. It lives at
`data/decks/<slug>/studies/NNN-<kebab>.json`, tracked, gated by `validate-study`, and it is
what `deck-branch propose --study NNN` requires. The goldfish's read is at most one labelled
fact inside it.

## Why

The bench's decision verbs — `try`, `net-change`, `propose`, the ledger — read two goldfish
runs, the bill, Forge and mana, and nothing a person wrote. The strategist, the rules agent,
the analyst and the scout wrote nothing that survived a session. The pilot said "the goldfish
is not the decider" at least fourteen times in forty-six days; the words that kept it one
lived in `CLAUDE.md`, the vision, the doctor's charter and the release skill. The study puts
the decision where the evidence is and makes judgment a gated artifact like a measurement.

## The shape

Borrowed from `sim_findings.json` (`pilot_findings.py`): a COMPUTED skeleton and an AUTHORED
half the skeleton never touches.

**Computed on `new`, `facts --write` and `install`:**

- `facts[]` — each `{id, kind, text, tier, source, as_of?, card?, figure?, ci95?, n?}`, ids
  `S-<sha8>-NN` (a hash of deck, study and the list studied). Tiers: `verified` ✓ a
  checker-passed stack; `data` ◆ a tracked record; `coach` ★ a person's or an agent's
  judgment. Kinds: `record` (games and causes), `deckcheck`, `recon`, `oracle`, `protected`,
  `log` (the pilot's own sentence naming the card, by game), `card_read` (the debrief's
  over/under/as-expected/missed, by game), `stack`, `engine_stage`, `context`, `queue`,
  `decision`, `staged`, `price`, `neighbours` (nearest in-deck cards in the ability space),
  `coverage` (seen / partial / dark / invisible), and LAST `goldfish_profile` and
  `goldfish_row` under the asterisk.
- `goldfish_coverage` — `{named, seen, partial, dark, invisible, unknown, share}` over the
  cards the study names.
- `decklist_sha256`, `base_decklist_sha256`, `absent[]` (a source that is not here, with its
  reason — never a zero), `_meta`.

**Authored, carried forward:**

- `question`, `cards`, `swap {out, in}`, `opened`.
- `contributions[]` — `{agent, model, spawned_at, elapsed_s, slice, text, cites {facts,
  strategy, rules}, commands?, prescription?, from?}`. The agent's text is a QUOTED SOURCE,
  kept as written; its citations are extracted and checked.
- `call` — `{decision: approve|decline|no-change|open, decided_by[], pilot_words, reading,
  at}`. `decided_by` draws on `pilot, log, strategist, rules-question, data-analyst,
  card-scout, deck-doctor, recon, stacks, engine, context, goldfish, forge, mana,
  deckcheck`.
- `decided_at` — stamped by `install`, which freezes the facts.

## The verbs

```bash
manamap pilot study <slug> new "<question>" [--branch B] [--out A --in B] [--card N…]
manamap pilot study <slug> show NNN [--facts] [--json]      # --facts: the slice an agent reads
manamap pilot study <slug> add NNN --agent strategist|rules-question|data-analyst|card-scout|deck-doctor \
    (--from FILE | --text "…") [--model M] [--spawned-at T] [--elapsed S] [--slice "…"] [--command C…]
manamap pilot study <slug> call NNN --decision approve|decline|no-change|open \
    --decided-by pilot,strategist,log --pilot-words "<verbatim>" [--reading "…"]
manamap pilot study <slug> install NNN                        # strict validate, stamp, freeze
manamap pilot study <slug> facts NNN [--write] [--force]      # recompute; a decided study needs --force
manamap pilot study <slug> list
manamap pilot card-dossier <slug> "<card>" [--branch B] [--json]   # the per-card query the skeleton is built from
manamap pilot deck-notes <slug> tally                         # the debrief's card reads counted across games
manamap pilot validate-study <slug> [NNN]
```

`study` runs warm over the API as `list` and `show` only; the writing verbs run in the
terminal.

## The gate (`validate-study`)

- A decided study names `decided_by` and carries `pilot_words`.
- **The goldfish is never the sole decider**, is admitted only beside `pilot`, and never
  when a named card's coverage reads `dark` or `invisible`.
- A `strategist` contribution cites at least one `strategy:<id>`; a `rules-question`
  contribution quotes at least one CR number; every cite is well-formed and every cited
  fact exists.
- Every number in `call.reading` appears in a fact it cites by id.
- `swap.out` is a card the deck, the branch, or the branch's staged cuts actually had.
- `install` additionally requires a decided call and coverage over every named card.

## Who reads it

`deck_branch.review` puts the branch's latest study on every branch row (`info.json`, the
manifest, `deck-info`'s BRANCHES block, the pages); `deck_info.compose` carries `studies[]`;
the Deck Context renders a generated **Studies** section; `deck-info`'s NEXT asks for a study
on an open branch that has none; `propose --study` requires an approved, installed one and
freezes its call under `accepted_on`; the decisions ledger carries `decided_by`.

## How Jarvis uses it

`/jarvis` Story 7: open the study, hand each agent `show --facts`, keep every answer with
`add`, present ONE card at a time with the specific cut and the full rationale, take the
pilot's call in his words, `install`, stage on approval. NO CHANGE is a valid answer. The
goldfish's row is asked for only when every named card reads `seen` and the study wants it.

## The first study

`heliod/studies/001`: Dawn's Truce for Flawless Maneuver on `b3-v1`, approved 2026-11-21,
decided by the pilot, the deck-doctor, the data-analyst and the log; the strategist's
Teferi's Protection refused as a fourth Game Changer; the goldfish saw neither card.
