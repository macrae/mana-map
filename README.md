# Mana Map

[![tests](https://github.com/macrae/mana-map/actions/workflows/test.yml/badge.svg)](https://github.com/macrae/mana-map/actions/workflows/test.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![the map](https://img.shields.io/badge/live-the%20card%20map-8b5cf6)](https://macrae.github.io/mana-map/viz/index.html)

**A workbench for crafting, experimenting, researching and analysing Commander decks.**
It is built around one idea: **a claim about a deck is worth what the experiment behind it
is worth.** `docs/vision.md` is the page everything else is written against.

At the centre is **the swap loop**, on two engines with different jobs:

- **A seeded, PAIRED Monte Carlo goldfish** — the decision instrument. `try` plays the
  current list and a candidate through the same games, seed for seed, and answers with an
  interval on the difference in about ten seconds.
- **Forge**, the real rules engine, run headless and seeded — a **targeted probe** since
  2026-10-04. It answers what the goldfish cannot: does the AI actually play this card,
  and what happens against blockers and removal.

Around them sit the things that make an experiment mean something: a deterministic builder,
a rules-citation loop for lines that must be *proven* rather than measured, dated web
reconnaissance, deterministic card mining over 34,955 cards, and a frontend that surfaces
the results.

Optimised for one player; open-sourced so anyone can stand up their own bench, not so
anyone else is supported.

## The hypothesis loop

```
  a question                     →  an experiment              →  a result you can cite
  "is this swap better?"            try --out A --in B            damage@T10 +0.83 [+0.68, +0.98], paired
  "does the AI play this card?"     forge-cast-check --card X     drawn 28, cast 0 — held
  "is this line lethal?"            /resolve-stack                ✓ or refuted, with CR cites
  "how fast does it go off?"        goldfish                      mean t4.19, 89% by t6
  "what do strong lists run?"       deck-recon, /prescribe        ranked, cited, skeptic-checked
  "what would fix this axis?"       deck-audit + card-search      candidates that move the number
  "is this change worth buying?"    net-change, then propose      a trade, priced, with a bill
```

**`try` is the flagship.** `manamap pilot try <slug> --out "A" --in "B"` — about ten
seconds, nothing written: every card in and out with its roles and what the goldfish can
see of it, the pilot's keep list (`protected.json`), colour sources before and after, and
the `net-change` rows with a **paired** interval on the difference. Every goldfish game
takes its own seed and the two lists are aligned slot for slot, so the noise of two
independent samples cancels — a list against itself reads exactly zero. A swap that
survives becomes a branch (`--stage NAME`), and `net-change` grades it on one
pre-registered objective plus twelve Holm-corrected rows. Never a comparison of two
marginal intervals: that is the overlap fallacy, and the key that reported it was removed.

## Six pages over one data layer

**The workbench** (`viz/workbench.html`) — **the landing page**: every deck you own, in
racks by whether it is sleeved, **waiting on cardboard**, on the bench or history; or as
one fleet table across record, stages, evidence, table and open work — sortable by
*recently played*, *needs game logs*, *needs analysis*, *optimisations identified* and
*waiting on cardboard*. Each deck carries a derived **next**, and three named links: its
Pilot's Operating Handbook, its dossier, and where it sits on the map.

**The card atlas** — every Magic oracle card (~35,000) embedded by two small neural nets.
It opens on **one card**: hover it, click a relation, and its neighbours join a
force-directed graph you grow by clicking. Load one of your own decks and it lights up with
its commander ringed. The 34,955-point atlas is one click away, and drifts slowly at
altitude, settling as you zoom in to read. Three relations, each precomputed so a click is
instant: **similar** (embedding neighbours), **synergy** (rule-based complements, each edge
labelled with its rule), **outclassed by** (strictly-better replacements). Boot costs 1.9 MB.

**The deck page** (`viz/deck.html?deck=<slug>`) — the workbench surface: what to do next,
where the deck stands, every list it has been, what limits it, the engine, **the
experiments and simulation runs with their intervals**, prescriptions, the captain's log,
open questions, and the deck's own constellation. It renders `info.json` — the shape
`deck-info` composes — rather than re-deriving anything, so it cannot disagree with the
command that owns each figure.

**The branch workbench** (`viz/branch.html?deck=<slug>&branch=<name>`) — one candidate 99
and the decision about it: what it would become and what it was accepted on, the verdict,
every measured row with its definition and a plain-language reading, reward / risk / cost,
and the bill. A branch is a deck that does not exist, so it gets its own page rather than
growing into the dossier.

**Curate** (`viz/library.html`) — the library across piles, at a size you can read. The
atlas's drawer keeps one card; this cuts forty. A pile rail, a grid with multi-select and
bulk move/remove, a pinned pane, and sort plus colour/type/role filters over the same card
index the atlas boots on — fetched *after* the first render, so a failed fetch degrades to
the piles, the name box and the three factless sorts rather than to a blank page. A card
the index cannot resolve gets its own chip and sorts **last**, never as 0.

**The embedding-space appendix** (`viz/spaces.html`) — what each embedding space is for and
how they differ, because "similar" means two different things in this repo and the wrong one
returns arbitrary same-colour cards. Linked from the shared nav on every page.

## The commands behind it

- `try <slug> --out A --in B` — the swap loop: a paired goldfish screen in ~10 s.
- `net-change <slug> --branch B` — a candidate 99 graded: one primary, twelve exploratory
  rows, Holm-corrected, paired per game. A Forge loss beside it is a warning, not a block.
- `goldfish` — seeded Monte Carlo resource development, 10,000 games in seconds. Every channel
  it models is **opt-in per deck**, so what it can see is a thing you declare and can check.
- `forge-cast-check <slug> --card X` — the Forge probe: does the AI play this card?
- `simulate` / `experiment` — Forge games against a named pod, when a question needs a
  table; every rate with its interval, graded only beside an A/A at the same N.
- `build <slug> --commander "<name>"` — **the one command**: a brief becomes a legal,
  bracket-gated, *measured* 99 on the bench in six stages, about ten seconds, no agents.
  Omit `--commander` and it proposes three and halts. The dev batch is the **goldfish**, not
  Forge — a twelve-minute Forge batch is ~20 games, whose minimum detectable difference is 42
  points. (`build-deck` is the underlying
  scoring step and is still callable on its own.)
- `deck-audit` — 16 axes, each carrying the verbatim `strategy.md` quote that sets its target.
- `card-search` — deterministic mining over the corpus: identity, oracle/name regex, role,
  cmc, and `--owned` against your physical boxes.
- `deck-info <slug>` — the whole join, and a derived **next**.
- `deck-branch` — a candidate 99 you cannot yet sleeve: stage swaps, measure it against the
  deck with `net-change`, then **`propose`** it as the next version and wait for the
  cardboard. The decision is frozen; the blocker is recomputed from your boxes on every
  read, so a proposal un-blocks itself when a card lands in one.
- `deck-version` — every list the deck has been, from git, joined to the games played on it;
  `deck-version <slug> paper <ref>` is how you say *this one is sleeved*.
- `check-in` — a paper list you typed up becomes `decklist.txt`: it diffs, and **refuses a
  silently-wrong list** rather than guessing which side is right.
- `model-coverage` — what the model cannot see, before the games: seen / **dark** (feeds a
  channel that is off) / invisible.
- `deck-state` / `deck-delete` — archive, retire, supersede, revive; and the one destructive
  verb, which refuses a deck that was ever sleeved, played or published.
- `regen` — rebuild the fleet in dependency order after a model change, parallel across decks
  but never splitting the games inside one run, so it stays bit-identical.
- `deck-notes add` → `/debrief` → `/captains-log` → `/prescribe` — the table, structured,
  read, then answered.
- `/resolve-stack` — a board (authored, or **lifted from a simulated game**) resolved with
  Comprehensive Rules citations and adversarially checked. The ✓ tier.
- `/analyze-engine` — the deck's machine as eight stages, solid where a stack proves a line.
- `/write-manual` + `/poh-procedures` → `build-poh` — the notes a person writes, then the
  handbook.

Under it all is a **three-tier evidence contract** that never moves: ✓ rules-verified, ◆
data-derived (seeded where randomness is involved), ★ coaching — labelled judgment, never
disguised as measurement. **A figure travels with its interval, its N and its limits, or it
does not travel** — enforced in code, not by convention. Every agent returns JSON a
validator checks; the Python makes zero LLM calls; the deployed site and your machine run
the same code.

**Four things this is honest about.** Forge's AI pilots every seat *including yours*, and
rates itself "poor to ok in control, pretty bad for combo" — a sentence quoted verbatim in
every run record, which makes a control deck's win rate a lower bound on the pilot; that is
why Forge is a probe. **Five decks carry a captain's log — 19 games, every entry debriefed**
(Ur-Dragon 7, Edgar 6, Gishath 2, Goblin Storm 2, Heliod 2, as of 2026-10-05) — which is
nineteen, not a sample: the log,
debrief and prescription surfaces are built and tested, and barely used. **Six of fourteen
decks are marked as built in paper** (Edgar, Gishath, Goblin Storm, Heliod, Sharknado,
Ur-Dragon); whether a deck exists as cardboard is an assertion only the pilot can make, so
an unlocked deck says it is unlocked rather than being assumed playable — and that one
authored flag is what decides whether the whole chain runs for it automatically.

And **the goldfish has no blockers and only models what a deck declares.** It cannot price
a lord's static pump or a blocker's worth, so its verdict on board quality is not evidence. Every channel — draw, combat,
Treasure, sacrifice, the copy commander, the spell count — is off until that deck's
`goldfish_targets.json` switches it on, so a card feeding an off channel measures as exactly
nothing and looks identical to a card that does not help. `model-coverage <slug>` is the
preflight that names those cards, and it exists because that confusion has cost this project
a whole branch.

**The Pilot's Operating Handbook** (`manuals/p/<slug>.html`, from `manamap pilot
build-poh`) — each deck's self-contained printable page: the game plan, the mulligan, the
verified lines argued, the engine, the numbers with their intervals, and the emergency and
normal procedures a person writes. **No `<script>` anywhere**, so it rebuilds
byte-identically and is trustworthy offline and in print — which is exactly why live map
embeds go on the dossier instead. The *legacy magazine* it replaced was deleted on
2026-09-13; what it measured is kept in `docs/gotchas-magazine-legacy.md`.

---

# For users

## Explore the card map

**Two commands. No install, no pipeline, no API keys.** Every data file the frontend
needs is committed.

```bash
git clone git@github.com:macrae/mana-map.git && cd mana-map
python3 -m http.server 8000
```

Open <http://localhost:8000/viz/index.html>.

*Contributing? [CONTRIBUTING.md](CONTRIBUTING.md) is two commands and a list of landmines.
[docs/README.md](docs/README.md) sorts the documentation into current reference and
historical design records, which saves reading 12,000 lines of the latter by accident.*

Three things to know:

- **Serve from the repo root.** The page fetches `../data/*`, so `viz/` and `data/` must
  stay top-level siblings. Opening `viz/index.html` as a `file://` URL fails on CORS.
- The clone carries **250 MB of tracked data** (and `git clone` transfers rather more
  than that — the history is ~254 MB), but **discovery boots on 1.9 MB gzipped** — a slim
  card index plus a precomputed neighbour table. The heavy artifacts load only if you ask
  for what needs them: the 3.0 MB projection when you open the atlas, the 17.9 MB embedding
  matrix never on the discovery path at all. That the *clone* is large is deliberate — see
  [Landmines](#landmines). (Two of the tracked files,
  `combo_details.json` and `card_roles.json`, are for the deck builder and the agents; the
  browser never fetches them.)
- d3 loads from a CDN, so the page needs internet even though the data doesn't.

What you get: two maps (one clustered by color and type, one by what cards *do*), three
relations on every card — **similar** via embedding neighbours, **synergy** via rule-based
complementarity, **outclassed by** via the obsolescence index (these are different
algorithms, see `docs/architecture.md`) — a **Build** mode that lights up a deck or a pool inside the 34K atlas, and obsolescence
badges for cards with strictly-better replacements.

Build shows a deck's footprint in card space — its role histogram, mana curve segmented by
the current overlay, colour load and verified lines — and hands work back out as a
`brief.json` the deterministic builder reads. It does **not** score cards: evaluation comes
from the pipeline through the agent loop, so there is exactly one scorer.

The atlas is a launchpad: clicking a relation there carries you into the walk seeded on that
card, rather than doing something subtly different because you happened to be in a different
mode.

## Set up the bench for your own decks

This half is more involved, and honest about why: **the agent phases need
[Claude Code](https://claude.com/claude-code).** The Python in this repo makes *zero* LLM
calls — it is deterministic infrastructure (fetching, simulating, validating, rendering)
that AI agents drive from the outside, and most of the bench (`deck-info`, `deck-version`,
`deck-notes`, `goldfish`, `simulate`, `model-coverage`, `regen`, every validator) is pure CLI.
The agent routines are what cost tokens — the doctor and its skeptic, the resolver and its
checker, the engineer and its critic, the strategy researcher, the notes writer, the handbook's
procedures author, the debrief, the captain's log and the cartographer — and an invocation
cache is what makes iterating on them affordable. Eighteen agent charters live in
`.claude/agents/`, twenty-one skills in `.claude/skills/`. `docs/agent-cost.md` has the
breakdown; **two deliberate opt-in exceptions** are the only LLM calls reachable from Python
itself: `serve.py`'s `ask` bridge, which shells out to `claude -p` as a polled job, and
`mm ask`, whose SDK is an optional extra the core install does not pull.

### 1. Environment

```bash
make setup
```

That is the whole step. It checks for **Python 3.10 exactly** — PyTorch publishes no wheels
for 3.13+, and the pins (`sentence-transformers<4`, `numpy<2`) target torch 2.2.2 — then
installs in the one order that works and downloads chromium for the browser tests.

If your 3.10 is not called `python3.10` on `PATH` (a conda or pyenv build, say):

```bash
make setup PYTHON310=$(pyenv which python3.10)
```

By hand it is three commands, and the **order is not cosmetic**: pacmap pulls numba, and
installing it afterwards triggers a source build of LLVM.

```bash
python3.10 -m venv .venv
.venv/bin/pip install llvmlite==0.41.1 numba==0.58.1   # must come first
.venv/bin/pip install -e ".[dev]"
```

Developed on macOS/arm64. Linux should work apart from MPS device selection; Windows is
untested.

### 2. One-time databases

```bash
.venv/bin/manamap pilot download-rules      # the Comprehensive Rules text
.venv/bin/manamap pilot build-rules-db      # ~3.9K chunks, chunk ID = rule number
.venv/bin/manamap pilot build-strategy-db   # the strategy companion's RAG index
```

All three outputs are gitignored and must be built locally. `CR_RULES_URL` in `config.py` is pinned to
a specific rules release — update it when Wizards ships a new one.

### 3. Your deck

```bash
mkdir -p data/decks/<slug>/{stacks,decisions}
# write data/decks/<slug>/decklist.txt
.venv/bin/manamap pilot fetch-deck <slug>
.venv/bin/manamap pilot validate-deck <slug>
```

**Use a Moxfield export.** Lines like `1 Zada, Hedron Grinder (SLD) 2406 *F*` carry the
set, collector number and foil marker, and `fetch-deck` resolves those *first* — so the
deck page shows the actual cards in your deck, with the right art. A name-only list works
but yields default reprints.

`fetch-deck` short-circuits when the decklist hasn't changed; use `--force` after oracle
errata.

### 4. The files you write by hand

Nothing scaffolds these. `data/decks/goblin-storm/` is the worked reference.

| File | Why it's manual |
|---|---|
| `decklist.txt` | It's your deck — **unless you let the builder write it**, see below |
| `goldfish_targets.json` | Which key-piece sets are worth simulating is a judgment call — and which channels the model may switch on for this deck. `/build-deck` derives a first version from the plan's declared engines; no agent ever edits it afterwards |
| `deck_versions.json` | Authored, and the home of two claims only you can make: `paper` (this exact list is **sleeved**, which is what puts the deck on the automatic chain) and `lifecycle` (archived / retired / superseded). Written through `deck-version paper` and `deck-state`, never by hand |
| `issue.json` | **Optional now.** It carried the magazine's authored identity; with that renderer deleted, only the display name is still read, and most decks have none |

### 4b. Or don't write a decklist at all

If you have a commander in mind rather than a list, the builder makes one:

```bash
# author data/decks/<slug>/brief.json: commander, bracket (1-5), playstyle
.venv/bin/manamap pilot build-deck <slug> --write-decklist   # deterministic, no agents
.venv/bin/manamap pilot fetch-deck <slug>
```

That alone produces a legal, tier-conditioned, goldfishable 99. Running the `/build-deck`
skill on top adds the agent loop, which is what makes it *good* rather than merely legal —
and the whole path from brief to a finished deck is proven (hapatra was built this way).

### 5. The lifecycle, then the loop

Run the skills from Claude Code in the repo root (each is in `.claude/skills/`).
**`/publish-deck` sequences the lifecycle** and `manamap pilot deck-info <slug>` tells you
where a deck stands and what to do next; start with both.

| Step | Produces | Pure CLI? |
|---|---|---|
| `/build-deck` | A 99 from a brief: pool → architect ⇄ critic, bracket-gated | no |
| `manamap pilot bracket-check <slug>` | Computed bracket floor + its evidence | **yes** |
| `manamap pilot goldfish <slug>` | Resource curves from 10k seeded games | **yes** |
| `manamap pilot deck-map <slug>` | The constellation: local layout + clusters | **yes** |
| `/analyze-engine` | The engine: stages, lines, what a stack actually proves | no |
| `/resolve-stack` | A verified line: resolver → validator → adversarial checker | no |
| `manamap pilot deck-audit <slug>` | 16 cited axes plus engine activation | **yes** |
| `manamap pilot model-coverage <slug>` | What the model cannot see, before the games: seen / dark / invisible | **yes** |
| `/write-manual` | The pilot's notes: game plan, mulligan, line intros, threats, matchups | no |
| `/poh-procedures` | The half a person writes: emergency and normal procedures, rules of engagement | no |
| `manamap pilot build-poh <slug>` + `build-index` | The Pilot's Operating Handbook (`manuals/p/`, deterministic, no `<script>`) | **yes** |

Then the loop the bench exists for — all CLI except the two agents:

| | | |
|---|---|---|
| `manamap pilot deck-branch <slug> new/stage/…` → `net-change --branch` | a candidate 99, measured against the deck | **yes** |
| `manamap pilot deck-branch <slug> propose <name> --as v1.0.2` | accept it, and wait for the cardboard | **yes** |
| `manamap pilot deck-version <slug> [tag …]` | commit the list; every version numbered from git | **yes** |
| `manamap pilot fetch-opponent "<commander>"` / `simulate <slug> --vs <pod> --games N` | your table, in Forge, seeded | **yes** |
| `manamap pilot check-in <slug> --from <file>` | a paper list becomes `decklist.txt`; refuses a silently-wrong one | **yes** |
| `manamap pilot deck-version <slug> paper <ref>` | **this list is sleeved** — the one claim that puts the deck on the automatic chain | **yes** |
| `manamap pilot deck-notes <slug> add "…" --result win\|loss --cause <code>` | the captain's log; `--cause` is a closed vocabulary so the dossier can count how games end | **yes** |
| `/debrief <slug>` | the note, structured and routed | no |
| `/captains-log <slug>` | each night plain, and **the read** — what the games have taught | no |
| `/prescribe <slug> "<question>"` | the doctor's answer, priced and skeptic-checked | no |
| `manamap pilot sim-scenario <slug> <run> --game G --turn T --stack` → `/resolve-stack` | a simulated board, proven | mixed |

---

# For developers

## The shape

Two pipelines, same pattern: each stage writes artifacts the next one reads, and every
stage is independently runnable and testable.

```
Card map    download → extract → preprocess → train ×2 → embed → reduce
                     → combos → export → synergy → power-creep → regions → card-roles
                     → viz-index → eval-embeddings   (the quality gate, step 15)

Build       brief.json → build-deck → bracket-check → architect ⇄ critic → decklist

Experiment  decklist → fetch-deck → goldfish ─┐
            pod       → fetch-opponent ───────┼→ simulate / experiment → parse → analysis
                                              └→ sim-scenario → /resolve-stack → ✓

Change      decklist → deck-branch new → stage → net-change → propose → merge → a version
                                                        └ blocked on cardboard, derived

Fleet       a model change → regen (dependency order, parallel across decks)
            goldfish → mana-analysis → net-change → diagnose → benchmark → deck-info

Surface     artifacts → deck-info --write → info.json ─┐
            build-index → index.json ──────────────────┴→ viz/deck.html
```

`manamap run` drives the first (15 steps, ~40–60 min, internet at two of them).
`manamap pilot <cmd>` drives the rest (**123 pilot subcommands** against 28 top-level ones).
All constants live in `src/manamap/config.py`; both CLIs are registry-driven with lazy
imports.

### The warm worker, and the read-only MCP server

`manamap serve` does three jobs, and only the first is obvious. It serves `viz/` and a local
`/api` the deployed site does not have — and it is also a **warm worker**: with it running,
every read-only `manamap pilot <cmd>` routes through `/api/cli` and skips the cold import,
byte-identically. Measured: `query-rules` 6.93s → 0.16s, `deck-facts` 1.44s → 0.14s,
`deck-audit` 2.26s → 0.59s. It **fails open** — no server, or any error at all, and the
command runs locally exactly as before. `MANAMAP_NO_DAEMON=1` opts out.

One gotcha that costs ten confusing minutes every time: **the server holds the old modules
until you restart it.** After editing Python, restart `serve` or set `MANAMAP_NO_DAEMON=1`,
or you will measure the code you just replaced.

`.mcp.json` registers `manamap.mcp_server`, which hands Claude Code seven **read-only** tools
over the same warm process — `deck_state`, `fleet`, `search_docs`, `search_code`, `stats`,
`run_command`, `command_help` — so an agent gets structured data instead of parsing prose
(`deck-status heliod` is 2.9s cold, 0.003s warm, byte-identical). There is no MCP SDK
dependency: the protocol is JSON-RPC over stdio and the subset a tool server needs is ~150
lines, the same reasoning that keeps scipy out of `sim/stats.py`. **It cannot write**, and the
gate is `serve._cli` imported rather than restated, so the two surfaces cannot drift apart.

## Forge — the rules engine, as a probe

*Code: `src/manamap/sim/{forge,parse,experiment,bridge,opponents,validate_sim,engine_casts}.py`.
Design, the spike and the verdict: `docs/simulation.md`.*

**Since 2026-10-04 Forge answers narrow questions; it does not decide.** Overnight pod runs
left the decision loop — a day per answer, an MDE of ~0.14 at 200 games, and an AI that
mis-pilots some decks into floors — so a swap is screened by `try` on the paired goldfish,
and Forge's first job is `forge-cast-check`: does the AI play this card at all. What
follows is how the harness works, which is still how every probe works.

**Forge was chosen by measurement, not preference.** Three things were checked before
committing to it: every log line parses, 4-seat Commander runs headless, and `-s` makes a
run byte-replayable. Writing a rules engine was shelved for one narrow deterministic case.

**A run is seeded.** `simulate` converts each seat's `decklist.txt` to a Forge `.dck`
*through the repo's own parser* — so a deck analysed and a deck simulated can never
disagree about what is in it — then runs N Commander games across J JVMs. The default seed
derives from the configuration, so **the default replays**; `--seed` asks for a new sample.
Job *i* runs `seed_base + i`, and a same-id re-run is refused without `--force`.

**The record is one tracked JSON.** Outcomes, per-game rows, every seat's decklist sha,
Forge and Java versions, wall time, the seeds, and an `analysis` block with Wilson
intervals for rates and normal intervals for means. `validate-sim` **re-derives that
analysis from the kept logs** where they exist and form-checks where they do not, so a
figure in the record is not merely asserted.

**Two turn counts, and they are not the same.** `round` is the winner's own turn count
(Forge's `Game Outcome: Turn N`); `global_turn` is the game's last `Turn:` line. In a
4-seat game round 8 is global turn ~32.

**Three things the parser gets right on purpose.** Tokens are reported two honest ways —
`token_resolutions` (creation abilities that resolved; blind to X and doubling) and
`tokens_observed` (distinct ids seen acting) — because Forge names a token on first *use*,
never on creation. Damage figures see **damage only**: a drain kill shows in `life_by_turn`
and `eliminated_how`, never in a damage total. And **commander damage is per defender**,
because CR 903.10a asks for 21 from one commander on one *player* — 60 damage spread over
three seats wins nothing.

**Every aggregate carries median, min, max and an interval.** A mean over a skewed sample
is a true number that describes no game: one measured arm read mean 17.42 with a **median
of 0**, the whole difference being two games out of twelve.

**The bridge closes the loop.** `sim-scenario <slug> <run> --game G --turn T --step S`
lifts a board out of a simulated game into a `game_state` v2 scenario, which
`/resolve-stack` then proves with rules citations. Life and lands are exact; hand size is
an estimate; every approximation is written into `extras.reconstruction_notes`. This has
run for real: radagast stack 008 is a board lifted from a simulated game, resolved and
checker-passed in three iterations, and the checker caught two triggers the author missed
that the log confirms.

**The AI caveat is not a footnote, and it is not one caveat.** Every seat is a Forge AI
including yours, and Forge's own rating — "poor to ok in control, pretty bad for combo" — is
quoted verbatim in every run record's `assumptions`. Measured: no AI profile flies a hold-up
deck better than Default (Default 3/6, Experimental 2/6, Reckless 2/6 over seeded games), so
Default stays the default. Three sharper consequences, each of which has cost a real
conclusion:

- **A result on a deck whose engine the AI never cast is a floor, and the record says so.**
  One seat cast Wheel of Fortune once and Windfall never in 60 games while *discarding*
  Windfall three times — the log's own statement that the card was held and passed over.
  `record["engine_casts"]` carries per-card cast / activated / discarded for your seat, and
  `simulate` prints "held and never cast" at its tail. Check it before reading any rate.
- **The AI will not sacrifice for a benefit its evaluator cannot price**, so a Forge result
  on a sacrifice deck is a floor. `Ashnod's Altar` is free and was **0 for 59 castings**;
  `Indulgent Aristocrat` costs {2}, puts a visible counter on the board, and activates 0.41
  per cast. Cost is not the discriminator — a visible board payoff is. Prefer a trigger over
  an activation.
- **Sometimes it is not a floor but a different deck.** Zada, Hedron Grinder copies a
  single-target spell across your board; over 60 games she was cast 100 times and her trigger
  fired **21**, because the evaluator prices `Brute Force` on Zada exactly as it prices it on
  anything else. Every card was cast, just never at her. So that deck's three measured win
  rates are void rather than conservative, and the A/B is not rescued by comparing them. The
  preflight is one grep for the commander's trigger in the logs; the route to evidence is
  `sim-scenario --stack` on one of the 21 real boards, proven by citation.

## The goldfish — the other engine

*Code: `src/manamap/pilot/goldfish*.py`. The channel-by-channel reference:
`docs/simulation.md`.*

Forge plays real games slowly; the goldfish plays 10,000 seeded hands of **resource
development against nobody**, in seconds. It answers questions about a curve — how fast does
the mana arrive, when is the engine assembled, what does the board look like on turn eight —
and it is wrong to ask it anything about a table.

**Everything it models is opt-in per deck**, declared in that deck's
`goldfish_targets.json`. Draw, combat, Treasure, sacrifice and deaths, discard, the spell
count and storm, magecraft, and four commander abilities that only one corpus card each
possesses are all separate flags. This is not configurability for its own sake: a channel
that is on for every deck is a channel that has to be right for every deck, and the
alternative — a card silently measuring as zero because nothing reads it — is the single most
expensive failure mode this repo has. Hence two rules, both enforced by tests:

- **A flag the model sets is a claim the model must act on.** `treasure_doubler` once shipped
  set-and-never-read; fifteen candidate cards came back byte-identical.
- **Teach the casting predicate in the same commit as the ability.** Every casting loop selects
  on a channel, so a card matching none of them sits in hand for ten turns while its profile
  says precisely what it would have done. Found five times in one session, and only recognised
  as a class on the fourth.

**Three limits worth stating up front**, because they decide which engine to ask. It has **no
blockers**, so its verdict on board *quality* is not evidence — a go-wide refactor it
preferred on damage, kill rate and card advantage lost 31/400 to 50/400 in Forge, because 1/1
tokens do not connect. **Lands enter untapped, always**, so it cannot rank two lands that make
the same colours; `mana-analysis` and `mana-fit` are deterministic for exactly that reason and
are the whole of the evidence for a land swap. And `meta.model_version` is a sha over the
model's own source, so **a model change makes every derived figure in the fleet stale** and
says so rather than quietly reporting the old number.

## The embedding models

*A technical overview. Code: `src/manamap/training/{model,train,train_ability}.py`,
`src/manamap/ingest/{extract,preprocess}.py`, `src/manamap/analysis/eval_embeddings.py`.
Every constant named below lives in `config.py`.*

Two lightweight fusion MLPs (~180K params each) produce the 128-dim embeddings; the text
encoder stays frozen. They answer different questions and are not interchangeable. The
**layout** model organises the map by colour and type and feeds the projection only. The
**function** model answers whether two cards do the same job, and is the sole source of
similarity — the *similar* relation, the walk and drill all read it whichever map is on screen.

That split exists because the alternative was measured and was bad: when similarity followed
the displayed map, the colour/type space was using 3.9 of its 128 dimensions and *Doubling
Season*'s nearest neighbours came back as arbitrary green enchantments. `manamap
eval-embeddings` (step 15) scores every space against a hand-authored golden set so a claim
like that is a number rather than an opinion.

### The problem

34,890 Magic cards, each a short piece of natural language plus a dozen structured
attributes, into a metric space where "these two cards do the same job" is a nearest-neighbour
query. There is no click-stream and no relevance judgements — supervision has to be
manufactured from the cards themselves, which is most of what makes this interesting.

### How a card is decomposed

One card becomes nine parallel inputs. Nothing is learned end-to-end from raw text; the
sentence encoder is frozen and everything else is a small learned table over an explicit
feature.

| block | shape | how it is built |
|---|---|---|
| frozen text | 384 | `all-MiniLM-L6-v2` over a synthesised sentence (below) |
| supertype | 1 → 16 | `nn.Embedding(10, 16)` — Creature, Land, Instant… |
| rarity | 1 → 8 | `nn.Embedding(7, 8)` |
| colour identity | 1 → 32 / 8 | `nn.Embedding(33, ·)` over the 32 observed WUBRG subsets |
| layout | 1 → 16 | `nn.Embedding(18, 16)` — normal, split, transform, adventure… |
| continuous | 2 | normalised CMC; normalised EDHREC rank |
| keywords | 50 | multi-hot over the 50 most frequent keywords |
| mechanical tags | 33 | multi-hot, regex over oracle text (function model only) |
| structured | 15 | power/toughness (3) + mana pips (6) + colour features (6) |

Two details that are load-bearing rather than incidental:

**The sentence is synthesised, and the card's name is deliberately excluded.**
`build_embedding_text` emits `"{type_line}. Cost {mana_cost}. {P}/{T}. {oracle_text}.
Keywords: {…}"`. The name used to lead the string and was buying similarity off shared
tokens rather than shared function — *Rhystic Study* matched *White Rhystic Study* at 0.951,
*Sol Ring* matched *Sisay's Ring*. A name is also a large fraction of a short card: *Sol
Ring*'s entire text is eleven words, three of them the name. Dropping it moved held-out
recall@10 from 0.187 to 0.248 and median rank from 159 to 129.

**Continuous features use fixed scales, never per-run min-max.** EDHREC rank divides by a
hardcoded 50,000, power/toughness by 15. A per-run normalisation makes the same card's
features differ between pipeline runs, which silently destroys comparability between two
runs' embeddings — this was a real bug.

### Architecture

A fusion MLP. Categorical blocks go through embedding tables, the two high-cardinality
multi-hot blocks (keywords, mechanical tags) through `Linear + ReLU`, and everything is
concatenated into one wide vector fed to a three-layer trunk:

```
concat[…] → Linear(d_in, 256) → ReLU → Dropout(0.1)
          → Linear(256, 128)  → ReLU → Dropout(0.1)
          → Linear(128, d_out)
```

**181,272 trainable parameters** for the layout model, **192,672** for the function model.
That is small on purpose: the frozen 384-dim MiniLM output is the only component that needed
scale, and it is amortised across every run.

### The output split — the one piece of real design

The layout model returns `F.normalize(x)`, 128 dims, and that is the whole story.

The function model does something else. Its trunk emits 96 dims and a **separate
`Linear(384, 32)` with no ReLU** projects the frozen text into the remaining 32. Each half
is L2-normalised independently, then scaled by `√(1−W)` and `√W` with `W = 0.3`:

```python
learned = F.normalize(x) * sqrt(1 - W)
text    = F.normalize(text_proj(text_emb)) * sqrt(W)
return torch.cat([learned, text], dim=1)      # already unit-norm
```

Because the squared weights sum to 1, the concatenation is unit-norm and the dot product of
two cards is **exactly**

```
sim(a, b) = 0.7 · cos_learned(a, b) + 0.3 · cos_text(a, b)
```

This exists because of a measured failure. The previous function model scored **0.093
recall@10 against 0.187 for the frozen text it was built from** — training was subtractive,
and the model had quietly learned to discard the only signal that was working. The split
makes discarding it *structurally impossible*: the text's contribution is a fixed fraction
set by architecture, not a fraction the optimiser is free to drive to zero. The rectifier is
omitted from `text_proj` for the same reason — this half exists to preserve the text
geometry, and a ReLU folds half of it away.

### Objectives

**Layout model — `TripletMarginLoss`, margin 0.3.** Positives are drawn from the same
`(supertype, primary_colour)` group with two fallbacks; negatives must differ on *both*.
This task is nearly trivial, which is the point — its only job is to give PaCMAP something
with legible colour/type structure to project. Its effective dimensionality of 3.9/128 is
not a defect for that job; it *is* the job.

**Function model — symmetric in-batch InfoNCE, τ = 0.05.** The dataset yields
`(anchor, positive)` pairs only; the batch supplies negatives.

```python
scores = (anchors @ positives.T) / temperature
labels = torch.arange(len(scores))
loss   = 0.5 * (cross_entropy(scores, labels) + cross_entropy(scores.T, labels))
```

At batch 256 that is **255 negatives per anchor for one forward pass**, against the old
triplet loss's single mined negative. The replacement was diagnostic, not fashionable: a
margin loss stops producing gradient the moment it is satisfied, which for a task this easy
was around epoch 3, so nothing pressured the model to preserve structure *within* a class.

### Positive mining

With no labels, the positive-selection rule is effectively the loss function. Three tiers,
per anchor:

1. **Rarest specific role first.** A 53-role taxonomy (`ROLE_PATTERNS`) gives a *specific*
   role to 73.2% of the 31,830 commander-legal cards, at 1.62 specific roles each
   (`card_roles.json`'s `meta.specific_coverage`; total coverage including the
   `threat:body` fallback is 89.6%). Roles are sorted by group size ascending — two cards
   sharing `doubler:tokens` (11 cards) say far more about each other than two sharing
   `value:etb` (5,580), so the positive is spent on the anchor's most specific claim.
2. **≥2 shared mechanical tags** (the old rule, now a fallback). It covered only 46.9% of
   the corpus, so for most cards it *was* the random tier wearing a better name.
3. **Random.**

`ROLE_BODY_FALLBACK` is excluded deliberately: it labels all 19,050 creatures, so "shares
this role" would be barely narrower than "is a creature" and would rebuild the trivial task
this design exists to escape. Mining is scoped to each split's own indices — a validation
positive drawn from the training set is leakage.

**No hard-negative mining, stated as a non-change.** Random in-batch negatives are safe
here: measured on this corpus, a random pair at batch 256 has a 0.004% chance of being a
true near-duplicate, about 0.01 false negatives per anchor. That would *not* survive hard
mining, which selects nearest-non-positives by construction while 39% of cards have a text
neighbour above 0.75 — it needs a similarity ceiling, and that is a second variable.

Optimiser: Adam, lr 1e-3, batch 256, 10% validation, early stopping on patience 5, seed 42.

### Similarity ranking, and how it is served

Embeddings are L2-normalised at build time, so cosine is a plain dot product and top-k is
`argpartition` over one matrix-vector product — no index structure, no approximation, 34,890
rows is small.

Serving is the constrained part, because the browser must branch **synchronously** mid-gesture
and cannot download a 16.8 MB float matrix. `neighbours.bin` precomputes, per card, 12
similar + 10 synergy + 5 obsoleted-by row ids (`NEIGHBOURS_K_*` in `config.py`): `uint16`
ids, similarities quantised to `uint8` — **1.9 MB gzipped for the whole discovery boot,
against 18.4 MB before.**

The quantised value is used for edge length only, and **ordering is array order.** Re-sorting
client-side by the lossy value changes the top-10 for roughly two thirds of cards, because
the space is a narrow cone — median pairwise cosine 0.714, so 8 bits over the observed range
is coarser than the gaps being ranked. It would read as a model regression rather than a
precision bug. The header carries a SHA-256 of the embeddings it was built from and a test
fails if they diverge, because a stale table parses fine and answers confidently.

The 2D map is a separate artifact: PaCMAP, `n_components=2, random_state=42`, over the
**layout** embeddings only. Nothing reads the projection for similarity.

### Evaluation

`manamap eval-embeddings` scores every space against `data/eval/similarity_golden.json` — 40
hand-authored groups, 12 dev / 28 test. It must **stay** hand-authored: training mines its
positives from roles and tags, so an eval derived from those would only measure whether
training memorised its own supervision.

Three metrics, deliberately including two geometric ones, because recall alone hid the
collapse for months:

- **recall@k / median rank** against the golden groups.
- **effective dimensionality** — participation ratio of the PCA spectrum, `(Σλ)²/Σλ²`. Reads
  as "how many of the nominal dimensions are actually in use"; equals *d* for an isotropic
  *d*-dimensional cloud and 1 for a line.
- **neighbour spread** — mean cosine gap between the 1st and 50th neighbour. Near zero means
  the top neighbours are indistinguishable and whichever one is returned first is an artefact
  of float ordering.

Current shipped artifacts, test split:

| space | dim | eff. dim | spread | r@10 | r@50 | med. rank |
|---|---:|---:|---:|---:|---:|---:|
| layout (colour+type) | 128 | 3.89 | 0.0061 | 0.086 | 0.139 | 1148 |
| function (ability) | 128 | 27.31 | 0.0323 | 0.232 | 0.464 | **76** |
| text baseline (frozen MiniLM) | 384 | 51.39 | 0.1341 | **0.244** | 0.414 | 126 |

**Read that table honestly.** Against the previous function model — 5.97 effective dims,
0.093 recall@10, median rank 995 — the rebuild is a large win, and the current space is
clearly better at depth: r@50 0.464 vs 0.414, median rank 76 vs 126. But it is still **0.012
behind the frozen text at r@10**, and the eval prints a warning saying so on every run. It
is not fixed. `tests/test_embedding_quality.py` also holds a deliberately still-failing
`xfail(strict=True)` gate on neighbour spread — 0.0323 against a 0.05 target — whose
threshold was not lowered to match the result.

Two standing rules around this harness:

- **Do not tune hyperparameters on it.** Sweeping the text weight looked like a win (0.258
  r@10) until selecting on `dev` picked a different value and the two splits disagreed. At
  this sample size those differences are noise.
- **Quote the test split.** `dev` was consumed while diagnosing.

## Extension points

| To add… | Touch |
|---|---|
| A pipeline step | `STEPS` in `pipeline.py`; the module exposes `main()` |
| A pilot command | `PILOT_STEPS` + `_DECK_COMMANDS` + argparse in `pilot/registry.py`; module exposes `main(args)` |
| A deck lifecycle phase | `STAGES` in `pilot/deck_status.py`, or the next person will not find it |
| A pod seat | `manamap pilot fetch-opponent "<commander>"`, or a `decklist.txt` under `data/opponents/<slug>/` |
| A figure the sim reports | `game_facts` + `aggregate` in `sim/parse.py`, then re-derive every run with `--analyze` — the record is compared against the logs, so an added key must be backfilled |
| A panel on the deck page | a `*Panel(d)` function in `viz/js/deck-view.js` returning `''` when its artifact is absent, plus the artifact's filename in `deck_manifest.gather_entries` if a browser cannot list it |
| A field the deck page reads | `deck_info.compose`, then `deck-info <slug> --write` for every deck — `info.json` is committed and staleness-gated |
| A section of the handbook | `poh_spec.SECTIONS` **and** `poh.RENDERERS` — the spec declares ten sections and the map holds seven, so a section added to one alone never renders |
| A data file the viz reads | The `DATA` map in `viz/js/mana-map.js`, plus a `.gitignore` negation |
| A goldfish channel | a reader in `goldfish_profiles.py`, **the casting predicate in `goldfish_turn.py` in the same commit**, a flag the deck declares, an entry in `model_coverage`, and a corpus sweep in the commit message. Skip any one and the channel measures as zero on every deck |
| A metric the branch report prints | `net_change.METRICS` — a test asserts it matches `ROWS` in both directions, because a figure whose definition a reader has to look up gets guessed at |
| A synergy rule, tag, or threshold | `config.py`, nowhere else |
| A deckbuilding role | `ROLE_PATTERNS` in `config.py`, then re-run `manamap card-roles` |
| A bracket rule | `BRACKETS` / `COMBO_BRACKET_TAGS` / `MASS_LAND_DENIAL` in `config.py` |
| Builder tuning | `DECK_ROLE_BUDGET` / `DECK_BUILD_WEIGHTS` in `config.py` |
| An agent cache routine | `AGENT_ROUTINES` in `config.py` |

**One thing that is someone's, not the project's.** `data/collection/*.txt` lists a
physical card collection — `deck-history` reads it to tell a swap you already own from
one you would have to buy, and it is the only ownership question left in the repo. The
tracked files are the maintainer's, kept as a worked example. Point
`MANAMAP_COLLECTION_DIR` at your own boxes, or at nothing: an absent directory means no
ownership claim rather than an error.

## Three contracts worth understanding

**Evidence tiers.** Every artifact and every figure carries a tier: ✓ rules-verified,
◆ data-derived (seeded where randomness is involved; *sampled* said out loud where it cannot
be), ★ coaching. A validator per artifact enforces that nothing claims a tier it was not
granted — costume never earns the badge.

**Validation gates — form in code, meaning by agent, publication by verdict.** A mechanical
validator checks structure: does every step cite a rule, does that rule exist, is the quote
a verbatim substring. A *separate adversarial agent* then fetches each full rule and judges
whether it actually supports the claim. Only `pass` renders. Failed artifacts are kept —
they document open questions.

**Determinism.** Agents return JSON and never write HTML. That keeps every renderer a pure
function of committed artifacts, byte-identical on rebuild — and asserted from *outside* the
code that asserts it: CI runs `make manuals` and then `git diff --exit-code`, because
otherwise the Makefile is the one thing nothing checks (and it was, until `make manuals` broke
on CI's first run). It is also *why* goldfish and Forge runs are seeded, why a dated artifact
carries an authored date rather than a generated one, and why image URLs get their
cache-busters stripped.

## Testing

```bash
make test          # the fast tier — what you run all day
make test-fleet    # the slow tier: every producer re-run per deck, fleet constants
make prepush       # both; before a push. CI runs both.
make test-browser  # the playwright suite, local only
pytest -m forge    # one real Forge game; needs ~/.mana-map/forge
```

**Counts and runtimes live in `docs/testing.md` and nowhere else**, with the markers, the
eight skip conditions, the regenerate-and-compare cache and the rules for writing a test.
A fresh clone skips what it cannot build and says which command would enable it, so skips
there are expected; `make test-fresh` is the one to trust before a PR.

## Landmines

The ones that cost the most; `CLAUDE.md` has one line per rule and the gotchas pages under
`docs/` hold the measurements.

- **`python -m manamap.pipeline` starts the full 40–60 minute run** with no arguments and
  no confirmation, overwriting trained models. Use the `manamap` CLI.
- **Never put `data/` on Git LFS** — GitHub Pages serves pointer files, not content.
- **Index alignment**: `projection[i]` ≡ `cards.csv[i]` ≡ `embeddings[i]`, positionally.
  Never partially regenerate after a card-count change.
- **An edit to any of the four goldfish files is a model change** — its bytes are the
  `model_version` stamp — so it means `manamap pilot regen --jobs 8 && make manuals`.
- **`manamap serve` holds the old modules.** Restart it after editing Python.
- **Bump `?v=N`** on the script and CSS tags in `viz/index.html` after any frontend edit.
- **Cache ordering for agents**: check → spawn → write → validate → **record last**.
- **Scryfall leaves `mana_cost` empty on transform and MDFC layouts** — read
  `common.front_field(card, "mana_cost")`, never `card["mana_cost"]`.
- **Ownership means a BOX**, never deck membership (`collection.owned_names()`).

## Deployment

GitHub Pages serves the repo directly. There is no root index; the entry points are
**`/viz/workbench.html` (the landing page — start here)**, `/viz/index.html` (the card
atlas), `/viz/deck.html?deck=<slug>` (one deck's dossier),
`/viz/branch.html?deck=<slug>&branch=<name>` (one candidate 99), `/viz/library.html`
(Curate), `/viz/spaces.html` (the embedding-space appendix) and `/manuals/p/<slug>.html`
(a deck's Pilot's Operating Handbook). Pushing to `main` deploys.

**The version list is committed and does render.** `deck-version` derives it by walking git,
and the commit that changes `decklist.txt` receives its sha *after* anything written in the
same commit — so `versions.json` is written on the *following* commit rather than the same
one, and the newest row can lag by one until then. That is a lag, not the structural blocker
this section used to describe: the panel renders from the tracked file like every other panel,
and a panel whose artifact is genuinely absent still renders nothing rather than an error.

## Where to read next

`docs/README.md` indexes and sorts all of it — current reference above, historical design
records below, which saves reading 12,000 lines of the latter by accident.

| Doc | Covers |
|---|---|
| `docs/vision.md` | **Start here.** Who this is for, what the bench does, what is live / legacy / next |
| `docs/prd.md` | **What is being built**: three environments, five epics, the metrics catalog, and four resolved decisions. `vision.md` says what the bench *is*; this says where it is going |
| `CLAUDE.md` | Orientation, environment, gotchas — the densest single page |
| `PLAN.md` | Current state and what's next |
| `/publish-deck` | The deck lifecycle: every phase in order, with its gate |
| `docs/simulation.md` | Both engines: Forge's spike and harness, the parser, the pod, the bridge — and the goldfish channel-by-channel |
| `docs/pilot.md` | Evidence contract, the bench's commands and artifacts, rules and strategy DBs |
| `docs/architecture.md` | Models, mechanical tags, synergy rules, power creep, regions |
| `docs/pipeline.md` | All 15 steps: inputs, outputs, runtimes, when to re-run |
| `docs/data-artifacts.md` | Every `data/` file: producer, size, git status, consumers |
| `docs/viz.md` | Frontend structure, the `window.MM` API, Pages layout |
| `docs/testing.md` | Test layout, skip markers, conventions — **and the only stated runtimes** |
| `docs/agent-cost.md` | Where LLM spend lives, per-routine costs, the cache |

**The gotcha pages are the expensive part.** `CLAUDE.md` loads into every session; these do
not, and they hold every measurement this project has paid for, verbatim. Read the one that
covers what you are about to touch.

| Page | Read before touching |
|---|---|
| `docs/gotchas-bench.md` | `src/manamap/pilot/`, `src/manamap/sim/` — much the largest |
| `docs/gotchas-viz.md` | anything under `viz/` |
| `docs/gotchas-evidence.md` | a validator, a citation, `engine.json` |
| `docs/gotchas-analysis.md` | `src/manamap/analysis/` — synergy, power creep, roles, regions |
| `docs/gotchas-magazine-legacy.md` | the deleted magazine renderer, kept because it was measured |

Historical design records, for provenance only — none of it describes live code:
`docs/history/manual-v5-spec.md` (the compact deck page that replaced the magazine),
`docs/history/agent-audit-2026-08-19.md` (the audit behind the pivot), and
`docs/prd-2026-08.md` (the superseded PRD that ~27 `PRD-v1 §N` citations resolve against).
`STYLEv3.md`, the magazine's constitution, was **deleted 2026-08-25**; its `STYLEv3 §N`
comments resolve through `git show 23e8cec:STYLEv3.md`.

## Non-goals

**Lint and formatting** are intentionally absent. Match the surrounding style; the test
suite is the gate. Adding a formatter to 20,000 lines would produce one enormous diff and
no information.

CI *was* on this list, on the grounds that a single-author project does not need it. It is
here now — `.github/workflows/test.yml` runs the fast suite and checks that every deck page
still rebuilds byte-identically — because a pull request from someone else arrives
unverified otherwise.

---

Card images and card text are property of Wizards of the Coast. This is unofficial fan
content permitted under the Wizards of the Coast Fan Content Policy, not approved or
endorsed by Wizards.
