# Mana Map

Two things live here, sharing one data layer and one CLI:

**An interactive card map** — every Magic: The Gathering oracle card (~34,300) embedded
by two small neural nets, projected to 2D, and served as a browsable atlas with a deck
builder, synergy lookup, and power-creep detection.

**Pilot's Manual** — a magazine generator. Point it at one Commander deck and it produces
a self-contained web issue: combo lines whose every step cites the Comprehensive Rules and
survived an adversarial checker, resource curves from a seeded simulation, and coaching
that says out loud when it's coaching.

The fastest way to understand the second one is to read an issue:
**Seven issues live** — [001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) · [002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) · [003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) · [004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) · [005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) · [006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) · [007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html)
· [the newsstand](https://macrae.github.io/mana-map/manuals/index.html)

Vol. 001 was hand-built. **Vol. 002 was not** — the deck was generated from a three-line
brief naming a commander and a power bracket, and the issue's headline finding is that the
deck's own combo count was wrong.

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

Three things to know:

- **Serve from the repo root.** The page fetches `../data/*`, so `viz/` and `data/` must
  stay top-level siblings. Opening `viz/index.html` as a `file://` URL fails on CORS.
- The clone carries ~123 MB of tracked data; the map loads ~100 MB of it into the browser.
  That's deliberate — see [Landmines](#landmines). (Two of the tracked files,
  `combo_details.json` and `card_roles.json`, are for the deck builder and the agents; the
  browser never fetches them.)
- Plotly loads from a CDN, so the page needs internet even though the data doesn't.

What you get: two maps (one clustered by color and type, one by what cards *do*), Find
Similar via embedding neighbours, Find Synergies via rule-based complementarity — these
are different algorithms, see `docs/architecture.md` — a deck builder across 8 formats
with a six-factor recommender, and obsolescence badges for cards with strictly-better
replacements.

## Generate a manual for your own deck

This half is more involved, and honest about why: **the manual pipeline needs
[Claude Code](https://claude.com/claude-code).** The Python in this repo makes *zero* LLM
calls — it is deterministic infrastructure (fetching, simulating, validating, rendering)
that AI agents drive from the outside. A full generation runs ~330k tokens across four
serially-dependent agents; an invocation cache is what makes iterating on it affordable.

### 1. Environment

**Python 3.10 specifically** — PyTorch has no wheels for 3.13/3.14, and the pins
(`sentence-transformers<4`, `numpy<2`) target torch 2.2.2. `pyproject.toml` says `>=3.10`,
which is more permissive than reality.

```bash
python3.10 -m venv .venv
.venv/bin/pip install llvmlite==0.41.1 numba==0.58.1   # must come first on macOS
.venv/bin/pip install -e ".[dev]"
```

That ordering is not cosmetic: pacmap pulls numba, and installing it afterwards triggers a
source build of LLVM.

### 2. One-time databases

```bash
.venv/bin/manamap pilot download-rules      # the Comprehensive Rules text
.venv/bin/manamap pilot build-rules-db      # ~3.9K chunks, chunk ID = rule number
.venv/bin/manamap pilot build-strategy-db   # the strategy companion's RAG index
```

Both are gitignored and must be built locally. `CR_RULES_URL` in `config.py` is pinned to
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
manual shows the actual cards in your deck, with the right artist and the right art. A
name-only list works but yields default reprints and a visibly weaker issue.

`fetch-deck` short-circuits when the decklist hasn't changed; use `--force` after oracle
errata.

### 4. The files you write by hand

Nothing scaffolds these. `data/decks/goblin-storm/` is the worked reference.

| File | Why it's manual |
|---|---|
| `decklist.txt` | It's your deck — **unless you let the builder write it**, see below |
| `issue.json` | Volume, date, cover price, next issue. **The build hard-exits without it** — a *generated* date would break byte-identical rebuilds |
| `goldfish_targets.json` | Which key-piece sets are worth simulating is a judgment call — though `/build-deck` derives it from the plan's declared engines |

### 4b. Or don't write a decklist at all

If you have a commander in mind rather than a list, the builder makes one:

```bash
# author data/decks/<slug>/brief.json: commander, bracket (1-5), playstyle
.venv/bin/manamap pilot build-deck <slug> --write-decklist   # deterministic, no agents
.venv/bin/manamap pilot fetch-deck <slug>
```

That alone produces a legal, tier-conditioned, goldfishable 99. Running the `/build-deck`
skill on top adds the agent loop, which is what makes it *good* rather than merely legal —
and the whole path from brief to published issue is proven: see Vol. 002.

### 5. The agent phases

Run these from Claude Code in the repo root. Each is a skill in `.claude/skills/`:

| Step | Produces | Pure CLI? |
|---|---|---|
| `/build-deck` | A 99 from a brief: pool → architect ⇄ critic, bracket-gated | no |
| `manamap pilot bracket-check <slug>` | Computed bracket floor + its evidence | **yes** |
| `/resolve-stack` | A verified combo line: resolver → validator → adversarial checker | no |
| `manamap pilot goldfish <slug>` | Resource curves from 10k seeded games | **yes** |
| `/write-manual` | Strategic frame, coaching, body prose | no |
| `/design-issue` | The issue plan — cover, departments, headlines | no |
| `manamap pilot validate-issue <slug>` | Form gate over the plan | **yes** |

### 6. Build and read

```bash
.venv/bin/manamap pilot build-manual <slug>
.venv/bin/manamap pilot build-index          # run after build-manual
open manuals/<slug>.html
```

Rendering is free, deterministic and repeatable — the same artifacts always produce the
same bytes.

---

# For developers

## The shape

Two pipelines, same pattern: each stage writes artifacts the next one reads, and every
stage is independently runnable and testable.

```
Card map   download → extract → preprocess → train ×2 → embed → reduce
                    → combos → export → synergy → power-creep → regions → card-roles → viz

Build      brief.json → build-deck → bracket-check → architect ⇄ critic → decklist

Manual     fetch-deck → goldfish + RAG DBs → agents author JSON
                    → validators gate → renderer builds → GitHub Pages
```

`manamap run` drives the first (14 steps, ~40–60 min, internet at two of them).
`manamap pilot <cmd>` drives the second (26 subcommands). All constants live in
`src/manamap/config.py`; both CLIs are registry-driven with lazy imports.

Two lightweight fusion MLPs (~180K params each) produce the 128-dim embeddings; the text
encoder stays frozen. They answer different questions and are not interchangeable. The
**layout** model organises the map by colour and type and feeds the projection only. The
**function** model answers whether two cards do the same job, and is the sole source of
similarity — Find Similar, the walk and drill all read it whichever map is on screen.

That split exists because the alternative was measured and was bad: when similarity followed
the displayed map, the colour/type space was using 3.2 of its 128 dimensions and *Doubling
Season*'s nearest neighbours came back as arbitrary green enchantments. `manamap
eval-embeddings` (step 14) scores every space against a hand-authored golden set so a claim
like that is a number rather than an opinion.

## Extension points

| To add… | Touch |
|---|---|
| A pipeline step | `STEPS` in `pipeline.py`; the module exposes `main()` |
| A pilot command | `PILOT_STEPS` + `_DECK_COMMANDS` + argparse in `pilot/registry.py`; module exposes `main(args)` |
| A magazine department | `DEPARTMENTS` in `pilot/issue_spec.py` — changes every issue; treat it like `config.py` |
| A data file the viz reads | The `DATA` map in `viz/js/mana-map.js`, plus a `.gitignore` negation |
| A synergy rule, tag, or threshold | `config.py`, nowhere else |
| A deckbuilding role | `ROLE_PATTERNS` in `config.py`, then re-run `manamap card-roles` |
| A bracket rule | `BRACKETS` / `COMBO_BRACKET_TAGS` / `MASS_LAND_DENIAL` in `config.py` |
| Builder tuning | `DECK_ROLE_BUDGET` / `DECK_BUILD_WEIGHTS` in `config.py` |
| An agent cache routine | `AGENT_ROUTINES` in `config.py` |

## Three contracts worth understanding

**Evidence tiers.** Every section of a manual wears a badge: ✓ rules-verified,
◆ data-derived, ★ coaching. `validate-issue` enforces that a department cannot claim a
tier the system didn't grant it — costume never earns the badge.

**Validation gates — form in code, meaning by agent, publication by verdict.** A mechanical
validator checks structure: does every step cite a rule, does that rule exist, is the quote
a verbatim substring. A *separate adversarial agent* then fetches each full rule and judges
whether it actually supports the claim. Only `pass` renders. Failed artifacts are kept —
they document open questions.

**Determinism.** Agents return JSON and never write HTML. That keeps the renderer a pure
function of committed artifacts, byte-identical on rebuild and enforced by tests — and it
is *why* the issue date is authored, why goldfish is seeded, and why image URLs get their
cache-busters stripped.

## Testing

```bash
.venv/bin/python -m pytest        # 1,055 tests (1,016 fast + 39 browser)
```

325 card-pipeline + 601 pilot. Five skip markers in `tests/conftest.py` gate on the last
artifact of each stage, so **skips on a fresh clone are expected and correct**. Unit tests
build inline fixtures — no fixture files. Paths always come from `manamap.config`, so the
suite is CWD-independent and honours `MANAMAP_DATA_DIR`.

## Landmines

- **`python -m manamap.pipeline` starts the full 40–60 minute run** with no arguments and
  no confirmation, overwriting trained models. Use the `manamap` CLI.
- **Never put `data/` on Git LFS.** GitHub Pages serves LFS pointer files, not content —
  it would silently break every fetch on the deployed site. The ~123 MB of tracked JSON
  is deliberate.
- **Index alignment**: `projection[i]` ≡ `cards.csv[i]` ≡ `embeddings[i]`, positionally
  (card names duplicate). Never partially regenerate after a card-count change.
- **Cache ordering**: check → spawn → write → validate → **record last**. Recording before
  validating poisons the cache.
- **Bump `?v=N`** on the script and CSS tags in `viz/index.html` after any frontend edit.
- **The combo graph is format-agnostic.** Commander Spellbook lines may quietly assume a
  card is *your commander* — `"Infinite commander casts"` in `produces` is the tell.
  Verify before publishing; Judge's Desk Case A-004 is the cautionary tale.
- **Rebuild the strategy DB after editing `strategy.md`** — the loader hard-errors on a
  sha256 mismatch.

## Deployment

GitHub Pages serves the repo directly. There is no root index; the two entry points are
`/viz/index.html` and `/manuals/index.html`. Pushing to `main` deploys.

## Where to read next

| Doc | Covers |
|---|---|
| `CLAUDE.md` | Orientation, environment, gotchas — the densest single page |
| `PLAN.md` | Current state and what's next |
| `STYLEv3.md` | The magazine's editorial and design constitution |
| `docs/architecture.md` | Models, mechanical tags, synergy rules, power creep, regions |
| `docs/pipeline.md` | All 14 steps: inputs, outputs, runtimes, when to re-run |
| `docs/data-artifacts.md` | Every `data/` file: producer, size, git status, consumers |
| `docs/viz.md` | Frontend structure, the `window.MM` API, Pages layout |
| `docs/testing.md` | Test layout, skip markers, conventions |
| `docs/pilot.md` | Evidence contract, rules and strategy DBs, the magazine layer |
| `docs/deck-builder-v2.md` | The deck builder: bracket engine, role taxonomy, architect ⇄ critic loop |
| `docs/agent-cost.md` | Where LLM spend lives, per-routine costs, the cache |

## Non-goals

Lint, formatting and CI are intentionally absent — this is a single-author project and the
test suite is the gate. Revisit if that changes.

---

Card images and card text are property of Wizards of the Coast. This is unofficial fan
content permitted under the Wizards of the Coast Fan Content Policy, not approved or
endorsed by Wizards.
