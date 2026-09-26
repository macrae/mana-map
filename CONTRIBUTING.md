# Contributing

Two commands to a working checkout, and one to know whether you broke anything:

```bash
make setup          # Python 3.10 venv, deps in the order that works, chromium
make test           # the inner loop
```

**Runtimes and test counts live in `docs/testing.md` and nowhere else** — including
here. Every file that restated them drifted, this one worst of all: it said the inner
loop was ~20 s while the measured figure was minutes.

Then read the landmines below. They are short, they are all things that have
actually happened here, and none of them is guessable from the code.

---

## What you can do without anything special

You do not need an API key. There isn't one — **the Python in this repo makes
zero LLM calls.** You do not need a GPU, and you do not need to run the pipeline.

**With nothing but `git clone` and Python's own web server:**

```bash
manamap serve            # viz + the local /api that makes Build's agents work.
                         # ALSO a warm worker: every read-only `manamap pilot` command
                         # routes through it and skips the cold import, byte-identically.
                         # It holds the OLD modules after you edit Python — restart it,
                         # or set MANAMAP_NO_DAEMON=1, or you measure the old code.
python3 -m http.server 8000   # or plain static — everything but the agent half
# localhost:8000/viz/workbench.html        THE LANDING PAGE — every deck, racked and tabled
# localhost:8000/viz/index.html            the card map, all three modes
# localhost:8000/viz/deck.html?deck=heliod one deck's dossier
# localhost:8000/manuals/p/heliod.html     its Pilot's Operating Handbook (printable, no JS)
# localhost:8000/viz/branch.html?deck=heliod&branch=splendor-v2   a candidate 99
# localhost:8000/viz/library.html          Curate — the library across piles
# localhost:8000/viz/spaces.html           what each embedding space is for
```

The map boots on 1.9 MB because the artifacts it needs are tracked on purpose.
(The legacy magazine rack at `manuals/index.html` was **deleted 2026-09-13** along
with the renderer; `manuals/` now holds `p/` and one stylesheet.)

**After `make setup`, with no pipeline run:** the large majority of the fast tests —
every deterministic `manamap pilot` subcommand that reads a deck rather than the
corpus, and a byte-identical re-render of every deck page. A fresh clone runs green
and *faster* than a developed checkout, by construction.

**Needs `manamap run`** (~40–60 min, downloads ~56 MB, internet): anything
reading `data/cards.csv` — `bracket-check`, `build-deck`, `pool-facts`,
`fetch-deck`'s card-pool checks — plus retraining and `eval-embeddings`. Those
cases skip without it, every one labelled with the command that would enable it.

**Needs [Claude Code](https://claude.com/claude-code):** the agent phases only —
generating a *new* deck's prose, engine model, debrief or prescription. Everything
deterministic is a CLI subcommand. If you do not have it, you can still work on
the frontend, the models, the pipeline, all 102 pilot subcommands, the Forge
harness, the renderer and the tests, which is nearly all of the code.

---

## Tests

```bash
make test           # non-browser, non-forge, parallel, cached
make test-fresh     # same with nothing cached — trust this
make test-browser   # playwright against a real Chromium
make test-all       # test-fresh + test-browser
pytest -m forge     # one real Forge game; needs ~/.mana-map/forge
```

`pytest` on its own is `make test`. Some useful variants:

```bash
pytest -n0 -k some_name    # one test: skip the worker startup
pytest -m browser -n 4     # the viz suite
pytest -m ""               # literally everything
pytest --lf                # only what failed last time
```

### Why some tests skip, and the one kind you should look at

Two reasons, and the run tells you which. **Data gates** — seven markers
(`requires_data`, `requires_rules`, `requires_deck`, `requires_strategy`,
`requires_roles`, `requires_rulings`, `requires_branch`) — mean an artifact the test
needs is gitignored and you have not generated it: expected on a fresh clone, and
correct.

**The regenerate-and-compare cache** is the other. Six test files recompute an
artifact and compare it to the tracked copy; `test_pilot_artifact_freshness` alone
re-runs 36 goldfish targets at 10,000 seeded games each. Those are pure functions of
files in the repo, so when no input has moved they are skipped and the run says
so:

```
176 test(s) served from the regenerate-and-compare cache (unchanged inputs).
```

It is keyed on the **content** of the inputs *and* the source of the code that
produces them, recorded only when a test passes, and stored in gitignored
`.pytest_cache/` so it can never travel to another machine or into CI. Run
`make test-fresh` before you open a PR anyway. If you add a test of this shape,
call `unchanged(...)` with every file it depends on and err toward naming a
whole directory: naming too many costs a re-run, naming too few silently serves
a stale pass.

---

## The landmines

Each of these is a real defect that shipped.

**Cache-bust the frontend, all together.** Any change under `viz/` needs `?v=N`
bumped on the changed `<script>`/`<link>` tags in `viz/index.html` **and**
`viz/deck.html`. The nine script busts in `index.html` must move as one — a test
asserts it, because a mismatched pair is how `build.js` ends up calling a stale
`mana-map.js`. `manuals/page.css` is content-addressed instead, so editing
`poh_design.py` obliges you to rebuild **every handbook**.

**A handbook must equal a fresh render of its artifacts.** `make manuals` is free
and deterministic; run it after touching the renderer and commit the result.
Four deck pages once spent days serving content their own artifacts no longer
supported, and a stale page renders perfectly. CI asserts this from outside the
code that asserts it: `make manuals`, then `git diff --exit-code`.

**Never put `data/` on Git LFS.** GitHub Pages serves LFS pointers, which would
break the deployed site for everyone. The large tracked JSON is deliberate.

**`data/` index alignment.** `projection[i]`, `cards.csv[i]` and `embeddings[i]`
are the same card. Never partially regenerate after the card count changes — go
back to the changed pipeline step and run forward from there.

**Editing an agent charter is expensive.** `.claude/agents/*.md` files are inputs
to a content-addressed cache over agent output. Changing one — even a typo —
invalidates that agent's routines across every deck, and re-running them
costs real money. If the fix is cosmetic, say so in the PR. Never "re-record" a cache
entry to make the board green; that is the one rule this project holds without
exception.

**Serve from the repo root.** `viz/` and `data/` must stay top-level siblings;
every fetch is `../data/<file>`.

---

## Sending a change

Branch, commit, open a PR. CI runs `make test` and checks that the handbooks still
rebuild byte-identically. (The maintainer commits straight to `main`; the remote's
pull-request prompt is advisory, not a gate. For anyone else a PR is how the change
gets reviewed at all.)

**Commit messages here are longer than usual and that is on purpose.** The
history is the project's real design record: what was measured, what was tried
and rejected, and what the number was. If you fixed something, the useful commit
says what the symptom was and how you know it is gone. A one-line "fix bug" is
accepted; it is just worth less to the next person.

There is no linter and no formatter — match the surrounding style. Comments here
tend to explain *why*, and often cite a measurement; that is the house voice and
you are welcome to write in it.

## Where to read next

`docs/README.md` indexes everything and separates current reference from
historical design records. The two most useful starting points are
`docs/pipeline.md` (the 15 steps, what each produces) and `docs/testing.md`
(how the suite is organised, the counts, the runtimes, and the lessons about
testing this thing that were learned the hard way). Before touching a subsystem,
read its gotcha page — `docs/gotchas-bench.md` for `pilot/` and `sim/`,
`docs/gotchas-viz.md` for `viz/`, `docs/gotchas-evidence.md` before adding a
validator. They hold the measurement behind every rule above.

`CLAUDE.md` is the densest engineering knowledge in the repo — roughly a hundred
paragraph-length post-mortems of real bugs. The filename says it is an
instruction file for an AI agent, and it is; read it anyway, because it is also
the closest thing to an architecture rationale.

## Code of conduct

Be decent. Assume good faith, argue about the work rather than the person, and
take a maintainer's "no" without a fight — this is a hobby project and its
scope is allowed to be narrower than your idea is good.
