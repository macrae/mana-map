# Contributing

Two commands to a working checkout, and one to know whether you broke anything:

```bash
make setup          # Python 3.10 venv, deps in the order that works, chromium
make test           # the inner loop: ~20 s
```

Then read the landmines below. They are short, they are all things that have
actually happened here, and none of them is guessable from the code.

---

## What you can do without anything special

You do not need an API key. There isn't one — **the Python in this repo makes
zero LLM calls.** You do not need a GPU, and you do not need to run the pipeline.

**With nothing but `git clone` and Python's own web server:**

```bash
manamap serve            # viz + the local /api that makes Build's agents work
python3 -m http.server 8000   # or plain static — everything but the agent half
# localhost:8000/viz/workbench.html        THE LANDING PAGE — every deck, racked and tabled
# localhost:8000/viz/index.html            the card map, all three modes
# localhost:8000/viz/deck.html?deck=heliod one deck's dossier
# localhost:8000/manuals/p/heliod.html     its Pilot's Manual (printable, no JS)
# localhost:8000/manuals/index.html        the legacy magazine rack (frozen, unlinked)
```

The map boots on 1.8 MB because the artifacts it needs are tracked on purpose.

**After `make setup`, with no pipeline run:** **1,360 of the fast tests** —
91%, measured on an actual fresh clone —
every deterministic `manamap pilot` subcommand that reads a deck rather than the
corpus, and a byte-identical re-render of every deck page.

**Needs `manamap run`** (~40–60 min, downloads ~56 MB, internet): anything
reading `data/cards.csv` — `bracket-check`, `build-deck`, `pool-facts`,
`fetch-deck`'s card-pool checks — plus retraining and `eval-embeddings`. 129
cases skip without it, every one labelled with the command that would enable it.

**Needs [Claude Code](https://claude.com/claude-code):** the agent phases only —
generating a *new* deck's prose, engine model, debrief or prescription. Everything
deterministic is a CLI subcommand. If you do not have it, you can still work on
the frontend, the models, the pipeline, the 61 pilot subcommands, the Forge
harness, the renderer and the tests, which is nearly all of the code.

---

## Tests

```bash
make test           # non-browser, parallel, cached          ~20 s
make test-fresh     # same with nothing cached — trust this  ~30 s
make test-browser   # playwright against a real Chromium     ~4 min
make test-all       # test-fresh + test-browser
```

`pytest` on its own is `make test`. Some useful variants:

```bash
pytest -n0 -k some_name    # one test: skip the worker startup
pytest -m browser -n 4     # the viz suite
pytest -m ""               # literally everything
pytest --lf                # only what failed last time
```

### Why some tests skip, and the one kind you should look at

Two reasons, and the run tells you which. **Data gates** (`requires_data`,
`requires_rules`, `requires_strategy`) mean an artifact the test needs is
gitignored and you have not generated it — expected on a fresh clone, and
correct.

**The regenerate-and-compare cache** is the other. Four test files recompute an
artifact and compare it to the tracked copy; `test_pilot_artifact_freshness`
alone re-runs 90,000 seeded goldfish simulations. Those are pure functions of
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
`mana-map.js`. `manuals/magazine.css` is content-addressed instead, so editing
`design.py` obliges you to rebuild every manual **and the newsstand**.

**A manual must equal a fresh render of its artifacts.** `make manuals` is free
and deterministic; run it after touching the renderer and commit the result.
Four deck pages once spent days serving content their own artifacts no longer
supported, and a stale manual renders perfectly.

**Never put `data/` on Git LFS.** GitHub Pages serves LFS pointers, which would
break the deployed site for everyone. The large tracked JSON is deliberate.

**`data/` index alignment.** `projection[i]`, `cards.csv[i]` and `embeddings[i]`
are the same card. Never partially regenerate after the card count changes — go
back to the changed pipeline step and run forward from there.

**Editing an agent charter is expensive.** `.claude/agents/*.md` files are inputs
to a content-addressed cache over agent output. Changing one — even a typo —
invalidates that agent's routines across every deck, and re-running them
costs real money. (Since 2026-08-19 the board is red fleet-wide by decision; a
charter edit does not make it redder.) If the fix is cosmetic, say so in the PR. Never "re-record" a cache entry to make the board
green; that is the one rule this project holds without exception.

**Serve from the repo root.** `viz/` and `data/` must stay top-level siblings;
every fetch is `../data/<file>`.

---

## Sending a change

Branch, commit, open a PR. **`main` is protected** — it takes an approving review
and signed commits, so a direct push will be refused even if you have write
access. CI runs `make test` and checks that the manuals still rebuild
byte-identically.

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
(how the suite is organised, and eight lessons about testing this thing that
were learned the hard way).

`CLAUDE.md` is the densest engineering knowledge in the repo — roughly a hundred
paragraph-length post-mortems of real bugs. The filename says it is an
instruction file for an AI agent, and it is; read it anyway, because it is also
the closest thing to an architecture rationale.

## Code of conduct

Be decent. Assume good faith, argue about the work rather than the person, and
take a maintainer's "no" without a fight — this is a hobby project and its
scope is allowed to be narrower than your idea is good.
