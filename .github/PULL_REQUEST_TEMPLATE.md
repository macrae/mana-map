<!-- Thanks for contributing. CONTRIBUTING.md has the setup and the landmines. -->

## What this changes, and why

<!-- One or two sentences. If it fixes something, say what the symptom was —
     this repo's commit history is a record of measured defects, not a changelog,
     and yours will be read by whoever hits the next one. -->

## How you know it works

<!-- A number, a command output, a before/after. "Tested manually" is fine for a
     doc fix and not for anything else. -->

---

- [ ] `make test` passes
- [ ] `make test-fresh` passes, if you touched anything the test cache covers
      (the artifact/manual freshness tests, the tracked-artifact validators)

**Only if your change touches these:**

- [ ] **Frontend (`viz/`)** — bumped `?v=N` on the changed script/CSS tags, and
      on *all nine* script tags in `viz/index.html` together if any moved. Ran
      `make test-browser`.
- [ ] **The renderer (`build_manual.py`, `design.py`, `issue_spec.py`)** — ran
      `make manuals` and committed the result. Every published issue must equal
      a fresh render, and the stylesheet is content-addressed, so a `design.py`
      edit moves the hash in all ten pages.
- [ ] **`data/` artifacts** — did NOT add them to Git LFS. Pages serves LFS
      pointers and it would break the deployed site.
- [ ] **An agent charter (`.claude/agents/*.md`)** — you have read the note in
      CONTRIBUTING.md about what this invalidates. A one-word fix here costs a
      re-spawn across nine decks.
