---
name: research-strategy
description: Run a strategy research pass — the strategy-researcher agent searches online sources, expands data/strategy/strategy.md, and logs the amendments in CHANGELOG.md; the loop validates, guards the write scope, and rebuilds the strategy DB. Use when the user wants strategy topics researched, the strategy doc deepened, or new schools of thought incorporated.
---

# Research pass on the strategy companion (the doc-update loop)

Updates `data/strategy/strategy.md` + `CHANGELOG.md` via the `strategy-researcher`
agent and rebuilds the RAG DB. The founder's review surface is `git diff
data/strategy/` — substance is human-judged; this loop enforces form and scope.

## Loop (max STRATEGY_RESEARCH_MAX_ITERATIONS = 3, from config.py)

1. **Topics**: from the user ("research commander politics norms") or the
   standing brief ("expand and deepen every section"). Include any `gaps` lists
   from recent `strategic_frame.json` files — deck consultations generate the
   best research topics.
2. **Snapshot**: note `git status --porcelain` before spawning (pre-existing
   dirt must not be blamed on the agent).
3. **Research**: spawn `strategy-researcher` with `MODE: research`, the topics,
   and the doc paths. It edits the doc + changelog directly (the one
   write-scoped agent) and reports amendments + sources in its final message.
4. **Scope guard**: `git status --porcelain` again — revert any change outside
   `data/strategy/` (`git checkout -- <path>`) and tell the user it happened.
5. **Form gate**: `.venv/bin/manamap pilot validate-strategy`. On failure,
   re-spawn the agent with the validator errors; do not hand-fix content
   yourself beyond mechanical formatting.
6. **Rebuild**: `.venv/bin/manamap pilot build-strategy-db` (the DB refuses to
   serve a stale doc, so this step is not optional).
7. **Report**: present `git diff --stat data/strategy/` plus the agent's
   amendments summary and open questions. Commit only when the user asks —
   the diff IS the founder red-line surface.

## Notes

- The agent must verify URLs by fetching them; treat "source could not be
  verified" flags in its report as review items, not failures.
- Videos can only be cited via transcripts/articles — never by title alone.
- If the pass stalls after 3 iterations, save nothing new (revert), report why.
