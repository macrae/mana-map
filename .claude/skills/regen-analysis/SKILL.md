---
name: regen-analysis
description: Fast re-run of the analysis-only pipeline steps (synergy graph, obsolescence index, map regions) after editing SYNERGY_RULES, obsolescence thresholds, or region clustering/naming parameters. No retraining involved.
---

# Regenerate analysis artifacts (steps 10–12)

```bash
.venv/bin/manamap synergy && .venv/bin/manamap power-creep && .venv/bin/manamap cluster-regions
```

Runs in ~2 minutes; uses existing embeddings, no retraining.

## When this is the right tool

- Edited `SYNERGY_RULES` in `config.py` → `synergy` (and optionally the other two)
- Edited `OBSOLESCENCE_*` thresholds → `power-creep`
- Edited region params (`REGION_*`, HDBSCAN sizes) → `cluster-regions`

If `MECHANICAL_TAGS` changed, this is NOT enough — use the retrain skill (tag dims invalidate the ability model).

## What changes on disk

All three outputs are **git-tracked** (the deployed viz fetches them):
`data/synergy_graph.json` (~8–27MB), `data/obsolescence_index.json` (~5–8MB), `data/regions_default.json` + `data/regions_ability.json` (~16–27KB).

Commit them with the config change that motivated the regen. Diff sanity checks:
- synergy: `python -c "import json; g=json.load(open('data/synergy_graph.json')); print(len(g))"` — card count with partners should be stable ± a few %
- power-creep: integration test asserts 5,000–16,000 flagged cards
- verify with `.venv/bin/python -m pytest tests/test_pipeline_integration.py`
