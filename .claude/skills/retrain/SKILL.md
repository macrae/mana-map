---
name: retrain
description: Retrain the two embedding models and regenerate everything downstream. Use after changing MECHANICAL_TAGS, model architecture, training hyperparameters, or when embeddings need refreshing on new card data.
---

# Retrain the models

```bash
.venv/bin/manamap run --from preprocess   # if features/tags changed (MECHANICAL_TAGS etc.)
.venv/bin/manamap run --from train        # if only training/model code changed
```

Either path continues automatically through embed → reduce → export → synergy → power-creep → cluster-regions, keeping all artifacts consistent. Takes ~20–40 min on Apple Silicon (MPS).

## Critical rules

- Changing `MECHANICAL_TAGS` in `config.py` changes `MECHANICAL_TAG_DIM` → the old `model_ability.pt` cannot load. You MUST start from `preprocess`.
- Checkpoints are gitignored — a retrain permanently replaces them. Confirm with the user before starting unless they asked explicitly.
- Expected convergence: Color+Type stops ~epoch 7 near-zero loss (normal); ability model stops ~epoch 16, best val_loss ~0.05. Wildly different numbers = investigate before proceeding.
- The git-tracked viz artifacts (projections, .bin files, graphs, regions) change after a retrain — the map layout will look different. Mention this to the user; push only when they're ready to deploy.

## Verification

```bash
.venv/bin/python -m pytest                 # full suite, 261 tests
```
Then serve the viz (`python -m http.server 8000`, from repo root) and spot-check: map renders, Find Similar returns sensible neighbors, region labels appear.
