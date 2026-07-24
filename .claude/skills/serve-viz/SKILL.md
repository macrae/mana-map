---
name: serve-viz
description: Serve and verify the Mana Map visualization locally. Use when the user wants to see the map, test viz changes, or debug frontend behavior.
---

# Serve the visualization

```bash
python -m http.server 8000    # MUST run from the repo root
# open http://localhost:8000/viz/index.html
```

The JS fetches `../data/<file>` relative to `viz/index.html` — serving from anywhere but the repo root 404s every data file. `file://` also fails (CORS).

## Verification checklist

All 9 data fetches should return 200:

```bash
for f in projection_2d.json projection_2d_ability.json embeddings.bin embeddings_ability.bin \
         regions_default.json regions_ability.json obsolescence_index.json synergy_graph.json combo_graph.json; do
  curl -s -o /dev/null -w "%{http_code}  $f\n" "http://localhost:8000/data/$f"
done
```

In the browser: map renders with region labels, map selector switches Color+Type ↔ Abilities, card click opens the detail panel, deck builder loads, no console errors.

## After changing JS or CSS

Bump the `?v=N` cache-bust on the changed file's tag in `viz/index.html` (JS scripts and the CSS link each carry one). Browsers and GitHub Pages cache aggressively — skipping the bump ships stale code. Frontend reference: `docs/viz.md`.
