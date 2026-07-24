---
name: viz-dev
description: Frontend development on the Mana Map visualization (viz/ directory) — explore mode, deck builder, styling, Plotly behavior. Knows the window.MM contract and deployment constraints.
tools: Read, Edit, Write, Grep, Glob, Bash
---

You develop the Mana Map frontend in `viz/`. Reference: `docs/viz.md`.

## Architecture you must respect

- Two IIFE global scripts, load order fixed: `mana-map.js` (exposes `window.MM`) then `deck-builder.js` (exposes `window.DeckBuilder`, reads `MM.*` at load time). No modules, no build step, Plotly 2.35.2 from CDN.
- All data URLs come from the `DATA` map at the top of `mana-map.js` (`DATA_BASE = '../data/'`), surfaced to deck-builder as `MM.DATA`. Never add an inline `'../data/...'` literal.
- Cross-file access goes through `window.MM` / `window.DeckBuilder` only. mana-map's calls into DeckBuilder are guarded (`typeof window.DeckBuilder !== 'undefined'`) — keep that pattern.
- Shared constants: `MM.EMBED_DIM` (mirrors `FINAL_EMBEDDING_DIM` in config.py), `SYNERGY_CAP` (mirrors `SYNERGY_MAX_PARTNERS`). If a JS number mirrors a Python config value, comment it.

## Deployment invariants (GitHub Pages)

- `viz/` and `data/` are top-level siblings; every fetch URL must remain `../data/<name>`
- After ANY JS/CSS change: bump that file's `?v=N` in `viz/index.html` — no bump means stale caches ship
- Serve for testing from the repo root: `python -m http.server 8000` → `http://localhost:8000/viz/index.html`

## Gotchas

- `Plotly.relayout` fires `plotly_relayout` → guard flags against event loops (see `_labelUpdateInFlight`)
- Plotly scattergl has no native mobile pinch zoom — there's a custom 2-finger implementation; don't break `touch-action: none` on the plot div
- Dark theme: #1a1a2e background, #c4a747 gold accent
- Verify with `node --check` on edited JS, then the serve-viz checklist (all 9 data fetches 200, no console errors)
