/* drill.js — dive into a selection and re-map it from the real embeddings.
 *
 * The world map is one PaCMAP layout of 34,322 cards at n_neighbors=10. That is the
 * regime that preserves global shape by squashing local shape: inside any one cluster,
 * the distances you see are mostly the projection's compromises, not the model's opinion.
 * Drilling recomputes a layout for just the selected cards from the 128-d embeddings, so
 * the structure that was compressed out becomes the whole view.
 *
 * The animation is not decoration. Points start from exactly where they sit on the world
 * map and relax toward the new layout over ~90 frames, so you can see which cards were
 * already neighbours and which ones travel. A cut would throw that information away.
 *
 * HONESTY CONSTRAINT — the one that governs this file:
 *   A drilled position is LOCAL. The same card sits somewhere else on the world map, and
 *   the two coordinate systems mean different things. There must be no state in which
 *   both are on screen without the breadcrumb saying which you are looking at. Every
 *   entry point routes through enter(), every exit through pop(), and both maintain the
 *   bar. If you add a third path, maintain it there too.
 *
 * Contract with mana-map.js:
 *   isActive()          -> is a local layout on screen right now
 *   getOverlayTraces()  -> the drill trace (one trace, per-point colour array)
 *   getDimmedIndices()  -> always null; the world is hidden outright, not dimmed
 *   getContourSource()  -> {x, y} in local coords, so Topo re-bins over the subset
 *   hidesWorld()        -> render() sets the 34K base traces invisible while true
 *
 * Row indices are indices into MM.allData == projection_2d.json == cards.csv row order,
 * which is also what embeddings.bin and regions_*.json membership are keyed by.
 */
(function () {
  'use strict';

  // Above ~2k the per-frame Plotly.restyle dominates and the settle stops reading as
  // motion. Measured: restyle on a 1,200-point scattergl trace runs ~32 ms median with
  // the full world still loaded. The cap is honest rather than silent — enter() says so
  // in the breadcrumb when it truncates.
  const MAX_DRILL = 2000;
  const SETTLE_FRAMES = 90;
  const PAIRS_PER_POINT = 8;
  const PAD = 0.06;              // fraction of the view box left as margin

  let stack = [];                // [{label, indices}] — level 0 is the world
  let pos = null;                // Float32Array [x0,y0,x1,y1,...] in plot units
  let indices = null;            // row indices of the drilled subset
  let colors = null;             // per-point colour, matched to the map's colour-by
  let emb = null;                // Float32Array view of embeddings.bin
  let rafId = null;
  let frame = 0;
  let box = null;                // {x0,y0,x1,y1} target box in plot units
  let truncatedFrom = 0;
  let candidate = null;          // a pending selection offered in the bar

  function isActive() { return indices !== null; }

  // ── Geometry ────────────────────────────────────────────────────────────

  // Embedding rows are L2-normalised at export, so the dot product IS the cosine and
  // the true chord distance on the unit sphere is sqrt(2 - 2cos). That is a real metric
  // in the model's own space — the thing the 2D projection can only approximate.
  function targetDistance(a, b) {
    const dim = MM.EMBED_DIM;
    const oa = indices[a] * dim;
    const ob = indices[b] * dim;
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += emb[oa + i] * emb[ob + i];
    if (dot > 1) dot = 1; else if (dot < -1) dot = -1;
    return Math.sqrt(2 - 2 * dot);
  }

  // Cap a set without biasing it. `slice(0, N)` takes the first N rows in cards.csv
  // order — Scryfall's export order — so a truncated drill of a 3,434-card region showed
  // whichever cards happened to be exported first, not a sample of the region. The
  // breadcrumb said "showing 2000 of 3434" and was honest about the count while silent
  // about the bias. An even stride keeps the shape of the set.
  function sampleEvenly(rows, cap) {
    if (rows.length <= cap) return rows.slice();
    const out = new Array(cap);
    const stride = rows.length / cap;
    for (let i = 0; i < cap; i++) out[i] = rows[Math.floor(i * stride)];
    return out;
  }

  function currentViewBox() {
    const gd = document.getElementById('plot');
    const xr = (gd && gd._fullLayout) ? gd._fullLayout.xaxis.range : [-50, 50];
    const yr = (gd && gd._fullLayout) ? gd._fullLayout.yaxis.range : [-50, 50];
    const x0 = Math.min(xr[0], xr[1]), x1 = Math.max(xr[0], xr[1]);
    const y0 = Math.min(yr[0], yr[1]), y1 = Math.max(yr[0], yr[1]);
    const px = (x1 - x0) * PAD, py = (y1 - y0) * PAD;
    return { x0: x0 + px, y0: y0 + py, x1: x1 - px, y1: y1 - py };
  }

  // Fit the cloud to the target box, preserving aspect. The camera never moves during a
  // drill — the points expand to fill the view instead, which costs no relayout and
  // keeps the axes' scaleanchor constraint out of the frame loop entirely.
  function normalise() {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < pos.length; i += 2) {
      if (pos[i] < minX) minX = pos[i];
      if (pos[i] > maxX) maxX = pos[i];
      if (pos[i + 1] < minY) minY = pos[i + 1];
      if (pos[i + 1] > maxY) maxY = pos[i + 1];
    }
    const w = Math.max(maxX - minX, 1e-6), h = Math.max(maxY - minY, 1e-6);
    const scale = Math.min((box.x1 - box.x0) / w, (box.y1 - box.y0) / h);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    const tx = (box.x0 + box.x1) / 2, ty = (box.y0 + box.y1) / 2;
    for (let i = 0; i < pos.length; i += 2) {
      pos[i] = tx + (pos[i] - cx) * scale;
      pos[i + 1] = ty + (pos[i + 1] - cy) * scale;
    }
  }

  // One stochastic stress-majorization sweep. Each point is pulled toward its correct
  // distance from a handful of random partners; the residual is what makes the cloud
  // wobble as it settles rather than snapping into place. alpha decays to zero, so the
  // motion has weight at the start and none at the end.
  function step(alpha) {
    const n = indices.length;
    // Chord distances live in [0, 2]; the box is tens of plot units wide. Scale so a
    // maximally-dissimilar pair wants roughly the full width of the view.
    const unit = Math.min(box.x1 - box.x0, box.y1 - box.y0) / 2;
    for (let a = 0; a < n; a++) {
      for (let k = 0; k < PAIRS_PER_POINT; k++) {
        const b = (Math.random() * n) | 0;
        if (b === a) continue;
        const want = targetDistance(a, b) * unit;
        let dx = pos[2 * a] - pos[2 * b];
        let dy = pos[2 * a + 1] - pos[2 * b + 1];
        let d = Math.sqrt(dx * dx + dy * dy);
        if (d < 1e-6) { dx = (Math.random() - 0.5) * 1e-3; dy = (Math.random() - 0.5) * 1e-3; d = 1e-6; }
        const push = ((d - want) / d) * 0.25 * alpha;
        const mx = dx * push, my = dy * push;
        pos[2 * a] -= mx; pos[2 * a + 1] -= my;
        pos[2 * b] += mx; pos[2 * b + 1] += my;
      }
    }
  }

  // ── The frame loop ──────────────────────────────────────────────────────

  function pushPositions() {
    const n = indices.length;
    const xs = new Array(n), ys = new Array(n);
    for (let i = 0; i < n; i++) { xs[i] = pos[2 * i]; ys[i] = pos[2 * i + 1]; }
    // Move one layer's points without touching the other 34,322 — rebuilding every frame
    // is the whole cost this avoids. `updateLayerBy` matches on `_isDrill` rather than a
    // trace index, which is why the Plotly-era `drillTraceIndex` scan went with it.
    if (MM.mapRenderer) MM.mapRenderer.updateLayerBy('_isDrill', { x: xs, y: ys });
  }

  function animate() {
    rafId = null;
    if (!isActive()) return;
    frame += 1;
    // Ease-out: big moves first, then settle. 1 - (t^3) keeps early frames lively.
    const t = Math.min(frame / SETTLE_FRAMES, 1);
    const alpha = 1 - t * t * t;
    step(alpha);
    normalise();
    pushPositions();
    if (t < 1) {
      rafId = requestAnimationFrame(animate);
    } else {
      settle();
    }
  }

  // requestAnimationFrame does not fire in a hidden tab. Without this, switching away
  // mid-flight leaves the layout frozen part-way — points at meaningless intermediate
  // positions, no contours, no arrival — and it never recovers, because the callback
  // that would have scheduled the next frame never runs. So: run the remaining
  // relaxation without painting and land on the settled layout.
  function finishNow() {
    stop();
    while (frame < SETTLE_FRAMES) {
      frame += 1;
      const t = frame / SETTLE_FRAMES;
      step(1 - t * t * t);
    }
    normalise();
    pushPositions();
    settle();
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden && isActive() && rafId !== null) finishNow();
  });

  // Arrival: bring back the things that cannot live inside a frame loop. Contours are
  // histogram2dcontour — main-thread SVG over the whole subset — and region labels are
  // annotations on a 150 ms debounce, so both would stutter the settle. They come back
  // once, together, and that lands as arriving somewhere.
  function settle() {
    MM.render();
    // Tell the renderer the positions moved. Its quadtree signature is layer lengths plus
    // endpoint ids — none of which change while a drill mutates coordinates in place — so
    // without this the tree still holds the world-seeded positions the drill started from
    // and every hover and click lands on the wrong card, or on nothing. That is what "the
    // points settle and then I can't interact with them" was.
    if (MM.mapRenderer) MM.mapRenderer.reindex();
    renderBar();
  }

  function stop() {
    if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  }

  // ── Entering and leaving ────────────────────────────────────────────────

  async function enter(rowIndices, label) {
    const unique = Array.from(new Set(rowIndices)).filter(i => MM.allData[i]);
    if (unique.length < 3) { MM.setStatus('Need at least 3 cards to drill.'); return; }

    if (!emb) {
      MM.setStatus('Loading embeddings…');
      emb = await MM.getEmbeddings();
      if (!emb) { MM.setStatus('Embeddings unavailable — cannot re-map.'); return; }
    }

    truncatedFrom = unique.length > MAX_DRILL ? unique.length : 0;
    indices = sampleEvenly(unique, MAX_DRILL);
    stack.push({ label: label, indices: indices.slice() });

    // Seed from where the points actually are on screen. This is what makes the
    // transition read as a dive into the map rather than a cut to a new one.
    pos = new Float32Array(indices.length * 2);
    for (let i = 0; i < indices.length; i++) {
      const d = MM.allData[indices[i]];
      pos[2 * i] = d.x;
      pos[2 * i + 1] = d.y;
    }
    colors = indices.map(i => MM.categoryColor(MM.allData[i]));
    box = currentViewBox();
    candidate = null;
    frame = 0;

    MM.render();          // world hidden, drill trace created at world positions
    renderBar();
    stop();
    if (document.hidden) {
      // Started in a background tab (or a headless check): no frames will come, so
      // land on the settled layout rather than sitting at the world positions.
      rafId = 0;
      finishNow();
    } else {
      rafId = requestAnimationFrame(animate);
    }
  }

  function pop() {
    stop();
    stack.pop();
    if (!stack.length) { clearAll(); return; }
    const level = stack[stack.length - 1];
    // Re-enter the parent level from the world so its layout is recomputed rather than
    // remembered — the box may have changed, and a stale cached layout would be a
    // position that no longer means anything.
    stack.pop();
    enter(level.indices, level.label);
  }

  function clearAll() {
    stop();
    stack = [];
    indices = null;
    pos = null;
    colors = null;
    truncatedFrom = 0;
    candidate = null;
    MM.render();
    renderBar();
  }

  // ── Entry points ────────────────────────────────────────────────────────

  // A box-select offers a drill rather than performing one: the same gesture already
  // feeds the 8-card detail stack, and silently hijacking it would be worse than a
  // button. Also the only place that reports how many cards the box actually caught.
  function offer(rowIndices, label) {
    candidate = { indices: rowIndices, label: label };
    renderBar();
  }

  function takeOffer() {
    if (candidate) enter(candidate.indices, candidate.label);
  }

  function dismissOffer() {
    candidate = null;
    renderBar();
  }

  // Region drill. Membership comes from regions_*.json, which since the clustering
  // stopped discarding its labels can say exactly which cards HDBSCAN put in a region —
  // rather than guessing by proximity to a centroid.
  async function enterRegion(regionId) {
    const data = await MM.getRegionData();
    if (!data || !data.membership) {
      MM.setStatus('This map has no region membership — re-run `manamap cluster-regions`.');
      return;
    }
    const m = /^l(\d)_(\d+)$/.exec(regionId);
    if (!m) return;
    const level = 'l' + m[1];
    const cid = parseInt(m[2], 10);
    const labels = data.membership[level];
    const rows = [];
    for (let i = 0; i < labels.length; i++) if (labels[i] === cid) rows.push(i);
    const region = data.regions.find(r => r.id === regionId);
    enter(rows, region ? region.label : regionId);
  }

  // Refuses rather than truncates. Drilling "everything" is not a thing: with no filters
  // this is all 34,322 cards, and a 2,000-card sample of the entire universe re-maps into
  // an incoherent pile whichever way you sample it. The other three entries — box-select,
  // a region label, a Find Similar result — all hand over a set that means something.
  function enterFiltered() {
    const rows = [];
    const all = MM.allData;
    // The row index is passed too: a search narrows by ROW, not by any property
    // of the record, so a predicate that only sees the card cannot honour it.
    for (let i = 0; i < all.length; i++) if (MM.passesFilters(all[i], i)) rows.push(i);
    if (rows.length > MAX_DRILL) {
      MM.setStatus(rows.length.toLocaleString() + ' cards is too many to re-map — filter below ' +
        MAX_DRILL.toLocaleString() + ', or box-select, or click a region label.');
      return;
    }
    enter(rows, MM.filterLabel());
  }

  // ── Rendering ───────────────────────────────────────────────────────────

  function getOverlayTraces() {
    if (!isActive()) return [];
    const n = indices.length;
    const xs = new Array(n), ys = new Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = pos[2 * i];
      ys[i] = pos[2 * i + 1];
    }
    // One trace, per-point colour array — so a frame is a single restyle. Splitting by
    // category would multiply the per-frame Plotly calls by the number of groups.
    return [{
      type: 'scattergl',
      mode: 'markers',
      name: 'Local layout (' + n + ')',
      x: xs,
      y: ys,
      customdata: indices.slice(),
      hoverinfo: 'none',
      marker: { size: 6, opacity: 0.95, color: colors, line: { color: '#0d0d1a', width: 0.5 } },
      _isDrill: true,
      _isDeckOverlay: true,
    }];
  }

  function getDimmedIndices() { return null; }

  // Where a card is *right now*, in whatever coordinate system is on screen. Returns
  // null for a card that is not in the drilled subset — it has no local position, and
  // inventing one (or falling back to its world position) would put a marker somewhere
  // that means nothing. Callers must handle null rather than defaulting.
  function localPosition(rowIdx) {
    if (!isActive()) return null;
    const i = indices.indexOf(rowIdx);
    return i === -1 ? null : [pos[2 * i], pos[2 * i + 1]];
  }

  function hidesWorld() { return isActive(); }

  function getContourSource() {
    if (!isActive()) return null;
    const out = new Array(indices.length);
    for (let i = 0; i < indices.length; i++) out[i] = { x: pos[2 * i], y: pos[2 * i + 1] };
    return out;
  }

  function renderBar() {
    const el = document.getElementById('drillBar');
    if (!el) return;

    if (!isActive() && !candidate) { el.style.display = 'none'; el.innerHTML = ''; return; }
    el.style.display = '';

    if (!isActive() && candidate) {
      el.innerHTML =
        '<span class="drill-count">' + candidate.indices.length + ' cards selected</span>' +
        '<button class="drill-go" onclick="Drill.take()">Drill in →</button>' +
        '<button class="drill-x" onclick="Drill.dismiss()">×</button>';
      return;
    }

    const crumbs = ['<button class="drill-crumb" onclick="Drill.home()">World</button>'];
    stack.forEach(function (level, i) {
      crumbs.push('<span class="drill-sep">›</span>');
      crumbs.push('<span class="drill-crumb is-here">' + MM.escHtml(level.label) +
                  ' · ' + level.indices.length + '</span>');
    });

    el.innerHTML =
      crumbs.join('') +
      '<span class="drill-local" title="These positions were recomputed from the 128-d ' +
      'embeddings for this selection. They are not the same coordinates as the world map.">' +
      'local layout</span>' +
      (truncatedFrom
        ? '<span class="drill-warn">showing ' + MAX_DRILL + ' of ' + truncatedFrom + '</span>'
        : '') +
      '<button class="drill-x" onclick="Drill.back()">Back ↩</button>';
  }

  window.Drill = {
    enter,
    enterRegion,
    enterFiltered,
    offer,
    take: takeOffer,
    dismiss: dismissOffer,
    back: pop,
    home: clearAll,
    isActive,
    hidesWorld,
    getOverlayTraces,
    getDimmedIndices,
    getContourSource,
    localPosition,
    sampleEvenly,
    renderBar,
    MAX_DRILL,
  };
})();
