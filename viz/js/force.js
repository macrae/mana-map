/* force.js — the walk through card space.
 *
 * A fourth map mode, and the first thing here that is not Plotly. Cards become nodes,
 * true 128-d cosine distance becomes link length, and a velocity-Verlet simulation gives
 * the graph weight: it settles, it wobbles, you can grab a node and fling it and the rest
 * of the graph follows. Then you branch — click a card and its nearest neighbours in the
 * full 34,322-card corpus are pulled in, the simulation reheats, and the graph grows in
 * the direction you were curious about. That is the ride.
 *
 * WHY IT IS NOT PLOTLY. Plotly draws a scatter; it has no notion of a node you can drag
 * or an edge that means something. This is also the first half of the renderer migration
 * (see the plan): canvas + d3-zoom + d3-quadtree, built somewhere nothing can regress,
 * so the pieces are proven before they go under the map itself.
 *
 * WHAT THE PICTURE CLAIMS. Distance here is the model's 128-d cosine, not the PaCMAP
 * projection — so two cards touching on this graph really are alike to the model, which
 * is a stronger claim than the world map makes. But it is still a 2-D embedding of a
 * high-dimensional space: the force layout satisfies link lengths approximately and
 * nothing more. Read adjacency, not absolute position, and never read the axes — there
 * are none, deliberately.
 *
 * Contract: window.Force — enter(indices, label) · exit() · isActive() · seedFrom()
 */
(function () {
  'use strict';

  // Node cap. The simulation and the canvas both scale much further than this; what does
  // not scale is *reading* the thing. Past a few hundred nodes the graph is a hairball
  // and the walk stops being legible. The cap is announced, never silent.
  const MAX_NODES = 500;
  const LINKS_PER_NODE = 3;      // k-nearest within the graph, so structure is visible
  const BRANCH_K = 6;            // neighbours pulled in when you branch from a card
  const TRAIL_MAX = 40;

  // Feel. These are the numbers that decide whether the graph has weight or just twitches.
  // velocityDecay is friction: d3's default 0.4 settles fast and dead, 0.22 keeps inertia
  // so a flung node swings. alphaDecay lower than default keeps it alive long enough to
  // watch it arrange itself.
  const PHYSICS = { velocityDecay: 0.22, alphaDecay: 0.015, charge: -110, linkScale: 190 };

  let canvas = null, ctx = null, dpr = 1;
  let sim = null, nodes = [], links = [], byIdx = new Map();
  let transform = null;          // d3.zoomIdentity once d3 is loaded
  let zoomBehaviour = null;      // the ONE instance — programmatic transforms must
                                 // go through it or its internal state desyncs and
                                 // the next wheel event snaps the view back
  let emb = null, dim = 0;
  let active = false;
  let hovered = null, pinned = null;
  let trail = [];
  let label = '';
  let truncatedFrom = 0;

  function isActive() { return active; }

  // ── Geometry in the model's own space ───────────────────────────────────

  // Rows are L2-normalised at export, so a dot product IS the cosine. Chord distance
  // sqrt(2-2cos) is the true Euclidean distance on the unit sphere — a real metric, which
  // is what makes a link length mean something.
  function chord(rowA, rowB) {
    const oa = rowA * dim, ob = rowB * dim;
    let dot = 0;
    for (let i = 0; i < dim; i++) dot += emb[oa + i] * emb[ob + i];
    if (dot > 1) dot = 1; else if (dot < -1) dot = -1;
    return Math.sqrt(2 - 2 * dot);
  }

  // k nearest to `row` across the whole corpus. One linear pass over 34,322 x 128 floats
  // — about 10 ms, which is why branching can be a click and not a spinner.
  function nearestInCorpus(row, k, exclude) {
    const n = MM.allData.length;
    const best = [];
    const o = row * dim;
    for (let j = 0; j < n; j++) {
      if (j === row || exclude.has(j)) continue;
      const oj = j * dim;
      let dot = 0;
      for (let i = 0; i < dim; i++) dot += emb[o + i] * emb[oj + i];
      if (best.length < k) {
        best.push({ j: j, dot: dot });
        if (best.length === k) best.sort((a, b) => a.dot - b.dot);
      } else if (dot > best[0].dot) {
        best[0] = { j: j, dot: dot };
        best.sort((a, b) => a.dot - b.dot);
      }
    }
    return best.sort((a, b) => b.dot - a.dot).map(b => b.j);
  }

  // ── Graph construction ──────────────────────────────────────────────────

  function makeNode(row, seed) {
    const d = MM.allData[row];
    return {
      row: row,
      name: d.n,
      color: MM.categoryColor(d),
      r: seed ? 6 : 4.5,
      seed: !!seed,
      // Start at the world-map position so the graph unfolds out of the map rather than
      // appearing from nowhere — plus jitter, which is not cosmetic. d3 only assigns
      // initial positions to nodes that lack x/y, so seeding them all by hand means
      // seeding a degenerate cluster with every node coincident. Some regions really are
      // that degenerate: the White Sorceries filament is 187 cards spanning 0.1 x 0.0 on
      // the world map, and without jitter the whole graph collapsed to a single point.
      x: d.x * 8 + (Math.random() - 0.5) * 60,
      y: d.y * 8 + (Math.random() - 0.5) * 60,
    };
  }

  // k-nearest links *within* the current node set. All-pairs is O(n^2 d): at the 500-node
  // cap that is 32M multiply-adds, ~100 ms, paid once per graph build.
  function linkWithin(nodeList) {
    const out = [];
    const seen = new Set();
    for (let a = 0; a < nodeList.length; a++) {
      const cand = [];
      for (let b = 0; b < nodeList.length; b++) {
        if (a === b) continue;
        cand.push({ b: b, d: chord(nodeList[a].row, nodeList[b].row) });
      }
      cand.sort((p, q) => p.d - q.d);
      for (let k = 0; k < Math.min(LINKS_PER_NODE, cand.length); k++) {
        const b = cand[k].b;
        const key = a < b ? a + ':' + b : b + ':' + a;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ source: nodeList[a], target: nodeList[b], d: cand[k].d });
      }
    }
    return out;
  }

  // ── Canvas ──────────────────────────────────────────────────────────────

  function ensureCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.id = 'forceCanvas';
    canvas.className = 'force-canvas';
    document.getElementById('plot').appendChild(canvas);
    ctx = canvas.getContext('2d');
    transform = d3.zoomIdentity;

    zoomBehaviour = d3.zoom().scaleExtent([0.02, 12]).on('zoom', function (ev) {
      transform = ev.transform;
      draw();
    });

    // drag before zoom, and `subject` returns undefined on empty space so the gesture
    // falls through to the zoom behaviour. This is the standard way to let one canvas
    // both drag nodes and pan.
    const drag = d3.drag()
      .subject(function (ev) { return pick(ev.x, ev.y); })
      .on('start', function (ev) {
        if (!ev.active) sim.alphaTarget(0.25).restart();
        ev.subject.fx = ev.subject.x;
        ev.subject.fy = ev.subject.y;
      })
      .on('drag', function (ev) {
        const p = transform.invert([ev.x, ev.y]);
        ev.subject.fx = p[0];
        ev.subject.fy = p[1];
      })
      .on('end', function (ev) {
        if (!ev.active) sim.alphaTarget(0);
        // Release rather than pin: the node keeps the velocity the simulation gave it,
        // so a flung card carries momentum and the graph swings after it.
        ev.subject.fx = null;
        ev.subject.fy = null;
      });

    d3.select(canvas).call(drag).call(zoomBehaviour);

    canvas.addEventListener('mousemove', function (e) {
      const r = canvas.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (hit !== hovered) { hovered = hit || null; draw(); renderPanel(); }
      canvas.style.cursor = hit ? 'pointer' : 'grab';
    });

    canvas.addEventListener('click', function (e) {
      const r = canvas.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (hit) branchFrom(hit);
    });

    canvas.addEventListener('mouseleave', function () {
      if (hovered) { hovered = null; draw(); renderPanel(); }
    });
  }

  function resize() {
    if (!canvas) return;
    const host = document.getElementById('plot');
    const w = host.clientWidth, h = host.clientHeight;
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // crisp on retina; without this it is soft
    draw();
  }

  // Frame the whole graph. Called after the layout settles and from the Fit button —
  // the extent is emergent, so it can only be measured, never predicted.
  function bbox() {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }
    return { minX, maxX, minY, maxY, w: maxX - minX, h: maxY - minY };
  }

  function fitToGraph(animate) {
    if (!canvas || !nodes.length) return;
    const b = bbox();
    if (!isFinite(b.w) || !isFinite(b.h)) return;
    const minX = b.minX, maxX = b.maxX, minY = b.minY, maxY = b.maxY;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const gw = Math.max(maxX - minX, 1), gh = Math.max(maxY - minY, 1);
    // Cap the zoom-in: a 59-unit-wide graph blown up to fill a 1439px canvas is not
    // more readable, just bigger.
    const k = Math.min(w / (gw * 1.18), h / (gh * 1.18), 1.6);
    const t2 = d3.zoomIdentity
      .translate(w / 2 - k * (minX + maxX) / 2, h / 2 - k * (minY + maxY) / 2)
      .scale(k);
    const sel = d3.select(canvas);
    if (animate === false) sel.call(zoomBehaviour.transform, t2);
    else sel.transition().duration(450).call(zoomBehaviour.transform, t2);
  }

  // Screen pixel -> node, through the zoom transform. A linear scan is honest at 500
  // nodes (0.02 ms); the quadtree earns its keep at 34,322, which is where it goes next.
  function pick(px, py) {
    if (!nodes.length) return null;
    const p = transform.invert([px, py]);
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      const dx = n.x - p[0], dy = n.y - p[1];
      const d2 = dx * dx + dy * dy;
      const hit = (n.r + 7) / transform.k;
      const rr = hit * hit;
      if (d2 < rr && d2 < bestD) { bestD = d2; best = n; }
    }
    return best;
  }

  // ── Drawing ─────────────────────────────────────────────────────────────

  function draw() {
    if (!ctx || !canvas) return;
    const w = canvas.width / dpr, h = canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    // Links, faint, brighter the closer the pair. A short bright edge is the model
    // saying "these two are nearly the same card".
    ctx.lineWidth = 1 / transform.k;
    for (const l of links) {
      const closeness = Math.max(0, 1 - l.d / 1.4);
      ctx.strokeStyle = 'rgba(122,138,196,' + (0.08 + closeness * 0.42).toFixed(3) + ')';
      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);
      ctx.stroke();
    }

    // The trail — where the walk has been.
    if (trail.length > 1) {
      ctx.strokeStyle = 'rgba(196,167,71,0.55)';
      ctx.lineWidth = 2 / transform.k;
      ctx.beginPath();
      trail.forEach(function (n, i) { i ? ctx.lineTo(n.x, n.y) : ctx.moveTo(n.x, n.y); });
      ctx.stroke();
    }

    // Radius divided by the zoom scale, so a card is the same size on screen however far
    // in you are. Without this, zooming to fit a tight cluster turns every node into a
    // dinner plate — which is exactly what the White Sorceries filament did.
    const onTrail = new Set(trail);
    for (const n of nodes) {
      const r = n.r / transform.k;
      ctx.beginPath();
      ctx.moveTo(n.x + r, n.y);
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.closePath();
      ctx.fillStyle = n.color;
      ctx.fill();
      // Ring only what is meaningful right now. Ringing every seed was noise: on a fresh
      // walk every node is a seed, so the ring said nothing.
      if (n === hovered || n === pinned || onTrail.has(n)) {
        ctx.lineWidth = (n === hovered ? 2.5 : 1.6) / transform.k;
        ctx.strokeStyle = n === hovered ? '#fff' : '#c4a747';
        ctx.stroke();
      }
    }
    ctx.restore();

    // Labels in screen space so they stay legible at any zoom — the thing Plotly's
    // annotations could never do without a relayout.
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    const named = hovered ? [hovered] : (pinned ? [pinned] : []);
    for (const n of named) {
      const p = transform.apply([n.x, n.y]);
      if (p[0] < -40 || p[0] > w + 40 || p[1] < -20 || p[1] > h + 20) continue;
      const text = n.name;
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = 'rgba(22,33,62,0.88)';
      ctx.fillRect(p[0] - tw / 2 - 5, p[1] - 26, tw + 10, 16);
      ctx.fillStyle = n === hovered ? '#fff' : '#c4a747';
      ctx.fillText(text, p[0], p[1] - 14);
    }
  }

  // ── Simulation ──────────────────────────────────────────────────────────

  function restart(reheat) {
    if (!sim) {
      sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).distance(l => l.d * PHYSICS.linkScale).strength(0.55))
        .force('charge', d3.forceManyBody().strength(PHYSICS.charge).distanceMax(900))
        .force('collide', d3.forceCollide().radius(n => n.r + 3).strength(0.85))
        .force('center', d3.forceCenter(0, 0).strength(0.04))
        .velocityDecay(PHYSICS.velocityDecay)
        .alphaDecay(PHYSICS.alphaDecay)
        .on('tick', draw)
        // Fit when the layout stops, not on a timer. alphaDecay 0.015 gives an ~8 s
        // settle, so an early fit frames a graph that then grows out of the viewport —
        // which is exactly what left the canvas looking empty the first time.
        .on('end', function () { fitToGraph(true); });
    } else {
      sim.nodes(nodes);
      sim.force('link').links(links).distance(l => l.d * PHYSICS.linkScale);
      sim.force('charge').strength(PHYSICS.charge);
    }
    sim.alpha(reheat == null ? 1 : reheat).restart();
  }

  // ── Entering, branching, leaving ────────────────────────────────────────

  async function enter(rowIndices, seedLabel) {
    // Re-entering with no explicit seed and a graph already built: pick up where you
    // left off rather than starting over.
    if (!rowIndices && nodes.length) {
      active = true;
      ensureCanvas();
      if (canvas) canvas.style.display = '';
      resize();
      restart(0.15);          // a nudge, so it is visibly alive without rearranging
      renderPanel();
      MM.setStatus(nodes.length + ' cards · picked up where you left off');
      return;
    }

    const src = rowIndices && rowIndices.length ? rowIndices : seedFrom();
    const unique = Array.from(new Set(src)).filter(i => MM.allData[i]);
    if (unique.length < 2) {
      // Nothing selected. An error message here would be a dead end — offer somewhere to
      // go instead. A walk has to start from a set, so hand over the sets that exist.
      active = true;
      ensureCanvas();
      if (canvas) canvas.style.display = '';
      resize();
      nodes = []; links = []; trail = [];
      draw();
      await renderEmptyState();
      MM.setStatus('Pick a starting point for the walk.');
      return;
    }

    if (!emb) {
      MM.setStatus('Loading embeddings…');
      emb = await MM.getEmbeddings();
      if (!emb) { MM.setStatus('Embeddings unavailable — the walk needs them.'); return; }
      dim = MM.EMBED_DIM;
    }

    truncatedFrom = unique.length > MAX_NODES ? unique.length : 0;
    const rows = truncatedFrom ? unique.slice(0, MAX_NODES) : unique;
    label = seedLabel || 'Selection';

    nodes = rows.map(r => makeNode(r, true));
    byIdx = new Map(nodes.map(n => [n.row, n]));
    links = linkWithin(nodes);
    trail = [];
    pinned = null;
    hovered = null;

    active = true;
    ensureCanvas();
    if (canvas) canvas.style.display = '';
    resize();
    restart(1);
    // Keep it framed while it settles — cheap, and the graph is legible the whole way
    // rather than drifting off-screen for the first few seconds. `sim.on('end')` does
    // the final, authoritative fit.
    let fits = 0;
    const keepFramed = setInterval(function () {
      if (!active || ++fits > 14) { clearInterval(keepFramed); return; }
      fitToGraph(false);
    }, 550);
    renderPanel();
    MM.setStatus(nodes.length + ' cards · link length is 128-dim cosine distance · click a card to branch');
  }

  // The walk. Pull in the clicked card's nearest neighbours from the whole corpus, link
  // them, and reheat — the graph grows toward whatever you were curious about.
  function branchFrom(node) {
    if (!emb) return;
    pinned = node;
    if (trail[trail.length - 1] !== node) {
      trail.push(node);
      if (trail.length > TRAIL_MAX) trail.shift();
    }

    if (nodes.length >= MAX_NODES) {
      MM.setStatus('At the ' + MAX_NODES + '-card cap — trim the walk or start a new one.');
      renderPanel();
      draw();
      return;
    }

    const have = new Set(nodes.map(n => n.row));
    const room = Math.min(BRANCH_K, MAX_NODES - nodes.length);
    const found = nearestInCorpus(node.row, room, have);
    if (!found.length) { renderPanel(); return; }

    for (const row of found) {
      const n = makeNode(row, false);
      // Born beside their parent, not at their world position — a new node should appear
      // to come *out of* the card you clicked.
      n.x = node.x + (Math.random() - 0.5) * 40;
      n.y = node.y + (Math.random() - 0.5) * 40;
      nodes.push(n);
      byIdx.set(row, n);
      links.push({ source: node, target: n, d: chord(node.row, row) });
    }

    restart(0.6);
    renderPanel();
    MM.setStatus('Branched from ' + node.name + ' — ' + nodes.length + ' cards on the graph');
  }

  // What a walk starts from, in priority order. Everything that already knows how to
  // produce a set of cards feeds this: browse, the 8-card stack, the drill offer.
  function seedFrom() {
    if (MM.browseSet && MM.browseSet.indices.length) return MM.browseSet.indices.slice(0, MAX_NODES);
    const sel = MM.selectedRows();
    if (sel.length) return sel;
    return [];
  }

  function reheat() { if (sim) sim.alpha(0.8).restart(); }

  // Halt the layout without leaving the mode. The physics sliders reheat by design,
  // so without this there is no way to hold the graph still and look at it.
  function freeze() { if (sim) { sim.alpha(0); sim.stop(); } draw(); }

  function clearTrail() { trail = []; draw(); renderPanel(); }

  // Leaving keeps the graph. Rebuilding a walk you spent minutes growing, just because
  // you looked at the map, is the wrong trade — `enter()` restores it.
  //
  // Note what this does NOT do: touch canvas.style.display. `#plot:not(.force-mode)`
  // already hides it, and an inline `display:none` set here survived re-entry, so the
  // graph rebuilt correctly into a 0x0 hidden canvas and the mode looked dead.
  function exit() {
    active = false;
    if (sim) sim.stop();
  }

  function focusCard(row) {
    const n = byIdx.get(row);
    if (n) branchFrom(n);
  }

  // ── Panel ───────────────────────────────────────────────────────────────

  // Somewhere to start. Decks come from the tracked manifest; regions from the HDBSCAN
  // membership the clustering now keeps. Both are one click.
  async function renderEmptyState() {
    const el = document.getElementById('deckInner');
    if (!el) return;
    document.getElementById('deckPanel').classList.add('open');

    let decks = [];
    try {
      const doc = await (await fetch('../data/decks/index.json')).json();
      decks = doc.decks || [];
    } catch (e) { /* the walk works without them */ }

    let regions = [];
    try {
      const rd = await MM.getRegionData();
      if (rd && rd.membership) {
        regions = rd.regions
          .filter(r => r.level === 1 && r.count >= 60 && r.count <= MAX_NODES)
          .sort((a, b) => b.count - a.count)
          .slice(0, 8);
      }
    } catch (e) { /* likewise */ }

    el.innerHTML =
      '<div class="deck-header"><h2>The Walk</h2>' +
      '<button class="detail-close" onclick="Force.close()" title="Close">×</button></div>' +
      '<div class="deck-section"><div class="deck-empty">' +
        'A walk starts from a set of cards. Pick one below — or box-select on the map, ' +
        'or use Find Similar, then come back.' +
      '</div></div>' +
      (decks.length ? '<div class="deck-section">' +
        '<div class="deck-section-title">Walk a deck</div>' +
        decks.map(d => '<div class="lens-cand" onclick="Force.walkDeck(' +
            JSON.stringify(d.slug).replace(/"/g, '&quot;') + ',' +
            JSON.stringify(d.deck_name).replace(/"/g, '&quot;') + ')">' +
          '<span class="lens-cand-name">' + MM.escHtml(d.deck_name) + '</span>' +
          '<span class="lens-chip">Vol. ' + String(d.volume).padStart(3, '0') + '</span>' +
        '</div>').join('') + '</div>' : '') +
      (regions.length ? '<div class="deck-section">' +
        '<div class="deck-section-title">Walk a region</div>' +
        regions.map(r => '<div class="lens-cand" onclick="Force.walkRegion(' +
            JSON.stringify(r.id).replace(/"/g, '&quot;') + ')">' +
          '<span class="lens-cand-name">' + MM.escHtml(r.short || r.label) + '</span>' +
          '<span class="lens-chip">' + r.count + '</span>' +
        '</div>').join('') + '</div>' : '');
  }

  async function walkDeck(slug, name) {
    try {
      const doc = await (await fetch('../data/decks/' + slug + '/cards.json')).json();
      const names = new Set(doc.cards.filter(c => !c.is_sideboard).map(c => c.name));
      const rows = [];
      MM.allData.forEach((d, i) => { if (names.has(d.n)) rows.push(i); });
      enter(rows, name || slug);
    } catch (e) { MM.setStatus('Could not load ' + slug); }
  }

  async function walkRegion(regionId) {
    const rd = await MM.getRegionData();
    if (!rd || !rd.membership) return;
    const m = /^l(\d)_(\d+)$/.exec(regionId);
    if (!m) return;
    const labels = rd.membership['l' + m[1]];
    const cid = parseInt(m[2], 10);
    const rows = [];
    for (let i = 0; i < labels.length; i++) if (labels[i] === cid) rows.push(i);
    const region = rd.regions.find(r => r.id === regionId);
    enter(rows, region ? region.label : regionId);
  }

  function renderPanel() {
    const el = document.getElementById('deckInner');
    if (!el || !active) return;
    // A "0 CARDS / 0 LINKS" panel is a dead end. Whoever calls this with an empty graph
    // wants the empty state, not an empty scoreboard — routing it here rather than at
    // each call site means a new caller cannot reintroduce the dead end.
    if (!nodes.length) { renderEmptyState(); return; }
    document.getElementById('deckPanel').classList.add('open');

    const h = hovered || pinned;
    let html =
      '<div class="deck-header"><h2>The Walk</h2>' +
      '<button class="detail-close" onclick="Force.close()" title="Close">×</button></div>' +
      '<div class="deck-section">' +
        '<div class="lens-title">' + MM.escHtml(label) + '</div>' +
        '<div class="lens-stats">' +
          '<div class="lens-stat"><div class="lens-stat-n">' + nodes.length + '</div><div class="lens-stat-l">cards</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + links.length + '</div><div class="lens-stat-l">links</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + trail.length + '</div><div class="lens-stat-l">visited</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + MAX_NODES + '</div><div class="lens-stat-l">cap</div></div>' +
        '</div>' +
        (truncatedFrom ? '<div class="lens-note">seeded with ' + MAX_NODES + ' of ' + truncatedFrom + '</div>' : '') +
        '<div class="lens-note">Drag a card to fling it · click to branch · scroll to zoom</div>' +
      '</div>';

    if (h) {
      html += '<div class="deck-section"><div class="deck-section-title">Under the cursor</div>' +
        '<div class="lens-title">' + MM.escHtml(h.name) + '</div>' +
        '<div class="lens-sub">' + MM.escHtml(MM.allData[h.row].t || '') + '</div>' +
        '<button class="lens-btn" onclick="MM.selectByName(' +
          JSON.stringify(h.name).replace(/"/g, '&quot;') + ')">Open the card →</button>' +
        '</div>';
    }

    html +=
      '<div class="deck-section"><div class="deck-section-title">Physics</div>' +
        slider('linkScale', 'link length', 60, 400, PHYSICS.linkScale) +
        slider('charge', 'repulsion', -400, -20, PHYSICS.charge) +
        slider('velocityDecay', 'friction', 5, 90, Math.round(PHYSICS.velocityDecay * 100)) +
        '<button class="lens-btn" onclick="Force.fit()">Fit to the graph</button>' +
        '<button class="lens-btn" onclick="Force.reheat()">Reheat</button>' +
      '</div>';

    if (trail.length) {
      html += '<div class="deck-section"><div class="deck-section-title">The walk ' +
        '<span>' + trail.length + '</span></div>' +
        trail.slice().reverse().map(n =>
          '<div class="lens-cand"><span class="lens-cand-name">' + MM.escHtml(n.name) + '</span></div>'
        ).join('') +
        '<button class="lens-btn" onclick="Force.clearTrail()">Clear the trail</button></div>';
    }

    el.innerHTML = html;
  }

  function slider(key, name, min, max, val) {
    return '<div class="force-slider"><label>' + name + '<span>' + val + '</span></label>' +
      '<input type="range" min="' + min + '" max="' + max + '" value="' + val +
      '" oninput="Force.tune(\'' + key + '\', this.value, this)"></div>';
  }

  function tune(key, value, el) {
    const v = parseFloat(value);
    if (el && el.previousElementSibling) el.previousElementSibling.querySelector('span').textContent = v;
    if (key === 'velocityDecay') { PHYSICS.velocityDecay = v / 100; sim.velocityDecay(PHYSICS.velocityDecay); }
    else if (key === 'charge') { PHYSICS.charge = v; sim.force('charge').strength(v); }
    else if (key === 'linkScale') { PHYSICS.linkScale = v; sim.force('link').distance(l => l.d * v); }
    sim.alpha(Math.max(sim.alpha(), 0.35)).restart();
  }

  function close() {
    document.getElementById('modeSelect').value = 'explore';
    MM.setMode('explore');
  }

  window.addEventListener('resize', function () { if (active) resize(); });

  window.Force = {
    enter, exit, isActive, seedFrom, focusCard,
    reheat, freeze, clearTrail, tune, close, renderPanel, bbox,
    walkDeck, walkRegion,
    fit: function () { fitToGraph(true); },
    get nodeCount() { return nodes.length; },
    get linkCount() { return links.length; },
    get trailLength() { return trail.length; },
    // Exposed for the browser tests: a link's `d` must be the 128-d chord distance, not
    // anything derived from screen position. Chord distance on a unit sphere is bounded
    // by [0, 2], which is the assertion.
    linkStats() {
      if (!links.length) return null;
      let min = Infinity, max = -Infinity, sum = 0;
      for (const l of links) { if (l.d < min) min = l.d; if (l.d > max) max = l.d; sum += l.d; }
      return { min: min, max: max, mean: sum / links.length, n: links.length };
    },
    MAX_NODES,
  };
})();
