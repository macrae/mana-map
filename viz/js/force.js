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
  // Persistent name labels: how many at most, and how much clear space each needs. 14 is
  // about what a 1440px canvas holds before names start reading as texture rather than
  // words; the gap keeps neighbours from touching.
  // Synchronous ticks run before the first paint so the graph arrives settled. Capped
  // because this is main-thread work: 500 nodes x 400 ticks is the worst case and still
  // well under a frame budget's worth of stall at these sizes.
  const SETTLE_TICKS = 400;
  const LABEL_MAX = 14;
  const LABEL_GAP = 6;
  let lastLabelCount = 0;   // exposed for the browser tests; canvas text is unassertable
  let lastEdgeLabelCount = 0;
  // Reason labels are only legible while the graph is small. Past this the edges are
  // too short and too many, and naming them turns the canvas into texture.
  const EDGE_LABEL_MAX_NODES = 60;
  const EDGE_LABEL_MAX = 8;

  // Feel. These are the numbers that decide whether the graph has weight or just twitches.
  // velocityDecay is friction: d3's default 0.4 settles fast and dead, 0.22 keeps inertia
  // so a flung node swings.
  //
  // alphaDecay was 0.015 — an ~8 second settle — so that the initial layout could be
  // watched arranging itself. It no longer animates into place: `enter` pre-settles it
  // synchronously before the first paint. All that remained was a graph that kept
  // drifting under the cursor for eight seconds after every branch, which reads as
  // clunky rather than alive. 0.08 settles in a bit over a second: new cards still fly
  // out and find their place, then stop. Dragging is unaffected — it holds alphaTarget
  // above zero for as long as you hold the node.
  const PHYSICS = { velocityDecay: 0.22, alphaDecay: 0.08, charge: -110, linkScale: 190 };

  let chrome = 'walk';
  // Which rows came from a loaded deck, and which one is its commander. Nodes pulled in
  // by branching are deliberately NOT in here — that difference is the whole point of
  // loading a deck: you can see at a glance what you brought and what you found.
  let deckRows = null;
  let commanderRow = -1;
  // Set by any real pan/zoom/drag. While false the graph may frame itself; once true it
  // never moves the camera again without being asked.
  let userAdjusted = false;
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

  // k-nearest now comes from MM.nearestTo — this file had its own copy, mana-map.js had
  // another inside findSimilarCards, and the neighbourhood walk would have been a third.
  // `respectFilters: false` on purpose: a graph you are branching through should not
  // change shape because someone toggled Lands off in the toolbar.
  async function nearestInCorpus(row, k, exclude) {
    const hits = await MM.nearestTo(row, k, { exclude: exclude, respectFilters: false });
    return hits.map(h => h.i);
  }

  // ── Graph construction ──────────────────────────────────────────────────

  function makeNode(row, seed) {
    // cardRecord, not allData: discovery boots on viz_index (0.56 MB) and the 2.9 MB
    // projection lands behind it. Reading allData directly meant a node could not be
    // built until the big fetch finished — and `nearestTo` failed the same way, but
    // *silently*, returning [] so the first click simply did nothing.
    const d = MM.cardRecord(row);
    // World coordinates only exist once the projection has arrived. Without them the
    // node starts at the origin and the jitter below does the work, which is fine —
    // the seeded position is an aesthetic (the graph unfolding out of the map), not
    // something the layout needs.
    const wx = typeof d.x === 'number' ? d.x * 8 : 0;
    const wy = typeof d.y === 'number' ? d.y * 8 : 0;
    const inDeck = !!(deckRows && deckRows.has(row));
    const isCommander = row === commanderRow;
    return {
      row: row,
      name: d.n,
      color: MM.categoryColor(d),
      r: isCommander ? 9 : (inDeck ? 6 : 4.5),
      seed: !!seed,
      deck: inDeck,
      commander: isCommander,
      // Start at the world-map position so the graph unfolds out of the map rather than
      // appearing from nowhere — plus jitter, which is not cosmetic. d3 only assigns
      // initial positions to nodes that lack x/y, so seeding them all by hand means
      // seeding a degenerate cluster with every node coincident. Some regions really are
      // that degenerate: the White Sorceries filament is 187 cards spanning 0.1 x 0.0 on
      // the world map, and without jitter the whole graph collapsed to a single point.
      x: wx + (Math.random() - 0.5) * 60,
      y: wy + (Math.random() - 0.5) * 60,
    };
  }

  // Links *within* the current node set. Two paths, and the cheap one is the default:
  // the precomputed table already knows each card's 12 nearest, so intersecting that with
  // the seed set is a lookup rather than an O(n^2 d) scan. All-pairs at the 500-node cap
  // was 32M multiply-adds and needed the whole embedding matrix resident first.
  function linkWithin(nodeList) {
    if (!emb) return linkWithinFromTable(nodeList);
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
      // `sourceEvent` is null for programmatic transforms and set for real gestures. Once
      // you have touched the camera it is yours: auto-fit stops competing with you. This
      // is the whole of "zooming while the graph moves zooms back out" — a settle-time
      // fit was overwriting the transform mid-gesture.
      if (ev.sourceEvent) userAdjusted = true;
      draw();
    });

    // drag before zoom, and `subject` returns undefined on empty space so the gesture
    // falls through to the zoom behaviour. This is the standard way to let one canvas
    // both drag nodes and pan.
    const drag = d3.drag()
      .subject(function (ev) { return pick(ev.x, ev.y); })
      .on('start', function (ev) {
        userAdjusted = true;
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

    // clickDistance is the whole reason clicking a card felt unreliable.
    //
    // d3-drag defaults it to 0, which means ANY pointer movement between mousedown and
    // mouseup — one pixel of hand tremor — makes d3 install a capture-phase suppressor
    // that swallows the subsequent `click` event outright. The handler below never runs,
    // nothing expands, and clicking again "fixes" it only because that click happened to
    // be steadier. Verified with real mouse input: two clicks on a node delivered zero
    // click events and added zero nodes.
    //
    // 6px is a deliberate tap tolerance: below it you meant to click, above it you meant
    // to fling the card.
    d3.select(canvas).call(drag.clickDistance(6)).call(zoomBehaviour);

    canvas.addEventListener('mousemove', function (e) {
      const r = canvas.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top);
      if (hit !== hovered) { hovered = hit || null; draw(); renderPanel(); }
      canvas.style.cursor = hit ? 'pointer' : 'grab';
      if (hit) MM.showCardPopup(hit.row, e.clientX, e.clientY);
      else MM.hideCardPopup();
    });

    // Leaving the canvas fires no mousemove, so without this the last hovered card keeps
    // its ring and its popup stays open over the graph with no way to dismiss it.
    canvas.addEventListener('mouseleave', function () {
      if (hovered) { hovered = null; draw(); renderPanel(); }
      MM.hideCardPopup();
    });

    canvas.addEventListener('click', function (e) {
      const r = canvas.getBoundingClientRect();
      // Fall back to whatever is hovered: the simulation is still running after a branch,
      // so a node can drift out from under the cursor between press and release. The
      // highlighted card is the one the user was aiming at.
      const hit = pick(e.clientX - r.left, e.clientY - r.top) || hovered;
      if (hit) branchFrom(hit);
    });

    canvas.addEventListener('mouseleave', function () {
      MM.hideCardPopup();
      if (hovered) { hovered = null; draw(); renderPanel(); }
    });

    // Track the plot's box. `inset: 0` would size the canvas to its parent, but resize()
    // sets an explicit inline width/height for devicePixelRatio, which then does NOT
    // follow the parent. enter() resizes before the side panel opens, so the canvas kept
    // the full-width size and overhung the panel by ~420px at z-index 10 — swallowing
    // every click on the deck menu underneath it. The panel looked fine and was inert.
    if (window.ResizeObserver) {
      let pending = null;
      new ResizeObserver(function () {
        clearTimeout(pending);
        pending = setTimeout(function () { if (active) resize(); }, 60);
      }).observe(document.getElementById('plot'));
    }
  }

  // Only the backing store is set here. The element's SIZE comes from `inset: 0` in CSS
  // and nothing else.
  //
  // Setting canvas.style.width/height instead decoupled the element from its parent: the
  // side panel opens after enter() has already resized, so the canvas kept the full-width
  // size, overhung the 420px panel at z-index 10, and silently swallowed every click on
  // the deck menu underneath. The panel looked perfect and was completely inert. A
  // ResizeObserver is not a fix for that — RO callbacks are throttled in background tabs
  // exactly like rAF and CSS transitions, so the overhang can outlive them. CSS geometry
  // cannot get out of sync; a JS mirror of it always can.
  function resize() {
    if (!canvas) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
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

  function fitToGraph(animate, auto) {
    if (!canvas || !nodes.length) return;
    // An automatic fit is a suggestion; a user gesture is an instruction.
    if (auto && userAdjusted) return;
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
    //
    // Deck edges — both ends from a loaded decklist — are drawn warm and heavier, so the
    // deck reads as a structure you can see through the exploration rather than dissolving
    // into it the moment you branch. Everything cool and thin is something you found.
    for (const l of links) {
      const inDeck = l.source.deck && l.target.deck;
      const closeness = Math.max(0, 1 - l.d / 1.4);
      ctx.lineWidth = (inDeck ? 1.7 : 1) / transform.k;
      // Three relations, three inks: deck structure warm gold, synergy violet,
      // similarity the default cool blue. Colour carries the relation so the reason
      // labels below only have to carry the detail.
      if (inDeck) {
        ctx.strokeStyle = 'rgba(196,167,71,' + (0.22 + closeness * 0.45).toFixed(3) + ')';
      } else if (l.rel === 'synergy') {
        ctx.strokeStyle = 'rgba(168,120,214,0.55)';
      } else if (l.rel === 'obsolete') {
        ctx.strokeStyle = 'rgba(214,120,120,0.5)';
      } else {
        ctx.strokeStyle = 'rgba(122,138,196,' + (0.08 + closeness * 0.42).toFixed(3) + ')';
      }
      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);
      ctx.stroke();
    }
    ctx.lineWidth = 1 / transform.k;

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
      // Cards you brought keep their full colour; cards you found are washed out, so a
      // loaded deck stays legible as you explore outward from it.
      ctx.globalAlpha = (deckRows && !n.deck) ? 0.5 : 1;
      ctx.fillStyle = n.color;
      ctx.fill();
      ctx.globalAlpha = 1;
      // The commander gets a permanent gold ring — it is the one card the deck is built
      // around, and it should be findable without hunting.
      if (n.commander) {
        ctx.lineWidth = 2.5 / transform.k;
        ctx.strokeStyle = '#c4a747';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(n.x, n.y, (n.r + 4) / transform.k, 0, Math.PI * 2);
        ctx.lineWidth = 1.2 / transform.k;
        ctx.strokeStyle = 'rgba(196,167,71,0.55)';
        ctx.stroke();
      } else if (n.deck) {
        ctx.lineWidth = 1.2 / transform.k;
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.stroke();
      }
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
    //
    // A representative sample, not every card. Labelling all 500 is an unreadable smear
    // and labelling only the hovered one means the graph tells you nothing until you
    // touch it. So: the cards you are interacting with always get a name, then as many
    // others as physically fit, chosen greedily and rejected on collision — which is why
    // the set thins out when the graph is dense and fills in as you zoom in, without any
    // zoom logic of its own.
    ctx.font = '12px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';

    const priority = [];
    if (hovered) priority.push(hovered);
    if (pinned && pinned !== hovered) priority.push(pinned);
    for (let i = trail.length - 1; i >= 0; i--) {
      if (priority.indexOf(trail[i]) === -1) priority.push(trail[i]);
    }
    // Commander, then the rest of the deck, then seeds — on a loaded deck those are the
    // cards worth naming before anything you wandered into.
    for (const n of nodes) if (n.commander && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (n.deck && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (n.seed && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (priority.indexOf(n) === -1) priority.push(n);

    const placed = [];
    let drawn = 0;
    lastLabelCount = 0;

    // Synergy edges say WHY. Placed first and into the same collision set as the node
    // labels, so a reason can never sit on top of a card name — and dropped entirely
    // once the graph is dense, because a wall of text is worse than no text.
    lastEdgeLabelCount = 0;
    if (nodes.length <= EDGE_LABEL_MAX_NODES) {
      ctx.font = '10px system-ui, -apple-system, sans-serif';
      for (const l of links) {
        if (lastEdgeLabelCount >= EDGE_LABEL_MAX) break;
        if (!l.reason) continue;
        const a = transform.apply([l.source.x, l.source.y]);
        const b = transform.apply([l.target.x, l.target.y]);
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        if (mx < 0 || mx > w || my < 0 || my > h) continue;
        const tw = ctx.measureText(l.reason).width;
        const box = { x0: mx - tw / 2 - 4, x1: mx + tw / 2 + 4, y0: my - 8, y1: my + 5 };
        let clash = false;
        for (const bb of placed) {
          if (box.x0 < bb.x1 + LABEL_GAP && box.x1 > bb.x0 - LABEL_GAP &&
              box.y0 < bb.y1 + LABEL_GAP && box.y1 > bb.y0 - LABEL_GAP) { clash = true; break; }
        }
        if (clash) continue;
        placed.push(box);
        lastEdgeLabelCount++;
        ctx.fillStyle = 'rgba(22,33,62,0.82)';
        ctx.fillRect(box.x0, box.y0, tw + 8, 13);
        ctx.fillStyle = 'rgba(180,140,220,0.92)';
        ctx.fillText(l.reason, mx, my + 2);
      }
      ctx.font = '12px system-ui, -apple-system, sans-serif';
    }
    for (const n of priority) {
      if (drawn >= LABEL_MAX) break;
      const p = transform.apply([n.x, n.y]);
      if (p[0] < 0 || p[0] > w || p[1] < 0 || p[1] > h) continue;
      const tw = ctx.measureText(n.name).width;
      const box = { x0: p[0] - tw / 2 - 5, x1: p[0] + tw / 2 + 5, y0: p[1] - 26, y1: p[1] - 10 };
      let clash = false;
      for (const b of placed) {
        if (box.x0 < b.x1 + LABEL_GAP && box.x1 > b.x0 - LABEL_GAP &&
            box.y0 < b.y1 + LABEL_GAP && box.y1 > b.y0 - LABEL_GAP) { clash = true; break; }
      }
      // The hovered and pinned cards are never suppressed — you asked about those.
      if (clash && n !== hovered && n !== pinned) continue;
      placed.push(box);
      drawn++;
      lastLabelCount = drawn;

      const focus = n === hovered || n === pinned;
      ctx.fillStyle = focus ? 'rgba(22,33,62,0.92)' : 'rgba(22,33,62,0.72)';
      ctx.fillRect(box.x0, box.y0, tw + 10, 16);
      ctx.fillStyle = n === hovered ? '#fff' : (focus ? '#c4a747' : 'rgba(196,167,71,0.72)');
      ctx.fillText(n.name, p[0], p[1] - 14);
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
        .on('end', function () { fitToGraph(true, true); });
    } else {
      sim.nodes(nodes);
      sim.force('link').links(links).distance(l => l.d * PHYSICS.linkScale);
      sim.force('charge').strength(PHYSICS.charge);
    }
    sim.alpha(reheat == null ? 1 : reheat).restart();
  }

  // ── Entering, branching, leaving ────────────────────────────────────────

  // `opts.chrome === 'discovery'` means someone else owns the side panel. One engine,
  // two chromes — a second force simulation for the landing would be the duplicate-kNN
  // mistake this codebase has already had to undo twice.
  async function enter(rowIndices, seedLabel, opts) {
    chrome = (opts && opts.chrome) || 'walk';
    if (opts && opts.deck) {
      deckRows = opts.deck.rows || null;
      commanderRow = typeof opts.deck.commander === 'number' ? opts.deck.commander : -1;
    } else if (opts && opts.deck === null) {
      deckRows = null; commanderRow = -1;
    }
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
    const unique = Array.from(new Set(src)).filter(i => MM.cardRecord(i));
    // One card is a legitimate starting point — it is THE starting point for discovery.
    // This used to demand two and fall through to a deck/region menu, which is why
    // landing on a single random card was impossible.
    if (unique.length < 1) {
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

    // Embeddings are a gate only where they earn it, which is NOT the landing.
    //
    // A single seed has no intra-graph links to compute, so waiting on 16.8 MB of
    // incompressible float32 before showing one card was pure dead time. A *seeded*
    // walk is different: `linkWithinFromTable` only links cards whose precomputed top-12
    // happen to also be in the set, which for a 97-card deck is 38 links instead of ~290
    // — a visibly sparser, worse graph. The browser suite caught exactly that.
    if (unique.length > 1 && !emb) {
      MM.setStatus('Loading embeddings…');
      emb = await MM.getEmbeddings();
      if (emb) dim = MM.EMBED_DIM;
    }
    // No background prefetch on the landing path, deliberately. Nothing in discovery
    // reads `emb` — branching comes from the table and a single seed has no intra-graph
    // links — so speculatively pulling 16.8 MB would be 16.8 MB spent on nothing. It also
    // showed up as contention: with the fetch in place two browser tests passed alone and
    // failed in the full run.

    truncatedFrom = unique.length > MAX_NODES ? unique.length : 0;
    // Even stride, not the first N — see Drill.sampleEvenly. Seeding a walk with the
    // first 500 rows of a 3,434-card region seeds it with Scryfall's export order.
    const rows = window.Drill ? window.Drill.sampleEvenly(unique, MAX_NODES)
                              : unique.slice(0, MAX_NODES);
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
    userAdjusted = false;      // a brand-new graph is allowed to frame itself, once

    // Pre-settle the layout BEFORE the first paint.
    //
    // `sim.tick()` advances the simulation without dispatching tick events, so nothing
    // draws. The graph therefore *arrives* arranged instead of being watched to arrange
    // itself — which is what made loading a deck look broken: a hundred nodes seeded at
    // scaled world coordinates appeared as a distorted smear, collapsed inward over
    // several seconds, and re-framed fourteen times on the way, with the user unable to
    // touch anything until it stopped.
    //
    // A few hundred synchronous ticks cost a few milliseconds at these sizes and buy the
    // entire settling animation up front.
    restart(1);
    sim.stop();
    const ticks = Math.min(SETTLE_TICKS, 60 + nodes.length * 3);
    for (let i = 0; i < ticks; i++) sim.tick();

    fitToGraph(false, true);   // one fit, on a layout that is already still
    draw();
    // Left stopped on purpose. Dragging reheats it (`alphaTarget` on drag start) and
    // branching reheats it (`restart(0.6)`), so it is alive exactly when something is
    // happening and perfectly steady the rest of the time — which is what "snappy" means
    // here. A graph that keeps drifting under the cursor is not responsive, it is busy.
    sim.stop();

    // A single seed is a landing: pin it so it draws as card art rather than a 6 px dot.
    if (nodes.length === 1) pinned = nodes[0];
    renderPanel();
    if (chrome !== 'discovery') {
      MM.setStatus(nodes.length + ' cards · link length is 128-dim cosine distance · click a card to branch');
    }
  }

  // The walk. Pull in the clicked card's neighbours and reheat — the graph grows toward
  // whatever you were curious about.
  //
  // Synchronous on purpose. This used to `await nearestInCorpus`, which scanned the
  // 34,322 x 128 embedding matrix on the main thread and could not run at all until
  // 16.8 MB of incompressible float32 had downloaded. An await inside a click is what
  // makes a graph feel laggy rather than physical; the precomputed table removes both.
  function branchFrom(node, relation) {
    pinned = node;
    // Tell Discovery which card is now open, so the panel shows THIS card's art, its
    // relation counts, and a Keep button that adds the card you actually clicked.
    if (chrome === 'discovery' && window.Discovery) Discovery.focus(node.row);
    if (trail[trail.length - 1] !== node) {
      trail.push(node);
      if (trail.length > TRAIL_MAX) trail.shift();
    }

    const rel = relation || 'similar';
    const found = Discovery.neighbours(node.row, rel);
    if (!found.length) {
      MM.setStatus(node.name + ' has no ' + rel + ' neighbours — try another relation.');
      renderPanel();
      return;
    }

    // Cross-links FIRST, and this is the fix for a real defect rather than a nicety.
    // Branching used to skip every neighbour already on the graph, and only ever add
    // parent->child edges. From a multi-seed start that was invisible because `enter()`
    // ran `linkWithin` over the seeds; from a SINGLE seed it meant every graph was a
    // pure tree forever — no cycles, no cross-links, and two near-duplicates reached
    // down different branches sitting far apart with nothing between them. Which is the
    // opposite of this file's whole thesis, "read adjacency, not absolute position".
    let added = 0;
    for (const nb of found) {
      const existing = byIdx.get(nb.row);
      if (!existing || existing === node) continue;
      if (hasLink(node, existing)) continue;
      links.push({ source: node, target: existing, d: edgeLength(nb), rel: nb.relation,
                   reason: nb.reason || null });
      added++;
    }

    const room = Math.max(0, MAX_NODES - nodes.length);
    if (!room && !added) {
      MM.setStatus('At the ' + MAX_NODES + '-card cap — trim the walk or start a new one.');
      renderPanel();
      draw();
      return;
    }

    let grown = 0;
    for (const nb of found) {
      if (grown >= Math.min(BRANCH_K, room)) break;
      if (byIdx.has(nb.row)) continue;
      const n = makeNode(nb.row, false);
      // Born beside their parent, not at their world position — a new node should appear
      // to come *out of* the card you clicked.
      n.x = node.x + (Math.random() - 0.5) * 40;
      n.y = node.y + (Math.random() - 0.5) * 40;
      nodes.push(n);
      byIdx.set(nb.row, n);
      links.push({ source: node, target: n, d: edgeLength(nb), rel: nb.relation,
                   reason: nb.reason || null });
      grown++;
    }

    restart(0.6);
    renderPanel();
    const bits = [];
    if (grown) bits.push(grown + ' new');
    if (added) bits.push(added + ' link' + (added === 1 ? '' : 's') + ' to cards already here');
    MM.setStatus('Branched from ' + node.name + ' by ' + rel +
                 (bits.length ? ' — ' + bits.join(', ') : '') +
                 ' · ' + nodes.length + ' cards');
  }

  function hasLink(a, b) {
    for (const l of links) {
      const s = l.source.row !== undefined ? l.source.row : l.source;
      const t = l.target.row !== undefined ? l.target.row : l.target;
      if ((s === a.row && t === b.row) || (s === b.row && t === a.row)) return true;
    }
    return false;
  }

  // Chord distance from the stored cosine. Synergy and obsolescence are rule-based, not
  // metric, so they carry a fixed nominal similarity — a synergy edge is a claim about
  // function, and pretending it has a measured length would be a lie in pixels.
  function edgeLength(nb) {
    const c = Math.max(-1, Math.min(1, nb.sim));
    return Math.sqrt(Math.max(0, 2 - 2 * c));
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

  // Back to the menu. Without this the walk is a one-way door: the deck/region list only
  // appears when the graph is empty, and since re-entry restores the graph, the first set
  // you pick is the only set you can ever pick.
  // Branch a row that is already on the graph — how Discovery hands its landing card
  // over: enter() seeds the single node, this opens it.
  // Table-driven equivalent: whichever of a card's precomputed neighbours are also in
  // this graph become links. Sparser than the all-pairs version by design — it only knows
  // the top 12 — but it needs no embedding matrix and no scan.
  function linkWithinFromTable(nodeList) {
    const pos = new Map();
    nodeList.forEach(function (n, i) { pos.set(n.row, i); });
    const out = [];
    const seen = new Set();
    for (const n of nodeList) {
      let made = 0;
      for (const nb of Discovery.neighbours(n.row, 'similar')) {
        if (made >= LINKS_PER_NODE) break;
        const j = pos.get(nb.row);
        if (j === undefined || nodeList[j] === n) continue;
        const a = pos.get(n.row);
        const key = a < j ? a + ':' + j : j + ':' + a;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ source: n, target: nodeList[j], d: edgeLength(nb) });
        made++;
      }
    }
    return out;
  }

  function hasRow(row) { return byIdx.has(row); }

  function branchByRow(row, relation) {
    const n = byIdx.get(row);
    if (n) branchFrom(n, relation);
  }

  // `quiet` skips the empty-state menu: Discovery clears the graph only to immediately
  // reseed it with a new landing card, and flashing a deck/region picker in between
  // would be a menu nobody asked for.
  function newWalk(quiet) {
    if (sim) sim.stop();
    nodes = []; links = []; trail = [];
    byIdx = new Map();
    hovered = null; pinned = null;
    truncatedFrom = 0;
    label = '';
    deckRows = null; commanderRow = -1;
    userAdjusted = false;
    if (canvas) { transform = d3.zoomIdentity; draw(); }
    if (quiet) return;
    renderEmptyState();
    MM.setStatus('Pick a starting point for the walk.');
  }

  // Leaving keeps the graph. Rebuilding a walk you spent minutes growing, just because
  // you looked at the map, is the wrong trade — `enter()` restores it.
  //
  // Note what this does NOT do: touch canvas.style.display. `#plot:not(.force-mode)`
  // already hides it, and an inline `display:none` set here survived re-entry, so the
  // graph rebuilt correctly into a 0x0 hidden canvas and the mode looked dead.
  function exit() {
    active = false;
    MM.hideCardPopup();
    if (sim) sim.stop();
  }

  function focusCard(row) {
    const n = byIdx.get(row);
    if (n) branchFrom(n);
  }

  // Pin WITHOUT branching. An import wants the commander centred and open, not the deck
  // silently grown by six cards it did not contain — which is what focusCard does, and
  // did: importing a 129-card deck produced a 135-node graph.
  function pinCard(row) {
    const n = byIdx.get(row);
    if (!n) return;
    pinned = n;
    if (chrome === 'discovery' && window.Discovery) Discovery.focus(row);
    if (trail[trail.length - 1] !== n) trail.push(n);
    draw();
    renderPanel();
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
    // Discovery owns the side panel in its chrome. Drawing the walk panel over it would
    // erase the landing controls, the tray and the import box every time the graph
    // reheats — which it does on every branch.
    if (chrome === 'discovery') { if (window.Discovery) Discovery.render(); return; }
    // A "0 CARDS / 0 LINKS" panel is a dead end. Whoever calls this with an empty graph
    // wants the empty state, not an empty scoreboard — routing it here rather than at
    // each call site means a new caller cannot reintroduce the dead end.
    if (!nodes.length) { renderEmptyState(); return; }
    document.getElementById('deckPanel').classList.add('open');

    // Pinned wins over hovered. It used to be the other way round, which meant "click a
    // card to open its details" evaporated the instant the cursor moved one pixel off
    // the node — the panel would flick to whatever you happened to be passing over.
    // Hover is a preview of somewhere you might go; the pin is where you are.
    const h = pinned || hovered;
    let html =
      '<div class="deck-header"><h2>The Walk</h2>' +
      '<button class="lens-btn lens-btn-inline" onclick="Force.newWalk()" ' +
        'title="Pick a different deck or region">New walk ↺</button>' +
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

    // The card itself, rendered here rather than in #detailPanel — force mode hides that
    // panel, so the old "Open the card →" button pushed the card into an invisible element
    // and the reader saw nothing (it then popped open on leaving the Walk, which was
    // worse than nothing). MM.buildCardDetailHtml is the same markup Explore uses, so
    // there is one card renderer rather than two that drift.
    if (h) {
      html += '<div class="deck-section">' +
        '<div class="deck-section-title">' + (h === pinned ? 'Pinned' : 'Under the cursor') + '</div>' +
        '<div class="lens-title">' + MM.escHtml(h.name) + '</div>' +
        MM.buildCardDetailHtml(MM.cardRecord(h.row), h.row) +
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
    enter, exit, isActive, seedFrom, focusCard, pinCard,
    reheat, freeze, clearTrail, newWalk, tune, close, renderPanel, bbox,
    walkDeck, walkRegion, branchByRow, hasRow,
    // The rows currently on the graph, for Explore's orientation lens: the graph encodes
    // adjacency and has no absolute position, so "where does this sit in card space" is a
    // question only the world map can answer.
    rows() { return nodes.map(function (n) { return n.row; }); },
    pinnedRow() { return pinned ? pinned.row : -1; },
    // An explicit request, so it overrides "the user has taken the camera".
    fit: function () { userAdjusted = false; fitToGraph(true); userAdjusted = true; },
    get nodeCount() { return nodes.length; },
    get linkCount() { return links.length; },
    get trailLength() { return trail.length; },
    // Exposed for the browser tests: a link's `d` must be the 128-d chord distance, not
    // anything derived from screen position. Chord distance on a unit sphere is bounded
    // by [0, 2], which is the assertion.
    // Canvas text cannot be queried, so the count of labels the last draw actually placed
    // is the only way a test can tell "a representative sample" from "one" or "all".
    get labelCount() { return lastLabelCount; },
    get edgeLabelCount() { return lastEdgeLabelCount; },
    // For the browser tests: how the graph is split between what you loaded and what you
    // found, which is the thing the visual language is expressing.
    membership() {
      let deck = 0, explored = 0, commander = 0, deckLinks = 0;
      for (const n of nodes) { if (n.commander) commander++; if (n.deck) deck++; else explored++; }
      for (const l of links) if (l.source.deck && l.target.deck) deckLinks++;
      return { deck: deck, explored: explored, commander: commander,
               deckLinks: deckLinks, links: links.length };
    },
    LABEL_MAX,
    linkStats() {
      if (!links.length) return null;
      let min = Infinity, max = -Infinity, sum = 0;
      for (const l of links) { if (l.d < min) min = l.d; if (l.d > max) max = l.d; sum += l.d; }
      return { min: min, max: max, mean: sum / links.length, n: links.length };
    },
    MAX_NODES,
  };
})();
