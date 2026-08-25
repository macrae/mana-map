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
  // The zoom behaviour's own ceiling — `scaleExtent([0.02, 12])`. Fit may go
  // wherever a drag could; anything less makes the button refuse to do its job.
  const MAX_FIT_SCALE = 12;
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
  // Verified-line edges are short on purpose: `d` drives both ink brightness and the
  // link force's target distance, so combo pieces sit together and read as a unit.
  const VERIFIED_EDGE_LENGTH = 0.35;

  // Clear space a node needs around its DRAWN edge, in screen pixels.
  //
  // A node is drawn at a SCREEN-CONSTANT radius (`n.r / transform.k`) while d3's collide
  // force works in WORLD units, so a fixed world radius makes the on-screen gap depend on
  // whatever zoom the graph happened to be fitted at. Measured on a 78-node deck fitted at
  // k=0.505: collide radius `n.r + 3` = 9 world units = **4.5 screen px** between nodes
  // drawn **12 px wide**. They overlapped, and because `pick` awards the hover to the
  // nearest CENTRE, a node buried in a dense patch could never win: 20 of 78 were
  // unreachable at any cursor position, Sol Ring among them.
  //
  // The radius is therefore derived from the live zoom — see `collideRadius`.
  const NODE_CLEARANCE_PX = 5;

  // Camera motion. A slight overshoot — the camera passes its target and settles back —
  // reads as physical on a surface that is itself moving. d3's default overshoot (1.70)
  // is a bounce you notice; this is one you only feel.
  const FIT_MS = 520;
  const FIT_OVERSHOOT = 0.9;

  // Which rows came from a loaded deck, and which one is its commander. Nodes pulled in
  // by branching are deliberately NOT in here — that difference is the whole point of
  // loading a deck: you can see at a glance what you brought and what you found.
  let deckRows = null;
  let commanderRow = -1;
  // Set by any real pan/zoom/drag. While false the graph may frame itself; once true it
  // never moves the camera again without being asked.
  let userAdjusted = false;
  let canvas = null, ctx = null;
  let surface = null, cam = null;
  let sim = null, nodes = [], links = [], byIdx = new Map();
  let transform = null;          // d3.zoomIdentity once d3 is loaded
  let zoomBehaviour = null;      // the ONE instance — programmatic transforms must
                                 // go through it or its internal state desyncs and
                                 // the next wheel event snaps the view back
  let emb = null, dim = 0;
  let active = false;
  let hovered = null, pinned = null;
  // The verified line under the spotlight: a Set of rows and the line's id, or null.
  // Deliberately NOT node references (unlike `hovered`/`pinned`) — a line is chosen in
  // the sidebar from a manifest that speaks rows, and nodes are rebuilt on every reseed.
  let lineRows = null, lineId = null;
  /* A GROUP spotlight is a different question from a LINE spotlight, so it gets
   * its own state rather than borrowing `lineRows`. A line is a claim about
   * EDGES — which cards talk to each other — so it mutes every edge that is not
   * part of it. A group ("show me the ramp", "show me the three-drops") is a
   * claim about NODES, and muting the deck's verified lines while you look at
   * its ramp would hide the thing that makes the graph worth reading. Same
   * dimming, deliberately different scope. */
  let groupRows = null, groupLabel = null;
  // The deck's verified lines, as {id, title, pairs:[[rowA,rowB],…]}. Held so a reseed
  // can re-inject them — links point at live node objects, so they cannot outlive nodes.
  let deckLines = null;
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
    // Surface and camera from Stage — identical to the atlas renderer's, which is why
    // they are no longer written twice. Only the scale extent differs: a 500-node graph
    // has no business zooming to 400x.
    surface = Stage.surface(document.getElementById('plot'), 'force-canvas');
    canvas = surface.canvas;
    canvas.id = 'forceCanvas';
    ctx = surface.ctx;

    cam = Stage.camera({
      canvas: canvas,
      scaleExtent: [0.02, 12],
      // Double-click expands a card's neighbours here, so d3-zoom must not also
      // zoom on it. Suppressed where the behaviour is installed, not at a distance.
      dblclickZoom: false,
      onZoom: function (t, byUser) {
        const prev = transform;
        transform = t;
        // Once you have MOVED the camera it is yours: auto-fit stops competing with you.
        //
        // "Moved" is load-bearing, and it was missing. d3-zoom treats a bare mousedown as
        // the start of a pan and fires a zoom event for it with `sourceEvent` set — so a
        // plain CLICK marked the camera user-owned and silently disabled every auto-fit
        // for the rest of the session. That is why the first expansion on the Discover
        // landing left its neighbours off screen and never recovered: the graph had grown
        // past a k=12 single-card fit, and nothing was allowed to reframe it.
        //
        // A gesture that does not change the transform is not an adjustment.
        if (byUser && prev &&
            (Math.abs(t.k - prev.k) > 1e-6 ||
             Math.abs(t.x - prev.x) > 0.5 || Math.abs(t.y - prev.y) > 0.5)) {
          userAdjusted = true;
        }
        draw();
      },
    });
    zoomBehaviour = cam.behaviour;
    transform = cam.transform;

    // drag before zoom, and `subject` returns undefined on empty space so the gesture
    // falls through to the zoom behaviour. This is the standard way to let one canvas
    // both drag nodes and pan.
    const drag = d3.drag()
      .subject(function (ev) { return pick(ev.x, ev.y); })
      .on('start', function (ev) {
        // NOT `userAdjusted = true` here. Drag-start fires on mousedown over a node,
        // before any movement — so a plain CLICK claimed the camera and disabled every
        // auto-fit for the rest of the session. That is the whole of "the neighbours are
        // off screen and never come back": one click, and nothing was allowed to reframe
        // the graph again. Ownership is claimed in `drag` below, once something moves.
        if (!ev.active) sim.alphaTarget(0.25).restart();
        ev.subject.fx = ev.subject.x;
        ev.subject.fy = ev.subject.y;
      })
      .on('drag', function (ev) {
        // A real drag IS a deliberate arrangement of the view: stop auto-fitting under it.
        userAdjusted = true;
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
    /* AND TAKE dblclick.zoom BACK OFF, AGAIN.
     *
     * Stage already removed it when it created the behaviour, but the line above
     * re-installs the whole behaviour to get drag ordered before zoom — and a
     * re-install restores every listener it owns, including this one. Removing it once,
     * at the source, looked correct and was silently undone one line later.
     *
     * It is not only a stray zoom: d3's handler runs first and moves `transform`, so the
     * expand handler's `pick` then resolves stale click coordinates against a camera that
     * has already jumped, and finds nothing. The double-click appeared to do nothing at
     * all while the view zoomed in — which is exactly how it was reported. */
    d3.select(canvas).on('dblclick.zoom', null);

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
      // SELECT, do not grow. Expanding on a single click meant there was no way to
      // simply read a card: every look cost you six new nodes and a re-layout.
      if (hit) selectCard(hit.row);
    });

    /* Double-click EXPANDS. The first click of the double has already selected the card,
     * so the intermediate state is the correct one rather than merely harmless — no
     * debounce timer is needed and the select feels instant.
     *
     * d3-zoom installs its own `dblclick.zoom`; Stage is told not to (see
     * `dblclickZoom: false` below), or every expand would also zoom the camera. */
    canvas.addEventListener('dblclick', function (e) {
      const r = canvas.getBoundingClientRect();
      const hit = pick(e.clientX - r.left, e.clientY - r.top) || pinned || hovered;
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
    if (surface) surface.resize(draw);
  }

  // Frame the whole graph. Called after the layout settles and from the Fit button —
  // the extent is emergent, so it can only be measured, never predicted.
  function bbox(only) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (only && !only.has(n.row)) continue;
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }
    return { minX, maxX, minY, maxY, w: maxX - minX, h: maxY - minY };
  }

  /* Keep the growing graph in frame WHILE it settles.
   *
   * The only refit after a branch was the simulation's `end` handler, and `restart(0.6)`
   * at `alphaDecay 0.08` takes about 1.3 seconds to get there. For that whole second the
   * six cards you just asked for are outside the viewport — which is not a slow fit, it
   * is a missing one. Easing the camera toward the live extent on the way makes the
   * expansion read as the graph growing out to meet you.
   *
   * Throttled because a bbox per frame at 60fps is wasted work at this size, and eased
   * rather than snapped so the motion is continuous with the simulation's own.
   * `userAdjusted` gates it: once the camera is yours nothing competes for it. */
  const FOLLOW_EVERY = 4;        // ticks
  const FOLLOW_EASE = 0.18;      // fraction of the remaining distance per update
  let followTick = 0;
  let followRuns = 0;    // test visibility: how often the follow camera actually moved

  function followGraph() {
    if (userAdjusted || !canvas || !nodes.length || !transform) return;
    if (++followTick % FOLLOW_EVERY) return;
    const b = bbox();
    if (!isFinite(b.w) || !isFinite(b.h)) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const gw = Math.max(b.w, 1) * 1.18, gh = Math.max(b.h, 1) * 1.18;
    const k = Math.min(w / gw, h / gh, MAX_FIT_SCALE);
    const tx = w / 2 - k * (b.minX + b.maxX) / 2;
    const ty = h / 2 - k * (b.minY + b.maxY) / 2;
    // Nothing to chase: within a pixel and a per-mille of scale, leave it alone so a
    // settled graph is not permanently nudged.
    if (Math.abs(k - transform.k) < transform.k * 0.001 &&
        Math.abs(tx - transform.x) < 1 && Math.abs(ty - transform.y) < 1) return;
    const e = FOLLOW_EASE;
    const next = d3.zoomIdentity
      .translate(transform.x + (tx - transform.x) * e, transform.y + (ty - transform.y) * e)
      .scale(transform.k + (k - transform.k) * e);
    // Through the behaviour, never by assignment — d3 keeps its own copy on the node.
    followRuns++;
    d3.select(canvas).call(zoomBehaviour.transform, next);
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
    // Cap the zoom-in at the zoom behaviour's OWN ceiling, so a fit can go anywhere the
    // user could drag to. It used to cap at 1.6 on the reasoning that "a 59-unit-wide
    // graph blown up to fill a 1439px canvas is not more readable, just bigger" — which
    // is wrong here, because node radius and label text are drawn in SCREEN space
    // (`r / transform.k`). Zooming in does not enlarge anything; it only spreads the
    // nodes apart, which is exactly what a label needs.
    //
    // Measured: a 31-node graph spanning 42x49 world units framed at k=1.6 is ~67px
    // wide, its neighbours ~12px apart against ~100px labels, so every label collided
    // and `labelCount` read 1. The cap, not the collision rule, was the problem.
    const k = Math.min(w / (gw * 1.18), h / (gh * 1.18), MAX_FIT_SCALE);
    const t2 = d3.zoomIdentity
      .translate(w / 2 - k * (minX + maxX) / 2, h / 2 - k * (minY + maxY) / 2)
      .scale(k);
    const sel = d3.select(canvas);
    if (animate === false) sel.call(zoomBehaviour.transform, t2);
    else sel.transition().duration(FIT_MS).ease(d3.easeBackOut.overshoot(FIT_OVERSHOOT))
            .call(zoomBehaviour.transform, t2);
  }

  // Screen pixel -> node, through the zoom transform. A linear scan is honest at 500
  // nodes (0.02 ms); the quadtree earns its keep at 34,322, which is where it goes next.
  /* Collide radius in WORLD units for the zoom the graph is currently viewed at.
   *
   * `n.r` is a screen radius, so the world radius it occupies is `n.r / k`. Adding the
   * clearance in the same space keeps the on-screen gap constant however the graph is
   * framed, which is what makes every node reachable by `pick`. Guarded against a
   * missing/zero transform because forces are constructed before the first fit. */
  function collideRadius(n) {
    const k = (transform && transform.k) ? transform.k : 1;
    return (n.r + NODE_CLEARANCE_PX) / k;
  }

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
    const w = surface.width, h = surface.height;
    const closeWorld = surface.open(transform);

    // Links, faint, brighter the closer the pair. A short bright edge is the model
    // saying "these two are nearly the same card".
    //
    // Deck edges — both ends from a loaded decklist — are drawn warm and heavier, so the
    // deck reads as a structure you can see through the exploration rather than dissolving
    // into it the moment you branch. Everything cool and thin is something you found.
    // The relation inks live in Stage now, so the atlas can draw the same edge and mean
    // the same thing by it. Deck edges — both ends from a loaded decklist — stay warm and
    // heavier, so the deck reads as a structure you can see through the exploration
    // rather than dissolving into it the moment you branch.
    Stage.drawEdges(ctx, links, function (n) { return [n.x, n.y]; }, transform.k, {
      width: 1,
      relOf: function (l) {
        // With a line under the spotlight, an edge is either part of it or it is
        // scenery. `relOf` is the documented hook for deciding an edge's kind from
        // context rather than storing it, and "which line am I looking at" is context.
        if (lineRows) return l.line === lineId ? 'verified' : 'muted';
        /* A GROUP spotlight mutes the edges it has nothing to do with, and
         * keeps the ones that TOUCH it.
         *
         * The first cut left every edge alone, on the reasoning that a group
         * is a claim about nodes while a line is a claim about edges. That was
         * half right and looked wrong: with all the edges at full strength the
         * graph stayed a bright web while the nodes receded, so the spotlight
         * barely read. Measured at the dimmed nodes, the ink moved 6% because
         * edges converge on every node centre.
         *
         * Keeping the edges that touch the lit set is the honest middle: an
         * edge INTO your ramp is part of what you asked about; an edge between
         * two cards you did not ask about is scenery. */
        if (groupRows) {
          if (!groupRows.has(l.source.row) && !groupRows.has(l.target.row)) return 'muted';
        }
        // At rest a verified edge is still visible — you should be able to see the deck's
        // lines without clicking — but quiet, so selecting one is a visible change.
        if (l.rel === 'verified') return 'verifiedQuiet';
        return (l.source.deck && l.target.deck) ? 'deck' : l.rel;
      },
      weightOf: function (l, rel) {
        if (rel === 'verified') return 2.4;
        if (rel === 'verifiedQuiet') return 1.4;
        if (rel === 'muted') return 1;
        return rel === 'deck' ? 1.7 : 1;
      },
    });

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
      // Spotlight: with a line active everything else recedes so the line reads at a
      // glance. Otherwise the original two states — what you brought vs what you found.
      /* ONE alpha for the whole node, fill AND rim.
       *
       * This used to dim only the fill and then reset to 1 before the rings,
       * so in Build — where every deck card carries a white rim — a spotlight
       * left 96 bright outlines on screen and receded almost nothing. Measured
       * while adding the group spotlight: the ink AT THE DIMMED NODES moved
       * 3.6% when two thirds of them were supposed to be dark. The rim was the
       * signal, and it was never dimming. `setLine` had the same defect since
       * it shipped.
       *
       * The commander's gold ring is deliberately exempt below: it answers
       * "where is my commander", which stays true while you look at something
       * else. */
      const spot = lineRows ? (lineRows.has(n.row) ? 1 : 0.15)
                 : groupRows ? (groupRows.has(n.row) ? 1 : 0.15)
                             : ((deckRows && !n.deck) ? 0.5 : 1);
      ctx.globalAlpha = spot;
      ctx.fillStyle = n.color;
      ctx.fill();
      // The commander gets a permanent gold ring — it is the one card the deck is built
      // around, and it should be findable without hunting.
      if (n.commander) {
        ctx.globalAlpha = 1;   // always findable, spotlight or not
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
      ctx.globalAlpha = 1;
      if (lineRows && lineRows.has(n.row)) {
        ctx.lineWidth = 2.5 / transform.k;
        ctx.strokeStyle = '#4CAF50';
        ctx.stroke();
      } else if (groupRows && groupRows.has(n.row)) {
        // NOT the line green. A group spotlight and a line spotlight must not
        // look the same, or "these cards are your ramp" reads as "these cards
        // are a rules-verified line" — which is the evidence contract leaking
        // into a colour.
        ctx.lineWidth = 2 / transform.k;
        ctx.strokeStyle = '#fff';
        ctx.stroke();
      }
      if (n === hovered || n === pinned || onTrail.has(n)) {
        ctx.lineWidth = (n === hovered ? 2.5 : 1.6) / transform.k;
        ctx.strokeStyle = n === hovered ? '#fff' : '#c4a747';
        ctx.stroke();
      }
    }
    closeWorld();

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
    // A line you asked to see must be named. Ahead of the trail and the commander,
    // because while the spotlight is on it is the only thing being asked about.
    if (lineRows) {
      for (const n of nodes) {
        if (lineRows.has(n.row) && priority.indexOf(n) === -1) priority.push(n);
      }
    } else if (groupRows) {
      for (const n of nodes) {
        if (groupRows.has(n.row) && priority.indexOf(n) === -1) priority.push(n);
      }
    }
    for (let i = trail.length - 1; i >= 0; i--) {
      if (priority.indexOf(trail[i]) === -1) priority.push(trail[i]);
    }
    // Commander, then the rest of the deck, then seeds — on a loaded deck those are the
    // cards worth naming before anything you wandered into.
    for (const n of nodes) if (n.commander && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (n.deck && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (n.seed && priority.indexOf(n) === -1) priority.push(n);
    for (const n of nodes) if (priority.indexOf(n) === -1) priority.push(n);

    // One collision set, shared by edge labels and node labels, so a synergy reason can
    // never sit on top of a card name. `Stage.placer` is the same greedy AABB pass the
    // atlas uses for region labels — it was copied there by hand when that renderer was
    // written, and there is one of it now.
    const place = Stage.placer(LABEL_GAP);
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
        if (!place.claim(box)) continue;
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
      // The hovered and pinned cards are never suppressed — you asked about those.
      // Nor is a card in the line under the spotlight, for the same reason.
      const keep = n === hovered || n === pinned || !!(lineRows && lineRows.has(n.row));
      if (!place.claim(box, keep)) continue;
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
        .force('collide', d3.forceCollide().radius(collideRadius).strength(0.85))
        .force('center', d3.forceCenter(0, 0).strength(0.04))
        .velocityDecay(PHYSICS.velocityDecay)
        .alphaDecay(PHYSICS.alphaDecay)
        .on('tick', function () { draw(); followGraph(); })
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

  // One engine, one chrome. `opts.chrome` used to select between this file owning the
  // side panel and Discovery owning it; The Walk is gone and Discovery always owns it, so
  // the option is accepted and ignored rather than removed from every call site.
  async function enter(rowIndices, seedLabel, opts) {
    if (opts && opts.deck) {
      deckRows = opts.deck.rows || null;
      commanderRow = typeof opts.deck.commander === 'number' ? opts.deck.commander : -1;
      deckLines = opts.deck.lines || null;
    } else if (opts && opts.deck === null) {
      deckRows = null; commanderRow = -1; deckLines = null;
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
      renderPanel();
      MM.setStatus('Pick a card, a deck or a region to start from.');
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
    injectLines();
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

    // SECOND PASS, and it is not optional. `collideRadius` is expressed in world units
    // derived from the live zoom, but the zoom is only known once the graph has been
    // framed — the first settle above ran at the identity transform. Re-settle against
    // the scale the graph is actually viewed at, then reframe. Two passes converge; a
    // third moved nothing measurable. Without this the collide is computed for a zoom
    // nobody is looking at and nodes overlap on screen exactly as before.
    // `d3.forceCollide` reads its radius accessor once, in `initialize` — NOT per tick.
    // Re-setting the force is therefore the only thing that re-evaluates it against the
    // new zoom. Merely calling `sim.alpha().restart()` left the radii computed at the
    // identity transform, which is why the first attempt at this changed nothing.
    sim.force('collide', d3.forceCollide().radius(collideRadius).strength(0.85));
    sim.alpha(0.3).restart();
    sim.stop();
    for (let i = 0; i < Math.min(SETTLE_TICKS, 40 + nodes.length * 2); i++) sim.tick();
    fitToGraph(false, true);
    draw();
    // Left stopped on purpose. Dragging reheats it (`alphaTarget` on drag start) and
    // branching reheats it (`restart(0.6)`), so it is alive exactly when something is
    // happening and perfectly steady the rest of the time — which is what "snappy" means
    // here. A graph that keeps drifting under the cursor is not responsive, it is busy.
    sim.stop();

    // A single seed is a landing: pin it so it draws as card art rather than a 6 px dot.
    if (nodes.length === 1) pinned = nodes[0];
    renderPanel();
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
    // Note the card; the PANEL is `renderPanel`'s decision, and it asks the mode.
    if (window.Discovery) Discovery.setCurrent(node.row);
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

  // Take a card that is NOT on the graph and add it to the graph you already have.
  //
  // This exists because the alternative was destroying your work. Branching from a card
  // found in the atlas used to go through `Discovery.show`, which calls `newWalk(true)` —
  // so clicking a relation in Explore on any card you had not already walked to silently
  // threw away the entire graph. `focus()` carries a comment warning about exactly that
  // hazard; the Explore path took the destructive branch anyway.
  //
  // The node is born at the graph's centre of mass rather than at its world position:
  // seeding from `makeNode`'s world coordinates would drop it wherever the atlas happens
  // to put it, which against an established cluster is usually off screen. Arriving in the
  // middle and being pushed out by the physics reads as joining; arriving 4,000 units away
  // reads as a bug.
  function adoptRow(row) {
    let n = byIdx.get(row);
    if (n) return n;
    n = makeNode(row, false);
    let cx = 0, cy = 0;
    for (const m of nodes) { cx += m.x; cy += m.y; }
    if (nodes.length) { cx /= nodes.length; cy /= nodes.length; }
    n.x = cx + (Math.random() - 0.5) * 40;
    n.y = cy + (Math.random() - 0.5) * 40;
    nodes.push(n);
    byIdx.set(row, n);
    // Link it to whatever it already belongs beside, so it lands attached rather than
    // drifting. `branchFrom` then adds the links for the relation actually asked for.
    let made = 0;
    for (const nb of Discovery.neighbours(row, 'similar')) {
      if (made >= LINKS_PER_NODE) break;
      const other = byIdx.get(nb.row);
      if (!other || other === n || hasLink(n, other)) continue;
      links.push({ source: n, target: other, d: edgeLength(nb), rel: nb.relation,
                   reason: nb.reason || null });
      made++;
    }
    return n;
  }

  /* A rules-verified line is a claim about which cards talk to each other, and the graph
   * had no way to say it: its links come only from embedding similarity, so two combo
   * pieces that are not near-neighbours had no edge at all. These are injected as real
   * links, so they join the simulation and pull their endpoints together — which is the
   * point. Deduped against similarity links via `hasLink`, so a pair that is already
   * connected keeps one edge and gains the line's identity. */
  function injectLines() {
    if (!deckLines) return;
    for (const line of deckLines) {
      /* The pairs this line knows the DIRECTION of, as "from-row:to-row".
       *
       * A verified line is a clique over the cards its stack names, and a
       * clique has no arrows — `{source, target}` is whichever order the pair
       * was built in. `engine.json` is the one artifact that says which way a
       * resource moves, so only the pairs that span its `from` and `to` stages
       * get an arrowhead. The rest stay undirected, which is honest: for those
       * pairs the direction is genuinely not known. */
      const dir = new Set((line.directed || []).map(function (p) { return p[0] + ':' + p[1]; }));
      for (const pair of (line.pairs || [])) {
        const a = byIdx.get(pair[0]), b = byIdx.get(pair[1]);
        if (!a || !b || a === b) continue;
        const fwd = dir.has(pair[0] + ':' + pair[1]);
        const rev = dir.has(pair[1] + ':' + pair[0]);
        const existing = findLink(a, b);
        if (existing) {
          // Keep the geometry, adopt the meaning: a verified line outranks "these two
          // cards are similar" as a reason for an edge to exist.
          existing.rel = 'verified';
          existing.line = line.id;
          existing.reason = line.title || existing.reason;
          if (fwd || rev) {
            // Point the link the way the engine points. An existing similarity
            // edge was built in arbitrary order, so the endpoints are swapped
            // when the engine disagrees rather than drawing an arrow backwards.
            if (rev && existing.source === a) {
              const t = existing.source; existing.source = existing.target; existing.target = t;
            } else if (fwd && existing.source === b) {
              const t = existing.source; existing.source = existing.target; existing.target = t;
            }
            existing.dir = true;
            existing.carries = line.carries || null;
            if (line.carries) existing.reason = line.carries;
          }
          continue;
        }
        const src = rev ? b : a, dst = rev ? a : b;
        links.push({ source: src, target: dst, d: VERIFIED_EDGE_LENGTH,
                     rel: 'verified', line: line.id,
                     dir: !!(fwd || rev),
                     carries: (fwd || rev) ? (line.carries || null) : null,
                     // The `carries` noun beats the stack title as an edge label:
                     // "bodies" says what moves, where the title says what was
                     // asked. The title is still on the panel row.
                     reason: ((fwd || rev) && line.carries) ? line.carries : (line.title || null) });
      }
    }
  }

  function findLink(a, b) {
    for (const l of links) {
      const s = l.source.row !== undefined ? l.source.row : l.source;
      const t2 = l.target.row !== undefined ? l.target.row : l.target;
      if ((s === a.row && t2 === b.row) || (s === b.row && t2 === a.row)) return l;
    }
    return null;
  }

  function branchByRow(row, relation) {
    if (nodes.length >= MAX_NODES && !byIdx.has(row)) {
      MM.setStatus('At the ' + MAX_NODES + '-card cap — trim the walk or start a new one.');
      return;
    }
    branchFrom(adoptRow(row), relation);
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
    deckRows = null; commanderRow = -1; deckLines = null;
    // A spotlight must not survive into a graph that no longer contains its line.
    lineRows = null; lineId = null;
    groupRows = null; groupLabel = null;
    userAdjusted = false;
    if (canvas) { transform = d3.zoomIdentity; draw(); }
    if (quiet) return;
    renderPanel();
    MM.setStatus('Pick a card, a deck or a region to start from.');
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
  /* Move the gold ring without touching the graph. Re-seeding to change a commander
   * would throw away everything you had branched to. */
  /* Put a rules-verified line under a spotlight. A restyle, never a reseed — the same
   * discipline as `setCommander` below, and for the same reason: rebuilding the graph to
   * answer "show me this line" would throw away everything branched to since it loaded.
   * No reheat either. The nodes must not move, or the click reads as the graph exploding
   * rather than as an answer.
   *
   * Rows, not node objects: the line is picked in the sidebar from a manifest that speaks
   * row indices, and nodes are rebuilt on every reseed. */
  function setLine(rows, id, opts) {
    lineRows = (rows && rows.length) ? new Set(rows) : null;
    lineId = lineRows ? (id == null ? null : id) : null;
    if (lineRows && opts && opts.fit) {
      // Frame the line itself, not the deck. `userAdjusted` is deliberately ignored:
      // this is an explicit request, not an automatic fit.
      fitToRows(lineRows);
    }
    draw();
    renderPanel();
  }

  function clearLine() { setLine(null); }

  /* Spotlight a GROUP of cards — a role family, a mana-value bucket, a colour.
   * Restyle only, exactly like `setLine`: no reseed, no reheat, no camera move.
   * The nodes must not move, or a click on a bar chart reads as the graph
   * exploding rather than as an answer.
   *
   * No `fit`, and that is deliberate where `setLine` offers one: a line is a
   * handful of cards worth framing, a group is routinely a third of the deck
   * and framing it is just the fit you already had. */
  function setGroup(rows, label) {
    groupRows = (rows && rows.length) ? new Set(rows) : null;
    groupLabel = groupRows ? (label || null) : null;
    draw();
    renderPanel();
  }

  function clearGroup() { setGroup(null); }

  /* Frame a subset. Same maths as `fitToGraph`, which drives the camera through
   * `zoomBehaviour.transform` rather than `Stage.camera` — programmatic transforms have
   * to go through the one zoom instance or its internal state desyncs and the next wheel
   * event snaps the view back. */
  function fitToRows(rowSet) {
    if (!canvas || !nodes.length) return;
    const b = bbox(rowSet);
    if (!isFinite(b.w) || !isFinite(b.h)) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    // A line can be two adjacent cards, which would otherwise frame at absurd zoom.
    // Pad generously so the line lands in the middle of its neighbourhood, in context.
    const pad = Math.max(b.w, b.h, 1) * 0.6 + 8;
    const minX = b.minX - pad, maxX = b.maxX + pad;
    const minY = b.minY - pad, maxY = b.maxY + pad;
    const gw = Math.max(maxX - minX, 1), gh = Math.max(maxY - minY, 1);
    const k = Math.min(w / (gw * 1.18), h / (gh * 1.18), MAX_FIT_SCALE);
    const t2 = d3.zoomIdentity
      .translate(w / 2 - k * (minX + maxX) / 2, h / 2 - k * (minY + maxY) / 2)
      .scale(k);
    d3.select(canvas).transition().duration(FIT_MS)
      .ease(d3.easeBackOut.overshoot(FIT_OVERSHOOT))
      .call(zoomBehaviour.transform, t2);
  }

  function setCommander(row) {
    commanderRow = typeof row === 'number' ? row : -1;
    for (const n of nodes) {
      n.commander = n.row === commanderRow;
      n.r = n.commander ? 9 : (n.deck ? 6 : 4.5);
    }
    draw();
    renderPanel();
  }

  /* SELECT a card without growing the graph.
   *
   * The sibling of `pinCard`, minus the trail push. A single click is "let me look at
   * this", and the breadcrumb records where you WENT — expansions. Clicking around a
   * 78-node deck to read cards would otherwise fill the trail with places you never
   * travelled to, and the gold path stops meaning anything.
   */
  function selectCard(row) {
    const n = byIdx.get(row);
    if (!n) return;
    pinned = n;
    if (window.Discovery) Discovery.setCurrent(row);
    draw();
    renderPanel();
  }

  function pinCard(row) {
    const n = byIdx.get(row);
    if (!n) return;
    pinned = n;
    if (window.Discovery) Discovery.setCurrent(row);
    if (trail[trail.length - 1] !== n) trail.push(n);
    draw();
    renderPanel();
  }

  // ── Panel ───────────────────────────────────────────────────────────────

  // Somewhere to start. Decks come from the tracked manifest; regions from the HDBSCAN
  // membership the clustering now keeps. Both are one click.
  // `renderEmptyState` and `walkDeck` lived here. The empty state was the walk panel's
  // deck+region picker; Discovery's panel now carries both, and `Discovery.onDeckPick` is
  // strictly better than `walkDeck` was — it passes `opts.deck`, so a loaded deck gets the
  // commander ring and the deck ink that `walkDeck` never set.
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

  // ONE PANEL. This used to branch: Discovery owned the side panel in discovery chrome,
  // and in walk chrome this function rendered a second panel of its own — the scoreboard,
  // Fit/Reheat/New walk, the physics sliders and the trail. The Walk was Discover with
  // different chrome (two behaviours and a status string, across four `chrome ===` reads),
  // so it was deleted and its panel's keepers moved into `Discovery.render`: the
  // scoreboard, Fit, Reheat, Start over, the trail, and the truncation notice — which had
  // never been shown in discovery chrome at all, so a >500-card import was silently cut.
  //
  // The physics sliders were NOT ported. They were a debugging surface and nothing else in
  // the product exposes tuning; `Force.tune` remains for the console.
  // WHO OWNS THE PANEL depends on the mode, not on a flag passed at `enter()`. The old
  // `chrome` option was that flag, and collapsing it to "Discovery always owns it" was
  // wrong the moment Build started seeding the graph too: every reheat repainted Build's
  // roles, curve and verified lines with Discover's landing controls. One engine, two
  // owners, and the engine has to ask which.
  function renderPanel() {
    if (!active) return;
    if (window.MM && MM.mode === 'build') {
      if (window.Build && window.Build.renderPanel) window.Build.renderPanel();
      return;
    }
    if (window.Discovery) Discovery.render();
  }

  function tune(key, value, el) {
    const v = parseFloat(value);
    if (el && el.previousElementSibling) el.previousElementSibling.querySelector('span').textContent = v;
    if (key === 'velocityDecay') { PHYSICS.velocityDecay = v / 100; sim.velocityDecay(PHYSICS.velocityDecay); }
    else if (key === 'charge') { PHYSICS.charge = v; sim.force('charge').strength(v); }
    else if (key === 'linkScale') { PHYSICS.linkScale = v; sim.force('link').distance(l => l.d * v); }
    sim.alpha(Math.max(sim.alpha(), 0.35)).restart();
  }

  window.addEventListener('resize', function () { if (active) resize(); });

  // force.js is where the graph physically lives — d3 owns these nodes and mutates them
  // every tick — so it registers itself as Session's storage rather than Session keeping a
  // second copy that could drift. Session is the interface everything else asks.
  if (window.Session) {
    Session.useGraph({
      rows: function () { return nodes.map(function (n) { return n.row; }); },
      links: function () {
        return links.map(function (l) {
          return { a: l.source.row, b: l.target.row, rel: l.rel || 'similar',
                   reason: l.reason || null, d: l.d };
        });
      },
      has: hasRow,
      grow: branchByRow,
    });
  }

  window.Force = {
    enter, exit, isActive, seedFrom, focusCard, pinCard,
    reheat, freeze, clearTrail, newWalk, tune, renderPanel, bbox,
    setLine, clearLine, setGroup, clearGroup, selectCard,
    walkRegion, branchByRow, hasRow, setCommander,
    /* Adopt WITHOUT branching. `branchByRow` was the only way in, and it also
     * pulls the row's relations — right when the pilot clicked a relation, wrong
     * when they named a card to add: the graph would grow by twelve when one was
     * asked for. Exported so "add this card where it belongs" is sayable on its
     * own, since that is the only shape of growth that cannot delete anything. */
    adopt(row) { const n = adoptRow(row); restart(0.3); draw(); return !!n; },
    // The rows currently on the graph, for Explore's orientation lens: the graph encodes
    // adjacency and has no absolute position, so "where does this sit in card space" is a
    // question only the world map can answer.
    rows() { return nodes.map(function (n) { return n.row; }); },
    // The typed links, as ROWS rather than node objects — so a consumer that has no
    // simulation (the atlas, which draws the same relations at world positions) can read
    // them without touching d3's mutable bodies.
    links() {
      return links.map(function (l) {
        return { a: l.source.row, b: l.target.row, rel: l.rel || 'similar',
                 reason: l.reason || null, d: l.d, line: l.line || null,
                 // `a -> b` is only a claim when `dir` is set; otherwise the
                 // order is whichever way the pair happened to be built.
                 dir: !!l.dir, carries: l.carries || null };
      });
    },
    pinnedRow() { return pinned ? pinned.row : -1; },
    /* Node positions in SCREEN space, for the browser tests. Whether a card can be
     * hovered is decided by whether its drawn circle is buried under a neighbour's —
     * `pick` awards the hover to the nearest centre — and that is a geometric fact the
     * canvas cannot be asked about directly. */
    screenNodes() {
      return nodes.map(function (n) {
        const p = transform ? transform.apply([n.x, n.y]) : [n.x, n.y];
        // The drawn screen radius IS n.r — world radius is n.r/k and screen = world*k.
        return { row: n.row, name: n.name, x: p[0], y: p[1], r: n.r };
      });
    },
    // Which verified line is under the spotlight, and how many of its cards are on the
    // graph. Canvas ink is unassertable, so the browser tests read state through here.
    /* Has the user taken the camera? Auto-fit and the follow camera both stand down
     * when this is true, so a test that cannot see it cannot tell "the fit is broken"
     * from "the fit correctly declined". */
    get cameraOwnedByUser() { return userAdjusted; },
    get followCount() { return followRuns; },
    get activeLine() { return lineId; },
    get lineRowCount() { return lineRows ? lineRows.size : 0; },
    // The rows themselves, not just how many. A pixel test cannot assert "the line
    // stopped shouting" from the whole canvas — clearing the spotlight un-mutes every
    // OTHER verified line, and their combined quiet ink replaces the one bright line's
    // almost exactly (measured on goblin-storm: 868 green px spotlit, 833 cleared, while
    // the line's own box went 1024 -> 301). So the test has to look where the line IS,
    // and to do that it needs to know which cards it joins.
    get spotlitRows() { return lineRows ? Array.from(lineRows) : []; },
    get verifiedLinkCount() {
      return links.filter(function (l) { return l.rel === 'verified'; }).length;
    },
    // An explicit request, so it overrides "the user has taken the camera".
    fit: function () { userAdjusted = false; fitToGraph(true); userAdjusted = true; },
    get nodeCount() { return nodes.length; },
    get linkCount() { return links.length; },
    get trailLength() { return trail.length; },
    // Where the walk has been, for the panel. The walk panel rendered this itself; with
    // one panel it has to be readable from outside.
    trailNames() { return trail.map(function (n) { return n.name; }); },
    // How many cards were dropped to fit MAX_NODES. Surfaced by the panel, and it never
    // was in discovery chrome — a >500-card import was silently truncated.
    get truncatedFrom() { return truncatedFrom; },
    get label() { return label; },
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
