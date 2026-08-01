/* stage.js — the primitives both canvas renderers were writing twice.
 *
 * There are two canvas renderers in this project and there always will be, because they
 * answer different questions: `render/canvas.js` draws 34,322 cards at fixed world
 * positions (where does this sit), `force.js` draws a few hundred at simulated positions
 * (what is this next to). What they do NOT need is two of everything underneath.
 *
 * Before this file they shared zero lines while separately implementing eight things:
 * canvas creation and devicePixelRatio resize (character-for-character identical, both
 * carrying the same "cost an afternoon" comment), d3-zoom wiring, the draw prologue,
 * world→screen, screen→world, fit-to-extent, the `/transform.k` constant-screen-size
 * trick, and greedy AABB label collision — which `render/canvas.js` openly noted it had
 * copied from The Walk.
 *
 * WHAT IS SHARED AND WHAT IS NOT. Stage owns the *surface*: pixels, gestures, geometry.
 * It does not own positions, and that is the load-bearing decision. `force.js` mutates
 * node x/y on every simulation tick; `canvas.js` never touches its points and moves only
 * the transform. An abstraction that owned positions would have to serve both and would
 * become a union of two designs. So Stage never stores a coordinate — callers paint
 * inside a transform Stage sets up, and hand it screen-space boxes when they want labels
 * placed.
 *
 * Nothing here knows what a card is.
 */
window.Stage = (function () {
  'use strict';

  // ── Surface ───────────────────────────────────────────────────────────
  //
  // Only the backing store is sized here — the element's box comes from CSS. A <canvas>
  // is a replaced element, so an inline width decouples it from its parent and it stops
  // following the layout. Both renderers learned that separately; it is written once now.
  function surface(host, className) {
    const canvas = document.createElement('canvas');
    canvas.className = className;
    host.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let dpr = 1;

    function resize(onResized) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return false;          // hidden: leave the backing store alone
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (onResized) onResized();
      return true;
    }

    return {
      canvas: canvas,
      ctx: ctx,
      resize: resize,
      get width() { return canvas.clientWidth; },
      get height() { return canvas.clientHeight; },
      destroy: function () { if (canvas.parentNode) canvas.parentNode.removeChild(canvas); },
      /* Open a frame in world space and return the closer. Every draw loop began with
       * this exact four-line prologue. */
      open: function (transform) {
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
        ctx.save();
        ctx.translate(transform.x, transform.y);
        ctx.scale(transform.k, transform.k);
        return function close() { ctx.restore(); };
      },
    };
  }

  // ── Camera ────────────────────────────────────────────────────────────
  //
  // d3-zoom, wired the same way twice. The differences that remain are real and stay
  // parameters: the scale extent (the atlas zooms to 400x, a 500-node graph to 12x) and
  // what each renderer does on a zoom event.
  function camera(opts) {
    const canvas = opts.canvas;
    let transform = d3.zoomIdentity;

    const behaviour = d3.zoom()
      .scaleExtent(opts.scaleExtent || [0.2, 400])
      .on('zoom', function (ev) {
        transform = ev.transform;
        // `sourceEvent` is null for programmatic transforms and set for real gestures —
        // the difference between "the code moved the camera" and "the user did". Callers
        // that auto-fit need it to stop competing with the user.
        opts.onZoom(transform, !!ev.sourceEvent);
      });
    if (opts.filter) behaviour.filter(opts.filter);
    d3.select(canvas).call(behaviour);

    /* ALWAYS go through the behaviour, never assign `transform` directly: d3 keeps its
     * own copy on the node, so a direct assignment is silently reverted by the next
     * gesture. Both renderers hit this and both left a comment about it. */
    function set(t, animate) {
      const sel = d3.select(canvas);
      if (animate) sel.transition().duration(animate === true ? 450 : animate).call(behaviour.transform, t);
      else sel.call(behaviour.transform, t);
    }

    return {
      get transform() { return transform; },
      behaviour: behaviour,
      set: set,
      reset: function () { set(d3.zoomIdentity); },
      /* The fit both renderers computed by hand: isotropic scale to cover an extent, then
       * centre it. `pad` shrinks the fit; `maxScale` caps zoom-in so a two-node graph does
       * not fill the screen at 40x. Returns the transform rather than applying it, because
       * one caller wants it animated and the other does not. */
      fitTransform: function (box, w, h, o) {
        o = o || {};
        const dx = Math.max(box.x1 - box.x0, 1e-6);
        const dy = Math.max(box.y1 - box.y0, 1e-6);
        let k = Math.min(w / dx, h / dy) * (o.pad == null ? 0.92 : o.pad);
        if (o.maxScale) k = Math.min(k, o.maxScale);
        if (o.minScale) k = Math.max(k, o.minScale);
        const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
        return d3.zoomIdentity.translate(w / 2 - k * cx, h / 2 - k * cy).scale(k);
      },
    };
  }

  // ── Labels ────────────────────────────────────────────────────────────
  //
  // Greedy first-come placement with axis-aligned-box rejection, in SCREEN space.
  //
  // Screen space is not a detail. An earlier version of the atlas compared world-space
  // centroids (±40 units) against pixel widths (~150px), so almost everything collided
  // with almost everything and 2 of 19 labels survived. Boxes must be measured where they
  // are drawn.
  //
  // Callers pass boxes in draw priority order — the first to ask for space gets it — and
  // may mark an entry `keep` to make it un-rejectable (the hovered and pinned cards are
  // never suppressed; you asked about those). Returns a boolean per input.
  /* Incremental: ask for space one box at a time and draw as you go. The Walk needs this
   * shape rather than a batch — it draws each label the moment it is accepted, caps on
   * the number *accepted* rather than attempted, and shares ONE collision set between
   * edge labels and node labels so a synergy reason can never sit on a card name. */
  function placer(gap) {
    const g = gap == null ? 6 : gap;
    const placed = [];
    return {
      /* `keep` forces acceptance — the hovered and pinned cards are never suppressed,
       * because you asked about those. */
      claim: function (b, keep) {
        for (const p of placed) {
          if (b.x0 < p.x1 + g && b.x1 > p.x0 - g &&
              b.y0 < p.y1 + g && b.y1 > p.y0 - g) { if (!keep) return false; break; }
        }
        placed.push(b);
        return true;
      },
      get count() { return placed.length; },
    };
  }

  /* Batch convenience over the same placer, for callers that know every box up front —
   * the atlas measures all its region labels in the DOM before deciding. */
  function placeLabels(boxes, gap) {
    const p = placer(gap);
    const out = new Array(boxes.length);
    for (let i = 0; i < boxes.length; i++) out[i] = p.claim(boxes[i], boxes[i].keep);
    out.placed = p.count;
    return out;
  }

  // ── Typed edges ───────────────────────────────────────────────────────
  //
  // The relation inks, in one place, because an edge means the same thing on both
  // surfaces and the atlas could not draw one at all.
  //
  // What the atlas had was `mode: 'lines'` — a single flattened polyline with one colour
  // for the whole layer, excluded from the quadtree so it could never be hovered. That is
  // enough to draw the Deck Lens's verified-line edges and not enough for anything that
  // has to say *why* two cards are connected. `force.js` had the real model
  // (`{source, target, rel, reason}`) and kept it to itself.
  //
  // Colour carries the relation so a label only has to carry the detail.
  const INK = {
    deck: function (c) { return 'rgba(196,167,71,' + (0.22 + c * 0.45).toFixed(3) + ')'; },
    synergy: function () { return 'rgba(168,120,214,0.55)'; },
    obsolete: function () { return 'rgba(214,120,120,0.5)'; },
    similar: function (c) { return 'rgba(122,138,196,' + (0.08 + c * 0.42).toFixed(3) + ')'; },
  };

  function edgeInk(rel, closeness) {
    return (INK[rel] || INK.similar)(closeness == null ? 0.5 : closeness);
  }

  /* Draw typed edges in world space. `at(node)` yields [x, y] so this works for both a
   * simulation node (positions on the object) and an atlas row (positions in a table) —
   * Stage still never stores a coordinate. `k` is the live zoom scale, so line widths
   * divide by it and stay constant on screen. */
  function drawEdges(ctx, edges, at, k, o) {
    o = o || {};
    const base = o.width || 1;
    for (const e of edges) {
      const a = at(e.source), b = at(e.target);
      if (!a || !b) continue;
      const closeness = e.d == null ? 0.5 : Math.max(0, 1 - e.d / 1.4);
      // `relOf` lets the caller decide an edge's kind from context rather than storing it.
      // The Walk needs it: an edge is a *deck* edge when both endpoints came from a loaded
      // decklist, which is a property of the graph, not of the link.
      const rel = o.relOf ? o.relOf(e) : e.rel;
      ctx.lineWidth = ((o.weightOf ? o.weightOf(e, rel) : e.weight) || base) / k;
      ctx.strokeStyle = e.ink || edgeInk(rel, closeness);
      ctx.beginPath();
      if (o.curve) {
        // A straight line between two far-apart cards reads as an assertion about the
        // space between them. A shallow arc reads as a connection. Used on the atlas,
        // where a relation can span a quarter of the map.
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const nx = -(b[1] - a[1]), ny = b[0] - a[0];
        const len = Math.hypot(nx, ny) || 1;
        const bow = Math.hypot(b[0] - a[0], b[1] - a[1]) * (o.curve === true ? 0.12 : o.curve);
        ctx.moveTo(a[0], a[1]);
        ctx.quadraticCurveTo(mx + (nx / len) * bow, my + (ny / len) * bow, b[0], b[1]);
      } else {
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
    }
    ctx.lineWidth = base / k;
  }

  return {
    surface: surface,
    camera: camera,
    placer: placer,
    placeLabels: placeLabels,
    edgeInk: edgeInk,
    drawEdges: drawEdges,
    INK: INK,
  };
})();
