/* render/canvas.js — the map, drawn by us instead of by Plotly.
 *
 * Phase 2 of the migration (see the plan). The Walk proved the pieces on 500 nodes;
 * this is the same machinery pointed at 34,322. Measured on this data before it was
 * written, because the whole decision rested on the numbers:
 *
 *   draw 34,322 points, one path per colour group   7.8 ms   (128 fps)
 *   draw 34,322 points, one arc call each          16.9 ms   (59 fps)
 *   quadtree build, once at load                   23.5 ms
 *   hover pick                                     ~0 ms
 *   box-select over 22,161 caught points            4.5 ms   (Plotly: 138 ms)
 *
 * THE LAYER FORMAT IS THE TRACE FORMAT. A layer is a Plotly-shaped object — `{x, y,
 * customdata, name, visible, mode, marker: {size, color, opacity, symbol, line}}` — so
 * `render()` builds one structure and hands it to whichever renderer is active. There is
 * no adapter to write now and delete later, and no second source of truth for what a
 * layer is.
 *
 * Supported: markers and lines, per-point colour AND opacity arrays (the deck builder's
 * dimming), scalar opacity, the four symbols in use (circle, diamond, star, square),
 * marker outlines, `visible: false`, and density contours via d3-contourDensity. Plotly's
 * `histogram2dcontour` layer is dropped on sight — the canvas computes its own density
 * rather than being handed pre-binned data.
 *
 * Coordinates. `world` is data space (projection_2d units, roughly ±60). `screen` is CSS
 * pixels. `transform` is the d3-zoom transform between them, and it is the ONLY thing that
 * moves — points are never mutated, unlike The Walk where the simulation owns positions.
 */
(function () {
  'use strict';

  const SYMBOL = { circle: 0, square: 1, diamond: 2, star: 3 };

  function create() {
    let host = null, canvas = null, ctx = null;
    let surface = null, cam = null;
    let layers = [];
    let transform = null;
    let zoomBehaviour = null;
    let tree = null;                 // d3.quadtree over every pickable point
    let rowPos = new Map();          // row id → its stored {x, y}, rebuilt with the tree
    let baseFit = null;              // world→screen fit computed once from the data extent
    const handlers = { click: [], hover: [], unhover: [], camera: [], select: [] };
    let raf = null;
    let selectMode = false;          // shift held: drag draws a marquee instead of panning
    let marquee = null;              // {x0, y0, x1, y1} in screen px, while dragging
    let labelHost = null;
    let lastVisibleLabels = 0;            // DOM layer for region labels
    let labels = [];                 // [{x, y, text, size, colour, id}]
    let contour = null;              // cached d3-contourDensity paths
    let contourKey = null;
    let showContours = false;

    function emit(name, arg) { for (const fn of handlers[name] || []) fn(arg); }

    // ── Setup ────────────────────────────────────────────────────────────

    function init(hostEl) {
      host = hostEl;
      // Surface and camera come from Stage — the canvas element, the devicePixelRatio
      // resize and the d3-zoom wiring were identical to The Walk's, down to the comment
      // about <canvas> being a replaced element.
      surface = Stage.surface(hostEl, 'map-canvas');
      canvas = surface.canvas;
      ctx = surface.ctx;

      cam = Stage.camera({
        canvas: canvas,
        scaleExtent: [0.2, 400],
        onZoom: function (t) { transform = t; schedule(); emit('camera', getCamera()); },
        // Shift takes the drag away from the zoom behaviour so it can draw a marquee.
        filter: function (ev) {
          if (selectMode && (ev.type === 'mousedown' || ev.type === 'touchstart')) return false;
          return !ev.ctrlKey && !ev.button;
        },
      });
      zoomBehaviour = cam.behaviour;
      transform = cam.transform;

      canvas.addEventListener('mousedown', function (e) {
        if (!selectMode) return;
        marquee = { x0: e.offsetX, y0: e.offsetY, x1: e.offsetX, y1: e.offsetY };
        e.preventDefault();
      });
      window.addEventListener('mousemove', function (e) {
        if (!marquee || !canvas) return;
        const r = canvas.getBoundingClientRect();
        marquee.x1 = e.clientX - r.left;
        marquee.y1 = e.clientY - r.top;
        schedule();
      });
      window.addEventListener('mouseup', function () {
        if (!marquee) return;
        const m = marquee;
        marquee = null;
        schedule();
        // A click, not a drag. Let the click handler own it.
        if (Math.abs(m.x1 - m.x0) < 4 && Math.abs(m.y1 - m.y0) < 4) return;
        // THE 138 ms operation, measured at 4.5 ms here over 22,161 caught points.
        emit('select', { rows: pickRect(m.x0, m.y0, m.x1, m.y1) });
      });

      canvas.addEventListener('mousemove', function (e) {
        const hit = pick(e.offsetX, e.offsetY);
        setHover(hit);
        // The cursor is the cheapest affordance on the surface and the map had none: 34,322
        // points that all look equally inert until you happen to click one. `grab` says the
        // canvas pans, which is true and is not the interesting half.
        if (!selectMode) canvas.style.cursor = hit == null ? 'grab' : 'pointer';
        if (hit == null) emit('unhover', null);
        else emit('hover', { row: hit, clientX: e.clientX, clientY: e.clientY });
      });
      canvas.addEventListener('mouseleave', function () { setHover(null); emit('unhover', null); });
      canvas.addEventListener('click', function (e) {
        const hit = pick(e.offsetX, e.offsetY);
        if (hit != null) {
          const p = rowPos.get(hit);
          if (p) { ripples.push({ x: p.x, y: p.y, t: performance.now() }); startTicking(); }
          emit('click', { row: hit, shiftKey: e.shiftKey, event: e });
        }
      });

      // rAF does not fire in a hidden tab, so the ticker stops on its own — but nothing
      // restarts it, and the map would come back frozen. `clock` only advances while
      // frames run, so it resumes in phase rather than snapping forward by however long
      // the tab was in the background.
      document.addEventListener('visibilitychange', function () {
        if (!document.hidden) startTicking();
      });

      resize();
      return api;
    }

    // The DPR-correct resize lives in Stage now; the "a <canvas> is a replaced element,
    // so an inline width stops it following the layout" lesson is recorded there.
    function resize() {
      if (surface) surface.resize(draw);
    }

    function destroy() {
      ticking = false;               // or the loop keeps calling draw() against a null ctx
      hoverRow = null;
      ripples = [];
      if (surface) surface.destroy();
      surface = cam = canvas = ctx = host = tree = null;
      layers = [];
    }

    // ── Layers ───────────────────────────────────────────────────────────

    function setLayers(next) {
      layers = (next || []).filter(function (l) {
        if (l && l.type === 'histogram2dcontour') {
          // Topo. Moves to d3-contour in Phase 3; dropping it loudly beats drawing
          // something that is not a density estimate.
          if (!setLayers._warnedContour) {
            console.warn('[canvas] contours not implemented yet — Topo is Plotly-only');
            setLayers._warnedContour = true;
          }
          return false;
        }
        return !!l;
      });
      if (!baseFit) fitToData();
      buildTree();
      ensureContours();
      // Draw synchronously, not via rAF. setLayers is a discrete state change — a filter
      // toggled, a search typed — and the caller wants the result now. Only pan/zoom
      // coalesces through schedule(), because that fires many events per gesture.
      //
      // It also matters for correctness, not just latency: rAF does not fire in a hidden
      // tab, so an rAF-only draw leaves the canvas blank until the tab is focused, and
      // leaves the browser tests unable to see anything at all.
      draw();
    }

    // One fit, from the widest layer, so the map opens showing everything. After that the
    // camera is the user's and nothing recomputes it — the equivalent of Plotly autorange
    // happening exactly once, which is the behaviour `render()` had to fake with keepX/keepY.
    function fitToData() {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, n = 0;
      for (const l of layers) {
        if (!l.x) continue;
        for (let i = 0; i < l.x.length; i++) {
          const x = l.x[i], y = l.y[i];
          if (x == null || y == null) continue;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          n++;
        }
      }
      if (!n || !isFinite(minX)) return;
      baseFit = { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
      const w = canvas.clientWidth || 1000, h = canvas.clientHeight || 700;
      // One scale for both axes: the projection is isotropic, which is what Plotly's
      // `scaleanchor: 'y'` was buying. Here it is free.
      const k = Math.min(w / (maxX - minX || 1), h / (maxY - minY || 1)) * 0.92;
      baseFit.k = k;
      baseFit.tx = w / 2 - k * (minX + maxX) / 2;
      baseFit.ty = h / 2 - k * (minY + maxY) / 2;
      // The pivot the ambient swirl turns about, and the radius that normalises it. Derived
      // from tx/k rather than assumed to be the viewport centre — they coincide at the fit
      // and stop coinciding the moment `resize` changes the box under a live camera.
      baseFit.cx = baseFit.tx + k * (minX + maxX) / 2;
      baseFit.cy = baseFit.ty + k * (minY + maxY) / 2;
      baseFit.rMax = Math.hypot(k * (maxX - minX), k * (maxY - minY)) / 2 || 1;
    }

    // The STATIC base-fit mapping: data units → fitted screen space, no time term. This is
    // what the whole file used to mean by "world→screen". Anything that must be stable
    // across frames — the cached density field, the camera target for a region, the span
    // the label crossfade keys on — uses these two and not `proj` below.
    function wx(x) { return baseFit.tx + baseFit.k * x; }
    function wy(y) { return baseFit.ty + baseFit.k * y; }

    // ── Ambient motion: the galaxy layer ─────────────────────────────────
    //
    // WHY THIS LIVES IN THE PROJECTION AND NOT IN THE DATA. The obvious way to make the
    // atlas drift is to move the points. It is also the one way that cannot work here:
    // `buildTree` costs 23.5 ms and its signature is deliberately blind to positions, so
    // per-frame mutation either rebuilds the quadtree 30 times a second or leaves it
    // answering with where the cards used to be — the exact failure `reindex()` exists to
    // document. Everything anchored to world coordinates (region labels, the search
    // highlight, the selection ring, drill's local layout) would detach for the same
    // reason.
    //
    // So the points never move. `proj` is `wx`/`wy` plus a time term, and every consumer
    // already goes through that one function. The cost is that `pick` has to invert it —
    // which is why the motion is built to BE invertible.
    //
    // WHY IT IS A SWAY AND NOT A ROTATION. A galaxy that actually rotates winds up: give
    // inner orbits a shorter period and after a few minutes the arms have wrapped and
    // cards that PaCMAP placed side by side are a quarter of the map apart. That is the
    // winding problem, and on a map whose entire content is "what is near what" it is not
    // a cosmetic issue — the picture would be lying. A rigid rotation avoids the shear but
    // costs the other thing: spatial memory. Turn the whole atlas 90° and the region you
    // learned was north is now west.
    //
    // The motion here is therefore BOUNDED — a differential sway, a couple of degrees
    // either side of home, that always returns. Kepler survives where it is legible: the
    // PERIOD varies with radius (T ∝ a^1.5, so the rim is slower than the core) and phase
    // lags outward, so at any instant the field is a shallow spiral, slowly unwinding and
    // rewinding. That reads as orbital motion. It just never accumulates.
    //
    // WHERE IT APPLIES. The same argument the halo already makes, in the same units:
    // atmosphere belongs at altitude, where a card is one pixel and only the shape of the
    // cloud is legible. Zoomed in, Explore has to converge on Discover and Build — plain
    // crisp dots, and a card you are trying to click must not be moving. `ambient()` is
    // gone by the time a region fills the screen, so points slide home as you approach.
    const MOTION = {
      AMP: 0.145,        // rad — peak swirl angle before the radial falloff
      SOFT: 0.5,         // A(u) = AMP · SOFT/(u+SOFT): outer arcs travel further in pixels
      PERIOD: 27000,     // ms — the reference orbital period
      KEP_U0: 0.35,      // softening, or the core would spin infinitely fast at u→0
      KEP_EXP: 1.5,      // Kepler's third law: T ∝ a^(3/2)
      PHASE: 2.4,        // rad of phase lag per unit radius — this is what makes it a spiral
      DRIFT: 0.011,      // bounded Lissajous drift of the whole field, as a fraction of extent
      DRIFT_PX: 41000, DRIFT_PY: 57000,   // deliberately not commensurate: it never loops
      BREATHE: 0.12, BREATHE_MS: 31000,   // the halo inhales; one scalar, not 34,322
      K1: 6,             // gone by a region — the same altitude the aura fades at
      FRAME_MS: 32,      // ~30 fps while something is under the cursor
      IDLE_MS: 50,       // ~20 fps for the ambient sway alone — see the pacing note in tick()
      BINS: 96,          // radial bins for the rotation table (see below)
      GAIN_MS: 700,      // how long motion takes to arrive or leave when toggled
    };

    // Reduced motion is a system-level request not to animate, and an ambient drift is
    // exactly what it is about. Honoured as the DEFAULT rather than a hard override, so
    // `setMotion(true)` still works for anyone who asks for it explicitly.
    const reduceMotion = typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let motionEnabled = !reduceMotion;
    let clock = 0;                   // accumulated ANIMATED ms — not wall time, so pausing
    let lastTick = 0;                // in a hidden tab resumes in phase instead of jumping
    let gain = 0, gainTarget = motionEnabled ? 1 : 0;
    let ticking = false, lastDraw = 0;
    let hoverRow = null, hoverAt = 0;
    let ripples = [];                // {x, y, t} in base-fit space

    // Per-point trigonometry would be 34,322 sin/cos per frame. It is not needed: the
    // swirl angle is a function of RADIUS alone, so one table of (cos, sin) per radial bin
    // is built once a frame and every point does a sqrt and a lookup.
    const rotC = new Float32Array(MOTION.BINS), rotS = new Float32Array(MOTION.BINS);
    let swirlLive = false, driftX = 0, driftY = 0;
    // The largest distance any point can currently be drawn from where it is stored. Box
    // select needs it to prune conservatively; it is a by-product of building the table.
    let maxShift = 0;

    // 1 at the whole-map fit, 0 once a region fills the screen — the inverse ramp the aura
    // uses, times the eased on/off gain.
    function ambient() {
      if (gain <= 0.001 || !baseFit) return 0;
      return gain * (1 - ramp(transform.k, 1.0, MOTION.K1));
    }

    function updateMotion() {
      const a = ambient();
      swirlLive = a > 0.002 && !!baseFit && baseFit.rMax > 0;
      if (!swirlLive) { driftX = driftY = 0; maxShift = 0; return; }
      const TAU = Math.PI * 2;
      let worst = 0;
      for (let i = 0; i < MOTION.BINS; i++) {
        const u = i / (MOTION.BINS - 1);
        const period = MOTION.PERIOD * Math.pow(u + MOTION.KEP_U0, MOTION.KEP_EXP);
        const amp = a * MOTION.AMP * (MOTION.SOFT / (u + MOTION.SOFT));
        const th = amp * Math.sin(TAU * clock / period + MOTION.PHASE * u);
        rotC[i] = Math.cos(th);
        rotS[i] = Math.sin(th);
        // chord length for a rotation of th at this radius
        const d = 2 * u * baseFit.rMax * Math.abs(Math.sin(th / 2));
        if (d > worst) worst = d;
      }
      const ext = baseFit.rMax;
      driftX = a * MOTION.DRIFT * ext * Math.sin(TAU * clock / MOTION.DRIFT_PX);
      driftY = a * MOTION.DRIFT * ext * Math.sin(TAU * clock / MOTION.DRIFT_PY + 1.1);
      maxShift = worst + Math.hypot(driftX, driftY);
    }

    // Scratch registers rather than a returned pair: this runs 34,322 times a frame, and a
    // two-element array each time is a million short-lived objects a second.
    let PX = 0, PY = 0;

    function proj(x, y) {
      const sx = baseFit.tx + baseFit.k * x, sy = baseFit.ty + baseFit.k * y;
      if (!swirlLive) { PX = sx; PY = sy; return; }
      const dx = sx - baseFit.cx, dy = sy - baseFit.cy;
      let i = (Math.sqrt(dx * dx + dy * dy) / baseFit.rMax) * (MOTION.BINS - 1);
      i = i > MOTION.BINS - 1 ? MOTION.BINS - 1 : (i > 0 ? i | 0 : 0);
      const c = rotC[i], s = rotS[i];
      PX = baseFit.cx + dx * c - dy * s + driftX;
      PY = baseFit.cy + dx * s + dy * c + driftY;
    }

    function projPt(x, y) { proj(x, y); return [PX, PY]; }

    // Hovering is a state, not an event: the ring animates, so entering a card has to start
    // the clock and re-entering the SAME card must not restart it (a mousemove fires many
    // times over one point, and a ring that restarts its grow-in on every pixel of travel
    // flickers instead of settling).
    function setHover(row) {
      if (row === hoverRow) return;
      hoverRow = row;
      hoverAt = performance.now();
      if (row != null) startTicking(); else schedule();
    }

    /* The exact inverse, and it exists BECAUSE the swirl is a rotation about a fixed
     * centre: rotation preserves radius, so the bin a point landed in is recoverable from
     * where it landed. Un-drift first (drift is applied last going forward), read the bin
     * off the rotated radius, rotate back by the same angle. No search, no approximation —
     * which is what lets hover and click keep hitting the card under the cursor while the
     * field is moving. A displacement-based fudge here would be wrong exactly where the
     * map is densest. */
    function unproj(sx, sy) {
      let px = sx, py = sy;
      if (swirlLive) {
        px -= driftX; py -= driftY;
        const dx = px - baseFit.cx, dy = py - baseFit.cy;
        let i = (Math.sqrt(dx * dx + dy * dy) / baseFit.rMax) * (MOTION.BINS - 1);
        i = i > MOTION.BINS - 1 ? MOTION.BINS - 1 : (i > 0 ? i | 0 : 0);
        const c = rotC[i], s = -rotS[i];
        px = baseFit.cx + dx * c - dy * s;
        py = baseFit.cy + dx * s + dy * c;
      }
      return [(px - baseFit.tx) / baseFit.k, (py - baseFit.ty) / baseFit.k];
    }

    // Is there anything to animate? A continuous rAF over 34,322 points is not free, so it
    // runs only when it is both wanted and visible: `offsetParent` is null in the graph
    // modes (`#plot.force-mode` hides this canvas) and rAF does not fire in a hidden tab
    // anyway, so both are cheap outs rather than optimisations.
    function wantsFrames() {
      if (!canvas || !baseFit || document.hidden) return false;
      // The CANVAS, not the host. `#plot` is shared: it hosts the force graph too and stays
      // visible in Discover and Build, where `#plot.force-mode .map-canvas {display:none}`
      // hides only this surface. Asking the host meant the ticker happily animated 34,890
      // points into a canvas nobody could see, in the two modes that do not use it.
      if (canvas.offsetParent === null) return false;
      if (gain !== gainTarget) return true;
      return (gainTarget > 0 && ambient() > 0.002) || hoverRow != null || ripples.length > 0;
    }

    function startTicking() {
      if (ticking || !wantsFrames()) return;
      ticking = true;
      lastTick = performance.now();
      requestAnimationFrame(tick);
    }

    function tick(now) {
      if (!ticking) return;
      /* Clamped so a backgrounded tab cannot fast-forward the sway by however many minutes
       * it was away — but clamped LOOSELY. At 64 ms this doubles as a frame-rate governor:
       * anything rendering slower than 15 fps (a headless browser throttles rAF to about
       * 4) advances `clock` at a fraction of wall time, so the animation silently runs in
       * slow motion on exactly the machines that are already struggling. 200 ms still
       * bounds the return-from-hidden jump to something invisible. */
      const dt = Math.min(200, now - lastTick);
      lastTick = now;
      if (gain !== gainTarget) {
        const step = dt / MOTION.GAIN_MS;
        gain = gainTarget > gain ? Math.min(gainTarget, gain + step)
                                 : Math.max(gainTarget, gain - step);
      }
      if (gain > 0) clock += dt;
      ripples = ripples.filter(function (r) { return now - r.t < 620; });
      if (!wantsFrames()) { ticking = false; draw(); return; }
      /* Frame pacing rather than every rAF, and the budget is spent where it shows. A full
       * draw is ~9.9 ms over 34,890 points, so this loop is not free and should not pretend
       * to be. The ambient sway travels 1–3 px per SECOND: at 20 fps that is a tenth of a
       * pixel between frames, which is not a thing an eye can resolve, so paying 60 fps for
       * it buys nothing. The ring and the ripple do move fast enough to judder — they get
       * the higher rate, for the fraction of a second they exist. */
      const budget = (hoverRow != null || ripples.length) ? MOTION.FRAME_MS : MOTION.IDLE_MS;
      if (now - lastDraw >= budget) draw();
      requestAnimationFrame(tick);
    }

    // Rebuilding the quadtree costs 23.5 ms at 34,322 points, and setLayers runs on every
    // filter, search keystroke and panel change — most of which do not alter the pickable
    // set at all. Key on what the tree actually depends on and skip the rest.
    let treeKey = null;

    // The signature is deliberately cheap — layer lengths and endpoint ids — because a
    // rebuild costs 23.5 ms and `setLayers` runs on every filter and search keystroke.
    // The cost of cheap is that it cannot see POSITIONS move: drill mutates coordinates
    // in place through `updateLayerBy` for 90 frames while every one of these fields
    // stays identical, so the tree kept answering with where the cards used to be and the
    // map stopped hit-testing. `reindex()` is the explicit "positions moved" signal, for
    // callers that know they did it. Called once at settle, never per frame.
    function reindex() { treeKey = null; buildTree(); }

    function treeSignature() {
      let sig = '';
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines' || l.mode === 'edges') continue;
        const cd = l.customdata;
        sig += l.x.length + ':' + cd[0] + ':' + cd[cd.length - 1] + '|';
      }
      return sig;
    }

    function buildTree() {
      const sig = treeSignature();
      if (sig === treeKey && tree) return;
      treeKey = sig;
      const pts = [];
      // row → stored position, for the things that know a card by id and have to draw AT
      // it: the hover ring and the click ripple. The tree can answer "what is near here"
      // and not "where is row 8,412", and a linear scan of 34,322 per frame to find out is
      // the kind of thing that only shows up on someone else's laptop.
      rowPos = new Map();
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines' || l.mode === 'edges') continue;
        for (let i = 0; i < l.x.length; i++) {
          if (l.x[i] == null) continue;
          const p = { x: l.x[i], y: l.y[i], row: l.customdata[i] };
          pts.push(p);
          rowPos.set(p.row, p);
        }
      }
      tree = d3.quadtree().x(function (d) { return d.x; }).y(function (d) { return d.y; }).addAll(pts);
    }

    // ── Drawing ──────────────────────────────────────────────────────────

    // contourDensity is the heaviest thing here, so it recomputes only when the drawn
    // point set actually changes — the same signature the quadtree uses.
    function ensureContours() {
      if (!showContours) return;
      const sig = treeSignature();
      if (sig === contourKey && contour) return;
      contourKey = sig;
      rebuildContours();
    }

    function schedule() {
      if (raf !== null) return;
      raf = requestAnimationFrame(function () { raf = null; draw(); });
    }

    function draw() {
      if (!ctx || !canvas || !baseFit) return;
      lastDraw = performance.now();
      // The time term is resolved ONCE per frame and every projection in the frame reads
      // the same table. Sampling the clock per layer would shear the overlays against the
      // base scatter by however long the frame took to draw.
      updateMotion();
      const w = surface.width, h = surface.height;
      // Shared prologue — clear, save, translate, scale — and its matching close.
      const close = surface.open(transform);

      if (showContours && contour) drawContours();

      for (const l of layers) {
        if (l.visible === false || !l.x || !l.x.length) continue;
        if (l.mode === 'edges') drawTypedEdges(l);
        else if (l.mode === 'lines') drawLines(l);
        else drawMarkers(l);
      }
      drawHoverRing();
      drawRipples();
      close();
      drawFalloff(w, h);

      if (marquee) drawMarquee();
      positionLabels();
      // A draw can be what makes motion wanted again — the first paint after boot, a mode
      // switch back to Explore, a hover. Cheaper to ask here than to remember every caller.
      startTicking();
    }

    // Screen space, outside the transform — a selection rectangle is a gesture, not a
    // thing in the data.
    function drawMarquee() {
      const x = Math.min(marquee.x0, marquee.x1), y = Math.min(marquee.y0, marquee.y1);
      const w = Math.abs(marquee.x1 - marquee.x0), h = Math.abs(marquee.y1 - marquee.y0);
      ctx.fillStyle = 'rgba(196,167,71,0.10)';
      ctx.strokeStyle = '#c4a747';
      ctx.lineWidth = 1;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }

    // Density, replacing Plotly's histogram2dcontour. Computed in base-fit space and drawn
    // inside the transform, so it zooms with the points instead of re-binning per frame —
    // Plotly's version auto-binned to whatever extent it was handed, which is why its
    // contour levels were never comparable between filters.
    function rebuildContours() {
      const pts = [];
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines' || l.mode === 'edges') continue;
        for (let i = 0; i < l.x.length; i++) if (l.x[i] != null) pts.push(l);
        break;                       // the base scatter only; overlays are not density
      }
      const src = [];
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines' || l.mode === 'edges') continue;
        for (let i = 0; i < l.x.length; i++) {
          if (l.x[i] != null) src.push([wx(l.x[i]), wy(l.y[i])]);
        }
      }
      if (!src.length) { contour = null; return; }
      const w = canvas.clientWidth, h = canvas.clientHeight;
      contour = d3.contourDensity()
        .x(function (d) { return d[0]; })
        .y(function (d) { return d[1]; })
        .size([Math.ceil(w), Math.ceil(h)])
        .bandwidth(14)
        .thresholds(14)(src);
    }

    function drawContours() {
      const path = d3.geoPath(null, ctx);
      const max = contour.length ? contour[contour.length - 1].value : 1;
      // The density field is computed once and cached against the point-set signature, so
      // it cannot follow a per-radius swirl — but it must not visibly detach from the
      // points it describes either. It is carried by ONE rigid rotation taken at the bulk
      // radius, which tracks the mass of the field; the residual at the extreme radii is
      // bounded by the swirl amplitude, i.e. a few pixels at the fit, on a field whose
      // kernel is 14 px wide. Recomputing contourDensity per frame is not on the table.
      ctx.save();
      if (swirlLive) {
        const i = Math.round(0.55 * (MOTION.BINS - 1));
        ctx.translate(baseFit.cx + driftX, baseFit.cy + driftY);
        ctx.rotate(Math.atan2(rotS[i], rotC[i]));
        ctx.translate(-baseFit.cx, -baseFit.cy);
      }
      for (const c of contour) {
        const a = Math.min(0.34, 0.05 + 0.30 * (c.value / (max || 1)));
        ctx.fillStyle = 'rgba(120,100,200,' + a.toFixed(3) + ')';
        ctx.beginPath();
        path(c);
        ctx.fill();
      }
      ctx.restore();
    }

    // Typed edges: a relation drawn between two cards, coloured by what the relation IS.
    //
    // This is not the same thing as a `lines` layer, and the difference is the point. A
    // `lines` layer is one flattened polyline with a single colour for the whole layer —
    // enough for the Deck Lens's verified-line edges, and structurally unable to say that
    // THIS edge is a synergy and THAT one is an obsolescence. An `edges` layer carries
    // `[{source: [x, y], target: [x, y], rel, reason, d}]` and hands the inks to Stage, so
    // an edge means the same thing here as it does on the graph.
    //
    // Coordinates are explicit rather than row indices: the renderer still knows nothing
    // about cards, and the producer already has the positions.
    function drawTypedEdges(l) {
      if (!l.edges || !l.edges.length) return;
      ctx.globalAlpha = l.opacity == null ? 1 : l.opacity;
      Stage.drawEdges(ctx, l.edges, function (pt) { return projPt(pt[0], pt[1]); },
                      transform.k, { width: (l.line && l.line.width) || 1, curve: l.curve });
      ctx.globalAlpha = 1;
    }

    function drawLines(l) {
      const line = l.line || {};
      ctx.strokeStyle = line.color || '#888';
      ctx.lineWidth = (line.width || 1) / transform.k;
      ctx.globalAlpha = l.opacity == null ? 1 : l.opacity;
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < l.x.length; i++) {
        if (l.x[i] == null) { pen = false; continue; }   // null separators break segments
        proj(l.x[i], l.y[i]);
        if (pen) ctx.lineTo(PX, PY); else { ctx.moveTo(PX, PY); pen = true; }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Batched: one path per colour, filled once. Measured 7.8 ms for 34,322 points versus
    // 16.9 ms issuing a fill per point — the difference between 128 fps and 59.
    /* TWO ramps, pulling opposite ways, because zooming in changes two things.
     *
     * Points draw at a constant SCREEN size, so zooming spreads them apart without making
     * any of them brighter — a dense field reads as grey haze far out and as sparse grey
     * dots up close, which is the same dimness twice. `closeness` fixes that: the further
     * in you are, the fewer cards are on screen and the more each can assert itself.
     *
     * `aura` runs the OTHER way, and that is the correction. Zoomed in, this view has to
     * converge on Discover and Build — one force engine, drawing plain crisp dots with no
     * halo at all (grep force.js for shadowBlur: there is none). A halo that grew as you
     * approached made close-up Explore look like a different application from the two
     * modes it hands off to, and it sat on top of exactly the points you were trying to
     * read. Atmosphere belongs at altitude: far out, where a single card is a pixel and
     * the shape of the cloud is the only readable thing.
     *
     * Both are in the units `transform.k` actually uses: this zoom is RELATIVE TO
     * `baseFit`, so k=1 is the whole-map fit, not an absolute data→pixel scale (see
     * `pick`, which multiplies the two back together). Measured on the atlas: fit k=1, a
     * region fills the screen near k=3, a neighbourhood near k=15, a street near k=45.
     * Getting the units wrong is silent — constants in absolute scale (25/240) leave the
     * ramp flat at 0 everywhere, and on/off then measures pixel-identical, which reads as
     * "the halo does nothing" rather than "the halo never ran".
     */
    const CLOSE_K0 = 1.6, CLOSE_K1 = 15;   // brightness: none at the fit, full by a neighbourhood
    const AURA_K1 = 6;                     // atmosphere: full at the fit, gone by a region

    function ramp(k, k0, k1) {
      if (k <= k0) return 0;
      if (k >= k1) return 1;
      return Math.log(k / k0) / Math.log(k1 / k0);
    }

    // 0 at the whole-map fit, 1 once a neighbourhood fills the screen.
    function closeness() { return ramp(transform.k, CLOSE_K0, CLOSE_K1); }

    /* The inverse, and capped LOW — measured, because "some aura" and "a wash" are only a
     * factor of two apart. Alpha coverage of the canvas at the fitted view, against a
     * 4.5% no-aura baseline: cap 0.55 / radius 3.2x gave 68.4%, i.e. two thirds of the
     * screen carrying ink and the clusters lost inside it; 0.35/2.4x gave 50.9%;
     * 0.25/1.9x gives 35.9%, where each cluster reads as a lit island and the space
     * between them stays black. Full-strength ink is 4.2% in every case — this only ever
     * moves the halo, never the cards. */
    /* The breathe rides on top and is deliberately ONE SCALAR: the halo already pools
     * additively where the field is dense, so modulating its strength makes the whole cloud
     * inhale without touching a single per-point value. A per-card twinkle would mean
     * rebuilding 34,322 alphas every frame on exactly the array path `dimsAll()` exists to
     * avoid. Centred on 1.0, so the measured 0.25 cap is still the mean and the coverage
     * figures above still describe the average frame. */
    function auraLevel() {
      const base = 0.25 * (1 - ramp(transform.k, 1.0, AURA_K1));
      if (gain <= 0.001) return base;
      const breathe = 1 + gain * MOTION.BREATHE * Math.sin(Math.PI * 2 * clock / MOTION.BREATHE_MS);
      return base * breathe;
    }

    /* Distance falloff — the other half of "brighter the closer, fading further out".
     *
     * The halo above makes everything brighter at once; this puts the brightness
     * somewhere, by letting the field fall off toward the edges of the viewport so what
     * you have centred reads as the thing you are looking at.
     *
     * It is drawn in SCREEN space as one radial gradient rather than as a per-point
     * alpha, and that is the whole reason it is affordable: a distance-from-centre ramp
     * computed per point is 34,322 distances plus a rebuilt colour bucket every frame,
     * on the same array path `dimsAll()` exists to avoid. One gradient fill is O(1) and
     * says the same thing.
     *
     * Tied to the same boost, so at the whole-map fit it does not draw at all — a
     * vignette over the entire atlas would just be a dimmer atlas.
     */
    function drawFalloff(w, h) {
      // Rides the aura, not the closeness: the vignette is atmosphere too, and Discover
      // and Build have none. Zoomed in it must be gone.
      const boost = auraLevel();
      if (boost < 0.02) return;
      const cx = w / 2, cy = h / 2;
      const inner = Math.min(w, h) * 0.28;
      const outer = Math.hypot(w, h) / 2;
      const g = ctx.createRadialGradient(cx, cy, inner, cx, cy, outer);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(0.55, 'rgba(0,0,0,' + (boost * 0.16).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(0,0,0,' + (boost * 0.46).toFixed(3) + ')');
      ctx.save();
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
      ctx.restore();
    }

    function drawMarkers(l) {
      const m = l.marker || {};
      const near = closeness();
      const aura = m.glow ? auraLevel() : 0;
      // Up to +60% radius and a brighter core as you close in — this half is unchanged,
      // and is what "brighter the closer" asked for.
      const size = (m.size == null ? 3 : m.size) * (1 + near * 0.6) / transform.k;
      const lift = (a) => Math.min(1, a * (1 + near * 0.55));
      const sym = SYMBOL[m.symbol] == null ? 0 : SYMBOL[m.symbol];
      const perColour = Array.isArray(m.color);
      const perAlpha = Array.isArray(m.opacity);

      if (!perColour && !perAlpha) {
        const base = m.opacity == null ? 1 : m.opacity;
        const idx = new Array(l.x.length);
        for (let i = 0; i < l.x.length; i++) idx[i] = i;
        // The halo, drawn first and underneath: same points, wider and faint. Additive
        // blending makes overlapping haloes pool where the space is dense, so a cluster
        // glows as a region rather than as a heap of separate dots.
        if (aura > 0.02) {
          ctx.save();
          ctx.globalCompositeOperation = 'lighter';
          ctx.globalAlpha = base * aura * 0.20;
          strokeFill(l, idx, m.color || '#666', size * 1.9, sym, m);
          ctx.restore();
        }
        ctx.globalAlpha = lift(base);
        strokeFill(l, idx, m.color || '#666', size, sym, m);
        ctx.globalAlpha = 1;
        return;
      }

      // Batch on (colour, opacity). Both are per-point at most a handful of distinct
      // values — the deck builder dims to exactly two, the palettes have 6–20 colours —
      // so this stays a few fills, never 34,322.
      const buckets = new Map();
      for (let i = 0; i < l.x.length; i++) {
        const c = perColour ? (m.color[i] || '#666') : (m.color || '#666');
        const a = perAlpha ? m.opacity[i] : (m.opacity == null ? 1 : m.opacity);
        const key = c + '@' + a;
        let b = buckets.get(key);
        if (!b) { b = { c: c, a: a, idx: [] }; buckets.set(key, b); }
        b.idx.push(i);
      }
      buckets.forEach(function (b) {
        if (aura > 0.02 && b.a > 0.2) {
          ctx.save();
          ctx.globalCompositeOperation = 'lighter';
          ctx.globalAlpha = b.a * aura * 0.20;
          strokeFill(l, b.idx, b.c, size * 1.9, sym, m);
          ctx.restore();
        }
        ctx.globalAlpha = lift(b.a);
        strokeFill(l, b.idx, b.c, size, sym, m);
      });
      ctx.globalAlpha = 1;
    }

    /* ── Touch: what the map does back ───────────────────────────────────
     *
     * Both of these are drawn INSIDE the world transform at a radius divided by
     * `transform.k`, so they hold a constant size on screen at every zoom, and both are
     * positioned through `proj` — a ring that did not carry the ambient term would sit
     * beside the card it is meant to be pointing at.
     *
     * They are also the only two things in this renderer that animate on their own, which
     * is why `wantsFrames` counts them: a hover or a click keeps the ticker alive for as
     * long as the animation lasts and no longer.
     */
    function drawHoverRing() {
      if (hoverRow == null) return;
      const p = rowPos.get(hoverRow);
      // The hovered card can stop being drawn under you — a filter toggle rebuilds the
      // pickable set. Forget it rather than just skipping the ring, or `wantsFrames` keeps
      // reporting a live hover and the ticker never stands down.
      if (!p) { hoverRow = null; return; }
      const age = performance.now() - hoverAt;
      const grow = 1 - Math.exp(-age / 90);          // arrives quickly, settles
      const pulse = 0.5 + 0.5 * Math.sin(age / 260); // and then breathes
      const r = (7 + 2.2 * pulse) * grow / transform.k;
      proj(p.x, p.y);
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = 'rgba(196,167,71,' + (0.75 * grow).toFixed(3) + ')';
      ctx.lineWidth = 1.6 / transform.k;
      ctx.beginPath();
      ctx.arc(PX, PY, r, 0, Math.PI * 2);
      ctx.stroke();
      // A second, fainter ring further out: one circle reads as a cursor, two read as a
      // thing being lit up.
      ctx.strokeStyle = 'rgba(196,167,71,' + (0.20 * grow * pulse).toFixed(3) + ')';
      ctx.lineWidth = 1 / transform.k;
      ctx.beginPath();
      ctx.arc(PX, PY, r * 2.1, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    function drawRipples() {
      if (!ripples.length) return;
      const now = performance.now();
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const rp of ripples) {
        const t = (now - rp.t) / 620;
        if (t < 0 || t > 1) continue;
        const ease = 1 - Math.pow(1 - t, 3);         // fast out, slow settle
        proj(rp.x, rp.y);
        ctx.strokeStyle = 'rgba(196,167,71,' + (0.55 * (1 - t)).toFixed(3) + ')';
        ctx.lineWidth = 1.8 * (1 - t) / transform.k;
        ctx.beginPath();
        ctx.arc(PX, PY, (5 + 34 * ease) / transform.k, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }

    function strokeFill(l, idx, colour, size, sym, m) {
      ctx.beginPath();
      for (const i of idx) {
        if (l.x[i] == null) continue;
        proj(l.x[i], l.y[i]);
        addSymbol(PX, PY, size, sym);
      }
      if (colour && colour !== 'rgba(0,0,0,0)') { ctx.fillStyle = colour; ctx.fill(); }
      if (m.line && m.line.color) {
        ctx.strokeStyle = m.line.color;
        ctx.lineWidth = (m.line.width || 1) / transform.k;
        ctx.stroke();
      }
    }

    function addSymbol(px, py, r, sym) {
      if (sym === 1) { ctx.moveTo(px - r, py - r); ctx.rect(px - r, py - r, r * 2, r * 2); return; }
      if (sym === 2) {
        ctx.moveTo(px, py - r); ctx.lineTo(px + r, py);
        ctx.lineTo(px, py + r); ctx.lineTo(px - r, py); ctx.closePath(); return;
      }
      if (sym === 3) {
        for (let k = 0; k < 10; k++) {
          const rad = k % 2 ? r * 0.45 : r;
          const a = -Math.PI / 2 + k * Math.PI / 5;
          const x = px + Math.cos(a) * rad, y = py + Math.sin(a) * rad;
          k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath(); return;
      }
      // moveTo before arc: without it the arc connects from the previous subpath's end
      // and every marker renders as a wedge. Cost an afternoon in The Walk.
      ctx.moveTo(px + r, py);
      ctx.arc(px, py, r, 0, Math.PI * 2);
    }

    // ── Region labels, as DOM ────────────────────────────────────────────
    //
    // Plotly drew these as layout annotations, which meant a relayout to change one, no
    // transition (the crossfade was an rgba() alpha baked into a colour string on a 150 ms
    // debounce, so labels popped), and no click target — clicking a region needed a 30-line
    // hit-test against annotation anchors. As DOM they get a real CSS transition and a
    // real click handler for free.
    function setAnnotations(list) {
      // Sorted big-first: size tracks region level, so the L0 headline labels claim their
      // space before the L1 detail labels compete for it.
      //
      // A `priority` term was tried here, to let a focused region's children outrank the
      // faint context labels around them. Measured on the densest case available (l0_4,
      // six L1 children): 6 children placed with it and 6 without — identical. The
      // collision pass was not what was limiting them, so the knob went back out rather
      // than ship as an unfalsifiable improvement.
      labels = (list || []).slice().sort((a, b) => (b.size || 0) - (a.size || 0));
      // Drop any cached pixel size: callers may hand back the same objects with a new
      // font-size (the level crossfade does exactly that), and a stale width silently
      // mis-places the collision boxes rather than failing.
      for (const a of labels) { a._w = null; a._h = null; }
      if (!labelHost) {
        labelHost = document.createElement('div');
        labelHost.className = 'map-labels';
        host.appendChild(labelHost);
      }
      labelHost.innerHTML = labels.map(function (a, i) {
        // The level rides through as a class so CSS can decide which labels are CONTROLS.
        // They are DOM buttons over the canvas, so every label is also a hole in the map:
        // a card underneath one cannot be hovered at all — the move lands on the button
        // and the canvas gets `mouseleave`. Countries and states earn that cost because
        // clicking them navigates; neighbourhoods are captions and give the pixels back.
        // The dark ring that keeps a name legible over a dense cluster, at the label's own
        // strength so a faint label stays faint instead of becoming a dark blob.
        const oa = (a.outline == null ? 1 : a.outline);
        const ring = 'rgba(8,10,20,' + (0.95 * oa).toFixed(2) + ')';
        const soft = 'rgba(8,10,20,' + (0.8 * oa).toFixed(2) + ')';
        const shadow = [
          '-1px -1px 0 ' + ring, '1px -1px 0 ' + ring,
          '-1px 1px 0 ' + ring, '1px 1px 0 ' + ring,
          '0 -1px 0 ' + ring, '0 1px 0 ' + ring,
          '-1px 0 0 ' + ring, '1px 0 0 ' + ring,
          '0 0 6px ' + soft,
        ].join(',');
        return '<button class="map-label map-label-l' + (a.level == null ? 1 : a.level) +
          '" data-i="' + i + '" data-id="' + (a.id || '') + '"' +
          ' style="font-size:' + a.size + 'px;color:' + a.colour +
          ';text-shadow:' + shadow + '">' +
          (a.text || '').replace(/[&<>"]/g, function (c) {
            return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c];
          }) + '</button>';
      }).join('');
      Array.prototype.forEach.call(labelHost.children, function (el) {
        el.addEventListener('click', function (e) {
          e.stopPropagation();
          emit('click', { regionId: el.dataset.id, row: null });
        });
      });
      positionLabels();
    }

    // Position, then hide whatever would overlap something already placed.
    //
    // Collision MUST be evaluated here rather than in setAnnotations, in PIXELS: the
    // annotations carry world coordinates (`region.cx/cy`), and comparing those against
    // label widths in pixels is a units error that rejects essentially everything — world
    // coords span about ±40 while a label is 150px wide, so every box overlaps every other.
    // Doing it here also makes it zoom-responsive for free: labels that collide zoomed out
    // separate as you zoom in, which is exactly the behaviour you want and what force.js
    // has always done for node labels.
    /* MEASURING and PLACING are separated, and the split is what makes an animated map
     * affordable. `offsetWidth` on a hidden element forces a synchronous layout; doing that
     * for every label on every frame of a 30 fps ambient loop is a self-inflicted jank
     * source that has nothing to do with the canvas. A label's PIXEL SIZE only changes when
     * its text or font-size changes, so it is measured once and cached on the label; only
     * its position, and therefore the collision pass, is recomputed per frame. */
    function positionLabels() {
      if (!labelHost || !labels.length || !baseFit) return;
      const kids = labelHost.children;
      const n = Math.min(kids.length, labels.length);
      const boxes = [];
      for (let i = 0; i < n; i++) {
        const el = kids[i];
        const p = dataToPixel(labels[i].x, labels[i].y);
        el.style.transform = 'translate(-50%,-50%) translate(' +
          Math.round(p[0]) + 'px,' + Math.round(p[1]) + 'px)';
        // Un-hidden unconditionally: the collision pass below re-decides every frame, so a
        // label rejected last frame has to be eligible again this one. This is a style
        // WRITE, which is free; only the measurement below reads back and forces layout.
        el.style.display = '';
        if (labels[i]._w == null) {
          labels[i]._w = el.offsetWidth || (labels[i].text || '').length * 7;
          labels[i]._h = el.offsetHeight || 16;
        }
        const w = labels[i]._w, h = labels[i]._h;
        boxes.push({ x0: p[0] - w / 2, x1: p[0] + w / 2, y0: p[1] - h / 2, y1: p[1] + h / 2 });
      }
      // The greedy AABB pass is `Stage.placeLabels` — the same one The Walk uses for node
      // and edge labels. It was copied here by hand when this renderer was written, and a
      // comment said so; now there is one of it. Screen space, always: an earlier version
      // compared world centroids against pixel widths and 2 of 19 labels survived.
      const ok = Stage.placeLabels(boxes, 3);
      for (let i = 0; i < n; i++) if (!ok[i]) kids[i].style.display = 'none';
      lastVisibleLabels = ok.placed;
    }

    // Move one layer's points without rebuilding anything else. This is what
    // `Plotly.restyle` bought: drill's animation pushes 90 frames of new positions, and
    // going through setLayers would rebuild all 34,322 base-layer arrays each frame.
    // Matched on a flag (`_isDrill`) rather than a name, because names carry live counts.
    function updateLayerBy(flag, patch) {
      for (const l of layers) {
        if (!l[flag]) continue;
        if (patch.x) l.x = patch.x;
        if (patch.y) l.y = patch.y;
        if (patch.customdata) l.customdata = patch.customdata;
        schedule();
        return true;
      }
      return false;
    }

    // ── Hit testing ──────────────────────────────────────────────────────

    // Screen → the row index under it, or null. The quadtree searches in WORLD space, so
    // the radius has to be converted back through both transforms.
    function pick(px, py, radiusPx) {
      if (!tree || !baseFit) return null;
      const p = transform.invert([px, py]);
      // Undo the ambient swirl before asking the tree, which stores UNMOVED positions. The
      // inverse is exact (see `unproj`), so this keeps working while the field drifts —
      // the alternative, hit-testing against where the cards are stored while drawing them
      // somewhere else, is off by several pixels at the fit, which is several cards.
      const q = unproj(p[0], p[1]);
      const r = (radiusPx == null ? 8 : radiusPx) / (baseFit.k * transform.k);
      const hit = tree.find(q[0], q[1], r);
      return hit ? hit.row : null;
    }

    // Every row inside a screen rectangle. This is the 138 ms operation; measured at
    // 4.5 ms here over 22,161 caught points.
    function pickRect(x0, y0, x1, y1) {
      if (!tree || !baseFit) return [];
      // `transform.invert` lands in base-fit space, which is where the swirl acts — so the
      // marquee is already in the right coordinates to test against directly.
      const a = transform.invert([Math.min(x0, x1), Math.min(y0, y1)]);
      const b = transform.invert([Math.max(x0, x1), Math.max(y0, y1)]);
      // A rotation maps a rectangle to something that is not a rectangle, so the stored
      // positions inside a screen box are NOT an axis-aligned range any more. Pruning gets
      // padded by the largest displacement the swirl can currently produce (conservative,
      // never misses), and membership is then decided by projecting each candidate FORWARD
      // and testing where it is actually drawn. Exact, and it costs one projection per
      // point the quadtree already had to visit.
      const pad = maxShift / baseFit.k;
      const wx0 = (a[0] - baseFit.tx) / baseFit.k - pad;
      const wy0 = (a[1] - baseFit.ty) / baseFit.k - pad;
      const wx1 = (b[0] - baseFit.tx) / baseFit.k + pad;
      const wy1 = (b[1] - baseFit.ty) / baseFit.k + pad;
      const out = [];
      tree.visit(function (node, nx0, ny0, nx1, ny1) {
        if (!node.length) {
          do {
            const d = node.data;
            if (d.x >= wx0 && d.x < wx1 && d.y >= wy0 && d.y < wy1) {
              proj(d.x, d.y);
              if (PX >= a[0] && PX < b[0] && PY >= a[1] && PY < b[1]) out.push(d.row);
            }
          } while (node = node.next);
        }
        return nx0 > wx1 || ny0 > wy1 || nx1 < wx0 || ny1 < wy0;
      });
      return out;
    }

    // ── Camera ───────────────────────────────────────────────────────────

    // Reported in DATA units, matching what `_fullLayout.xaxis.range` gave, so the region
    // label crossfade and everything else keyed on `visibleSpan` needs no changes.
    function getCamera() {
      if (!canvas || !baseFit) return null;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const a = transform.invert([0, 0]), b = transform.invert([w, h]);
      return {
        x: [(a[0] - baseFit.tx) / baseFit.k, (b[0] - baseFit.tx) / baseFit.k],
        y: [(b[1] - baseFit.ty) / baseFit.k, (a[1] - baseFit.ty) / baseFit.k],
      };
    }

    function setCamera(range, opts) {
      if (!canvas || !baseFit) return;
      const o = opts || {};
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const sx0 = wx(range.x[0]), sx1 = wx(range.x[1]);
      const sy0 = wy(range.y[1]), sy1 = wy(range.y[0]);
      const k = Math.min(w / Math.abs(sx1 - sx0 || 1), h / Math.abs(sy1 - sy0 || 1));
      const t = d3.zoomIdentity
        .translate(w / 2 - k * (sx0 + sx1) / 2, h / 2 - k * (sy0 + sy1) / 2)
        .scale(k);
      const sel = d3.select(canvas);
      // A d3 transition is driven by rAF, which does not run in a hidden tab — so an
      // animated camera move there simply never happens, silently. Everything else in
      // this file that touches rAF has needed the same guard; a camera that does not
      // arrive is worse than one that arrives without easing.
      if (o.animate && !document.hidden) {
        sel.transition().duration(o.duration || 420).call(zoomBehaviour.transform, t);
      } else {
        sel.call(zoomBehaviour.transform, t);
      }
    }

    // Data → screen pixels, for anything positioned over the map (region labels) — and the
    // canonical "where is this card right now" for callers and tests. It carries the
    // ambient term, so a click aimed through it lands on the card it aimed at.
    function dataToPixel(x, y) {
      if (!baseFit) return null;
      return transform.apply(projPt(x, y));
    }

    const api = {
      init, destroy, resize, setLayers,
      draw: schedule, drawNow: draw,
      pick, pickRect, getCamera, setCamera, dataToPixel,
      setAnnotations, updateLayerBy, reindex,
      get visibleLabelCount() { return lastVisibleLabels; },
      setSelectMode: function (on) { selectMode = !!on; if (canvas) canvas.style.cursor = on ? 'crosshair' : 'grab'; },
      /* Ambient motion, on or off, eased rather than snapped — dropping ~34,000 points back
       * to their stored positions in one frame reads as a glitch, and turning it off is
       * usually a deliberate act (a preference, a precision task) that deserves to look
       * deliberate. Reading `motion` back reports what is CONFIGURED; `motionLevel` reports
       * what is currently on screen, which is also zero when you are zoomed in. */
      setMotion: function (on) {
        motionEnabled = !!on;
        gainTarget = motionEnabled ? 1 : 0;
        if (motionEnabled) startTicking(); else schedule();
      },
      get motion() { return motionEnabled; },
      // "Is the map moving right now" — which is false when it is zoomed in, switched off,
      // in a background tab, or hidden behind the graph modes. Deliberately not just
      // `ambient()`: a level of 1 on a canvas nobody is drawing to is not a true answer.
      get motionLevel() { return wantsFrames() ? ambient() : 0; },
      // Where a card is being drawn RIGHT NOW, in data units — the ambient term applied.
      // Anything that has to align with a moving point and cannot go through
      // `dataToPixel` (an overlay in world space) asks here rather than guessing.
      projected: function (x, y) {
        if (!baseFit) return null;
        const p = projPt(x, y);
        return [(p[0] - baseFit.tx) / baseFit.k, (p[1] - baseFit.ty) / baseFit.k];
      },
      setHover: setHover,
      setContours: function (on) {
        showContours = !!on;
        if (showContours) { contourKey = null; ensureContours(); }
        draw();
      },
      // Refit to the data extent AND drop the user's zoom. Both halves matter: `baseFit`
      // is the world→screen fit and `transform` is the zoom on top of it, so keeping the
      // transform would frame the old camera over new coordinates. This is the map switch
      // — the one case that SHOULD forget where you were, because the coordinates
      // themselves changed. Under Plotly the same thing was expressed by clearing
      // `plotInitialized` so the next `react` autoranged.
      //
      // Pushed through `zoomBehaviour.transform` rather than assigning `transform`, or
      // d3 keeps its own internal copy and the next gesture jumps back.
      fitToData: function () {
        baseFit = null;
        fitToData();
        if (canvas && zoomBehaviour) {
          d3.select(canvas).call(zoomBehaviour.transform, d3.zoomIdentity);
        }
        draw();
      },
      on: function (name, fn) { (handlers[name] = handlers[name] || []).push(fn); return api; },
      get canvas() { return canvas; },
      // What is currently drawn. This is the canvas's answer to Plotly's `gd.data`, and it
      // exists because the browser suite legitimately needs to assert on the layer list —
      // that drilling hides the base layers, that the deck overlay is present, that the
      // browse marker moved. Losing that with the renderer would have meant deleting real
      // coverage rather than porting it. Read-only by convention: mutate through
      // `setLayers`/`updateLayerBy` so the quadtree signature stays honest.
      get layers() { return layers; },
      get layerCount() { return layers.length; },
      get pointCount() { return tree ? tree.size() : 0; },
    };
    return api;
  }

  window.MapCanvas = { create: create };
})();
