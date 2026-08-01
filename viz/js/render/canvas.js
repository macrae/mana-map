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
        if (hit == null) emit('unhover', null);
        else emit('hover', { row: hit, clientX: e.clientX, clientY: e.clientY });
      });
      canvas.addEventListener('mouseleave', function () { emit('unhover', null); });
      canvas.addEventListener('click', function (e) {
        const hit = pick(e.offsetX, e.offsetY);
        if (hit != null) emit('click', { row: hit, shiftKey: e.shiftKey, event: e });
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
    }

    function wx(x) { return baseFit.tx + baseFit.k * x; }
    function wy(y) { return baseFit.ty + baseFit.k * y; }

    // Rebuilding the quadtree costs 23.5 ms at 34,322 points, and setLayers runs on every
    // filter, search keystroke and panel change — most of which do not alter the pickable
    // set at all. Key on what the tree actually depends on and skip the rest.
    let treeKey = null;

    function treeSignature() {
      let sig = '';
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines') continue;
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
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines') continue;
        for (let i = 0; i < l.x.length; i++) {
          if (l.x[i] == null) continue;
          pts.push({ x: l.x[i], y: l.y[i], row: l.customdata[i] });
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
      const w = surface.width, h = surface.height;
      // Shared prologue — clear, save, translate, scale — and its matching close.
      const close = surface.open(transform);

      if (showContours && contour) drawContours();

      for (const l of layers) {
        if (l.visible === false || !l.x || !l.x.length) continue;
        if (l.mode === 'lines') drawLines(l);
        else drawMarkers(l);
      }
      close();

      if (marquee) drawMarquee();
      positionLabels();
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
        if (l.visible === false || !l.customdata || l.mode === 'lines') continue;
        for (let i = 0; i < l.x.length; i++) if (l.x[i] != null) pts.push(l);
        break;                       // the base scatter only; overlays are not density
      }
      const src = [];
      for (const l of layers) {
        if (l.visible === false || !l.customdata || l.mode === 'lines') continue;
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
      for (const c of contour) {
        const a = Math.min(0.34, 0.05 + 0.30 * (c.value / (max || 1)));
        ctx.fillStyle = 'rgba(120,100,200,' + a.toFixed(3) + ')';
        ctx.beginPath();
        path(c);
        ctx.fill();
      }
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
        const px = wx(l.x[i]), py = wy(l.y[i]);
        if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Batched: one path per colour, filled once. Measured 7.8 ms for 34,322 points versus
    // 16.9 ms issuing a fill per point — the difference between 128 fps and 59.
    function drawMarkers(l) {
      const m = l.marker || {};
      const size = (m.size == null ? 3 : m.size) / transform.k;
      const sym = SYMBOL[m.symbol] == null ? 0 : SYMBOL[m.symbol];
      const perColour = Array.isArray(m.color);
      const perAlpha = Array.isArray(m.opacity);

      if (!perColour && !perAlpha) {
        ctx.globalAlpha = m.opacity == null ? 1 : m.opacity;
        const idx = new Array(l.x.length);
        for (let i = 0; i < l.x.length; i++) idx[i] = i;
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
        ctx.globalAlpha = b.a;
        strokeFill(l, b.idx, b.c, size, sym, m);
      });
      ctx.globalAlpha = 1;
    }

    function strokeFill(l, idx, colour, size, sym, m) {
      ctx.beginPath();
      for (const i of idx) {
        if (l.x[i] == null) continue;
        addSymbol(wx(l.x[i]), wy(l.y[i]), size, sym);
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
      labels = (list || []).slice().sort((a, b) => (b.size || 0) - (a.size || 0));
      if (!labelHost) {
        labelHost = document.createElement('div');
        labelHost.className = 'map-labels';
        host.appendChild(labelHost);
      }
      labelHost.innerHTML = labels.map(function (a, i) {
        return '<button class="map-label" data-i="' + i + '" data-id="' + (a.id || '') + '"' +
          ' style="font-size:' + a.size + 'px;color:' + a.colour + '">' +
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
        // offsetWidth is 0 while hidden, so un-hide before measuring.
        el.style.display = '';
        const w = el.offsetWidth || (labels[i].text || '').length * 7;
        const h = el.offsetHeight || 16;
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
      const wxp = (p[0] - baseFit.tx) / baseFit.k;
      const wyp = (p[1] - baseFit.ty) / baseFit.k;
      const r = (radiusPx == null ? 8 : radiusPx) / (baseFit.k * transform.k);
      const hit = tree.find(wxp, wyp, r);
      return hit ? hit.row : null;
    }

    // Every row inside a screen rectangle. This is the 138 ms operation; measured at
    // 4.5 ms here over 22,161 caught points.
    function pickRect(x0, y0, x1, y1) {
      if (!tree || !baseFit) return [];
      const a = transform.invert([Math.min(x0, x1), Math.min(y0, y1)]);
      const b = transform.invert([Math.max(x0, x1), Math.max(y0, y1)]);
      const wx0 = (a[0] - baseFit.tx) / baseFit.k, wy0 = (a[1] - baseFit.ty) / baseFit.k;
      const wx1 = (b[0] - baseFit.tx) / baseFit.k, wy1 = (b[1] - baseFit.ty) / baseFit.k;
      const out = [];
      tree.visit(function (node, nx0, ny0, nx1, ny1) {
        if (!node.length) {
          do {
            const d = node.data;
            if (d.x >= wx0 && d.x < wx1 && d.y >= wy0 && d.y < wy1) out.push(d.row);
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

    // Data → screen pixels, for anything positioned over the map (region labels).
    function dataToPixel(x, y) {
      if (!baseFit) return null;
      return transform.apply([wx(x), wy(y)]);
    }

    const api = {
      init, destroy, resize, setLayers,
      draw: schedule, drawNow: draw,
      pick, pickRect, getCamera, setCamera, dataToPixel,
      setAnnotations, updateLayerBy,
      get visibleLabelCount() { return lastVisibleLabels; },
      setSelectMode: function (on) { selectMode = !!on; if (canvas) canvas.style.cursor = on ? 'crosshair' : 'grab'; },
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
