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
 * Supported because the map uses them: markers and lines, per-point colour arrays, scalar
 * opacity, the four symbols in use (circle, diamond, star, square), marker outlines, and
 * `visible: false`. NOT supported, deliberately: per-point opacity arrays (the deck
 * builder's dimming — it still runs on Plotly), and `histogram2dcontour` (Topo, which
 * moves to d3-contour in Phase 3). `setLayers` warns rather than silently mis-drawing.
 *
 * Coordinates. `world` is data space (projection_2d units, roughly ±60). `screen` is CSS
 * pixels. `transform` is the d3-zoom transform between them, and it is the ONLY thing that
 * moves — points are never mutated, unlike The Walk where the simulation owns positions.
 */
(function () {
  'use strict';

  const SYMBOL = { circle: 0, square: 1, diamond: 2, star: 3 };

  function create() {
    let host = null, canvas = null, ctx = null, dpr = 1;
    let layers = [];
    let transform = null;
    let zoomBehaviour = null;
    let tree = null;                 // d3.quadtree over every pickable point
    let baseFit = null;              // world→screen fit computed once from the data extent
    const handlers = { click: [], hover: [], unhover: [], camera: [], select: [] };
    let raf = null;

    function emit(name, arg) { for (const fn of handlers[name] || []) fn(arg); }

    // ── Setup ────────────────────────────────────────────────────────────

    function init(hostEl) {
      host = hostEl;
      canvas = document.createElement('canvas');
      canvas.className = 'map-canvas';
      host.appendChild(canvas);
      ctx = canvas.getContext('2d');
      transform = d3.zoomIdentity;

      zoomBehaviour = d3.zoom().scaleExtent([0.2, 400]).on('zoom', function (ev) {
        transform = ev.transform;
        schedule();
        emit('camera', getCamera());
      });
      d3.select(canvas).call(zoomBehaviour);

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

    // Only the backing store is set here — the element's size comes from CSS. A <canvas>
    // is a replaced element, so an inline width decouples it from its parent and it stops
    // following the layout. That cost an afternoon in The Walk; it is not repeated.
    function resize() {
      if (!canvas) return;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (!w || !h) return;
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function destroy() {
      if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
      canvas = ctx = host = tree = null;
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
        if (l && l.marker && Array.isArray(l.marker.opacity)) {
          if (!setLayers._warnedOpacity) {
            console.warn('[canvas] per-point opacity not implemented — layer drawn at full opacity');
            setLayers._warnedOpacity = true;
          }
        }
        return !!l;
      });
      if (!baseFit) fitToData();
      buildTree();
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

    function schedule() {
      if (raf !== null) return;
      raf = requestAnimationFrame(function () { raf = null; draw(); });
    }

    function draw() {
      if (!ctx || !canvas || !baseFit) return;
      const w = canvas.width / dpr, h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      for (const l of layers) {
        if (l.visible === false || !l.x || !l.x.length) continue;
        if (l.mode === 'lines') drawLines(l);
        else drawMarkers(l);
      }
      ctx.restore();
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
      const perPoint = Array.isArray(m.color);
      ctx.globalAlpha = (Array.isArray(m.opacity) || m.opacity == null) ? 1 : m.opacity;

      if (perPoint) {
        // Group by colour so a per-point palette still batches. The map has 6–20 distinct
        // colours, never 34,322.
        const byColour = new Map();
        for (let i = 0; i < l.x.length; i++) {
          const c = m.color[i] || '#666';
          let arr = byColour.get(c);
          if (!arr) { arr = []; byColour.set(c, arr); }
          arr.push(i);
        }
        byColour.forEach(function (idx, colour) { strokeFill(l, idx, colour, size, sym, m); });
      } else {
        const idx = new Array(l.x.length);
        for (let i = 0; i < l.x.length; i++) idx[i] = i;
        strokeFill(l, idx, m.color || '#666', size, sym, m);
      }
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
      if (o.animate) sel.transition().duration(o.duration || 420).call(zoomBehaviour.transform, t);
      else sel.call(zoomBehaviour.transform, t);
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
      fitToData: function () { baseFit = null; fitToData(); draw(); },
      on: function (name, fn) { (handlers[name] = handlers[name] || []).push(fn); return api; },
      get canvas() { return canvas; },
      get layerCount() { return layers.length; },
      get pointCount() { return tree ? tree.size() : 0; },
    };
    return api;
  }

  window.MapCanvas = { create: create };
})();
