/**
 * spaces-view.js — the embedding-space reference page.
 *
 * Draws `data/eval/space_projections.json` as five side-by-side scatters and
 * renders the measured tables. No `window.MM`: this page has no atlas, no card
 * index and no deck file, the same way `workbench.js` and `deck-view.js` stand
 * alone.
 *
 * WHY THE NUMBERS ARE IN THIS FILE. There is no metrics artifact to read — the
 * eval prints a report and writes nothing. So they are a SNAPSHOT, carried with
 * the command that reproduces them and the date they were taken, and the page
 * says so. When a metrics artifact exists this should read it instead.
 */
(function () {
  'use strict';

  var DATA_VERSION = 9;                       // mirrors mana-map.js
  var URL = '../data/eval/space_projections.json?v=' + DATA_VERSION;

  /* Measured 2026-09-01 by `manamap eval-embeddings`, held-out test split.
   * Order is the argument: incumbent, the two baselines, the challenger. */
  var HEADLINE = [
    ['space', 'dim', 'eff. dims', 'spread', 'recall@10', 'centroid headroom', 'hard-neg sep'],
    ['function (ability)', '128', '27.31', '0.0323', '0.232', '0.019', '0.0133'],
    ['text baseline (frozen MiniLM)', '384', '51.39', '0.1341', '0.244', '0.075', '0.0197'],
    ['layout (color+type)', '128', '3.89', '0.0061', '0.086', '0.150', '0.0891'],
    ['vae (masked imputation)', '128', '5.71', '0.0454', '0.167', '0.092', '0.0064'],
    ['cardbert (masked fields)', '128', '16.72', '0.1347', '0.103', '0.976', '0.0377']
  ];

  /* cardbert vs function, 95% CI on the DIFFERENCE. Every row excludes zero. */
  var POOLS = [
    ['candidates', 'FUNCTION — 28 groups', 'THEME — 55 groups'],
    ['100', '0.759 vs 0.964 → −0.205', '0.537 vs 0.443 → +0.094'],
    ['500', '0.519 vs 0.794 → −0.275', '0.303 vs 0.152 → +0.151'],
    ['2000', '0.317 vs 0.562 → −0.245', '0.127 vs 0.053 → +0.074'],
    ['10000', '0.201 vs 0.363 → −0.163', '0.031 vs 0.021 → +0.010']
  ];

  /* Presentation order and the one-line character of each space. */
  var ORDER = [
    ['function (ability)', 'the incumbent. Trained on role and tag positives — so function is what it knows and tribe is what it discards.'],
    ['cardbert (masked fields)', '73 typed fields and 6 text spans, masked-group imputation. Better at tribe, worse at function.'],
    ['text baseline (frozen MiniLM)', 'frozen sentence vectors, untrained. Still wins commander search.'],
    ['vae (masked imputation)', 'beaten by a RANDOM projection on theme. Retrieves better than cardbert and maps far worse.'],
    ['layout (color+type)', 'colour and type only, 3.89 of 128 dimensions. The floor.']
  ];

  var PAL = ['#3B7DB5', '#C4762A', '#2D8A5E', '#B24B57', '#7B5EA8', '#B08A2B',
             '#4AA0A8', '#A55A8E', '#6B8F3A', '#8A6A55', '#5B6BA8', '#C05F3C'];
  var FIXED = { W: '#D8CB9A', U: '#5B93C4', B: '#6E6763', R: '#C4675A',
                G: '#5E9A6B', C: '#A9A296', multi: '#B08A2B' };

  var data = null, key = 'identity', dot = 2, panels = [];

  function table(el, rows) {
    var head = rows[0], body = rows.slice(1);
    el.innerHTML =
      '<thead><tr>' + head.map(function (h) { return '<th>' + h + '</th>'; }).join('') +
      '</tr></thead><tbody>' + body.map(function (r) {
        var cls = /cardbert/.test(r[0]) ? ' class="hi"' : '';
        return '<tr' + cls + '>' + r.map(function (c, i) {
          return '<td' + (i ? ' class="num"' : '') + '>' + c + '</td>';
        }).join('') + '</tr>';
      }).join('') + '</tbody>';
  }

  function palette(values) {
    var counts = {};
    values.forEach(function (v) { if (v) counts[v] = (counts[v] || 0) + 1; });
    var order = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; });
    var map = {};
    order.forEach(function (k, i) { map[k] = FIXED[k] || PAL[i % PAL.length]; });
    return { map: map, order: order.slice(0, 12) };
  }

  function legend(p) {
    document.getElementById('legend').innerHTML =
      p.order.map(function (k) {
        return '<span><i style="background:' + p.map[k] + '"></i>' + k + '</span>';
      }).join('') +
      '<span><i style="background:#9aa39c;opacity:.35"></i>other / none</span>';
  }

  function draw(cv, pts, p) {
    var vals = data.facts[key];
    var w = cv.clientWidth, h = Math.round(w * 0.78);
    var r = window.devicePixelRatio || 1;
    cv.width = w * r; cv.height = h * r; cv.style.height = h + 'px';
    var g = cv.getContext('2d');
    g.setTransform(r, 0, 0, r, 0, 0);
    g.clearRect(0, 0, w, h);
    var pad = 10, sc = Math.min(w - pad * 2, h - pad * 2) / 2000;
    var ox = w / 2, oy = h / 2, top = {};
    p.order.forEach(function (k) { top[k] = 1; });
    /* Two passes so the coloured groups sit ON TOP of the grey field —
       otherwise a large "other" group buries the thing being compared. */
    for (var pass = 0; pass < 2; pass++) {
      for (var i = 0; i < pts.length; i++) {
        var v = vals[i], hot = v && top[v];
        if ((pass === 0) === !!hot) continue;
        g.fillStyle = hot ? p.map[v] : '#9aa39c';
        g.globalAlpha = hot ? 0.78 : 0.16;
        g.beginPath();
        g.arc(ox + pts[i][0] * sc, oy + pts[i][1] * sc, dot, 0, 6.2832);
        g.fill();
      }
    }
    g.globalAlpha = 1;
  }

  function redraw() {
    if (!data) return;
    var p = palette(data.facts[key]);
    legend(p);
    panels.forEach(function (x) { draw(x.cv, x.pts, p); });
  }

  function build() {
    var grid = document.getElementById('grid');
    ORDER.forEach(function (entry) {
      var label = entry[0], note = entry[1];
      var space = data.spaces[label];
      if (!space) return;
      var el = document.createElement('div');
      el.className = 'sp';
      el.innerHTML = '<h3>' + label + '</h3><p class="note">' + note + '</p>';
      var cv = document.createElement('canvas');
      el.appendChild(cv);
      grid.appendChild(el);
      panels.push({ cv: cv, pts: space.points });
    });
  }

  function wire(sel, apply) {
    var all = document.querySelectorAll(sel);
    Array.prototype.forEach.call(all, function (b) {
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(all, function (o) { o.classList.remove('on'); });
        b.classList.add('on');
        apply(b);
        redraw();
      });
    });
  }

  table(document.getElementById('headline'), HEADLINE);
  table(document.getElementById('pools'), POOLS);

  fetch(URL)
    .then(function (r) {
      if (!r.ok) throw new Error('projections ' + r.status);
      return r.json();
    })
    .then(function (d) {
      data = d;
      build();
      redraw();
      document.getElementById('status').textContent =
        d.cards.toLocaleString() + ' cards · ' + Object.keys(d.spaces).length +
        ' spaces · identical PaCMAP settings and seed';
    })
    .catch(function (e) {
      /* Names the command, because the artifact is regenerable and its absence
         on a fresh clone is expected rather than broken. */
      document.getElementById('status').textContent =
        'Could not load the projections (' + e.message +
        ') — run `manamap project-spaces`.';
    });

  wire('#controls button[data-k]', function (b) { key = b.dataset.k; });
  wire('#controls button[data-s]', function (b) { dot = +b.dataset.s; });
  window.addEventListener('resize', redraw);
})();
