/* The Fleet — every deck side by side, read from ONE committed artifact.
 *
 * `data/fleet/analytics.json` (manamap pilot fleet-analytics --write) is the
 * whole input: per-deck profiles with intervals, per-bracket bands and
 * placements, and the comparisons. This page COMPUTES NOTHING. It sorts, it
 * draws, it prints what the artifact says — including every `absent_because`
 * — and the one thing it adds is a leaderboard of `rating.json` files, which
 * are ★ coaching written per deck by the rating agent and shown with that badge.
 *
 * No d3. The curves are hand-written SVG over the benchmark's own series
 * (constellationPanel's precedent on the deck page), the interval bars are a
 * CSS span beside the interval PRINTED AS TEXT — the bar decorates, the text is
 * the figure. The shared formatters live in `api.js` (`window.Fmt`).
 */
(function () {
  'use strict';

  var F = window.Fmt;
  var esc = F.esc, num = F.num, pct = F.pct, getJSON = F.getJSON;
  var ARTIFACT = '../data/fleet/analytics.json';
  var DECKS = '../data/decks/';

  var DTI_GROUPS = [['R', 'Resources'], ['A', 'Access'], ['P', 'Pressure'],
                    ['I', 'Interaction'], ['S', 'Resilience']];

  // ── Figures ──────────────────────────────────────────────────────────────

  function isFig(x) { return !!x && typeof x === 'object' && 'value' in x && 'source' in x; }
  function isAbsent(x) { return !!x && typeof x === 'object' && 'absent_because' in x; }

  function dash(reason) {
    return '<span class="t-none" title="' + esc(reason || 'absent') + '">—</span>';
  }

  /* A rate with its interval, as text. `[lo, hi]` always printed; the n rides on
   * the title so the cell stays one line. */
  function rate(cell, digits) {
    if (!isFig(cell)) return dash(cell && cell.absent_because);
    var d = digits === undefined ? 1 : digits;
    var s = pct(cell.value, d);
    if (cell.ci95) s += ' <span class="ev">[' + pct(cell.ci95[0], d) + ', ' + pct(cell.ci95[1], d) + ']</span>';
    var title = (cell.n ? 'n = ' + cell.n + '; ' : '') + (cell.basis || cell.source);
    return '<span title="' + esc(title) + '">' + s + '</span>'
      + (cell.stale ? ' <span class="t-warn">older list</span>' : '');
  }

  function plain(cell, digits) {
    if (!isFig(cell)) return dash(cell && cell.absent_because);
    return '<span title="' + esc(cell.basis || cell.source) + '">' + num(cell.value, digits) + '</span>';
  }

  /* The interval bar: a track, the interval as a filled span, the value as a
   * marker. Pure decoration beside `rate()`'s text, so a reader with no CSS
   * still has the figure. `max` is the axis end (1 for a rate). */
  function ciBar(cell, max) {
    if (!isFig(cell)) return '';
    var m = max || 1, v = cell.value / m * 100;
    var lo = cell.ci95 ? cell.ci95[0] / m * 100 : v, hi = cell.ci95 ? cell.ci95[1] / m * 100 : v;
    return '<span class="ci-bar" aria-hidden="true"><i style="left:' + lo.toFixed(2) + '%;width:'
      + Math.max(0.5, hi - lo).toFixed(2) + '%"></i><b style="left:' + v.toFixed(2) + '%"></b></span>';
  }

  function money(cell) {
    if (!isFig(cell)) return dash(cell && cell.absent_because);
    return '<span title="' + esc(cell.basis || '') + '">$' + Math.round(cell.value / 100) + '</span>'
      + (cell.stale ? ' <span class="t-warn">older list</span>' : '');
  }

  function dtiCell(d) {
    if (isAbsent(d)) return dash(d.absent_because);
    var s = esc(d.score.value) + '/' + esc(d.max_score) + ' B' + esc(d.bracket);
    if (d.stale_list) s += ' <span class="t-warn">older list</span>';
    return s + ' <span class="ev">' + esc(d.as_of) + '</span>';
  }

  // ── State ────────────────────────────────────────────────────────────────

  var SORTS = {
    bracket: { label: 'By bracket', key: function (r) { return [r.apart ? 1 : 0, r.bracket.group, r.slug]; } },
    dti: { label: 'DTI', key: function (r) { return [r.apart ? 1 : 0, isAbsent(r.dti) ? 1 : 0, -(isAbsent(r.dti) ? 0 : r.dti.score.value)]; } },
    kill8: { label: 'Kill by 8', key: function (r) { var c = fig(r, 'benchmark', 'kill_by_turn_8'); return [r.apart ? 1 : 0, c ? 0 : 1, -(c ? c.value : 0)]; } },
    stall: { label: 'Stall', key: function (r) { var c = fig(r, 'diagnostic', 'stall_two_in_a_row'); return [r.apart ? 1 : 0, c ? 0 : 1, c ? c.value : 0]; } },
    price: { label: 'Price', key: function (r) { var c = fig(r, 'price', 'total_cents'); return [r.apart ? 1 : 0, c ? 0 : 1, -(c ? c.value : 0)]; } },
    record: { label: 'Games logged', key: function (r) { return [r.apart ? 1 : 0, -(r.record.games || 0)]; } }
  };

  function fig(r, block, key) {
    var b = r[block];
    if (!b || isAbsent(b)) return null;
    return isFig(b[key]) ? b[key] : null;
  }

  function cmpKey(a, b) {
    for (var i = 0; i < Math.max(a.length, b.length); i++) {
      var x = a[i] === undefined ? 0 : a[i], y = b[i] === undefined ? 0 : b[i];
      if (x !== y) return x < y ? -1 : 1;
    }
    return 0;
  }

  function readState() {
    var p = new URLSearchParams(window.location.search);
    return { sort: SORTS[p.get('sort')] ? p.get('sort') : 'bracket',
             bracket: p.get('bracket') || '' };
  }

  function writeState(s) {
    var p = new URLSearchParams(window.location.search);
    if (s.sort && s.sort !== 'bracket') p.set('sort', s.sort); else p.delete('sort');
    if (s.bracket) p.set('bracket', s.bracket); else p.delete('bracket');
    var q = p.toString();
    history.replaceState(null, '', window.location.pathname + (q ? '?' + q : ''));
  }

  if (window.PageState) PageState.register(function () {
    var st = readState();
    var s = { view: st.sort, filters: st.bracket ? { bracket: st.bracket } : {} };
    if (window.Session && Session.library) s.library = Session.library.names;
    return s;
  });

  // ── (a) The fleet table ──────────────────────────────────────────────────

  function surveyCell(sv) {
    if (isAbsent(sv)) return dash(sv.absent_because);
    var keys = ['power', 'resilience', 'consistency', 'fun', 'interaction'];
    var bits = keys.filter(function (k) { return sv[k]; }).map(function (k) {
      return '<span title="' + esc(k + ': mean ' + sv[k].mean + ', n ' + sv[k].n + ', ' + sv[k].min + '-' + sv[k].max) + '">'
        + esc(k.slice(0, 3)) + ' ' + num(sv[k].mean, 1) + '</span>';
    });
    return bits.join(' · ') + ' <span class="ev">n=' + esc(sv.games_surveyed) + '</span>';
  }

  // The grade a reader sees: `grade_display` is the range when the deck's
  // instruments diverge ("D–C"), the letter otherwise (Rating v2, 2026-10-19).
  function gradeShown(rt) { return rt.grade_display || rt.grade; }

  function ratingCell(rt) {
    if (!rt) return dash('no rating.json — the rating is ★ coaching, written by the rating agent');
    return '<span class="chip fl-star" title="' + esc(rt.label || '') + '">★ ' + esc(gradeShown(rt)) + '</span> '
      + num(rt.score, 1) + (rt.rank ? ' <span class="ev">' + esc(rt.rank.position) + '/' + esc(rt.rank.of) + '</span>' : '');
  }

  // Rating v2's extra chips: the probability of first place, the concordance
  // level with every instrument's rank in the tooltip, and the nearest flip.
  function ratingExtras(rt) {
    var out = '';
    var acc = rt.rank && rt.rank.acceptability;
    if (acc && acc['1'] != null) {
      out += '<span class="chip fl-acc" title="' + esc(Object.keys(acc).map(function (p) { return 'P(' + p + ') ' + pct(acc[p], 0); }).join(', ')) + '">P(1st) ' + pct(acc['1'], 0) + '</span>';
    }
    var cc = rt.concordance;
    if (cc && cc.level) {
      var inst = cc.instruments || {};
      var tip = Object.keys(inst).map(function (i) { return i + ' #' + inst[i]; }).join(', ')
        + (cc.outliers && cc.outliers.length ? '; outliers: ' + cc.outliers.map(function (o) { return o.instrument; }).join(', ') : '')
        + (cc.group && cc.group.kendall_w != null ? '; group W ' + cc.group.kendall_w + ' p ' + cc.group.p : '');
      out += '<span class="chip fl-cc" data-level="' + esc(cc.level) + '" title="' + esc(tip) + '">' + esc(cc.level) + '</span>';
    }
    return out;
  }

  function sensitivityLine(rt) {
    var s = rt.sensitivity;
    if (!s || !s.flips || !s.flips.length) return '';
    var f = s.flips[0];
    var loo = s.leave_one_out && s.leave_one_out[f.component];
    return '<div class="fl-words ev">'
      + (loo ? 'without ' + esc(f.component) + ' ' + num(loo.score, 1) + ' ' + esc(loo.grade) + '; ' : '')
      + 'grade flips to ' + esc(f.grade_at_flip) + ' at ' + esc(f.component) + ' share ' + num(f.share_at_flip, 2)
      + ' (now ' + num(f.share_now, 2) + ')</div>';
  }

  function tableRow(r, rt) {
    var slug = encodeURIComponent(r.slug);
    var b = r.bracket;
    var group = b.group === 'unbracketed' ? '<span class="t-none">—</span>'
      : esc(b.group) + ' <span class="ev">' + esc(b.group_basis) + '</span>';
    var floor = b.floor ? 'B' + esc(b.floor) : '<span class="t-none">—</span>';
    var rec = r.record;
    var record = rec.games
      ? rec.games + ' · ' + rec.win + '–' + rec.loss + (rec.draw ? '–' + rec.draw : '')
      : '<span class="t-none">none</span>';
    return '<tr class="t-head' + (r.apart ? ' is-dead' : '') + '">'
      + '<th scope="row"><a href="deck.html?deck=' + slug + '">' + esc(r.slug) + '</a>'
      + (r.format ? ' <span class="ev">' + esc(r.format) + '</span>' : '')
      + (r.apart ? ' <span class="t-none">apart</span>' : '') + '</th>'
      + '<td>' + group + '</td>'
      + '<td>' + floor + '</td>'
      + '<td>' + dtiCell(r.dti) + '</td>'
      + '<td>' + rate(fig(r, 'benchmark', 'kill_by_turn_8') || (isAbsent(r.benchmark) ? r.benchmark : null)) + '</td>'
      + '<td>' + rate(fig(r, 'diagnostic', 'stall_two_in_a_row') || (isAbsent(r.diagnostic) ? r.diagnostic : null)) + '</td>'
      + '<td>' + rate(fig(r, 'table', 'win_rate') || (isAbsent(r.table) ? r.table : null)) + '</td>'
      + '<td>' + money(fig(r, 'price', 'total_cents') || (isAbsent(r.price) ? r.price : null)) + '</td>'
      + '<td>' + record + '</td>'
      + '<td>' + surveyCell(r.survey) + '</td>'
      + '<td>' + ratingCell(rt) + '</td>'
      + '</tr>';
  }

  function totals(rows) {
    var g = 0, w = 0, l = 0, dti = 0, rated = 0;
    rows.forEach(function (r) {
      g += r.record.games || 0; w += r.record.win || 0; l += r.record.loss || 0;
      if (!isAbsent(r.dti)) dti += 1;
    });
    return '<tr class="t-total"><th scope="row">' + rows.length + ' sleeved</th>'
      + '<td></td><td></td><td>' + dti + ' with a DTI</td><td></td><td></td><td></td><td></td>'
      + '<td>' + (g ? g + ' · ' + w + '–' + l : '<span class="t-none">none</span>') + '</td>'
      + '<td></td><td></td></tr>';
  }

  /* THE FLEET IS THE SLEEVED DECKS (2026-10-19): the artifact carries a row for
   * every deck with a cards.json, and marks `member` on the sleeved, bracketed
   * Commander ones; this page draws members only. A bench deck has its own page. */
  function members(doc) {
    return doc.decks.filter(function (r) { return r.member === true; });
  }

  function fleetTable(doc, ratings, state) {
    var rows = members(doc).filter(function (r) { return !state.bracket || r.bracket.group === state.bracket; });
    var s = SORTS[state.sort] || SORTS.bracket;
    rows = rows.slice().sort(function (a, b) {
      return cmpKey(s.key(a), s.key(b)) || (a.slug < b.slug ? -1 : 1);
    });
    var tabs = Object.keys(SORTS).map(function (k) {
      return '<button class="wb-sort' + (k === state.sort ? ' is-on' : '') + '" data-sort="' + k + '">'
        + esc(SORTS[k].label) + '</button>';
    }).join('');
    var groups = [''].concat(Object.keys(doc.groups));
    var filt = groups.map(function (g) {
      return '<button class="wb-view' + (g === state.bracket ? ' is-on' : '') + '" data-bracket="' + esc(g) + '">'
        + (g ? esc(g) : 'All') + '</button>';
    }).join('');
    return '<section class="wb-rack" id="fleet-table"><h2>The fleet'
      + ' <span class="wb-count">' + rows.length + '</span></h2>'
      + '<p class="wb-blurb">The sleeved decks only. Every figure with its interval where one exists; a dash is an absence with its reason on hover. '
      + 'The rating column is ★ coaching from each deck’s <code>rating.json</code>, never a measurement.</p>'
      + '<div class="wb-views">' + filt + '</div><div class="wb-sorts">' + tabs + '</div>'
      + '<div class="wb-tablewrap"><table class="wb-table"><thead><tr>'
      + '<th scope="col">Deck</th><th scope="col" title="the pilot’s declared target; floor when none">Bracket</th>'
      + '<th scope="col">Floor</th><th scope="col" title="DeckCheck Threat Index, dated">DTI</th>'
      + '<th scope="col" title="benchmark: lethal by turn 8, unopposed, frozen harness">Kill by 8</th>'
      + '<th scope="col" title="diagnostic: two stalled turns in a row">Stall</th>'
      + '<th scope="col" title="Forge win rate at the standard table — the AI’s piloting, not a power read">Table</th>'
      + '<th scope="col">Price</th><th scope="col">Record</th>'
      + '<th scope="col" title="the pilot’s own 1-5 scores, authored">Survey</th>'
      + '<th scope="col" title="★ coaching, per deck">Rating</th>'
      + '</tr></thead><tbody>'
      + rows.map(function (r) { return tableRow(r, ratings[r.slug]); }).join('')
      + totals(rows)
      + '</tbody></table></div></section>';
  }

  // ── (b) By bracket ───────────────────────────────────────────────────────

  function bandRow(name, band, measure) {
    var label = esc(name.replace(/_/g, ' '));
    if (isAbsent(band)) {
      return '<tr><th scope="row">' + label + '</th><td colspan="2" class="t-none">' + esc(band.absent_because) + '</td></tr>';
    }
    var f = /rate|kill|stall|missed|mulligan/.test(name) ? function (v) { return pct(v, 1); }
      : /answer_cards|dti_score|classes/.test(name) ? function (v) { return num(v, 0); }
      : function (v) { return num(v, 2); };
    return '<tr><th scope="row">' + label + '</th>'
      + '<td>' + f(band.min) + ' · <b>' + f(band.median) + '</b> · ' + f(band.max) + '</td>'
      + '<td class="ev">n=' + band.n + '</td></tr>';
  }

  function bracketPanels(doc) {
    var by = {};
    doc.decks.forEach(function (r) { by[r.slug] = r; });
    var html = Object.keys(doc.groups).map(function (g) {
      var spec = doc.groups[g];
      var members = spec.members.map(function (s) {
        var r = by[s], pl = (r.placements && r.placements.group) || {};
        var words = Object.keys(pl).map(function (k) {
          var cls = /better/.test(pl[k]) ? 't-ok' : /worse/.test(pl[k]) ? 't-warn' : /no band/.test(pl[k]) ? 't-none' : '';
          return '<span class="chip ' + cls + '" title="' + esc(k) + '">' + esc(k.replace(/_/g, ' ')) + ': ' + esc(pl[k]) + '</span>';
        }).join('');
        return '<div class="fl-member"><a href="deck.html?deck=' + encodeURIComponent(s) + '">' + esc(s) + '</a>'
          + (r.stage ? ' <span class="ev">' + esc(r.stage) + '</span>' : '')
          + '<div class="fl-words">' + (words || '<span class="t-none">no placements</span>') + '</div></div>';
      }).join('');
      var bands = Object.keys(spec.bands).map(function (k) {
        return bandRow(k, spec.bands[k], k);
      }).join('');
      return '<section class="wb-rack" id="fleet-' + esc(g) + '"><h2>' + esc(g)
        + ' <span class="wb-count">' + spec.n + '</span></h2>'
        + (spec.n < doc.meta.min_group
            ? '<p class="wb-blurb t-warn">' + spec.n + ' member(s): no band below ' + doc.meta.min_group + '.</p>'
            : '<p class="wb-blurb">min · <b>median</b> · max over the members that carry each measure.</p>')
        + '<div class="rap-wrap"><table class="rap"><thead><tr><th>Measure</th><th>Band</th><th></th></tr></thead><tbody>'
        + bands + '</tbody></table></div>'
        + '<div class="fl-members">' + members + '</div></section>';
    }).join('');
    return html;
  }

  // ── (c) Curves, grades, floor vs DTI, interval bars ──────────────────────

  var W = 220, H = 120, PAD = 22;

  function linePath(series, yMax) {
    var pts = [];
    for (var t = 1; t <= 10; t++) {
      var v = series[String(t)];
      if (v === undefined || v === null) continue;
      var x = PAD + (t - 1) / 9 * (W - PAD - 6);
      var y = H - PAD + 4 - (v / yMax) * (H - PAD - 8);
      pts.push((pts.length ? 'L' : 'M') + x.toFixed(1) + ',' + y.toFixed(1));
    }
    return pts.join(' ');
  }

  function smallMultiple(slug, series, yMax, fmt) {
    var dots = '';
    for (var t = 1; t <= 10; t++) {
      var v = series[String(t)];
      if (v === undefined || v === null) continue;
      var x = PAD + (t - 1) / 9 * (W - PAD - 6);
      var y = H - PAD + 4 - (v / yMax) * (H - PAD - 8);
      dots += '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="2.2"><title>turn ' + t + ': ' + fmt(v) + '</title></circle>';
    }
    return '<figure class="fl-multiple"><figcaption><a href="deck.html?deck=' + encodeURIComponent(slug) + '">' + esc(slug) + '</a></figcaption>'
      + '<svg class="fl-svg" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(slug) + '">'
      + '<line x1="' + PAD + '" y1="' + (H - PAD + 4) + '" x2="' + (W - 6) + '" y2="' + (H - PAD + 4) + '" class="fl-axis"/>'
      + '<line x1="' + PAD + '" y1="' + (H - PAD + 4) + '" x2="' + PAD + '" y2="6" class="fl-axis"/>'
      + '<text x="' + (PAD - 3) + '" y="10" class="fl-tick">' + esc(fmt(yMax)) + '</text>'
      + '<text x="' + (W - 6) + '" y="' + (H - 4) + '" class="fl-tick fl-tick-x">T10</text>'
      + '<path d="' + linePath(series, yMax) + '" class="fl-line"/>' + dots + '</svg></figure>';
  }

  function curvesPanel(id, title, blurb, curves, yMax, fmt, ms) {
    var slugs = Object.keys(curves).filter(function (s) { return !ms || ms[s]; });
    if (!slugs.length) {
      return '<section class="wb-rack" id="' + id + '"><h2>' + esc(title) + '</h2>'
        + '<p class="wb-blurb t-none">no curves on file — `manamap pilot regen --only benchmark` writes the series</p></section>';
    }
    return '<section class="wb-rack" id="' + id + '"><h2>' + esc(title) + ' <span class="wb-count">' + slugs.length + '</span></h2>'
      + '<p class="wb-blurb">' + esc(blurb) + '</p><div class="fl-multiples">'
      + slugs.map(function (s) { return smallMultiple(s, curves[s], yMax, fmt); }).join('') + '</div></section>';
  }

  function memberSet(doc) {
    var m = {};
    members(doc).forEach(function (r) { m[r.slug] = true; });
    return m;
  }

  function gradesHeatmap(doc) {
    var ms = memberSet(doc);
    var g = doc.comparisons.benchmark_grades, rows = g.rows || {}, slugs = Object.keys(rows).filter(function (s) { return ms[s]; });
    if (!slugs.length) return '';
    var keys = DTI_GROUPS.reduce(function (acc, grp) { return acc.concat(g.groups[grp[0]] || []); }, []);
    var head = '<tr><th>Deck</th>' + keys.map(function (k) { return '<th title="' + esc(k) + '">' + esc(k) + '</th>'; }).join('') + '</tr>';
    var body = slugs.map(function (s) {
      return '<tr><th scope="row"><a href="deck.html?deck=' + encodeURIComponent(s) + '">' + esc(s) + '</a></th>'
        + keys.map(function (k) {
          var gr = rows[s][k];
          return '<td><span class="chip dti-chip" data-grade="' + esc(gr || '') + '">' + esc(gr || '—') + '</span></td>';
        }).join('') + '</tr>';
    }).join('');
    var dist = keys.map(function (k) {
      var d = g.distribution[k] || {};
      return '<td class="ev">' + ['S', 'A', 'B', 'C', 'F'].map(function (x) { return d[x] ? x + d[x] : ''; }).filter(Boolean).join(' ') + '</td>';
    }).join('');
    return '<section class="wb-rack" id="fleet-grades"><h2>DeckCheck’s twelve grades</h2>'
      + '<p class="wb-blurb">' + DTI_GROUPS.map(function (x) { return esc(x[0] + ' ' + x[1]); }).join(' · ')
      + '. An LLM’s read, dated; only S and F are coloured.</p>'
      + '<div class="rap-wrap"><table class="rap fl-heat"><thead>' + head + '</thead><tbody>' + body
      + '<tr class="t-total"><th scope="row">spread</th>' + dist + '</tr></tbody></table></div></section>';
  }

  function floorDtiTable(doc) {
    var ms = memberSet(doc);
    var rows = (doc.comparisons.floor_vs_dti || []).filter(function (r) { return ms[r.slug]; });
    var sp = doc.comparisons.spearman || {};
    function spearman(name, cell) {
      if (!cell) return '';
      if (cell.verdict === 'measured') {
        return '<p class="ev">Spearman(' + esc(name) + ') rho ' + num(cell.rho, 2) + ' on n=' + cell.n
          + ', critical ' + num(cell.critical_rho, 2) + (cell.significant ? ' — significant' : ' — not significant') + '</p>';
      }
      return '<p class="ev"><b>' + esc(cell.verdict) + '</b> — ' + esc(cell.why) + '. What it would take: ' + esc(cell.what_it_would_take) + '.</p>';
    }
    var body = rows.map(function (r) {
      var cls = r.relation === 'equal' ? 't-ok' : r.relation === 'dti_lower' ? 't-warn' : '';
      return '<tr><th scope="row"><a href="deck.html?deck=' + encodeURIComponent(r.slug) + '">' + esc(r.slug) + '</a></th>'
        + '<td>B' + esc(r.floor) + (r.target ? ' <span class="ev">target B' + esc(r.target) + '</span>' : '') + '</td>'
        + '<td>B' + esc(r.dti_bracket) + ' <span class="ev">' + esc(r.dti_score) + '/96</span></td>'
        + '<td class="' + cls + '">' + esc((r.relation || '').replace(/_/g, ' ')) + '</td>'
        + '<td class="ev">' + esc(r.reading_as_of) + (r.stale_list ? ' <span class="t-warn">older list</span>' : '') + '</td></tr>';
    }).join('');
    return '<section class="wb-rack" id="fleet-floor-dti"><h2>Our floor against their bracket</h2>'
      + '<p class="wb-blurb">The floor is what the list’s contents force, card by card; the DTI bracket is DeckCheck’s read. The floor gates, the DTI never does.</p>'
      + (rows.length
          ? '<div class="tablewrap"><table class="netchange"><thead><tr><th>Deck</th><th>Floor</th><th>DTI</th><th>Relation</th><th>Read</th></tr></thead><tbody>' + body + '</tbody></table></div>'
          : '<p class="t-none">no deck has a DTI reading on file</p>')
      + spearman('DTI, kill by 8', sp.dti_vs_kill_by_8) + spearman('DTI, Forge win rate', sp.dti_vs_table_win_rate)
      + '</section>';
  }

  function intervalBars(doc) {
    function block(title, pick, max, what) {
      var rows = members(doc).filter(function (r) { return pick(r); }).sort(function (a, b) {
        var x = pick(a).value, y = pick(b).value;
        return x === y ? (a.slug < b.slug ? -1 : 1) : (x > y ? -1 : 1);
      });
      if (!rows.length) return '';
      return '<h3 class="fl-h3">' + esc(title) + '</h3><p class="ev">' + esc(what) + '</p>'
        + rows.map(function (r) {
          var c = pick(r);
          return '<div class="fl-bar-row"><span class="fl-bar-name"><a href="deck.html?deck=' + encodeURIComponent(r.slug) + '">' + esc(r.slug) + '</a></span>'
            + ciBar(c, max) + '<span class="fl-bar-fig">' + rate(c) + '</span></div>';
        }).join('');
    }
    return '<section class="wb-rack" id="fleet-intervals"><h2>Intervals</h2>'
      + '<p class="wb-blurb">The bar is the 95% interval, the mark is the value; the text beside it is the figure.</p>'
      + block('Kill by turn 8', function (r) { return fig(r, 'benchmark', 'kill_by_turn_8'); }, 1,
              'benchmark: unopposed, frozen harness, n = iterations')
      + block('Forge win rate at the standard table', function (r) { return fig(r, 'table', 'win_rate'); }, 1,
              'the AI’s piloting of the list it played; a fair share of four seats is 25%')
      + block('Two stalled turns in a row', function (r) { return fig(r, 'diagnostic', 'stall_two_in_a_row'); }, 0.5,
              'diagnostic: lower is better; axis to 50%')
      + '</section>';
  }

  // ── (d) Trends ───────────────────────────────────────────────────────────

  function dtiHistory(doc) {
    var ms = memberSet(doc);
    var h = doc.comparisons.dti_history || {}, slugs = Object.keys(h).filter(function (s) { return ms[s]; });
    if (!slugs.length) return '';
    var body = slugs.map(function (s) {
      return h[s].map(function (x, i) {
        return '<tr' + (x.current_list ? '' : ' class="is-dead"') + '>'
          + (i === 0 ? '<th scope="row" rowspan="' + h[s].length + '"><a href="deck.html?deck=' + encodeURIComponent(s) + '">' + esc(s) + '</a></th>' : '')
          + '<td>' + esc(x.as_of) + '</td><td>' + esc(x.score) + '/96</td><td>B' + esc(x.bracket) + '</td>'
          + '<td class="ev">' + (x.current_list ? 'current list' : 'older list') + '</td></tr>';
      }).join('');
    }).join('');
    return '<section class="wb-rack" id="fleet-dti-history"><h2>DTI readings over time</h2>'
      + '<p class="wb-blurb">Every reading is kept; the omni over readings of one list is ours, not DeckCheck’s.</p>'
      + '<div class="rap-wrap"><table class="rap"><thead><tr><th>Deck</th><th>Read</th><th>DTI</th><th>Bracket</th><th></th></tr></thead><tbody>' + body + '</tbody></table></div></section>';
  }

  function recordByVersion(doc) {
    var rows = [];
    members(doc).forEach(function (r) {
      (r.record.by_version || []).forEach(function (v) {
        var feel = Object.keys(v.feel || {}).map(function (k) {
          return esc(k.slice(0, 3)) + ' ' + num(v.feel[k].mean, 1) + '<span class="ev">·' + v.feel[k].n + '</span>';
        }).join(' ');
        rows.push('<tr><th scope="row"><a href="deck.html?deck=' + encodeURIComponent(r.slug) + '">' + esc(r.slug) + '</a></th>'
          + '<td>' + (v.release ? esc(v.release) : 'V' + esc(v.version)) + ' <span class="ev">' + esc(v.date || '') + '</span></td>'
          + '<td>' + v.games + ' · ' + v.win + '–' + v.loss + (v.draw ? '–' + v.draw : '') + '</td>'
          + '<td>' + (feel || '<span class="t-none">—</span>') + '</td></tr>');
      });
      if (r.record.unmatched_games && r.record.unmatched_games.length) {
        rows.push('<tr class="is-dead"><th scope="row">' + esc(r.slug) + '</th><td class="t-none">no version matches</td>'
          + '<td>' + r.record.unmatched_games.length + ' game(s)</td><td></td></tr>');
      }
    });
    if (!rows.length) return '';
    return '<section class="wb-rack" id="fleet-record"><h2>Record by version</h2>'
      + '<p class="wb-blurb">Games joined to the list they were played on, by sha; the survey means ride beside the record.</p>'
      + '<div class="rap-wrap"><table class="rap"><thead><tr><th>Deck</th><th>Version</th><th>Record</th><th>Survey</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div></section>';
  }

  // ── (e) The rating leaderboard ───────────────────────────────────────────

  function ratingBoard(doc, ratings) {
    var rated = members(doc).filter(function (r) { return ratings[r.slug]; });
    var head = '<section class="wb-rack" id="fleet-rating"><h2>Rating <span class="chip fl-star">★ COACHING</span></h2>';
    if (!rated.length) {
      return head + '<p class="wb-blurb t-none" id="fleet-no-rating">no ratings yet — the rating is ★ coaching, written by the rating agent (`/rate-deck`), over the ◆ figures above.</p></section>';
    }
    var groups = {};
    rated.forEach(function (r) { var k = ratings[r.slug].bracket ? 'B' + ratings[r.slug].bracket.key : r.bracket.group; (groups[k] = groups[k] || []).push(r); });
    var html = Object.keys(groups).sort().map(function (g) {
      var rows = groups[g].sort(function (a, b) {
        var x = ratings[a.slug].score, y = ratings[b.slug].score;
        return x === y ? (a.slug < b.slug ? -1 : 1) : (x > y ? -1 : 1);
      });
      return '<h3 class="fl-h3">' + esc(g) + '</h3>' + rows.map(function (r) {
        var rt = ratings[r.slug];
        var comps = (rt.components || []).map(function (c) {
          return '<span class="chip" title="' + esc(c.method || '') + '">' + esc(c.name) + ' ' + num(c.sub_score, 0) + '</span>';
        }).join('');
        var withheld = (rt.withheld || []).map(function (w) { return esc(w.component); }).join(', ');
        var level = rt.uncertainty && rt.uncertainty.level ? Math.round(rt.uncertainty.level * 100) + '% ' : '';
        return '<div class="fl-rating-row"><span class="chip fl-star">★ ' + esc(gradeShown(rt)) + '</span> '
          + '<a href="deck.html?deck=' + encodeURIComponent(r.slug) + '">' + esc(r.slug) + '</a> '
          + '<b>' + num(rt.score, 1) + '</b>'
          + (rt.score_ci ? ' <span class="ev">' + level + '[' + num(rt.score_ci[0], 1) + ', ' + num(rt.score_ci[1], 1) + ']</span>' : '')
          + (rt.rank ? ' <span class="ev">' + esc(rt.rank.position) + ' of ' + esc(rt.rank.of) + '</span>' : '')
          + ' ' + ratingExtras(rt)
          + '<div class="fl-words">' + comps + (withheld ? '<span class="chip t-none">withheld: ' + withheld + '</span>' : '') + '</div>'
          + sensitivityLine(rt) + '</div>';
      }).join('');
    }).join('');
    var label = ratings[rated[0].slug].label || '';
    return head + '<p class="wb-blurb">' + rated.length + ' of ' + members(doc).length + ' sleeved decks rated. ' + esc(label) + '</p>' + html + '</section>';
  }

  // ── (f) Limits ───────────────────────────────────────────────────────────

  function limitsPanel(doc) {
    return '<section class="wb-rack" id="fleet-limits"><h2>Limits</h2><ul class="fl-limits">'
      + (doc.limits || []).map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('')
      + '</ul><p class="ev">harness v' + esc(doc.meta.harness.version) + ', ' + esc(doc.meta.harness.iterations)
      + ' games per deck, seed ' + esc(doc.meta.harness.seed) + '. ' + esc(doc.meta.no_composite) + '</p></section>';
  }

  // ── Render and boot ──────────────────────────────────────────────────────

  function render(doc, ratings, state) {
    var c = doc.comparisons, ms = memberSet(doc);
    document.getElementById('fleet').innerHTML =
      fleetTable(doc, ratings, state)
      + bracketPanels(doc)
      + ratingBoard(doc, ratings)
      + floorDtiTable(doc)
      + intervalBars(doc)
      + curvesPanel('fleet-kill', 'Kill by turn', 'Share of games with lethal by each turn, unopposed, under the frozen harness.',
                    c.kill_curves || {}, 1, function (v) { return pct(v, 0); }, ms)
      + curvesPanel('fleet-mana', 'Mana by turn', 'Mean available mana each turn under the frozen harness.',
                    c.mana_curves || {}, 10, function (v) { return num(v, 1); }, ms)
      + curvesPanel('fleet-land', 'Land drops by turn', 'Share of games that hit the land drop each turn.',
                    c.land_curves || {}, 1, function (v) { return pct(v, 0); }, ms)
      + gradesHeatmap(doc)
      + dtiHistory(doc)
      + recordByVersion(doc)
      + limitsPanel(doc);
    document.getElementById('status').textContent = '';
  }

  getJSON(ARTIFACT).then(function (doc) {
    if (!doc || !doc.decks) {
      document.getElementById('status').textContent =
        'No fleet artifact — run `manamap pilot fleet-analytics --write`.';
      return;
    }
    return Promise.all(doc.decks.map(function (r) {
      return getJSON(DECKS + r.slug + '/rating.json');
    })).then(function (list) {
      var ratings = {};
      doc.decks.forEach(function (r, i) { if (list[i] && list[i].tier === '★') ratings[r.slug] = list[i]; });
      var state = readState();
      render(doc, ratings, state);
      document.getElementById('fleet').addEventListener('click', function (ev) {
        var s = ev.target.closest && ev.target.closest('[data-sort]');
        var b = ev.target.closest && ev.target.closest('[data-bracket]');
        if (!s && !b) return;
        if (s) state.sort = s.getAttribute('data-sort');
        if (b) state.bracket = b.getAttribute('data-bracket');
        writeState(state);
        render(doc, ratings, state);
      });
    });
  });
})();
