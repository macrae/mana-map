/* Deck dossier — renders a deck's committed pilot artifacts.
 *
 * Every number on this page comes out of data/decks/<slug>/*.json. Nothing is
 * recomputed in the browser and nothing is hardcoded: the manual renders those same
 * artifacts as ◆ reproducible evidence, so a second implementation that drifted would
 * quietly break the tier contract. If a figure is missing from the artifact, the panel
 * says so rather than inventing it — the same posture build_manual.py takes with [TODO].
 *
 * Slug comes from ?deck=<slug>, which is also the first URL state this frontend has.
 */
(function () {
  'use strict';

  var BASE = '../data/decks/';

  // Panels are optional by design: a deck without decisions or a build plan shows
  // fewer panels, never an error.
  var FILES = {
    issue: 'issue.json', bracket: 'bracket_report.json',
    goldfish: 'goldfish_metrics.json', mana: 'mana_analysis.json',
    considering: 'considering.json', tutors: 'tutor_guide.json',
    cards: 'cards.json', buildPlan: 'build_plan.json',
    deckMap: 'deck_map.json'
  };

  function esc(v) {
    return String(v === undefined || v === null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
  }
  function pct(x, digits) {
    return typeof x === 'number' ? (x * 100).toFixed(digits === undefined ? 1 : digits) + '%' : '—';
  }
  function num(x, digits) {
    return typeof x === 'number' ? x.toFixed(digits === undefined ? 0 : digits) : '—';
  }

  function getJSON(url) {
    return fetch(url).then(function (r) { return r.ok ? r.json() : null; })
                     .catch(function () { return null; });
  }

  function badge(tier) {
    var g = { verified: '✓ RULES-VERIFIED', data: '◆ DATA-DERIVED', coach: '★ COACHING' };
    return '<span class="badge badge-' + tier + '">' + g[tier] + '</span>';
  }

  function meter(label, value, accent) {
    var w = Math.max(0, Math.min(1, typeof value === 'number' ? value : 0)) * 100;
    return '<div class="meter"><div class="meter-label"><span>' + esc(label) +
      '</span><span class="val">' + pct(value) + '</span></div>' +
      '<div class="meter-track"><div class="meter-fill" style="width:' + w.toFixed(1) +
      '%' + (accent ? ';background:' + accent : '') + '"></div></div></div>';
  }

  function facts(pairs) {
    var rows = pairs.filter(function (p) { return p[1] !== undefined && p[1] !== null; })
      .map(function (p) {
        return '<dt>' + esc(p[0]) + '</dt><dd>' + esc(p[1]) + '</dd>';
      }).join('');
    return '<div class="facts"><dl>' + rows + '</dl></div>';
  }

  function panel(id, title, promise, tiers, accent, body) {
    return '<section class="panel' + (id === 'kill' || id === 'ten' ? ' wide' : '') +
      '" id="panel-' + id + '" style="--accent:' + accent + '">' +
      '<h2>' + esc(title) + '</h2>' +
      '<div class="promise">' + esc(promise) + ' ' +
      tiers.map(badge).join(' ') + '</div>' + body + '</section>';
  }

  // ── Panels ───────────────────────────────────────────────────────────

  function bracketPanel(d) {
    var b = d.bracket;
    if (!b) return '';
    var drivers = (b.drivers || []).map(function (dr) {
      return '<li><b>Forces ' + esc(dr.forces) + '</b> — <span class="chip">' +
        esc(dr.signal) + '</span><span class="ev">' + esc(dr.detail) + '</span></li>';
    }).join('');
    var body =
      '<div style="font-family:var(--display);font-size:2.6em;line-height:1">' +
        esc(b.floor) + ' <span style="font-size:.4em;color:var(--ink-soft)">' +
        esc(b.floor_name || '') + '</span></div>' +
      '<p class="ev">A floor is what the contents are consistent with — never a verdict. ' +
        'Tutor density is reported, never scored.</p>' +
      facts([
        ['Game Changers', (b.game_changers || []).length],
        ['Two-card infinites', (b.two_card_infinites || []).length],
        ['Tutors (reported)', (b.tutors || []).length],
        ['Mass land denial', (b.mass_land_denial || []).length],
        ['Combos contained', b.combo_count]
      ]) +
      (drivers ? '<h3 class="promise" style="margin-top:14px">What drives it</h3>' +
        '<ul class="stack-list">' + drivers + '</ul>' : '');
    return panel('bracket', 'Bracket Floor', 'What table is this deck for?',
                 ['data'], 'var(--stamp-red)', body);
  }

  function manaPanel(d) {
    var m = d.mana;
    if (!m) return '';
    var L = m.lands || {}, src = (m.sources || {}).lands || {};
    var pAll = (m.on_curve_probability || {}).with_rocks_and_dorks || {};
    var pLand = (m.on_curve_probability || {}).lands_only || {};
    var colours = Object.keys(m.pips || {}).sort();
    var meters = colours.map(function (c) {
      return meter(c + ' on curve (lands + ramp)', pAll[c]);
    }).join('');
    var rows = colours.map(function (c) {
      var sh = (m.shares || {})[c] || {};
      return '<tr><th>' + esc(c) + '</th><td>' + num((m.pips[c] || {}).total_pips, 0) +
        '</td><td>' + esc(src[c] || 0) + '</td><td>' + pct(pLand[c]) + '</td><td>' +
        pct(pAll[c]) + '</td><td>' + pct(sh.pip_share, 0) + ' / ' +
        pct(sh.source_share, 0) + '</td></tr>';
    }).join('');
    var classes = Object.keys(L.classes || {}).map(function (k) {
      return '<span class="chip">' + esc(k) + ' ' + esc(L.classes[k]) + '</span>';
    }).join('');
    var notes = (m.notes || []).map(function (n) {
      return '<li>' + esc(n) + '</li>';
    }).join('');
    var body = meters +
      '<table class="data"><tr><th>Colour</th><th>Pips</th><th>Land sources</th>' +
      '<th>On curve (lands)</th><th>+ ramp</th><th>Pip / source share</th></tr>' +
      rows + '</table>' +
      facts([
        ['Lands', L.total], ['Distinct land cards', L.entries],
        ['Enter tapped', L.enters_tapped],
        ['Rocks', (m.ramp || {})['ramp:rock']], ['Dorks', (m.ramp || {})['ramp:dork']],
        ['Land ramp', (m.ramp || {})['ramp:land']]
      ]) +
      '<div style="margin-top:10px">' + classes + '</div>' +
      (notes ? '<h3 class="promise" style="margin-top:14px">What the audit flags</h3>' +
        '<ul class="stack-list">' + notes + '</ul>' : '') +
      '<div class="assumptions"><b>Hypergeometric draws, not games.</b><ul>' +
        (m.assumptions || []).map(function (a) { return '<li>' + esc(a) + '</li>'; }).join('') +
      '</ul></div>';
    return panel('mana', 'Sources Say', 'Does this mana base keep its promises?',
                 ['data'], 'var(--y2k-blue)', body);
  }

  function goldfishPanel(d) {
    var g = d.goldfish;
    if (!g) return '';
    var m = g.metrics || {}, meta = g.meta || {};
    var oh = m.opening_hand || {}, cmd = m.commander || {};
    var targets = (m.targets || []).map(function (t) {
      return meter(t.label, t.by_turn_6_rate, 'var(--tier-coach)');
    }).join('');
    var turns = Object.keys(m.land_drop_hit_rate_by_turn || {})
      .sort(function (a, b) { return +a - +b; });
    var row = function (label, fn) {
      return '<tr><th>' + label + '</th>' + turns.map(function (t) {
        return '<td>' + fn(t) + '</td>';
      }).join('') + '</tr>';
    };
    var body =
      meter('Keepable first sevens', oh.keep_first_seven_rate) +
      (cmd.cast_by_turn_6_rate !== undefined
        ? meter('Commander cast by turn 6', cmd.cast_by_turn_6_rate) : '') +
      targets +
      '<table class="data"><tr><th>Turn</th>' +
        turns.map(function (t) { return '<th>' + esc(t) + '</th>'; }).join('') + '</tr>' +
        row('Land drop', function (t) { return pct(m.land_drop_hit_rate_by_turn[t], 0); }) +
        row('Mean mana', function (t) { return num((m.mean_available_mana_by_turn || {})[t], 1); }) +
        row('Mean bodies', function (t) { return num((m.mean_bodies_by_turn || {})[t], 1); }) +
      '</table>' +
      facts([
        ['Iterations', (meta.iterations || 0).toLocaleString()],
        ['Seed', meta.seed],
        ['Commander mean cast', num(cmd.mean_cast_turn, 3)],
        ['Mean mulligans', num(oh.mean_mulligans, 2)]
      ]) +
      '<div class="assumptions"><b>Resource development, not full games.</b><ul>' +
        (meta.model_assumptions || []).map(function (a) {
          return '<li>' + esc(a) + '</li>';
        }).join('') +
      '</ul></div>';
    return panel('goldfish', 'By the Numbers', 'What can I expect, turn by turn?',
                 ['data'], 'var(--tier-data)', body);
  }

  function tenPanel(d) {
    var c = d.considering;
    if (!c) return '';
    var items = (c.ten || []).map(function (e, i) {
      var ev = e.evidence || {};
      var bits = [];
      (ev.combo_lines_opened || []).forEach(function (l) {
        bits.push('◆ completes ' + (l.cards || []).join(' + ') + ' (' + (l.status || '') + ')');
      });
      (ev.obsoletes || []).forEach(function (o) { bits.push('◆ obsoletes ' + o); });
      if ((ev.synergy_partners_in_deck || []).length) {
        bits.push('◆ synergy: ' + ev.synergy_partners_in_deck.join(', '));
      }
      if (ev.edhrec_rank) bits.push('◆ EDHREC rank ' + ev.edhrec_rank.toLocaleString());
      return '<li><span class="rank">' + (i + 1) + '.</span><b>' + esc(e.card) + '</b> ' +
        (e.role ? '<span class="chip">' + esc(e.role) + '</span>' : '') +
        (bits.length ? '<span class="ev">' + esc(bits.join(' · ')) + '</span>' : '') +
        '<span class="ev">★ ' + esc(e.why) + '</span>' +
        (e.natural_cut ? '<span class="ev">Natural cut: ' + esc(e.natural_cut) + '</span>' : '') +
        '</li>';
    }).join('');
    var body = '<p>' + esc(c.assessment) + '</p><ul class="stack-list">' + items + '</ul>';
    return panel('ten', 'The Short List', 'The only ten worth your sleeves.',
                 ['data', 'coach'], 'var(--tier-coach)', body);
  }

  function tutorPanel(d) {
    var t = d.tutors;
    if (!t || !(t.tutors || []).length) return '';
    var items = t.tutors.map(function (e) {
      var wishes = (e.targets || []).map(function (w) {
        return '<div class="step"><span class="n">→</span><b>' + esc(w.fetch) + '</b>' +
          '<span class="ev">' + esc(w.scenario) + '</span>' +
          '<span class="ev">' + esc(w.why) + '</span></div>';
      }).join('');
      return '<details class="case"><summary>' + esc(e.card) +
        ' <span class="chip">' + (e.targets || []).length + ' wishes</span></summary>' +
        '<div class="body">' + wishes +
        (e.notes ? '<p class="ev">' + esc(e.notes) + '</p>' : '') + '</div></details>';
    }).join('');
    var body = '<p>' + esc(t.assessment) + '</p>' + items;
    return panel('tutors', 'Fetch Quests', 'One wish per tutor.',
                 ['coach'], 'var(--tier-coach)', body);
  }

  function stacksPanel(d) {
    if (!d.stacks.length) return '';
    var cases = d.stacks.map(function (s) {
      var steps = ((s.resolution || {}).steps || []).map(function (st) {
        var cites = (st.citations || []).map(function (c) {
          return '<div class="cite"><b>CR ' + esc(c.rule) + '</b> — “' + esc(c.quote) + '”</div>';
        }).join('');
        return '<div class="step"><span class="n">' + esc(st.n) + '</span>' +
          esc(st.action) + '<span class="ev">' + esc(st.effect) + '</span>' + cites + '</div>';
      }).join('');
      var ck = s.checker || {};
      return '<details class="case"><summary>Case A-' + esc(s.id) + ' · ' + esc(s.title) +
        ' <span class="chip">' + esc(ck.iterations) + ' review cycle(s)</span></summary>' +
        '<div class="body"><p class="ev">' +
          esc((s.scenario || {}).question || '') + '</p>' + steps +
        '<p class="ev"><b>Result.</b> ' +
          esc(((s.resolution || {}).final_state || {}).summary || '') + '</p></div></details>';
    }).join('');
    return panel('kill', 'The Kill', 'Every claim, with the rule text behind it.',
                 ['verified'], 'var(--tier-verified)', cases);
  }

  function buildPlanPanel(d) {
    // Only hapatra has one today; the panel appears when the artifact does.
    var p = d.buildPlan;
    if (!p) return '';
    var slots = (p.slots || []).slice(0, 12).map(function (s) {
      var alts = (s.alternates || []).map(function (a) {
        return esc(a.name) + ' (Δ' + num(a.delta, 3) + ')';
      }).join(', ');
      return '<tr><td>' + esc(s.name) + '</td><td>' + esc(s.role || '') + '</td><td>' +
        num(s.score, 3) + '</td><td class="ev">' + alts + '</td></tr>';
    }).join('');
    var body = '<p class="ev">This deck came out of the deterministic builder. Showing the ' +
      'first 12 of ' + (p.slots || []).length + ' scored slots with their runners-up — a ' +
      'small delta means the scorer was nearly indifferent.</p>' +
      '<table class="data"><tr><th>Card</th><th>Role</th><th>Score</th>' +
      '<th>Runners-up</th></tr>' + slots + '</table>';
    return panel('build', 'The Builder\'s Record', 'Why did each slot get filled this way?',
                 ['data'], 'var(--slime-green)', body);
  }


  // ── The constellation ────────────────────────────────────────────────
  //
  // The same `deck_map.json` the magazine renders, drawn the same way and in the
  // SAME COLOURS — `CITY_INK` here is a transcription of `pilot/design.py`'s list,
  // and the two must not drift, or the printed map and the site disagree about
  // which territory is which while both look correct.
  //
  // What the page adds over the printed one is the thing print cannot do: hover a
  // point and the card tells you its name and its city. That was the founder's
  // first requirement for putting the map on the site at all.
  var CITY_INK = [
    ['#E4007C', '#FF66B8'], ['#1B4FD8', '#6E93F5'], ['#3FBF3F', '#8BE28B'],
    ['#C8A03C', '#EBD08A'], ['#7B2D8B', '#B77BC4'], ['#E4002B', '#FF7A93'],
    ['#0FA3A3', '#68DADA']
  ];

  function constellationPanel(d) {
    var map = d.deckMap;
    if (!map || !map.cards || !map.cards.length) return '';
    var W = 900, H = 560, PAD = 66;
    var xs = map.cards.map(function (c) { return c.x; });
    var ys = map.cards.map(function (c) { return c.y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    var k = Math.min((W - 2 * PAD) / spanX, (H - 2 * PAD) / spanY);
    var ox = (W - spanX * k) / 2 - minX * k, oy = (H - spanY * k) / 2 - minY * k;
    var at = function (c) { return [c.x * k + ox, c.y * k + oy]; };
    var ink = function (i, shade) { return CITY_INK[i % CITY_INK.length][shade || 0]; };

    var parts = ['<svg class="deck-constellation" viewBox="0 0 ' + W + ' ' + H + '">',
      '<defs><filter id="dcb" x="-30%" y="-30%" width="160%" height="160%">' +
      '<feGaussianBlur stdDeviation="18"/></filter></defs>',
      '<rect width="' + W + '" height="' + H + '" fill="#0B0A14"/>'];

    // Density: one soft disc per neighbourhood, at its centroid. A hull in SVG
    // that the page also has to hit-test is not worth the code — the disc reads
    // the same at this size and every point stays independently hoverable.
    var lobes = {};
    map.cards.forEach(function (c) {
      var key = c.city + ':' + (c.hood || 0);
      (lobes[key] = lobes[key] || []).push(at(c));
    });
    parts.push('<g filter="url(#dcb)" opacity="0.5">');
    Object.keys(lobes).forEach(function (key) {
      var pts = lobes[key];
      var cx = pts.reduce(function (a, p) { return a + p[0]; }, 0) / pts.length;
      var cy = pts.reduce(function (a, p) { return a + p[1]; }, 0) / pts.length;
      var r = 34 + Math.max.apply(null, pts.map(function (p) {
        return Math.sqrt((p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy));
      }));
      parts.push('<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) +
                 '" r="' + Math.min(r, 150).toFixed(1) + '" fill="' +
                 ink(+key.split(':')[0]) + '"/>');
    });
    parts.push('</g>');

    (map.edges || []).forEach(function (e) {
      var a = map.cards[e.a], b = map.cards[e.b];
      if (!a || !b) return;
      var pa = at(a), pb = at(b), same = a.city === b.city;
      parts.push('<line x1="' + pa[0].toFixed(1) + '" y1="' + pa[1].toFixed(1) +
        '" x2="' + pb[0].toFixed(1) + '" y2="' + pb[1].toFixed(1) + '" stroke="' +
        (same ? ink(a.city, 1) : '#8A93B5') + '" stroke-opacity="' +
        (same ? 0.36 : 0.2) + '" stroke-width="1"/>');
    });

    map.cards.forEach(function (c) {
      var p = at(c), r = c.commander ? 6 : (c.verified ? 5.5 : 4);
      // <title> is the hover, and it costs nothing: no JS, no tooltip layer, and
      // it survives with scripting off. The city name travels with the card, which
      // is the association the whole page exists to teach.
      var city = (map.regions || []).filter(function (g) {
        return g.level === 0 && g.id === 'city-' + c.city; })[0] || {};
      parts.push('<g class="dc-card"><circle cx="' + p[0].toFixed(1) + '" cy="' +
        p[1].toFixed(1) + '" r="' + r + '" fill="' +
        (c.commander ? '#FFD800' : ink(c.city, 1)) + '"' +
        (c.commander ? ' stroke="#FFD800" stroke-width="2.5" fill-opacity="1"' :
         (c.verified ? ' stroke="#fff" stroke-width="1.5"' : '')) +
        '/><circle class="dc-hit" cx="' + p[0].toFixed(1) + '" cy="' +
        p[1].toFixed(1) + '" r="11" fill="transparent"><title>' + esc(c.name) +
        ' — ' + esc(city.label || city.fallback || '') +
        (c.verified ? ' · in a verified line' : '') + '</title></circle></g>');
    });

    (map.regions || []).filter(function (g) { return g.level === 0; })
      .forEach(function (g) {
        var members = map.cards.filter(function (c) {
          return 'city-' + c.city === g.id; }).map(at);
        if (!members.length) return;
        var cx = members.reduce(function (a, p) { return a + p[0]; }, 0) / members.length;
        var cy = members.reduce(function (a, p) { return a + p[1]; }, 0) / members.length;
        var i = +g.id.split('-').pop();
        parts.push('<text x="' + cx.toFixed(1) + '" y="' + cy.toFixed(1) +
          '" text-anchor="middle" class="dc-label" stroke="#0B0A14" stroke-width="5" ' +
          'paint-order="stroke" fill="' + ink(i, 1) + '">' +
          esc((g.label || g.fallback || '').toUpperCase()) + '</text>');
      });
    parts.push('</svg>');

    var legend = (map.regions || []).filter(function (g) { return g.level === 0; })
      .sort(function (a, b) { return b.count - a.count; })
      .map(function (g) {
        var i = +g.id.split('-').pop();
        return '<li><i style="background:' + ink(i, 1) + '"></i><b>' +
          esc(g.label || g.fallback) + '</b> <span class="ev">' + g.count +
          ' cards' + (g.verified_count ? ' · ✓' + g.verified_count : '') + '</span>' +
          (g.gloss ? '<br><span class="ev">' + esc(g.gloss) + '</span>' : '') + '</li>';
      }).join('');

    var body = '<p class="ev">The deck re-laid-out from its own cards in the 128-dim ' +
      'ability space, then clustered. Hover any point for the card and its city. ' +
      'Positions are LOCAL to this deck — they are not positions on the 34,890-card ' +
      'atlas.</p>' + parts.join('') + '<ul class="dc-legend">' + legend + '</ul>';
    return panel('constellation', 'The Constellation',
                 'What shape is this deck?', ['data'], 'var(--hot-magenta)', body);
  }

  // ── Assembly ─────────────────────────────────────────────────────────

  function render(slug, d) {
    var issue = d.issue || {};
    document.title = (issue.deck_name || slug) + ' — Deck Dossier';
    document.getElementById('deckName').textContent = issue.deck_name || slug;
    document.getElementById('commanderLine').textContent =
      (issue.commander || '') + (issue.volume ? ' · Pilot\'s Manual Vol. ' +
        String(issue.volume).padStart(3, '0') : '');
    // A deck is loadable here as soon as it has a cards.json; a magazine issue is a
    // separate, later, expensive step. Linking to `../manuals/<slug>.html`
    // unconditionally sent every unpublished deck to a 404 — the manifest carries
    // `published` precisely so this link can tell the two apart.
    var link = document.getElementById('issueLink');
    if (!d.published) {
      link.hidden = true;
      link.removeAttribute('href');
    } else {
      link.hidden = false;
      link.href = '../manuals/' + slug + '.html';
    }
    // The map's Deck Lens reads ?deck=<slug> on entry, so this deep-links straight
    // into the overlay rather than dropping the reader on an unfiltered map.
    document.getElementById('lensLink').href = 'index.html?deck=' + encodeURIComponent(slug);

    var html = [
      constellationPanel(d), bracketPanel(d), manaPanel(d), goldfishPanel(d),
      tenPanel(d), tutorPanel(d), buildPlanPanel(d), stacksPanel(d)
    ].filter(Boolean).join('');
    document.getElementById('panels').innerHTML = html;
    document.getElementById('status').textContent =
      d.stacks.length + ' verified line(s) · artifacts read from data/decks/' + slug + '/';
  }

  function pickerHTML(decks, current) {
    return 'Deck: ' + decks.map(function (d) {
      var label = 'Vol. ' + String(d.volume).padStart(3, '0') + ' ' + d.slug;
      return d.slug === current ? '<b>' + esc(label) + '</b>'
        : '<a href="?deck=' + esc(d.slug) + '">' + esc(label) + '</a>';
    }).join(' · ');
  }

  function boot() {
    // The manifest is written by `manamap pilot build-index` from the same scan
    // that builds the newsstand: it carries the deck list and each deck's passing
    // stack filenames, because a browser can list neither.
    getJSON(BASE + 'index.json').then(function (manifest) {
      if (!manifest || !manifest.decks || !manifest.decks.length) {
        document.getElementById('status').textContent =
          'No deck manifest — run `manamap pilot build-index`.';
        return;
      }
      var decks = manifest.decks;
      var want = new URLSearchParams(location.search).get('deck');
      var entry = decks.filter(function (d) { return d.slug === want; })[0] || decks[0];
      document.getElementById('deckPicker').innerHTML = pickerHTML(decks, entry.slug);

      var jobs = Object.keys(FILES).map(function (k) {
        return getJSON(BASE + entry.slug + '/' + FILES[k])
          .then(function (v) { return [k, v]; });
      });
      var stackJobs = (entry.stack_files || []).map(function (f) {
        return getJSON(BASE + entry.slug + '/stacks/' + f);
      });

      return Promise.all([Promise.all(jobs), Promise.all(stackJobs)])
        .then(function (both) {
          var d = { stacks: both[1].filter(Boolean) };
          both[0].forEach(function (p) { d[p[0]] = p[1]; });
          // From the MANIFEST, not from issue.json — an unpublished deck has no
          // issue.json at all, so reading the flag off `d.issue` would always be
          // undefined and the dead-link guard would never fire.
          d.published = entry.published !== false;
          render(entry.slug, d);
        });
    }).catch(function (e) {
      document.getElementById('status').textContent = 'Could not load the dossier: ' + e;
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else { boot(); }
})();
