/* The Workbench — the front door.
 *
 * One question, asked before any other: WHICH DECKS CAN I PLAY TONIGHT. That is
 * not derivable from any artifact, because it is a fact about cardboard — so it
 * is authored, in `deck_versions.json`'s `paper` block, and it arrives here
 * through `data/decks/index.json` as `locked`.
 *
 * Two racks, and the split is the whole point. LOCKED decks are built in paper
 * and sleeved; everything else is on the bench — build plans, broken-down lists,
 * decks being researched. A deck that exists only as JSON and a deck you can put
 * on a table are different objects, and the old picker (one line of text links)
 * could not tell you which was which.
 *
 * Nothing is computed here. The manifest is written by `build-index` and the
 * per-deck chips come from `info.json`, which `deck-info --write` composes — the
 * same rule deck-view.js follows, for the same reason: a second implementation
 * that drifted would quietly break the evidence contract.
 */
(function () {
  'use strict';

  var MANIFEST = '../data/decks/index.json';
  var BASE = '../data/decks/';

  function esc(v) {
    return String(v === undefined || v === null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
  }

  /* `cache: 'no-cache'` forces revalidation on every artifact fetch, and it is
   * load-bearing rather than defensive. These files are NOT content-addressed the
   * way `MM.DATA`'s URLs are — locking a deck rewrites `index.json` and bumps no
   * version constant anywhere — so a heuristically-cached copy shows yesterday's
   * answer with no way for the page to know. Caught in a browser: two decks were
   * locked on disk and the rack still said "no deck is marked as built in paper".
   * They are small JSON; a revalidation round-trip is the right trade. */
  function getJSON(url) {
    return fetch(url, { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  function chip(text, kind) {
    return '<span class="wb-chip' + (kind ? ' wb-' + kind : '') + '">' + esc(text) + '</span>';
  }

  /* The lock, said in one phrase. `in_sync` is tri-state — true, false, or null
   * when the lock names a version git no longer carries — and null must not read
   * as "fine". */
  function lockChips(paper) {
    if (!paper) return '';
    if (paper.unresolved) return chip('SLEEVED V' + paper.version + ' · not in git', 'warn');
    if (paper.in_sync) return chip('V' + paper.version + ' · in sync', 'ok');
    var d = paper.drift || { pull: [], add: [] };
    var behind = paper.versions_behind;
    return chip('V' + paper.version + (behind ? ' · ' + behind + ' behind' : ' · drifted'), 'warn')
         + chip('pull ' + d.pull.length + ' · add ' + d.add.length, 'warn');
  }

  /* What the bench knows about this deck. Counts only — a number here is an
   * invitation to open the deck, never a claim about it. */
  function evidenceChips(e, info) {
    var out = '';
    // `record`, NOT `status`. `deck-info` writes the games under `info.record`
    // and uses `info.status` for the stage counts, so reading `status.games`
    // was a chip that had never rendered once — dead code that looked live,
    // and by the time it was found two decks had a real logged game the front
    // door structurally could not show. A logged game is the whole point of
    // the bench; it does not get to be the thing that silently goes missing.
    var r = (info && info.record) || {};
    if (r.games) {
      var rec = [r.win || 0, r.loss || 0].join('–');
      out += chip(r.games + ' game' + (r.games === 1 ? '' : 's') + ' · ' + rec, 'ok');
    }
    if (e.verified) out += chip(e.verified + ' verified line' + (e.verified === 1 ? '' : 's'));
    if ((e.sim_runs || []).length) out += chip((e.sim_runs || []).length + ' sim');
    if ((e.experiments || []).length) out += chip((e.experiments || []).length + ' experiment');
    if ((e.prescriptions || []).length) out += chip((e.prescriptions || []).length + ' question');
    return out;
  }

  function card(e, info) {
    // `status` is the lifecycle TRIPLE for a dead deck and null for a live one —
    // [id, headline, blurb]. Only the headline belongs on a card.
    var dead = Array.isArray(e.status) ? e.status : null;
    var art = e.image
      ? '<img class="wb-art" src="' + esc(e.image) + '" alt="" loading="lazy">'
      : '<div class="wb-art wb-art-none"></div>';
    // THREE DESTINATIONS, ALL NAMED — and no card-sized hit area at all.
    //
    // The card used to be one big link to the dossier with the manual as a small
    // link inside it, because "a card that opens two different things depending
    // on where you click is the interaction bug this repo has already fixed
    // once, in the atlas." That rule is right and this honours it rather than
    // working around it: the answer to "I want the art to open the manual" is
    // not to make the art mean something different from the body — it is to
    // stop anything being implicit. Every destination is a labelled link, so
    // there is no invisible target left to be surprised by.
    //
    // The title goes to the MANUAL because that is what a pilot opens a deck to
    // read. The map link carries `?deck=`, the documented inbound contract that
    // lands in Build with the deck loaded rather than on an unfiltered atlas.
    var slug = encodeURIComponent(e.slug);
    var hasPage = !!(e.has && e.has.page);
    var title = esc(e.deck_name || e.slug);
    var heading = hasPage
      ? '<h3><a class="wb-title" href="../manuals/p/' + slug + '.html">' + title + '</a></h3>'
      : '<h3>' + title + '</h3>';
    // A link to a page that does not exist is worse than no link, so the manual
    // is omitted rather than dead when a deck has none — the same rule the
    // dossier's own manual link follows.
    var links = '<nav class="wb-links">'
      + (hasPage ? '<a href="../manuals/p/' + slug + '.html">Manual</a>' : '')
      + '<a href="deck.html?deck=' + slug + '">Dossier</a>'
      + '<a href="index.html?deck=' + slug + '">On the map</a>'
      + '</nav>';
    return '<div class="wb-card' + (dead ? ' is-dead' : '') + '">'
      + art
      + '<div class="wb-body">'
      +   heading
      +   '<div class="wb-sub">' + esc(e.commander || '') + '</div>'
      +   (dead ? '<div class="wb-dead">' + esc(dead[1]) + '</div>' : '')
      +   '<div class="wb-chips">' + lockChips(e.paper) + evidenceChips(e, info) + '</div>'
      +   links
      + '</div></div>';
  }

  /* ── THE FLEET TABLE ────────────────────────────────────────────────────
   *
   * Every figure below was ALREADY on the wire. This page fetches one
   * `info.json` per deck and used exactly one field of it; the rest was
   * downloaded, parsed and thrown away on every load. So the table costs no
   * new request, no new artifact, and no new computation — it stops
   * discarding.
   *
   * `next` is the payoff. `deck-info` already derives a to-do list per deck,
   * each line naming the command that would settle it (deck_info.py:254-315),
   * and it is committed to `info.json` and read by nothing. It is the last
   * column because it is the answer to the question the page exists to ask.
   */

  function num(v, dash) { return (v === 0 || v) ? String(v) : (dash || '—'); }

  /* Days since an ISO date, or null. The table sorts on the raw date and
   * PRINTS this, because "2026-08-22" is a fact and "2d ago" is the reading. */
  function ago(iso) {
    if (!iso) return null;
    var d = Math.floor((Date.now() - Date.parse(iso + 'T00:00:00')) / 86400000);
    if (!isFinite(d) || d < 0) return null;
    return d === 0 ? 'today' : d === 1 ? 'yesterday' : d + 'd ago';
  }

  /* The four questions, as predicates over (entry, info). Each returns a sort
   * key where SMALLER sorts first, so the deck that most needs you is at the
   * top. Dead decks always sink: history is not a queue. */
  var SORTS = {
    played: {
      label: 'Recently played',
      key: function (e, i) {
        var d = i && i.record && i.record.last_played;
        return [e.status ? 1 : 0, d ? -Date.parse(d) : 1e15];
      }
    },
    logs: {
      label: 'Needs game logs',
      key: function (e, i) {
        var g = (i && i.record && i.record.games) || 0;
        // A sleeved deck with no games is the sharpest case: you can play it
        // tonight and nothing knows how it plays. An unlocked deck with no
        // games is just unbuilt, which is a different problem.
        return [e.status ? 1 : 0, g, e.locked ? 0 : 1];
      }
    },
    analysis: {
      label: 'Needs analysis',
      key: function (e, i) {
        var st = (i && i.status) || {};
        var missing = (st.of || 0) - (st.complete || 0);
        var bad = ((i && i.engine && i.engine.critic === 'fail') ? 1 : 0)
                + ((i && i.diagnosis && i.diagnosis.stale) ? 1 : 0)
                + ((st.invalid || []).length) + ((st.stale || []).length);
        return [e.status ? 1 : 0, -(missing + bad * 3)];
      }
    },
    optimisations: {
      label: 'Optimisations identified',
      key: function (e, i) {
        var p = (i && i.prescriptions) || {};
        var open = (p.count || 0) - (p.answered || 0);
        var under = (i && i.audit && (i.audit.under || []).length) || 0;
        var qs = (i && i.open_questions || []).length;
        return [e.status ? 1 : 0, -(open * 10 + under * 2 + qs)];
      }
    }
  };

  function cmpKey(a, b) {
    for (var i = 0; i < Math.max(a.length, b.length); i++) {
      var x = a[i] === undefined ? 0 : a[i], y = b[i] === undefined ? 0 : b[i];
      if (x !== y) return x < y ? -1 : 1;
    }
    return 0;
  }

  function row(e, info) {
    var i = info || {};
    var rec = i.record || {}, st = i.status || {}, eng = i.engine || {};
    var dg = i.diagnosis || {}, sim = i.simulation || {}, pr = i.prescriptions || {};
    var slug = encodeURIComponent(e.slug);

    var record = rec.games
      ? esc(rec.games + ' · ' + (rec.win || 0) + '–' + (rec.loss || 0))
      : '<span class="t-none">none</span>';
    var when = ago(rec.last_played);

    var lock = e.paper
      ? (e.paper.unresolved ? '<span class="t-warn">V' + num(e.paper.version) + ' · ?</span>'
         : e.paper.in_sync ? '<span class="t-ok">V' + num(e.paper.version) + '</span>'
         : '<span class="t-warn">V' + num(e.paper.version) + ' · ' + num(e.paper.versions_behind)
           + ' behind</span>')
      : '<span class="t-none">—</span>';

    // Verified lines come from the MANIFEST (a count of passing stacks); the
    // engine ratio comes from info.json and answers a different question —
    // how much of the model is proved. Both, because they disagree usefully.
    var evidence = num(e.verified, '0') + ' ✓';
    if (eng.lines) evidence += ' · ' + num(eng.verified_lines, '0') + '/' + num(eng.lines);
    if (eng.critic === 'fail') evidence += ' <span class="t-warn">critic</span>';
    if (dg.skeptic === 'fail') evidence += ' <span class="t-warn">skeptic</span>';

    var table = sim.games
      ? esc(sim.win_rate + ' · n' + sim.games)
        + (sim.stale ? ' <span class="t-warn">stale</span>' : '')
      : '<span class="t-none">—</span>';

    var openQ = (i.open_questions || []).length;
    var openP = (pr.count || 0) - (pr.answered || 0);
    var open = openQ || openP
      ? esc((openQ ? openQ + 'q' : '') + (openQ && openP ? ' · ' : '') + (openP ? openP + '?' : ''))
      : '<span class="t-none">—</span>';

    // THE NEXT STEP GETS ITS OWN ROW, spanning the table.
    //
    // It was a column first, and that was the wrong shape: every other column
    // is `nowrap`, so their sum pushed past 100% and the browser put the ONE
    // thing the page exists to say behind a horizontal scrollbar. A percentage
    // width could not win that fight. The deeper mistake was treating a
    // SENTENCE as a cell — it is prose, it wants the full measure, and giving
    // it a row of its own makes the metrics above it scannable instead of
    // squeezed.
    var next = (i.next && i.next.length) ? esc(i.next[0]) : '';
    var dead = e.status ? ' is-dead' : '';

    return '<tr class="t-head' + dead + '">'
      + '<th scope="row"><a href="' + (e.has && e.has.page
            ? '../manuals/p/' + slug + '.html' : 'deck.html?deck=' + slug) + '">'
        + esc(e.deck_name || e.slug) + '</a>'
        + '<span class="t-sub">' + esc(e.commander || '') + '</span></th>'
      + '<td>' + lock + '</td>'
      + '<td>' + record + (when ? '<span class="t-sub">' + esc(when) + '</span>' : '') + '</td>'
      + '<td>' + num(st.complete, '—') + (st.of ? '/' + st.of : '') + '</td>'
      + '<td>' + evidence + '</td>'
      + '<td>' + table + '</td>'
      + '<td>' + open + '</td>'
      + '</tr>'
      + (next ? '<tr class="t-nextrow' + dead + '"><td colspan="7" class="t-next">'
                + '<a href="deck.html?deck=' + slug + '">' + next + '</a></td></tr>'
              : '');
  }

  /* One aggregate row. Sums only what SUMS — games, wins, losses, verified
   * lines, open questions. Deliberately no averaged win rate: the runs have
   * different Ns against different pods, and a mean of rates would be a number
   * no simulation ever measured. */
  function totals(decks, infos) {
    var g = 0, w = 0, l = 0, v = 0, q = 0, sims = 0;
    decks.forEach(function (e) {
      var i = infos[e.slug] || {}, r = i.record || {};
      g += r.games || 0; w += r.win || 0; l += r.loss || 0;
      v += e.verified || 0; q += (i.open_questions || []).length;
      sims += (e.sim_runs || []).length;
    });
    return '<tr class="t-total"><th scope="row">' + decks.length + ' decks</th>'
      + '<td></td>'
      + '<td>' + (g ? g + ' · ' + w + '–' + l : '<span class="t-none">none</span>') + '</td>'
      + '<td></td><td>' + v + ' ✓</td><td>' + sims + ' runs</td>'
      + '<td>' + q + 'q</td></tr>';
  }

  function fleetTable(decks, infos, sortKey) {
    var s = SORTS[sortKey] || SORTS.played;
    var sorted = decks.slice().sort(function (a, b) {
      return cmpKey(s.key(a, infos[a.slug]), s.key(b, infos[b.slug]))
        || (a.slug < b.slug ? -1 : 1);
    });
    var tabs = Object.keys(SORTS).map(function (k) {
      return '<button class="wb-sort' + (k === sortKey ? ' is-on' : '')
        + '" data-sort="' + k + '">' + esc(SORTS[k].label) + '</button>';
    }).join('');
    return '<section class="wb-rack"><div class="wb-sorts">' + tabs + '</div>'
      + '<div class="wb-tablewrap"><table class="wb-table"><thead><tr>'
      + '<th scope="col">Deck</th><th scope="col">Paper</th><th scope="col">Record</th>'
      + '<th scope="col">Stages</th><th scope="col">Evidence</th><th scope="col">Table</th>'
      + '<th scope="col">Open</th>'
      + '</tr></thead><tbody>'
      + sorted.map(function (e) { return row(e, infos[e.slug]); }).join('')
      + totals(decks, infos)
      + '</tbody></table></div></section>';
  }

  function rack(title, blurb, entries, infos) {
    if (!entries.length) return '';
    return '<section class="wb-rack">'
      + '<h2>' + esc(title) + ' <span class="wb-count">' + entries.length + '</span></h2>'
      + '<p class="wb-blurb">' + esc(blurb) + '</p>'
      + '<div class="wb-grid">'
      + entries.map(function (e) { return card(e, infos[e.slug]); }).join('')
      + '</div></section>';
  }

  /* View and sort live in the URL, not in a closure, so a fleet sorted by
   * "needs game logs" is a link somebody can send. Same reason `?deck=` and
   * `?mode=` are URL params on the map: a view worth reaching twice is worth
   * addressing. `replaceState` rather than `pushState` — flipping a sort is
   * not a navigation and should not stack up in the back button. */
  function readState() {
    var p = new URLSearchParams(window.location.search);
    var view = p.get('view') === 'table' ? 'table' : 'racks';
    var sort = SORTS[p.get('sort')] ? p.get('sort') : 'played';
    return { view: view, sort: sort };
  }

  function writeState(s) {
    var p = new URLSearchParams(window.location.search);
    if (s.view === 'table') { p.set('view', 'table'); p.set('sort', s.sort); }
    else { p.delete('view'); p.delete('sort'); }
    var q = p.toString();
    history.replaceState(null, '', window.location.pathname + (q ? '?' + q : ''));
  }

  function render(decks, infos, state) {
    var locked = decks.filter(function (e) { return e.locked; });
    // A deck that has been broken down or retired is not "on the bench" waiting
    // for work — it is history, and it sorts last so the rack reads as a queue.
    var rest = decks.filter(function (e) { return !e.locked; });
    var live = rest.filter(function (e) { return !e.status; });
    var dead = rest.filter(function (e) { return e.status; });

    var toggle = '<div class="wb-views">'
      + '<button class="wb-view' + (state.view === 'racks' ? ' is-on' : '')
      + '" data-view="racks">Racks</button>'
      + '<button class="wb-view' + (state.view === 'table' ? ' is-on' : '')
      + '" data-view="table">Fleet table</button></div>';

    var html;
    if (state.view === 'table') {
      html = toggle + fleetTable(decks, infos, state.sort);
    } else {
      html = toggle
        + rack('Locked', 'Built in paper and sleeved — you can play these tonight.',
               locked, infos)
        + rack('On the bench', 'Lists, build plans and decks under research. Nothing here is '
               + 'sleeved yet.', live, infos)
        + rack('History', 'Broken down for parts, superseded, or retired. Kept as published.',
               dead, infos);
      if (!locked.length) {
        html = toggle + '<p class="wb-empty">No deck is marked as built in paper yet. '
             + '<code>manamap pilot deck-version &lt;slug&gt; paper</code> marks the version '
             + 'you have sleeved, once its list is checked in.</p>'
             + html.slice(toggle.length);
      }
    }
    document.getElementById('racks').innerHTML = html;
    document.getElementById('status').textContent =
      decks.length + ' deck(s) · ' + locked.length + ' locked';
  }

  getJSON(MANIFEST).then(function (m) {
    if (!m || !m.decks) {
      document.getElementById('status').textContent =
        'No deck manifest — run `manamap pilot build-index`.';
      return;
    }
    var decks = m.decks;
    // `info.json` carries the games and the record. Fetched per deck rather than
    // folded into the manifest, because `deck-info` owns those figures and a
    // second copy in the manifest is a second thing to go stale.
    return Promise.all(decks.map(function (e) {
      return getJSON(BASE + e.slug + '/info.json');
    })).then(function (list) {
      var infos = {};
      decks.forEach(function (e, i) { infos[e.slug] = list[i]; });

      var state = readState();
      render(decks, infos, state);

      // One delegated listener on the container, because `render` replaces its
      // whole innerHTML — per-button handlers would be re-bound on every
      // repaint and leak the ones they replaced.
      document.getElementById('racks').addEventListener('click', function (ev) {
        var v = ev.target.closest && ev.target.closest('[data-view]');
        var s = ev.target.closest && ev.target.closest('[data-sort]');
        if (!v && !s) return;
        if (v) state.view = v.getAttribute('data-view');
        if (s) { state.view = 'table'; state.sort = s.getAttribute('data-sort'); }
        writeState(state);
        render(decks, infos, state);
      });
    });
  });
})();
