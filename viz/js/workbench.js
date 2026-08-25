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

  function rack(title, blurb, entries, infos) {
    if (!entries.length) return '';
    return '<section class="wb-rack">'
      + '<h2>' + esc(title) + ' <span class="wb-count">' + entries.length + '</span></h2>'
      + '<p class="wb-blurb">' + esc(blurb) + '</p>'
      + '<div class="wb-grid">'
      + entries.map(function (e) { return card(e, infos[e.slug]); }).join('')
      + '</div></section>';
  }

  function render(decks, infos) {
    var locked = decks.filter(function (e) { return e.locked; });
    // A deck that has been broken down or retired is not "on the bench" waiting
    // for work — it is history, and it sorts last so the rack reads as a queue.
    var rest = decks.filter(function (e) { return !e.locked; });
    var live = rest.filter(function (e) { return !e.status; });
    var dead = rest.filter(function (e) { return e.status; });

    var html = rack('Locked', 'Built in paper and sleeved — you can play these tonight.',
                    locked, infos)
      + rack('On the bench', 'Lists, build plans and decks under research. Nothing here is '
             + 'sleeved yet.', live, infos)
      + rack('History', 'Broken down for parts, superseded, or retired. Kept as published.',
             dead, infos);

    if (!locked.length) {
      html = '<p class="wb-empty">No deck is marked as built in paper yet. '
           + '<code>manamap pilot deck-version &lt;slug&gt; paper</code> marks the version '
           + 'you have sleeved, once its list is checked in.</p>' + html;
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
      render(decks, infos);
    });
  });
})();
