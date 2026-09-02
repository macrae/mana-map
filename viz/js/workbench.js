/* The Workbench — the front door.
 *
 * One question, asked before any other: WHICH DECKS CAN I PLAY TONIGHT. That is
 * not derivable from any artifact, because it is a fact about cardboard — so it
 * is authored, in `deck_versions.json`'s `paper` block, and it arrives here
 * through `data/decks/index.json` as `locked`.
 *
 * Three racks, and the split is the whole point. SLEEVED decks are built in paper
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

  // Cache-busted, because the manifest GREW A KEY. CLAUDE.md's rule is to bump
  // whenever a consumer would draw a different conclusion from the bytes, and
  // "no drafts" versus "one draft" is exactly that — a browser holding the old
  // shape would show an empty bench and be quietly wrong about your work.
  // Bump this when the manifest's shape changes again.
  //
  // 2 -> 3: the manifest grew `version`, the latest release tag, carried for
  // EVERY deck rather than only the sleeved ones. A browser holding the old
  // shape stamps no version on any unsleeved deck's art — and the stamp is how
  // this page now says which list a deck is, so the old bytes are quietly wrong
  // about the thing the page exists to tell you.
  var MANIFEST_VERSION = 3;
  var MANIFEST = '../data/decks/index.json?v=' + MANIFEST_VERSION;
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

  function chip(text, kind, html) {
    // `html` is opt-in and used by ONE caller, for the pin glyph. Everything
    // else still escapes, because a deck slug and a card name reach this from
    // artifacts and a chip is not a place to start trusting input.
    return '<span class="wb-chip' + (kind ? ' wb-' + kind : '') + '">'
         + (html ? text : esc(text)) + '</span>';
  }

  /* THE RELEASE TAG, NOT THE ORDINAL. "V3" is the number git can derive and
   * means nothing to a reader; `v1.0.2` is what the pilot ships and asks for.
   * Falls back to the ordinal when a list carries no release tag — a real state,
   * not an error, so it degrades to the number rather than to nothing. */
  function versionName(v) {
    if (!v) return '';
    return v.release ? v.release : (v.version ? 'V' + v.version : '');
  }

  /* THE STAMP OVER THE ART, and the pin is the load-bearing half of it.
   *
   * 📌 MEANS CARDBOARD AND NOTHING ELSE — this exact 99 is in sleeves and level
   * with the repo. Every other state gets the tag WITHOUT the pin, because the
   * pin is the answer to the one question this page exists to ask and spending
   * it on "there is a version tag" would make it decorative. A drifted lock is
   * amber and says how far; an unsleeved deck shows its latest release plainly.
   *
   * `in_sync` is tri-state — true, false, or null when the lock names a version
   * git no longer carries — and null must not read as "fine". */
  function versionStamp(e) {
    var paper = e.paper;
    if (paper) {
      var name = versionName(paper);
      if (paper.unresolved) {
        return '<span class="wb-stamp is-warn">' + esc(name) + ' · not in git</span>';
      }
      if (paper.in_sync) {
        return '<span class="wb-stamp is-ok"><span class="wb-pin" '
             + 'aria-hidden="true">\uD83D\uDCCC</span>' + esc(name) + '</span>';
      }
      return '<span class="wb-stamp is-warn">' + esc(name)
           + (paper.versions_behind ? ' · ' + paper.versions_behind + ' behind'
                                    : ' · drifted') + '</span>';
    }
    var tag = versionName(e.version);
    return tag ? '<span class="wb-stamp">' + esc(tag) + '</span>' : '';
  }

  /* What is left for the chip row once the version has moved onto the art: the
   * pull list, which is an ERRAND rather than an identity. */
  function lockChips(paper) {
    if (!paper || paper.unresolved || paper.in_sync) return '';
    var d = paper.drift || { pull: [], add: [] };
    return chip('pull ' + d.pull.length + ' · add ' + d.add.length, 'warn');
  }

  /* AN OPEN PROPOSAL, which the landing page could not see at all. A branch
   * surfaced here only as prose in `info.next[0]`, in the fleet table, if it
   * happened to sort first — so the one thing the pilot is actively waiting on
   * was the least visible thing on the bench. */
  function proposalOf(info) {
    return ((info && info.branches) || []).filter(function (b) {
      return b.proposal && b.state !== 'MERGED';
    })[0] || null;
  }

  function proposalChip(info) {
    var b = proposalOf(info);
    if (!b) return '';
    var left = (b.pull_list || {}).blocking || 0;
    return chip(b.proposal.as_version + ' proposed' + (left ? ' · ' + left + ' out' : ' · ready'),
                left ? 'warn' : 'ok');
  }

  /* The fallback subtitle: what the title has not already said.
   *
   * `colour_identity` and `size` are on every `info.json` without exception —
   * `deck-info` derives both from `cards.json` — so this cannot be the empty
   * string on a deck that has a list. On a deck that has none it is, and an
   * empty subtitle is the correct answer to "what else is there to say". */
  function identityLine(info) {
    var id = (info && info.colour_identity) || [];
    var size = info && info.size;
    var bits = [];
    if (id.length) bits.push(id.join(''));
    if (size) bits.push(size + ' cards');
    return esc(bits.join(' \u00b7 '));
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
      // RECENCY RIDES WITH THE RECORD. `ago()` was written for the fleet table
      // and the card threw it away, so a deck played last night and a deck
      // played in July read identically — on the page whose only question is
      // which deck to spend tonight on. A record with no date is a fact about
      // the past presented as a fact about now.
      var when = ago(r.last_played);
      out += chip(r.games + ' game' + (r.games === 1 ? '' : 's') + ' · ' + rec
                  + (when ? ' · ' + when : ''), 'ok');
    }
    /* `N verified lines` IS GONE, from here and from the fleet table's evidence
     * column. It counted passing stack scenarios — how many rules questions
     * this deck happens to have written up — which says nothing about the deck
     * and everything about which evenings the pilot spent on citations. The
     * ENGINE ratio (`verified_lines/lines`) stays in the table: that one answers
     * how much of the model is proved, which is a different question and has a
     * denominator. */
    /* A BRANCH IS WORK IN FLIGHT, and the front door could only see the subset
     * that had reached a proposal. Edgar carries six open branches and the card
     * showed none of them, so the deck with the most work under way looked
     * exactly like one with none. Neutral, deliberately: `proposalChip` is amber
     * or green because it is BLOCKED ON CARDBOARD, and having work open is not
     * being blocked. */
    var open = ((info && info.branches) || []).filter(function (b) {
      return b.state !== 'MERGED' && !b.proposal;
    }).length;
    if (open) out += chip('⑂ ' + open + ' branch' + (open === 1 ? '' : 'es'));
    if ((e.sim_runs || []).length) out += chip((e.sim_runs || []).length + ' sim');
    if ((e.experiments || []).length) out += chip((e.experiments || []).length + ' experiment');
    if ((e.prescriptions || []).length) out += chip((e.prescriptions || []).length + ' question');
    /* NOTHING TO SHOW IS TWO DIFFERENT DECKS, and a bare card cannot tell them
     * apart. A deck built this afternoon and a deck nobody has touched in six
     * months both render as a title and no chips — so the newest thing on the
     * bench looks exactly like the most neglected, which is the front door
     * getting the one question it exists to answer backwards.
     *
     * Said only when there is genuinely nothing else on the card: the moment a
     * deck has a game, a line or a run, its own evidence is the better answer
     * and this would just be noise beside it. `todo` comes from
     * `deck_status.STAGES` by way of `info.json` — the same list the dossier's
     * absent sections read, so the two surfaces cannot disagree about how far
     * along a deck is. */
    if (!out) {
      var todo = ((info && info.status && info.status.todo) || []).length;
      var done = (info && info.status && info.status.complete) || 0;
      if (todo) out += chip('new — ' + done + ' of ' + ((info.status.of) || 0)
                            + ', nothing measured yet');
    }
    return out;
  }

  /* MOVING AND REMOVING A DECK, from the page that shows the racks.
   *
   * ABSENT WITHOUT A LOCAL SERVER, never present-and-broken. `manamap serve`
   * has an `/api` the deployed GitHub Pages site does not, and the repo's
   * standing rule for that gap is `shell.js:consider` and `deck-view.js:130`:
   * a control that cannot work does not render. A disabled button on the
   * deployed page would be a promise the page cannot keep.
   *
   * THE DELETE VERDICT IS NOT COMPUTED HERE. `entry.deletable` and
   * `undeletable_because` come from `deck_delete.blockers` by way of the
   * manifest. Re-deriving "never sleeved, never played, never published" from
   * `locked` and `record.games` would be a second implementation of the
   * refusal, free to disagree with the command's — and the only way anyone
   * would find out is by pressing a button that then refuses.
   */
  function actions(e) {
    if (!window.Api || !Api.ready) return '';
    var slug = esc(e.slug);
    var out = '';
    if (e.status) {
      out += '<button class="wb-act" data-act="revive" data-slug="' + slug
           + '">Restore</button>';
    } else {
      out += '<button class="wb-act" data-act="archive" data-slug="' + slug
           + '">Archive</button>';
    }
    /* DELETE IS OFFERED OR IT IS NOT — no greyed button, and no marker either.
     *
     * A first cut printed a small "kept" beside the reasons, and it rendered on
     * NINE OF TEN CARDS: almost every deck was sleeved, played or published, so
     * the word said nothing about any particular deck and cost a line on all of
     * them. A label that is always there is chrome. The absence is the answer,
     * and `deck-delete <slug>` prints the reasons in full for anyone who wants
     * them named. */
    if (e.deletable) {
      out += '<button class="wb-act is-danger" data-act="delete" data-slug="'
           + slug + '">Delete</button>';
    }
    /* ONE GROUP, pushed right with `margin-left: auto` — NOT a flex spacer
     * between the links and the buttons. `.wb-links` wraps, and a `flex: 1 1
     * auto` spacer inside a wrapping row eats the remaining width and shoves
     * whatever follows onto its own line, so "kept" rendered as a stray word
     * under the destinations. `margin-left: auto` pushes right when there is
     * room and wraps the whole group cleanly when there is not. */
    return '<span class="wb-acts">' + out + '</span>';
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
      + actions(e)
      + '</nav>';
    /* A DECK WITH NO AUTHORED NAME PRINTS ITS COMMANDER TWICE.
     * `build_index` falls back to the commander for `deck_name`, so four decks
     * rendered "Zur the Enchanter" as both the title and the line under it — a
     * subtitle carrying no information at all. Where they differ the commander
     * is the right subtitle; where they are the same, the colour identity and
     * the size are facts the title does not already state. */
    var sub = (e.commander && e.commander !== (e.deck_name || ''))
      ? esc(e.commander)
      : identityLine(info);
    return '<div class="wb-card' + (dead ? ' is-dead' : '') + '">'
      + '<div class="wb-artwrap">' + art + versionStamp(e) + '</div>'
      + '<div class="wb-body">'
      +   heading
      +   '<div class="wb-sub">' + sub + '</div>'
      +   (dead ? '<div class="wb-dead">' + esc(dead[1]) + '</div>' : '')
      +   '<div class="wb-chips">' + lockChips(e.paper) + proposalChip(info)
      +     evidenceChips(e, info) + '</div>'
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
    cardboard: {
      label: 'Waiting on cardboard',
      key: function (e, i) {
        var b = proposalOf(i);
        // Proposals first, the closest to ready first inside that — a deck one
        // card away is a different errand from one eight cards away.
        return [e.status ? 1 : 0, b ? 0 : 1,
                b ? ((b.pull_list || {}).blocking || 0) : 1e9];
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

    // Same name the card above shows. Two surfaces on one page calling the same
    // list by two different names — "v1.0.2" on the card and "V3" in the table —
    // reads as two different things.
    var lockName = e.paper
      ? (e.paper.release ? e.paper.release : 'V' + num(e.paper.version))
      : '';
    var lock = e.paper
      ? (e.paper.unresolved ? '<span class="t-warn">' + esc(lockName) + ' · ?</span>'
         : e.paper.in_sync ? '<span class="t-ok">' + esc(lockName) + '</span>'
         : '<span class="t-warn">' + esc(lockName) + ' · ' + num(e.paper.versions_behind)
           + ' behind</span>')
      : '<span class="t-none">—</span>';

    // THE ENGINE RATIO ONLY. This used to lead with the MANIFEST's raw count of
    // passing stack scenarios and the comment said the two "disagree usefully".
    // They do not: a bare count of write-ups has no denominator, so it measures
    // how many evenings went into citations rather than anything about the deck.
    // The ratio answers how much of the engine model is PROVED, which is the
    // question that count was standing in for.
    var evidence = eng.lines
      ? num(eng.verified_lines, '0') + '/' + num(eng.lines) + ' ✓'
      : '<span class="t-none">—</span>';
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

  /* One aggregate row. Sums only what SUMS — games, wins, losses, engine lines,
   * open questions. Deliberately no averaged win rate: the runs have different
   * Ns against different pods, and a mean of rates would be a number no
   * simulation ever measured.
   *
   * BOTH HALVES OF THE RATIO, and that is the point. The evidence column now
   * carries `verified/total` per deck, and this row summed the NUMERATORS and
   * printed "23 ✓" — a total with its denominator thrown away, which is the
   * failure `net_change.METRICS` exists to prevent: a figure a reader has to
   * reconstruct gets guessed at, and the guesses go one way. `23/61` says how
   * much of the fleet's engine model is proved; `23` says nothing. */
  function totals(decks, infos) {
    var g = 0, w = 0, l = 0, v = 0, lines = 0, q = 0, sims = 0;
    decks.forEach(function (e) {
      var i = infos[e.slug] || {}, r = i.record || {}, eng = i.engine || {};
      g += r.games || 0; w += r.win || 0; l += r.loss || 0;
      v += eng.verified_lines || 0;
      lines += eng.lines || 0;
      q += (i.open_questions || []).length;
      sims += (e.sim_runs || []).length;
    });
    return '<tr class="t-total"><th scope="row">' + decks.length + ' decks</th>'
      + '<td></td>'
      + '<td>' + (g ? g + ' · ' + w + '–' + l : '<span class="t-none">none</span>') + '</td>'
      + '<td></td>'
      + '<td>' + (lines ? v + '/' + lines + ' ✓' : '<span class="t-none">—</span>')
      + '</td><td>' + sims + ' runs</td>'
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

  /* IN PROGRESS — decks that exist as a brief and no more.
   *
   * A partial build is work you must be able to put down and find again, so it
   * has to appear here or it is lost the moment the tab closes. `build-index`
   * files these under `drafts` rather than `decks`, because everything that
   * consumes `decks` assumes a 99 and a draft has none — the deck picker, the
   * dossier and `deck-info` would all offer something they cannot load.
   *
   * FIRST in the racks, and only when there are any. It is the one thing on
   * this page that is waiting on YOU rather than reporting a measurement.
   */
  function draftRack(drafts) {
    drafts = drafts || [];
    if (!drafts.length) return '';
    var tiles = drafts.map(function (d) {
      var bits = [];
      if (d.theme) bits.push(esc(d.theme));
      if (d.kept) bits.push(d.kept + ' kept');
      if (d.bracket) bits.push('bracket ' + d.bracket);
      return '<article class="wb-card wb-draft">'
        + '<div class="wb-body">'
        + '<div class="wb-title"><a href="index.html?mode=build&draft='
        + encodeURIComponent(d.slug) + '">' + esc(d.deck_name || d.slug) + '</a></div>'
        + '<div class="wb-sub">' + esc(d.commander || '') + '</div>'
        + (bits.length ? '<div class="wb-chips"><span class="wb-chip">'
            + bits.join('</span><span class="wb-chip">') + '</span></div>' : '')
        + '<div class="wb-sub">started ' + esc(d.started || '') + '</div>'
        + '</div></article>';
    }).join('');
    return '<section class="wb-rack">'
      + '<h2>In progress <span class="wb-count">' + drafts.length + '</span></h2>'
      + '<p class="wb-blurb">Started and not finished. Pick one up where you left it.</p>'
      + '<div class="wb-grid">' + tiles + '</div></section>';
  }

  /* `collapsed` renders the rack as a <details>, shut. Only the archive uses it.
   *
   * A rack is a QUEUE — the racks above it are things to do, and the archive is
   * the one that is finished. Left open it was the largest block on the page and
   * the last thing under the eye, so the front door's final impression was of
   * work the pilot had deliberately stopped. Shut, with its count on the summary,
   * it says how much history there is without spending the screen on it.
   *
   * <details> rather than a JS toggle: it is keyboard-reachable, it survives the
   * `innerHTML` replacement `render` does on every repaint with no state to
   * thread, and Cmd-F finds text inside a closed one in every current browser. */
  function rack(title, blurb, entries, infos, collapsed) {
    if (!entries.length) return '';
    var head = '<h2>' + esc(title)
      + ' <span class="wb-count">' + entries.length + '</span></h2>'
      + '<p class="wb-blurb">' + esc(blurb) + '</p>';
    var body = '<div class="wb-grid">'
      + entries.map(function (e) { return card(e, infos[e.slug]); }).join('')
      + '</div>';
    if (!collapsed) {
      return '<section class="wb-rack">' + head + body + '</section>';
    }
    return '<details class="wb-rack wb-fold"><summary>' + head + '</summary>'
      + body + '</details>';
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

  function render(decks, infos, state, drafts) {
    /* DEATH FIRST, THEN THE LOCK — and the order is the whole correctness of
     * this page. `locked` used to be filtered over the WHOLE list, so a deck
     * carrying both a lifecycle status and a paper lock landed in SLEEVED and
     * could never reach the archive. That is not hypothetical: yawgmoth-swarm
     * was broken down for parts while still locked at V5, and the one screen
     * whose job is answering WHAT CAN I PLAY TONIGHT said "you can play these
     * tonight" over a box of nothing. It is the hapatra failure
     * (`common.DECK_STATUSES`) one surface later.
     *
     * `set_lifecycle` now withdraws the lock when a deck is archived, so the
     * contradiction should not reach the wire — but the page must not depend on
     * a command having been run correctly to be right about cardboard. */
    var dead = decks.filter(function (e) { return e.status; });
    var living = decks.filter(function (e) { return !e.status; });
    var locked = living.filter(function (e) { return e.locked; });
    var live = living.filter(function (e) { return !e.locked; });

    var toggle = '<div class="wb-views">'
      + '<button class="wb-view' + (state.view === 'racks' ? ' is-on' : '')
      + '" data-view="racks">Racks</button>'
      + '<button class="wb-view' + (state.view === 'table' ? ' is-on' : '')
      + '" data-view="table">Fleet table</button></div>';

    var html;
    if (state.view === 'table') {
      html = toggle + fleetTable(decks, infos, state.sort);
    } else {
      // WAITING ON CARDBOARD SITS ABOVE SLEEVED, because it is the only rack
      // that names something the pilot is blocked on rather than something they
      // could do. A deck appears here AND in its own rack — the chip marks it
      // wherever it is, the rack collects everything you are waiting for.
      var waiting = living.filter(function (e) {
        return proposalOf(infos[e.slug]);
      });
      html = toggle
        + draftRack(drafts)
        + rack('Waiting on cardboard',
               'Decided, measured and accepted — these are lists you cannot '
               + 'sleeve yet. The pull list is on each deck\u2019s dossier.',
               waiting, infos)
        + rack('Sleeved', 'Built in paper — you can play these tonight.',
               locked, infos)
        + rack('On the bench', 'Lists, build plans and decks under research. Nothing here is '
               + 'sleeved yet.', live, infos)
        + rack('Archive',
               'Broken down for parts, superseded, or retired \u2014 and builds you '
               + 'put down. Kept as published, so you can come back and read them.',
               dead, infos, true);
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

  /* THE PROBE IS AWAITED, not fired and forgotten.
   *
   * It used to run at the bottom of this file with no one waiting on it, while
   * `render` ran the moment the manifest arrived — so whether the Archive and
   * Delete controls appeared depended on WHICH FETCH FINISHED FIRST. Caught in
   * a real browser: the buttons rendered on one load and were absent on the
   * next, same page, same server. A control that is sometimes there is worse
   * than one that is never there, because the absence stops being information.
   *
   * `Api.probe()` resolves to a boolean and NEVER rejects — "no server" is an
   * answer, not an error (`api.js:37`) — so waiting on it costs a deployed page
   * one 404 round-trip and cannot hang the render. */
  var ready = (window.Api && Api.probe) ? Api.probe() : Promise.resolve(false);

  Promise.all([getJSON(MANIFEST), ready]).then(function (both) {
    var m = both[0];
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
      render(decks, infos, state, m.drafts || []);

      // One delegated listener on the container, because `render` replaces its
      // whole innerHTML — per-button handlers would be re-bound on every
      // repaint and leak the ones they replaced.
      document.getElementById('racks').addEventListener('click', function (ev) {
        var v = ev.target.closest && ev.target.closest('[data-view]');
        var s = ev.target.closest && ev.target.closest('[data-sort]');
        var a = ev.target.closest && ev.target.closest('[data-act]');
        if (a) return fleetAction(a);
        if (!v && !s) return;
        if (v) state.view = v.getAttribute('data-view');
        if (s) { state.view = 'table'; state.sort = s.getAttribute('data-sort'); }
        writeState(state);
        render(decks, infos, state, m.drafts || []);
      });
    });
  });

  /* One verb, then a full reload.
   *
   * RELOAD RATHER THAN PATCHING THE PAGE'S OWN STATE. Archiving a deck rewrites
   * `info.json` AND the manifest AND — through `deck_is_apart` — what every
   * other deck's branch pull list says about whose cards are free. A page that
   * moved one card between two racks in memory would be showing a fleet that
   * no longer matches the artifacts it claims to render, which is the whole
   * failure this surface exists to end.
   */
  function fleetAction(btn) {
    var act = btn.getAttribute('data-act');
    var slug = btn.getAttribute('data-slug');
    if (!window.Api || !Api.ready) {
      alert('This needs a local server — run `manamap serve`.');
      return;
    }
    var call;
    if (act === 'delete') {
      // NAMED, AND TYPED BACK. The server requires `confirm` to equal the slug
      // for its own reasons; asking for it here means the pilot reads the name
      // of the thing they are removing rather than clicking past a yes/no.
      var typed = prompt('Delete ' + slug + ' permanently?\n\n'
        + 'Its directory and its Pilot\u2019s Manual page are removed and staged '
        + 'in git \u2014 not committed, so you can still review or undo it.\n\n'
        + 'Type the deck slug to confirm:');
      if (typed !== slug) return;
      call = Api.call('deck/delete', { slug: slug, confirm: slug });
    } else {
      var reason = act === 'revive' ? null
        : prompt('Archive ' + slug + '. Why? (optional \u2014 a note about the '
                 + 'decision, not a claim about the cards)') || '';
      if (reason === null && act !== 'revive') return;
      call = Api.call('deck/state',
                      { slug: slug, action: act, reason: reason || '' });
    }
    var label = btn.textContent;
    btn.disabled = true;
    btn.textContent = '\u2026';
    call.then(function () { location.reload(); })
        .catch(function (err) {
          btn.disabled = false;
          btn.textContent = label;
          // The server's sentence, VERBATIM. `deck_delete.blockers` and
          // `set_lifecycle` both refuse with a paragraph naming what to do
          // instead — "archive it", "withdraw the lock first" — and
          // paraphrasing it here would throw away the instruction and leave
          // only the rejection.
          alert(slug + ': ' + (err && err.message ? err.message : err));
        });
  }

  // The library drawer (shell.js) mounts itself on every surface and its buttons
  // reach the local server; without a probe here `Api.ready` was permanently
  // false and the drawer reported "needs a local server" while one was running.
  // The probe now happens above and the first render WAITS on it, which serves
  // the drawer as well — `probe()` memoises, so there is only ever one request.
})();
