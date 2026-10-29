/* api.js — is there a machine behind this page?
 *
 * `manamap serve` exposes `/api/` beside the static files. GitHub Pages does
 * not. So the same page is two products, and this module is the one place that
 * knows which one it is currently being.
 *
 * THE DIFFERENCE IS A FEATURE OF THE PAGE, NOT AN ACCIDENT OF THE ENVIRONMENT.
 * `CLAUDE.md` argued against a local bridge on exactly this ground — "a local
 * bridge means the deployed site and your machine run different code, and only
 * one of them is the one you test" — and the answer is not to pretend the two
 * are the same. It is that every affordance which needs the API is EXPLICITLY
 * gated on `Api.ready`, and the static build says what is missing instead of
 * offering a button that does nothing. The PRD's words: "agent-dependent
 * affordances are absent, not broken."
 *
 * PROBED ONCE, CACHED, AND NEVER RETRIED IN A LOOP. A page served from Pages
 * would otherwise spend its life issuing failing requests to an origin that has
 * no server. One probe, one answer, and a `refresh()` for the case where
 * somebody starts the server while the page is open.
 *
 * The browser makes NO model calls here and never will. It asks the local
 * server to run a named command, exactly as a terminal would; the server's
 * allow-list decides what a name means. Nothing in this file is a model client.
 */
/* ── The deck manifest's cache bust — ONE constant, four readers ──────────────
 *
 * `data/decks/index.json` was fetched with `?v=3` (workbench.js), `?v=2`
 * (build.js), `?v=` + MM.DATA_VERSION (discovery.js) and `cache: 'no-cache'`
 * (branch-view.js): four schemes for one file, so a manifest shape change
 * reached some readers and not others. Two of them were already wrong.
 *
 * It is NOT `MM.DATA_VERSION`, which is deliberately separate: that one means
 * "a consumer would draw a different conclusion from the card-map bytes" and
 * moves on a retrain. The manifest moves when its SHAPE changes. Sharing one
 * number would bust 60 MB of embeddings because a deck gained a key.
 *
 * Bump when a key is added, removed or given a new meaning.
 *   2 -> 3: the manifest grew `version`, the latest release tag, for EVERY deck
 *           rather than only the sleeved ones. A browser holding the old shape
 *           stamps no version on any unsleeved deck's art — and the stamp is
 *           how the workbench says which list a deck is.
 *   6 -> 7: every entry carries `format` (and a 60-card deck `commander: null`).
 *           The pages now READ it — Build's legality lens, the deck page's n/a
 *           panels, the workbench subtitle — and a cached manifest without the
 *           key would read a Modern deck as Commander.
 *   7 -> 8: an entry carries `links` (`{moxfield: {url, id, as_of}}`) when the
 *           deck has a `links.json`. The workbench card and the deck page's
 *           cover READ it, and a cached manifest would hide a recorded link.
 *   8 -> 9: entries carry `stage`; the Build/Discover pickers filter on it.
 *   9 -> 10: `links` may carry `deckcheck`, each link may carry its
 *            `decklist_sha256`/`version` stamp, and `has.deckcheck` says whether
 *            a Threat Index is on file (2026-10-17).
 *   10 -> 11: every entry carries `branches[]` (name, state, opened, add, out,
 *            verdict, objective grade, as_version, measured) and `branches_open`;
 *            the workbench chip and the deck page's rack read them (2026-10-19).
 *   11 -> 12: every entry carries `cards` — every name and every face in the
 *            deck's cards.json, commander included — for the Atlas's "in these
 *            decks" pill, and `has.rating` says a ★ rating.json is on file so
 *            the workbench fetches it for its grade chip only where one exists
 *            (2026-10-20). A cached manifest without `cards` would show no deck
 *            on any card, which reads as "in no deck".
 */
window.MANIFEST_VERSION = 12;
window.MANIFEST_URL = '../data/decks/index.json?v=' + window.MANIFEST_VERSION;

/* ── A branch's state badge — ONE home (2026-10-20) ──────────────────────────
 *
 * The deck page and the branch page each carried a copy of the state → badge
 * mapping, and the workbench's git-style branch block would have been the
 * third. The state is `deck_branch.branch_state`'s word, derived when
 * info.json was written; this only prints it. `b.base_moved`, when set, adds
 * the warning the deck page shows beside it. */
window.branchStateBadge = function (b) {
  var e = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]; }); };
  var st = String((b && b.state) || '');
  var key = /^MERGED/.test(st) ? 'merged' : /^PROPOSED/.test(st) ? 'proposed'
          : /^REJECT/.test(st) ? 'rejected' : /^WITHDRAW/.test(st) ? 'withdrawn' : 'open';
  return '<span class="br-state" data-state="' + key + '">' + e(st || 'OPEN') + '</span>'
       + (b && b.base_moved ? ' <span class="br-state" data-state="base-moved" title="' + e(b.base_moved) + '">base moved</span>' : '');
};

/* ── A deck's live branches, as rows — ONE renderer, two surfaces (2026-10-28) ──
 *
 * The workbench drew these privately (`branchRow` in workbench.js); Build's Change
 * tab lists the same deck's live branches, and a copy there would have been the
 * second reading of one artifact. Moved here beside `branchStateBadge`, which it
 * calls. The rows read `info.json`'s `branches` (`deck_info._branches`) and print
 * them; nothing is derived.
 *
 * `opts.withDeck` prefixes the deck's name (the workbench's fleet-wide rack).
 * `opts.profile` adds one line of what was MEASURED — how many rows rose, fell and
 * made no call, and the guard's row when the branch named one — for Build, where
 * the pilot is deciding what to change next. The workbench leaves it off: there
 * the pilot called the deltas "messy, illegible" (2026-10-20). Never the
 * report's merge word as a headline — a branch is read as a whole profile. */
window.BranchRows = (function () {
  var e = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]; }); };

  /* The branches that are WORK: open or proposed. Merged, rejected and withdrawn
   * ones live in the deck page's decided fold. Newest first, as `deck_info._branches`
   * orders them. */
  function live(info) {
    return ((info && info.branches) || []).filter(function (b) {
      return b && !/^(MERGED|REJECT|WITHDRAW)/.test(String(b.state || ''));
    });
  }

  function profileLine(b) {
    var v = b.verdict;
    var bits = [];
    if (v && (v.rose != null || v.fell != null)) {
      bits.push((v.rose || 0) + ' better · ' + (v.fell || 0) + ' worse · ' + (v.no_call || 0) + ' no call');
    } else {
      bits.push('not measured yet');
    }
    var o = b.objective;
    if (o && o.axis) {
      var g = b.objective_grade || {};
      bits.push('guard ' + o.axis + ' ' + o.op + ' ' + o.value +
        (g.state ? ': ' + g.state + (typeof g.reading === 'number' ? ' (' + (+g.reading.toFixed(3)) + ')' : '') : ''));
    }
    return '<p class="wb-br-profile">' + e(bits.join(' · ')) + '</p>';
  }

  /* ONE BRANCH, IN PLAIN WORDS. The headline is Jarvis's summary when one is on the
   * branch (`deck-branch <slug> summary <name> --title --blurb`): the branch in
   * words, then one sentence on what it would do and why. Without one the
   * branch's name stands in and the row says so. */
  function row(deck, b, opts) {
    opts = opts || {};
    var slug = encodeURIComponent(deck.slug), name = encodeURIComponent(b.name);
    var href = 'branch.html?deck=' + slug + '&branch=' + name;
    var sm = b.summary || {};
    var head = (opts.withDeck ? '<a class="wb-br-deck" href="deck.html?deck=' + slug + '">' + e(deck.deck_name || deck.slug) + '</a> ' : '')
      + '<a class="wb-br-name" href="' + href + '">' + e(sm.title || b.name) + '</a> '
      + window.branchStateBadge(b)
      + (b.proposal && b.proposal.as_version ? ' <span class="wb-br-as">proposed as ' + e(b.proposal.as_version) + '</span>' : '')
      + (sm.title ? ' <span class="wb-br-slug" title="the branch’s name on the bench">' + e(b.name) + '</span>' : '');
    var blurb = sm.blurb
      ? '<p class="wb-br-blurb">' + e(sm.blurb) + '</p>'
      : '<p class="wb-br-blurb wb-br-unsummarised">No summary yet — ask Jarvis to review the branch and it writes one.</p>';
    var links = '<nav class="wb-br-links">'
      + '<a href="' + href + '#swaps">the swaps</a>'
      + '<a href="' + href + '#measured">what was measured</a>'
      + '<a href="' + href + '#net-change">the net change</a>'
      + '<a href="deck.html?deck=' + slug + '#panel-branches">all of this deck’s branches</a>'
      + '</nav>';
    return '<div class="wb-branch-row" data-state="' + e(String(b.state || 'OPEN').toLowerCase()) + '">'
      + '<div class="wb-br-head">' + head + '</div>'
      + blurb
      + (opts.profile ? profileLine(b) : '')
      + links + '</div>';
  }

  return { live: live, row: row };
})();

/* ── A deck's identity facts — ONE reading, two surfaces (2026-10-28) ─────────
 *
 * The workbench card and Build's deck header both say which deck this is: its
 * name, its version, its rung, its bracket, its ★ grade, its record and
 * DeckCheck's Threat Index. The workbench had the only reading, private to its
 * IIFE; Build's header would have been a second. These return DATA (text,
 * title, warn) and each page wraps them in its own markup.
 *
 * ABSENT MEANS ABSENT: every reader returns null when the file does not carry
 * the figure, and the caller omits it. Nothing here computes a score — every
 * value is a field of `index.json` (the manifest entry), `info.json`
 * (`deck-info --write`) or `rating.json`, printed. */
window.DeckFacts = (function () {
  /* The release tag, not the ordinal; the ordinal when there is no tag. */
  function versionName(v) {
    if (!v) return '';
    return v.release ? v.release : (v.version ? 'V' + v.version : '');
  }

  /* THE DECK'S NAME. `build_index` falls back to the commander when no authored
   * `deck_name` exists (issue.json), so sharknado reads "Shabraz, the Skyshark"
   * — the commander twice. Where the name IS the commander, the slug is the one
   * name the bench actually calls the deck by, and it is printed as a name
   * ("sharknado" -> "Sharknado"); `authored` says which one this is. */
  function displayName(e) {
    if (!e) return { text: '', authored: false };
    var name = e.deck_name || '';
    if (name && name !== e.commander) return { text: name, authored: true };
    var slug = String(e.slug || '');
    var words = slug.split('-').filter(Boolean).map(function (w) {
      return w.charAt(0).toUpperCase() + w.slice(1);
    });
    return { text: words.join(' ') || name, authored: false };
  }

  /* The version a reader should know this list by. A paper lock wins (it is what
   * is in the sleeves); its drift is a warning. Otherwise the latest release. */
  function version(e) {
    if (!e) return null;
    var p = e.paper;
    if (p && (p.release || p.version)) {
      var name = versionName(p);
      if (p.unresolved) return { text: name, title: 'paper lock ' + name + ' — not in git', warn: true };
      if (p.in_sync === false) {
        return { text: name + (p.versions_behind ? ' · ' + p.versions_behind + ' behind' : ' · drifted'),
                 title: 'the sleeves hold ' + name + '; the list has moved since', warn: true };
      }
      return { text: name, title: 'sleeved as V' + p.version + (p.in_sync ? ', in sync with the list' : ''),
               warn: false };
    }
    var tag = versionName(e.version);
    return tag ? { text: tag, title: 'latest release', warn: false } : null;
  }

  /* The ladder rung: Sleeved / On the bench / Brewing, or null for a pile. */
  function rung(e) {
    var r = window.deckRung ? window.deckRung(e) : (e && e.stage);
    var label = r && window.DECK_RUNGS ? window.DECK_RUNGS[r] : null;
    return label ? { key: r, text: label } : null;
  }

  /* The bracket the deck is built for (the floor when no target is set), with the
   * floor and DeckCheck's read in the title. Warn when the floor outgrew the
   * target or DeckCheck read an older list. */
  function bracket(info) {
    var b = (info && info.bracket) || {}, dc = info && info.deckcheck;
    var n = b.target || b.floor || (dc && dc.bracket);
    if (!n) return null;
    var title = [];
    if (b.floor) title.push('floor ' + b.floor + (b.floor_name ? ' (' + b.floor_name + ')' : ''));
    if (b.target) title.push('target ' + b.target);
    if (dc && dc.bracket) title.push('DeckCheck B' + dc.bracket + (dc.score != null ? ' ' + dc.score + '/' + dc.max_score : ''));
    return { text: 'B' + n, title: title.join(' · '),
             warn: b.within_target === false || !!(dc && dc.stale_list) };
  }

  /* The ★ grade from rating.json — ★ coaching over ◆ inputs; the rank names its n. */
  function grade(rt) {
    if (!rt) return null;
    var g = rt.grade_display || rt.grade;
    if (!g) return null;
    var rk = rt.rank || {};
    var title = 'rating ' + (rt.score != null ? String(rt.score) : '') + (rk.position ? ' · ' + rk.position + ' of ' + rk.of
              + ' in ' + ((rt.bracket && rt.bracket.key) ? 'B' + rt.bracket.key : 'its bracket') : '')
              + ' · ★ coaching over ◆ inputs';
    return { text: '★ ' + g, title: title };
  }

  /* W–L from the captain's log (`info.record`), a draw column only when one
   * exists. No logged game is no record — null, never "0–0". */
  function record(info) {
    var r = (info && info.record) || {};
    if (!r.games) return null;
    var wl = [r.win || 0, r.loss || 0].concat(r.draw ? [r.draw] : []).join('–');
    var games = r.games + ' game' + (r.games === 1 ? '' : 's');
    return { text: wl, games: games, last: r.last_played || null,
             title: games + (r.last_played ? ', last ' + r.last_played : '') };
  }

  /* DeckCheck's Threat Index — dated external evidence, with its as_of. Warn when
   * it describes an older list or is past `max_age_days`. */
  function dti(info) {
    var dc = info && info.deckcheck;
    if (!dc || dc.score == null) return null;
    var warn = '';
    if (dc.stale_list) warn = 'older list';
    else if (dc.as_of && dc.max_age_days) {
      var days = Math.floor((Date.now() - Date.parse(dc.as_of)) / 86400000);
      if (days > dc.max_age_days) warn = days + ' d old';
    }
    return { text: 'DTI ' + dc.score + '/' + dc.max_score, as_of: dc.as_of || null, warn: warn,
             title: 'DeckCheck Threat Index' + (dc.bracket ? ', B' + dc.bracket : '')
                    + (dc.as_of ? ', read ' + dc.as_of : '') + ' — external evidence, never a gate' };
  }

  return { versionName: versionName, displayName: displayName, version: version, rung: rung,
           bracket: bracket, grade: grade, record: record, dti: dti };
})();

/* ── Land on the hash after an async render (2026-10-20) ─────────────────────
 *
 * The deck and branch pages build their panels after a fetch, so the browser's
 * own jump to `#panel-branches` or `#measured` fires before the element exists
 * and every deep link from the workbench landed at the top. A chip that names
 * something must LEAD there; this is the other half of that rule. Safe to call
 * more than once — it scrolls only when the target is on the page. */
window.scrollToHash = function () {
  var id = (location.hash || '').slice(1);
  if (!id) return false;
  var el = document.getElementById(id);
  if (!el) return false;
  el.scrollIntoView({block: 'start'});
  return true;
};

/* ── The shared formatters (2026-10-18) ──────────────────────────────────────
 *
 * `esc`, `num`, `pct`, `signed` and `getJSON` were copied into workbench.js,
 * deck-view.js and branch-view.js with two different `num` signatures between
 * them. The fleet page is the first to read them from here; the three private
 * copies stay until each page is moved over on its own (deck-view.js:640 warns
 * why a second `pct` in scope hoists over the first). Semantics are deck-view's:
 * `num(x, digits)` fixed decimals, `pct(x, digits)` 0-1 to "x.x%", `signed`
 * with its sign, `getJSON` revalidating and resolving null on any failure.
 */
window.Fmt = {
  esc: function (v) {
    return String(v === undefined || v === null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
  },
  num: function (x, digits) {
    if (x === undefined || x === null || isNaN(x)) return '—';
    return Number(x).toFixed(digits === undefined ? 2 : digits);
  },
  pct: function (x, digits) {
    if (x === undefined || x === null || isNaN(x)) return '—';
    return (Number(x) * 100).toFixed(digits === undefined ? 1 : digits) + '%';
  },
  signed: function (x, digits) {
    if (x === undefined || x === null || isNaN(x)) return '—';
    var s = Number(x).toFixed(digits === undefined ? 2 : digits);
    return (x > 0 ? '+' : '') + s;
  },
  getJSON: function (url) {
    return fetch(url, { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }
};

/* ── What a legal deck IS, per format — mirrors `pilot/formats.py:FORMATS` ────
 *
 * MIRRORS PYTHON, and `tests/test_viz_format_specs.py` reads this literal as JSON
 * and holds every format and every field to `formats.FORMATS`: `size` is
 * `deck_size`, `exact` is `exact_size`, `copies` is `max_copies`, `sideboard` is
 * `sideboard_size`. A silent duplicate of a config table is how the two sides
 * drift, so the literal stays strict JSON (quoted keys, one format per line) for
 * the test to parse, and nothing else in `viz/` may hard-code a 99 or a 60.
 *
 * Here rather than in `mana-map.js` because all seven pages need it and only one of
 * them has an `MM`: the deck page and the workbench load `api.js` and never
 * `mana-map.js`. `MM.FORMAT_SPECS` is the same object, aliased at export.
 */
window.FORMAT_SPECS = {
  "commander": {"name": "Commander", "size": 100, "exact": true, "copies": 1, "commanders": 1, "sideboard": 0},
  "standard": {"name": "Standard", "size": 60, "exact": false, "copies": 4, "commanders": 0, "sideboard": 15},
  "modern": {"name": "Modern", "size": 60, "exact": false, "copies": 4, "commanders": 0, "sideboard": 15},
  "pioneer": {"name": "Pioneer", "size": 60, "exact": false, "copies": 4, "commanders": 0, "sideboard": 15},
  "pauper": {"name": "Pauper", "size": 60, "exact": false, "copies": 4, "commanders": 0, "sideboard": 15}
};
Object.keys(window.FORMAT_SPECS).forEach(function (k) { Object.freeze(window.FORMAT_SPECS[k]); });
Object.freeze(window.FORMAT_SPECS);

/* A format's spec by key. No key means Commander — every deck that predates the
 * field is one — and so does a key this table does not know, because a page that
 * threw on it would render nothing at all. Python refuses unknown names
 * (`formats.get`); the browser only ever reads names Python already accepted. */
window.formatSpec = function (key) {
  return window.FORMAT_SPECS[String(key || '').toLowerCase()] || window.FORMAT_SPECS.commander;
};

/* ── Which decks you can WORK ON — the ladder rung, `promote.stage` ───────────
 *
 * The manifest's `stage` is "dev" | "bench" | "sleeved", or null for a deck in
 * a pile (broken down, retired, superseded). ONE predicate for every picker, so
 * Build and Discover can never disagree about which decks are on offer. A
 * manifest older than `stage` falls back to the old reading — any lifecycle
 * status is no rung, a lock is sleeved, everything else the bench — never to
 * "show everything", which is the bug this exists to end. */
window.DECK_RUNGS = {dev: 'Brewing', bench: 'On the bench', sleeved: 'Sleeved'};
window.deckRung = function (e) {
  if (!e) return null;
  return 'stage' in e ? e.stage : (e.status ? null : (e.locked ? 'sleeved' : 'bench'));
};
window.isWorkableDeck = function (e) {
  return !!window.DECK_RUNGS[window.deckRung(e)];
};

/* `<option>`s for a format picker — ONE renderer for both pickers (Build's new-deck
 * form and Discover's "60 cards and no commander" hint), so a format reads the
 * same wherever it is chosen: "Modern — 60+, 4-of". `keys` defaults to all five. */
window.formatOptionsHtml = function (keys, selected) {
  var esc = function (s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  };
  return (keys || Object.keys(window.FORMAT_SPECS)).map(function (k) {
    var s = window.FORMAT_SPECS[k];
    if (!s) return '';
    return '<option value="' + esc(k) + '"' + (k === selected ? ' selected' : '') + '>' +
      esc(s.name) + ' — ' + s.size + (s.exact ? '' : '+') +
      (s.copies === 1 ? ', singleton' : ', ' + s.copies + '-of') + '</option>';
  }).join('');
};

window.Api = (function () {
  'use strict';

  var state = { probed: false, ready: false, commands: [], reason: null };
  var probing = null;

  function base() {
    // Same origin, always. A configurable host would be the first step toward
    // this page talking to a machine that is not yours.
    return '/api';
  }

  /* Is the local API there? Resolves to a boolean and never rejects — "no
   * server" is an ANSWER, not an error, and every caller treats it as one. */
  function probe() {
    if (state.probed) return Promise.resolve(state.ready);
    if (probing) return probing;
    probing = fetch(base() + '/health', { method: 'GET', cache: 'no-store' })
      // A 404 is the DEPLOYED shape, not a malfunction: a static host serves
      // its own not-found page for `/api/health`. Reporting that as "the API
      // answered oddly" is the message every visitor to the published site
      // would see, and it describes a fault that is not happening.
      .then(function (r) { return r.ok ? r.json() : 'absent'; })
      .then(function (doc) {
        if (doc === 'absent') {
          state.ready = false;
          state.reason = 'no local server — run `manamap serve`';
          return false;
        }
        state.ready = !!(doc && doc.result && doc.result.ok);
        state.reason = state.ready ? null
          : 'something is answering /api but it is not this bench';
        return state.ready ? fetch(base() + '/', { cache: 'no-store' })
          .then(function (r) { return r.json(); })
          .then(function (d) { state.commands = d.commands || []; return true; })
          .catch(function () { return true; }) : false;
      })
      .catch(function () {
        // The expected path on a deployed page. Not an error, not logged as one.
        state.ready = false;
        state.reason = 'no local server — run `manamap serve`';
        return false;
      })
      .then(function (ok) { state.probed = true; probing = null; return ok; });
    return probing;
  }

  /* Run one allow-listed command. Rejects with a readable message.
   *
   * Callers must have checked `ready` first: this is deliberately NOT a
   * silent no-op when the API is absent, because a promise that resolves to
   * nothing is how an affordance ends up looking broken instead of absent.
   */
  function call(name, payload) {
    return fetch(base() + '/' + name, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      body: JSON.stringify(payload || {}),
    }).then(function (r) {
      return r.json().then(function (doc) {
        if (!r.ok || doc.error) throw new Error(doc.error || ('HTTP ' + r.status));
        return doc.result;
      });
    });
  }

  function has(name) { return state.commands.indexOf(name) !== -1; }

  return {
    probe: probe,
    call: call,
    has: has,
    get ready() { return state.ready; },
    get probed() { return state.probed; },
    get commands() { return state.commands.slice(); },
    get reason() { return state.reason; },
    // For the case where the server starts after the page loads.
    refresh: function () { state.probed = false; probing = null; return probe(); },
  };
})();
