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
    buildPlan: 'build_plan.json', deckMap: 'deck_map.json',
    // The join: `deck-info --write`'s shape, composed from every other artifact.
    // The page renders it rather than re-deriving anything, so it cannot disagree
    // with the command that owns each figure. `cards.json` was fetched here for
    // years and read by nothing; it is gone.
    info: 'info.json', engine: 'engine.json', versions: 'versions.json',
    // Fixed filename, so no manifest entry is needed — the browser cannot
    // list `threat/` but it does not have to.
    targeting: 'threat/targeting.json'
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

  /* ── An absent section says what it is and how to get it ────────────────
   *
   * Every panel below opened with `if (!x) return ''`, so a section with no
   * artifact VANISHED. On a deck with everything that is right — the page shows
   * what has been measured. On a deck you built five minutes ago it is wrong in
   * a way that matters: `zur-enchantress` rendered nine fewer sections than
   * `radagast` and said nothing about the difference, so a NEW deck was
   * indistinguishable from a BROKEN one, and the page gave no hint that the
   * missing half is a sequence you can walk.
   *
   * The command comes from `deck_status.STAGES` by way of `info.status.todo`.
   * It is deliberately NOT a lookup table here: that sequence is the one thing
   * `deck_status` exists to be the single statement of, and a copy in JavaScript
   * would be the second — free to drift, and drifting silently.
   */
  function todoFor(d, stage) {
    var todo = (((d.info || {}).status || {}).todo) || [];
    for (var i = 0; i < todo.length; i++) if (todo[i].stage === stage) return todo[i];
    return null;
  }

  /* A command you can take away. There is no Run button: these write tracked
   * artifacts and some of them cost 45 minutes or a quarter of a million
   * tokens, so the page hands you the line rather than spending on your behalf.
   */
  function command(how) {
    if (!how) return '';
    if (how.indexOf('AUTHORED:') === 0) {
      return '<p class="ev todo-authored">' + esc(how) + '</p>';
    }
    return '<button class="todo-cmd" title="Copy" onclick="Deck.copy(this)">' +
      '<code>' + esc(how) + '</code><span class="todo-copy">copy</span></button>';
  }

  /* What the local server will run, if there is one. Populated once, from the
   * server's own table — the page does not decide what is cheap enough to
   * press, because that judgement belongs beside the commands. */
  var measures = null;
  var drafts = [];          // authored files the server will DRAFT, never author

  function runnable(d, stage) {
    if (!measures || !measures[stage] || !window.Api || !Api.ready) return null;
    var m = measures[stage];
    // `needs` is reported, not assumed. A button that fails because of an
    // unstated dependency is worse than one that explains itself, and two of
    // the four measurements genuinely have an order.
    var missing = (m.needs || []).filter(function (n) {
      var stem = n.replace('.json', '');
      var byArtifact = { goldfish_targets: 'targets', goldfish_metrics: 'goldfish' };
      var s = byArtifact[stem];
      return s ? todoFor(d, s) !== null : false;
    });
    return { what: m.what, blockedBy: missing };
  }

  function absent(id, title, promise, accent, stage, d, what) {
    var t = todoFor(d, stage);
    // Not a lifecycle stage, or the deck has it and something else is wrong:
    // stay silent rather than inventing a reason.
    if (!t) return '';
    var run = runnable(d, stage);
    var action = '';
    if (run && !run.blockedBy.length) {
      action = '<button class="todo-run" onclick="Deck.measure(this, ' +
        JSON.stringify(d.slug).replace(/"/g, '&quot;') + ', ' +
        JSON.stringify(stage).replace(/"/g, '&quot;') + ')">Measure it now</button>';
    } else if (run) {
      /* A BLOCKED PANEL THAT CAN UNBLOCK ITSELF SAYS SO.
       *
       * "Needs goldfish_targets.json first" is true and is a dead end: that
       * file is authored, and the pilot's next act was to invent a JSON shape
       * from nothing. The server will DRAFT it — derived from contained combo
       * lines and role axes, marked as a draft, and reported as an unedited
       * scaffold by the validator on every run until somebody rewrites it. So
       * the blocker becomes the button, and the button is honest about handing
       * back a draft rather than an answer. */
      var blocker = { 'goldfish_targets.json': 'targets' }[run.blockedBy[0]];
      action = '<p class="ev todo-blocked">Needs ' + esc(run.blockedBy.join(', ')) +
        ' first.</p>';
      if (blocker && drafts.indexOf(blocker) !== -1 && todoFor(d, blocker)) {
        action += '<button class="todo-run todo-draft" onclick="Deck.draft(this, ' +
          JSON.stringify(d.slug).replace(/"/g, '&quot;') + ', ' +
          JSON.stringify(blocker).replace(/"/g, '&quot;') +
          ')">Draft it — a starting file to edit</button>';
      }
    }
    return panel(id, title, promise, [], accent,
      '<div class="todo"><p class="todo-what">Not yet on this deck — ' +
      esc(what || t.what) + '</p>' + action + command(t.how) + '</div>');
  }

  /* ── THE BRIEF: what the deck is TRYING to be ──────────────────────────
   *
   * A staged, tracked artifact that had no surface anywhere. `deck_status`
   * reports it, `build_deck` consumes it, and neither `info.json` nor this page
   * had ever shown one — so the only way to read a deck's own intent was to open
   * the JSON. It is rendered FIRST among the reference panels because it is the
   * thing every other panel is evidence for or against.
   *
   * It carries NO tier badge on purpose. A brief is neither measured (◆) nor
   * rules-verified (✓) — it is authored intent, and giving it a badge would put
   * an evidence mark on a statement of what somebody wants.
   *
   * THE DECK AND ITS BRIEF MAY DISAGREE, and during a refactor they always do.
   * The panel says so rather than implying the 99 already matches.
   */
  /* ── BRANCHES: a candidate 99 the pilot cannot yet sleeve ──────────────
   *
   * Sits beside the brief, because a branch is what the brief looks like as a
   * list. It renders the four-way sourcing split and, above all, THE BUY LIST —
   * the question a pilot actually has in front of a candidate deck is "what is
   * this going to cost me", and until now nothing answered it anywhere.
   *
   * No tier badge, same reason as the brief: a branch is intent, not evidence.
   */
  /* Three states where there were two. `mergeable` / `N to source` could not
   * tell a decision from an experiment — both rendered as a shopping list. The
   * state comes from `deck_branch.branch_state`, which derives it and stores
   * nothing, so a proposal un-blocks itself when a card lands in a box. */
  /* THE PULL LIST. `source()` has computed this since branches shipped and
   * nothing presented it as a list you take to a shop or a box. The buckets are
   * separate because they cost different things: BUY is money, UNSLEEVE takes a
   * deck apart, PROXY is a decision already recorded, FREE is cardboard already
   * in a pile. One number read as one problem. */
  var PULL = [
    ['buy', 'To buy', 'nobody owns a copy'],
    ['unsleeve', 'To unsleeve', 'out of a deck that is still together'],
    ['proxy', 'Proxied', 'you decided to move these across your own decks'],
    ['free', 'Free', 'only in a broken-down or retired deck'],
    ['box', 'From a box', 'already yours']
  ];

  function pullList(pl) {
    var out = '<h4>The pull list <span class="ev">' +
      (pl.blocking ? pl.blocking + ' still blocking' : 'nothing left to find') +
      '</span></h4>';
    if (pl.procurement && pl.procurement.note) {
      out += '<p class="ev">' + esc(pl.procurement.note) + '</p>';
    }
    PULL.forEach(function (g) {
      var rows = pl[g[0]] || [];
      if (!rows.length) return;
      out += '<h5 class="pull-head">' + esc(g[1]) + ' <span class="ev">(' +
        rows.length + ') ' + esc(g[2]) + '</span></h5>' +
        '<ul class="pull-list pull-' + g[0] + '">' + rows.map(function (r) {
          return '<li>' + esc(r.name) + ((r.where || []).length
            ? ' <span class="ev">' + esc(r.where.join(', ')) + '</span>' : '') +
            '</li>';
        }).join('') + '</ul>';
    });
    return out;
  }

  function branchBadge(b) {
    if (b.proposal && b.state) {
      var cls = b.state === 'PROPOSED · READY' ? 'branch-ok'
        : (b.state === 'PROPOSED · BLOCKED' ? 'branch-wait' : 'branch-block');
      return '<span class="' + cls + '">' + esc(b.state) + ' \u2192 ' +
             esc(b.proposal.as_version) + '</span>';
    }
    return b.mergeable ? '<span class="branch-ok">mergeable</span>'
      : '<span class="branch-block">' + (b.unsourced || []).length +
        ' to source</span>';
  }

  function branchPanel(d) {
    var bs = ((d.info || {}).branches) || [];
    if (!bs.length) return '';
    var body = bs.map(function (b) {
      if (b.unreadable) {
        return '<div class="branch"><h3>' + esc(b.name) +
               '</h3><p class="ev">This branch\u2019s list will not parse.</p></div>';
      }
      var c = b.counts || {};
      // THE DOSSIER SUMMARISES; branch.html decides. This panel is about a deck
      // that EXISTS and a branch is a proposal with a bill attached, so the
      // decision surface is its own page and this links to it rather than
      // growing into the busiest panel here.
      var href = 'branch.html?deck=' + encodeURIComponent(d.slug) +
                 '&branch=' + encodeURIComponent(b.name);
      var out = '<div class="branch"><h3><a href="' + href + '">' + esc(b.name) +
        '</a>' +
        branchBadge(b) + '</h3>';
      if (b.why) out += '<p class="branch-why">' + esc(b.why) + '</p>';
      out += facts([
        ['opened', b.opened || '\u2014'],
        ['vs the current list', '+' + b.add + ' \u2212' + b.out],
        ['size', String(b.size)],
        ['already in the deck', String(c.in_deck || 0)],
        ['in a box', String(c.box || 0)],
        ['sleeved in another deck', String(c.elsewhere || 0)],
        ['to buy', String(c.buy || 0)],
      ]);
      if (b.proposal && b.pull_list) {
        out += pullList(b.pull_list);
      } else if ((b.unsourced || []).length) {
        out += '<h4>Not yet sourced <span class="ev">' + b.unsourced.length +
               '</span></h4>' + list(b.unsourced.map(esc), 'branch-buy');
      }
      out += '<p class="ev branch-note">A branch is a candidate list, not the deck. ' +
             'It merges only when every added card is sourced \u2014 ' +
             '<code>deck-branch ' + esc(d.slug) + ' merge ' + esc(b.name) + '</code>.<br>' +
             '<a href="' + href + '">The net change, and whether it met its ' +
             'objective \u2192</a></p>';
      return out + '</div>';
    }).join('');
    // NAMED FOR WHAT THEY ARE. "Branches" reads as part of the deck; these are
    // proposals that may never be built, and one of them — ur-dragon's treasure
    // refactor — was measured, found worse and deleted after its design brief
    // had already spent weeks describing the deck on this very page.
    return panel('branches',
                 bs.length === 1 ? 'Further exploration' : 'Further explorations',
                 'Candidate lists that are NOT this deck. Each links to its own '
                 + 'net-change report \u2014 what it costs, what it buys, and '
                 + 'whether it met the objective it declared.',
                 [], '#7ba05b', body);
  }

  /* ── THE ROSTER: the whole list, grouped like The 99 ───────────────────
   *
   * The dossier could show you a picture of the deck (the constellation), what
   * it is trying to be (the brief) and what it costs (the branch) — and never
   * the cards. `cards.json` was dropped from this page's fetches years ago as
   * "read by nothing", and nothing replaced it.
   *
   * IT IS THE CONSTELLATION'S OTHER FORM, not a second artifact. Both read
   * `deck_map.json`, both group by city, and both take their colour from
   * `CITY_INK` AT THE SAME INDEX — which is the load-bearing part, copied from
   * `design.py:city_head`'s argument: a grid grouped by city under plain
   * headings is a second taxonomy the reader has to reconcile with the picture
   * beside it. Same ink, same index, by construction.
   *
   * NO ENGINE-STAGE CHIPS, unlike the printed `card_tile`. A branch inherits the
   * DECK's `engine.json`, which describes the old list, so a stage chip on a
   * card the model never placed would be a claim nobody made.
   */
  function rosterFor(d) {
    // A branch roster when there is one, otherwise the deck's own. The branch is
    // the more interesting answer: it is the list you cannot yet sleeve.
    var bs = ((d.info || {}).branches) || [];
    var b = bs.filter(function (x) { return !x.unreadable && x.cards; })[0];
    if (b && d.branchMap) return { map: d.branchMap, branch: b };
    return { map: d.deckMap, branch: null };
  }

  function cardRef(name, prov) {
    // Hover preview is CSS-only, the way the manual does it — no positioning
    // code, no tooltip layer. The image comes from `Shell.cardImageUrl`, which
    // already carries the double-faced-card front-face retry.
    var url = (window.Shell && Shell.cardImageUrl)
      ? Shell.cardImageUrl(name, 'normal') : null;
    var pop = url ? '<img class="card-pop" src="' + esc(url) + '" alt="" loading="lazy">' : '';
    var mark = '', tail = '';
    if (prov) {
      if (prov.is_new) mark = '<span class="rost-new" title="new in this branch">+</span>';
      if (prov.state === 'buy') tail = '<span class="rost-buy">to buy</span>';
      else if (prov.state === 'box') tail = '<span class="rost-have">in a box</span>';
      else if (prov.state === 'elsewhere') {
        /* `free` means every deck holding it is broken down or retired — the
         * card is loose, so it costs nothing and blocks nothing. Rendered the
         * same as a contested one it read as a deck to take apart. */
        var who = (prov.where || []).map(function (h) {
          return h.slug + (h.locked ? ' ◆' : '') + (h.apart ? ' (in a pile)' : '');
        }).join(', ');
        tail = '<span class="rost-' + (prov.free ? 'have' : 'elsewhere') + '">' +
               esc(who) + '</span>';
      }
    }
    return '<li class="rost-row">' + mark +
      '<a class="cardref" tabindex="0">' + esc(name) + pop + '</a>' + tail + '</li>';
  }

  function rosterPanel(d) {
    var got = rosterFor(d);
    var map = got.map;
    if (!map || !map.cards || !map.regions)
      return absent('roster', 'The roster', 'Every card, grouped by what it is like.',
                    '#8a7fd0', 'map', d);
    var prov = {};
    (got.branch ? got.branch.cards : []).forEach(function (r) { prov[r.name] = r; });
    var cityOf = {};
    map.cards.forEach(function (c) { cityOf[c.name] = c.city; });
    // Ordered by size, exactly as `render_the_99` does: the deck's centre of
    // mass leads.
    var cities = map.regions.filter(function (r) { return r.level === 0; })
      .slice().sort(function (a, b) {
        return (b.count - a.count) || (a.id < b.id ? -1 : 1);
      });
    var seen = {}, body = '';
    cities.forEach(function (r) {
      var index = parseInt(String(r.id).split('-').pop(), 10);
      var members = map.cards.filter(function (c) {
        return c.city === index && !c.commander;
      });
      if (!members.length) return;
      var pair = CITY_INK[index % CITY_INK.length];
      var seal = r.verified_count
        ? '<span class="rost-seal" title="named in a verified line">✓' +
          r.verified_count + '</span>' : '';
      body += '<h3 class="rost-city" style="--city:' + pair[0] + ';--city-lt:' + pair[1] + '">' +
        '<span class="rost-chip"></span>' + esc(r.label || r.fallback || r.id) +
        '<span class="rost-count">' + members.length + '</span>' + seal + '</h3>';
      if (r.gloss) body += '<p class="rost-gloss">' + esc(r.gloss) + '</p>';
      body += '<ul class="rost-list">' + members.map(function (c) {
        seen[c.name] = 1;
        return cardRef(c.name, prov[c.name]);
      }).join('') + '</ul>';
    });
    // A CARD THE MAP COULD NOT PLACE STILL GETS A SEAT. `render_the_99` ends
    // with the same group, and a roster that silently drops a card is worse
    // than no roster.
    var stray = map.cards.filter(function (c) {
      return !c.commander && !seen[c.name];
    });
    if (stray.length) {
      body += '<h3 class="rost-city rost-stray"><span class="rost-chip"></span>Unplaced' +
        '<span class="rost-count">' + stray.length + '</span></h3>' +
        '<ul class="rost-list">' + stray.map(function (c) {
          return cardRef(c.name, prov[c.name]);
        }).join('') + '</ul>';
    }
    var cmdr = map.cards.filter(function (c) { return c.commander; });
    if (cmdr.length)
      body = '<p class="rost-cmdr">' + cmdr.map(function (c) {
        return cardRef(c.name, prov[c.name]);
      }).join('').replace(/<\/?li[^>]*>/g, '') + '</p>' + body;
    var title = got.branch ? 'The roster — branch ' + esc(got.branch.name) : 'The roster';
    var promise = got.branch
      ? 'Every card in the candidate list, and where it comes from.'
      : 'Every card, grouped by what it is like.';
    return panel('roster', title, promise, [], '#8a7fd0', body);
  }

  /* ── THE VITALS ────────────────────────────────────────────────────────
   *
   * Strategy-relative by construction: every figure is read against what THIS
   * deck declares, and nothing here ranks it against another deck. That is what
   * makes a grade possible at all — the aggregate refusal in `benchmark.py`
   * stands, because `speed` spans 400x across the fleet and would rank a combo
   * deck last for playing correctly.
   *
   * The fleet band is shown as CONTEXT and never as the verdict.
   */
  // `pct` already exists at the top of this file and handles a non-number. A
  // second `function pct` here would HOIST OVER IT and silently change every
  // other panel's formatting — `node --check` is perfectly happy with two.
  /* Like `facts`, but the VALUE is already HTML — a bold figure and a muted
   * interval. `facts` escapes, correctly, because its twelve other callers pass
   * plain text; widening it would make every one of them a place markup could
   * arrive from data. */
  function vitRows(pairs) {
    var rows = pairs.filter(function (p) { return p[1] != null; })
      .map(function (p) {
        return '<dt>' + esc(p[0]) + '</dt><dd>' + p[1] + '</dd>';
      }).join('');
    return '<div class="facts"><dl>' + rows + '</dl></div>';
  }

  function reading(r) {
    if (!r) return '<span class="ev">—</span>';
    return '<b>' + pct(r.rate) + '</b> <span class="vit-ci">[' +
      pct(r.ci95[0]) + ', ' + pct(r.ci95[1]) + ']</span>';
  }

  function vitalsPanel(d) {
    var v = (d.info || {}).diagnostic;
    if (!v) return absent('vitals', 'The vitals',
                          'Engine online, stall risk and the mana under both.',
                          '#4c8fbd', 'diagnostic', d);
    var out = '';
    var e = v.engine || {};
    out += '<h3>Engine</h3>';
    if (!e.available) {
      out += '<p class="ev vit-absent">Not measured — ' + esc(e.why || '') + '</p>';
    } else {
      var rows = ['3', '5', '8'].map(function (t) {
        return ['online by turn ' + t, reading((e.online_by_turn || {})[t])];
      });
      if (e.any_route_by_turn)
        rows.push(['any kill route by 8', reading(e.any_route_by_turn['8'])]);
      out += vitRows(rows);
      if (e.bottleneck)
        out += '<p class="vit-bottleneck"><b>Bottleneck:</b> ' +
          esc(e.bottleneck.label) + ' — ' + reading(e.bottleneck.by_turn_three) +
          ' by turn three</p>';
      out += '<p class="ev">Measured against what this deck DECLARES, and counted ' +
             'over per-iteration assembly — not composed from marginals, which ' +
             'would understate an engine whose components share cards.</p>';
    }
    var s = v.stall || {};
    if (s.two_in_a_row) {
      out += '<h3>Stall</h3>' + vitRows([
        ['two turns in a row with nothing castable', reading(s.two_in_a_row)],
        ['mana-short / hand-empty',
         esc((s.cause || {}).mana_short + ' / ' + (s.cause || {}).hand_empty)],
      ]);
      if (s.fleet) out += '<p class="ev">' + esc(s.fleet) + '</p>';
    }
    var m = v.mana || {};
    if (m.missed_land_drop_by_five) {
      out += '<h3>Mana</h3>' + vitRows([
        ['missed a land drop by turn 5', reading(m.missed_land_drop_by_five)],
        ['mulliganed', reading(m.mulliganed)],
      ]);
      if (m.fleet) out += '<p class="ev">' + esc(m.fleet) + '</p>';
      if (m.correlated) out += '<p class="ev vit-note">' + esc(m.correlated) + '</p>';
    }
    var h = v.harness || {};
    out += '<p class="ev vit-note">' + (h.iterations || '?') +
      ' games, seed ' + (h.seed || '?') + '. No pod and no opponent: this ' +
      'measures a DECK, not a table, and is not a win rate.</p>';
    return panel('vitals', 'The vitals',
                 'Engine online, stall risk and the mana under both.',
                 [], '#4c8fbd', out);
  }

  function briefPanel(d) {
    // From `d.info`, not a second fetch of brief.json: info.json is this page's
    // data model and the composition already carries the brief, so reading the
    // raw file here would be a second source free to disagree with the first.
    var b = (d.info || {}).brief;
    if (!b) return absent('brief', 'The brief', 'What this deck is trying to be.',
                          '#c4a747', 'brief', d);
    var out = '';
    if (b.playstyle) out += '<p class="brief-lede">' + esc(b.playstyle) + '</p>';
    if (b.commander_rationale)
      out += '<h3>Why this commander</h3><p>' + esc(b.commander_rationale) + '</p>';
    if (b.mana) out += '<h3>Mana</h3><p>' + esc(b.mana) + '</p>';
    if (b.win_conditions) out += '<h3>How it wins</h3><p>' + esc(b.win_conditions) + '</p>';
    if ((b.design_rules || []).length)
      out += '<h3>Design rules</h3><ol class="brief-rules">' +
        b.design_rules.map(function (r) { return '<li>' + esc(r) + '</li>'; }).join('') +
        '</ol>';
    var t = b.targets || {}, keys = Object.keys(t);
    if (keys.length) {
      out += '<h3>Targets</h3>' + facts(keys.map(function (k) {
        var v = t[k];
        if (typeof v === 'number' && v > 0 && v < 1) v = pct(v, 1);
        return [k.replace(/_/g, ' '), String(v)];
      }));
    }
    if ((b.must_include || []).length)
      out += '<h3>Must include <span class="ev">' + b.must_include.length + '</span></h3>' +
        list(b.must_include.map(esc), 'brief-in');
    if ((b.must_exclude || []).length)
      out += '<h3>Must exclude <span class="ev">' + b.must_exclude.length + '</span></h3>' +
        list(b.must_exclude.map(esc), 'brief-out');
    if (b.notes) out += '<h3>Notes</h3><p class="ev">' + esc(b.notes) + '</p>';
    out += '<p class="ev brief-caveat">Authored intent, not a measurement. The 99 and the ' +
           'brief may disagree — during a rebuild they will.</p>';
    return panel('brief', 'The brief', 'What this deck is trying to be.', [], '#c4a747', out);
  }

  function bracketPanel(d) {
    var b = d.bracket;
    if (!b) return absent('bracket', 'Bracket', 'The power floor the contents are consistent with.',
      'var(--tier-data)', 'bracket', d, 'the computed power floor, and what drives it');
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
    if (!m) return absent('mana', 'The mana', 'Hypergeometric colour sources.',
      'var(--tier-data)', 'mana', d, 'colour sources and castability, priced against Karsten');
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
    if (!g) return absent('goldfish', 'The goldfish', 'Seeded Monte Carlo over resource development.',
      'var(--tier-data)', 'goldfish', d, 'how fast it develops, over 10,000 seeded games');
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
    if (!t || !(t.tutors || []).length) return absent('tutors', 'The tutors', 'What to fetch, and when.',
      'var(--tier-coach)', 'tutors', d, 'what each tutor should go and get, by board state');
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
    if (!d.stacks.length) return absent('kill', 'The lines', 'Rules-verified, or not claimed.',
      'var(--tier-verified)', 'stacks', d, 'a board resolved step by step with rules citations — the only fact tier');
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
    if (!map || !map.cards || !map.cards.length)
      return absent('map', 'The constellation', "This deck's own layout, in function space.",
        'var(--tier-data)', 'map', d, 'where this deck sits in function space, cut into cities');
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


  // ── The workbench half ───────────────────────────────────────────────
  //
  // Everything below reads `info.json` — the shape `deck-info` composes and
  // `deck-info --write` puts on disk. It is deliberately NOT re-derived here: the
  // CLI owns each figure, and a second implementation in the browser is how the
  // deck-builder's scorer drifted from the pipeline's on five of six factors.

  function list(items, cls) {
    return '<ul class="' + (cls || 'wb-list') + '">' +
      items.map(function (i) { return '<li>' + i + '</li>'; }).join('') + '</ul>';
  }

  // A mean without its interval is a number that describes no game. Measured on
  // kianne: arm B's commander damage read mean 17.42 with a median of 0, the whole
  // difference being two games out of twelve.
  function figure(m) {
    if (!m || typeof m.mean !== 'number') return '—';
    var out = String(m.mean);
    if (typeof m.median === 'number' && m.median !== m.mean) out += ' (med ' + m.median + ')';
    if (m.ci95) out += ' ci95 [' + m.ci95[0] + ', ' + m.ci95[1] + ']';
    if (m.n) out += ' n=' + m.n;
    return out;
  }

  function nextPanel(d) {
    var info = d.info;
    if (!info || !info.next || !info.next.length) return '';
    return panel('next', 'What to do next',
      'Derived from a condition that is true right now.', ['data'],
      'var(--tier-data)',
      list(info.next.map(esc)) +
      '<p class="ev">Every line names the command that would settle it. None of it is ' +
      'judgment about the deck — that is the doctor\'s, behind /prescribe.</p>');
  }

  function recordPanel(d) {
    // `versions.json` is built at DEPLOY time, not committed: versions are a git
    // walk and the commit that changes decklist.txt gets its sha after anything
    // written in the same commit, so a committed copy is one version behind
    // forever. Absent locally is normal — the panel simply does not render, which
    // is what every panel here does when its artifact is missing.
    var v = d.versions;
    if (!v || !v.versions || !v.versions.length) return '';
    var rows = v.versions.map(function (ver) {
      var cur = ver.version === v.current_version;
      var rec = ver.record || {};
      return (cur ? '<b>' : '') + 'V' + ver.version + (cur ? '</b>' : '') +
        ' <span class="ev">' + esc(ver.first_date || '') + '</span> ' +
        (ver.tags && ver.tags.length ? '<span class="chip">' + esc(ver.tags.join(', ')) + '</span> ' : '') +
        // What MOVED is the whole point of a version list. `in`/`out` are card
        // names; the counts are what fits on a row, and the subject says why.
        ((ver['in'] || []).length || (ver.out || []).length
          ? ' <span class="chip">+' + (ver['in'] || []).length +
            ' \u2212' + (ver.out || []).length + '</span>' : '') +
        (ver.games ? ' ' + ver.games + ' game(s) · ' + (rec.win || 0) + 'W ' + (rec.loss || 0) + 'L'
                   : ' <span class="ev">no games</span>') +
        (ver.subject ? '<span class="ev">' + esc(ver.subject) + '</span>' : '');
    });
    var body = list(rows);
    if (v.unmatched_log_entries && v.unmatched_log_entries.length) {
      body += '<p class="ev">' + v.unmatched_log_entries.length + ' logged game(s) played ' +
        'on an uncommitted list — reported unmatched rather than guessed.</p>';
    }
    return panel('record', 'Every list this deck has been',
      'Numbered from git, joined to the log by the decklist sha.', ['data'],
      'var(--tier-data)', body);
  }

  function statusPanel(d) {
    var info = d.info;
    if (!info || !info.status) return '';
    var st = info.status, b = info.bracket || {};
    var rows = [['Stages complete', st.complete + ' / ' + st.of]];
    if (st.stale && st.stale.length) rows.push(['STALE', st.stale.join(', ')]);
    if (st.invalid && st.invalid.length) rows.push(['INVALID', st.invalid.join(', ')]);
    if (st.missing && st.missing.length) rows.push(['Missing', st.missing.join(', ')]);
    if (b.floor) rows.push(['Bracket floor', b.floor + (b.floor_name ? ' (' + b.floor_name + ')' : '')]);
    if (b.target) rows.push(['Target', b.target + (b.within_target ? ' ✓' : ' ✗')]);
    var r = info.record || {};
    rows.push(['Games logged', r.games || 0]);
    if (r.games) rows.push(['Record', r.win + 'W ' + r.loss + 'L' + (r.draw ? ' ' + r.draw + 'D' : '')]);
    return panel('status', 'Where it stands',
      'Lifecycle, gates and the record.', ['data'], 'var(--tier-data)', facts(rows));
  }

  function auditPanel(d) {
    var a = (d.info || {}).audit;
    if (!a) return '';
    var rows = [];
    if (a.archetype) rows.push(['Archetype', a.archetype]);
    if (a.under && a.under.length) rows.push(['Under', a.under.join(', ')]);
    if (a.over && a.over.length) rows.push(['Over', a.over.join(', ')]);
    var diag = (d.info || {}).diagnosis;
    var body = facts(rows);
    if (diag) {
      body += '<p class="ev"><b>Diagnosis</b> (skeptic ' + esc(diag.skeptic) +
        (diag.stale ? ', STALE' : '') + '): ' + esc(diag.verdict || '') + '</p>';
    }
    return panel('audit', 'What limits it',
      'Sixteen cited axes; each target quotes strategy.md.', ['data', 'coach'],
      'var(--tier-coach)', body);
  }

  function enginePanel(d) {
    var e = d.engine;
    if (!e) return absent('engine', 'The engine', 'How the deck actually runs.',
      'var(--tier-coach)', 'engine', d, 'the stages it converts through, and what carries between them');
    var lines = (e.lines || []).map(function (l) {
      var proved = !!l.verified_by;
      return '<span class="chip">' + (proved ? '✓' : '·') + '</span> ' +
        esc(l.from) + ' → ' + esc(l.to) +
        (l.carries ? ' <i>(' + esc(l.carries) + ')</i>' : '') +
        (proved ? ' <span class="ev">' + esc(l.verified_by) + '</span>'
                : ' <span class="ev">a reading, not a proof</span>');
    });
    var body = '<p class="ev">' + esc(e.thesis || '') + '</p>' + list(lines);
    var v = (e.critic || {}).verdict;
    if (v) body += '<p class="ev">Critic verdict: <b>' + esc(v) + '</b></p>';
    return panel('engine', 'The engine',
      'Eight stages, and what moves between them.', ['verified', 'data', 'coach'],
      'var(--tier-verified)', body);
  }

  function tablePanel(d) {
    var runs = d.sims || [], exps = d.experiments || [];
    // A measurement made against a list the deck no longer holds is still a
    // measurement — of a deck that is gone. `info.json` detects it from the sha
    // every run record stamps, and it says so here rather than letting a precise
    // number pass for a current one.
    var simStale = ((d.info || {}).simulation || {}).stale;
    var expStale = (((d.info || {}).experiments || {}).latest || {}).stale;
    if (!runs.length && !exps.length)
      return absent('table', 'At the table', 'Forge, seeded, against your own pod.',
        'var(--tier-data)', 'sim', d);
    
    var body = '';
    runs.forEach(function (run) {
      // A BRANCH SEAT IS FLATTENED FOR FORGE (`@` -> `-`), so looking the seat
      // up by its raw slug finds nothing — the same mistake that printed
      // `wins 0` for a list that had won eleven.
      var seats = (run.analysis || {}).seats || {};
      var me = seats[run.slug] || seats[String(run.slug).replace('@', '-')] || {};
      var rows = [
        ['Table', (run.seats || []).slice(1).map(function (s) { return s.slug; }).join(', ')],
        ['Games', run.games_completed],
        ['Win rate', me.win_rate + (me.win_rate_ci95 ? ' ci95 [' + me.win_rate_ci95.join(', ') + ']' : '')],
        ['Eliminated turn', figure(me.eliminated_turn)],
        ['Combat damage dealt', figure(me.combat_damage_dealt_to_players)]
      ];
      // A WIN RATE NEVER TRAVELS WITHOUT THE PILOTING READING. Forge's AI is
      // untrained and measured at two thirds of a land drop per turn; the number
      // is only readable beside whether OUR seat was handled like the table.
      var pq = ((d.info || {}).simulation || {}).piloting;
      if (pq) {
        rows.push(['AI piloted our seat',
          (pq.comparable ? 'comparably' : 'WORSE than the pod') +
          ' — ' + Math.round(pq.lands_ratio * 100) + '% of the pod\'s land drops']);
      }
      var cd = me.commander_damage;
      if (cd) {
        rows.push(['Cmdr damage on one seat', figure(cd.max_on_one_defender)]);
        rows.push(['Games reaching 21', cd.games_reaching_21 + ' / ' + run.games_completed]);
      }
      body += '<h3 class="slug-line">' + esc(run.run_id.slice(0, 46)) +
        (simStale ? ' <span class="chip stale">stale</span>' : '') + '</h3>' + facts(rows);
    });
    exps.forEach(function (x) {
      var d = x.delta || {}, w = d.win_rate || {}, pw = d.power || {};
      // The interval is on the DIFFERENCE, which is the quantity anyone actually
      // wants and the thing the old artifact could not state — it compared the two
      // arms' marginal intervals and read an overlap as "noise", which is the
      // overlap fallacy. And the minimum detectable difference answers the question
      // a null result raises: could this experiment have found anything at all?
      var band = w.ci95_diff
        ? '[' + w.ci95_diff[0].toFixed(3) + ', ' + w.ci95_diff[1].toFixed(3) + ']'
        : '—';
      var rows = [['A', w.a], ['B', w.b], ['Δ win rate', w.diff],
                  ['Δ 95% interval', band],
                  ['Games per arm', x.games_per_arm]];
      if (pw.minimum_detectable_difference !== undefined &&
          pw.minimum_detectable_difference !== null) {
        rows.push(['Smallest detectable Δ', '±' + pw.minimum_detectable_difference]);
      }
      body += '<h3 class="slug-line">' + esc(x.question || '') +
        (expStale ? ' <span class="chip stale">stale</span>' : '') + '</h3>' + facts(rows) +
        '<p class="ev">' + esc(d.reading || '') + '</p>';
    });
    if (simStale || expStale) {
      body += '<p class="ev"><b>Marked stale:</b> these games were played on a list ' +
        'this deck no longer holds. The figures are true about the deck that played ' +
        'them and say nothing about the current one — re-run <code>simulate</code> ' +
        'or <code>experiment</code> to measure this list.</p>';
    }
    body += '<p class="ev"><b>Every seat is a Forge AI, including this deck.</b> Forge ' +
      'rates its own AI "poor to ok in control, pretty bad for combo", so a control ' +
      'deck\'s rate is a lower bound and a combo deck\'s is not a measurement. A ' +
      'figure without its interval and its N is not a figure.</p>';
    return panel('table', 'At a table',
      'Seeded Forge games against the pilot\'s own pod.', ['data'],
      'var(--tier-data)', body);
  }

  function targetingPanel(d) {
    var doc = d.targeting;
    if (!doc) return '';
    var pol = doc.forge_ai_targeting_policy || {};
    var rows = Object.keys(pol).map(function (k) {
      var h = pol[k];
      return [h.hypothesis,
              (h.rate * 100).toFixed(1) + '% ' +
              'ci95 [' + (h.ci95[0] * 100).toFixed(0) + ', ' + (h.ci95[1] * 100).toFixed(0) + ']' +
              '  vs ' + (h.uniform_expected_rate * 100).toFixed(1) + '% at random' +
              '  p ' + h.permutation_p];
    });
    var body = facts(rows);
    var c = doc.when_the_hypotheses_disagree;
    if (c) {
      // The honest half. The two leading hypotheses agree most of the time, so
      // the headline rate cannot separate them; only the disagreements can.
      body += '<h3 class="slug-line">Where "biggest threat" and "easiest kill" ' +
        'name different seats (' + c.decisions + ' decisions)</h3>' +
        facts(['most_damage_dealt', 'lowest_life', 'neither'].map(function (k) {
          var s = c[k];
          return [k.replace(/_/g, ' '),
                  (s.rate * 100).toFixed(1) + '% ci95 [' +
                  (s.ci95[0] * 100).toFixed(0) + ', ' + (s.ci95[1] * 100).toFixed(0) + ']'];
        })) +
        '<p class="ev">' + esc(c.note) + '</p>';
    }
    body += '<p class="ev"><b>This measures Forge\'s AI, not human politics.</b> ' +
      'There are no deals here, no grudges, no table talk, and no player who ' +
      'remembers what you did last turn. Four fixed decks in four fixed seats, so ' +
      'any policy measured is confounded with deck identity and turn order — it is ' +
      'a statement about this pod, not about Commander.</p>';
    return panel('targeting', 'Who the table attacks',
                 doc.decisions + ' targeting decisions across ' + doc.games +
                 ' simulated games. Opponent modelling, not equilibrium.',
                 ['data'], 'y2k-blue', body);
  }

  function askedPanel(d) {
    var rx = d.prescriptions || [];
    if (!rx.length) return '';
    var items = rx.map(function (p) {
      var answered = p.add_candidates !== undefined && p.add_candidates !== null;
      return '<b>' + esc(p.prompt || p.id) + '</b> — ' +
        (answered ? (p.add_candidates || []).length + ' add(s), ' +
                    (p.cut_candidates || []).length + ' cut(s), skeptic ' +
                    esc((p.skeptic || {}).verdict)
                  : '<i>open — no answer yet</i>');
    });
    return panel('asked', 'Asked and answered',
      'One question to the doctor, scoped and skeptic-checked.', ['data', 'coach'],
      'var(--tier-coach)', list(items));
  }

  /* THE CASE FILE — what a pilot sitting down with this deck needs first.
   *
   * IT DESCRIBES THE CHECKED-IN LIST AND NOTHING ELSE. The dossier once opened
   * by calling ur-dragon a "non-creature treasure engine with an RGW-only mana
   * base" — the design brief of a BRANCH that was measured, found worse and
   * deleted, still sitting in `brief.json` and composed into `info.json` by
   * `deck-info`. The deck it described has 24 creatures, runs two black duals
   * and holds none of the cards the brief named. Explorations belong at the
   * FOOT of this page, behind their own net-change report; the top is the deck
   * you can actually shuffle.
   *
   * STALENESS IS SHOWN, NOT HIDDEN. `deck_status` already computes which
   * artifacts describe an older list — `info.status.stale` — and the page used
   * to render those sections with no mark at all, which is how prose about a
   * different deck reads as current.
   */
  function caseFilePanel(d) {
    var info = d.info || {};
    if (!info.slug) return '';
    var v = info.version || {};
    var rec = info.record || {};
    var paper = (d.entry || {}).paper;
    var stale = (info.status || {}).stale || [];
    var eng = info.engine || {};

    var head = '<div class="case-grid">';
    head += caseFact('Commander', (info.commander || []).join(' / ') || '\u2014');
    head += caseFact('Colours', (info.colour_identity || []).join('') || 'C');
    head += caseFact('The 99', info.size + ' cards \u00b7 ' + info.lands + ' lands');
    head += caseFact('Sleeved',
      paper ? 'V' + paper.version + (paper.built_at ? ' \u00b7 ' + paper.built_at : '')
            : 'not marked as built in paper');
    /* A PROPOSAL BELONGS IN THE CASE FILE, and the branch panel stays at the
     * foot where it was deliberately put ("a branch is a deck that does not
     * exist"). The difference is that a PROPOSED branch is not speculative —
     * the pilot has accepted it and is waiting on cardboard — so the fact that
     * this deck is about to become v1.0.2 is a fact about the deck. */
    var proposed = ((info.branches || []).filter(function (b) {
      return b.proposal && b.state !== 'MERGED';
    }))[0];
    if (proposed) {
      var pl = proposed.pull_list || {};
      head += caseFact('Proposed',
        proposed.proposal.as_version + ' \u00b7 ' + esc(proposed.state) +
        (pl.blocking ? ' \u00b7 ' + pl.blocking + ' card(s) still to find' : ''));
    }
    head += caseFact('Record', rec.games
      ? rec.games + ' game(s) \u00b7 ' + (rec.win || 0) + 'W ' + (rec.loss || 0) + 'L'
      : 'no games logged');
    var br = info.bracket || {};
    head += caseFact('Bracket', br.floor != null
      ? 'floor ' + br.floor + (br.floor_name ? ' (' + br.floor_name + ')' : '') +
        ' \u00b7 target ' + (br.target != null ? br.target : '\u2014')
      : '\u2014');
    head += '</div>';

    var body = head;
    // THE THESIS, and it must say whose it is. A one-line engine thesis is the
    // fastest true sentence about a deck; an absent one says so rather than
    // leaving the reader to assume the deck has no plan.
    if (eng.thesis) {
      body += '<p class="case-thesis">' + esc(eng.thesis) + '</p>';
      body += '<p class="ev">\u2605 the engineer\u2019s reading of the machine' +
              (eng.critic ? ' \u00b7 critic: ' + esc(eng.critic) : '') + '</p>';
    } else {
      body += '<p class="ev">No engine model yet \u2014 nothing here states what ' +
              'this deck is trying to do. <code>/analyze-engine ' +
              esc(info.slug) + '</code></p>';
    }
    if (!info.brief || !Object.keys(info.brief || {}).length) {
      body += '<p class="ev">No brief authored for this list. A brief is the ' +
              'written intent a build starts from; this deck has none, which is ' +
              'absent rather than empty.</p>';
    }
    if (stale.length) {
      body += '<p class="case-stale">\u26a0 ' + esc(stale.join(', ')) +
        ' describe an older list. Those sections below are history, not the ' +
        'deck as it stands.</p>';
    }
    return panel('casefile', 'Case file',
      'The deck as checked in \u2014 explorations are at the foot of this page.',
      ['data'], 'var(--tier-data)', body);
  }

  function caseFact(k, v) {
    return '<div class="case-cell"><span class="case-k">' + esc(k) +
           '</span><span class="case-v">' + esc(v) + '</span></div>';
  }

  function logPanel(d) {
    var entries = d.log || [];
    // A LOCKED deck with no games is not "nothing to show" — it is the most
    // actionable state on the page, and a panel that simply vanishes says
    // nothing at all. An unlocked deck genuinely has no table to log from, so
    // that one still renders no panel.
    if (!entries.length) {
      if (!(d.entry && d.entry.locked)) return '';
      return panel('log', 'The captain\'s log',
        'What happened at the table, in the pilot\'s words.', ['coach'],
        'var(--tier-coach)',
        '<p>No games logged on this deck yet. This deck is sleeved, so the next '
        + 'one you play lands here — stamped with the exact list you played it '
        + 'on, so its record attaches to that version and not to the deck in '
        + 'general.</p>'
        + '<p class="ev">Log a game: <code>manamap pilot deck-notes '
        + esc(d.slug || '&lt;slug&gt;')
        + ' add "…" --result win|loss --opponents 3</code>. The debrief agent '
        + 'then reads it and routes what it raises to the loop that can settle '
        + 'it — a goldfish run, a rules resolution, a question to the doctor.</p>');
    }
    // SUMMARY FIRST, THEN THE ENTRIES. The log is the only thing on this page
    // written by a person who was actually at the table, and it was rendered as
    // an undifferentiated list halfway down. What a reader wants first is the
    // shape — how many games, how they went, how many still have nobody's
    // reading on them — and the entries on demand.
    var notes = (d.debrief || {}).entries || {};
    var wins = 0, losses = 0, undebriefed = 0;
    entries.forEach(function (e) {
      if (e.result === 'win') wins++;
      else if (e.result === 'loss') losses++;
      if (!notes[e.id]) undebriefed++;
    });
    var last = entries[entries.length - 1] || {};
    var summary = '<div class="log-sum">'
      + '<span class="log-n">' + entries.length + '</span> game(s) logged'
      + ' \u00b7 <b>' + wins + 'W ' + losses + 'L</b>'
      + (last.at ? ' \u00b7 last ' + esc(last.at.slice(0, 10)) : '')
      + (undebriefed ? ' \u00b7 <span class="log-todo">' + undebriefed
                       + ' not yet debriefed</span>' : '')
      + '</div>';
    // The most recent entry rides ABOVE the fold, because "what happened last
    // time" is the question that brought the pilot here.
    if (last.text) {
      summary += '<p class="log-last"><b>' + esc((last.at || '').slice(0, 10))
        + '</b> ' + (last.result ? '<span class="chip">' + esc(last.result)
                                   + '</span> ' : '')
        + esc(last.text) + '</p>';
      var ln = notes[last.id];
      if (ln && ln.summary) {
        summary += '<p class="ev">' + esc(ln.summary) + '</p>';
      }
    }
    var items = entries.slice().reverse().map(function (e) {
      var n = notes[e.id];
      return '<b>' + esc(e.at ? e.at.slice(0, 10) : e.id) + '</b> ' +
        (e.result ? '<span class="chip">' + esc(e.result) + '</span> ' : '') +
        esc(e.text || '') +
        (n ? '<span class="ev">' + esc(n.summary || '') + '</span>'
           : '<span class="ev">not yet debriefed</span>');
    });
    var body = summary;
    if (entries.length > 1) {
      body += '<details class="log-all"><summary>All ' + entries.length
        + ' entries</summary>' + list(items) + '</details>';
    }
    return panel('log', 'The captain\'s log',
      'What happened at the table, in the pilot\'s words.', ['coach'],
      'var(--tier-coach)', body);
  }

  function questionsPanel(d) {
    var qs = (d.info || {}).open_questions || [];
    if (!qs.length) return '';
    var items = qs.map(function (q) {
      return '<span class="chip">' + esc(q.settled_by || '?') + '</span> ' +
        esc(q.question) + '<span class="ev">from ' + esc(q.from) + '</span>';
    });
    return panel('questions', 'Open questions',
      'What nobody has settled yet, and which loop would settle it.', ['coach'],
      'var(--tier-coach)', list(items));
  }

  // ── Assembly ─────────────────────────────────────────────────────────

  function render(slug, d) {
    var issue = d.issue || {};
    document.title = (issue.deck_name || slug) + ' — Deck Dossier';
    document.getElementById('deckName').textContent = issue.deck_name || slug;
    // The commander and the version — NOT "Pilot's Manual Vol. 006". Volume
    // numbers belong to the magazine this surface replaced, and the version is
    // both truer and more useful: it is the list the games attach to.
    // `info.json` deliberately omits the version — it is a git walk, and a
    // committed copy is one commit behind forever. The manifest's `paper` block
    // carries the SLEEVED version, which is the more useful fact anyway: it is
    // the list the games attach to.
    var paper = (d.entry || {}).paper;
    document.getElementById('commanderLine').textContent =
      (issue.commander || '') + (paper ? ' \u00b7 sleeved V' + paper.version : '');
    // A deck is loadable here as soon as it has a cards.json; a magazine issue is a
    // separate, later, expensive step. Linking to `../manuals/<slug>.html`
    // unconditionally sent every unpublished deck to a 404 — the manifest carries
    // `published` precisely so this link can tell the two apart.
    // A deck that no longer exists as cardboard says so, rather than inviting
    // someone to sleeve it. The flag rides in the manifest.
    var life = (d.info || {}).lifecycle;
    if (life) {
      var head = document.querySelector('.dossier-head');
      var flag = document.createElement('div');
      flag.className = 'slug-line';
      flag.style.cssText = 'color:var(--tier-coach);margin-top:6px';
      flag.textContent = '⚑ ' + life.headline + ' — ' + life.body;
      head.firstElementChild.appendChild(flag);
    }

    // No magazine link. The surfaces are the workbench, the map, the dossier and
    // the Pilot's Manual — the magazine is not a product any more, so pointing a
    // pilot at it is offering them a page nobody maintains. The RENDERER is not
    // deleted and the issues stay on disk: they are the record of what was
    // published, and `issue.json` still carries the deck's `status`, which
    // `deck_lifecycle` reads. What goes is the invitation, not the archive.
    var link = document.getElementById('issueLink');
    if (link) link.remove();
    // The compact Pilot's Manual lives under manuals/p/. Hidden rather than dead
    // when a deck has none, because a link that 404s is worse than a link that is
    // not there.
    var manual = document.getElementById('manualLink');
    if (manual) {
      // `has` rides on the MANIFEST entry, not on the artifacts object —
      // `d` is keyed by artifact name. Reading `d.has` was always undefined,
      // so the link silently hid itself on every deck that had a manual.
      if (d.entry && d.entry.has && d.entry.has.page) {
        manual.hidden = false;
        manual.href = '../manuals/p/' + slug + '.html';
      } else {
        manual.hidden = true;
        manual.removeAttribute('href');
      }
    }
    // The map's Deck Lens reads ?deck=<slug> on entry, so this deep-links straight
    // into the overlay rather than dropping the reader on an unfiltered map.
    document.getElementById('lensLink').href = 'index.html?deck=' + encodeURIComponent(slug);

    // Workbench first, reference second: the questions a pilot sits down with are
    // "where is this, and what do I do", not "what shape is the mana curve".
    // THE CASE FILE AND THE LOG LEAD, AND THE EXPLORATIONS COME LAST.
    //
    // The old order opened with `next` and buried the log at position nine,
    // between the threat table and the open questions — so the one artifact
    // written by a person who was at the table sat below four derived ones. And
    // `branchPanel` sat in the middle of the reference half, where a PROPOSAL
    // read as a property of the deck. A branch is a deck that does not exist;
    // it belongs at the foot, behind its own net-change report.
    var html = [
      caseFilePanel(d), logPanel(d), nextPanel(d),
      statusPanel(d), recordPanel(d), auditPanel(d),
      enginePanel(d), tablePanel(d), targetingPanel(d), askedPanel(d),
      questionsPanel(d),
      briefPanel(d), vitalsPanel(d), constellationPanel(d), rosterPanel(d),
      bracketPanel(d), manaPanel(d), goldfishPanel(d),
      tenPanel(d), tutorPanel(d), buildPlanPanel(d), stacksPanel(d),
      branchPanel(d)
    ].filter(Boolean).join('');
    document.getElementById('panels').innerHTML = html;
    var bits = [d.stacks.length + ' verified line(s)'];
    if ((d.sims || []).length) bits.push(d.sims.length + ' sim run(s)');
    if ((d.experiments || []).length) bits.push(d.experiments.length + ' experiment(s)');
    if (!d.info) bits.push('no info.json — run `deck-info ' + slug + ' --write`');
    document.getElementById('status').textContent =
      bits.join(' · ') + ' · artifacts read from data/decks/' + slug + '/';
  }

  function pickerHTML(decks, current) {
    // No volume numbers. "Vol. 999 KIANNE" is the magazine's sentinel for an
    // unnumbered issue leaking onto the workbench, and volume/issue/newsstand is
    // vocabulary the product retired. A deck has a name; that is the label.
    // Locked decks lead, because that is the question the front door asks.
    var sorted = decks.slice().sort(function (a, b) {
      return (b.locked ? 1 : 0) - (a.locked ? 1 : 0) || a.slug.localeCompare(b.slug);
    });
    return 'Decks: ' + sorted.map(function (d) {
      var label = d.slug + (d.locked ? ' \u25c6' : '');
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
      // The manifest names these because a browser cannot list a directory. Without
      // it the page can fetch only what it knows the name of, which is why the
      // dossier showed no simulation, no experiment and no prescription for a year.
      function dirJobs(key, dir) {
        return (entry[key] || []).map(function (f) {
          return getJSON(BASE + entry.slug + '/' + dir + '/' + f);
        });
      }
      var simJobs = dirJobs('sim_runs', 'sim');
      var expJobs = dirJobs('experiments', 'experiments');
      var rxJobs = dirJobs('prescriptions', 'prescriptions');
      var logJob = (entry.has || {}).log
        ? fetch(BASE + entry.slug + '/log.jsonl').then(function (r) {
            return r.ok ? r.text() : '';
          }).catch(function () { return ''; })
        : Promise.resolve('');
      var debriefJob = (entry.has || {}).log_annotations
        ? getJSON(BASE + entry.slug + '/log_annotations.json') : Promise.resolve(null);

      return Promise.all([Promise.all(jobs), Promise.all(stackJobs),
                          Promise.all(simJobs), Promise.all(expJobs),
                          Promise.all(rxJobs), logJob, debriefJob])
        .then(function (both) {
          var d = { stacks: both[1].filter(Boolean) };
          both[0].forEach(function (p) { d[p[0]] = p[1]; });
          d.sims = both[2].filter(Boolean);
          d.experiments = both[3].filter(Boolean);
          d.prescriptions = both[4].filter(Boolean);
          // JSONL, one game per line — the log is append-only and authored.
          d.log = String(both[5] || '').split('\n').filter(Boolean).map(function (ln) {
            try { return JSON.parse(ln); } catch (e) { return null; }
          }).filter(Boolean);
          d.debrief = both[6];
          // From the MANIFEST, not from issue.json — an unpublished deck has no
          // issue.json at all, so reading the flag off `d.issue` would always be
          // undefined and the dead-link guard would never fire.
          d.published = entry.published !== false;
          // The manifest entry itself, for the panels that need to know what
          // KIND of deck this is rather than what artifacts it has — whether it
          // is sleeved, above all. An empty log means something different on a
          // locked deck than on a build plan.
          d.entry = entry;
          d.slug = entry.slug;
          // A BRANCH'S MAP NEEDS A SECOND ROUND TRIP, and it is chained rather
          // than batched because the branch's NAME only arrives with info.json,
          // which is in the batch above. The alternative is naming branch files
          // in the manifest the way `stack_files` are — worth doing if branches
          // ever multiply, and overkill for a fetch that only fires on a deck
          // that has one.
          var bs = ((d.info || {}).branches) || [];
          var b = bs.filter(function (x) { return !x.unreadable; })[0];
          if (!b) { render(entry.slug, d); return; }
          getJSON(BASE + entry.slug + '/branches/' + b.name + '/deck_map.json')
            .then(function (m) { d.branchMap = m; })
            .catch(function () { d.branchMap = null; })
            .then(function () { render(entry.slug, d); });
        });
    }).catch(function (e) {
      document.getElementById('status').textContent = 'Could not load the dossier: ' + e;
    });
  }

  /* The one thing this page needs a global for: the copy button on an absent
   * section's command. Assigned here, at the end, rather than referenced from
   * anything that runs during evaluation — the boot-order trap `mana-map.js`
   * documents (touching a global inside the IIFE that defines it aborts the
   * IIFE, so the global is never exported and every later file fails too). */
  window.Deck = {
    /* Run one deterministic measurement and show the result.
     *
     * A RELOAD, not a re-render. The dossier composes fifteen artifacts at
     * boot and a measurement writes two of them (its own, plus the `info.json`
     * the server refreshes); patching the live object would be a second,
     * partial version of that composition, free to disagree with the one the
     * page was built from. Two seconds of work is not the place to invent a
     * cache-coherence problem. */
    measure: function (btn, slug, stage) {
      if (btn.disabled) return;
      btn.disabled = true;
      var was = btn.textContent;
      btn.textContent = 'Measuring…';
      Api.call('deck/measure', { slug: slug, stage: stage }).then(function () {
        btn.textContent = 'Done — reloading';
        location.reload();
      }).catch(function (e) {
        // Named, never silent: the server refuses for real reasons (a missing
        // dependency, a stage it will not run) and every one of them is
        // something the pilot can act on.
        btn.disabled = false;
        btn.textContent = was;
        var p = document.createElement('p');
        p.className = 'ev todo-blocked';
        p.textContent = String(e && e.message ? e.message : e);
        btn.parentElement.insertBefore(p, btn.nextSibling);
      });
    },
    /* Write a starting version of an authored file. Same shape as `measure`,
     * and separate on purpose: one runs a measurement, the other hands you a
     * draft of something only you can finish, and a single verb for both would
     * make a draft look like a result. */
    draft: function (btn, slug, stage) {
      if (btn.disabled) return;
      btn.disabled = true;
      var was = btn.textContent;
      btn.textContent = 'Drafting…';
      Api.call('deck/scaffold', { slug: slug, stage: stage }).then(function () {
        btn.textContent = 'Drafted — reloading';
        location.reload();
      }).catch(function (e) {
        btn.disabled = false;
        btn.textContent = was;
        var p = document.createElement('p');
        p.className = 'ev todo-blocked';
        p.textContent = String(e && e.message ? e.message : e);
        btn.parentElement.insertBefore(p, btn.nextSibling);
      });
    },
    copy: function (btn) {
      var text = (btn.querySelector('code') || {}).textContent || '';
      var say = btn.querySelector('.todo-copy');
      var done = function (ok) {
        if (!say) return;
        say.textContent = ok ? 'copied' : 'select and copy';
        setTimeout(function () { say.textContent = 'copy'; }, 1600);
      };
      // `navigator.clipboard` needs a secure context, and this page is served
      // over plain http from localhost — which IS secure by the spec, but not
      // everywhere it might be opened. Report the failure rather than looking
      // like a button that does nothing.
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); },
                                                 function () { done(false); });
      } else { done(false); }
    },
  };

  /* Ask ONCE whether there is a server, and what it will run.
   *
   * Both are absent on the deployed site, and that is a supported shape rather
   * than a fault: the page renders its static half and prints the command. The
   * boot does not WAIT on this — a probe that has to answer before anything
   * draws would make the dossier's speed depend on a server it may not have.
   */
  function boot_() {
    if (window.Api && Api.probe) {
      Api.probe().then(function (ok) {
        if (!ok) return null;
        return Api.call('deck/measures', {}).then(function (r) {
          measures = r.stages || null;
          drafts = r.drafts || [];
        });
      }).catch(function () { measures = null; })
        .then(function () { boot(); });
      return;
    }
    boot();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot_);
  } else { boot_(); }
})();
