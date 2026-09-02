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

  /* THE NINE SECTIONS OF THE FILE, mirrored from `page_spec.DOSSIER_SECTIONS`.
   *
   * A dossier is not a report. A report has a conclusion; a dossier has a LATEST
   * ENTRY, and its order encodes that: the cover sheet is thirty seconds at the
   * top, the assessment is last and SEPARATE — because a file where the
   * analyst's opinion is mixed into the record loses trust, which is exactly
   * what the old page did by rendering the diagnosis verdict as one inline
   * sentence inside the audit panel.
   *
   * PYTHON OWNS THE ORDER. This is a transcription and a test locks the two
   * together, the same contract `decklist.js` lives under — because the order
   * used to be an anonymous array literal down in `render()` with no statement
   * of it anywhere. */
  var DOSSIER = [
    ['cover', 'Cover sheet',
     'Who, what state, and the three numbers. Thirty seconds.', ['data']],
    ['rap-sheet', 'Rap sheet',
     'Every version this deck has been: what changed, why, and what happened.',
     ['data', 'coach']],
    ['associates', 'Known associates',
     'The 99 by the job each card does, and the ones that decide games.', ['data']],
    ['vitals', 'Vitals',
     'The seeded measurements, and what they do not model.', ['data']],
    ['priors', 'Priors',
     'Every game played, one row, with how it ended.', ['coach']],
    ['logs', "Captain's logs",
     "The night as a ship's log; the pilot's own words underneath, unedited.", ['coach']],
    ['exhibits', 'Exhibits',
     'The evidence, attached whole and stamped with the list it describes.',
     ['verified', 'data']],
    ['leads', 'Open leads',
     'What is unresolved, and which loop would settle it.', ['coach']],
    ['assessment', "Analyst's assessment",
     "The custodian's current read, dated. Previous reads kept underneath.",
     ['coach']]
  ];

  function sect(id) {
    for (var i = 0; i < DOSSIER.length; i++) {
      if (DOSSIER[i][0] === id) return DOSSIER[i];
    }
    return [id, id, '', []];
  }

  /* A dossier section, as opposed to `panel()`'s reference panel. Same shell —
   * so the tier badges, the `--accent` hook and every existing style keep
   * working — with the title and promise coming from the registry rather than
   * from the call site, which is what stops the two drifting. */
  function section(id, accent, body) {
    var sp = sect(id);
    return panel(id, sp[1], sp[2], sp[3], accent, body);
  }

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
    // HOW EACH GAME ENDED — authored by the pilot, keyed by log entry id.
    // Separate from `log.jsonl` because that file is append-only and eleven
    // games were logged before the field existed.
    causes: 'log_causes.json',
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

  /* SECTIONS THAT MUST NOT SHARE A COLUMN. `.panels` is a two-column grid, and
   * a record TABLE in one column wraps every cell into a stack — the rap sheet
   * rendered its card names one per line down a 250px column and read as a
   * wall. Membership is a property of the content, not of the id, but the id is
   * what the caller has. */
  var WIDE = { kill: 1, ten: 1, cover: 1, 'rap-sheet': 1, priors: 1 };

  function panel(id, title, promise, tiers, accent, body) {
    return '<section class="panel' + (WIDE[id] ? ' wide' : '') +
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
    //
    // THE HREF IS `Shell.cardHref`, NOT A SECOND OPINION ABOUT SCRYFALL'S URL
    // FORMAT. This anchor had no `href` at all — it was a `tabindex` hook for
    // the hover, so the roster showed you a card and could not take you to one.
    // branch.html renders its names through `Shell.cardLink`, which builds the
    // same URL; the two presentations differ (that page has no room for a
    // 488px pop beside sixty diff rows) but the destination is defined once.
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
    var href = (window.Shell && Shell.cardHref)
      ? ' href="' + esc(Shell.cardHref(name)) + '" target="_blank" rel="noopener"'
      : '';
    return '<li class="rost-row">' + mark +
      '<a class="cardref"' + href + ' tabindex="0">' + esc(name) + pop + '</a>' +
      tail + '</li>';
  }

  function rosterPanel(d) {
    var got = rosterFor(d);
    var map = got.map;
    if (!map || !map.cards || !map.regions)
      return absent('associates', sect('associates')[1], sect('associates')[2],
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
      /* THE MEMBERS FOLD, PER CITY. Ninety-nine cards inline is 3,111 pixels,
       * and the roster's job is to show the deck's SHAPE — which cities exist,
       * how big each is, what each is for. Folding per city rather than all at
       * once keeps it scannable: you open the one you came for. */
      body += '<details class="rost-fold"><summary>' + members.length
        + ' card(s)</summary><ul class="rost-list">' + members.map(function (c) {
        seen[c.name] = 1;
        return cardRef(c.name, prov[c.name]);
      }).join('') + '</ul></details>';
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
        '<details class="rost-fold"><summary>' + stray.length
        + ' card(s)</summary><ul class="rost-list">' + stray.map(function (c) {
          return cardRef(c.name, prov[c.name]);
        }).join('') + '</ul></details>';
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
    // Title still carries the branch name when one is being viewed; the
    // registry supplies the promise and the tier.
    return panel('associates', title, sect('associates')[2],
                 sect('associates')[3], '#8a7fd0', body);
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
    /* `diagnostic.json` IS NOT A LIFECYCLE STAGE — it is a gated artifact with
     * no `STAGES` row, so `absent()` finds no todo for it and returns ''. That
     * made the vitals tab vanish on NINE OF TEN decks: only ur-dragon has been
     * diagnosed, and the other nine said nothing about the difference between
     * "healthy" and "never measured". Named here instead. */
    if (!v) {
      return section('vitals', '#4c8fbd',
        '<p class="ev">Not measured. The vitals are the seeded diagnostic — ' +
        'engine online by turn, stall risk, and the mana under both — and ' +
        'nothing has run it against this list. ' +
        '<code>manamap pilot diagnose ' + esc((d.info || {}).slug || '') +
        ' --write</code></p>');
    }
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

  /* THE MODEL RENDERER. One function for every converted panel.
   *
   * The dossier's four fattest panels were 3,000-4,000px because nothing
   * upstream had an opinion about what mattered, so each rendered its artifact
   * at whatever length it happened to be. `deck_model` now weights every fact
   * headline / body / detail, and this is the only thing that reads that
   * weighting: headline large and inline, body compact, detail behind a
   * disclosure. A panel CANNOT dump its artifact any more, whatever it wants. */
  function factValue(f) {
    var v = f.value;
    if (Array.isArray(v)) {
      return '<ul class="m-list">' + v.map(function (x) {
        return '<li>' + esc(x) + '</li>';
      }).join('') + '</ul>';
    }
    if (v && typeof v === 'object') {
      var ks = Object.keys(v).sort(function (a, b) { return (+a || 0) - (+b || 0); });
      return '<table class="data"><tr>' + ks.map(function (k) {
        return '<th>' + esc(k) + '</th>';
      }).join('') + '</tr><tr>' + ks.map(function (k) {
        return '<td>' + esc(typeof v[k] === 'number' ? num(v[k], 2) : v[k]) + '</td>';
      }).join('') + '</tr></table>';
    }
    return esc(v) + (f.unit && f.unit !== '%' ? ' ' + esc(f.unit)
                     : (f.unit === '%' ? '%' : ''));
  }

  function factLabel(k) {
    return k.indexOf('target:') === 0 ? k.slice(7)
      : k.replace(/_/g, ' ').replace(/^./, function (c) { return c.toUpperCase(); });
  }

  /* A figure and its DEFINITION, together. "Every figure carries its
   * definition, in the report that prints it" — a number a reader has to look
   * up elsewhere gets guessed at, and a mean has been read as a rate here
   * before. The definition is the `title` and the caption, not a footnote. */
  function factRow(k, f, big) {
    if (f.absent_because) {
      return '<p class="m-absent"><b>' + esc(factLabel(k)) + '</b> — not measured: '
        + esc(f.absent_because) + '</p>';
    }
    var ci = f.ci95 ? ' <span class="m-ci">[' + num(f.ci95[0], 3) + ', '
                      + num(f.ci95[1], 3) + ']</span>' : '';
    var n = f.n ? ' <span class="m-n">n=' + esc(f.n) + '</span>' : '';
    if (big) {
      return '<div class="m-head"><span class="m-v">' + factValue(f) + '</span>'
        + ci + n + '<span class="m-k">' + esc(factLabel(k)) + '</span>'
        + '<span class="m-def">' + esc(f.definition) + '</span></div>';
    }
    return '<div class="m-row" title="' + esc(f.definition) + '">'
      + '<span class="m-rk">' + esc(factLabel(k)) + '</span>'
      + '<span class="m-rv">' + factValue(f) + ci + '</span></div>';
  }

  function modelBlock(blk) {
    if (!blk) { return ''; }
    var facts = blk.facts || {};
    var pick = function (w) {
      return Object.keys(facts).filter(function (k) {
        return (facts[k] || {}).weight === w;
      });
    };
    var out = '';
    pick('headline').forEach(function (k) { out += factRow(k, facts[k], true); });
    var body = pick('body');
    if (body.length) {
      out += '<div class="m-body">'
        + body.map(function (k) { return factRow(k, facts[k], false); }).join('')
        + '</div>';
    }
    /* DETAIL IS NEVER INLINE. That is the rule the weight vocabulary exists to
     * enforce, and it is where the assumptions and the per-turn curves live. */
    var detail = pick('detail');
    if (detail.length) {
      out += '<details class="m-detail"><summary>' + detail.length
        + ' more — assumptions, distributions and curves</summary>'
        + detail.map(function (k) {
            return '<div class="m-d"><b>' + esc(factLabel(k)) + '</b>'
              + '<p class="ev">' + esc(facts[k].definition) + '</p>'
              + factValue(facts[k]) + '</div>';
          }).join('')
        + '</details>';
    }
    if (blk.definition) {
      out += '<p class="ev m-scope">' + esc(blk.definition) + '</p>';
    }
    return out;
  }

  function goldfishPanel(d) {
    /* READS THE MODEL, not the artifact. It used to render two meters, every
     * target, a three-by-ten table, a facts list and all 28 model assumptions
     * inline — 3,833 pixels, with nothing marked as mattering more than
     * anything else. */
    var blk = ((d.info || {}).model || {}).goldfish;
    if (!blk) return absent('goldfish', 'The goldfish', 'Seeded Monte Carlo over resource development.',
      'var(--tier-data)', 'goldfish', d, 'how fast it develops, over 10,000 seeded games');
    return panel('goldfish', 'By the Numbers', 'What can I expect, turn by turn?',
                 ['data'], 'var(--tier-data)', modelBlock(blk));
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
    /* THE ASSESSMENT IS ONE UNBROKEN PARAGRAPH — around two thousand characters
     * of coaching, rendered raw, and it was the whole visible bulk of this
     * panel. It is worth reading and it is not what a pilot opens the page for,
     * so the LEAD stays and the rest folds: the same headline/body/detail split
     * `deck_model` makes structural, applied to prose that has no model yet. */
    var lead = '', rest = '';
    var text = String(t.assessment || '');
    var cut = text.indexOf('. ');
    if (cut > 0 && text.length > 260) {
      lead = text.slice(0, cut + 1);
      rest = text.slice(cut + 2);
    } else {
      lead = text;
    }
    var body = '<div class="m-head"><span class="m-v">' + (t.tutors || []).length
      + '</span><span class="m-k">tutor(s)</span>'
      + '<span class="m-def">what each one should go and get, by board state</span></div>'
      + (lead ? '<p>' + esc(lead) + '</p>' : '')
      + (rest ? '<details class="m-detail"><summary>the rest of the read</summary>'
                + '<p>' + esc(rest) + '</p></details>' : '')
      + items;
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

  function logPanel(d) {
    var entries = d.log || [];
    /* THE SECTION IS A TAB IN A FILE, so it renders empty rather than vanishing.
     * A file whose tabs disappear when a drawer is empty is the "cleaned up into
     * a narrative" failure this page exists to avoid. */
    if (!entries.length) {
      return section('logs', 'var(--tier-coach)',
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

    var notes = (d.debrief || {}).entries || {};
    var nights = (d.captainsLog || {}).nights || {};
    var wins = 0, losses = 0, undebriefed = 0;
    entries.forEach(function (e) {
      if (e.result === 'win') wins++;
      else if (e.result === 'loss') losses++;
      if (!notes[e.id]) undebriefed++;
    });
    var last = entries[entries.length - 1] || {};
    var body = '<div class="log-sum">'
      + '<span class="log-n">' + entries.length + '</span> game(s) logged'
      + ' · <b>' + wins + 'W ' + losses + 'L</b>'
      + (last.at ? ' · last ' + esc(last.at.slice(0, 10)) : '')
      + (undebriefed ? ' · <span class="log-todo">' + undebriefed
                       + ' not yet debriefed</span>' : '')
      + '</div>';

    /* THE RAW NOTE, ALWAYS REACHABLE. Every entry gets one of these whether or
     * not it has been rendered as a log — a game that exists and is invisible is
     * the one outcome this section forbids. */
    function rawNote(e) {
      var chips = [e.result, (d.causes || {}).entries
                             && ((d.causes.entries[e.id] || {}).cause)]
        .filter(Boolean).map(esc).join(' · ');
      var n = notes[e.id];
      return '<details class="log-raw"><summary>the note as written — '
        + esc((e.at || '').slice(0, 10))
        + (e.opponents ? ', ' + esc(e.opponents) + ' opponents' : '')
        + '</summary>'
        + '<p class="log-raw-text">' + esc(e.text || '') + '</p>'
        + (chips ? '<p class="ev">' + chips + '</p>' : '')
        + (n && n.summary ? '<p class="ev">' + esc(n.summary) + '</p>'
                          : '<p class="ev">not yet debriefed</p>')
        + '</details>';
    }

    function block(b) {
      if (!b) { return ''; }
      var out = '<p class="pl-head">' + esc(b.header || '') + '</p>'
        + '<div class="pl-body">';
      [['Situation', b.situation], ['Narrative', b.narrative]]
        .forEach(function (pair) {
          if (pair[1]) {
            out += '<h4>' + pair[0] + '</h4><p>' + esc(pair[1]) + '</p>';
          }
        });
      if ((b.assessment || []).length) {
        out += '<h4>Assessment</h4><p>'
          + b.assessment.map(function (a) { return esc(a.text || ''); }).join(' ')
          + '</p>';
      }
      if ((b.orders || []).length) {
        out += '<h4>Orders</h4>' + list(b.orders.map(function (o) {
          return '<span class="pl-station">' + esc(o.station || '') + '</span> '
            + esc(o.text || '');
        }));
      }
      if (b.coda) { out += '<h4>Coda</h4><p class="pl-coda">' + esc(b.coda) + '</p>'; }
      return out + '</div>'
        + (b.supplementals || []).map(function (sup) {
            return '<div class="pl-sup">' + block(sup) + '</div>';
          }).join('');
    }

    /* Newest night first — "what happened last time" is the question that
     * brought the pilot here. */
    var byId = {};
    entries.forEach(function (e) { byId[e.id] = e; });
    var keys = Object.keys(nights).sort().reverse();
    var placed = {};
    keys.forEach(function (k) {
      (nights[k].source_ids || []).forEach(function (id) { placed[id] = true; });
    });

    /* NEWEST NIGHT OPEN, THE REST FOLDED. Three full logs inline is 4,143px,
     * and "what happened last time" is the question that brought the pilot
     * here — the earlier nights are history and history folds. */
    var rendered = 0;
    keys.forEach(function (k) {
      var night = nights[k] || {};
      var ship = (night.logs || {}).ship;
      if (!ship) { return; }
      rendered++;
      var older = rendered > 1;
      var pos = night.position_in_evening;
      if (older && rendered === 2) { body += '<details class="pl-older"><summary>earlier nights</summary>'; }
      body += '<article class="pl">' + block(ship)
        + (pos && pos.of > 1
            ? '<p class="ev">game ' + esc(pos.n) + ' of ' + esc(pos.of)
              + ' that night' + (pos.after ? ', after ' + esc(pos.after) : '')
              + '</p>'
            : '')
        + (night.source_ids || []).map(function (id) {
            return byId[id] ? rawNote(byId[id]) : '';
          }).join('')
        + '</article>';
    });
    if (rendered > 1) { body += '</details>'; }

    /* A GAME WITH NO RENDERED LOG STILL SHOWS. Mirrors "not yet debriefed":
     * the absence is stated, the note is readable, and the command is named. */
    var unrendered = entries.filter(function (e) {
      return !placed[e.id] || !((nights[Object.keys(nights).filter(function (k) {
        return (nights[k].source_ids || []).indexOf(e.id) >= 0;
      })[0]] || {}).logs || {}).ship;
    });
    if (unrendered.length) {
      body += '<div class="pl-todo"><p class="ev">'
        + unrendered.length + ' night(s) not yet rendered as a log — '
        + '<code>/captains-log ' + esc(d.slug || '') + '</code></p>'
        + unrendered.slice().reverse().map(rawNote).join('') + '</div>';
    }
    return section('logs', 'var(--tier-coach)', body);
  }

  function questionsPanel(d) {
    var qs = (d.info || {}).open_questions || [];
    /* UNRESOLVED LEADS ARE PART OF THE FILE, not a footnote — so "nothing
     * open" is a statement worth making. It also distinguishes a deck nobody
     * has questioned from one whose questions have all been settled, which a
     * vanished section cannot. */
    if (!qs.length) {
      return section('leads', 'var(--tier-coach)',
        '<p class="ev">Nothing open. Questions arrive from the engine model, ' +
        'the diagnosis and the debrief — none of those has raised one against ' +
        'this list.</p>');
    }
    /* TWENTY-ONE QUESTIONS AT FULL LENGTH IS 3,936 PIXELS and answers nothing a
     * reader came for. What they came for is HOW MANY and WHICH LOOP SETTLES
     * THEM — the routes are a closed set, so they count. The questions
     * themselves are evidence and fold. */
    var byRoute = {};
    qs.forEach(function (q) {
      var r = q.settled_by || 'unrouted';
      (byRoute[r] = byRoute[r] || []).push(q);
    });
    var routes = Object.keys(byRoute).sort(function (a, b) {
      return byRoute[b].length - byRoute[a].length;
    });
    var head = '<div class="m-head"><span class="m-v">' + qs.length + '</span>'
      + '<span class="m-k">open lead(s)</span>'
      + '<span class="m-def">every one names the loop that would settle it</span></div>'
      + '<div class="m-body">' + routes.map(function (r) {
          return '<div class="m-row"><span class="m-rk">' + esc(r)
            + '</span><span class="m-rv">' + byRoute[r].length + '</span></div>';
        }).join('') + '</div>';
    var items = qs.map(function (q) {
      return '<span class="chip">' + esc(q.settled_by || '?') + '</span> ' +
        esc(q.question) + '<span class="ev">from ' + esc(q.from) + '</span>';
    });
    return section('leads', 'var(--tier-coach)', head
      + '<details class="m-detail"><summary>read all ' + qs.length
      + '</summary>' + list(items) + '</details>');
  }

  // ── Assembly ─────────────────────────────────────────────────────────

  // ══ THE COVER SHEET ═══════════════════════════════════════════════════
  //
  // The booking record: who, what state, three numbers, and nothing else. It
  // replaces `caseFilePanel`, which had grown into eight facts, a thesis, a
  // stale list and two absence notices — all useful, none of it something you
  // absorb in thirty seconds, which is the one job a cover sheet has.

  /* WHAT STATE THE FILE IS IN, in one stamped word.
   *
   * The lifecycle and the paper lock are the same question about cardboard
   * asked twice, and `deck_versions.json` already reconciles them — so this
   * reads the reconciled answer rather than re-deciding it. 📌 means SLEEVED
   * and nothing else, the rule the workbench rack lives under. */
  function stampOf(info, entry) {
    var life = info && info.lifecycle;
    if (life) {
      return { word: life.status === 'broken-down' ? 'COLD CASE'
                    : life.status === 'superseded' ? 'SUPERSEDED' : 'CLOSED',
               kind: 'dead', why: life.headline };
    }
    if (entry && entry.paper) {
      var v = entry.paper.release || ('V' + entry.paper.version);
      if (entry.paper.in_sync) {
        return { word: 'SLEEVED ' + v, kind: 'ok', pin: true,
                 why: 'built in paper and level with the repo' };
      }
      return { word: 'SLEEVED ' + v, kind: 'warn', pin: true,
               why: 'the cardboard has drifted from the list' };
    }
    return { word: 'UNDER INVESTIGATION', kind: 'open',
             why: 'nobody has said whether this exists in paper' };
  }

  /* One headline figure. THE DEFINITION TRAVELS WITH IT — a number a reader has
   * to look up elsewhere gets guessed at, and the guesses go one way. */
  function head(label, value, note) {
    return '<div class="cov-num"><div class="cov-k">' + esc(label) + '</div>' +
      '<div class="cov-v">' + (value === null || value === undefined
        ? '<span class="ev">not measured</span>' : esc(value)) + '</div>' +
      (note ? '<div class="cov-note">' + esc(note) + '</div>' : '') + '</div>';
  }

  /* THE STRICTER KEEP RULE, and the label says which.
   *
   * `goldfish.py` reports two and warns about the loose one in its own comment:
   * `keep_first_seven_rate` "sits near 100% inside the keep window for every
   * deck — informative about the mulligan rule, useless as a fitness signal."
   * A cover-sheet number that reads 98% on every deck in the fleet is not a
   * number, so this uses `keep_can_act_by_t3_rate`. */
  function keepRate(d) {
    var oh = ((d.goldfish || {}).metrics || {}).opening_hand || {};
    var r = oh.keep_can_act_by_t3_rate;
    return (r === undefined || r === null) ? null : Math.round(r * 100) + '%';
  }

  function versionName(v) {
    return (v && v.tags && v.tags.length) ? v.tags[0] : ('V' + (v || {}).version);
  }

  /* The version this deck IS right now.
   *
   * `current_version` is a TOP-LEVEL ordinal on the document, not a flag on the
   * row — a first cut read `v.current`, which is undefined on every row, so
   * every deck resolved to "no version" and the cover sheet reported its record
   * as unmeasured on a deck with three logged games. The fallback is the last
   * row rather than nothing, because a deck with an uncommitted list still has
   * a newest version. */
  function currentVersion(d) {
    var doc = d.versions || {}, vs = doc.versions || [];
    if (!vs.length) return null;
    for (var i = vs.length - 1; i >= 0; i--) {
      if (vs[i].version === doc.current_version) return vs[i];
    }
    return vs[vs.length - 1];
  }

  /* Games played ON one exact list. `versions.json` already carries `record`
   * and `games` per row — `deck_versions.report()` does the join from `log_ids`
   * — so this reads it rather than re-deriving it in the browser. */
  function versionRecord(v) {
    if (!v) return null;
    var r = v.record;
    if (!r) return null;
    var games = v.games !== undefined ? v.games
      : (r.win || 0) + (r.loss || 0) + (r.draw || 0);
    if (!games) return null;
    return { games: games, win: r.win || 0, loss: r.loss || 0 };
  }

  function coverPanel(d) {
    var info = d.info || {}, entry = d.entry || {};
    var g = info.goldfish || {}, rec = info.record || {};
    var stamp = stampOf(info, entry);

    var art = entry.image
      ? '<img class="cov-shot" src="' + esc(entry.image) + '" alt="" loading="lazy">'
      : '<div class="cov-shot cov-shot-none"></div>';

    var marks = [(info.colour_identity || []).join(''),
                 info.size ? info.size + ' cards' : null,
                 info.lands ? info.lands + ' lands' : null]
      .filter(Boolean).join(' · ');

    var ident = '<div class="cov-id">' +
      '<div class="cov-alias">' + esc(entry.deck_name || info.slug || '') + '</div>' +
      '<div class="cov-sub">' + esc((info.commander || []).join(' // ')) + '</div>' +
      '<dl class="cov-book">' +
        '<dt>Case no.</dt><dd>' + esc(info.slug || '') + '</dd>' +
        '<dt>Marks</dt><dd>' + esc(marks) + '</dd>' +
        '<dt>First booked</dt><dd>' + (rec.first_played
          ? esc(rec.first_played)
          : '<span class="ev">never played</span>') + '</dd>' +
      '</dl></div>';

    /* THE MO — what this deck is designed to do, in one line. `engine.thesis`
     * is the only sentence in the file that answers it, and when there is none
     * the absence is NAMED: a cover sheet with no MO is a file nobody has
     * opened, which is worth saying out loud rather than leaving blank. */
    var mo = info.engine && info.engine.thesis
      ? '<p class="cov-mo">' + esc(info.engine.thesis) + '</p>'
      : '<p class="cov-mo cov-mo-none">No engine model yet — nothing here ' +
        'states what this deck is trying to do. <code>/analyze-engine ' +
        esc(info.slug || '') + '</code></p>';
    /* THE BRIEF'S ABSENCE, said here because it is a statement about what the
     * deck is TRYING to be. A brief is the written intent a build starts from;
     * a deck with none is absent rather than empty, and the difference matters
     * during a refactor when the brief and the 99 legitimately disagree. */
    var briefless = (!info.brief || !info.brief.playstyle)
      ? '<p class="ev cov-brief">No brief authored for this list. A brief is ' +
        'the written intent a build starts from; this deck has none, which is ' +
        'absent rather than empty.</p>'
      : '';

    /* THE WORD AND ITS MEASURE, together and never apart. `engine_health` is a
     * VERDICT, which this repo normally refuses to publish — the obsolescence
     * index shipped one over a measure and 36.5% of 22,753 pairs failed a
     * purely mechanical check. It ships here only because the bands are a named
     * constant the pilot can move and the rate rides beside the word. */
    var h = info.engine_health, health = '';
    if (h && h.word) {
      health = '<div class="cov-health is-' + esc(h.word.toLowerCase()) + '"' +
        ' title="' + esc((h.bands || []).map(function (b) {
          return b[1] + ' ≥ ' + Math.round(b[0] * 100) + '%';
        }).join(' · ')) + '">' +
        '<span class="cov-health-w">' + esc(h.word) + '</span>' +
        '<span class="cov-health-y">' + esc(h.why || '') + '</span></div>';
    } else if (h) {
      health = '<div class="cov-health is-none"><span class="cov-health-w">' +
        'ENGINE NOT RATED</span><span class="cov-health-y">' +
        esc(h.why || '') + '</span></div>';
    }

    /* THREE NUMBERS, AND THE RECORD IS THIS VERSION'S — not the lifetime total.
     * A deck that went 0–3 on a list you have since replaced is a fact about a
     * deck that no longer exists, and filing it on the cover sheet under
     * "record" is the most misleading thing this page could do. */
    var cur = currentVersion(d);
    var vrec = versionRecord(cur);
    var nums = '<div class="cov-nums">' +
      head('Commander by T6',
           (g.commander_cast_by_turn_6_pct === undefined ||
            g.commander_cast_by_turn_6_pct === null)
             ? null : g.commander_cast_by_turn_6_pct + '%',
           'seeded goldfish') +
      head('Keepable sevens', keepRate(d), 'can act by turn three') +
      head('Record, ' + (cur ? versionName(cur) : 'this version'),
           vrec ? (vrec.games + ' · ' + vrec.win + '–' + vrec.loss) : null,
           cur ? 'games on this exact list' : 'no version resolved') +
      '</div>';

    var body = '<div class="cov">' + art +
      '<div class="cov-main">' +
        '<div class="cov-stamp is-' + stamp.kind + '" title="' + esc(stamp.why || '') + '">' +
          (stamp.pin ? '<span class="wb-pin" aria-hidden="true">📌</span>' : '') +
          esc(stamp.word) + '</div>' +
        ident + mo + briefless + health + nums +
      '</div></div>';
    return section('cover', 'var(--tier-data)', body);
  }

  // ══ THE RAP SHEET ═════════════════════════════════════════════════════
  //
  // One row per version: what changed, why, what was expected, what happened.
  // ROWS ARE ADDED, NEVER EDITED — that is what makes the file a record rather
  // than a summary, and it is why a row where `observed` contradicts `expected`
  // is the most valuable thing on the page rather than an embarrassment.
  //
  // EVERY COLUMN WAS ALREADY ON THE WIRE. `versions.json` is fetched on every
  // load and carries `in[]` and `out[]` as CARD NAMES, `notes`, `first_date`,
  // `tags` and `log_ids`. The old `recordPanel` rendered `+7 −5` and threw the
  // names away, with a comment admitting it: "the counts are what fits on a
  // row." They fit in a table.

  /* The pilot's own sentence about why a version happened. Authored, in
   * `deck_versions.json`'s tag note — Ur-Dragon's v1.0.1 reads "PATCH: the mana
   * base only… Priced on the goldfish before applying — turn-five land drop
   * 52.5% to 65.1%", which is the WHY and the EXPECTED in one breath. */
  function versionNote(d, v) {
    /* THE NOTE IS ON THE TAG, not on the version row. `deck_versions` keeps
     * versions DERIVED (a git walk) and tags AUTHORED, and the sentence the
     * pilot wrote about why a version happened belongs to the name they gave
     * it — so it is looked up through the tag rather than read off the row,
     * whose own `notes` key is null on every deck in the fleet. */
    var tags = (d.versions || {}).tags || {};
    for (var i = 0; i < (v.tags || []).length; i++) {
      var t = tags[v.tags[i]];
      if (t && t.note) return t.note;
    }
    return '';
  }

  /* IN and OUT, inline and wrapping rather than one name per line.
   *
   * A vertical list looks right on a three-card patch and becomes a wall on a
   * real one — Ur-Dragon's v1.0.2 moved 18 in and 17 out, which is 35 lines in
   * a table cell. Inline, they wrap to the measure and the row stays a row.
   * Cut cards keep their names struck through rather than vanishing: what LEFT
   * is half of what a version IS. */
  var SWATCH_MAX = 14;

  function swatch(names, mark, label) {
    if (!names || !names.length) return '';
    var shown = names.slice(0, SWATCH_MAX);
    var rest = names.length - shown.length;
    return '<div class="rap-side"><span class="rap-side-k">' + esc(label) +
      ' ' + names.length + '</span><ul class="rap-cards">' +
      shown.map(function (c) {
        return '<li class="' + mark + '">' + esc(c) + '</li>';
      }).join('') +
      (rest > 0 ? '<li class="rap-more">+' + rest + ' more</li>' : '') +
      '</ul></div>';
  }

  function rapSheetPanel(d) {
    var vs = ((d.versions || {}).versions) || [];
    if (!vs.length) {
      return section('rap-sheet', 'var(--tier-data)',
        '<p class="ev">No committed versions yet. A version is a commit whose ' +
        'parsed decklist differs from the one before it, so the first one ' +
        'arrives when this list is committed.</p>');
    }
    var cur = currentVersion(d);
    /* WHICH ROW IS SLEEVED. `paper` is a TOP-LEVEL block on the document naming
     * one version, not a flag on the row — 📌 means cardboard and belongs on
     * exactly one line of the rap sheet. */
    var paperV = ((d.versions || {}).paper || {}).version;
    var rows = vs.slice().reverse().map(function (v) {
      var why = versionNote(d, v);
      var rec = versionRecord(v);
      var isNow = cur && v.version === cur.version;
      /* THE HONEST COLUMN. Most versions have no games, and saying so is the
       * point: an untested hypothesis is the normal state of a deck between
       * pod nights, and a table that hid it would read as if every change had
       * been validated. */
      var observed = rec
        ? '<b>' + rec.games + ' game' + (rec.games === 1 ? '' : 's') + '</b> · ' +
          rec.win + '–' + rec.loss
        : '<span class="ev">not played on this list</span>';
      /* THE BASELINE ROW IS NOT A CHANGE. V1 is the first tracked list, so its
       * `in` is the whole deck — 95 names for Ur-Dragon — and listing them
       * reads as a 95-card swap. What the row means is "this is where the
       * numbering starts", and the size says it in four words. */
      var baseline = !(v.out || []).length && (v['in'] || []).length > 40;
      return '<tr' + (isNow ? ' class="rap-now"' : '') + '>' +
        '<th scope="row">' + esc(versionName(v)) +
          (isNow ? ' <span class="chip">current</span>' : '') +
          (paperV === v.version
            ? ' <span class="wb-pin" aria-hidden="true">📌</span>' : '') +
        '</th>' +
        '<td class="rap-when">' + esc(v.first_date || '') + '</td>' +
        '<td class="rap-diff">' +
          (baseline
            ? '<span class="ev">the first tracked list — ' +
              (v.size || (v['in'] || []).length) + ' cards, where the ' +
              'numbering starts</span>'
            : swatch(v['in'], 'rap-in', 'in') + swatch(v.out, 'rap-out', 'out') ||
              '<span class="ev">no card moved</span>') +
        '</td>' +
        '<td class="rap-why">' + (why ? esc(why)
          : '<span class="ev">no note — <code>deck-version ' +
            esc((d.info || {}).slug || '') + ' tag</code> records why</span>') +
        '</td>' +
        '<td class="rap-obs">' + observed + '</td></tr>';
    }).join('');

    return section('rap-sheet', 'var(--tier-data)',
      '<div class="rap-wrap"><table class="rap">' +
      '<thead><tr><th scope="col">Version</th><th scope="col">Date</th>' +
      '<th scope="col">What changed</th><th scope="col">Why, and what was expected</th>' +
      '<th scope="col">Observed</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="ev rap-foot">Rows are added, never edited. Where <b>observed</b> ' +
      'contradicts what was expected, that disagreement is the record — it is ' +
      'not corrected here.</p>');
  }

  // ══ PRIORS ════════════════════════════════════════════════════════════
  //
  // Every game, one row, with how it ended. The log has held `opponents` and
  // `tags` since it shipped and the page rendered neither; `result` was only
  // ever shown as a running total. A game is a record.

  var CAUSE_GLOSS = {
    'mana-drought': 'colour or land screw',
    'removal': 'picked apart one card at a time',
    'wipe': 'a board wipe, and no rebuild',
    'combo': 'an opponent comboed off',
    'politics': 'the table converged',
    'raced': 'someone was simply faster',
    'stalled': 'the engine never assembled',
    'won': 'the deck closed the game'
  };

  /* Which version was on the table. The log stamps `decklist_sha256`; the
   * version list carries every byte-sha that maps to each version. That join is
   * the whole reason the log carries a sha at all. */
  function versionOfSha(d, sha) {
    var vs = ((d.versions || {}).versions) || [];
    for (var i = 0; i < vs.length; i++) {
      var m = vs[i].decklist_sha256s || [];
      if (m.indexOf(sha) >= 0 || vs[i].decklist_sha256 === sha) return vs[i];
    }
    return null;
  }

  function priorsPanel(d) {
    var log = d.log || [];
    if (!log.length) {
      return section('priors', 'var(--tier-coach)',
        '<p class="ev">No games logged. <code>manamap pilot deck-notes ' +
        esc((d.info || {}).slug || '') +
        ' add "…" --result win|loss --opponents 3 --cause &lt;code&gt;</code></p>');
    }
    var causes = ((d.causes || {}).entries) || {};
    var rows = log.slice().reverse().map(function (e) {
      var v = versionOfSha(d, e.decklist_sha256);
      var c = causes[e.id];
      return '<tr>' +
        '<th scope="row">' + esc(e.at ? e.at.slice(0, 10) : '') + '</th>' +
        '<td>' + (e.opponents ? esc(e.opponents + 1) + '-player' :
                  '<span class="ev">—</span>') + '</td>' +
        '<td>' + (v ? esc(versionName(v)) :
                  '<span class="ev">unmatched list</span>') + '</td>' +
        '<td class="pri-' + esc(e.result || 'none') + '">' +
          esc((e.result || '—').toUpperCase()) + '</td>' +
        '<td>' + (c
          ? '<span class="pri-cause" title="' + esc(c.note || '') + '">' +
            esc(c.cause) + '</span> <span class="ev">' +
            esc(CAUSE_GLOSS[c.cause] || '') + '</span>'
          : '<span class="ev">no cause filed</span>') + '</td></tr>';
    }).join('');

    /* THE ROLL-UP IS THE POINT OF CODING THE CAUSE. Three losses to `removal`
     * and three to `mana-drought` are two different decks with the same record,
     * and prose cannot be counted. Composed by `deck-info`, not here, so the
     * rows and the totals cannot disagree about the vocabulary. */
    var counts = ((d.info || {}).record || {}).cause_counts || {};
    var keys = Object.keys(counts);
    var roll = keys.length
      ? '<p class="pri-roll">' + keys.sort(function (a, b) {
          return counts[b] - counts[a] || (a < b ? -1 : 1);
        }).map(function (k) {
          return '<span class="chip">' + esc(k) + ' ×' + counts[k] + '</span>';
        }).join(' ') + '</p>'
      : '';

    return section('priors', 'var(--tier-coach)',
      '<div class="rap-wrap"><table class="rap pri">' +
      '<thead><tr><th scope="col">Date</th><th scope="col">Pod</th>' +
      '<th scope="col">Version</th><th scope="col">Result</th>' +
      '<th scope="col">How it ended</th></tr></thead><tbody>' + rows +
      '</tbody></table></div>' + roll +
      '<p class="ev">The cause is the pilot’s own claim about their own game, ' +
      'from a closed vocabulary so the counts above mean something. Nothing ' +
      'derives it.</p>');
  }

  // ══ THE ANALYST'S ASSESSMENT ══════════════════════════════════════════
  //
  // LAST, AND SEPARATE. A dossier where the analyst's opinion is mixed into the
  // record loses trust — and that is exactly what the old page did, rendering
  // the diagnosis verdict as one inline sentence inside the audit panel, three
  // lines under a measured figure and in the same typeface.

  function assessmentPanel(d) {
    var dg = (d.info || {}).diagnosis;
    if (!dg) {
      /* NO READ IS ITSELF A FACT ABOUT THE FILE, and the last section is where
       * a reader looks for one. `absent()` returns '' when the stage is not in
       * the deck's todo list — right for a reference panel, wrong for a tab. */
      return section('assessment', 'var(--tier-coach)',
        '<p class="ev">Nobody has read this deck yet. The measurements above ' +
        'are what it is; an assessment is what someone thinks of it, and there ' +
        'is none on file. <code>/diagnose-deck ' +
        esc((d.info || {}).slug || '') + '</code></p>');
    }
    var cur = currentVersion(d);
    var head = '<div class="as-head">' +
      '<span class="as-verdict">' + esc(dg.verdict || 'no verdict stated') + '</span>' +
      '</div>';
    var meta = '<dl class="facts">' +
      '<dt>Read against</dt><dd>' + (dg.stale
        ? '<span class="chip stale">a list this deck no longer runs</span>'
        : esc(cur ? versionName(cur) : 'the current list')) + '</dd>' +
      '<dt>Skeptic</dt><dd>' + esc(dg.skeptic || 'not run') + '</dd>' +
      '</dl>';
    /* STALE IS NOT WRONG. The read was true about the list it was made against;
     * it just does not describe this one. Saying "superseded" rather than
     * greying it out is the difference between a file and a dashboard. */
    var note = dg.stale
      ? '<p class="as-stale">Superseded by a later list. Kept as written — it ' +
        'was true of the deck it was read against. <code>/diagnose-deck ' +
        esc((d.info || {}).slug || '') + '</code> for a current read.</p>'
      : '';
    return section('assessment', 'var(--tier-coach)', head + meta + note);
  }

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
    /* THE FILE, then everything else.
     *
     * The nine dossier sections come first, in `DOSSIER` order — cover sheet at
     * the top because the whole point of one is that you absorb it in thirty
     * seconds, and the assessment last because a file where the analyst's
     * opinion sits inside the record loses trust.
     *
     * The remaining panels are APPENDED UNCHANGED beneath them. They are the
     * evidence and they will become numbered exhibits; until then nothing is
     * lost and nothing is half-converted, which is the only way to move a page
     * this size without a window where it renders neither shape. */
    var html = [
      coverPanel(d),
      rapSheetPanel(d),
      rosterPanel(d),
      vitalsPanel(d),
      priorsPanel(d),
      logPanel(d),
      questionsPanel(d),
      assessmentPanel(d),

      // ── the exhibits, not yet numbered ──
      nextPanel(d), statusPanel(d), auditPanel(d),
      enginePanel(d), tablePanel(d), targetingPanel(d), askedPanel(d),
      briefPanel(d), constellationPanel(d),
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
      // GATED ON THE MANIFEST, not fetched unconditionally like FILES: a deck
      // with no logged games has no artifact, and a 404 read as "absent" is the
      // ambiguity the `has` map exists to resolve.
      var picardJob = (entry.has || {}).captains_log
        ? getJSON(BASE + entry.slug + '/captains_log.json') : Promise.resolve(null);

      return Promise.all([Promise.all(jobs), Promise.all(stackJobs),
                          Promise.all(simJobs), Promise.all(expJobs),
                          Promise.all(rxJobs), logJob, debriefJob, picardJob])
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
          d.captainsLog = both[7];
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
