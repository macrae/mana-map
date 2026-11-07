/* The branch workbench: one screen for the decision a refactor asks you to make.
 *
 * WHY THIS IS ITS OWN PAGE. The dossier already renders nine panels about a deck
 * that EXISTS. A branch is a deck that does not — a proposal with a bill attached
 * — and the question it asks is a different one: not "where does this deck
 * stand" but "is this change worth the money". Growing the dossier's read-only
 * branch panel into that would have made the busiest panel on the page a tenth.
 *
 * IT RENDERS WHAT IT IS GIVEN AND COMPUTES NOTHING. Every figure comes from
 * `net_change.json`, which `manamap pilot net-change --write` produces and
 * `validate-net-change` gates. A second opinion computed here could disagree with
 * the artifact the pilot acted on, which is the divergence this repo has paid for
 * more than once.
 */
(function () {
  'use strict';

  var DATA = '../data/';
  // This branch's own `cards.json` rows by name, so its gallery and hovers draw its art.
  var branchRows = {};

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function getJSON(url) {
    return fetch(url, { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  /* THE REPORT, READ STRICTLY. A 404 is a branch nobody has measured. A report
   * that will not parse, or a request that failed, is something else, and saying
   * "not measured" over it sends the pilot to re-run a report that exists. */
  function getReport(url) {
    return fetch(url, { cache: 'no-cache' }).then(function (r) {
      if (r.status === 404) return { doc: null };
      if (!r.ok) return { doc: null, failed: 'the server answered ' + r.status };
      return r.json().then(function (doc) { return { doc: doc }; },
                           function () { return { doc: null, failed: 'net_change.json will not parse' }; });
    }).catch(function (e) { return { doc: null, failed: 'the request failed: ' + e.message }; });
  }

  /* A CARD NAME IS A LINK EVERYWHERE ON THIS PAGE, and it is `Shell`'s so that
   * deck.html and workbench.html get the same one. This file used to carry its
   * own `cardref` + `card-pop` markup inside `cardList` and nowhere else, so the
   * bill showed you a card and the diff — the panel whose entire content is
   * card names — showed you sixty strings. */
  function card(name) {
    return window.Shell && Shell.cardLink ? Shell.cardLink(name) : esc(name);
  }

  function num(v, dp) {
    if (v == null) return '—';
    return Number(v).toFixed(dp == null ? 3 : dp);
  }

  function signed(v, dp) {
    if (v == null) return '—';
    var s = Number(v).toFixed(dp == null ? 3 : dp);
    return (Number(v) > 0 ? '+' : '') + s;
  }

  /* ── panels ─────────────────────────────────────────────────────────── */

  /* AN ABSENT SECTION MUST SAY WHAT IT IS AND HOW TO GET IT. A branch with no
   * report is indistinguishable from a broken page otherwise — the defect the
   * dossier's `todo` block already exists to prevent. */
  function absent(title, what, how) {
    return '<section class="panel absent"><h2>' + esc(title) + '</h2>' +
           '<p class="ev">' + esc(what) + '</p>' +
           (how ? '<pre class="cmd">' + esc(how) + '</pre>' : '') + '</section>';
  }

  function objectivePanel(meta, nc) {
    var o = (nc && nc.objective) || (meta && meta.objective);
    var g = nc && nc.objective_grade;
    if (!o) {
      return '<div class="guard none" id="guard"><p class="guard-line"><b>Guard:</b> none set.</p></div>';
    }
    var state = g ? g.state : 'not measured';
    var cls = { 'met': 'met', 'not met': 'failed',
                'not resolvable': 'unresolved' }[state] || 'unknown';
    var word = { 'met': 'Guard met', 'not met': 'Guard not met',
                 'not resolvable': 'Guard not resolved',
                 'not measured': 'Guard not measured' }[state] || 'Guard';
    var out = '<div class="guard ' + cls + '" id="guard"><p class="guard-line"><b>' + esc(word) +
      ':</b> <code>' + esc(o.axis) + ' ' + esc(o.op) + ' ' + esc(o.value) +
      (o.pod ? ' @' + esc(o.pod) : '') + '</code>' +
      (g && g.reading != null ? ', read <b>' + esc(num(g.reading)) + '</b>' : '') + '</p>';
    if (g) {
      if (g.why) out += '<p class="ev">' + esc(g.why) + '</p>';
      // A FORGE OBJECTIVE TRAVELS WITH THE INTERVAL ON THE DIFFERENCE and the
      // table's null — the two numbers the reading was actually read against.
      if (g.difference && g.difference.ci95) {
        out += '<p class="ev">vs the deck: &Delta; ' + signed(g.difference.delta) +
          ' &nbsp;·&nbsp; CI [' + signed(g.difference.ci95[0]) + ', ' +
          signed(g.difference.ci95[1]) + ']' +
          (g.difference.excludes_zero ? ' — excludes zero' : ' — spans zero') + '</p>';
      }
      if (g.null && g.null.rate != null) {
        out += '<p class="ev">the table\'s null is ' + num(g.null.rate) +
          ' (' + esc(String(g.null.games)) + ' games) at ' + esc(g.null.pod) + '</p>';
      }
    }
    if (o.why) out += '<p class="ev">Written when the branch was opened: ' + esc(o.why) + '</p>';
    return out + '</div>';
  }

  /* THE ROWS, GROUPED BY WHAT THEY MEAN AT A TABLE. Mirrors
   * `net_change.ROW_GROUPS` (a test holds the two equal). A row named nowhere
   * lands in Other, so a new metric cannot drop off the page by being forgotten. */
  var ROW_GROUPS = [
    ['Speed', ['killed by T6', 'killed by T10', 'damage @T10']],
    ['Board', ['board power @T6', 'hoard @T6', 'hoard @T10']],
    ['Mana and hands', ['stall, 2 in a row', 'missed drop by T5', 'mulliganed',
                        'hand can act by T3', "kept hand passes the pilot's keep rule"]],
    ['Cards and answers', ['extra cards by T8', 'interaction affordable @T6',
                           'finisher wheel by T8', 'wheels cast by T10',
                           'dead wheels in hand @T5']],
    ['Big bodies', ['a 6+ drop out by T6', 'turns saved on big bodies', 'experience @T7']]
  ];
  var READ_WORD = { better: 'Better', worse: 'Worse', noise: 'No call' };

  /* DISPLAY ONLY, the same rule as `net_change.format_row`: a rate as a percentage
   * and its change in points, a mean to the precision its size earns. The verdict
   * is the artifact's and is never recomputed. */
  function rowFigures(r) {
    var rate = r.unit === 'rate', scale = rate ? 100 : 1;
    var top = Math.max(Math.abs(r.champion), Math.abs(r.branch));
    var dp = rate ? 1 : top < 1 ? 3 : top < 10 ? 2 : 1;
    function fig(v) { return num(v * scale, dp) + (rate ? '%' : ''); }
    // ZERO CARRIES ITS SIGN, as the terminal's does: an A/A reads +0.0, not 0.0.
    function sign(v) { var x = v * scale; return (x >= 0 ? '+' : '') + x.toFixed(dp); }
    var ci = r.ci95_diff;
    return {
      now: fig(r.champion), branch: fig(r.branch),
      change: sign(r.delta) + (rate ? ' pts' : ''),
      range: ci ? '[' + sign(ci[0]) + ', ' + sign(ci[1]) + ']' : '',
      read: READ_WORD[r.verdict] || r.verdict,
      resolves: r.mde == null ? '' : num(r.mde * scale, dp) + (rate ? ' pts' : '')
    };
  }

  function defId(name) {
    return 'def-' + String(name).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  }

  function profileCount(table) {
    var n = { better: 0, worse: 0, noise: 0 };
    table.forEach(function (r) { if (r.verdict in n) n[r.verdict]++; });
    return n;
  }

  function rowHtml(r) {
    var f = rowFigures(r);
    return '<tr class="v-' + esc(r.verdict) + '">' +
      '<th scope="row" class="nc-name"><a href="#' + defId(r.measure) + '">' + esc(r.measure) + '</a>' +
      (r.better_is === 'lower' ? ' <span class="nc-low" title="Lower is better on this row">lower is better</span>' : '') + '</th>' +
      '<td class="n nc-ab">' + esc(f.now) + ' <span class="nc-arrow">&rarr;</span> ' + esc(f.branch) + '</td>' +
      '<td class="n nc-change">' + esc(f.change) + '</td>' +
      '<td class="n nc-range">' + esc(f.range) + '</td>' +
      '<td class="verdict"><b>' + esc(f.read) + '</b>' +
      (r.verdict === 'noise' && f.resolves
        ? ' <span class="ev nc-res">resolves ' + esc(f.resolves) + '</span>' : '') +
      '</td></tr>';
  }

  /* THE PROFILE LEADS, and no single word sits on top of it. The bench judges the
   * whole profile, so the first thing read is the count and the rows; the rule's
   * stated word and the guard sit directly under them, labelled as what they are.
   * `#measured` is the deep link the workbench's rack points at. */
  function tablePanel(nc, meta) {
    if (!nc) return '';
    var table = nc.table || [];
    var byName = {}, placed = {};
    table.forEach(function (r) { byName[r.measure] = r; });
    var groups = ROW_GROUPS.map(function (g) {
      var rows = g[1].filter(function (n) { return byName[n]; }).map(function (n) {
        placed[n] = 1; return byName[n];
      });
      return [g[0], rows];
    });
    groups.push(['Other', table.filter(function (r) { return !placed[r.measure]; })]);
    var body = groups.filter(function (g) { return g[1].length; }).map(function (g) {
      return '<tbody><tr class="nc-group"><th colspan="5" scope="rowgroup">' + esc(g[0]) + '</th></tr>' +
        g[1].map(rowHtml).join('') + '</tbody>';
    }).join('');
    var n = profileCount(table);
    var rec = nc.recommendation || {};
    return '<section class="panel nc-profile" id="measured">' +
      '<h2>The profile</h2>' +
      '<p class="nc-count"><span class="c-better"><b>' + n.better + '</b> better</span>' +
      '<span class="c-worse"><b>' + n.worse + '</b> worse</span>' +
      '<span class="c-noise"><b>' + n.noise + '</b> no call</span></p>' +
      (rec.forge_warning ? '<p class="warn nc-forge-warn" id="forgeWarning">' + esc(rec.forge_warning) + '</p>' : '') +
      '<div class="tablewrap"><table class="netchange nc-table"><thead><tr>' +
      '<th scope="col">Measure</th><th class="n" scope="col">Now &rarr; Branch</th>' +
      '<th class="n" scope="col">Change</th><th class="n" scope="col">95% range</th>' +
      '<th scope="col">Read</th></tr></thead>' + body + '</table></div>' +
      '<p class="ev nc-key"><b>No call:</b> smaller than this run can resolve, which is not the same as ' +
      'no change. <b>Pts:</b> games per 100. The range is the 95% interval on the difference; ' +
      'the read is Holm-corrected across the ' + table.length + ' rows. ' +
      esc((nc.harness.iterations || 0).toLocaleString()) + ' games per list, same seeds.' +
      (nc.design && nc.design.primary ? ' The primary is ' + esc(nc.design.primary) + '.' : '') + '</p>' +
      objectivePanel(meta, nc) + verdictPanel(nc) +
      '</section>';
  }

  /* DEFINITIONS AT THE FOOT, ONE PANEL. Twelve folds on twelve rows put the same
   * question on the page twelve times; each row name links to its entry here, so a
   * definition is one click away and the table still scans in one pass. */
  function definitionsPanel(nc) {
    if (!nc) return '';
    var defs = nc.definitions || {};
    var n = (nc.table || []).length;
    function fix(t) { return String(t == null ? '' : t).replace('The sixteen rows', 'The ' + n + ' rows'); }
    var rows = (nc.table || []).filter(function (r) { return r.what; }).map(function (r) {
      return '<div class="nc-def" id="' + defId(r.measure) + '"><dt>' + esc(r.measure) +
        ' <span class="ev">' + esc(r.unit) + ', ' + esc(r.better_is) + ' is better</span></dt><dd>' +
        (r.reads_as ? '<p class="nc-here">' + esc(r.reads_as) + '</p>' : '') +
        '<p>' + esc(r.what) + '</p>' +
        (r.scale ? '<p class="ev">Scale: ' + esc(r.scale) + '</p>' : '') +
        '<p class="ev"><b>Why:</b> ' + esc(r.why_we_care) + '</p></dd></div>';
    }).join('');
    var derived = (defs.derived || []).map(function (d) {
      return '<div class="nc-def"><dt>' + esc(d.name) + '</dt><dd><p>' + esc(fix(d.what)) +
        '</p><p class="ev"><b>Why:</b> ' + esc(fix(d.why)) + '</p></dd></div>';
    }).join('');
    var fe = defs.forge_endpoints || {};
    var ep = nc.forge && nc.forge.available ? Object.keys(fe).map(function (k) {
      var d = fe[k] || {};
      return '<div class="nc-def"><dt>' + esc(k) + '</dt><dd><p>' + esc(d.what) + '</p>' +
        (d.why ? '<p class="ev"><b>Why:</b> ' + esc(d.why) + '</p>' : '') + '</dd></div>';
    }).join('') : '';
    return '<section class="panel nc-defs" id="definitions"><h2>What the figures mean</h2><dl>' + rows +
      (derived ? '</dl><h3>The figures that are not rows</h3><dl>' + derived : '') +
      (ep ? '</dl><h3>Forge</h3><dl>' + ep : '') + '</dl></section>';
  }

  /* WHAT `recommend` SAID THAT NOTHING ELSE ON THE PAGE SAYS. The reward and paid
   * rows are the table's and the unmeasured rows are the blind spots', so only the
   * unresolved line (which row came closest) and the structural one remain. */
  var RISK_COPY = { unresolved: 'no answer', structural: 'true of the whole harness' };

  function otherRisks(nc) {
    var r = nc && nc.recommendation;
    var rows = ((r && r.risk) || []).filter(function (x) {
      return x.kind !== 'paid' && x.kind !== 'unmeasured';
    });
    if (!rows.length) return '';
    return '<h3>Also true of this run</h3><ul class="ledger-list">' + rows.map(function (x) {
      return '<li class="risk k-' + esc(x.kind) + '"><span class="kind">' +
        esc(RISK_COPY[x.kind] || x.kind) + '</span><b>' + esc(x.what) + '</b>' +
        (x.detail ? '<span class="reads">' + esc(x.detail) + '</span>' : '') +
        (x.why_it_matters ? '<span class="ev">' + esc(x.why_it_matters) + '</span>' : '') +
        '</li>';
    }).join('') + '</ul>';
  }

  /* THE MANA, which this page used to define and never show. For a land swap it is
   * the whole of the evidence: the rows cannot rank two lands that make the same
   * colours. A colour the deck never touches is not a row. */
  var COLOUR_NAME = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };

  function manaPanel(nc) {
    var m = nc && nc.mana;
    if (!m) return '';
    if (!m.available) {
      return absent('The mana', m.why || 'Not measured for this report.', null).replace('<section class="panel absent"', '<section class="panel absent" id="mana"');
    }
    function ab(pair, fn) {
      return esc(fn(pair[0])) + ' <span class="nc-arrow">&rarr;</span> ' + esc(fn(pair[1]));
    }
    function same(v) { return v; }
    function pct(v) { return v == null ? '—' : Math.round(v * 100) + '%'; }
    function gap(v) { return (v > 0 ? '+' : '') + v; }
    var rows = (m.colours || []).filter(function (r) {
      return r.target[0] || r.target[1] || r.have[0] || r.have[1];
    }).map(function (r) {
      var oc = (m.on_curve || {})[r.colour] || [null, null];
      return '<tr><th scope="row">' + esc(COLOUR_NAME[r.colour] || r.colour) + '</th>' +
        '<td class="n">' + ab(r.have, same) + '</td><td class="n">' + ab(r.target, same) + '</td>' +
        '<td class="n">' + ab(r.gap, gap) + '</td><td class="n">' + ab(oc, pct) + '</td></tr>';
    }).join('');
    var life = m.life || [];
    var la = life[0] || {}, lb = life[1] || {};
    function lifeRow(key, label) {
      if (la[key] == null || lb[key] == null) return '<li>' + esc(label) + ': not measured</li>';
      return '<li>' + esc(label) + ': ' + esc(la[key]) + ' &rarr; ' + esc(lb[key]) + '</li>';
    }
    return '<section class="panel nc-mana" id="mana"><h2>The mana</h2>' +
      '<p class="ev">Deterministic, so no interval. For a land swap this table is the evidence: the rows above cannot rank two lands that make the same colours.</p>' +
      '<div class="nc-duo"><div class="tablewrap"><table class="netchange nc-mana-table"><thead><tr><th scope="col">Colour</th>' +
      '<th class="n" scope="col">Sources</th><th class="n" scope="col">Target</th>' +
      '<th class="n" scope="col">Gap</th><th class="n" scope="col">On curve</th></tr></thead><tbody>' + rows + '</tbody></table></div><div>' +
      '<ul class="nc-facts"><li>Lands: ' + ab(m.lands, same) + '</li><li>Always tapped: ' + ab(m.enters_tapped_always, same) + '</li>' +
      (life.length ? lifeRow('recurring_per_tap_cycle', 'Life, every tap-cycle') + lifeRow('one_time_on_entry', 'Life, once on entry') : '') + '</ul>' +
      '<p class="ev"><b>On curve:</b> the chance the base casts a spell of that colour on the turn Karsten’s table sizes it for. The gap is the figure, not the source count: a branch that changes its spells moves the target under the base. ' +
      (life.length ? 'The two life figures are never summed: a painland charges on every activation, a fetch or a shock once. ' : '') +
      (m.note ? esc(m.note) : '') + '</p></div></div></section>';
  }

  /* THE MISMATCH RENDERS ABOVE THE RATE, for the reason the CLI prints it
   * there: underneath, it reads as a footnote to a number already taken in.
   * An arm listed here has no Forge run on the list this page is about, so its
   * figure describes a DIFFERENT ninety-nine. */
  function mismatchNote(f) {
    var m = f.list_mismatch, out = '';
    if (!m) return '';
    Object.keys(m).forEach(function (arm) {
      var r = m[arm];
      out += '<p class="warn">' + esc(arm.toUpperCase()) +
        ' MEASURED ON A DIFFERENT LIST — ' + r.games + ' game(s) on ' +
        esc((r.played || []).join(', ')) + ', ' + esc(r.on_disk) +
        ' on disk. ' + esc(r.reads_as) + '</p>';
    });
    return out;
  }

  function forgePanel(nc) {
    if (!nc || !nc.forge) return '';
    var f = nc.forge;
    if (!f.available) {
      return absent('The real table', f.why, null).replace('<section class="panel absent"', '<section class="panel absent" id="forge"');
    }
    function routes(r) {
      var keys = Object.keys(r.won_by || {});
      if (!keys.length) return '';
      return '<div class="ev">' + keys.map(function (k) {
        return esc(k) + ' &times;' + r.won_by[k];
      }).join(' · ') + '</div>';
    }
    var under = f.mde != null && Math.abs(f.delta) < f.mde;
    return '<section class="panel" id="forge"><h2>The real table</h2>' +
      mismatchNote(f) +
      (f.pod ? '<p class="ev">at <b>' + esc(f.pod) + '</b> — ' + esc(f.basis || '') +
        '</p>' : '') +
      '<div class="forge"><div><div class="who">The deck</div><div class="big">' +
      f.champion.wins + '/' + f.champion.games + '</div>' + routes(f.champion) +
      '</div><div><div class="who">The branch</div><div class="big">' +
      f.branch.wins + '/' + f.branch.games + '</div>' + routes(f.branch) +
      '</div></div>' +
      '<p class="ev">&Delta; ' + signed(f.delta) + ' &nbsp;·&nbsp; CI [' +
      signed(f.ci95[0]) + ', ' + signed(f.ci95[1]) + '] &nbsp;·&nbsp; MDE ' +
      esc(f.mde) + '</p>' +
      (under ? '<p class="warn">UNDERPOWERED — this run could only resolve a ' +
        'difference of ' + esc(f.mde) + '. It rules out a large effect and ' +
        'cannot say which list is better.</p>' : '') +
      (f.null && f.null.rate != null ? '<p class="ev">the table\'s null is <b>' +
        num(f.null.rate) + '</b> — what our decks score in seat 0 here (' +
        esc(String(f.null.games)) + ' games); the deck reads ' +
        Math.round(100 * f.champion.rate / f.null.rate) + '% of it, the branch ' +
        Math.round(100 * f.branch.rate / f.null.rate) + '%</p>' : '') +
      '<p class="ev">' + esc(f.caveat || '') + '</p></section>';
  }

  /* THE BILL, and the five states are not five kinds of purchase. `elsewhere`
   * is a card sleeved in another deck — a trade-off, not money — and folding it
   * into `buy` reads as "spend on something already in the house".
   *
   * `free` was carved out of `elsewhere` on 2026-09-12 (#25). A card whose only
   * holders are broken down or retired costs NOTHING: there is no deck to
   * unsleeve and nothing to buy, and `source()`'s blocker already excluded it
   * while this tile still counted it. Measured on ur-dragon/eminence-v3:
   * elsewhere 6, of which 1 was in `sisay`, retired and in a pile — so the page
   * said "sleeved in another deck: 6" when five were contested. */
  var STATE_COPY = {
    in_deck: 'already in the deck',
    box: 'in a box — you own these',
    elsewhere: 'sleeved in a deck that is still together',
    free: 'loose — its deck is broken down',
    buy: 'to buy'
  };

  function billPanel(nc, meta) {
    var bill = nc && nc.bill;
    if (!bill) return '';
    var c = bill.counts || {};
    var owned = (c.in_deck || 0) + (c.box || 0) + (c.elsewhere || 0) +
                (c.free || 0);
    var total = Object.keys(c).reduce(function (a, k) { return a + c[k]; }, 0);
    var tiles = Object.keys(STATE_COPY).map(function (k) {
      return '<div class="tile s-' + k + '"><div class="k">' + (c[k] || 0) +
             '</div><div class="t">' + esc(STATE_COPY[k]) + '</div></div>';
    }).join('');
    var buy = (bill.cards || []).filter(function (r) { return r.state === 'buy'; });
    var elsewhere = (bill.cards || []).filter(function (r) {
      return r.state === 'elsewhere' && !r.free;
    });
    var cost = nc.recommendation && nc.recommendation.cost;
    var rc = cost || {};
    var raid = rc.free_to_raid || [];
    var where = elsewhere.length || raid.length
      ? '<details class="nc-where"><summary>Where the owned cards are</summary>' +
        cardList('To unsleeve from other decks', elsewhere) +
        (raid.length ? '<p class="ev">Free to take, because the deck it sits in is ' +
          'already apart: ' + esc(raid.map(function (x) {
            return x.name + ' (' + x.decks.join(', ') + ')';
          }).join('; ')) + '.</p>' : '') + '</details>'
      : '';
    return '<section class="panel" id="bill"><h2>The bill</h2>' +
      '<div class="tiles">' + tiles + '</div>' +
      (rc.reads_as ? '<p class="ev">' + esc(rc.reads_as) + '</p>' : '') +
      (rc.mergeable === false ? '<p class="ev warn">NOT MERGEABLE YET — <code>deck-branch ' +
        'merge</code> refuses while any card is unsourced. That is a question about ' +
        'cardboard, not about whether the branch is right.</p>' : '') +
      '<p class="lede">You already own <b>' + owned + ' of ' + total + '</b>.</p>' +
      (buy.length
        ? '<p class="ev bill-price" id="billPrice">' +
          (cost && typeof cost.buy_cents === 'number' ? pricedLine(cost) : 'pricing…') +
          '</p>'
        : '') +
      '<div class="nc-duo"><div>' + cardList('To buy', buy) + '</div><div>' + buyListBlock(buy) + '</div></div>' + where +
      '</section>';
  }

  /* THE BILL IN DOLLARS (2026-10-09). This paragraph used to say "Prices are
   * stripped from the card corpus by design, so this page cannot give you a
   * figure" — true of cards.csv, and it stayed true of the page after
   * `manamap pilot prices` started writing a dated `prices.json` beside each
   * list. The figure is that file's and nothing else's: dated evidence quoted
   * with its date and source, never a live lookup, never a graded row.
   *
   * Two readers of one file, same arithmetic as `net_change._priced_bill`:
   * `nm_cents × quantity` over the BUY rows, and a card the file does not
   * price is NAMED, never counted as zero. `net_change.json` carries the
   * figure in `recommendation.cost` when `net-change --write` ran after the
   * prices were written, and that paints first so the line is there without
   * a fetch; then `wireBillPrice` reads the branch's own `prices.json` and
   * repaints from it, because the file can be younger than the report (the
   * usual case the day prices land) or gone since. No file: the command that
   * writes one. Nothing to buy: no line at all — "≈ $0" is not a figure. */
  function dollars(cents) {
    var s = (cents / 100).toFixed(2);
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function pricedLine(p) {
    var unpriced = p.buy_unpriced || [];
    return '<b>≈ $' + dollars(p.buy_cents) + ' to buy</b> (as of ' +
      esc(p.prices_as_of) + ', ' + esc(p.prices_source) +
      (unpriced.length
        ? '; ' + unpriced.length + ' unpriced: ' + esc(unpriced.join(', '))
        : '') + ')';
  }

  function noPricesLine(slug, branch) {
    return 'No prices for this branch yet — ' +
      '<button class="todo-cmd" data-act="copy-cmd" title="Copy"><code>' +
      esc('manamap pilot prices ' + slug + ' --branch ' + branch + ' --write') +
      '</code><span class="todo-copy">copy</span></button> ' +
      '(Scryfall’s daily USD, or Mana Pool with a token).';
  }

  /* Mirrors `net_change._priced_bill` — keep the two in step. */
  function priceBill(prices, buy) {
    var cents = 0, unpriced = [];
    buy.map(function (r) { return r.name; }).sort().forEach(function (name) {
      var row = prices.cards[name] || {};
      var nm = row.nm_cents;
      if (typeof nm === 'number' && isFinite(nm)) {
        cents += nm * (parseInt(row.quantity, 10) || 1);
      } else {
        unpriced.push(name);
      }
    });
    return { buy_cents: cents, buy_unpriced: unpriced,
             prices_as_of: prices.as_of, prices_source: prices.source };
  }

  function wireBillPrice(slug, branch, buy) {
    var host = document.getElementById('billPrice');
    if (!host || !buy.length) return;
    var base = DATA + 'decks/' + encodeURIComponent(slug) + '/branches/' +
               encodeURIComponent(branch) + '/';
    getJSON(base + 'prices.json').then(function (doc) {
      var priced = doc && doc.cards && typeof doc.cards === 'object';
      host.innerHTML = priced ? pricedLine(priceBill(doc, buy))
                              : noPricesLine(slug, branch);
    });
  }

  /* THE BILL AS A PASTE (Area B4, 2026-10-09). Mana Pool's mass entry
   * (manapool.com/add-deck) takes a pasted decklist and nothing else — no URL
   * prefill — so the most this page can do is put the text on the clipboard and
   * open the page. With a local server the text is the server's
   * (`branch/buy-list`, the same function `manamap pilot buy-list --json`
   * prints); on the static site it is computed here from the bill's BUY rows
   * plus the branch's cards.json for the printings, and shown in a <pre> with a
   * plain Copy. Both forms go through the same line shape as
   * `check_in.decklist_line`: `N Name`, or `N Name (SET) CN`, `*F*` riding along. */
  var MANAPOOL_URL = 'https://manapool.com/add-deck';

  function buyListBlock(buy) {
    if (!buy.length) return '';
    return '<div class="buylist" id="buyList">' +
      '<h3>Buy list <span class="ev">(for Mana Pool)</span></h3>' +
      '<p class="ev"><label><input type="checkbox" id="buyExact"> exact printings ' +
      '<code>N Name (SET) CN</code></label></p>' +
      '<p class="ev buylist-hint">paste there, choose exact printings</p>' +
      '<div class="buylist-body"><p class="ev">preparing…</p></div>' +
      '</div>';
  }

  function buyLine(r, exact) {
    var s = (r.quantity || 1) + ' ' + r.name;
    if (exact && r.set && r.collector_number) {
      s += ' (' + String(r.set).toUpperCase() + ') ' + r.collector_number;
    }
    if (r.foil) s += ' *F*';
    return s;
  }

  /* The deck page's copy idiom (`Deck.copy`): `navigator.clipboard` needs a
   * secure context, and the failure is reported rather than looking like a
   * button that does nothing. Resolves to whether it copied. */
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(function () { return true; },
                                                      function () { return false; });
    }
    return Promise.resolve(false);
  }

  function wireBuyList(slug, branch, buy) {
    var host = document.getElementById('buyList');
    if (!host) return;
    var body = host.querySelector('.buylist-body');
    var exactBox = document.getElementById('buyExact');
    function exact() { return !!(exactBox && exactBox.checked); }
    function say(btn, ok) {
      var s = host.querySelector('.buylist-say');
      if (!s) return;
      s.textContent = ok ? 'copied' : 'could not copy — select and copy';
      setTimeout(function () { s.textContent = ''; }, 2400);
    }

    if (window.Api && Api.ready && Api.has('branch/buy-list')) {
      body.innerHTML =
        '<button class="todo-cmd" data-act="buy-copy">' +
        '<code>Copy for Mana Pool</code><span class="todo-copy buylist-say"></span></button>';
      host.addEventListener('click', function (ev) {
        var btn = ev.target.closest && ev.target.closest('[data-act="buy-copy"]');
        if (!btn) return;
        btn.disabled = true;
        Api.call('branch/buy-list', { slug: slug, branch: branch, exact: exact() })
          .then(function (doc) {
            return copyText(doc.text || '').then(function (ok) {
              say(btn, ok);
              // `noopener`: the shop's tab must not get a handle on this page.
              window.open(MANAPOOL_URL, '_blank', 'noopener');
            });
          })
          .catch(function (e) {
            btn.insertAdjacentHTML('afterend',
              '<p class="ev todo-blocked">' + esc(e.message) + '</p>');
          })
          .then(function () { btn.disabled = false; });
      });
      return;
    }

    // THE STATIC SITE: no server, so the printings come from the branch's own
    // cards.json and the list is shown rather than only copied.
    var base = DATA + 'decks/' + encodeURIComponent(slug) + '/branches/' +
               encodeURIComponent(branch) + '/';
    getJSON(base + 'cards.json').then(function (doc) {
      var byName = {};
      ((doc && doc.cards) || []).forEach(function (c) {
        if (!c || !c.name) return;
        byName[c.name] = c;
        String(c.name).split(' // ').forEach(function (f) {
          if (!byName[f]) byName[f] = c;
        });
      });
      var rows = buy.map(function (r) {
        var c = byName[r.name] || {};
        return { name: r.name, quantity: c.quantity || 1, set: c.set,
                 collector_number: c.collector_number, foil: !!c.foil };
      });
      function text() {
        return rows.map(function (r) { return buyLine(r, exact()); }).join('\n');
      }
      function paint() {
        body.innerHTML =
          '<button class="todo-cmd" data-act="buy-copy"><code>Copy</code>' +
          '<span class="todo-copy buylist-say"></span></button>' +
          '<pre class="cmd buylist-text">' + esc(text()) + '</pre>' +
          (doc ? '' : '<p class="ev">No cards.json on this branch, so no printings: ' +
                      'the plain form only.</p>');
      }
      paint();
      if (exactBox) exactBox.addEventListener('change', paint);
      host.addEventListener('click', function (ev) {
        var btn = ev.target.closest && ev.target.closest('[data-act="buy-copy"]');
        if (!btn) return;
        copyText(text()).then(function (ok) { say(btn, ok); });
      });
    });
  }

  /* Names through `card()`, which is `Shell.cardLink`. This list used to build
   * its own `cardref` + `card-pop` markup — a second hover mechanism living in
   * one function on one panel, while every other card name on the page was
   * inert text. One idiom, one home. */
  function cardList(title, rows) {
    if (!rows.length) return '';
    return '<h3>' + esc(title) + ' <span class="ev">(' + rows.length + ')</span></h3>' +
      '<ul class="cardlist">' + rows.map(function (r) {
        return '<li>' + card(r.name) +
          (r.where && r.where.length
            ? ' <span class="ev">in ' + esc(r.where.map(function (w) {
                /* One shape, two kinds: a box names itself, a deck names its
                 * slug and whether it is sleeved or already in a pile. */
                return w.kind === 'box' ? w.name
                  : w.slug + (w.locked ? ' (locked)' : '') +
                    (w.apart ? ' (in a pile)' : '');
              }).join(', ')) + '</span>' : '') + '</li>';
      }).join('') + '</ul>';
  }

  function trailPanel(meta) {
    if (!meta) return '';
    var commits = meta.commits || [];
    var out = '<section class="panel" id="trail"><h2>The trail</h2>';
    out += '<p class="ev">Opened ' + esc(meta.opened) + ' from V' +
           esc(meta.base_version) + '.</p>';
    if (!commits.length) {
      out += '<p class="ev">No commits yet. A commit freezes one exact list with ' +
             'a message — and is allowed while cards are still unsourced, ' +
             'because the gap between deciding and merging is cardboard.</p>';
    }
    commits.forEach(function (c, i) {
      out += '<div class="commit"><div class="ev">#' + (i + 1) + ' &nbsp; ' +
        esc(c.at) + ' &nbsp; <code>' + esc(c.decklist_sha256.slice(0, 12)) +
        '</code></div><p>' + esc(c.message) + '</p></div>';
    });
    if (meta.merged) {
      out += '<p class="merged">MERGED ' + esc(meta.merged.at) +
             ' into the list after V' + esc(meta.merged.into_version_before) +
             '.</p>';
    }
    return out + '</section>';
  }

  /* BASE MOVED. A bench deck is edited in place between saves, so the deck can
   * move under a branch without its version changing, and the branch's
   * comparison is then against a list the deck no longer runs. Whether it moved
   * is `deck_branch.base_moved`'s sentence (derived when `deck-info` writes the
   * branch row into info.json); this page shows it, one line, and decides
   * nothing. Absent key, absent line: a branch cut before 2026-10-10 recorded
   * no base list. */
  function wireBaseMoved(slug, name) {
    getJSON(DATA + 'decks/' + encodeURIComponent(slug) + '/info.json').then(function (info) {
      var rows = ((info && info.branches) || []);
      var row = rows.filter(function (b) { return b && b.name === name; })[0];
      var mount = document.getElementById('objective');
      if (!mount) return;
      // THE STATE, FOR EVERY BRANCH (2026-10-19). `proposalPanel` renders only a
      // proposal, so an open branch never said it was open — the word comes
      // from `deck_branch.branch_state` on info.json, and this prints it.
      if (row && row.state && !document.getElementById('branchState')) {
        var st = String(row.state);
        var key = /^MERGED/.test(st) ? 'merged' : /^PROPOSED/.test(st) ? 'proposed'
                : /^REJECT/.test(st) ? 'rejected' : /^WITHDRAW/.test(st) ? 'withdrawn' : 'open';
        var hint = key === 'open'
          ? ' \u2014 an experiment, not a decision. <code>manamap pilot deck-branch ' + esc(slug)
            + ' propose ' + esc(name) + ' --as vX.Y.Z</code> makes it one.'
          : (row.state_why ? ' \u2014 ' + esc(row.state_why) : '');
        mount.insertAdjacentHTML('afterbegin',
          '<p class="ev" id="branchState"><span class="br-state" data-state="' + key + '">' + esc(st) + '</span>' + hint + '</p>');
      }
      if (row && row.base_moved && !document.getElementById('baseMoved')) {
        mount.insertAdjacentHTML('afterbegin',
          '<p class="ev warn" id="baseMoved">' + esc(row.base_moved) + '</p>');
      }
      var panels = document.getElementById('slotEvidence') || document.getElementById('panels');
      if (!panels) return;
      /* AT THE TABLE (2026-10-21). The pilot's own per-game counts for this deck
       * — when the commanders landed, when the first wheel fired and whether a
       * commander was there to be paid, what the kept seven held — read off
       * `info.record.marks`. They are the deck's games, not this branch's (a
       * branch is not sleeved), and the page says so; they sit ABOVE the goldfish
       * table because they are the instrument the pilot trusts. Absent, absent. */
      var rec = (info && info.record) || {};
      var mk = rec.marks || {};
      if (rec.marks_n && !document.getElementById('table-marks')) {
        function turn(key, label) {
          var c = mk[key];
          if (!c) return '';
          return '<li>' + esc(label) + ': ' + (c.mean == null ? 'never' : 'turn ' + esc(String(c.mean)) + ' on average')
            + (c.never ? ', never in ' + esc(String(c.never)) : '') + ' <span class="ev">(n ' + esc(String(c.n)) + ')</span></li>';
        }
        var items = turn('commander_turn', 'first commander resolved')
          + turn('both_turn', 'both commanders out')
          + turn('first_wheel_turn', 'first wheel resolved')
          + (mk.wheel_with_commander_rate != null
             ? '<li>first wheel landed with a commander out: ' + esc(String(Math.round(mk.wheel_with_commander_rate * 100))) + '%</li>' : '')
          + (mk.kept_hand_passes_rule_rate != null
             ? '<li>kept seven passed the keep rule: ' + esc(String(Math.round(mk.kept_hand_passes_rule_rate * 100))) + '%</li>' : '')
          + (mk.commander_removed
             ? '<li>a commander removed ' + esc(String(mk.commander_removed.mean)) + ' times a game</li>' : '');
        panels.insertAdjacentHTML('afterbegin',
          '<section class="panel" id="table-marks"><h2>At the table</h2>'
          + '<p class="ev">The pilot’s own counts over ' + esc(String(rec.marks_n)) + ' marked game'
          + (rec.marks_n === 1 ? '' : 's') + ' on the deck as sleeved — not this branch’s list. '
          + 'Authored after each game; nothing here is simulated.</p><ul>' + items + '</ul></section>');
      }
      // PROVEN ON THIS BRANCH: the stacks whose scenario names a card only the
      // branch brings (`deck_branch.review`), with the checker's verdict.
      if (row && row.stacks && row.stacks.length && !document.getElementById('branchStacks')) {
        panels.insertAdjacentHTML('beforeend',
          '<section class="panel stacks" id="branchStacks"><h2>Proven on this branch</h2>'
          + '<p class="lede">Stack scenarios that name a card only this branch brings \u2014 a line the current deck cannot run until the branch merges.</p><ul>'
          + row.stacks.map(function (h) {
              return '<li><a href="deck.html?deck=' + encodeURIComponent(slug) + '#panel-stacks"><code>' + esc(h.id) + '</code></a> '
                   + esc(h.title || '') + ' <span class="br-state" data-state="' + (h.checker === 'pass' ? 'merged' : 'rejected') + '">'
                   + esc(h.checker || 'unchecked') + '</span> <span class="ev">' + esc((h.names || []).join(', ')) + '</span></li>';
            }).join('') + '</ul></section>');
      }
      // THE SIBLINGS: the deck's other branches, newest first, with their state.
      var others = rows.filter(function (b) { return b && b.name && b.name !== name && !b.unreadable; });
      if (others.length && !document.getElementById('branchSiblings')) {
        (document.getElementById('slotActions') || panels).insertAdjacentHTML('beforeend',
          '<section class="panel siblings" id="branchSiblings"><h2>The deck\u2019s other branches</h2><ul class="br-siblings">'
          + others.map(function (b) {
              var st = String(b.state || ''), v = b.verdict || {};
              var key = /^MERGED/.test(st) ? 'merged' : /^PROPOSED/.test(st) ? 'proposed'
                      : /^REJECT/.test(st) ? 'rejected' : /^WITHDRAW/.test(st) ? 'withdrawn' : 'open';
              return '<li><a href="branch.html?deck=' + encodeURIComponent(slug) + '&branch=' + encodeURIComponent(b.name) + '">'
                   + esc(b.name) + '</a> <span class="br-state" data-state="' + key + '">' + esc(st || 'OPEN') + '</span>'
                   + (v.state ? ' <span class="ev">' + esc(v.state) + '</span>' : '')
                   + ' ' + askBranchHtml(slug, b.name) + '</li>';
            }).join('')
          + '</ul><p class="ev"><a href="deck.html?deck=' + encodeURIComponent(slug) + '#panel-branches">\u2190 Branches, on the deck page</a></p></section>');
      }
    });
  }

  // WHAT THE GOLDFISH CANNOT SEE: net-change's own `blind_spots`, rendered
  // beside the table so a "worse" row is read with the cards it is blind to.
  function blindSpotsPanel(nc) {
    if (!nc) return '';
    var bs = nc.blind_spots || [];
    var limits = nc.limits || [];
    if (!bs.length && !limits.length) return '';
    var why = ((nc.recommendation || {}).risk || []).filter(function (r) {
      return r.kind === 'unmeasured' && r.why_it_matters;
    })[0];
    return '<section class="panel blind" id="blindSpots"><h2>What this cannot see</h2>' +
      '<div class="nc-duo"><div>' + (bs.length ? '<p class="ev">Cards in the change whose effect no row can price. The figure is a floor for them, not a zero' +
        (why ? ': ' + esc(why.why_it_matters) : '') + '.</p><ul class="nc-blind">' +
        bs.map(function (b) {
          return '<li><b>' + (b.cards || []).map(card).join(', ') + '</b>' +
            '<span class="nc-blind-why">' + esc(b.why || b.detail || '') + '</span></li>';
        }).join('') + '</ul>' : '') + '</div><div>' +
      otherRisks(nc) +
      (limits.length ? '<h3>What no goldfish can tell you</h3><ul class="nc-limits">' +
        limits.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('') + '</ul>' : '') +
      '</div></div></section>';
  }

  // WHAT FORGE'S AI DID WITH EACH ADD: `cast_proofs.json`, one row per card,
  // the CLI's own verdict line. A HELD add is a card the AI will not play —
  // never a reason to cut, and a hole in any Forge arm.
  function wireCastProofs(base) {
    getJSON(base + 'cast_proofs.json').then(function (cp) {
      var cards = cp && cp.cards;
      var panels = document.getElementById('slotEvidence') || document.getElementById('panels');
      if (!cards || !panels || document.getElementById('castProofs')) return;
      var names = Object.keys(cards).sort();
      panels.insertAdjacentHTML('beforeend',
        '<section class="panel cast" id="castProofs"><h2>What Forge\u2019s AI did with each add</h2>'
        + '<p class="lede">Two-seat shells, ' + esc(cp.harness || '') + (cp.as_of ? ', ' + esc(cp.as_of) : '')
        + '. PLAYED means cast or activated when drawn; HELD means castable and never cast \u2014 the AI will not play it, a hint or a patch before any Forge arm.</p>'
        + '<ul class="br-cast-list">' + names.map(function (n) {
            var r = cards[n] || {}, v = String(r.verdict || '');
            var key = /^PLAYED/.test(v) ? 'played' : /^HELD/.test(v) ? 'held' : 'unplayed';
            return '<li><span class="br-cast" data-state="' + key + '">' + esc(key.toUpperCase()) + '</span> <b>' + esc(n) + '</b> '
                 + '<span class="ev">drawn ' + esc(r.drawn_games != null ? r.drawn_games : '?') + 'g \u00b7 cast ' + esc(r.cast != null ? r.cast : '?')
                 + ' \u00b7 castable-uncast ' + esc(r.castable_uncast_turns != null ? r.castable_uncast_turns : '?') + 't</span></li>';
          }).join('') + '</ul></section>');
    });
  }

  // THE PROFILE'S OWN FIGURES: the goldfish series the net-change rows do not
  // carry (cumulative drain, bodies, extra cards), champion beside branch, each
  // with its model version. Printed, never differenced: the paired interval is
  // net-change's job and lives in the Measured table where it exists.
  function wireProfileFigures(slug, name, base) {
    Promise.all([getJSON(DATA + 'decks/' + encodeURIComponent(slug) + '/goldfish_metrics.json'),
                 getJSON(base + 'goldfish_metrics.json')]).then(function (got) {
      var champ = got[0], br = got[1];
      var panels = document.getElementById('slotEvidence') || document.getElementById('panels');
      if (!champ || !br || !panels || document.getElementById('profileFigures')) return;
      var cm = champ.metrics || champ, bm = br.metrics || br;
      function at(m, path, t) {
        var o = m; for (var i = 0; i < path.length; i++) { o = o && o[path[i]]; }
        return o && o[String(t)] != null ? num(o[String(t)], 2) : '\u2014';
      }
      var rows = [
        ['cumulative drain by T8', ['drain', 'mean_cumulative_drain_by_turn'], 8],
        ['cumulative drain by T10', ['drain', 'mean_cumulative_drain_by_turn'], 10],
        ['extra cards by T10', ['mean_extra_cards_drawn_by_turn'], 10],
        ['bodies by T6', ['mean_bodies_by_turn'], 6],
        ['bodies by T10', ['mean_bodies_by_turn'], 10],
        ['sacrifices by T10', ['mean_sacrifices_by_turn'], 10]
      ].filter(function (r) { return at(cm, r[1], r[2]) !== '\u2014' || at(bm, r[1], r[2]) !== '\u2014'; });
      if (!rows.length) return;
      var cv = (champ.meta || {}).model_version || '', bv = (br.meta || {}).model_version || '';
      panels.insertAdjacentHTML('beforeend',
        '<section class="panel profile" id="profileFigures"><h2>The deck\u2019s own goldfish, beside the branch\u2019s</h2>'
        + '<p class="lede">Series the net-change table does not carry, read from each list\u2019s goldfish artifact. Two readings side by side, not a difference: the paired interval is the Measured table\u2019s job.</p>'
        + '<table class="netchange"><thead><tr><th>measure</th><th>champion</th><th>branch</th></tr></thead><tbody>'
        + rows.map(function (r) { return '<tr><td>' + esc(r[0]) + '</td><td>' + at(cm, r[1], r[2]) + '</td><td>' + at(bm, r[1], r[2]) + '</td></tr>'; }).join('')
        + '</tbody></table><p class="ev">model ' + esc(String(cv).slice(0, 12)) + (bv && bv !== cv ? ' vs ' + esc(String(bv).slice(0, 12)) + ' \u2014 two models, read with care' : '') + '</p></section>');
    });
  }

  /* THE RULE'S WORD, UNDER THE ROWS IT SUMMARISES. `recommend` derives it from the
   * same document the table renders, so the two cannot disagree; it no longer sits
   * on top because the bench judges the profile, not the word. The rows it names
   * are already above, so it does not list them again. */
  /* THE STATE BANNER. A branch had two observable conditions before `propose`
   * existed — the directory is there, or `merged` is — so one the pilot had
   * DECIDED ON rendered identically to a half-finished experiment. This is the
   * decision; `verdictPanel` below it is the report the decision rests on, and
   * the two stay separate on purpose: one is what the bench measured, the other
   * is what a human did about it. */
  var STATE_CLASS = {
    'PROPOSED \u00b7 READY': 'met',
    'PROPOSED \u00b7 BLOCKED': 'unresolved',
    'PROPOSED \u00b7 STALE': 'failed',
    'PROPOSED \u00b7 OUTRUN': 'failed',
    'MERGED': 'met',
    'OPEN': 'unknown'
  };

  function proposalPanel(meta, slug, name) {
    var p = meta && meta.proposal;
    if (!p) return '';
    /* The live STATE is derived server-side by `deck_branch.branch_state` and
     * travels on `info.json`; this page has only `branch.json`, so it renders
     * what the proposal FROZE and leaves blocked-ness to the deck page. Whether
     * a card is in a box is a fact about a collection this page cannot read. */
    var a = p.accepted_on || {};
    var out = '<section class="panel proposal ' +
      (STATE_CLASS[meta.state] || 'unresolved') + '">' +
      '<h2>Proposed as ' + esc(p.as_version) + '</h2>' +
      '<p class="lede">Accepted ' + esc(p.at) +
      (p.why ? ' \u2014 ' + esc(p.why) : '') + '</p>';
    if (a.state) {
      out += '<p class="ev">On the net change: <b>' + esc(a.state) + '</b>' +
        (a.objective ? ', objective <code>' + esc(a.objective.axis) + ' ' +
          esc(a.objective.op) + ' ' + esc(a.objective.value) + '</code> ' +
          esc(String(a.grade || '').toUpperCase()) +
          (a.reading != null ? ' at ' + num(a.reading) : '') : '') + '.</p>';
    }
    if (p.forced_reason) {
      out += '<p class="ev warn">Accepted over a DO NOT MERGE \u2014 ' +
             esc(p.forced_reason) + '</p>';
    }
    if (p.proxy && p.proxy.length) {
      out += '<p class="ev">Proxying ' + p.proxy.length +
             ' across your own decks: ' + esc(p.proxy.join(', ')) + '.</p>';
    }
    if (p.procurement && p.procurement.note) {
      out += '<p class="ev">Procurement: ' + esc(p.procurement.note) + '</p>';
    }
    out += '<p class="ev">Nothing is merged. What is left is cardboard \u2014 the ' +
      'deck\u2019s dossier carries the pull list, because what you still need ' +
      'depends on your boxes and this page cannot read them. ' +
      '<code>manamap pilot deck-branch ' + esc(slug) + ' merge ' + esc(name) +
      ' --write</code></p>';
    return out + '</section>';
  }

  function verdictPanel(nc) {
    var r = nc && nc.recommendation;
    if (!r) return '';
    var cls = { 'merge': 'met', 'a trade': 'unresolved',
                'do not merge': 'failed', 'inconclusive': 'unknown',
                'no objective': 'unknown' }[r.state] || 'unknown';
    var because = String(r.because || '');
    if (r.forge_warning) because = because.replace(r.forge_warning, '').trim();
    var out = '<div class="verdict ' + cls + '" id="verdict"><p class="verdict-why">' +
      (r.state === 'no objective' ? '' : 'The rule\u2019s word is <b>' + esc(r.state) + '</b>. ') +
      esc(because) + '</p>';
    (r.notes || []).forEach(function (n) {
      out += '<p class="ev">' + esc(n) + '</p>';
    });
    return out + '</div>';
  }

  /* THE STAGING AREA. Every row carries what the card COSTS as well as what it
   * gains, badged apart — the pre-repair index rendered both the same red, so a
   * card that charged you something looked identical to one that gave you
   * something. Nothing here is measured; the panel says so. */
  function swapsPanel(doc, slug, branch) {
    if (!doc) return '';
    var out = '<section class="panel swaps"><h2>Swaps to consider</h2>';
    if (!doc.swaps.length) {
      out += '<p class="ev">Nothing in this list has a candidate above the ' +
             'strength floor. That is a real answer.</p>';
      return out + '</section>';
    }
    out += '<p class="ev">Ranked by the obsolescence index\u2019s own strength. ' +
           'It PROPOSES; nothing here has been measured.</p>' +
           '<details class="nc-fold"' + (location.hash === '#swaps' ? ' open' : '') + '><summary>' +
           doc.swaps.length + ' candidate' + (doc.swaps.length === 1 ? '' : 's') +
           '</summary><ul class="swaplist">';
    doc.swaps.forEach(function (r) {
      var band = r.strength >= 0.65 ? 'strong'
               : r.strength >= 0.4 ? 'mild' : 'weak';
      out += '<li class="swap"><div class="swap-head">' +
        '<span class="strength ' + band + '" title="0 = two different cards ' +
        'that sort near each other. 1 = strictly better, cheaper, no strings. ' +
        'The shipped data tops out at 0.95.">' + esc(r.strength.toFixed(2)) +
        '</span> <span class="out">' + card(r.out) + '</span>' +
        ' <span class="arrow">&rarr;</span> <span class="in">' + card(r['in']) +
        '</span> <span class="src">' + esc(r.source || 'buy') + '</span></div>';
      (r.gains || []).forEach(function (g) {
        out += '<span class="badge gain">+ ' + esc(g) + '</span>';
      });
      (r.costs || []).forEach(function (c) {
        out += '<span class="badge cost">\u2212 ' + esc(c) + '</span>';
      });
      (r.narrows || []).forEach(function (n) {
        out += '<span class="badge cost">narrower: ' + esc(n) + '</span>';
      });
      if (r.played_more === false) {
        out += '<span class="badge rank">played less</span>';
      }
      if (r.roles_disjoint) {
        out += '<p class="ev warn">These two share no job (' +
          esc((r.roles_out || []).join('/')) + ' \u2192 ' +
          esc((r.roles_in || []).join('/')) + '). The SEARCH failed, not the ' +
          'comparison \u2014 read both cards.</p>';
      }
      if (r.newly_combat_gated) {
        out += '<p class="ev warn">It has to connect, and the card it replaces ' +
          'does not. Efficient in a vacuum, wrong axis for a deck that does ' +
          'not attack.</p>';
      }
      if (branch) {
        out += '<div class="swap-act">' +
          '<button type="button" data-act="stage" data-out="' + esc(r.out) +
          '" data-in="' + esc(r['in']) + '" data-strength="' + esc(r.strength) +
          '">accept</button></div>';
      }
      out += '</li>';
    });
    out += '</ul>';
    (doc.notes || []).forEach(function (n) {
      out += '<p class="ev">' + esc(n) + '</p>';
    });
    return out + '</section>';
  }

  /* THE MERGE-REQUEST DIFF, and it is the first thing on the page after the
   * verdict — the deck equivalent of "Files changed". The panel it replaced
   * read `branch.json`'s `staged` list, so a branch opened with
   * `new --from <list>` — which stages nothing and sets the whole 99 at once —
   * rendered as "The change (0)" while the report beneath it measured all 35
   * cards. The cards coming IN showed up on the bill; the cards going OUT
   * appeared nowhere on the page. A diff that omits its deletions is not a diff.
   *
   * COUNTS ARE NAMES, THE DECK SIZE IS COPIES, and they are different numbers.
   * Cutting one of four Swamps removes no NAME, so "18 out, 21 in" sits happily
   * beside "100 -> 100 cards" and both are true. The header says both, because
   * a reader given only the first will read it as a deck that grew by three.
   *
   * Lands and spells stay apart for the reason the evidence does: a spell swap
   * moves the sampled rows, a land swap moves only the deterministic mana
   * block, and rendered together a land pass borrows a spell pass's credit. */
  var STATE_WORD = { buy: 'buy', box: 'in a box', elsewhere: 'in another deck',
                     in_deck: 'already here' };

  /* THE REASON PRINTS ONCE, against the card that was argued FOR. A staged swap
   * is one decision about two cards; printing its `why` on both sides shows the
   * reader the same sentence twice and still leaves them guessing which removal
   * paid for which addition. So the `-` row points at its opposite number and
   * the `+` row carries the argument. */
  function diffRows(rows, sign) {
    var adding = sign === '+';
    return '<ul class="difflist">' + rows.map(function (r) {
      var tag = r.state
        ? '<span class="chip ' + esc(r.state) + '">' +
          esc(STATE_WORD[r.state] || r.state) + '</span>' : '';
      var link = r.pair
        ? '<span class="paired">' + (adding ? '\u2190 ' : '\u2192 ') +
          card(r.pair) + '</span>' : '';
      return '<li class="' + (adding ? 'add' : 'del') + '">' +
             '<span class="sign">' + sign + '</span>' +
             '<span class="cname">' + card(r.name) + '</span>' +
             '<span class="mv">mv' + (r.cmc == null ? '?' : r.cmc) + '</span>' +
             tag + link +
             (adding && r.why && !isAutoWhy(r.why) ? '<span class="ev why">' + esc(r.why) + '</span>' : '') +
             '</li>';
    }).join('') + '</ul>';
  }

  /* `try --stage` WRITES ITS OWN SCREENING LINE AS THE REASON. It repeats on every
   * swap of a run and argues nothing, so it collapses to one line under the lists. */
  function isAutoWhy(why) { return /^try: /.test(String(why || '')); }

  function alpha(rows) {
    return rows.slice().sort(function (a, b) { return String(a.name).localeCompare(String(b.name)); });
  }

  /* THE SAME ROWS AS PICTURES, because a name is not a card you can judge.
   *
   * `diffRows` answers "what moved and why" and answers it well; it cannot
   * answer "is this card any good", which is the question in front of a
   * thirty-five card change. The hover pop exists on every name here, but it
   * shows ONE card at 244px and costs a deliberate hover each — thirty-five
   * hovers to review a branch is not a review, it is a chore, and the fallback
   * was a link OUT of the workbench to Scryfall.
   *
   * Reuses `.lib-grid` / `.lib-open` / `.lib-name` from the library drawer
   * rather than inventing a second card tile: the box is already reserved at
   * 488/680 there (without it every tile measures ~0 and the grid reflows as
   * each image lands), the art already comes from `Shell.cardImageUrl` by NAME,
   * and `Shell.wireCardArt` already carries the DFC front-face retry.
   *
   * `loading="lazy"` and the hidden-by-default view together are what keep this
   * honest: the gallery costs nothing until it is opened, so a page whose job
   * is a table of numbers does not fetch seventy images to show none of them. */
  function galleryRows(rows, sign) {
    if (!rows.length) return '';
    var adding = sign === '+';
    var art = window.Shell && Shell.cardImageUrl;
    return '<div class="lib-grid diffgrid">' + rows.map(function (r) {
      var href = (window.Shell && Shell.cardHref)
        ? ' href="' + esc(Shell.cardHref(r.name)) + '" target="_blank" rel="noopener"'
        : '';
      /* `data-src`, AND NO `loading="lazy"`. Both halves of that are measured.
       *
       * The drawer's tiles use `loading="lazy"` and it works there because they
       * are parsed into a container that is about to be shown. These are parsed
       * into one that is `display:none` — the list view is the default — and an
       * image that is never laid out never intersects anything, so Chrome marks
       * it "not needed" and REVEALING THE CONTAINER DOES NOT RE-ARM IT. Measured
       * on this page: 36 tiles in the viewport with `src` set, 0 requested, held
       * for 4s; removing the `loading` attribute and re-assigning the identical
       * URL loaded it at 146x204 immediately.
       *
       * So the deferral is explicit instead. `Shell.queueArt` promotes `data-src` the
       * first time the gallery is opened, which is the property that actually
       * mattered — a page whose job is a table of numbers fetches no card art
       * until someone asks to see cards. What is lost is progressive loading
       * WITHIN an open gallery, which is a real cost and a smaller one than a
       * grid of empty boxes. */
      var img = art
        ? '<img data-src="' + esc(Shell.cardImageUrl(r.name, undefined, branchRows[r.name])) + '" alt="' +
          esc(r.name) + '">'
        : '';
      var tag = r.state
        ? '<span class="chip ' + esc(r.state) + '">' +
          esc(STATE_WORD[r.state] || r.state) + '</span>' : '';
      return '<div class="lib-tile difftile ' + (adding ? 'add' : 'del') + '">' +
             '<a class="lib-open"' + href + ' title="' + esc(r.name) + '">' + img +
             '<span class="lib-name">' + esc(r.name) + '</span></a>' +
             '<span class="difftile-sign">' + sign + '</span>' + tag +
             (adding && r.why && !isAutoWhy(r.why) ? '<span class="ev why">' + esc(r.why) + '</span>' : '') +
             '</div>';
    }).join('') + '</div>';
  }

  /* "Swamp 4 -> 2", in the removals column. The card is still in the list, so
   * it gets no +/- sign of its own — what changed is how many. */
  function countRows(rows) {
    if (!rows.length) return '';
    return '<ul class="difflist">' + rows.map(function (r) {
      var down = r.delta < 0;
      return '<li class="' + (down ? 'del' : 'add') + ' countrow">' +
             '<span class="sign">' + (down ? '\u2212' : '+') + '</span>' +
             '<span class="cname">' + card(r.name) + '</span>' +
             '<span class="mv">' + r.from + ' \u2192 ' + r.to + '</span>' +
             '<span class="chip count">' + (down ? '' : '+') + r.delta +
             ' cop' + (Math.abs(r.delta) === 1 ? 'y' : 'ies') + '</span></li>';
    }).join('') + '</ul>';
  }

  function netChangePanel(meta, nc) {
    var d = nc && nc.changes && nc.changes.diff;
    if (!d) return stagedFallback(meta, nc);
    var c = d.counts;
    /* THE HEADLINE COUNTS COPIES, because that is what you sleeve and it is the
     * only pair of numbers that balances. It used to count NAMES and read
     * "18 out, 21 in" for a change that was 21 for 21 — cutting one of two
     * Plains and two of four Swamps removes three cards and no names, so three
     * removals were missing from the page. The name counts stay underneath,
     * where the gap between the two is an explanation rather than a puzzle. */
    var co = c.out_copies == null ? c.out : c.out_copies;
    var ci = c.in_copies == null ? c['in'] : c.in_copies;
    /* LIST OR GALLERY, and the list stays the default. The numbers are what
     * this page is for; the pictures are what a card decision is made on. Both
     * are rendered and CSS picks — the gallery's images are `loading="lazy"`,
     * so the one you are not looking at costs nothing. */
    var out = '<section class="panel netchange" id="net-change" data-view="list">' +
      '<h2>The cards' +
      '<span class="viewtoggle">' +
      '<button type="button" class="vt is-on" data-view="list">list</button>' +
      '<button type="button" class="vt" data-view="gallery">gallery</button>' +
      '</span></h2>' +
      '<p class="headline"><span class="del">\u2212 ' + co + ' out</span>' +
      '<span class="add">+ ' + ci + ' in</span>' +
      '<span class="ev">' + d.base_size + ' \u2192 ' + d.size + ' cards' +
      (d.base_size === d.size ? '' : '  \u26a0 the deck changed size') +
      '</span></p>' +
      '<p class="ev">' + co + ' cards out and ' + ci + ' in, across ' +
      c.out + ' name(s) removed and ' + c['in'] + ' added: ' + d.base_names +
      ' \u2192 ' + d.names + ' distinct cards, ' + d.base_size + ' \u2192 ' +
      d.size + ' copies. A basic cut from four to two changes no name.</p>';

    [['Non-lands', 'spell', c.spells_out, c.spells_in],
     ['Lands', 'land', c.lands_out, c.lands_in]].forEach(function (g) {
      var outs = alpha(d.out.filter(function (r) { return r.kind === g[1]; }));
      var ins = alpha(d['in'].filter(function (r) { return r.kind === g[1]; }));
      if (!outs.length && !ins.length) return;
      /* A COPY COUNT THAT MOVED IS A REMOVAL TOO. Rendered in the column it
       * belongs to rather than a third list, so the two sides still add up. */
      var chg = (d.changed || []).filter(function (r) { return r.kind === g[1]; });
      var lost = chg.filter(function (r) { return r.delta < 0; });
      var gained = chg.filter(function (r) { return r.delta > 0; });
      if (!outs.length && !ins.length && !chg.length) return;
      /* COPIES, THE SAME UNIT AS THE HEADLINE. Counting rows here gave the
       * lands "(-3 / +4)" while the panel above said 21 for 21 — two units in
       * one panel, which is the defect `net_change.METRICS` exists to stop.
       * A name gone takes all its copies with it; a name that merely shrank
       * takes the difference. */
      var copiesOut = outs.reduce(function (a, r) { return a + (r.copies || 1); }, 0) +
        lost.reduce(function (a, r) { return a - r.delta; }, 0);
      var copiesIn = ins.reduce(function (a, r) { return a + (r.copies || 1); }, 0) +
        gained.reduce(function (a, r) { return a + r.delta; }, 0);
      out += '<h3>' + esc(g[0]) + ' <span class="ev">(\u2212' + copiesOut +
             ' / +' + copiesIn + ')</span></h3><div class="diffcols">' +
             '<div>' + (outs.length || lost.length
                        ? diffRows(outs, '\u2212') + galleryRows(outs, '\u2212') +
                          countRows(lost)
                        : '<p class="ev">nothing removed</p>') + '</div>' +
             '<div>' + (ins.length || gained.length
                        ? diffRows(ins, '+') + galleryRows(ins, '+') + countRows(gained)
                        : '<p class="ev">nothing added</p>') + '</div>' +
             '</div>';
    });

    /* HOW MUCH OF THIS NOBODY ARGUED FOR, CARD BY CARD. A `why` is written when
     * a swap is STAGED, before any figure below exists, so it cannot have been
     * fitted to them. A branch opened from a whole list has none, and that is
     * worth saying out loud rather than rendering as blank space. */
    var autoRows = d['in'].filter(function (r) { return isAutoWhy(r.why); });
    if (autoRows.length) {
      out += '<details class="nc-auto"><summary>' + autoRows.length +
        ' add' + (autoRows.length === 1 ? '' : 's') + ' carry only the automatic note from <code>try</code></summary><ul>' +
        alpha(autoRows).map(function (r) {
          return '<li><b>' + esc(r.name) + '</b> <span class="ev">' + esc(r.why) + '</span></li>';
        }).join('') + '</ul></details>';
    }
    var u = d.unexplained || {};
    if (u.out || u['in']) {
      out += '<p class="ev unexplained">' + (u.out + u['in']) +
        ' of ' + (c.out + c['in']) + ' cards carry no recorded reason \u2014 ' +
        'they arrived with the branch\u2019s opening list rather than through ' +
        '<code>deck-branch stage</code>, which is what records a why. The ' +
        'branch\u2019s own rationale is at the top of this page.</p>';
    }
    return out + '</section>';
  }

  /* A branch measured before `card_diff` shipped has no `diff` block. Render
   * what it does have rather than an empty page. */
  function stagedFallback(meta, nc) {
    var ch = nc && nc.changes;
    var groups = ch
      ? [['Spells', ch.spells || []], ['Lands', ch.lands || []]]
      : [['Staged', (meta && meta.staged) || []]];
    var total = groups.reduce(function (a, g) { return a + g[1].length; }, 0);
    if (!total) return '';
    var out = '<section class="panel staged"><h2>The change (' + total +
      ')</h2><p class="ev">Staged swaps only \u2014 this branch was measured ' +
      'before the full diff was recorded.</p>';
    groups.forEach(function (g) {
      if (!g[1].length) return;
      out += '<h3>' + esc(g[0]) + ' <span class="ev">(' + g[1].length +
             ')</span></h3><ul class="swaplist">';
      g[1].forEach(function (r) {
        out += '<li><span class="pair"><span class="out">\u2212 ' + card(r.out) +
               '</span><span class="in">+ ' + card(r['in']) + '</span></span>' +
               (r.why ? '<span class="ev">' + esc(r.why) + '</span>' : '') +
               '</li>';
      });
      out += '</ul>';
    });
    return out + '</section>';
  }

  /* THE TWO VERBS THE PAGE WAS MISSING (2026-10-10): Measure and Unstage.
   * Both endpoints shipped with the branch workbench (`branch/net-change`,
   * `branch/stage {undo}`) and nothing here called them, so a branch started
   * from Build's tray had to go back to the terminal to be measured. Gated on
   * `Api.ready`: the deployed page has no server, and the panels above already
   * print the command for that case. */
  function benchPanel(meta, slug, branch) {
    if (!window.Api || !Api.ready) return '';
    var staged = (meta && meta.staged) || [];
    var rows = staged.map(function (r) {
      return '<li class="staged-row"><span class="pair"><span class="out">− ' + card(r.out) +
             '</span><span class="in">+ ' + card(r['in']) + '</span></span>' +
             '<button type="button" data-act="unstage" data-out="' + esc(r.out) +
             '" data-in="' + esc(r['in']) + '">unstage</button></li>';
    }).join('');
    return '<section class="panel bench-actions" id="benchActions"><h2>On the bench</h2>' +
      '<p class="ev">Measure runs the paired net change for this list as it stands ' +
      '(~15s, no model call) and writes <code>net_change.json</code>.</p>' +
      '<div class="swap-act"><button type="button" data-act="measure">Measure</button>' +
      ' <span class="ev" id="measureState"></span></div>' +
      (rows ? '<details class="nc-fold"' + (staged.length <= 8 ? ' open' : '') + '><summary>Staged swaps (' +
              staged.length + ')</summary><ul class="swaplist">' + rows + '</ul></details>'
            : '<p class="ev">Nothing staged.</p>') +
      '</section>';
  }

  function pollJob(id, done) {
    (function tick() {
      Api.call('job', { id: id }).then(function (row) {
        if (row.state === 'running') { setTimeout(tick, 1000); return; }
        done(row);
      }).catch(function (e) { done({ state: 'failed', error: e.message }); });
    })();
  }

  function measure(btn) {
    var q = new URLSearchParams(location.search);
    var say = document.getElementById('measureState');
    var started = Date.now();
    var tick = setInterval(function () {
      if (say) say.textContent = 'measuring… ' + Math.round((Date.now() - started) / 1000) + ' s';
    }, 1000);
    btn.disabled = true;
    if (say) say.textContent = 'measuring…';
    Api.call('branch/net-change', { slug: q.get('deck'), branch: q.get('branch') })
      .then(function (job) {
        pollJob(job.id, function (row) {
          clearInterval(tick);
          btn.disabled = false;
          if (row.state !== 'done') {
            if (say) say.textContent = 'failed: ' + (row.error || 'unknown');
            return;
          }
          // Re-render from the artifact the job just wrote — the page renders
          // committed files and nothing else, so it reads the new one.
          main();
        });
      })
      .catch(function (e) {
        clearInterval(tick);
        btn.disabled = false;
        if (say) say.textContent = e.message;
      });
  }

  function unstage(btn) {
    var q = new URLSearchParams(location.search);
    btn.disabled = true;
    btn.textContent = 'unstaging…';
    // `card` rather than `in`, which is a keyword — `serve._branch_stage`.
    Api.call('branch/stage', {
      slug: q.get('deck'), branch: q.get('branch'), undo: true,
      out: btn.getAttribute('data-out'), card: btn.getAttribute('data-in')
    }).then(function () {
      main();
    }).catch(function (e) {
      btn.disabled = false;
      btn.textContent = 'unstage';
      btn.insertAdjacentHTML('afterend', '<p class="ev todo-blocked">' + esc(e.message) + '</p>');
    });
  }

  /* ── boot ───────────────────────────────────────────────────────────── */

  function askBranchHtml(slug, name) {
    if (!window.PageState || !PageState.askButton) return '';
    var deck = window.DeckFacts ? DeckFacts.displayName({ slug: slug }).text : slug;
    return PageState.askButton({ text: PageState.askText.branch(name, deck), deck: slug,
                                 branch: name, label: 'Ask Jarvis', cls: 'br-jarvis' });
  }

  function main() {
    var q = new URLSearchParams(location.search);
    var slug = q.get('deck');
    var name = q.get('branch');
    var head = document.getElementById('branchName');
    if (!slug || !name) {
      head.textContent = 'Which branch?';
      document.getElementById('panels').innerHTML = absent(
        'No branch named',
        'This page needs both a deck and a branch.',
        'branch.html?deck=<slug>&branch=<name>');
      return;
    }
    document.title = slug + '@' + name + ' — Mana Map';
    head.textContent = slug + ' @ ' + name;
    var link = document.getElementById('deckLink');
    link.href = 'deck.html?deck=' + encodeURIComponent(slug);
    // SEND TO JARVIS on this branch (2026-10-28) — the page's own row. The deck's
    // name is the slug printed as one (DeckFacts.displayName), which is what the
    // pilot calls it; page-state.js owns the button and the prompt.
    if (window.PageState && PageState.askButton && !document.getElementById('branchAsk')) {
      link.parentNode.insertAdjacentHTML('beforeend', '<span id="branchAsk">' + askBranchHtml(slug, name) + '</span>');
    }

    var base = DATA + 'decks/' + encodeURIComponent(slug) + '/branches/' +
               encodeURIComponent(name) + '/';
    Promise.all([getJSON(base + 'branch.json'), getReport(base + 'net_change.json')])
      .then(function (got) {
        var meta = got[0], nc = got[1].doc, failed = got[1].failed;
        if (!meta) {
          document.getElementById('panels').innerHTML = absent(
            'No such branch',
            'Nothing is tracked at ' + base + '.',
            'manamap pilot deck-branch ' + slug + ' list');
          return;
        }
        // THE SUMMARY, NOT THE AUTOMATIC NOTE. `meta.why` on a branch opened by
        // `try --stage` is the screening line of its first swap.
        var sum = meta.summary || {};
        if (sum.title) document.getElementById('branchName').textContent = sum.title;
        var sub = document.getElementById('branchSub');
        if (sub) sub.textContent = sum.title ? slug + ' @ ' + name : '';
        document.getElementById('branchWhy').textContent =
          sum.blurb || (isAutoWhy(meta.why) ? '' : (meta.why || ''));
        document.getElementById('objective').innerHTML = proposalPanel(meta, slug, name);
        var unread = failed ? absent('The report could not be read',
          'A net_change.json exists but did not load (' + failed + '). This is not an unmeasured branch; re-running only replaces the file.',
          'manamap pilot net-change ' + slug + ' --branch ' + name + ' --write') : '';
        document.getElementById('panels').innerHTML = (nc
          ? (tablePanel(nc, meta) + blindSpotsPanel(nc) + manaPanel(nc) + billPanel(nc, meta) +
             forgePanel(nc) + netChangePanel(meta, nc) +
             '<div id="slotEvidence"></div>' + trailPanel(meta) +
             benchPanel(meta, slug, name) + '<div id="slotActions"></div>' + definitionsPanel(nc))
          : (unread || (objectivePanel(meta, nc) + absent(
              'The net change',
              'This branch has not been measured against the deck yet. Until it ' +
              'is, there is nothing here to decide on.',
              'manamap pilot net-change ' + slug + ' --branch ' + name + ' --write'))) +
            '<div id="slotEvidence"></div>' + trailPanel(meta) + benchPanel(meta, slug, name) +
            '<div id="slotActions"></div>');
        paintNav();
        wireViewToggle();
        wireBaseMoved(slug, name);
        wireCastProofs(base);
        wireProfileFigures(slug, name, base);
        // Late panels arrive by fetch; the nav follows them in.
        var slot = document.getElementById('panels');
        if (window.MutationObserver && slot && !slot._navWatch) {
          slot._navWatch = new MutationObserver(function () { paintNav(); });
          slot._navWatch.observe(slot, { childList: true, subtree: true });
        }
        if (nc && nc.bill) {
          var buy = (nc.bill.cards || []).filter(function (r) {
            return r.state === 'buy';
          });
          wireBuyList(slug, name, buy);
          wireBillPrice(slug, name, buy);
        }
        // THE DEEP LINK LANDS (2026-10-20): the workbench links `#measured`,
        // `#net-change` and `#swaps`; the browser's own jump fired before
        // these panels existed. Once here, and again when the swaps panel
        // fills in, since that one arrives last.
        if (window.scrollToHash) window.scrollToHash();
        loadSwaps(slug, name);
      });
  }

  /* ONE DELEGATED LISTENER, on the container that survives every re-render.
   * The toggle flips `data-view` on the section and CSS does the rest, so
   * switching costs no re-render and loses no scroll position.
   *
   * `Shell.wireCardArt` is what gives these tiles the DFC front-face retry —
   * without it a double-faced card shows a permanently empty box, because the
   * corpus keys "A // B" and Scryfall 404s on that form. */
  /* WHAT `loading="lazy"` WAS SUPPOSED TO DO, done by hand — and PACED, because
   * the first version that worked got throttled.
   *
   * Two measured failures are behind this shape, and both are why the queue that
   * answers them now lives in `shell.js` rather than here. `loading="lazy"` never
   * fires at all: these tiles are parsed into a `display:none` container (the
   * list view is the default), an image that is never laid out never intersects
   * anything, and revealing the container does not re-arm it — 36 tiles in the
   * viewport with `src` set, 0 requested, held for 4s. Then promoting all seventy
   * at once DID load them and Scryfall answered thirty-five: the rest errored and
   * `onImageError` stripped them to empty boxes. Seventy parallel requests at a
   * public API is exactly what the drawer's `lazy` was quietly preventing.
   *
   * THE QUEUE MOVED TO `shell.js` WHEN THE SECOND CALLER ARRIVED. Curate shows
   * 190 tiles, squarely in the throttled regime this measured, and a second copy
   * of a paced fetch is how one copy stops matching the API — the same argument
   * that kept `onImageError` shared. `Shell.queueArt` IS this function: promote
   * in document order, one every `ART_GAP` ms, the `data-src` consumed as it is
   * queued so re-opening the gallery adds nothing and a part-loaded grid is never
   * restarted. It gained one thing in the move — a detached <img> is dropped
   * without spending the gap — and the gap itself went 70ms to 110ms, inside
   * Scryfall's published 50-100ms guidance where 70 (~14/s) was not. Seventy
   * cards now land in about eight seconds instead of five.
   *
   * What stays HERE is the property that actually mattered: the deferral is the
   * CALL SITE, not the queue. A page of numbers fetches no art until the gallery
   * is asked for. */

  /* THE JUMP NAV, the dossier's own. Built from what actually rendered, and repainted
   * as the late panels land, so a link never points at a section that is not there. */
  var NAV = [['measured', 'Profile'], ['blindSpots', 'Cannot see'], ['mana', 'Mana'],
             ['bill', 'Cost'], ['forge', 'Forge'], ['net-change', 'Cards'],
             ['table-marks', 'At the table'], ['castProofs', 'Cast proofs'],
             ['branchStacks', 'Proven'], ['profileFigures', 'Goldfish'],
             ['trail', 'Trail'], ['benchActions', 'Bench'], ['swaps', 'Swaps'],
             ['branchSiblings', 'Other branches'], ['definitions', 'Definitions']];

  function paintNav() {
    var host = document.getElementById('branchNav');
    if (!host) return;
    var here = NAV.filter(function (n) { return document.getElementById(n[0]); });
    host.innerHTML = here.map(function (n) {
      return '<a href="#' + n[0] + '" data-ch="' + n[0] + '">' + esc(n[1]) + '</a>';
    }).join('');
    host.hidden = !here.length;
    if (!window.IntersectionObserver) return;
    if (host._io) host._io.disconnect();
    var io = host._io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        Array.prototype.forEach.call(host.querySelectorAll('a'), function (a) {
          a.classList.toggle('is-here', a.getAttribute('data-ch') === e.target.id);
        });
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    here.forEach(function (n) { io.observe(document.getElementById(n[0])); });
  }

  function wireViewToggle() {
    var panels = document.getElementById('panels');
    if (!panels) return;
    if (window.Shell && Shell.wireCardArt) Shell.wireCardArt(panels);
    if (panels.dataset.viewWired) return;
    panels.dataset.viewWired = '1';
    panels.addEventListener('click', function (ev) {
      var b = ev.target && ev.target.closest && ev.target.closest('button.vt');
      if (!b) return;
      var sec = b.closest('section.netchange');
      if (!sec) return;
      var view = b.getAttribute('data-view');
      sec.setAttribute('data-view', view);
      if (view === 'gallery' && window.Shell) Shell.queueArt(sec);
      Array.prototype.forEach.call(sec.querySelectorAll('button.vt'), function (o) {
        o.classList.toggle('is-on', o === b);
      });
    });
  }

  /* THE SERVER HALF, and it is strictly additive. Everything above renders from
   * committed artifacts and works on a deployed page with no server at all;
   * these are the verbs, and each one is gated on `Api.ready` with the CLI
   * command named where it is absent — this page shipped loading `api.js` and
   * never probing, so `Api.ready` was permanently false and the library
   * drawer's own button reported "needs a local server" while one was running. */
  function loadSwaps(slug, branch) {
    var host = document.createElement('div');
    host.id = 'swaps';
    (document.getElementById('slotActions') || document.getElementById('panels')).insertAdjacentElement('afterbegin', host);
    if (!window.Api || !Api.ready) {
      host.innerHTML = absent(
        'Swaps to consider',
        'What in this list has a cheaper card doing its job. This one needs a ' +
        'local server; the report above does not.',
        'manamap pilot upgrades ' + slug + ' --branch ' + branch);
      return;
    }
    Api.call('branch/upgrades', { slug: slug, branch: branch })
      .then(function (doc) {
        host.innerHTML = swapsPanel(doc, slug, branch);
        if (window.scrollToHash) window.scrollToHash();
      })
      .catch(function (e) {
        host.innerHTML = absent('Swaps to consider', e.message,
          'manamap pilot upgrades ' + slug + ' --branch ' + branch);
      });
  }

  /* Delegated, because `swapsPanel` replaces the whole subtree. */
  function wire() {
    document.addEventListener('click', function (ev) {
      // A COMMAND IS HANDED TO YOU, not run for you (`deck-view.js`'s `command`
      // idiom): the button copies its <code> and says so.
      var cmd = ev.target.closest && ev.target.closest('[data-act="copy-cmd"]');
      if (cmd) {
        var code = cmd.querySelector('code'), say = cmd.querySelector('.todo-copy');
        copyText(code ? code.textContent : '').then(function (ok) {
          if (!say) return;
          say.textContent = ok ? 'copied' : 'could not copy';
          setTimeout(function () { say.textContent = 'copy'; }, 2400);
        });
        return;
      }
      var verb = ev.target.closest && ev.target.closest('[data-act="measure"], [data-act="unstage"]');
      if (verb && !verb.disabled) {
        if (verb.getAttribute('data-act') === 'measure') measure(verb); else unstage(verb);
        return;
      }
      var btn = ev.target.closest && ev.target.closest('[data-act="stage"]');
      if (!btn) return;
      var q = new URLSearchParams(location.search);
      btn.disabled = true;
      btn.textContent = 'staging\u2026';
      Api.call('branch/stage', {
        slug: q.get('deck'), branch: q.get('branch'),
        out: btn.getAttribute('data-out'), card: btn.getAttribute('data-in'),
        strength: btn.getAttribute('data-strength')
      }).then(function () {
        // A full reload rather than a partial repaint: staging changes the
        // list, which changes every panel on this page.
        location.reload();
      }).catch(function (e) {
        btn.disabled = false;
        btn.textContent = 'accept';
        btn.insertAdjacentHTML('afterend',
          '<p class="ev todo-blocked">' + esc(e.message) + '</p>');
      });
    });
  }

  // Exposed for the browser suite. The absent-objective render is a CONTRACT —
  // a branch with none must say so rather than drop the panel — and every real
  // branch is now required to state one, so the case cannot be reached through
  // live data. Testing it needs the builder, not a deck.
  window.Branch = {
    __proposalPanel: proposalPanel,
    __objectivePanel: objectivePanel,
    __verdictPanel: verdictPanel,
    __swapsPanel: swapsPanel,
    __buyLine: buyLine,
    __priceBill: priceBill
  };

  /* WHAT THIS TAB HAS OPEN, for Jarvis (PRD v2 Step 7; `page-state.js`): the
   * deck and the branch, both read from the URL the page itself renders from. */
  if (window.PageState) PageState.register(function () {
    var q = new URLSearchParams(location.search);
    var s = { deck: q.get('deck') || undefined, branch: q.get('branch') || undefined };
    if (window.Session && Session.library) s.library = Session.library.names;
    return s;
  });

  if (window.Shell && Shell.mount) Shell.mount();
  wire();
  (function () {
    var q = new URLSearchParams(location.search);
    if (!q.get('deck') || !q.get('branch')) return;
    getJSON(DATA + 'decks/' + encodeURIComponent(q.get('deck')) + '/branches/' +
            encodeURIComponent(q.get('branch')) + '/cards.json').then(function (doc) {
      ((doc && doc.cards) || []).forEach(function (c) { if (c && c.name) branchRows[c.name] = c; });
      if (window.Shell && Shell.setDeckRows) Shell.setDeckRows(branchRows);
    }).catch(function () { /* no cards.json: hovers fall back to the default print */ });
  })();
  // Probe WITHOUT blocking the first paint — the artifacts render either way.
  if (window.Api && Api.probe) { Api.probe().then(main, main); } else { main(); }
})();
