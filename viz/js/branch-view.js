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
      return '<section class="panel objective none"><h2>Objective</h2>' +
        '<p class="ev">This branch states none. It predates the requirement, so ' +
        'it cannot be graded — it can only be described.</p>' +
        '<p class="ev">A branch without a testable objective gets judged on ' +
        'whether it did what it does, which is not the same question as ' +
        'whether it was worth doing.</p></section>';
    }
    var state = g ? g.state : 'not measured';
    var cls = { 'met': 'met', 'not met': 'failed',
                'not resolvable': 'unresolved' }[state] || 'unknown';
    var out = '<section class="panel objective ' + cls + '"><h2>Objective</h2>' +
      '<p class="obj-expr"><code>' + esc(o.axis) + ' ' + esc(o.op) + ' ' +
      esc(o.value) + '</code></p>';
    if (o.why) out += '<p class="ev">' + esc(o.why) + '</p>';
    if (g) {
      out += '<p class="obj-result"><span class="obj-reading">' +
             esc(num(g.reading)) + '</span> &rarr; <b>' +
             esc(state.toUpperCase()) + '</b></p>';
      if (g.why) out += '<p class="ev">' + esc(g.why) + '</p>';
    }
    return out + '</section>';
  }

  function tablePanel(nc) {
    if (!nc) return '';
    var rows = (nc.table || []).map(function (r) {
      return '<tr class="v-' + esc(r.verdict) + '"><td>' + esc(r.measure) +
        '</td><td class="n">' + num(r.champion) + '</td><td class="n">' +
        num(r.branch) + '</td><td class="n">' + signed(r.delta) +
        '</td><td class="verdict">' + esc(r.verdict) +
        (r.verdict === 'noise' ? ' <span class="ev">MDE ' + num(r.mde) + '</span>' : '') +
        '</td></tr>';
    }).join('');
    return '<section class="panel"><h2>Measured</h2>' +
      '<p class="ev">' + esc(nc.harness.iterations.toLocaleString()) +
      ' games each, shared seed. A delta smaller than the run could detect is ' +
      'marked noise rather than ranked.</p>' +
      '<div class="tablewrap"><table class="netchange"><thead><tr><th>Measure</th>' +
      '<th class="n">Deck</th><th class="n">Branch</th><th class="n">&Delta;</th>' +
      '<th>Verdict</th></tr></thead><tbody>' + rows + '</tbody></table></div></section>';
  }

  function liftPanel(nc) {
    if (!nc || !nc.engine_lift) return '';
    var out = '<section class="panel"><h2>Does the engine make it win?</h2>' +
      '<p class="ev">The kill rate in games where a list’s declared engine ' +
      'came online by turn three, minus the rate in games where it did not. ' +
      'The only test of whether a stated engine does anything.</p><div class="lifts">';
    ['champion', 'branch'].forEach(function (who) {
      var e = nc.engine_lift[who] || {};
      var label = who === 'champion' ? 'The deck' : 'The branch';
      if (!e.available) {
        out += '<div class="lift"><div class="who">' + esc(label) + '</div>' +
               '<p class="ev">' + esc(e.why || 'not measured') + '</p></div>';
        return;
      }
      out += '<div class="lift ' + (e.lift > 0 ? 'good' : 'bad') + '">' +
        '<div class="who">' + esc(label) + '</div>' +
        '<div class="big">' + signed(e.lift) + '</div>' +
        '<div class="ev">' + num(e.offline.kill_rate) + ' &rarr; ' +
        num(e.online.kill_rate) + ' &nbsp;·&nbsp; CI [' + signed(e.ci95[0]) +
        ', ' + signed(e.ci95[1]) + ']' +
        (e.excludes_zero ? '' : ' &nbsp;·&nbsp; spans zero') + '</div>' +
        '<p class="ev">' + esc(e.reading) + '</p></div>';
    });
    return out + '</div></section>';
  }

  function forgePanel(nc) {
    if (!nc || !nc.forge) return '';
    var f = nc.forge;
    if (!f.available) {
      return absent('The real table', f.why, null);
    }
    function routes(r) {
      var keys = Object.keys(r.won_by || {});
      if (!keys.length) return '';
      return '<div class="ev">' + keys.map(function (k) {
        return esc(k) + ' &times;' + r.won_by[k];
      }).join(' · ') + '</div>';
    }
    var under = f.mde != null && Math.abs(f.delta) < f.mde;
    return '<section class="panel"><h2>The real table</h2>' +
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
      '<p class="ev">' + esc(f.caveat || '') + '</p></section>';
  }

  /* THE BILL, and the four states are not four kinds of purchase. `elsewhere` is
   * a card sleeved in another deck — a trade-off, not money — and folding it into
   * `buy` reads as "spend on something already in the house". */
  var STATE_COPY = {
    in_deck: 'already in the deck',
    box: 'in a box — you own these',
    elsewhere: 'sleeved in another deck — a trade-off, not a purchase',
    buy: 'to buy'
  };

  function billPanel(nc, meta) {
    var bill = nc && nc.bill;
    if (!bill) return '';
    var c = bill.counts || {};
    var owned = (c.in_deck || 0) + (c.box || 0) + (c.elsewhere || 0);
    var total = Object.keys(c).reduce(function (a, k) { return a + c[k]; }, 0);
    var tiles = Object.keys(STATE_COPY).map(function (k) {
      return '<div class="tile s-' + k + '"><div class="k">' + (c[k] || 0) +
             '</div><div class="t">' + esc(STATE_COPY[k]) + '</div></div>';
    }).join('');
    var buy = (bill.cards || []).filter(function (r) { return r.state === 'buy'; });
    var elsewhere = (bill.cards || []).filter(function (r) { return r.state === 'elsewhere'; });
    return '<section class="panel"><h2>The bill</h2>' +
      '<div class="tiles">' + tiles + '</div>' +
      '<p class="lede">You already own <b>' + owned + ' of ' + total + '</b>.</p>' +
      '<p class="ev">Prices are stripped from the card corpus by design, so this ' +
      'page cannot give you a figure. It can tell you exactly what to price.</p>' +
      cardList('To buy', buy) + cardList('To unsleeve from other decks', elsewhere) +
      '</section>';
  }

  /* Art through `Shell.cardImageUrl`, which already carries the DFC front-face
   * retry — a NAME is the only card identity this page ever has, which is what
   * lets it draw real cards with no corpus loaded. The hover is CSS-only,
   * ported from the roster: no positioning code, no tooltip layer. */
  function cardList(title, rows) {
    if (!rows.length) return '';
    return '<h3>' + esc(title) + ' <span class="ev">(' + rows.length + ')</span></h3>' +
      '<ul class="cardlist">' + rows.map(function (r) {
        var img = window.Shell ? Shell.cardImageUrl(r.name, 'small') : '';
        return '<li><a class="cardref" href="index.html?cards=' +
          encodeURIComponent(r.name) + '">' + esc(r.name) +
          (img ? '<span class="card-pop"><img loading="lazy" width="488" ' +
                 'height="680" src="' + esc(img) + '" alt=""></span>' : '') +
          '</a>' + (r.where && r.where.length
            ? ' <span class="ev">in ' + esc(r.where.map(function (w) {
                return w.slug + (w.locked ? ' (locked)' : '');
              }).join(', ')) + '</span>' : '') + '</li>';
      }).join('') + '</ul>';
  }

  function trailPanel(meta) {
    if (!meta) return '';
    var commits = meta.commits || [];
    var out = '<section class="panel"><h2>The trail</h2>';
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

  function limitsPanel(nc) {
    if (!nc || !nc.limits || !nc.limits.length) return '';
    return '<section class="panel limits"><h2>What this cannot tell you</h2><ul>' +
      nc.limits.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('') +
      '</ul></section>';
  }

  /* ── boot ───────────────────────────────────────────────────────────── */

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

    var base = DATA + 'decks/' + encodeURIComponent(slug) + '/branches/' +
               encodeURIComponent(name) + '/';
    Promise.all([getJSON(base + 'branch.json'), getJSON(base + 'net_change.json')])
      .then(function (got) {
        var meta = got[0], nc = got[1];
        if (!meta) {
          document.getElementById('panels').innerHTML = absent(
            'No such branch',
            'Nothing is tracked at ' + base + '.',
            'manamap pilot deck-branch ' + slug + ' list');
          return;
        }
        document.getElementById('branchWhy').textContent = meta.why || '';
        document.getElementById('objective').innerHTML = objectivePanel(meta, nc);
        document.getElementById('panels').innerHTML = nc
          ? (tablePanel(nc) + liftPanel(nc) + forgePanel(nc) +
             billPanel(nc, meta) + trailPanel(meta) + limitsPanel(nc))
          : (trailPanel(meta) + absent(
              'The net change',
              'This branch has not been measured against the deck yet. Until it ' +
              'is, there is nothing here to decide on.',
              'manamap pilot net-change ' + slug + ' --branch ' + name + ' --write'));
      });
  }

  if (window.Shell && Shell.mount) Shell.mount();
  main();
})();
