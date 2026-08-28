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

  /* EVERY ROW CARRIES ITS OWN DEFINITION, because a figure whose meaning is not
   * on the page beside it gets guessed at, and the guesses go one way: a mean
   * read as a rate, a clock read as a win rate, a hoard read as mana. All three
   * have happened on this bench. `reads_as` and `what`/`why_we_care` are built
   * by `net_change.build` and travel in the artifact, so this page still
   * computes nothing. */
  function tablePanel(nc) {
    if (!nc) return '';
    var rows = (nc.table || []).map(function (r) {
      var head = '<tr class="v-' + esc(r.verdict) + '"><td>' + esc(r.measure) +
        (r.better_is ? ' <span class="ev">' + esc(r.better_is) +
                       ' is better</span>' : '') +
        '</td><td class="n">' + num(r.champion) + '</td><td class="n">' +
        num(r.branch) + '</td><td class="n">' + signed(r.delta) +
        '</td><td class="verdict">' + esc(r.verdict) +
        (r.verdict === 'noise' ? ' <span class="ev">MDE ' + num(r.mde) + '</span>' : '') +
        '</td></tr>';
      if (!r.reads_as && !r.what) return head;
      return head + '<tr class="reading"><td colspan="5">' +
        (r.reads_as ? '<p class="reads">' + esc(r.reads_as) + '</p>' : '') +
        (r.what ? '<details><summary>What is this, and why do we care?</summary>' +
          '<p>' + esc(r.what) + '</p>' +
          (r.scale ? '<p class="ev">Scale: ' + esc(r.scale) + '</p>' : '') +
          '<p class="ev"><b>Why:</b> ' + esc(r.why_we_care) + '</p></details>' : '') +
        '</td></tr>';
    }).join('');
    var derived = ((nc.definitions || {}).derived || []).map(function (d) {
      return '<details><summary>' + esc(d.name) + '</summary><p>' +
        esc(d.what) + '</p><p class="ev"><b>Why:</b> ' + esc(d.why) +
        '</p></details>';
    }).join('');
    return '<section class="panel"><h2>Measured</h2>' +
      '<p class="ev">' + esc(nc.harness.iterations.toLocaleString()) +
      ' games each, shared seed. A delta smaller than the run could detect is ' +
      'marked noise rather than ranked — that is no answer, not no change.</p>' +
      '<div class="tablewrap"><table class="netchange"><thead><tr><th>Measure</th>' +
      '<th class="n">Deck</th><th class="n">Branch</th><th class="n">&Delta;</th>' +
      '<th>Verdict</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      (derived ? '<h3>The figures that are not rows</h3>' + derived : '') +
      '</section>';
  }

  /* VALUE AND RISK, SIDE BY SIDE AND NEVER MERGED. The four risk kinds are the
   * point: a row that fell is a priced cost, a row inside the MDE is an open
   * question, and an effect this harness cannot see is neither. All three read
   * alike as prose, so each one is badged with what kind of claim it is. */
  var RISK_COPY = {
    paid: 'measured cost',
    unresolved: 'no answer',
    unmeasured: 'the model cannot see this',
    structural: 'true of the whole harness'
  };

  function ledgerPanel(nc) {
    var r = nc && nc.recommendation;
    if (!r || (!r.reward && !r.risk)) return '';
    var out = '<section class="panel ledger-panel"><h2>Reward, risk and cost</h2>';

    out += '<h3>The reward <span class="ev">(' + (r.reward || []).length +
           ')</span></h3>';
    if (!(r.reward || []).length) {
      out += '<p class="ev">Nothing beat its own minimum detectable ' +
             'difference.</p>';
    } else {
      out += '<ul class="ledger-list">' + r.reward.map(function (x) {
        return '<li class="gain"><b>' + esc(x.measure) + '</b>' +
          '<span class="reads">' + esc(x.reads_as) + '</span>' +
          (x.why_we_care ? '<span class="ev">' + esc(x.why_we_care) +
                           '</span>' : '') + '</li>';
      }).join('') + '</ul>';
    }

    out += '<h3>The risk <span class="ev">(' + (r.risk || []).length +
           ')</span></h3><ul class="ledger-list">' +
      (r.risk || []).map(function (x) {
        return '<li class="risk k-' + esc(x.kind) + '">' +
          '<span class="kind">' + esc(RISK_COPY[x.kind] || x.kind) + '</span>' +
          '<b>' + esc(x.what) + '</b>' +
          (x.detail ? '<span class="reads">' + esc(x.detail) + '</span>' : '') +
          (x.cards && x.cards.length
            ? '<span class="ev">' + esc(x.cards.join(', ')) + '</span>' : '') +
          (x.why_it_matters ? '<span class="ev">' + esc(x.why_it_matters) +
                              '</span>' : '') + '</li>';
      }).join('') + '</ul>';

    var c = r.cost;
    if (c) {
      out += '<h3>The cost</h3><p class="lede">' + esc(c.reads_as) + '</p>';
      if (c.mergeable === false) {
        out += '<p class="ev warn">NOT MERGEABLE YET — <code>deck-branch ' +
               'merge</code> refuses while any card is unsourced. That is a ' +
               'question about cardboard, not about whether the branch is ' +
               'right.</p>';
      }
      if ((c.free_to_raid || []).length) {
        out += '<p class="ev">Free to take, because the deck it sits in is ' +
               'already apart: ' + esc(c.free_to_raid.map(function (x) {
                 return x.name + ' (' + x.decks.join(', ') + ')';
               }).join('; ')) + '.</p>';
      }
    }
    return out + '</section>';
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

  /* THE VERDICT LEADS, because it is the question the page exists to answer.
   * It is derived by `net_change.recommend` from the same document the table
   * below it renders, so the two can never disagree — the reason this page
   * computes nothing itself. */
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
    var out = '<section class="panel verdict ' + cls + '"><h2>' +
      esc(r.state.toUpperCase()) + '</h2>' +
      '<p class="verdict-why">' + esc(r.because) + '</p>';
    var groups = [['rose', 'Rose'], ['fell', 'Fell'],
                  ['no_call', 'Could not tell apart']];
    out += '<ul class="ledger">';
    groups.forEach(function (g) {
      var rows = r[g[0]] || [];
      if (!rows.length) return;
      out += '<li class="led-' + g[0] + '"><b>' + esc(g[1]) + '</b> ' +
             esc(rows.join(' · ')) + '</li>';
    });
    out += '</ul>';
    (r.notes || []).forEach(function (n) {
      out += '<p class="ev">' + esc(n) + '</p>';
    });
    return out + '</section>';
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
           'It PROPOSES; nothing here has been measured.</p><ul class="swaplist">';
    doc.swaps.forEach(function (r) {
      var band = r.strength >= 0.65 ? 'strong'
               : r.strength >= 0.4 ? 'mild' : 'weak';
      out += '<li class="swap"><div class="swap-head">' +
        '<span class="strength ' + band + '" title="0 = two different cards ' +
        'that sort near each other. 1 = strictly better, cheaper, no strings. ' +
        'The shipped data tops out at 0.95.">' + esc(r.strength.toFixed(2)) +
        '</span> <span class="out">' + esc(r.out) + '</span>' +
        ' <span class="arrow">&rarr;</span> <span class="in">' + esc(r['in']) +
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

  /* THE SWAPS, SPLIT THE WAY THE EVIDENCE SPLITS. A spell swap moves the nine
   * sampled rows; a land swap moves only the deterministic mana block, because
   * the model has no tapped state and cannot rank two lands that make the same
   * colours. Rendered together, a land pass borrows credit from a spell pass.
   * Each `why` was written when the swap was staged, before any figure above
   * existed, so it cannot have been fitted to them. */
  function stagedPanel(meta, nc) {
    var ch = nc && nc.changes;
    var groups = ch
      ? [['Spells', ch.spells || []], ['Lands', ch.lands || []]]
      : [['Staged', (meta && meta.staged) || []]];
    var total = groups.reduce(function (a, g) { return a + g[1].length; }, 0);
    if (!total) return '';
    var out = '<section class="panel staged"><h2>The change (' + total +
      ')</h2><p class="ev">A challenger starts as a copy of the deck. These are ' +
      'the swaps that make it a different one, each with the reason it was ' +
      'staged.</p>';
    groups.forEach(function (g) {
      if (!g[1].length) return;
      out += '<h3>' + esc(g[0]) + ' <span class="ev">(' + g[1].length +
             ')</span></h3><ul class="swaplist">';
      g[1].forEach(function (r) {
        out += '<li><span class="pair"><span class="out">\u2212 ' + esc(r.out) +
               '</span><span class="in">+ ' + esc(r['in']) + '</span></span>' +
               (r.why ? '<span class="ev">' + esc(r.why) + '</span>' : '') +
               '</li>';
      });
      out += '</ul>';
    });
    return out + '</section>';
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
        document.getElementById('objective').innerHTML =
          proposalPanel(meta, slug, name) + verdictPanel(nc) +
          objectivePanel(meta, nc);
        document.getElementById('panels').innerHTML = stagedPanel(meta, nc) + (nc
          ? (tablePanel(nc) + ledgerPanel(nc) + forgePanel(nc) +
             billPanel(nc, meta) + trailPanel(meta) + limitsPanel(nc))
          : (trailPanel(meta) + absent(
              'The net change',
              'This branch has not been measured against the deck yet. Until it ' +
              'is, there is nothing here to decide on.',
              'manamap pilot net-change ' + slug + ' --branch ' + name + ' --write')));
        loadSwaps(slug, name);
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
    document.getElementById('panels').appendChild(host);
    if (!window.Api || !Api.ready) {
      host.innerHTML = absent(
        'Swaps to consider',
        'What in this list has a cheaper card doing its job. This one needs a ' +
        'local server; the report above does not.',
        'manamap pilot upgrades ' + slug + ' --branch ' + branch);
      return;
    }
    Api.call('branch/upgrades', { slug: slug, branch: branch })
      .then(function (doc) { host.innerHTML = swapsPanel(doc, slug, branch); })
      .catch(function (e) {
        host.innerHTML = absent('Swaps to consider', e.message,
          'manamap pilot upgrades ' + slug + ' --branch ' + branch);
      });
  }

  /* Delegated, because `swapsPanel` replaces the whole subtree. */
  function wire() {
    document.addEventListener('click', function (ev) {
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
    __swapsPanel: swapsPanel
  };

  if (window.Shell && Shell.mount) Shell.mount();
  wire();
  // Probe WITHOUT blocking the first paint — the artifacts render either way.
  if (window.Api && Api.probe) { Api.probe().then(main, main); } else { main(); }
})();
