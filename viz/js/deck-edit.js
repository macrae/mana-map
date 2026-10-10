/* deck-edit.js — edit a bench or brewing deck in place, from Build (window.DeckEdit).
 *
 * THE PILOT'S RULING (2026-10-10): a deck on the bench or brewing is where he
 * experiments, so a change lands STRAIGHT ON THE DECK with live before/after
 * numbers, undo/redo, and "Save version" (one git commit with a note). A
 * SLEEVED deck's list is cardboard: the same tray offers "Start a branch"
 * instead. An archived deck is read-only. `docs/viz.md` "Editing a deck".
 *
 * THE MODE IS A PREDICATE, NOT A GUESS. `deckRung` / `isWorkableDeck` (api.js)
 * decide which rung a deck is on; `Api.ready` decides whether there is a
 * machine to write with. Every rule about WHAT is legal lives in Python
 * (`deck_edit.plan`): the tray shows the server's blocking sentences verbatim
 * and never re-derives one, so the page and `manamap pilot edit` cannot
 * disagree. The static site has no API, and says so in one sentence.
 *
 * Loaded by index.html after build.js. Nothing here runs during mana-map.js's
 * boot: every entry point is called by Build (attach, trayHtml) or by the card
 * panel (cardButtons) at render time, behind a `window.DeckEdit` guard.
 */
window.DeckEdit = (function () {
  'use strict';

  /* How long the tray waits after the last change before it asks for numbers.
   * A tray is usually built in two clicks (a cut, then an add); previewing the
   * half-built one would spend a goldfish run on a list nobody meant. */
  var PREVIEW_DELAY_MS = 600;
  var POLL_MS = 900;
  var STORE = 'mm.deckEdit.';

  var S = fresh(null);

  function fresh(slug) {
    return {
      slug: slug, entry: null, active: null, mode: 'off',
      baseSha: null, history: null, axes: null,
      tray: [], pendingOut: null,
      seq: 0, timer: null, preview: null, previewState: 'idle', previewErr: null,
      previewFor: null, gf: null,
      busy: null, measure: null, notice: null, error: null, saved: null,
      branchDone: null,
      form: { note: '', branch: '', axis: '', op: '>=', value: '', why: '' },
    };
  }

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function spec() {
    return (S.active && S.active.spec) || window.formatSpec('commander');
  }

  /* ── the mode ─────────────────────────────────────────────────────────── */

  function computeMode() {
    if (!S.entry) return 'off';
    if (!window.isWorkableDeck(S.entry)) return 'readonly';
    if (!window.Api || !Api.ready) return 'noapi';
    // A server started before this sprint answers /api but has no edit verbs.
    if (Api.commands.length && !Api.has('deck/edit')) return 'oldserver';
    return window.deckRung(S.entry) === 'sleeved' ? 'branch' : 'edit';
  }

  function editing() { return S.mode === 'edit' || S.mode === 'branch'; }

  /* Build calls this whenever a deck is loaded — and again after a reload of
   * the SAME deck, which must keep the tray and the history it already has. */
  function attach(entry, active) {
    var same = S.slug && entry && S.slug === entry.slug;
    if (!same) {
      if (S.timer) clearTimeout(S.timer);
      S = fresh(entry ? entry.slug : null);
    }
    S.entry = entry || null;
    S.active = active || null;
    // Build renders its panel right after this returns, so the synchronous
    // paths paint nothing themselves; only the probe's answer repaints.
    if (!entry) { S.mode = 'off'; return; }
    if (!window.isWorkableDeck(entry)) { S.mode = 'readonly'; return; }
    if (same && editing()) return;
    S.mode = 'probing';
    var probe = window.Api && Api.probe ? Api.probe() : Promise.resolve(false);
    probe.then(function () {
      if (!S.entry || S.entry.slug !== entry.slug) return;
      S.mode = computeMode();
      if (editing()) {
        loadHistory(true);
        if (S.mode === 'branch') loadAxes();
      }
      repaintPanel();
    });
  }

  function detach() {
    if (S.timer) clearTimeout(S.timer);
    S = fresh(null);
  }

  /* The panel is Build's; a mode change re-renders it whole so the tray's
   * section appears or goes. Every smaller change paints only its own part. */
  function repaintPanel() {
    if (window.Build && Build.renderPanel && window.MM && MM.mode === 'build') Build.renderPanel();
    paintCards();
  }

  /* ── server state: history, axes ──────────────────────────────────────── */

  function loadHistory(restore) {
    var slug = S.slug;
    return Api.call('deck/edit/history', { slug: slug }).then(function (h) {
      if (S.slug !== slug) return;
      S.history = h;
      if (!S.baseSha || restore) S.baseSha = h.decklist_sha256;
      if (restore) restoreTray();
      paintAll();
      if (restore && S.tray.length) schedulePreview();
    }).catch(function (e) {
      if (S.slug !== slug) return;
      S.error = 'Could not read the edit history: ' + e.message;
      paint('status');
    });
  }

  function loadAxes() {
    var slug = S.slug;
    Api.call('branch/axes', { slug: slug }).then(function (doc) {
      if (S.slug !== slug) return;
      S.axes = doc.axes || [];
      if (!S.form.axis && S.axes.length) {
        var k8 = S.axes.filter(function (a) { return a.axis === 'kill_by_8'; })[0];
        pickAxis((k8 || S.axes[0]).axis);
      }
      paint('form');
    }).catch(function () { S.axes = []; paint('form'); });
  }

  function pickAxis(axis) {
    S.form.axis = axis;
    var row = (S.axes || []).filter(function (a) { return a.axis === axis; })[0];
    if (!row) return;
    S.form.op = row.lower_is_better ? '<=' : '>=';
    if (row.current != null) S.form.value = String(+row.current.toFixed(3));
  }

  /* ── the tray: persisted per list ─────────────────────────────────────── */

  /* Keyed on the slug AND the list's sha: a tray built against one list means
   * nothing against another, so a tray whose sha moved is dropped, not shown. */
  function storeKey() { return STORE + S.slug + '.' + S.baseSha; }

  function persist() {
    if (!S.slug || !S.baseSha) return;
    try {
      if (S.tray.length || S.pendingOut) {
        sessionStorage.setItem(storeKey(), JSON.stringify({ tray: S.tray, pendingOut: S.pendingOut }));
      } else {
        sessionStorage.removeItem(storeKey());
      }
    } catch (e) { /* private mode, quota: the tray just does not survive a reload */ }
  }

  function restoreTray() {
    try {
      var prefix = STORE + S.slug + '.';
      var keep = storeKey();
      for (var i = sessionStorage.length - 1; i >= 0; i--) {
        var k = sessionStorage.key(i);
        if (k && k.indexOf(prefix) === 0 && k !== keep) sessionStorage.removeItem(k);
      }
      var doc = JSON.parse(sessionStorage.getItem(keep) || 'null');
      if (doc && Array.isArray(doc.tray) && !S.tray.length) {
        S.tray = doc.tray.filter(function (t) { return t && (t.op === 'cut' || t.op === 'add' || t.op === 'swap'); });
        S.pendingOut = doc.pendingOut || null;
      }
    } catch (e) { /* unreadable: start empty */ }
  }

  function clearTray() {
    S.tray = [];
    S.pendingOut = null;
    S.preview = null;
    S.gf = null;
    S.previewState = 'idle';
    S.seq++;
    if (S.timer) clearTimeout(S.timer);
    persist();
  }

  function trayChanged() {
    S.error = null;
    S.branchDone = null;
    persist();
    paint('rows');
    paint('actions');
    paintCards();
    schedulePreview();
  }

  function findIdx(op, key, name) {
    for (var i = 0; i < S.tray.length; i++) {
      if (S.tray[i].op === op && S.tray[i][key] === name) return i;
    }
    return -1;
  }

  function cut(name) {
    var a = findIdx('add', 'card', name);
    if (a !== -1) { S.tray.splice(a, 1); trayChanged(); return; }
    if (findIdx('cut', 'card', name) !== -1 || findIdx('swap', 'out', name) !== -1) return;
    S.tray.push({ op: 'cut', card: name, qty: 1 });
    trayChanged();
  }

  function add(name) {
    if (S.pendingOut) {
      S.tray.push({ op: 'swap', out: S.pendingOut, 'in': name });
      S.pendingOut = null;
      trayChanged();
      return;
    }
    var c = findIdx('cut', 'card', name);
    if (c !== -1) { S.tray.splice(c, 1); trayChanged(); return; }
    if (findIdx('add', 'card', name) !== -1 || findIdx('swap', 'in', name) !== -1) return;
    S.tray.push({ op: 'add', card: name, qty: 1 });
    trayChanged();
  }

  function swapFor(name) {
    S.pendingOut = S.pendingOut === name ? null : name;
    persist();
    paint('rows');
    paintCards();
  }

  function remove(i) {
    if (i >= 0 && i < S.tray.length) { S.tray.splice(i, 1); trayChanged(); }
  }

  function setQty(i, n) {
    var t = S.tray[i];
    n = parseInt(n, 10);
    if (!t || t.op === 'swap' || !(n >= 1)) return;
    t.qty = n;
    trayChanged();
  }

  /* The tray as the server's op list. `card` is the wire's word for a name. */
  function wireOps() {
    return S.tray.map(function (t) {
      return t.op === 'swap' ? { op: 'swap', out: t.out, 'in': t['in'] }
                             : { op: t.op, card: t.card, qty: t.qty || 1 };
    });
  }

  function balance() {
    var outs = 0, ins = 0;
    S.tray.forEach(function (t) {
      if (t.op === 'swap') { outs++; ins++; }
      else if (t.op === 'cut') outs += t.qty || 1;
      else ins += t.qty || 1;
    });
    var copies = S.active ? S.active.copies : null;
    var after = copies == null ? null : copies - outs + ins;
    return { outs: outs, ins: ins, after: after };
  }

  function balanceLine() {
    var b = balance(), sp = spec();
    var head = b.outs + ' out, ' + b.ins + ' in';
    if (b.after == null) return head;
    var ok = sp.exact ? b.after === sp.size : b.after >= sp.size;
    if (ok) return head + ' — ' + b.after + ' cards';
    return head + ' — ' + (sp.exact ? '' : b.after + ' cards; ') + sp.name + ' needs ' +
      sp.size + (sp.exact ? '' : '+');
  }

  /* ── the preview ──────────────────────────────────────────────────────── */

  function schedulePreview() {
    if (S.timer) clearTimeout(S.timer);
    S.seq++;
    if (!S.tray.length) {
      S.preview = null; S.gf = null; S.previewState = 'idle';
      paint('preview'); paint('actions');
      return;
    }
    S.previewState = 'waiting';
    paint('preview');
    S.timer = setTimeout(runPreview, PREVIEW_DELAY_MS);
  }

  function runPreview() {
    var seq = ++S.seq, slug = S.slug, ops = wireOps();
    S.previewState = 'loading';
    paint('preview');
    Api.call('deck/edit/preview', { slug: slug, ops: ops }).then(function (res) {
      if (seq !== S.seq || slug !== S.slug) return;     // a newer tray replaced this one
      S.preview = res;
      S.previewFor = JSON.stringify(ops);
      S.previewState = 'done';
      S.gf = res.goldfish || null;
      paint('preview'); paint('actions');
      if (res.job && res.job.id) {
        pollJob(res.job.id, function (row) {
          if (seq !== S.seq || slug !== S.slug) return;
          if (row.state !== 'done') {
            S.gf = { absent: 'the goldfish run failed: ' + (row.error || 'unknown') };
          } else if (row.result && row.result.superseded) {
            return;                                      // the server says a newer one won
          } else {
            S.gf = (row.result && row.result.goldfish) || { absent: 'no goldfish result' };
          }
          paint('preview');
        });
      }
    }).catch(function (e) {
      if (seq !== S.seq || slug !== S.slug) return;
      S.previewState = 'error';
      S.previewErr = e.message;
      paint('preview'); paint('actions');
    });
  }

  /* The preview's blocking list, but only for the tray it was asked about:
   * an older tray's refusal must not disable Apply on a newer one. */
  function blocking() {
    if (!S.preview || S.previewFor !== JSON.stringify(wireOps())) return [];
    return S.preview.blocking || [];
  }

  function pollJob(id, done) {
    (function tick() {
      Api.call('job', { id: id }).then(function (row) {
        if (row.state === 'running') { setTimeout(tick, POLL_MS); return; }
        done(row);
      }).catch(function (e) { done({ state: 'failed', error: e.message }); });
    })();
  }

  /* ── apply, undo, redo, save ──────────────────────────────────────────── */

  function moves(diff) {
    var parts = [];
    Object.keys((diff && diff.out) || {}).forEach(function (n) { parts.push('− ' + n); });
    Object.keys((diff && diff['in']) || {}).forEach(function (n) { parts.push('+ ' + n); });
    return parts.join(', ') || 'no change';
  }

  function landed(verb, res) {
    S.baseSha = res.after_sha;
    clearTray();
    S.notice = verb + ': ' + moves(res.diff || (res.entry && res.entry.diff));
    if (res.job && res.job.id) startMeasuring(res.job);
    loadHistory(false);
  }

  function apply() {
    if (S.busy || !S.tray.length || blocking().length) return;
    S.busy = 'apply'; S.error = null;
    paint('actions');
    Api.call('deck/edit', { slug: S.slug, ops: wireOps(), expect_sha: S.baseSha })
      .then(function (res) { landed('Applied', res); })
      .catch(function (e) { S.error = e.message; })
      .then(function () { S.busy = null; paintAll(); });
  }

  function step(kind) {
    if (S.busy) return;
    S.busy = kind; S.error = null;
    paint('actions');
    Api.call('deck/edit/' + kind, { slug: S.slug, expect_sha: S.baseSha })
      .then(function (res) { landed(kind === 'undo' ? 'Undone' : 'Redone', res); })
      .catch(function (e) { S.error = e.message; })
      .then(function () { S.busy = null; paintAll(); });
  }

  /* THE REBUILD, polled. While it runs the panel's figures are the list BEFORE
   * the edit, and the status says so; when it lands Build reloads the deck and
   * the stamp (`cards.json`'s `decklist_sha256`) is checked against the list. */
  function startMeasuring(job) {
    var slug = S.slug;
    var m = { id: job.id, started: Date.now(), tick: null };
    S.measure = m;
    m.tick = setInterval(function () { if (S.measure === m) paint('status'); else clearInterval(m.tick); }, 1000);
    paint('status');
    pollJob(job.id, function (row) {
      clearInterval(m.tick);
      if (S.slug !== slug || S.measure !== m) return;
      S.measure = null;
      var r = row.result || {};
      if (row.state !== 'done') S.error = 'The rebuild failed: ' + (row.error || 'unknown');
      else if (r.behind) S.error = r.behind;
      else if (r.failures && r.failures.length) S.error = 'Rebuilt with failures: ' + r.failures.join('; ');
      var reload = window.Build && Build.reloadDeck ? Build.reloadDeck() : Promise.resolve();
      Promise.resolve(reload).then(function () { loadHistory(false); }, function () { paintAll(); });
    });
  }

  function sinceSave() {
    var s = S.history && S.history.since_save;
    if (!s) return null;
    var sum = function (o) { return Object.keys(o || {}).reduce(function (a, k) { return a + o[k]; }, 0); };
    return { ins: sum(s['in']), outs: sum(s.out) };
  }

  function save() {
    var note = S.form.note.trim();
    if (S.busy || !note || S.measure) return;
    S.busy = 'save'; S.error = null; S.saved = null;
    paint('actions'); paint('form');
    var slug = S.slug;
    Api.call('deck/save-version', { slug: slug, note: note, confirm: slug }).then(function (job) {
      pollJob(job.id, function (row) {
        if (S.slug !== slug) return;
        S.busy = null;
        if (row.state === 'done') {
          var r = row.result || {};
          S.saved = 'Saved as V' + r.version + ' — ' + String(r.commit || '').slice(0, 10) +
            (r.keeper ? ' · next: ' + r.keeper : '');
          S.form.note = '';
        } else {
          S.error = row.error || 'save failed';
        }
        loadHistory(false);
        paintAll();
      });
    }).catch(function (e) { S.busy = null; S.error = e.message; paintAll(); });
  }

  /* ── branch mode: a sleeved deck changes through a branch ─────────────── */

  /* Staging is 1-for-1, so the tray must pair. A swap is a pair already; the
   * remaining cuts and adds pair in tray order, copy by copy. */
  function pairs() {
    var out = [], inn = [], got = [];
    S.tray.forEach(function (t) {
      var q = t.qty || 1;
      if (t.op === 'swap') got.push({ out: t.out, card: t['in'] });
      else for (var i = 0; i < q; i++) (t.op === 'cut' ? out : inn).push(t.card);
    });
    if (out.length !== inn.length) {
      var b = balance();
      return { error: b.outs + ' out, ' + b.ins + ' in — a branch stages 1-for-1 swaps, ' +
        'so every card out needs one card in' };
    }
    for (var j = 0; j < out.length; j++) got.push({ out: out[j], card: inn[j] });
    return { pairs: got };
  }

  function startBranch() {
    if (S.busy) return;
    var p = pairs();
    var name = S.form.branch.trim();
    var value = S.form.value.trim();
    var why = null;
    if (!S.tray.length) why = 'The tray is empty — cut and add the cards the branch should try.';
    else if (p.error) why = p.error;
    else if (!name) why = 'Name the branch.';
    else if (!S.form.axis || value === '' || isNaN(+value)) {
      why = 'Pick the objective it must meet — an axis, a direction and a number.';
    }
    if (why) { S.error = why; paint('status'); return; }
    var slug = S.slug, objective = S.form.axis + ' ' + S.form.op + ' ' + value;
    S.busy = 'branch'; S.error = null;
    paint('actions');
    Api.call('branch/new', { slug: slug, name: name, objective: objective,
                             why: S.form.why.trim() || 'started from Build' })
      .then(function () {
        var chain = Promise.resolve(), done = 0;
        p.pairs.forEach(function (pr) {
          chain = chain.then(function () {
            return Api.call('branch/stage', { slug: slug, branch: name, out: pr.out, card: pr.card })
              .then(function () { done++; });
          });
        });
        return chain.then(function () { return done; }, function (e) {
          e.staged = done;
          throw e;
        });
      })
      .then(function (n) {
        S.branchDone = { name: name, staged: n };
        clearTray();
      })
      .catch(function (e) {
        S.error = e.message + (e.staged != null ? ' (' + e.staged + ' staged before it stopped)' : '');
        if (e.staged != null) S.branchDone = { name: name, staged: e.staged, partial: true };
      })
      .then(function () { S.busy = null; paintAll(); });
  }

  /* ── HTML ─────────────────────────────────────────────────────────────── */

  function btn(act, label, opts) {
    opts = opts || {};
    return '<button class="lens-btn de-btn' + (opts.cls ? ' ' + opts.cls : '') + '" data-de-act="' + act + '"' +
      (opts.card != null ? ' data-card="' + esc(opts.card) + '"' : '') +
      (opts.i != null ? ' data-i="' + opts.i + '"' : '') +
      (opts.disabled ? ' disabled' : '') +
      (opts.title ? ' title="' + esc(opts.title) + '"' : '') + '>' + esc(label) + '</button>';
  }

  /* The card panel's half: Add to deck / Cut / Swap for…. Empty outside edit
   * and branch mode, so the library's "+ Deck" behaviour is untouched there. */
  function cardButtons(row) {
    if (!editing() || !window.MM || MM.mode !== 'build') return '';
    var d = MM.cardRecord && MM.cardRecord(row);
    if (!d || !d.n) return '';
    return '<div class="de-card" data-row="' + (+row) + '">' + cardInner(d.n) + '</div>';
  }

  function cardInner(name) {
    var inDeck = !!(window.Build && Build.hasCard && Build.hasCard(name));
    var html = '<div class="de-card-row">';
    if (inDeck) {
      if (findIdx('cut', 'card', name) !== -1 || findIdx('swap', 'out', name) !== -1) {
        html += '<span class="de-pending">− in the tray</span>';
      } else {
        html += btn('cut', 'Cut', { card: name, cls: 'de-cut' }) +
          btn('swapfor', S.pendingOut === name ? 'Swapping… (cancel)' : 'Swap for…',
              { card: name, cls: 'de-swapfor' });
      }
    } else if (findIdx('add', 'card', name) !== -1 || findIdx('swap', 'in', name) !== -1) {
      html += '<span class="de-pending">+ in the tray</span>';
    } else {
      html += btn('add', S.pendingOut ? 'Bring in for ' + S.pendingOut : 'Add to deck',
                  { card: name, cls: 'de-add' });
    }
    return html + '</div>';
  }

  function paintCards() {
    var els = document.querySelectorAll('.de-card[data-row]');
    for (var i = 0; i < els.length; i++) {
      var d = window.MM && MM.cardRecord ? MM.cardRecord(+els[i].getAttribute('data-row')) : null;
      els[i].innerHTML = d && editing() ? cardInner(d.n) : '';
    }
  }

  function trayHtml() {
    if (!S.entry || S.mode === 'off') return '';
    var e = S.entry;
    if (S.mode === 'readonly') {
      // The manifest's `status` is `[status, HEADLINE, sentence]` (or a bare string
      // from an older one).
      var status = Array.isArray(e.status) ? e.status[0] : e.status;
      return '<div class="deck-section de-readonly" id="deckEditReadonly"><p class="lens-note">' +
        esc(e.deck_name || e.slug) + ' is archived (' + esc(status || 'archived') +
        ') — read only. Revive it to edit: <code>manamap pilot deck-state ' + esc(e.slug) +
        ' revive --reason "…"</code></p></div>';
    }
    if (S.mode === 'probing') return '';
    if (S.mode === 'noapi') {
      return '<div class="deck-section de-noapi" id="deckEditNoApi"><p class="lens-note">' +
        'Editing needs the local bench — run <code>manamap serve</code> and open this page from it.' +
        '</p></div>';
    }
    if (S.mode === 'oldserver') {
      return '<div class="deck-section de-noapi" id="deckEditNoApi"><p class="lens-note">' +
        'This <code>manamap serve</code> predates editing — restart it and reload the page.' +
        '</p></div>';
    }
    var title = S.mode === 'branch'
      ? 'Change it on a branch <span>sleeved — the list is cardboard</span>'
      : 'Edit this deck <span>' + esc(window.DECK_RUNGS[window.deckRung(e)] || '') + '</span>';
    return '<div class="deck-section de-tray" id="deckEditTray" data-mode="' + S.mode + '">' +
      '<div class="deck-section-title">' + title + '</div>' +
      '<div id="deStatus">' + statusInner() + '</div>' +
      '<div id="deRows">' + rowsInner() + '</div>' +
      '<div id="dePreview">' + previewInner() + '</div>' +
      '<div id="deActions">' + actionsInner() + '</div>' +
      '<div id="deForm">' + formInner() + '</div>' +
      '<div id="deHistory">' + historyInner() + '</div>' +
      '</div>';
  }

  var PARTS = { status: ['deStatus', statusInner], rows: ['deRows', rowsInner],
                preview: ['dePreview', previewInner], actions: ['deActions', actionsInner],
                form: ['deForm', formInner], history: ['deHistory', historyInner] };

  function paint(part) {
    var p = PARTS[part];
    var el = p && document.getElementById(p[0]);
    if (el) el.innerHTML = p[1]();
  }

  function paintAll() {
    if (!document.getElementById('deckEditTray')) { repaintPanel(); return; }
    Object.keys(PARTS).forEach(paint);
    paintCards();
  }

  function statusInner() {
    var out = '';
    if (S.measure) {
      out += '<p class="de-measuring">measuring… ' + Math.round((Date.now() - S.measure.started) / 1000) +
        ' s — the figures in this panel are the list before this change</p>';
    } else if (S.active && S.baseSha && S.active.listSha && S.active.listSha !== S.baseSha) {
      out += '<p class="de-stale">The figures in this panel are for an earlier list (cards.json is ' +
        'behind decklist.txt) — <code>manamap pilot edit ' + esc(S.slug) + ' --rebuild</code></p>';
    }
    if (S.notice) out += '<p class="de-notice">' + esc(S.notice) + '</p>';
    if (S.saved) out += '<p class="de-notice">' + esc(S.saved) + '</p>';
    if (S.branchDone) {
      var url = 'branch.html?deck=' + encodeURIComponent(S.slug) + '&branch=' + encodeURIComponent(S.branchDone.name);
      out += '<p class="de-notice">Branch <b>' + esc(S.branchDone.name) + '</b> ' +
        (S.branchDone.partial ? 'opened, ' : 'opened with ') + S.branchDone.staged + ' swap(s) staged — ' +
        '<a class="de-branch-link" href="' + esc(url) + '">measure it on the branch page →</a></p>';
    }
    if (S.error) out += '<p class="de-error">' + esc(S.error) + '</p>';
    return out;
  }

  function rowsInner() {
    var flexible = !spec().exact;
    var out = '';
    if (S.pendingOut) {
      out += '<p class="lens-note de-pending-out">Swapping out <b>' + esc(S.pendingOut) +
        '</b> — open the card to bring in and press its button. ' +
        btn('cancelswap', 'cancel', { cls: 'lens-btn-inline' }) + '</p>';
    }
    if (!S.tray.length) {
      return out + '<p class="lens-note de-empty">Open a card: <b>Cut</b> or <b>Swap for…</b> ' +
        'one in the deck, <b>Add to deck</b> one that is not. Nothing is written until ' +
        (S.mode === 'branch' ? 'you start the branch.' : 'Apply.') + '</p>';
    }
    out += '<ul class="de-rows">' + S.tray.map(function (t, i) {
      var label;
      if (t.op === 'swap') {
        label = '<span class="de-out">− ' + esc(t.out) + '</span> / <span class="de-in">+ ' + esc(t['in']) + '</span>';
      } else {
        var sign = t.op === 'cut' ? '−' : '+';
        label = '<span class="' + (t.op === 'cut' ? 'de-out' : 'de-in') + '">' + sign + ' ' +
          (flexible ? '<input class="de-qty" type="number" min="1" max="99" value="' + (t.qty || 1) +
                      '" data-de-qty="' + i + '" aria-label="copies"> ' : '') + esc(t.card) + '</span>';
      }
      return '<li class="de-row">' + label + btn('remove', '×', { i: i, cls: 'de-x', title: 'Take out of the tray' }) + '</li>';
    }).join('') + '</ul>';
    out += '<p class="de-balance">' + esc(balanceLine()) + '</p>';
    return out;
  }

  function signed(v, dp) {
    if (typeof v !== 'number' || !isFinite(v)) return '—';
    var s = v.toFixed(dp == null ? 3 : dp);
    return v > 0 ? '+' + s : s;
  }

  function num(v, dp) {
    return typeof v === 'number' && isFinite(v) ? v.toFixed(dp == null ? 3 : dp) : '—';
  }

  function dollars(c) {
    var v = (c || 0) / 100;
    return (v < 0 ? '−$' : (v > 0 ? '+$' : '$')) + Math.abs(v).toFixed(2);
  }

  function absentLine(label, block) {
    return '<li><span class="de-k">' + esc(label) + '</span> <span class="de-absent">' + esc(block.absent) + '</span></li>';
  }

  function instantHtml(p) {
    var out = '';
    if ((p.blocking || []).length) {
      out += '<div class="de-blocking"><b>Apply is blocked:</b><ul>' +
        p.blocking.map(function (b) { return '<li>' + esc(b) + '</li>'; }).join('') + '</ul></div>';
    }
    if ((p.warnings || []).length) {
      out += '<ul class="de-warnings">' + p.warnings.map(function (w) {
        return '<li>warning: ' + esc(w) + '</li>'; }).join('') + '</ul>';
    }
    var li = [];
    if (p.size) li.push('<li><span class="de-k">size</span> ' + p.size.before + ' → ' + p.size.after + '</li>');
    var cs = p.colour_sources;
    if (cs && cs.absent) li.push(absentLine('colour sources', cs));
    else if (cs) {
      li.push('<li><span class="de-k">colour sources</span> ' + Object.keys(cs).map(function (c) {
        var r = cs[c];
        return '<span class="de-col' + (r.short_after > 0 ? ' is-short' : '') + '">' + esc(c) + ' ' +
          r.before + (r.after !== r.before ? '→' + r.after : '') + '/' + r.target + '</span>';
      }).join(' ') + '</li>');
    }
    var cv = p.curve;
    if (cv && cv.absent) li.push(absentLine('curve', cv));
    else if (cv && cv.before) {
      var moved = Object.keys(cv.after).filter(function (k) { return (cv.after[k] || 0) !== (cv.before[k] || 0); })
        .map(function (k) { return (k === 'unknown' ? 'unknown' : 'MV' + k + (k === '7' ? '+' : '')) + ' ' +
                                   (cv.before[k] || 0) + '→' + cv.after[k]; });
      li.push('<li><span class="de-k">curve</span> ' + esc(moved.join(', ') || 'unchanged') + '</li>');
    }
    var cb = p.combos;
    if (cb && cb.absent) li.push(absentLine('combos', cb));
    else if (cb) {
      var names = function (rows) {
        return rows.slice(0, 3).map(function (r) { return (r.cards || []).join(' + '); }).join('; ') +
          (rows.length > 3 ? '; …' : '');
      };
      li.push('<li><span class="de-k">combos</span> ' + cb.before + ' → ' + cb.after +
        (cb.gained.length ? ' · gained ' + cb.gained.length + ': ' + esc(names(cb.gained)) : '') +
        (cb.lost.length ? ' · lost ' + cb.lost.length + ': ' + esc(names(cb.lost)) : '') + '</li>');
    }
    li.push('<li><span class="de-k">keep list</span> ' + ((p.keep_list_hits || []).length
      ? esc(p.keep_list_hits.length + ' protected card(s) cut') : 'clear') + '</li>');
    var pr = p.price;
    if (pr && pr.absent) li.push(absentLine('price', pr));
    else if (pr) {
      li.push('<li><span class="de-k">price</span> ' + dollars(pr.delta_cents) +
        ' <span class="de-asof">as of ' + esc((pr.dates && pr.dates.length ? pr.dates : [pr.as_of]).join(', ')) +
        (pr.source ? ', ' + esc(pr.source) : '') + '</span>' +
        (pr.unpriced && pr.unpriced.length ? ' · unpriced: ' + esc(pr.unpriced.join(', ')) : '') + '</li>');
    }
    var ro = p.roles;
    if (ro && !ro.absent && Object.keys(ro).length) {
      li.push('<li><span class="de-k">roles</span> ' + esc(Object.keys(ro).map(function (r) {
        return r + ' ' + (ro[r] > 0 ? '+' : '−') + Math.abs(ro[r]); }).join(', ')) + '</li>');
    }
    return out + '<ul class="de-instant">' + li.join('') + '</ul>';
  }

  /* The paired goldfish table: champion -> after, the delta, the interval ON THE
   * DIFFERENCE, and the verdict — the same rows `try` prints, with its trust line
   * and the caveat that the goldfish says nothing about board quality. */
  function goldfishHtml(g) {
    if (!g) return '';
    if (g.pending) return '<p class="lens-note de-gf-wait">goldfish: 10,000 paired games on both lists…</p>';
    if (g.absent) return '<p class="lens-note de-gf-absent">goldfish: ' + esc(g.absent) + '</p>';
    var rows = (g.table || []).map(function (r) {
      var ci = r.ci95_diff;
      var verdict = r.verdict === 'noise' ? 'no call' : r.verdict;
      return '<tr class="de-v-' + esc(r.verdict) + '"><td>' + esc(r.measure) + '</td>' +
        '<td>' + num(r.champion) + ' → ' + num(r.branch) + '</td>' +
        '<td>' + signed(r.delta) + '</td>' +
        '<td class="de-ci">' + (ci ? '[' + signed(ci[0]) + ', ' + signed(ci[1]) + ']' : '') + '</td>' +
        '<td>' + esc(verdict || '') + '</td></tr>';
    }).join('');
    return '<table class="de-gf"><thead><tr><th>measure</th><th>champion → after</th>' +
      '<th>Δ</th><th>95% on the difference</th><th>verdict</th></tr></thead><tbody>' + rows +
      '</tbody></table>' +
      '<p class="de-call">⇒ ' + esc(String(g.call || '').toUpperCase()) +
      ((g.better || []).length ? ': better on ' + esc(g.better.join(', ')) : '') +
      ((g.worse || []).length ? '; worse on ' + esc(g.worse.join(', ')) : '') + '</p>' +
      (g.trust ? '<p class="lens-note de-trust">' + esc(g.trust) + '</p>' : '') +
      (g.caveat ? '<p class="lens-note de-caveat">' + esc(g.caveat) + '</p>' : '');
  }

  function previewInner() {
    if (!S.tray.length) return '';
    if (S.previewState === 'error') return '<p class="de-error">The preview failed: ' + esc(S.previewErr) + '</p>';
    var stale = S.previewState === 'waiting' || S.previewState === 'loading';
    if (!S.preview) return '<p class="lens-note">reading the change…</p>';
    return '<div class="de-preview' + (stale ? ' is-stale' : '') + '">' +
      (stale ? '<p class="lens-note">updating for the new tray…</p>' : '') +
      instantHtml(S.preview) + goldfishHtml(S.gf) + '</div>';
  }

  function actionsInner() {
    var h = S.history || {};
    var block = blocking();
    if (S.mode === 'branch') {
      return btn('branch', S.busy === 'branch' ? 'Starting the branch…' : 'Start a branch',
                 { disabled: !!S.busy || !S.tray.length, cls: 'de-start-branch' });
    }
    var out = '<div class="de-actions">' +
      btn('apply', S.busy === 'apply' ? 'Applying…' : 'Apply',
          { disabled: !!S.busy || !S.tray.length || block.length > 0, cls: 'de-apply',
            title: block.length ? block.join(' · ') : 'Write this change to the deck (undoable)' }) +
      btn('undo', 'Undo', { disabled: !!S.busy || !(h.undo > 0), cls: 'de-undo' }) +
      btn('redo', 'Redo', { disabled: !!S.busy || !(h.redo > 0), cls: 'de-redo' }) +
      '</div>';
    if (block.length) {
      out += '<p class="de-error de-why">Apply is blocked: ' + esc(block.join(' · ')) + '</p>';
    }
    return out;
  }

  function formInner() {
    if (S.mode === 'branch') {
      var axes = S.axes || [];
      var opts = axes.map(function (a) {
        return '<option value="' + esc(a.axis) + '"' + (a.axis === S.form.axis ? ' selected' : '') + '>' +
          esc(a.axis) + (a.current != null ? ' (now ' + (+a.current.toFixed(3)) + ')' : '') +
          (a.needs ? ' — needs ' + esc(a.needs) : '') + '</option>';
      }).join('');
      return '<div class="de-form de-branch-form">' +
        '<label>Branch name <input data-de-field="branch" value="' + esc(S.form.branch) + '" placeholder="e.g. ramp-v1"></label>' +
        '<label>What it is for <input data-de-field="why" value="' + esc(S.form.why) + '" placeholder="one sentence"></label>' +
        '<label>Objective <span class="de-objective">' +
          '<select data-de-field="axis">' + (opts || '<option value="">loading the axes…</option>') + '</select>' +
          '<select data-de-field="op"><option' + (S.form.op === '>=' ? ' selected' : '') + '>&gt;=</option>' +
          '<option' + (S.form.op === '<=' ? ' selected' : '') + '>&lt;=</option></select>' +
          '<input data-de-field="value" value="' + esc(S.form.value) + '" size="6"></span></label>' +
        '<p class="lens-note">Pre-registered: the branch is graded on this one objective. ' +
        'Staging is 1-for-1, so the tray must pair every cut with an add.</p></div>';
    }
    var since = sinceSave();
    var ver = S.history && S.history.saved_version;
    var sinceLine = since
      ? '+' + since.ins + ' −' + since.outs + ' since ' + (ver != null ? 'V' + ver : 'the last commit')
      : 'nothing saved yet to compare against';
    return '<div class="de-form de-save-form">' +
      '<div class="deck-section-title">Save version <span class="de-since">' + esc(sinceLine) + '</span></div>' +
      '<input data-de-field="note" value="' + esc(S.form.note) + '" placeholder="the version’s note (required)">' +
      btn('save', S.busy === 'save' ? 'Saving…' : 'Save version',
          { disabled: !!S.busy || !S.form.note.trim() || !!S.measure, cls: 'de-save',
            title: 'One git commit of this deck’s paths, with your note' }) +
      '</div>';
  }

  function historyInner() {
    if (S.mode !== 'edit' || !S.history) return '';
    var rows = [], since = 0;
    for (var i = 0; i < (S.history.entries || []).length; i++) {
      var e = S.history.entries[i];
      if (e.kind === 'save') {
        rows.push('<li class="de-h-save">— saved' + (e.version != null ? ' V' + esc(e.version) : '') +
          ' ' + esc(e.at || '') + (e.note ? ' · ' + esc(e.note) : '') + '</li>');
        break;
      }
      since++;
      rows.push('<li class="de-h-' + esc(e.kind) + '"><span class="de-h-at">' + esc(String(e.at || '').replace('T', ' ')) +
        '</span> ' + esc(e.kind) + (e.source ? ' · ' + esc(e.source) : '') + ' · ' + esc(moves(e.diff)) +
        (e.note ? ' — ' + esc(e.note) : '') + '</li>');
    }
    return '<details class="de-history"><summary>History — ' + since + ' change(s) since the last save' +
      ' · undo ' + (S.history.undo || 0) + ', redo ' + (S.history.redo || 0) +
      (S.history.in_sync === false ? ' · the list moved outside the editor' : '') + '</summary>' +
      (rows.length ? '<ul>' + rows.join('') + '</ul>' : '<p class="lens-note">No edits yet.</p>') + '</details>';
  }

  /* ── events: delegated once, so a repaint never leaks a listener ──────── */

  document.addEventListener('click', function (ev) {
    var b = ev.target && ev.target.closest && ev.target.closest('[data-de-act]');
    if (!b || b.disabled) return;
    var act = b.getAttribute('data-de-act');
    var card = b.getAttribute('data-card');
    var i = +b.getAttribute('data-i');
    ev.preventDefault();
    if (act === 'cut') cut(card);
    else if (act === 'add') add(card);
    else if (act === 'swapfor') swapFor(card);
    else if (act === 'cancelswap') swapFor(S.pendingOut);
    else if (act === 'remove') remove(i);
    else if (act === 'apply') apply();
    else if (act === 'undo' || act === 'redo') step(act);
    else if (act === 'save') save();
    else if (act === 'branch') startBranch();
  });

  document.addEventListener('input', function (ev) {
    var f = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-de-field');
    if (!f || !(f in S.form)) return;
    S.form[f] = ev.target.value;
    if (f === 'note') {
      var s = document.querySelector('#deForm .de-save');
      if (s) s.disabled = !!S.busy || !S.form.note.trim() || !!S.measure;
    }
  });

  document.addEventListener('change', function (ev) {
    var t = ev.target;
    if (!t || !t.getAttribute) return;
    if (t.getAttribute('data-de-field') === 'axis') { pickAxis(t.value); paint('form'); return; }
    if (t.getAttribute('data-de-field') === 'op') { S.form.op = t.value; return; }
    var q = t.getAttribute('data-de-qty');
    if (q != null) setQty(+q, t.value);
  });

  return {
    attach: attach,
    detach: detach,
    cardButtons: cardButtons,
    trayHtml: trayHtml,
    get mode() { return S.mode; },
    // Read-only probes for the browser suite.
    get tray() { return JSON.parse(JSON.stringify(S.tray)); },
    get baseSha() { return S.baseSha; },
    __wireOps: wireOps,
    __balanceLine: balanceLine,
  };
})();
