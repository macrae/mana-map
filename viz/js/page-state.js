/* page-state.js — tell the local bench what this tab has open, for Jarvis.
 *
 * PRD v2 Step 7. "Is this card any good here?" names neither card nor deck; the
 * page does. A page registers ONE function that describes itself:
 *
 *   PageState.register(function () {
 *     return { deck: slug, mode: 'build', focus: 'Windfall', selected: [...] };
 *   });
 *
 * and this module does the rest: it polls that function, and when the snapshot
 * CHANGES (or the tab gains or loses focus) it POSTs it to `serve`'s `page/state`,
 * which keeps the latest per tab for `manamap pilot page-state`.
 *
 * PULL, NOT PUSH, on purpose: the Atlas has a dozen places that change the focus or
 * the selection, and a beacon wired into each is a beacon one of them forgets.
 * Polling a pure read of the page's state cannot miss a path.
 *
 * Fails silent and stays silent: with no local `serve` (the deployed site, a plain
 * static server) the probe answers no and nothing is ever sent. A snapshot that
 * throws is skipped, never surfaced — this must never break the page it describes.
 * Fields serve does not know are dropped there (`pilot/page_state.FIELDS`).
 */
window.PageState = (function () {
  'use strict';

  var POLL_MS = 1500;
  var snapshotFn = null;
  var last = '';
  var timer = null;
  var off = false;

  var tab = (function () {
    var id = null;
    try { id = sessionStorage.getItem('mm.tab'); } catch (e) { /* private mode */ }
    if (!id) {
      id = Math.random().toString(36).slice(2, 10);
      try { sessionStorage.setItem('mm.tab', id); } catch (e) { /* fine: one id per load */ }
    }
    return id;
  })();

  function pageName() {
    var file = window.location.pathname.split('/').pop() || 'index.html';
    var name = file.replace(/\.html$/, '');
    return name === 'index' ? 'atlas' : name;
  }

  function collect() {
    var own = {};
    if (snapshotFn) {
      try { own = snapshotFn() || {}; } catch (e) { own = {}; }
    }
    var s = {
      page: pageName(),
      url: window.location.pathname + window.location.search,
      title: document.title,
    };
    Object.keys(own).forEach(function (k) { s[k] = own[k]; });
    s.visible = document.visibilityState === 'visible';
    s.focused = s.visible && document.hasFocus();
    return s;
  }

  function tick() {
    if (off || !window.Api) return;
    Api.probe().then(function (ok) {
      if (!ok) { stop(); return; }
      if (!Api.has('page/state') && Api.commands.length) { stop(); return; }
      var s = collect();
      var key = JSON.stringify(s);
      if (key === last) return;
      last = key;
      Api.call('page/state', { tab: tab, state: s }).catch(function () { /* next change retries */ last = ''; });
    });
  }

  function start() {
    if (timer || off) return;
    tick();
    timer = setInterval(tick, POLL_MS);
    window.addEventListener('focus', tick);
    window.addEventListener('blur', tick);
    document.addEventListener('visibilitychange', tick);
  }

  function stop() {
    off = true;
    if (timer) clearInterval(timer);
    timer = null;
  }

  /* One describer per page; a second register replaces the first. */
  function register(fn) {
    snapshotFn = typeof fn === 'function' ? fn : null;
    last = '';
    start();
  }

  return {
    register: register,
    /* Send now (a page that just changed something big need not wait a poll). */
    poke: tick,
    get tab() { return tab; },
    /* For tests: the snapshot as it would be sent. */
    collect: collect,
  };
})();
