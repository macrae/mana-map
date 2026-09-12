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
 */
window.MANIFEST_VERSION = 3;
window.MANIFEST_URL = '../data/decks/index.json?v=' + window.MANIFEST_VERSION;

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
