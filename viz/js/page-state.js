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
 *
 * SEND TO JARVIS (2026-10-28). The pilot talks to Jarvis in the Claude Code
 * terminal and looks at these pages; a question about the thing on screen should
 * not have to be retyped there. `askButton(spec)` is a small button any page puts
 * beside a card, a swap or a branch; a click opens a one-line prompt prefilled
 * with a question for that thing (`askText`), editable. Send does two things:
 *
 *   - copies `/jarvis <text>` to the clipboard (the Clipboard API, then
 *     `execCommand('copy')`, then the text shown SELECTED for ⌘C) and says
 *     "copied — paste it into Claude Code";
 *   - under a local serve, posts the snapshot with an `ask` — `{text, page,
 *     deck, cards, branch}` — so `manamap pilot page-state` prints it as the
 *     pending question and Jarvis answers it (`--clear-ask` marks it taken). On a
 *     static host there is no serve, and the note says the clipboard is the
 *     whole of it. A serve too old to keep an ask answers without one, and the
 *     note says to restart it rather than claiming the question was held.
 *
 * The page never calls an LLM: the ask is a note left for Jarvis, and the
 * pilot's paste is what starts the turn.
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

  /* ── Send to Jarvis ─────────────────────────────────────────────────── */

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function joinAnd(parts) {
    if (parts.length < 2) return parts.join('');
    return parts.slice(0, -1).join(', ') + ' and ' + parts[parts.length - 1];
  }

  /* THE PREFILLED QUESTION, one phrasing per thing, here so every page asks the
   * same way. `deck` is the deck's display name (DeckFacts.displayName). */
  var askText = {
    card: function (card, deck) {
      return deck ? 'What does ' + card + ' do for ' + deck + '?'
                  : 'What does ' + card + ' do, and which of my decks wants it?';
    },
    /* The tray's rows as deck-edit.js keeps them: {op:'swap', out, in} | {op:'cut'|'add', card}. */
    change: function (rows, deck) {
      var parts = (rows || []).map(function (t) {
        if (t.op === 'swap') return 'swap ' + t.out + ' for ' + t['in'];
        return (t.op === 'cut' ? 'cut ' : 'add ') + t.card;
      });
      return 'Should I ' + joinAnd(parts) + (deck ? ' in ' + deck : '') + '?';
    },
    branch: function (name, deck) {
      return 'Review branch ' + name + (deck ? ' on ' + deck : '');
    },
  };

  /* The button. `spec` = {text, deck, cards, branch, label?, title?}; `deck` is the
   * SLUG (what page-state and Jarvis resolve), the name is already in `text`. */
  function askButton(spec) {
    // The button's own style, injected HERE: it lived only behind the prompt's open, so
    // every Ask Jarvis button drew as a bare white browser button until the first click
    // (the card panel, 2026-10-29).
    ensureStyle();
    var payload = { text: spec.text || '', deck: spec.deck || '', branch: spec.branch || '',
                    cards: spec.cards || [] };
    return '<button type="button" class="mm-jarvis-btn' + (spec.cls ? ' ' + esc(spec.cls) : '') +
      '" data-jarvis-ask="' + esc(JSON.stringify(payload)) + '" title="' +
      esc(spec.title || 'Ask Jarvis about this — copies the question for Claude Code') + '">' +
      esc(spec.label || 'Ask Jarvis') + '</button>';
  }

  var STYLE = '' +
    '.mm-jarvis-btn{background:transparent;border:1px solid #c4a74766;color:#c4a747;font:inherit;' +
    'font-size:0.72rem;letter-spacing:0.02em;padding:2px 8px;border-radius:4px;cursor:pointer;margin:2px 4px 2px 0}' +
    '.mm-jarvis-btn:hover,.mm-jarvis-btn:focus-visible{border-color:#c4a747;background:#c4a7471a}' +
    '.mm-ask{position:fixed;z-index:10000;width:min(460px,calc(100vw - 16px));background:#1a1a2e;' +
    'border:1px solid #c4a747;border-radius:6px;padding:10px;box-shadow:0 8px 28px #000a;color:#e6e6ee;' +
    'font:13px/1.4 Inter,system-ui,sans-serif}' +
    '.mm-ask-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;' +
    'color:#c4a747;font-size:0.72rem;letter-spacing:0.06em;text-transform:uppercase}' +
    '.mm-ask-row{display:flex;gap:6px}' +
    '.mm-ask-text{flex:1;min-width:0;background:#10101f;border:1px solid #3a3a5a;color:#f0f0f6;' +
    'border-radius:4px;padding:5px 7px;font:inherit}' +
    '.mm-ask-text:focus{outline:none;border-color:#c4a747}' +
    '.mm-ask button{background:#c4a747;color:#1a1a2e;border:0;border-radius:4px;padding:4px 10px;' +
    'font:inherit;font-weight:600;cursor:pointer}' +
    '.mm-ask button.mm-ask-close{background:transparent;color:#aaa;padding:0 4px;font-size:1rem}' +
    '.mm-ask-note{margin:6px 0 0;font-size:0.78rem;color:#b9b9c9}' +
    '.mm-ask-note:empty{display:none}' +
    '.mm-ask-note.warn{color:#e8b86a}';

  function ensureStyle() {
    if (document.getElementById('mmAskStyle')) return;
    var st = document.createElement('style');
    st.id = 'mmAskStyle';
    st.textContent = STYLE;
    (document.head || document.documentElement).appendChild(st);
  }

  var box = null;      // the open prompt, or null
  var lastAsk = null;  // what the last send did, for the browser suite

  function closeAsk() {
    if (box && box.parentNode) box.parentNode.removeChild(box);
    box = null;
  }

  function place(anchor) {
    if (!box) return;
    var r = anchor && anchor.getBoundingClientRect ? anchor.getBoundingClientRect() : null;
    var w = box.offsetWidth, h = box.offsetHeight;
    var vw = window.innerWidth, vh = window.innerHeight;
    var left = r ? r.left : (vw - w) / 2;
    var top = r ? r.bottom + 6 : vh / 3;
    if (top + h > vh - 8 && r) top = Math.max(8, r.top - h - 6);
    box.style.left = Math.max(8, Math.min(left, vw - w - 8)) + 'px';
    box.style.top = Math.max(8, Math.min(top, vh - h - 8)) + 'px';
  }

  /* Open the prompt for `spec`, beside `anchor` (the button), or centred. */
  function openAsk(spec, anchor) {
    ensureStyle();
    closeAsk();
    box = document.createElement('div');
    box.className = 'mm-ask';
    box.id = 'mmAsk';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-label', 'Ask Jarvis');
    box.innerHTML = '<div class="mm-ask-head"><span>Ask Jarvis</span>' +
      '<button type="button" class="mm-ask-close" aria-label="close">×</button></div>' +
      '<div class="mm-ask-row"><input class="mm-ask-text" type="text" spellcheck="true">' +
      '<button type="button" class="mm-ask-send">Send</button></div>' +
      '<p class="mm-ask-note"></p>';
    box.__spec = spec;
    document.body.appendChild(box);
    var input = box.querySelector('.mm-ask-text');
    input.value = spec.text || '';
    place(anchor);
    input.focus();
    input.select();
    // The pages' own keys (Escape's peel chain, the arrows, Shift's marquee) are
    // document-level listeners; keys typed here are this prompt's.
    box.addEventListener('keydown', function (ev) {
      ev.stopPropagation();
      if (ev.key === 'Escape') { ev.preventDefault(); closeAsk(); }
      else if (ev.key === 'Enter' && !ev.isComposing) { ev.preventDefault(); send(); }
    });
    box.querySelector('.mm-ask-send').addEventListener('click', send);
    box.querySelector('.mm-ask-close').addEventListener('click', closeAsk);
    return box;
  }

  function note(text, warn) {
    var el = box && box.querySelector('.mm-ask-note');
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('warn', !!warn);
  }

  /* 'copied' | 'manual'. Never rejects. */
  function copyText(text) {
    function legacy() {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      return ok ? 'copied' : 'manual';
    }
    if (navigator.clipboard && navigator.clipboard.writeText && window.isSecureContext) {
      return navigator.clipboard.writeText(text).then(function () { return 'copied'; }, legacy);
    }
    return Promise.resolve(legacy());
  }

  /* 'held' (serve kept it) | 'old' (serve answered without it) | 'static' (no serve)
   * | 'failed' (serve refused). Never rejects. */
  function post(ask) {
    if (!window.Api || !Api.probe) return Promise.resolve('static');
    return Api.probe().then(function (ok) {
      if (!ok) return 'static';
      if (Api.commands.length && !Api.has('page/state')) return 'old';
      var s = collect();
      s.ask = ask;
      return Api.call('page/state', { tab: tab, state: s }).then(function (row) {
        return row && row.ask && row.ask.text ? 'held' : 'old';
      }, function (e) { lastAsk && (lastAsk.error = e.message); return 'failed'; });
    });
  }

  function send() {
    if (!box) return Promise.resolve(null);
    var input = box.querySelector('.mm-ask-text');
    var text = String(input.value || '').replace(/\s+/g, ' ').trim();
    if (!text) { note('Type a question first.', true); return Promise.resolve(null); }
    var spec = box.__spec || {};
    var ask = { text: text, page: pageName(), deck: spec.deck || undefined,
                branch: spec.branch || undefined,
                cards: spec.cards && spec.cards.length ? spec.cards : undefined };
    var clip = '/jarvis ' + text;
    lastAsk = { ask: ask, clip: clip, posted: null, copied: null };
    var mine = box;
    note('Sending…');
    return Promise.all([post(ask), copyText(clip)]).then(function (got) {
      lastAsk.posted = got[0];
      lastAsk.copied = got[1];
      if (box !== mine) return lastAsk;
      var held = got[0] === 'held';
      var where = held ? ' Jarvis will also see it as the page’s question.'
        : got[0] === 'static' ? ' No local serve here, so the clipboard is the whole of it.'
        : got[0] === 'old' ? ' This serve predates page questions — restart it so Jarvis sees it too.'
        : ' serve did not take it' + (lastAsk.error ? ' (' + lastAsk.error + ')' : '') + '.';
      if (got[1] === 'copied') {
        note('Copied — paste it into Claude Code.' + where, !held && got[0] !== 'static');
      } else {
        // The clipboard refused: show the exact text, selected, for ⌘C.
        input.value = clip;
        input.setAttribute('readonly', '');
        input.focus();
        input.select();
        note('The clipboard refused — the text is selected: press ⌘C, then paste it into Claude Code.' +
             where, true);
      }
      return lastAsk;
    });
  }

  document.addEventListener('click', function (ev) {
    var t = ev.target;
    var b = t && t.closest && t.closest('[data-jarvis-ask]');
    if (b) {
      ev.preventDefault();
      ev.stopPropagation();
      var spec = {};
      try { spec = JSON.parse(b.getAttribute('data-jarvis-ask')) || {}; } catch (e) { spec = {}; }
      openAsk(spec, b);
      return;
    }
    // A click outside closes the prompt, unless it was inside it.
    if (box && !(t && t.closest && t.closest('.mm-ask'))) closeAsk();
  }, true);

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
    /* Send to Jarvis: the button, the prompt, the phrasing. */
    askButton: askButton,
    askText: askText,
    openAsk: openAsk,
    closeAsk: closeAsk,
    sendAsk: send,
    /* For tests: what the last send did ({ask, clip, posted, copied}). */
    get lastAsk() { return lastAsk; },
  };
})();
