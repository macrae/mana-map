/* shell.js — one navigation strip, on every surface.
 *
 * The PRD calls this the core deliverable: "one product, not four tools."
 * Before it there were three pages with three different headers, each linking
 * to a different subset of the others — and `workbench.html`, which IS the
 * workbench, linked to the atlas under the name "the bench".
 *
 * WHAT IT IS: a thin strip above whatever header a page already has. Global
 * navigation and the library live here; page-specific actions (Pilot's Manual,
 * See it on the map) stay in the page's own `.head-links`, because they are
 * about the thing you are looking at rather than about where you are.
 *
 * NO ARTIFACT DEPENDENCIES, which is what lets it be everywhere. It reads no
 * `MM`, no card index, no deck file: `deck.html` and `workbench.html` do not
 * draw cards, and a shell that needed 0.56 MB of corpus would either weigh those
 * pages down or exist on one page and stop being a shell.
 *
 * IT DOES NOW READ `Session`, and the reason is the drawer. `session.js` is ~10
 * KB with no data behind it, and it is loaded on all three surfaces so that the
 * library can be OPENED anywhere rather than merely counted anywhere. The
 * `localStorage` read below survives as the fallback for the moment before
 * Session boots, so the strip still cannot break a page's first paint.
 *
 * THE COUNT IS LEGIBLE WITHOUT A CORPUS precisely because the library stores
 * NAMES. That is what makes the fallback possible, and — more importantly — what
 * lets the drawer render real card art on a page that has never heard of a card:
 * Scryfall's image endpoint takes a name. Had it stored row indices (the form
 * that got the previous attempt deleted) neither would work.
 *
 * ONE COUNT, ONE LIST. The strip and the drawer both read `libraryNames()`, so
 * "2 in your library" and the number of tiles cannot disagree — they are the
 * same array. They used to be `localStorage.cards.length` against
 * `Session.library.size`, computed from two representations that were not
 * inverses of each other.
 *
 * VOCABULARY, fixed here because this is the one place they are all named
 * together, and the relationship between the first two is the product's whole
 * shape:
 *
 *   THE WORKBENCH  the landing page, and the thing all of this IS — your decks
 *                  and everything the bench knows about them. Not one tool
 *                  among several; the surface the tools sit on.
 *   THE ATLAS      a TOOL on the workbench — 34,890 cards in embedding space.
 *   YOUR LIBRARY   the cards you are gathering into a deck. In Magic your
 *                  library IS your deck, which is why it is not a "basket".
 *
 * The PRD flags a naming collision (§4.2: "ManaMap" is used for both the
 * product and the embedding surface, resolve before public docs). This is the
 * resolution: the product is the Workbench, the embedding surface is the Atlas,
 * and Mana Map is the wordmark over both.
 */
window.Shell = (function () {
  'use strict';

  var STORE_KEY = 'manamap-library';

  var SURFACES = [
    { id: 'bench', href: 'workbench.html', label: 'Workbench',
      hint: 'your decks, and everything the bench knows about them' },
    { id: 'atlas', href: 'index.html', label: 'Atlas',
      hint: 'a tool on the workbench: all 34,890 cards in space' },
  ];

  /* How many cards are in the library, without resolving any of them.
   *
   * Tolerant on purpose: a missing key, a disabled localStorage, a document
   * from a schema this build does not know — every one of those is "no number
   * to show", never an exception out of a page's first paint.
   */
  function libraryNames() {
    // Session is the writer, so it is the truth the moment it exists. The stored
    // document is the same list — it is read only in the window before Session
    // has booted, and on the impossible page that never loads it.
    if (window.Session && Session.library) {
      try { return Session.library.names; } catch (e) { /* fall through */ }
    }
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return [];
      var doc = JSON.parse(raw);
      return doc && Array.isArray(doc.cards) ? doc.cards.slice() : [];
    } catch (e) {
      return [];
    }
  }

  function libraryCount() { return libraryNames().length; }

  /* THE CARD IMAGE, FROM A NAME AND NOTHING ELSE.
   *
   * `viz_index.json` carries no Scryfall id, no set code and no image URI — its
   * own docstring says the card image already shows all of that, "which is also
   * why the landing can paint from a name alone". So a name is the only card
   * identity this page ever has, and Scryfall's `cards/named` endpoint is the
   * only thing that turns one into a picture.
   *
   * `version=small` (146x204) rather than `normal` (488x680): the detail panel
   * draws ONE card and the popup preloads two, while a drawer draws the whole
   * library at once. This is the first place in the repo to fire dozens of image
   * requests in a breath, which is also why the tiles are lazy.
   */
  function cardImageUrl(name, version) {
    return 'https://api.scryfall.com/cards/named?exact='
      + encodeURIComponent(name) + '&format=image&version=' + (version || 'small');
  }

  /* A DOUBLE-FACED CARD 404s ON ITS OWN NAME. The corpus keys the full
   * `A // B` form — it is the graph key everywhere — and Scryfall answers some
   * of those with a 404 while resolving the front face alone (measured on
   * `Disciple of Freyalise // Garden of Freyalise`). So the first failure retries
   * the front face, and only the second gives up to a name plate. Both existing
   * image sites route through here and inherit the retry. */
  function imgFallback() {
    return 'if(!this.dataset.retried&&this.alt.indexOf(\' // \')>0)'
      + '{this.dataset.retried=1;this.src=' + JSON.stringify('https://api.scryfall.com/cards/named?exact=')
      + '+encodeURIComponent(this.alt.split(\' // \')[0])+' + JSON.stringify('&format=image&version=small') + ';}'
      + 'else{this.onerror=null;this.parentElement.classList.add(\'lib-tile-noart\');'
      + 'this.remove();}';
  }

  /* ── the drawer ─────────────────────────────────────────────────────────
   *
   * The count used to be a LINK TO `index.html`, which is the one thing it could
   * least afford to be: you clicked "2 in your library" and arrived at a page
   * that showed you nothing about your library. The library had no content view
   * anywhere in the frontend — you could add to it, you were told how many
   * things were in it, and the only way to see WHAT was to export a JSON file.
   *
   * It opens over the page rather than living inside one of them, because the
   * library is the connective tissue between surfaces (PRD 7.1) and belongs to
   * none of them. Same reason it is in the shell and not in `discovery.js`.
   */
  var open = false;

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function jsStr(s) { return esc(JSON.stringify(String(s))); }

  /* Is the card actually IN this corpus? A tile for a name the atlas cannot
   * place is drawn anyway and says so — a library that quietly comes back two
   * cards short is indistinguishable from one that came back whole, and the
   * previous design proved it by dropping such names into a status line that the
   * next status message overwrote. */
  function entryList() {
    if (window.Session && Session.library && Session.library.entries) {
      try { return Session.library.entries; } catch (e) { /* fall through */ }
    }
    return libraryNames().map(function (n) { return { name: n, row: -1 }; });
  }

  function tile(e) {
    // "Not in this corpus" is only sayable where there IS a corpus. On the
    // workbench and the deck page nothing has resolved a name, so every row is
    // -1 and flagging them would libel the whole library.
    var canTell = !!(window.Session && Session.library && Session.library.resolvable);
    var known = !canTell || e.row >= 0;
    return '<div class="lib-tile' + (known ? '' : ' lib-tile-unknown') + '">'
      + '<button class="lib-drop" title="Take out of your library" '
      +   'onclick="Shell.drop(' + jsStr(e.name) + ')">&times;</button>'
      + '<button class="lib-open" title="' + esc(e.name) + '" '
      +   'onclick="Shell.open(' + jsStr(e.name) + ')">'
      +   '<img src="' + esc(cardImageUrl(e.name)) + '" alt="' + esc(e.name) + '" '
      +     'loading="lazy" onerror="' + imgFallback() + '">'
      +   '<span class="lib-name">' + esc(e.name) + '</span>'
      + '</button>'
      + (known ? '' : '<span class="lib-flag">not in this corpus</span>')
      + '</div>';
  }

  function drawerHtml() {
    var es = entryList();
    var head = '<div class="lib-head">'
      + '<span class="lib-title">Your library</span>'
      + '<span class="lib-n">' + es.length + ' card' + (es.length === 1 ? '' : 's') + '</span>'
      + '<span class="lib-actions">'
      +   (es.length ? '<button class="lib-btn" onclick="Shell.clear()">Clear</button>' : '')
      +   '<button class="lib-btn" onclick="Shell.toggle()">Close</button>'
      + '</span></div>';
    if (!es.length) {
      return head + '<p class="lib-empty">Nothing kept yet. Open a card in the '
        + '<a href="index.html">Atlas</a> and press <b>Keep this card</b> — '
        + 'what you gather here becomes the deck you are building.</p>';
    }
    return head + '<div class="lib-grid">' + es.map(tile).join('') + '</div>';
  }

  function renderDrawer() {
    var el = document.getElementById('shell-drawer');
    if (!el) return;
    el.innerHTML = open ? drawerHtml() : '';
    el.style.display = open ? '' : 'none';
  }

  function currentSurface() {
    var file = (location.pathname.split('/').pop() || 'index.html');
    if (file === 'workbench.html') return 'bench';
    if (file === 'deck.html') return 'deck';
    return 'atlas';
  }

  function render() {
    var strip = document.getElementById('shell');
    if (!strip) return;
    var here = currentSurface();
    var n = libraryCount();

    var links = SURFACES.map(function (s) {
      // The surface you are on is NOT a link. A nav that offers to take you
      // where you already are teaches the reader the nav is decorative.
      var on = s.id === here;
      return on
        ? '<span class="shell-here" title="' + esc(s.hint) + '">' + esc(s.label) + '</span>'
        : '<a href="' + s.href + '" title="' + esc(s.hint) + '">' + esc(s.label) + '</a>';
    });

    // A deck page is a place, so it gets a crumb rather than a fourth surface:
    // there are three surfaces and any number of decks.
    if (here === 'deck') {
      var slug = new URLSearchParams(location.search).get('deck');
      links.push('<span class="shell-here">' + esc(slug || 'deck') + '</span>');
    }

    strip.innerHTML =
      '<div class="shell-brand">Mana&nbsp;Map</div>' +
      '<nav class="shell-nav">' + links.join('<span class="shell-sep">·</span>') + '</nav>' +
      '<div class="shell-library">' +
        '<button class="shell-lib-btn' + (open ? ' is-open' : '') + '"' +
          ' aria-expanded="' + (open ? 'true' : 'false') + '"' +
          ' title="' + (n ? 'the cards you are gathering — click to see them'
                          : 'keep cards in the Atlas and they collect here') + '"' +
          ' onclick="Shell.toggle()">' +
          '<span class="shell-lib-mark">&#9635;</span> ' +
          (n ? n + ' in your library' : 'library empty') +
        '</button>' +
      '</div>';
    renderDrawer();
  }

  /* Put the strip at the top of the page, above whatever header exists.
   *
   * Injected rather than written into three HTML files, so the vocabulary and
   * the link set cannot drift between surfaces — which is the exact way they
   * drifted in the first place.
   */
  function mount() {
    if (!document.getElementById('shell')) {
      var strip = document.createElement('div');
      strip.id = 'shell';
      strip.className = 'shell';
      document.body.insertBefore(strip, document.body.firstChild);
    }
    // The drawer is a SIBLING under the strip rather than a child of it: the
    // strip is a flex row and a full-width panel inside it would be a column in
    // that row. Inserted after, so it pushes the page down instead of covering
    // the thing you were reading.
    if (!document.getElementById('shell-drawer')) {
      var d = document.createElement('div');
      d.id = 'shell-drawer';
      d.className = 'lib-drawer';
      d.style.display = 'none';
      var strip2 = document.getElementById('shell');
      strip2.parentNode.insertBefore(d, strip2.nextSibling);
    }
    render();
  }

  /* ── what the drawer's controls do ──────────────────────────────────────
   *
   * Every one of these writes through `Session`, which saves and then emits, and
   * the emit calls `render()` right back. So the strip count, the tile count and
   * the store move together by construction — there is no second path where one
   * of them could be updated and another forgotten, which is the whole class of
   * bug this replaced. */
  function toggleDrawer() { open = !open; render(); }

  function drop(name) {
    if (window.Session && Session.library) Session.library.remove(name);
    else return;
    render();
  }

  /* Opening a card is the Atlas's job, so off the Atlas this is a link rather
   * than an action. `?cards=` is an existing inbound contract — the same one a
   * named-card walk uses — so there is no new plumbing and no second name reader. */
  function openCard(name) {
    // `MM.openCard` routes by mode and never touches the graph — see its note.
    // It answers false when the corpus has not booted, which is a reason to
    // navigate rather than to do nothing.
    if (window.MM && MM.openCard && MM.openCard(name)) {
      open = false;
      render();
      return;
    }
    location.href = 'index.html?cards=' + encodeURIComponent(name);
  }

  /* CLEAR ASKS FIRST. It wipes work that took ten minutes to gather, it used to
   * be unconfirmed, and it lived inside a collapsed block labelled "Start
   * somewhere else" — a destructive control mislabelled by its container. */
  function clearAll() {
    var n = libraryCount();
    if (!n) return;
    if (!window.confirm('Take all ' + n + ' card' + (n === 1 ? '' : 's')
        + ' out of your library? This cannot be undone.')) return;
    if (window.Session && Session.library) Session.library.clear();
    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }

  // `refresh` is for the one page that can CHANGE the library while you are on
  // it. The other two only ever read it.
  return {
    refresh: render,
    libraryCount: libraryCount,
    libraryNames: libraryNames,
    cardImageUrl: cardImageUrl,
    mount: mount,
    toggle: toggleDrawer,
    drop: drop,
    open: openCard,
    clear: clearAll,
    get isOpen() { return open; },
  };
})();
