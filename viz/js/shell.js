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
 * ZERO DEPENDENCIES, and that is what lets it be everywhere. It does not read
 * `MM`, `Session` or any artifact: `deck.html` and `workbench.html` load
 * neither the atlas nor the session, and a shell that needed them would either
 * pull 0.56 MB of card index onto two pages that do not draw cards, or exist on
 * only one page and stop being a shell.
 *
 * THE LIBRARY COUNT IS READ STRAIGHT FROM `localStorage`, and it is legible
 * there precisely because the library stores NAMES. A count of names needs no
 * corpus to resolve against — so the number is available on a page that has
 * never heard of a card. Had it stored row indices (the form that got the
 * previous attempt deleted) this would have been impossible without loading the
 * whole index, which is a second, quieter argument for names.
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
  function libraryCount() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return 0;
      var doc = JSON.parse(raw);
      return doc && Array.isArray(doc.cards) ? doc.cards.length : 0;
    } catch (e) {
      return 0;
    }
  }

  function currentSurface() {
    var file = (location.pathname.split('/').pop() || 'index.html');
    if (file === 'workbench.html') return 'bench';
    if (file === 'deck.html') return 'deck';
    return 'atlas';
  }

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
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
      '<div class="shell-library" title="' +
        (n ? 'cards you are gathering — open the Atlas to work with them'
           : 'keep cards in the Atlas and they collect here') + '">' +
        (n ? '<a href="index.html">' + n + ' in your library</a>'
           : '<span class="shell-empty">library empty</span>') +
      '</div>';
  }

  /* Put the strip at the top of the page, above whatever header exists.
   *
   * Injected rather than written into three HTML files, so the vocabulary and
   * the link set cannot drift between surfaces — which is the exact way they
   * drifted in the first place.
   */
  function mount() {
    if (document.getElementById('shell')) return render();
    var strip = document.createElement('div');
    strip.id = 'shell';
    strip.className = 'shell';
    document.body.insertBefore(strip, document.body.firstChild);
    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }

  // `refresh` is for the one page that can CHANGE the library while you are on
  // it. The other two only ever read it.
  return { refresh: render, libraryCount: libraryCount, mount: mount };
})();
