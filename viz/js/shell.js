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
    // AN APPENDIX, NOT A THIRD PLACE TO WORK. The Atlas offers a similarity
    // toggle and cannot explain what it is toggling; this is where that lives.
    // Marked `appendix` so it renders after a divider and in a quieter weight —
    // giving a reference page the same visual rank as the two surfaces you
    // actually work on would misdescribe what the nav is for.
    { id: 'spaces', href: 'spaces.html', label: 'Spaces', appendix: true,
      hint: 'reference: where each embedding space comes from, what its metrics '
          + 'mean, and which one to ask' },
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
      if (!doc) return [];
      // BOTH SHAPES. v1 kept a flat `cards`; v2 keeps `zones[].cards`. This
      // fallback was written against v1 and not revisited when piles landed,
      // so a v2 document read as EMPTY — the strip saying "library empty" over
      // a full library, in exactly the window before Session boots where this
      // is the only reader there is. A fallback that silently answers zero is
      // worse than no fallback, because zero is a plausible answer.
      if (Array.isArray(doc.cards)) return doc.cards.slice();
      return (doc.zones || []).reduce(function (all, z) {
        return all.concat(z.cards || []);
      }, []);
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
  /* A DELEGATED LISTENER, NOT AN INLINE HANDLER, and the first version is why.
   *
   * It built the handler as a string containing `JSON.stringify(...)` output —
   * so the attribute value carried a literal `"`, which TERMINATED the
   * attribute. The handler was truncated mid-block, every tile threw
   * `Unexpected end of input`, and the DFC retry it exists to perform never ran
   * once. `mana-map.js`'s two inline handlers already know this and hand-escape
   * with `.replace(/"/g, '&quot;')`; this needs no escaping because nothing is
   * interpolated into markup at all.
   *
   * `error` does not bubble, so the listener is registered in the CAPTURE phase
   * on the drawer — one listener for the whole grid rather than one per tile.
   */
  /* ── a card name you can inspect ────────────────────────────────────────
   *
   * EVERY REPORT NAMES CARDS AND NONE OF THEM SHOWED ONE. The branch diff lists
   * 39 cards, the bill lists 21, the pull list says what to buy — and a pilot
   * reading "Scion of Opulence" against "Pitiless Plunderer" had to leave the
   * page to find out what either one does. The magazine renderer solved this
   * with `a.cardref` and a hover pop; the workbench pages never got it.
   *
   * BY NAME, WHICH IS THE ONLY KEY THESE PAGES HAVE. `net_change.json` carries
   * names, not Scryfall ids or image URIs, and that is the same property that
   * lets the library drawer draw art on a page that has never loaded the
   * corpus: `cards/named?exact=` takes a name, so nothing has to be joined.
   *
   * THE IMAGE IS FETCHED ON HOVER AND NOT BEFORE. A branch page names ~60
   * cards; eager art would be 60 requests against a public API for a page whose
   * job is a table of numbers. One delegated listener, one reused <img>, and a
   * request only for the card actually being inspected.
   */
  function cardHref(name) {
    return 'https://scryfall.com/search?q=' +
      encodeURIComponent('!"' + String(name) + '"');
  }

  /* Returns MARKUP, because every caller builds strings. `esc` covers the text
   * and the data attribute; the href is URL-encoded. */
  function cardLink(name, cls) {
    var n = String(name == null ? '' : name);
    if (!n) return '';
    return '<a class="cardname' + (cls ? ' ' + cls : '') + '" target="_blank"' +
      ' rel="noopener" href="' + esc(cardHref(n)) + '"' +
      ' data-card="' + esc(n) + '">' + esc(n) + '</a>';
  }

  var popEl = null;

  function cardPop() {
    if (popEl) return popEl;
    popEl = document.createElement('img');
    popEl.className = 'cardpop';
    popEl.alt = '';
    popEl.hidden = true;
    // A DFC 404s on its full `A // B` name and resolves on the front face —
    // the same retry `onImageError` performs for the drawer tiles.
    popEl.addEventListener('error', function () {
      var n = popEl.dataset.card || '';
      if (!popEl.dataset.retried && n.indexOf(' // ') > 0) {
        popEl.dataset.retried = '1';
        popEl.src = cardImageUrl(n.split(' // ')[0], 'normal');
        return;
      }
      popEl.hidden = true;
    });
    document.body.appendChild(popEl);
    return popEl;
  }

  var popAnchor = null;

  /* Flip above when there is no room below, and clamp to the viewport so a card
   * named at the right edge does not open off-screen. */
  function placePop(a) {
    var el = cardPop();
    var r = a.getBoundingClientRect();
    var top = r.bottom + 8;
    if (top + 300 > window.innerHeight && r.top > 320) top = r.top - 308;
    el.style.top = Math.max(4, top) + 'px';
    el.style.left = Math.min(r.left, window.innerWidth - 230) + 'px';
  }

  function onCardOver(ev) {
    var a = ev.target && ev.target.closest && ev.target.closest('a.cardname');
    if (!a) return;
    var name = a.getAttribute('data-card');
    if (!name) return;
    var el = cardPop();
    if (el.dataset.card !== name) {
      el.dataset.card = name;
      delete el.dataset.retried;
      el.src = cardImageUrl(name, 'normal');
    }
    popAnchor = a;
    placePop(a);
    el.hidden = false;
  }

  function onCardOut(ev) {
    var a = ev.target && ev.target.closest && ev.target.closest('a.cardname');
    if (a && popEl) { popEl.hidden = true; popAnchor = null; }
  }

  function onImageError(ev) {
    var img = ev.target;
    if (!img || img.tagName !== 'IMG' || !img.closest('.lib-tile')) return;
    var name = img.getAttribute('alt') || '';
    // A double-faced card 404s on its own full `A // B` name — the corpus keys
    // that form because it is the graph key — while resolving on the front face
    // alone. Measured on `Disciple of Freyalise // Garden of Freyalise`.
    if (!img.dataset.retried && name.indexOf(' // ') > 0) {
      img.dataset.retried = '1';
      img.src = cardImageUrl(name.split(' // ')[0]);
      return;
    }
    // Out of retries: leave the reserved box and the name, which is the whole
    // of what the tile has to say.
    var tile = img.closest('.lib-tile');
    if (tile) tile.classList.add('lib-tile-noart');
    img.remove();
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

  function lib() { return (window.Session && Session.library) || null; }

  function tile(e) {
    // "Not in this corpus" is only sayable where there IS a corpus.
    var canTell = !!(lib() && lib().resolvable);
    var known = !canTell || e.row >= 0;
    var others = (lib() ? lib().zones : []).filter(function (z) {
      return z.name !== (lib() && lib().active);
    });
    // MOVE IS A SELECT, NOT A DRAG. One card lives in one pile, so the whole
    // operation is "which pile" — and a select says the available answers out
    // loud, works by keyboard, and needs no drop target to aim at.
    var move = others.length
      ? '<select class="lib-move" title="Move to another pile" '
        + 'onchange="Shell.move(' + jsStr(e.name) + ', this.value); this.value=\'\'">'
        + '<option value="">move&hellip;</option>'
        + others.map(function (z) {
            return '<option value="' + esc(z.name) + '">' + esc(z.name) + '</option>';
          }).join('')
        + '</select>'
      : '';
    return '<div class="lib-tile' + (known ? '' : ' lib-tile-unknown') + '">'
      + '<button class="lib-drop" title="Take out of your library" '
      +   'onclick="Shell.drop(' + jsStr(e.name) + ')">&times;</button>'
      + '<button class="lib-open" title="' + esc(e.name) + '" '
      +   'onclick="Shell.open(' + jsStr(e.name) + ')">'
      +   '<img src="' + esc(cardImageUrl(e.name)) + '" alt="' + esc(e.name) + '" '
      +     'loading="lazy">'
      +   '<span class="lib-name">' + esc(e.name) + '</span>'
      + '</button>'
      + (known ? '' : '<span class="lib-flag">not in this corpus</span>')
      + move
      + '</div>';
  }

  /* The piles, as tabs. The count on each is the whole point of having them —
   * a zone you cannot size at a glance is a folder, not a working surface. */
  function zoneBar() {
    var L = lib();
    if (!L || !L.zones) return '';
    var active = L.active;
    return '<div class="lib-zones">'
      + L.zones.map(function (z) {
          return '<button class="lib-zone' + (z.name === active ? ' is-active' : '') + '"'
            + ' onclick="Shell.zone(' + jsStr(z.name) + ')">'
            + esc(z.name) + '<span class="lib-zone-n">' + z.count + '</span></button>';
        }).join('')
      + '<button class="lib-zone lib-zone-new" title="Start another pile" '
      +   'onclick="Shell.newZone()">+</button>'
      + '</div>';
  }

  function drawerHtml() {
    var L = lib();
    var all = entryList();
    var active = L ? L.active : null;
    var shown = active ? all.filter(function (e) { return e.zone === active; }) : all;
    var head = '<div class="lib-head">'
      + '<span class="lib-title">Your library</span>'
      + '<span class="lib-n">' + all.length + ' card' + (all.length === 1 ? '' : 's')
      +   (L && L.zones.length > 1 ? ' in ' + L.zones.length + ' piles' : '') + '</span>'
      + '<span class="lib-actions">'
      +   (active ? '<button class="lib-btn" onclick="Shell.renameZone()">Rename</button>'
      +             '<button class="lib-btn" onclick="Shell.dropZone()">Delete pile</button>' : '')
      +   (shown.length ? '<button class="lib-btn" onclick="Shell.consider()">Consider for a deck…</button>' : '')
      +   (shown.length ? '<button class="lib-btn" onclick="Shell.treat()">Treat a deck…</button>' : '')
      +   (shown.length ? '<button class="lib-btn" onclick="Shell.clear()">Empty pile</button>' : '')
      +   '<button class="lib-btn" onclick="Shell.toggle()">Close</button>'
      + '</span></div>' + zoneBar();
    if (!shown.length) {
      return head + '<p class="lib-empty">'
        + (all.length
            ? 'Nothing in <b>' + esc(active) + '</b> yet. Cards you keep land in the '
              + 'pile that is open, so switch to another tab or keep something here.'
            : 'Nothing kept yet. Open a card in the <a href="index.html">Atlas</a> and '
              + 'press <b>Keep this card</b> — what you gather here becomes the deck '
              + 'you are building.')
        + '</p>';
    }
    return head + '<div class="lib-grid">' + shown.map(tile).join('') + '</div>';
  }

  // One delegated listener for the whole grid; `error` does not bubble, so it
  // is registered in the capture phase. Wired once, on first render.
  var errorsWired = false;

  function renderDrawer() {
    var el = document.getElementById('shell-drawer');
    if (!el) return;
    if (!errorsWired) { el.addEventListener('error', onImageError, true); errorsWired = true; }
    el.innerHTML = open ? drawerHtml() : '';
    el.style.display = open ? '' : 'none';
  }

  function currentSurface() {
    var file = (location.pathname.split('/').pop() || 'index.html');
    if (file === 'workbench.html') return 'bench';
    if (file === 'deck.html') return 'deck';
    if (file === 'spaces.html') return 'spaces';
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

    // The appendix hangs off the end behind its own divider rather than joining
    // the run of surfaces, so the nav still reads as "two places, plus a
    // reference" instead of three equal destinations.
    var main = [], extra = [];
    SURFACES.forEach(function (s, i) {
      (s.appendix ? extra : main).push(links[i]);
    });
    if (here === 'deck') { main = links.slice(); extra = []; }

    strip.innerHTML =
      '<div class="shell-brand">Mana&nbsp;Map</div>' +
      '<nav class="shell-nav">' + main.join('<span class="shell-sep">·</span>') +
        (extra.length
          ? '<span class="shell-sep shell-appendix-sep">|</span>' +
            '<span class="shell-appendix">' + extra.join('') + '</span>'
          : '') +
      '</nav>' +
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

  /* CLEAR ASKS FIRST, and it now names the PILE rather than the library. It
   * wipes work that took ten minutes to gather; with zones it could wipe work
   * that has nothing to do with what you are looking at, so it is scoped to
   * the open tab and says so in the question. */
  function clearAll() {
    var L = lib();
    if (!L) return;
    var zone = L.active;
    var n = L.entries.filter(function (e) { return e.zone === zone; }).length;
    if (!n) return;
    if (!window.confirm('Take all ' + n + ' card' + (n === 1 ? '' : 's')
        + ' out of "' + zone + '"? This cannot be undone.')) return;
    L.clearZone(zone);
    render();
  }

  /* ── the piles ──────────────────────────────────────────────────────────
   *
   * `prompt` and `confirm` rather than inline editors, deliberately: these are
   * rare, one-field operations on a page whose job is showing cards, and a
   * bespoke rename widget would be more code than the feature. */
  function zone(name) {
    if (lib()) lib().setActive(name);
    render();
  }

  function newZone() {
    var name = window.prompt(
      'Name the pile — what are you collecting?\n\n'
      + 'Cards you keep land in whichever pile is open.', '');
    if (name === null) return;
    if (lib() && !lib().addZone(name)) {
      if (String(name).trim()) window.alert('There is already a pile called "'
        + String(name).trim() + '".');
      return;
    }
    render();
  }

  /* HAND THE OPEN PILE OVER AS CANDIDATES — not as a deck.
   *
   * `build/save` already sends the library to a brief's `must_include`, which
   * PROMISES the cards are in the 99. This is the other claim: consider these.
   * It writes `data/decks/<slug>/pool.txt`, which `manamap pilot candidates
   * --pool library` reads to substitute each card in and measure what it does.
   * The two must not share a slot — afterwards nothing could tell a card you
   * committed to from one you were only weighing.
   *
   * ONE PILE, ONE QUESTION. It sends `zoneNames` and not the whole library, the
   * same scope rule a brief keeps: exporting every pile would put an artifact
   * collection into a Zur build.
   */
  function consider() {
    // TWO WAYS TO GET THIS LINE WRONG AND BOTH SHIPPED HERE.
    // `var lib = lib()` shadows the accessor and throws on the call. And
    // `zoneNames` is a GETTER on Session.library (session.js), not a method —
    // calling it threw `TypeError: store.zoneNames is not a function` on every
    // press of this button, so the pile -> pool.txt pipe had never once worked.
    // `node --check` passes both happily; nothing tested `consider()` or
    // `pool/save`, which is why it took an audit rather than a use to find.
    var store = lib();
    var names = store ? store.zoneNames : [];
    if (!names.length) { return; }
    if (!window.Api || !Api.ready) {
      // Absent, never broken: a deployed page has no server and says so rather
      // than offering a control that quietly does nothing.
      alert('This needs a local server — run `manamap serve`.');
      return;
    }
    var slug = prompt('Consider these ' + names.length + ' card(s) for which deck?');
    if (!slug) { return; }
    Api.call('pool/save', { slug: slug.trim(), cards: names })
      .then(function (r) {
        alert('Saved ' + r.cards + ' candidate(s) to ' + r.path +
              '\n\nNext:\n  ' + r.next);
      })
      .catch(function (e) { alert('Could not save the pool: ' + e.message); });
  }

  /* OPEN A CHALLENGER FROM THIS PILE.
   *
   * `consider` hands a pile over as candidates and stops there — the pilot then
   * goes to a terminal, opens a branch, stages swaps and measures. This does the
   * first step so the rest has somewhere to happen: a branch identical to the
   * deck, the pile attached, and the workbench open on it.
   *
   * IT ASKS FOR AN OBJECTIVE and refuses without one. That is not ceremony: the
   * Ur-Dragon treasure branch stated "treasure is the engine", achieved it 4.4x
   * over, and lost on the purpose nobody wrote down. A branch that cannot be
   * falsified gets graded on whether it did what it does.
   */
  function treat() {
    var store = lib();
    var names = store ? store.zoneNames : [];
    if (!names.length) { return; }
    if (!window.Api || !Api.ready) {
      alert('This needs a local server — run `manamap serve`.');
      return;
    }
    Api.call('decks', {}).then(function (r) {
      var decks = (r && r.decks ? r.decks : r) || [];
      var slugs = decks.map(function (d) { return d.slug || d; });
      var slug = prompt('Treat which deck with these ' + names.length +
                        ' card(s)?\n\n' + slugs.join(', '));
      if (!slug) { return; }
      var branch = prompt('Name the branch:', 'treatment');
      if (!branch) { return; }
      var why = prompt('In a sentence, what is this treatment for?') || '';
      return objectiveFor(slug.trim(), why).then(function (obj) {
        if (!obj) { return; }
        return Api.call('branch/new', {
          slug: slug.trim(), name: branch.trim(), objective: obj,
          why: why, cards: names
        }).then(function (got) { location.href = got.url; });
      });
    }).catch(function (e) { alert('Could not open the branch: ' + e.message); });
  }

  /* THE DOCTOR PROPOSES, THE PILOT CONFIRMS.
   *
   * `<measure> <op> <number>` is a vocabulary, and asking a pilot to produce one
   * from memory in a `prompt()` is asking them to guess at `OBJECTIVE_AXES` and
   * at what number their deck could plausibly reach. The doctor reads the
   * deck's CURRENT readings and proposes one axis with that reading beside it;
   * the pilot accepts it or types their own. It writes nothing either way.
   *
   * A REFUSAL IS AN ANSWER. "Make it better" names no axis, and the charter
   * returns `axis: null` with the alternatives rather than guessing — so this
   * falls through to the manual prompt with the doctor's own reasoning shown,
   * which is more than the pilot had before they asked.
   */
  function objectiveFor(slug, direction) {
    var manual = function (hint) {
      return prompt(
        (hint ? hint + '\n\n' : '') +
        'The objective it must meet — <measure> <op> <number>.\n\n' +
        'e.g. hoard_8 >= 6.0   ·   kill_by_8 >= 0.30   ·   stall <= 0.03\n\n' +
        'A branch that cannot be falsified gets graded on whether it did what ' +
        'it does, not on whether it was worth doing.');
    };
    if (!direction || !Api.has('branch/objective')) {
      return Promise.resolve(manual());
    }
    if (!confirm('Ask the deck doctor to turn that into a measurable ' +
                 'objective?\n\n' + OBJECTIVE_PRICE +
                 '\n\nCancel to type one yourself.')) {
      return Promise.resolve(manual());
    }
    return Api.call('branch/objective', { slug: slug, direction: direction })
      .then(function (job) { return pollJob(job.id); })
      .then(function (row) {
        var got = readObjective(row);
        if (!got || !got.objective || !got.objective.axis) {
          return manual('The doctor could not resolve that into one axis' +
                        (got && got.unresolved ? ':\n' + got.unresolved : '.'));
        }
        var o = got.objective;
        var expr = o.axis + ' ' + o.op + ' ' + o.value;
        var msg = 'The doctor proposes:\n\n  ' + expr +
          (got.current_reading != null
            ? '\n  (now ' + got.current_reading + ')' : '') +
          (got.why ? '\n\n' + got.why : '') +
          '\n\nOK to use it. Cancel to type your own.';
        return confirm(msg) ? expr : manual();
      })
      .catch(function (e) { return manual('The doctor could not be reached: ' +
                                          e.message); });
  }

  var OBJECTIVE_PRICE = '~8-15k tokens. It writes nothing; you confirm.';

  function pollJob(id) {
    return new Promise(function (resolve, reject) {
      var tick = function () {
        Api.call('job', { id: id }).then(function (row) {
          if (row.state === 'running') { setTimeout(tick, 2000); return; }
          if (row.state === 'failed') { reject(new Error(row.error || 'failed')); return; }
          resolve(row);
        }).catch(reject);
      };
      tick();
    });
  }

  /* The agent returns a PATH and a summary, per the shared charter — never the
   * JSON inline. Its own object is the thing to read; this pulls the first
   * JSON object out of the reply if the agent inlined one anyway, and returns
   * null rather than guessing when it did not. */
  function readObjective(row) {
    var text = String(row && row.output || '');
    var i = text.indexOf('{');
    while (i !== -1) {
      for (var j = text.length; j > i; j--) {
        if (text.charAt(j - 1) !== '}') { continue; }
        try {
          var got = JSON.parse(text.slice(i, j));
          if (got && typeof got === 'object' && 'objective' in got) { return got; }
        } catch (e) { /* not this span */ }
      }
      i = text.indexOf('{', i + 1);
    }
    return null;
  }

  function renameZone() {
    var L = lib();
    if (!L) return;
    var name = window.prompt('Rename "' + L.active + '" to:', L.active);
    if (name === null || String(name).trim() === L.active) return;
    if (!L.renameZone(L.active, name)) {
      window.alert('That name is empty or already taken.');
      return;
    }
    render();
  }

  /* Deleting a pile MOVES its cards, and the confirmation says where they go —
   * the alternative reads as "delete these cards", which is not what happens
   * and not what anyone wants to find out afterwards. */
  function dropZone() {
    var L = lib();
    if (!L) return;
    var zones = L.zones;
    if (zones.length < 2) {
      window.alert('This is the only pile — a library needs somewhere to put '
        + 'the next card. Make another first.');
      return;
    }
    var here = L.active;
    var n = L.entries.filter(function (e) { return e.zone === here; }).length;
    var to = zones.filter(function (z) { return z.name !== here; })[0].name;
    if (!window.confirm('Delete the pile "' + here + '"?'
        + (n ? '\n\nIts ' + n + ' card' + (n === 1 ? '' : 's') + ' move to "'
               + to + '" — nothing is thrown away.' : ''))) return;
    L.removeZone(here);
    render();
  }

  function move(name, zoneName) {
    if (!zoneName || !lib()) return;
    lib().move(name, zoneName);
    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }

  // `refresh` is for the one page that can CHANGE the library while you are on
  // it. The other two only ever read it.
  document.addEventListener('mouseover', onCardOver, { passive: true });
  document.addEventListener('mouseout', onCardOut, { passive: true });
  /* REPOSITION ON SCROLL, DO NOT HIDE. Hiding raced with the thing that opens
   * it: a programmatic hover scrolls the link into view first, and that scroll
   * event landed AFTER the mouseover, so the popup opened and was closed again
   * in the same breath. Measured in the browser suite, where `hover()` left the
   * element present and hidden every time while a hand-dispatched mouseover
   * worked. Following the anchor is also what a reader expects. */
  document.addEventListener('scroll', function () {
    if (!popEl || popEl.hidden) return;
    if (popAnchor && popAnchor.isConnected) placePop(popAnchor);
    else { popEl.hidden = true; popAnchor = null; }
  }, { capture: true, passive: true });

  return {
    refresh: render,
    cardLink: cardLink,
    cardHref: cardHref,
    libraryCount: libraryCount,
    libraryNames: libraryNames,
    cardImageUrl: cardImageUrl,
    mount: mount,
    toggle: toggleDrawer,
    drop: drop,
    zone: zone,
    newZone: newZone,
    consider: consider,
    treat: treat,
    // Exposed for the browser suite: the confirm-and-fall-through path
    // is the half of `treat` worth driving, and the rest is prompts.
    __objectiveFor: objectiveFor,
    renameZone: renameZone,
    dropZone: dropZone,
    move: move,
    open: openCard,
    clear: clearAll,
    get isOpen() { return open; },
  };
})();
