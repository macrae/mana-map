/* session.js — what you are holding, independent of how it is being drawn.
 *
 * The problem this solves is countable. Before it there were EIGHT "set of cards"
 * containers (`selectedCards`, `browseSet`, `orientation.rows`, `Discovery.tray`,
 * `Force.nodes`, `deckState.seeds`, `DeckMap.active.main`, `Drill.indices`) and SEVEN
 * different answers to "which card is selected right now" — `selectedCards[topCardIndex]`,
 * `browseSet.indices[pos]`, `orientation.anchor`, `Discovery.current`, `Force.pinned`,
 * `Force.hovered`, `deckState.commander`. Nothing reconciled them. `discovery.js` has a
 * comment enumerating six of them and apologising for adding a seventh.
 *
 * That is why the modes felt like different products: a click meant a different thing in
 * each because each was writing to a different place.
 *
 * WHAT SESSION OWNS: the focus (one card, one answer) and the LIBRARY (the cards you
 * kept). It also owns the *question* "what graph am I holding" — but see below.
 *
 * It is a LIBRARY, not a basket or a tray. In Magic your library IS your deck, and the
 * cards you gather while brewing are the deck you are gathering. "Basket" is shopping
 * vocabulary for an act that is not shopping. Distinct from THE BENCH, which is every
 * deck you own — one word per thing.
 *
 * WHAT SESSION DOES NOT OWN, DELIBERATELY: the graph's storage. `force.js` holds nodes as
 * live simulation bodies — d3 mutates x/y/vx/vy on every tick and owns their identity —
 * so lifting membership into a second array would create exactly the duplicate-model
 * problem this file exists to remove. Instead Session *delegates* graph reads to whatever
 * is registered as the provider, and every consumer asks Session. One interface, one
 * storage, no copy to drift.
 *
 * The thing that motivated this: Explore's `orientation` was a SNAPSHOT of `Force.rows()`
 * copied at mode-entry that never tracked the graph afterwards. So the atlas showed a
 * photograph of your walk rather than the walk. Reading through Session makes it live.
 *
 * Out of scope on purpose: `deckState`, `DeckMap.active` and `Drill.indices` are different
 * concepts (a deck under construction, a published decklist, a re-layout subset), and
 * folding them in would make this unshippable without making anything better.
 */
window.Session = (function () {
  'use strict';

  let provider = null;          // the graph's storage — force.js registers itself
  let focusRow = -1;
  /* THE LIBRARY IS A LIST OF NAMES; A ROW IS A RESOLUTION OF A NAME.
   *
   * Entries are `{name, row}` and `row` may be -1. That one shape settles a bug
   * class this file had shipped three of at once, all of them "the count and the
   * contents are two different questions":
   *
   *   - `save()` could not run without the corpus, because it derived names from
   *     rows. A page that never loads `viz_index` therefore could not REMOVE a
   *     card, silently — and two of three surfaces are such pages.
   *   - A name this corpus cannot resolve was dropped from memory and LEFT in
   *     the store, so the strip counted it forever and nothing could show it.
   *   - Two rows sharing a name (there are 38 such names in 34,890 rows —
   *     `Savage Lands`, `Everythingamajig`, …) persisted as two entries and
   *     restored as one, permanently.
   *
   * Names are canonical, so the count is `entries.length` on every surface and
   * the grid renders the same list. They agree by construction rather than by
   * two readers being careful.
   */
  const entries = [];      // [{name, row, zone}] — row -1 = not in this corpus
  const zones = [];        // ordered zone names; zones[0] is where new cards land
  let activeZone = null;   // the pile you are looking at and keeping into
  const listeners = [];

  function emit(what) {
    // The shell shows the library count and lives on every surface, including
    // two that load neither Session nor the atlas. It is notified here rather
    // than subscribing, because it must not depend on Session existing — the
    // guard is what lets one strip serve three pages.
    if (what === 'library' && window.Shell && Shell.refresh) {
      try { Shell.refresh(); } catch (e) { /* the strip is never load-bearing */ }
    }
    for (const fn of listeners) {
      try { fn(what); } catch (e) { console.error('[session] listener failed', e); }
    }
  }

  /* Register the graph's storage. Kept as an interface rather than an import so the
   * layouts stay swappable: a world-coordinate layout and a force layout are two views of
   * the same membership, and neither should be Session's dependency. */
  function useGraph(p) { provider = p; }

  // ── The graph, read through one interface ───────────────────────────────

  function rows() { return provider ? provider.rows() : []; }
  function links() { return provider ? provider.links() : []; }
  function has(row) { return provider ? provider.has(row) : false; }
  function size() { return rows().length; }

  /* Grow from a card by a relation. The single write path — `MM.relate` and every panel
   * button funnel here, so "growing must never be able to delete" is enforced in one
   * place rather than re-argued at each call site. */
  function grow(row, rel) {
    if (!provider) return false;
    provider.grow(row, rel || 'similar');
    setFocus(row);
    emit('graph');
    return true;
  }

  // ── Focus: ONE answer to "which card am I looking at" ───────────────────

  function setFocus(row) {
    if (typeof row !== 'number' || row < 0 || row === focusRow) return;
    focusRow = row;
    emit('focus');
  }

  // ── The commander ───────────────────────────────────────────────────────
  //
  // One card, and everything reads it from here: the gold ring on the graph, the colour
  // identity that decides what you may legally play, and the exported brief. The builder
  // kept its own `deckState.commander` and the graph kept `commanderRow`, so designating
  // one meant writing two places and hoping.

  let commanderRow = -1;

  function setCommander(row) {
    commanderRow = (typeof row === 'number' && row >= 0) ? row : -1;
    emit('commander');
  }

  /* ── PERSISTENCE ────────────────────────────────────────────────────────
   *
   * The library survives navigation between surfaces, which is the whole point
   * of it (PRD §7.1: "the connective tissue for the whole flow").
   *
   * IT STORES CARD NAMES. NEVER ROW INDICES. This is not a preference — the
   * previous attempt at this, `localStorage['manamap-deck']`, was DELETED for
   * storing raw positional row indices with no schema version. A Scryfall
   * refresh reorders `cards.csv`, every row index shifts, and a saved deck
   * silently reinterprets as different cards: no error, no warning, a plausible
   * wrong answer. It also did no range check, so entering Build before the
   * projection landed threw. Names are the stable key, the full "A // B" form is
   * already the graph-key convention, and a name that no longer resolves is
   * REPORTED rather than dropped.
   *
   * The resolver is INJECTED, exactly as the graph provider is, and for a
   * concrete reason: `session.js` loads before `mana-map.js`, so reaching for
   * `MM.*` here would run inside that module's IIFE before `window.MM` exists —
   * the boot-order failure that once took out four files at once. Discovery
   * owns the name index, so Discovery hands it over when it has one, and
   * restore happens at that moment rather than at load.
   */
  const STORE_KEY = 'manamap-library';
  const SCHEMA = 2;
  //: The pile a v1 library lands in, and the one a fresh library starts with.
  //: "Unsorted" rather than "Library": the whole thing is the library, and a
  //: pile named after its container tells the reader nothing about what is in
  //: it. It also says the honest thing about cards kept before zones existed —
  //: they were never sorted, because there was nowhere to sort them to.
  const DEFAULT_ZONE = 'Unsorted';

  let cards = null;               // {nameOf, rowOf, fingerprint}
  let lastRestore = null;         // what happened, for a surface that wants to say

  function storedCorpus() {
    const doc = readStore();
    return doc && !doc.bad ? (doc.corpus || null) : null;
  }

  function save() {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        v: SCHEMA,
        // A WRITE FROM A PAGE WITH NO CORPUS MUST NOT ERASE THE FINGERPRINT.
        // Taking a card out of the library on the workbench used to stamp
        // `corpus: null` over a real value, because that page has no index to
        // fingerprint — losing the one thing that can later EXPLAIN why a name
        // stopped resolving. Absent knowledge is not knowledge of absence.
        corpus: cards && cards.fingerprint ? cards.fingerprint() : storedCorpus(),
        active: activeZone,
        zones: zones.map(function (z) {
          return {
            name: z,
            cards: entries.filter(function (e) { return e.zone === z; })
                          .map(function (e) { return e.name; }),
          };
        }),
      }));
    } catch (e) {
      // A full or disabled localStorage must not break the session. Losing the
      // saved copy is survivable; throwing out of a click handler is not.
      console.warn('[session] could not save the library', e);
    }
  }

  /* One name, one entry, whatever case it arrived in.
   *
   * The corpus carries 38 names on more than one row, so two DIFFERENT rows can
   * be the same card to anyone building a deck. Keying on the name is what makes
   * "keep this twice" a no-op instead of an entry the store holds and the panel
   * cannot show. */
  function indexOfName(name) {
    const want = String(name).trim().toLowerCase();
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].name.toLowerCase() === want) return i;
    }
    return -1;
  }

  function resolve(name) {
    if (!cards || !cards.rowOf) return -1;
    const row = cards.rowOf(name);
    return typeof row === 'number' && row >= 0 ? row : -1;
  }

  /* Accept a row or a name, because both are real ways a card arrives.
   *
   * A row comes from the atlas (a click, the Keep button); a name comes from a
   * decklist, a brief, or a page with no corpus loaded at all. Resolving here
   * rather than at each call site is what lets the drawer work on the workbench,
   * where there are no rows to speak of. */
  function nameOf(rowOrName) {
    if (typeof rowOrName === 'number') {
      return cards && cards.nameOf ? cards.nameOf(rowOrName) : null;
    }
    const s = String(rowOrName == null ? '' : rowOrName).trim();
    return s || null;
  }

  /* Read the stored document. Names only — resolving is a separate question and
   * on two of three surfaces there is nothing to resolve against. */
  function readStore() {
    let raw = null;
    try {
      raw = typeof localStorage === 'undefined' ? null : localStorage.getItem(STORE_KEY);
    } catch (e) { raw = null; }
    if (!raw) return null;
    let doc;
    try { doc = JSON.parse(raw); } catch (e) { doc = null; }
    if (!doc) return { bad: null };
    // A v1 document is UPGRADED, not ignored. The "unknown schema is left on
    // disk" rule protects a NEWER save from an OLDER build; this is the other
    // direction, where the data is ours and readable and refusing it would
    // silently empty a real library. One pile, every card, nothing dropped.
    if (doc.v === 1 && Array.isArray(doc.cards)) {
      return {
        v: SCHEMA, corpus: doc.corpus || null, active: DEFAULT_ZONE,
        zones: [{ name: DEFAULT_ZONE, cards: doc.cards.slice() }],
        migrated_from: 1,
      };
    }
    if (doc.v !== SCHEMA || !Array.isArray(doc.zones)) return { bad: doc.v };
    return doc;
  }

  /* Load names and their piles, WITHOUT a corpus. This runs at module
   * evaluation, so the library exists on every page that loads this file. */
  function loadNames() {
    const doc = readStore();
    if (!doc || doc.bad !== undefined) {
      if (!zones.length) { zones.push(DEFAULT_ZONE); activeZone = DEFAULT_ZONE; }
      // The bad marker is RETURNED, not swallowed. A caller reporting "schema
      // null" for a document that plainly says `v: 99` cannot tell "there was
      // nothing" from "there was something I refused to guess at", and the
      // second is the whole point of refusing.
      return doc;
    }
    entries.length = 0;
    zones.length = 0;
    for (const z of doc.zones) {
      const name = String((z && z.name) || '').trim();
      if (!name || zones.indexOf(name) !== -1) continue;
      zones.push(name);
      for (const card of (z.cards || [])) {
        // Dedupe across the WHOLE library, not per zone: one card, one pile.
        if (!card || indexOfName(card) !== -1) continue;
        entries.push({ name: String(card), row: -1, zone: name });
      }
    }
    if (!zones.length) zones.push(DEFAULT_ZONE);
    activeZone = zones.indexOf(doc.active) !== -1 ? doc.active : zones[0];
    return doc;
  }

  /* Register the card index and resolve what is already held.
   *
   * Returns the restore report rather than emitting only, because the caller is
   * the surface that can SHOW it — and a library that silently comes back two
   * cards short is indistinguishable from one that came back whole. */
  function useCards(api) {
    cards = api;
    const before = entries.length;
    const doc = loadNames();            // re-read: another tab may have written
    if (!doc) return (lastRestore = { restored: 0, missing: [], schema: null });
    if (doc.bad !== undefined) {
      return (lastRestore = { restored: 0, missing: [], schema: doc.bad });
    }

    const missing = [];
    for (const e of entries) {
      // A NAME THIS CORPUS CANNOT RESOLVE IS KEPT, NOT DROPPED. It used to be
      // discarded from memory and left in the store, which is the worst of
      // both: the strip counted a card no surface could show, forever.
      e.row = resolve(e.name);
      if (e.row < 0) missing.push(e.name);
    }
    lastRestore = {
      restored: entries.filter(function (e) { return e.row >= 0; }).length,
      missing: missing,
      migrated: doc.migrated_from || null,
      corpusChanged: !!(doc.corpus && cards && cards.fingerprint &&
                        doc.corpus !== cards.fingerprint()),
    };
    // RECONCILE THE STORE ONCE — after a migration, or when a duplicate or a
    // malformed zone was dropped on the way in. Skipped otherwise, so an
    // ordinary boot touches nothing.
    if (doc.migrated_from || entries.length !== before) save();
    if (entries.length) emit('library');
    return lastRestore;
  }

  // ── The library: the cards you kept, and the thing you export ──────────

  function inLibrary(rowOrName) {
    if (typeof rowOrName === 'number') {
      if (rowOrName < 0) return false;
      for (const e of entries) if (e.row === rowOrName) return true;
      // A row whose name is held under a different row is still HELD. The corpus
      // has 38 names on more than one row, and "do I have this card" is a
      // question about the card, not about which printing you clicked.
      const n = nameOf(rowOrName);
      return !!n && indexOfName(n) !== -1;
    }
    return indexOfName(rowOrName) !== -1;
  }

  /* SAVE BEFORE EMIT, and the order is load-bearing.
   *
   * The shell shows the library count and can read it straight from
   * `localStorage` — which is what lets it live on a page that has not finished
   * booting Session. Emitting first meant it read the PREVIOUS save: four cards
   * in, three on the strip, permanently one behind. It looked right at a glance
   * because the number was plausible and only wrong by one, and it is invisible
   * unless you compare the strip against the store.
   *
   * So: persist, then announce. Any listener that reads the stored form is then
   * reading the state the event is telling it about. */
  function commit() { save(); emit('library'); }

  /* ADD AND REMOVE ARE SEPARATE ACTS, and `toggle` is the one that needs a
   * reason to exist rather than the other way round.
   *
   * For a long time `toggle` was the only writer, so every caller adding a LIST
   * of cards was running "flip each of these" — and a list containing the same
   * card twice, or two names resolving to one row, quietly removed it again.
   * `build.js:resumeDraft` shipped exactly that: it restored a draft's
   * must-includes by toggling, so a repeated name cancelled itself out and the
   * status line then reported the count it had INTENDED to keep. */
  function addToLibrary(rowOrName, zone) {
    const name = nameOf(rowOrName);
    if (!name) return false;
    const at = indexOfName(name);
    if (at !== -1) {
      /* ALREADY HELD — and what that means depends on whether a pile was NAMED.
       *
       * With no zone the caller is saying "keep this", and a card already kept
       * is already kept. With an explicit zone they are saying "put it HERE",
       * and library-wide dedupe made that a silent no-op: resuming a draft
       * whose must-include sat in another pile gathered fewer cards than the
       * brief named, reported the brief's count, and left the deck's own card
       * filed under something else. Measured: a two-card brief gathered one.
       */
      const target = zones.indexOf(zone) !== -1 ? zone : null;
      if (target && entries[at].zone !== target) {
        entries[at].zone = target;
        commit();
        return true;
      }
      return false;
    }
    entries.push({
      name: name,
      row: typeof rowOrName === 'number' ? rowOrName : resolve(name),
      // A card lands in the pile you are LOOKING at. Keeping is a gesture made
      // from somewhere, and "somewhere" is the zone tab that is open.
      zone: zones.indexOf(zone) !== -1 ? zone : activeZone,
    });
    commit();
    return true;
  }

  function removeFromLibrary(rowOrName) {
    const name = nameOf(rowOrName);
    let at = name ? indexOfName(name) : -1;
    if (at === -1 && typeof rowOrName === 'number') {
      at = entries.findIndex(function (e) { return e.row === rowOrName; });
    }
    if (at === -1) return false;
    entries.splice(at, 1);
    commit();
    return true;
  }

  function toggleLibrary(rowOrName) {
    return inLibrary(rowOrName)
      ? (removeFromLibrary(rowOrName), false)
      : addToLibrary(rowOrName);
  }

  /* ── the piles ──────────────────────────────────────────────────────────
   *
   * ONE CARD, ONE ZONE. A pile is the physical metaphor the library already
   * runs on — the cards you are gathering — and a card is in one pile. That
   * keeps every count unambiguous and makes "remove" mean one thing; the cost
   * is that considering a card for two decks means picking one, which is what
   * `move` is for.
   */
  function zoneCounts() {
    return zones.map(function (z) {
      return { name: z,
               count: entries.filter(function (e) { return e.zone === z; }).length };
    });
  }

  function addZone(name) {
    const clean = String(name || '').trim().slice(0, 40);
    if (!clean || zones.indexOf(clean) !== -1) return false;
    zones.push(clean);
    activeZone = clean;
    commit();
    return true;
  }

  function setActiveZone(name) {
    if (zones.indexOf(name) === -1) return false;
    activeZone = name;
    commit();               // the open pile is state worth surviving a reload
    return true;
  }

  function renameZone(from, to) {
    const clean = String(to || '').trim().slice(0, 40);
    const at = zones.indexOf(from);
    if (at === -1 || !clean || zones.indexOf(clean) !== -1) return false;
    zones[at] = clean;
    for (const e of entries) if (e.zone === from) e.zone = clean;
    if (activeZone === from) activeZone = clean;
    commit();
    return true;
  }

  /* Removing a pile MOVES its cards rather than destroying them. Deleting a
   * label is a statement about the label; the cards were kept on purpose and
   * nothing here asked about them. The last zone cannot go — a library with no
   * pile has nowhere to put the next card. */
  function removeZone(name) {
    const at = zones.indexOf(name);
    if (at === -1 || zones.length < 2) return false;
    zones.splice(at, 1);
    const fallback = zones[0];
    for (const e of entries) if (e.zone === name) e.zone = fallback;
    if (activeZone === name) activeZone = fallback;
    commit();
    return true;
  }

  function moveCard(name, zone) {
    const at = indexOfName(name);
    if (at === -1 || zones.indexOf(zone) === -1) return false;
    entries[at].zone = zone;
    commit();
    return true;
  }

  function clearZone(name) {
    const target = zones.indexOf(name) !== -1 ? name : activeZone;
    for (let i = entries.length - 1; i >= 0; i--) {
      if (entries[i].zone === target) entries.splice(i, 1);
    }
    commit();
  }

  function clearLibrary() { entries.length = 0; commit(); }

  loadNames();   // the library exists before anything has resolved it

  /* THE LIBRARY IS CROSS-SURFACE, AND SO ARE TABS. Keeping a card in the Atlas
   * while the Workbench sits open in another tab used to leave that tab's strip
   * and drawer showing the library as it was when the page loaded — quietly
   * wrong, and exactly the "two readings of one library" failure this file was
   * rewritten to end, one window over.
   *
   * `storage` fires only in the OTHER documents, never the writer, so this
   * cannot loop. Rows are re-resolved where there is an index to resolve
   * against, and left alone where there is not. */
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('storage', function (ev) {
      if (ev.key && ev.key !== STORE_KEY) return;
      loadNames();
      if (cards) for (const e of entries) e.row = resolve(e.name);
      emit('library');
    });
  }

  return {
    useGraph: useGraph,
    on: function (fn) { listeners.push(fn); },
    // graph
    rows: rows,
    links: links,
    has: has,
    size: size,
    grow: grow,
    // focus
    get focus() { return focusRow; },
    setFocus: setFocus,
    // commander
    get commander() { return commanderRow; },
    setCommander: setCommander,
    // the library
    library: {
      /* `names` is THE list and THE count — one question, one answer, on every
       * surface. `list` stays the resolved rows, because the graph and the map
       * speak rows and a card this corpus has never heard of has none. The two
       * can differ in length, and that difference is the thing the drawer draws
       * rather than the thing that used to make two counters disagree. */
      get names() { return entries.map(function (e) { return e.name; }); },
      get entries() {
        return entries.map(function (e) {
          return { name: e.name, row: e.row, zone: e.zone };
        });
      },
      /* The ACTIVE pile's names — what a brief exports, because a brief is one
       * deck and the library is now several. `names` stays the whole library,
       * which is what the strip counts. */
      get zoneNames() {
        return entries.filter(function (e) { return e.zone === activeZone; })
                      .map(function (e) { return e.name; });
      },
      get zones() { return zoneCounts(); },
      get active() { return activeZone; },
      setActive: setActiveZone,
      addZone: addZone,
      renameZone: renameZone,
      removeZone: removeZone,
      move: moveCard,
      clearZone: clearZone,
      get list() {
        return entries.filter(function (e) { return e.row >= 0; })
                      .map(function (e) { return e.row; });
      },
      /* IS THERE A CORPUS TO BE ABSENT FROM? Without one, every entry's row is
       * -1 because nothing has resolved it — not because the card is unknown.
       * A drawer that read `row < 0` as "not in this corpus" would libel every
       * card in the library on the two surfaces that never load the index. */
      get resolvable() { return !!(cards && cards.rowOf); },
      has: inLibrary,
      add: addToLibrary,
      remove: removeFromLibrary,
      toggle: toggleLibrary,
      clear: clearLibrary,
      get size() { return entries.length; },
    },
    useCards: useCards,
    get restoreReport() { return lastRestore; },
  };
})();
