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
  const library = [];
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
  const SCHEMA = 1;

  let cards = null;               // {nameOf, rowOf, fingerprint}
  let lastRestore = null;         // what happened, for a surface that wants to say

  function save() {
    if (!cards || typeof localStorage === 'undefined') return;
    try {
      const names = library.map(cards.nameOf).filter(Boolean);
      localStorage.setItem(STORE_KEY, JSON.stringify({
        v: SCHEMA,
        corpus: cards.fingerprint ? cards.fingerprint() : null,
        cards: names,
      }));
    } catch (e) {
      // A full or disabled localStorage must not break the session. Losing the
      // saved copy is survivable; throwing out of a click handler is not.
      console.warn('[session] could not save the library', e);
    }
  }

  /* Register the card index and restore whatever was saved.
   *
   * Returns the restore report rather than emitting only, because the caller is
   * the surface that can SHOW it — and a library that silently comes back two
   * cards short is indistinguishable from one that came back whole. */
  function useCards(api) {
    cards = api;
    let raw = null;
    try {
      raw = typeof localStorage === 'undefined' ? null : localStorage.getItem(STORE_KEY);
    } catch (e) { raw = null; }
    if (!raw) return (lastRestore = { restored: 0, missing: [], schema: null });

    let doc;
    try { doc = JSON.parse(raw); } catch (e) { doc = null; }
    // An unknown schema is not upgraded and not guessed at. It is left on disk
    // and ignored, so a newer build's data survives an older build reading it.
    if (!doc || doc.v !== SCHEMA || !Array.isArray(doc.cards)) {
      return (lastRestore = { restored: 0, missing: [], schema: doc && doc.v });
    }

    const missing = [];
    library.length = 0;
    for (const name of doc.cards) {
      const row = cards.rowOf(name);
      // The range check the deleted version did not have. `rowOf` answers -1 for
      // a name this corpus does not carry, and a card that left the corpus is a
      // fact worth saying rather than a silent shortfall.
      if (typeof row !== 'number' || row < 0) { missing.push(name); continue; }
      if (library.indexOf(row) === -1) library.push(row);
    }
    lastRestore = {
      restored: library.length,
      missing: missing,
      // Informational: names are stable, so a corpus change does not invalidate
      // the save. It explains a shortfall rather than causing one.
      corpusChanged: !!(doc.corpus && cards.fingerprint &&
                        doc.corpus !== cards.fingerprint()),
    };
    if (library.length) emit('library');
    return lastRestore;
  }

  // ── The library: the cards you kept, and the thing you export ──────────

  function inLibrary(row) { return library.indexOf(row) !== -1; }

  /* SAVE BEFORE EMIT, and the order is load-bearing.
   *
   * The shell shows the library count and reads it straight from
   * `localStorage` — which is what lets it live on two pages that never load
   * Session. Emitting first meant it read the PREVIOUS save: four cards in,
   * three on the strip, permanently one behind. It looked right at a glance
   * because the number was plausible and only wrong by one, and it is invisible
   * unless you compare the strip against the store.
   *
   * So: persist, then announce. Any listener that reads the stored form is then
   * reading the state the event is telling it about. */
  function toggleLibrary(row) {
    const at = library.indexOf(row);
    if (at === -1) library.push(row); else library.splice(at, 1);
    save();
    emit('library');
    return at === -1;
  }

  function clearLibrary() { library.length = 0; save(); emit('library'); }

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
      get list() { return library.slice(); },
      has: inLibrary,
      toggle: toggleLibrary,
      clear: clearLibrary,
      get size() { return library.length; },
    },
    useCards: useCards,
    get restoreReport() { return lastRestore; },
  };
})();
