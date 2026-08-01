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
 * WHAT SESSION OWNS: the focus (one card, one answer) and the tray (the cards you kept).
 * It also owns the *question* "what graph am I holding" — but see below.
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
  const tray = [];
  const listeners = [];

  function emit(what) {
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

  // ── Tray: the cards you kept, and the thing you export ──────────────────

  function inTray(row) { return tray.indexOf(row) !== -1; }

  function toggleTray(row) {
    const at = tray.indexOf(row);
    if (at === -1) tray.push(row); else tray.splice(at, 1);
    emit('tray');
    return at === -1;
  }

  function clearTray() { tray.length = 0; emit('tray'); }

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
    // tray
    tray: {
      get list() { return tray.slice(); },
      has: inTray,
      toggle: toggleTray,
      clear: clearTray,
      get size() { return tray.length; },
    },
  };
})();
