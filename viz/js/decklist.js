/* Decklist parsing, browser side.
 *
 * A second implementation of `src/manamap/pilot/fetch_deck.py:parse_decklist`, which is
 * a thing this codebase has learned to be suspicious of: it recently deleted a duplicate
 * k-NN that had quietly diverged for years behind a comment claiming the two had been
 * consolidated. So the parity is not a promise, it is a test —
 * `tests/test_decklist_parity.py` runs both against the same hand-authored fixtures in
 * `tests/fixtures/decklists/`.
 *
 * **The contract is a projection.** Only `{name, quantity, is_commander, board}`
 * has to match. Python additionally resolves printings against Scryfall and tracks
 * `foil`; the viz has no use for any of it, so this strips the annotation and throws it
 * away. That is deliberate risk reduction rather than laziness — the printing regex is
 * exactly where the one real hazard lives, and the safest way to not reimplement a
 * hazard is to not reimplement the feature.
 *
 * ONE EXCEPTION, opt-in (2026-10-09): `parse(text, {printings: true})` keeps `foil`,
 * `set` and `collector_number`, and `render(entries)` writes them back as
 * `check_in.render_decklist` does — for the deck page's static "Copy for Moxfield",
 * which has no server to run `deck-export`. That path is held to the CLI's bytes on
 * every tracked deck by `tests/test_viz_moxfield.py`, not by the parity fixtures.
 *
 * `board` is `'main'` or `'side'` (2026-10-09). Both parsers used to stop reading at a
 * `Sideboard:` line; now the section switches and every entry says where it sits, so a
 * 60-card list with its fifteen imports whole. A caller that wants the deck filters on
 * `board === 'main'`, as Python's `parse_mainboard` does.
 *
 * The hazard, for the record: Python's `_PRINTING_RE` is anchored to `$`, so `*F*` and
 * `*CMDR*` must be stripped from the end of the line BEFORE it runs. Reverse those two
 * steps and every foil line silently keeps "(2X2) 117" inside the card name. Same order
 * is preserved here for the same reason.
 */
window.Decklist = (function () {
  'use strict';

  // Mirrors COMMANDER/MAIN/SIDEBOARD_SECTION_MARKERS in pilot/common.py. If those grow a
  // member and these do not, an imported deck silently files a whole section as mainboard.
  const COMMANDER = new Set(['commander', 'commanders']);
  const MAIN = new Set(['deck', 'mainboard', 'main']);
  const SIDEBOARD = new Set(['sideboard', 'side', 'maybeboard', 'considering']);

  const PRINTING = /\s+\(([A-Z0-9]{2,6})\)\s+([\w-]+)$/;
  // Mirrors `_FOIL_MARKERS` in pilot/fetch_deck.py (`*E*` is etched).
  const FOIL_MARKERS = ['*F*', '*E*'];

  function stripSuffix(line, marker) {
    const upper = line.toUpperCase();
    if (!upper.endsWith(marker)) return { line: line, found: false };
    return { line: line.slice(0, upper.lastIndexOf(marker)).trim(), found: true };
  }

  // A comment line whose WHOLE text is a section marker, or null. Moxfield and
  // Archidekt write the commander under `// COMMANDER`, and both parsers used to
  // strip `//` lines before ever testing for a marker — so the header was
  // swallowed and the list imported with no commander. The `//` test stays
  // anchored to line start (a DFC name carries ` // ` inline); this notices when
  // a comment IS a marker. Whole text only, never a prefix, so a real note like
  // `// commander is the wincon` stays a note.
  function commentMarker(line) {
    if (!line.startsWith('//') && !line.startsWith('#')) return null;
    const body = line.replace(/^[/#]+/, '').trim().toLowerCase().replace(/:+$/, '');
    return (COMMANDER.has(body) || MAIN.has(body) || SIDEBOARD.has(body)) ? body : null;
  }

  /* `opts.printings` keeps what the default parse throws away — `foil`, and `set`
   * (lower-cased, as Python stores it) plus `collector_number` when the line carries
   * a printing — so `render` below can write the list back the way
   * `check_in.render_decklist` does. OFF by default: the parity contract and every
   * import caller see exactly the entries they always did. The deck page's static
   * "Copy for Moxfield" is the one caller that asks, and
   * `tests/test_viz_moxfield.py` holds parse→render to `deck-export --format
   * moxfield` on every tracked deck. */
  function parse(text, opts) {
    const keep = !!(opts && opts.printings);
    const entries = [];
    let section = 'deck';
    // Whether the current section was entered through a comment header. It is the
    // only thing that gives a blank line meaning, and only inside such a section.
    let fromComment = false;

    for (const raw of String(text).split('\n')) {
      let line = raw.trim();
      if (!line) {
        // A blank line closes a comment-entered section and nothing else. Exports
        // that write `// COMMANDER` do not write a matching `// DECK`; the blank
        // IS the terminator. Everywhere else a blank line stays what it has always
        // been — nothing. EXCEPT a comment-entered sideboard: a sideboard is the
        // last section of every export, so nothing after it is mainboard, and a
        // blank inside it (Moxfield groups by type) must not refile the rest.
        if (fromComment && section !== 'side') { section = 'deck'; fromComment = false; }
        continue;
      }
      const marker = commentMarker(line);
      // A leading `//` is a comment; an inline one is a double-faced card name
      // ("Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"), which is why this
      // tests the start of the line rather than searching it.
      if (marker === null && (line.startsWith('#') || line.startsWith('//'))) continue;

      const lowered = marker !== null ? marker : line.toLowerCase().replace(/:+$/, '');
      if (COMMANDER.has(lowered)) { section = 'commander'; fromComment = marker !== null; continue; }
      if (MAIN.has(lowered)) { section = 'deck'; fromComment = marker !== null; continue; }
      // The sideboard is READ, not skipped: the section switches, a `Deck:` after it
      // returns to the mainboard, and every entry below carries `board: 'side'`.
      if (SIDEBOARD.has(lowered)) { section = 'side'; fromComment = marker !== null; continue; }

      let isCommander = section === 'commander';
      const cmdr = stripSuffix(line, '*CMDR*');
      if (cmdr.found) { isCommander = true; line = cmdr.line; }
      // Foil is stripped (and discarded unless `keep`) — but it MUST be stripped
      // here, before the printing suffix is removed below, or the `$` anchor never
      // matches. Both of Python's `_FOIL_MARKERS`, first match wins, as there.
      let foil = false;
      for (const fm of FOIL_MARKERS) {
        const f = stripSuffix(line, fm);
        if (f.found) { foil = true; line = f.line; break; }
      }

      let quantity = 1;
      let name = line;
      const parts = line.match(/^(\S+)\s+([\s\S]+)$/);
      if (parts) {
        const head = parts[1].toLowerCase().replace(/x+$/, '');
        if (/^\d+$/.test(head)) {
          quantity = parseInt(head, 10);
          name = parts[2].trim();
        }
      }

      const printing = name.match(PRINTING);
      if (printing) name = name.slice(0, name.length - printing[0].length).trim();

      const entry = {
        name: name,
        quantity: quantity,
        is_commander: isCommander,
        board: section === 'side' ? 'side' : 'main',
      };
      if (keep) {
        entry.foil = foil;
        if (printing) {
          entry.set = printing[1].toLowerCase();
          entry.collector_number = printing[2];
        }
      }
      entries.push(entry);
    }
    return entries;
  }

  /* One entry as one line — MIRRORS `pilot/check_in.py:render_line`:
   * `N Name [(SET) CN] [*F*] [*CMDR*]`, in that order and nothing after. */
  function renderLine(e, cmdrMarker) {
    let s = (parseInt(e.quantity, 10) || 1) + ' ' + e.name;
    if (e.set && e.collector_number) {
      s += ' (' + String(e.set).toUpperCase() + ') ' + e.collector_number;
    }
    if (e.foil) s += ' *F*';
    if (cmdrMarker) s += ' *CMDR*';
    return s;
  }

  // Python's `sorted(key=name)`: code-point order, never `localeCompare`, which
  // would put "Æther Vial" somewhere a Python sort does not.
  function byName(a, b) { return a.name < b.name ? -1 : a.name > b.name ? 1 : 0; }

  /* Entries back to text — MIRRORS `pilot/check_in.py:render_decklist`, which is
   * what `manamap pilot deck-export --format moxfield` prints and what Moxfield's
   * import box takes: `Commander:` block and a blank line when there is one,
   * `Deck:` sorted by name, then `Sideboard:` only when there is one. One
   * trailing newline. A stable sort keeps duplicate names in list order, as
   * Python's does. */
  function render(entries) {
    const board = e => e.board || 'main';
    const main = entries.filter(e => board(e) === 'main');
    const side = entries.filter(e => board(e) === 'side').slice().sort(byName);
    const cmds = main.filter(e => e.is_commander);
    const deck = main.filter(e => !e.is_commander).slice().sort(byName);
    const out = [];
    if (cmds.length) {
      out.push('Commander:');
      cmds.forEach(e => out.push(renderLine(e)));
      out.push('');
    }
    out.push('Deck:');
    deck.forEach(e => out.push(renderLine(e)));
    if (side.length) {
      out.push('');
      out.push('Sideboard:');
      side.forEach(e => out.push(renderLine(e, !!e.is_commander)));
    }
    return out.join('\n') + '\n';
  }

  return { parse: parse, render: render, renderLine: renderLine };
})();
