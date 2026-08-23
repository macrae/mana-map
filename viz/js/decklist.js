/* Decklist parsing, browser side.
 *
 * A second implementation of `src/manamap/pilot/fetch_deck.py:parse_decklist`, which is
 * a thing this codebase has learned to be suspicious of: it recently deleted a duplicate
 * k-NN that had quietly diverged for years behind a comment claiming the two had been
 * consolidated. So the parity is not a promise, it is a test —
 * `tests/test_decklist_parity.py` runs both against the same hand-authored fixtures in
 * `tests/fixtures/decklists/`.
 *
 * **The contract is a projection.** Only `{name, quantity, is_commander}`
 * has to match. Python additionally resolves printings against Scryfall and tracks
 * `foil`; the viz has no use for any of it, so this strips the annotation and throws it
 * away. That is deliberate risk reduction rather than laziness — the printing regex is
 * exactly where the one real hazard lives, and the safest way to not reimplement a
 * hazard is to not reimplement the feature.
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

  function parse(text) {
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
        // been — nothing.
        if (fromComment) { section = 'deck'; fromComment = false; }
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
      // There is no sideboard any more, but the MARKER still has to be consumed:
      // a pasted list carrying one would otherwise file every card after it as
      // mainboard. Stop reading — everything below the line is out of the deck.
      if (SIDEBOARD.has(lowered)) break;

      let isCommander = section === 'commander';
      const cmdr = stripSuffix(line, '*CMDR*');
      if (cmdr.found) { isCommander = true; line = cmdr.line; }
      // Foil is stripped and discarded — but it MUST be stripped here, before the
      // printing suffix is removed below, or the `$` anchor never matches.
      line = stripSuffix(line, '*F*').line;

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

      entries.push({
        name: name,
        quantity: quantity,
        is_commander: isCommander,
      });
    }
    return entries;
  }

  return { parse: parse };
})();
