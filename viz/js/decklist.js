/* Decklist parsing, browser side.
 *
 * A second implementation of `src/manamap/pilot/fetch_deck.py:parse_decklist`, which is
 * a thing this codebase has learned to be suspicious of: it recently deleted a duplicate
 * k-NN that had quietly diverged for years behind a comment claiming the two had been
 * consolidated. So the parity is not a promise, it is a test —
 * `tests/test_decklist_parity.py` runs both against the same hand-authored fixtures in
 * `tests/fixtures/decklists/`.
 *
 * **The contract is a projection.** Only `{name, quantity, is_commander, is_sideboard}`
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

  function parse(text) {
    const entries = [];
    let section = 'deck';

    for (const raw of String(text).split('\n')) {
      let line = raw.trim();
      // A leading `//` is a comment; an inline one is a double-faced card name
      // ("Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"), which is why this
      // tests the start of the line rather than searching it.
      if (!line || line.startsWith('#') || line.startsWith('//')) continue;

      const lowered = line.toLowerCase().replace(/:+$/, '');
      if (COMMANDER.has(lowered)) { section = 'commander'; continue; }
      if (MAIN.has(lowered)) { section = 'deck'; continue; }
      if (SIDEBOARD.has(lowered)) { section = 'sideboard'; continue; }

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
        is_sideboard: section === 'sideboard',
      });
    }
    return entries;
  }

  return { parse: parse };
})();
