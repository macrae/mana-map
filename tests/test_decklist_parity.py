"""Two decklist parsers, one contract.

`src/manamap/pilot/fetch_deck.py:parse_decklist` runs in the CLI; `viz/js/decklist.js`
runs in the browser so a pasted Moxfield export can light up as a graph without a
backend. Two hand-maintained implementations of one grammar drift — this repo recently
deleted a duplicate k-NN that had diverged for years behind a comment claiming the two
had been consolidated. So the parity is a test, not a promise.

Three deliberate choices, each of which is the point of the file:

- **The expected files are hand-authored.** Generating them from Python would make
  Python the oracle, and both parsers would then agree with Python's bugs.
- **The contract is a projection**, `{name, quantity, is_commander}`.
  Python also resolves printings against Scryfall and tracks `foil`; the viz needs none
  of it, so the JS side strips the annotation and discards it. The printing regex is
  where the one real hazard lives, and the safest way not to reimplement a hazard is not
  to reimplement the feature.
- **The browser side runs on a bare fixture page.** There is no `package.json` in this
  repo, so the only JS runtime is a browser — but booting the map would fetch megabytes
  to exercise a regex.

The hazard, guarded by `printings.txt`: Python's `_PRINTING_RE` is `$`-anchored, so
`*F*` and `*CMDR*` must come off the line *before* it runs. Reverse those two steps and
every foil line silently keeps "(2X2) 117" inside the card name.
"""

import json
from pathlib import Path

import pytest
from conftest_viz import BOOT_TIMEOUT_MS  # noqa: F401

from manamap.pilot.fetch_deck import parse_decklist

FIXTURES = Path(__file__).parent / "fixtures" / "decklists"
CONTRACT = ("name", "quantity", "is_commander")
CASES = sorted(p.stem for p in FIXTURES.glob("*.txt"))


def project(entries):
    return [{k: e[k] for k in CONTRACT} for e in entries]


def expected(case):
    return json.loads((FIXTURES / f"{case}.expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_python_matches_the_hand_authored_expectation(case):
    text = (FIXTURES / f"{case}.txt").read_text(encoding="utf-8")
    assert project(parse_decklist(text)) == expected(case)


@pytest.mark.browser
@pytest.mark.parametrize("case", CASES)
def test_javascript_matches_the_same_expectation(browser, viz_server, case):
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{viz_server}/tests/fixtures/parser_page.html")
    page.wait_for_function("() => !!window.Decklist", timeout=BOOT_TIMEOUT_MS)
    text = (FIXTURES / f"{case}.txt").read_text(encoding="utf-8")
    got = page.evaluate("t => Decklist.parse(t)", text)
    page.close()
    assert errors == []
    assert got == expected(case)


@pytest.mark.browser
def test_the_printing_hazard_is_actually_covered(browser, viz_server):
    """Prove the fixture would catch the bug rather than trusting that it would.

    Strips the markers in the WRONG order — printing first, then `*F*` — and asserts the
    result disagrees with the expectation. A guard nobody has seen fail is a guess.
    """
    page = browser.new_page()
    page.goto(f"{viz_server}/tests/fixtures/parser_page.html")
    page.wait_for_function("() => !!window.Decklist", timeout=BOOT_TIMEOUT_MS)
    wrong = page.evaluate("""() => {
        const line = '1 Lightning Bolt (2X2) 117 *F*';
        // Wrong order: printing regex first, while the line still ends with *F*.
        const printing = /\\s+\\(([A-Z0-9]{2,6})\\)\\s+([\\w-]+)$/;
        let name = line.replace(/^\\S+\\s+/, '');
        const m = name.match(printing);
        return m ? name.slice(0, m.index).trim() : name.replace(/\\s*\\*F\\*$/i, '').trim();
    }""")
    page.close()
    assert wrong == "Lightning Bolt (2X2) 117", (
        "the wrong order should leave the printing inside the name — if it does not, "
        "printings.txt is no longer exercising the hazard"
    )
    assert wrong != expected("printings")[1]["name"]
