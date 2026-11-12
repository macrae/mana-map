"""No test splits `decklist.txt` lines by hand (2026-11-12).

When Meren's art was pinned, every line became `1 Animate Dead (SOC) 207` and five
test files that split lines on the first space stopped matching names; the gate failed
twenty-odd tests picking cards the deck already ran. `conftest.decklist_names` reads a
list through the production parser. This holds the hand split out of `tests/`. Proved
by re-introducing the bug: put the old line back in any test and this fails naming it.
"""
import re
from pathlib import Path

HAND_SPLIT = re.compile(r"""\.split\(\s*["'] ["']\s*,\s*1\s*\)\s*\[\s*1\s*\]""")


def test_no_test_splits_a_decklist_line_by_hand():
    here = Path(__file__)
    offenders = []
    checked = 0
    for path in sorted(here.parent.glob("*.py")):
        if path == here:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            if HAND_SPLIT.search(line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert checked >= 100, "the scan read almost nothing"
    assert not offenders, ("read the list with `conftest.decklist_names(slug)`, which "
                           "parses printings and foil marks:\n  " + "\n  ".join(offenders))
