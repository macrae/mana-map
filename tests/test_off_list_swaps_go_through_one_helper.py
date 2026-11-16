"""No test hand-rolls "put the card back if the pilot cut it" (2026-11-16).

`None if "<card>" in have else try_swap.apply_swaps(…)` runs on the tracked list while
the deck carries the card, and needs the Scryfall dump the day the pilot cuts it — which
CI's push job does not have. Craterhoof left Meren in V22 (2026-11-13) and the push job
stayed red three days; Plaguecrafter did the same on 2026-11-09. `conftest.with_card`
skips where there is no dump. This holds the conditional out of `tests/`. Proved by
re-introducing the bug: put the old line back in any test and this fails naming it.
"""
import re
from pathlib import Path

# the conditional, written either way round: `… in have else try_swap.apply_swaps` or
# `if "<card>" not in have:` followed by an apply_swaps on the next lines
HAND_ROLLED = re.compile(r"""in\s+\w+\s+else\s+try_swap\.apply_swaps|not\s+in\s+have\s*:""")


def test_no_test_hand_rolls_an_off_list_swap():
    here = Path(__file__)
    offenders = []
    checked = 0
    for path in sorted(here.parent.glob("*.py")):
        if path == here:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        if "apply_swaps" not in text:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if HAND_ROLLED.search(line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert checked >= 100, "the scan read almost nothing"
    assert not offenders, ("put a cut card back with `conftest.with_card(slug, card, out)`, "
                           "which skips where CI has no Scryfall dump:\n  "
                           + "\n  ".join(offenders))
