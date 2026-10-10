"""`viz/js/api.js`'s FORMAT_SPECS is `pilot/formats.py:FORMATS`, field for field.

The browser needs a format's size, copy limit, commander count and sideboard to say
"60 / 60+", to hide "Set as commander" on a Modern deck and to size Discover's brief
(Area C7, 2026-10-09). It cannot import Python, so the table is mirrored in
JavaScript — and a silent mirror of a config table is how the two sides drift. This
test reads the literal as JSON and holds every format and every field to the Python
source of truth, in both directions: a format added on one side only fails here.
"""

import json
import re
from pathlib import Path

from manamap.pilot import formats

API_JS = Path(__file__).resolve().parents[1] / "viz" / "js" / "api.js"

# `size` is the pilot's count (the commander included), `exact` says whether it
# is a minimum, `copies` is `max_copies` (singleton -> 1, else 4).
FIELDS = {
    "name": lambda s: s.name,
    "size": lambda s: s.deck_size,
    "exact": lambda s: s.exact_size,
    "copies": lambda s: s.max_copies,
    "commanders": lambda s: s.commanders,
    "sideboard": lambda s: s.sideboard_size,
}


def _js_specs():
    src = API_JS.read_text(encoding="utf-8")
    m = re.search(r"window\.FORMAT_SPECS\s*=\s*(\{.*?\n\});", src, flags=re.S)
    assert m, "api.js no longer declares `window.FORMAT_SPECS = {...};`"
    return json.loads(m.group(1))


def test_the_js_table_names_exactly_the_python_formats():
    assert set(_js_specs()) == set(formats.FORMATS)


def test_every_field_of_every_format_agrees_with_python():
    js = _js_specs()
    checked = 0
    for key, spec in formats.FORMATS.items():
        assert set(js[key]) == set(FIELDS), (key, sorted(js[key]))
        for field, read in FIELDS.items():
            assert js[key][field] == read(spec), (
                f"{key}.{field}: api.js says {js[key][field]!r}, formats.py says {read(spec)!r}")
            checked += 1
    assert checked >= len(FIELDS) * 5


def test_the_test_catches_a_drifted_value(monkeypatch):
    """Prove the comparison bites: a JS table claiming Modern is exactly 60 fails."""
    drifted = _js_specs()
    drifted["modern"]["exact"] = True
    monkeypatch.setattr(__import__(__name__), "_js_specs", lambda: drifted)
    try:
        test_every_field_of_every_format_agrees_with_python()
    except AssertionError as exc:
        assert "modern.exact" in str(exc)
    else:
        raise AssertionError("a drifted field passed the comparison")
