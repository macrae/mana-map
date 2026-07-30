"""The map's camera must survive a re-render.

`render()` hands `Plotly.react` a freshly built layout object, and `react` replaces
layout wholesale. A layout with no explicit axis range therefore resets the viewport to
autorange — which meant filtering and zooming were mutually destructive: every supertype
toggle, colour-by change, search keystroke and deck-panel mutation silently threw the
camera away.

It was invisible because nothing failed. Measured in a browser before the fix: zoom to a
span of 20.5, call `MM.render()`, read 116.6. The same block that reads the live span to
pick region labels was reading a value the next statement destroyed.

These are source assertions, not behavioural ones — there is no JS runtime in the suite,
and the pattern follows `test_viz_deck_lens.py`. They are worth having anyway: the failure
mode is a silent omission, and an omission is exactly what a source check catches.
"""

import re

from manamap.config import VIZ_DIR

MAP_JS = VIZ_DIR / "js" / "mana-map.js"


def _render_body() -> str:
    """The body of render(), from its declaration to the config line."""
    src = MAP_JS.read_text(encoding="utf-8")
    start = src.index("function render()")
    end = src.index("const config = {", start)
    return src[start:end]


def test_render_writes_the_live_range_back_into_the_layout():
    body = _render_body()
    assert "range: keepX" in body, "xaxis range is not preserved across render()"
    assert "range: keepY" in body, "yaxis range is not preserved across render()"


def test_range_capture_is_gated_on_plot_initialised():
    """Two cases must still auto-fit, and both are keyed off `plotInitialized`.

    The first render has no camera to keep. A map switch changes the coordinates
    themselves — `applyProjection` clears the flag for exactly this reason — so
    holding the old range there would frame the wrong part of a different map.
    """
    body = _render_body()
    capture = re.search(r"if \((.*?)\) \{\n\s+const xr = plotEl\._fullLayout", body)
    assert capture, "could not find the range-capture guard in render()"
    assert "plotInitialized" in capture.group(1), (
        "range capture is not gated on plotInitialized — a map switch would keep a "
        "stale camera over new coordinates"
    )


def test_scaleanchor_survives_the_preserved_branch():
    """Both axis branches must keep `scaleanchor`.

    The map is a projection: dropping the constraint on one branch would let Plotly
    scale x and y independently and silently distort every distance on screen.
    """
    body = _render_body()
    xaxis_lines = [ln for ln in body.splitlines() if "xaxis:" in ln or "scaleanchor" in ln]
    joined = " ".join(xaxis_lines)
    assert joined.count("scaleanchor") == 2, (
        f"expected scaleanchor on both xaxis branches, found {joined.count('scaleanchor')}"
    )
