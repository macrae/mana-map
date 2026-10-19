"""`viz/fleet.html` — the Fleet, the seventh surface.

One committed artifact (`data/fleet/analytics.json`) drawn seven ways: the
table, the bracket racks, the rating leaderboard, floor against DTI, the
interval bars, the curves, the grades. The page computes nothing.

Static assertions only; the behaviour lives in the browser suite. What can rot
here is the nav wiring, a cache-bust out of step, a script order the shell
needs, and the one rule the page exists under — no arithmetic over figures.
"""

import re

import pytest

from manamap.config import DATA_DIR

pytestmark = pytest.mark.regression

VIZ = DATA_DIR.parent / "viz"
PAGE = VIZ / "fleet.html"
SCRIPT = VIZ / "js" / "fleet-view.js"
SHELL = VIZ / "js" / "shell.js"
API = VIZ / "js" / "api.js"


def test_the_page_and_its_script_exist():
    assert PAGE.is_file() and SCRIPT.is_file()


def test_the_page_busts_all_of_its_own_assets_at_one_version():
    html = PAGE.read_text()
    busts = re.findall(r"\?v=(\d+)", html)
    assert len(set(busts)) == 1, busts
    assert len(busts) == 6, "one stylesheet and five scripts"


def test_the_scripts_load_in_the_order_the_shell_requires():
    html = PAGE.read_text()
    order = re.findall(r'<script src="js/([a-z-]+)\.js', html)
    assert order == ["api", "page-state", "session", "shell", "fleet-view"]


def test_fleet_is_in_the_shared_nav_and_the_surface_is_recognised():
    js = SHELL.read_text()
    assert "href: 'fleet.html'" in js and "id: 'fleet'" in js
    assert "if (file === 'fleet.html') return 'fleet';" in js


def test_no_other_surface_hand_links_the_page():
    """The strip is the only nav; a hand link is a second nav to rot."""
    for other in VIZ.glob("*.html"):
        if other.name in ("fleet.html", "workbench.html"):
            continue
        assert 'href="fleet.html"' not in other.read_text(), other.name


def test_the_page_reads_the_formatters_from_api_and_keeps_no_copy():
    js = SCRIPT.read_text()
    assert "window.Fmt" in js
    assert "function esc(" not in js and "function getJSON(" not in js
    assert "window.Fmt = {" in API.read_text()


def test_the_page_never_adds_or_averages_a_figure():
    """The one rule: sort and draw, never compute. No `+=` on a figure, no
    `/ rows.length`, no Math.* over a value — the totals row only counts games
    and the artifact's own counts."""
    js = SCRIPT.read_text()
    assert "Math.round(cell.value / 100)" in js, "cents to dollars is the only arithmetic on a figure"
    body = js.replace("Math.round(cell.value / 100)", "")
    assert not re.search(r"\.value\s*[-+*/]\s*[a-z]", body.replace("cell.value / m", "")), \
        "a figure is used in arithmetic beyond cents→dollars and the bar's percent"
    assert "reduce(" not in body.replace("DTI_GROUPS.reduce", "")
    assert "/ rows.length" not in body and "/ slugs.length" not in body


def test_the_rating_is_read_per_deck_and_labelled_coaching():
    js = SCRIPT.read_text()
    assert "'/rating.json'" in js
    assert "tier === '★'" in js, "a file without the ★ tier is not a rating"
    assert "★ COACHING" in js and "fleet-no-rating" in js


def test_no_d3_and_no_mana_map_on_this_page():
    srcs = re.findall(r'<script src="([^"]+)"', PAGE.read_text())
    assert not any("d3" in x or "mana-map.js" in x for x in srcs), srcs
