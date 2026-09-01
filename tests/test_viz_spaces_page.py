"""`viz/spaces.html` — the embedding-space reference page.

Static assertions only. The page is prose plus five canvases over a committed
artifact; what can rot is the artifact going missing, a cache-bust drifting out
of step, or a number here disagreeing with the code it describes.
"""

import json
import re

import pytest

from manamap.config import DATA_DIR

PAGE = DATA_DIR.parent / "viz" / "spaces.html"
SCRIPT = DATA_DIR.parent / "viz" / "js" / "spaces-view.js"
PROJECTIONS = DATA_DIR / "eval" / "space_projections.json"


def test_the_page_and_its_script_exist():
    assert PAGE.exists() and SCRIPT.exists()


def test_the_projections_artifact_is_present_and_covers_every_drawn_space():
    """The page renders a COMMITTED artifact, the same contract every other viz
    surface follows — a browser cannot run PaCMAP."""
    if not PROJECTIONS.exists():
        pytest.skip("run `manamap project-spaces`")
    data = json.loads(PROJECTIONS.read_text())
    script = SCRIPT.read_text()
    drawn = re.findall(r"\['([^']+ \([^)]+\))',", script)
    assert drawn, "no spaces listed in ORDER"
    for label in drawn:
        assert label in data["spaces"], f"{label} is drawn but absent from the artifact"


def test_every_projected_space_has_the_same_card_count():
    """One sample across all five is what makes the comparison fair; a panel with
    a different sample would be a different picture of a different corpus."""
    if not PROJECTIONS.exists():
        pytest.skip("run `manamap project-spaces`")
    data = json.loads(PROJECTIONS.read_text())
    counts = {k: len(v["points"]) for k, v in data["spaces"].items()}
    assert len(set(counts.values())) == 1, counts
    assert set(counts.values()) == {data["cards"]}
    for fact in data["facts"].values():
        assert len(fact) == data["cards"]


def test_the_page_busts_all_of_its_own_assets():
    """A cached page pins the script `?v=`, which pins every data fetch."""
    versions = set(re.findall(r"\?v=(\d+)", PAGE.read_text()))
    assert len(versions) == 1, f"spaces.html busts inconsistently: {versions}"


def test_the_default_it_documents_is_the_default_in_code():
    """The page tells a reader `function` is the default and nothing is cut over.
    If that ever stops being true the page becomes confidently wrong, which is
    worse than absent."""
    from manamap import spaces

    assert spaces.DEFAULT == "function"
    assert "Nothing has been cut over" in PAGE.read_text()


def test_it_names_the_command_that_reproduces_its_numbers():
    """The figures are a SNAPSHOT — there is no metrics artifact to read — so the
    page has to say how to regenerate them rather than presenting them as live."""
    text = PAGE.read_text()
    assert "manamap eval-embeddings" in text
    assert "snapshot" in text.lower()


def test_it_carries_the_interval_rule():
    """This repo's rule: a comparison carries the interval on the DIFFERENCE, and
    two overlapping marginal intervals imply nothing."""
    assert "interval on the difference" in PAGE.read_text().lower()


def test_the_reference_page_is_in_the_shared_nav_not_a_bespoke_link():
    """`shell.js` puts one strip on every surface. A page linked only from its
    neighbour is reachable from one place and invisible from the Atlas, which is
    where the toggle it explains actually lives."""
    shell = (DATA_DIR.parent / "viz" / "js" / "shell.js").read_text()
    assert "spaces.html" in shell, "the reference page is not in SURFACES"
    assert "appendix: true" in shell, (
        "it should render as an appendix, not as a third place to work")
    assert "if (file === 'spaces.html') return 'spaces';" in shell, (
        "currentSurface cannot recognise the page, so its own nav entry would "
        "render as a link back to itself")


def test_no_surface_carries_a_second_nav_to_the_reference_page():
    """Two navs two lines apart is one of them being ignored."""
    viz = DATA_DIR.parent / "viz"
    for page in ("workbench.html", "spaces.html", "index.html"):
        text = (viz / page).read_text()
        assert 'href="spaces.html"' not in text, (
            f"{page} hand-links the reference page; shell.js already does")
