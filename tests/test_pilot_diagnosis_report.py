"""diagnosis-report: the artifact rendered as something a human reads.

The renderer must never become a second place that derives figures — that is how
two views of the same artifact drift. It reads the committed JSON and nothing
else, so these tests are about ordering, escaping and honesty, not arithmetic.
"""

import json

from manamap.pilot import diagnosis_report as dr

from conftest import requires_deck


def _doc(**over):
    base = {
        "slug": "test", "as_of_decklist_sha256": "abc123def456789",
        "verdict": "It grinds; it does not close.",
        "axes": [], "engine": {}, "lean_into": [], "cut_candidates": [],
        "add_candidates": [], "open_questions": [], "gaps": [],
    }
    base.update(over)
    return base


def _axis(name, verdict="adequate", value=5, reading="fine"):
    return {"axis": name, "verdict": verdict,
            "measured": {"value": value, "unit": "copies"}, "reading": reading}


def _cells(row):
    """Cell count of a markdown row, ignoring escaped pipes inside prose."""
    return row.replace("\\|", "").count("|")


def test_a_minimal_diagnosis_renders():
    text = dr.render(_doc())
    assert text.startswith("# test — deck diagnosis")
    assert "It grinds; it does not close." in text


def test_a_fail_verdict_is_announced_at_the_top():
    """A reader must not reach the recommendations before the caveat."""
    text = dr.render(_doc(skeptic={"verdict": "fail", "iterations": 3,
                                   "findings": [{"status": "over-claimed",
                                                 "where": "axes[0]", "note": "x"}]}))
    head = text.split("## The verdict")[0]
    assert "`fail`" in head
    assert "documents what could not be grounded" in head


def test_skeptic_findings_render_last_and_open_ones_first():
    """Burying an adversary's open findings under what they qualify would make
    this a sales document."""
    doc = _doc(skeptic={"verdict": "fail", "iterations": 3, "findings": [
        {"status": "supported", "where": "axes[1]", "note": "SUPPORTED-NOTE"},
        {"status": "over-claimed", "where": "axes[0]", "note": "OPEN-NOTE"},
    ]})
    text = dr.render(doc)
    assert text.index("## Skeptic findings") > text.index("## The axes")
    assert text.index("OPEN-NOTE") < text.index("SUPPORTED-NOTE")
    assert "### Open" in text and "### Confirmed" in text


def test_axes_render_in_reading_order_not_artifact_order():
    doc = _doc(axes=[_axis("power"), _axis("mana-base"), _axis("consistency")])
    text = dr.render(doc)
    assert text.index("**mana-base**") < text.index("**consistency**") < text.index("**power**")


def test_an_axis_not_in_the_reading_order_still_renders():
    doc = _doc(axes=[_axis("brand-new-axis")])
    assert "**brand-new-axis**" in dr.render(doc)


def test_weaknesses_are_called_out_by_name():
    doc = _doc(axes=[_axis("threat-density", "weakness"), _axis("power", "adequate")])
    text = dr.render(doc)
    assert "Called out as a weakness or liability:" in text
    assert "`threat-density`" in text.split("Called out")[1].split("\n")[0]


def test_pipes_in_prose_do_not_break_the_table():
    """A reading containing a pipe would otherwise split a markdown cell."""
    doc = _doc(axes=[_axis("ramp", reading="rocks | dorks | rituals")])
    row = [ln for ln in dr.render(doc).splitlines() if "**ramp**" in ln][0]
    assert _cells(row) == 5, row
    assert "\\|" in row


def test_newlines_in_prose_do_not_break_the_table():
    doc = _doc(axes=[_axis("ramp", reading="line one\nline two")])
    row = [ln for ln in dr.render(doc).splitlines() if "**ramp**" in ln][0]
    assert "line one line two" in row


def test_cuts_render_hardest_last():
    doc = _doc(cut_candidates=[
        {"card": "Painful One", "difficulty": "painful", "why": "w",
         "cost_of_cutting": "c"},
        {"card": "Easy One", "difficulty": "easy", "why": "w", "cost_of_cutting": "c"},
    ])
    text = dr.render(doc)
    assert text.index("Easy One") < text.index("Painful One")


def test_a_cut_touching_a_verified_stack_says_so():
    doc = _doc(cut_candidates=[
        {"card": "Load Bearing", "difficulty": "painful", "why": "w",
         "cost_of_cutting": "c", "orphans_stack": ["005", "007"]}])
    assert "**Touches verified stack(s):** 005, 007" in dr.render(doc)


def test_citation_ids_are_shown_but_not_the_quotes():
    """The quote lives in the JSON; the report names what was cited."""
    doc = _doc(lean_into=[{"what": "the engine", "why": "it grinds", "citations": [
        {"rule": "strategy:deckbuilding.ratios", "quote": "a very long verbatim span"}]}])
    text = dr.render(doc)
    assert "cites strategy:deckbuilding.ratios" in text
    assert "a very long verbatim span" not in text


def test_a_single_point_of_failure_with_no_closer_says_so():
    doc = _doc(engine={"declared": "d", "components": [],
                       "single_points_of_failure": [
                           {"component": "the kill", "why": "one card", "closers": []}]})
    assert "*No closer found.*" in dr.render(doc)


def test_open_questions_show_where_they_route():
    doc = _doc(open_questions=[{"question": "does it loop?",
                                "settled_by": "resolve-stack",
                                "why_it_matters": "the bracket"}])
    text = dr.render(doc)
    assert "**`resolve-stack`**" in text and "does it loop?" in text


def test_render_is_deterministic():
    doc = _doc(axes=[_axis("ramp"), _axis("power")], gaps=["a", "b"])
    assert dr.render(doc) == dr.render(json.loads(json.dumps(doc)))


@requires_deck
def test_renders_the_real_yawgmoth_diagnosis():
    from manamap.pilot.common import deck_dir, load_json
    doc = load_json(deck_dir("yawgmoth-swarm") / "diagnosis.json", default=None)
    if doc is None:
        return  # deck present but not yet diagnosed
    text = dr.render(doc)
    assert "# yawgmoth-swarm — deck diagnosis" in text
    # Every table row must have the same cell count as its header.
    for block in text.split("\n\n"):
        rows = [r for r in block.splitlines() if r.startswith("|")]
        if len(rows) > 2:
            width = _cells(rows[0])
            assert all(_cells(r) == width for r in rows), block[:200]
