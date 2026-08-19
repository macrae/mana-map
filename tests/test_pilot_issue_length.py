"""`issue-length`: the measure the cuts are argued against.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.
"""

import json

from manamap.pilot import issue_length as il


def test_folded_case_files_cost_words_but_no_scroll():
    """The whole reason the report gives two numbers. Judge's Desk was 21% of the
    issue's words and 2.4% of its scroll because every case is collapsed — read
    only the first number and you go and cut the appendix, which is the one
    department that costs the reader nothing and holds all the proof."""
    markup = (
        '<section class="dept" id="the-kill">one two three</section>'
        '<section class="dept" id="judges-desk">visible'
        '<details class="dossier"><summary>tap</summary>'
        'a b c d e f g h</details></section>')
    rows = {sid: (il.words(f), il.visible_words(f)) for sid, f in il.sections(markup)}
    assert rows["the-kill"] == (3, 3)
    # "visible" + the summary's "tap" survive; the eight folded words do not.
    assert rows["judges-desk"] == (10, 2)


def test_an_open_details_is_counted_as_visible():
    """`open` means it is on screen, so it costs scroll like anything else."""
    frag = '<details open><summary>s</summary>a b c</details>'
    assert il.visible_words(frag) == il.words(frag) == 4


def test_the_report_is_read_only_and_needs_no_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(il, "MANUALS_DIR", tmp_path)
    (tmp_path / "x.html").write_text(
        '<section class="dept" id="cover">a b</section>')
    doc = il.measure("x")
    assert doc["totals"] == {"words": 2, "visible_words": 2, "folded_words": 0,
                             "bytes": 46}
    assert "screens" not in doc["totals"]      # --rendered is opt-in
    assert json.dumps(doc)                     # serialisable for --json
