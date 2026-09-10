"""`sim-progress` must look where `simulate` actually wrote.

A branch is a seat: `simulate zur-enchantress@drain-v2` is valid and its logs
land under `branches/drain-v2/sim/logs`, because `forge._out_dir` scopes a branch
run structurally so its record cannot overwrite the champion's. `sim-progress`
built `DECKS_DIR / slug` from the RAW argument, so it looked in a literal
`data/decks/zur-enchantress@drain-v2/` — a directory that has never existed — and
reported "no simulation logs" while the run it was asked about was two hours into
producing them.

The second half is quieter and worse: the "ours" hint was the CLI argument, but
Forge names the seat `mm-zur-enchantress-drain-v2` (`@` becomes `-`), so the hint
matched nothing and the report's `<-` marker fell to whichever token sorted
first — a progress view that points at an opponent.
"""

import types

import pytest

from manamap.sim import forge, progress


def test_a_branch_seat_resolves_to_the_branch_directory(tmp_path, monkeypatch):
    """Drives `progress.main`. With the old `DECKS_DIR / slug` it asks for
    `zur-enchantress@drain-v2` and this fails."""
    # The reader hands the WHOLE seat token to `forge.seat_home` (2026-09-10);
    # it no longer splits it itself, because the writer does not either.
    asked = []

    def fake_seat_home(slug):
        asked.append(slug)
        return tmp_path

    monkeypatch.setattr(progress.forge, "seat_home", fake_seat_home)
    args = types.SimpleNamespace(slug="zur-enchantress@drain-v2", run=None, all=False)
    with pytest.raises(SystemExit):        # no logs in tmp_path; resolution is the point
        progress.main(args)
    assert asked == ["zur-enchantress@drain-v2"], asked
    assert forge.split_seat("zur-enchantress@drain-v2") == ("zur-enchantress", "drain-v2")


def test_a_plain_slug_still_resolves_to_the_deck(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(progress.forge, "seat_home",
                        lambda slug: (asked.append(slug), tmp_path)[1])
    args = types.SimpleNamespace(slug="heliod", run=None, all=False)
    with pytest.raises(SystemExit):
        progress.main(args)
    assert asked == ["heliod"], asked


def test_it_looks_under_both_sim_and_experiments(tmp_path, monkeypatch):
    """`experiment` writes elsewhere than `simulate`; both were searched before
    and both must still be."""
    monkeypatch.setattr(progress.forge, "seat_home", lambda slug: tmp_path)
    (tmp_path / "experiments" / "logs" / "a-run").mkdir(parents=True)
    (tmp_path / "experiments" / "logs" / "a-run" / "a-part-00.log").write_text("")
    args = types.SimpleNamespace(slug="heliod", run=None, all=False)
    progress.main(args)          # finds the experiments tree; must not raise


@pytest.mark.parametrize("slug,expected", [
    ("zur-enchantress@drain-v2", "zur-enchantress-drain-v2"),
    ("heliod", "heliod"),
])
def test_the_ours_hint_is_the_forge_seat_token_not_the_cli_argument(slug, expected):
    """Forge writes `Ai(1)-mm-zur-enchantress-drain-v2`. A hint of
    `zur-enchantress@drain-v2` matches no seat in the log."""
    assert forge.deck_meta_name(slug) == expected


def _code_of(fn):
    """Source with comment lines stripped — the first version of this test
    asserted against a string that appeared only in a COMMENT explaining the bug,
    which is a test reading its own documentation."""
    import inspect

    return "\n".join(l for l in inspect.getsource(fn).splitlines()
                     if not l.lstrip().startswith("#"))


def test_the_report_is_handed_the_token_form():
    src = _code_of(progress.main)
    assert "forge.deck_meta_name(slug)" in src, (
        "main passes the raw slug as ours_hint — the marker will point at an opponent")
    assert "DECKS_DIR / slug" not in src, (
        "main still builds a path from the raw argument")


def test_reader_and_writer_agree_on_where_a_run_lives():
    """The property that broke: `forge._out_dir` decides where a run is WRITTEN
    and `progress.main` decides where it is READ. Two spellings of the same
    question is how they drifted."""
    # `seat_home` became the ONE resolution on 2026-09-09 (an opponent can be the
    # subject, so the writer stopped calling `deck_dir` directly). The reader
    # must call the same function, not the two calls it is built from.
    assert "seat_home" in _code_of(progress.main)
    assert "seat_home" in _code_of(forge._out_dir)
    assert "deck_dir(" not in _code_of(progress.main), (
        "the reader resolves through deck_dir and cannot see an opponent-subject run")
