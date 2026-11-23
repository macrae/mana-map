"""The test selector (2026-11-22): a change runs the tests that can see it, and nothing
it cannot place is ever narrowed. A missed edge is not a red test — it is a stale PASS —
so the rules that widen the selection are tested as hard as the ones that narrow it."""
import coverage

from manamap import test_impact as ti

MAP = {"files": {
    "src/manamap/pilot/bracket.py": ["tests/test_pilot_bracket.py", "tests/test_pilot_build_deck.py"],
    "src/manamap/pilot/deck_edit.py": ["tests/test_pilot_deck_edit.py"],
}}


def test_a_mapped_change_selects_exactly_the_tests_that_executed_it():
    got = ti.affected_tests(["src/manamap/pilot/bracket.py"], measured=MAP)
    assert got == ["tests/test_pilot_bracket.py", "tests/test_pilot_build_deck.py"]
    both = ti.affected_tests(["src/manamap/pilot/bracket.py", "src/manamap/pilot/deck_edit.py"], measured=MAP)
    assert "tests/test_pilot_deck_edit.py" in both and len(both) == 3


def test_what_cannot_be_narrowed_runs_the_whole_tier():
    """config.py's constants run at import, outside every test's measured lines; a shared
    helper or a non-Python file under the package can reach anything."""
    for path in ("src/manamap/config.py", "tests/conftest.py", "tests/core_contracts.py",
                 "src/manamap/pilot/some_data.json", "tools/docs_index_sizes.py", "Makefile"):
        assert ti.affected_tests([path], measured=MAP) is None, path
    assert ti.affected_tests(["src/manamap/pilot/bracket.py", "src/manamap/config.py"],
                             measured=MAP) is None, "one unplaceable path widens the whole run"


def test_a_changed_test_selects_itself_and_its_importers():
    got = ti.affected_tests(["tests/test_pilot_engine_graph.py"], measured=MAP)
    assert "tests/test_pilot_engine_graph.py" in got
    # test_pilot_engine_targets imports test_pilot_engine_graph's fixture.
    assert "tests/test_pilot_engine_targets.py" in got


def test_a_module_the_map_never_saw_falls_back_to_the_import_graph():
    got = ti.affected_tests(["src/manamap/pilot/bracket.py"], measured={"files": {}})
    assert "tests/test_pilot_bracket.py" in got, "the static graph must still find its direct tests"


def test_a_module_path_string_is_an_edge(tmp_path):
    """`registry.py` loads every subcommand by its dotted name; dropping string edges
    would make a subcommand's tests invisible to a change in its module."""
    f = tmp_path / "reg.py"
    f.write_text('STEPS = [("bracket-check", "manamap.pilot.bracket", "help")]\n'
                 'import importlib\n')
    mods, _ = ti._refs(f, "manamap.pilot")
    assert "manamap.pilot.bracket" in mods
    reg = ti._graph()["manamap.pilot.registry"]
    assert "manamap.pilot.bracket" in reg


def test_a_test_that_drives_the_cli_depends_on_the_registry():
    deps = ti._test_map()["test_pilot_check_in.py"]
    assert "manamap.pilot.registry" in deps


def test_the_shared_harness_does_not_hand_its_edges_to_every_test():
    """`from conftest import requires_corpus` used to make a test inherit ~200 modules."""
    deps = ti._test_map()["test_mechanical_tags.py"]
    assert "manamap.pilot.try_swap" not in deps


def test_the_map_is_built_from_per_test_contexts(tmp_path):
    basename = tmp_path / ".coverage"
    data = coverage.CoverageData(basename=str(basename))
    src = str(ti.SRC / "manamap" / "pilot" / "bracket.py")
    data.set_context("tests/test_pilot_bracket.py::test_floor|run")
    data.add_lines({src: [10, 11]})
    data.set_context("")                               # import-time lines belong to no test
    data.add_lines({str(ti.SRC / "manamap" / "config.py"): [1]})
    data.write()
    doc = ti.build_map(basename, out=tmp_path / "map.json")
    assert doc["files"] == {"src/manamap/pilot/bracket.py": ["tests/test_pilot_bracket.py"]}
    assert (tmp_path / "map.json").exists() and doc["built_at_commit"]
