"""Forge scenario slices (PRD v2 Step 5): the runner jar, its argv, its output.

The unit half needs no Forge. The `forge`-marked test plays one real two-seat slice
(a few seconds plus the JVM boot) — opt in with `pytest -m forge`.
"""
import json

import pytest

from conftest import requires_deck
from manamap.sim import slice as sl


def test_the_runner_is_not_part_of_the_pod_fingerprint():
    """A class in data/forge_patches/ joins the `-tl<sha8>` every pod run records. The
    runner never executes in a `sim` game, so it lives apart and builds its own jar."""
    from manamap.sim import telemetry

    assert sl.SOURCE.parent.name == "forge_driver"
    assert sl.SOURCE.name not in telemetry.PATCHES
    assert all("ScenarioSlice" not in p["entry"] for p in telemetry.PATCHES.values())


def test_the_argv_puts_the_runner_beside_the_forge_jar_and_names_every_case(tmp_path):
    argv = sl.command(["a.dck", "b.dck"], [("A", 3, tmp_path / "a3"), ("B", 1, tmp_path / "b1")],
                      rounds=2, timeout=60, jar="/x/forge.jar", home=tmp_path)
    cp = argv[argv.index("-cp") + 1]
    assert cp == f"/x/forge.jar:{tmp_path / sl.JAR_NAME}"
    assert argv[argv.index(cp) + 1] == sl.MAIN
    assert argv[argv.index("--decks") + 1] == "a.dck,b.dck"
    assert argv[argv.index("--rounds") + 1] == "2"
    cases = [argv[i + 1] for i, a in enumerate(argv) if a == "--case"]
    assert cases == [f"A:3={tmp_path / 'a3'}", f"B:1={tmp_path / 'b1'}"]


def test_only_marked_lines_are_records_and_forge_noise_is_ignored():
    rec = {"label": "A", "seed": 1, "end": [], "stopped": True}
    text = "Language loaded\nMMSLICE " + json.dumps(rec) + "\nThe card X was not assigned\n"
    assert sl.parse_output(text) == [rec]


def test_a_runner_built_from_another_source_is_rebuilt(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(sl, "source_sha", lambda: "new")
    monkeypatch.setattr(sl, "build", lambda home=None: built.append(home) or tmp_path / "j")
    monkeypatch.setattr(sl, "built_sha", lambda home=None: "old")
    sl.ensure_built(tmp_path)
    assert built == [tmp_path]
    monkeypatch.setattr(sl, "built_sha", lambda home=None: "new")
    sl.ensure_built(tmp_path)
    assert built == [tmp_path], "a current runner is not rebuilt"


@pytest.mark.forge
@requires_deck
def test_one_real_slice_applies_the_board_and_stops_after_one_round():
    from manamap import config

    if not list(config.FORGE_HOME.glob("forge-gui-desktop-*-jar-with-dependencies.jar")):
        pytest.skip("Forge is not installed at FORGE_HOME (docs/simulation.md)")
    state = "\n".join([
        "activeplayer=p0", "activephase=MAIN1", "turn=6",
        "p0life=40", "p0hand=Windfall",
        "p0library=" + ";".join(["Island"] * 15),
        "p0battlefield=Island;Island;Mountain;Plains;Command Tower",
        "p1life=33", "p1hand=Plains",
        "p1library=" + ";".join(["Plains"] * 15),
        "p1battlefield=Plains;Plains;Grizzly Bears",
    ])
    recs = sl.run(["sharknado", "giada-angels"], [("A", 1, state), ("A", 2, state)], rounds=1)
    assert [r["seed"] for r in recs] == [1, 2] and all("error" not in r for r in recs)
    for r in recs:
        assert r["start"][0]["life"] == 40 and r["start"][1]["life"] == 33   # the board applied
        assert r["start_turn"] == 6 and r["stop_turn"] == 8                   # 2 seats, 1 round
        assert r["stopped"] is True and r["ended_at_turn"] == 8
        assert r["log"], "the game log came back"
        assert len(r["end"]) == 2, "every seat reports, lost or not"
