"""The patched log formatter: its fingerprint, its three states, its tag, its stamp.

Each test is proven by the bug it guards: a fingerprint that hashed the repo instead of
the jar (the card-override module's original sin), a tag that stayed empty under the
patched jar (two runs of one configuration writing one path), a stamp that a plain run
would carry (every earlier record reddened), and a `lines: 0` stamp (a run CLAIMING the
patch whose log shows it never ran).
"""
import json
import zipfile

import pytest

from manamap.sim import forge, telemetry as tl, validate_sim


@pytest.fixture
def jars(tmp_path, monkeypatch):
    """A pristine jar and a manifest registering one patched build."""
    home = tmp_path / "forge"
    home.mkdir()
    pristine = home / "forge-gui-desktop-9.9.9-jar-with-dependencies.jar"
    with zipfile.ZipFile(pristine, "w") as z:
        z.writestr(tl.CLASS_ENTRY, b"pristine bytes")
        z.writestr("other.class", b"x")
    patched = tl.PATCH_DIR  # noqa: F841 — the real dir is never touched below
    pdir = tmp_path / "forge_patches"
    pdir.mkdir()
    (pdir / "GameLogFormatter.java").write_text("// patched")
    man = {"forge": {"version": "9.9.9"}, "pristine_sha": tl._sha(b"pristine bytes"),
           "source_sha": tl._sha(b"// patched"), "patched_shas": [tl._sha(b"patched bytes")]}
    (pdir / "manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(tl, "PATCH_DIR", pdir)
    monkeypatch.setattr(tl, "SOURCE", pdir / "GameLogFormatter.java")
    monkeypatch.setattr(tl, "MANIFEST", pdir / "manifest.json")
    monkeypatch.setattr(forge, "FORGE_HOME", home)
    monkeypatch.setattr(tl, "pristine_jar", lambda home=None: pristine)
    return home, pristine


def _write_patched(home, blob):
    out = home / "forge-gui-desktop-9.9.9-mm-telemetry.jar"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(tl.CLASS_ENTRY, blob)
    return out


def test_no_patched_jar_is_a_plain_run_not_a_refusal(jars):
    home, pristine = jars
    assert tl.installed(home) is None
    jar, fp = tl.jar_for_run(home)
    assert jar == pristine and fp is None


def test_a_registered_build_is_the_fingerprint_and_the_jar_for_the_run(jars):
    home, pristine = jars
    out = _write_patched(home, b"patched bytes")
    fp = tl.installed(home)
    assert fp and fp["sha"] == tl._sha(b"patched bytes") and fp["jar"] == out.name
    jar, fp2 = tl.jar_for_run(home)
    assert jar == out and fp2 == fp
    # --plain-jar is always available, and says so by stamping nothing
    assert tl.jar_for_run(home, plain=True) == (pristine, None)


def test_a_copy_that_was_never_patched_reads_as_plain(jars):
    home, _ = jars
    _write_patched(home, b"pristine bytes")
    assert tl.installed(home) is None


def test_an_unregistered_class_is_the_only_refusal(jars):
    """The bug: a jar somebody rebuilt on another JDK, or with another patch, would
    have written a log of a shape nothing on disk describes, under a tag that
    claimed a registered build."""
    home, _ = jars
    _write_patched(home, b"somebody else's bytes")
    with pytest.raises(tl.EngineMismatch):
        tl.installed(home)
    with pytest.raises(tl.EngineMismatch):
        tl.jar_for_run(home)


def test_the_fingerprint_reads_the_jar_not_the_repo(jars):
    """Editing the tracked source must not move the fingerprint of a jar that was
    not rebuilt — the card-override module hashed the working tree once and stamped
    it as provenance."""
    home, _ = jars
    _write_patched(home, b"patched bytes")
    before = tl.installed(home)["sha"]
    tl.SOURCE.write_text("// patched, then edited")
    assert tl.installed(home)["sha"] == before


def test_the_tag_is_empty_at_the_default_and_carries_eight_hex_otherwise():
    assert forge.telemetry_tag(None) == "" and forge.telemetry_tag("") == ""
    assert forge.telemetry_tag("fe64c50348a1") == "-tlfe64c503"


def test_the_run_id_carries_the_formatter_and_old_ids_still_resolve(tmp_path, monkeypatch):
    from manamap import config
    data = tmp_path / "data"
    for s in ("mine", "rival"):
        (data / "decks" / s).mkdir(parents=True)
        (data / "decks" / s / "decklist.txt").write_text("1 Sol Ring\n1 Radagast of Rhosgobel *CMDR*\n")
    monkeypatch.setattr(config, "DECKS_DIR", data / "decks")
    monkeypatch.setattr("manamap.config.DECKS_DIR", data / "decks")
    plain = forge.run_id("mine", ["rival"], 20, seed=7)
    patched = forge.run_id("mine", ["rival"], 20, seed=7, telemetry="fe64c50348a1")
    assert plain.endswith("-s7") and patched == plain + "-tlfe64c503", \
        "two runs of one configuration under two jars are two paths"
    # and through the path `run` actually takes, where the pod tag sits before it
    for_plain = forge.run_id_for("mine", ["rival"], 20, 7, None, None, None)
    for_patched = forge.run_id_for("mine", ["rival"], 20, 7, None, None, None, telemetry="fe64c50348a1")
    assert for_patched == for_plain + "-tlfe64c503"


def test_a_plain_record_carries_no_stamp_and_a_stamp_must_be_comparable():
    assert validate_sim._telemetry_errors({}) == []
    assert validate_sim._telemetry_errors({"telemetry": None}) == []
    ok = {"telemetry": {"sha": "fe64c50348a1", "class": tl.CLASS_ENTRY, "lines": 87}}
    assert validate_sim._telemetry_errors(ok) == []
    assert validate_sim._telemetry_errors({"telemetry": {"sha": "zz", "class": "c", "lines": 1}})
    assert validate_sim._telemetry_errors({"telemetry": {"sha": "fe64c50348a1"}})


def test_a_patched_stamp_over_a_log_with_no_new_lines_is_refused():
    """A record that claims the patch while its logs carry two transitions was not
    played under the patch — `lines` is the receipt, and zero is not one."""
    bad = {"telemetry": {"sha": "fe64c50348a1", "class": tl.CLASS_ENTRY, "lines": 0}}
    assert any("lines is 0" in e for e in validate_sim._telemetry_errors(bad))


def test_the_parser_takes_the_owner_suffix_and_ignores_its_absence():
    from manamap.sim import parse
    shipped = parse._event("Zone Change: Sol Ring (12) was put into Graveyard from Battlefield.", parse._new_game())
    patched = parse._event("Zone Change: Sol Ring (12) was put into Hand from Library. owner Ai(1)-mm-goblin-storm", parse._new_game())
    assert shipped["kind"] == "zone" and "owner" not in shipped
    assert patched["from"] == "Library" and patched["to"] == "Hand" and patched["owner"] == "Ai(1)-mm-goblin-storm"
    assert tl.count_new_lines(["Zone Change: A (1) was put into Hand from Library. owner X\n"
                               "Zone Change: B (2) was put into Graveyard from Battlefield. owner X\n"]) == 1


def test_the_tracked_manifest_registers_the_source_it_ships_with():
    """The real data/forge_patches/: the manifest's source sha is the sha of the .java
    beside it, so an edited source that was never rebuilt is visible."""
    man = tl.manifest()
    assert man and man["source_sha"] == tl.source_sha(), \
        "edit GameLogFormatter.java -> forge-telemetry --build -> commit the manifest"
    assert man["patched_shas"] and all(len(s) == 12 for s in man["patched_shas"])
