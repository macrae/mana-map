"""The install/verify boundary between the repo and the Forge engine.

Every test here drives `forge_pilot` against a SYNTHETIC zip in `tmp_path` rather than
the real install, so the suite can prove the guard without a four-minute reinstall and
without depending on whether this machine happens to have overrides applied right now.
"""

import zipfile

import pytest

from manamap.sim import forge_pilot as fp


# --------------------------------------------------------------------------- fixtures

#: A card script shaped like the ones this actually overrides — a targeting line with a
#: `ValidTgts$` clause naming a creature.
PUMP = ("Name:Test Pump\n"
        "ManaCost:R\n"
        "Types:Instant\n"
        "A:SP$ Pump | ValidTgts$ Creature | NumAtt$ +3 | SpellDescription$ Pump it.\n"
        "Oracle:Target creature gets +3/+0 until end of turn.\n")

#: The shape that must be REFUSED. `forge.ai.ability.CopyPermanentAi` never reads
#: `AITgts$` (2.0.14 bytecode), and the corpus never pairs them.
COPY = ("Name:Test Copy\n"
        "ManaCost:1 R\n"
        "Types:Instant\n"
        "A:SP$ CopyPermanent | ValidTgts$ Creature | SpellDescription$ Copy it.\n"
        "Oracle:Create a token that's a copy of target creature.\n")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """A synthetic Forge tree: a pristine zip, an empty override dir, and the module
    pointed at both. Returns a small helper object."""
    cardsfolder = tmp_path / "res" / "cardsfolder" / "cardsfolder.zip"
    cardsfolder.parent.mkdir(parents=True)
    with zipfile.ZipFile(cardsfolder, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("t/test_pump.txt", PUMP)
        z.writestr("t/test_copy.txt", COPY)
        z.writestr("f/filler.txt", "Name:Filler\nTypes:Land\nOracle:\n")
    overrides = tmp_path / "overrides" / "cards"
    overrides.mkdir(parents=True)
    monkeypatch.setattr(fp, "CARDSFOLDER", cardsfolder)
    monkeypatch.setattr(fp, "PRISTINE", cardsfolder.with_suffix(".zip.orig"))
    monkeypatch.setattr(fp, "OVERRIDE_DIR", overrides)

    class Engine:
        zip_path = cardsfolder
        override_dir = overrides

        @staticmethod
        def entry(name):
            with zipfile.ZipFile(cardsfolder) as z:
                return z.read(name).decode()

    return Engine()


# ------------------------------------------------------------------- absent is absent

def test_no_overrides_reads_as_none_not_as_an_empty_fingerprint(engine):
    """ABSENT MEANS ABSENT. A checkout with no overrides must be
    byte-indistinguishable from the world before this module existed, or every
    historical record acquires a key it never had."""
    assert fp.declared() is None
    assert fp.installed() is None
    agrees, d, i = fp.verify()
    assert agrees and d is None and i is None
    fp.require_agreement()          # must not raise


# ------------------------------------------------------------- the engine is the truth

def test_installed_reads_the_engine_not_the_repo(engine):
    """THE DEFECT THIS MODULE EXISTS FOR.

    `card_overrides()` shipped hashing the repo working tree, so restoring the pristine
    zip left every run stamping a fingerprint it did not earn. Write an override to the
    repo WITHOUT installing it: `declared` must see it and `installed` must not.

    Re-introducing the bug is making `installed()` hash `_override_files()`.
    """
    written, _ = fp.generate_overrides(["test_pump"])
    assert written == ["test_pump"]

    assert fp.declared()["n"] == 1, "the repo declares it"
    # The engine still carries the SHIPPED script, whose bytes differ.
    assert fp.installed()["sha"] != fp.declared()["sha"], (
        "installed() must read the engine; if it hashes the repo these are equal and "
        "the guard is inert")
    agrees, _, _ = fp.verify()
    assert agrees is False
    with pytest.raises(fp.EngineMismatch):
        fp.require_agreement()


def test_install_makes_the_engine_agree(engine):
    fp.generate_overrides(["test_pump"])
    fp.install()
    agrees, d, i = fp.verify()
    assert agrees, (d, i)
    assert d["sha"] == i["sha"]
    fp.require_agreement()
    assert "AITgts$ Ally.YouCtrl" in engine.entry("t/test_pump.txt")


def test_install_is_idempotent(engine):
    """Installing twice equals installing once — the property that makes this safe to
    call from a fleet sweep.

    NOTE this test is WEAK on its own and says so. `install` replaces whole zip entries
    rather than editing them, so overlaying the current zip produces identical bytes and
    this passes either way — I proved that by re-introducing the bug and watching it
    stay green. The real hazard is the next test.
    """
    fp.generate_overrides(["test_pump"])
    fp.install()
    once = fp.installed()["sha"]
    body = engine.entry("t/test_pump.txt")
    fp.install()
    fp.install()
    assert fp.installed()["sha"] == once
    assert engine.entry("t/test_pump.txt") == body
    assert body.count("AITgts$") == 1


def test_a_withdrawn_override_reverts_because_install_rebuilds_from_pristine(engine):
    """THE HAZARD THAT MAKES `PRISTINE` LOAD-BEARING, and the one my first attempt at
    this missed.

    An install that overlaid the CURRENT zip would leave a withdrawn override in the
    engine forever: nothing restores the shipped script, so a card the repo no longer
    declares keeps its hint and every later run is steered by a rule that has been
    deleted — while `verify()` reports agreement, because it only compares the scripts
    the repo still declares.

    That is how the four `CopyPermanent` cards would have survived their own withdrawal
    on 2026-09-28.

    Re-introducing the bug is opening `CARDSFOLDER` instead of `PRISTINE` in `install`.
    """
    with zipfile.ZipFile(engine.zip_path, "a") as z:
        z.writestr("t/test_other.txt", PUMP.replace("Test Pump", "Test Other"))

    fp.generate_overrides(["test_pump", "test_other"])
    fp.install()
    assert "AITgts$" in engine.entry("t/test_pump.txt")
    assert "AITgts$" in engine.entry("t/test_other.txt")

    # The pilot withdraws one override — exactly what happened to the four copy spells.
    (engine.override_dir / "t" / "test_other.txt").unlink()
    fp.install()

    assert "AITgts$" in engine.entry("t/test_pump.txt"), "the kept override survives"
    assert "AITgts$" not in engine.entry("t/test_other.txt"), (
        "a WITHDRAWN override is still in the engine — install is overlaying the "
        "current zip instead of rebuilding from pristine, so a deleted rule keeps "
        "steering every run while verify() reports agreement")
    assert fp.verify()[0], "and the engine still agrees with what the repo declares"


def test_uninstall_restores_the_shipped_script(engine):
    fp.generate_overrides(["test_pump"])
    fp.install()
    assert "AITgts$" in engine.entry("t/test_pump.txt")
    fp.uninstall()
    assert "AITgts$" not in engine.entry("t/test_pump.txt")
    assert engine.entry("t/test_pump.txt") == PUMP


def test_install_refuses_to_bake_an_override_into_the_pristine_baseline(engine):
    """A first install with no `.orig` copies the current zip as the baseline — which
    would make the distortion permanent and invisible if the current zip is already
    overridden. Refuse instead."""
    fp.generate_overrides(["test_pump"])
    fp.install()
    fp.PRISTINE.unlink()            # simulate losing the baseline
    with pytest.raises(fp.EngineMismatch, match="permanent"):
        fp.install()


def test_the_zip_survives_an_install_whole(engine):
    """An install rewrites every entry. Losing one would be silent and catastrophic."""
    with zipfile.ZipFile(engine.zip_path) as z:
        before = sorted(z.namelist())
        raw_before = sum(i.file_size for i in z.infolist())
    fp.generate_overrides(["test_pump"])
    fp.install()
    with zipfile.ZipFile(engine.zip_path) as z:
        assert sorted(z.namelist()) == before
        # Only the one patched entry may have grown.
        assert sum(i.file_size for i in z.infolist()) > raw_before


# ------------------------------------------------------- what cannot be steered at all

def test_generate_refuses_a_copypermanent_card(engine):
    """`CopyPermanentAi` NEVER reads `AITgts$` — verified in the 2.0.14 bytecode, where
    the class references `AILogic` and the sixteen logic names it implements and
    `AITgts` is not among its strings. The corpus agrees: `AITgts$` sits beside
    `Destroy` 22 times, `ChangeZone` 12, `Pump` 7, `CopyPermanent` zero.

    Writing one anyway is a flag the engine cannot act on — the failure
    `tests/test_metric_hygiene.py` exists for. It must be named, not written.
    """
    written, unsteerable = fp.generate_overrides(["test_pump", "test_copy"])
    assert written == ["test_pump"]
    assert unsteerable == ["test_copy"]
    assert not (engine.override_dir / "t" / "test_copy.txt").exists(), (
        "an unsteerable override was written — it implies a capability we do not have")


def test_generate_derives_from_the_shipped_script_and_changes_only_the_aim(engine):
    """An override is DERIVED so it cannot silently diverge from the card. `ValidTgts$`
    is never touched, so nothing LEGAL changes — a human pilot's options are identical
    and only the AI's choice narrows."""
    fp.generate_overrides(["test_pump"])
    got = (engine.override_dir / "t" / "test_pump.txt").read_text()
    assert "AITgts$ Ally.YouCtrl" in got
    assert "ValidTgts$ Creature" in got, "the legal target set must be untouched"
    # Every other line survives verbatim, in order.
    shipped = [ln for ln in PUMP.splitlines() if "ValidTgts$" not in ln]
    assert [ln for ln in got.splitlines() if "ValidTgts$" not in ln] == shipped
    assert got.count("AITgts$") == 1


def test_generate_refuses_a_card_with_no_single_targeting_line(engine):
    """Refusing beats guessing which line meant it."""
    with zipfile.ZipFile(engine.zip_path, "a") as z:
        z.writestr("t/two_targets.txt",
                   "Name:Two\nTypes:Instant\n"
                   "A:SP$ Pump | ValidTgts$ Creature | NumAtt$ +1\n"
                   "A:SP$ Pump | ValidTgts$ Creature | NumAtt$ +2\n")
    with pytest.raises(ValueError, match="exactly one"):
        fp.generate_overrides(["two_targets"])


def test_the_overridden_set_names_no_copypermanent_card():
    """Drives the REAL corpus: the shipped set must contain nothing unsteerable, or the
    directory implies a steering capability the engine does not have."""
    if not (fp.PRISTINE.is_file() or fp.CARDSFOLDER.is_file()):
        pytest.skip("Forge is not installed on this machine")
    assert len(fp.OVERRIDDEN) >= 11
    for stem in ("molten_duplication", "heat_shimmer", "electroduplicate",
                 "kindle_the_inner_flame"):
        assert stem not in fp.OVERRIDDEN, (
            f"{stem} is SP$ CopyPermanent and cannot be steered; it was removed on "
            f"2026-09-28 after the bytecode check and must not come back")
