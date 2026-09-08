"""The commander-ability band: a ceiling and a floor instead of a figure.

A deck can DECLARE a commander ability the corpus cannot be swept for --
`model_commander_animate`, `model_commander_attack_tutor` -- and the goldfish
then applies it every turn it can afford to. Forge's AI fired zur's animate in
FIVE of 119 games. Neither instrument is the table, so a declared ability is
reported as a pair.

The bugs this file exists for, each of which shipped once:

1. `BAND_ROWS` reused `candidates.OBJECTIVE_AXES`, which maps the DIAGNOSTIC's
   shape (`output.kill_by_turn`) and not this module's (`combat.kill_by_
   turn_rate`). Every lookup missed and the band rendered with NO ROWS -- which
   a reader takes for "no difference", not "nothing was read".
2. The declaration body sat inside the `elif targets_path.exists()` branch, so a
   caller supplying the document directly -- which is exactly what the floor run
   does -- skipped it. `model_combat` and `model_draw` fell back to False and the
   floor was measured under a DIFFERENT MODEL. It read kill_by_8 0.102 against a
   true floor of 0.219, blaming the ability for a gap that was mostly combat
   being switched off.
3. `declared_drain` and `declared_deaths` were bound only inside that same
   branch while their four siblings were bound outside it. Latent for as long as
   every deck had a targets file; an UnboundLocalError the moment one did not.
"""

import json

import pytest

from manamap.pilot import common, goldfish

from conftest import requires_data, requires_deck

BAND = "commander_ability_band"


def _declaring_decks():
    """Every deck whose targets file declares an ability the band covers."""
    out = []
    for d in sorted(common.DECKS_DIR.glob("*/goldfish_targets.json")):
        try:
            doc = json.loads(d.read_text())
        except (OSError, ValueError):
            continue
        if any(doc.get(k) for k in goldfish._DECLARED_ABILITIES):
            out.append(d.parent.name)
    return out


# ── 1. The document, where one is declared ────────────────────────────────

@requires_data
@requires_deck
def test_a_declared_ability_gets_both_ends_and_the_rows_are_not_empty():
    """Bug 1: an empty `rows` reads as "no difference" and must not be possible."""
    slugs = _declaring_decks()
    if not slugs:
        pytest.skip("no deck declares a commander ability")
    checked = 0
    for slug in slugs:
        doc = goldfish.run(slug, iterations=200, quiet=True)
        band = doc.get(BAND)
        assert band, f"{slug} declares an ability and got no band"
        assert "unavailable" not in band, (
            f"{slug}: the floor run failed -- {band.get('unavailable')}. It is "
            f"swallowed so it cannot break the primary run, which is also how "
            f"an empty band would ship in silence.")
        assert band["rows"], (
            f"{slug}: a band with no rows. Every BAND_ROWS path missed the "
            f"metrics document -- see bug 1 above.")
        for axis, row in band["rows"].items():
            assert axis in goldfish.BAND_ROWS
            assert row["owed_to_the_ability"] == pytest.approx(
                row["ceiling"] - row["floor"], abs=1e-4), (
                f"{slug}/{axis}: `owed` must be the difference it names")
        checked += 1
    assert checked >= 1


@requires_data
@requires_deck
def test_a_deck_declaring_nothing_has_no_band_at_all():
    """Absent means absent. A zero-width band on an ordinary commander is noise
    on every other deck's page."""
    declaring = set(_declaring_decks())
    checked = 0
    for d in sorted(common.DECKS_DIR.glob("*/goldfish_targets.json")):
        slug = d.parent.name
        if slug in declaring:
            continue
        assert BAND not in goldfish.run(slug, iterations=100, quiet=True), (
            f"{slug} declares no commander ability and must carry no band")
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1


# ── 2. The floor runs the SAME MODEL, minus one ability ───────────────────

@requires_data
@requires_deck
def test_supplying_the_decks_own_declaration_changes_nothing():
    """Bug 2, driven through the production entry point.

    `_targets_doc` is the seam the floor run uses. Handed the file's OWN
    contents it must produce the file's own answer -- if it does not, the seam
    is dropping declarations, and the floor is a different model rather than
    the same model with one ability off.

    Re-introduce the bug (move the declaration body back inside the `elif`) and
    this fails on `model_combat` alone.
    """
    slugs = _declaring_decks() or [d.parent.name for d in
                                   sorted(common.DECKS_DIR.glob("*/goldfish_targets.json"))[:1]]
    checked = 0
    for slug in slugs:
        own = json.loads((common.DECKS_DIR / slug / "goldfish_targets.json").read_text())
        a = goldfish.run(slug, iterations=200, quiet=True, _band=False)
        b = goldfish.run(slug, iterations=200, quiet=True, _band=False, _targets_doc=own)
        assert a["metrics"] == b["metrics"], (
            f"{slug}: the same declaration by two routes gave two answers")
        checked += 1
    assert checked >= 1


@requires_data
@requires_deck
def test_stripping_the_ability_moves_the_deck_and_only_downwards():
    """The floor must be a REAL floor: the ability is a benefit, so switching it
    off cannot make the deck faster. A floor above its ceiling means the two
    runs differ by something other than the ability."""
    slugs = _declaring_decks()
    if not slugs:
        pytest.skip("no deck declares a commander ability")
    checked = 0
    for slug in slugs:
        band = goldfish.run(slug, iterations=200, quiet=True)[BAND]
        moved = [a for a, r in band["rows"].items() if r["floor"] != r["ceiling"]]
        assert moved, (
            f"{slug}: switching {band['abilities']} off moved nothing. Either "
            f"the ability is unread by the simulation -- the set-and-unread "
            f"class -- or the strip did not reach it.")
        for axis, row in band["rows"].items():
            assert row["floor"] <= row["ceiling"] + 1e-9, (
                f"{slug}/{axis}: floor {row['floor']} above ceiling "
                f"{row['ceiling']} -- the two runs differ by more than the ability")
        checked += 1
    assert checked >= 1


# ── 3. The bindings that were latent ──────────────────────────────────────

@requires_data
@requires_deck
def test_a_declaration_less_document_does_not_raise():
    """Bug 3. Four `declared_*` names were bound before the branch and two were
    not, so any path that skipped the file read died on the two."""
    slug = sorted(common.DECKS_DIR.glob("*/goldfish_targets.json"))[0].parent.name
    doc = goldfish.run(slug, iterations=100, quiet=True, _band=False, _targets_doc={})
    assert doc["metrics"], "an empty declaration is a valid deck, not a crash"


# ── 4. Cost: the floor run is a SECOND full simulation ────────────────────

@requires_data
@requires_deck
def test_the_looping_callers_do_not_pay_for_a_floor_they_discard():
    """`diagnostic.run_on` is called once per card by a candidate sweep, and
    `benchmark` once per deck by the fleet. Both discard the band and both
    override the model flags, so a band taken there would describe the harness
    rather than the deck — at the price of doubling every run.

    Asserted on the SOURCE because the cost is the point: a test that only
    checked the returned document would pass while paying for it.
    """
    import inspect

    from manamap.pilot import benchmark, calibrate, diagnostic
    checked = 0
    for mod, fn in ((diagnostic, "run_on"), (benchmark, None), (calibrate, None)):
        src = inspect.getsource(getattr(mod, fn) if fn else mod)
        for chunk in src.split("goldfish.run(")[1:]:
            assert "_band=False" in chunk.split(")")[0] + chunk[:400], (
                f"{mod.__name__}: a goldfish.run call that does not opt out of "
                f"the band — it will pay for a floor run nothing reads")
            checked += 1
    assert checked >= 3
