"""`sim/validate_sim.py` — the gate on every Forge run record.

THIS FILE IS NEW, which is the first thing worth recording about it: the module that
form-checks every measurement in the repo had no test file of its own. It was exercised
only incidentally, through `test_sim_parse`, `test_sim_experiment` and the fleet sweep in
`test_pilot_tracked_artifacts_validate` — so its own predicates were never driven directly,
and a checker with no checker is how `card_overrides` came to be absent from `REQUIRED`
without anybody noticing.
"""

# --------------------------------------------------------------- the harness stamp

def test_a_record_with_no_card_overrides_is_fine():
    """ABSENT MEANS ABSENT. A plain run has no such key and every record written before
    2026-09-28 lacks it, so `card_overrides` is deliberately NOT in `REQUIRED` —
    demanding it would redden history to no purpose."""
    from manamap.sim.validate_sim import _card_overrides_errors

    assert _card_overrides_errors({}) == []
    assert _card_overrides_errors({"card_overrides": None}) == []


def test_a_harness_stamp_must_be_comparable():
    """A stamp that cannot be compared is not provenance — the whole failure this key was
    added to fix was a fingerprint nobody could check."""
    from manamap.sim.validate_sim import _card_overrides_errors

    ok = {"card_overrides": {"sha": "60636e9e5565", "n": 2, "cards": ["a", "b"]}}
    assert _card_overrides_errors(ok) == []

    bad = _card_overrides_errors({"card_overrides": {"n": 2, "cards": ["a", "b"]}})
    assert any("sha" in e for e in bad), bad

    bad = _card_overrides_errors({"card_overrides": {"sha": "nothex", "n": 1, "cards": ["a"]}})
    assert any("12-hex" in e for e in bad), bad

    bad = _card_overrides_errors({"card_overrides": {"sha": "60636e9e5565", "n": 5,
                                                    "cards": ["a", "b"]}})
    assert any("claims n=5" in e for e in bad), bad


def test_a_record_stamped_while_the_engine_disagreed_is_refused():
    """`agrees: false` means the engine carried neither Forge's own scripts nor the ones
    the repo declares, so the rate describes NEITHER arm. `forge.run` refuses to start in
    that state, so a record carrying it predates the guard or was written by hand."""
    from manamap.sim.validate_sim import _card_overrides_errors

    errs = _card_overrides_errors({"card_overrides": {
        "sha": "deadbeefcafe", "n": 1, "cards": ["x"], "agrees": False}})
    assert any("describes neither arm" in e for e in errs), errs
    # agrees: true is the ordinary case and says nothing.
    assert _card_overrides_errors({"card_overrides": {
        "sha": "deadbeefcafe", "n": 1, "cards": ["x"], "agrees": True}}) == []


def test_the_check_does_not_fire_across_the_tracked_fleet():
    """MEASURED AGAINST EVERY TRACKED RECORD BEFORE SHIPPING, which is this repo's bar —
    six proposed validators have been rejected here for firing on correct data."""
    import glob
    import json

    from manamap.sim.validate_sim import _card_overrides_errors

    checked, bad = 0, []
    for p in sorted(glob.glob("data/decks/**/sim/*.json", recursive=True)
                    + glob.glob("data/opponents/**/sim/*.json", recursive=True)):
        if "/logs/" in p:
            continue
        checked += 1
        errs = _card_overrides_errors(json.load(open(p)))
        if errs:
            bad.append((p, errs))
    assert checked >= 20, f"expected the tracked corpus, swept {checked}"
    assert not bad, f"the harness check fires on tracked records: {bad}"


def test_an_experiment_records_the_harness_it_flew_under():
    """An experiment is the one place two lists are compared under a CONTROLLED
    instrument, and it recorded every part of that instrument except the card scripts —
    so an A/B could run half-overridden with nothing on disk saying so, on the exact
    command whose purpose is that the arms differ in the list and nothing else.

    Asserts the WIRING: an edit that drops the key leaves every other test here green.
    """
    import inspect

    from manamap.sim import experiment

    src = inspect.getsource(experiment)
    assert '"card_overrides": card_overrides(),' in src, (
        "the experiment record must stamp the harness both arms flew under")
