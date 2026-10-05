"""The `forge` half of `pilot_policy.json` — a per-deck AI profile, compiled and gated.

Forge's `res/ai/*.ai` files are 121 `KEY=VALUE` knobs and `AiProfileUtil.getAvailableProfiles`
is `new File(res/ai).list()`, so any file dropped there becomes a name `-a` accepts, and
`loadProfile` falls back to `AiProps.getDefault()` per key — a PARTIAL file is legal. That is
Forge's own supported lever for piloting, and it needs no engine patching at all.
"""

import pathlib

import pytest

from manamap.pilot import forge_ai as fa, pilot_policy as pp
from manamap.sim import forge, forge_pilot as fp


def _policy(**forge_keys):
    return {"rules": [{"id": "r", "why": "because", "when": {"channel": "spell_token_copy"},
                       "hold_until": {"other_creatures": 0}}],
            "forge": dict(forge_keys)}


# ------------------------------------------------------------------ the constants

def test_the_tracked_key_set_still_matches_the_engine():
    """THE `pod_behaviour.POD` PATTERN: a constant derived from something external gets a
    test that RE-DERIVES it and fails on drift, so it cannot outlive its evidence.

    121 keys are tracked so a policy can be validated on a machine with no Forge install.
    A Forge upgrade that adds or renames one must fail here rather than silently accept a
    key the engine no longer reads, or refuse one it does.
    """
    import re

    default_ai = fp.AI_DIR / "Default.ai"
    if not default_ai.is_file():
        pytest.skip("Forge is not installed on this machine")
    live = dict(re.findall(r"^([A-Z_0-9]+)=(.*)$", default_ai.read_text(), re.M))
    assert set(live) == set(fa.FORGE_AI_DEFAULTS), (
        f"only in engine: {sorted(set(live) - set(fa.FORGE_AI_DEFAULTS))}; "
        f"only tracked: {sorted(set(fa.FORGE_AI_DEFAULTS) - set(live))}")
    for key, raw in live.items():
        tracked, kind = fa.FORGE_AI_DEFAULTS[key]
        got = {"true": True, "false": False}.get(raw, raw)
        if kind == "int":
            got = int(raw)
        assert got == tracked, f"{key}: engine {raw!r}, tracked {tracked!r}"


def test_every_master_toggle_really_gates_its_family():
    """The masters are Forge's own words — "Master toggle for the following options",
    "If disabled, the following three options do nothing" — and the families are derived by
    POSITION in the file, so a reordering must be caught."""
    import re

    default_ai = fp.AI_DIR / "Default.ai"
    if not default_ai.is_file():
        pytest.skip("Forge is not installed on this machine")
    order = [m.group(1) for m in
             re.finditer(r"^([A-Z_0-9]+)=", default_ai.read_text(), re.M)]
    for master, children in fa.FORGE_AI_MASTERS.items():
        i = order.index(master)
        assert order[i + 1: i + 1 + len(children)] == list(children), (
            f"{master}'s family is no longer the {len(children)} keys after it: "
            f"{order[i + 1: i + 1 + len(children)]}")


def test_play_aggro_is_deliberately_not_a_master():
    """Its family is not a clean gate: `CHANCE_TO_ATTACK_INTO_TRADE` "works even if not
    playing all-out aggro, e.g. PLAY_AGGRO disabled", while
    `ATTACK_INTO_TRADE_WHEN_TAPPED_OUT` is "ignored if PLAY_AGGRO is globally enabled" — an
    INVERSE relation. Encoding it would refuse a correct policy."""
    assert "PLAY_AGGRO" not in fa.FORGE_AI_MASTERS


# ------------------------------------------------------------------- validation

def test_a_gated_key_without_its_master_is_refused():
    """THE BLOCKER. `SACRIFICE_DEFAULT_PREF_ENABLE` is FALSE in Default.ai, so a policy
    setting `SACRIFICE_DEFAULT_PREF_MAX_CMC` on a Default base compiles, runs 400 games and
    returns no difference — and the loop records UNPROVEN about a knob that was never on.

    That is "A CARD THE MODEL CANNOT READ LOOKS EXACTLY LIKE A CARD THAT DOES NOT HELP",
    one layer up, and it is worth an error rather than a comment.
    """
    with pytest.raises(pp.PolicyError, match="gated by SACRIFICE_DEFAULT_PREF_ENABLE"):
        fa.validate_forge(_policy(SACRIFICE_DEFAULT_PREF_MAX_CMC=3))
    # With the master on, it is accepted.
    fa.validate_forge(_policy(SACRIFICE_DEFAULT_PREF_ENABLE=True,
                              SACRIFICE_DEFAULT_PREF_MAX_CMC=3))
    # Explicitly switching the master OFF is refused too, though by the OTHER check and
    # that is the better message: False is already the default, so the line changes nothing
    # whatever else the policy says. Both errors are correct; the restated-default one
    # fires first because it is the more basic complaint.
    with pytest.raises(pp.PolicyError, match="changes nothing"):
        fa.validate_forge(_policy(SACRIFICE_DEFAULT_PREF_ENABLE=False,
                                  SACRIFICE_DEFAULT_PREF_MAX_CMC=3))


def test_a_key_forge_does_not_read_is_refused():
    with pytest.raises(pp.PolicyError, match="not an AiProps key"):
        fa.validate_forge(_policy(PLAY_LIKE_A_PRO=True))


def test_restating_a_default_is_refused():
    """A policy that restates a default is a rule nobody can measure: the A/B has two
    identical arms and reports noise as a finding."""
    with pytest.raises(pp.PolicyError, match="already 4 in Default.ai"):
        fa.validate_forge(_policy(MULLIGAN_THRESHOLD=4))


def test_the_wrong_type_is_refused():
    with pytest.raises(pp.PolicyError, match="is a flag"):
        fa.validate_forge(_policy(PLAY_AGGRO=1))
    with pytest.raises(pp.PolicyError, match="is a number"):
        fa.validate_forge(_policy(MULLIGAN_THRESHOLD=True))


def test_no_forge_section_is_absent_not_empty():
    assert fa.validate_forge({"rules": []}) == {}
    assert fa.forge_fingerprint({"rules": []}) is None


# -------------------------------------------------------------------- compiling

def test_the_compiled_profile_is_the_diff_and_shows_the_direction():
    """Partial on purpose — Forge falls back per key — so the file reads as the piloting
    decision. And it prints the DEFAULT beside the new value, because writing the first
    trial policy I set `MIN_COUNT_FOR_STORM_SPELLS: 2` under a `why` that said "lowering
    it"; the default is 1, so the value did the opposite of its stated reason. Nothing
    mechanical can check a reason against a direction; printing both lets a human.
    """
    doc = _policy(MULLIGAN_THRESHOLD=6)
    text = fa.compile_profile(doc, name="mm-test")
    body = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert body == ["MULLIGAN_THRESHOLD=6"], body
    assert "Default.ai has 4; this sets 6" in text
    assert "mm-test" in text
    # Booleans render as Forge spells them, not as Python.
    t2 = fa.compile_profile(_policy(PLAY_AGGRO=True))
    assert "PLAY_AGGRO=true" in t2 and "True" not in t2


def test_the_fingerprint_moves_with_a_VALUE_not_just_a_name():
    """`profile_tag` carries the profile NAME, which is enough for Forge's four fixed
    profiles and useless for one `mm-<slug>` whose content changes per experiment. Without
    this, iterating knob values would write one run id and the second run would be refused
    as an existing measurement — the third instance of the omission `profile_tag` and
    `clock_tag` were each written to fix."""
    a = fa.forge_fingerprint(_policy(MULLIGAN_THRESHOLD=6))
    b = fa.forge_fingerprint(_policy(MULLIGAN_THRESHOLD=5))
    assert a["sha"] != b["sha"]
    assert a["keys"] == ["MULLIGAN_THRESHOLD"] and a["n"] == 1


@pytest.mark.regression
def test_the_run_id_carries_the_profile_content():
    opp = ["sythis-enchantress"]
    # A REAL deck: `run_id_for` derives a config digest from every seat's decklist.
    plain = forge.run_id_for("goblin-storm", opp, 10, 1, None, None, 600)
    with_ai = forge.run_id_for("goblin-storm", opp, 10, 1, None, None, 600,
                               None, "deadbeefcafe")
    assert plain != with_ai
    assert with_ai.endswith("-aideadbeef")
    assert not plain.endswith("-ai"), "a plain run's id must be unchanged"


def test_forges_own_profiles_are_not_content_hashed():
    """They are named in the id by `profile_tag` already, they do not change, and hashing
    them would rename every historical record."""
    assert fp.profile_content_sha("Experimental") is None
    assert fp.profile_content_sha("Default") is None
    assert fp.profile_content_sha(None) is None


def test_a_branch_inherits_the_decks_profile():
    """Same reason `pilot_policy.load` inherits the policy: two candidate 99s are only
    comparable if the same hand pilots both."""
    assert fp.profile_name("goblin-storm@copy-burst-v1") == "mm-goblin-storm-copy-burst-v1"
    # `declared_profile` reads through `deck_file`, so the branch resolves the deck's file.
    text, fpr = fp.declared_profile("goblin-storm", "copy-burst-v1")
    deck_text, deck_fpr = fp.declared_profile("goblin-storm")
    assert (text, fpr) == (deck_text, deck_fpr)


# --------------------------------------------------- which module is model-facing

def test_the_forge_knobs_are_NOT_in_the_goldfish_model_stamp():
    """THE SPLIT IS LOAD-BEARING AND THIS IS WHAT HOLDS IT.

    `pilot_policy.py` is in `goldfish._MODEL_FILES` because the policy layer decides whether
    a card is cast at all. `model_version` is a sha over the WHOLE FILE, coarse on purpose.
    So while the 121 Forge constants lived in `pilot_policy.py`, every knob edit staled all
    37 goldfish artifacts — it happened within the hour of adding them: the stamp moved
    c42ed8ea2ec3 -> e41cebc10cf0 and ~25 fleet tests went red, for constants the goldfish
    does not read and cannot read.

    A knob edit must stale Forge runs and leave the goldfish alone. Moving `forge_ai.py`
    into `_MODEL_FILES` would silently undo that, and nothing else would notice.
    """
    from manamap.pilot import goldfish

    assert "pilot_policy.py" in goldfish._MODEL_FILES, (
        "the policy layer decides what gets cast; it must stale derived figures")
    assert "forge_ai.py" not in goldfish._MODEL_FILES, (
        "the Forge knobs are not read by the goldfish — putting them in the model stamp "
        "makes every knob edit regenerate the whole fleet for nothing")


def test_the_goldfish_does_not_import_the_forge_half():
    """The stamp only tells the truth if the dependency really is one-way."""
    import inspect

    from manamap.pilot import goldfish, goldfish_turn

    for mod in (goldfish, goldfish_turn):
        src = inspect.getsource(mod)
        assert "forge_ai" not in src, (
            f"{mod.__name__} imports forge_ai — then a knob edit really would change a "
            f"goldfish figure, and it would be outside the model stamp")
