"""`model-coverage` — what the goldfish cannot see, said BEFORE the games.

Every expensive fidelity surprise on this bench was found after the run:
eminence (a 400-game Forge arm), the token doublers, the fetchlands. The
`*_not_modelled` lists could not have warned, because they are only populated
for a channel the deck has ALREADY opted into — a deck that models nothing
reports nothing missing.

MEASURED across the fleet when this shipped: 236 DARK cards, and the only two
decks under five were edgar-vampires and ur-dragon — the two where the problem
had already been hit the expensive way and the flags turned on.

    blar             23 dark   combat (20)
    gishath          34 dark   combat (33)
    goblin-storm     36 dark   combat (28)
    kinnan           37 dark   combat (32)
    yawgmoth-swarm   39 dark   combat (28)
"""

import pytest

from manamap.pilot import model_coverage as mc

SLUG = "gishath"


def _report(slug=SLUG, branch=None):
    try:
        return mc.analyze(slug, branch)
    except FileNotFoundError:
        pytest.skip(f"{slug} has no cards.json")


def test_every_channel_names_a_real_flag_or_is_always_on():
    """A channel pointing at a flag `goldfish` never reads would be a report
    about a switch that does not exist."""
    from manamap.pilot import goldfish
    import inspect

    source = inspect.getsource(goldfish)
    checked = 0
    for channel, flag in mc.CHANNELS.items():
        if flag is None:
            continue
        assert f'targets_doc.get("{flag}")' in source, f"{channel} -> {flag}"
        checked += 1
    assert checked >= 4


def test_dark_is_per_channel_not_per_card():
    """THE BUG THIS SHIPPED WITH FOR ONE ITERATION. Nearly every creature feeds
    `bodies`, which is always on, so "seen if anything is active" reported
    gishath — a deck that opts into NOTHING — as 0 dark. A card is dark when
    ANYTHING about it is unread, whatever else is read.
    """
    report = _report()
    assert report["counts"]["dark"] > 0, "a deck with no flags cannot be all-seen"
    dark = [r for r in report["cards"] if r["state"] == "dark"]
    # …and the proof it is per-channel: at least one dark card also has an
    # ACTIVE channel. Under the per-card reading it would have been "seen".
    assert any(r["active"] for r in dark), "no dark card has an active channel"


def test_a_deck_with_a_flag_on_is_not_dark_for_that_channel():
    """The control. Without it the test above passes on a broken analyzer that
    calls everything dark."""
    report = _report()
    for row in report["cards"]:
        for channel in row["dark_channels"]:
            flag = mc.CHANNELS[channel]
            assert flag and not report["flags"][flag], (
                f"{row['name']}: {channel} is dark but {flag} is on")


def test_the_declaration_is_resolved_the_way_the_simulation_resolves_it():
    """RE-INTRODUCING A REAL BUG. `goldfish` reads the declaration through
    `common.deck_file`, which FALLS BACK from a branch to the deck — a branch
    has its own measurements but not its own authored files. Reading
    `deck_dir(slug, branch)` directly instead reported ur-dragon@landbase-v1 as
    30 cards dark when the declaration the simulation actually used makes it 2.

    A coverage report that disagrees with the model it reports on is worse than
    no report, so this asserts the two agree.
    """
    from manamap.config import DECKS_DIR

    slug = "ur-dragon"
    if not (DECKS_DIR / slug / "cards.json").exists():
        pytest.skip("ur-dragon absent")
    branches = [b.name for b in (DECKS_DIR / slug / "branches").glob("*")
                if b.is_dir() and not (b / "goldfish_targets.json").exists()
                and (b / "cards.json").exists()]
    if not branches:
        pytest.skip("no branch without its own declaration")
    deck = mc.analyze(slug)
    for branch in branches:
        assert mc.analyze(slug, branch)["flags"] == deck["flags"], (
            f"{slug}@{branch} resolved a different declaration than the deck")


def test_states_are_exhaustive_and_exclusive():
    report = _report()
    counts = report["counts"]
    assert sum(counts.values()) == report["distinct"] == len(report["cards"])
    assert set(counts) == {"seen", "dark", "invisible"}


def test_the_headline_is_empty_when_nothing_is_dark_and_names_the_branch():
    """It is printed as a preflight by `goldfish` and `net-change`, so silence
    has to mean "nothing to say" rather than "not computed"."""
    clean = {"counts": {"dark": 0}, "distinct": 9, "slug": "x",
             "branch": None, "dark_by_channel": {}}
    assert mc.headline(clean) == ""
    noisy = dict(clean, counts={"dark": 3},
                 dark_by_channel={"combat": ["a", "b", "c"]}, branch="b1")
    line = mc.headline(noisy)
    assert "3 of 9" in line and "combat" in line and "--branch b1" in line


def test_an_invisible_card_is_not_reported_as_dark():
    """A removal spell in a resource model is CORRECTLY invisible, and calling
    that a defect would make the report noise."""
    report = _report()
    for row in report["cards"]:
        if row["state"] == "invisible":
            assert not row["possible"] and not row["dark_channels"], row["name"]


def test_card_value_classifies_with_the_deck_pool_like_build_library_does():
    """`classify`'s docstring says the pool exists for exactly one card class
    and that "without it every fetch is a colourless land that never produces
    anything". `goldfish.build_library` always passes it; `card_value` did not.

    HONEST SCOPE: this changes nothing today. `card_value` reads only `is_land`
    and the visibility predicate off the classified dict, and `_is_visible`
    returns True for ANY land, so no fetch was ever dropped from a ranking. It
    is fixed so the next field read off that dict is not silently wrong — the
    same shape as the defect that made `mana-fit` score a strict improvement as
    a five-colour regression.
    """
    import inspect

    from manamap.pilot import card_value

    source = inspect.getsource(card_value)
    assert "goldfish.classify(c, pool=land_pool)" in source
    assert "goldfish.classify(c)" not in source, "a pool-less call came back"


# ── the sentinel that read as a capability ──────────────────────────────────

def test_an_unmodelled_treasure_trigger_is_not_a_dark_channel():
    """A SENTINEL MEANING "I CANNOT MODEL THIS" IS NOT EVIDENCE THAT A FLAG
    WOULD HELP.

    `treasure_trigger` is a string and one of its values is "unmodelled". The
    channel test read it for truthiness, so a card the parser had explicitly
    given up on counted as feeding the treasure channel — and the report then
    told the pilot to set `model_treasures` to light it up.

    Setting it does nothing. Measured on zur-enchantress 2026-09-04: both its
    treasure cards are in this state, the flag returned BYTE-IDENTICAL figures,
    and the hoard was 0.0 at every one of the ten turns. Across the fleet, 21
    cards in 7 decks — including Goldspan Dragon and Old Gnawbone on ur-dragon,
    whose genuinely nonzero hoard comes from its other sources entirely.

    That inflates DARK, which this module's own docstring calls "the thing that
    invalidates a run". Overstating it makes an honest run look untrustworthy
    and sends the reader to a switch that is not connected to anything.

    The draw channel never had this bug because it tests NAMED KEYS rather than
    the truthiness of a free-form field — `draw` carries the same "unmodelled"
    sentinel and correctly ignores it. This aligns treasure with draw.
    """
    from manamap.pilot.model_coverage import channels_for

    def profile(**over):
        base = {"is_land": False, "produces": 0, "reduces": None,
                "scales_with_colors": False, "creature_bodies": 0, "bodies": 0,
                "tutor": None, "treasure_n": 0, "treasure_trigger": None,
                "treasure_bonus": False, "treasure_doubler": False,
                "token_doubler": False, "sac_outlet": False,
                "combat": {}, "draw": {}, "death": {}}
        base.update(over)
        return base

    # The exact shape of Black Market Connections and Goldspan Dragon.
    unmodelled = profile(treasure_trigger="unmodelled", treasure_n=1)
    assert "treasure" not in channels_for(unmodelled), (
        "a trigger the turn loop never branches on was reported as a channel "
        "a flag would switch on")

    # THE CONTROL. Without it this change would have silently switched the
    # channel off for every deck that legitimately uses it.
    for trigger in ("upkeep", "landfall", "etb", "cast"):
        assert "treasure" in channels_for(
            profile(treasure_trigger=trigger, treasure_n=1)), trigger

    # And the paths that do not depend on a trigger at all still count.
    assert "treasure" in channels_for(profile(treasure_doubler=True))
    assert "treasure" in channels_for(profile(token_doubler=True))
    assert "treasure" in channels_for(profile(treasure_bonus=True))


def test_the_modelled_trigger_set_matches_what_goldfish_branches_on():
    """A COPY OF A VOCABULARY ROTS UNLESS SOMETHING COMPARES IT.

    `_MODELLED_TREASURE_TRIGGERS` restates the values `goldfish`'s turn loop
    branches on. Adding a trigger there and not here would put the card back in
    the invisible bucket — the same silence, one door over — so the two are
    compared against the source rather than trusted to stay in step.
    """
    import re
    from pathlib import Path

    from manamap.pilot.model_coverage import _MODELLED_TREASURE_TRIGGERS

    src = (Path(__file__).resolve().parent.parent
           / "src/manamap/pilot/goldfish.py").read_text(encoding="utf-8")
    branched = set()
    for m in re.finditer(r'card\["treasure_trigger"\]\s+in\s+\(([^)]*)\)', src):
        branched |= set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert branched, "the sweep found no treasure_trigger branch to compare against"
    assert branched == set(_MODELLED_TREASURE_TRIGGERS), (
        f"goldfish branches on {sorted(branched)} but model_coverage believes "
        f"{sorted(_MODELLED_TREASURE_TRIGGERS)} — a card whose trigger is in "
        f"one set and not the other is misfiled in the coverage report")


# ── read correctly, never played ────────────────────────────────────────────

def test_no_deck_computes_an_effect_it_never_gets_to_apply():
    """A CARD CAN BE READ CORRECTLY AND NEVER CAST.

    Every casting loop in `simulate_once`'s main phase selects on a channel:
    draws, ramps, makes Treasure, has a body. A card matching none of them sits
    in hand for ten turns while its profile says exactly what it would have
    done — and the figure that results is not low, it is a different number
    about a different deck.

    `goldfish.py` already carried a comment about patching this once, for damage
    doublers: "Gratuitous Violence and Dictate of the Twin Gods are
    enchantments, read correctly and never cast." It was patched case by case,
    and nothing looked for the next one.

    The fleet sweep on 2026-09-04 found 228 never-cast cards across ten decks
    (26%), of which four were live losses:

        zur-enchantress  Sanctum of Stone Fangs, Northern Air Temple — found
            first, and they were the reason two Shrines measured as EXACTLY
            nothing. Casting them nearly doubled the deck's drain.
        zur-enchantress  Steel of the Godhead, Sheltered by Ghosts — Auras that
            grant lifelink.
        edgar-vampires   Ashnod's Altar, Altar of Dementia — free sacrifice
            outlets with no body. `free_sac_outlet` was set inside the BODIES
            loop, so a SLEEVED deck ran its engine on 2 of its 4 outlets.

    The other 224 are correct: a counterspell should never be cast in a
    goldfish, and this module's docstring says so. The defect is only ever a
    card that is never cast AND feeds a channel that is switched ON.
    """
    import glob
    import pathlib as _pl

    from manamap.pilot.model_coverage import silent_losses

    root = _pl.Path(__file__).resolve().parent.parent
    decks = sorted(_pl.Path(f).parent.name
                   for f in glob.glob(str(root / "data/decks/*/cards.json")))
    if len(decks) < 5:
        pytest.skip("deck data not present")

    checked, offenders = 0, {}
    for slug in decks:
        losses = silent_losses(slug)
        checked += 1
        if losses:
            offenders[slug] = losses
    assert checked >= 5, "the loop stopped checking"
    assert not offenders, (
        "these cards feed a channel the deck switched ON and no casting loop "
        "will ever select them, so the model computes an effect it never "
        "applies:\n" + "\n".join(
            f"  {s}: " + ", ".join(f"{n} ({'/'.join(ch)})" for n, ch in v)
            for s, v in offenders.items()))
