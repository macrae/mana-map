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
