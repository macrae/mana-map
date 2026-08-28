"""`mana-fit` — right-size the mana to the spells actually in the list.

THE STEP THAT WAS MISSING. `mana-analysis` measures the gap and stops, so every
refactor ended with a shortfall and no answer — and the shortfall MOVES the
moment a spell changes. Cutting three counterspells takes blue pips out; adding
six dorks puts sources in; a manabase fitted before those changes is fitted to a
list nobody is playing.
"""

import pytest

from conftest import requires_data, requires_deck
from manamap.pilot import mana_fit

SLUG = "ur-dragon"


@requires_data
@requires_deck
def test_it_composes_mana_analysis_rather_than_recomputing_it():
    """THE DEFECT THIS EXISTS AGAINST, and it was in the first cut of this file.

    `shortfall` recomputed the source counts by running `land_colors` over every
    card and reported 53 red sources against `mana-analysis`'s 27 — because
    `land_colors` applied to a spell answers a question nobody asked, and
    because that module gates production by the commander's colour identity and
    classifies a non-land producer through `nonland_producer_kind`. Two modules
    that can disagree about one number is the divergence this repo keeps paying
    for, so there is one owner and this one reads it.
    """
    from manamap.pilot import mana_analysis
    theirs = mana_analysis.analyze(SLUG)
    ours = mana_fit.shortfall(SLUG)
    checked = 0
    for c in "WUBRG":
        assert ours["colours"][c]["have"] == theirs["sources"]["total"][c], c
        assert ours["colours"][c]["target"] == theirs["source_targets"][c], c
        checked += 1
    assert checked == 5
    assert ours["lands"] == theirs["lands"]["total"]


@requires_data
@requires_deck
def test_the_gap_is_have_minus_target_and_carries_its_sign():
    """Direction is load-bearing: without it a colour ABOVE target reads as a
    problem, and the cut scan would offer the wrong lands."""
    doc = mana_fit.shortfall(SLUG)
    for c, row in doc["colours"].items():
        assert row["short"] == row["have"] - row["target"], c
    short = {c for c, r in doc["colours"].items() if r["short"] < 0}
    got = mana_fit.propose(SLUG, limit=3)
    assert set(got["short"]) == short
    assert not (set(got["short"]) & set(got["over"])), "a colour is one or the other"


@requires_data
@requires_deck
def test_it_only_proposes_cards_that_cover_a_colour_the_list_is_short_of():
    doc = mana_fit.propose(SLUG, limit=8)
    checked = 0
    for kind, rows in doc["add"].items():
        for r in rows:
            assert r["covers"], f"{r['name']} covers nothing"
            assert set(r["covers"]) <= set(doc["short"]), r["name"]
            assert set(r["covers"]) <= set(r["colours"])
            checked += 1
    assert checked >= 5, "no proposals at all on a deck with a known shortfall"


@requires_data
@requires_deck
def test_a_card_already_in_the_list_is_never_proposed():
    from manamap.pilot.common import deck_dir
    from manamap.pilot.fetch_deck import parse_decklist
    held = {e["name"] for e in parse_decklist(
        (deck_dir(SLUG) / "decklist.txt").read_text())}
    doc = mana_fit.propose(SLUG, limit=20)
    for rows in doc["add"].values():
        for r in rows:
            assert r["name"] not in held, r["name"]


@requires_data
@requires_deck
def test_more_colours_covered_outranks_fewer_and_untapped_outranks_tapped():
    """A five-colour land is five fixes in one slot; a basic is one. And a land
    that always enters tapped costs a turn, so it ranks below an untapped source
    covering the same colours."""
    doc = mana_fit.propose(SLUG, limit=25)
    rows = doc["add"]["land"]
    assert len(rows) >= 4
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True), "not ranked by coverage"
    for r in rows:
        expected = len(r["covers"]) - (mana_fit.TAPPED_PENALTY if r["tapped"] else 0)
        assert r["score"] == pytest.approx(expected)


@requires_data
@requires_deck
def test_owned_only_asks_the_boxes():
    from manamap.pilot import collection
    owned = collection.owned_names()
    doc = mana_fit.propose(SLUG, owned_only=True, limit=20)
    checked = 0
    for rows in doc["add"].values():
        for r in rows:
            assert r["name"] in owned and r["owned"] is True
            checked += 1
    assert checked >= 3


@requires_data
@requires_deck
def test_a_land_feeding_a_short_colour_is_never_a_cut_however_slow_it_is():
    """The cut scan exists to find lands that fix nothing scarce. A land that
    touches a short colour is not a candidate even if it always enters tapped —
    proposing it would trade a real source for tempo the deck cannot spend."""
    doc = mana_fit.propose(SLUG, limit=5)
    for r in doc["cut"]:
        assert not (set(r["colours"]) & set(doc["short"])), r["name"]
        assert r["why"]


@requires_data
@requires_deck
def test_a_splash_names_itself_because_its_target_is_the_loudest_number():
    """A Karsten target of 30 sources driven by a single {B}{B} spell reads
    identically to one driven by thirty black cards. Cutting the card is usually
    cheaper than buying the sources, and the report has to say which it is."""
    doc = mana_fit.propose(SLUG, limit=3)
    splashes = [c for c, r in doc["colours"].items() if r["splash"]]
    for c in splashes:
        row = doc["colours"][c]
        assert row["pip_share"] < mana_fit.SPLASH_PIP_SHARE
        assert row["pip_cards"] >= 1
    if splashes:
        assert any("SPLASH" in n for n in doc["notes"])


@requires_data
@requires_deck
def test_the_notes_refuse_to_be_read_as_a_measurement():
    doc = mana_fit.propose(SLUG, limit=3)
    blob = " ".join(doc["notes"])
    assert "THESE ARE PROPOSALS" in blob
    assert "net-change" in blob and "candidates" in blob
    assert "hypergeometric" in blob.lower()


@requires_data
@requires_deck
def test_it_renders_without_throwing_on_a_deck_and_on_a_branch():
    from manamap.pilot import deck_branch
    assert "MANA FIT" in mana_fit.format_report(mana_fit.propose(SLUG, limit=3))
    for b in deck_branch.names(SLUG):
        out = mana_fit.format_report(mana_fit.propose(SLUG, branch=b, limit=3))
        assert f"{SLUG}/{b}" in out
        break


# ── the reminder-text defect this pass found ─────────────────────────────

@requires_data
def test_reminder_text_is_not_this_cards_own_ability():
    """PROSPEROUS INNKEEPER HAS NO MANA ABILITY AT ALL. It creates a Treasure,
    and the Treasure's reminder text — `(It's an artifact with "{T}, Sacrifice
    this token: Add one mana of any color.")` — was being read as the creature's
    own, making it a five-colour source AND a `ramp:dork`. 24 cards in the
    corpus read colours from reminder text alone, every one a Treasure-maker,
    and Goldvein Pick and Prying Blade are in zur-enchantress today.

    Parentheses in oracle text are always reminder text, so stripping them is
    exact rather than a heuristic. Proven by re-introducing the bug: with the
    reminder left in, both of these become five-colour sources.
    """
    from manamap.pilot import card_pool, mana_analysis, manabase
    o = card_pool.corpus_oracle()
    pool = card_pool.load_pool()
    checked = 0
    for name in ("Prosperous Innkeeper", "Goldvein Pick"):
        if name not in pool:
            continue
        card = dict(pool[name], name=name, oracle_text=o.get(name, ""))
        assert mana_analysis.nonland_producer_kind(card) is None, name
        # ...and with the reminder text left in, it wrongly reads as a source.
        with_reminder = dict(card, oracle_text=o.get(name, "").replace("(", " ").replace(")", " "))
        assert mana_analysis.nonland_producer_kind(with_reminder) is not None, (
            f"{name} no longer exercises the defect")
        checked += 1
    assert checked >= 1

    # A REAL mana ability still reads. Sol Ring is deliberately absent from
    # this list: it makes {C}, so an empty COLOUR set is the right answer and
    # asserting otherwise would test the wrong thing.
    for name in ("Birds of Paradise", "Command Tower", "Arcane Signet"):
        card = dict(pool[name], name=name, oracle_text=o.get(name, ""))
        assert manabase.land_colors(card), name
    assert mana_analysis.nonland_producer_kind(
        dict(pool["Sol Ring"], name="Sol Ring",
             oracle_text=o.get("Sol Ring", ""))) == "ramp:rock"


@requires_data
def test_the_reminder_fix_is_scoped_to_the_cards_the_sweep_found():
    """A PATTERN SHIPS WITH ITS SWEEP. If this count moves, something else now
    parses differently and has to be read card by card before it is believed."""
    import re
    from manamap.pilot import card_pool, manabase
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()
    paren = re.compile(r"\([^)]*\)")
    moved = 0
    for name, info in pool.items():
        text = o.get(name) or ""
        if "(" not in text:
            continue
        stripped = dict(info, name=name, oracle_text=paren.sub(" ", text))
        loud = dict(info, name=name, oracle_text=text.replace("(", " ").replace(")", " "))
        if manabase.land_colors(loud) and not manabase.land_colors(stripped):
            moved += 1
    # THE NUMBER MOVED WHEN THE FALLBACK WENT, and that is the finding rather
    # than a regression. With the colour-identity fallback in place a card whose
    # only coloured text was a reminder still got colours BOTH ways — from the
    # reminder, or from its own identity — so the two readings agreed and it did
    # not count. 24 was what the defect looked like through the fallback; 356 is
    # its real size. Both fixes were needed and neither alone was enough.
    assert 300 <= moved <= 420, (
        f"{moved} cards read mana colours from reminder text alone; measured at "
        f"356 once the colour-identity fallback was removed. A different number "
        f"means the parser changed and the tail needs reading again.")


@requires_data
def test_a_land_is_credited_only_with_mana_it_can_actually_make():
    """THE COLOUR-IDENTITY FALLBACK IS GONE, and the set it was presumed to
    exist for was EMPTY.

    `land_colors` used to end: if nothing parsed, credit the card with its own
    colour identity. Enumerated before removal — 60 lands relied on it, 59 of
    them `{T}: Add {C}` taking their colour from an unrelated ACTIVATION COST,
    and the 60th with no `add` clause at all. Of 129 counted non-lands, ZERO had
    a coloured `add` the parser should have caught.

    Found independently by two `deck-doctor` runs: goblin-storm's red land count
    was 33 and is 31; yawgmoth-swarm's Pawn of Ulamog was a black source because
    it is a black card.
    """
    from manamap.pilot import card_pool, manabase
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()

    def colours(name):
        return manabase.land_colors(
            dict(pool[name], name=name, oracle_text=o.get(name, "")))

    # Taps for {C}; the colour was coming from a pump/sacrifice cost.
    for name in ("Goblin Burrows", "Kher Keep", "Blighted Woodland"):
        if name in pool:
            assert not colours(name), f"{name} makes only colourless mana"
    # ...and a real dual, a real five-colour land and a real dork still read.
    assert colours("Command Tower") == set("WUBRG")
    assert colours("Blood Crypt") == {"B", "R"}
    assert colours("Birds of Paradise") == set("WUBRG")


@requires_data
def test_the_fallback_removal_is_measured_not_assumed():
    """A REMOVAL SHIPS WITH ITS SWEEP, same rule as a widening. If any land ever
    needs the fallback again, this fails and the tail gets read card by card
    instead of the behaviour being restored on a hunch."""
    import re
    from manamap.pilot import card_pool, manabase
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()
    add = re.compile(r"add[^.\n]*(\{[wubrg]\}|one mana of any colou?r)", re.I)
    stranded = []
    checked = 0
    for name, info in pool.items():
        if "Land" not in (info.get("type_line") or ""):
            continue
        checked += 1
        text = o.get(name) or ""
        card = dict(info, name=name, oracle_text=text)
        # RESTRICTED MANA IS DELIBERATELY NOTHING — Haven of the Spirit Dragon
        # taps for {C} in a Vampire deck, and counting it as five sources is how
        # a mana base looks fine and cannot cast its spells. 35 lands say "spend
        # this mana only" and are correctly excluded; the first cut of this test
        # missed that and reported them as a regression.
        if manabase.RESTRICTED_MANA in text.lower():
            continue
        # A PROMISE IN REMINDER TEXT IS NOT A PROMISE. `Treasure Map` says
        # "one mana of any color" only inside the Treasure token's reminder;
        # reading it here would assert back the exact defect this file's other
        # test exists against.
        text = manabase._REMINDER_RE.sub(" ", text)
        # A land whose text promises unrestricted coloured mana must still read
        # as producing some.
        if add.search(text) and not manabase.land_colors(card):
            stranded.append(name)
    assert checked >= 1000, "the corpus is not loaded"
    assert not stranded, (
        f"{len(stranded)} land(s) promise coloured mana and now read as making "
        f"none: {stranded[:8]}")
