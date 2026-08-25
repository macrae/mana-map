"""Commander search: the parts that decide a ranking, gated.

Everything network-shaped is faked here. The point is not to re-measure the
embedding — `eval-commander-search` does that against a frozen pool — but to pin
the decisions around it: which cards enter a centroid, what identity a seed has,
and the one that would be invisible if it broke, that the eval and the product
share a single implementation.
"""

import random

import numpy as np
import pytest

from conftest import requires_data
from manamap.analysis import commander_search as cs

pytestmark = [requires_data]


@pytest.fixture(scope="module")
def corpus():
    try:
        return cs.Corpus()
    except FileNotFoundError:
        pytest.skip("cards.csv not built — `manamap extract`")


# ── One implementation, not two ────────────────────────────────────────────


def test_the_eval_scores_with_the_product_code():
    """The eval measures the product, so it must not own a second centroid.

    Two copies drift the first time either is tuned, and then the accuracy in
    the commit message belongs to code nobody ships. Same argument the deck
    builder makes for having exactly one scorer — and that one was written down
    only after a JS re-implementation had already diverged.
    """
    from manamap.analysis import eval_commander_search as ecs

    assert ecs.centroid is cs.centroid
    assert ecs.type_controlled_rows is cs.type_controlled_rows
    assert ecs.Corpus is cs.Corpus
    assert ecs.BASIC_LANDS is cs.BASIC_LANDS


# ── What goes into a centroid ──────────────────────────────────────────────


def test_basics_are_dropped_and_utility_lands_are_kept(corpus):
    """§6.1 step 2. Basics would pull every centroid toward one point; Command
    Tower carries real signal about what a deck is doing."""
    rows, missing = corpus.rows(["Plains", "Island", "Command Tower", "Sol Ring"])
    assert corpus.by_name["Command Tower"] in rows
    assert corpus.by_name["Sol Ring"] in rows
    assert corpus.by_name["Plains"] not in rows
    assert missing == []


def test_an_unresolved_name_is_reported_not_silently_dropped(corpus):
    """A seed that half-resolves and says nothing produces a confident ranking
    over a query the user did not ask."""
    rows, missing = corpus.rows(["Sol Ring", "Not A Real Card"])
    assert len(rows) == 1
    assert missing == ["Not A Real Card"]


def test_a_duplicate_name_counts_once(corpus):
    """A centroid is a mean; listing a card twice would weight it twice and let
    a repeated name quietly steer the ranking."""
    rows, _ = corpus.rows(["Sol Ring", "Sol Ring", "Command Tower"])
    assert len(rows) == 2


# ── Identity, which chooses the whole candidate pool ───────────────────────


def test_identity_is_the_union_of_the_seed(corpus):
    """DERIVED, never authored — one black card makes a seed black-inclusive,
    because a deck holding it would be."""
    rows, _ = corpus.rows(["Swords to Plowshares", "Counterspell", "Dark Ritual"])
    assert cs.seed_identity(rows, corpus) == frozenset("wub")


def test_identity_codes_are_wubrg_ordered():
    """`ub` and `bu` are not the same URL, so the order is not cosmetic."""
    assert cs.identity_code(frozenset("ub")) == "ub"
    assert cs.identity_code(frozenset("bu")) == "ub"
    assert cs.identity_code(frozenset("rgw")) == "wrg"
    assert cs.identity_code(frozenset()) == "colorless"


def test_comma_separated_identities_parse(corpus):
    """`cards.csv` stores `"B, G"`. Reading that as one token is a bug this repo
    has already shipped once — `--identity GU` matched only colourless cards."""
    row = corpus.by_name.get("Deathrite Shaman")
    if row is None:
        pytest.skip("Deathrite Shaman not in this corpus")
    assert corpus.identities[row] == frozenset("bg")


def test_three_colour_identities_get_their_edhrec_name():
    """EDHREC 403s on `wur` and answers on `jeskai`. Every other arity takes a
    colour code — probed, not assumed — so this mapping covers exactly ten."""
    from manamap.ingest import edhrec

    assert edhrec.identity_segment("wur") == "jeskai"
    assert edhrec.identity_segment("wbg") == "abzan"
    assert edhrec.identity_segment("w") == "w"
    assert edhrec.identity_segment("rg") == "rg"
    assert edhrec.identity_segment("wubr") == "wubr"
    assert edhrec.identity_segment("wubrg") == "wubrg"


# ── The ranking itself, on a fake network ──────────────────────────────────


def _fake_edhrec(monkeypatch, corpus, decks):
    from manamap.ingest import edhrec

    monkeypatch.setattr(edhrec, "top_commanders", lambda ident, limit=100: list(decks))
    monkeypatch.setattr(edhrec, "average_deck",
                        lambda name: {"slug": name, "commander": name,
                                      "cards": [(n, 1) for n in decks[name]]})


def test_the_closest_deck_ranks_first(monkeypatch, corpus):
    """End to end with the network faked: a candidate whose deck IS the seed
    must outrank one built from unrelated cards."""
    seed = ["Sol Ring", "Arcane Signet", "Command Tower", "Swords to Plowshares"]
    far = ["Llanowar Elves", "Giant Growth", "Rampant Growth", "Overrun"]
    _fake_edhrec(monkeypatch, corpus, {"Twin": seed, "Stranger": far})

    out = cs.search(seed, corpus=corpus, limit=5)
    assert [r["commander"] for r in out["results"]][0] == "Twin"
    assert out["results"][0]["score"] > out["results"][1]["score"]


def test_the_result_carries_the_caveat_and_the_shared_cards(monkeypatch, corpus):
    """§6.3: the UI must invite inspection rather than assert a match. The
    shared-card list is what makes inspection possible without a second query,
    and the caveat travels with the data rather than living in one renderer."""
    seed = ["Sol Ring", "Command Tower", "Arcane Signet"]
    _fake_edhrec(monkeypatch, corpus, {"Twin": seed})
    out = cs.search(seed, corpus=corpus)
    assert "discovery aid" in out["caveat"]
    assert set(out["results"][0]["shared"]) >= {"Sol Ring", "Command Tower"}


def test_an_empty_seed_refuses_rather_than_ranking_nothing(corpus):
    with pytest.raises(SystemExit):
        cs.search(["Not A Real Card"], corpus=corpus)


# ── Type control ───────────────────────────────────────────────────────────


def test_type_control_never_empties_a_reference(corpus):
    """A deck holding none of the seed's types stays rankable — badly, which is
    informative — instead of dropping out of the denominator silently."""
    creatures = [i for i, t in enumerate(corpus.types) if t == "Creature"][:20]
    picked = cs.type_controlled_rows(creatures, corpus.types,
                                     {"Instant": 1.0}, random.Random(0))
    assert picked


def test_composition_sums_to_one(corpus):
    rows, _ = corpus.rows(["Sol Ring", "Swords to Plowshares", "Llanowar Elves"])
    comp = cs.composition(rows, corpus.types)
    assert abs(sum(comp.values()) - 1.0) < 1e-9


# ── The default space is a measurement, and it should be hard to change ────


def test_the_default_space_is_text_and_says_why():
    """Not a convenience. The trained space measures 0.410 top-1 against the
    frozen text baseline's 0.584 over ten held-out draws, ranges not
    overlapping. When Track A2 lands this flips — and the eval says so first.
    """
    import inspect

    assert cs.SPACES["text"].name.startswith("text")
    doc = inspect.getdoc(cs) or ""
    assert "0.584" in doc and "0.410" in doc, (
        "the default-space decision must carry its numbers, or the next reader "
        "has to take it on faith")
