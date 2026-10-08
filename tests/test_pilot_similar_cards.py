"""similar-cards: nearest neighbours in the ability space, under card-search's rules.

The unit half drives `similar` over a five-card corpus built here, so the filters and
the centroid are proved without tracked data. The `requires_data` half asks the real
corpus one question whose answer is not in doubt.
"""

import numpy as np
import pandas as pd
import pytest

from manamap.pilot import similar_cards as sc

from conftest import requires_data


def _rec(ident, legal=True, gc=False, type_line="Sorcery"):
    return {"color_identity": set(ident), "legal": legal, "game_changer": gc,
            "type_line": type_line, "cmc": 3.0, "mana_cost": "{2}{U}",
            "edhrec_rank": 100, "set_code": "tst", "first_released_at": None}


@pytest.fixture
def tiny_corpus(monkeypatch):
    """Five cards. Seed `Wheel` points along x; `Twin` (blue) and `Redwheel` (red)
    sit near it, `Banned` is nearest of all but illegal, `Far` points the other way.
    `Wheel` appears twice — `cards.csv` repeats names, and the first row wins."""
    names = ["Wheel", "Twin", "Redwheel", "Banned", "Far", "Wheel", "Front // Back"]
    vecs = np.array([[1, 0], [0.95, 0.31], [0.9, 0.44], [0.99, 0.14], [-1, 0],
                     [1, 0], [0.8, 0.6]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pool = {"Wheel": _rec("U"), "Twin": _rec("U"), "Redwheel": _rec("R"),
            "Banned": _rec("U", legal=False), "Far": _rec("U"),
            "Front // Back": _rec("U", gc=True)}
    monkeypatch.setattr(sc, "load_frame", lambda: pd.DataFrame({"name": names}))
    monkeypatch.setattr(sc, "load_pool", lambda: pool)
    monkeypatch.setattr(sc, "ability_embeddings", lambda: vecs)
    monkeypatch.setattr(sc, "corpus_oracle", lambda: {n: "" for n in names})
    monkeypatch.setattr(sc, "load_card_roles",
                        lambda: {"Wheel": ["draw:wheel"], "Twin": ["draw:wheel"],
                                 "Far": ["removal"]})
    return names


def test_ranks_by_proximity_and_never_returns_the_seed(tiny_corpus):
    rows, meta = sc.similar(["Wheel"], limit=10)
    got = [r["name"] for r in rows]
    assert "Wheel" not in got
    assert got[0] == "Twin" and got[-1] == "Far"
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert meta["ranked_against"] == "the seed"


def test_illegal_cards_never_rank_and_are_counted(tiny_corpus):
    """`Banned` is the single nearest card; legality is a filter, not a note."""
    rows, meta = sc.similar(["Wheel"], limit=10)
    assert "Banned" not in [r["name"] for r in rows]
    assert meta["commander_illegal_skipped"] == 1


def test_identity_is_a_subset_filter(tiny_corpus):
    rows, _ = sc.similar(["Wheel"], identity={"U"}, limit=10)
    assert "Redwheel" not in [r["name"] for r in rows]
    rows, _ = sc.similar(["Wheel"], identity={"U", "R"}, limit=10)
    assert "Redwheel" in [r["name"] for r in rows]


def test_exclusion_and_game_changers(tiny_corpus):
    rows, _ = sc.similar(["Wheel"], exclude={"Twin"}, allow_game_changers=False, limit=10)
    names = [r["name"] for r in rows]
    assert "Twin" not in names and "Front // Back" not in names


def test_a_repeated_name_ranks_once(tiny_corpus):
    rows, _ = sc.similar(["Twin"], limit=10)
    assert [r["name"] for r in rows].count("Wheel") == 1


def test_several_seeds_rank_against_their_centroid(tiny_corpus):
    """Wheel + Redwheel's centroid lies between them; Twin is nearest it, and
    neither seed comes back."""
    rows, meta = sc.similar(["Wheel", "Redwheel"], limit=10)
    names = [r["name"] for r in rows]
    assert "Wheel" not in names and "Redwheel" not in names
    assert names[0] == "Twin" and meta["ranked_against"] == "the seeds' centroid"


def test_the_reason_is_the_shared_role(tiny_corpus):
    rows, _ = sc.similar(["Wheel"], limit=10)
    by = {r["name"]: r for r in rows}
    assert by["Twin"]["shared_roles"] == ["draw:wheel"]
    assert by["Far"]["shared_roles"] == [] and by["Far"]["roles"] == ["removal"]
    assert by["Twin"]["atlas"].endswith("?cards=Twin")


def test_seeds_resolve_by_face_and_unique_prefix(tiny_corpus):
    assert sc.resolve_seeds(["back"], tiny_corpus) == [6]
    assert sc.resolve_seeds(["fron"], tiny_corpus) == [6]
    assert sc.resolve_seeds(["WHEEL"], tiny_corpus) == [0]


def test_an_unresolved_or_ambiguous_seed_is_a_refusal_naming_candidates(tiny_corpus):
    """A seed silently dropped would move the centroid without saying so."""
    with pytest.raises(SystemExit, match="Twinn"):
        sc.resolve_seeds(["Twinn"], tiny_corpus)
    with pytest.raises(SystemExit, match="names 2 cards"):
        sc.resolve_seeds(["W"], ["Wheel", "Whirl"])


def test_misaligned_embeddings_are_a_refusal(tiny_corpus, monkeypatch):
    monkeypatch.setattr(sc, "ability_embeddings", lambda: np.zeros((2, 2), np.float32))
    with pytest.raises(SystemExit, match="index alignment"):
        sc.similar(["Wheel"])


@requires_data
def test_the_real_corpus_puts_a_wheel_near_a_wheel():
    rows, meta = sc.similar(["Windfall"], identity={"U", "R"}, limit=15)
    names = [r["name"] for r in rows]
    assert "Wheel of Fortune" in names
    assert all(set(r["color_identity"]) <= {"U", "R"} for r in rows)
    assert meta["returned"] == len(rows) == 15
