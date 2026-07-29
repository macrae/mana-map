"""Tests for the card-reference matcher.

The matcher's contract is asymmetric: false positives are safe (one
unnecessary regeneration), false negatives are the quality risk. Every test
that pins a non-match is really pinning "the stopword list did its job";
every match test pins the conservative side.
"""

from manamap.pilot import card_refs as cr


DECK = [
    "Gishath, Sun's Avatar",
    "Forerunner of the Empire",
    "Monster Manual // Zoological Study",
    "Beast Within",
    "Sol Ring",
    "Blood Crypt",
]


def refs(text):
    return cr.text_refs(text, DECK)


# ── Full names and possessives ───────────────────────────────────────────


def test_full_name_matches():
    assert "Sol Ring" in refs("tap Sol Ring for two")


def test_possessive_matches():
    assert "Gishath, Sun's Avatar" in refs("Gishath, Sun's Avatar's trigger fires")


def test_case_insensitive():
    assert "Sol Ring" in refs("SOL RING untaps")


# ── Double-faced names ───────────────────────────────────────────────────


def test_front_face_matches():
    assert "Monster Manual // Zoological Study" in refs("cast Monster Manual on turn three")


def test_back_face_matches():
    assert "Monster Manual // Zoological Study" in refs("Zoological Study digs two deep")


# ── Distinctive tokens (the nickname gap) ────────────────────────────────


def test_distinctive_token_matches_nickname():
    """'the Forerunner ping web' must pin Forerunner of the Empire."""
    assert "Forerunner of the Empire" in refs("the Forerunner ping web re-arms")


def test_short_tokens_do_not_match():
    """'Sol'/'Ring' are under the length floor; only the full name matches."""
    assert "Sol Ring" not in refs("the ring of stones surrounds the solar altar")


def test_stopworded_token_does_not_match():
    """'Avatar' is stopworded — tribal words appear constantly in prose."""
    assert "Gishath, Sun's Avatar" not in refs("a big avatar of the tables")


def test_stopworded_card_still_matches_full_name():
    """Blood Crypt's tokens are all stopworded/short; full-name still works."""
    assert "Blood Crypt" in refs("fetch Blood Crypt untapped")
    assert "Blood Crypt" not in refs("first blood goes to the crypt keeper")


def test_over_trigger_is_by_design():
    """'within' is a distinctive token of Beast Within — matching ordinary
    prose is the accepted false-positive cost of never under-invalidating."""
    assert "Beast Within" in refs("nothing within the loop can stop it")


# ── Probes ───────────────────────────────────────────────────────────────


def test_name_probes_shape():
    probes = cr.name_probes("Forerunner of the Empire")
    assert "forerunner of the empire" in probes
    assert "forerunner" in probes
    assert "empire" in probes
    assert "of" not in probes


def test_dfc_probes_include_both_faces():
    probes = cr.name_probes("Monster Manual // Zoological Study")
    assert "monster manual" in probes
    assert "zoological study" in probes


# ── Artifact-level extraction ────────────────────────────────────────────


def test_artifact_refs_walks_whole_doc():
    doc = {"a": {"nested": ["Sol Ring on turn one"]}, "b": "no cards here"}
    assert cr.artifact_card_refs(doc, DECK) == ["Sol Ring"]


def test_refs_by_key_scopes_per_key():
    doc = {"how_it_wins": "Gishath, Sun's Avatar connects",
           "mulligan": "keep hands with Sol Ring",
           "unowned": "Forerunner of the Empire"}
    by_key = cr.artifact_card_refs_by_key(doc, ["how_it_wins", "mulligan"], DECK)
    assert by_key["how_it_wins"] == ["Gishath, Sun's Avatar"]
    assert by_key["mulligan"] == ["Sol Ring"]
    assert "unowned" not in by_key


def test_card_roles_keys_count_as_refs():
    """card_roles is keyed by exact names — keys serialize into the JSON."""
    doc = {"card_roles": {"Sol Ring": "ramp"}}
    assert "Sol Ring" in cr.artifact_card_refs(doc, DECK)


def test_deck_card_names():
    deck_doc = {"cards": [{"name": "Sol Ring"}, {"name": "Beast Within"}, {"name": None}]}
    assert cr.deck_card_names(deck_doc) == ["Beast Within", "Sol Ring"]
