"""Tests for the deterministic manual renderer (pilot build_manual)."""

from manamap.pilot.build_manual import render_manual


def deck_doc():
    return {
        "deck": "test-deck",
        "cards": [
            {"name": "Wort, Boggart Auntie", "is_commander": True, "quantity": 1,
             "image": "https://img/wort.jpg"},
            {"name": "Skirk Prospector", "is_commander": False, "quantity": 1,
             "image": "https://img/skirk.jpg"},
            {"name": "Mountain", "is_commander": False, "quantity": 30, "image": None},
        ],
    }


def verified_stack():
    return {
        "id": "001",
        "slug": "storm-count",
        "deck": "test-deck",
        "title": "Storm count with Empty the Warrens",
        "scenario": {
            "stack": [{"pos": 0, "object": "Empty the Warrens", "controller": "you"}],
            "question": "How many goblins?",
        },
        "resolution": {
            "steps": [
                {"n": 1, "action": "Storm triggers", "effect": "4 copies",
                 "citations": [{"rule": "702.40a", "quote": "copy it for each other spell"}]}
            ],
            "final_state": {"summary": "10 goblins on board."},
        },
        "checker": {"verdict": "pass", "iterations": 2, "findings": []},
    }


PROSE = {
    "cover": {"tagline": "Goblins all the way down", "identity": "A storm deck."},
    "how_it_wins": "Cast cheap spells.\n\nThen Empty the Warrens.",
    "combo_lines": {"001": "The classic line."},
    "card_roles": {"Skirk Prospector": "Sac outlet and mana engine."},
    "mulligan": "Keep lands.",
    "upgrades": "None needed.",
}

SYNERGY = {"Skirk Prospector": [{"partner": "X", "score": 3, "synergies": ["Sac + Death Trigger"]}]}


def test_full_render_contains_all_sections():
    html_out = render_manual("test-deck", deck_doc(), [verified_stack()], PROSE, SYNERGY)
    for expected in [
        "Wort, Boggart Auntie",                      # cover title
        "Goblins all the way down",                  # tagline
        "How the Deck Wins",
        "Storm count with Empty the Warrens",        # verified stack spread
        "✓ verified · 2 iteration(s)",
        "702.40a",                                   # citation footnote
        "copy it for each other spell",              # verbatim quote
        "Sac outlet and mana engine.",               # card role prose
        "Sac + Death Trigger",                       # synergy label
        "Mulligan Guide",
        "Upgrade Paths",
    ]:
        assert expected in html_out, f"missing: {expected}"


def test_render_is_deterministic():
    a = render_manual("test-deck", deck_doc(), [verified_stack()], PROSE, SYNERGY)
    b = render_manual("test-deck", deck_doc(), [verified_stack()], PROSE, SYNERGY)
    assert a == b


def test_missing_prose_renders_todo_not_crash():
    html_out = render_manual("test-deck", deck_doc(), [verified_stack()], {}, {})
    assert "[TODO:" in html_out
    assert "How the Deck Wins" in html_out


def test_no_verified_stacks_renders_placeholder():
    html_out = render_manual("test-deck", deck_doc(), [], PROSE, SYNERGY)
    assert "no verified stack scenarios yet" in html_out


def test_html_escaping():
    doc = deck_doc()
    doc["cards"][1]["name"] = 'Skirk <script>alert("x")</script>'
    html_out = render_manual("test-deck", doc, [], PROSE, {})
    assert "<script>alert" not in html_out
    assert "&lt;script&gt;" in html_out
