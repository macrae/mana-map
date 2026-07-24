"""Tests for the deterministic manual renderer (pilot build_manual)."""

from manamap.pilot.build_manual import render_manual

GOLDFISH_FIXTURE = {
    "meta": {
        "commander": "Wort, Boggart Auntie", "commander_cmc": 4, "seed": 42,
        "iterations": 1000, "max_turn": 4, "deck": "test-deck",
        "model_assumptions": ["Simulates resource development, not full games."],
    },
    "metrics": {
        "iterations": 1000,
        "opening_hand": {"land_histogram": {"3": 500}, "keep_first_seven_rate": 0.9,
                          "mean_mulligans": 0.1},
        "land_drop_hit_rate_by_turn": {"1": 0.99, "2": 0.95, "3": 0.9, "4": 0.85},
        "mean_available_mana_by_turn": {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.1},
        "commander": {"cast_turn_histogram": {"4": 700}, "mean_cast_turn": 4.3,
                       "median_cast_turn": 4, "cast_by_turn_6_rate": 0.9},
        "mean_bodies_by_turn": {"1": 0.2, "2": 1.1, "3": 2.4, "4": 4.0},
        "targets": [{"label": "Cantrip drawn", "assembled_rate": 0.8,
                      "mean_turn": 3.2, "by_turn_6_rate": 0.5}],
    },
}

DECISION_FIXTURE = {
    "id": "001", "slug": "open-mana-signal", "deck": "test-deck", "kind": "decision",
    "title": "Open red with five bodies: cast now or hold?",
    "scenario": {
        "board": {"you": "Zada + 4 tokens, {R} open", "table": "Player 3 at 12 life, sweeper mana up"},
        "question": "Cast the cantrip now or hold for the sweeper?",
    },
    "branches": [
        {"choice": "Cast now", "line": "Fire the cantrip pre-combat.",
         "signals": "Announces the engine is live.", "coalition_risk": "High — you become the threat.",
         "coaching": "Only right if you can win off the draws."},
        {"choice": "Hold", "line": "Pass with mana up.",
         "signals": "Looks like interaction.", "coalition_risk": "Low.",
         "coaching": "Default line into open sweeper mana."},
    ],
    "recommendation": {"choice": "Hold", "rationale": "The sweeper undoes everything; patience wins."},
}


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


def test_tier_legend_on_cover():
    html_out = render_manual("test-deck", deck_doc(), [], PROSE, {})
    for expected in ["✓ RULES-VERIFIED", "◆ DATA-DERIVED", "★ COACHING", "How to read this manual"]:
        assert expected in html_out


def test_goldfish_section_renders():
    html_out = render_manual("test-deck", deck_doc(), [], PROSE, {}, goldfish=GOLDFISH_FIXTURE)
    for expected in [
        "Goldfish Numbers",
        "turn 4.3",                      # mean cast turn
        "90%",                           # cast by turn 6
        "Cantrip drawn",                 # target row
        "Simulates resource development, not full games.",  # assumptions
    ]:
        assert expected in html_out, f"missing: {expected}"


def test_goldfish_absent_renders_nothing():
    html_out = render_manual("test-deck", deck_doc(), [], PROSE, {}, goldfish=None)
    assert "Goldfish Numbers" not in html_out


def test_decision_spread_renders():
    html_out = render_manual("test-deck", deck_doc(), [], PROSE, {},
                             decisions=[DECISION_FIXTURE])
    for expected in [
        "Playing the Table",
        "Open red with five bodies: cast now or hold?",
        "★ RECOMMENDED LINE",
        "Coalition risk",
        "Looks like interaction.",
        "patience wins",
    ]:
        assert expected in html_out, f"missing: {expected}"
    # The recommended branch is "Hold" — verify the highlight lands on it.
    assert 'class="branch recommended"' in html_out


def test_v2_determinism():
    kwargs = dict(goldfish=GOLDFISH_FIXTURE, decisions=[DECISION_FIXTURE])
    a = render_manual("test-deck", deck_doc(), [verified_stack()], PROSE, SYNERGY, **kwargs)
    b = render_manual("test-deck", deck_doc(), [verified_stack()], PROSE, SYNERGY, **kwargs)
    assert a == b
