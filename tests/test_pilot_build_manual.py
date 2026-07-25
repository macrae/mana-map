"""Issue renderer: department completeness, contract integrity, determinism, escaping."""
import copy

from manamap.pilot.build_manual import render_issue
from manamap.pilot.design import esc
from manamap.pilot.issue_spec import DEPARTMENT_BY_ID, DEPARTMENT_IDS

ISSUE = {
    "volume": 1,
    "issue_date": "August 2026",
    "cover_price": "$4.95",
    "deck_name": "TEST DECK",
    "commander": "Test Commander",
    "cover_tagline": "A tagline",
    "next_issue": "NEXT DECK",
}


def deck_doc():
    return {
        "decklist_sha256": "abc123def456789",
        "cards": [
            {"name": "Test Commander", "is_commander": True, "is_sideboard": False,
             "quantity": 1, "mana_cost": "{3}{R}", "cmc": 4.0,
             "type_line": "Legendary Creature — Goblin", "image": "https://img/cmd.jpg",
             "color_identity": ["R"], "scryfall_uri": "https://scryfall/cmd"},
            {"name": "Sac Outlet", "is_commander": False, "is_sideboard": False,
             "quantity": 1, "type_line": "Creature — Goblin", "image": "https://img/sac.jpg"},
            {"name": "Payoff Engine", "is_commander": False, "is_sideboard": False,
             "quantity": 1, "type_line": "Enchantment", "image": "https://img/pay.jpg"},
        ],
    }


def verified_stack():
    return {
        "id": "001", "title": "Storm count with Empty the Warrens",
        "rules_version": "June 19, 2026",
        "scenario": {"question": "How many copies?", "stack": [{"pos": 0, "object": "X"}]},
        "resolution": {
            "steps": [{
                "n": 1, "action": "Storm triggers.", "effect": "Three copies.",
                "citations": [{"rule": "702.40a",
                               "quote": "copy it for each other spell"}],
            }],
            "final_state": {"summary": "Eight tokens."},
        },
        "checker": {"verdict": "pass", "iterations": 2, "findings": []},
    }


DECISION = {
    "id": "001", "kind": "decision", "title": "Hold or deploy?",
    "scenario": {"board": {"you": ["Two tokens"]}, "question": "What's your play?"},
    "branches": [
        {"choice": "Deploy", "line": "Cast it", "signals": "Aggressive",
         "coalition_risk": "Low", "coaching": "Usually right"},
        {"choice": "Hold", "line": "Pass", "signals": "Quiet",
         "coalition_risk": "None", "coaching": "Too slow"},
    ],
    "recommendation": {"choice": "Deploy", "rationale": "Tempo wins here."},
}

GOLDFISH = {
    "meta": {"iterations": 10000, "seed": 42, "decklist_sha256": "abc123def456789",
             "model_assumptions": ["No interaction is modeled.", "No removal."]},
    "metrics": {
        "commander": {"mean_cast_turn": 4.35, "median_cast_turn": 4,
                      "cast_by_turn_6_rate": 0.9},
        "opening_hand": {"keep_first_seven_rate": 0.791, "mean_mulligans": 0.26},
        "land_drop_hit_rate_by_turn": {"1": 1.0, "2": 0.9},
        "mean_available_mana_by_turn": {"1": 1.04, "2": 1.98},
        "mean_bodies_by_turn": {"1": 0.2, "2": 0.9},
        "targets": [{"label": "Engine online", "assembled_rate": 0.66, "mean_turn": 5.2}],
    },
}

PROSE = {
    "how_it_wins": "Bodies first, then the engine.",
    "combo_lines": {"001": "Intro prose for line 001."},
    "card_roles": {"Sac Outlet": "Sac outlet and mana engine.",
                   "Payoff Engine": "Drains the table."},
    "mulligan": "Keep hands with bodies.",
    "upgrades": "Consider these swaps.",
    "threat_assessment": "You flip to archenemy here.",
    "matchups": "Against sweepers, bank cards.",
}

SYNERGY = {"Sac Outlet": [{"partner": "Payoff Engine", "rule": "sacrifice:death-trigger"}]}

PLAN = {
    "slug": "test-deck",
    "angle": "A test angle.",
    "cover": {"kicker": "VERIFIED", "dominant_coverline": "THE BIG LINE",
              "teases": ["A specific tease"], "violators": [{"text": "NEW!"}]},
    "departments": [
        {"id": "cover"},
        {"id": "contents"},
        {"id": "first-turns", "kicker": "THE PLAN", "headline": "GOBLINS DOWN",
         "dek": "A dek.",
         "captions": {"Test Commander": "**THE MULTIPLIER:** she changes everything."},
         "pilot_tips": [{"card": "Sac Outlet", "text": "Play him turn one."}],
         "pull_quote": "A quotable line.",
         "callouts": [{"n": 1, "title": "BODIES FIRST", "text": "Make Goblins."}]},
        {"id": "command-zone", "kicker": "FORMAT", "headline": "THE ZONE",
         "dek": "A dek.", "body": "Command zone body copy.\n\nSecond paragraph."},
        {"id": "by-the-numbers", "kicker": "LAB", "headline": "TEN THOUSAND",
         "dek": "A dek."},
        {"id": "the-kill", "kicker": "VERIFIED", "headline": "THE PAYOFF", "dek": "A dek."},
        {"id": "politics-table", "kicker": "READ", "headline": "WHEN THEY TURN",
         "dek": "A dek."},
        {"id": "whats-your-play", "kicker": "YOUR MOVE", "headline": "TWO TURNS",
         "dek": "A dek."},
        {"id": "know-your-enemy", "kicker": "SCOUTING", "headline": "WHO BEATS YOU",
         "dek": "A dek.",
         "threats": [{"archetype": "Stax", "meter_label": "Threat", "rate": 0.8,
                      "read": "This is the one.", "outs": ["Vandalblast"]}]},
        {"id": "the-99", "kicker": "ROSTER", "headline": "EVERY SLOT", "dek": "A dek.",
         "roster": [{"role": "The engine", "cards": ["Payoff Engine"]}]},
        {"id": "featured-artist", "kicker": "THE GALLERY", "headline": "WHO PAINTED THIS",
         "dek": "A dek."},
        {"id": "keep-or-ship", "kicker": "DRILL", "headline": "FOUR SEVENS", "dek": "A dek.",
         "hands": [{"verdict": "KEEP", "cards": ["Mountain"], "why": "Enough mana."}]},
        {"id": "upgrade-watch", "kicker": "INSIDE", "headline": "SHOPPING LIST",
         "dek": "A dek."},
        {"id": "judges-desk", "kicker": "PROVE IT", "headline": "CASE FILES",
         "dek": "A dek."},
        {"id": "back-page"},
    ],
}


def render(**overrides):
    kwargs = dict(
        issue=ISSUE, plan=PLAN, deck_doc=deck_doc(),
        stacks=[verified_stack()], prose_doc=PROSE, synergy=SYNERGY,
        goldfish=GOLDFISH, decisions=[DECISION],
    )
    kwargs.update(overrides)
    return render_issue(**kwargs)


# ── Structure ────────────────────────────────────────────────────────────


def test_all_departments_render():
    html_out = render()
    for dept_id in DEPARTMENT_IDS:
        assert f'id="{dept_id}"' in html_out, f"missing department {dept_id}"


def test_department_titles_render():
    """Every department names itself — except the cover, which wears the masthead."""
    html_out = render()
    for dept_id in DEPARTMENT_IDS:
        if dept_id == "cover":
            continue
        assert esc(DEPARTMENT_BY_ID[dept_id]["title"]) in html_out


def test_issue_identity_on_cover():
    html_out = render()
    assert "VOL. 001" in html_out
    assert "August 2026" in html_out
    assert "$4.95" in html_out
    assert "THE BIG LINE" in html_out
    assert "A specific tease" in html_out
    assert "MANA MAP" in html_out


def test_plan_copy_renders():
    html_out = render()
    assert "GOBLINS DOWN" in html_out          # headline
    assert "THE PLAN" in html_out              # kicker
    assert "BODIES FIRST" in html_out          # callout
    assert "Play him turn one." in html_out    # pilot tip
    assert "A quotable line." in html_out      # pull quote
    assert "THE MULTIPLIER:" in html_out       # caption lead-in


def test_next_issue_and_colophon():
    html_out = render()
    assert "NEXT DECK" in html_out
    assert "abc123def456" in html_out          # decklist sha, truncated
    assert "Fan Content Policy" in html_out


# ── Contract integrity ───────────────────────────────────────────────────


def test_judges_desk_reproduces_citations_verbatim():
    html_out = render()
    assert "702.40a" in html_out
    assert "copy it for each other spell" in html_out
    assert 'id="case-001"' in html_out


def test_the_kill_points_at_the_dossier():
    html_out = render()
    assert "#case-001" in html_out


def test_checker_iteration_count_is_published():
    html_out = render()
    assert "2 review cycle(s)" in html_out


def test_tier_badges_come_from_the_department_system():
    """Costume never earns the badge — badges are not plan-controlled."""
    html_out = render()
    assert "RULES-VERIFIED" in html_out
    assert "DATA-DERIVED" in html_out
    assert "COACHING" in html_out


def test_goldfish_assumptions_always_render():
    html_out = render()
    assert "No interaction is modeled." in html_out
    assert "resource development, not full games" in html_out


def test_missing_prose_renders_visible_todo():
    html_out = render(prose_doc={})
    assert "todo" in html_out.lower()


def test_missing_goldfish_does_not_break_build():
    html_out = render(goldfish=None)
    assert 'id="by-the-numbers"' in html_out


def test_no_plan_still_renders_every_department():
    """Departments never vanish silently, even with no issue plan."""
    html_out = render(plan={})
    for dept_id in DEPARTMENT_IDS:
        assert f'id="{dept_id}"' in html_out


def test_no_stacks_renders_todo_not_silence():
    html_out = render(stacks=[])
    assert 'id="the-kill"' in html_out
    assert 'id="judges-desk"' in html_out


# ── Roster, decisions, threats ───────────────────────────────────────────


def test_roster_groups_lead_and_remainder_falls_into_depth():
    html_out = render()
    assert "<h3>The engine</h3>" in html_out
    assert "<h3>Depth</h3>" in html_out          # Sac Outlet wasn't named in the roster
    assert "Sac outlet and mana engine." in html_out


def test_decision_recommendation_follows_branches():
    """The reader commits before we answer (STYLEv3 §5.1)."""
    html_out = render()
    assert html_out.index("Usually right") < html_out.index("Tempo wins here.")


def test_threat_box_renders_with_meter():
    html_out = render()
    assert "Stax" in html_out
    assert "Vandalblast" in html_out
    assert "meter-track" in html_out


def test_keep_or_ship_hands_render():
    html_out = render()
    assert "Enough mana." in html_out


def test_command_zone_tax_ladder_renders():
    """The Commander Mandate: the tax ladder is the department's signature."""
    html_out = render()
    assert "tax-ladder" in html_out
    assert "Commander tax" in html_out
    assert "Commander File" in html_out


# ── Determinism & escaping ───────────────────────────────────────────────


def test_render_is_deterministic():
    assert render() == render()


def test_html_is_escaped():
    doc = deck_doc()
    doc["cards"][1]["name"] = 'Evil <script>alert("x")</script>'
    prose = dict(PROSE, card_roles={'Evil <script>alert("x")</script>': "Nasty."})
    html_out = render(deck_doc=doc, prose_doc=prose)
    assert "<script>alert" not in html_out
    assert "&lt;script&gt;" in html_out


# ── Featured Artist ──────────────────────────────────────────────────────


def artist_deck():
    """A deck with a clear standout artist and a secondary cluster."""
    doc = deck_doc()
    for card in doc["cards"]:
        card.update(artist="Lead Artist", set="sld", set_name="Secret Lair Drop",
                    collector_number="100", border_color="borderless",
                    frame_effects=["inverted"], finishes=["foil"], foil=True,
                    art_crop="https://img/crop.jpg")
    for i, name in enumerate(["Extra One", "Extra Two", "Extra Three"]):
        doc["cards"].append({
            "name": name, "is_commander": False, "is_sideboard": False, "quantity": 1,
            "type_line": "Instant", "image": f"https://img/{i}.jpg",
            "artist": "Other Artist", "set": "plst", "set_name": "The List",
            "collector_number": f"X-{i}", "border_color": "black",
            "frame_effects": [], "finishes": ["nonfoil"], "foil": False,
        })
    return doc


def test_featured_artist_renders_standout_and_gallery():
    plan = copy.deepcopy(PLAN)
    for dept in plan["departments"]:
        if dept["id"] == "featured-artist":
            dept["featured"] = {"artist": "Lead Artist", "note": "One drop, bought whole."}
            dept["also_worth_noting"] = [{"artist": "Other Artist", "note": "Three more."}]
    html_out = render(plan=plan, deck_doc=artist_deck())
    assert 'id="featured-artist"' in html_out
    assert "Lead Artist" in html_out
    assert "One drop, bought whole." in html_out
    assert "Every Lead Artist card in the deck" in html_out
    assert "Secret Lair Drop" in html_out          # printing credit in tiles


def test_featured_artist_leads_with_the_commander():
    html_out = render(plan=PLAN, deck_doc=artist_deck())
    section = html_out[html_out.index('id="featured-artist"'):]
    hero = section[:section.index("Every ")]
    assert "Test Commander" in hero


def test_featured_artist_surfaces_honesty_notes():
    """Counting caveats travel with the numbers onto the page."""
    doc = artist_deck()
    doc["cards"][1]["quantity"] = 22          # a basic-land-style multi-copy entry
    html_out = render(deck_doc=doc)
    assert "How these numbers are counted" in html_out
    assert "counted once" in html_out


def test_featured_artist_handles_a_deck_with_no_standout():
    """Every artist unique — the department still renders, telling breadth."""
    doc = deck_doc()
    for i, card in enumerate(doc["cards"]):
        card.update(artist=f"Artist {i}", set="plst", set_name="The List",
                    collector_number=f"X-{i}", border_color="black",
                    frame_effects=[], finishes=["nonfoil"], foil=False)
    html_out = render(deck_doc=doc)
    assert 'id="featured-artist"' in html_out
    assert "Art File" in html_out              # facts box still renders


def test_featured_artist_works_without_artist_data():
    """Older decks have no printing metadata — must not crash."""
    html_out = render(deck_doc=deck_doc())
    assert 'id="featured-artist"' in html_out


def test_featured_artist_sits_between_the_99_and_keep_or_ship():
    html_out = render()
    assert (html_out.index('id="the-99"')
            < html_out.index('id="featured-artist"')
            < html_out.index('id="keep-or-ship"'))


# ── No department silently drops validated furniture ─────────────────────


def test_every_department_renders_its_plan_furniture():
    """validate_issue accepts tips/captions for any department, so every
    renderer must show them — otherwise validated content vanishes silently."""
    from manamap.pilot.issue_spec import NO_FURNITURE_DEPARTMENTS

    plan = copy.deepcopy(PLAN)
    furnished = [d for d in plan["departments"]
                 if d["id"] not in NO_FURNITURE_DEPARTMENTS]
    for dept in furnished:
        dept["pilot_tips"] = [{"card": "Sac Outlet",
                               "text": f"Tip for {dept['id']}."}]
        dept["pull_quote"] = f"Quote for {dept['id']}."
    html_out = render(plan=plan)
    for dept in furnished:
        assert f"Tip for {dept['id']}." in html_out, f"{dept['id']} dropped its tip"
        assert f"Quote for {dept['id']}." in html_out, f"{dept['id']} dropped its quote"


def test_the_99_renders_captions_and_tips():
    plan = copy.deepcopy(PLAN)
    for dept in plan["departments"]:
        if dept["id"] == "the-99":
            dept["captions"] = {"Sac Outlet": "**THE OUTLET:** it sacrifices."}
            dept["pilot_tips"] = [{"card": "Payoff Engine", "text": "Deploy it late."}]
    html_out = render(plan=plan)
    assert "THE OUTLET:" in html_out
    assert "Deploy it late." in html_out


def test_newsstand_survives_a_deck_without_a_commander_metric(tmp_path, monkeypatch):
    """One malformed goldfish artifact must not take down the whole rack."""
    import json
    from manamap.pilot import build_index

    decks = tmp_path / "decks"
    base = decks / "d"
    base.mkdir(parents=True)
    (base / "cards.json").write_text(json.dumps(
        {"deck": "d", "cards": [{"name": "X", "is_commander": True}]}))
    # No "commander" key — a deck with nothing flagged as commander.
    (base / "goldfish_metrics.json").write_text(json.dumps({"meta": {}, "metrics": {}}))
    (base / "stacks").mkdir()
    (base / "decisions").mkdir()
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    (manuals / "d.html").write_text("<html></html>")

    monkeypatch.setattr(build_index, "DECKS_DIR", decks)
    monkeypatch.setattr(build_index, "MANUALS_DIR", manuals)
    entries = build_index.gather_entries()
    assert len(entries) == 1 and entries[0]["mean_cast"] is None
    assert "d.html" in build_index.render_index(entries)
