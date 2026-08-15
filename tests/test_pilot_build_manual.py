"""Issue renderer: department completeness, contract integrity, determinism, escaping."""
import copy

from manamap.pilot.build_manual import render_issue
from manamap.pilot.design import esc
from manamap.pilot.issue_spec import (
    DEPARTMENT_BY_ID,
    DEPARTMENT_IDS,
    OPTIONAL_DEPARTMENTS,
)

# Departments every issue carries. An OPTIONAL department renders only when the
# plan opts in, so "every department always renders" stopped being true when the
# first one was added — see `issue_spec.OPTIONAL_DEPARTMENTS`. The tests below
# assert the new contract in BOTH directions, which is stronger than the old one:
# required always renders, optional renders exactly when asked for.
REQUIRED_IDS = [d for d in DEPARTMENT_IDS if d not in OPTIONAL_DEPARTMENTS]

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
            {"name": "Test Commander", "is_commander": True,
             "quantity": 1, "mana_cost": "{3}{R}", "cmc": 4.0,
             "type_line": "Legendary Creature — Goblin", "image": "https://img/cmd.jpg",
             "color_identity": ["R"], "scryfall_uri": "https://scryfall/cmd"},
            {"name": "Sac Outlet", "is_commander": False,
             "quantity": 1, "type_line": "Creature — Goblin", "image": "https://img/sac.jpg"},
            {"name": "Payoff Engine", "is_commander": False,
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

# Real synergy_graph.json entry shape: {partner, score, synergies}. The fixture used
# to invent a "rule" key that matched a renderer bug, so no test ever noticed that
# every chip on every published manual rendered empty. Keep this shaped like the
# artifact — analysis/synergy.py:155 is the producer.
SYNERGY = {
    "Sac Outlet": [
        {"partner": "Payoff Engine", "score": 4,
         "synergies": ["Sacrifice + Death Trigger", "Tokens + Sacrifice"]},
    ]
}

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
        # The whole of Act III, in the one department the three Coach openers
        # merged into. It carries all three of their bodies — the threat boxes
        # here, `threat_assessment` and `matchups` from the prose doc, and the
        # tutor guide from its own artifact — with no `subheads`, which is the
        # fallback path a plan written before that key gets.
        {"id": "at-the-table", "kicker": "READ", "headline": "WHEN THEY TURN",
         "dek": "A dek.",
         "threats": [{"archetype": "Stax", "meter_label": "Threat", "rate": 0.8,
                      "read": "This is the one.", "outs": ["Vandalblast"]}]},
        {"id": "whats-your-play", "kicker": "YOUR MOVE", "headline": "TWO TURNS",
         "dek": "A dek."},
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
    for dept_id in REQUIRED_IDS:
        assert f'id="{dept_id}"' in html_out, f"missing department {dept_id}"


def test_department_titles_render():
    """Every department names itself — except the cover, which wears the masthead."""
    html_out = render()
    for dept_id in REQUIRED_IDS:
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


def test_synergy_chips_render_from_the_graphs_real_keys():
    """Chips read `synergies`; there has never been a `rule` key on this artifact."""
    html_out = render()
    assert '<span class="chip">Sacrifice + Death Trigger</span>' in html_out


def test_synergy_chips_cap_at_two_per_tile():
    busy = {"Sac Outlet": [{"partner": "P", "score": 9,
                            "synergies": ["Aaa", "Bbb", "Ccc", "Ddd"]}]}
    html_out = render(synergy=busy)
    assert html_out.count('<span class="chip">Aaa</span>') == 1
    assert "Ccc" not in html_out and "Ddd" not in html_out


def test_absent_card_roles_says_so_instead_of_rendering_blank_tiles():
    """card_roles is a dict so it cannot route through prose(); it must still TODO.

    A grid of empty blurbs reads as "these need no explanation", not as
    "nobody wrote this yet".
    """
    prose = {k: v for k, v in PROSE.items() if k != "card_roles"}
    html_out = render(prose_doc=prose)
    the_99 = html_out.split('id="the-99"')[1].split("</section>")[0]
    assert "todo" in the_99.lower()


def test_missing_goldfish_does_not_break_build():
    html_out = render(goldfish=None)
    assert 'id="by-the-numbers"' in html_out


def test_no_plan_still_renders_every_department():
    """Departments never vanish silently, even with no issue plan."""
    html_out = render(plan={})
    for dept_id in REQUIRED_IDS:
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
            "name": name, "is_commander": False, "quantity": 1,
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


def test_departments_render_in_canonical_arc_order():
    """Render order mirrors issue_spec.DEPARTMENTS — the single source of truth
    for the STYLEv3 §5 three-act arc."""
    html_out = render()
    positions = [html_out.index(f'id="{dept_id}"') for dept_id in REQUIRED_IDS]
    assert positions == sorted(positions)


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


# ── The sideboard section ────────────────────────────────────────────────

SIDEBOARD = {
    "slug": "test-deck",
    "assessment": "One flex slot for graveyard-light metas.",
    "swaps": [{"in": "Payoff Engine", "out": "Sac Outlet", "role": "draw:engine",
               "when": "against graveyard-light tables",
               "why": "Instant speed matters on the turn that matters.",
               "bracket_delta": {"before": 3, "after": 4}}],
    "opens_lines": [{"cards": ["Payoff Engine", "Sac Outlet"],
                     "why_plausible": "Both present once swapped.",
                     "status": "needs a stack scenario"}],
    "long_term_defaults": [{"card": "Payoff Engine", "verdict": "keep-in-sideboard",
                            "why": "Only better when the meta is graveyard-light."}],
}


# ── Evidence linkification (renderer-provided navigation, STYLEv3 §8.4) ──


def test_prose_stack_references_become_case_links():
    html_out = render(prose_doc={**PROSE, "how_it_wins": "See stack 001 for proof."})
    assert '<a class="xref" href="#case-001">stack 001</a>' in html_out


def test_prose_cr_references_link_to_judges_desk():
    html_out = render(prose_doc={**PROSE, "how_it_wins": "Per CR 603.2h it triggers."})
    assert '<a class="xref" href="#judges-desk">CR 603.2h</a>' in html_out


def test_linkifier_runs_after_escaping():
    """A hostile prose string must be escaped BEFORE linkification."""
    html_out = render(prose_doc={**PROSE,
                                 "how_it_wins": '<script>alert("stack 001")</script>'})
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_cite_blocks_are_never_linkified():
    """Judge's Desk citations keep their verbatim CR text link-free."""
    html_out = render()
    import re as _re
    for cite in _re.findall(r'<div class="cite">.*?</div>', html_out):
        assert "xref" not in cite


def test_judges_desk_cases_are_collapsible_with_backlinks():
    html_out = render()
    assert '<details class="dossier" id="case-001">' in html_out
    assert 'href="#line-001">↩' in html_out


def test_kill_articles_carry_line_anchors():
    assert 'id="line-001"' in render()


def test_floating_contents_button_renders_once():
    html_out = render()
    assert html_out.count('class="toc-float"') == 1


def test_masthead_columnists_render_in_contents():
    html_out = render()
    assert "Ledger" in html_out and "Vera Dictum" in html_out and "Brightside" in html_out


# ── The Flight Plan: acts, bylines, section terminology ──────────────────


def test_acts_cover_every_section_exactly_once():
    """The five acts partition DEPARTMENT_IDS after cover/contents — a section
    outside an act would vanish from the Flight Plan."""
    from manamap.pilot.issue_spec import ACTS

    flattened = [d for _, ids in ACTS for d in ids]
    assert flattened == [d for d in DEPARTMENT_IDS if d not in ("cover", "contents")]


def test_flight_plan_groups_sections_under_act_headers():
    from manamap.pilot.issue_spec import ACTS

    html_out = render()
    for act_title, _ in ACTS:
        assert f'<h3 class="toc-act-title">{act_title}</h3>' in html_out


def test_bylines_render_in_section_heads_and_flight_plan():
    html_out = render()
    assert html_out.count('<div class="byline">by Coach Sunny Brightside</div>') >= 4
    assert '<div class="byline">by Counselor Vera Dictum</div>' in html_out
    assert '<span class="toc-byline">&quot;Ledger&quot; Lin Marginal</span>' in html_out


def test_reader_facing_chrome_never_says_department():
    """The reader's word is Section (grouped into acts); 'Department' is
    internal vocabulary only."""
    html_out = render()
    assert "Department" not in html_out


# ── Card links: tiles, hrefs, hover previews ─────────────────────────────


def test_card_mentions_link_to_their_99_tile_with_preview():
    prose = dict(PROSE, how_it_wins="Lead with Sac Outlet, then Payoff Engine.")
    html_out = render(prose_doc=prose)
    assert '<a class="cardref" href="#card-sac-outlet">Sac Outlet' in html_out
    assert '<img class="card-pop" src="https://img/sac.jpg"' in html_out
    assert 'id="card-sac-outlet"' in html_out  # the tile target exists


def test_commander_mentions_link_to_command_zone_not_a_tile():
    prose = dict(PROSE, how_it_wins="Test Commander leads from the zone.")
    html_out = render(prose_doc=prose)
    assert '<a class="cardref" href="#command-zone">Test Commander' in html_out
    assert 'id="card-test-commander"' not in html_out  # no tile is minted


def test_card_linker_matches_only_whole_names():
    prose = dict(PROSE, how_it_wins="A Sac Outletter is not the Sac Outlet.")
    html_out = render(prose_doc=prose)
    assert "Sac Outletter is not" in html_out  # unlinked: trailing word chars
    assert html_out.count('href="#card-sac-outlet"') >= 1


def test_card_linker_runs_after_escaping_with_hostile_names():
    doc = deck_doc()
    doc["cards"][1]["name"] = 'Evil <script>alert("x")</script>'
    prose = dict(PROSE,
                 how_it_wins='Cast Evil <script>alert("x")</script> early.',
                 card_roles={'Evil <script>alert("x")</script>': "Nasty."})
    html_out = render(deck_doc=doc, prose_doc=prose)
    assert "<script>alert" not in html_out
    assert "&lt;script&gt;" in html_out


def test_cite_blocks_carry_no_card_links():
    import re as _re

    doc = deck_doc()
    stack = verified_stack()
    stack["resolution"]["steps"][0]["citations"][0]["quote"] = (
        "copy it for each other Sac Outlet cast before it")
    html_out = render(deck_doc=doc, stacks=[stack])
    for cite in _re.findall(r'<div class="cite">.*?</div>', html_out, _re.S):
        assert "cardref" not in cite and "xref" not in cite


# ── The three v3.3 sections ──────────────────────────────────────────────


def test_the_tutor_guide_renders_wishes_and_a_no_tutor_fallback():
    guide = {"slug": "x", "assessment": "Wishes.",
             "tutors": [{"card": "Sac Outlet", "targets": [
                 {"scenario": "Turn 3.", "fetch": "Payoff Engine",
                  "why": "It wins."}]}], "gaps": []}
    html_out = render(tutor_guide=guide)
    assert "<b>Fetch:</b>" in html_out
    with_none = render()
    assert "No tutors in this 99" in with_none


def test_art_break_renders_after_sources_say():
    mana = {"lands": {"total": 24, "enters_tapped": 12, "classes": {}},
            "sources": {}, "pips": {}, "on_curve_probability": {},
            "shares": {}, "source_targets": {}, "ramp": {},
            "assumptions": [], "notes": []}
    html_out = render(mana=mana)
    pos_sources = html_out.index('id="sources-say"')
    pos_numbers = html_out.index('id="by-the-numbers"')
    # `BREATHER_AFTER` now holds two ids and The Kill's break comes FIRST in the
    # document, so this must find the break that follows Sources Say rather than
    # the first one on the page. The Kill's own break is asserted below.
    pos_break = html_out.index('class="art-break"', pos_sources)
    assert pos_sources < pos_break < pos_numbers


def test_the_kill_gets_a_breather_because_it_is_the_peak():
    """It is the mid-book peak and it ran straight into the next department's
    opener, so the issue's loudest moment had nowhere to land. A breather after a
    peak is not a rhythm exception — it is what makes it a peak (STYLEv3 6)."""
    from manamap.pilot.issue_spec import BREATHER_AFTER

    assert "the-kill" in BREATHER_AFTER
    # The break is commander art plus one computed Ledger line, so it needs the
    # mana analysis — with none, no break renders anywhere, which is also true of
    # the Sources Say one and is why that test supplies the same fixture.
    html_out = render(mana={"lands": {"total": 24, "enters_tapped": 12, "classes": {}},
                            "sources": {}, "pips": {}, "on_curve_probability": {},
                            "shares": {}, "source_targets": {}, "ramp": {},
                            "assumptions": [], "notes": []})
    pos_kill = html_out.index('id="the-kill"')
    # Whichever department follows — the fixture predates the Act III merge and
    # carries the three sections it replaced, so naming one here would pin the
    # test to a migration that is still in flight.
    pos_next = html_out.index('<section class="dept"', pos_kill + 1)
    assert pos_kill < html_out.index('class="art-break"', pos_kill) < pos_next


# ── The board, stated before anything argues about it ────────────────────
#
# Every stack file carries board / hand / graveyard / mana_available / stack.
# Until the founder read a shipped issue, the renderer emitted exactly ONE of
# them (`question`), so a reader met a hundred-word question about a board they
# had never been shown. These pin the block and the two shapes it must survive.


def test_board_block_renders_every_side_of_the_scenario():
    from manamap.pilot.build_manual import render_board_block
    out = render_board_block({
        "board": {
            "you": ["Sac Outlet (2/2)", "Payoff Engine (enchantment)", "Swamp",
                    "Insect token (already sacrificed to pay the cost)"],
            "opponents": [{"name": "P2", "life": 33, "board": ["a 4/4 creature"]}],
        },
        "hand": ["Dark Ritual"], "graveyard": ["Bloodghast"], "mana_available": "{2}{B}",
        "stack": [{"pos": 0, "object": "Bottom thing"}, {"pos": 1, "object": "Top thing"}],
    })
    for expected in ("Sac Outlet", "Payoff Engine", "Swamp", "P2", "33 life",
                     "a 4/4 creature", "Dark Ritual", "Bloodghast", "{2}{B}"):
        assert esc(expected) in out, expected
    # The cost payment is listed but is NOT on the battlefield — folding it in
    # changes the body count, which is what these engines are bounded by.
    assert "Already paid" in out
    # pos 0 is the BOTTOM (docs/pilot.md), so the reader's first question —
    # what resolves next — must be rendered first.
    assert out.index("Top thing") < out.index("Bottom thing")
    assert "resolves first" in out


def test_board_block_survives_a_decision_scenario():
    """Decisions write `you` as prose and their stack as bare strings; stacks
    write `you` as entries and the stack as dicts. Both are in the corpus."""
    from manamap.pilot.build_manual import render_board_block
    out = render_board_block({
        "board": {"you": "Turn 8, eight lands, Pantlaza out.",
                  "table": "Two blue seats have passed."},
        "stack": ["Eminence trigger — will create one token"],
    })
    assert "Turn 8, eight lands" in out
    assert "Two blue seats have passed." in out
    assert "Eminence trigger" in out


def test_board_block_is_empty_rather_than_a_placeholder():
    from manamap.pilot.build_manual import render_board_block
    assert render_board_block({}) == ""
    assert render_board_block(None) == ""


def test_seat_keys_are_never_shown_to_the_reader():
    """`scenario_facts` emits machine keys because agents read it. The page
    must not carry a snake_case identifier."""
    from manamap.pilot.build_manual import render_board_block
    out = render_board_block({
        "board": {"you": [], "opponent_a": ["no permanents"]},
        "extras": {"life_totals": {"opponent_a": 40}},
    })
    assert "Opponent A" in out and "opponent_a" not in out


def test_no_python_repr_reaches_the_page():
    """The shipped bug: `", ".join(map(str, value))` over a list of dicts put
    `{'seat': 'A — Azorius flash control', 'life': 35}` on the published page."""
    html_out = render()
    for leak in ("{'seat'", "{&#x27;seat&#x27;", "{'life'", "{&#x27;life&#x27;"):
        assert leak not in html_out, leak


def test_the_resolver_brief_lives_in_the_case_file_not_the_read_through():
    """`scenario.question` is authored FOR THE RESOLVER, not for a reader.

    A real one runs 113 words and says "confirm each is a Dinosaur creature card
    and that Cultivate/Mountain/Path go to the bottom in random order" — an
    instruction to an agent, printed verbatim in the middle of the cover story.
    It belongs in Judge's Desk, which is the collapsed record and one tap away.
    """
    html_out = render()
    kill = html_out[html_out.index('id="the-kill"'):html_out.index('id="judges-desk"')]
    desk = html_out[html_out.index('id="judges-desk"'):]
    assert esc("How many copies?") not in kill, "the resolver brief is in the read-through"
    assert esc("How many copies?") in desk, "the resolver brief was dropped, not moved"


# ── Optional departments: the other half of the contract ────────────────


# `OPTIONAL_DEPARTMENTS` is EMPTY as of the two finished migrations, and iterating
# it is how these three tests silently stopped testing anything. They inject a
# synthetic member instead, so the mechanism stays covered whether or not a
# department happens to be mid-pilot today — which is the only state in which a
# bug here would ever ship.
OPTIONAL_UNDER_TEST = "featured-artist"


def _as_optional(monkeypatch, dept_id=OPTIONAL_UNDER_TEST):
    """Declare one real department optional, in both modules that read the set.

    Both do `from .issue_spec import OPTIONAL_DEPARTMENTS`, so the binding to
    patch is the consumer's own global — patching `issue_spec` would not be seen.
    """
    from manamap.pilot import build_manual as bm
    monkeypatch.setattr(bm, "OPTIONAL_DEPARTMENTS", frozenset({dept_id}))
    return dept_id


def test_an_optional_department_is_absent_when_the_plan_omits_it(monkeypatch):
    """`OPTIONAL_DEPARTMENTS` exists so a department can be piloted on one deck.

    A department arriving in `DEPARTMENTS` used to invalidate every issue plan at
    once, which meant a new one landed on nine decks or on none. Optional means an
    older plan without it stays valid — and the renderer has to agree, or an issue
    that never opted in prints an empty department with a [TODO] in it.
    """
    dept_id = _as_optional(monkeypatch)
    html_out = render(plan={})
    assert f'id="{dept_id}"' not in html_out, (
        f"{dept_id} rendered into an issue whose plan does not carry it")


def test_an_optional_department_leaves_no_dead_link_in_the_flight_plan(monkeypatch):
    """The Flight Plan must skip what the body skipped.

    Caught in review rather than by a test: the body correctly omitted the section
    and the contents page went on linking to it, producing a dead anchor in eight
    issues — in the one department whose entire job is telling a reader where
    things are.
    """
    dept_id = _as_optional(monkeypatch)
    html_out = render(plan={})
    assert f'href="#{dept_id}"' not in html_out, (
        f"the Flight Plan links to {dept_id}, which this issue does not carry")


def test_an_optional_department_renders_when_the_plan_asks_for_it(monkeypatch):
    """And the positive case, or the two tests above pass on a renderer that
    dropped optional departments entirely."""
    dept_id = _as_optional(monkeypatch)
    html_out = render(plan={"departments": [
        {"id": dept_id, "kicker": "K", "headline": "H", "dek": "D"}]})
    assert f'id="{dept_id}"' in html_out
    # Through the renderer's own escaper: a title carrying an apostrophe reaches
    # the page as &#x27; and would fail a raw match.
    assert esc(DEPARTMENT_BY_ID[dept_id]["title"]) in html_out


def test_the_optional_set_is_empty_until_something_is_being_piloted():
    """Not a style rule — the constant's own contract, made mechanical.

    "An id should be REMOVED as soon as every deck has it." Five ids have passed
    through here across two migrations and both are finished, so the set is empty
    and all nine issues validate against the full canonical list again. This test
    is meant to FAIL while a department is mid-pilot: delete it for the duration,
    or better, read the failure as the reminder to finish the migration.
    """
    assert OPTIONAL_DEPARTMENTS == frozenset(), (
        f"still piloting {sorted(OPTIONAL_DEPARTMENTS)} — if every deck now "
        f"carries these, take them out of the set; a permanently optional "
        f"department is one nobody committed to")


def test_at_the_table_merges_three_coach_sections_under_one_opener():
    """The Act III merge: one department head, three bodies, no lost content.

    Three consecutive Coach departments were three openers, three bylines and
    three folios answering the same question. The merge has to drop the CHROME
    and keep everything a reader was getting — so this asserts both halves: the
    threat boxes, the matchups prose and the tutor guide all still render, and the
    editor's own sub-headlines survive instead of being replaced by the department
    titles they came from.
    """
    guide = {"slug": "x", "assessment": "Wishes.",
             "tutors": [{"card": "Sac Outlet", "targets": [
                 {"scenario": "Turn 3.", "fetch": "Payoff Engine",
                  "why": "It wins."}]}], "gaps": []}
    plan = dict(PLAN, departments=[
        {"id": "cover"}, {"id": "contents"},
        {"id": "at-the-table", "kicker": "THE READ", "headline": "PAPER CAMOUFLAGE",
         "dek": "A dek.",
         "subheads": {"enemy": {"headline": "ONE HYDRA HOLDS THE SKY"},
                      "tutors": {"headline": "TWO TUTORS, THREE REFUSALS"}},
         "threats": [{"archetype": "Stax", "meter_label": "Threat", "level": 4,
                      "read": "This is the one.", "outs": ["Vandalblast"]}]},
        {"id": "back-page"},
    ])
    html_out = render(plan=plan, tutor_guide=guide)

    # One opener for the whole act. Counting `dept-title` would count every
    # REQUIRED department the renderer emits from the spec regardless of the plan
    # — the merge's claim is about Act III, so ask Act III.
    assert html_out.count('id="at-the-table"') == 1
    # The three it replaced are gone from the SPEC, not merely from this plan —
    # asserting they do not render would now pass on any renderer at all.
    for gone in ("politics-table", "know-your-enemy", "fetch-quests"):
        assert gone not in DEPARTMENT_IDS, (
            f"{gone} is back in DEPARTMENTS; the Act III merge is what deleted it")
        assert f'id="{gone}"' not in html_out

    # ...and every body that used to have its own department still renders.
    assert "Stax" in html_out                    # the threat boxes
    assert "<b>Fetch:</b>" in html_out           # the tutor guide
    assert "ONE HYDRA HOLDS THE SKY" in html_out
    assert "TWO TUTORS, THREE REFUSALS" in html_out
    assert html_out.count('class="act-sub"') == 2


def test_a_merged_department_with_no_subheads_falls_back_to_plain_titles():
    """An `at-the-table` plan that never wrote sub-headlines still reads.

    The fallback matters because the merge is mid-migration: a plan generated
    before the subhead key existed must render section names rather than two
    unlabelled rules in the middle of a department.
    """
    guide = {"slug": "x", "tutors": [], "gaps": []}
    plan = dict(PLAN, departments=[
        {"id": "cover"}, {"id": "contents"},
        {"id": "at-the-table", "kicker": "K", "headline": "H", "dek": "D",
         "threats": [{"archetype": "Stax", "meter_label": "Threat", "level": 2,
                      "read": "r", "outs": ["o"]}]},
        {"id": "back-page"},
    ])
    html_out = render(plan=plan, tutor_guide=guide)
    assert "Know Your Enemy" in html_out and "Fetch Quests" in html_out


# ── Magazine v5: the front of book, the schematic, and the theatre ──────
#
# Five editorial defects the founder found by reading a shipped issue end to end.
# Each test below pins the behaviour that fixed one, and where the fix is visual
# the test reads the STRUCTURE the visuals depend on — a rendered pixel is the
# browser suite's job, but "the selector counts the right children" is not.


def test_a_stack_entry_is_named_under_either_corpus_key():
    """11 entries use `item` where 53 use `object`, and reading only `object`
    printed those eleven as an empty <b></b> — an unnamed thing on the stack, in
    the department that exists to show what is on the stack."""
    from manamap.pilot.build_manual import render_board_block, stack_entry_text

    assert stack_entry_text({"object": "Sol Ring"}) == "Sol Ring"
    assert stack_entry_text({"item": "Craterhoof Behemoth"}) == "Craterhoof Behemoth"
    assert stack_entry_text({}) == "" and stack_entry_text(None) == ""
    out = render_board_block({"stack": [{"pos": 0, "item": "Ambush Hoof"}]})
    assert "Ambush Hoof" in out
    assert "<b></b>" not in out


def test_the_theatre_lights_the_tab_that_matches_the_step():
    """The rail's label was a CHILD of the tab list, so every `:nth-child(I)`
    rule landed one tab early — step 4 showing while tab 3 was lit. The label is
    a sibling now, and this pins it: a tab and its note must share an index."""
    from manamap.pilot.design import _theatre_rules, stack_theatre

    html = stack_theatre("003", [{"action": "a", "effect": "e"}] * 3)
    tabs = html.split('<nav class="th-rail"', 1)[1]
    assert tabs.count('class="th-tab"') == 3
    # The label sits outside the nav; if it ever moves back in, the first child
    # of .th-rail stops being a tab and every generated rule is off by one.
    assert 'class="th-rail-lbl"' in html.split('<nav class="th-rail"', 1)[0]
    rules = _theatre_rules(3)
    assert ".th-railwrap .th-tab:nth-child(2)" in rules
    assert ".th-body .th-note:nth-child(2)" in rules


def test_the_theatre_needs_no_javascript_and_opens_on_a_valid_view():
    """An issue is a standalone printable file with no scripts. The mechanism is
    radio inputs, and step 1 is checked in the MARKUP — so CSS-off, print and
    screen-reader readers get a real first view rather than a blank stage."""
    from manamap.pilot.design import stack_theatre

    html = stack_theatre("005", [{"action": f"step {i}"} for i in range(1, 5)])
    assert "<script" not in html and "onclick" not in html
    assert html.count('type="radio"') == 4
    assert html.count(" checked") == 1 and 'id="th-005-1" checked' in html
    # Every step's prose is in the document, not fetched or revealed by code.
    for i in range(1, 5):
        assert f"step {i}" in html


def test_the_theatre_caps_its_tabs_and_says_so():
    """A silent truncation reads as "that is all of them" — which is the exact
    failure the constellation's unplaced-cards caption was added to prevent."""
    from manamap.pilot.design import THEATRE_MAX_STEPS, stack_theatre

    n = THEATRE_MAX_STEPS + 3
    html = stack_theatre("x", [{"action": f"s{i}"} for i in range(n)])
    assert html.count('class="th-tab"') == THEATRE_MAX_STEPS
    assert html.count('class="th-note"') == n      # every step still prints
    assert "3 further steps" in html


def test_the_theatre_shows_a_card_only_when_the_step_names_one():
    from manamap.pilot.design import stack_theatre

    cards = [{"name": "Craterhoof Behemoth", "image": "hoof.jpg"}]
    html = stack_theatre("003", [
        {"action": "Craterhoof Behemoth resolves and its ETB counts itself."},
        {"action": "State-based actions are checked."},
    ], cards)
    assert "hoof.jpg" in html
    assert "Craterhoof Behemoth" in html
    assert "Step 2" in html          # the unnamed step falls back to its number


def test_the_engine_schematic_labels_what_each_arrow_carries():
    """A block diagram says two stages are related; a schematic says what moves
    between them. Derived labels are marked italic and counted in the caption."""
    from manamap.pilot.design import engine_figure, engine_flow, line_carries

    doc = {
        "stages": [{"stage": "fuel", "label": "BODIES", "cards": ["a", "b"]},
                   {"stage": "wincon", "label": "THE HOOF", "cards": ["c"]}],
        "lines": [{"from": "fuel", "to": "wincon", "verified_by": "003"}],
    }
    assert line_carries({"from": "fuel"}) == ("bodies", False)
    assert line_carries({"from": "fuel", "carries": "counters"}) == ("counters", True)
    svg = engine_flow(doc)
    assert ">bodies<" in svg
    assert "FEEDS IT" in svg and "ENDS IT" in svg      # the triad, named in place
    assert "1 arrow label is set in italic" in engine_figure(doc)


def test_the_schematic_separates_two_lines_between_the_same_pair():
    """radagast declares `fuel -> wincon` twice on two different stacks. Drawn at
    identical coordinates they are one arrow, and the caption's count then
    disagrees with the number a reader can find on the page."""
    from manamap.pilot.design import engine_flow

    doc = {
        "stages": [{"stage": "fuel", "label": "F", "cards": ["a"]},
                   {"stage": "wincon", "label": "W", "cards": ["b"]}],
        "lines": [{"from": "fuel", "to": "wincon", "verified_by": "003"},
                  {"from": "fuel", "to": "wincon", "verified_by": "005"}],
    }
    paths = [p for p in engine_flow(doc).split("<path ")[1:]]
    assert len(paths) == 2
    assert paths[0][:80] != paths[1][:80], "the two arrows are drawn identically"


def test_an_unverified_engine_line_is_still_drawn_dashed():
    """The dashed line is the contract the panel is bound by; the schematic
    rewrite must not have quietly turned every arrow solid."""
    from manamap.pilot.design import engine_flow

    doc = {
        "stages": [{"stage": "mana", "label": "M", "cards": ["a"]},
                   {"stage": "wincon", "label": "W", "cards": ["b"]}],
        "lines": [{"from": "mana", "to": "wincon", "verified_by": None}],
    }
    assert "stroke-dasharray" in engine_flow(doc)


def test_the_game_plan_states_its_conditions_without_being_asked():
    """The rail is emitted by the RENDERER. A department whose promise is "why
    it's going to work" will not volunteer what it assumed away, so the plan is
    not allowed to be the thing that decides whether the caveat runs."""
    from manamap.pilot.design import not_modelled_rail

    out = not_modelled_rail(
        ["The floor loses."],
        [{"question": f"q{i}"} for i in range(5)],
        "10,000 runs of resource development, not of games.",
    )
    assert "The floor loses." in out
    assert out.count("<li>") == 1 + 3 + 1        # authored + capped + scope
    assert "2 further questions" in out
    assert not_modelled_rail([], [], None) == ""


def test_the_card_tile_wears_its_engine_stage_in_the_schematic_ink():
    """The chip and the bay must agree by construction — two palettes for one
    taxonomy is two legends that can never agree."""
    from manamap.pilot.design import ENGINE_STAGE_INK, card_tile

    card = {"name": "Scute Swarm", "image": "s.jpg"}
    lit = card_tile(card, {}, {}, stage="fuel")
    assert ENGINE_STAGE_INK["fuel"] in lit and ">fuel<" in lit
    # No engine model, or a card the model does not place: no chip, nothing else
    # changes. An absent chip is a finding, not a hole.
    plain = card_tile(card, {}, {})
    assert "chip stage" not in plain


def test_the_panel_opens_on_a_hot_take_and_someone_answers_it():
    from manamap.pilot.validate_issue import _hot_take_errors

    good = [
        {"voice": "Coach Sunny Brightside", "kind": "hot-take", "text": "t"},
        {"voice": "Counselor Vera Dictum", "responds_to": "hot-take", "text": "t"},
    ]
    assert _hot_take_errors(good) == []
    assert _hot_take_errors([{"voice": "Coach Sunny Brightside", "text": "t"}])
    # The Coach owns it: it is a ★ judgment and the other two answer it.
    wrong_voice = copy.deepcopy(good)
    wrong_voice[0]["voice"] = "Counselor Vera Dictum"
    assert any("hot take is" in e for e in _hot_take_errors(wrong_voice))
    # A take nobody answers is an epigraph, not a conversation.
    unanswered = copy.deepcopy(good)
    del unanswered[1]["responds_to"]
    assert any("epigraph" in e for e in _hot_take_errors(unanswered))
    # One take, argued with — not three takes in a row.
    twice = copy.deepcopy(good)
    twice[1]["kind"] = "hot-take"
    assert any("second hot take" in e for e in _hot_take_errors(twice))


def test_the_editors_letter_rail_falls_back_to_the_department_deks():
    """Derived beats blank and authored beats derived — but the rail is what
    makes the letter a preview, so it must never be empty."""
    from manamap.pilot.build_manual import letter_teases

    plan = {"departments": [
        {"id": "editors-letter"},
        {"id": "command-zone", "dek": "Meet the wizard."},
        {"id": "first-turns", "dek": "The plan, stated."},
        {"id": "the-99", "dek": "Roll call."},
    ]}
    assert letter_teases(plan) == [
        ("The Command Zone", "Meet the wizard."),
        ("The Game Plan", "The plan, stated."),
        ("The 99", "Roll call."),
    ]
    plan["departments"][0]["in_this_issue"] = [
        {"department": "the-kill", "line": "Four ways it ends."}]
    assert letter_teases(plan) == [("The Kill", "Four ways it ends.")]


def test_the_pilots_log_runs_behind_the_99():
    """The panel is the densest thing in the issue and every move it makes refers
    to material the reader must already have met. Order is encoded in exactly one
    place, so this is the only place it can be asserted."""
    order = DEPARTMENT_IDS
    for earlier in ("command-zone", "first-turns", "the-99"):
        assert order.index(earlier) < order.index("pilots-log"), earlier
    assert order.index("editors-letter") < order.index("command-zone")
    assert order.index("pilots-log") < order.index("keep-or-ship")


def test_short_list_cards_get_a_hover_preview_that_leaves_the_magazine():
    """The Short List's ten are the only card names in an issue with no tile to
    link to — they are, by definition, cards the deck does not run. Before the
    art sidecar they were the only names a reader could not look at, in the one
    department whose whole job is showing you cards you do not own."""
    from manamap.pilot.build_manual import (
        card_linkify, clear_card_links, set_card_links)
    from manamap.pilot.design import esc

    deck = [{"name": "Sol Ring", "image": "sol.jpg", "is_commander": False}]
    art = {"Seedborn Muse": {"image": "muse.jpg",
                             "scryfall_uri": "https://scryfall.com/card/x"}}
    set_card_links(deck, offdeck=art)
    try:
        out = card_linkify(esc("Seedborn Muse beside Sol Ring."))
        assert 'class="cardref offdeck"' in out
        assert 'href="https://scryfall.com/card/x"' in out
        assert 'target="_blank"' in out and 'rel="noopener"' in out
        assert 'src="muse.jpg"' in out
        # The in-deck card keeps its tile link and is NOT marked off-deck.
        assert 'href="#card-sol-ring"' in out
        assert out.count("cardref offdeck") == 1
    finally:
        clear_card_links()


def test_a_short_list_card_the_deck_already_runs_points_at_its_tile():
    """An analyst may list a card the deck has since picked up. In-deck always
    wins: sending a reader off-site for a card printed two pages away is worse
    than no link at all."""
    from manamap.pilot.build_manual import (
        card_linkify, clear_card_links, set_card_links)
    from manamap.pilot.design import esc

    deck = [{"name": "Vigor", "image": "vigor.jpg", "is_commander": False}]
    set_card_links(deck, offdeck={"Vigor": {"image": "other.jpg",
                                            "scryfall_uri": "https://x"}})
    try:
        out = card_linkify(esc("Vigor."))
        assert "offdeck" not in out
        assert 'href="#card-vigor"' in out and 'src="vigor.jpg"' in out
    finally:
        clear_card_links()


def test_the_short_list_art_sidecar_reads_only_the_ten():
    """`natural_cut` names a card that IS in the 99 and is already linked to its
    own tile — resolving it here would mint a second, external link to a card
    the reader can reach on the same page."""
    from manamap.pilot.short_list_art import names_from

    analysis = {"ten": [
        {"card": "Seedborn Muse", "natural_cut": "Fertile Ground"},
        {"card": "Vigor"},
        {"card": "Seedborn Muse"},          # de-duplicated, order preserved
    ]}
    assert names_from(analysis) == ["Seedborn Muse", "Vigor"]
    assert names_from({}) == []


# ── Magazine v6: length is a measured quantity ───────────────────────────
#
# Vol. 009 reached 43,494 words and 74.5 screens of scroll — 62 A4 pages, where a
# real issue is 30–50 including full-page art. These pin the structural cuts, so
# the length cannot drift back the way it drifted there: one department at a
# time, each addition defensible on its own.


def test_the_proof_is_printed_exactly_once():
    """The Kill and Judge's Desk carried the IDENTICAL 120 CR citations and all 73
    rules, because the stack theatre put the full record into the read-through —
    which is the failure mode §5.1 names for that section by name. Judge's Desk
    holds the citations; The Kill holds the walkthrough and points at them."""
    from manamap.pilot.design import stack_theatre

    steps = [{"action": "Cast it.", "effect": "It resolves.",
              "citations": [{"rule": "601.2", "quote": "To cast a spell…"},
                            {"rule": "608.2", "quote": "…the spell resolves."}]}]
    html_out = stack_theatre("003", steps)
    assert "CR 601.2" not in html_out and "To cast a spell" not in html_out
    assert "Cast it." in html_out and "It resolves." in html_out
    # The count survives, so the reader knows a record exists and how big it is.
    assert "2 citations on the record" in html_out


def test_a_case_row_carries_counts_and_never_a_derived_holding():
    """The index says how much record is behind each case. It does NOT say what
    the case held: deriving that from `final_state.summary` was measured against
    the corpus and removed (see the note above render_the_kill), and a wrong
    verdict in the department that exists for correctness is worse than none."""
    desk = render().split('id="judges-desk"', 1)[1]
    assert 'class="case-row"' in desk
    assert 'class="case-id"' in desk and 'class="case-meta"' in desk
    assert "step" in desk and "citation" in desk


def test_a_stack_title_is_split_into_a_headline_and_the_question():
    """Stack titles are written for the resolver — a median of 74 characters and
    up to 157 — and ran three lines of display type at feature size. They almost
    all carry a real headline before a colon; the question becomes the dek."""
    from manamap.pilot.build_manual import stack_headline

    head, tail = stack_headline(
        "The Frostfang trap: flashed in after blockers are declared, does "
        "deathtouch apply to damage already lined up?")
    assert head == "The Frostfang trap"
    assert tail.startswith("flashed in after blockers")
    # Six of the 54 presentable stacks carry no colon; they keep their whole text.
    assert stack_headline("One ping, five payoffs") == ("One ping, five payoffs", "")
    # A leading fragment too short to be a headline is not one — a title like
    # "X: y" must not become the headline "X".
    assert stack_headline("A: something") == ("A: something", "")


def test_the_board_renders_as_a_strip_and_keeps_every_field():
    """It shipped as a grid of bordered cards holding definition lists and came
    to 3,782px on radagast — taller than the stack theatre it introduces. Same
    fields, one line per seat."""
    from manamap.pilot.build_manual import render_board_block

    out = render_board_block({
        "board": {
            "you": ["Sac Outlet (2/2)", "Payoff Engine (enchantment)", "Swamp",
                    "Insect token (already sacrificed to pay the cost)"],
            "opponents": [{"name": "P2", "life": 33, "board": ["a 4/4"]}],
        },
        "hand": ["Dark Ritual"], "graveyard": ["Bloodghast"],
        "mana_available": "{2}{B}",
    })
    assert 'class="seats"' in out and 'class="seat"' in out
    assert "<dl>" not in out
    for kept in ("Sac Outlet", "Payoff Engine", "Swamp", "Dark Ritual",
                 "Bloodghast", "{2}{B}", "P2", "33 life", "a 4/4"):
        assert esc(kept) in out, kept
    # The cost payment keeps its OWN labelled run — folding it into Permanents
    # would change the body count, which is what these engines are bounded by.
    assert "Already paid" in out
