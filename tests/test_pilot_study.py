"""The study: one question, the facts from every source, the agents' answers kept,
the call and who decided. `validate_study` is the gate the pilot asked for
fourteen times: a decision names what decided it, and the goldfish is never that
alone, nor at all on a card it cannot read.

Every test here is proved by the bug it refuses: drop the rule and the test reds.
"""

import json

import pytest

from manamap.pilot import card_dossier, deck_notes, study as st, validate_study as vs


def _cards(*names, commander="Zada, Hedron Grinder"):
    rows = [{"name": commander, "type_line": "Legendary Creature — Goblin", "oracle_text": "", "is_commander": True,
             "quantity": 1, "cmc": 4, "mana_cost": "{3}{R}", "color_identity": ["R"]}]
    for n in names:
        rows.append({"name": n, "type_line": "Instant", "oracle_text": "Counter target spell.", "quantity": 1,
                     "cmc": 2, "mana_cost": "{1}{R}", "color_identity": ["R"], "is_commander": False})
    return rows


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """A deck with a log, a debrief, a passing stack and a branch — no corpus, no
    embeddings, no goldfish declaration: the dossier must list what is absent and
    never crash on it."""
    base = tmp_path / "decks" / "x"
    base.mkdir(parents=True)
    names = ["Haze of Rage", "Swords to Plowshares", "Flawless Maneuver"]
    (base / "decklist.txt").write_text("1 Zada, Hedron Grinder *CMDR*\n" + "".join(f"1 {n}\n" for n in names))
    (base / "cards.json").write_text(json.dumps({"cards": _cards(*names), "decklist_sha256": "a" * 64}))
    (base / "log.jsonl").write_text(
        json.dumps({"id": "001", "at": "2026-11-01T20:00", "result": "loss", "opponents": 3,
                    "decklist_sha256": "a" * 64, "tags": [],
                    "text": "Lost to a wipe. Flawless Maneuver sat in hand; I never had the mana spare."}) + "\n"
        + json.dumps({"id": "002", "at": "2026-11-08T20:00", "result": "win", "opponents": 3,
                      "decklist_sha256": "a" * 64, "tags": [], "text": "Won on Zada. Haze of Rage did the work."}) + "\n")
    (base / "log_annotations.json").write_text(json.dumps({"slug": "x", "entries": {
        "001": {"summary": "s", "cards": [{"card": "Flawless Maneuver", "read": "under", "why": "'never had the mana spare'"},
                                           {"card": "Haze of Rage", "read": "as-expected", "why": "fine"}]},
        "002": {"summary": "s", "cards": [{"card": "Haze of Rage", "read": "over", "why": "'did the work'"}]}}}))
    (base / "log_causes.json").write_text(json.dumps({"slug": "x", "entries": {
        "001": {"cause": "wipe", "note": "", "at": "2026-11-02"}, "002": {"cause": "won", "note": "", "at": "2026-11-09"}}}))
    (base / "stacks").mkdir()
    (base / "stacks" / "001-haze.json").write_text(json.dumps({
        "id": "001", "title": "Haze of Rage on a wide board", "deck": "x",
        "scenario": {"board": {"you": ["Zada, Hedron Grinder"]}, "hand": ["Haze of Rage"], "stack": []},
        "checker": {"verdict": "pass"}}))
    b = base / "branches" / "b1"
    b.mkdir(parents=True)
    (b / "decklist.txt").write_text("1 Zada, Hedron Grinder *CMDR*\n1 Haze of Rage\n1 Swords to Plowshares\n1 Dawn's Truce\n")
    (b / "cards.json").write_text(json.dumps({"cards": _cards("Haze of Rage", "Swords to Plowshares", "Dawn's Truce"),
                                              "decklist_sha256": "b" * 64}))
    (b / "branch.json").write_text(json.dumps({"slug": "x", "branch": "b1", "v": 2, "opened": "2026-11-20",
                                               "why": "protection", "staged": [
                                                   {"at": "2026-11-21", "out": "Flawless Maneuver", "in": "Dawn's Truce",
                                                    "why": "Maneuver is creatures-only"}]}))
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path / "decks")
    # no composer, no goldfish on this deck: the record comes from the log alone
    monkeypatch.setattr("manamap.pilot.deck_info.compose",
                        lambda slug, **kw: {"record": {"games": 2, "win": 1, "loss": 1, "draw": 0,
                                                       "cause_counts": {"wipe": 1, "won": 1}, "undebriefed": [],
                                                       "last_played": "2026-11-08"}})
    return base


# ── the tally and the dossier ─────────────────────────────────────────────────

def test_the_tally_counts_the_debriefs_reads_per_card_across_games_and_never_a_rate(deck):
    t = deck_notes.tally("x")
    assert t["games"] == 2 and t["annotated"] == 2
    fm = t["cards"]["Flawless Maneuver"]
    assert fm["under"] == 1 and fm["games"] == ["001"]
    hz = t["cards"]["Haze of Rage"]
    assert hz["over"] == 1 and hz["as-expected"] == 1 and hz["games"] == ["001", "002"]
    assert not any(isinstance(v, float) for row in t["cards"].values() for v in row.values() if not isinstance(v, list)), (
        "the tally is counts with game ids, never a rate over three games")
    assert t["by_result"]["loss"]["causes"] == {"wipe": 1}


def test_card_dossier_lists_log_reads_per_game_with_the_pilots_words_and_what_is_absent(deck):
    d = card_dossier.dossier("x", "Flawless Maneuver", "b1")
    kinds = [f["kind"] for f in d["facts"]]
    assert "log" in kinds and "card_read" in kinds and "staged" in kinds
    quote = next(f for f in d["facts"] if f["kind"] == "log")
    assert "never had the mana spare" in quote["text"] and quote["game"] == "001" and quote["tier"] == "coach"
    read = next(f for f in d["facts"] if f["kind"] == "card_read")
    assert read["read"] == "under" and read["source"] == "log_annotations.json#001"
    assert d["in_deck"] is True and d["in_branch"] is False
    # the embedding space is either read (a `neighbours` fact) or ABSENT with a reason —
    # never a silent zero; which one depends on whether this checkout has the corpus
    assert (any(f["kind"] == "neighbours" for f in d["facts"])
            or any(a["source"] == "similar-cards" for a in d["absent"]))
    # a passing stack is the one ✓ tier a card can carry
    h = card_dossier.dossier("x", "Haze of Rage")
    stack = next(f for f in h["facts"] if f["kind"] == "stack")
    assert stack["tier"] == "verified" and stack["stack"] == "001"


def test_the_coverage_block_is_the_first_thing_an_instrument_prints(deck):
    text, cov = card_dossier.coverage_block("x", ["Haze of Rage", "Dawn's Truce"], "b1")
    assert text.startswith("* ONE INSTRUMENT") and card_dossier.ASTERISK in text
    assert cov["named"] == 2


# ── the study skeleton and its freeze ─────────────────────────────────────────

def _study(deck):
    return st.new("x", "Dawn's Truce for Flawless Maneuver on b1?", branch="b1",
                  out="Flawless Maneuver", in_="Dawn's Truce")


def test_a_new_study_is_a_computed_skeleton_with_ids_tiers_and_sources(deck):
    doc = _study(deck)
    assert doc["id"] == "001" and doc["call"]["decision"] == "open" and doc["decklist_sha256"]
    ids = [f["id"] for f in doc["facts"]]
    assert ids and len(set(ids)) == len(ids) and all(st.FACT_ID_RE.fullmatch(i) for i in ids)
    assert {f["tier"] for f in doc["facts"]} <= {"verified", "data", "coach"}
    assert doc["facts"][0]["kind"] == "record" and "2 game(s)" in doc["facts"][0]["text"]
    # one staging note names both cards and reaches the study ONCE
    staged = [f for f in doc["facts"] if f["kind"] == "staged"]
    assert len(staged) == 1
    # two studies of one deck never share a fact id
    other = st.new("x", "another question", cards=["Haze of Rage"])
    assert not set(ids) & {f["id"] for f in other["facts"]}
    assert vs.validate(doc, "x") == ([], [])


def test_a_decided_study_is_frozen_unless_forced(deck):
    doc = _study(deck)
    st.write("x", doc)
    st.set_call(doc, "approve", ["pilot", "log"], "do it")
    st.write("x", doc)
    p, _notes = st.install("x", "001")
    frozen = json.loads(p.read_text())
    assert frozen["decided_at"] and st.is_frozen(frozen)
    # the CLI refuses to recompute the record a call rested on
    args = type("A", (), {"slug": "x", "action": "facts", "ref": "001", "write": True, "force": False, "as_json": False})()
    with pytest.raises(SystemExit, match="decided"):
        st.main(args)
    args.force = True
    st.main(args)                                       # forced: allowed, and says so
    # and a late contribution belongs in a new study
    args = type("A", (), {"slug": "x", "action": "add", "ref": "001", "agent": "strategist", "text": "late",
                          "source_file": None, "as_json": False})()
    with pytest.raises(SystemExit, match="decided"):
        st.main(args)


def test_a_study_whose_list_moved_while_it_ran_installs_with_its_citations_intact(deck):
    """THE BENCH FLOW (2026-11-22): a bench deck is edited in place swap by swap as the
    pilot approves each one, so by the call the list is not the one studied. Install
    rebuilt the skeleton on the moved list, every fact id hashes the list, and every
    citation the agents made went dangling — meren-recursion's study 001 could not be
    installed. Proved by re-introducing the bug: always re-skeleton in `install`."""
    doc = st.new("x", "is the deck light on counters?", cards=["Haze of Rage"])
    fid = doc["facts"][0]["id"]
    doc["contributions"].append({"agent": "deck-doctor", "text": f"per {fid}, yes",
                                 "cites": {"facts": [fid], "strategy": [], "rules": []}})
    st.set_call(doc, "approve", ["pilot", "deck-doctor"], "approve, make the swap")
    doc["call"]["reading"] = f"the record {fid} carries it"
    st.write("x", doc)
    studied = doc["decklist_sha256"]
    (deck / "decklist.txt").write_text((deck / "decklist.txt").read_text()
                                       .replace("Swords to Plowshares", "Dawn's Truce"))
    cards = json.loads((deck / "cards.json").read_text())
    cards["decklist_sha256"] = "c" * 64
    (deck / "cards.json").write_text(json.dumps(cards))
    p, _notes = st.install("x", doc["id"])
    frozen = json.loads(p.read_text())
    assert frozen["decklist_sha256"] == studied, "the record names the list that was studied"
    assert frozen["decided_on_decklist_sha256"] not in (None, studied), "and the list the call produced"
    assert fid in {f["id"] for f in frozen["facts"]}, "the facts the agents cited are the facts kept"


# ── the gate ──────────────────────────────────────────────────────────────────

def _decided(deck, by, cards_state=None):
    doc = _study(deck)
    if cards_state:
        for f in doc["facts"]:
            if f["kind"] == "coverage" and f.get("card") in cards_state:
                f["state"] = cards_state[f["card"]]
                f["text"] = f"goldfish: {f['state']}"
    st.set_call(doc, "approve", by, "yes, do it")
    return doc


def test_validate_study_refuses_a_decision_with_no_decided_by(deck):
    doc = _study(deck)
    doc["call"] = {"decision": "approve", "decided_by": [], "pilot_words": "yes"}
    errs, _ = vs.validate(doc, "x")
    assert any("decided_by is empty" in e for e in errs), errs
    doc["call"] = {"decision": "approve", "decided_by": ["pilot"], "pilot_words": ""}
    errs, _ = vs.validate(doc, "x")
    assert any("pilot_words is empty" in e for e in errs), errs


def test_goldfish_may_never_be_the_sole_decider(deck):
    doc = _decided(deck, ["goldfish"], {"Flawless Maneuver": "seen", "Dawn's Truce": "seen"})
    errs, _ = vs.validate(doc, "x")
    assert any("sole decider" in e for e in errs), errs
    doc = _decided(deck, ["goldfish", "log"], {"Flawless Maneuver": "seen", "Dawn's Truce": "seen"})
    errs, _ = vs.validate(doc, "x")
    assert any("only beside `pilot`" in e for e in errs), errs


def test_goldfish_in_decided_by_is_refused_when_a_named_card_is_dark(deck):
    # the fixture's cards have no coverage fact at all (no declaration) — refused for that
    doc = _decided(deck, ["pilot", "goldfish"])
    errs, _ = vs.validate(doc, "x")
    assert any("goldfish" in e for e in errs), errs
    # give both cards a coverage fact: DARK is refused, SEEN is admitted
    for card in ("Flawless Maneuver", "Dawn's Truce"):
        doc["facts"].append({"id": st.fact_id("x", "001", doc["decklist_sha256"], 90 + len(card)), "kind": "coverage",
                             "text": "goldfish: dark", "tier": "data", "source": "model_coverage.card_state",
                             "card": card, "state": "dark"})
    errs, _ = vs.validate(doc, "x")
    assert any("cannot read" in e for e in errs), errs
    for f in doc["facts"]:
        if f["kind"] == "coverage":
            f["state"] = "seen"
    errs, _ = vs.validate(doc, "x")
    assert errs == [], errs


def test_an_uncited_figure_in_the_reading_is_refused_and_ids_are_not_numbers(deck):
    doc = _decided(deck, ["pilot", "log"])
    rec = doc["facts"][0]["id"]
    doc["call"]["reading"] = f"two games, one lost to a wipe ({rec})"
    assert vs.validate(doc, "x")[0] == []
    doc["call"]["reading"] = f"the deck wins 73% of games ({rec})"
    errs, _ = vs.validate(doc, "x")
    assert any("73" in e and "not in any cited fact" in e for e in errs), errs
    doc["call"]["reading"] = "see S-deadbeef-01"
    errs, _ = vs.validate(doc, "x")
    assert any("not a fact of this study" in e for e in errs), errs
    # the digits inside a fact id are never read as a figure (this was the first bug)
    doc["call"]["reading"] = f"as the record says ({rec})"
    assert vs.validate(doc, "x")[0] == []


def test_a_rules_cite_outside_RULE_ID_RE_and_an_ungrounded_strategist_are_refused(deck):
    doc = _decided(deck, ["pilot", "strategist"])
    st.add_contribution(doc, "strategist", "Protection first (strategy:multiplayer.commander-insurance).")
    st.add_contribution(doc, "rules-question", "Yes: 614.5 and 616.1 govern it; ruling 2023-04-14 agrees.")
    errs, _ = vs.validate(doc, "x")
    assert errs == [], errs
    c = doc["contributions"][0]["cites"]
    assert c["strategy"] == ["strategy:multiplayer.commander-insurance"]
    assert doc["contributions"][1]["cites"]["rules"] == ["614.5", "616.1"], "a log id like 003 or a year is never a rule"
    doc["contributions"][1]["cites"]["rules"] = ["61.4"]
    errs, _ = vs.validate(doc, "x")
    assert any("not a CR rule number" in e for e in errs), errs
    st.add_contribution(doc, "strategist", "Just do it, it feels right.")
    errs, _ = vs.validate(doc, "x")
    assert any("no strategy:<id> citation" in e for e in errs), errs
    with pytest.raises(SystemExit, match="not one of"):
        st.add_contribution(doc, "goldfish", "I am not an agent.")


def test_a_backfilled_study_reads_an_uncited_answer_as_a_note_and_a_new_one_as_an_error(deck):
    """The hand-saved answers of 2026-11-20/21 predate the citation contract; a study
    opened since holds its agents to it."""
    doc = _decided(deck, ["pilot", "strategist"])
    st.add_contribution(doc, "strategist", "Just do it, it feels right.")
    errs, notes = vs.validate(doc, "x")
    assert any("no strategy:<id> citation" in e for e in errs) and not notes
    doc["backfilled"] = True
    errs, notes = vs.validate(doc, "x")
    assert errs == [] and any("no strategy:<id> citation" in n for n in notes)


def test_the_swap_must_name_a_card_the_list_or_the_branch_actually_had(deck):
    doc = _decided(deck, ["pilot"])
    assert vs.validate(doc, "x")[0] == []
    doc["swap"]["out"] = "Black Lotus"
    errs, _ = vs.validate(doc, "x")
    assert any("Black Lotus" in e and "neither the deck" in e for e in errs), errs


def test_validate_study_main_signals_failure_through_the_exit_code(deck, capsys):
    doc = _decided(deck, ["pilot"])
    doc["call"]["decided_by"] = []
    st.write("x", doc)
    with pytest.raises(SystemExit) as exit_:
        vs.main(type("A", (), {"slug": "x"})())
    assert exit_.value.code == 1 and "FAIL" in capsys.readouterr().out


def test_study_is_a_read_only_action_on_the_warm_worker():
    """`study` writes on new/facts/add/call/install, so only `list` and `show` may
    run over the API; the gate pins every other action shut."""
    from manamap import serve
    assert "study" in serve.CLI_READONLY and "card-dossier" in serve.CLI_READONLY
    assert serve.CLI_READONLY_ACTIONS["study"] == frozenset({None, "list", "show"})
