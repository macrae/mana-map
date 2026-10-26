"""`forge-cast-check`: prove the Forge AI plays a card before a branch depends on it.

The counting is driven through the production parser on a synthetic log in Forge's own
line shapes (the telemetry patch's owner-bearing zone lines), and the shell is built from
the real edgar-vampires deck. The bug this exists for: Toxic Deluge, drawn 28 times and
cast 0 across a 200-game branch arm, read as "castable" because its script was unflagged.
"""
import pytest

from manamap import config
from manamap.sim import cast_check as cc

from conftest import requires_corpus, requires_deck

OURS = "Ai(1)-mm-castcheck-edgar-vampires"
OPP = "Ai(2)-mm-giada-angels"
HEAD = [f"Mulligan: {OURS} has kept a hand of 7 cards", f"Mulligan: {OPP} has kept a hand of 7 cards"]
TAIL = ["Game Outcome: Turn 9", f"Game Outcome: {OPP} has won because all opponents have lost",
        "Game Result: Game 1 ended in 100 ms."]


def _game(lines):
    return "\n".join(HEAD + lines + TAIL) + "\n"


def _held_game():
    """Deluge drawn on turn 1, three lands by our turn 5, never cast, held at the end."""
    return _game([
        f"Turn: Turn 1 ({OURS})", f"Zone Change: Toxic Deluge (211) was put into Hand from Library. owner {OURS}",
        f"Land: {OURS} played Swamp (1)", f"Turn: Turn 2 ({OPP})",
        f"Turn: Turn 3 ({OURS})", f"Land: {OURS} played Swamp (2)", f"Turn: Turn 4 ({OPP})",
        f"Turn: Turn 5 ({OURS})", f"Land: {OURS} played Swamp (3)", f"Turn: Turn 6 ({OPP})",
        f"Turn: Turn 7 ({OURS})", f"Land: {OURS} played Swamp (4)", f"Turn: Turn 8 ({OPP})"])


def _cast_game():
    return _game([
        f"Turn: Turn 1 ({OURS})", f"Zone Change: Toxic Deluge (211) was put into Hand from Library. owner {OURS}",
        f"Land: {OURS} played Swamp (1)", f"Turn: Turn 2 ({OPP})",
        f"Turn: Turn 3 ({OURS})", f"Land: {OURS} played Swamp (2)", f"Turn: Turn 4 ({OPP})",
        f"Turn: Turn 5 ({OURS})", f"Land: {OURS} played Swamp (3)",
        f"Add To Stack: {OURS} cast Toxic Deluge (211)",
        f"Zone Change: Toxic Deluge (211) was put into Stack from Hand. owner {OURS}",
        f"Resolve Stack: Toxic Deluge (211) - All creatures get -X/-X until end of turn.",
        f"Zone Change: Toxic Deluge (211) was put into Graveyard from Stack. owner {OURS}",
        f"Turn: Turn 6 ({OPP})"])


def test_a_held_card_is_counted_as_castable_and_uncast_and_the_verdict_says_held():
    c = cc.count(_held_game(), "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["games"] == 1 and c["drawn_games"] == 1 and c["drawn"] == 1
    assert c["cast"] == 0 and c["activated"] == 0 and c["discarded"] == 0
    assert c["castable_uncast_turns"] == 2 and c["held_castable_games"] == 1 and c["held_at_end_games"] == 1
    assert cc.verdict(c).startswith("HELD:")


def test_a_cast_card_is_counted_and_the_verdict_says_played():
    c = cc.count(_cast_game() + _held_game(), "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["games"] == 2 and c["cast"] == 1 and c["drawn_games"] == 2
    assert c["castable_uncast_turns"] == 2, "the held game's turns, not the cast game's turn of casting"
    assert cc.verdict(c).startswith("PLAYED: cast 1")


def test_the_opponents_copy_is_not_ours():
    """The bug: counting every `cast Toxic Deluge` line — jarad runs it too, and the branch
    arm's logs carried the opponent's casts beside our zero."""
    text = _game([f"Turn: Turn 1 ({OPP})", f"Zone Change: Toxic Deluge (99) was put into Hand from Library. owner {OPP}",
                  f"Add To Stack: {OPP} cast Toxic Deluge (99)", f"Turn: Turn 2 ({OURS})"])
    c = cc.count(text, "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["cast"] == 0 and c["drawn_games"] == 0
    assert cc.verdict(c).startswith("NOT DRAWN")


@requires_corpus
@requires_deck
@pytest.mark.skipif(not (config.DECKS_DIR / "edgar-vampires" / "cards.json").exists(), reason="requires edgar-vampires")
def test_the_shell_is_the_decks_commander_copies_filler_and_basics_of_the_cards_colours():
    text, facts = cc.shell_decklist("edgar-vampires", "Toxic Deluge", copies=4)
    lines = text.splitlines()
    assert lines[:2] == ["Commander:", "1 Edgar Markov"] and "4 Toxic Deluge" in lines
    assert facts["lands"] == 36 and facts["basics"] == {"Swamp": 36}, "Deluge is mono-black; its basics are Swamps"
    assert facts["copies"] + facts["filler"] + facts["lands"] == 99 and facts["cmc"] == 3.0
    with pytest.raises(SystemExit):
        cc.shell_decklist("edgar-vampires", "Counterspell")          # outside the identity
    with pytest.raises(SystemExit):
        cc.shell_decklist("edgar-vampires", "Not A Card")


# ── the gate (2026-10-01): diagnosis, CAST-LATE, the proof file, the validator ───────────

from manamap.sim import forge_cards as _fc

needs_forge = pytest.mark.skipif(not _fc.installed(), reason="requires a Forge install (card scripts)")


@needs_forge
def test_the_diagnosis_names_the_known_classes_on_the_real_scripts():
    """Each class was found after a night of games: Vish Kal and Deluge (X priced before
    its cost, no IsCurse), Altar / Teferi's Protection / Swat (no AI logic for the shape),
    Bastion (a cheap do-nothing-now permanent). SHIPPED scripts carry the defects; the
    INSTALLED ones carry the hints, so a hint already applied is not reported."""
    shipped = {n: cc.diagnose(n, installed_copy=False)["classes"] for n in
               ("Toxic Deluge", "Vish Kal, Blood Arbiter", "Altar of Dementia", "Teferi's Protection",
                "Deflecting Swat", "Bastion of Remembrance", "Vein Ripper")}
    assert {"x-priced-before-cost", "no-iscurse"} <= set(shipped["Toxic Deluge"])
    assert "no-iscurse" in shipped["Vish Kal, Blood Arbiter"]
    assert shipped["Altar of Dementia"] == ["removedeck-all", "no-ai-logic"]
    assert shipped["Teferi's Protection"] == ["no-ai-logic"]
    assert shipped["Deflecting Swat"] == ["no-ai-logic", "reactive"]
    assert shipped["Bastion of Remembrance"] == ["permanent-cast-priority"]
    assert shipped["Vein Ripper"] == ["unknown"], "a card with no defect says unknown, never guesses"
    d = cc.diagnose("Toxic Deluge")
    assert any("IsCurse" in r for r in d["remedies"]) and any("PumpAllAi" in r for r in d["remedies"])
    assert cc.diagnose("Not A Card")["classes"] == ["no-script"]


def _script(monkeypatch, text):
    from manamap.sim import forge_cards
    monkeypatch.setattr(forge_cards, "script", lambda card, installed_copy=True: text)


def test_an_authors_gate_is_named_with_the_drop_svars_remedy(monkeypatch):
    """Wheel of Fortune read "no known class" with 0 casts in 5 drawn games (answers-v1,
    2026-10-25): its script's NeedsToPlayVar (two or fewer other cards in hand) never
    opened in the shell. Proven by deleting the SVar line: the class goes."""
    wheel = ("Name:Wheel of Fortune\nTypes:Sorcery\n"
             "A:SP$ Discard | Mode$ Hand | Defined$ Player | SubAbility$ DBEachDraw\n"
             "SVar:NeedsToPlayVar:Y LE2\n")
    _script(monkeypatch, wheel)
    d = cc.diagnose("Wheel of Fortune")
    assert d["classes"] == ["author-gate"]
    assert any("drop_svars" in r and "Y LE2" in r for r in d["remedies"])
    _script(monkeypatch, wheel.replace("SVar:NeedsToPlayVar:Y LE2\n", ""))
    assert cc.diagnose("Wheel of Fortune")["classes"] == ["unknown"]


@pytest.mark.parametrize("line,types,want", [
    ("A:SP$ Counter | TargetType$ Spell | ValidTgts$ Card.nonCreature", "Instant", True),
    ("A:SP$ Pump | ValidTgts$ Creature | KW$ Hexproof & Indestructible", "Instant", True),
    ("A:SP$ Pump | ValidTgts$ Creature | NumAtt$ +3", "Instant", False),
    ("A:SP$ Pump | ValidTgts$ Creature | KW$ Hexproof", "Sorcery", False),
])
def test_an_answer_only_card_is_reactive_and_a_pump_is_not(monkeypatch, line, types, want):
    """Negate, Stubborn Denial, Loran's Escape and Deflecting Swat read HELD on
    sharknado with nothing to answer (2026-10-25); a +3/+0 trick is not an answer."""
    _script(monkeypatch, f"Name:X\nTypes:{types}\n{line}\n")
    assert ("reactive" in cc.diagnose("X")["classes"]) is want


def test_the_documented_limitations_replace_a_wrong_remedy(monkeypatch):
    """Ghostly Pilferer's self-unblockable Effect was told to take the `Protection`
    logic, which would cast it against a wipe; March's X phase-out had no class at all.
    Both are documented limitations (the pilot's call, 2026-10-25), never patched."""
    _script(monkeypatch, "Name:Ghostly Pilferer\nTypes:Creature Spirit Rogue\n"
            "A:AB$ Effect | Cost$ Discard<1/Card> | RememberObjects$ Self | StaticAbilities$ Unblockable\n"
            "SVar:Unblockable:Mode$ CantBlockBy | ValidAttacker$ Card.IsRemembered\n")
    d = cc.diagnose("Ghostly Pilferer")
    assert d["remedies"] == [cc.LIMITATIONS["self-unblockable-effect"]]
    _script(monkeypatch, "Name:March of Swirling Mist\nTypes:Instant\n"
            "A:SP$ Phases | ValidTgts$ Creature | TargetMax$ X\nSVar:X:Count$xPaid\n")
    d = cc.diagnose("March of Swirling Mist")
    assert cc.LIMITATIONS["phases-x"] in d["remedies"] and "reactive" in d["classes"]


def test_cast_late_is_bastions_shape_and_played_needs_a_play():
    late = {"games": 8, "drawn_games": 8, "drawn": 8, "cast": 2, "activated": 0, "triggered": 5, "discarded": 1,
            "castable_uncast_turns": 20, "held_castable_games": 6, "held_at_end_games": 5, "timeouts": 0}
    assert cc.verdict_word(late) == "CAST-LATE"
    assert cc.verdict_word({**late, "cast": 5}) == "PLAYED", "cast in more than half the games it was drawn"
    assert cc.verdict_word({**late, "castable_uncast_turns": 4}) == "PLAYED", "two slow casts are not a pattern"
    assert cc.verdict_word({**late, "cast": 0}) == "HELD"
    assert cc.verdict_word({**late, "cast": 0, "held_castable_games": 0, "castable_uncast_turns": 0}) == "UNPLAYED"
    assert set(cc.VERDICTS) == {"PLAYED", "CAST-LATE", "HELD", "UNPLAYED", "NOT DRAWN"}


def _proof_row(word, cast=1):
    return {"drawn_games": 8, "drawn": 9, "cast": cast, "activated": 0, "triggered": 0, "discarded": 0,
            "castable_uncast_turns": 3, "held_castable_games": 1, "held_at_end_games": 0,
            "verdict": f"{word}: …", "verdict_word": word, "classes": ["unknown"]}


def _proofs(tmp_path, monkeypatch, harness, rows):
    import json
    from manamap import config
    decks = tmp_path / "decks"; b = decks / "d" / "branches" / "x"; b.mkdir(parents=True, exist_ok=True)
    (decks / "d" / "decklist.txt").write_text("Commander:\n1 Edgar Markov\n\nDeck:\n1 Sol Ring\n1 Swamp\n")
    (b / "decklist.txt").write_text("Commander:\n1 Edgar Markov\n\nDeck:\n1 Toxic Deluge\n1 Vein Ripper\n")
    doc = {"slug": "d", "branch": "x", "as_of": "2026-10-01", "harness": harness,
           "shell": {"vs": "giada-angels", "games": 8, "copies": 4, "clock": 300, "seed": 4343},
           "cards": rows, "limits": ["a shell"]}
    (b / cc.PROOFS).write_text(json.dumps(doc))
    monkeypatch.setattr(config, "DECKS_DIR", decks); monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    return doc


HARNESS = {"overrides": "aaaa", "profile": "mm-d", "profile_sha": "bbbb", "patches": "cccc", "jar": "j.jar", "forge": {"version": "2.0.14"}}


def test_the_gate_reads_every_add_against_the_current_harness(tmp_path, monkeypatch):
    """A proof is a measurement under one harness. The bug this guards: accepting a PLAYED
    row taken under another overrides sha — the branch arm would then pair with a
    champion the proof never saw."""
    _proofs(tmp_path, monkeypatch, HARNESS, {"Toxic Deluge": _proof_row("HELD", cast=0), "Vein Ripper": _proof_row("PLAYED")})
    s = cc.status("d", "x", HARNESS)
    assert s["proven"] == ["Vein Ripper"] and s["held"] == ["Toxic Deluge"] and s["unproven"] == [] and s["harness_matches"] is True
    stale = cc.status("d", "x", {**HARNESS, "overrides": "ffff"})
    assert stale["unproven"] == ["Toxic Deluge", "Vein Ripper"] and stale["proven"] == [] and stale["harness_matches"] is False
    _proofs(tmp_path, monkeypatch, HARNESS, {"Vein Ripper": _proof_row("CAST-LATE", cast=2)})
    s2 = cc.status("d", "x", HARNESS)
    assert s2["late"] == ["Vein Ripper"] and s2["unproven"] == ["Toxic Deluge"]
    assert cc.adds("d", "x") == ["Toxic Deluge", "Vein Ripper"]


def test_the_validator_passes_a_good_file_and_fails_the_broken_shapes(tmp_path, monkeypatch):
    from manamap.pilot import validate_cast_proofs as v
    doc = _proofs(tmp_path, monkeypatch, HARNESS, {"Toxic Deluge": _proof_row("HELD", cast=0), "Vein Ripper": _proof_row("PLAYED")})
    errs, warns = v.validate("d", "x", doc)
    assert errs == [] and warns == []
    import json
    bad = json.loads(json.dumps(doc))
    bad["harness"].pop("overrides")
    bad["cards"]["Vein Ripper"]["verdict_word"] = "MAYBE"
    bad["cards"]["Toxic Deluge"]["cast"] = 3                       # HELD with plays
    bad["cards"]["Sol Ring"] = _proof_row("PLAYED", cast=0)        # PLAYED with none, and not an add
    errs, warns = v.validate("d", "x", bad)
    assert any("harness stamp lacks" in e for e in errs) and any("not in" in e for e in errs) \
        and any("HELD with 3" in e for e in errs) and any("PLAYED with zero" in e for e in errs)
    assert any("no longer adds" in w for w in warns)
    errs2, _ = v.validate("d", "y", doc)
    assert any("lives under" in e for e in errs2)


# ── the gate in simulate, the record block, the post-hoc reader ─────────────────────────

def _args(**kw):
    base = dict(slug="d@x", opponents=["giada-angels"], games=1, jobs=1, clock=60, seed=None, force=False,
                dry_run=False, profile=None, pod=None, list=False, analyze=None, detect=None, anyway=False, vs=None)
    base.update(kw)
    return type("Args", (), base)()


class Ran(Exception):
    """Raised by the stubbed runner the moment simulate would have started the JVMs."""


def _stub_forge(monkeypatch):
    """simulate's main with no JVM: the table resolves to one seat, the power preflight
    is silent, and `run` records the kwargs it was handed instead of playing games."""
    from manamap.sim import forge, power
    seen = {}
    def fake_run(slug, opponents, **kw):
        seen.update(kw); seen["slug"] = slug
        raise Ran()                       # the arm "started"; the post-run tail is not under test
    monkeypatch.setattr(forge, "run", fake_run)
    monkeypatch.setattr(forge, "resolve_table", lambda args: (args.opponents, None))
    monkeypatch.setattr(power, "null_rate", lambda pod: (None, None))
    monkeypatch.setattr(power, "baseline_rate", lambda slug, opps: (0.25, None))
    monkeypatch.setattr(power, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(power, "refuse_if_underpowered", lambda *a, **k: None)
    # the post-run tail reads the record for the engine-casts print; make it a no-op
    monkeypatch.setattr(forge, "_print_engine_casts", lambda *a, **k: None, raising=False)
    return seen


def test_simulate_refuses_a_branch_arm_with_a_held_or_unproven_add_and_anyway_records_the_floor(tmp_path, monkeypatch, capsys):
    """THE GATE. The bug this guards is the whole finding: a five-hour arm on a list whose
    add the AI never plays, found afterwards. Held -> refused; unproven -> refused; a
    proof under another harness -> refused as unproven; --anyway -> runs, and the record
    block names the slots as floors."""
    from manamap.sim import forge, cast_check
    _proofs(tmp_path, monkeypatch, HARNESS, {"Toxic Deluge": _proof_row("HELD", cast=0), "Vein Ripper": _proof_row("PLAYED")})
    monkeypatch.setattr(cast_check, "current_harness", lambda slug, branch=None, profile=None: dict(HARNESS))
    seen = _stub_forge(monkeypatch)
    with pytest.raises(SystemExit) as e:
        forge.main(_args())
    assert "REFUSED" in str(e.value) and "Toxic Deluge is HELD" in str(e.value) and "forge-cast-check d --branch x --adds" in str(e.value)
    assert "slug" not in seen, "the arm must not have started"
    # a proof taken under another harness is UNPROVEN, not played
    monkeypatch.setattr(cast_check, "current_harness", lambda slug, branch=None, profile=None: {**HARNESS, "patches": "zzzz"})
    with pytest.raises(SystemExit) as e2:
        forge.main(_args())
    assert "UNPROVEN under this harness" in str(e2.value) and "ANOTHER harness" in str(e2.value)
    # --anyway: the arm runs and the record carries the block
    monkeypatch.setattr(cast_check, "current_harness", lambda slug, branch=None, profile=None: dict(HARNESS))
    with pytest.raises(Ran):
        forge.main(_args(anyway=True))
    assert seen["slug"] == "d@x" and seen["cast_proofs"]["held"] == ["Toxic Deluge"] and seen["cast_proofs"]["anyway"] is True
    assert "RUNNING ANYWAY" in capsys.readouterr().out
    # every add PLAYED -> one line and the arm runs, cast_proofs says so
    _proofs(tmp_path, monkeypatch, HARNESS, {"Toxic Deluge": _proof_row("PLAYED"), "Vein Ripper": _proof_row("PLAYED")})
    seen.clear()
    with pytest.raises(Ran):
        forge.main(_args())
    assert seen["cast_proofs"]["proven"] == ["Toxic Deluge", "Vein Ripper"] and not seen["cast_proofs"]["held"]
    assert "2/2 adds PLAYED" in capsys.readouterr().out
    # a DECK seat is not gated
    seen.clear()
    with pytest.raises(Ran):
        forge.main(_args(slug="d"))
    assert seen["slug"] == "d" and seen["cast_proofs"] is None


def test_validate_sim_checks_the_block_only_where_present():
    from manamap.sim import validate_sim
    base = {"proven": ["A"], "held": [], "late": [], "unproven": [], "as_of": "2026-10-01",
            "harness_matches": True, "harness": {}, "anyway": False}
    def errs(block):
        return validate_sim._cast_proofs_errors({"cast_proofs": block})
    assert errs(None) == [] and errs(base) == []
    assert any("wrong shape" in e for e in errs({"proven": []}))
    assert any("unless --anyway" in e for e in errs({**base, "held": ["Toxic Deluge"]}))
    assert errs({**base, "held": ["Toxic Deluge"], "anyway": True}) == []


def test_the_post_hoc_reader_labels_an_arm_run_before_the_gate_existed(tmp_path, monkeypatch):
    """drain-v1's arm ran before the gate: its record has no block, but engine_casts says
    Toxic Deluge was in hand in 28 games and cast 0, and Bastion was cast 10 of 30 while
    castable on 48 turns. The reader labels both from the record alone."""
    from manamap.pilot import net_change
    from manamap.sim import forge, cast_check
    _proofs(tmp_path, monkeypatch, HARNESS, {"Toxic Deluge": _proof_row("HELD", cast=0), "Vein Ripper": _proof_row("PLAYED")})
    rec = {"run_id": "r1", "cast_proofs": None,
           "engine_casts": {"by_card": {"Toxic Deluge": {"cast": 0, "activated": 0, "discarded": 0, "in_hand_games": 28, "castable_uncast": 84},
                                        "Vein Ripper": {"cast": 11, "activated": 0, "discarded": 5, "in_hand_games": 26, "castable_uncast": 14}}}}
    monkeypatch.setattr(forge, "list_runs", lambda seat: [rec])
    got = net_change._cast_proofs_from_runs("d", "x", ["r1"])
    assert got["held"] == ["Toxic Deluge"] and got["late"] == [] and got["floor"] is True and "FLOOR" in got["reads_as"]
    assert got["gated_runs"] == [] and got["adds"] == ["Toxic Deluge", "Vein Ripper"]
    rec2 = {"run_id": "r2", "cast_proofs": {"proven": ["Toxic Deluge", "Vein Ripper"], "held": [], "late": [], "unproven": [],
                                             "as_of": "2026-10-01", "harness_matches": True, "harness": {}, "anyway": False},
            "engine_casts": {"by_card": {"Toxic Deluge": {"cast": 3, "activated": 0, "discarded": 0, "in_hand_games": 5, "castable_uncast": 2}}}}
    monkeypatch.setattr(forge, "list_runs", lambda seat: [rec2])
    got2 = net_change._cast_proofs_from_runs("d", "x", ["r2"])
    assert got2["floor"] is False and got2["gated_runs"] == ["r2"]
    assert net_change._cast_proofs_from_runs("d", "x", []) is None


def test_run_many_reports_each_shell_as_it_finishes_and_counts_games_on_the_band(tmp_path, monkeypatch):
    """Live, not blind (2026-10-18). Before this, `run_many` buffered every shell and the
    command printed after the last one — fourteen shells of silence. Now every shell gets
    its own log file the band counts finished games from, and `on_done` fires per shell in
    COMPLETION order with its index. Proven by a fake `run` that finishes out of order."""
    import json as _json
    import pathlib
    import time as _time
    from manamap import progress as _progress
    monkeypatch.setattr(_progress, "DIR", tmp_path / "band")
    monkeypatch.setattr(cc, "read_proofs", lambda slug, branch: None)
    monkeypatch.setattr(cc, "current_harness", lambda slug, branch=None, profile=None: dict(HARNESS))
    seen_logs = {}

    def fake_run(slug, card, branch=None, log_path=None, games=8, **kw):
        assert log_path is not None, "a shell must stream its log for the band to count"
        seen_logs[card] = pathlib.Path(log_path)
        _time.sleep(0.3 if card == "Alpha" else 0.0)      # Alpha finishes LAST
        pathlib.Path(log_path).write_text("".join(f"Game Result: Game {g} ended in a win\n" for g in range(1, games + 1)))
        return {"card": card, "verdict_word": "PLAYED", "counts": {"cast": 1}, "diagnosis": {"classes": []}}, "log"
    monkeypatch.setattr(cc, "run", fake_run)
    order = []
    kept, docs, current = cc.run_many("d", ["Alpha", "Beta"], branch="x", jobs=2, games=8,
                                      on_done=lambda card, doc, i, n: order.append((card, i, n)))
    assert order == [("Beta", 1, 2), ("Alpha", 2, 2)], "rows fire as shells finish, not in list order"
    assert set(docs) == {"Alpha", "Beta"} and kept == {}
    assert len({p.parent for p in seen_logs.values()}) == 1 and len(set(seen_logs.values())) == 2
    band = list((tmp_path / "band").glob("cast-check-*.json"))
    assert len(band) == 1
    doc = _json.loads(band[0].read_text())
    assert doc["unit"] == "games" and doc["total"] == 16 and doc["done"] == 16 and doc["state"] == "passed"
    assert doc["label"] == "forge-cast-check d@x" and "2/2 cards" in doc["detail"]


def test_run_many_removes_its_shell_logs_when_done(tmp_path, monkeypatch):
    """The streamed logs exist for the band; the proof carries the counts. A run that left
    them would pile up a few hundred KB per card per run under the system temp dir."""
    import pathlib
    from manamap import progress as _progress
    monkeypatch.setattr(_progress, "DIR", tmp_path / "band")
    monkeypatch.setattr(cc, "read_proofs", lambda slug, branch: None)
    monkeypatch.setattr(cc, "current_harness", lambda slug, branch=None, profile=None: dict(HARNESS))
    seen = {}

    def fake_run(slug, card, branch=None, log_path=None, games=8, **kw):
        seen["dir"] = pathlib.Path(log_path).parent
        pathlib.Path(log_path).write_text("Game Result: Game 1 ended in a win\n")
        return {"card": card, "verdict_word": "PLAYED", "counts": {}, "diagnosis": {"classes": []}}, "log"
    monkeypatch.setattr(cc, "run", fake_run)
    cc.run_many("d", ["Alpha"], branch="x", jobs=1, games=1)
    assert seen["dir"].name.startswith("mm-castcheck-") and not seen["dir"].exists()


def test_a_land_add_is_listed_and_never_proven_held_or_unproven(tmp_path, monkeypatch, capsys):
    """2026-10-18, torbran-v1: four lands and Command Beacon read HELD (drawn 12-14 games,
    cast 0) because Forge PLAYS a land and never casts it — the shell cannot read a land
    drop, and the gate would have refused the arm over a Verge. Lands are named, left out
    of the proof set and out of the gate; a branch that adds only lands has nothing to prove."""
    import json
    _proofs(tmp_path, monkeypatch, HARNESS, {"Vein Ripper": _proof_row("PLAYED")})
    b = config.DECKS_DIR / "d" / "branches" / "x"
    (b / "decklist.txt").write_text("Commander:\n1 Edgar Markov\n\nDeck:\n1 Toxic Deluge\n1 Vein Ripper\n1 Command Beacon\n")
    (b / "cards.json").write_text(json.dumps({"cards": [
        {"name": "Edgar Markov", "type_line": "Legendary Creature — Vampire Knight", "is_commander": True},
        {"name": "Toxic Deluge", "type_line": "Sorcery"}, {"name": "Vein Ripper", "type_line": "Creature — Vampire"},
        {"name": "Command Beacon", "type_line": "Land"}]}))
    assert cc.adds("d", "x") == ["Command Beacon", "Toxic Deluge", "Vein Ripper"]
    assert cc.land_adds("d", "x") == {"Command Beacon"} and cc.spell_adds("d", "x") == ["Toxic Deluge", "Vein Ripper"]
    s = cc.status("d", "x", HARNESS)
    assert s["lands"] == ["Command Beacon"]
    assert "Command Beacon" not in s["proven"] + s["held"] + s["late"] + s["unproven"]
    assert s["proven"] == ["Vein Ripper"] and s["unproven"] == ["Toxic Deluge"]
    # only lands -> nothing the shell can prove, said so, no shell started
    (b / "decklist.txt").write_text("Commander:\n1 Edgar Markov\n\nDeck:\n1 Sol Ring\n1 Swamp\n1 Command Beacon\n")
    monkeypatch.setattr(cc, "run_many", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a shell ran")))
    import argparse
    with pytest.raises(SystemExit) as e:
        cc.main(argparse.Namespace(slug="d", branch="x", adds=True))
    assert "only lands" in str(e.value) and "lands not proven here (1): Command Beacon" in capsys.readouterr().out


def test_our_own_gate_hint_is_not_an_authors_gate(monkeypatch):
    """The first cut of `author-gate` fired on Incendiary Command's installed script,
    whose only NeedsToPlay is OUR hint (2026-10-25). The class reads lines the SHIPPED
    script carries."""
    from manamap.sim import forge_cards
    shipped = "Name:C\\nTypes:Sorcery\\nA:SP$ Charm | Choices$ A,B\\n"
    installed = shipped + "SVar:NeedsToPlay:Creature.IsCommander+YouCtrl\\n"
    monkeypatch.setattr(forge_cards, "script",
                        lambda card, installed_copy=False: installed if installed_copy else shipped)
    assert "author-gate" not in cc.diagnose("C")["classes"]
