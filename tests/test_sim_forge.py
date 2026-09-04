"""The Forge harness (S1): the parts we own, tested without the engine — plus one
opt-in game that needs it.

Pinned: a seat's .dck comes from the repo's own decklist parser and carries the
commander; data/opponents/ outranks data/decks/ for a seat name; the run id moves with
any seat's decklist; N games split evenly across JVMs with no empty job; the argv is
what the spike ran; the outcome parser reads winner, ROUND (Forge's `Game Outcome:
Turn N` is the winner's own turn count) and GLOBAL turn (the last `Turn:` line), how
each seat lost, and ms; a dry run writes nothing; a second identical sample is a
second file, never an overwrite.
"""

import hashlib
import json
from pathlib import Path

import pytest

from manamap.sim import forge

from conftest import requires_deck

FIX = Path(__file__).parent / "fixtures" / "forge" / "four-seat-two-games.log"
DECK = "1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n30 Forest\n"


@pytest.fixture
def seats(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "decks" / "mine").mkdir(parents=True)
    (data / "decks" / "mine" / "decklist.txt").write_text(DECK)
    (data / "decks" / "rival").mkdir(parents=True)
    (data / "decks" / "rival" / "decklist.txt").write_text(DECK.replace("Radagast of Rhosgobel", "Edgar Markov"))
    (data / "opponents" / "rival").mkdir(parents=True)
    (data / "opponents" / "rival" / "decklist.txt").write_text("1 Yawgmoth, Thran Physician *CMDR*\n1 Swamp\n")
    monkeypatch.setattr(forge, "DECKS_DIR", data / "decks")
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", data / "decks")
    return data


def test_dck_comes_from_the_repo_parser_and_carries_the_commander(seats):
    text = forge.to_dck("mine")
    assert text.startswith("[metadata]\nName=mm-mine\n[Commander]\n1 Radagast of Rhosgobel\n[Main]\n")
    assert "30 Forest" in text and "*CMDR*" not in text
    (seats / "decks" / "mine" / "decklist.txt").write_text("1 Forest\n")
    with pytest.raises(SystemExit):
        forge.to_dck("mine")


def test_an_opponent_dir_outranks_a_deck_of_the_same_name(seats):
    assert forge.seat_dir("rival") == seats / "opponents" / "rival"
    assert "Yawgmoth" in forge.to_dck("rival")
    with pytest.raises(SystemExit):
        forge.seat_dir("nobody")


def test_the_run_id_moves_with_any_seats_decklist_and_carries_the_seed(seats):
    a = forge.run_id("mine", ["rival"], 20)
    assert a.startswith("rival-n20-") and forge.run_id("mine", ["rival"], 20) == a
    assert a.endswith(f"-s{forge.default_seed('mine', ['rival'])}"), "the default seed is derived, so the default replays"
    assert forge.run_id("mine", ["rival"], 20, seed=7).endswith("-s7")
    (seats / "opponents" / "rival" / "decklist.txt").write_text("1 Yawgmoth, Thran Physician *CMDR*\n1 Swamp\n1 Sol Ring\n")
    assert forge.run_id("mine", ["rival"], 20) != a
    assert forge.run_id("mine", ["rival"], 21) != a


def test_games_split_evenly_with_no_empty_job():
    assert forge.split_games(20, 8) == [3, 3, 3, 3, 2, 2, 2, 2]
    assert forge.split_games(5, 8) == [1, 1, 1, 1, 1]
    assert forge.split_games(1, 3) == [1]
    assert sum(forge.split_games(500, 7)) == 500


def test_the_argv_is_what_the_spike_ran(tmp_path):
    jar = tmp_path / "forge-gui-desktop-9.9.9-jar-with-dependencies.jar"
    argv = forge.command(["mm-a", "mm-b", "mm-c"], 7, 300, jar=jar, seed=42)
    assert argv[:1] == ["java"] and "-jar" in argv and str(jar) in argv
    i = argv.index("sim")
    assert argv[i:] == ["sim", "-d", "mm-a", "mm-b", "mm-c", "-f", "commander", "-n", "7", "-c", "300", "-s", "42"]
    assert "-s" not in forge.command(["mm-a"], 1, 120, jar=jar), "no seed flag without a seed"


def test_outcomes_parse_winner_round_global_turn_and_losses():
    games = forge.parse_outcomes(FIX.read_text())
    assert len(games) == 2
    g1, g2 = games
    assert g1["winner"] == "Ai(1)-radagast" and g2["winner"] == "Ai(2)-edgar-vampires"
    assert g1["round"] == 20 and g1["global_turn"] == 39, "round is Forge's count; global is the last Turn: line"
    assert set(g1["lost"]) == {"Ai(2)-edgar-vampires", "Ai(3)-yawgmoth-swarm", "Ai(4)-heliod"}
    assert all(v == "life total reached 0" for v in g1["lost"].values())
    assert g1["ms"] == 32987 and g2["round"] == 19


def test_an_alternate_win_condition_is_a_win_not_a_draw():
    """Measured on the first tracked run: heliod won twice by Approach of the Second
    Sun and both read as draws, because the line is "has won due to effect of", not
    "has won because"."""
    text = ("Turn: Turn 42 (Ai(4)-mm-heliod)\nGame Outcome: Turn 21\n"
            "Game Outcome: Ai(1)-mm-radagast has lost because an opponent has won by spell 'Approach of the Second Sun'\n"
            "Game Outcome: Ai(4)-mm-heliod has won due to effect of 'Approach of the Second Sun'\n"
            "Game Result: Game 1 ended in 1000 ms. Ai(4)-mm-heliod has won!\n")
    g = forge.parse_outcomes(text)[0]
    assert g["winner"] == "Ai(4)-mm-heliod" and not g["draw"]
    assert g["won_by"] == "effect of 'Approach of the Second Sun'"
    assert g["lost"]["Ai(1)-mm-radagast"].startswith("an opponent has won by spell")


def test_forge_seat_labels_map_back_to_slugs_FROM_ANY_SEAT():
    """THE MAP IS A CROSS PRODUCT because the decks ROTATE through the seats.

    It used to be 1:1 — deck i is `Ai(i+1)` — which was true only while every
    deck sat in a fixed chair. Forge gives the first `-d` the first turn every
    game, so a fixed order handed our deck a permanent positional advantage or
    disadvantage; rotating the order fixes that and makes `Ai(k)` a property of
    the GAME rather than of the deck.

    A label carries the deck's own name, so position never had to be part of the
    key. Every index maps for every deck, and a lookup is correct under any
    rotation.
    """
    label = forge._seat_label(["mm-radagast", "mm-edgar-vampires"])
    for k in (1, 2):
        assert label[f"Ai({k})-mm-radagast"] == "radagast"
        assert label[f"Ai({k})-mm-edgar-vampires"] == "edgar-vampires"
    assert len(label) == 4, "every seat index for every deck, and nothing else"

    # A CONSUMER MUST NOT TREAT THIS AS AN ENUMERATION OF SEATS. `bridge` did —
    # it zipped these keys against the record's seat list and paired
    # `Ai(2)-<deck 0>` with seat 1, giving every seat the wrong decklist and
    # commander. There are N x N keys for N seats and that is the point.
    assert len(label) == 2 * 2


def test_a_dry_run_writes_nothing_and_the_same_seed_is_a_replay_not_a_sample(seats, tmp_path):
    home = tmp_path / "forge"; home.mkdir()
    (home / "forge-gui-desktop-0.0.1-jar-with-dependencies.jar").write_text("")
    decks = tmp_path / "forgedecks"
    path, rec = forge.run("mine", ["rival"], games=4, jobs=2, dry_run=True, home=home, decks_dir=decks)
    assert rec["games_per_job"] == [2, 2] and len(rec["commands"]) == 2
    base = forge.default_seed("mine", ["rival"])
    assert rec["seeds"] == [base, base + 1], "job i runs seed_base + i"
    assert all("-s" in c for c in rec["commands"])
    assert path.name.startswith("rival-n4-") and not path.exists()
    assert (decks / "mm-mine.dck").exists() and (decks / "mm-rival.dck").exists(), "decks are installed even on a dry run"
    assert not (seats / "decks" / "mine" / "sim" / "logs").exists()
    path.parent.mkdir(parents=True); path.write_text("{}")
    with pytest.raises(SystemExit):                       # same config + seed = the same bytes
        forge.run("mine", ["rival"], games=4, jobs=2, home=home, decks_dir=decks)
    path2, _ = forge.run("mine", ["rival"], games=4, jobs=2, seed=99, dry_run=True, home=home, decks_dir=decks)
    # THE SEED IS IN THE ID BUT NO LONGER AT THE END OF IT. Since 2026-08-30 the
    # id carries the pilots too — `-pod<Profile>` for a non-default pod, which is
    # now the standard — so this asserts the seed is PRESENT and that a new seed
    # is a new path, rather than pinning the id's tail.
    assert "-s99" in path2.name and path2 != path, "a new seed is a new sample"


@pytest.mark.forge
@requires_deck
def test_one_real_two_seat_game_records_a_run(tmp_path, monkeypatch):
    """Plays ONE real game (≈6–10 s). Opt in with `pytest -m forge`."""
    from manamap import config
    if not list(config.FORGE_HOME.glob("forge-gui-desktop-*-jar-with-dependencies.jar")):
        pytest.skip("Forge is not installed at FORGE_HOME (docs/simulation.md)")
    # write the record under a scratch copy of the deck dir so the repo is untouched
    import shutil
    data = tmp_path / "data"; (data / "decks").mkdir(parents=True)
    for s in ("radagast", "edgar-vampires"):
        (data / "decks" / s).mkdir()
        shutil.copy(config.DECKS_DIR / s / "decklist.txt", data / "decks" / s / "decklist.txt")
    monkeypatch.setattr(forge, "DECKS_DIR", data / "decks")
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", data / "decks")
    path, rec = forge.run("radagast", ["edgar-vampires"], games=1, jobs=1, clock=120)
    assert rec["games_completed"] == 1 and path.exists()
    o = rec["outcomes"][0]
    assert o["winner"] in ("radagast", "edgar-vampires", None)
    assert o["global_turn"] and o["round"] and o["global_turn"] >= o["round"]
    assert rec["engine"]["forge"]["version"] and any("SEEDED" in a for a in rec["assumptions"])
    assert rec["seeds"] == [rec["seed_base"]] and o["seed"] == rec["seed_base"] and o["game_in_job"] == 1


# ── the runaway job, and the clock that does not end a game ──


def test_the_job_cap_kills_the_pathological_runs_and_nothing_else():
    """THE HOURS WERE HERE. Forge's `-c` is a FutureTask timeout: it ends the
    game's ACCOUNTING and not its AI thread, so a job could run unbounded.
    Measured across the eighteen tracked runs, wall time against the cap:

        zur-enchantress  n=20   15220s   killed at 1470s   (4.2 HOURS)
        yawgmoth-swarm   n=20   13372s   killed at 1470s   (3.7 HOURS)
        edgar-vampires   n=400  11243s   survives (cap 26220s)
        ur-dragon        n=100   3720s   survives (cap  6870s)
        …sixteen others, all survive

    7.1 hours across the tracked set, and the discrimination is what matters:
    a cap that also killed the healthy 400-game run would be useless.
    """
    import glob
    import os

    from manamap.sim import forge

    killed, survived = [], []
    for path in sorted(glob.glob("data/decks/*/sim/*.json")):
        try:
            doc = json.loads(open(path).read())
        except Exception:                          # noqa: BLE001 - defensive
            continue
        games, wall = doc.get("games_requested"), doc.get("wall_seconds")
        if not (games and wall):
            continue
        cap = (int(doc.get("clock_seconds") or 300)
               * max(forge.split_games(int(games), int(doc.get("jobs") or 7)))
               * forge.TIMEOUT_SLACK)
        cap = int(cap) + forge.TIMEOUT_FLOOR
        name = f"{os.path.basename(os.path.dirname(os.path.dirname(path)))} n={games}"
        (killed if wall > cap else survived).append(name)

    if not (killed or survived):
        pytest.skip("no tracked Forge runs")
    assert len(survived) >= 10, f"the cap is too tight: only {len(survived)} survive"
    # A run whose games genuinely take the clock must not be killed: the 400-game
    # arm is the longest LEGITIMATE run on disk and is the real control here.
    assert not any("n=400" in n for n in killed), f"killed a healthy long run: {killed}"


def test_a_run_record_always_says_whether_a_job_was_truncated():
    """ABSENT MEANS ABSENT. A missing key reads as "no truncation" and "nobody
    looked" identically, so the list is always present and empty when clean."""
    import inspect

    from manamap.sim import forge

    source = inspect.getsource(forge.run)
    assert '"truncated_jobs": timed_out' in source


def test_the_assumptions_no_longer_claim_a_clock_hit_game_is_a_draw():
    """MEASURED AND WRONG for as long as it shipped: `summary.draws` is 0 on
    every tracked run including the two with three clock-hit games each, and
    edgar-vampires n=400 carries 75 clock-hit games (19%) all with winners. A
    record that states its own assumption incorrectly is worse than one that
    states nothing.
    """
    from manamap.sim import forge

    text = " ".join(forge.ASSUMPTIONS)
    # THE SUBSTANCE, not the sentence. The original wording ("NOT recorded as a
    # draw") was itself a half-truth once the mechanism was decompiled: Forge
    # really does call `setGameOver(Draw)`, and then prints a `has won` line for
    # every surviving seat, which is why a naive parse credited one of them. The
    # assumption now says both halves, so pinning the old phrasing would force
    # the text back to the less accurate version.
    assert "recorded as a draw, not dropped" not in text
    assert "truncated" in text, "the record's name for a clock-out must be stated"
    assert "EXCLUDED from the win rate" in text
    assert "no winner" in text or "with no winner" in text


def test_oversubscribing_the_machine_censors_games():
    """The evidence `default_jobs` rests on, re-derived from the tracked runs.

    Forge's `-c` is WALL time, so a JVM scheduled onto an efficiency core runs
    the same game at roughly half speed and its games hit the clock. A truncated
    game has no winner and is EXCLUDED from the win rate, so oversubscribing
    does not merely run slower — it censors the sample, and it censors it in a
    way indistinguishable in the record from a genuinely stalled game.

    Asserted as a RATIO rather than as fixed rates, because both move as runs
    accumulate: the docstring originally said 4 jobs truncate 0%, which was true
    of the two runs that existed then and is not true now (3.4% over seven).
    """
    import collections
    import glob
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    by_jobs = collections.defaultdict(lambda: [0, 0])
    for path in sorted(glob.glob(str(root / "data/decks/*/sim/*.json"))):
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        n = doc.get("games_completed") or 0
        if not n or doc.get("jobs") is None:
            continue
        by_jobs[doc["jobs"]][0] += n
        by_jobs[doc["jobs"]][1] += doc["summary"].get("truncated") or 0

    assert 4 in by_jobs and 7 in by_jobs, dict(by_jobs)
    low = by_jobs[4][1] / by_jobs[4][0]
    high = by_jobs[7][1] / by_jobs[7][0]
    assert by_jobs[4][0] >= 200 and by_jobs[7][0] >= 200, "too few games to compare"
    assert high > low * 2, (
        f"4 jobs censor {low:.1%}, 7 jobs censor {high:.1%} — if that gap has "
        f"closed, `default_jobs` needs re-deriving rather than trusting")
