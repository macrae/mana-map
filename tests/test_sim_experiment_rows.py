"""AN EXPERIMENT'S ARMS, READ WHOLE (2026-10-22, the finisher-v1 A/B).

Three defects one 300-a-side A/B exposed at once:

- `experiment --analyze` mapped OUR seat to `Ai(1)` alone, while the run rotates the
  seats every job, so the re-derivation scored some other seat as ours in three games
  out of four and read the A/B as 0.04 a side where the run had read 0.151 / 0.111.
- answers, counterspells, protection and the wheels reached no experiment row, so the
  pilot's question ("has answers, interactions, resolve") had no figure at a table.
- `net-change` read `sim/` records only, so the A/B reached no branch report at all.
"""

import json

from manamap.pilot import net_change
from manamap.sim import experiment as ex

SLUG = "xdeck"


def _log(seats, winner):
    head = [f"Mulligan: {s} has kept a hand of 7 cards" for s in seats]
    head.append(f"Turn: Turn 1 ({seats[0]})")
    tail = [f"Game Outcome: {s} has lost because life total reached 0"
            for s in seats if s != winner]
    return "\n".join(head + ["Game Outcome: Turn 1",
                             f"Game Outcome: {winner} has won because all opponents have lost",
                             *tail,
                             f"Game Result: Game 1 ended in 100 ms. {winner} has won!"])


def test_analyze_scores_our_seat_wherever_the_rotation_put_it(tmp_path, monkeypatch):
    """Our seat sits at Ai(2) and wins. The old map knew only Ai(1)-<us>, so this
    arm re-derived as zero wins. Re-introduce `label[f"Ai(1)-{meta}"] = slug` in
    `analyze` and the win disappears."""
    monkeypatch.setattr(ex, "deck_dir", lambda slug, branch=None: tmp_path)
    monkeypatch.setattr(ex, "_commanders_by_slug", lambda opps: {})
    exp = tmp_path / ex.EXP_DIR
    (exp / "logs" / "e1").mkdir(parents=True)
    arm = {"ref": "working", "label": "w", "decklist_sha256": "x",
           "decklist_text": "Commander:\n1 Someone\n\nDeck:\n1 Sol Ring\n"}
    (exp / "e1.json").write_text(json.dumps({
        "opponents": [{"slug": "opp"}],
        "arms": {"a": dict(arm), "b": dict(arm)}}))
    for letter in ("a", "b"):
        seats = ["Ai(1)-mm-opp", f"Ai(2)-mm-x-{SLUG}-{letter}"]
        (exp / "logs" / "e1" / f"{letter}-part-00.log").write_text(_log(seats, seats[1]))
    _, doc = ex.analyze(SLUG, "e1")
    for letter in ("a", "b"):
        assert doc["arms"][letter]["analysis"]["seats"][SLUG]["wins"] == 1, (
            "a rotated seat scored as somebody else's")


def _games(values, key):
    return [{"per_seat": {SLUG: {key: v}}} for v in values]


def test_answers_and_wheel_rows_reach_the_delta_with_an_interval():
    a, b = {"games": 4, "seats": {SLUG: {}}}, {"games": 4, "seats": {SLUG: {}}}
    for key in ("answers_cast", "wheels_with_commander", "interaction_cast"):
        d = ex.delta(a, b, SLUG, _games([0, 1, 1, 0], key), _games([2, 3, 2, 3], key))
        row = d[key]
        assert row["a"] == 0.5 and row["b"] == 2.5 and row["diff"] == 2.0
        assert row["ci95_diff"] and row["n_a"] == 4, f"{key} has no interval"
    # an arm scored before the rows existed carries no value: no row, never a zero
    d = ex.delta(a, b, SLUG, _games([1, 2], "casts"), _games([1, 2], "casts"))
    assert "answers_cast" not in d


def _experiment(tmp_path, aa=False):
    exp = tmp_path / "data" / "decks" / SLUG / "experiments"
    exp.mkdir(parents=True)
    def arm(ref, sha, wins):
        return {"ref": ref, "decklist_sha256": sha,
                "analysis": {"games": 2, "decided": 2,
                             "seats": {SLUG: {"wins": wins}}},
                "games_detail": [{"winner": SLUG, "per_seat": {SLUG: {"answers_cast": 1,
                                                                      "wheels_cast": 2,
                                                                      "wheels_with_commander": 1}}},
                                 {"winner": "opp", "per_seat": {SLUG: {"answers_cast": 0}}}]}
    (exp / "e1.json").write_text(json.dumps({
        "experiment_id": "e1", "pod": {"name": "t"}, "design": {"aa": aa},
        "arms": {"a": arm("working", "LIVE", 1), "b": arm("@br", "BR", 2)}}))


def test_net_change_reads_an_experiments_arms_as_runs_of_each_list(tmp_path, monkeypatch):
    _experiment(tmp_path)
    monkeypatch.chdir(tmp_path)
    br = list(net_change._records(f"data/decks/{SLUG}/branches/br/sim/*.json", "xdeck-br", "BR"))
    assert [d["run_id"] for _, d in br] == ["e1#b"]
    doc = br[0][1]
    assert doc["analysis"]["seats"]["xdeck-br"]["wins"] == 2, "the arm was not re-keyed to the branch"
    assert doc["games"][0]["winner"] == "xdeck-br"
    assert doc["wheel_timing"]["per_game"] == [{"answers_cast": 1, "wheels_cast": 2,
                                                "wheels_with_commander": 1}]
    champ = list(net_change._records(f"data/decks/{SLUG}/sim/*.json", SLUG, "LIVE"))
    assert [d["run_id"] for _, d in champ] == ["e1#a"]
    # an arm on a list that is no longer on disk is not the deck's run
    assert not list(net_change._records(f"data/decks/{SLUG}/sim/*.json", SLUG, "OTHER"))


def test_an_a_a_experiment_is_never_pooled_into_a_branch_report(tmp_path, monkeypatch):
    _experiment(tmp_path, aa=True)
    monkeypatch.chdir(tmp_path)
    assert not list(net_change._records(f"data/decks/{SLUG}/sim/*.json", SLUG, "LIVE"))
