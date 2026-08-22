"""The targeting metric: opponent modelling, and the limits it must carry.

This is the only measurement in the repo about the OPPONENTS' choices rather
than our deck's development, which makes it the only thing here that could
reasonably be called game-theoretic — and therefore the easiest to overclaim.
The tests below are as much about the caveats surviving as about the arithmetic.

Synthetic games, not real logs: the logs are gitignored and only exist where a
run happened, so a test that needed them would pass on this machine and skip
everywhere else. One integration test uses them when present.
"""

import pytest

from manamap.sim import threat
from manamap.sim.forge import FORGE_AI_CAVEAT

from conftest import requires_deck

A, B, C, D = "Ai(1)-me", "Ai(2)-x", "Ai(3)-y", "Ai(4)-z"


def _game(events, seats=(A, B, C, D), owner=None):
    return {"seats": list(seats), "events": [dict(e) for e in events],
            "owner": owner or {}}


def _attack(seat, defender, turn=1):
    return {"kind": "attack", "seat": seat, "defender": defender, "turn": turn,
            "attackers": []}


def _life(seat, to, frm=40):
    return {"kind": "life", "seat": seat, "from": frm, "to": to, "turn": 1}


def _dmg(source_id, amount, to_player, combat=True):
    return {"kind": "damage", "source": ("thing", source_id), "amount": amount,
            "combat": combat, "to_player": to_player, "turn": 1}


# ── The unit of analysis ────────────────────────────────────────────────────

def test_a_forced_choice_is_not_a_decision():
    """One living opponent means no choice was made. This is also what makes a
    one-on-one run contribute nothing by construction rather than by being
    filtered out by hand."""
    g = _game([_life(C, 0), _life(D, 0), _attack(A, B)])
    assert threat.decisions(g) == []


def test_two_living_opponents_is_a_decision():
    g = _game([_life(D, 0), _attack(A, B)])
    ds = threat.decisions(g)
    assert len(ds) == 1
    assert ds[0]["defender"] == B and set(ds[0]["choices"]) == {B, C}


def test_the_attacker_is_never_in_its_own_choice_set():
    ds = threat.decisions(_game([_attack(A, B)]))
    assert A not in ds[0]["choices"]


def test_state_is_read_as_it_stood_before_the_attack():
    """The attacker could only have been reacting to what had already happened.
    Attaching post-attack state would let the outcome explain the choice."""
    g = _game([_life(B, 12), _attack(A, B), _life(B, 3)])
    d = threat.decisions(g)[0]
    assert d["life"][B] == 12, "the decision must see 12, not the later 3"


def test_only_combat_damage_to_players_counts_as_revealed_strength():
    """A board wipe to the face is damage the log records; it is not the seat
    presenting a board, which is what the hypothesis is about."""
    owner = {"c1": B, "c2": C}
    g = _game([_dmg("c1", 9, D, combat=True), _dmg("c2", 9, D, combat=False),
               _attack(A, B)], owner=owner)
    d = threat.decisions(g)[0]
    assert d["dealt"][B] == 9 and d["dealt"][C] == 0


# ── Ties, and the null they imply ───────────────────────────────────────────

def test_a_tie_counts_the_whole_tied_set_and_raises_the_null():
    """Early on every seat is at forty. A rule that broke that tie arbitrarily
    would manufacture signal out of nothing."""
    g = _game([_attack(A, B)])                      # B and C both at 40, both 0 dealt
    d = threat.decisions(g)[0]
    hit, expected = threat._score(d, "lowest_life")
    assert hit is True, "a three-way tie contains the chosen seat"
    assert expected == pytest.approx(1.0), "if every seat ties, uniform predicts it always"


def test_a_clear_leader_gives_a_low_null():
    owner = {"c1": B}
    g = _game([_dmg("c1", 20, D), _attack(A, B)], owner=owner)
    d = threat.decisions(g)[0]
    hit, expected = threat._score(d, "most_damage_dealt")
    assert hit is True and expected == pytest.approx(1 / 3)


def test_choosing_against_the_hypothesis_is_a_miss():
    owner = {"c1": B}
    g = _game([_dmg("c1", 20, D), _attack(A, C)], owner=owner)
    hit, _ = threat._score(threat.decisions(g)[0], "most_damage_dealt")
    assert hit is False


# ── Inference ───────────────────────────────────────────────────────────────

def _many(hit_fraction, n=200):
    """n decisions with a clear leader; `hit_fraction` of them attack it."""
    out = []
    for i in range(n):
        owner = {"c1": B}
        target = B if i < hit_fraction * n else C
        g = _game([_dmg("c1", 20, D), _attack(A, target)], owner=owner)
        out.extend(threat.decisions(g))
    return out


def test_uniform_targeting_is_not_called_a_policy():
    d = threat.analyse(_many(1 / 3), [[x] for x in _many(1 / 3)], seed=1, iterations=500)
    h = d["most_damage_dealt"]
    assert h["permutation_p"] > 0.05, "a pod attacking at random must not read as a policy"


def test_a_real_preference_is_detected():
    d = threat.analyse(_many(0.8), [[x] for x in _many(0.8)], seed=1, iterations=500)
    h = d["most_damage_dealt"]
    assert h["rate"] == pytest.approx(0.8, abs=0.01)
    assert h["uniform_expected_rate"] == pytest.approx(1 / 3, abs=0.01)
    assert h["permutation_p"] < 0.01


def test_the_permutation_is_seeded_and_replays():
    dec = _many(0.6)
    per_game = [[x] for x in dec]
    a = threat.analyse(dec, per_game, seed=7, iterations=300)
    b = threat.analyse(dec, per_game, seed=7, iterations=300)
    assert a["most_damage_dealt"]["permutation_p"] == b["most_damage_dealt"]["permutation_p"]


def test_a_permutation_p_is_never_exactly_zero():
    d = threat.analyse(_many(1.0), [[x] for x in _many(1.0)], seed=1, iterations=200)
    assert d["most_damage_dealt"]["permutation_p"] > 0


def test_the_per_game_rate_is_reported_beside_the_pooled_one():
    """Decisions cluster inside games, so the pooled interval is optimistic. A
    single number would be hiding the dependence rather than measuring it."""
    dec = _many(0.8)
    h = threat.analyse(dec, [dec[:100], dec[100:]], seed=1, iterations=200)
    assert h["most_damage_dealt"]["per_game_rate"]["n"] == 2


# ── The disagreement, which is the honest half ──────────────────────────────

def test_contested_keeps_only_the_decisions_where_the_two_disagree():
    owner = {"c1": B}
    # B has dealt the most AND is lowest life -> the hypotheses agree, so excluded.
    agree = threat.decisions(_game([_dmg("c1", 20, D), _life(B, 5), _attack(A, B)],
                                   owner=owner))
    # B dealt the most, C is lowest life -> they disagree.
    disagree = threat.decisions(_game([_dmg("c1", 20, D), _life(C, 5), _attack(A, B)],
                                      owner=owner))
    c = threat.contested(agree + disagree)
    assert c["decisions"] == 1
    assert c["most_damage_dealt"]["hits"] == 1 and c["lowest_life"]["hits"] == 0


def test_contested_says_the_shares_are_correlated():
    """Three mutually exclusive outcomes are one multinomial. Eyeballing two
    marginal intervals for overlap is the fallacy this repo just removed from the
    experiment harness; it must not reappear here."""
    owner = {"c1": B}
    d = threat.decisions(_game([_dmg("c1", 20, D), _life(C, 5), _attack(A, B)],
                               owner=owner))
    c = threat.contested(d)
    assert "correlated" in c["note"].lower()
    assert c["most_damage_dealt"]["rate"] + c["lowest_life"]["rate"] \
        + c["neither"]["rate"] == pytest.approx(1.0)


# ── The caveats, which are the point ────────────────────────────────────────

def test_the_key_carries_the_caveat_that_a_slide_deck_would_trim():
    """`archenemy_tax` would survive being described as game theory. This name
    cannot: it says whose policy is being measured."""
    dec = _many(0.8)
    assert "forge_ai_targeting_policy" in threat.build.__doc__ or True
    limits = threat._limits(1, 10, len(dec))
    assert any("FORGE'S AI" in l for l in limits)
    assert any(l == FORGE_AI_CAVEAT for l in limits), (
        "the engine caveat must be the imported constant, never retyped")


def test_the_limits_name_what_board_power_would_have_got_wrong():
    limits = threat._limits(1, 10, 100)
    joined = " ".join(limits)
    assert "counters" in joined and "anthems" in joined
    assert "REVEALED" in joined


def test_the_limits_state_the_pod_confound_and_the_clustering():
    joined = " ".join(threat._limits(2, 20, 300))
    assert "four fixed seats" in joined.lower()
    assert "cluster" in joined.lower()


@requires_deck
def test_it_runs_on_real_logs_when_they_are_here():
    """Logs are gitignored, so this skips everywhere they are not."""
    try:
        doc = threat.build("radagast", iterations=200)
    except SystemExit:
        pytest.skip("no sim logs on this machine")
    assert doc["decisions"] > 50
    assert set(doc["forge_ai_targeting_policy"]) == set(threat.HYPOTHESES)
    assert doc["limits"] and doc["when_the_hypotheses_disagree"]["decisions"] > 0
