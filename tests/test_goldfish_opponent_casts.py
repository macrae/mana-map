"""Our draws on their turns — Rhystic Study and Mind's Eye.

Rhystic Study triggers on an opponent CASTING, and the goldfish has no
opponents, so it was never cast. The cast count is now MEASURED (Forge, via
`sim/pod_behaviour`), and the tax rides the authored TITHE_PAY_RATE. Mind's Eye
was refused whole because an unbudgeted read is seven free cards a wheel; it now
pays from the mana left at the end of the turn.
"""

import copy

from conftest import requires_data, requires_deck

from manamap import config
from manamap.pilot import card_pool, goldfish
from manamap.sim import pod_behaviour


def test_the_cast_rate_is_the_measured_one():
    """The constant moves with its evidence or this fails."""
    assert config.OPPONENT_CASTS_PER_TURN == pod_behaviour.POD["casts_per_turn"]


def _ev(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.event_payoffs({"name": name, "oracle_text": text})


@requires_data
def test_both_are_read():
    rs = _ev("Rhystic Study")
    assert rs["per_opponent_cast_draw"] == 1 and rs["per_opponent_cast_taxed"] is True
    me = _ev("Mind's Eye")
    assert me["per_opponent_draw_paid_draw"] == 1 and me["per_opponent_draw_paid_cost"] == 1
    # A free per-draw draw is still the unpaid field, untouched.
    assert me["per_opponent_draw_our_draw"] == 0


@requires_data
@requires_deck
def test_rhystic_study_draws_in_a_game():
    """Same seeds, Rhystic Study's text blanked or live. Extra cards by turn
    eight must rise. Re-introduce the bug (skip the opponent-cast block) and the
    arms read identical."""
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")
    assert any(c["name"] == "Rhystic Study" for c in doc["cards"])

    def extra(blank):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if blank and c["name"] == "Rhystic Study":
                c["oracle_text"] = "Flash"
        m = goldfish.run("sharknado", doc=d, iterations=1500, seed=7,
                         quiet=True, _band=False)["metrics"]
        return m["mean_extra_cards_drawn_by_turn"]["8"]

    assert extra(blank=False) > extra(blank=True) + 0.1
