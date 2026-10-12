"""Life gain as a counted EVENT — Shabraz's life per draw, Archangel of Thune.

Before 2026-10-11 `event_payoffs` refused "you gain N life" ("this model has no
own life total"), so Shabraz's half of every draw went uncounted, and Archangel
of Thune — "whenever you gain life, put a +1/+1 counter on each creature you
control" — read as a 3/4 lifelink flyer. The drain pillar already counted gain
EVENTS for Marauding Blight-Priest; it now counts Shabraz's too, and Archangel
spends them.
"""

import copy

from conftest import requires_data, requires_deck

from manamap.pilot import card_pool, goldfish


def _card(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return {"name": name, "oracle_text": text}


@requires_data
def test_the_gain_and_its_payoffs_are_read():
    assert goldfish.event_payoffs(_card("Shabraz, the Skyshark"))["per_draw_gain"] == 1
    assert goldfish.drain_profile(_card("Archangel of Thune"))["gain_team_counters"] == 1
    assert goldfish.drain_profile(_card("Ajani's Pridemate"))["gain_self_counter"] == 1
    # Brallin gains nothing; a discard trigger without the clause stays zero.
    assert goldfish.event_payoffs(_card("Brallin, Skyshark Rider"))["per_discard_gain"] == 0


@requires_data
def test_once_a_turn_and_paid_shapes_are_not_read_as_per_event():
    # "that many … once each turn" and a paid draw are named, not guessed.
    para = goldfish.drain_profile(_card("Nykthos Paragon"))
    assert para["gain_team_counters"] == 0 and para["gain_self_counter"] == 0


@requires_data
@requires_deck
def test_every_draw_is_a_gain_event_that_grows_the_board():
    """Same seeds, sharknado with the drain channel on: a dozen creatures carry
    Archangel's line or none do. The commanders' counters by turn eight must
    rise. Re-introduce the bug (drop `event_gain_events` from the pillar) and
    the board only grows on lifelink connects, a small fraction of this."""
    import json

    from manamap.pilot.common import deck_file, load_deck_cards

    targets = json.loads(deck_file("sharknado", "goldfish_targets.json").read_text())
    targets["model_drain"] = True
    doc = load_deck_cards("sharknado")
    line = "Whenever you gain life, put a +1/+1 counter on each creature you control."

    def counters(text_for_twelve):
        d = copy.deepcopy(doc)
        n = 0
        for c in d["cards"]:
            if ("Creature" in (c.get("type_line") or "") and n < 12
                    and c["name"] not in ("Shabraz, the Skyshark", "Brallin, Skyshark Rider")):
                c["oracle_text"] = text_for_twelve
                n += 1
        assert n == 12
        m = goldfish.run("sharknado", doc=d, iterations=600, seed=7, quiet=True,
                         _band=False, _targets_doc=targets)["metrics"]
        return m["discard"]["mean_commander_counters_by_turn"]["8"]

    plain, thune = counters("Flying"), counters("Flying\n" + line)
    assert thune > plain * 1.15, (plain, thune)
