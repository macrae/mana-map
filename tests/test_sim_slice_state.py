"""game_state v2 -> Forge `[state]` (PRD v2 Step 5, the converter). Unit tests on
inline boards; the decklists and the corpus are faked."""
import pytest

from manamap.sim import slice_state as ss

DECKS = {
    "shark": (["Brallin, Skyshark Rider", "Shabraz, the Skyshark", "Windfall", "Sol Ring",
               "Lightning Greaves", "Hengegate Pathway // Mistgate Pathway"] + ["Island"] * 6,
              ["Brallin, Skyshark Rider", "Shabraz, the Skyshark"]),
    "angels": (["Giada, Font of Hope", "Swords to Plowshares", "Serra Angel"] + ["Plains"] * 8,
               ["Giada, Font of Hope"]),
}
KNOWN = {n for cards, _ in DECKS.values() for n in cards} | {"Hengegate Pathway", "Grizzly Bears"}


@pytest.fixture(autouse=True)
def fake_decks(monkeypatch):
    monkeypatch.setattr(ss, "_deck_copies", lambda slug: (list(DECKS[slug][0]), list(DECKS[slug][1])))


def board(**over):
    b = {"version": 2, "turn": 6, "active_seat": "you", "phase": "precombat main",
         "stack": [], "actions": [],
         "seats": [
             {"seat": "you", "life": 37,
              "board": [{"name": "Brallin, Skyshark Rider", "tapped": True,
                         "counters": {"P1P1": 2}}, "Island", "Island"],
              "hand": ["Windfall"]},
             {"seat": "opp1", "life": 40, "board": ["Serra Angel", "Plains"],
              "hand": {"unknown": 2}},
         ]}
    b.update(over)
    return b


def parse(text):
    return dict(line.split("=", 1) for line in text.strip().splitlines())


def convert(b=None, seed=1):
    return ss.to_forge_state(b or board(), {"you": "shark", "opp1": "angels"}, seed, known_names=KNOWN)


def test_you_are_p0_the_board_life_and_turn_carry_over_with_their_modifiers():
    st, _ = convert()
    s = parse(st)
    assert s["activeplayer"] == "p0" and s["activephase"] == "MAIN1" and s["turn"] == "6"
    assert s["p0life"] == "37" and s["p1life"] == "40"
    bf = s["p0battlefield"].split(";")
    assert bf[0] == "Brallin, Skyshark Rider|Tapped|Counters:P1P1=2|IsCommander"
    assert s["p0hand"] == "Windfall"
    assert s["p0command"] == "Shabraz, the Skyshark|IsCommander"   # the other commander stays home


def test_hidden_cards_come_from_the_seats_own_list_never_twice_and_vary_by_seed():
    a, notes = convert(seed=1)
    s = parse(a)
    hand = s["p1hand"].split(";")
    lib = s["p1library"].split(";")
    assert len(hand) == 2 and all(h in DECKS["angels"][0] for h in hand)
    # everything the seat has, copy for copy: the 11-card list, commander in the command zone
    on = ["Serra Angel", "Plains"] + hand + lib + ["Giada, Font of Hope"]
    assert sorted(on) == sorted(DECKS["angels"][0])
    assert any("2 unknown hand card(s)" in n for n in notes)
    orders = {parse(convert(seed=k)[0])["p1library"] for k in range(1, 8)}
    assert len(orders) > 1, "the library is shuffled per seed"
    assert parse(convert(seed=3)[0]) == parse(convert(seed=3)[0]), "and the same seed is the same board"


def test_named_library_cards_go_on_top_and_a_count_caps_the_rest():
    b = board()
    b["seats"][0]["library"] = {"top": ["Sol Ring"], "count": 3}
    lib = parse(convert(b)[0])["p0library"].split(";")
    assert lib[0] == "Sol Ring" and len(lib) == 3


def test_a_double_faced_card_goes_to_forge_by_its_front_face():
    b = board()
    b["seats"][0]["board"].append("Hengegate Pathway // Mistgate Pathway")
    assert "Hengegate Pathway" in parse(convert(b)[0])["p0battlefield"].split(";")
    assert "//" not in convert(b)[0]


def test_a_name_forge_would_skip_silently_is_a_refusal():
    b = board()
    b["seats"][1]["board"].append("Serra Angle")
    with pytest.raises(ss.StateError, match="Serra Angle"):
        convert(b)


def test_a_token_without_a_script_is_a_note_never_a_silent_drop():
    b = board()
    b["seats"][0]["board"].append({"name": "Shark", "token": True})
    b["seats"][0]["board"].append({"name": "Thopter", "token": True, "forge_token": "c_1_1_a_thopter_flying"})
    st, notes = convert(b)
    assert any("token 'Shark'" in n for n in notes)
    assert "t:c_1_1_a_thopter_flying" in parse(st)["p0battlefield"]


def test_mid_combat_needs_two_seats_and_every_seat_needs_a_deck():
    b = board(phase="combat", step="declare blockers")
    assert parse(convert(b)[0])["activephase"] == "COMBAT_DECLARE_BLOCKERS"
    b["seats"].append({"seat": "opp2", "life": 40, "board": []})
    with pytest.raises(ss.StateError, match="no deck named"):
        convert(b)
    with pytest.raises(ss.StateError, match="two seats"):
        ss.to_forge_state(b, {"you": "shark", "opp1": "angels", "opp2": "angels"}, 1, known_names=KNOWN)


def test_a_stack_is_named_as_left_out():
    b = board(stack=[{"name": "Windfall", "controller": "you"}])
    assert any("stack" in n for n in convert(b)[1])
