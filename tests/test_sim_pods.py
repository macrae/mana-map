"""Named tables (sim/pods.py).

The standard pod was a SENTENCE in `docs/simulation.md` and three `--vs` flags
typed by hand; `grep STANDARD_POD` found an AI profile and no roster. These
pin the two properties that make a pod file safe to adopt: it expands to the
same ordered slugs and therefore the SAME RUN ID, and a per-seat AI profile
cannot silently re-base a record that already exists.
"""

import json
import pathlib
from types import SimpleNamespace

import pytest

from manamap.sim import forge, pods

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(pods, "PODS_DIR", tmp_path)
    return tmp_path


def _write(dir_, name, seats, **extra):
    doc = {"name": name, "seats": seats, **extra}
    (dir_ / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


# ── the property that makes a pod safe to adopt ─────────────────────────────

def test_a_pod_expands_to_the_same_run_id_as_the_flags_it_replaces():
    """`--pod standard` is a SPELLING, not a new measurement.

    The run id is built from the ordered opponent slugs, so a convenience that
    reordered or renamed them would silently re-base every record made through
    it — and nothing downstream would notice, because a run id looks like a run
    id. This is the whole safety argument for the feature.
    """
    flags, flag_profiles = forge.resolve_table(
        SimpleNamespace(vs=["giada-angels", "baylen-tokens", "abaddon"], pod=None))
    podded, pod_profiles = forge.resolve_table(
        SimpleNamespace(vs=[], pod="standard"))
    assert flags == podded, "a pod must expand to the same ordered slugs"

    # Through `run_id_for`, which is what `run` itself calls — a test that
    # recomputed the rule would be testing itself.
    by_flag = forge.run_id_for("edgar-vampires", flags, 100, 1121742080,
                               None, flag_profiles, 600)
    by_pod = forge.run_id_for("edgar-vampires", podded, 100, 1121742080,
                              None, pod_profiles, 600)
    assert by_flag == by_pod

    # And it is the id of a record that already exists on disk.
    tracked = (ROOT / "data" / "decks" / "edgar-vampires" / "sim" /
               f"{by_flag}.json")
    assert tracked.exists(), by_flag


def test_the_standard_pod_is_the_three_decks_the_docs_name():
    assert pods.seats("standard") == ["giada-angels", "baylen-tokens", "abaddon"]
    assert pods.profiles("standard") is None, \
        "the standard pod expresses no profile opinion — --vs-profile decides"


def test_the_vito_era_pod_exists_so_an_old_record_can_be_reproduced():
    """A superseded table is kept, not deleted — a record must stay replayable.

    Every Forge record made before 2026-09-02 was measured against it, and vito
    is bracket 4 with 13 two-card infinites at a bracket-3 table.
    """
    assert "vito" in pods.seats("vito-era")
    assert "vito" not in pods.seats("standard")
    doc = pods.load("vito-era")
    assert "life loss" in doc["note"].lower(), "the note must say WHY it was replaced"


# ── the per-seat profile, and the tag that keeps records apart ──────────────

def test_a_uniform_pod_profile_is_indistinguishable_from_the_flag():
    """Every seat agreeing must not invent a new run id.

    Otherwise adopting pod files would rename every record that could have been
    made with `--vs-profile Cautious`, for no change in what was measured.
    """
    assert forge.pod_tag_name({"a": "Cautious", "b": "Cautious"}, ["a", "b"]) \
        == "Cautious"
    assert forge.pod_tag_name("Experimental", ["a", "b"]) == "Experimental"
    # A seat the map omits falls back to the standard rather than to Default,
    # so a pod that sets ONE seat does not quietly re-pilot the others.
    assert forge.pod_tag_name({}, ["a", "b"]) == forge.STANDARD_POD_PROFILE


def test_a_mixed_table_gets_its_own_run_id():
    """Two tables that play differently must not write the same path.

    The same silent-overwrite `profile_tag` was written for: the digest is over
    decklists, opponents, games and seed, none of which move when a pilot does.
    """
    mixed = forge.pod_tag_name({"a": "Cautious"}, ["a", "b"])
    assert mixed.startswith("Mixed") and len(mixed) == len("Mixed") + 8

    other = forge.pod_tag_name({"b": "Cautious"}, ["a", "b"])
    assert other.startswith("Mixed") and other != mixed, \
        "which seat is cautious changes the table"

    # Stable: the same map twice is the same id.
    assert forge.pod_tag_name({"a": "Cautious"}, ["a", "b"]) == mixed


def test_a_seat_keeps_its_own_pilot_through_the_rotation():
    """Seats rotate per job; profiles are index-aligned with `-d` and must follow.

    Rotating decks without rotating profiles hands our seat's pilot to whichever
    deck landed in slot 0 — a worse bug than the seat bias rotation exists for.
    """
    rot = ["baylen-tokens", "abaddon", "mine", "giada-angels"]
    got = forge._profiles_for(rot, "mine", "Default", {"abaddon": "Reckless"})
    assert got == [forge.STANDARD_POD_PROFILE, "Reckless", "Default",
                   forge.STANDARD_POD_PROFILE]


# ── the resolver both commands share ────────────────────────────────────────

def test_flags_and_a_pod_together_are_refused():
    """Two sources for one list is how a seat goes missing silently."""
    args = SimpleNamespace(vs=["giada-angels"], pod="standard")
    with pytest.raises(SystemExit) as exc:
        forge.resolve_table(args)
    assert "pick one" in str(exc.value)


def test_the_resolver_passes_bare_flags_straight_through():
    got, profiles = forge.resolve_table(
        SimpleNamespace(vs=["a", "b"], pod=None))
    assert got == ["a", "b"] and profiles is None


def test_an_unknown_pod_names_the_ones_that_exist():
    with pytest.raises(SystemExit) as exc:
        forge.resolve_table(SimpleNamespace(vs=[], pod="no-such-pod"))
    assert "standard" in str(exc.value), "the error must name a way forward"


def test_experiment_resolves_a_table_the_same_way_simulate_does():
    """One resolver, because these two disagreeing is not hypothetical.

    They already disagreed about the pod's AI profile, and that made every
    controlled A/B controlled against a table the deck is never measured on.
    """
    from manamap.sim import experiment

    assert experiment.forge_resolve_table is forge.resolve_table


# ── the file is form-checked, and pod SIZE is free ──────────────────────────

def test_a_pod_may_seat_three_four_or_five_players(sandbox):
    """B-2 asks for configurable pod size; a pod is a list, so it already is."""
    for n, expected in ((2, 3), (3, 4), (4, 5)):
        seats = [{"slug": f"opp{i}"} for i in range(n)]
        _write(sandbox, f"pod{n}", seats)
        assert pods.compose(f"pod{n}")["players"] == expected


def test_a_seat_named_twice_is_refused(sandbox):
    """Forge installs one .dck per slug, so a repeat is one deck in two chairs."""
    _write(sandbox, "twice", [{"slug": "a"}, {"slug": "a"}])
    with pytest.raises(pods.PodError) as exc:
        pods.seats("twice")
    assert "twice" in str(exc.value) or "a" in str(exc.value)


def test_an_unknown_seat_key_is_refused(sandbox):
    """A typo'd key would be silently ignored, which is how a profile goes unread."""
    _write(sandbox, "typo", [{"slug": "a", "profil": "Cautious"}])
    with pytest.raises(pods.PodError) as exc:
        pods.load("typo")
    assert "profil" in str(exc.value)


def test_a_pod_with_no_seats_is_refused(sandbox):
    _write(sandbox, "empty", [])
    with pytest.raises(pods.PodError):
        pods.load("empty")


def test_composition_is_what_a_result_should_be_reported_by(sandbox):
    """B-2's last clause: by pod COMPOSITION, not only in aggregate."""
    _write(sandbox, "mix", [
        {"slug": "b", "archetype": "tokens", "bracket": 3},
        {"slug": "a", "archetype": "control", "bracket": 4}])
    info = pods.compose("mix")
    assert info["composition"] == ["control", "tokens"], "sorted, so shapes compare"
    assert info["brackets"] == [3, 4]
    assert info["players"] == 3


def test_every_pod_on_disk_loads():
    names = pods.available()
    assert len(names) >= 2, "the guard iterated almost nothing"
    for name in names:
        doc = pods.load(name)
        assert doc["seats"] and doc.get("note"), f"{name} has no note"
        for seat in doc["seats"]:
            assert seat.get("archetype"), f"{name}/{seat['slug']} has no archetype"


# ── the tables that ship ────────────────────────────────────────────────────

def test_the_pods_on_disk_cover_three_four_and_five_players():
    """Pod SIZE is a variable now, and the sizes are the ones actually played.

    Alex's house is three-player and Oliver's is five; every Forge record before
    pods existed was four, because four was the only thing anyone typed.
    """
    sizes = {name: pods.compose(name)["players"] for name in pods.available()}
    assert 5 in sizes.values(), "no five-player table, and Oliver's is five"
    assert 4 in sizes.values()
    assert sizes["five-player"] == 5


def test_the_playgroup_pod_names_only_seats_the_log_names():
    """Every commander here is written down in a real game entry.

    Three of the playgroup's decks are deliberately ABSENT — Tom's blue-black,
    Tom's enchantment deck, Alex's fight-based green — because the log never
    names their commanders. An invented seat in a file called `playgroup` is an
    invention that goes invisible.
    """
    doc = pods.load("playgroup")
    assert [s["slug"] for s in doc["seats"]] == \
        ["krenko-tokens", "purphoros-pingers", "tannuk-warp"]
    for seat in doc["seats"]:
        assert seat.get("note"), f"{seat['slug']} must say who plays it"
    assert "does not name" in doc["note"] or "never names" in doc["note"]

    # And it is red-dense, which is an observation and not a choice this made.
    assert all("mono-red" in s["archetype"] for s in doc["seats"])


def test_a_coverage_seat_never_claims_to_be_someones_deck():
    """`value-chains` is archetypes, and says so rather than implying a table."""
    doc = pods.load("value-chains")
    assert "deck anyone at the table plays" in doc["note"]
    # No bracket is claimed, because none was verified.
    assert not any(s.get("bracket") for s in doc["seats"])


def test_every_seat_a_pod_names_exists_on_disk():
    """A pod naming a seat that is not fetched fails at the JVM, late and loudly.

    `forge.seat_dir` resolves it, so this is the cheap version of that check —
    and the one that fails on a fresh clone rather than after a pod is installed.
    """
    from manamap.config import DATA_DIR

    checked = 0
    for name in pods.available():
        for slug in pods.seats(name):
            seat = DATA_DIR / "opponents" / slug
            deck = DATA_DIR / "decks" / slug
            assert seat.exists() or deck.exists(), f"{name}: no seat for {slug!r}"
            checked += 1
    assert checked >= 10, "the guard iterated almost nothing"
