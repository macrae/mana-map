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
    assert set(sizes.values()) >= {3, 4, 5}, sizes
    assert sizes["playgroup"] == 5, "Oliver's table is five"
    assert sizes["playgroup-small"] == 3, "Alex's is three"


def test_the_playgroup_seats_each_say_where_they_came_from():
    """Two archetypes are the log's words and two are the pilot's.

    The commanders for those two were chosen for the ARCHETYPE and not to
    identify anyone's list, which is the trade the pilot asked for — "the exact
    pod players is less important than just having calibrated teams playing
    against each other". Every seat says which it is, so the approximation
    cannot quietly harden into a claim.
    """
    doc = pods.load("playgroup")
    for seat in doc["seats"]:
        assert seat.get("note"), f"{seat['slug']} must say where it came from"
    assert "approximate" in doc["note"]
    assert "calibrated teams" in doc["note"], \
        "the reason the mapping is loose belongs in the file"

    # The table the log records is red-dense, and three of the four seats are.
    reds = [s for s in doc["seats"] if "red" in s["archetype"]]
    assert len(reds) >= 3, "goblin-storm 002: 'the red density took the game over'"


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


# ── the table a record faced ────────────────────────────────────────────────

def test_a_record_says_which_table_it_faced():
    """Until pods existed, the only record of the table was the run id.

    So "report by pod composition" (B-2) had nothing to group on. `pod.named`
    is the load-bearing field: True when the pilot passed `--pod` (a fact about
    the run), False when it was inferred from the opponents against today's
    files (a reading of an older record).
    """
    import glob

    runs = sorted(glob.glob(str(ROOT / "data/decks/*/sim/*.json")))
    assert len(runs) >= 15
    named, inferred, unmatched = 0, 0, 0
    for path in runs:
        pod = json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("pod")
        if pod is None:
            unmatched += 1
            continue
        assert set(pod) == {"name", "named", "players", "composition", "brackets"}
        assert pod["name"] in pods.available()
        assert pod["composition"] == sorted(pod["composition"])
        named += bool(pod["named"])
        inferred += not pod["named"]

    # Most records predate `--pod` and are readings; the playgroup calibration
    # was the first run made THROUGH a pod and is a stamp.
    assert inferred >= 15, "the backfill reached almost nothing"
    assert named >= 1, "the calibration run stamped its pod and should say so"


def test_most_of_the_record_faced_the_table_the_docs_call_unfair():
    """The readout the stamp exists to make possible.

    `vito` is bracket 4 with 13 two-card infinites at a bracket-3 table, wins by
    life loss the damage parser cannot see, and was dropped as the default on
    2026-09-02. Knowing how much of the evidence base was measured against it
    was previously invisible without reading nineteen run ids by hand.
    """
    import collections
    import glob

    seen = collections.Counter()
    for path in sorted(glob.glob(str(ROOT / "data/decks/*/sim/*.json"))):
        pod = json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("pod")
        seen[(pod or {}).get("name")] += 1
    assert seen["vito-era"] > seen["standard"], \
        "if this flips, the fleet has been re-measured and the caveat is stale"


def test_an_ad_hoc_table_matches_nothing_rather_than_the_nearest_pod():
    """A near miss must be no match. Matching loosely would mislabel a record."""
    assert pods.match(["giada-angels", "baylen-tokens"]) is None
    assert pods.match(["abaddon", "baylen-tokens", "giada-angels"]) is None, \
        "order is part of the identity — it is the run id"
    assert pods.match(["giada-angels", "baylen-tokens", "abaddon"]) == "standard"


def test_a_stamped_pod_is_not_overwritten_by_a_reading():
    """`--analyze` backfills a reading and must never downgrade a fact."""
    stamped = pods.record_for("standard", [])
    assert stamped["named"] is True
    reading = pods.record_for(None, ["giada-angels", "baylen-tokens", "abaddon"])
    assert reading["named"] is False and reading["name"] == "standard"


# ── calibration: what the null actually is ──────────────────────────────────

def test_a_measured_pod_reports_a_null_that_is_not_one_over_n():
    """The reason this exists. A four-player win rate reads against 0.25 unless
    something says otherwise, and until now nothing did.

    Measured over the tracked records, `standard` gives its top seat more than
    twice its fair share and its bottom seat a fifth of one, and the subject
    chair lands well under a quarter. A deck scoring 0.16 there is AT the
    table's typical subject rate, not two thirds below a quarter — and that is
    the whole difference between a bad deck and an uneven table.
    """
    cal = pods.calibration("standard")
    assert cal["measured"] and cal["runs"] >= 3
    assert cal["fair_share"] == 0.25

    null = cal["subject_null"]
    assert null["rate"] < cal["fair_share"], \
        "if the subject seat reaches its fair share the caveat is stale"
    assert null["games"] == cal["games"]

    rates = [r["rate"] for r in cal["seats"]]
    assert rates == sorted(rates, reverse=True), "seats sort by rate"
    assert sum(r["wins"] for r in cal["seats"]) <= cal["games"] * len(cal["seats"])


def test_the_table_that_replaced_the_unfair_one_is_also_uneven():
    """`vito-era` was dropped for being unfair. `standard` is uneven too.

    Not the same unfairness — the dominant seat moved from vito to giada-angels
    — but a seat taking more than twice its share and another taking a fifth is
    not a level table, and every win rate measured on it is relative to that.
    `baylen-tokens` is the floor in BOTH pods, which is a fact about that deck
    and Forge's AI rather than about either table.
    """
    for name in ("standard", "vito-era"):
        cal = pods.calibration(name)
        assert cal["measured"], name
        assert cal["balance"]["dominant"] or cal["balance"]["floor"], \
            f"{name} reads as level — re-check, this was not true when written"
        assert "baylen-tokens" in cal["balance"]["floor"], name


def test_an_unplayed_table_says_it_has_no_null_rather_than_assuming_one():
    """Absent means absent. A pod nobody has run has no baseline at all.

    `playgroup` was the subject here until it was actually calibrated, which is
    the right way for this assertion to move: a table gains a null by being
    played, and the test follows the tables that have not been.
    """
    unplayed = [n for n in pods.available()
                if not pods.calibration(n)["measured"]]
    assert unplayed, "every table has a null now — pick a fresh one or drop this"
    cal = pods.calibration(unplayed[0])
    assert cal["measured"] is False
    assert cal["seats"] == []
    assert "no null" in cal["note"]
    assert "subject_null" not in cal, "an unmeasured null must not be a number"


def test_the_calibration_carries_the_limits_of_its_own_pooling():
    """It pools runs that differ in N, clock, profile and subject deck."""
    cal = pods.calibration("vito-era")
    text = " ".join(cal["limits"])
    assert "exchangeability" in text and "Markov chain" in text
    assert "SUBJECT pools OUR decks" in text, \
        "the subject null describes the fleet as much as the table"
    assert "Truncated" in text


def test_calibration_is_derived_and_stores_nothing():
    """It moves as runs accumulate, so storing it would date immediately."""
    doc = pods.load("standard")
    assert "calibration" not in doc and "baseline" not in doc
    for seat in doc["seats"]:
        assert "rate" not in seat and "win_rate" not in seat


def test_a_seat_that_won_nothing_reports_a_share_of_zero_not_none():
    """`0.0` is falsy AND a measurement.

    The first cut wrote `if rate and fair`, so a deck that went 0 for 39 printed
    "Nonex fair" — the absent-versus-zero confusion this repo keeps paying for,
    pointing the other way. A seat winning nothing is the most informative
    reading a calibration produces and must not render as unknown.
    """
    cal = pods.calibration("playgroup")
    if not cal["measured"]:
        pytest.skip("nothing has faced playgroup yet")
    for row in cal["seats"]:
        if row["rate"] == 0:
            assert row["share_of_fair"] == 0.0, row
    assert all(r["share_of_fair"] is not None for r in cal["seats"])


def test_a_null_pooled_from_one_deck_says_so():
    """A baseline from a single deck is that deck's record wearing the table's name.

    It cannot separate an uneven table from a bad deck, which is exactly the
    reading 0.000 over 39 games invites.
    """
    cal = pods.calibration("playgroup")
    if not cal["measured"]:
        pytest.skip("nothing has faced playgroup yet")
    text = pods.format_calibration(cal)
    if len(cal["decks"]) < 2:
        assert "ONE DECK ONLY" in text
