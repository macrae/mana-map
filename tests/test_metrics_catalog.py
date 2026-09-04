"""The metrics catalog, checked against the artifacts it describes.

A catalog that is only prose rots one entry at a time and nothing notices — the
same failure `test_docs_counts.py` exists for, one layer up. So every claim here
is checked in BOTH directions against real tracked artifacts: a figure the
catalog says is published must actually be there, and one it calls unavailable
must not be.

The unavailable entries are the ones worth guarding hardest. They are the
honest answer to "does the simulation measure X", and the failure mode they
prevent is somebody adding a plausible zero.
"""

import glob
import json
import pathlib

import pytest

from manamap import metrics
from manamap.metrics import (CATALOG, DERIVABLE, ENGINES, GROUPS, OPT_IN,
                             PUBLISHED, STATUSES, UNAVAILABLE)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _runs():
    return sorted(glob.glob(str(ROOT / "data" / "decks" / "*" / "sim" / "*.json")))


def _goldfish():
    return sorted(glob.glob(str(ROOT / "data" / "decks" / "*" / "goldfish_metrics.json")))


def _dig(doc, path):
    """Walk a dotted path, returning a sentinel rather than raising."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ── shape ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(CATALOG))
def test_every_entry_is_well_formed(name):
    row = CATALOG[name]
    assert row["group"] in GROUPS, name
    assert row["status"] in STATUSES, name
    assert row["engine"] in ENGINES, name
    assert row["definition"].strip().endswith("."), f"{name}: definitions are sentences"


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_a_figure_that_exists_names_where_it_comes_from(name):
    """B-3's actual ask: the definition AND the events it derives from."""
    row = CATALOG[name]
    if row["status"] == UNAVAILABLE:
        assert row["engine"] is None, f"{name} is unavailable and names an engine"
        assert row["source"] is None, f"{name} is unavailable and names a source"
        assert row.get("absent", "").strip(), f"{name} is unavailable and says no why"
    else:
        assert row["engine"] is not None, name
        assert row["source"] and row["source"].strip(), name
        assert "absent" not in row, f"{name} is {row['status']} and carries a reason"


def test_absent_means_absent_and_never_zero():
    """A figure nobody measured must not have a value at all.

    `0.0` is a measurement and a reader cannot tell it from one. So an
    unavailable entry carries no engine, no source and no number — only a reason.
    """
    for name, row in metrics.unavailable().items():
        assert set(row) == {"group", "definition", "status", "engine", "source",
                            "absent"}, name
        assert row["engine"] is None and row["source"] is None, name


def test_the_catalog_covers_every_group_the_prd_names():
    seen = {row["group"] for row in CATALOG.values()}
    assert seen == set(GROUPS)
    assert len(CATALOG) >= 25, "PRD §14 lists 27 metrics"


def test_coverage_adds_up():
    total = sum(n for group in metrics.coverage().values() for n in group.values())
    assert total == len(CATALOG)


# ── the claims, against real artifacts ──────────────────────────────────────

FORGE_SEAT_KEYS = {
    "mulligan rate": ("mulligans_taken", "mulligan_kept"),
    "counter frequency": ("counter_events", "mass_counter_events",
                          "proliferate_events"),
    "win rate": ("win_rate", "win_rate_ci95"),
    "damage by source": ("combat_damage_dealt_to_players",
                         "noncombat_damage_dealt_to_players",
                         "combat_damage_taken", "damage_dealt_total"),
}


@pytest.mark.parametrize("name", sorted(FORGE_SEAT_KEYS))
def test_a_published_forge_figure_is_in_every_tracked_record(name):
    """The claim, checked against the artifact rather than against the comment."""
    runs = _runs()
    assert len(runs) >= 15, "the guard iterated almost nothing"
    assert CATALOG[name]["status"] == PUBLISHED
    assert CATALOG[name]["engine"] in ("forge", "both")

    for path in runs:
        seats = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for slug, seat in seats["analysis"]["seats"].items():
            for key in FORGE_SEAT_KEYS[name]:
                assert key in seat, f"{pathlib.Path(path).name} / {slug}: {key}"


GOLDFISH_PATHS = {
    "mean available mana by turn": "mean_available_mana_by_turn",
    "keepable sevens": "opening_hand.keep_first_seven_rate",
    "mulligan rate": "opening_hand.mean_mulligans",
    "commander resolve turn": "commander.mean_cast_turn",
    "missed land drops": "land_drop_hit_rate_by_turn",
    "bodies by turn": "mean_bodies_by_turn",
    "first payoff turn": "targets",
}

#: The channels behind a per-deck `model_*` flag, with the coverage MEASURED
#: when this was written. An opt-in figure must be in some artifacts and absent
#: from others — if it reaches all of them it is no longer opt-in, and if it
#: reaches none the channel is dead.
GOLDFISH_OPT_IN = {
    "cards drawn per game": "mean_extra_cards_drawn_by_turn",
    "turn to lethal, goldfish": "combat.mean_kill_turn",
}


@pytest.mark.parametrize("name", sorted(GOLDFISH_PATHS))
def test_a_published_goldfish_figure_is_in_every_tracked_artifact(name):
    docs = _goldfish()
    assert len(docs) >= 8, "the guard iterated almost nothing"
    assert CATALOG[name]["status"] == PUBLISHED
    assert CATALOG[name]["engine"] in ("goldfish", "both")

    for path in docs:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        found = _dig(doc["metrics"], GOLDFISH_PATHS[name])
        assert found is not None, f"{pathlib.Path(path).parent.name}: {name}"


@pytest.mark.parametrize("name", sorted(GOLDFISH_OPT_IN))
def test_an_opt_in_figure_is_present_on_some_decks_and_absent_on_others(name):
    """OPT_IN is a claim about a SPLIT, so both sides are asserted.

    This state exists because the first cut of the catalog called both of these
    PUBLISHED and the guard caught it: `mean_extra_cards_drawn_by_turn` is in one
    of the ten tracked goldfish artifacts and `combat.mean_kill_turn` in two.
    A boolean would have hidden that, and a figure half the fleet lacks is
    exactly what `benchmark` overrides the declarations to avoid.
    """
    docs = _goldfish()
    assert len(docs) >= 8, "the guard iterated almost nothing"
    assert CATALOG[name]["status"] == OPT_IN

    have = [p for p in docs
            if _dig(json.loads(pathlib.Path(p).read_text(encoding="utf-8"))["metrics"],
                    GOLDFISH_OPT_IN[name]) is not None]
    assert have, f"{name}: no deck opts in — the channel is dead, not opt-in"
    assert len(have) < len(docs), \
        f"{name}: every deck has it, so it is PUBLISHED rather than opt-in"


def test_the_unavailable_figures_are_genuinely_not_in_a_record():
    """The inverse, and the one that stops a plausible zero being added.

    If a key with one of these names ever appears in a run record, either the
    gap closed — in which case the catalog entry is stale and must be rewritten
    with its source — or something is publishing a number it cannot measure.
    """
    banned = {
        "creature power distribution": ("creature_power", "power_p50",
                                        "power_percentiles"),
        "tokens by type": ("tokens_by_type", "treasure_tokens", "blood_tokens"),
        "commander uptime": ("commander_uptime", "turns_with_commander"),
        "draw-engine uptime": ("draw_engine_uptime",),
        "anthem-adjusted power": ("anthem_adjusted_power",),
        "opposing threats answered": ("threats_answered", "permanents_removed"),
    }
    for name in banned:
        assert CATALOG[name]["status"] == UNAVAILABLE, name

    for path in _runs():
        text = pathlib.Path(path).read_text(encoding="utf-8")
        for name, keys in banned.items():
            for key in keys:
                assert f'"{key}"' not in text, \
                    f"{pathlib.Path(path).name} publishes {key}, which the " \
                    f"catalog calls unavailable under {name!r}"


def test_post_wipe_recovery_is_unavailable_even_though_wipe_recovery_exists():
    """The subtlest entry, and the reason `status` is not a presence check.

    `analysis.wipe_recovery` IS published — it measures damage on the wipe turn
    and over the two turns after. The PRD asks for turns to return to pre-wipe
    BOARD POWER, which is a different quantity and an impossible one: Forge logs
    a permanent leaving the battlefield and never one arriving.
    """
    assert CATALOG["post-wipe recovery"]["status"] == UNAVAILABLE
    assert "board" in CATALOG["post-wipe recovery"]["absent"]

    runs = _runs()
    with_block = 0
    for path in runs:
        doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        wipe = doc["analysis"].get("wipe_recovery")
        if wipe:
            with_block += 1
            assert "damage_on_wipe" in wipe or wipe.get("available") is False
            assert "board_power_before" not in wipe
    assert with_block >= 1, "no record carried a wipe_recovery block at all"


# ── the derivable ones are honestly labelled ────────────────────────────────

def test_a_derivable_figure_names_data_that_is_actually_in_the_record():
    """DERIVABLE is a promise that the raw data is there. Checked."""
    runs = _runs()
    assert runs
    doc = json.loads(pathlib.Path(runs[-1]).read_text(encoding="utf-8"))

    # placement: elimination turns per seat, per game.
    assert any("eliminated_turn" in p
               for g in doc["games"] for p in g["per_seat"].values())
    # seat effect: the seat order and the winner, per game — and `seat_order`
    # arrived WITH seat rotation, so only records made after it carry one. The
    # catalog says three of nineteen; the guard asserts the shape rather than
    # the count, which moves every time a run is made.
    with_order = [p for p in runs
                  if any("seat_order" in o for o in
                         json.loads(pathlib.Path(p).read_text(encoding="utf-8"))["outcomes"])]
    assert with_order, "no record carries seat_order at all"
    assert len(with_order) < len(runs), \
        "every record carries it now — the catalog's caveat is stale"
    newest = json.loads(pathlib.Path(with_order[-1]).read_text(encoding="utf-8"))
    assert all("seat_order" in o and "winner" in o for o in newest["outcomes"])

    for name in ("placement", "seat effect"):
        assert CATALOG[name]["status"] == DERIVABLE, name
        assert "nothing" in CATALOG[name]["source"], \
            f"{name} must say what is missing, not just where the data is"


# ── the split between the engines is the measured one ───────────────────────

def test_everything_about_the_library_or_a_board_arrival_is_off_forge():
    """The measurement that decides half the catalog, asserted as policy.

    Forge emits two zone transitions and neither is `from Library` or
    `to Battlefield`. So no entry may claim Forge for drawing, tutoring,
    recursion, or what arrives on a board — and the ones that need those are
    goldfish-only or unavailable.
    """
    library_bound = ("cards drawn per game", "turns with empty hand",
                     "draw-engine uptime")
    board_bound = ("bodies by turn", "post-wipe recovery", "commander uptime",
                   "creature power distribution", "anthem-adjusted power")

    for name in library_bound + board_bound:
        engine = CATALOG[name]["engine"]
        assert engine in ("goldfish", None), f"{name} claims {engine}"


def test_the_zone_limit_is_quoted_from_one_constant():
    """A number restated in six places is six things to keep true."""
    citing = [n for n, r in CATALOG.items()
              if metrics.FORGE_ZONE_LIMIT in (r.get("caveat") or "")
              or metrics.FORGE_ZONE_LIMIT in (r.get("absent") or "")]
    assert len(citing) >= 4, "the constant exists to be shared"
    assert "100-game pod run" in metrics.FORGE_ZONE_LIMIT


# ── it stays joined to the report that already had definitions ──────────────

def test_the_net_change_registry_is_still_the_branch_reports_own():
    """One catalog, and one report-row registry, deliberately not merged.

    `net_change.METRICS` defines the THIRTEEN ROWS of a branch report, keyed by
    the label the report prints. This catalog defines the figures a simulation
    measures, keyed by PRD §14's names. Collapsing them would force one key
    space on two questions; what matters is that both exist and both are
    checked, which they are — see `test_pilot_net_change.py`.
    """
    from manamap.pilot import net_change

    assert set(net_change.METRICS) == {label for label, *_ in net_change.ROWS}
    for spec in net_change.METRICS.values():
        assert spec["unit"] in ("rate", "mean")


# ── the catalog answers the question the PRD asked ──────────────────────────

def test_every_metric_a_pod_night_problem_needs_is_in_the_catalog():
    """`PROBLEMS` is a mapping onto `CATALOG`, so a rename must break it.

    Without this the table rots into prose: a metric gets renamed, the row keeps
    naming the old one, and "can the bench measure this" silently answers about
    a figure that no longer exists.
    """
    for problem, decks, needs in metrics.PROBLEMS:
        assert decks, problem
        assert needs, problem
        for need in needs:
            assert need in CATALOG, f"{problem!r} needs {need!r}, not in the catalog"


def test_the_answer_to_each_problem_is_derived_not_typed():
    rows = metrics.answerable()
    assert len(rows) == len(metrics.PROBLEMS)
    for row in rows:
        available = [n for n in row["needs"]
                     if CATALOG[n]["status"] != UNAVAILABLE]
        assert row["have"] == available, row["problem"]
        assert row["state"] == ("full" if len(available) == len(row["needs"])
                               else "none" if not available else "partial")


def test_the_honest_headline_is_that_most_of_them_are_not_answerable():
    """The finding, pinned so a regression in either direction is visible.

    Two of the six pod-night problems are fully answerable today, two partly,
    and two not at all — and all three RESILIENCE metrics are unavailable, which
    is why "board dies to wipes, no value" has no route. If this number moves,
    either a gap closed (good, and the catalog must say how) or a figure was
    added that cannot be measured.
    """
    rows = metrics.answerable()
    by_state = {s: sum(1 for r in rows if r["state"] == s)
                for s in ("full", "partial", "none")}
    assert by_state == {"full": 2, "partial": 2, "none": 2}, by_state

    resilience = metrics.by_group("resilience")
    assert all(r["status"] == UNAVAILABLE for r in resilience.values())
    assert len(resilience) == 3


def test_the_catalog_renders_without_a_deck_on_disk():
    """The CLI view is prose over the module and must not touch an artifact."""
    text = metrics.format_catalog()
    for group in GROUPS:
        assert group.upper().replace("_", " ") in text
    assert "unavailable" in text and "opt-in" in text
    assert metrics.format_problems().count("PRD") == 1
