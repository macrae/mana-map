"""Browser behaviour: the fleet page (viz/fleet.html).

THE BEHAVIOUR CORE (2026-11-22). Split from `test_viz_behaviour.py` (272 tests) and cut
to what a page must DO: boot without a JS error, read and write the right files,
land deep links, and show exactly what the artifacts hold. Wording, layout, pixels,
timing and JS source text are not tested here — the pages change fast and those
tests broke on every change. A new test asserts a behaviour, never a phrase.
"""
from __future__ import annotations
import base64
import json
import pathlib
import re
import pytest
from conftest import A_BRANCH, requires_branch, requires_deck
from manamap.config import SYNERGY_RULES
from conftest_viz import await_projection  # noqa: F401
from conftest_viz import (  # noqa: F401
    BOOT_TIMEOUT_MS, _record, canvas_page, corpus_count, discover_page, page,
    idle, serve_watchlist_fixture, still_page,
)
pytestmark = pytest.mark.browser


def _fig(value, **kw):
    d = {"value": value, "source": "benchmark.json#x", "n": 1000}
    d.update(kw)
    return d


def _fleet_row(slug, group="B3", basis="target", dti=None, k8=0.5, apart=False, member=True):
    bench = {"kill_by_turn_8": _fig(k8, ci95=[k8 - 0.02, k8 + 0.02]),
             "kill_by_turn_6": _fig(k8 / 2, ci95=[k8 / 2 - 0.02, k8 / 2 + 0.02]),
             "missed_land_drop_rate": _fig(0.2, ci95=[0.18, 0.22])}
    return {"slug": slug, "format": None, "stage": "sleeved" if member else "bench", "apart": apart,
            "lifecycle": None, "member": member and not apart, "decklist_sha256": "a" * 64,
            "bracket": {"floor": 3, "floor_name": "Upgraded", "target": 3 if basis == "target" else None,
                        "group": group, "group_basis": basis, "dti_bracket": 3 if dti else None},
            "benchmark": bench,
            "diagnostic": {"absent_because": "no diagnostic.json", "source": "diagnostic.json"},
            "own_goldfish": {"comparable": False},
            "dti": ({"score": _fig(dti, as_of="2026-10-18"), "max_score": 96, "bracket": 3, "as_of": "2026-10-18",
                     "stale_list": False, "readings_n": 1, "benchmarks": {"R1": "A", "I2": "F"},
                     "omni": {}, "floor_vs_dti": {"relation": "equal"}}
                    if dti is not None else
                    {"absent_because": "no deckcheck.json — scan on DeckCheck", "source": "deckcheck.json"}),
            "table": {"absent_because": "no Forge run on file", "source": "info.json#simulation"},
            "record": {"games": 2, "win": 1, "loss": 1, "draw": 0, "last_played": "2026-10-01",
                       "cause_counts": {}, "by_version": [], "unmatched_games": []},
            "survey": {"absent_because": "no --feel scores logged", "source": "log.jsonl#feel"},
            "price": {"absent_because": "no prices.json", "source": "prices.json"},
            "placements": {"group": {"kill_by_turn_8": "no band (n=2)"}, "fleet": {}},
            "rating": None}


def _fleet_doc(rows):
    groups = {}
    for r in rows:
        if r["member"]:
            groups.setdefault(r["bracket"]["group"], []).append(r["slug"])
    return {"schema_version": 1, "decks": rows,
            "groups": {g: {"members": m, "n": len(m),
                           "bands": {"kill_by_turn_8": ({"min": 0.4, "median": 0.5, "max": 0.6, "n": 3} if len(m) >= 3
                                                        else {"absent_because": f"fewer than 3 decks carry it (n={len(m)})",
                                                              "source": "benchmark.kill_by_turn_8"})}}
                       for g, m in groups.items()},
            "fleet_bands": {},
            "comparisons": {"floor_vs_dti": [], "dti_history": {}, "benchmark_grades": {"rows": {}, "distribution": {}, "groups": {}, "group_names": {}},
                            "kill_curves": {r["slug"]: {str(t): 0.05 * t for t in range(1, 11)} for r in rows},
                            "mana_curves": {}, "land_curves": {}, "prices": [],
                            "spearman": {"dti_vs_kill_by_8": {"verdict": "NOT ANSWERABLE", "n": 1, "why": "needs 10",
                                                              "what_it_would_take": "more decks"}}},
            "limits": ["Benchmark figures are an unopposed ceiling."],
            "meta": {"harness": {"version": 1, "iterations": 1000, "seed": 42}, "min_group": 3,
                     "no_composite": "no score here"}}


def _fleet(browser, viz_server, doc, ratings=None):
    page = browser.new_page()
    import json as _json
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/data/fleet/analytics.json*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=_json.dumps(doc)))
    page.route("**/data/decks/index.json*", lambda route: route.fulfill(
        status=200, content_type="application/json", body=_json.dumps({"decks": []})))

    def _rating(route):
        slug = route.request.url.split("/")[-2]
        body = (ratings or {}).get(slug)
        if body is None:
            route.fulfill(status=404, body="")
        else:
            route.fulfill(status=200, content_type="application/json", body=_json.dumps(body))
    page.route("**/data/decks/*/rating.json", _rating)
    page.goto(f"{viz_server}/viz/fleet.html")
    idle(page)
    page.js_errors = errors
    return page


def test_the_fleet_table_draws_one_row_per_deck_and_an_absence_as_a_dash_with_its_reason(browser, viz_server):
    doc = _fleet_doc([_fleet_row("alpha", dti=40), _fleet_row("bravo"), _fleet_row("charlie", group="B4", apart=True),
                      _fleet_row("delta", member=False, k8=0.95)])
    page = _fleet(browser, viz_server, doc)
    try:
        rows = page.query_selector_all("#fleet-table tbody tr.t-head")
        assert len(rows) == 2, "the fleet is the sleeved decks: the apart deck and the bench deck are not drawn"
        assert "delta" not in page.text_content("#fleet-table") and "charlie" not in page.text_content("#fleet-table")
        assert page.text_content("#fleet-table tr.t-total th").strip() == "2 sleeved"
        assert len(page.query_selector_all("#fleet-kill .fl-multiple")) == 2
        # bravo has no DTI: a dash whose title is the artifact's reason, never a zero.
        bravo = [r for r in rows if "bravo" in r.text_content()][0]
        dash = bravo.query_selector("td:nth-child(4) .t-none")
        assert dash is not None and "scan on DeckCheck" in dash.get_attribute("title")
        assert "40/96 B3" in page.text_content("#fleet-table")
        # The kill-by-8 figure prints its interval as text.
        assert "[48.0%, 52.0%]" in page.text_content("#fleet-table")
        assert page.js_errors == []
    finally:
        page.close()


def test_a_rating_file_is_drawn_only_with_the_coaching_tier_and_sorted_by_score(browser, viz_server):
    doc = _fleet_doc([_fleet_row("alpha"), _fleet_row("bravo"), _fleet_row("charlie")])
    rating = {"tier": "★", "label": "This rating is ★ coaching.", "score": 66.4, "score_ci": [60.0, 70.0],
              "grade": "B", "grade_display": "B", "bracket": {"key": 3},
              "rank": {"position": 1, "of": 3, "acceptability": {"1": 0.62, "2": 0.3, "3": 0.08}},
              "uncertainty": {"level": 0.9},
              "concordance": {"level": "mixed", "instruments": {"speed": 1.0, "mana": 2.0}, "outliers": [], "group": None},
              "sensitivity": {"leave_one_out": {"speed": {"score": 58.1, "grade": "C"}},
                              "flips": [{"component": "speed", "share_now": 0.2, "share_at_flip": 0.31,
                                         "boundary": 65, "grade_at_flip": "C", "direction": "raise"}]},
              "components": [{"name": "speed", "sub_score": 71, "method": "m"}],
              "withheld": [{"component": "forge", "reason": "no run"}]}
    divergent = dict(rating, score=70.1, grade="B", grade_display="C–B",
                     rank={"position": 1, "of": 3, "acceptability": {"1": 0.9, "2": 0.1, "3": 0.0}},
                     concordance={"level": "divergent", "instruments": {"speed": 3.0, "dti": 1.0},
                                  "outliers": [{"instrument": "dti", "rank": 1.0, "side": "better"}],
                                  "group": {"kendall_w": 0.12, "p": 0.69}})
    page = _fleet(browser, viz_server, doc, ratings={
        "alpha": rating,
        "bravo": dict(rating, tier="◆", score=99.0),       # not a rating: no ★ tier
        "charlie": divergent,
    })
    try:
        rows = page.query_selector_all("#fleet-rating .fl-rating-row")
        assert [r.text_content().split()[2] for r in rows] == ["charlie", "alpha"]
        assert "★ COACHING" in page.text_content("#fleet-rating h2")
        assert "2 of 3 sleeved decks rated" in page.text_content("#fleet-rating")
        assert "withheld: forge" in page.text_content("#fleet-rating")
        # Rating v2: the divergent deck shows its grade RANGE, every row its P(1st), its
        # concordance level, the 90% label on the interval and the nearest flip.
        assert rows[0].text_content().startswith("★ C–B") and rows[1].text_content().startswith("★ B")
        assert "P(1st) 90%" in rows[0].text_content() and "P(1st) 62%" in rows[1].text_content()
        assert page.query_selector('#fleet-rating .fl-cc[data-level="divergent"]') is not None
        assert "dti" in page.query_selector('#fleet-rating .fl-cc[data-level="divergent"]').get_attribute("title")
        assert "90% [60.0, 70.0]" in rows[1].text_content()
        assert "without speed 58.1 C" in rows[1].text_content() and "share 0.31" in rows[1].text_content()
        # The table's rating column carries the badge for rated decks and a dash otherwise.
        assert page.query_selector_all("#fleet-table .fl-star").__len__() == 2
        assert "★ C–B" in page.text_content("#fleet-table")
        assert page.js_errors == []
    finally:
        page.close()
