"""`close` — the diagnostic's bottleneck becomes a candidate pool."""

import json

import pytest

from conftest import requires_data, requires_deck
from manamap.pilot import close
from conftest import ROOT

SLUG, BRANCH = "ur-dragon", "treasure-v2"


@requires_deck
def test_the_component_defaults_to_the_diagnostics_own_bottleneck():
    """Read it, never recompute it. `diagnostic.engine()` owns that figure and a
    second opinion here would be a second answer to one question."""
    got = close.close(SLUG, BRANCH, limit=4)
    assert got["component"]["why"] == "the diagnostic's own bottleneck"
    diag = json.loads(
        (close.deck_dir(SLUG, BRANCH) / "diagnostic.json").read_text())
    assert (got["component"]["label"]
            == diag["engine"]["bottleneck"]["label"])


@requires_deck
def test_naming_a_component_matches_the_shortest_hit():
    """Longest-first is the `_named_axis` rule: `interaction` and
    `interaction-breadth` are both real names one module over, and a
    shortest-first scan silently checks the wrong one."""
    got = close.close(SLUG, BRANCH, component="MULTIPLIER", limit=4)
    assert "MULTIPLIER" in got["component"]["label"]
    assert "named on the command line" in got["component"]["why"]
    with pytest.raises(SystemExit) as e:
        close.close(SLUG, BRANCH, component="no such component")
    assert "declares" in str(e.value)


@requires_deck
def test_the_signature_is_the_rarest_shared_role_not_the_commonest():
    """MEASURED BEFORE IT SHIPPED.

    `_closers` takes the most COMMON shared role, which for any component
    holding two creatures is `threat:body` — 62.4% of the classified corpus.
    Asked to widen ur-dragon's multipliers it offered Birds of Paradise. Across
    all 91 declared components on the fleet the most-common pick lands above 40%
    for 29 of them; rarest-first lands there for 7.
    """
    got = close.close(SLUG, BRANCH, component="MULTIPLIER", limit=8)
    sig = got["routes"]["signature"]
    assert sig["available"], sig.get("why")
    assert sig["signature"] == "doubler:tokens", sig["signature"]
    # A signature that small is the point: it is a real job, not "is a creature".
    assert sig["total"] < 50, sig["total"]


@requires_data
def test_the_broad_signature_guard_is_rederived_from_the_fleet():
    """So the constant cannot outlive its evidence — the `BROAD_GROUP` pattern.

    Asserts the two things that justify the rule: rarest-first is a large
    improvement over most-common, and the guard's threshold sits above what
    rarest-first normally produces so it fires on the genuine no-signature case
    rather than on correct data.
    """
    import collections
    import glob

    from manamap.pilot.common import load_card_roles, load_deck_cards
    roles = load_card_roles()
    freq = close._role_frequency(roles)
    common, rarest = [], []
    for tp in sorted(glob.glob(str(ROOT / "data/decks/*/goldfish_targets.json"))):
        slug = tp.split("/")[2]
        try:
            held = {c["name"] for c in load_deck_cards(slug)["cards"]}
        except FileNotFoundError:
            # ONLY a missing fixture is skippable. A bare `except Exception`
            # made a deck that CRASHES the code under test indistinguishable
            # from one that is absent — the failure this suite exists to see.
            continue
        for t in (json.load(open(tp)).get("targets") or []):
            for g in (t.get("need") or []):
                present = [n for n in (g.get("any_of") or []) if n in held]
                if len(present) < 2:
                    continue
                rc = collections.Counter(r for n in present for r in roles.get(n, []))
                shared = [r for r, n in rc.items() if n > 1]
                if not shared:
                    continue
                common.append(freq[max(shared, key=lambda r: rc[r])])
                rarest.append(freq[min(shared, key=lambda r: freq[r])])
    if len(rarest) < 20:
        pytest.skip("too few declared components on this checkout to calibrate")
    over_c = sum(1 for x in common if x > close.SIGNATURE_MAX_SHARE)
    over_r = sum(1 for x in rarest if x > close.SIGNATURE_MAX_SHARE)
    assert over_r * 2 < over_c, (
        f"rarest-first stopped being an improvement: {over_r} vs {over_c} of "
        f"{len(rarest)} components land on a role over "
        f"{close.SIGNATURE_MAX_SHARE:.0%} of the corpus")
    assert over_r / len(rarest) < 0.25, (
        f"the guard now fires on {over_r}/{len(rarest)} components — at that "
        f"rate it is firing on correct data, which is worse than no guard")


@requires_deck
def test_the_two_routes_disagree_and_the_overlap_is_marked():
    """ROUTE-DISAGREEMENT CONTROL. If they agreed everywhere one is redundant
    and nothing in the output would tell you which."""
    got = close.close(SLUG, BRANCH, component="MULTIPLIER", limit=12)
    fn = {r["name"] for r in got["routes"]["function"]["cards"]}
    sg = {r["name"] for r in got["routes"]["signature"]["cards"]}
    assert fn and sg
    assert fn - sg, "the function route found nothing the signature route missed"
    assert sg - fn, "the signature route found nothing the function route missed"
    assert set(got["both_routes"]) == fn & sg
    assert got["pool"] == sorted(fn | sg)


@requires_deck
def test_it_never_offers_a_card_the_deck_already_holds_or_cannot_play():
    got = close.close(SLUG, BRANCH, limit=20)
    from manamap.pilot import card_pool
    from manamap.pilot.common import load_deck_cards
    held = {c["name"] for c in load_deck_cards(SLUG, BRANCH)["cards"]}
    pool = card_pool.load_pool() or {}
    for name in got["pool"]:
        assert name not in held, name
        # Halving Season and Hosting Season rank top of the raw centroid and are
        # ACORN cards — commander-illegal. A miner that returns them looks
        # cleverer and is wrong.
        assert pool[name]["legal"], name


@requires_deck
def test_the_function_route_degrades_rather_than_raising(monkeypatch):
    """FRESH-CLONE CONTROL. `embeddings_ability.npy` is gitignored, so this
    route is simply absent on a fresh checkout. It must still return a list and
    NAME the missing route — a miner that silently returns half its answer is
    the `libraryNames` fallback bug, where a plausible short answer is worse
    than an honest refusal."""
    from manamap import config

    class _Gone:
        def exists(self):
            return False
        name = "embeddings_ability.npy"

    monkeypatch.setattr(config, "ABILITY_EMBEDDINGS_PATH", _Gone())
    got = close.close(SLUG, BRANCH, component="MULTIPLIER", limit=6)
    fn = got["routes"]["function"]
    assert fn["available"] is False
    assert "gitignored" in fn["why"]
    assert got["routes"]["signature"]["available"], "the other route died too"
    assert got["pool"], "no candidates at all on a fresh clone"
