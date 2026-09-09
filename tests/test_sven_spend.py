"""What Sven costs, and the ceiling he cannot spend past.

The account behind this has NO AUTO-RELOAD: when the balance is gone it is gone.
That changes what the tracking is for — not accounting after the fact, but a
refusal before the next call.

Every figure here is an ESTIMATE from a price table with a date on it, and the
tests assert that it says so. A number that looks authoritative and is not is
worse than no number, which is the same discipline this repo applies to a rate
without its interval.
"""

import json

import pytest

from manamap.sven import llm, spend


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(spend, "LEDGER", tmp_path / "spend.jsonl")
    monkeypatch.delenv("MANAMAP_SVEN_BUDGET_USD", raising=False)


# ── the estimate ──────────────────────────────────────────────────────────

def test_a_typical_turn_is_a_fraction_of_a_cent():
    """The premise of running Haiku by default. If this is ever wrong by an
    order of magnitude, the whole cost argument for the design changes."""
    cost = spend.cost_of(llm.FAST_MODEL, {"in": 5000, "out": 300})
    assert 0.001 < cost < 0.02, f"a Haiku turn costs {cost}"


def test_an_unpriced_model_is_absent_not_free():
    """`None`, never `0.0`. A zero is a measurement and a reader cannot tell it
    from one — the oldest rule on this bench, applied to money."""
    assert spend.cost_of("some-model-with-no-price", {"in": 1000, "out": 100}) is None


def test_unpriced_turns_are_counted_but_never_folded_into_the_total():
    spend.record(llm.FAST_MODEL, {"in": 1000, "out": 100})
    spend.record("unknown-model", {"in": 1_000_000, "out": 1_000_000})
    got = spend.total()
    assert got["turns"] == 2
    assert got["unpriced"] == 1
    assert got["estimated_usd"] < 0.01, "an unpriced turn leaked into the total"


def test_every_report_says_the_figure_is_an_estimate():
    """The console is the authority on what was charged. A local ledger that
    presents itself as a bill is the thing that gets believed at the wrong
    moment."""
    spend.record(llm.FAST_MODEL, {"in": 1000, "out": 100})
    text = spend.report()
    assert "ESTIMATED" in text or "estimated" in text
    assert "console" in text.lower()


# ── the ledger ────────────────────────────────────────────────────────────

def test_a_turn_is_recorded_with_what_it_was_asked():
    spend.record(llm.FAST_MODEL, {"in": 2000, "out": 100}, "is zur ready?")
    row = json.loads(spend.LEDGER.read_text().splitlines()[0])
    assert row["model"] == llm.FAST_MODEL
    assert row["usage"] == {"in": 2000, "out": 100}
    assert row["question"] == "is zur ready?"
    assert row["estimated_usd"] > 0


def test_a_ledger_that_cannot_be_written_does_not_fail_the_turn(monkeypatch):
    """The pilot has already paid for the answer by the time this runs."""
    monkeypatch.setattr(spend, "LEDGER",
                        spend.Path("/nonexistent-root/nope/spend.jsonl"))
    assert spend.record(llm.FAST_MODEL, {"in": 100, "out": 10}) is not None


def test_a_corrupt_line_does_not_break_the_total():
    spend.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    spend.LEDGER.write_text('{"broken\n' + json.dumps(
        {"at": 1, "model": llm.FAST_MODEL, "usage": {"in": 1000, "out": 100},
         "estimated_usd": 0.0015}) + "\n")
    assert spend.total()["turns"] == 1


# ── the ceiling ───────────────────────────────────────────────────────────

def test_the_ceiling_refuses_once_it_is_passed():
    for _ in range(50):
        spend.record(llm.DEEP_MODEL, {"in": 1_000_000, "out": 100_000})
    with pytest.raises(spend.BudgetExceeded, match="ceiling"):
        spend.check()


def test_the_refusal_says_how_to_raise_it():
    """A guard with no override is a guard that gets deleted."""
    for _ in range(50):
        spend.record(llm.DEEP_MODEL, {"in": 1_000_000, "out": 100_000})
    with pytest.raises(spend.BudgetExceeded) as caught:
        spend.check()
    assert "MANAMAP_SVEN_BUDGET_USD" in str(caught.value)


def test_the_ceiling_can_be_disabled_and_raised(monkeypatch):
    for _ in range(50):
        spend.record(llm.DEEP_MODEL, {"in": 1_000_000, "out": 100_000})
    monkeypatch.setenv("MANAMAP_SVEN_BUDGET_USD", "0")
    spend.check()                              # 0 disables; must not raise
    monkeypatch.setenv("MANAMAP_SVEN_BUDGET_USD", "10000")
    spend.check()


def test_the_ceiling_catches_a_runaway_WITHIN_ONE_TURN_not_before():
    """THE LIMITATION, asserted so nobody mistakes this for a hard cap.

    `check()` runs before a turn and reads what has already been spent, so a
    single catastrophic turn — one that blows the whole ceiling by itself —
    completes and IS PAID FOR, and only the turn after it is refused. Bounding
    a turn before it happens would mean predicting its output length, which
    cannot be done.

    What this stops is a LOOP: the second iteration and everything after. That
    is the failure that happens while nobody is watching, and it is the one
    worth stopping.
    """
    spend.check()                              # clean ledger, no refusal
    spend.record(llm.DEEP_MODEL, {"in": 100_000_000, "out": 10_000_000})
    assert spend.total()["estimated_usd"] > spend.budget()
    with pytest.raises(spend.BudgetExceeded):
        spend.check()                          # the NEXT turn is refused


# ── the loop honours it ───────────────────────────────────────────────────

def test_the_loop_refuses_before_it_builds_a_transport():
    """Refused BEFORE the transport exists, so an over-budget question cannot
    reach the network at all. The scripted transport here has an empty script:
    any use of it raises."""
    from manamap.sven import loop

    for _ in range(50):
        spend.record(llm.DEEP_MODEL, {"in": 1_000_000, "out": 100_000})
    empty = llm.ScriptedTurn([])
    frames = dict((k, d) for k, d in loop.run("anything", turn=empty))
    assert empty.seen == [], "the transport was constructed despite the ceiling"
    assert "budget" in frames["done"]["uncacheable"]
