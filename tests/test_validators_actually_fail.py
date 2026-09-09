"""A validator that prints FAIL must exit non-zero.

`registry.run_pilot_step` calls `main(args)` and DISCARDS the return value.
Twenty-one validators go through `common.report_errors`, which calls
`sys.exit(1)`. Two hand-rolled the same tail and wrote `return 1` — so they
printed

    FAIL ur-dragon captain's log (3 error(s)):
      - nights[2026-08-22].logs['ship'] is not a log kind

and exited 0. A gate that reports failure and success in the same breath.

It was invisible for as long as the two artifacts were valid, and it surfaced
the moment a schema change made them invalid: the CLI said FAIL, the test suite
stayed green, and the tracked-artifact sweep passed five stale files.

`docs/pilot.md` states the rule this file now enforces: "Read exit codes
directly. `| tail` swallows them, which has burned this repo four times."
"""

import re

import pytest

from manamap import config
from manamap.pilot.deck_status import VALIDATED

VALIDATORS = sorted(set(VALIDATED.values()))


@pytest.mark.parametrize("dotted", VALIDATORS, ids=lambda d: d.rsplit(".", 1)[-1])
def test_a_validator_signals_failure_through_the_exit_code(dotted):
    """Either it delegates to `report_errors`, or it calls `sys.exit` itself.

    A bare `return 1` reaches a dispatcher that throws it away, which is the
    defect this file exists for.
    """
    import importlib
    import inspect

    module = importlib.import_module(dotted)
    src = inspect.getsource(module)

    # THE DEFECT, stated exactly: a validator that signals failure by RETURNING
    # a code. Not "does main call sys.exit" — several validators delegate to a
    # shared tail or to another module's `main`, and a check that demanded the
    # call appear in `main` itself fired on `validate_deck_map`, which is
    # correct and one line long.
    offenders = [n for n, line in enumerate(src.splitlines(), 1)
                 if re.fullmatch(r"\s+return 1\s*", line)]
    assert not offenders, (
        f"{dotted} returns 1 at line(s) {offenders} — `run_pilot_step` calls "
        f"`main(args)` and DISCARDS the result, so this prints FAIL and exits 0. "
        f"Use `common.report_errors` or `sys.exit(1)`.")
    assert "report_errors" in src or "sys.exit" in src or "main_validate" in src, (
        f"{dotted} has no visible failure path at all")


def test_the_dispatcher_still_discards_return_values():
    """The reason the rule above exists, pinned. If `run_pilot_step` ever starts
    propagating a return code, this fails and the rule can relax — which is the
    honest way for a workaround to expire."""
    import inspect

    from manamap.pilot import registry

    src = inspect.getsource(registry.run_pilot_step)
    assert "return\n" in src and "main(args)" in src
    assert "sys.exit(importlib" not in src and "return importlib" not in src, (
        "the dispatcher now propagates a code — revisit test_validators_actually_fail")
