"""`deck-branch stage` takes one swap a call (2026-11-23): `--out A --in B --out C --in D`
kept only the last pair and staged one swap where the pilot asked for two, silently. A
second use of a once-per-call flag is refused, never overwritten."""
import argparse

import pytest

from manamap.pilot.registry import _OncePerCall


def _parser():
    p = argparse.ArgumentParser(prog="stage")
    p.add_argument("--out", dest="swap_out", default=None, action=_OncePerCall)
    p.add_argument("--in", dest="swap_in", default=None, action=_OncePerCall)
    return p


def test_one_swap_parses():
    a = _parser().parse_args(["--out", "Island", "--in", "Port Town"])
    assert (a.swap_out, a.swap_in) == ("Island", "Port Town")


def test_a_second_pair_is_refused_not_overwritten():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--out", "Island", "--in", "Port Town",
                              "--out", "Island", "--in", "City of Brass"])
