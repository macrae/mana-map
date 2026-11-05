"""The report a pilot reads off `manamap pilot try`, driven end to end."""
from argparse import Namespace

from conftest import requires_data, requires_deck


@requires_data
@requires_deck
def test_the_terminal_report_renders(capsys):
    """`--json` was the only path anything drove, so the text report died on a NameError:
    `render` used `net_change`, and only the functions above it imported the module."""
    from manamap.pilot import try_swap
    args = Namespace(slug="sharknado", branch="draw-engine-v1", out=["Spectator Seating"],
                     in_=["Prosperity"], iterations=50, seed=1, json=False)
    try_swap.main(args)
    out = capsys.readouterr().out
    assert "GOLDFISH" in out and "Prosperity" in out
