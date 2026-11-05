"""The report a pilot reads off `manamap pilot try`, driven end to end."""
from argparse import Namespace

from conftest import requires_data, requires_deck


@requires_data
@requires_deck
def test_the_terminal_report_renders(capsys):
    """`--json` was the only path anything drove, so the text report died on a NameError:
    `render` used `net_change`, and only the functions above it imported the module."""
    import json

    from manamap.config import DECKS_DIR
    from manamap.pilot import try_swap
    # A land of the branch's own, so a change to the list cannot break a test of the report.
    doc = json.loads((DECKS_DIR / "sharknado/branches/draw-engine-v1/cards.json").read_text(encoding="utf-8"))
    land = sorted(c["name"] for c in doc["cards"] if "Land" in (c.get("type_line") or "") and "Basic" not in c["type_line"])[0]
    args = Namespace(slug="sharknado", branch="draw-engine-v1", out=[land],
                     in_=["Divination"], iterations=50, seed=1, json=False)
    try_swap.main(args)
    out = capsys.readouterr().out
    assert "GOLDFISH" in out and "Divination" in out
