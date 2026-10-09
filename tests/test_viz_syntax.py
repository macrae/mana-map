"""Every script under viz/js parses (2026-10-09).

Two agents each added a `const pr` to the same function of `mana-map.js`; the
cherry-picks merged without a textual conflict and the page died at parse time
("Identifier 'pr' has already been declared"), which no unit test saw and every
browser test reported as "the projection never loaded". A parse check is a second
of `node --check`, so it runs in the unit tier whenever node is on the PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

VIZ_JS = Path(__file__).resolve().parents[1] / "viz" / "js"
SCRIPTS = sorted(VIZ_JS.glob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the PATH")
@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_every_viz_script_parses(script):
    proc = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr.strip()


def test_the_check_covers_the_scripts_the_pages_load():
    assert len(SCRIPTS) >= 10, [s.name for s in SCRIPTS]
