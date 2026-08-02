"""Every pilot module must be able to run, not merely to import.

`build_deck.build()` called `load_card_roles` and `load_combo_details` without
importing either. The module imported fine, every unit test passed — because all
of them exercise the pure helpers — and `manamap pilot build-deck` died with a
NameError on the first real invocation. A module whose entry point nothing calls
is a module nothing tests.

This scan is the cheap general guard: it walks each module's AST and reports any
name that is loaded but never bound by an import, assignment, def, class,
argument, comprehension or except-clause. It catches the whole class of defect
across the subsystem without needing artifacts.
"""

import ast
import builtins
import pathlib

import pytest

PILOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "manamap"
MODULES = sorted(PILOT.rglob("*.py"))


def _bound_names(tree):
    bound = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    return bound


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_module_has_no_unbound_names(path):
    tree = ast.parse(path.read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - _bound_names(tree))
    assert not missing, f"{path.name} uses undefined name(s): {missing}"
