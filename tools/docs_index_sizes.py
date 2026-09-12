"""Re-derive the size column in `docs/README.md`. Run after editing any doc.

The column was stale on seventeen of its rows and mixed two units — the gotchas
rows quoted BULLET counts (99 for a file of 1,731 lines) while every other row
quoted lines. `test_docs_counts.test_the_docs_index_is_complete_and_its_sizes_are_real`
gates it now, and this is how you satisfy the gate without counting by hand.

The unit is LINES, one unit, every row. `make docs-sizes`.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    index = ROOT / "docs" / "README.md"
    text = index.read_text(encoding="utf-8")
    changed = []

    def fix(match):
        name, stated = match.group(1), match.group(2)
        target = ROOT / "docs" / name
        if not target.exists():          # history rows carry a path prefix
            return match.group(0)
        real = str(len(target.read_text(encoding="utf-8").splitlines()))
        if real != stated.replace(",", ""):
            changed.append(f"{name}: {stated} -> {real}")
        return match.group(0).replace(f"| {stated} |", f"| {real} |", 1)

    out = re.sub(r"\[([a-z0-9\-.]+\.md)\]\(\1\)\*{0,2} \| ([\d,]+) \|", fix, text)
    if out != text:
        index.write_text(out, encoding="utf-8")
    for line in changed:
        print(f"  {line}")
    print(f"{len(changed)} row(s) updated in docs/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
