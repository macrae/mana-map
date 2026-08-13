"""Pilot: form-check a named deck map.

Split from `merge_deck_map` because the pilot registry dispatches
`module.main(args)` and has no `module:function` form — and because validating is
a thing you want to do without a merge in front of it, on a map an agent named
three sessions ago.

The checks live in `merge_deck_map.validate`; this is the CLI tail.
"""

from manamap.pilot.merge_deck_map import main_validate


def main(args):
    main_validate(args)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-deck-map <slug>`.")
