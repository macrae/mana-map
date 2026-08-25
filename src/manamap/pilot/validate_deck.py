"""Pilot: validate a fetched deck against its format's rules.

Every rule here reads `pilot/formats.py` rather than a literal. It used to
hardcode `100`, `1..2 commanders` and singleton inline — correct for Commander,
and the only copy of those numbers that no shared constant could reach.
"""

from manamap.pilot import formats
from manamap.pilot.common import load_deck_cards, report_errors


def validate(doc, spec=None):
    """Return a list of human-readable error strings (empty = valid).

    Sideboard entries (tokens, art cards, spare copies) are excluded from the
    size, singleton and colour-identity checks.
    """
    spec = spec or formats.DEFAULT
    errors = []
    cards = doc.get("cards", [])
    total = sum(c.get("quantity", 0) for c in cards)
    if total != spec.deck_size:
        errors.append(f"Deck has {total} cards, expected exactly {spec.deck_size}")

    commanders = [c for c in cards if c.get("is_commander")]
    if spec.commanders:
        # The upper bound is `commanders + 1` rather than a literal 2: Commander
        # requires one and allows a partner pair, so the allowance is "one more
        # than required" and stays true if a format ever required two.
        lo, hi = spec.commanders, spec.commanders + 1
        if not lo <= len(commanders) <= hi:
            errors.append(
                f"Expected {lo}-{hi} commanders flagged, found {len(commanders)} "
                f"({', '.join(c['name'] for c in commanders) or 'none'})"
            )

    if spec.singleton:
        for c in cards:
            is_basic = spec.basics_exempt and "Basic" in c.get("type_line", "")
            if c.get("quantity", 0) > spec.max_copies and not is_basic:
                errors.append(f"Singleton violation: {c['name']} x{c['quantity']}")

    if spec.colour_identity and commanders:
        identity = set()
        for c in commanders:
            identity.update(c.get("color_identity", []))
        for c in cards:
            outside = set(c.get("color_identity", [])) - identity
            if outside:
                errors.append(
                    f"Color identity violation: {c['name']} is {sorted(outside)} "
                    f"outside commander identity {sorted(identity) or ['C']}"
                )
    return errors


def main(args):
    spec = formats.get(getattr(args, "format", None))
    doc = load_deck_cards(args.slug)
    errors = validate(doc, spec)
    report_errors(args.slug, errors)
    commanders = [c["name"] for c in doc["cards"] if c["is_commander"]]
    print(f"OK: {spec.deck_size} cards ({spec.name}), commander: {', '.join(commanders)}")
