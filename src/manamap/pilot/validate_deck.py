"""Pilot: validate a fetched deck's Commander invariants."""

from manamap.pilot.common import load_deck_cards, report_errors


def validate(doc):
    """Return a list of human-readable error strings (empty = valid).

    Sideboard entries (tokens, art cards, spare copies) are excluded from the
    100-card, singleton, and color-identity checks.
    """
    errors = []
    cards = doc.get("cards", [])
    total = sum(c.get("quantity", 0) for c in cards)
    if total != 100:
        errors.append(f"Deck has {total} cards, expected exactly 100")

    commanders = [c for c in cards if c.get("is_commander")]
    if not 1 <= len(commanders) <= 2:
        errors.append(
            f"Expected 1-2 commanders flagged, found {len(commanders)} "
            f"({', '.join(c['name'] for c in commanders) or 'none'})"
        )

    for c in cards:
        is_basic = "Basic" in c.get("type_line", "")
        if c.get("quantity", 0) > 1 and not is_basic:
            errors.append(f"Singleton violation: {c['name']} x{c['quantity']}")

    if commanders:
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
    doc = load_deck_cards(args.slug)
    errors = validate(doc)
    report_errors(args.slug, errors)
    commanders = [c["name"] for c in doc["cards"] if c["is_commander"]]
    print(f"OK: 100 cards, commander: {', '.join(commanders)}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-deck <slug>`.")
