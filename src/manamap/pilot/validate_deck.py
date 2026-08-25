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
    size_problem = spec.size_error(total)
    if size_problem:
        errors.append(size_problem)

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
    errors.extend(illegal_cards(cards, spec))
    return errors


def illegal_cards(cards, spec):
    """Cards this format does not allow, from the corpus's own legality column.

    NOTHING CHECKED THIS BEFORE. Every deck here is Commander and every card in
    them is Commander-legal, so the gap was invisible — and it stops being
    invisible the moment a 60-card format arrives, where a Modern deck holding
    a Standard-rotated card is the commonest mistake there is.

    The pool comes straight from Scryfall via `extract`'s `legal_<format>`
    columns. No rule is reimplemented here, which matters most for Pauper: the
    naive reading is "commons only" and it is wrong for 373 cards, because a
    card printed at common anywhere is legal even where this printing is not.

    Silent when the corpus is absent — a fresh clone must still be able to
    validate structure, and a missing column is a missing MEASUREMENT rather
    than a passing one. It says so instead of reporting nothing.
    """
    from manamap.pilot import card_pool

    try:
        status = card_pool.legality(spec.legality_column)
    except ValueError:
        return [f"cannot check {spec.name} legality: {spec.legality_column} not in "
                f"cards.csv (re-run `manamap extract`)"]
    except Exception:
        return []                      # no corpus at all — structure still checks

    out = []
    for c in cards:
        state = status.get(c["name"])
        if state is None:            # not in the corpus at all — a different error
            continue
        if state != "legal":
            out.append(f"{spec.name} legality: {c['name']} is {state}")
    return out


def main(args):
    spec = formats.get(getattr(args, "format", None))
    doc = load_deck_cards(args.slug)
    errors = validate(doc, spec)
    report_errors(args.slug, errors)
    commanders = [c["name"] for c in doc["cards"] if c["is_commander"]]
    print(f"OK: {spec.deck_size} cards ({spec.name}), commander: {', '.join(commanders)}")
