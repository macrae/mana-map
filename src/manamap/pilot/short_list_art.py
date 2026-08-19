"""Pilot: card art for The Short List's ten, so the reader can see what is offered.

LEGACY (2026-08-19): the magazine renderer. It still renders the nine frozen issues from
artifacts nothing regenerates any more (issue_plan.json, the panel keys,
card_roles/mana_base/upgrades, considering.json), and it is replaced by the compact deck
page in docs/manual-v5-spec.md. Do not extend it; internals below are accurate for what it
does.

Every other card the magazine names is in the 99, so `build_manual`'s card linker
already has its image from `cards.json` and can pop a preview on hover. The Short
List is the one department whose subject is deliberately OUTSIDE the deck — ten
cards you do not own yet — and those ten were the only card names in the issue a
reader could not look at. A recommendation you cannot see is a recommendation you
have to go and google, which is the reader leaving the magazine to read it.

Deterministic and network-only-here: it resolves names against Scryfall exactly
the way `fetch_deck` resolves the 99, and writes a tracked sidecar. Nothing is
computed at view time and the renderer never touches the network.

**A SIDECAR, not a field on `considering.json`.** The Short List artifact belongs
to `short-list-analyst` and is fingerprinted by the cache per file; a deterministic
process writing into it would either invalidate the routine on every art refresh or
force the analyst to author image URLs it has no business knowing. Same separation
`merge_deck_map` makes between what was measured and what was named.

Missing sidecar, missing card, missing image: the name renders as plain text
exactly as it does today. Art is an enhancement, never a precondition.
"""

import json

from manamap.pilot.common import deck_dir, resolve_out_path  # noqa: F401
from manamap.pilot.fetch_deck import fetch_collection, stable_image_url

ARTIFACT = "considering_art.json"


def names_from(analysis):
    """The ten card names, in list order, de-duplicated.

    Only the `card` field. `natural_cut` names a card that IS in the 99 and is
    already linked to its own tile by the renderer — resolving it here would mint
    a second, external link to a card the reader can reach on the same page.
    """
    seen, out = set(), []
    for entry in analysis.get("ten") or []:
        name = (entry or {}).get("card")
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def shape(card):
    """The few fields a hover preview and a credit line need — nothing else.

    Deliberately not `fetch_deck.shape_card`: that projects the full cards.json
    schema (legality, quantity, commander flag, faces) and every one of those
    fields would be a claim about a card that is not in this deck.
    """
    images = card.get("image_uris") or {}
    if not images and card.get("card_faces"):
        images = (card["card_faces"][0] or {}).get("image_uris") or {}
    return {
        "name": card.get("name"),
        "image": stable_image_url(images.get("normal")),
        "art_crop": stable_image_url(images.get("art_crop")),
        "scryfall_uri": card.get("scryfall_uri"),
        "type_line": card.get("type_line", ""),
        "mana_cost": card.get("mana_cost", ""),
        "artist": card.get("artist"),
        "set_name": card.get("set_name"),
    }


def build(slug):
    """Resolve the ten and write the sidecar. Returns (doc, not_found)."""
    base = deck_dir(slug)
    source = base / "considering.json"
    if not source.exists():
        raise SystemExit(f"{source} not found — run the-ten for this deck first")
    with open(source) as f:
        analysis = json.load(f)

    names = names_from(analysis)
    if not names:
        raise SystemExit(f"{source} lists no cards under `ten`")

    by_name, not_found = fetch_collection(names)

    # Retry unresolved double-faced names on the FRONT FACE alone. The repo's
    # convention is the full `A // B` form (it is the graph key), and Scryfall's
    # collection endpoint does not accept it for every layout — measured: radagast's
    # "Disciple of Freyalise // Garden of Freyalise" 404s under the full name and
    # resolves instantly under "Disciple of Freyalise". The card is keyed under the
    # name the analyst wrote either way, so nothing downstream has to know.
    # Scryfall answers a front-face query with the card under its CANONICAL full
    # name, so the results merge straight in — looking the reply up by the face we
    # asked about finds nothing and silently drops the card, which is what the
    # first version of this did while reporting "9 of 9".
    retry = {n.split(" // ")[0]: n for n in not_found if " // " in n}
    if retry:
        faces, still_missing = fetch_collection(sorted(retry))
        by_name.update(faces)
        resolved = {full for face, full in retry.items()
                    if face not in still_missing}
        not_found = [n for n in not_found if n not in resolved]

    cards = {}
    for name in names:
        card = by_name.get(name.lower())
        if card:
            cards[name] = shape(card)

    # Sorted so the file is stable under re-fetch: the analyst may reorder the
    # ten between passes and a diff full of moved blocks hides the real change.
    doc = {"slug": slug, "cards": dict(sorted(cards.items()))}
    return doc, not_found


def main(args):
    slug = args.slug
    doc, not_found = build(slug)
    path = deck_dir(slug) / ARTIFACT
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    missing = [n for n, c in doc["cards"].items() if not c.get("image")]
    print(f"{slug}: resolved {len(doc['cards'])} of "
          f"{len(doc['cards']) + len(not_found)} Short List card(s) → {path}")
    if not_found:
        print(f"  NOT FOUND on Scryfall: {', '.join(sorted(not_found))}")
    if missing:
        print(f"  resolved but no image: {', '.join(sorted(missing))}")
    return 0
