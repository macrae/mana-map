"""Pilot: form-check `deck_recon.json` — the doctor's dated web reconnaissance.

Recon was the one deck artifact with no gate. It is also the artifact whose whole
value is that every card it names is REAL: `deck-doctor.md`'s MODE recon rule 5 says
each card must be checked "in this commander's colour identity, Commander-legal, and
not already in the 99" before it is recorded, and until now nothing verified that the
agent had done so. On the kinnan recon I checked 110 cards by hand; it passed. That
is exactly the work this repo says must become a mechanical check, or it is re-spent
on every run.

**What is deliberately NOT checked, and why** — each of these fires on correct data,
which trains a reader to ignore the gate:

  * **"Not already in the 99."** It is the charter's own rule and it would fail 4 of
    the 5 tracked recons on 37 card-instances. Recon is DATED and the decklist moves
    under it: kianne's own `_checked_against` note records that its ownership checks
    ran against a V2 working draft that has since been restored to V1. A card
    correctly outside the 99 in August is inside it in September without the artifact
    changing a byte. Reported as a WARN line, never an error.
  * **Age.** `RECON_MAX_AGE_DAYS` is judged by the SKILL because `deck_audit` is
    deterministic and never reads the clock. A time-dependent verdict would flip a
    tracked artifact from green to red with no input change, which also breaks the
    regenerate-and-compare test cache — it keys on file bytes, so nothing would
    invalidate and the failure would arrive out of nowhere.
  * **Required optional keys.** `ownership`, `_checked_against` and
    `commander_text_verified` each appear on exactly ONE deck. They are things a
    particular run had reason to record, not a schema.

`ownership`, when present, IS checked — a claim about what the pilot has is
falsifiable, and `pilot.collection` is the one reader that can falsify it.
"""

import json
from datetime import date

from manamap.pilot import collection
from manamap.pilot.card_pool import load_pool
from manamap.pilot.card_search import commander_identity, deck_names
from manamap.pilot.common import deck_dir, expand_faces, load_json, report_errors

REQUIRED_KEYS = ("slug", "commander", "as_of", "consensus", "findings", "sources")

# The charter's vocabulary (`.claude/agents/deck-doctor.md`, MODE recon).
CONFIDENCE = {"widely agreed", "contested", "one source"}


def _named_cards(doc):
    """`{card name: [where it was named]}` across every finding."""
    out = {}
    for i, f in enumerate(doc.get("findings") or []):
        for name in f.get("cards") or []:
            out.setdefault(name, []).append(f"findings[{i}]")
    return out


def _validate_structure(doc):
    """Only what would make every later check crash or cascade.

    Kept separate from the field checks because the first version returned early on
    ANY shape error, and a `confidence` typo then suppressed a nonexistent-card
    error in the same file — proven when the gate was re-broken four ways and
    reported one. An early return must be about cascade, not about severity.
    """
    errors = [f"missing required key {key!r}" for key in REQUIRED_KEYS if key not in doc]
    if not isinstance(doc.get("findings"), list):
        errors.append("findings is not a list")
    return errors


def _validate_shape(doc, slug):
    errors = []
    if doc.get("slug") != slug:
        errors.append(f"slug is {doc.get('slug')!r} but the artifact lives in {slug}/")
    try:
        date.fromisoformat(str(doc.get("as_of")))
    except (TypeError, ValueError):
        errors.append(f"as_of {doc.get('as_of')!r} is not an ISO date — recon is "
                      f"perishable and the reader has to know how old it is")
    if not doc.get("findings"):
        errors.append("findings is empty — a recon with no findings is a gap report, "
                      "and should say so in `gaps` rather than claim to be recon")
    for i, f in enumerate(doc.get("findings") or []):
        if not f.get("claim"):
            errors.append(f"findings[{i}]: no claim")
        conf = f.get("confidence")
        if conf not in CONFIDENCE:
            errors.append(f"findings[{i}]: confidence {conf!r} not in {sorted(CONFIDENCE)}")
    return errors


def _validate_sources(doc):
    """Every URL a finding leans on must appear in the top-level source list.

    The charter requires each cited URL to have been FETCHED; the top-level list is
    where that is recorded with a title and what it contributed. A finding citing a
    URL that appears nowhere else is a citation nobody can audit.
    """
    errors = []
    known = {s.get("url") for s in (doc.get("sources") or []) if isinstance(s, dict)}
    for i, f in enumerate(doc.get("findings") or []):
        for url in f.get("sources") or []:
            if url not in known:
                errors.append(f"findings[{i}]: source {url} is not in the top-level "
                              f"sources list, so nothing records that it was fetched")
    for i, s in enumerate(doc.get("sources") or []):
        if isinstance(s, dict) and not s.get("url"):
            errors.append(f"sources[{i}]: no url")
    return errors


def _resolve(pool, name):
    """A named card's corpus row, accepting either FACE of a double-faced card.

    `load_pool` keys only on the joined `"A // B"` form, so a recon naming
    `Legion's Landing` — which is how everyone writes it, and how the decklist writes
    it — looked like a card that does not exist. It failed on three real cards across
    two tracked recons the first time this validator ran, and every one was correct
    data. `expand_faces` is the repo's answer to this everywhere else.
    """
    rec = pool.get(name)
    if rec is not None:
        return rec
    faces = expand_faces(name)
    for key, row in pool.items():
        if faces & expand_faces(key):
            return row
    return None


def _validate_cards(doc, slug, pool):
    """Every named card real, Commander-legal, and inside the commander's identity."""
    errors = []
    if pool is None:                       # pragma: no cover — fresh clone
        return errors
    try:
        identity = commander_identity(slug)
    except FileNotFoundError:              # pragma: no cover — no cards.json yet
        identity = None
    for name, where in sorted(_named_cards(doc).items()):
        rec = _resolve(pool, name)
        if rec is None:
            errors.append(f"{where[0]}: {name!r} is not in cards.csv — a recommendation "
                          f"nobody can look up is not a recommendation")
            continue
        if not rec["legal"]:
            errors.append(f"{where[0]}: {name!r} is not Commander-legal")
        if identity is not None and not rec["color_identity"] <= identity:
            errors.append(f"{where[0]}: {name!r} is {''.join(sorted(rec['color_identity']))}, "
                          f"outside {slug}'s {''.join(sorted(identity)) or 'colourless'} identity")
    return errors


def _validate_ownership(doc):
    """`ownership` is a falsifiable claim about the pilot's cards, so falsify it.

    "Owned" means in a BOX — `data/decks/` holds build plans as well as assembled
    decks and nothing tells them apart, so deck membership is deliberately not
    counted (see `pilot.collection`). A recon that says a card is owned when no box
    holds it has sent the pilot to the shelf for something that is not there.

    This check's first real catch was not an agent error: it reported the tracked
    collection as two boxes short of the physical one.
    """
    errors = []
    own = doc.get("ownership")
    if not isinstance(own, dict):
        return errors
    have = collection.owned_names()
    for name, claimed in sorted(own.items()):
        actual = bool(expand_faces(name) & have)
        if bool(claimed) != actual:
            errors.append(
                f"ownership[{name!r}] says {bool(claimed)} but no box in "
                f"COLLECTION_DIR holds it (deck membership does not count)")
    named = set(_named_cards(doc))
    missing = sorted(named - set(own))
    if missing:
        errors.append(f"{len(missing)} card(s) named in findings are absent from "
                      f"`ownership`, so their cost is unstated: {', '.join(missing[:5])}")
    return errors


def in_the_99(doc, slug):
    """Cards the recon names that the deck already runs — a WARN, never an error.

    See the module docstring: recon is dated, the decklist moves under it, and this
    fires on 4 of 5 tracked recons for cards that were correctly outside the 99 when
    the artifact was written.
    """
    try:
        have = deck_names(slug)
    except FileNotFoundError:
        return []
    return sorted(n for n in _named_cards(doc) if expand_faces(n) & have)


def validate(doc, slug, pool=None):
    errors = _validate_structure(doc)
    if errors:
        return errors                      # everything below would cascade
    errors = _validate_shape(doc, slug)
    errors += _validate_sources(doc)
    errors += _validate_cards(doc, slug, load_pool() if pool is None else pool)
    errors += _validate_ownership(doc)
    return errors


def main(args):
    base = deck_dir(args.slug)
    path = base / "deck_recon.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the `deck-doctor` agent in MODE recon "
            f"(`/diagnose-deck {args.slug}` sequences it), then copy its "
            f"`.agent-out/deck-doctor-recon.json` here.")
    doc = load_json(path) or {}
    errors = validate(doc, args.slug)
    stale = in_the_99(doc, args.slug) if not errors else []
    for name in stale:
        print(f"WARN {name} is already in the 99 — recon is dated ({doc.get('as_of')}) "
              f"and the decklist moved under it")
    cards = len(_named_cards(doc))
    report_errors(
        path.name, errors,
        f"OK   {path.name} — as_of {doc.get('as_of')}, {len(doc.get('findings') or [])} "
        f"finding(s), {len(doc.get('sources') or [])} source(s), {cards} card(s) checked"
        + (f", {len(stale)} since added to the 99" if stale else "")
        + "; dated meta claims ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-recon <slug>`.")
