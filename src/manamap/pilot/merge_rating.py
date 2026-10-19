"""Pilot: merge the rating agents' hand-offs into `rating.json`.

THE SKELETON IS RECOMPUTED, ALWAYS — it is a pure function of the tracked
artifacts and the scorecard — and the agent's hand-off contributes only what
it owns: `deck-rater` writes `judgment` (sub_score, argument, citations) and
`open_questions`; `rating-skeptic` writes `verifier` (verdict, findings). The
validator runs BEFORE the write and refuses on any error, so a judgment that
invents a figure, cites a log entry that does not exist, or scores outside
[0, 100] lands nothing (the `merge_sim_findings` pattern).

`iterations` counts rater passes; the skeptic's merge does not bump it.
"""

from manamap import config
from manamap.pilot import rating, validate_rating
from manamap.pilot.common import load_json
from manamap.pilot.validate_stack import load_strategy_sections

RATER_FILE = "deck-rater.json"
SKEPTIC_FILE = "rating-skeptic.json"


def merge(slug, verifier=False):
    base = config.DECKS_DIR / slug
    name = SKEPTIC_FILE if verifier else RATER_FILE
    handoff = load_json(base / ".agent-out" / name)
    if handoff is None:
        raise SystemExit(f"no {base / '.agent-out' / name} — spawn the "
                         f"`{'rating-skeptic' if verifier else 'deck-rater'}` agent first; "
                         f"it writes there and returns the path")
    previous = load_json(base / rating.RATING_FILE)
    doc = rating.skeleton(slug)
    if verifier:
        if not previous or not previous.get("judgment"):
            raise SystemExit("no judgment to verify — merge the rater's hand-off first")
        doc = rating.carry_forward(doc, previous)
        v = handoff.get("verifier") or handoff
        doc["verifier"] = {"verdict": v.get("verdict"),
                           "findings": [{"where": f.get("where"), "status": f.get("status"), "note": f.get("note")}
                                        for f in v.get("findings") or []],
                           "iterations": doc.get("iterations") or 0}
    else:
        j = handoff.get("judgment") or {}
        judgment = {"sub_score": j.get("sub_score"), "argument": j.get("argument"),
                    "citations": [{"kind": c.get("kind"), "ref": c.get("ref"), "quote": c.get("quote")}
                                  for c in j.get("citations") or []],
                    "against": {"decklist_sha256": doc["decklist_sha256"], "spec_sha": doc["model_version"]}}
        doc["withheld"] = [w for w in doc["withheld"] if w["component"] != "judgment"]
        doc["open_questions"] = [{"question": q.get("question"), "settled_by": q.get("settled_by")}
                                 for q in handoff.get("open_questions") or []]
        # A NEW LOOP STARTS AT ONE. `iterations` counts rater passes within one
        # loop; a judgment merged after a loop already reached the cap (a forced
        # re-rate, `cache-clear` then a fresh rater) is pass one of a new loop,
        # not pass four of the old one.
        prev_iter = (previous or {}).get("iterations") or 0
        doc["iterations"] = prev_iter + 1 if prev_iter < config.RATING_MAX_ITERATIONS else 1
        doc["verifier"] = None      # a new judgment is unverified until the skeptic looks again
        rating._finish(doc, judgment)
    doc = rating.ranked(doc)
    try:
        sections = load_strategy_sections()
    except Exception:  # noqa: BLE001
        sections = None
    errors, _notes = validate_rating.validate(slug, doc, fresh=None, strategy_sections=sections)
    if errors:
        raise SystemExit("refusing to merge — the hand-off does not hold to the record:\n  - "
                         + "\n  - ".join(errors))
    path = rating.write(slug, doc)
    return doc, path


def main(args):
    doc, path = merge(args.slug, verifier=bool(getattr(args, "verifier", False)))
    what = "verifier" if getattr(args, "verifier", False) else "judgment"
    print(f"merged the {what} into {path}")
    rating._print_doc(doc)
    print(f"  next: `manamap pilot validate-rating {args.slug}`"
          + ("" if getattr(args, "verifier", False) else f", then spawn `rating-skeptic` and "
             f"`manamap pilot merge-rating {args.slug} --verifier`"))
