"""Pilot: form-check a deck's engine model — re-derive, never trust.

`engine.json` is read by three magazine columnists, The Short List and the
builder, and it is drawn. A stage that is wrong about the job therefore
propagates into several places at once, and a `verified_by` that does not hold
turns a coaching line into a false claim about the rules. So this checks the
things that are mechanically checkable and re-derives every figure rather than
believing the file.

WHAT IS DELIBERATELY NOT CHECKED, and why it will keep looking tempting:

  **That the cards in a stage share a role axis.** That check was prototyped
  fleet-wide against `goldfish_targets.json` and REJECTED — see
  `validate_goldfish_targets`'s docstring. It fired hardest on the most correct
  groups: 1-of-6 on goblin-storm's cantrip group where every member is right, and
  it flagged Howling Mine and Font of Mythos as outliers in heliod's draw engine.
  The reason generalises exactly to stages: `ROLE_PATTERNS` answers "what job does
  this card do in a 99", and a stage is a DECK-SPECIFIC functional set with no
  taxonomy axis. `fodder` is not a role. A validator that fires on correct data
  trains its reader to ignore it, and this repo has now learned that four times.

  **Whether a stage's reading is right.** Not mechanically checkable, which is
  what `engine-critic` is for.
"""

import json

from manamap.pilot.build_index import line_cards
from manamap.pilot.common import (
    deck_dir,
    load_deck_cards,
    load_json,
    presentable,
    report_errors,
)
from manamap.pilot.validate_diagnosis import REQUIRED_QUESTION_KEYS, SETTLED_BY

ARTIFACT = "engine.json"

# The closed stage set. Ordered as the engine runs, because the renderer draws it
# left to right in this order and a second ordering somewhere else would be a
# second answer to "what comes first".
STAGES = ["mana", "ignition", "fuel", "fodder", "conversion",
          "output", "protection", "wincon"]

AGREEMENTS = {"full", "partial", "cuts-across"}
EVIDENCE_KINDS = {"stack", "combo", "role", "city"}
CRITIC_STATUSES = {"supported", "unjustified", "miscounted", "mis-cited",
                   "over-claimed", "unverified-line", "contradicts-artifact"}


def _passing_stacks(slug, deck_names):
    """id -> the cards its scenario actually names. The fact tier, indexed."""
    out = {}
    directory = deck_dir(slug) / "stacks"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text())
        if not presentable(doc):
            continue
        sid = doc.get("id") or path.stem[:3]
        out[sid] = set(line_cards(doc.get("scenario") or {}, deck_names))
    return out


def validate(slug, doc=None):
    base = deck_dir(slug)
    if doc is None:
        doc = load_json(base / ARTIFACT)
    if not doc:
        raise SystemExit(
            f"{base / ARTIFACT} not found — spawn deck-engineer for {slug} first.")

    deck = load_deck_cards(slug)
    deck_names = {c["name"] for c in deck["cards"]}
    stacks = _passing_stacks(slug, deck_names)
    errors = []

    if not (doc.get("thesis") or "").strip():
        errors.append("thesis is empty — the model has to say what the engine IS "
                      "in one sentence before it lists parts")

    stages = doc.get("stages") or []
    if not stages:
        errors.append("no stages — an engine model with no stages is a card list")

    seen_stage, seen_label, placed = set(), set(), {}
    for i, stage in enumerate(stages):
        where = f"stages[{i}]"
        name = stage.get("stage")
        if name not in STAGES:
            errors.append(f"{where}: stage {name!r} is not one of {STAGES}")
        elif name in seen_stage:
            errors.append(f"{where}: stage {name!r} declared twice — one block per "
                          f"stage, or the renderer draws the same box twice")
        seen_stage.add(name)

        label = (stage.get("label") or "").strip()
        if not label:
            errors.append(f"{where}: no label")
        elif label.upper() in seen_label:
            errors.append(f"{where}: label {label!r} is already used")
        seen_label.add(label.upper())

        if not (stage.get("what_it_does") or "").strip():
            errors.append(f"{where}: what_it_does is empty")

        cards = stage.get("cards") or []
        if not cards:
            errors.append(f"{where}: names no cards")
        for card in cards:
            if card not in deck_names:
                errors.append(f"{where}: names {card!r}, which is not in the 99")
            elif card in placed:
                errors.append(f"{where}: {card!r} is already in stage "
                              f"{placed[card]!r} — a card does one job here")
            else:
                placed[card] = name

        spof = stage.get("single_point_of_failure")
        if spof and spof not in deck_names:
            errors.append(f"{where}: single_point_of_failure {spof!r} not in the 99")

        for j, ev in enumerate(stage.get("evidence") or []):
            spot = f"{where}.evidence[{j}]"
            kind = ev.get("kind")
            if kind not in EVIDENCE_KINDS:
                errors.append(f"{spot}: kind {kind!r} not in {sorted(EVIDENCE_KINDS)}")
            if kind == "stack":
                sid = str(ev.get("id", ""))
                if sid not in stacks:
                    errors.append(f"{spot}: cites stack {sid!r}, which is not a "
                                  f"checker-passed stack of this deck")
            if kind == "city" and ev.get("agreement") not in AGREEMENTS:
                errors.append(f"{spot}: agreement {ev.get('agreement')!r} not in "
                              f"{sorted(AGREEMENTS)}")

    # Completeness. The failure that matters, because it is invisible: a model
    # that forgot eleven cards reads exactly as complete as one that placed them.
    unassigned = doc.get("unassigned") or []
    for i, entry in enumerate(unassigned):
        card = entry.get("card")
        if card not in deck_names:
            errors.append(f"unassigned[{i}]: {card!r} is not in the 99")
        elif card in placed:
            errors.append(f"unassigned[{i}]: {card!r} is also in stage {placed[card]!r}")
        if not (entry.get("why") or "").strip():
            errors.append(f"unassigned[{i}]: no reason given for leaving "
                          f"{card!r} out of the engine")
    accounted = set(placed) | {e.get("card") for e in unassigned}
    missing = sorted(deck_names - accounted)
    if missing:
        errors.append(
            f"{len(missing)} card(s) in the 99 appear in no stage and in no "
            f"unassigned entry: {', '.join(missing[:8])}"
            + (" …" if len(missing) > 8 else ""))

    # Lines. `verified_by` is the claim this validator exists to police.
    for i, line in enumerate(doc.get("lines") or []):
        where = f"lines[{i}]"
        for end in ("from", "to"):
            if line.get(end) not in STAGES:
                errors.append(f"{where}: {end}={line.get(end)!r} is not a stage")
        via = line.get("via") or []
        if not via:
            errors.append(f"{where}: names no cards in `via`")
        for card in via:
            if card not in deck_names:
                errors.append(f"{where}: via names {card!r}, not in the 99")
        sid = line.get("verified_by")
        if sid is None:
            continue
        sid = str(sid)
        if sid not in stacks:
            errors.append(f"{where}: verified_by {sid!r} is not a checker-passed "
                          f"stack — a line is a CLAIM until one passes, and null "
                          f"is the honest value")
            continue
        # The check nothing else performs: the stack must actually be about
        # these cards. A passing stack id attached to an unrelated line is a
        # citation that reads as proof and is not.
        absent = [c for c in via if c not in stacks[sid]]
        if absent:
            errors.append(
                f"{where}: stack {sid} does not name {', '.join(absent)} in its "
                f"scenario, so it cannot verify this line")

    # A line's cards must live in the stages it connects, and this check exists
    # because `engine-critic` found three lines without it. The stack check above
    # proves the CARDS were on a verified board together; it says nothing about
    # whether they are the cards of the stages the arrow is drawn between. A
    # `fuel -> conversion` arrow whose via cards are both in `conversion` is a
    # picture of a connection that the model does not actually claim, and it
    # renders as solid green — proof, for a relation nobody asserted.
    for i, line in enumerate(doc.get("lines") or []):
        via = set(line.get("via") or [])
        if not via:
            continue
        for end in ("from", "to"):
            stage_name = line.get(end)
            if stage_name not in STAGES:
                continue
            members = {c for c, st in placed.items() if st == stage_name}
            if members and not (via & members):
                errors.append(
                    f"lines[{i}]: {end}={stage_name!r} contains none of "
                    f"{', '.join(sorted(via))} — an arrow between two stages has "
                    f"to be about a card in each of them")

    for i, q in enumerate(doc.get("open_questions") or []):
        missing_keys = REQUIRED_QUESTION_KEYS - set(q)
        if missing_keys:
            errors.append(f"open_questions[{i}]: missing {sorted(missing_keys)}")
        if q.get("settled_by") not in SETTLED_BY:
            errors.append(f"open_questions[{i}]: settled_by "
                          f"{q.get('settled_by')!r} not in {sorted(SETTLED_BY)}")

    for i, edit in enumerate(doc.get("proposed_goldfish_edits") or []):
        for key in ("target", "change", "why"):
            if not (edit.get(key) or "").strip():
                errors.append(f"proposed_goldfish_edits[{i}]: {key} is empty")

    critic = doc.get("critic")
    if critic:
        if critic.get("verdict") not in {"pass", "fail"}:
            errors.append(f"critic.verdict {critic.get('verdict')!r} must be "
                          f"pass or fail")
        for i, f in enumerate(critic.get("findings") or []):
            if f.get("status") not in CRITIC_STATUSES:
                errors.append(f"critic.findings[{i}]: status {f.get('status')!r} "
                              f"not in {sorted(CRITIC_STATUSES)}")
    return errors, doc, placed


def main(args):
    errors, doc, placed = validate(args.slug)
    stages = doc.get("stages") or []
    verified = sum(1 for l in doc.get("lines") or [] if l.get("verified_by"))
    total = len(doc.get("lines") or [])
    report_errors(
        f"engine model for {args.slug}", errors,
        f"OK   engine model for {args.slug} — {len(stages)} stage(s), "
        f"{len(placed)} card(s) placed, {len(doc.get('unassigned') or [])} "
        f"unassigned, {verified}/{total} line(s) verified by a passing stack")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-engine <slug>`.")
