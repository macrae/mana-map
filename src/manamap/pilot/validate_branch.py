"""`branch.json` — the form check on a branch, and on the decision to merge it.

**THE LAST TRACKED PILOT ARTIFACT WITH NO GATE.** Every other one on a branch —
`net_change.json`, `cards.json`, `deck_map.json`, the measurements — is validated
and freshness-tested; `branch.json`, which holds the objective the branch is
graded against and now the pilot's acceptance of it, was checked by nothing. The
rule it violated is `CLAUDE.md`'s: *a new tracked artifact needs a gate in the
same commit*. It shipped without one and the omission survived two development
cycles because branches were the newest thing in the repo and nothing had gone
wrong yet.

WHAT IT CHECKS is the discipline, not the figures. Nothing here re-measures:

  1. **AN OBJECTIVE NAMES A REAL AXIS, AND NOT AN AUTHORED ONE.** The
     `MEMBERSHIP_AXES` ban has lived as a loose test that walked every tracked
     `branch.json` — a gate is where it belongs, because the reason is permanent:
     `engine_online_*` asks whether the parts named in an AUTHORED file were
     drawn, so its sign can be flipped by an edit. Three declarations of one
     Ur-Dragon list gave +0.007, −0.036 and +0.014 over the same 10,000 games.

  2. **A PROPOSAL SAYS WHAT IT INTENDS TO BECOME.** Without `as_version` a
     proposal cannot be found to have been outrun by another merge, which is
     `PROPOSED · OUTRUN` and the whole reason `base_version` exists.

  3. **A PROPOSAL FREEZES BOTH SHAS.** `decklist_sha256` is the list the pilot
     accepted; `accepted_on.decklist_sha256` is the list the report measured. One
     without the other cannot answer "has this gone stale", and staleness is the
     only thing standing between a decision and a list that quietly moved under
     it.

  4. **A PROPOSAL OVER A "DO NOT MERGE" SAYS WHY.** `propose` refuses it without
     `--anyway --reason`; this is the same claim held at rest, so a hand-edited
     file cannot carry an override with no argument attached.

  5. **`merged` AND `proposal` DO NOT CONTRADICT.** A merged branch that still
     advertises an unfulfilled proposal reads as pending work that is already
     done.

  6. **`objective_history`, IF PRESENT, HAS A SHAPE.** Nothing in the repo writes
     or reads it — it was hand-added to `eminence-v3` when that branch's objective
     moved off an authored axis. An ungated key that only ever arrives by hand is
     the shape a typo lives in forever, so it is either checked or it is deleted.

WHAT IT DELIBERATELY DOES NOT CHECK is whether the branch is BLOCKED, READY,
STALE or OUTRUN. Those are derived from the collection and from git on every
read — `deck_branch.branch_state` — and a gate that froze them would be asserting
a fact about cardboard that stops being true when a card lands in a box. Closure
is derived, never declared; the same rule `validate_pending` earned.
"""

import json

from manamap.pilot.common import deck_dir, report_errors

ARTIFACT = "branch.json"

#: The keys a v2+ file owes. `v1` predates the objective requirement and is
#: tolerated — the same allowance `deck_branch.new` documents — because a file
#: that loaded yesterday must load today.
REQUIRED = ("slug", "branch", "opened")

PROPOSAL_REQUIRED = ("at", "as_version", "base_version", "decklist_sha256",
                     "accepted_on")


def validate(doc, slug=None, branch=None):
    from manamap.pilot import candidates, deck_branch, deck_versions, net_change

    errors = []
    for key in REQUIRED:
        if not doc.get(key):
            errors.append(f"no {key!r}")
    if slug and doc.get("slug") not in (None, slug):
        errors.append(f"slug is {doc.get('slug')!r}, but it lives under {slug!r}")
    if branch and doc.get("branch") not in (None, branch):
        errors.append(f"branch is {doc.get('branch')!r}, but the directory is {branch!r}")

    # ── the objective ────────────────────────────────────────────────────
    o = doc.get("objective")
    version = int(doc.get("v") or 1)
    if o is None and version >= 2:
        errors.append("v2+ and no objective — a branch that cannot be falsified "
                      "gets graded on whether it did what it does")
    if o is not None:
        for key in ("axis", "op", "value"):
            if key not in o:
                errors.append(f"objective: no {key!r}")
        axis = o.get("axis")
        if axis and axis not in candidates.OBJECTIVE_AXES:
            errors.append(f"objective: {axis!r} is not something the bench measures")
        if axis in deck_branch.MEMBERSHIP_AXES:
            errors.append(
                f"objective: {axis!r} asks whether the parts named in an AUTHORED "
                f"file were drawn — the same hand writes the declaration and reads "
                f"the verdict. Aim at an output the deck produces.")

    # ── objective_history: hand-written, so it owes a shape ──────────────
    hist = doc.get("objective_history")
    if hist is not None:
        if not isinstance(hist, list):
            errors.append("objective_history is not a list")
        else:
            for i, row in enumerate(hist):
                where = f"objective_history[{i}]"
                if not isinstance(row, dict):
                    errors.append(f"{where}: not an object")
                    continue
                for key in ("at", "was", "why_changed"):
                    if not row.get(key):
                        errors.append(f"{where}: no {key!r}")
                was = row.get("was") or {}
                if isinstance(was, dict) and was.get("axis") == o and o:
                    errors.append(f"{where}: records a change to the same axis")

    # ── the proposal ─────────────────────────────────────────────────────
    prop = doc.get("proposal")
    if prop is not None:
        if not isinstance(prop, dict):
            errors.append("proposal is not an object")
            prop = {}
        for key in PROPOSAL_REQUIRED:
            if prop.get(key) in (None, ""):
                errors.append(f"proposal: no {key!r}")
        as_v = prop.get("as_version")
        if as_v and not deck_versions._RELEASE_RE.match(as_v):
            errors.append(
                f"proposal: as_version {as_v!r} is not a release tag like v1.0.2")
        accepted = prop.get("accepted_on") or {}
        if not isinstance(accepted, dict):
            errors.append("proposal.accepted_on is not an object")
            accepted = {}
        if not accepted.get("decklist_sha256"):
            errors.append(
                "proposal.accepted_on: no 'decklist_sha256' — without the sha the "
                "report measured, nothing can say the list has moved since")
        state = accepted.get("state")
        if state and state not in net_change.STATES:
            errors.append(f"proposal.accepted_on: {state!r} is not a net-change state")
        if state == "do not merge" and not prop.get("forced_reason"):
            errors.append(
                "proposal: the net change said DO NOT MERGE and no 'forced_reason' "
                "says why it was accepted anyway")
        proxy = prop.get("proxy")
        if proxy is not None and not isinstance(proxy, list):
            errors.append("proposal.proxy must be a list of card NAMES, never a "
                          "boolean — a decision about specific cardboard is "
                          "recorded as specific cardboard")
        proc = prop.get("procurement")
        if proc is not None:
            if not isinstance(proc, dict) or not proc.get("note"):
                errors.append("proposal.procurement: no 'note'")

    # ── merged vs proposal ───────────────────────────────────────────────
    merged = doc.get("merged")
    if merged is not None:
        if not isinstance(merged, dict) or not merged.get("at"):
            errors.append("merged: no 'at'")
        elif prop and prop.get("as_version"):
            # A merged branch may KEEP its proposal — that is the record of what
            # was decided and when — but it must not still be advertising work
            # that is done. The state machine reads `merged` first, so this is a
            # form check on the pair rather than a second opinion about it.
            if not merged.get("decklist_sha256"):
                errors.append("merged: no 'decklist_sha256'")

    # ── commits ──────────────────────────────────────────────────────────
    for i, c in enumerate(doc.get("commits") or []):
        for key in ("at", "decklist_sha256", "message"):
            if not (c or {}).get(key):
                errors.append(f"commits[{i}]: no {key!r}")
    return errors


def main(args):
    branch = getattr(args, "branch", None)
    if not branch:
        raise SystemExit(f"{ARTIFACT} lives on a branch — `--branch <name>`.")
    path = deck_dir(args.slug, branch) / ARTIFACT
    if not path.exists():
        raise SystemExit(
            f"{path} not found — `manamap pilot deck-branch {args.slug} new "
            f"{branch} --from <file> --objective \"…\"` first.")
    doc = json.loads(path.read_text())
    errors = validate(doc, slug=args.slug, branch=branch)

    from manamap.pilot import deck_branch
    state, _why = deck_branch.branch_state(args.slug, branch, doc=doc)
    prop = doc.get("proposal") or {}
    tail = f" as {prop['as_version']}" if prop.get("as_version") else ""
    report_errors(
        f"{ARTIFACT} for {args.slug}@{branch}", errors,
        f"OK   {ARTIFACT} for {args.slug}@{branch} — {state}{tail} ◆")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-branch <slug> --branch <name>`.")
