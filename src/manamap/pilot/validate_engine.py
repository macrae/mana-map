"""Pilot: hold a deck's engine record to a fresh computation — re-derive, never trust.

    manamap pilot validate-engine <slug>

`engine.json` v2 (2026-11-22; `engine_graph`, `docs/research/engine-model.md`) has two
halves. Everything but `reading` is COMPUTED from the list and the cards' Forge
scripts, so the gate is the same one `combos.json` has: rebuild it and compare, key by
key — a committed record that differs from a fresh run is stale or hand-edited, and
either is refused. `reading` is AUTHORED — the engineer's and the pilot's words — and
is checked for form only: it must say which list it was written for, say `stale` when
that is not the current one, name only lines and cards the record has, and carry a
critic verdict that is `pass` or `fail` when it carries one. A v1 file (no `_meta.v`)
is refused with the command that migrates it.

WHAT IS DELIBERATELY NOT CHECKED, and why it will keep looking tempting: whether a
stack that NAMES a line's cards SUPPORTS the line. Two real radagast lines once passed
every mechanical check while citing a stack that showed the opposite — one claimed
Castle Garenbrig paid for Craterhoof, citing a stack that leaves Garenbrig untapped.
A checker-passed stack is evidence that a BOARD resolved a certain way; reading it as
"A feeds B" is an inference, and inference is the critic's job. The computed `feeds`
edge is a different claim — the filter accepts the subject — and is checked by
recomputation, not by the stack. Do not close the gap with string matching on
resolution prose: the same wrong line survives a rephrase. (The v1 gate's two rejected
prototypes — a stage/role-axis check that fired on correct groups, and an acting-stage
check that fired 46 times across 57 verified lines — are recorded in
`docs/gotchas-bench.md`; the computed stages have no authored placement to check.)
"""
from manamap.config import ENGINE_STAGE_RULES
from manamap.pilot.common import deck_dir, load_json, report_errors

ARTIFACT = "engine.json"
#: The stage vocabulary in the PAGE's order (`poh_spec.STAGE_ORDER` mirrors it);
#: `config.ENGINE_STAGE_RULES` is the same eight in rule-priority order.
STAGES = ("mana", "ignition", "fuel", "fodder", "conversion", "output", "protection", "wincon")
assert set(STAGES) == set(ENGINE_STAGE_RULES)
AUTHORED = "reading"
CRITIC_VERDICTS = {"pass", "fail"}
#: The critic's finding statuses, unchanged from the v1 charter.
CRITIC_STATUSES = {"supported", "unjustified", "miscounted", "mis-cited",
                   "over-claimed", "unverified-line", "contradicts-artifact"}
REQUIRED_READING_KEYS = ("written_for", "stale", "names_departed", "thesis", "lines",
                         "stage_overrides", "open_questions", "critic")


def _differences(fresh, stored, prefix=""):
    """Every key where the committed computed part differs from a fresh build —
    named, so the error says WHAT moved rather than that something did."""
    out = []
    keys = sorted(set(fresh) | set(stored))
    for k in keys:
        a, b = fresh.get(k), stored.get(k)
        where = f"{prefix}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out.extend(_differences(a, b, where + "."))
        elif a != b:
            if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
                out.append(f"{where}: {len(b)} entr{'y' if len(b) == 1 else 'ies'} on disk, {len(a)} fresh")
            else:
                out.append(f"{where} differs")
    return out


def validate(slug, doc=None):
    """Returns `(errors, doc, placed)` — the same triple the v1 gate returned, so
    `deck_status.VALIDATED` and the callers that unpack it keep working."""
    from manamap.pilot import engine_graph
    base = deck_dir(slug)
    if doc is None:
        doc = load_json(base / ARTIFACT)
    if not doc:
        raise SystemExit(f"{base / ARTIFACT} not found — `manamap pilot engine-graph {slug} --write` builds it.")
    errors = []
    meta = doc.get("_meta") or {}
    if meta.get("v") != 2:
        errors.append(f"engine.json is a v1 model (agent-placed stages) — "
                      f"`manamap pilot engine-graph {slug} --write` recomputes it and keeps the "
                      f"thesis, the line notes and the critic as the first reading")
        return errors, doc, {}

    fresh = engine_graph.build(slug)
    stored = engine_graph.computed_part(doc)
    fresh_c = engine_graph.computed_part(fresh)
    if stored.get("_meta", {}).get("forge_digest") != fresh_c["_meta"]["forge_digest"] \
            and fresh_c["_meta"]["forge_digest"] is None:
        # No Forge here: the stored record was read from scripts this machine does
        # not have. Not stale — unverifiable, and said so.
        errors.append("cannot recompute: Forge is not installed here, and the record was read from "
                      "its scripts (`_meta.forge_digest`); nothing else is checked")
        return errors, doc, doc.get("placed") or {}
    diffs = _differences(fresh_c, stored)
    if diffs:
        head = "; ".join(diffs[:6]) + (" …" if len(diffs) > 6 else "")
        why = ("the list moved" if stored.get("decklist_sha256") != fresh_c["decklist_sha256"]
               else "the parser or the scripts moved" if stored.get("_meta", {}).get("ability_graph_version")
               != fresh_c["_meta"]["ability_graph_version"] or stored.get("_meta", {}).get("forge_digest")
               != fresh_c["_meta"]["forge_digest"]
               else "a hand edit, or a code change without a rebuild")
        errors.append(f"computed record differs from a fresh build ({why}): {head} — "
                      f"`manamap pilot engine-graph {slug} --write`")

    r = doc.get(AUTHORED)
    if r is None:
        errors.append("no `reading` block — `engine-graph --write` writes an empty one; the engineer fills it")
        return errors, doc, doc.get("placed") or {}
    if not isinstance(r, dict):
        errors.append("`reading` must be an object")
        return errors, doc, doc.get("placed") or {}
    for k in REQUIRED_READING_KEYS:
        if k not in r:
            errors.append(f"reading.{k} is missing")
    now = doc.get("decklist_sha256")
    if r.get("written_for") and now and r["written_for"] != now and r.get("stale") is not True:
        errors.append(f"reading.written_for {str(r['written_for'])[:12]} is not the current list "
                      f"{str(now)[:12]} but reading.stale is not true — a reading of another list must say so")
    if r.get("written_for") == now and r.get("stale") is True:
        errors.append("reading.stale is true for the current list")
    line_ids = {l.get("id") for l in doc.get("lines") or []}
    comp_ids = {c.get("id") for c in doc.get("components") or []}
    for i, l in enumerate(r.get("lines") or []):
        ref = l.get("line")
        if ref not in line_ids and ref not in comp_ids:
            errors.append(f"reading.lines[{i}] names {ref!r}, which is neither a computed line nor a component")
    names = {n["id"] for n in (doc.get("graph") or {}).get("nodes") or [] if n.get("zone") in ("command", "library")}
    for i, o in enumerate(r.get("stage_overrides") or []):
        if o.get("card") not in names:
            errors.append(f"reading.stage_overrides[{i}] names {o.get('card')!r}, not in the 99")
        if not (o.get("why") or "").strip():
            errors.append(f"reading.stage_overrides[{i}]: no why")
        if o.get("stage") not in ENGINE_STAGE_RULES:
            errors.append(f"reading.stage_overrides[{i}]: stage {o.get('stage')!r} is not a stage")
    critic = r.get("critic")
    if critic:
        if critic.get("verdict") not in CRITIC_VERDICTS:
            errors.append(f"reading.critic.verdict {critic.get('verdict')!r} must be pass or fail")
        for i, f in enumerate(critic.get("findings") or []):
            if f.get("status") not in CRITIC_STATUSES:
                errors.append(f"reading.critic.findings[{i}]: status {f.get('status')!r} not in {sorted(CRITIC_STATUSES)}")
    return errors, doc, doc.get("placed") or {}


def main(args):
    errors, doc, _ = validate(args.slug)
    report_errors(f"engine.json for {args.slug}", errors)
    r = doc.get(AUTHORED) or {}
    n_lines = len(doc.get("lines") or [])
    n_ver = sum(1 for l in doc.get("lines") or [] if l.get("verified_by"))
    print(f"OK engine.json for {args.slug}: v2, {len(doc.get('components') or [])} components, "
          f"{n_ver}/{n_lines} lines stack-stamped, {len(doc.get('spof') or [])} single points of failure; "
          f"reading {'STALE' if r.get('stale') else 'current' if r.get('thesis') else 'empty'}"
          + (f", critic {r['critic'].get('verdict')}" if r.get("critic") else ""))
    return 0
