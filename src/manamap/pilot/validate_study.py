"""The gate on `studies/NNN-*.json`: a decision names what decided it, the goldfish
is never that alone, and every citation resolves.

Modelled on `validate_sim_findings`: fact ids are the study's own; a figure in
Jarvis's `reading` must appear in a fact it cites. An agent's contribution is a
QUOTED SOURCE, kept verbatim like a log entry, so its numbers are not held to the
facts — its citations are: a strategist with no `strategy:` section and a
rules-question with no CR number have not done the job their charters name.

THE RULE THIS FILE EXISTS FOR. `decided_by` is required on every decided study,
and `goldfish` may appear in it only beside `pilot`, and only when every card the
study names is one the model reads. A card the model cannot see cannot have been
decided by the model; the pilot said so fourteen times before this gate existed.
"""

import re

from manamap.pilot import study as st
from manamap.pilot.common import RULE_ID_RE, STRATEGY_ID_RE, deck_dir, load_json, report_errors
from manamap.pilot.validate_sim_findings import numbers_in

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)%?(?![\w.])")
_ID = re.compile(r"^\d{3}$")
TIERS = ("verified", "data", "coach")


def _base_names(slug, branch):
    """Every name the swap's OUT card could legitimately be: the deck's list, the
    branch's list, and the branch's own staged OUTs (a backfilled study names a card
    the branch has already cut)."""
    names = set()
    for b in (None, branch) if branch else (None,):
        try:
            doc = load_json(deck_dir(slug, b) / "cards.json") or {}
        except FileNotFoundError:
            continue
        for c in doc.get("cards") or []:
            names.add((c.get("name") or "").lower())
            for face in (c.get("name") or "").split(" // "):
                names.add(face.strip().lower())
    if branch:
        try:
            bdoc = load_json(deck_dir(slug, branch) / "branch.json") or {}
            names |= {(s.get("out") or "").lower() for s in bdoc.get("staged") or []}
        except FileNotFoundError:
            pass
    return names


def validate(doc, slug=None, strict=False):
    errors, notes = [], []
    if not isinstance(doc, dict) or "question" not in doc or "facts" not in doc:
        return ["not a study document (needs `question` and `facts`)"], notes
    sid = str(doc.get("id") or "")
    if not _ID.match(sid):
        errors.append(f"id {sid!r} is not a zero-padded three-digit id")
    if slug and doc.get("slug") != slug:
        errors.append(f"slug {doc.get('slug')!r} is not {slug!r}")
    if not str(doc.get("question") or "").strip():
        errors.append("the question is empty")
    if not doc.get("decklist_sha256"):
        errors.append("no decklist_sha256 — the study does not say which list it studied "
                      "(the branch is gone, or the facts were never computed)")

    facts = {}
    for i, f in enumerate(doc.get("facts") or []):
        if not isinstance(f, dict):
            errors.append(f"facts[{i}] is not an object")
            continue
        for k in ("id", "kind", "text", "tier", "source"):
            if not f.get(k):
                errors.append(f"facts[{i}]: no {k!r}")
        if f.get("id") and not st.FACT_ID_RE.fullmatch(str(f["id"])):
            errors.append(f"facts[{i}].id {f['id']!r} is not of the form S-<sha8>-NN")
        if f.get("tier") not in TIERS:
            errors.append(f"facts[{i}].tier {f.get('tier')!r} is not one of {TIERS}")
        if f.get("id"):
            if f["id"] in facts:
                errors.append(f"fact id {f['id']} appears twice")
            facts[f["id"]] = f

    # the call
    call = doc.get("call") or {}
    decision = call.get("decision", "open")
    if decision not in st.DECISIONS:
        errors.append(f"call.decision {decision!r} is not one of {st.DECISIONS}")
    by = list(call.get("decided_by") or [])
    bad = [b for b in by if b not in st.DECIDED_BY]
    if bad:
        errors.append(f"call.decided_by {bad} not in {st.DECIDED_BY}")
    if decision != "open":
        if not by:
            errors.append("a decided study must say what decided it (call.decided_by is empty)")
        if not str(call.get("pilot_words") or "").strip():
            errors.append("a decided study carries the pilot's words verbatim (call.pilot_words is empty)")
    if "goldfish" in by:
        if by == ["goldfish"]:
            errors.append("the goldfish may never be the sole decider — it has no opponents, no blockers "
                          "and no removal; name the pilot and the source that read the cards")
        elif "pilot" not in by:
            errors.append("the goldfish is admitted to decided_by only beside `pilot`")
        unread = [f.get("card") for f in facts.values()
                  if f.get("kind") == "coverage" and f.get("state") not in ("seen", "partial")]
        if unread:
            errors.append(f"decided_by names the goldfish while it cannot read {sorted(set(unread))} — "
                          f"a card the model does not see was not decided by the model")
        named = set(doc.get("cards") or [])
        covered = {f.get("card") for f in facts.values() if f.get("kind") == "coverage"}
        if named - covered:
            errors.append(f"decided_by names the goldfish with no coverage fact for {sorted(named - covered)}")
    # Jarvis's reading is held to the facts it cites
    reading = str(call.get("reading") or "")
    if reading:
        cited = set(st.FACT_ID_RE.findall(reading))
        missing = sorted(c for c in cited if c not in facts)
        if missing:
            errors.append(f"call.reading cites {missing} — not a fact of this study")
        allowed = set()
        for c in cited:
            if c in facts:
                allowed |= numbers_in({k: v for k, v in facts[c].items() if k not in ("id", "kind", "source", "stamp")})
        # the ids themselves carry digits (`S-1a9fa010-08`); scan the prose without them
        for m in _NUM.finditer(st.FACT_ID_RE.sub(" ", reading)):
            v = round(float(m.group(1)), 4)
            if v not in allowed and v not in (0.0, 1.0):
                errors.append(f"call.reading: the number {m.group(0)} is not in any cited fact — "
                              f"a figure in the reading is the record's, cited by id")

    # the contributions: quoted sources, with their citations checked
    for i, c in enumerate(doc.get("contributions") or []):
        where = f"contributions[{i}]"
        if c.get("agent") not in st.AGENTS:
            errors.append(f"{where}.agent {c.get('agent')!r} is not one of {st.AGENTS}")
        if not str(c.get("text") or "").strip():
            errors.append(f"{where}: empty text")
        cites = c.get("cites") or {}
        for fid in cites.get("facts") or []:
            if fid not in facts:
                errors.append(f"{where} cites fact {fid} — not a fact of this study")
        for s_ in cites.get("strategy") or []:
            if not STRATEGY_ID_RE.match(s_):
                errors.append(f"{where} cites {s_!r} — not a strategy:<id>")
        for r in cites.get("rules") or []:
            if not RULE_ID_RE.match(r):
                errors.append(f"{where} cites {r!r} — not a CR rule number")
        if c.get("agent") == "strategist" and not (cites.get("strategy") or []):
            errors.append(f"{where}: a strategist argument with no strategy:<id> citation — "
                          f"the charter grounds every framework claim in the companion")
        if c.get("agent") == "rules-question" and not (cites.get("rules") or []):
            errors.append(f"{where}: a rules answer with no CR number quoted")
        if c.get("agent") == "data-analyst" and not (c.get("commands") or []):
            notes.append(f"{where}: an analyst answer naming no commands — its figures cannot be re-run")

    # the swap names a card the list (or the branch's staged OUTs) actually has
    swap = doc.get("swap") or {}
    if swap.get("out") and slug:
        try:
            base = _base_names(slug, doc.get("branch"))
        except FileNotFoundError:
            base = set()
        if base and swap["out"].lower() not in base:
            errors.append(f"swap.out {swap['out']!r} is in neither the deck, the branch nor its staged cuts")

    if strict:
        if decision == "open":
            errors.append("install needs a decided study — call it first")
        if (doc.get("goldfish_coverage") or {}).get("named") != len(doc.get("cards") or []):
            errors.append("goldfish_coverage does not cover every named card — recompute the facts")
    return errors, notes


def main(args):
    slug = args.slug
    ref = getattr(args, "ref", None) or getattr(args, "study", None)
    docs = [st.read(slug, ref)] if ref else st.list_studies(slug)
    if not docs:
        print(f"{slug}: no studies")
        return
    errors, notes = [], []
    for doc in docs:
        e, n = validate(doc, slug)
        errors += [f"study {doc.get('id')}: {x}" for x in e]
        notes += [f"study {doc.get('id')}: {x}" for x in n]
    for n_ in notes:
        print(f"  ! {n_}")
    report_errors(f"{slug}: studies", errors)
    decided = sum(1 for d in docs if (d.get("call") or {}).get("decision", "open") != "open")
    print(f"{slug}: {len(docs)} study(ies), {decided} decided, OK")
