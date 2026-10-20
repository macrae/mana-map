"""Pilot: the gate on `rating.json` — ★ coaching held to its ◆ inputs.

RECOMPUTE, NEVER TRUST. The skeleton (`rating.skeleton`) is a pure function of
the tracked artifacts and the scorecard's constants, so every code component,
the withheld list, the code score, the bracket and the limits must equal a
fresh computation byte for byte. The agent owns `judgment`, `open_questions`
and (through the skeptic) `verifier`; the gate holds those to the record:

- `tier` is "★" and the label is the one sentence `RATING_LABEL` renders;
- the judgment's sub-score is a whole number in [0, 100], its weight is the
  scorecard's, and the composite, interval, grade and rank recompute from it;
- every citation resolves — a `log:<id>` exists and its quote is in the
  entry's text, a `strategy:<id>` quote is verbatim (the stack validator's
  contract), a `stack:<NNN>` exists and passed its checker, an `engine:<stage>`
  and a `context:<section>` exist and carry the quote, a `skeleton:<component>`
  is present — and at least two kinds are cited;
- every number in the argument appears in cited material or in the skeleton
  (the `sim_findings` rule), and the argument is within the word budget;
- the verifier's statuses are from the closed set, every finding says `where`,
  and `pass` means every finding is `supported`.

STALENESS IS REPORTED, NEVER FAILED: when the artifact describes another list,
another fleet build or another spec, the recompute cannot be compared, so the
gate runs the form checks only and says STALE. Absent file: "absent means
absent", exit 0.
"""

import hashlib
import json
import re

from manamap import config
from manamap.pilot import deck_context, rating
from manamap.pilot.agent_cache import checker_passed
from manamap.pilot.common import deck_dir, load_json, measured_sha, report_errors
from manamap.pilot.deck_notes import read_log
from manamap.pilot.validate_sim_findings import numbers_in
from manamap.pilot.validate_stack import _normalize_ws, load_strategy_sections, validate_citations

REQUIRED = ("slug", "tier", "label", "bracket", "score", "score_point", "code_score", "score_ci",
            "resolution", "grade", "grade_range", "grade_display", "grade_probabilities", "uncertainty",
            "sensitivity", "concordance", "rank", "components", "judgment", "withheld", "limits", "context",
            "decklist_sha256", "fleet_sha", "model_version", "spec_version", "open_questions",
            "iterations", "verifier")
CITATION_KINDS = ("log", "strategy", "stack", "engine", "context", "skeleton")
VERIFIER_STATUSES = ("supported", "unjustified", "miscounted", "mis-cited", "over-claimed",
                     "too-generous", "too-harsh", "contradicts-artifact")
SETTLED_BY = ("resolve-stack", "research-strategy", "goldfish", "diagnose", "rate")
CODE_KEYS = ("components", "withheld", "code_score", "score_point", "bracket", "limits", "context",
             "uncertainty", "sensitivity")
#: The keys the GROUP's simulation fills: they move when a neighbour's judgment lands.
GROUP_KEYS = ("score", "score_ci", "grade", "grade_range", "grade_display", "grade_probabilities",
              "concordance")
#: How an argument may name an outlier instrument (lower-cased substrings).
INSTRUMENT_WORDS = {"speed": ("speed",), "mana": ("mana",), "resilience": ("resilience",),
                    "construction": ("construction",), "dti": ("dti", "deckcheck", "threat index"),
                    "table_record": ("table record", "table_record", "record"),
                    "pilot_feel": ("pilot feel", "pilot_feel", "survey", "feel")}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_code(doc):
    """The code-owned view, with the withheld judgment entry removed so a
    skeleton and a judged document compare on the same keys."""
    out = {k: json.loads(json.dumps(doc.get(k))) for k in CODE_KEYS}
    out["withheld"] = [w for w in out["withheld"] or [] if w.get("component") != "judgment"]
    # `sub_ci` is the group simulation's and `weight_effective` depends on the
    # judgment's presence; both are compared under GROUP_KEYS and the weights check.
    out["components"] = [{k: v for k, v in c.items() if k not in ("weight_effective", "sub_ci")}
                         for c in out["components"] or []]
    return out


def _group_view(doc):
    out = {k: json.loads(json.dumps(doc.get(k))) for k in GROUP_KEYS}
    out["rank"] = {k: v for k, v in (doc.get("rank") or {}).items() if k != "basis"}
    out["sub_ci"] = {c["name"]: c.get("sub_ci") for c in doc.get("components") or []}
    return out


def _check_citation(c, where, slug, errors, strategy_sections, skeleton_names):
    kind, ref, quote = c.get("kind"), c.get("ref"), c.get("quote")
    if kind not in CITATION_KINDS:
        errors.append(f"{where}: citation kind {kind!r} is not one of {', '.join(CITATION_KINDS)}")
        return
    if not ref or not isinstance(quote, str) or not quote.strip():
        errors.append(f"{where}: a citation needs ref and quote")
        return
    base = deck_dir(slug)
    if kind == "strategy":
        validate_citations([{"rule": ref, "quote": quote}], {}, where, errors, strategy_sections)
    elif kind == "log":
        if ref == "log.jsonl":
            return
        entries = {e["id"]: e for e in read_log(slug)}
        eid = ref.split(":", 1)[-1]
        if eid not in entries:
            errors.append(f"{where}: cites log entry {eid!r}, which is not in the log")
        elif _normalize_ws(quote) not in _normalize_ws(entries[eid].get("text") or ""):
            errors.append(f"{where}: quote is not in log entry {eid}'s text: {quote[:60]!r}")
    elif kind == "stack":
        sid = ref.split(":", 1)[-1]
        hits = sorted((base / "stacks").glob(f"{sid}-*.json")) if (base / "stacks").is_dir() else []
        if not hits:
            errors.append(f"{where}: cites stack {sid}, which does not exist")
            return
        sdoc = load_json(hits[0]) or {}
        if not checker_passed(sdoc):
            errors.append(f"{where}: cites stack {sid}, which has not passed its checker")
        # ensure_ascii=False on BOTH halves: the title used to be dumped with the
        # default escaping, so a title carrying an em dash ("… or not — verified
        # before it is counted", stack 012) became "—" in the haystack and no
        # verbatim quote of it could ever match (found 2026-10-20).
        hay = _normalize_ws(json.dumps(sdoc.get("title", ""), ensure_ascii=False)
                            + json.dumps(sdoc.get("resolution", {}), ensure_ascii=False))
        if _normalize_ws(quote) not in hay:
            errors.append(f"{where}: quote is not in stack {sid}'s title or resolution: {quote[:60]!r}")
    elif kind == "engine":
        eng = load_json(base / "engine.json") or {}
        stage = ref.split(":", 1)[-1]
        rows = {s.get("stage"): s for s in eng.get("stages") or []}
        hay = ""
        if stage == "thesis":
            hay = eng.get("thesis") or ""
        elif stage in rows:
            hay = json.dumps(rows[stage], ensure_ascii=False)
        else:
            errors.append(f"{where}: cites engine stage {stage!r}, which engine.json lacks")
            return
        if _normalize_ws(quote) not in _normalize_ws(hay):
            errors.append(f"{where}: quote is not in engine.json's {stage}: {quote[:60]!r}")
    elif kind == "context":
        p = deck_context.path(slug)
        if not p.exists():
            errors.append(f"{where}: cites CONTEXT.md, which the deck lacks")
            return
        secs = deck_context.sections(p.read_text())
        key = ref.split(":", 1)[-1]
        if key not in secs:
            errors.append(f"{where}: cites context section {key!r}, which CONTEXT.md lacks")
        elif _normalize_ws(quote) not in _normalize_ws(secs[key]):
            errors.append(f"{where}: quote is not in CONTEXT.md's {key}: {quote[:60]!r}")
    elif kind == "skeleton":
        name = ref.split(":", 1)[-1]
        if name not in skeleton_names:
            errors.append(f"{where}: cites skeleton component {name!r}, which is withheld or unknown")


def cc_level(doc):
    return (doc.get("concordance") or {}).get("level")


def validate(slug, doc, *, fresh=None, strategy_sections=None):
    """`(errors, notes)`. `fresh` is the recomputed skeleton (with rank) or None
    when the document is stale and cannot be compared."""
    errors, notes = [], []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing {k!r}")
    if errors:
        return errors, notes
    if doc["tier"] != rating.TIER:
        errors.append(f"tier {doc['tier']!r} — a rating is ★ coaching, nothing else")
    n = (doc.get("rank") or {}).get("of")
    if n is not None and doc["label"] != rating.RATING_LABEL.format(n=n):
        errors.append("label is not the one sentence RATING_LABEL renders for this group")
    if doc["spec_version"] != rating.RATING_SPEC_VERSION:
        errors.append(f"spec_version {doc['spec_version']} != {rating.RATING_SPEC_VERSION}")
    if fresh is not None:
        if _strip_code(doc) != _strip_code(fresh):
            errors.append("a code component, the withheld list, the code score, the point score, the bracket, "
                          "the limits, the context, the uncertainty or the sensitivity differ from a fresh "
                          "computation — the agent may not touch them")
        if _group_view(doc) != _group_view(fresh):
            errors.append("the score, its interval, the grade, the rank or the concordance do not match the "
                          "group's current simulation — the group moved; `rating <slug> --write`")
    u = doc.get("uncertainty") or {}
    if u and (u.get("seed") != rating.RATING_MC_SEED or u.get("k") != rating.RATING_MC_K
              or u.get("kappa") != rating.WEIGHT_CONCENTRATION or u.get("level") != rating.RATING_MC_LEVEL):
        errors.append("uncertainty names a seed, K, κ or level that is not the scorecard's")
    # Weights and the composite.
    comps = doc["components"] or []
    j = doc["judgment"]
    names = [c["name"] for c in comps] + (["judgment"] if j else [])
    for c in comps:
        if c.get("weight") != rating.WEIGHTS.get(c["name"]):
            errors.append(f"components.{c['name']}: weight {c.get('weight')} != WEIGHTS")
        if c.get("weight_validity", 1.0) not in (1.0, rating.SPEED_VALIDITY_ALT_WIN):
            errors.append(f"components.{c['name']}: weight_validity {c.get('weight_validity')!r} is not 1.0 "
                          f"or SPEED_VALIDITY_ALT_WIN")
        if not (0 <= c.get("sub_score", -1) <= 100):
            errors.append(f"components.{c['name']}: sub_score outside [0, 100]")
    weighted = {c["name"]: rating.WEIGHTS[c["name"]] * c.get("weight_validity", 1.0) for c in comps
                if c["name"] in rating.WEIGHTS}
    if j:
        weighted["judgment"] = rating.WEIGHTS["judgment"]
    total = sum(weighted.values())
    eff = sum((c.get("weight_effective") or 0) for c in comps) + ((j or {}).get("weight_effective") or 0)
    if names and abs(eff - 1.0) > 1e-3:
        errors.append(f"effective weights sum to {eff:.4f}, not 1")
    if j is not None:
        if not isinstance(j, dict):
            errors.append("judgment is not an object")
        else:
            sub = j.get("sub_score")
            if isinstance(sub, bool) or not isinstance(sub, int) or not 0 <= sub <= 100:
                errors.append(f"judgment.sub_score {sub!r} must be a whole number in [0, 100]")
            if j.get("weight") != rating.WEIGHTS["judgment"]:
                errors.append("judgment.weight is not the scorecard's")
            against = j.get("against") or {}
            if against.get("decklist_sha256") != doc["decklist_sha256"] or against.get("spec_sha") != doc["model_version"]:
                errors.append("judgment.against does not name this list and this spec")
            arg = j.get("argument") or ""
            words = len(arg.split())
            if words > rating.RATING_ARGUMENT_MAX_WORDS:
                errors.append(f"argument is {words} words; the budget is {rating.RATING_ARGUMENT_MAX_WORDS}")
            if not arg.strip():
                errors.append("judgment without an argument")
            cites = j.get("citations") or []
            kinds = {c.get("kind") for c in cites}
            if len(cites) < 2 or len(kinds) < 2:
                errors.append("the judgment needs at least two citations of at least two kinds")
            skeleton_names = {c["name"] for c in comps}
            for i, c in enumerate(cites):
                _check_citation(c, f"judgment.citations[{i}]", slug, errors, strategy_sections, skeleton_names)
            # Every number in the argument is somewhere it can be checked.
            allowed = set()
            for c in cites:
                allowed |= numbers_in({"quote": c.get("quote") or ""})
            allowed |= numbers_in({"components": comps, "score": doc["code_score"], "point": doc.get("score_point"),
                                   "mc": doc.get("score"), "ci": doc.get("score_ci"), "bracket": doc["bracket"],
                                   "rank": {k: v for k, v in (doc.get("rank") or {}).items() if k != "basis"},
                                   "context": doc["context"], "grade_probabilities": doc.get("grade_probabilities"),
                                   "sensitivity": doc.get("sensitivity"), "concordance": doc.get("concordance")})
            allowed |= {0.0, 1.0, float(sub or 0)}
            for m in _NUM.finditer(arg):
                v = round(float(m.group(0)), 4)
                if v not in allowed and round(v, 1) not in allowed and round(v) not in allowed:
                    errors.append(f"argument quotes {m.group(0)}, which appears in no cited material and no skeleton figure")
            # A divergent deck's argument must address every outlier instrument.
            cc = doc.get("concordance") or {}
            if cc.get("level") == "divergent":
                low = arg.lower()
                cited = {c.get("ref", "").split(":", 1)[-1] for c in cites if c.get("kind") == "skeleton"}
                for o in cc.get("outliers") or []:
                    inst = o.get("instrument")
                    if inst == "judgment" or inst in cited:
                        continue
                    if not any(w in low for w in INSTRUMENT_WORDS.get(inst, (inst,))):
                        errors.append(f"concordance is divergent and the argument never addresses the outlier "
                                      f"instrument {inst!r} (rank {o.get('rank')} of {cc.get('of')}, the {o.get('side')} side)")
        # The point score recomputes from the parts; the MC score is held to the fresh build above.
        if comps or j:
            total = total or 1.0
            point = (sum(weighted[c["name"]] * c["sub_score"] for c in comps if c["name"] in weighted)
                     + (rating.WEIGHTS["judgment"] * j["sub_score"] if isinstance(j, dict) and isinstance(j.get("sub_score"), int) else 0)) / total
            if doc.get("score_point") is None or abs(doc["score_point"] - round(point, 1)) > 0.051:
                errors.append(f"score_point {doc.get('score_point')} does not recompute from the components ({round(point, 1)})")
            if doc["grade"] != rating.grade_of(doc["score"] if doc["score"] is not None else 0):
                errors.append(f"grade {doc['grade']} is not the band for {doc['score']}")
            gr = doc.get("grade_range") or [None, None]
            want = f"{gr[0]}–{gr[1]}" if (cc_level(doc) == "divergent" and gr[0] != gr[1]) else doc["grade"]
            if doc.get("grade_display") != want:
                errors.append(f"grade_display {doc.get('grade_display')!r} should read {want!r} "
                              f"(concordance {cc_level(doc)}, range {gr})")
    for q in doc.get("open_questions") or []:
        if not isinstance(q, dict) or q.get("settled_by") not in SETTLED_BY:
            errors.append(f"open_questions: settled_by must be one of {', '.join(SETTLED_BY)}")
    v = doc.get("verifier")
    if v is not None:
        if v.get("verdict") not in ("pass", "fail"):
            errors.append("verifier.verdict must be pass or fail")
        findings = v.get("findings") or []
        for i, f in enumerate(findings):
            if f.get("status") not in VERIFIER_STATUSES:
                errors.append(f"verifier.findings[{i}]: status {f.get('status')!r} is not in the closed set")
            if not f.get("where"):
                errors.append(f"verifier.findings[{i}]: no `where`")
        if v.get("verdict") == "pass" and any(f.get("status") != "supported" for f in findings):
            errors.append("verifier says pass beside a finding that is not supported")
        if (doc.get("iterations") or 0) > config.RATING_MAX_ITERATIONS:
            errors.append(f"iterations {doc.get('iterations')} > RATING_MAX_ITERATIONS")
    return errors, notes


def staleness(slug, doc):
    """Why the document cannot be recomputed against, or None."""
    why = []
    if doc.get("decklist_sha256") != measured_sha(slug):
        why.append("the list moved")
    if config.FLEET_ANALYTICS_PATH.exists():
        fsha = hashlib.sha256(config.FLEET_ANALYTICS_PATH.read_bytes()).hexdigest()[:12]
        if doc.get("fleet_sha") != fsha:
            why.append("the fleet artifact was rebuilt")
    else:
        why.append("no fleet artifact")
    if doc.get("model_version") != rating.spec_sha():
        why.append("the scorecard's spec changed")
    return ", ".join(why) or None


def main(args=None):
    slug = getattr(args, "slug", None)
    p = deck_dir(slug) / rating.RATING_FILE
    if not p.is_file():
        print(f"{slug}: no {rating.RATING_FILE} — not rated (absent means absent)")
        return
    doc = load_json(p)
    stale = staleness(slug, doc)
    fresh = None
    if stale is None:
        fresh = rating.build(slug, previous=doc)
    else:
        print(f"STALE {slug}: {stale} — form-only check; `/rate-deck {slug}` re-rates")
    try:
        sections = load_strategy_sections()
    except Exception:  # noqa: BLE001 — unavailable DB errors only if a strategy citation appears
        sections = None
    errors, notes = validate(slug, doc, fresh=fresh, strategy_sections=sections)
    report_errors(f"{slug} — {rating.RATING_FILE}", errors)
    j = doc.get("judgment")
    v = doc.get("verifier") or {}
    print(f"OK   {slug} — {rating.RATING_FILE}: {doc.get('grade_display') or doc['grade']} in Bracket {doc['bracket']['key']} "
          f"({(doc.get('rank') or {}).get('position')} of {(doc.get('rank') or {}).get('of')}), score {doc['score']}; "
          f"{'judged' if j else 'skeleton only'}"
          + (f", verifier {v.get('verdict')}" if v else "") + "; ★ over ◆"
          + (" (STALE)" if stale else ""))
