"""The gate on `sim_findings.json`: the prose names nothing the record did not.

Modelled on `validate_debrief` and `validate_captains_log`: every citation in
the prose must be a finding id in THAT run; every number in the prose must
appear in a cited finding's figure, interval or N; `settled_by` is a closed set; a
`so_what` with no citation is an opinion about a decklist, which four other
artifacts already are. An EMPTY prose block is fine — the skeleton is honest on
its own — and a skeleton-only file passes on every deck.
"""

import re

from manamap.pilot import pilot_findings as pf
from manamap.pilot.common import deck_dir, load_json, report_errors

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)%?(?![\w.])")
_FID = re.compile(r"\bF-[0-9a-f]{8}-\d{2}\b")


def _numbers_in(finding):
    """Every number a finding carries, in the forms prose might quote."""
    vals = set()

    def take(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.add(round(float(x), 4))
            vals.add(round(float(x) * 100, 1))          # a rate quoted as a percent
            vals.add(round(float(x) * 100))
        elif isinstance(x, dict):
            for v in x.values():
                take(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                take(v)
        elif isinstance(x, str):
            for m in _NUM.finditer(x):
                vals.add(round(float(m.group(1)), 4))
    for k, v in finding.items():
        if k not in ("id", "kind", "source", "basis"):
            take(v)
    return vals


def validate(doc, slug=None):
    errors, notes = [], []
    if not isinstance(doc, dict) or "runs" not in doc:
        return ["not a sim_findings document"], notes
    for rid, run in (doc.get("runs") or {}).items():
        where = f"runs[{rid[:24]}]"
        findings = {f["id"]: f for f in run.get("findings") or []}
        for f in findings.values():
            for k in ("id", "kind", "text"):
                if k not in f:
                    errors.append(f"{where}: a finding with no {k!r}")
        prose = run.get("prose") or {}
        unknown = set(prose) - set(pf.PROSE_KEYS)
        if unknown:
            errors.append(f"{where}: prose carries {sorted(unknown)}; only {pf.PROSE_KEYS}")
        items = []
        if prose.get("reading"):
            items.append(("reading", prose["reading"], None))
        for i, sw in enumerate(prose.get("so_what") or []):
            if not (sw.get("cites") or []):
                errors.append(f"{where}: so_what[{i}] cites nothing — a claim with no finding "
                              f"behind it is an opinion about a decklist")
            items.append((f"so_what[{i}]", sw.get("text") or "", sw.get("cites") or []))
        for i, q in enumerate(prose.get("open_questions") or []):
            if q.get("settled_by") not in pf.SETTLED_BY:
                errors.append(f"{where}: open_questions[{i}].settled_by {q.get('settled_by')!r} "
                              f"is not one of {pf.SETTLED_BY}")
            items.append((f"open_questions[{i}]", q.get("question") or "", q.get("cites") or []))
        for label, text, cites in items:
            cited = set(cites or []) | set(_FID.findall(text))
            missing = [c for c in cited if c not in findings]
            if missing:
                errors.append(f"{where}.{label}: cites {missing} — not a finding of this run")
            allowed_nums = set()
            for c in cited:
                if c in findings:
                    allowed_nums |= _numbers_in(findings[c])
            for m in _NUM.finditer(text):
                v = round(float(m.group(1)), 4)
                if v not in allowed_nums and v not in (0.0, 1.0):
                    errors.append(f"{where}.{label}: the number {m.group(0)} is not in any cited "
                                  f"finding — every figure in the prose is the record's")
    # CARD NAMES ARE NOT CHECKED. A heuristic that guesses which capitalised
    # words are card names would fire on correct prose, and a validator that
    # fires on correct data is worse than none; the numbers are the claim that
    # can be held to the record, and they are.
    return errors, notes


def main(args):
    slug = args.slug
    p = deck_dir(slug) / pf.ARTIFACT
    if not p.exists():
        print(f"{slug}: no {pf.ARTIFACT}")
        return
    doc = load_json(p) or {}
    errors, notes = validate(doc, slug)
    for n in notes:
        print(f"  ! {n}")
    report_errors(f"{slug}: {pf.ARTIFACT}", errors)
    print(f"{slug}: {pf.ARTIFACT} — {len(doc.get('runs') or {})} run(s), "
          f"{sum(1 for r in doc['runs'].values() if r.get('prose'))} with prose, OK")
