"""Pilot: form-check `deckcheck.json` — DeckCheck's Threat Index as dated evidence.

FORM ONLY, NEVER FRESHNESS OR REACHABILITY: the keys, the vocabularies, the ranges,
the dates, the hosts, the order of readings, no adjacent duplicates, and that the
derived `omni` re-adds from the readings. Three things are WARNED, not failed, because
they are facts about time rather than form: the newest reading describes an older
list than `cards.json`; the stored floor is not the current `bracket_report.json`'s;
DeckCheck's bracket sits below our floor (their copy is probably older). Absent file:
"absent means absent", exit 0 — a deck nobody scanned is not a defect.
"""

import re
from datetime import date, datetime
from urllib.parse import urlsplit

from manamap import config
from manamap.pilot import dti
from manamap.pilot.common import deck_dir, load_json, measured_sha, report_errors
from manamap.pilot.deck_link import ID_RE, SERVICES

REQUIRED = ("schema_version", "slug", "branch", "source", "last_pulled", "readings",
            "omni", "floor_vs_dti")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
HOSTS = SERVICES["deckcheck"]["hosts"]


def _iso_date(s):
    try:
        date.fromisoformat(str(s))
        return True
    except (TypeError, ValueError):
        return False


def _iso_timestamp(s):
    try:
        datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _forbidden(obj, trail="$"):
    """Every path at which a forbidden key appears, anywhere in the document."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in dti.FORBIDDEN_KEYS:
                out.append(f"{trail}.{k}")
            out += _forbidden(v, f"{trail}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _forbidden(v, f"{trail}[{i}]")
    return out


def _reading_errors(i, r):
    e = []
    tag = f"readings[{i}]"
    if not isinstance(r, dict):
        return [f"{tag} is not an object"]
    for k in dti.READING_KEYS:
        if k not in r:
            e.append(f"{tag} lacks {k!r}")
    for k in sorted(set(r) - set(dti.READING_KEYS)):
        e.append(f"{tag} carries an unknown key {k!r}")
    if "as_of" in r and not _iso_date(r["as_of"]):
        e.append(f"{tag}.as_of {r['as_of']!r} is not an ISO date")
    if r.get("source_updated_at") is not None and not _iso_timestamp(r["source_updated_at"]):
        e.append(f"{tag}.source_updated_at {r['source_updated_at']!r} is not an ISO timestamp")
    if r.get("decklist_sha256") is not None and not SHA_RE.match(str(r["decklist_sha256"])):
        e.append(f"{tag}.decklist_sha256 is not a sha256")
    if "deckcheck_id" in r and not ID_RE.match(str(r["deckcheck_id"])):
        e.append(f"{tag}.deckcheck_id {r['deckcheck_id']!r} does not match [A-Za-z0-9_-]+")
    if "url" in r:
        parts = urlsplit(str(r["url"]))
        if parts.scheme != "https" or parts.hostname not in HOSTS or parts.query:
            e.append(f"{tag}.url {r['url']!r} is not https on {', '.join(sorted(HOSTS))} "
                     f"without a query string")
    ms = r.get("max_score")
    if ms != dti.MAX_SCORE:
        e.append(f"{tag}.max_score {ms!r} is not {dti.MAX_SCORE}")
    s = r.get("score")
    if isinstance(s, bool) or not isinstance(s, int) or not 0 <= s <= dti.MAX_SCORE:
        e.append(f"{tag}.score {s!r} is not an int in [0, {dti.MAX_SCORE}]")
    b = r.get("bracket")
    if isinstance(b, bool) or not isinstance(b, int) or b not in config.BRACKETS:
        e.append(f"{tag}.bracket {b!r} is not one of {sorted(config.BRACKETS)}")
    bm = r.get("benchmarks")
    if not isinstance(bm, dict) or set(bm) != dti.BENCHMARKS:
        e.append(f"{tag}.benchmarks is not exactly the twelve "
                 f"{', '.join(sorted(dti.BENCHMARKS))}")
    else:
        bad = {k: v for k, v in bm.items() if v not in dti.GRADES}
        if bad:
            e.append(f"{tag}.benchmarks carries grades outside {dti.GRADES}: {bad}")
    return e


def validate(slug, branch, doc, *, measured=None, bracket_report=None):
    """`(errors, warnings)`. Errors are form; warnings are time."""
    errors, warnings = [], []
    if not isinstance(doc, dict):
        return [f"{dti.ARTIFACT} is not an object"], []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"lacks {k!r}")
    if doc.get("schema_version") != dti.SCHEMA_VERSION:
        errors.append(f"schema_version {doc.get('schema_version')!r} is not {dti.SCHEMA_VERSION}")
    if doc.get("slug") != slug:
        errors.append(f"slug {doc.get('slug')!r} is not the directory's ({slug!r})")
    if doc.get("branch") != branch:
        errors.append(f"branch {doc.get('branch')!r} is not the directory's ({branch!r})")
    if doc.get("source") != dti.SOURCE:
        errors.append(f"source {doc.get('source')!r} is not {dti.SOURCE!r}")
    if "last_pulled" in doc and not _iso_date(doc["last_pulled"]):
        errors.append(f"last_pulled {doc['last_pulled']!r} is not an ISO date")
    for p in _forbidden(doc):
        errors.append(f"{p}: DeckCheck's analysis HTML is never stored")
    rs = doc.get("readings")
    if not isinstance(rs, list) or not rs:
        errors.append("readings is not a non-empty list")
        return errors, warnings
    for i, r in enumerate(rs):
        errors += _reading_errors(i, r)
    if errors:
        return errors, warnings
    for i in range(1, len(rs)):
        if rs[i]["as_of"] < rs[i - 1]["as_of"]:
            errors.append(f"readings[{i}] ({rs[i]['as_of']}) is dated before readings[{i - 1}]")
        if dti.same_reading(rs[i - 1], rs[i]):
            errors.append(f"readings[{i}] repeats readings[{i - 1}] — `append` dedupes these")
    o = doc.get("omni")
    if not isinstance(o, dict) or "decklist_sha256" not in o:
        errors.append("omni is not an object with a decklist_sha256")
    elif o != dti.omni(rs, o["decklist_sha256"]):
        errors.append("omni does not re-add from the readings of its list")
    f = doc.get("floor_vs_dti")
    if not isinstance(f, dict):
        errors.append("floor_vs_dti is not an object")
    elif "relation" in f and f["relation"] not in dti.RELATIONS:
        errors.append(f"floor_vs_dti.relation {f['relation']!r} is not one of {dti.RELATIONS}")
    if errors:
        return errors, warnings
    # Time, not form.
    newest = rs[-1]
    if measured and newest.get("decklist_sha256") and newest["decklist_sha256"] != measured:
        warnings.append(f"the newest reading ({newest['as_of']}) describes an older list — "
                        f"scan on DeckCheck, then `deckcheck {slug} --write`")
    if bracket_report and "floor" in (f or {}) and f["floor"] != bracket_report.get("floor"):
        warnings.append(f"floor_vs_dti.floor {f['floor']} is not the current floor "
                        f"{bracket_report.get('floor')} — `deckcheck {slug} --write` recomputes it")
    if (f or {}).get("relation") == "dti_lower":
        warnings.append("DeckCheck's bracket is below our floor — their copy is probably "
                        "an older list (compare source_updated_at); the floor is not lowered")
    return errors, warnings


def main(args=None):
    slug = getattr(args, "slug", None)
    branch = getattr(args, "branch", None)
    p = deck_dir(slug, branch) / dti.ARTIFACT
    if not p.is_file():
        print(f"{slug}: no {dti.ARTIFACT} — not scanned on DeckCheck (absent means absent)")
        return
    doc = load_json(p)
    report = load_json(deck_dir(slug, branch) / "bracket_report.json", None)
    errors, warnings = validate(slug, branch, doc, measured=measured_sha(slug, branch),
                                bracket_report=report if isinstance(report, dict) else None)
    for w in warnings:
        print(f"WARN {w}")
    report_errors(f"{slug} — {dti.ARTIFACT}", errors)
    r = dti.latest(doc)
    print(f"OK   {slug} — {dti.ARTIFACT}: {len(doc['readings'])} reading(s), latest "
          f"{r['as_of']} {r['score']}/{r['max_score']} B{r['bracket']}"
          + (f"; {len(warnings)} warning(s)" if warnings else "") + "; dated external evidence ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-deckcheck <slug>`.")
