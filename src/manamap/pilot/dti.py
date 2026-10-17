"""DeckCheck's Threat Index as DATED EXTERNAL EVIDENCE — `deckcheck.json`.

ONE FILE PER DECK, `data/decks/<slug>/deckcheck.json`, written by `manamap pilot
deckcheck <slug> --write` after the pilot has run a scan on DeckCheck (the bench cannot
trigger one — the API is read-only, and a scan costs credits the pilot spends by hand).
Every pull is a READING: the score (0–96), the bracket (1–5) and the twelve S–F
benchmark grades, stamped `as_of`, with DeckCheck's own `updatedAtUtc` beside it and
the sha of OUR list at the time. Readings are never edited or deleted; a rescan of the
same list is a new reading, because DeckCheck's read is an LLM's and moves between
scans (their own "Omni-Score" is the median across scans). Two blocks are DERIVED from
the readings and recomputed whenever the list moves:

  `omni`           median_low per benchmark, score and bracket over readings of the
                   SAME list — DeckCheck's Omni rule applied to our readings, said so;
                   their formula is unpublished and this is not their figure.
  `floor_vs_dti`   our deterministic floor (`bracket_report.json`, stamped) against
                   their bracket. `dti_lower` is a contradiction — DeckCheck can only
                   RAISE its own rules floor — and reads as "their copy is probably an
                   older list", never as a lowered floor.

THE FLOOR GATES; THE DTI NEVER DOES. `bracket.py` is deterministic and cites its
drivers; this is a second opinion, labelled external, quoted with its date, and absent
means absent — a deck nobody scanned has no file, not a zero. No `fullAnalysis` HTML is
ever stored (DeckCheck's terms, and it is prose). Nothing here reads the network or
the clock: `append` is handed `today`, and the age line is a READ-path concern
(`deck_info._next(today=…)`, the page), so the tracked `info.json` never depends on
the day it was written.
"""

import json
import statistics
from datetime import date

from manamap import config
from manamap.pilot.common import deck_dir, load_json, measured_sha

ARTIFACT = "deckcheck.json"
SOURCE = "deckcheck.co"
SCHEMA_VERSION = 1
MAX_SCORE = 96
GRADES = ("S", "A", "B", "C", "F")
GRADE_RANK = {"F": 1, "C": 2, "B": 3, "A": 4, "S": 5}
RANK_GRADE = {v: k for k, v in GRADE_RANK.items()}
#: DeckCheck's twelve benchmarks in five groups (https://deckcheck.co/commander-brackets).
BENCHMARK_GROUPS = {"R": ("R1", "R2"), "A": ("A1", "A2"), "P": ("P1", "P2", "P3"),
                    "I": ("I1", "I2"), "S": ("S1", "S2", "S3")}
GROUP_NAMES = {"R": "Resources", "A": "Access", "P": "Pressure", "I": "Interaction",
               "S": "Resilience"}
BENCHMARKS = frozenset(sum(BENCHMARK_GROUPS.values(), ()))
#: Never stored, under any spelling, anywhere in the document.
FORBIDDEN_KEYS = frozenset({"fullAnalysis", "full_analysis"})
RELATIONS = ("equal", "dti_higher", "dti_lower")
READING_KEYS = ("as_of", "source_updated_at", "decklist_sha256", "deckcheck_id", "url",
                "score", "max_score", "bracket", "benchmarks")
OMNI_RULE = ("median_low per benchmark, score and bracket over readings of this exact "
             "list — DeckCheck's Omni-Score rule applied to OUR readings; their formula "
             "is unpublished and this is not their figure")


def path(slug, branch=None):
    return deck_dir(slug, branch) / ARTIFACT


def load(slug, branch=None):
    doc = load_json(path(slug, branch), None)
    return doc if isinstance(doc, dict) else None


def _write(p, doc):
    p.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                 encoding="utf-8")


# ── one reading ────────────────────────────────────────────────────────────

def reading_from_payload(payload, *, deck_id, url, as_of, decklist_sha256):
    """The twelve keys and the three numbers, and nothing else of the document.

    Raises `ValueError` on a document with no `dti`, a benchmark set that is not
    exactly the twelve, or a grade outside the vocabulary — the seam that says
    when DeckCheck changes its index again (it has twice in a year)."""
    dti = (payload or {}).get("dti")
    if not isinstance(dti, dict):
        raise ValueError("the DeckCheck document carries no `dti` — not a Commander "
                         "deck, or not yet scanned")
    bm = dti.get("benchmarks") or {}
    if set(bm) != BENCHMARKS:
        missing, extra = sorted(BENCHMARKS - set(bm)), sorted(set(bm) - BENCHMARKS)
        raise ValueError(f"dti.benchmarks is not the twelve: missing {missing}, extra {extra}")
    bad = {k: v for k, v in bm.items() if v not in GRADES}
    if bad:
        raise ValueError(f"dti.benchmarks carries grades outside {GRADES}: {bad}")
    try:
        score, bracket = int(dti["score"]), int(dti["bracket"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"dti.score / dti.bracket unreadable: {exc}") from exc
    return {"as_of": as_of, "source_updated_at": payload.get("updatedAtUtc"),
            "decklist_sha256": decklist_sha256, "deckcheck_id": deck_id, "url": url,
            "score": score, "max_score": int(dti.get("maxScore") or MAX_SCORE),
            "bracket": bracket, "benchmarks": {k: bm[k] for k in sorted(bm)}}


def same_reading(a, b):
    """The dedupe key: DeckCheck's own edit time, the score, the bracket and every
    grade. `as_of` and our sha are not part of it — pulling the same answer on
    a later day, or after our list moved, adds nothing."""
    return all((a or {}).get(k) == (b or {}).get(k)
               for k in ("source_updated_at", "score", "bracket", "benchmarks"))


# ── the derived blocks ─────────────────────────────────────────────────────

def omni(readings, decklist_sha256):
    """Median_low over the readings of ONE list. `n == 0` is an absence with a
    reason, never a zero."""
    same = [r for r in (readings or [])
            if r.get("decklist_sha256") and r["decklist_sha256"] == decklist_sha256]
    if not same:
        return {"decklist_sha256": decklist_sha256, "n": 0,
                "absent_because": ("no reading of the current list — scan on DeckCheck, "
                                   "then `manamap pilot deckcheck <slug> --write`")}
    grades = {k: RANK_GRADE[statistics.median_low(GRADE_RANK[r["benchmarks"][k]] for r in same)]
              for k in sorted(BENCHMARKS)}
    return {"decklist_sha256": decklist_sha256, "n": len(same),
            "median_score": statistics.median_low(r["score"] for r in same),
            "median_bracket": statistics.median_low(r["bracket"] for r in same),
            "benchmarks": grades, "rule": OMNI_RULE}


def compare_floor(reading, bracket_report):
    """Our floor against their bracket, for the latest reading."""
    if not bracket_report or bracket_report.get("floor") is None:
        return {"absent_because": "no bracket_report.json — `manamap pilot bracket-check <slug>`"}
    if not reading:
        return {"absent_because": "no reading"}
    floor, theirs = int(bracket_report["floor"]), int(reading["bracket"])
    relation = "equal" if theirs == floor else "dti_higher" if theirs > floor else "dti_lower"
    out = {"floor": floor, "floor_name": bracket_report.get("floor_name"),
           "floor_decklist_sha256": bracket_report.get("decklist_sha256"),
           "their_bracket": theirs, "relation": relation, "reading_as_of": reading.get("as_of")}
    if relation == "dti_lower":
        out["note"] = ("DeckCheck can only RAISE its own rules floor, so a bracket below "
                       "ours means their copy is probably an older list — compare "
                       "source_updated_at; the floor is not lowered")
    return out


def latest(doc):
    rs = (doc or {}).get("readings") or []
    return rs[-1] if rs else None


def _derive(doc, slug, branch=None):
    sha = measured_sha(slug, branch)
    doc["omni"] = omni(doc.get("readings"), sha)
    report = load_json(deck_dir(slug, branch) / "bracket_report.json", None)
    doc["floor_vs_dti"] = compare_floor(latest(doc), report if isinstance(report, dict) else None)
    return doc


def _new(slug, branch):
    return {"schema_version": SCHEMA_VERSION, "slug": slug, "branch": branch,
            "source": SOURCE, "last_pulled": None, "readings": []}


def append(slug, reading, *, branch=None, today=None):
    """Add a reading unless it repeats the LAST one; recompute the derived blocks;
    bump `last_pulled` either way (a same-answer pull is a fact worth a date)."""
    p = path(slug, branch)
    doc = load(slug, branch) or _new(slug, branch)
    last = latest(doc)
    added = not (last and same_reading(last, reading))
    if added:
        doc["readings"].append(reading)
    doc["last_pulled"] = (today or date.today()).isoformat()
    _derive(doc, slug, branch)
    _write(p, doc)
    return {"path": p, "added": added, "readings_n": len(doc["readings"]),
            "deduped_against": None if added else last.get("as_of")}


def reconcile(slug, branch=None):
    """After the list moved: recompute `omni` and `floor_vs_dti` against the current
    `cards.json` and floor. Readings are untouched; writes only when a byte moves.
    None when there is no file."""
    p = path(slug, branch)
    doc = load(slug, branch)
    if doc is None:
        return None
    before = json.dumps(doc, sort_keys=True)
    _derive(doc, slug, branch)
    changed = json.dumps(doc, sort_keys=True) != before
    if changed:
        _write(p, doc)
    return {"path": p, "changed": changed,
            "readings_on_current_list": doc["omni"].get("n", 0)}


# ── reading it back ────────────────────────────────────────────────────────

def stale_list(doc, measured):
    """True when the newest reading describes another list; None when either
    side is absent (absent is not a match)."""
    r = latest(doc)
    if not r or not r.get("decklist_sha256") or not measured:
        return None
    return r["decklist_sha256"] != measured


def age_days(doc, today):
    r = latest(doc)
    if not r or not today:
        return None
    try:
        return (today - date.fromisoformat(r["as_of"])).days
    except (TypeError, ValueError):
        return None


def info_block(doc, measured):
    """What `deck_info.compose` embeds. No clock: `max_age_days` rides along so a
    page can judge age without restating the constant."""
    r = latest(doc)
    if not r:
        return None
    return {"score": r["score"], "max_score": r["max_score"], "bracket": r["bracket"],
            "as_of": r["as_of"], "source_updated_at": r.get("source_updated_at"),
            "url": r.get("url"), "stale_list": stale_list(doc, measured),
            "readings_n": len(doc.get("readings") or []),
            "omni": doc.get("omni"), "floor_vs_dti": doc.get("floor_vs_dti"),
            "max_age_days": config.DECKCHECK_MAX_AGE_DAYS}


def next_lines(block, slug, today=None, max_age=None):
    """The sha line whenever the reading describes an older list; the age line
    only when a clock is handed in and the reading is past `max_age`."""
    if not block:
        return []
    max_age = config.DECKCHECK_MAX_AGE_DAYS if max_age is None else max_age
    out = []
    fix = f"scan on DeckCheck, then `manamap pilot deckcheck {slug} --write`"
    if block.get("stale_list"):
        out.append(f"DTI ({block['as_of']}) describes an older list — {fix}")
    if today is not None:
        try:
            age = (today - date.fromisoformat(block["as_of"])).days
        except (TypeError, ValueError):
            age = None
        if age is not None and age > max_age and not block.get("stale_list"):
            out.append(f"DTI is {age} days old — {fix}")
    return out


def near(slug, within, decks_dir=None):
    """The fleet's decks within ±`within` DTI points of `slug`'s latest reading —
    a matched set from the pilot's own decks. Decks with no file are NAMED, never
    scored."""
    root = decks_dir or config.DECKS_DIR
    anchor = load(slug)
    a = latest(anchor)
    rows, without = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        other = load_json(d / ARTIFACT, None)
        r = latest(other) if isinstance(other, dict) else None
        if not r:
            without.append(d.name)
            continue
        if d.name == slug or a is None:
            continue
        delta = r["score"] - a["score"]
        if abs(delta) <= within:
            rows.append({"slug": d.name, "score": r["score"], "bracket": r["bracket"],
                         "delta": delta, "as_of": r["as_of"],
                         "stale_list": stale_list(other, measured_sha(d.name))})
    rows.sort(key=lambda x: (abs(x["delta"]), x["slug"]))
    return {"anchor": slug, "anchor_score": a["score"] if a else None,
            "anchor_as_of": a["as_of"] if a else None, "within": within,
            "rows": rows, "without_dti": without}


def format_near(doc):
    if doc["anchor_score"] is None:
        return (f"{doc['anchor']}: no DTI on file — scan on DeckCheck, then "
                f"`manamap pilot deckcheck {doc['anchor']} --write`")
    lines = [f"DECKS WITHIN ±{doc['within']} DTI OF {doc['anchor']} "
             f"({doc['anchor_score']}/{MAX_SCORE}, {doc['anchor_as_of']})"]
    for r in doc["rows"]:
        flag = "  (older list)" if r["stale_list"] else ""
        lines.append(f"  {r['slug']:20} {r['score']:>3}/{MAX_SCORE}  B{r['bracket']}  "
                     f"{r['delta']:+d}  ({r['as_of']}){flag}")
    if not doc["rows"]:
        lines.append("  none")
    if doc["without_dti"]:
        lines.append(f"  no DTI on file: {', '.join(doc['without_dti'])}")
    return "\n".join(lines)


def format_history(doc, measured):
    rs = (doc or {}).get("readings") or []
    if not rs:
        return "no readings"
    lines = [f"DTI READINGS — {doc['slug']} ({len(rs)}; last pulled {doc.get('last_pulled')})"]
    for r in rs:
        which = "current list" if r.get("decklist_sha256") == measured else "older list"
        lines.append(f"  {r['as_of']}  {r['score']:>3}/{r['max_score']}  B{r['bracket']}  "
                     f"{''.join(r['benchmarks'][k] for k in sorted(BENCHMARKS))}  "
                     f"{(r.get('decklist_sha256') or '')[:12]}  {which}")
    o = doc.get("omni") or {}
    if o.get("n"):
        lines.append(f"  omni over {o['n']} reading(s) of this list: {o['median_score']}/"
                     f"{MAX_SCORE} B{o['median_bracket']} — ours, not DeckCheck's")
    return "\n".join(lines)
