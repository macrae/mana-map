"""`manamap pilot pods <name> --power`: is this table evenly matched?

ONE ROW PER SEAT, three readings of power set side by side and never merged:

  typed     the bracket the pod file SAYS the seat is (hand-typed; may be absent —
            the real `playgroup` table carries none);
  floor     our deterministic bracket floor (`bracket.assess`) — read from the
            stamped `bracket_report.json` for one of our decks, COMPUTED from the
            seat's `decklist.txt` for an opponent (which has no report), and from
            the DeckCheck copy for a seat whose pod entry carries a `deckcheck` URL;
  DTI       DeckCheck's Threat Index — `deckcheck.json`'s latest reading for our
            decks (dated, with its history), or the live document's `dti` for a
            `deckcheck` seat (dated by the cache record, no history);

plus `share_of_fair` with its interval from `pods.calibration` where Forge has
played the table. Flags name the disagreements: a typed bracket off the floor or
the DTI, a DTI below the floor (their copy is probably an older list), a DTI that
describes an older list, a DTI past `DECKCHECK_MAX_AGE_DAYS` (only when a clock
is handed in). `matched_within` is max − min DTI over the seats that have one,
with the count that do not — the "evenly matched" figure, stated as a spread.

THE HONEST CELL. Spearman(DTI, share_of_fair) over the seats would say whether
DTI predicts who dominates a table. Below `calibrate.MIN_DECKS` it is reported
as NOT ANSWERABLE with the n, and a six-deck fleet can never seat ten at one
table — so this view REPORTS and never claims DTI predicts outcomes. The floor
gates; the DTI never does.

Read-only. A `deckcheck` seat is fetched through `pilot/deckcheck.py` (cached
fifteen minutes, never stored); offline it reads as an absence with its reason.
"""

import json
from datetime import date

from manamap import config
from manamap.pilot import calibrate
from manamap.pilot.common import deck_dir, load_json, measured_sha
from manamap.sim import pods

SOURCE_OURS = "bracket_report.json"
SOURCE_OPPONENT = "computed from decklist.txt (an EDHREC stand-in)"
SOURCE_DECKCHECK = "computed from the DeckCheck copy"


def _absent(reason):
    return {"absent_because": reason}


def _is_ours(slug):
    return (config.DECKS_DIR / slug / "cards.json").exists()


def _reference():
    """`bracket.load_reference()` or None when the corpus is absent."""
    from manamap.pilot import bracket
    try:
        return bracket.load_reference()
    except SystemExit:
        return None


def _floor_of(names, commanders, reference):
    from manamap.pilot import bracket
    if reference is None:
        return _absent("corpus not present — `manamap extract`, `card-roles`, `process-combos`")
    rep = bracket.assess(names, *reference, commanders)
    return {"floor": rep["floor"], "floor_name": rep["floor_name"],
            "drivers_n": len(rep.get("drivers") or [])}


def seat_entries(seat, fetch=None):
    """The seat's list as parse_decklist entries, and where it came from."""
    from manamap.pilot.fetch_deck import parse_mainboard
    url = seat.get("deckcheck")
    if url:
        from manamap.pilot import deckcheck
        from manamap import net
        try:
            doc = (fetch or deckcheck.fetch)(deck_url=url)
        except net.Offline as exc:
            return None, _absent(f"could not reach DeckCheck — {exc}"), None
        except deckcheck.Private as exc:
            return None, _absent(str(exc)), None
        return deckcheck.project(doc), SOURCE_DECKCHECK, doc
    from manamap.sim import forge
    try:
        d = forge.seat_dir(seat["slug"])
    except SystemExit as exc:
        return None, _absent(str(exc)), None
    text = (d / "decklist.txt").read_text(encoding="utf-8")
    return parse_mainboard(text), (SOURCE_OURS if _is_ours(seat["slug"]) else SOURCE_OPPONENT), None


def seat_floor(seat, entries=None, source=None, reference=None):
    """Our stamped report for one of our decks; computed for everyone else."""
    slug = seat["slug"]
    if source == SOURCE_OURS or (source is None and _is_ours(slug)):
        rep = load_json(deck_dir(slug) / "bracket_report.json", None)
        if isinstance(rep, dict) and rep.get("floor") is not None:
            return {"floor": rep["floor"], "floor_name": rep.get("floor_name"),
                    "drivers_n": len(rep.get("drivers") or []), "source": SOURCE_OURS,
                    "decklist_sha256": rep.get("decklist_sha256")}
    if isinstance(source, dict):                       # the list could not be had
        return dict(source, source="none")
    if entries is None:
        return _absent("no list to compute a floor from")
    names = [e["name"] for e in entries]
    commanders = [e["name"] for e in entries if e.get("is_commander")]
    out = _floor_of(names, commanders, reference if reference is not None else _reference())
    out["source"] = source or SOURCE_OPPONENT
    return out


def seat_dti(seat, doc=None, today=None):
    """Our `deckcheck.json` (dated, with history) or a `deckcheck` seat's live
    document (dated by its cache record). None when neither applies."""
    from manamap.pilot import dti
    slug = seat["slug"]
    if seat.get("deckcheck"):
        if not isinstance(doc, dict):
            return None
        try:
            r = dti.reading_from_payload(doc, deck_id=doc.get("id") or doc.get("publicId"),
                                         url=doc.get("publicUrl") or seat["deckcheck"],
                                         as_of=(today or date.today()).isoformat(),
                                         decklist_sha256=None)
        except ValueError as exc:
            return _absent(str(exc))
        from manamap.pilot import deckcheck
        read_at = deckcheck.read_at(deck_url=seat["deckcheck"])
        return {"score": r["score"], "max_score": r["max_score"], "bracket": r["bracket"],
                "benchmarks": r["benchmarks"], "as_of": (read_at or r["as_of"])[:10],
                "stale_list": None, "readings_n": None, "source": "deckcheck.co (live)",
                "age_days": None}
    if not _is_ours(slug):          # an opponent stand-in has no deck dir, no file
        return None
    file = dti.load(slug)
    r = dti.latest(file)
    if not r:
        return None
    return {"score": r["score"], "max_score": r["max_score"], "bracket": r["bracket"],
            "benchmarks": r["benchmarks"], "as_of": r["as_of"],
            "stale_list": dti.stale_list(file, measured_sha(slug)),
            "readings_n": len(file["readings"]), "source": "deckcheck.json",
            "age_days": dti.age_days(file, today)}


def _flags(seat, floor, dti_row, today):
    flags = []
    typed = seat.get("bracket")
    f = (floor or {}).get("floor")
    b = (dti_row or {}).get("bracket")
    if typed is not None and f is not None and typed != f:
        flags.append(f"typed B{typed} ≠ floor B{f}")
    if typed is not None and b is not None and typed != b:
        flags.append(f"typed B{typed} ≠ DTI B{b}")
    if f is not None and b is not None and b < f:
        flags.append(f"DTI B{b} below floor B{f} — their copy is probably an older list")
    if (dti_row or {}).get("stale_list"):
        flags.append("DTI describes an older list")
    age = (dti_row or {}).get("age_days")
    if today is not None and age is not None and age > config.DECKCHECK_MAX_AGE_DAYS:
        flags.append(f"DTI is {age} days old")
    return flags


def table(name, *, today=None, fetch=None, reference=None, calibration=None):
    """The whole table, read-only. Injectable fetch / reference / calibration
    for tests and for an offline run."""
    pod = pods.load(name)
    cal = calibration if calibration is not None else pods.calibration(name)
    by_seat = {r["seat"]: r for r in (cal.get("seats") or [])} if cal.get("measured") else {}
    ref = reference
    # CAN THE AI PILOT THE SEAT (2026-10-21): from the pod's seat audit where one is
    # on disk; absent, not "fine", where none is.
    from manamap.sim import seat_audit as _sa
    _audit = _sa.load(name)
    _pilot = {r["slug"]: r for r in (_audit or {}).get("seats", [])}
    rows = []
    for seat in pod["seats"]:
        entries, source, doc = seat_entries(seat, fetch=fetch)
        if entries is not None and ref is None and source != SOURCE_OURS:
            ref = _reference()
        floor = seat_floor(seat, entries, source, reference=ref)
        d = seat_dti(seat, doc, today=today)
        crow = by_seat.get(seat["slug"]) or (by_seat.get("SUBJECT") if _is_ours(seat["slug"]) else None)
        rows.append({"slug": seat["slug"], "archetype": seat.get("archetype"),
                     "typed_bracket": seat.get("bracket"),
                     "floor": floor, "dti": d,
                     "share_of_fair": (crow or {}).get("share_of_fair"),
                     "ci95": (crow or {}).get("ci95"), "games": (crow or {}).get("games"),
                     "flags": _flags(seat, floor, d, today),
                     "pilot": ({"verdict": _pilot[seat["slug"]]["verdict"],
                                "why": _pilot[seat["slug"]]["why"], "at": _audit["at"]}
                               if seat["slug"] in _pilot else None)})
    scores = [r["dti"]["score"] for r in rows if r["dti"] and r["dti"].get("score") is not None]
    matched = {"within": (max(scores) - min(scores)) if len(scores) >= 2 else None,
               "seats_with_dti": len(scores), "seats": len(rows)}
    pairs = [(r["dti"]["score"], r["share_of_fair"]) for r in rows
             if r["dti"] and r["dti"].get("score") is not None and r["share_of_fair"] is not None]
    n = len(pairs)
    if n < calibrate.MIN_DECKS:
        spear = {"verdict": "NOT ANSWERABLE", "n": n,
                 "why": (f"Spearman(DTI, share of fair) needs {calibrate.MIN_DECKS} seats with "
                         f"both a DTI and a Forge record at this table; this one has {n}"),
                 "what_it_would_take": "a DTI on every seat and Forge runs of the table, "
                                       "across enough tables to pool"}
    else:
        rho = calibrate.spearman([p[0] for p in pairs], [p[1] for p in pairs])
        crit = calibrate.critical_rho(n)
        spear = {"verdict": "measured", "n": n, "rho": round(rho, 3), "critical_rho": crit,
                 "significant": abs(rho) >= crit}
    return {"pod": name, "today": today.isoformat() if today else None, "seats": rows,
            "matched_within": matched,
            "balance": (cal.get("balance") if cal.get("measured")
                        else _absent("this table has never been simulated")),
            "spearman": spear,
            "limits": ["The floor gates; the DTI is dated external evidence and never does.",
                       "An opponent seat's floor is computed from an EDHREC stand-in unless the "
                       "seat carries a `deckcheck` URL to the friend's real list.",
                       "A `deckcheck` seat's DTI is read live and kept nowhere."]}


def format_table(doc):
    lines = [f"\n{doc['pod']} — POWER, seat by seat"]
    for r in doc["seats"]:
        typed = f"typed B{r['typed_bracket']}" if r["typed_bracket"] is not None else "typed —"
        f = r["floor"] or {}
        floor = f"floor B{f['floor']}" if f.get("floor") is not None else f"floor — ({f.get('absent_because', '')})"
        d = r["dti"]
        if d and d.get("score") is not None:
            dti = (f"DTI {d['score']}/{d['max_score']} B{d['bracket']} ({d['as_of']})"
                   + (" older list" if d.get("stale_list") else ""))
        elif d and d.get("absent_because"):
            dti = f"DTI — ({d['absent_because']})"
        else:
            dti = "DTI —"
        share = (f"{r['share_of_fair']}× fair [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
                 if r["share_of_fair"] is not None and r["ci95"] else "no Forge record")
        lines.append(f"  {r['slug']:<20} {typed} · {floor} · {dti} · {share}")
        for fl in r["flags"]:
            lines.append(f"      ! {fl}")
    m = doc["matched_within"]
    if m["within"] is not None:
        lines.append(f"\n  matched within {m['within']} DTI points over {m['seats_with_dti']} of "
                     f"{m['seats']} seats" + (f"; {m['seats'] - m['seats_with_dti']} without a DTI"
                                              if m["seats"] > m["seats_with_dti"] else ""))
    else:
        lines.append(f"\n  {m['seats_with_dti']} of {m['seats']} seats have a DTI — no spread to state")
    s = doc["spearman"]
    if s["verdict"] == "NOT ANSWERABLE":
        lines.append(f"  does DTI predict who dominates here? NOT ANSWERABLE at n={s['n']} — {s['why']}")
    else:
        lines.append(f"  Spearman(DTI, share of fair) = {s['rho']} over n={s['n']} "
                     f"({'clears' if s['significant'] else 'does not clear'} ±{s['critical_rho']})")
    b = doc["balance"]
    if isinstance(b, dict) and "absent_because" in b:
        lines.append(f"  balance: {b['absent_because']}")
    return "\n".join(lines)


def main(args):
    name = getattr(args, "name", None)
    if not name:
        raise SystemExit("`pods --power` needs a pod name — `manamap pilot pods` lists them")
    doc = table(name, today=date.today())
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(format_table(doc))
