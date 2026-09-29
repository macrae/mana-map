"""THE DECISION LEDGER: every decision about a deck, what the evidence said at
the moment it was taken, and — later — what actually happened.

WHY IT EXISTS. `deck_branch.propose` freezes what the pilot accepted
(`accepted_on`), and that was the whole record: nothing captured a rejection,
nothing re-measured a merged list, and nothing put the prediction beside the
outcome. Forty-two branches, nine merges, and not one line saying whether a
merge delivered what its report promised. A bench that cannot compare what it
predicted with what it got cannot know whether its instruments are any good —
and `calibrate.py`'s headline is that nothing here is validated.

`data/decks/<slug>/decisions.jsonl` is APPEND-ONLY, one JSON object per line,
the same discipline as the captain's log: a malformed line is an error, nothing
rewrites, ids are sequential. `branch.json` keeps `accepted_on` (it is what
`branch_state` reads); the ledger is the HISTORY, and it is the one place a
withdrawal or a rejection is recorded — so `withdraw` can keep leaving the
branch directory clean (no `withdrawn` key, no graveyard) and still not lose
the reason.

KINDS. `propose` / `amend` / `withdraw` / `reject` / `merge` are the branch
verbs; `experiment` is a campaign entry finishing; `adopt-policy` is a piloting
rule accepted; `outcome` is the realised figure joined to a `merge` by id.
`outcome` is computed, never authored: `decisions <slug> outcome` finds every
Forge run of the MERGED list at the predicted pod and harness, pools it, and
writes the realised rate, the realised difference against the pre-merge
champion's runs, and whether that landed inside the predicted interval. It is
idempotent by run id, and it is NOT a `regen` stage — an append-only file
cannot satisfy recompute-and-compare.
"""

import datetime
import json
import pathlib

from manamap.pilot.common import deck_dir, load_json

LEDGER = "decisions.jsonl"
KINDS = ("propose", "amend", "withdraw", "reject", "merge", "adopt-policy",
         "experiment", "outcome")
#: A kind that must say why.
NEEDS_REASON = ("withdraw", "reject")


def path(slug, base=None):
    """`base` is the deck directory when the caller already resolved it — the
    branch verbs pass theirs, so a test that points `deck_branch` at a
    temporary deck never writes a ledger into the real tree."""
    return (base or deck_dir(slug)) / LEDGER


def read(slug, base=None):
    """Every line, in order. A malformed line is an error, not a skip."""
    p = path(slug, base)
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{p}:{n}: not JSON ({exc}) — the ledger is append-only and a "
                             f"hand edit must keep one object per line")
    return out


def next_id(entries):
    return f"{max((int(e['id']) for e in entries), default=0) + 1:03d}"


def _deck_sha(slug, base=None):
    try:
        import hashlib
        text = ((base or deck_dir(slug)) / "decklist.txt").read_bytes()
        return hashlib.sha256(text).hexdigest()
    except Exception:                       # noqa: BLE001 — absent, never invented
        return None


def append(slug, kind, branch=None, at=None, base=None, **fields):
    """Append one line and return it. The deck's list sha is stamped at write.
    Refuses to invent a deck directory: the ledger lives beside a deck that exists."""
    if kind not in KINDS:
        raise SystemExit(f"decision kind {kind!r} is not one of {KINDS}")
    if kind in NEEDS_REASON and not str(fields.get("reason") or "").strip():
        raise SystemExit(f"a {kind} needs a reason — it is the one thing the branch "
                         f"directory will not keep")
    p = path(slug, base)
    if not p.parent.is_dir():
        raise SystemExit(f"{slug}: no deck directory at {p.parent} to keep a ledger in")
    entries = read(slug, base)
    entry = {"id": next_id(entries),
             "at": at or datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "kind": kind, "branch": branch,
             "deck_decklist_sha256": _deck_sha(slug, base)}
    entry.update({k: v for k, v in fields.items() if v is not None})
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def latest_for(slug, branch, base=None):
    """The most recent line about one branch, or None."""
    rows = [e for e in read(slug, base) if e.get("branch") == branch]
    return rows[-1] if rows else None


# ── the prediction: what the report said when the decision was taken ──────────

def prediction_from(nc):
    """The frozen prediction a `propose` / `merge` line carries, read from the
    branch's `net_change.json` at that moment. A superset of `accepted_on`:
    the objective and its grade, the harness, and — where the real table was
    measured — the Forge block's delta, interval, MDE, null and run ids."""
    if not nc:
        return None
    grade = nc.get("objective_grade") or {}
    f = nc.get("forge") or {}
    pred = {"endpoint": (nc.get("objective") or {}).get("axis"),
            "objective": nc.get("objective"),
            "grade": grade.get("state"), "reading": grade.get("reading"),
            "recommendation": (nc.get("recommendation") or {}).get("state"),
            "harness": nc.get("harness"),
            "measured_decklist_sha256": nc.get("decklist_sha256")}
    if f.get("available"):
        wr = (f.get("endpoints") or {}).get("forge.win_rate") or {}
        pred["forge"] = {
            "pod": f.get("pod"), "card_overrides": f.get("card_overrides"),
            "ai_profile": f.get("ai_profile"),
            "champion": f.get("champion"), "branch": f.get("branch"),
            "delta": f.get("delta"), "ci95": f.get("ci95"),
            "excludes_zero": f.get("excludes_zero"), "mde": f.get("mde"),
            "null": (f.get("null") or {}).get("rate"),
            "run_ids": f.get("run_ids"),
            "endpoints": {k: {"delta": v.get("delta"), "ci95": v.get("ci95"), "mde": v.get("mde")}
                          for k, v in (f.get("endpoints") or {}).items()
                          if v.get("delta") is not None} or None}
        if wr.get("delta") is not None and "delta" not in pred["forge"]:
            pred["forge"]["delta"] = wr["delta"]
    else:
        pred["forge"] = None
        pred["forge_why"] = f.get("why")
    return pred


# ── outcomes: what actually happened to a merged list ─────────────────────────

def _runs_of(slug, sha, pod, overrides, profile):
    """Deck-level run records and experiment arms that PLAYED `sha` at `pod`
    under the same harness — the population a realised figure is read from."""
    import glob
    from manamap.pilot import net_change
    base = deck_dir(slug)
    out = []
    for p in sorted(glob.glob(str(base / "sim" / "*.json"))):
        doc = load_json(pathlib.Path(p)) or {}
        rpod = doc.get("pod")
        rpod = (rpod.get("name") if isinstance(rpod, dict) else rpod)
        if rpod != pod:
            continue
        ov = (doc.get("card_overrides") or {}).get("sha") or None
        prof = ((doc.get("profiles") or ["Default"])[0]) or "Default"
        if ov != overrides or prof != (profile or "Default"):
            continue
        if net_change._seat_sha(doc, slug) != sha:
            continue
        a = doc.get("analysis") or {}
        seat = (a.get("seats") or {}).get(slug) or {}
        if seat.get("wins") is None:
            continue
        out.append({"run_id": doc.get("run_id"), "wins": seat["wins"],
                    "decided": a.get("decided", a.get("games") or 0), "games": a.get("games")})
    return out


def _pool(rows):
    w = sum(r["wins"] for r in rows)
    n = sum(r["decided"] for r in rows)
    return w, n


def outcome(slug, write=True):
    """For every `merge` without an `outcome`, look for runs of the merged list
    at the predicted pod and harness and record what they read. Returns the
    lines appended (or that would be)."""
    from manamap.sim import stats
    from manamap.pilot.deck_notes import read_log
    entries = read(slug)
    have = {e.get("of") for e in entries if e.get("kind") == "outcome"}
    added = []
    for m in [e for e in entries if e.get("kind") == "merge"]:
        if m["id"] in have:
            continue
        sha = m.get("decklist_sha256")
        pred = (m.get("prediction") or {})
        fp = pred.get("forge") or {}
        pod = fp.get("pod")
        if not (sha and pod):
            continue                    # no real-table prediction to close against
        rows = _runs_of(slug, sha, pod, fp.get("card_overrides"), fp.get("ai_profile"))
        if not rows:
            continue                    # nothing has measured the merged list yet
        w, n = _pool(rows)
        lo, hi = stats.wilson_bounds(w, n)
        realised = {"pod": pod, "wins": w, "decided": n, "rate": round(w / n, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "run_ids": [r["run_id"] for r in rows]}
        # THE REALISED DIFFERENCE, against the pre-merge champion's own runs —
        # the ones the prediction was scaled from.
        champ = fp.get("champion") or {}
        if champ.get("wins") is not None and champ.get("games"):
            d = stats.diff_proportions(champ["wins"], champ["games"], w, n)
            realised["difference"] = {"delta": d["diff"], "ci95": d["ci95"],
                                      "excludes_zero": d["excludes_zero"],
                                      "against": "the champion runs the prediction was read from"}
            ci = fp.get("ci95")
            inside = bool(ci and ci[0] <= d["diff"] <= ci[1])
        else:
            inside = None
        # THE REAL TABLE, joined by sha and shown beside — never pooled in.
        paper = [e for e in read_log(slug) if e.get("decklist_sha256") == sha]
        if paper:
            realised["paper"] = {"games": len(paper),
                                 "win": sum(1 for e in paper if e.get("result") == "win"),
                                 "loss": sum(1 for e in paper if e.get("result") == "loss"),
                                 "draw": sum(1 for e in paper if e.get("result") == "draw")}
        line = {"of": m["id"], "realised": realised,
                "predicted": {"delta": fp.get("delta"), "ci95": fp.get("ci95"),
                              "mde": fp.get("mde"), "null": fp.get("null")},
                "inside_prediction": inside}
        if write:
            added.append(append(slug, "outcome", branch=m.get("branch"), **line))
        else:
            added.append(dict(line, kind="outcome"))
    return added


def awaiting(slug):
    """Merges with no outcome yet, and whether a qualifying run exists."""
    entries = read(slug)
    have = {e.get("of") for e in entries if e.get("kind") == "outcome"}
    out = []
    for m in [e for e in entries if e.get("kind") == "merge" and e["id"] not in have]:
        fp = (m.get("prediction") or {}).get("forge") or {}
        pod = fp.get("pod")
        rows = _runs_of(slug, m.get("decklist_sha256"), pod, fp.get("card_overrides"),
                        fp.get("ai_profile")) if (pod and m.get("decklist_sha256")) else []
        out.append({"id": m["id"], "branch": m.get("branch"), "pod": pod,
                    "runs_of_merged_list": len(rows),
                    "closable": bool(rows)})
    return out


# ── backfill: the six proposals and nine merges that predate the ledger ───────

def backfill(slug):
    """Seed the ledger from `branch.json` files so the registry names an
    artifact that exists and every merge can get an outcome. Idempotent:
    a line already present (same kind, branch and date) is not written twice.
    Lines are marked `backfilled: true` and may lack a prediction."""
    import glob
    from manamap.pilot import deck_branch
    existing = read(slug)
    seen = {(e.get("kind"), e.get("branch"), (e.get("at") or "")[:10]) for e in existing}
    added = []
    for p in sorted(glob.glob(str(deck_dir(slug) / "branches" / "*" / "branch.json"))):
        doc = load_json(pathlib.Path(p)) or {}
        branch = doc.get("branch")
        nc = load_json(deck_dir(slug, branch) / "net_change.json") if branch else None
        prop = doc.get("proposal")
        if prop and ("propose", branch, str(prop.get("at") or "")[:10]) not in seen:
            acc = prop.get("accepted_on") or {}
            added.append(append(slug, "propose", branch=branch, at=prop.get("at"),
                                as_version=prop.get("as_version"),
                                reason=prop.get("why") or None,
                                forced_reason=prop.get("forced_reason"),
                                branch_decklist_sha256=prop.get("decklist_sha256"),
                                prediction={"endpoint": (acc.get("objective") or {}).get("axis"),
                                            "objective": acc.get("objective"),
                                            "grade": (acc.get("grade") or {}).get("state")
                                            if isinstance(acc.get("grade"), dict) else acc.get("grade"),
                                            "reading": acc.get("reading"),
                                            "recommendation": acc.get("state"),
                                            "harness": acc.get("harness"),
                                            "measured_decklist_sha256": acc.get("decklist_sha256"),
                                            "forge": None,
                                            "forge_why": "backfilled from accepted_on, which "
                                                         "carried no Forge block"},
                                backfilled=True))
        merged = doc.get("merged")
        if merged and ("merge", branch, str(merged.get("at") or "")[:10]) not in seen:
            pred = prediction_from(nc) if nc and nc.get("decklist_sha256") == merged.get("decklist_sha256") else None
            added.append(append(slug, "merge", branch=branch, at=merged.get("at"),
                                decklist_sha256=merged.get("decklist_sha256"),
                                into_version_before=merged.get("into_version_before"),
                                forced_reason=merged.get("forced_reason"),
                                prediction=pred,
                                prediction_note=(None if pred else
                                                 "net_change.json on disk describes a later "
                                                 "list than the one merged, so no prediction "
                                                 "is attributable to this merge"),
                                backfilled=True))
    return added


# ── CLI ───────────────────────────────────────────────────────────────────────

def _fmt(e):
    head = f"  {e['id']}  {str(e.get('at') or '')[:10]}  {e['kind']:12} {e.get('branch') or '—'}"
    bits = []
    if e.get("as_version"):
        bits.append(f"as {e['as_version']}")
    p = e.get("prediction") or {}
    if p.get("endpoint"):
        bits.append(f"{p['endpoint']} {p.get('grade')}")
    fp = p.get("forge") or {}
    if fp.get("delta") is not None:
        ci = fp.get("ci95") or [None, None]
        bits.append(f"forge Δ {fp['delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] at {fp.get('pod')}")
    if e.get("kind") == "outcome":
        r = e.get("realised") or {}
        bits.append(f"realised {r.get('wins')}/{r.get('decided')} = {r.get('rate')} at {r.get('pod')}")
        d = r.get("difference") or {}
        if d.get("delta") is not None:
            bits.append(f"Δ {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
                        + (" INSIDE the prediction" if e.get("inside_prediction")
                           else " OUTSIDE the prediction" if e.get("inside_prediction") is False
                           else ""))
        if r.get("paper"):
            pp = r["paper"]
            bits.append(f"paper {pp['win']}-{pp['loss']}-{pp['draw']} over {pp['games']}")
    if e.get("reason"):
        bits.append(f"because {e['reason'][:70]}")
    if e.get("forced_reason"):
        bits.append(f"FORCED: {e['forced_reason'][:60]}")
    if e.get("backfilled"):
        bits.append("(backfilled)")
    return head + ("\n           " + " · ".join(bits) if bits else "")


def main(args):
    slug = args.slug
    action = getattr(args, "action", None) or "list"
    if action == "backfill":
        added = backfill(slug)
        print(f"{slug}: {len(added)} line(s) backfilled from branch.json")
        for e in added:
            print(_fmt(e))
        return
    if action == "outcome":
        added = outcome(slug, write=not getattr(args, "dry_run", False))
        if not added:
            waiting = awaiting(slug)
            print(f"{slug}: nothing to close — "
                  + (f"{len(waiting)} merge(s) await a run of the merged list at their pod"
                     if waiting else "no merge awaits an outcome"))
            for w in waiting:
                print(f"    {w['id']} {w['branch']}: {w['runs_of_merged_list']} run(s) of the "
                      f"merged list at {w['pod']}"
                      + ("" if w["pod"] else " — no real-table prediction to close against"))
            return
        print(f"{slug}: {len(added)} outcome(s) recorded")
        for e in added:
            print(_fmt(e))
        return
    if action == "adopt-policy":
        e = append(slug, "adopt-policy", branch=None,
                   experiment_id=getattr(args, "from_experiment", None),
                   reason=getattr(args, "reason", None))
        print(_fmt(e))
        return
    entries = read(slug)
    if action == "show":
        wanted = getattr(args, "entry_id", None)
        e = next((x for x in entries if x["id"] == wanted), None)
        if not e:
            raise SystemExit(f"{slug}: no decision {wanted!r}")
        print(json.dumps(e, indent=2, ensure_ascii=False))
        return
    if getattr(args, "as_json", False):
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return
    if not entries:
        print(f"{slug}: no decisions recorded — `decisions {slug} backfill` seeds it from "
              f"branch.json; `deck-branch propose|withdraw|reject|merge` write it from now on")
        return
    print(f"DECISIONS — {slug} ({len(entries)})")
    for e in entries:
        print(_fmt(e))
    waiting = [w for w in awaiting(slug) if w["closable"]]
    if waiting:
        print(f"\n  {len(waiting)} merge(s) can be closed: `manamap pilot decisions {slug} outcome`")
