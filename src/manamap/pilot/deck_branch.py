"""Pilot: a candidate 99, measured and decided before it is the deck (`branches/<name>/`).

    manamap pilot deck-branch <slug> list | new | show | diff
                                     | stage | unstage | commit | log
                                     | propose | withdraw | reject | merge | delete

MAP OF THIS FILE (in order):
  BRANCH_FILE / NAME_RE  the on-disk names.
  names / meta / diff  the branch's list read in the deck's vocabulary (DFC faces
                    canonicalised); `diff` is `add` / `out` against the deck.
  report / review / one  one branch summarised for `list`, `info.json` and the
                    web roster: the study, the instrument's read, the diff.
  parse_objective / grade_objective  `"kill_by_8 >= 0.30"` or `forge.<axis> … @<pod>`
                    into {axis, op, value}; met / not met / not resolvable.
  champion_reading / _band_caveat  the anchor printed when a line is chosen.
  new / swap_entries / stage / unstage / commit / log  open a branch, stage one
                    swap at a time (keep list honoured), freeze a list.
  _validate_after_merge  what no longer validates once the list changed.
  BRANCH_STATES / branch_state  six derived states (OPEN .. REJECTED), never stored.
  propose / withdraw / reject / delete / merge  the merge-request stage, the
                    ledger verbs, and the write of `decklist.txt`.
  ── CLI ──          `_dispatch` parses the action; `main` runs it and, for the
                    verbs in MUTATES_THE_DOSSIER, rewrites `info.json` on any
                    live deck that has one (`refresh_dossier`).

THE GAP THIS FILLS. A deck had exactly two states: the list in `decklist.txt`,
and nothing. `decklist.txt` is tracked, so writing it MINTS A VERSION, and the
captain's log stamps games against versions — so a refactor the pilot has not
yet put in the sleeves had nowhere to live. The first one, a 34-out/35-in
Ur-Dragon treasure rebuild, was designed, measured, briefed and unappliable
(and, once measured, abandoned — which is the loop working).

A BRANCH IS A WHOLE LIST, NOT A QUEUE OF SWAPS. `pending.json` already holds
decided-but-unapplied in/out pairs and is right for a three-land swap decided on
a Tuesday. It is the wrong shape here for two reasons: 35 swaps written as pairs
is unreadable, and — the load-bearing one — **you can goldfish a list and you
cannot goldfish a swap queue**. Being measurable is the whole point.

WHY NOT A GIT BRANCH. Versions derive from `git log` over `decklist.txt`, so a
git branch would mint version numbers that never shipped, and checking one out
would move every OTHER deck's artifacts with it.

NO OWNERSHIP (2026-11-22). This module used to answer where every card would come
from — in the deck, in a box, sleeved in another deck, or a purchase — and refused
a merge until all were "sourced". The pilot ruled the bench has no notion of card
ownership: a branch is a list and a decision; whether the cards are in the
sleeves is his word, said at merge. The paper lock (`deck_versions.paper`) and
the pinned printings on a checked-in deck are the only physical facts kept.

THE THREE STEPS, AND WHY THERE ARE THREE. A COMMIT freezes one exact list with a
message — it says "this is the deck I am committed to running". A PROPOSAL says
"this is the deck's next version": it names the version the list means to
become, rests on an approved study, freezes the instrument's read beside it, and
is the merge-request stage. A MERGE says "this is the deck" and rewrites
`decklist.txt`, on the pilot's word that the list is in the sleeves.

EVERY STATE IS DERIVED AND NONE IS STORED — see `branch_state`. `validate_pending`
earned that rule: "closure is DERIVED, never declared … a hand-set flag is
precisely how HISTORY.md became append-only and append-forgotten."

MERGE IS THE MOMENT THE DECK CHANGES. It
reuses `check_in.analyze`'s blocking checks verbatim rather than restating them,
writes `decklist.txt`, and prints the commit command WITHOUT committing — the
commit is what `deck-version` numbers and what the log stamps games against, so
it stays the pilot's act. It never touches the `paper` block: three placeholder
locks were once withdrawn for being written to demonstrate machinery, and a
merge that silently claimed cardboard would be that mistake with a command
attached.
"""

import contextlib
import io
import datetime
import json
import hashlib
import re
import shutil

from manamap.pilot import check_in
from manamap.pilot import common as _common
from manamap.pilot.common import (
    BRANCHES_DIR,
    deck_dir,
    decklist_sha256,
    deck_lifecycle,
    expand_faces,
    load_json,
)
from manamap.pilot.deck_history import _entries

BRANCH_FILE = "branch.json"
#: A branch name is a path segment and a git-visible directory, so keep it to
#: what a shell and a filesystem both treat as one word.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,48}$")

#: NO SOURCING STATES (2026-11-22). The bench has no notion of card ownership:
#: a branch is a list and a decision, never a shopping trip. The paper lock
#: (`deck_versions.paper`) and pinned printings are the only physical facts kept.


def branch_root(slug):
    return deck_dir(slug) / BRANCHES_DIR


def names(slug):
    root = branch_root(slug)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "decklist.txt").exists())


def _list_text(slug, branch=None):
    return (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8")


def _canonical(slug, branch=None):
    """decklist name -> the name every other artifact uses.

    A DOUBLE-FACED CARD IS NAMED TWO WAYS AND BOTH ARE CORRECT. `cards.json` and
    everything derived from it (deck_map, the graph) key the joined `A // B`
    form; a decklist names whichever face the writer had in front of them, and
    for transform cards it HAS to — Scryfall answers the joined name with a 404
    and resolves the front face alone, so a decklist carrying `A // B` cannot be
    fetched at all.

    Left unreconciled the seam is silent and small: the roster joined
    `deck_map`'s names to this module's and lost exactly two cards, under-marking
    the additions 30 against 32 and the buy list 19 against 21. Nothing errors —
    two cards simply render unmarked.
    """
    doc = load_json(deck_dir(slug, branch) / "cards.json")
    if not doc:
        return {}
    out = {}
    for c in (doc.get("cards") or []):
        for face in expand_faces(c.get("name")):
            out[face] = c["name"]
    return out


def meta(slug, branch):
    return load_json(deck_dir(slug, branch) / BRANCH_FILE) or {}


def _named(slug, branch, text_slug, text_branch):
    """Both lists in one vocabulary, preferring what the resolver settled on."""
    canon = dict(_canonical(slug))
    canon.update(_canonical(slug, branch))
    def fix(entries):
        out = {}
        for n, q in entries.items():
            out[canon.get(n, n)] = out.get(canon.get(n, n), 0) + q
        return out
    return fix(text_slug), fix(text_branch)


def diff(slug, branch):
    """What the branch adds and drops against the deck's current list.

    Named for the hands rather than the diff, the same way `paper_state` is:
    `add` is what goes in, `out` is what comes out.
    """
    now, cand = _named(slug, branch,
                       _entries(_list_text(slug)), _entries(_list_text(slug, branch)))
    # COPIES and NAMES are different numbers and the repo has been bitten by
    # conflating them: `size` is what the shuffler sees (36 basics are 36 cards),
    # `names` is what you have to source (36 Forests are one line on a buy list).
    #
    # AND THE COMPARISON ITSELF MUST COUNT COPIES, which for a long time it did
    # not. `add`/`out` test NAME PRESENCE, so a name whose count merely MOVES is
    # in neither list: ur-dragon/final-v1 cut a Forest (2 -> 1) and a Mountain
    # (4 -> 3) alongside its twelve named cuts and published "+14 -12" on two
    # 100-card lists, which cannot be true and is the first thing a reader
    # notices. The named lists stay name-shaped — that is what a buy list and a
    # per-card row want — and the copy arithmetic moves into its own keys.
    #
    # `quantity` is the third kind of change, and it is the one with no name to
    # print on either side: same card, different count. It is reported rather
    # than folded into `add`/`out`, because a name in both lists at once would
    # break every consumer that pairs them.
    add = sorted(n for n in cand if n not in now)
    out = sorted(n for n in now if n not in cand)
    quantity = [{"name": n, "from": now[n], "to": cand[n]}
                for n in sorted(set(now) & set(cand)) if now[n] != cand[n]]
    return {"add": add, "out": out, "quantity": quantity,
            # THE COUNTS THAT BALANCE. On two lists of equal size these are
            # equal, and a test asserts exactly that.
            "add_copies": (sum(cand[n] for n in add)
                           + sum(q["to"] - q["from"] for q in quantity
                                 if q["to"] > q["from"])),
            "out_copies": (sum(now[n] for n in out)
                           + sum(q["from"] - q["to"] for q in quantity
                                 if q["from"] > q["to"])),
            "size": sum(cand.values()), "base_size": sum(now.values()),
            "names": len(cand), "base_names": len(now)}


def report(slug, branch=None):
    if branch:
        return {"slug": slug, "branches": [one(slug, branch)]}
    return {"slug": slug, "branches": [one(slug, b) for b in names(slug)]}


def review(slug, branch, doc=None, d=None):
    """What a reader needs to CHOOSE a branch to open, read off the branch's own
    files and never recomputed (2026-10-19): the objective and its grade, the
    net-change verdict with what rose and fell, whether the report describes the
    branch's current list, what Forge's AI did with each add, and the stacks
    whose scenario names an add. Absent files are absent keys, never zero.

    The pilot could not find a branch from the workbench, and the deck page's
    list showed every branch alike — no state for an open one, no objective, no
    verdict — so the one live branch sat among eleven stale experiments with
    nothing to tell them apart. These are the fields that tell them apart.
    """
    m = doc if doc is not None else meta(slug, branch)
    d = d if d is not None else diff(slug, branch)
    out = {}
    # THE STUDY FIRST (2026-11-21): the latest decided question about this branch —
    # the call, who decided, the pilot's words — is what a reader chooses on; the
    # instrument's read follows it.
    latest = latest_study(slug, branch)
    if latest:
        out["study"] = latest
    obj = m.get("objective") or {}
    if obj.get("axis"):
        out["objective"] = {k: obj.get(k) for k in ("axis", "op", "value", "pod") if obj.get(k) is not None}
    if isinstance(m.get("summary"), dict) and m["summary"].get("title"):
        out["summary"] = {k: m["summary"].get(k) for k in ("title", "blurb", "by", "at")}
    nc = load_json(deck_dir(slug, branch) / "net_change.json")
    if nc:
        rec = nc.get("recommendation") or {}
        from manamap.pilot.net_change import state_word  # local: net_change imports this module
        out["verdict"] = {"state": rec.get("state"), "word": state_word(rec),
                          "rose": len(rec.get("rose") or []), "fell": len(rec.get("fell") or []),
                          "no_call": len(rec.get("no_call") or [])}
        if nc.get("objective_grade"):
            out["objective_grade"] = {k: nc["objective_grade"].get(k) for k in ("state", "reading", "shortfall")}
        current = decklist_sha256(slug, branch)
        out["measured"] = bool(current) and nc.get("decklist_sha256") == current
        if not out["measured"]:
            out["measured_why"] = "net_change.json describes an older list of this branch"
        out["blind_spots"] = sorted({c for b in (nc.get("blind_spots") or []) if isinstance(b, dict)
                                     for c in (b.get("cards") or [])}) or None
        # THE MEASURED IMPACT, for the workbench (2026-10-20): the verdict with
        # the rows that rose and fell BY NAME, and the headline rows copied
        # verbatim from the table — delta, interval, MDE, verdict — so the
        # landing page shows a figure without fetching every branch's report.
        # Only `IMPACT_MEASURES` rows, in that order; a row the table lacks is
        # absent, never zero. The page renders this; it computes nothing.
        from manamap.pilot.net_change import IMPACT_MEASURES  # local: net_change imports this module
        rows = {str(r.get("measure")): r for r in (nc.get("table") or []) if isinstance(r, dict)}
        headline = [{"measure": m, "delta": rows[m].get("delta"), "ci95_diff": rows[m].get("ci95_diff"),
                     "mde": rows[m].get("mde"), "verdict": rows[m].get("verdict"),
                     "unit": rows[m].get("unit"), "better_is": rows[m].get("better_is")}
                    for m in IMPACT_MEASURES if m in rows]
        out["impact"] = {"state": rec.get("state"), "rose": list(rec.get("rose") or []),
                         "fell": list(rec.get("fell") or []), "headline": headline}
    cp = load_json(deck_dir(slug, branch) / "cast_proofs.json")
    if cp and isinstance(cp.get("cards"), dict):
        counts = {"played": 0, "held": 0, "unplayed": 0}
        for row in cp["cards"].values():
            v = str((row or {}).get("verdict") or "").upper()
            for k in counts:
                if v.startswith(k.upper()):
                    counts[k] += 1
                    break
        out["cast_proofs"] = {**counts, "as_of": cp.get("as_of"), "harness": cp.get("harness")}
    # ONLY THE CARDS THE DECK DOES NOT ALREADY RUN. A branch whose base moved
    # reads half the deck as "added", and every stack the deck owns would then
    # count for it; a stack is proven ON THIS BRANCH when it names a card only
    # the branch brings (012 names Enduring Tenacity, a mean-v1 add).
    try:
        in_deck = {c.get("name") for c in (load_json(deck_dir(slug) / "cards.json") or {}).get("cards") or []}
    except Exception:  # noqa: BLE001
        in_deck = set()
    adds = set(d["add"]) - in_deck
    if adds:
        hits = stacks_naming(slug, adds)
        if hits:
            out["stacks"] = hits
    return out


def study_row(doc):
    """One study as every surface summarises it: id, question, call, who decided."""
    call = doc.get("call") or {}
    cov = doc.get("goldfish_coverage") or {}
    return {"id": doc.get("id"), "question": doc.get("question"), "branch": doc.get("branch"),
            # the file under studies/, so a page can fetch the whole study by its row
            "file": (f"{doc.get('id')}-{doc['slug_name']}.json" if doc.get("slug_name") else None),
            "decision": call.get("decision", "open"), "decided_by": list(call.get("decided_by") or []),
            "pilot_words": call.get("pilot_words"), "decided_at": doc.get("decided_at"),
            "opened": doc.get("opened"), "swap": doc.get("swap"),
            "goldfish_saw": ((len(cov.get("seen") or []) + len(cov.get("partial") or []), cov.get("named"))
                             if cov.get("named") else None)}


def studies_of(slug):
    """Every study of a deck as rows, newest first; [] when there are none."""
    try:
        from manamap.pilot import study as st
        rows = [study_row(d) for d in st.list_studies(slug)]
    except Exception:                                   # noqa: BLE001 — no deck tree, no studies
        return []
    rows.sort(key=lambda r: (r.get("decided_at") or r.get("opened") or "", r.get("id") or ""), reverse=True)
    return rows


def latest_study(slug, branch):
    """The newest study about one branch, decided ones first; None when none."""
    rows = [r for r in studies_of(slug) if r.get("branch") == branch]
    if not rows:
        return None
    decided = [r for r in rows if r["decision"] != "open"]
    return (decided or rows)[0]


def stacks_naming(slug, names):
    """The deck's stack scenarios whose SCENARIO names any of `names`, with the
    checker's verdict: `[{id, title, checker, names}]`, in file order.

    ONE SCAN, TWO READERS. `review` asks it about a branch's adds; `card_dossier`
    asks it about one card. A second hand-rolled glob would have been the second
    copy of the "does a stack name this card" rule, and the first was already
    subtle (the scenario only — never the resolution prose, which names cards
    the board does not hold)."""
    names = [n for n in names if n]
    if not names:
        return []
    hits = []
    stacks = deck_dir(slug) / "stacks"
    if not stacks.is_dir():
        return hits
    for path in sorted(stacks.glob("*.json")):
        sd = load_json(path) or {}
        hay = json.dumps(sd.get("scenario") or {}, ensure_ascii=False).lower()
        named = sorted(n for n in names if n.lower() in hay)
        if named:
            hits.append({"id": sd.get("id") or path.name[:3], "title": sd.get("title"),
                         "checker": ((sd.get("checker") or {}).get("verdict")), "names": named})
    return hits


def one(slug, branch):
    """One branch, summarised for `list`, `info.json` and the web roster.

    PUBLIC because it has been called across a module boundary since it shipped —
    `deck_info._branches` reaches for it by its underscore name. A private helper
    that two modules depend on is a contract wearing a disclaimer.
    """
    m = meta(slug, branch)
    d = diff(slug, branch)
    add = set(d["add"])
    state, state_why = branch_state(slug, branch, doc=m)
    # Present only on a branch that recorded its base list (opened since
    # 2026-10-10): absent is "not recorded", None is "not moved".
    moved = ({"base_moved": base_moved(slug, branch, doc=m, state=state)}
             if m.get("base_sha256") else {})
    _now, cand = _named(slug, branch, _entries(_list_text(slug)), _entries(_list_text(slug, branch)))
    return {"name": branch, "opened": m.get("opened"), "why": m.get("why"),
            "state": state, "state_why": state_why, **moved,
            "proposal": m.get("proposal"),
            "base_version": m.get("base_version"),
            # COPIES, not names — `+14 -12` on two 100-card lists was this
            # number reading `len(add)` while two basic-land cuts had no name
            # to appear under.
            "size": d["size"], "add": d["add_copies"], "out": d["out_copies"],
            # Every card of the candidate list, and whether the branch brings it
            # (2026-11-22: no provenance — the bench does not know where a card is).
            "cards": [{"name": n, "is_new": n in add} for n in sorted(cand)],
            "has_cards": (deck_dir(slug, branch) / "cards.json").exists(),
            **review(slug, branch, doc=m, d=d)}


#: A BRANCH THAT CANNOT BE FALSIFIED GETS GRADED ON WHETHER IT DID WHAT IT DOES.
#: The Ur-Dragon treasure branch said "treasure is the engine" and achieved that —
#: 4.4x the hoard — while missing the purpose nobody wrote down, which was winning.
#: It was measured for a week before anyone could say it had failed, because there
#: was nothing it could fail against.
#:
#: So an objective names a MEASURE, a DIRECTION and a NUMBER, and the measure has to
#: be one the bench already computes: the vocabulary is `candidates.AXES`, which is a
#: registry, is independence-checked fleet-wide by `tests/test_metric_hygiene.py`, and
#: cannot silently grow a second name for the same thing.
_OBJECTIVE_RE = re.compile(
    r"^\s*((?:forge\.)?[a-z_0-9]+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)"
    r"\s*(?:@([a-z0-9][a-z0-9-]*))?\s*$")

_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}


#: AXES A BRANCH MAY NOT AIM AT, and the refusal is the most expensive lesson on
#: this page. A membership axis asks whether the parts named in
#: `goldfish_targets.json` were DRAWN — and that file is authored. The pilot who
#: sets the objective also writes the declaration it is graded against, so the
#: axis can be moved without touching a single card.
#:
#: MEASURED ON ONE LIST, ONE SEED, 10,000 GAMES. Three defensible declarations of
#: the same Ur-Dragon deck, each graded by the engine lift against kill-by-T8:
#:
#:   ramp + a loosely worded payoff        lift +0.007  [-0.003, +0.017]  spans zero
#:   discount + ramp + burn, all required  lift -0.036  [-0.052, -0.020]  REAL
#:   ramp + burn                           lift +0.014  [+0.005, +0.023]  REAL
#:
#: Same cards, same games. The middle row says assembling the engine makes the
#: deck WIN LESS, at an interval excluding zero, and it is not wrong — a second
#: cost reducer is a card and three mana that deals no damage when eminence
#: already pays the discount for free. But an axis whose SIGN a JSON edit can
#: flip is not something a spending decision may rest on.
#:
#: The declaration stays: it is a good description of a deck and it drives the
#: `*_assisted` figures and the target table, which are hypergeometric and real.
#: What it may not do is be the thing a branch is graded on. Aim at an OUTPUT —
#: damage, a clock, board power, the hoard — which no wording moves. The lift
#: itself was deleted from `net-change` the same day, for the same reason.
MEMBERSHIP_AXES = ("engine_online_3", "engine_online_5", "engine_online_8",
                   "any_route_8")


def parse_objective(text):
    """`"kill_by_8 >= 0.30"` -> {axis, op, value}. Raises with the vocabulary."""
    from manamap.pilot import candidates
    m = _OBJECTIVE_RE.match(text or "")
    if not m:
        raise SystemExit(
            f"--objective must read `<measure> <op> <number>`, e.g. "
            f'"kill_by_8 >= 0.30". Got: {text!r}')
    axis, op, value, pod = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    # THE REAL TABLE IN THE GRAMMAR. `forge.win_rate >= 0.25 @standard-v3` is
    # a claim about the list AT A TABLE, so the pod is part of it: a Forge axis
    # with no pod is refused, and a goldfish axis with one is refused too — the
    # goldfish has no table for the name to mean anything.
    if axis in candidates.FORGE_OBJECTIVE_AXES:
        if not pod:
            raise SystemExit(
                f"'{axis}' is measured at a TABLE, and the objective must name it: "
                f'"{axis} {op} {value:g} @standard-v3". A pod\'s null is a property '
                f"of the table with the subject in it, so the same rate means "
                f"different things at different pods (`manamap pilot pods`).")
        from manamap.sim import pods
        try:
            pods.load(pod)
        except Exception as exc:                  # PodError, or a bad file
            raise SystemExit(f"objective names pod {pod!r}: {exc}")
        return {"axis": axis, "op": op, "value": value, "pod": pod}
    if pod:
        raise SystemExit(
            f"'{axis}' is a goldfish axis and has no table — drop `@{pod}`, or aim "
            f"at a Forge endpoint: " + ", ".join(sorted(candidates.FORGE_OBJECTIVE_AXES)))
    if axis not in candidates.OBJECTIVE_AXES:
        raise SystemExit(
            f"'{axis}' is not something the bench measures. Pick one of:\n  "
            + "\n  ".join(sorted(candidates.OBJECTIVE_AXES))
            + "\n  " + "\n  ".join(f"{a} … @<pod>" for a in sorted(candidates.FORGE_OBJECTIVE_AXES))
            + "\n(`candidates.OBJECTIVE_AXES` — wider than what a SWEEP may rank "
              "on, because a correlated measure is a fine thing to aim at and a "
              "useless thing to sort by; `FORGE_OBJECTIVE_AXES` need a pod.)")
    if axis in MEMBERSHIP_AXES:
        raise SystemExit(
            f"'{axis}' asks whether the parts named in goldfish_targets.json "
            f"were DRAWN, and that file is authored — the same hand writes the "
            f"declaration and the objective, so this axis can be moved without "
            f"touching a card. Measured on ur-dragon: three defensible "
            f"declarations of one list gave lifts of +0.007 (spans zero), "
            f"-0.036 (REAL) and +0.014 (REAL) over the same 10,000 games.\n"
            f"Aim at an OUTPUT instead, which no wording moves:\n  "
            + "\n  ".join(sorted(a for a in candidates.OBJECTIVE_AXES
                                  if a not in MEMBERSHIP_AXES)))
    return {"axis": axis, "op": op, "value": value}


def grade_objective(objective, reading, mde=None, difference=None, null=None,
                    why_unmeasured=None):
    """met / not met / not resolvable, given a measured value.

    NOT RESOLVABLE IS A THIRD STATE AND IT IS THE HONEST ONE. A reading that
    misses by less than the run could detect has not failed — the run could not
    see it. Reporting that as "not met" is the same error as reporting a null as
    a finding, which this repo refuses in three other places.

    `difference` and `null` are copied through untouched: on a Forge axis the
    grade travels with the interval on the champion-to-branch difference and
    the table's null, so the number it was graded against is on the page.
    """
    if reading is None:
        return {"state": "not measured",
                "why": why_unmeasured or "the axis has no reading on this list"}
    hit = _OPS[objective["op"]](reading, objective["value"])
    miss = abs(reading - objective["value"])
    extra = {}
    if difference is not None:
        extra["difference"] = difference
    if null is not None:
        extra["null"] = null
    if not hit and mde is not None and miss < mde:
        return {"state": "not resolvable", "reading": reading, "shortfall": round(miss, 4),
                "why": (f"missed by {miss:.4f}, which is under the {mde:.4f} this run "
                        f"could detect — evidence of nothing, not evidence of failure"),
                **extra}
    return {"state": "met" if hit else "not met", "reading": reading,
            "shortfall": None if hit else round(miss, 4), **extra}


def champion_reading(slug, objective):
    """What the CHAMPION currently reads on this objective's axis, through the
    harness that will grade the branch — `net_change`'s own diagnostic, at its
    own seed, not `goldfish`'s.

    Returns lines to print. Never raises: a missing corpus or an axis the
    diagnostic has no cell for must not stop a branch being opened, so it says
    so and moves on.
    """
    from manamap.pilot import candidates, diagnostic, net_change
    axis = objective["axis"]
    if axis in candidates.FORGE_OBJECTIVE_AXES:
        return _champion_forge_reading(slug, objective)
    block, key, turn = candidates.OBJECTIVE_AXES.get(axis, (None, None, None))
    if not block:
        return [f"  champion: no diagnostic cell for {axis!r} — cannot anchor this line"]
    try:
        # `diagnostic.HARNESS` is where `net_change.build` takes both from, so
        # this is the same run the grading will make — not a near-miss of it.
        doc = diagnostic.run(slug, iterations=diagnostic.HARNESS["iterations"],
                             seed=diagnostic.HARNESS["seed"], quiet=True)
        cell = net_change._cell(doc, block, key, turn)
    except Exception as exc:                      # pragma: no cover - defensive
        return [f"  champion: could not be read ({exc.__class__.__name__}) — "
                f"the line is unanchored"]
    if not cell:
        needs = candidates.OBJECTIVE_NEEDS.get(axis)
        return [f"  champion: {axis} is absent from this deck's diagnostic — "
                f"the line is unanchored"
                + (f" (it needs \"{needs}\": true in goldfish_targets.json)" if needs else "")]
    now, mde = cell["rate"], diagnostic.mde(cell)
    gap = objective["value"] - now
    lines = [f"  champion:  {axis} reads {now:.4f} right now "
             f"(same harness, same seed as the grading run)"]
    if mde:
        lines.append(f"             this run can resolve {mde:.4f}; your line asks "
                     f"for a move of {gap:+.4f}")
        if abs(gap) < mde:
            lines.append("             THE LINE IS INSIDE THE NOISE — it cannot "
                         "come back MET or NOT MET on evidence")
    lines += _band_caveat(slug, axis, now)
    return lines


def _champion_forge_reading(slug, objective):
    """The anchor for a Forge objective: what the champion reads at that pod
    today, the table's null, and the MDE at the champion's decided count —
    printed where the line is chosen, for the same reason the goldfish anchor is."""
    from manamap.pilot import net_change
    from manamap.sim import power, stats
    axis, pod = objective["axis"], objective["pod"]
    got = net_change.champion_at(slug, pod)
    null, n_null = power.null_rate(pod)
    lines = []
    if not got:
        lines.append(f"  champion: no Forge run of this deck at {pod} — the line is "
                     f"unanchored; `manamap pilot simulate {slug} --pod {pod} --games N`")
    else:
        ep = (got.get("endpoints") or {}).get(axis) or {}
        now = ep.get("value")
        if now is None:
            lines.append(f"  champion: {axis} has no reading at {pod} "
                         f"({got.get('games')} decided games)")
        else:
            lines.append(f"  champion:  {axis} reads {now:.4f} at {pod} "
                         f"({got['games']} decided games over {got['runs']} run(s), "
                         f"{got['label']})")
            gap = objective["value"] - now
            mde = ep.get("mde")
            if mde:
                lines.append(f"             this many games can resolve {mde:.4f} against "
                             f"an equal branch arm; your line asks for a move of {gap:+.4f}")
                if abs(gap) < mde:
                    lines.append("             THE LINE IS INSIDE THE NOISE — it cannot "
                                 "come back MET or NOT MET on evidence")
    if null is not None:
        lines.append(f"             the table's null is {null:.3f} — what our decks score "
                     f"in seat 0 at {pod}")
    return lines


def _band_caveat(slug, axis, now):
    """How much of the number you are aiming at is owed to a DECLARED ability.

    zur-enchantress declares `model_commander_animate`; the goldfish fires it
    every turn it can afford and Forge's AI fired it in FIVE of 119 games. On
    `kill_by_8` that declaration is worth 0.162 of a reading of 0.381 — so a
    branch asking for +0.05 is asking for a third of what one unfireable ability
    already supplies, and TWENTY-FOUR zur branches were graded that way before
    anybody measured the floor.

    Measured on the DIAGNOSTIC's harness, which is the one doing the grading —
    reading the committed `goldfish_metrics.json` instead would mix two seeds and
    two iteration counts into one comparison, which is how three figures in one
    session got computed against the wrong baseline.
    """
    from manamap.pilot import diagnostic, goldfish
    if axis not in goldfish.BAND_ROWS:
        return []
    try:
        band = goldfish.run(slug, iterations=diagnostic.HARNESS["iterations"],
                            seed=diagnostic.HARNESS["seed"],
                            max_turn=diagnostic.HARNESS["max_turn"],
                            quiet=True).get("commander_ability_band")
    except Exception:                             # pragma: no cover - never block
        return []
    row = (band or {}).get("rows", {}).get(axis)
    if not row:
        return []
    owed = row["owed_to_the_ability"]
    share = (f", {owed / now:.0%} of the reading" if now else "")
    return [f"             BAND: {owed:+.4f} of this axis is owed to "
            f"{', '.join(band['abilities'])}{share}.",
            f"             Forge's AI fires it far less often than this model "
            f"does. Grade against the FLOOR",
            f"             ({row['floor']:.4f}) too, or the line rewards an "
            f"assumption rather than a card."]


def new(slug, branch, text, why=None, at=None, objective=None):
    if not NAME_RE.match(branch or ""):
        raise SystemExit(
            f"'{branch}' is not a usable branch name — lowercase letters, digits, "
            f"dot, dash and underscore, starting with a letter or digit.")
    root = branch_root(slug)
    path = root / branch
    if path.exists():
        raise SystemExit(f"{path} already exists — pick another name, or edit it in place.")
    # The SAME refusals a check-in gets. A branch that cannot become a deck is
    # not worth carrying, and finding out at merge time wastes the work in
    # between.
    checked = check_in.analyze(slug, text)
    if checked["blocking"]:
        raise SystemExit("Refusing to open the branch:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    # A WHOLE LIST SKIPS `stage`, so the keep list is checked here too: every card
    # the champion runs that this list does not.
    from manamap.pilot import protected
    if protected.names(slug):
        # The MAINBOARD keeps a card; a protected card moved to the sideboard
        # has left the deck.
        kept = {e["name"] for e in checked["entries"]
                if e.get("board", "main") == "main"}
        protected.refuse(slug, [n for n in _entries(_list_text(slug)) if n not in kept],
                         "to open the branch")
    path.mkdir(parents=True)
    # `checked["entries"]` carries the sideboard too (a constructed deck's
    # fifteen; nothing on a Commander list, where `analyze` drops a pasted
    # maybeboard with a warning), so the branch's file keeps the whole list.
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")
    from manamap.pilot import deck_versions
    doc = {"slug": slug, "branch": branch,
           # v2: an objective and a commit trail. v1 files (objective absent)
           # still load — the grading section is simply absent for them, which
           # is the honest report for a branch that never stated one.
           "v": 2,
           "opened": (at or datetime.date.today().isoformat()),
           "why": why or "",
           "objective": objective,
           "commits": [],
           # What it was branched FROM, so a stale branch is visible as one.
           "base_version": deck_versions.report(slug).get("current_version"),
           # ...and the deck's LIST at that moment. A bench deck is edited in
           # place between saves, so the version alone cannot see that the deck
           # moved under the branch; `base_moved` compares this to the list.
           "base_sha256": _common.decklist_sha256(slug)}
    (path / BRANCH_FILE).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return {"path": str(path), "warnings": checked["warnings"],
            # `cards` is the mainboard count; `count_copies(entries)` would
            # add the sideboard to the size of the deck.
            "size": checked["cards"]}


def _write_meta(slug, branch, doc):
    (branch_root(slug) / branch / BRANCH_FILE).write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def _parsed(slug, branch=None):
    """The decklist as ENTRIES, not as a name->copies map.

    `deck_history._entries` is the right reader for a diff and the wrong one
    here: it collapses to `{name: copies}` and throws away `is_commander`, the
    set code, the collector number and the foil marker. Rendering from that would
    drop the `Commander:` header — the list would still be 100 cards and would no
    longer be a Commander deck — and would silently re-resolve a Secret Lair to
    its cheapest reprint on the next `fetch-deck`.
    """
    from manamap.pilot.fetch_deck import parse_mainboard
    return parse_mainboard(_list_text(slug, branch))


def _sideboard(slug, branch=None):
    """The list's sideboard entries, for the RENDER path only.

    `stage` and `unstage` edit the mainboard (`_parsed`) and write the whole
    list back through `render_decklist`; a constructed deck's fifteen would
    vanish on the first swap unless the renderer is handed them. Every
    measurement still reads the mainboard. Empty on a Commander list.
    """
    from manamap.pilot.fetch_deck import parse_sideboard
    return parse_sideboard(_list_text(slug, branch))


def _resolve_in_list(entries, name):
    """Which entry is this card, allowing for either face of a DFC.

    Same seam `_canonical` closes one function up: the library holds `A // B`
    and a pilot types whichever face they are looking at.
    """
    for e in entries:
        if e["name"] == name:
            return e
    for e in entries:
        if name in expand_faces(e["name"]) or e["name"] in expand_faces(name):
            return e
    return None


def warn_if_proposed(slug, branch, verb, doc=None):
    """Say what editing a PROPOSED branch will do, before it does it.

    `stage` and `commit` on a branch in `PROPOSED · BLOCKED` succeeded with the
    ordinary "2 swap(s) staged" line. Nothing said the branch was proposed, or
    that committing would make it read `PROPOSED · STALE` — the proposal freezes
    `decklist_sha256` and `branch_state` compares it. The pilot found out from
    `deck-branch list`, after the fact (#46).

    A LINE, NOT A REFUSAL. Iterating on a proposed list is exactly the case the
    bench is supposed to support: a proposal is a decision about a version, not
    a lock on the files. `stage` already calls `meta()`, so the state costs
    nothing to read.
    """
    doc = meta(slug, branch) if doc is None else doc
    prop = (doc or {}).get("proposal")
    if not prop:
        return
    version = prop.get("as_version")
    print(f"  NOTE: {slug}/{branch} is PROPOSED as {version}. This {verb} makes "
          f"the proposal STALE — the list will no longer be the one accepted.")
    print(f"  Re-propose when the list settles: `net-change {slug} --branch "
          f"{branch} --write`, then `deck-branch {slug} propose {branch} "
          f"--as {version} --why \"…\"` (which AMENDS rather than refusing).")


#: The commander-cut refusal, one wording for `stage`, `try` and the in-place
#: editor (`deck_edit.plan`).
COMMANDER_REFUSAL = ("{name} is the COMMANDER. Changing it is a different deck, "
                     "not a swap — open a new branch from a new list.")


def swap_entries(slug, branch, entries, out_name, in_name):
    """One copy of OUT out, one of IN in, on parsed decklist ENTRIES — in memory.

    The whole of `stage`'s swap arithmetic and its refusals (not in the list, the
    commander, the pilot's keep list, already in the list), with nothing written.
    `try` calls it to measure a swap before anything exists on disk, and `stage`
    calls it to write one, so the two can never disagree about what a swap is.
    Returns `(staged_entries, out_entry, in_entry_or_None)`.
    """
    where = f"{slug}/{branch}" if branch else slug
    out_e = _resolve_in_list(entries, out_name)
    if out_e is None:
        raise SystemExit(
            f"{out_name!r} is not in {where} — nothing to swap out. "
            f"`deck-branch {slug} diff {branch}` shows what is.")
    if out_e.get("is_commander"):
        raise SystemExit(COMMANDER_REFUSAL.format(name=out_e["name"]))
    # THE PILOT'S KEEP LIST, before anything is written (draw-v1 cut Vish Kal).
    from manamap.pilot import protected
    protected.refuse(slug, [out_e["name"]], "that swap")
    in_e = _resolve_in_list(entries, in_name)
    # A basic already in the list is not a duplicate; it is one more copy. The
    # out side has known this since the branch tool shipped (decrement, never
    # delete); the in side refused "Forest" on a list holding six of them, so a
    # land pass that wanted a seventh could not be staged (gishath/mana-v1,
    # 2026-09-16). Same predicate as the manifest's furniture list.
    from manamap.pilot.deck_manifest import _BASIC_LANDS
    in_is_basic = in_e is not None and in_e["name"].lower() in _BASIC_LANDS
    if in_e is not None and not in_is_basic:
        raise SystemExit(f"{in_name!r} is already in {where}.")

    staged_entries = []
    for e in entries:
        if e is out_e:
            # Basics carry a quantity; a singleton does not. Decrement rather
            # than delete, or swapping one Mountain would cut all of them.
            # PER ENTRY, on purpose: removing ONE copy. Not `count_copies`.
            left = int(e.get("quantity") or 1) - 1
            if left > 0:
                staged_entries.append(dict(e, quantity=left))
            continue
        if e is in_e:
            staged_entries.append(dict(e, quantity=int(e.get("quantity") or 1) + 1))
            continue
        staged_entries.append(e)
    if in_e is None:
        staged_entries.append({"name": in_name, "quantity": 1})

    return staged_entries, out_e, in_e


def stage(slug, branch, out_name, in_name, strength=None, why=None, instrument=None):
    """One card out, one card in — the staging area, and its provenance.

    `why` is the PILOT'S reason, in words; `instrument` (2026-11-21) is what the
    goldfish read of the swap when `try --stage` wrote it, kept beside the reason
    and never in its place — `try` used to write its own verdict as the `why`.

    THE SWAP IS THE UNIT, not the card. A card added and a card cut are two edits
    a reader has to pair up by hand; a swap is one edit that already says what it
    displaced, and `net-change` can then name WHICH swaps bought the delta rather
    than reporting a list that changed somehow.

    ONE FOR ONE, ALWAYS. The list stays at its legal size through every
    intermediate state, so a branch is never briefly a 98 that some command
    measures. It also means the sweep in `candidates` prices exactly this — a
    substitution, which needs no placebo because the library never shrinks.

    It writes through `check_in.analyze`, so a staged swap gets the same refusals
    a paper list gets: singleton, size, commander, and a name the corpus does not
    know. Editing `decklist.txt` directly would skip all four.
    """
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    entries = _parsed(slug, branch)
    typed = _common.split_printing(in_name)
    in_name = typed["name"]
    if typed["set"] and in_name.split(" // ")[0].lower() == str(out_name).split(" // ")[0].lower():
        raise SystemExit(f"That only changes {in_name}'s art, which needs no branch swap — "
                         f"`manamap pilot art {slug} --branch {branch} --set "
                         f"\"{in_name}={typed['set'].upper()} {typed['collector_number']}\"`")
    staged_entries, out_e, in_e = swap_entries(slug, branch, entries, out_name, in_name)
    if typed["set"]:
        # The list keeps the printing; the staged row and every reader keep the name.
        for e in staged_entries:
            if e["name"] == in_name:
                e.update(set=typed["set"], collector_number=typed["collector_number"],
                         foil=typed["foil"])
    # The sideboard rides through untouched: a swap is a mainboard edit.
    text = check_in.render_decklist(staged_entries + _sideboard(slug, branch))
    checked = check_in.analyze(slug, text)
    if checked["blocking"]:
        raise SystemExit("Refusing to stage that swap:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")

    # MEASURE TWICE, said at the cheap step: staging never refuses (the gate is at
    # `simulate`, the expensive one), but it says now when the card has no PLAYED proof.
    try:
        from manamap.sim import cast_check as _cc
        _pr = ((_cc.read_proofs(slug, branch) or {}).get("cards") or {}).get(in_name)
        _w = (_pr or {}).get("verdict_word")
        if _w in ("HELD", "CAST-LATE"):
            checked["warnings"].append(f"{in_name} is {_w} in the branch's cast proofs — a Forge probe of this branch would refuse it")
        elif _w != "PLAYED":
            checked["warnings"].append(f"no cast proof for {in_name} — `forge-cast-check {slug} --branch {branch} --adds --write` before `simulate`")
    except Exception:                                  # noqa: BLE001 - no Forge, no warning
        pass
    doc = meta(slug, branch) or {"slug": slug, "branch": branch, "v": 2}
    entry = {"at": datetime.date.today().isoformat(),
             "out": out_e["name"], "in": in_name,
             "strength": strength, "why": why or ""}
    if instrument:
        entry["instrument"] = instrument
    doc.setdefault("staged", []).append(entry)
    _write_meta(slug, branch, doc)
    return {"slug": slug, "branch": branch, "out": out_e["name"], "in": in_name,
            "staged": len(doc["staged"]), "warnings": checked["warnings"]}


def unstage(slug, branch, out_name=None, in_name=None):
    """Put a staged swap back. Reverses the most recent match.

    A staging area you cannot back out of is a decision, not a draft.
    """
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc = meta(slug, branch) or {}
    staged = doc.get("staged") or []
    if not staged:
        raise SystemExit(f"Nothing staged on {slug}/{branch}.")
    hit = None
    for i in range(len(staged) - 1, -1, -1):
        row = staged[i]
        if ((out_name is None or row["out"] == out_name)
                and (in_name is None or row["in"] == in_name)):
            hit = i
            break
    if hit is None:
        raise SystemExit(
            f"No staged swap matches that on {slug}/{branch} — "
            f"`deck-branch {slug} log {branch}` lists them.")
    row = staged[hit]
    entries = _parsed(slug, branch)
    in_e = _resolve_in_list(entries, row["in"])
    if in_e is None:
        raise SystemExit(
            f"{row['in']} is no longer in the list, so this swap cannot be "
            f"reversed cleanly. Edit decklist.txt and drop the record by hand.")
    rebuilt = []
    for e in entries:
        if e is in_e:
            # PER ENTRY, on purpose: removing ONE copy. Not `count_copies`.
            left = int(e.get("quantity") or 1) - 1
            if left > 0:
                rebuilt.append(dict(e, quantity=left))
            continue
        rebuilt.append(e)
    back = _resolve_in_list(rebuilt, row["out"])
    if back is not None:
        rebuilt = [dict(e, quantity=int(e.get("quantity") or 1) + 1)
                   if e is back else e for e in rebuilt]
    else:
        rebuilt.append({"name": row["out"], "quantity": 1})

    checked = check_in.analyze(
        slug, check_in.render_decklist(rebuilt + _sideboard(slug, branch)))
    if checked["blocking"]:
        raise SystemExit("Refusing to unstage that swap:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")
    staged.pop(hit)
    doc["staged"] = staged
    _write_meta(slug, branch, doc)
    return {"slug": slug, "branch": branch, "out": row["out"], "in": row["in"],
            "staged": len(staged)}


def commit(slug, branch, message):
    """Freeze this candidate list with a message. NOT a merge.

    THE TWO-STEP IS THE PILOT'S OWN DISTINCTION: a commit says "this is the deck
    I am committed to running"; a merge says "this is the deck".

    It records the decklist sha, so a commit names one exact 99 and a later edit
    is visibly a different one.
    """
    import hashlib
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    if not (message or "").strip():
        raise SystemExit(
            "A commit needs a message. It is the only record of WHY this list "
            "and not the last one.")
    text = _list_text(slug, branch)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc = meta(slug, branch) or {"slug": slug, "branch": branch}
    commits = doc.setdefault("commits", [])
    if commits and commits[-1].get("decklist_sha256") == sha:
        raise SystemExit(
            f"The list has not changed since the last commit "
            f"({commits[-1]['at']}: {commits[-1]['message'][:48]}). "
            f"Edit the list, or say something new about the same one by "
            f"deleting that entry.")
    commits.append({"at": datetime.date.today().isoformat(),
                    "decklist_sha256": sha, "message": message.strip()})
    doc.setdefault("v", 2)
    _write_meta(slug, branch, doc)
    return {"slug": slug, "branch": branch, "n": len(commits),
            "decklist_sha256": sha, "message": message.strip()}


def log(slug, branch):
    """The branch's record — and, since 2026-09-12, its STAGED swaps.

    `unstage`'s refusal has always read "No staged swap matches that — `log`
    lists them". It never did: this returned `objective`, `why`, `opened`,
    `base_version`, `commits` and `merged`, and nothing else has ever read
    `staged`. A pilot following the refusal got a commit trail and no swaps, and
    the 21 staged swaps on `eminence-v3` were visible only through `net-change`
    or by opening `branch.json` (#25).

    `log` is the right home rather than re-pointing the refusal: it is the only
    read command that costs nothing, and `unstage` is the thing that needs the
    list.
    """
    doc = meta(slug, branch) or {}
    return {"slug": slug, "branch": branch,
            "objective": doc.get("objective"), "why": doc.get("why"),
            "opened": doc.get("opened"), "base_version": doc.get("base_version"),
            "staged": doc.get("staged") or [],
            "commits": doc.get("commits") or [], "merged": doc.get("merged")}


#: THE STATES A BRANCH CAN BE IN, and the first enumerated set it has ever had.
#: Before this, a branch had exactly two observable conditions — the directory
#: exists, or `merged` is present — and `delete` was the only code in the repo
#: that read `merged`. A branch the pilot had DECIDED ON was byte-identical to a
def _validate_after_merge(slug):
    """`[(artifact, first error)]` for everything that no longer validates.

    Deliberately reports rather than repairs. Every artifact here is either
    AUTHORED (`goldfish_targets.json`) or agent-written (`engine.json`,
    `tutor_guide.json`, `diagnosis.json`), and this repo's standing rule is that
    prose is never hand-patched to green a gate — "a fresh claim under an old
    byline".
    """
    import contextlib
    import importlib
    import io as _io

    from manamap.pilot import deck_status

    out = []
    for artifact, dotted in sorted(getattr(deck_status, "VALIDATED", {}).items()):
        if not (deck_dir(slug) / artifact).exists():
            continue
        # Every validator is a CLI: it prints its errors and exits non-zero.
        # Captured rather than re-implemented — a second copy of "what counts as
        # invalid" is the two-maps divergence `VALIDATED` was extracted to end.
        buf = _io.StringIO()
        try:
            module = importlib.import_module(dotted)
            with contextlib.redirect_stdout(buf):
                module.main(type("Args", (), {"slug": slug, "branch": None})())
        except SystemExit as exit_:
            if exit_.code:
                lines = [ln.strip() for ln in buf.getvalue().splitlines()
                         if ln.strip().startswith("-")]
                out.append((artifact, (lines[0] if lines else "fails its validator")[:130]))
        except Exception as exc:                        # noqa: BLE001 - env
            out.append((artifact, f"{type(exc).__name__}: {exc}"[:130]))
    return out


#: half-finished experiment nobody had looked at.
#:
#: Every one of these is DERIVED and none is stored. That is the rule
#: `validate_pending` earned the hard way — "closure is DERIVED, never declared …
#: a hand-set flag is precisely how HISTORY.md became append-only and
#: append-forgotten".
OPEN = "OPEN"
#: Accepted and not yet merged (2026-11-22: one state, where BLOCKED / READY used
#: to split on whether the cards were found — the bench no longer asks).
PROPOSED = "PROPOSED"
PROPOSED_STALE = "PROPOSED · STALE"
PROPOSED_OUTRUN = "PROPOSED · OUTRUN"
MERGED = "MERGED"
REJECTED = "REJECTED"

BRANCH_STATES = (OPEN, PROPOSED, PROPOSED_STALE, PROPOSED_OUTRUN, MERGED, REJECTED)


def is_decided(slug, branch, doc=None):
    """MERGED or REJECTED — a branch whose question has been answered.

    A DECIDED BRANCH IS HISTORY, NOT AN EXPERIMENT. Its measurement is the record
    the decision rested on (the ledger's `prediction` carries the same figures),
    so nothing regenerates it: `regen.targets` skips it, and the freshness gates
    leave its files as written. Re-measuring a merged branch against the list it
    became reads every delta as zero and overwrites the reading (known-issue 28,
    closed 2026-10-21); re-measuring a rejected one spends minutes on a list
    nobody will sleeve. The derivation is `branch_state`'s own for the two
    states, factored here so the two cannot disagree. A withdrawn branch is OPEN
    and stays measurable, as `withdraw` documents.
    """
    doc = meta(slug, branch) if doc is None else doc
    if doc.get("merged"):
        return True
    if doc.get("proposal"):
        return False
    from manamap.pilot import decisions
    last = decisions.latest_for(slug, branch, base=deck_dir(slug))
    return bool(last and last.get("kind") == "reject")


def branch_state(slug, branch, doc=None):
    """Which of `BRANCH_STATES`, and one sentence saying why.

    Computes nothing it can read; `doc` is accepted so `list` reads each branch
    file once.

    STALE and OUTRUN are the two that could not be asked before, and they are
    the merge-request half of this:

    * **STALE** — the list moved after the pilot accepted it. The proposal
      freezes `decklist_sha256`, so a swap staged afterwards is visible as one.
      Same idiom `commit` already uses to refuse an unchanged list.
    * **OUTRUN** — something else merged first, so the version this proposal
      claims is taken. `base_version` has been written by `new()` since branches
      shipped and **no code has ever compared it to anything**; this is the
      comparison it was written for.
    """
    doc = meta(slug, branch) if doc is None else doc
    if doc.get("merged"):
        m = doc["merged"]
        return MERGED, f"merged {m.get('at')} into the list after V{m.get('into_version_before')}"
    prop = doc.get("proposal")
    if not prop:
        # REJECTED is derived from the ledger's last word on this branch —
        # stored nowhere in the branch file, and undone by a later `propose`.
        from manamap.pilot import decisions
        last = decisions.latest_for(slug, branch, base=deck_dir(slug))
        if last and last.get("kind") == "reject":
            return REJECTED, f"rejected {str(last.get('at') or '')[:10]}: {last.get('reason', '')[:80]}"
        return OPEN, "no proposal — this is an experiment, not a decision"

    from manamap.pilot import deck_versions
    current = deck_versions.report(slug).get("current_version")
    if prop.get("base_version") != current:
        return (PROPOSED_OUTRUN,
                f"proposed against V{prop.get('base_version')} and the deck is now "
                f"V{current} — {prop.get('as_version')} may be taken; re-measure "
                f"and propose again")

    live = _sha_of_list(slug, branch)
    if prop.get("decklist_sha256") != live:
        return (PROPOSED_STALE,
                "the list has changed since it was proposed — re-run `net-change` "
                "and propose again, or `unstage` back to what was accepted")

    return (PROPOSED,
            f"accepted as {prop.get('as_version')} — `deck-branch {slug} merge {branch} --write` "
            f"on the pilot's word")


def base_moved(slug, branch, doc=None, state=None):
    """One line when the deck's list has moved since the branch was cut, else None.

    DERIVED, never stored, like `branch_state`. `new()` records the deck's
    `decklist_sha256` as `base_sha256`; a bench deck is edited IN PLACE between
    saves (`deck_edit`), so `base_version` alone cannot see the deck move under
    a branch, and the branch's diff and net change are then against a list the
    deck no longer runs. None for a branch that recorded no base (opened before
    2026-10-10), and for one already decided — a merge or a rejection is the
    end of the comparison, not a stale one.
    """
    doc = meta(slug, branch) if doc is None else doc
    base = (doc or {}).get("base_sha256")
    if not base:
        return None
    if state is None:
        state = branch_state(slug, branch, doc=doc)[0]
    if state in (MERGED, REJECTED):
        return None
    live = _common.decklist_sha256(slug)
    if not live or live == base:
        return None
    return (f"base moved since {base[:12]} — the deck's list changed after this "
            f"branch was cut, so its comparison is against an older deck")


def _sha_of_list(slug, branch):
    """The branch's list as it stands.

    `common.list_sha256` is THE algorithm — one definition, so the six that
    existed cannot drift again. The PATH still resolves through this module's
    own `deck_dir`, which is `common.deck_dir` imported: routing the path
    through `common.decklist_sha256` too would move the patch point that every
    branch test uses, and the consolidation is about the algorithm and the two
    meanings, not about which function object resolves a directory.
    """
    return _common.list_sha256(
        (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8"))


def _study_for(slug, branch, study_id):
    """The approved study a proposal rests on, read and checked (2026-11-21):
    it must be about THIS branch, decided `approve`, and about the list on disk.
    One seam, so a test can hand in a study without a deck tree."""
    from manamap.pilot import study as st
    if not study_id:
        raise SystemExit(
            f"A PROPOSAL RESTS ON A STUDY. Nothing has been decided about {slug}/{branch} in the "
            f"pilot's words — the question, the facts from every source, the agents' arguments, "
            f"the call and who decided:\n"
            f"  manamap pilot study {slug} list\n"
            f"  manamap pilot study {slug} new \"<question>\" --branch {branch}   # then add, call, install\n"
            f"  manamap pilot deck-branch {slug} propose {branch} --as <vX.Y.Z> --study <NNN>")
    doc = st.read(slug, study_id)
    if doc.get("branch") != branch:
        raise SystemExit(f"study {study_id} is about branch {doc.get('branch')!r}, not {branch!r}")
    call = doc.get("call") or {}
    if call.get("decision") != "approve":
        raise SystemExit(f"study {study_id} reads {call.get('decision', 'open')!r} — only an approved, "
                         f"installed study proposes (`study {slug} call {study_id} …`, then `install`)")
    if not doc.get("decided_at"):
        raise SystemExit(f"study {study_id} is not installed — `manamap pilot study {slug} install {study_id}`")
    live = _sha_of_list(slug, branch)
    if doc.get("decklist_sha256") != live:
        raise SystemExit(
            f"study {study_id} studied a different list than the one on disk — the list moved after "
            f"the call. Recompute its facts and read them again before proposing:\n"
            f"  manamap pilot study {slug} facts {study_id} --write --force")
    return doc


def propose(slug, branch, as_version, why=None, reason=None, at=None, study=None):
    """Accept this branch as the deck's next version; the merge is the pilot's word.

    IT RESTS ON A STUDY (2026-11-21), never on the goldfish's word. `--study NNN`
    names an approved, installed study of this branch; the instrument's read
    (`net_change.json`), when there is one, is frozen beside it under
    `accepted_on.instrument` and gates nothing. The old gate — refuse without a
    report, refuse on "do not merge" unless `--anyway --reason` — made a solitaire
    simulator with no opponents the arbiter of a purchase, and the pilot overrode
    it fourteen times.

    THE STAGE THE BENCH HAD NO ROOM FOR. `commit`'s own docstring names the gap —
    "a commit says 'this is the deck I am committed to running'; a merge says
    'this is the deck'. The gap between them is CARDBOARD" — and `deck_info._next`
    names it again: "a branch that is fully sourced is a decision waiting to be
    taken; one that is not is a shopping list." Neither had anywhere to put it, so
    a decided branch and an abandoned one rendered identically and the decision
    lived in the pilot's head until they opened another deck.

    IT MEASURES NOTHING. It records that a human said yes, freezes WHAT they said
    yes to, and gets out of the way. Everything downstream — blocked, ready,
    stale, outrun — is derived on read.

    Two shas, because they answer different questions. `decklist_sha256` is the
    list the pilot accepted; `accepted_on.decklist_sha256` is the list the report
    measured. They must match to propose, and the first going out of date
    afterwards is exactly what STALE means.
    """
    import datetime

    from manamap.pilot import deck_versions
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc = meta(slug, branch) or {"slug": slug, "branch": branch}
    if doc.get("merged"):
        raise SystemExit(
            f"{slug}/{branch} is already merged ({doc['merged'].get('at')}). "
            f"There is nothing left to propose.")
    from manamap.pilot import protected
    if protected.names(slug):
        protected.refuse(slug, diff(slug, branch).get("out") or [], f"to propose {branch}")
    # SAME VERSION AMENDS; A DIFFERENT ONE STILL REFUSES.
    #
    # This refused any second proposal, so the way back from `PROPOSED · STALE`
    # — which one `stage` and one `commit` are enough to cause — was `withdraw`
    # plus a `propose` carrying the WHOLE original reason retyped, because
    # `withdraw` discards `accepted_on`: the objective, its grade and its
    # reading at the moment the pilot said yes (#46).
    #
    # Amending keeps the version and appends the new reason to the old, so the
    # record reads as what it is: one decision, revised. A DIFFERENT `--as` is
    # a change of mind about what this list is meant to become, and that still
    # goes through `withdraw`.
    amending = False
    if doc.get("proposal"):
        held = doc["proposal"]
        if held.get("as_version") != as_version:
            raise SystemExit(
                f"{slug}/{branch} is already proposed as "
                f"{held.get('as_version')} ({held.get('at')}), and you asked for "
                f"{as_version}. Proposing a DIFFERENT version is a change of "
                f"mind about what this list becomes — "
                f"`deck-branch {slug} withdraw {branch}` first.\n"
                f"To revise the same proposal, pass --as {held.get('as_version')}.")
        amending = True

    # THE TAG IS `deck_versions`' VOCABULARY, NOT A SECOND ONE. Its regexes
    # already refuse `v1.2` and `1.2.3.4`, and a second copy here would drift.
    if not deck_versions._RELEASE_RE.match(as_version or ""):
        raise SystemExit(
            f"--as must be a release tag like `v1.0.2`. Got: {as_version!r}"
            + ("  (a near miss — releases are MAJOR.MINOR.PATCH)"
               if deck_versions._NEARLY_RELEASE_RE.match(as_version or "") else ""))
    existing = deck_versions.tags(slug) or {}
    if as_version in existing:
        raise SystemExit(
            f"{as_version} already names V{existing[as_version].get('version')} "
            f"on {slug}. A tag is a claim about one exact list — pick the next one.")

    # A PROPOSAL RESTS ON A STUDY (2026-11-21). The study is the decision in the
    # pilot's words with every source behind it; the instrument's report, if one
    # exists, is frozen beside it and gates nothing.
    sdoc = _study_for(slug, branch, study)
    live = _sha_of_list(slug, branch)
    nc = load_json(deck_dir(slug, branch) / "net_change.json") or {}
    rec = nc.get("recommendation") or {}
    # `--reason` ON PROPOSE IS ALMOST CERTAINLY A TYPO FOR `--why`.
    #
    # Both options live on the shared `deck-branch` parser — `--reason` is
    # merge's sourcing override, `--why` is new's and stage's. So `propose … --as
    # v1.2.2 --reason "…"` PARSED, proposed, and wrote `proposal.why: ""`. The
    # pilot's sentence went nowhere and was found by reading branch.json (#48).
    # Refused rather than silently mapped to `--why`.
    if reason:
        raise SystemExit(
            f"--reason is merge's sourcing override. Did you mean --why?\n"
            f"  manamap pilot deck-branch {slug} propose {branch} "
            f"--as {as_version} --study {study} --why \"{reason[:60]}\"")

    grade = nc.get("objective_grade") or {}
    call = sdoc.get("call") or {}
    held = doc.get("proposal") or {}
    # THE PREVIOUS ACCEPTANCE IS KEPT, not overwritten. `accepted_on` is the
    # only record of what the report said at the moment the pilot said yes, and
    # a revision that discards it leaves a decision with no evidence behind it.
    history = list(held.get("history") or [])
    if amending:
        history.append({k: v for k, v in held.items() if k != "history"})
    reasons = [r for r in (held.get("why") if amending else None, why) if r]
    prop = {
        "at": (at or datetime.date.today().isoformat()),
        "as_version": as_version,
        "why": "\n".join(reasons),
        "base_version": deck_versions.report(slug).get("current_version"),
        "decklist_sha256": live,
        # WHAT THE PILOT SAID YES TO: the study, in their words, and the
        # instrument's read beside it (null when the branch was never measured).
        "accepted_on": {
            "study": sdoc.get("id"),
            "question": sdoc.get("question"),
            "decision": call.get("decision"),
            "decided_by": list(call.get("decided_by") or []),
            "pilot_words": call.get("pilot_words"),
            "decklist_sha256": sdoc.get("decklist_sha256"),
            "instrument": ({
                "state": rec.get("state"),
                "objective": nc.get("objective"),
                "grade": grade.get("state"),
                "reading": grade.get("reading"),
                "harness": nc.get("harness"),
                "decklist_sha256": nc.get("decklist_sha256"),
                "current": nc.get("decklist_sha256") == live,
            } if nc else None),
        },
    }
    if history:
        # Oldest first. `branch_state` reads only the LIVE acceptance; this is
        # the record, and `validate_branch` checks its shape.
        prop["history"] = history
    if amending and not why:
        # Amending with no new reason keeps the old one rather than blanking it —
        # the same rule `set_paper` learned the hard way (#51).
        prop["why"] = held.get("why") or ""
    doc["proposal"] = prop
    doc["v"] = max(int(doc.get("v") or 2), 3)
    _write_meta(slug, branch, doc)
    # THE LEDGER LINE. `accepted_on` stays here (it is what `branch_state`
    # reads); the ledger is the history, and it carries the study's call and
    # the report's Forge block too, so a later `outcome` has a prediction to
    # close against.
    from manamap.pilot import decisions
    decisions.append(slug, "amend" if amending else "propose", branch=branch,
                     as_version=as_version, reason=prop.get("why") or None,
                     study=sdoc.get("id"), decided_by=list(call.get("decided_by") or []),
                     pilot_words=call.get("pilot_words") or None,
                     branch_decklist_sha256=live,
                     prediction=decisions.prediction_from(nc, sdoc), base=deck_dir(slug))
    return {"slug": slug, "branch": branch, "proposal": prop,
            "state": branch_state(slug, branch, doc=doc)}


def withdraw(slug, branch, reason=None):
    """Take the proposal back. The "changes requested" path.

    Deletes the block and nothing else — the branch stays open, measured and
    measurable, and `branch.json` gains no `withdrawn` key: a graveyard of
    withdrawn intentions in the branch file is the `HISTORY.md` failure again.
    THE REASON GOES IN THE LEDGER, which is the one place a withdrawal is
    recorded, so the decision is not lost either — it was, for the first
    forty-two branches, and nothing could say why a proposal went away.
    """
    doc = meta(slug, branch)
    if not doc:
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    if not doc.get("proposal"):
        raise SystemExit(f"{slug}/{branch} is not proposed — nothing to withdraw.")
    if not str(reason or "").strip():
        raise SystemExit("withdraw needs --reason: the branch directory will not keep "
                         "it, and the ledger is where a withdrawal is recorded")
    prop = doc.pop("proposal")
    _write_meta(slug, branch, doc)
    from manamap.pilot import decisions
    decisions.append(slug, "withdraw", branch=branch, reason=reason,
                     as_version=prop.get("as_version"), base=deck_dir(slug))
    return {"slug": slug, "branch": branch, "withdrew": prop}


def reject(slug, branch, reason=None):
    """Record that this branch is NOT going to be merged, and why.

    Appends to the ledger and touches nothing else: the branch stays on disk,
    measured, and `branch_state` derives REJECTED from the ledger's last word
    on it until a later `propose` says otherwise. A proposal still open is
    withdrawn first, under the same reason.
    """
    if not str(reason or "").strip():
        raise SystemExit("reject needs --reason: a refusal with no argument cannot be "
                         "revisited when the evidence changes")
    doc = meta(slug, branch)
    if not doc:
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    if doc.get("merged"):
        raise SystemExit(f"{slug}/{branch} is merged; a merge is undone by a new branch, "
                         f"not by rejecting the old one")
    if doc.get("proposal"):
        withdraw(slug, branch, reason=reason)
    from manamap.pilot import decisions
    nc = load_json(deck_dir(slug, branch) / "net_change.json") or None
    e = decisions.append(slug, "reject", branch=branch, reason=reason,
                         prediction=decisions.prediction_from(nc), base=deck_dir(slug))
    return {"slug": slug, "branch": branch, "decision": e,
            "state": branch_state(slug, branch)}


def summary(slug, branch, title=None, blurb=None, by="jarvis"):
    """THE BRANCH IN PLAIN WORDS (2026-10-20). The pilot read the workbench's
    git-style row and said it was "too git like — have Jarvis summarize the
    objective/name and quick description". The page makes no model call, so the
    summary is a committed artifact on `branch.json`: a `title` (the branch in
    words, not its slug) and a one-sentence `blurb` (what it would do and why),
    written by Jarvis when he reviews the branch and rendered by the workbench
    and the deck page as the row's headline. Figures stay in the measured
    fields; the summary names none, so it never goes stale against a re-run.
    """
    title = str(title or "").strip(); blurb = str(blurb or "").strip()
    if not title or not blurb:
        raise SystemExit("summary needs --title (the branch in words) and --blurb (one sentence)")
    if len(title) > 80 or len(blurb) > 320:
        raise SystemExit("summary: the title is at most 80 characters and the blurb 320 — it is a headline, not the why")
    if re.search(r"\d", title + blurb):
        raise SystemExit("summary: no digits — figures live in the measured fields and a summary that quotes one goes stale on the next run")
    doc = meta(slug, branch)
    if not doc:
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc["summary"] = {"title": title, "blurb": blurb, "by": by,
                      "at": datetime.date.today().isoformat()}
    _write_meta(slug, branch, doc)
    return doc["summary"]


def delete(slug, branch, force=False):
    """Remove a branch. Refuses an unmerged one without `--force`.

    A branch holds measurements that cost real time — a 100-game Forge run is
    45 minutes — so deleting one that was never merged is throwing away the
    evidence for a decision nobody recorded.
    """
    import shutil
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc = meta(slug, branch) or {}
    if not doc.get("merged") and not force:
        raise SystemExit(
            f"'{branch}' was never merged. Deleting it discards its "
            f"measurements and whatever decision they supported — "
            f"`--force` if that is what you mean.")
    shutil.rmtree(path)
    return {"slug": slug, "branch": branch, "deleted": str(path)}


def merge(slug, branch, write=False, run_chain=True):
    """Make the branch the deck's list. Refuses what it cannot honestly apply.

    NO SOURCING GATE (2026-11-22). Whether the cards are in the sleeves is the
    pilot's call and the pilot's word; the bench refuses only a list it cannot
    check in (`check_in.analyze`) and a cut the keep list forbids.
    """
    text = _list_text(slug, branch)
    checked = check_in.analyze(slug, text)
    blocking = list(checked["blocking"])
    # The keep list: the pilot releases a card by editing protected.json, never by a flag.
    from manamap.pilot import protected
    if protected.names(slug):
        blocking += protected.refusals(slug, diff(slug, branch).get("out") or [])
    out = {"slug": slug, "branch": branch, "blocking": blocking,
           "warnings": checked["warnings"], "written": False}
    if blocking or not write:
        return out
    # A BACKUP FIRST, the way `check_in.apply` does. Merge overwrites the deck's
    # tracked list; check-in has made a `.txt.bak` since it shipped and merge
    # never did, which is the more destructive of the two.
    #
    # THROUGH THE ONE WRITER (`deck_edit.write_list`): the lock, the atomic
    # replace, the `.txt.bak` and a journal line, so a merge can be undone on a
    # bench deck like any edit. On a sleeved deck this IS the publish path, so
    # no edit guard is applied — merge's own gates above are the guard.
    from manamap.pilot import deck_edit
    target = deck_dir(slug) / "decklist.txt"
    deck_edit.replace_list(slug, check_in.render_decklist(checked["entries"]),
                           source="merge", note=f"merge {branch}", path=target)
    out["written"] = True

    # THE CHAIN, because a merge that leaves the figures behind makes the deck
    # read stale forever — `goldfish_metrics.json` and `mana_analysis.json` stamp
    # the decklist sha. `check_in.apply` has run it since it shipped; merge
    # printed "next: fetch-deck" and hoped.
    # THE READING THE DECISION RESTED ON, read BEFORE anything rebuilds it. The
    # chain below re-measures the deck's branches against the new list, and this
    # branch IS the new list, so its net_change would read every delta as zero
    # (known-issue 28, closed 2026-10-21); regen skips it, and the ledger's
    # prediction is frozen from the file as it stood when the pilot said merge.
    # Read here, outside the chain, so a merge that runs no chain still has it.
    _nc_at_merge = load_json(deck_dir(slug, branch) / "net_change.json") or None
    if run_chain:
        from types import SimpleNamespace
        from manamap.pilot import fetch_deck, goldfish, mana_analysis
        ran = []
        # `fetch-deck` first and on its own: everything downstream reads
        # `cards.json`, and `regen`'s stages assume it is already current.
        for name, mod in (("fetch-deck", fetch_deck), ("goldfish", goldfish),
                          ("mana-analysis", mana_analysis)):
            try:
                mod.main(SimpleNamespace(slug=slug, branch=None))
                ran.append(name)
            except Exception as exc:                    # pragma: no cover - env
                out.setdefault("chain_failed", []).append(f"{name}: {exc}")
                break

        # THE REST OF THE BUILD, in `regen`'s dependency order rather than a
        # second hand-written list. The three above used to be the whole chain,
        # so a merge left `diagnostic.json`, `benchmark.json` and `info.json`
        # describing the PREVIOUS 99 — and the way anyone found out was a failing
        # test hours later. Reusing `regen` means this order has one home.
        if not out.get("chain_failed"):
            try:
                from manamap.pilot import regen
                # `slug=` scopes it to this deck, and `regen` fans out across the
                # deck's OPEN BRANCHES too — which is required, not incidental:
                # main just moved underneath every open branch, so each one's
                # diff and net_change describe a base that no longer exists.
                # Doing this by hand after the last two merges is what this line
                # replaces. `skip`: the chain above just ran the goldfish and
                # mana-analysis on the deck; regen ran them a second time.
                # `skip_branches`: this branch's `merged` key is written AFTER
                # the chain, so `targets` cannot yet see it is decided.
                regen.run(slug=slug, jobs=None, echo=lambda *a, **k: None,
                          skip=("goldfish", "mana-analysis"),
                          skip_branches=(branch,))
                ran.extend(n for n in regen.STAGE_NAMES if n not in ran)
            except Exception as exc:                    # pragma: no cover - env
                out.setdefault("chain_failed", []).append(f"regen: {exc}")

        # THE HANDBOOK AND THE MANIFEST. The handbook embeds the goldfish figures
        # and `build-index` carries the paper lock and the passing-stack list, so
        # both describe the previous 99 until they are re-rendered. Both went
        # stale on the last two merges and were caught by a freshness test rather
        # than by the command that made them stale.
        if not out.get("chain_failed"):
            from types import SimpleNamespace as _NS
            for name, dotted, kwargs in (
                    # `deck-map` is DETERMINISTIC and was being reported as stale
                    # rather than rebuilt — it re-lays the deck's own
                    # constellation from the new 99. Its city NAMES are
                    # agent-authored and are merged separately by
                    # `merge-deck-map`, so rebuilding here recomputes membership
                    # and leaves the naming pass to be reported below, which is
                    # the right split: measure automatically, name deliberately.
                    ("deck-map", "manamap.pilot.deck_map", {"slug": slug}),
                    # No handbook render: the rendered Pilot's Operating
                    # Handbook was retired 2026-11-22; the deck page is the one
                    # place a deck is read.
                    ("build-index", "manamap.pilot.deck_manifest", {})):
                try:
                    import importlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        importlib.import_module(dotted).main(_NS(**kwargs))
                    ran.append(name)
                except Exception as exc:                # pragma: no cover - env
                    out.setdefault("chain_failed", []).append(f"{name}: {exc}")
        out["chain"] = ran

        # AND THEN ASK WHETHER WHAT WE JUST BUILT IS VALID. A merge can leave an
        # AUTHORED file naming cards the new list does not run — the declaration
        # in `goldfish_targets.json` is the common one — and no amount of
        # rebuilding fixes that, because nobody but the pilot can say which
        # components the new deck has. Rebuilding is this command's job; saying
        # what it could not fix is the other half.
        out["invalid"] = _validate_after_merge(slug)

    # THE DECK CONTEXT, which no rebuild can rewrite: its blocks are refreshed and
    # the Keeper's `deck-change` pass is named for Jarvis to run next.
    from manamap.pilot import deck_context, deck_link
    out["context"] = deck_context.list_change(slug)
    # The copy on another site is behind the merged list (files only, 2026-10-17).
    out["link_drift"] = deck_link.drift_lines(slug)

    # THE BRANCH RECORDS THAT IT LANDED. Without this the branch survives
    # untouched, `diff` reads +0 -0 forever, and nothing links the resulting
    # version back to the work that produced it.
    from manamap.pilot import deck_versions
    doc = meta(slug, branch) or {"slug": slug, "branch": branch}
    doc["merged"] = {
        "at": datetime.date.today().isoformat(),
        "decklist_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        # The version this became. It is minted from git by the COMMIT the pilot
        # makes next, so at merge time we can only name what it was merged INTO.
        "into_version_before": deck_versions.report(slug).get("current_version"),
    }
    _write_meta(slug, branch, doc)
    # THE LEDGER LINE, with the report's prediction frozen — the figure the
    # merged list's own runs will later be read against by `decisions outcome`.
    from manamap.pilot import decisions
    _nc = _nc_at_merge
    _pred = decisions.prediction_from(_nc)
    # ABSENT MEANS ABSENT, WITH A STATED REASON. A merge normally freezes the
    # report's figures so `decisions outcome` has something to close against.
    # A branch that was never measured has nothing to freeze — sharknado's
    # swords-v1 was four swaps the pilot made in CARDBOARD, merged to record the
    # list being shuffled rather than to accept a measured proposal. Writing a
    # `prediction` there would be inventing one, and leaving the key out
    # silently made the entry indistinguishable from a merge whose prediction
    # was simply forgotten. So the absence is named.
    _note = None if _pred else (
        "no net_change.json on the branch at merge time, so there is no report to "
        "freeze — this merge records a list rather than accepting a measured "
        "proposal, and `decisions outcome` has nothing to close it against")
    decisions.append(slug, "merge", branch=branch,
                     decklist_sha256=doc["merged"]["decklist_sha256"],
                     into_version_before=doc["merged"]["into_version_before"],
                     as_version=(doc.get("proposal") or {}).get("as_version"),
                     prediction=_pred, prediction_note=_note, base=deck_dir(slug))

    # WHAT IS NOW STALE. The registry already carries the command for each stage,
    # so this reads `deck_status` rather than keeping a second list that can
    # disagree with it.
    #
    # THIS BLOCK HAD NEVER RUN. It called `deck_status.report(slug)` — a function
    # that does not exist; the module defines `status`, `fleet`, `_fleet_main`
    # and `main` — and the bare `except` below swallowed the AttributeError, so
    # `out["stale"]` was unconditionally `[]` and the "written against the
    # previous list" block in the CLI has never printed once. Two more bugs sat
    # behind it: rows key their name as `stage`, not `key`, and `status()`
    # returns a LIST rather than `{"stages": [...]}`. Nothing caught it because
    # no branch in this repo has ever been merged, and a bare `except` around
    # three chained assumptions cannot fail loudly enough to be noticed.
    #
    # The `except` stays — a missing corpus or strategy DB genuinely makes some
    # gates unrunnable and a merge must not die for it — but it is now narrow
    # enough that a typo cannot hide inside it.
    try:
        from manamap.pilot import deck_status
        rows = deck_status.status(slug)
    except Exception as exc:                            # pragma: no cover - env
        out["stale"] = []
        out["stale_unavailable"] = str(exc)
    else:
        out["stale"] = [{"stage": r["stage"], "how": r.get("how", "")}
                        for r in rows if r.get("state") == "STALE"]
    return out


# ── CLI ──────────────────────────────────────────────────────────────────

def _dispatch(args):
    slug, action = args.slug, args.action
    branch = getattr(args, "name", None)
    if action == "list":
        doc = report(slug)
        if getattr(args, "json", False):
            print(json.dumps(doc, indent=1)); return
        if not doc["branches"]:
            print(f"No branches on {slug}. "
                  f"`manamap pilot deck-branch {slug} new <name> --from <file>`")
            return
        print(f"BRANCHES — {slug} ({len(doc['branches'])})")
        for b in doc["branches"]:
            mark = b.get("state") or "OPEN"
            if b.get("proposal"):
                mark += f"  → {b['proposal']['as_version']}"
            print(f"  {b['name']:24} opened {b['opened']}  +{b['add']:<3} -{b['out']:<3} "
                  f"[{b['size']:>3}]  {mark}")
            if b["why"]:
                print(f"      {b['why'][:96]}")
            if b.get("base_moved"):
                print(f"      {b['base_moved']}")
        return
    if not branch:
        raise SystemExit(f"`deck-branch {slug} {action}` needs a branch name.")
    if action == "new":
        # THE OBJECTIVE IS OPTIONAL (the pilot, 2026-10-28). A branch is judged on
        # its whole paired profile — sixteen Holm-corrected rows — and an objective,
        # when one is given, is a GUARD: one row written down before the run that
        # the branch must not fail, shown among the others. It is still parsed by
        # `parse_objective` whenever it is given, so a guard names an axis the bench
        # computes. With none, `net-change` reports only what changed.
        raw = getattr(args, "objective", None)
        objective = None
        if raw:
            objective = parse_objective(raw)
            objective["why"] = getattr(args, "why", None) or ""
        # NO --from MEANS THE DECK'S OWN LIST. It used to reach `read_list(None)`
        # and die in a TypeError traceback (the Story 6 trial, 2026-10-07); the
        # list a branch usually starts from is the one the deck runs today.
        # `from_list`, not `source`: a local named `source` here made the whole
        # function's `source` a local, and `deck-branch <slug> source <name>`
        # died in UnboundLocalError (found on goblin-storm, 2026-10-18).
        from_list = getattr(args, "source", None)
        if not from_list:
            from_list = deck_dir(slug) / "decklist.txt"
            if not from_list.exists():
                raise SystemExit(f"{slug} has no decklist.txt to branch from — pass --from <file>")
            print(f"  (no --from: starting from {slug}'s own decklist.txt)")
        got = new(slug, branch, check_in.read_list(from_list),
                  why=getattr(args, "why", None), objective=objective)
        print(f"Opened {got['path']}  ({got['size']} cards)")
        if objective:
            print(f"  guard: {objective['axis']} {objective['op']} {objective['value']}"
                  + (f" @{objective['pod']}" if objective.get("pod") else ""))
        else:
            print("  guard: none — net-change reads the whole profile, sixteen rows, "
                  "Holm-corrected; add one with `deck-branch "
                  f"{slug} objective {branch} --objective \"<axis> <op> <n>\"`")
        # ANCHOR THE THRESHOLD TO THE CONTROL THAT WILL GRADE IT.
        #
        # `--objective` takes a bare number and the number has to come from
        # somewhere. Twice in one session it came from the wrong place: once
        # from a `goldfish` run (a different seed to net-change's, 2.438 against
        # 2.366) and once from a net-change table printed BEFORE a model change
        # moved the control 13 points underneath it. Both branches then read NOT
        # MET against lines that were never the right yardstick, and the second
        # one was actually an IMPROVEMENT on the control it was graded against.
        #
        # The fix is not a rule in a document. It is printing the number, here,
        # at the moment it is being chosen, from the same harness that will do
        # the grading. Cheap — one goldfish run — and it makes the mistake
        # visible instead of arriving hours later inside a verdict.
        for line in (champion_reading(slug, objective) if objective else []):
            print(line)
        for w in got["warnings"]:
            print(f"  warning: {w}")
        print(f"  next: `manamap pilot deck-branch {slug} source {branch}`")
        return
    if action in ("stage", "unstage"):
        out_name = getattr(args, "swap_out", None)
        in_name = getattr(args, "swap_in", None)
        if action == "stage" and not (out_name and in_name):
            raise SystemExit(
                f"A swap is ONE CARD OUT AND ONE CARD IN:\n"
                f'  manamap pilot deck-branch {slug} stage {branch} '
                f'--out "<card>" --in "<card>"\n'
                f"`manamap pilot upgrades {slug} --branch {branch}` proposes them.")
        if action == "stage":
            got = stage(slug, branch, out_name, in_name,
                        strength=getattr(args, "strength", None),
                        why=getattr(args, "why", None))
            print(f"Staged on {slug}/{branch}:  - {got['out']}  + {got['in']}")
        else:
            got = unstage(slug, branch, out_name, in_name)
            print(f"Unstaged on {slug}/{branch}:  + {got['out']}  - {got['in']}")
        print(f"  {got['staged']} swap(s) staged. Nothing is measured yet — "
              f"`manamap pilot net-change {slug} --branch {branch}`")
        for w in got.get("warnings") or []:
            print(f"  warning: {w}")
        warn_if_proposed(slug, branch, "edit")
        return
    if action == "commit":
        got = commit(slug, branch, getattr(args, "message", None))
        print(f"Committed #{got['n']} on {slug}/{branch}  "
              f"[{got['decklist_sha256'][:12]}]")
        print(f"  {got['message']}")
        print("  a commit is a decision on a list; the merge is the pilot's word")
        warn_if_proposed(slug, branch, "commit")
        return
    if action == "log":
        got = log(slug, branch)
        if getattr(args, "json", False):
            print(json.dumps(got, indent=1)); return
        o = got.get("objective") or {}
        print(f"BRANCH — {slug}/{branch}   opened {got['opened']}"
              f"   from V{got.get('base_version')}")
        if o:
            print(f"  objective: {o['axis']} {o['op']} {o['value']}"
                  + (f"   — {o['why']}" if o.get("why") else ""))
        else:
            print("  objective: none — judged on the whole profile, no guard row")
        # STAGED SWAPS, which `unstage`'s refusal has always claimed this
        # command lists and which it never did (#25). Uncommitted work is the
        # thing a pilot is most likely to be asking about.
        staged = got.get("staged") or []
        if staged:
            print(f"\n  STAGED, not yet committed ({len(staged)}):")
            for s in staged:
                line = f"    - {s.get('out')}  + {s.get('in')}"
                if s.get("strength") is not None:
                    line += f"   [strength {s['strength']}]"
                print(line)
                if s.get("why"):
                    print(f"        {s['why']}")
        if not got["commits"]:
            print("\n  no commits yet — `deck-branch "
                  f"{slug} commit {branch} -m \"…\"`")
        for i, c in enumerate(got["commits"], 1):
            print(f"\n  #{i}  {c['at']}  [{c['decklist_sha256'][:12]}]")
            print(f"      {c['message']}")
        m = got.get("merged")
        if m:
            print(f"\n  MERGED {m['at']} into the list after V"
                  f"{m.get('into_version_before')}")
        return
    if action == "propose":
        got = propose(slug, branch,
                      getattr(args, "as_version", None),
                      why=getattr(args, "why", None),
                      reason=getattr(args, "reason", None),
                      study=getattr(args, "study", None))
        if getattr(args, "json", False):
            print(json.dumps(got, indent=1)); return
        p = got["proposal"]
        state, why = got["state"]
        print(f"\n{state} — {slug}/{branch} as {p['as_version']}")
        print(f"  {why}")
        if p.get("why"):
            print(f"  {p['why']}")
        a = p["accepted_on"]
        print(f"\n  accepted on study {a.get('study')}: {a.get('decision')} — decided by "
              f"{', '.join(a.get('decided_by') or []) or '—'}")
        if a.get("pilot_words"):
            print(f"  pilot: “{a['pilot_words']}”")
        inst = a.get("instrument")
        if inst:
            from manamap.pilot.net_change import state_word
            print(f"  the goldfish read: {state_word(inst)}"
                  + (f", guard {inst['objective']['axis']} {inst['objective']['op']} "
                     f"{inst['objective']['value']} {str(inst.get('grade') or '').upper()}"
                     f" at {inst.get('reading')}" if inst.get("objective") else "")
                  + ("" if inst.get("current") else " (an older list of this branch)"))
        else:
            print("  the goldfish has not read this branch; the study decided")
        print(f"\n  Nothing is merged. On the pilot's word:\n"
              f"      manamap pilot deck-branch {slug} merge {branch} --write\n"
              f"  To take it back:  deck-branch {slug} withdraw {branch}")
        return
    if action == "withdraw":
        got = withdraw(slug, branch, reason=getattr(args, "reason", None))
        w = got["withdrew"]
        print(f"Withdrew the proposal on {slug}/{branch} "
              f"({w.get('as_version')}, made {w.get('at')}).")
        print("  The branch is untouched and still measured; the reason is in the ledger.")
        return
    if action == "reject":
        got = reject(slug, branch, reason=getattr(args, "reason", None))
        state, why = got["state"]
        print(f"{state} — {slug}/{branch}: {why}")
        print("  Recorded in decisions.jsonl; the branch stays on disk and measured, "
              "and a later `propose` reopens it.")
        return
    if action == "objective":
        got = reaim(slug, branch, getattr(args, "objective", None), why=getattr(args, "why", None))
        o = got["objective"]
        print(f"{slug}/{branch}: objective is now {o['axis']} {o['op']} {o['value']}"
              + (f" @{o['pod']}" if o.get("pod") else ""))
        prev = got["was"]
        if prev:
            print(f"  was {prev['axis']} {prev['op']} {prev['value']} — kept in branch.json's "
                  f"objective_history; `net-change --write` re-grades against the new line")
        for line in champion_reading(slug, o):
            print(line)
        return
    if action == "summary":
        got = summary(slug, branch, title=getattr(args, "title", None), blurb=getattr(args, "blurb", None))
        print(f"{slug}/{branch}: \"{got['title']}\" — {got['blurb']}")
        print("  Written to branch.json; the workbench and the deck page read it as the row's headline.")
        return
    if action == "delete":
        got = delete(slug, branch, force=getattr(args, "force", False))
        print(f"Deleted {got['deleted']}")
        return
    if action == "show":
        doc = meta(slug, branch)
        state, why = branch_state(slug, branch, doc=doc)
        prop = doc.get("proposal")
        print(f"BRANCH — {slug}/{branch}   {state}")
        print(f"  {why}")
        if prop:
            print(f"  proposed {prop['at']} as {prop['as_version']}"
                  + (f" — {prop['why']}" if prop.get("why") else ""))
            print()
        # The list itself still prints, because `show` has always been the way to
        # read a branch's 99 and a state line above it does not replace that.
        print(_list_text(slug, branch), end="")
        return
    if action == "diff":
        d = diff(slug, branch)
        if getattr(args, "json", False):
            print(json.dumps(d, indent=1)); return
        print(f"DIFF — {slug}/{branch} vs the current list "
              f"({d['base_size']} -> {d['size']} cards)")
        print(f"\n  OUT ({len(d['out'])}):")
        for n in d["out"]:
            print(f"    - {n}")
        print(f"\n  IN ({len(d['add'])}):")
        for n in d["add"]:
            print(f"    + {n}")
        if d["quantity"]:
            print(f"\n  COPIES ({len(d['quantity'])}) — same card, different count:")
            for q in d["quantity"]:
                print(f"    {'-' if q['to'] < q['from'] else '+'} {q['name']}"
                      f"  {q['from']} -> {q['to']}")
        print(f"\n  {d['out_copies']} copies out, {d['add_copies']} in")
        return
    if action == "merge":
        got = merge(slug, branch, write=getattr(args, "write", False))
        if got["blocking"]:
            print(f"REFUSED — {slug}/{branch} cannot be merged:")
            for b in got["blocking"]:
                print(f"  - {b}")
            raise SystemExit(1)
        for w in got["warnings"]:
            print(f"  warning: {w}")
        if not got["written"]:
            print(f"{slug}/{branch} is mergeable. `--write` applies it.")
            return
        print(f"Merged {branch} into {slug}/decklist.txt  "
              f"(previous list kept as decklist.txt.bak)")
        if got.get("chain"):
            print(f"  recomputed: {', '.join(got['chain'])}")
        for failed in got.get("chain_failed") or []:
            print(f"  CHAIN FAILED — {failed}")
        stale = got.get("stale") or []
        if stale:
            # WHAT THE MERGE COULD NOT REDO. The chain covers the deterministic
            # figures; every narrative artifact was written about the OLD list
            # and no command can regenerate it — that is an agent's work, and
            # naming it here is the difference between a merge that finishes and
            # one that quietly leaves the deck describing a deck you deleted.
            print(f"\n  STALE — written against the previous list ({len(stale)}):")
            for row in stale:
                print(f"    {row['stage']:10} {row['how']}")
        invalid = got.get("invalid") or []
        if invalid:
            # THE HALF A REBUILD CANNOT FIX. These are AUTHORED or agent-written
            # and name cards the new list no longer runs; regenerating them would
            # mean inventing which components the new deck has, which is the
            # pilot's call and an agent's work. Naming them here is the whole
            # point of running the build at merge time instead of finding out
            # from a failing test hours later.
            print(f"\n  INVALID after the merge ({len(invalid)}) — "
                  f"these need you or an agent, not a rebuild:")
            for artifact, why in invalid:
                print(f"    {artifact:24} {why}")
        from manamap.pilot import deck_context
        deck_context.print_list_change(got.get("context"))
        for line in got.get("link_drift") or []:
            print(f"  {line}")
        print("\n  NOT COMMITTED — the commit is what `deck-version` numbers and what the")
        print("  captain's log stamps games against, so it stays yours:")
        print(f"      git add data/decks/{slug} && \\")
        print(f"        git commit -m \"{slug}: merge branch {branch}\"")


#: Verbs that change what the DECK's `info.json` says about its branches.
#: `merge` is absent on purpose — it already runs `regen.run(slug=…)`, which
#: includes the `deck-info` stage.
MUTATES_THE_DOSSIER = frozenset(
    {"new", "stage", "unstage", "commit", "propose", "withdraw", "reject", "delete", "summary",
     "objective"})


def reaim(slug, branch, raw, why=None):
    """RE-AIM AN OPEN BRANCH (2026-10-21). `new` fixes the objective on the day the
    branch opens, and the day sharknado's finisher-v1 was staged the row it aimed at
    moved the wrong way BY CONSTRUCTION — "finisher wheel by T8" counts games where a
    wheel landed on a commander, so cutting wheels lowers it even when every wheel
    left is a better one. A pre-registered line is still the rule: this is refused
    once a proposal exists (the pilot accepted that line) or the branch merged, the
    previous line is kept in `objective_history` with the date, and `net-change
    --write` must be re-run to grade against the new one. Never on a decided branch.
    """
    doc = meta(slug, branch)
    if not doc:
        raise SystemExit(f"{slug}: no branch {branch!r}")
    if doc.get("merged"):
        raise SystemExit(f"{slug}/{branch} is merged — its objective is history")
    if doc.get("proposal"):
        raise SystemExit(f"{slug}/{branch} is proposed against its current objective — "
                         f"`withdraw` first if the line is wrong")
    if not raw:
        raise SystemExit("`--objective \"<axis> <op> <n>[ @pod]\"` is required")
    if not why:
        raise SystemExit("`--why \"…\"` is required — a moved pre-registered line says why it moved")
    objective = parse_objective(raw)
    objective["why"] = why
    was = doc.get("objective")
    # The shape `validate_branch` holds the key to: {at, was, why_changed}. Until
    # 2026-10-25 this wrote the old line flat with `replaced_at`, which the gate
    # refused, so every re-aimed branch failed its own gate (finisher-v1, answers-v1).
    doc.setdefault("objective_history", []).append(
        {"at": datetime.date.today().isoformat(), "was": was or {}, "why_changed": why})
    doc["objective"] = objective
    _write_meta(slug, branch, doc)
    return {"objective": objective, "was": was}


def refresh_dossier(slug):
    """Rewrite `info.json` after a branch write, on any live deck that has one.

    A branch is a directory, and `deck_info.compose` reads every one of them:
    the `branches` block and the derived `next` line both move when a branch is
    opened, staged, committed, proposed or deleted. So a branch write makes the
    DECK's tracked dossier stale without touching the deck's own list — and on
    2026-09-12 that is exactly what happened: `elenda-v1` was opened on
    edgar-vampires and `test_info_json_matches_a_fresh_run[edgar-vampires]` went
    red with the champion's 99 unchanged, two keys out of date.

    `merge` already learned this lesson and its own comment states it — "a merge
    left `diagnostic.json`, `benchmark.json` and `info.json` describing the
    PREVIOUS 99, and the way anyone found out was a failing test hours later."
    The cheap verbs never learned it.

    ONLY THE DOSSIER, NOT `regen.run`. Nothing a branch write does moves the
    deck's own measurements, so re-running the goldfish would cost seconds to
    reproduce identical bytes. `deck_info.compose` is 2.4s cold and 0.8s warm,
    measured, which is the whole added cost.

    WHEREVER `info.json` ALREADY EXISTS AND THE DECK IS NOT RETIRED — the rule
    `regen.targets` has applied since 2026-09-26 (REFRESH every live deck whose
    artifact exists; BOOTSTRAP only a sleeved one). This used to gate on
    `regen.is_pinned`, which left a BENCH deck's tracked, freshness-gated
    dossier stale after every branch opened or swap staged from the page —
    exactly the case the split was meant to stop. A missing `info.json` is still
    not created here: bootstrapping is regen's job, for sleeved decks.

    Public because `serve`'s `branch/new` and `branch/stage` call it too: the
    workbench writes branches without going through `main`.
    """
    import argparse

    from manamap.pilot import deck_info, regen

    try:
        base = deck_dir(slug)
    except FileNotFoundError:
        return
    if not (base / "info.json").exists() or not (base / "cards.json").exists():
        return
    if regen.is_retired(slug):
        return
    deck_info.main(argparse.Namespace(slug=slug, write=True, as_json=False,
                                      verify=False, branch=None))
    print(f"  rewrote data/decks/{slug}/info.json (the branch set changed)")


_refresh_dossier = refresh_dossier     # the old private name, kept for callers


def main(args):
    _dispatch(args)
    if getattr(args, "action", None) in MUTATES_THE_DOSSIER:
        refresh_dossier(args.slug)
