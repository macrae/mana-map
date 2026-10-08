"""THE HYPOTHESIS QUEUE (PRD v2 Phase 2, 2026-10-07): pilot feedback turned into
checked, testable claims, and what testing them found.

`data/queue.jsonl` is ONE fleet-wide, APPEND-ONLY file — the decisions ledger's
discipline, across every deck, because "what's in the queue?" is one question.
An item is a `hypothesis` line with an id (`Q001`…); every later line points at
it with `of`. STATE IS DERIVED, NEVER STORED (`state_of`), the rule
`sim/campaign.py` set: a line records what happened, and the state is what the
lines add up to.

THE LOOP. The Incubation Pod proposes (`hypothesis`) → the Challenger argues
against each one, ONE ROUND (`challenge`: promote / drop / revise) → a `revise`
gets one rebuttal from the pod (`rebut`: revise or withdraw) → a revision gets the
Challenger's SECOND LOOK (`recheck`: yes / no, 2026-10-07 — Sean's call; until then
a revision promoted on the pod's word alone, and two of the first four were later
killed) → it is promoted or dropped. The recheck is a glance, not a second round:
it cannot ask for another revision. `apply` writes the promote / drop that follows from those lines, so the
single round cannot be stretched by hand. Jarvis then tests a promoted item with
the lightest method that settles it (`result`) and Sean makes the call
(`decide`). Sean can reorder (`rank`) or remove (`kill`) anything at any time, and
add a claim of his own (`add`): it enters as a hypothesis `by: sean` and still
faces the Challenger — an item does not skip the round because of who wrote it.

Agents never write this file: they write a draft under `.agent-out/`, and
`queue apply <draft>` checks every transition before a line is appended.

ORDER is Sean's latest `rank`, then decks by how recently they were PLAYED (the
newest captain's-log entry), then age. An item no line has touched in
`EXPIRE_DAYS` reads EXPIRED — derived from the dates, never written, and any new
line on it revives it.
"""

import datetime
import json

from manamap import config
from manamap.pilot.common import deck_dir

PATH = config.DATA_DIR / "queue.jsonl"
KINDS = ("hypothesis", "challenge", "rebut", "recheck", "promote", "drop", "result",
         "decide", "rank", "kill")
#: How a hypothesis says it would be settled — the PRD's "lightest method".
METHODS = ("context", "argument", "data", "try", "rules", "strategy", "scenario")
#: Methods that cannot run yet. `scenario` left it on 2026-10-08, when scenario slices
#: (`manamap pilot scenario-ab`, the `scenario-sim` agent) became buildable.
METHODS_NOT_YET = {}
VERDICTS = ("promote", "drop", "revise")
RECHECK = ("yes", "no")
RESULT_VERDICTS = ("supported", "refuted", "inconclusive")
DECISIONS = ("stage", "drop", "watch", "more")
STATES = ("INCUBATING", "CHALLENGED", "PROMOTED", "DROPPED", "TESTED", "DECIDED",
          "KILLED", "EXPIRED")
LIVE = ("INCUBATING", "CHALLENGED", "PROMOTED", "TESTED")
CLOSED = ("DROPPED", "DECIDED", "KILLED")
EXPIRE_DAYS = 14
SHOW_PROMOTED = 10


# ── the file ──────────────────────────────────────────────────────────────────

def read(path=None):
    p = path or PATH
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{p}:{n}: not JSON ({exc}) — the queue is append-only; fix the line by hand")
    return out


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def next_id(lines):
    n = max((int(e["id"][1:]) for e in lines if e.get("kind") == "hypothesis"), default=0)
    return f"Q{n + 1:03d}"


def _write(lines, new, path=None):
    p = path or PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for e in new:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return new


# ── state ─────────────────────────────────────────────────────────────────────

def _day(at):
    return datetime.date.fromisoformat(str(at)[:10])


def items(lines):
    """`{id: hypothesis line}` in file order."""
    return {e["id"]: e for e in lines if e.get("kind") == "hypothesis"}


def history(qid, lines):
    return [e for e in lines if e.get("id") == qid or e.get("of") == qid]


def raw_state(hist):
    """The state the lines add up to, before expiry."""
    kinds = [e["kind"] for e in hist]
    last_decide = next((e for e in reversed(hist) if e["kind"] == "decide"), None)
    if "kill" in kinds:
        return "KILLED"
    if "drop" in kinds:
        return "DROPPED"
    if last_decide is not None:
        # `more` sends it back to be tested again; a result after it counts anew.
        after = hist[hist.index(last_decide) + 1:]
        if last_decide.get("decision") != "more":
            return "DECIDED"
        return "TESTED" if any(e["kind"] == "result" for e in after) else "PROMOTED"
    if "result" in kinds:
        return "TESTED"
    if "promote" in kinds:
        return "PROMOTED"
    if "challenge" in kinds:
        return "CHALLENGED"
    return "INCUBATING"


def state_of(qid, lines, today=None):
    """The derived state. `today=None` skips expiry (a rendering that must not
    change by itself overnight — the Deck Context's block)."""
    hist = history(qid, lines)
    if not hist:
        raise KeyError(qid)
    state = raw_state(hist)
    if today is not None and state in LIVE:
        if (today - _day(hist[-1]["at"])).days >= EXPIRE_DAYS:
            return "EXPIRED"
    return state


def current(qid, lines):
    """The hypothesis as it stands: a rebuttal's revision replaces its fields."""
    item = dict(items(lines)[qid])
    for e in history(qid, lines):
        if e["kind"] == "rebut" and e.get("action") == "revise":
            item.update({k: e[k] for k in ("claim", "expected_effect", "test") if e.get(k)})
    return item


def last_played():
    """`{slug: date}` of each deck's newest captain's-log entry."""
    from manamap.pilot import deck_notes

    out = {}
    if not config.DECKS_DIR.is_dir():
        return out
    for d in config.DECKS_DIR.iterdir():
        if d.is_dir() and (d / "log.jsonl").exists():
            try:
                log = deck_notes.read_log(d.name)
            except SystemExit:
                continue
            if log:
                out[d.name] = max(_day(e["at"]) for e in log if e.get("at"))
    return out


def ordered(lines, played=None):
    """Item ids in queue order: Sean's latest `rank`, then decks by how recently
    they were played, then age (oldest first)."""
    played = last_played() if played is None else played
    rank = next((e["order"] for e in reversed(lines) if e["kind"] == "rank"), [])
    pos = {q: i for i, q in enumerate(rank)}
    its = items(lines)
    never = datetime.date.min

    def key(qid):
        it = its[qid]
        return (0, pos[qid]) if qid in pos else (1, -played.get(it["deck"], never).toordinal(),
                                                 str(it["at"]), qid)
    return sorted(its, key=key)


# ── transitions ───────────────────────────────────────────────────────────────

def check(line, lines):
    """Why `line` may not be appended after `lines`, or None. The ONE statement of
    the legal transitions — `apply`, the Sean verbs and the validator all call it."""
    kind = line.get("kind")
    if kind not in KINDS:
        return f"unknown kind {kind!r}"
    if not line.get("at"):
        return "no `at`"
    its = items(lines)
    if kind == "rank":
        order = line.get("order") or []
        if not order:
            return "rank names no items"
        bad = [q for q in order if q not in its]
        if bad:
            return f"rank names unknown item(s) {', '.join(bad)}"
        if len(set(order)) != len(order):
            return "rank names an item twice"
        return None
    if kind == "hypothesis":
        for k in ("claim", "expected_effect", "deck"):
            if not str(line.get(k) or "").strip():
                return f"a hypothesis needs `{k}`"
        test = line.get("test") or {}
        if test.get("method") not in METHODS:
            return f"test.method must be one of {', '.join(METHODS)}"
        if test.get("method") in METHODS_NOT_YET:
            return METHODS_NOT_YET[test["method"]]
        if not str(test.get("how") or "").strip():
            return "test.how: say how the method would settle it"
        if not deck_dir(line["deck"]).is_dir():
            return f"no deck {line['deck']!r}"
        if line.get("id") != next_id(lines):
            return f"id {line.get('id')!r} is not the next id ({next_id(lines)})"
        return None

    qid = line.get("of")
    if qid not in its:
        return f"`of` {qid!r} names no item"
    hist = history(qid, lines)
    state = raw_state(hist)
    kinds = [e["kind"] for e in hist]
    if state in CLOSED and kind != "kill":
        return f"{qid} is {state} — closed"
    if kind == "kill":
        if state in ("KILLED",):
            return f"{qid} is already killed"
        return None if str(line.get("reason") or "").strip() else "a kill needs a reason"
    if kind == "challenge":
        if "challenge" in kinds:
            return f"{qid} has had its challenge — one round only"
        if line.get("verdict") not in VERDICTS:
            return f"a challenge's verdict is one of {', '.join(VERDICTS)}"
        return None if str(line.get("reason") or "").strip() else "a challenge needs a reason"
    if kind == "rebut":
        ch = next((e for e in hist if e["kind"] == "challenge"), None)
        if ch is None or ch.get("verdict") != "revise":
            return f"{qid}: only a `revise` challenge gets a rebuttal"
        if "rebut" in kinds:
            return f"{qid} has had its rebuttal — one round only"
        if line.get("action") not in ("revise", "withdraw"):
            return "a rebuttal's action is revise or withdraw"
        if line.get("action") == "revise":
            t = line.get("test") or {}
            if t and t.get("method") not in METHODS:
                return f"test.method must be one of {', '.join(METHODS)}"
            if t.get("method") in METHODS_NOT_YET:
                return METHODS_NOT_YET[t["method"]]
        return None if str(line.get("why") or "").strip() else "a rebuttal needs a why"
    if kind == "recheck":
        rb = next((e for e in hist if e["kind"] == "rebut"), None)
        if rb is None or rb.get("action") != "revise":
            return f"{qid}: only a revised hypothesis gets the second look"
        if "recheck" in kinds:
            return f"{qid} has had its second look — yes or no, once"
        if line.get("verdict") not in RECHECK:
            return "a recheck's verdict is yes or no — a glance, not another round"
        if line["verdict"] == "no" and not str(line.get("reason") or "").strip():
            return "a recheck that says no needs a reason"
        return None
    if kind in ("promote", "drop"):
        if state != "CHALLENGED":
            return f"{qid} is {state}: promote and drop follow the challenge round"
        if kind == "drop" and not str(line.get("reason") or "").strip():
            return "a drop needs a reason"
        return None
    if kind == "result":
        if state != "PROMOTED":
            return f"{qid} is {state}: a result follows a promote (or a `more` decision)"
        if line.get("verdict") not in RESULT_VERDICTS:
            return f"a result's verdict is one of {', '.join(RESULT_VERDICTS)}"
        if line.get("method") not in METHODS or line.get("method") in METHODS_NOT_YET:
            return "a result names the method it used"
        return None if str(line.get("answer") or "").strip() else "a result needs an answer"
    if kind == "decide":
        if state != "TESTED":
            return f"{qid} is {state}: Sean decides on a tested item"
        if line.get("decision") not in DECISIONS:
            return f"a decision is one of {', '.join(DECISIONS)}"
        return None
    return None


def append(new, lines=None, path=None):
    """Check each line against everything before it, then write all or nothing."""
    lines = list(read(path) if lines is None else lines)
    staged = list(lines)
    for e in new:
        e.setdefault("at", _now())
        why = check(e, staged)
        if why:
            raise SystemExit(f"FAIL queue: {why} — nothing was written")
        staged.append(e)
    return _write(lines, new, path)


def settle(qid, lines):
    """The promote / drop that the single challenge round implies, or None."""
    hist = history(qid, lines)
    if raw_state(hist) != "CHALLENGED":
        return None
    ch = next(e for e in hist if e["kind"] == "challenge")
    rb = next((e for e in hist if e["kind"] == "rebut"), None)
    if ch["verdict"] == "promote":
        return {"kind": "promote", "of": qid, "by": "jarvis", "why": "the challenger promoted it"}
    if ch["verdict"] == "drop":
        return {"kind": "drop", "of": qid, "by": "jarvis", "reason": f"challenger: {ch['reason']}"}
    if rb is None:
        return None                                     # waiting for the pod's one reply
    if rb["action"] == "withdraw":
        return {"kind": "drop", "of": qid, "by": "jarvis", "reason": f"withdrawn by the pod: {rb['why']}"}
    rc = next((e for e in hist if e["kind"] == "recheck"), None)
    if rc is None:
        return None                                     # waiting for the second look
    if rc["verdict"] == "no":
        return {"kind": "drop", "of": qid, "by": "jarvis",
                "reason": f"challenger's second look: {rc['reason']}"}
    return {"kind": "promote", "of": qid, "by": "jarvis",
            "why": "revised once after the challenge; the challenger's second look accepted it"}


def apply(draft, path=None):
    """A draft from `.agent-out/` → lines, then the settlement each implies."""
    lines = read(path)
    kind = draft.get("kind")
    new = []
    if kind == "incubation":
        staged = list(lines)
        for h in draft.get("hypotheses") or []:
            line = {"id": next_id(staged), "kind": "hypothesis", "deck": draft.get("deck"),
                    "by": draft.get("by", "incubation-pod"), "source": draft.get("source"),
                    "at": _now(), **{k: h.get(k) for k in ("claim", "expected_effect", "test", "why",
                                                           "ruled_out") if h.get(k) is not None}}
            new.append(line)
            staged.append(line)
    elif kind in ("challenge", "rebuttal", "recheck"):
        k = {"challenge": "challenge", "rebuttal": "rebut", "recheck": "recheck"}[kind]
        for it in draft.get("items") or []:
            new.append({"kind": k, "by": draft.get("by", "incubation-pod" if k == "rebut" else "challenger"),
                        **{x: v for x, v in it.items() if v is not None}})
    elif kind == "result":
        new.append({"kind": "result", "by": draft.get("by", "jarvis"),
                    **{x: draft.get(x) for x in ("of", "method", "answer", "evidence", "verdict")}})
    else:
        raise SystemExit("a draft's kind is incubation, challenge, rebuttal, recheck or result")
    if not new:
        raise SystemExit("the draft holds nothing to apply")
    written = append(new, lines, path)
    lines = lines + written
    settled = []
    for qid in dict.fromkeys(e.get("of") for e in written if e.get("of")):
        s = settle(qid, lines)
        if s:
            settled += append([s], lines, path)
            lines.append(s)
    return written + settled


def add(deck, claim, expected_effect, method, how, why=None, path=None):
    """Sean's own claim, straight into the queue as a hypothesis (INCUBATING).

    It goes through `check` like every other line, so it needs what a pod's
    hypothesis needs — a claim, its expected effect and the lightest test — and
    then waits for the Challenger like any other item."""
    lines = read(path)
    line = {"id": next_id(lines), "kind": "hypothesis", "deck": deck, "by": "sean",
            "source": "sean", "claim": claim, "expected_effect": expected_effect,
            "test": {"method": method, "how": how}}
    if why:
        line["why"] = why
    return append([line], lines, path)[0]


# ── reading it out ────────────────────────────────────────────────────────────

def rows(lines, today=None, deck=None, everything=False):
    today = today or datetime.date.today()
    out = []
    for qid in ordered(lines):
        it = current(qid, lines)
        if deck and it["deck"] != deck:
            continue
        st = state_of(qid, lines, today)
        if not everything and st not in LIVE:
            continue
        hist = history(qid, lines)
        res = next((e for e in reversed(hist) if e["kind"] == "result"), None)
        out.append({"id": qid, "deck": it["deck"], "state": st, "claim": it["claim"],
                    "expected_effect": it.get("expected_effect"),
                    "method": (it.get("test") or {}).get("method"),
                    "result": res and {k: res.get(k) for k in ("verdict", "answer", "method")},
                    "last": str(hist[-1]["at"])[:10]})
    return out


#: What Jarvis flags at the start of a deck conversation: a result waiting for
#: Sean's call, or an item gone quiet. Everything else waits until he asks.
WAITING = ("TESTED", "EXPIRED")


def waiting(lines, deck=None, today=None):
    """`[(qid, state, claim)]` for the items that need Sean, in queue order."""
    today = today or datetime.date.today()
    its = items(lines)
    out = []
    for qid in ordered(lines):
        if deck and its[qid]["deck"] != deck:
            continue
        st = state_of(qid, lines, today)
        if st in WAITING:
            out.append((qid, st, current(qid, lines)["claim"]))
    return out


def waiting_line(rows):
    """ONE line, or "" — the PRD's idle check is a nudge, never a report."""
    if not rows:
        return ""
    tested = [q for q, st, _ in rows if st == "TESTED"]
    expired = [q for q, st, _ in rows if st == "EXPIRED"]
    parts = []
    if tested:
        parts.append(f"{', '.join(tested)} tested, waiting for your call")
    if expired:
        parts.append(f"{', '.join(expired)} expired ({EXPIRE_DAYS} days untouched)")
    first = rows[0]
    hint = f' — {first[0]}: "{first[2][:80]}"' if len(rows) == 1 else ""
    return "queue: " + "; ".join(parts) + hint


def deck_block(slug, path=None):
    """The Deck Context's `queue` block: this deck's items, without expiry so the
    rendering does not change by itself overnight."""
    lines = read(path)
    mine = [q for q, it in items(lines).items() if it["deck"] == slug]
    if not mine:
        return "- Nothing queued for this deck yet. After a game, `/incubate` turns what you noticed into testable claims."
    out = []
    for qid in mine:
        st = state_of(qid, lines)
        if st in ("DROPPED", "KILLED"):
            continue
        it = current(qid, lines)
        res = next((e for e in reversed(history(qid, lines)) if e["kind"] == "result"), None)
        line = f"- **{qid}** {st.lower()} — {it['claim']}"
        if res:
            line += f" → _{res['verdict']}: {res['answer']}_"
        out.append(line)
    closed = sum(1 for q in mine if state_of(q, lines) in ("DROPPED", "KILLED"))
    if closed:
        out.append(f"- {closed} dropped or killed: `manamap pilot queue list --deck {slug} --all`")
    return "\n".join(out) or "- Nothing live in the queue for this deck."


def _print(rs):
    if not rs:
        print("the queue is empty — `/incubate` adds to it")
        return
    promoted = [r for r in rs if r["state"] == "PROMOTED"]
    hidden = {r["id"] for r in promoted[SHOW_PROMOTED:]}
    for r in rs:
        if r["id"] in hidden:
            continue
        print(f"{r['id']}  {r['state']:<10} {r['deck']:<16} {r['claim']}")
        if r["result"]:
            print(f"      → {r['result']['verdict']}: {r['result']['answer']}")
    if hidden:
        print(f"+{len(hidden)} more promoted — `manamap pilot queue list --all`")


def _refresh_contexts(qids):
    """A queue line changes that deck's `queue` block, so the Deck Context is
    refreshed here rather than left stale until the next regen."""
    from manamap.pilot import deck_context

    its = items(read())
    for slug in sorted({its[q]["deck"] for q in qids if q in its}):
        if deck_context.path(slug).exists():
            deck_context.refresh(slug)
            print(f"  context refreshed: {slug}")


def main(args):
    verb = getattr(args, "verb", None) or "list"
    rest = list(getattr(args, "rest", None) or [])
    lines = read()
    if verb == "list":
        rs = rows(lines, deck=getattr(args, "deck", None), everything=getattr(args, "all", False))
        print(json.dumps(rs, indent=2)) if getattr(args, "json", False) else _print(rs)
        return
    if verb == "waiting":
        line = waiting_line(waiting(lines, deck=getattr(args, "deck", None)))
        if line:
            print(line)
        return
    if verb == "show":
        if not rest or rest[0] not in items(lines):
            raise SystemExit("show Q### — an item id")
        qid = rest[0]
        out = {"item": current(qid, lines), "state": state_of(qid, lines, datetime.date.today()),
               "history": history(qid, lines)}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    if verb == "apply":
        if not rest:
            raise SystemExit("apply <draft.json>")
        with open(rest[0], encoding="utf-8") as f:
            draft = json.load(f)
        written = apply(draft)
        for e in written:
            print(f"  {e.get('id') or e.get('of')}  {e['kind']}"
                  + (f": {e.get('verdict') or e.get('reason') or e.get('claim') or ''}"[:100]))
        _refresh_contexts({e.get("id") or e.get("of") for e in written})
        return
    if verb == "add":
        e = add(getattr(args, "deck", None), getattr(args, "claim", None),
                getattr(args, "expect", None), getattr(args, "method", None),
                getattr(args, "how", None), why=getattr(args, "why", None))
        print(f"{e['id']}  added for {e['deck']} (INCUBATING): {e['claim']}")
        print(f"  next: the Challenger argues once — `/incubate` step 2 with {e['id']}")
        _refresh_contexts({e["id"]})
        return
    if verb == "rank":
        append([{"kind": "rank", "by": "sean", "order": rest}], lines)
        print("ranked: " + " ".join(rest))
        return
    if verb == "kill":
        if not rest:
            raise SystemExit("kill Q### --reason \"…\"")
        append([{"kind": "kill", "of": rest[0], "by": "sean", "reason": getattr(args, "reason", None)}], lines)
        print(f"killed {rest[0]}")
        _refresh_contexts({rest[0]})
        return
    if verb == "decide":
        if len(rest) < 2:
            raise SystemExit("decide Q### stage|drop|watch|more [--note \"…\"]")
        append([{"kind": "decide", "of": rest[0], "by": "sean", "decision": rest[1],
                 "note": getattr(args, "note", None)}], lines)
        print(f"{rest[0]}: {rest[1]}")
        _refresh_contexts({rest[0]})
        return
    raise SystemExit("verbs: list, waiting, show, add, apply, rank, kill, decide")
