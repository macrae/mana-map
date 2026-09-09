"""One turn of Sven: ask, run tools, narrate, answer.

A generator of wire frames rather than a function returning a string, because
the first token has to reach the terminal before the turn is finished. Every
caller — the CLI, the daemon endpoint, the tests — consumes the same frames, so
there is one description of what a turn emits.

THE ESCALATION IS A TOOL, NOT A CLASSIFIER. Sven runs on Haiku and decides for
himself when a question is beyond him; the turn is then re-run on Sonnet with
the tool results already gathered, so it costs one more step rather than a
restart. A heuristic guessing difficulty from the question text would be a
second thing to maintain, and it would be wrong in ways nobody could see.

The charter is read PER TURN through `mtime_memo`, so editing `prompt.md`
reaches the next question with no restart. That matters because the charter is
the part that will be edited daily and the loop is the part that will not.
"""

import json
from pathlib import Path

from manamap.pilot.common import mtime_memo
from manamap.sven import cache, core, llm, spend, tools

PROMPT_PATH = Path(__file__).with_name("prompt.md")

#: A turn that has run this many tool rounds stops and says what it has.
#: Not a safety net against a hostile model — a bound on a question that cannot
#: be answered, so it fails in seconds rather than in dollars.
MAX_ROUNDS = 8


def charter():
    """The system prompt, re-read whenever `prompt.md` changes on disk."""
    return mtime_memo(PROMPT_PATH, "sven:prompt", PROMPT_PATH.read_text, absent="")


def _tool_result(session, name, payload):
    """Run one tool call and shape its result for the model.

    Errors come back as CONTENT, not exceptions: the model should see argparse's
    own message and correct itself, which is one round trip and self-teaching.
    Hiding the error would make it guess again with no new information.
    """
    if name == "run_command":
        argv = [payload.get("command", ""), *(payload.get("args") or [])]
        got = session.call("run_readonly", argv=argv)
        if "error" in got:
            return got["error"], True
        text = got.get("stdout", "")
        if got.get("exit"):
            text = f"(exit {got['exit']})\n{text}"
        return _cap(text), False
    if name == "command_help":
        try:
            return _cap(tools.command_help(payload.get("command", ""))), False
        except ValueError as exc:
            return str(exc), True
    if name in ("deck_state", "fleet"):
        # ADVERTISED AND UNREACHABLE was the state that produced Sven's first
        # wrong answer: `tool_block` listed four tools, the dispatcher knew six,
        # and this function knew three. He asked "is zur ready", could not reach
        # `deck_state`, fell back to `deck-status`, and reported LIFECYCLE
        # STAGES as promotion GATES — a confident wrong answer with the real
        # blocker (56 cards to buy) never surfaced.
        got = session.call(name, **payload)
        if isinstance(got, dict) and "error" in got:
            return got["error"], True
        return json.dumps(got, indent=2, default=str), False
    if name == "stats":
        got = session.call("stat_test", kind=payload.get("fn"),
                           **(payload.get("args") or {}))
        if isinstance(got, dict) and "error" in got:
            return got["error"], True
        return json.dumps(got, indent=2, default=str), False
    return f"{name!r} is not a tool you have.", True


def _cap(text):
    if len(text) <= tools.TOOL_RESULT_CAP:
        return text
    head = text[:tools.TOOL_RESULT_CAP]
    dropped = text[tools.TOOL_RESULT_CAP:].count("\n") + 1
    return (f"{head}\n… [truncated — {dropped} more lines. Narrow it with "
            f"--limit, or ask for the specific thing you need.]")


def _label(name, payload):
    """What the pilot sees while a tool runs. A verb and a subject, no JSON."""
    if name == "run_command":
        argv = " ".join([payload.get("command", ""), *(payload.get("args") or [])])
        return f"reading {argv.strip()}"
    if name == "command_help":
        return f"checking {payload.get('command', '')} flags"
    if name == "deck_state":
        return f"reading {payload.get('slug', '?')}'s rung and gates"
    if name == "fleet":
        return "reading the fleet"
    if name == "stats":
        return f"computing {payload.get('fn', 'a statistic')}"
    if name == "escalate":
        return "this needs a closer look — escalating"
    return name


def run(question, *, turn=None, session=None, model=None, use_cache=True):
    """Yield `(kind, data)` frames for one question. The only entry point.

    `turn` is an `llm.Turn`; pass a `ScriptedTurn` to run with no key. `session`
    carries the fact cache between questions, which is what makes the second
    question about a deck instant.
    """
    session = session or core.Session()
    session.reset_turn()
    model = model or llm.FAST_MODEL

    # THE ANSWER CACHE IS CHECKED BEFORE THE MODEL IS EVEN CONSTRUCTED, which is
    # what makes a repeat question free rather than merely fast — no transport,
    # no key, no request.
    #
    # And it ALWAYS SAYS SO. `tests/conftest.py` made the same call for the
    # regenerate-and-compare cache — "hits are counted and printed, never
    # silent" — because a cache you cannot see is one you cannot trust. A pilot
    # who suspects a stale answer and has no way to check will stop believing
    # the fast ones too.
    if use_cache:
        stored = cached_answer(question, model=model)
        if stored:
            yield ("text", stored["answer"])
            yield ("note", "cached answer — nothing it read has changed "
                           "(--no-cache to re-ask)")
            yield ("done", {"summary": "served from cache", "model": model,
                            "escalated": False, "cached": True,
                            "cacheable": True, "uncacheable": [],
                            "tool_calls": stored.get("tool_calls", 0),
                            "failed": 0, "paths_touched": len(stored.get("touched") or []),
                            "facts": session.facts.stats()})
            return

    # THE CEILING IS CHECKED BEFORE THE TRANSPORT EXISTS. An overspend you
    # learn about afterwards is one you have already paid for, and the account
    # has no auto-reload.
    try:
        spend.check()
    except spend.BudgetExceeded as exc:
        yield ("error", str(exc))
        yield ("done", {"summary": "refused — budget ceiling", "model": model,
                        "escalated": False, "cacheable": False,
                        "uncacheable": ["budget"], "tool_calls": 0,
                        "failed": 0, "paths_touched": 0,
                        "facts": session.facts.stats()})
        return

    turn = turn or llm.default_turn()
    messages = [{"role": "user", "content": question}]
    tool_block = tools.tool_block()
    escalated = False
    answer = []
    spent = 0.0

    for round_no in range(MAX_ROUNDS):
        text_parts, calls, stop = [], [], None
        for event in turn.stream(system=charter(), messages=messages,
                                 tools=tool_block, model=model):
            kind = event.get("type")
            if kind == "text":
                text_parts.append(event["text"])
                yield ("text", event["text"])
            elif kind == "tool_use":
                calls.append(event)
            elif kind == "end":
                stop = event
                # The SDK hands usage back and this used to drop it on the
                # floor — `stop` was assigned and never read, so there was no
                # way to answer "what has Sven cost" except to open the console.
                turn_cost = spend.record(model, event.get("usage"), question)
                if turn_cost is not None:
                    spent += turn_cost

        if text_parts:
            answer.append("".join(text_parts))
        if not calls:
            break

        # An escalation ends this model's turn and re-runs the next round on the
        # stronger one, carrying everything already read.
        if any(c["name"] == "escalate" for c in calls) and not escalated:
            reason = next(c["input"].get("reason", "") for c in calls
                          if c["name"] == "escalate")
            escalated = True
            model = llm.DEEP_MODEL
            yield ("note", f"escalating to {model}: {reason}")

        assistant = [{"type": "tool_use", "id": c["id"], "name": c["name"],
                      "input": c["input"]} for c in calls]
        if text_parts:
            assistant.insert(0, {"type": "text", "text": "".join(text_parts)})
        messages.append({"role": "assistant", "content": assistant})

        results = []
        for call in calls:
            if call["name"] == "escalate":
                results.append({"type": "tool_result", "tool_use_id": call["id"],
                                "content": "Escalated. Answer with what you have."})
                continue
            yield ("tool", _label(call["name"], call["input"]))
            content, is_error = _tool_result(session, call["name"], call["input"])
            if is_error:
                yield ("error", content.splitlines()[0][:200])
            results.append({"type": "tool_result", "tool_use_id": call["id"],
                            "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        yield ("note", f"stopped after {MAX_ROUNDS} rounds of tool calls")

    final = "".join(answer)
    stats = session.stats()
    if use_cache and session.cacheable() and final.strip():
        # `touched` is stored, not just the signature it produced. A lookup has
        # to RECOMPUTE the signature from the same paths to know whether they
        # have moved since — storing only the digest would let us compare an old
        # digest against nothing.
        cache.answer_put(question, session.touched_signature(), model, final,
                         meta={"tool_calls": stats["tool_calls"],
                               "touched": sorted(session.touched)})
    yield ("done", {"summary": _summary(stats, model, escalated, spent),
                    "model": model, "escalated": escalated,
                    "estimated_usd": round(spent, 6), **stats})


def _summary(stats, model, escalated, spent=0.0):
    bits = [f"{stats['tool_calls']} tool call(s)"]
    facts = stats.get("facts") or {}
    if facts.get("hits"):
        bits.append(f"{facts['hits']} from cache")
    bits.append(model.split("-")[1] if "-" in model else model)
    if escalated:
        bits.append("escalated")
    if not stats["cacheable"] and stats["uncacheable"]:
        bits.append(f"not cached ({stats['uncacheable'][0]})")
    if spent:
        # Shown every turn, not on request. A cost you have to go and look up is
        # one you look up after it matters.
        bits.append(f"~${spent:.4f}")
    return " · ".join(bits)


def cached_answer(question, session=None, model=None):
    """A previously stored answer for this question, or None.

    Checked BEFORE a turn runs, which means the signature has to be computed
    from what the LAST answer touched rather than from what this one will. That
    is why the stored record carries its own signature: we recompute it and
    compare, so an answer whose inputs moved is a miss rather than a stale hit.
    """
    model = model or llm.FAST_MODEL
    for stored in _recent(question, model):
        touched = stored.get("touched")
        if touched is None:                 # a turn that read no file at all
            if stored.get("signature") == "no-files-read":
                return stored
            continue
        if cache.signature(touched) == stored.get("signature"):
            return stored
    return None


def _recent(question, model):
    """Stored answers to this question, newest first. Small by construction."""
    out = []
    for path in sorted(cache.SVEN_CACHE_DIR.glob("*.json"),
                       key=lambda p: -p.stat().st_mtime):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("model") == model and " ".join(
                (doc.get("question") or "").lower().split()) == " ".join(
                question.lower().split()):
            out.append(doc)
    return out
