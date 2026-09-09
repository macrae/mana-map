"""A turn of Sven's, with no model in it.

Everything here runs, and is tested, with no API key and no network. The model
is a face on top of this — `sven/llm.py` decides WHICH of these to call and how
to word the result; this module decides what they return and what that answer
depended on. Keeping the split sharp is what lets CI prove Sven's facts without
ever making a request.

THE ONE IDEA: a `Session` watches every tool call and accumulates the paths
those calls depended on. At the end of a turn `touched_signature()` is a digest
over exactly that set, and it is the second half of the answer cache's key. An
answer therefore invalidates itself when anything it looked at changes, without
anyone maintaining a list of what invalidates what.

Prewarming lives here too, for the same reason it exists: the first question of
a session should be as fast as the tenth, and the pilot should not be the one
paying to warm the process.
"""

import time

from manamap.sven import cache, tools


class Session:
    """One conversation's fact cache and dependency ledger.

    Long-lived: in the daemon a `Session` is created once and reused, so the
    fact cache spans questions. That is the point — the second question about a
    deck is answered from the first question's reads.
    """

    def __init__(self):
        self.facts = cache.FactCache()
        self.touched = set()
        self.calls = []
        # Why this turn may not be cached, if it may not be. A LIST OF REASONS
        # rather than a bool, so the client can say which one and the pilot is
        # never left wondering why a repeat question cost a model call.
        self.uncacheable = []

    # ── running a tool ────────────────────────────────────────────────────

    def call(self, name, **kw):
        """Run one tool, record what it depended on, and time it.

        A tool that RAISES is recorded too. Sven is told to report a refusal
        rather than route around it, and a refusal he cannot see is one he will
        route around.
        """
        started = time.time()
        try:
            value = self._dispatch(name, kw)
            error = None
        except Exception as exc:                   # noqa: BLE001 - reported, not swallowed
            value, error = None, f"{exc.__class__.__name__}: {exc}"
        entry = {"tool": name, "args": kw, "ms": round((time.time() - started) * 1000),
                 "error": error}
        self.calls.append(entry)
        if error:
            # A failed tool means the answer was built on less than it asked
            # for. Caching that would freeze a transient failure into a
            # permanent answer.
            self.uncacheable.append(f"{name} failed: {error}")
            return {"error": error}
        return value

    def _dispatch(self, name, kw):
        if name == "run_readonly":
            argv = [str(a) for a in (kw.get("argv") or [])]
            head = argv[1] if argv[:1] == ["pilot"] else (argv[0] if argv else "")
            if head in tools.VOLATILE:
                self.uncacheable.append(
                    f"{head} is true only at the moment it is asked")
            self.touched.update(str(p) for p in tools.depends_on(argv))
            return tools.run(argv, facts=self.facts)
        if name == "deck_state":
            # RESOLVE FIRST, then record what was touched. The other order meant
            # the dependency walk saw the pilot's shorthand ("zur") rather than
            # the deck ("zur-enchantress"), so the answer was keyed on the wrong
            # directory even when it worked.
            slug = tools.resolve_slug(kw["slug"])
            self.touched.update(str(p) for p in tools.depends_on(["deck-info", slug]))
            return tools.deck_state(slug, facts=self.facts)
        if name == "fleet":
            self.touched.update(str(p) for p in tools.depends_on(["deck-status"]))
            return tools.fleet(facts=self.facts)
        if name == "stat_test":
            # Pure arithmetic over numbers already in hand — depends on no file,
            # so it contributes nothing to the signature. That is correct and
            # worth stating: a Wilson interval over (27, 120) is the same
            # interval forever.
            return tools.stat_test(**kw)
        raise ValueError(
            f"{name!r} is not a tool Sven has. Available: "
            + ", ".join(["run_readonly", *sorted(tools.TIER1)]))

    # ── what the turn depended on ─────────────────────────────────────────

    def cacheable(self):
        """Whether this turn's answer may be stored. Reasons in `uncacheable`."""
        return not self.uncacheable

    def touched_signature(self):
        """A digest over every path this turn's tools read, or None if none did.

        `None` means the turn touched no file — a pure statistics question, say.
        Those are cacheable forever, and the caller signals that by passing the
        literal string below rather than a digest, so "depended on nothing" and
        "we forgot to record" can never look alike.
        """
        if not self.touched:
            return "no-files-read"
        return cache.signature(self.touched)

    def stats(self):
        return {"cacheable": self.cacheable(),
                "uncacheable": list(self.uncacheable),
                "tool_calls": len(self.calls),
                "failed": sum(1 for c in self.calls if c["error"]),
                "paths_touched": len(self.touched),
                "facts": self.facts.stats()}

    def reset_turn(self):
        """Clear the per-turn ledger, keeping the fact cache. Between questions."""
        self.touched.clear()
        self.calls.clear()
        self.uncacheable.clear()


def prewarm(session=None, decks=None):
    """Precompute the fleet snapshot so question #1 is as fast as question #10.

    Runs at daemon boot on a background thread. Every failure is swallowed on
    purpose: a prewarm is an optimisation, and one that can break startup is a
    liability. What it cannot do is hide a problem — the returned report says
    what failed, and `mm ask --status` prints it.
    """
    session = session or Session()
    report = {"decks": [], "errors": [], "seconds": None}
    started = time.time()
    try:
        rows = tools.fleet(facts=session.facts)
        names = decks or [r.get("slug") for r in rows if r.get("slug")]
    except Exception as exc:                       # noqa: BLE001
        report["errors"].append(f"fleet: {exc.__class__.__name__}: {exc}")
        names = decks or []
    for slug in names:
        try:
            tools.deck_state(slug, facts=session.facts)
            report["decks"].append(slug)
        except Exception as exc:                   # noqa: BLE001
            report["errors"].append(f"{slug}: {exc.__class__.__name__}: {exc}")
    report["seconds"] = round(time.time() - started, 2)
    return report
