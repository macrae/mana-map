"""ONE API. Two transports. No second list.

Sven reaches the bench through the Anthropic tool protocol; Claude Code reaches
it through MCP. Those are two wire formats for the same set of capabilities, and
until this module existed they were two HAND-WRITTEN LISTS that had already
drifted: the same tool was `run_command` in one and `run_readonly` in the other,
`stats` in one and `stat_test` in the other, and two capabilities existed on one
side only. A question Claude Code could answer, Sven could not.

So the capability is declared once, here, with its schema, its handler and the
rules for reading its result. `anthropic_block()` and `mcp_block()` are
RENDERINGS — they differ only in the spelling of one key (`input_schema` versus
`inputSchema`) — and `call()` is the single dispatcher. A test asserts the two
renderings carry the same names, because that is the property that decayed.

THE ANSWER CONTRACT, which is the point of centralising this at all:

  1. NEVER RETURN A BARE NUMBER THAT INVITES A COMPARISON. A win rate arrives
     with the comparisons already computed as intervals on the DIFFERENCE.
     Asked whether 25.2% was good, a model told to be careful with arithmetic
     answered "well below functional in a four-player pod" — where par is 25%.
  2. NEVER RETURN A BOOLEAN TO NARRATE. Handed `excludes_zero: false` the same
     model wrote "it excludes zero" about an interval that spans it. Booleans
     get a `reading`: a sentence to quote.
  3. NEVER LEAVE A GAP UNNAMED. Every partial payload carries `not_included` —
     what it does not cover and the tool that does. A gap is where a model
     narrates from nothing: asked about zur it invented "you died by turn 5-6"
     against a recorded median of 34.
  4. NEVER MAKE THE CALLER JOIN TWO FACTS TO SPOT A PROBLEM. The commander a
     deck declares and the commander Forge played arrive together, because six
     of zur's eight runs used the wrong one and nothing noticed for weeks.

Those four are why the answer to "how do we stop it inferring" is not a firmer
prompt. It is a payload with nothing left to infer.
"""

TOOLS = []


def tool(name, description, schema, handler, transports=("sven", "mcp")):
    TOOLS.append({"name": name, "description": description, "schema": schema,
                  "handler": handler, "transports": transports})


def _obj(**properties):
    required = [k for k, v in properties.items() if v.pop("_required", False)]
    return {"type": "object", "properties": properties,
            **({"required": required} if required else {})}


# ── the capabilities ──────────────────────────────────────────────────────

def _deck_state(slug, facts=None, **_):
    from manamap.sven import tools
    return tools.deck_state(slug, facts=facts)


def _fleet(facts=None, **_):
    from manamap.sven import tools
    return tools.fleet(facts=facts)


def _search(corpus, query, k=None, **_):
    from manamap.pilot import retrieve
    return [{"id": cid, "score": round(score, 4), "title": rec.get("title"),
             "source": rec.get("source"), "text": rec["text"]}
            for cid, rec, score in retrieve.search(corpus, query, k=k)]


def _stats(fn, args=None, **_):
    from manamap.sven import tools
    return tools.stat_test(fn, **(args or {}))


def _run_command(command, args=None, facts=None, **_):
    from manamap.sven import tools
    return tools.run([command, *(args or [])], facts=facts)


def _command_help(command, **_):
    from manamap.sven import tools
    return {"help": tools.command_help(command)}


def _register():
    from manamap.sven import tools

    commands = sorted(name for name, _ in tools.readonly_commands())
    table = "\n".join(f"  {n:16s}{d}" for n, d in sorted(tools.readonly_commands()))

    tool("deck_state",
         "WHERE ONE DECK STANDS, and the first call for any question about a "
         "single deck. Its rung on the dev -> bench -> sleeved ladder, the "
         "promotion gates it meets and which are blocking, the derived next "
         "action, its commander and composition, and its MEASURED FIGURES — the "
         "simulation with its win rate, interval, and the comparisons against "
         "par and against this table's null already computed.\n\n"
         "Read `simulation.comparisons.*.reading` as a sentence; do not restate "
         "it from `excludes_zero`. `simulation.stale` means the run played an "
         "older list — say so before quoting it. `simulation.runs_warning` means "
         "stored runs used the WRONG COMMANDER. `band` means the deck has no "
         "single kill number and both ends must be quoted. `not_included` names "
         "what this does not cover and which tool does.",
         _obj(slug={"type": "string", "_required": True}),
         _deck_state)

    tool("fleet",
         "Every deck, one row each, with its rung. The answer to 'what should I "
         "work on' and to anything spanning more than one deck.",
         _obj(), _fleet)

    tool("search_docs",
         "Semantic search over this repo's own docs — the 'why did we do it "
         "this way' corpus, including 121 KB of measurements this project has "
         "already paid for. Reach for it BEFORE changing a matcher, a metric or "
         "a validator: the answer is usually written down with the number it "
         "cost.",
         _obj(query={"type": "string", "_required": True},
              k={"type": "integer"}),
         lambda query, k=None, **_: _search("docs", query, k))

    tool("search_code",
         "Semantic search over src/. Retrieves 'what does this module do' well "
         "and 'where is X built' poorly — prefer grep when you know the symbol.",
         _obj(query={"type": "string", "_required": True},
              k={"type": "integer"}),
         lambda query, k=None, **_: _search("code", query, k))

    tool("stats",
         "Run a statistical test and get its exact return value. USE THIS "
         "RATHER THAN COMPUTING OR PARAPHRASING ONE — never state an interval, "
         "a power figure or an MDE in your own arithmetic.\n"
         "  wilson(k, n)                          a rate's 95% interval\n"
         "  diff_proportions(k_a, n_a, k_b, n_b)  Newcombe: the interval ON THE "
         "DIFFERENCE, the only correct answer to 'is this better'\n"
         "  diff_means(xs, ys)                    Welch\n"
         "  diff_medians(xs, ys)                  bootstrap, seeded\n"
         "  permutation_p(xs, ys)                 two-sided, seeded\n"
         "  power(p_a, p_b, n_a, n_b)             exact, not an approximation\n"
         "  mde(p_a, n_a)                         smallest detectable effect\n"
         "  games_needed(p_a, difference)         n per arm for 80% power",
         _obj(fn={"type": "string", "_required": True},
              args={"type": "object", "_required": True}),
         _stats)

    tool("run_command",
         "Run one read-only pilot command and get its terminal output. `args` is "
         "the argv after the command name. Refuses anything that writes, "
         "including a read-only command given --write. Prefer a shaped tool "
         "above when one fits — every wrong answer on record came from choosing "
         "a command and interpreting its prose when a tool held the value.\n\n"
         + table,
         _obj(command={"type": "string", "enum": commands, "_required": True},
              args={"type": "array", "items": {"type": "string"}}),
         _run_command)

    tool("command_help",
         "The full --help for one command. Progressive disclosure: read the "
         "flags for the one command you are about to use rather than guessing. "
         "A wrong flag costs a round trip; an invented one costs trust.",
         _obj(command={"type": "string", "enum": commands, "_required": True}),
         _command_help)

    tool("escalate",
         "Hand this question to a stronger model. Call it when the question "
         "needs judgement rather than lookup, when evidence conflicts, or when "
         "you are about to state a statistical conclusion. Everything already "
         "read is carried over, so it costs one step and not a restart.",
         _obj(reason={"type": "string", "_required": True}),
         None, transports=("sven",))


def _ensure():
    if not TOOLS:
        _register()
    return TOOLS


def anthropic_block():
    """Sven's tool list. `input_schema`, and includes `escalate`."""
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["schema"]}
            for t in _ensure() if "sven" in t["transports"]]


def mcp_block():
    """Claude Code's tool list. `inputSchema`, and no `escalate` — there is
    nothing above Claude Code to escalate to."""
    return [{"name": t["name"], "description": t["description"],
             "inputSchema": t["schema"]}
            for t in _ensure() if "mcp" in t["transports"]]


def call(name, arguments, facts=None):
    """Run one capability. Raises `ValueError` for a name that is not one.

    `facts` is the caller's fact cache, threaded through so a long-lived caller
    — the daemon's Sven session, the MCP server — gets the second question about
    a deck answered out of the first question's reads.
    """
    for entry in _ensure():
        if entry["name"] == name and entry["handler"] is not None:
            return entry["handler"](facts=facts, **(arguments or {}))
    raise ValueError(
        f"unknown tool {name!r}. Available: "
        + ", ".join(sorted(t["name"] for t in _ensure() if t["handler"])))
