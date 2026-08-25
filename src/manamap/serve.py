"""`manamap serve` — the viz, and a local API the deployed site does not have.

    $ manamap serve
      viz  http://localhost:8000/viz/workbench.html
      api  http://localhost:8000/api/   (local only)

THIS REVERSES A DECISION, AND THE REVERSAL IS DELIBERATE. `CLAUDE.md` recorded,
on 2026-08-01: *"The frontend never calls an LLM, and deployed == local … there
is no local-only bridge — because a local bridge means the deployed site and
your machine run different code, and only one of them is the one you test."*

The PRD overrules it (§5.2), and says why: *"when running locally, the agent is
embedded in the UI — not a separate terminal the user context-switches to.
Closing the gap between the front end and the agent back end is the single
largest UX requirement in this document."*

**Why it changed, in the owner's words:** *"I AM the sole user of this software
… I think having a build process produces higher quality code, so we will check
in and build what we can, but at the end of the day I want the software to work
how I need it to … and that's have sub-agents accessible for questions,
resolutions, etc. in the Build page."*

That settles what the static build IS. It is not a second audience to serve — it
is HYGIENE. Keeping the deployed path working keeps CI honest, keeps the
artifacts deterministic and keeps the deck pages readable offline, and those
disciplines are why the code is worth trusting. It is a by-product of building
well, not the target. The target is the agents being reachable from the page.

The old argument is answered rather than ignored:

- **The two builds do run different code, so the difference is a FEATURE of the
  page rather than an accident of the environment.** `/api/health` is probed
  once; a page that cannot reach it renders its static half and SAYS SO. There
  is no code path that silently behaves differently — an affordance is either
  present or explained.
- **The browser still makes no LLM calls.** It asks this server to run a
  command, exactly as a terminal would. Nothing here is a model client.
- **Most of Build needs no agent at all, which is what makes the agents
  affordable when they are needed.** The old note's own second bullet is the
  design: 69 pilot subcommands answer in JSON, instantly, for free.
  `archetypes`, `card-search`, `commander-search` and `build-deck` are all
  deterministic. Spending an agent on a question `card-search` answers is the
  waste that would make the agent path feel expensive; keeping them separate is
  what lets the expensive one be worth it.

**COMMANDS ARE AN ALLOW-LIST, NEVER A STRING TO RUN.** The API takes a command
NAME and a dict of arguments, looks the name up in `ENDPOINTS`, and calls a
Python function. Nothing is passed to a shell, no argument becomes a flag by
string interpolation, and a name that is not in the table is a 404 rather than
an attempt. A local server is still a server: something else on this machine can
reach it.

**Bound to 127.0.0.1.** Not configurable here on purpose — this is a personal
tool with no auth, and the one-line change that would expose it to a network is
a line somebody should have to write themselves, knowing why.
"""

import json
import threading

import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from manamap import console
from manamap.config import DATA_DIR

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


# ── The allow-list ─────────────────────────────────────────────────────────
#
# Each entry is (callable, {arg: coercion}). The coercion is what makes an
# argument safe: a value that will not coerce is a 400, so a string never
# reaches a place expecting an int and nothing arrives untyped.


def _str(v):
    return None if v is None else str(v)


def _int(v):
    return None if v in (None, "") else int(v)


def _bool(v):
    return bool(v) and str(v).lower() not in ("false", "0", "no")


def _strlist(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _archetypes(commander=None, theme=None, limit=12):
    from manamap.pilot import archetypes

    # A required argument is checked HERE, not left to crash three frames down.
    # Without it a missing commander reached `edhrec_slug(None)` and surfaced as
    # a 500 — a server fault for what is a caller's mistake.
    if not commander:
        raise ValueError("archetypes needs a commander")
    return archetypes.report(commander, theme=theme, limit=limit or 12)


def _commander_search(cards=(), space="text", limit=10, candidates=25,
                      controlled=True):
    from manamap.analysis import commander_search

    if not cards:
        raise ValueError("commander-search needs seed cards")

    return commander_search.search(list(cards), space=space, limit=limit or 10,
                                   per_identity=candidates or 25,
                                   controlled=controlled)


def _card_search(identity=None, oracle=(), role=None, cmc=None, limit=40):
    """Corpus mining. `card_search.search` owns every rule; this only passes
    arguments and shapes the reply for a browser."""
    from manamap.pilot.card_search import search

    rows, meta = search(
        identity=identity,
        oracle=list(oracle) or None,
        roles=[role] if role else None,
        cmc_max=int(cmc) if cmc not in (None, "") else None,
        limit=limit or 40,
    )
    # `rows` are pandas records; hand the browser plain names and the few fields
    # a candidate list needs, not a frame dump.
    def row(r):
        return {"name": r.get("name"), "type_line": r.get("type_line"),
                "cmc": r.get("cmc"), "identity": r.get("color_identity"),
                "roles": r.get("roles") or []}
    return {"meta": meta, "cards": [row(dict(r)) for r in rows]}


def _decks():
    """The manifest, so Build can list what is already on the bench."""
    path = DATA_DIR / "decks" / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"decks": []}


def _formats():
    from manamap.pilot import formats

    return {"formats": [
        {"key": k, "name": s.name, "deck_size": s.deck_size,
         "exact_size": s.exact_size, "singleton": s.singleton,
         "commanders": s.commanders, "colour_identity": s.colour_identity}
        for k, s in sorted(formats.FORMATS.items())]}


def _build_save(slug=None, commander=None, theme=None, bracket=None,
                library=(), fmt=None):
    """Create or update a DRAFT — a deck that exists as a brief and no more.

    §7.4 lands a new deck at v0.1.0, and this is the step before that: the point
    where a half-finished idea becomes something the Workbench can show you and
    you can come back to. Saving is idempotent, so the page can save on every
    change without a "is this the first time" branch.

    NOT A COMMITTED DECK. It writes `brief.json` and nothing else; there is no
    99, no `cards.json`, and `build_index` files it under `drafts` rather than
    `decks` for exactly that reason.
    """
    import json as _json

    from manamap.config import DECKS_DIR
    from manamap.pilot import formats

    if not slug or not str(slug).strip():
        raise ValueError("a draft needs a slug")
    slug = str(slug).strip().lower().replace(" ", "-")
    base = DECKS_DIR / slug
    base.mkdir(parents=True, exist_ok=True)
    path = base / "brief.json"
    # Read-modify-write, so saving a theme does not drop the library somebody
    # spent ten minutes gathering.
    brief = _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    brief.update({k: v for k, v in {
        "slug": slug, "commander": commander, "theme": theme,
        "format": fmt, "bracket": bracket,
    }.items() if v is not None})
    if library:
        brief["must_include"] = list(library)
    brief.setdefault("must_include", [])
    brief.setdefault("must_exclude", [])
    # Validate the MERGED brief, not the incoming patch. Requiring a commander
    # on every save meant `{"slug": …, "bracket": 4}` — a perfectly good update
    # to a draft that already names one — was refused for lacking a field it
    # was never going to send.
    spec = formats.get(brief.get("format"))
    if spec.commanders and not brief.get("commander"):
        raise ValueError(f"{spec.name} needs a commander")
    path.write_text(_json.dumps(brief, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return {"slug": slug, "path": str(path), "brief": brief, "draft": True}


def _build_run(slug=None):
    """Turn a draft into a 99 and write `decklist.txt`. Deterministic, no agent."""
    from manamap.pilot import build_deck

    if not slug:
        raise ValueError("build needs a slug")
    plan = build_deck.build(slug)
    return {"slug": slug, "commander": plan["commander"],
            "cards": len(plan["slots"]) + sum(plan["land_counts"].values()) + 1,
            "bracket": plan["bracket"],
            "role_budget_grounding": plan.get("role_budget_grounding"),
            "slots": [{"name": s["name"], "role": s.get("role")} for s in plan["slots"]]}


# ── Agent jobs ─────────────────────────────────────────────────────────────
#
# THE POINT OF THE BRIDGE. A deterministic command answers in milliseconds and
# returns; an agent takes minutes and costs real tokens, so it cannot be a
# request that blocks. It is a JOB: started, polled, and readable while it runs.
#
# Spawned through `claude -p`, in this repo, with the same charters the terminal
# uses (`.claude/agents/*.md`). That matters more than it looks: the agent
# reading `deck-doctor.md` from the page is THE SAME agent that reads it from a
# terminal, so nothing here is a second, weaker version of the loop that already
# produces artifacts. There is no separate prompt, no reimplementation.
#
# `subprocess` with an ARGUMENT LIST and never `shell=True`. The question is
# text the pilot typed; it goes to the model as one argument and is never seen
# by a shell.

JOBS = {}
_JOB_LOCK = threading.Lock()

#: What an agent costs, so the page can say it BEFORE spending it. From
#: `docs/agent-cost.md` — the cheapest routine measured 54.5k tokens and
#: `candidate-pool` is 235k. A button that spends a quarter of a million tokens
#: without saying so is the one thing this bridge must not become.
AGENT_COSTS = {
    "ask": "a few thousand tokens — a question, not a routine",
    "deck-doctor": "~50-90k tokens (diagnose); reads the deck's artifacts",
    "deck-analyst": "~235k tokens (candidate-pool) — the most expensive routine",
    "strategy-researcher": "~60k tokens, and it goes to the web",
}


def _spawn(job_id, prompt, agent=None, cwd=None):
    import subprocess

    cmd = ["claude", "-p", prompt]
    if agent:
        # The charter by NAME. Claude Code resolves it from `.claude/agents/`,
        # so the page cannot invent an agent that the terminal does not have.
        cmd += ["--agents", json.dumps({agent: {"description": agent}})]
    try:
        proc = subprocess.run(cmd, cwd=cwd or str(DATA_DIR.parent),
                              capture_output=True, text=True, timeout=1800)
        out, err, code = proc.stdout, proc.stderr, proc.returncode
    except FileNotFoundError:
        out, err, code = "", ("`claude` is not on PATH — the agent half of Build "
                              "needs Claude Code installed"), 127
    except Exception as exc:                    # noqa: BLE001
        out, err, code = "", f"{type(exc).__name__}: {exc}", 1
    with _JOB_LOCK:
        JOBS[job_id].update(state="done" if code == 0 else "failed",
                            output=out, error=err or None, code=code)


def _ask(question=None, agent=None):
    """Start an agent job. Returns immediately with an id to poll."""
    import uuid

    if not question:
        raise ValueError("no question")
    job_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        JOBS[job_id] = {"id": job_id, "state": "running", "agent": agent,
                        "question": question, "output": "", "error": None,
                        "cost": AGENT_COSTS.get(agent or "ask", AGENT_COSTS["ask"])}
    threading.Thread(target=_spawn, args=(job_id, question, agent),
                     daemon=True).start()
    return dict(JOBS[job_id])


def _job(id=None):
    """One job's state. Polled by the page while an agent runs."""
    with _JOB_LOCK:
        if id not in JOBS:
            raise ValueError(f"no job {id!r}")
        return dict(JOBS[id])


def _agents():
    """The charters this repo actually has, with what each costs.

    Read from disk rather than listed here, for the reason every registry in
    this repo is read rather than transcribed: a hardcoded list is a second
    place to remember, and it is the one that goes stale.
    """
    d = DATA_DIR.parent / ".claude" / "agents"
    names = sorted(p.stem for p in d.glob("*.md")) if d.is_dir() else []
    return {"agents": [{"name": n, "cost": AGENT_COSTS.get(n, "unmeasured")}
                       for n in names]}


#: name -> (function, {argument: coercion})
ENDPOINTS = {
    "health": (lambda: {"ok": True, "api": 1}, {}),
    "formats": (_formats, {}),
    "decks": (_decks, {}),
    "archetypes": (_archetypes, {"commander": _str, "theme": _str, "limit": _int}),
    "commander-search": (_commander_search, {
        "cards": _strlist, "space": _str, "limit": _int,
        "candidates": _int, "controlled": _bool}),
    "card-search": (_card_search, {
        "identity": _str, "oracle": _strlist, "role": _str,
        "cmc": _int, "limit": _int}),
    # Drafts: a deck you can put down and pick up.
    "build/save": (_build_save, {
        "slug": _str, "commander": _str, "theme": _str, "bracket": _int,
        "library": _strlist, "fmt": _str}),
    "build/run": (_build_run, {"slug": _str}),
    # The agent half. Started, then polled — an agent takes minutes.
    "agents": (_agents, {}),
    "ask": (_ask, {"question": _str, "agent": _str}),
    "job": (_job, {"id": _str}),
}


def call(name, payload):
    """Run one allow-listed command. Raises KeyError for an unknown name."""
    fn, spec = ENDPOINTS[name]
    kwargs = {}
    for key, coerce in spec.items():
        if key in payload:
            kwargs[key] = coerce(payload[key])
    return fn(**kwargs)


class Handler(SimpleHTTPRequestHandler):
    """Static files, plus `/api/<command>`.

    Everything that is not under `/api/` falls through to the normal static
    handler, so this is a drop-in replacement for `python -m http.server` run
    from the repo root — the same paths, the same `viz/` and `data/` siblings.
    """

    def _json(self, code, body):
        blob = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        # The API is local-only and unauthenticated, so it must never be cached
        # by anything: a stale build result is worse than a slow one.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/api":
            return self._json(200, {"commands": sorted(ENDPOINTS)})
        if self.path.startswith("/api/"):
            name = self.path[len("/api/"):].split("?")[0].rstrip("/")
            return self._run(name, {})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self.send_error(404)
        name = self.path[len("/api/"):].split("?")[0].rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "body is not JSON"})
        if not isinstance(payload, dict):
            return self._json(400, {"error": "body must be an object"})
        return self._run(name, payload)

    def _run(self, name, payload):
        if name not in ENDPOINTS:
            # A 404 rather than an attempt. The name is echoed so a typo is
            # obvious, and the list is offered so the page can discover what
            # this build supports rather than hardcoding it.
            return self._json(404, {"error": f"unknown command {name!r}",
                                    "commands": sorted(ENDPOINTS)})
        try:
            return self._json(200, {"ok": True, "command": name,
                                    "result": call(name, payload)})
        except (TypeError, ValueError) as exc:
            return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})
        except SystemExit as exc:
            # `SystemExit` is how the pilot layer reports a refusal — an unknown
            # theme, a missing brief. That is a 400, not a crashed server.
            return self._json(400, {"error": str(exc)})
        except Exception as exc:               # noqa: BLE001 — a server must not die
            console.err(traceback.format_exc())
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        """Quiet by default; the console layer owns what the terminal says."""
        if self.path.startswith("/api/"):
            console.err(f"  api {self.path} {args[1] if len(args) > 1 else ''}")


def serve(port=DEFAULT_PORT, root=None):
    from manamap.config import DATA_DIR as _d

    root = str(root or _d.parent)
    handler = partial(Handler, directory=root)
    httpd = ThreadingHTTPServer((HOST, port), handler)
    return httpd


def main(args=None):
    port = getattr(args, "port", None) or DEFAULT_PORT
    httpd = serve(port)
    base = f"http://{HOST}:{port}"
    print(f"manamap serve — {httpd.server_address[0]}:{httpd.server_address[1]}")
    print(f"  workbench  {base}/viz/workbench.html")
    print(f"  atlas      {base}/viz/index.html")
    print(f"  api        {base}/api/   ({len(ENDPOINTS)} commands, local only)")
    print("\n  The deployed site has no /api — Build shows its static half there,")
    print("  and says so rather than failing quietly.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
