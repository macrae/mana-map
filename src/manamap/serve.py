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
         "commanders": s.commanders, "colour_identity": s.colour_identity,
         "buildable": s.buildable}
        for k, s in sorted(formats.FORMATS.items())]}


def _resolve_commander(name):
    """A typed commander name -> the corpus's exact name, or a helpful refusal.

    TYPED BY A HUMAN, so it is matched the way a human means it: case and
    punctuation are ignored. The report that produced this was "zur, the
    enchanter" — lowercase, and with a comma that most legendary creatures do
    have. `Zur, Eternal Schemer` sits three rows away in the corpus WITH one;
    `Zur the Enchanter` has none. That is not a user error, it is a lookup that
    demanded more precision than the name itself carries.

    It also mattered that EDHREC already forgave it: `archetypes` answered
    happily for "zur, the enchanter" because `edhrec_slug` lowercases and strips
    punctuation. So the styles panel worked and the build failed, which is the
    worst arrangement — the part that looks like progress succeeds and the part
    that does the work does not.

    A miss SUGGESTS. "not in cards.csv" is accurate and useless; the corpus
    knows perfectly well what you probably meant.
    """
    import re

    from manamap.pilot import card_pool

    def key(v):
        return re.sub(r"[^a-z0-9]+", "", str(v).lower())

    wanted = key(name)
    if not wanted:
        return None, []

    # Legendary creatures first, because this is a COMMANDER field. Suggesting
    # Zuran Orb to someone typing "zur" is technically a prefix match and no
    # help at all.
    frame = card_pool.load_frame()
    legendary = {n for n, t in zip(frame["name"], frame["type_line"])
                 if "Legendary" in str(t) and "Creature" in str(t)}

    exact, near = None, []
    for real in card_pool.corpus_names():
        k = key(real)
        # An empty key matches everything — the corpus holds cards literally
        # named "_____", and `"".startswith("")` put them at the head of every
        # suggestion list.
        if not k:
            continue
        if k == wanted:
            exact = real
            break
        if k.startswith(wanted) or wanted.startswith(k):
            near.append(real)
    # Commanders before anything else, then shortest — the shortest legendary
    # starting with what you typed is almost always the one you meant.
    near.sort(key=lambda n: (n not in legendary, len(n)))
    return exact, near[:6]


def _commanders(q=None, limit=8):
    """Legendary creatures matching what has been typed so far. §7 — the picker.

    A TEXT BOX THAT REFUSES IS THE WRONG CONTROL. Typing "zur" produced a 400
    with suggestions nobody could click, no brief was written, and pressing
    Build then reported a missing file — three messages for one unfinished
    thought. A picker has no invalid state to report: you type, you see real
    commanders, you choose one, and the name that reaches disk is the corpus's.

    Ranked shortest-first among names that CONTAIN the query, with names that
    START with it first. "zur" should offer Zur the Enchanter before Zurgo
    Bellstriker, and both before a card that merely contains "zur" in the
    middle.
    """
    import re

    from manamap.pilot import card_pool

    q = re.sub(r"[^a-z0-9]+", "", str(q or "").lower())
    if len(q) < 2:
        return {"query": q, "commanders": []}

    frame = card_pool.load_frame()
    out = []
    for name, type_line in zip(frame["name"], frame["type_line"]):
        t = str(type_line)
        if "Legendary" not in t or "Creature" not in t:
            continue
        k = re.sub(r"[^a-z0-9]+", "", str(name).lower())
        if q in k:
            out.append((0 if k.startswith(q) else 1, len(name), name))
    seen, ranked = set(), []
    for _, _, name in sorted(out):
        if name not in seen:
            seen.add(name)
            ranked.append(name)
    return {"query": q, "commanders": ranked[:limit or 8]}


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
    if commander:
        # Resolved at SAVE, so the draft on disk carries the corpus's exact name
        # and every later step — build, validate, the manual — agrees about
        # which card this is.
        exact, near = _resolve_commander(commander)
        if not exact:
            raise ValueError(
                f"no card named {commander!r}"
                + (f" — did you mean {', '.join(near)}?" if near else
                   " — check the spelling against the card"))
        commander = exact
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
    import json as _json

    from manamap.config import DECKS_DIR
    from manamap.pilot import build_deck, formats

    if not slug:
        raise ValueError("build needs a slug")

    # Refuse a format the builder cannot build, HERE, with the reason. Without
    # this a Standard draft reached `load_brief` and came back "brief.json has
    # no commander" — true, useless, and blaming the brief for a limitation of
    # the builder. Defence in depth: the picker should not have offered it, and
    # this says so if something else did.
    path = DECKS_DIR / slug / "brief.json"
    if not path.exists():
        # `load_brief`'s own message tells the reader to "author it first" and
        # prints the JSON to write — correct in a terminal, nonsense in a
        # browser, where nobody has a file open. The endpoint owns the message
        # its caller can act on.
        raise ValueError(f"nothing saved for {slug!r} yet — choose a commander, "
                         f"and the draft saves itself")
    brief = _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    spec = formats.get(brief.get("format"))
    if not spec.buildable:
        raise ValueError(
            f"the builder cannot build {spec.name} yet — it is anchored on a "
            f"commander at every step (colour identity, the similarity seed, "
            f"the bracket engine, a 99-card mana base), and a constructed deck "
            f"has no such anchor. {spec.name} decks can be VALIDATED "
            f"(`validate-deck --format {brief.get('format')}`) and their pool "
            f"searched, but not built.")
    # BUILD AND *WRITE*. `build_deck.build()` computes a plan and returns it;
    # `build_deck.main()` is what persists one. Calling only the first reported
    # "100 cards, bracket 3" and left NOTHING on disk — success for work it had
    # thrown away, which is the worst thing a build button can do.
    #
    # Routed through `main` rather than reimplementing the writes here, so the
    # page and the CLI produce byte-identical artifacts: `main` also merges the
    # agent-authored keys an existing `build_plan.json` carries, and a second
    # writer would drop them.
    build_deck.main(type("A", (), {"slug": slug, "write_decklist": True})())

    plan = _json.loads((DECKS_DIR / slug / "build_plan.json").read_text(encoding="utf-8"))
    written = [p.name for p in sorted((DECKS_DIR / slug).iterdir())]
    return {"slug": slug, "commander": plan["commander"],
            "cards": len(plan["slots"]) + sum(plan["land_counts"].values()) + 1,
            "bracket": plan["bracket"],
            "role_budget_grounding": plan.get("role_budget_grounding"),
            # Kept cards the commander cannot legally play. The pilot chose them
            # on purpose, so the page has to say which ones could not come.
            "must_include_illegal": plan.get("must_include_illegal") or [],
            "written": written,
            "slots": [{"name": s["name"], "role": s.get("role")} for s in plan["slots"]]}


def _build_finish(slug=None, commit=False, message=None):
    """A draft becomes a tracked deck. The last step that needed a terminal.

    Two acts, and only the second is irreversible:

    1. `fetch-deck` resolves every name against Scryfall and writes
       `cards.json` — printings, images, the decklist sha. This is what moves
       the deck out of `manifest.drafts` and onto the bench, because
       `build_index` admits a deck when it has a `cards.json`.
    2. `build-index` rewrites the manifest so the page can see it.

    THE COMMIT IS SEPARATE AND OPT-IN. `decklist.txt` is tracked, so the commit
    is what `deck-version` NUMBERS and what the captain's log stamps games
    against — "check a deck in without committing and tonight's games attach to
    no version at all". That makes it load-bearing rather than bookkeeping, and
    load-bearing enough that a button should not do it by surprise. Without
    `commit`, the exact command is returned instead, which is also the honest
    answer for anyone who wants to write their own message.
    """
    import subprocess

    from manamap.config import DECKS_DIR
    from manamap.pilot import build_index, fetch_deck

    if not slug:
        raise ValueError("finish needs a slug")
    base = DECKS_DIR / slug
    if not (base / "decklist.txt").exists():
        raise ValueError(f"{slug} has no decklist yet — build it first")

    fetch_deck.main(type("A", (), {"slug": slug, "force": False})())
    build_index.main()

    out = {"slug": slug,
           "written": [p.name for p in sorted(base.iterdir())],
           "committed": False,
           "commit_command": f"git add data/decks/{slug} && git commit -m "
                             f"\"Build {slug}: first list\""}
    if not commit:
        out["note"] = ("committed nothing — `decklist.txt` is tracked, and the "
                       "commit is what numbers this V1 and what the captain's "
                       "log stamps games against. Run the command above, or ask "
                       "again with commit.")
        return out

    msg = message or f"Build {slug}: first list"
    try:
        subprocess.run(["git", "add", f"data/decks/{slug}"],
                       cwd=str(DECKS_DIR.parent.parent), check=True,
                       capture_output=True, text=True)
        r = subprocess.run(["git", "commit", "-m", msg],
                           cwd=str(DECKS_DIR.parent.parent),
                           capture_output=True, text=True)
        out["committed"] = r.returncode == 0
        out["git"] = (r.stdout or r.stderr).strip().splitlines()[:3]
    except Exception as exc:                    # noqa: BLE001
        out["git"] = [f"{type(exc).__name__}: {exc}"]
    return out


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
# ── Measuring a deck from the page ────────────────────────────────────────
#
# THE ALLOW-LIST IS THE POINT, and what it excludes is the argument for it.
# Every entry here is DETERMINISTIC, makes NO model call, and writes exactly one
# artifact. Measured end to end on a 100-card deck, INCLUDING the `info.json`
# refresh below: bracket 2.3s, map 1.9s. The refresh is the larger half of both
# (bracket-check 512ms, deck-map 1150ms, deck-info --write 1513ms) because it
# composes every artifact and runs the audit — which is worth knowing before
# anyone tries to make this feel instant by trimming the command instead.
#
# (An earlier version of this comment claimed ~200ms from a timing loop whose
# runs had all FAILED. A measurement of a command that did not do the work is
# not a measurement of the command.)
#
# What is NOT here, deliberately: `simulate` (45-62 minutes of Forge), every
# agent loop (`/analyze-engine`, `/resolve-stack` — the cheapest measured
# routine is 54.5k tokens and `candidate-pool` is 235k), and the authored files
# (`goldfish_targets.json`, `issue.json`), which are judgements a person makes
# rather than work a machine does. The dossier prints THOSE as text and names
# them, which is the whole reason a command travels with a stage: some of them
# are for reading, not for pressing.
#
# `needs` is checked and reported rather than assumed. `mana-analysis` embeds
# goldfish figures and must run after it; `goldfish` reads the authored
# declaration and cannot run without one. A button that fails because of an
# unstated dependency is worse than a button that explains itself.
class _Args:
    """An argparse namespace, without argparse.

    The pilot commands take `args` and read attributes off it. `type("A", (), {…})()`
    is the trick used twice above; naming it once is what stops the third caller
    inventing a fourth spelling — and `getattr` defaults matter, because these
    modules read optional flags (`--target`, `--json`) that a page never sends.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):        # every unset flag is simply absent
        return None


MEASURES = {
    "bracket": ("manamap.pilot.bracket", "bracket_report.json", (),
                "the computed power floor"),
    "map":     ("manamap.pilot.deck_map", "deck_map.json", (),
                "the deck's own constellation"),
    "goldfish": ("manamap.pilot.goldfish", "goldfish_metrics.json",
                 ("goldfish_targets.json",), "seeded Monte Carlo development"),
    "mana":    ("manamap.pilot.mana_analysis", "mana_analysis.json",
                ("goldfish_metrics.json",), "colour sources and castability"),
}


#: Authored files a page may DRAFT — never author. The distinction is the whole
#: point: `scaffold-targets` writes a starting shape derived from contained
#: combos and role axes, marked `"scaffolded": true`, and the validator says so
#: on every run until a person edits it. Nothing here writes a claim the pilot
#: has not made; it writes the blank page they were otherwise facing.
#:
#: `issue` is absent on purpose. Its three live keys are a deck's NAME and
#: whether it is still sleeved — a fact about cardboard that no command can
#: derive, and the exact class of claim the withdrawn rehearsal locks were
#: withdrawn for.
SCAFFOLDS = {
    "targets": ("manamap.pilot.scaffold_targets", "goldfish_targets.json"),
}


def _scaffold(slug=None, stage=None):
    """Write a starting version of an authored file, and refresh the dossier."""
    if not slug:
        raise ValueError("no slug")
    if stage not in SCAFFOLDS:
        raise ValueError(
            f"{stage!r} has no draft this page can write — it drafts "
            f"{', '.join(sorted(SCAFFOLDS))}. The rest are judgements somebody "
            f"has to make.")
    from manamap.config import DECKS_DIR

    import importlib

    module, artifact = SCAFFOLDS[stage]
    base = DECKS_DIR / slug
    if (base / artifact).exists():
        raise ValueError(
            f"{slug} already has {artifact} — a draft never overwrites one, "
            f"because an authored file is the thing no command can rebuild")
    importlib.import_module(module).scaffold(slug)

    from manamap.pilot import deck_info

    deck_info.main(_Args(slug=slug, write=True, as_json=False))
    return {"slug": slug, "stage": stage, "wrote": artifact, "draft": True,
            "info": json.loads((base / "info.json").read_text(encoding="utf-8"))}


def _measure(slug=None, stage=None):
    """Run one deterministic measurement and hand back the refreshed dossier.

    The refresh is not a convenience. `info.json` is composed from every other
    artifact, so a measurement leaves it stale by construction — and the page
    renders `info.json`. Running the command without re-emitting it would show
    the pilot a dossier that still says the thing they just measured is missing.
    """
    if not slug:
        raise ValueError("no slug")
    if stage not in MEASURES:
        raise ValueError(
            f"{stage!r} is not something this page may run — "
            f"it runs {', '.join(sorted(MEASURES))}. Everything else is an agent "
            f"loop, a 45-minute simulation, or a file somebody has to write.")
    from manamap.config import DECKS_DIR

    module, artifact, needs, what = MEASURES[stage]
    base = DECKS_DIR / slug
    if not (base / "cards.json").exists():
        raise ValueError(f"{slug} has no cards.json — finish the deck first")
    for need in needs:
        if not (base / need).exists():
            raise ValueError(
                f"{stage} needs {need} first, and this deck has none — "
                f"see the dossier's own note on that step")

    import importlib

    mod = importlib.import_module(module)
    mod.main(_Args(slug=slug))

    from manamap.pilot import deck_info

    deck_info.main(_Args(slug=slug, write=True, as_json=False))
    return {"slug": slug, "stage": stage, "wrote": artifact, "what": what,
            "info": json.loads((base / "info.json").read_text(encoding="utf-8"))}

ENDPOINTS = {
    "health": (lambda: {"ok": True, "api": 1}, {}),
    "formats": (_formats, {}),
    "decks": (_decks, {}),
    "commanders": (_commanders, {"q": _str, "limit": _int}),
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
    "build/finish": (_build_finish, {"slug": _str, "commit": _bool, "message": _str}),
    # The agent half. Started, then polled — an agent takes minutes.
    "agents": (_agents, {}),
    "ask": (_ask, {"question": _str, "agent": _str}),
    "job": (_job, {"id": _str}),
    # Deterministic measurement, no model call. See MEASURES.
    "deck/measure": (_measure, {"slug": _str, "stage": _str}),
    "deck/measures": (lambda: {"stages": {k: {"artifact": v[1], "needs": list(v[2]),
                                              "what": v[3]}
                                         for k, v in MEASURES.items()},
                               "drafts": sorted(SCAFFOLDS)}, {}),
    "deck/scaffold": (_scaffold, {"slug": _str, "stage": _str}),
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
        except RuntimeError as exc:
            # A DELIBERATE, ALREADY-READABLE FAILURE: something outside this
            # machine did not cooperate and the raiser wrote a sentence about
            # it. Prefixing the class name onto that sentence is how the page
            # came to say "ConnectionError: ('Connection aborted.',
            # RemoteDisconnected('Remote end closed connection without
            # response'))" — a true statement about a socket, offered as the
            # answer to "did my deck get built". The prefix survives below,
            # where an unexpected TYPE is genuinely the most useful clue.
            console.err(traceback.format_exc())
            return self._json(502, {"error": str(exc)})
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
