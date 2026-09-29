"""A PRE-REGISTERED CAMPAIGN: the overnight queue of Forge A/Bs.

WHY A FILE, AND WHY IT IS TRACKED. A powered A/B is ~400 games an arm at 0.7
games a minute — about ten hours per arm on this machine — so the loop that
"proposes, measures, records" cannot be a person typing `experiment` at eleven
at night. It is a queue. And a queue that is written down BEFORE the games is
the only kind that counts as pre-registration: the arms, the table, N, the
looks, the endpoint and the hypothesis are on record with a git date, and
`multiple comparisons` stops being a worry a reader has to take on trust.
`data/campaigns/<name>.json` is tracked for the same reason `data/pods/` is.

WHAT IT NEVER DOES. `run` starts experiments in order, resumes an unfinished
sequential one, and records each terminal result in the deck's decision ledger.
It does not propose, merge, or touch a decklist; a test holds it to that by
reading this module's imports. The pilot decides; the queue measures.

STATE IS DERIVED, NEVER STORED — the `validate_pending` rule. An entry is DONE
when its experiment record has a terminal status, RUNNING when the record says
so, STALE when a `working` ref it pinned at plan time has moved underneath it
(the games would measure a list nobody holds), and PENDING otherwise. Nothing
in the campaign file says which; the records do.
"""

import datetime
import json
import os

from manamap.config import DATA_DIR, SIM_GAME_CLOCK_SECONDS
from manamap.pilot.common import deck_dir, load_json

CAMPAIGNS_DIR = DATA_DIR / "campaigns"
STATES = ("PENDING", "RUNNING", "DONE", "STALE")
MAX_LOOKS = 4
#: The one endpoint an entry may register. `experiment.PRIMARY_ENDPOINT` is the
#: same word; restated here so a campaign file can be validated without Forge.
PRIMARY = "win_rate"
ENTRY_KEYS = {"id", "slug", "a", "b", "pod", "games", "looks", "primary", "hypothesis",
              "detect", "until_mde", "profile", "vs_profile", "profile_b", "aa", "anyway",
              "seed", "boundary", "resolved", "note"}
REQUIRED = ("id", "slug", "a", "b", "pod", "games", "hypothesis")


def path_for(name):
    return CAMPAIGNS_DIR / f"{name}.json"


def state_path(name):
    """Gitignored: pid and the entry in flight, for `sim-progress`."""
    return CAMPAIGNS_DIR / f"{name}.state.json"


def list_all():
    return sorted(p.stem for p in CAMPAIGNS_DIR.glob("*.json")
                  if not p.name.endswith(".state.json")) if CAMPAIGNS_DIR.is_dir() else []


def load(name):
    p = path_for(name)
    if not p.exists():
        known = ", ".join(list_all()) or "none"
        raise SystemExit(f"no campaign named {name!r} — known: {known}. A campaign is "
                         f"{CAMPAIGNS_DIR}/<name>.json")
    doc = load_json(p)
    errors = validate(doc)
    if errors:
        raise SystemExit(f"{p.name} fails its own form check:\n  - " + "\n  - ".join(errors))
    return doc


def validate(doc):
    """The gate. Form only: it runs on a checkout with no Forge and no logs."""
    errors = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if not doc.get("name"):
        errors.append("no name")
    if not str(doc.get("hypothesis") or "").strip():
        errors.append("no hypothesis — a campaign with no question is a batch, not a "
                      "pre-registration")
    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        return errors + ["entries: none — nothing to run"]
    seen = set()
    from manamap.sim import pods
    for i, e in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{where}: not an object")
            continue
        for k in REQUIRED:
            if not e.get(k) and e.get(k) != 0:
                errors.append(f"{where}: no {k!r}")
        unknown = set(e) - ENTRY_KEYS
        if unknown:
            errors.append(f"{where}: unknown key(s) {sorted(unknown)}")
        eid = e.get("id")
        if eid in seen:
            errors.append(f"{where}: duplicate id {eid!r}")
        seen.add(eid)
        if e.get("primary", PRIMARY) != PRIMARY:
            errors.append(f"{where}: primary {e.get('primary')!r} — only {PRIMARY!r} is "
                          f"a registered endpoint (`experiment.PRIMARY_ENDPOINT`)")
        looks = e.get("looks", 1)
        if not isinstance(looks, int) or not 1 <= looks <= MAX_LOOKS:
            errors.append(f"{where}: looks {looks!r} is not 1..{MAX_LOOKS}")
        games = e.get("games")
        if isinstance(games, int) and isinstance(looks, int) and looks and games % looks:
            errors.append(f"{where}: {games} games do not divide into {looks} looks")
        if e.get("aa") and e.get("a") != e.get("b"):
            errors.append(f"{where}: aa is one list twice; a={e.get('a')!r} b={e.get('b')!r}")
        if not e.get("aa") and not e.get("profile_b") and e.get("a") == e.get("b"):
            errors.append(f"{where}: a and b are the same ref with neither aa nor profile_b")
        pod = e.get("pod")
        if pod and not (pods.PODS_DIR / f"{pod}.json").is_file():
            errors.append(f"{where}: pod {pod!r} is not a table under data/pods/")
        slug = e.get("slug")
        if slug and not (deck_dir(slug) / "decklist.txt").exists():
            errors.append(f"{where}: no deck {slug!r}")
        r = e.get("resolved")
        if r is not None:
            for k in ("a_sha", "b_sha", "experiment_id", "harness", "planned"):
                if k not in r:
                    errors.append(f"{where}: resolved has no {k!r}")
    return errors


# ── resolution: refs -> shas, an entry -> a run id ────────────────────────────

def harness_fingerprint():
    """What the engine holds right now: the axes a record carries so that two
    runs pooled together were flown under one instrument."""
    from manamap.sim import forge
    try:
        ov = forge.card_overrides()
    except Exception:                       # noqa: BLE001 — no engine here
        ov = None
    try:
        build = forge.forge_version()
        if isinstance(build, dict):         # {version, build} — one hashable string
            build = f"{build.get('version')} {build.get('build')}".strip()
    except Exception:                       # noqa: BLE001
        build = None
    return {"forge": build, "card_overrides": (ov or {}).get("sha"),
            "clock_seconds": SIM_GAME_CLOCK_SECONDS}


def resolve_entry(entry):
    """Pin an entry: refs to shas, the run id it will write, the harness."""
    from manamap.sim import experiment as ex
    from manamap.sim import pods
    a = ex.resolve_arm(entry["slug"], entry["a"])
    b = ex.resolve_arm(entry["slug"], entry["b"])
    opponents = pods.seats(entry["pod"])
    hf = harness_fingerprint()
    seed = entry.get("seed")
    if seed is None:
        import hashlib
        seed = int(hashlib.sha256((a["decklist_sha256"] + b["decklist_sha256"]).encode())
                   .hexdigest()[:8], 16) % 2_000_000_000
    eid = ex.experiment_id(entry["slug"], a, b, opponents, int(entry["games"]), seed,
                           profile=entry.get("profile"),
                           vs_profile=entry.get("vs_profile") or ex.STANDARD_POD_PROFILE,
                           clock=SIM_GAME_CLOCK_SECONDS,
                           overrides_sha=hf["card_overrides"],
                           looks=int(entry.get("looks", 1) or 1),
                           aa=bool(entry.get("aa")), profile_b=entry.get("profile_b"))
    return {"a_sha": a["decklist_sha256"], "b_sha": b["decklist_sha256"],
            "seed": seed, "experiment_id": eid, "harness": hf,
            "planned": datetime.date.today().isoformat()}


def record_for(entry):
    r = entry.get("resolved") or {}
    if not r.get("experiment_id"):
        return None
    p = deck_dir(entry["slug"]) / "experiments" / f"{r['experiment_id']}.json"
    return load_json(p) if p.exists() else None


def state_of(entry):
    """DERIVED from the record and the lists, never read from the file."""
    from manamap.sim import experiment as ex
    r = entry.get("resolved")
    if not r:
        return "PENDING", "not planned yet — `campaign <name> plan`"
    doc = record_for(entry)
    if doc:
        st = doc.get("status") or "complete"
        if st == "running":
            done = len(doc.get("looks") or [])
            return "RUNNING", f"look {done} of {(doc.get('design') or {}).get('looks', 1)} recorded"
        return "DONE", f"{st}: {(doc.get('delta') or {}).get('reading', '')[:90]}"
    for arm, key in (("a", "a_sha"), ("b", "b_sha")):
        ref = entry[arm]
        if ref == "working" or str(ref).startswith("@"):
            try:
                now = ex.resolve_arm(entry["slug"], ref)["decklist_sha256"]
            except SystemExit:
                now = None
            if now != r.get(key):
                return "STALE", (f"arm {arm} ({ref}) was pinned at {str(r.get(key))[:12]} and "
                                 f"is {str(now)[:12]} now — re-plan, or the games measure a "
                                 f"list nobody holds")
    return "PENDING", "planned; not started"


# ── the verbs ─────────────────────────────────────────────────────────────────

def _fingerprint_key(entry):
    hf = (entry.get("resolved") or {}).get("harness") or {}
    return (hf.get("forge"), hf.get("card_overrides"), entry.get("pod"),
            hf.get("clock_seconds"))


def plan(name, write=True):
    """Resolve every entry, preflight it, prepend an A/A where the harness has
    none, and write `resolved` back — the one time this command writes the file."""
    from manamap.sim import power
    doc = load(name)
    lines = [f"CAMPAIGN {name} — {doc.get('hypothesis')}"]
    for e in doc["entries"]:
        e["resolved"] = resolve_entry(e)
    # A STANDING A/A PER HARNESS. Policy-off against policy-off at two seeds is
    # the only cheap detector of the censoring confound two records already
    # show (82 vs 73 decided games at one N); a campaign without one at each
    # harness it uses cannot tell a harness effect from a list effect.
    have_aa = {_fingerprint_key(e) for e in doc["entries"] if e.get("aa")}
    added = []
    for e in list(doc["entries"]):
        key = _fingerprint_key(e)
        if e.get("aa") or key in have_aa:
            continue
        aa = {"id": f"aa-{e['slug']}-{e['pod']}", "slug": e["slug"], "a": "working",
              "b": "working", "aa": True, "pod": e["pod"], "games": e["games"],
              "looks": e.get("looks", 1), "primary": PRIMARY,
              "hypothesis": (f"the harness's noise floor for {e['slug']} at {e['pod']}: "
                             f"one list, two seed bases — an interval that excludes zero "
                             f"here is a harness finding, not a deck finding")}
        aa["resolved"] = resolve_entry(aa)
        doc["entries"].insert(0, aa)
        have_aa.add(key)
        added.append(aa["id"])
    if added:
        lines.append(f"  prepended A/A entr{'y' if len(added) == 1 else 'ies'}: {', '.join(added)}")
    total_h = 0.0
    for e in doc["entries"]:
        r = e["resolved"]
        p_a, n_null = power.null_rate(e["pod"])
        pre = power.preflight(p_a, int(e["games"]), detect=e.get("detect"), arms=2, n_a=None)
        hours = 2 * int(e["games"]) / power.GAMES_PER_MINUTE / 60
        total_h += hours
        lines.append(f"\n  [{e['id']}] {e['slug']}: {e['a']} vs {e['b']} at {e['pod']}, "
                     f"{e['games']}/arm, {e.get('looks', 1)} look(s)"
                     + (" (A/A)" if e.get("aa") else "")
                     + f"  -> {r['experiment_id'][:70]}")
        lines.append(f"      {e['hypothesis']}")
        lines += ["    " + l for l in pre[:3]]
        if e.get("detect") is not None and p_a is not None and not e.get("anyway"):
            try:
                power.refuse_if_underpowered(p_a, int(e["games"]), e["detect"], False)
            except SystemExit as exc:
                raise SystemExit(f"[{e['id']}] {exc}\n(set \"anyway\": true on the entry to "
                                 f"run it as a screen)")
    lines.append(f"\n  about {total_h:.0f} h of Forge for the whole queue at "
                 f"{power.GAMES_PER_MINUTE} games/min — roughly "
                 f"{total_h / 10:.0f} night(s) of ten hours")
    if write:
        path_for(name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        lines.append(f"  written: {path_for(name)}")
    return doc, lines


def status(name):
    doc = load(name)
    rows = []
    for e in doc["entries"]:
        st, why = state_of(e)
        rows.append({"id": e["id"], "slug": e["slug"], "pod": e["pod"], "games": e["games"],
                     "looks": e.get("looks", 1), "aa": bool(e.get("aa")),
                     "state": st, "why": why,
                     "experiment_id": (e.get("resolved") or {}).get("experiment_id")})
    return rows


def run(name, only=None, dry_run=False):
    """In order: skip DONE and STALE, resume RUNNING, run PENDING. NOTHING MERGES."""
    from manamap.sim import experiment as ex
    from manamap.sim import pods
    doc = load(name)
    ran = []
    for e in doc["entries"]:
        if only and e["id"] not in only:
            continue
        st, why = state_of(e)
        if st in ("DONE", "STALE"):
            print(f"  [{e['id']}] {st} — {why}")
            continue
        if not e.get("resolved"):
            raise SystemExit(f"[{e['id']}] is not planned — `manamap pilot campaign {name} plan` first")
        hf_now, hf_then = harness_fingerprint(), e["resolved"]["harness"]
        if hf_now.get("card_overrides") != hf_then.get("card_overrides"):
            raise SystemExit(f"[{e['id']}] was planned under card overrides "
                             f"{hf_then.get('card_overrides')} and the engine carries "
                             f"{hf_now.get('card_overrides')} — `forge-install` to match, or "
                             f"re-plan; a run under another harness is another measurement")
        opponents = pods.seats(e["pod"])
        print(f"  [{e['id']}] {'RESUME' if st == 'RUNNING' else 'RUN'}: {e['slug']} "
              f"{e['a']} vs {e['b']} at {e['pod']} ({e['games']}/arm, "
              f"{e.get('looks', 1)} look(s))")
        if dry_run:
            ran.append((e["id"], "would run"))
            continue
        _write_state(name, e, "running")
        try:
            path, rec = ex.run(e["slug"], e["a"], e["b"], opponents, games=int(e["games"]),
                               seed=e["resolved"]["seed"], profile=e.get("profile"),
                               vs_profile=e.get("vs_profile"),
                               detect=e.get("detect"), anyway=bool(e.get("anyway")),
                               looks=int(e.get("looks", 1) or 1),
                               until_mde=e.get("until_mde"), aa=bool(e.get("aa")),
                               resume=(st == "RUNNING"),
                               boundary=e.get("boundary") or "obf",
                               profile_b=e.get("profile_b"), pod_name=e["pod"])
        finally:
            _write_state(name, e, "idle")
        _ledger(e, rec)
        ran.append((e["id"], rec.get("status")))
    return ran


def _ledger(entry, rec):
    """Record the terminal experiment in the deck's decision ledger, where
    the ledger exists. The queue MEASURES; the ledger is where a measurement
    is later read against the decision it informed."""
    try:
        from manamap.pilot import decisions
    except ImportError:                    # the ledger lands in A6
        return
    decisions.append(entry["slug"], "experiment", campaign=entry.get("id"),
                     experiment_id=rec.get("experiment_id"), status=rec.get("status"),
                     hypothesis=entry.get("hypothesis"),
                     prediction={"endpoint": PRIMARY,
                                 **{k: (rec.get("delta") or {}).get("win_rate", {}).get(k)
                                    for k in ("a", "b", "diff", "ci95_diff", "excludes_zero")}})


def _write_state(name, entry, phase):
    CAMPAIGNS_DIR.mkdir(parents=True, exist_ok=True)
    state_path(name).write_text(json.dumps({
        "campaign": name, "entry": entry["id"], "slug": entry["slug"], "phase": phase,
        "pid": os.getpid(), "at": datetime.datetime.now().isoformat(timespec="seconds"),
        "experiment_id": (entry.get("resolved") or {}).get("experiment_id")}, indent=1))


def live_for(slug):
    """The campaign entry in flight for a deck, if any — read by `sim-progress`."""
    for p in CAMPAIGNS_DIR.glob("*.state.json") if CAMPAIGNS_DIR.is_dir() else []:
        try:
            st = json.loads(p.read_text())
        except Exception:                   # noqa: BLE001
            continue
        if st.get("slug") == slug and st.get("phase") == "running":
            return st
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args):
    name = getattr(args, "name", None)
    if not name:
        names = list_all()
        print("CAMPAIGNS" + (f" ({len(names)})" if names else " — none; a campaign is "
                             f"data/campaigns/<name>.json"))
        for n in names:
            rows = status(n)
            counts = {s: sum(1 for r in rows if r["state"] == s) for s in STATES}
            print(f"  {n:32} " + "  ".join(f"{s.lower()} {c}" for s, c in counts.items() if c))
        return
    action = getattr(args, "action", None) or "status"
    if action == "plan":
        _, lines = plan(name, write=not getattr(args, "dry_run", False))
        print("\n".join(lines))
        return
    if action == "run":
        ran = run(name, only=getattr(args, "only", None) or None,
                  dry_run=getattr(args, "dry_run", False))
        print(f"\n  {len(ran)} entr{'y' if len(ran) == 1 else 'ies'} "
              f"{'would run' if getattr(args, 'dry_run', False) else 'ran'}; nothing merged — "
              f"read each with `experiment <slug> --list`, then decide")
        return
    rows = status(name)
    if getattr(args, "as_json", False):
        print(json.dumps(rows, indent=2))
        return
    doc = load(name)
    print(f"CAMPAIGN {name} — {doc.get('hypothesis')}\n")
    for r in rows:
        print(f"  {r['state']:8} [{r['id']}] {r['slug']} at {r['pod']}  {r['games']}/arm "
              f"x{r['looks']}{' A/A' if r['aa'] else ''}")
        print(f"           {r['why']}")
    live = [live_for(r["slug"]) for r in rows]
    if any(live):
        print("\n  in flight: " + ", ".join(l["entry"] for l in live if l))
