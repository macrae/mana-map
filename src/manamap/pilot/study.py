"""THE STUDY: one question about one deck, the evidence from every source with its
tier and stamp, the agents' arguments kept, the call, and who decided — in the
pilot's words. `data/decks/<slug>/studies/NNN-<kebab>.json`, tracked.

WHY IT EXISTS (2026-11-21). The bench's decision verbs — `try`, `net-change`,
`propose`, the ledger — read two goldfish runs, the bill, Forge and mana, and
nothing a person wrote. The strategist, the rules agent, the analyst and the
scout wrote nothing, so a month of argued one-card calls lived in gitignored
`.agent-out/*.md` files read by nobody, and the pilot had to say "the goldfish
is not the decider" fourteen times in forty-six days. The study is the unit of
Jarvis's work and the artifact a decision rests on. An instrument's row is one
fact in it, labelled with what the instrument could see.

THE SHAPE, borrowed from `sim_findings.json` (`pilot_findings.py`): a COMPUTED
skeleton of `facts[]` with ids (`S-<sha8>-NN`), each with `kind`, `text`, `tier`
(✓ `verified`, ◆ `data`, ★ `coach`), `source` and `as_of`; AUTHORED parts the
skeleton never touches — `question`, `cards`, `swap`, `contributions[]` (each
agent's answer, verbatim, with the agent type, model, timing and the slice it was
given) and `call` (`decision`, `decided_by`, `pilot_words`, `reading`). `facts`
recomputes on `facts --write` and `install`; `with_authored` carries the rest.

THE FREEZE. A decided study is a record: once `call.decision` is not `open` and
`install` has stamped `decided_at`, `facts --write` prints the diff and refuses
without `--force` — the rule a decided branch already lives under.

THE AGENTS WRITE NOTHING, STILL. Jarvis spawns each by type (the band times it)
and persists the answer with `study add --agent <type> --from <file>|--text`.
`validate_study.py` is the gate: `decided_by` is required and the goldfish may
never be its sole member.
"""

import datetime
import hashlib
import json
import re

from manamap.pilot import card_dossier
from manamap.pilot.common import deck_dir, decklist_sha256, load_json

STUDIES_DIR = "studies"
DECISIONS = ("approve", "decline", "no-change", "open")
#: Who or what a call may rest on. `pilot` is the pilot's own word; the rest are
#: the bench's sources. `goldfish` is admitted only beside `pilot`, and only when
#: every card the study names is one the model reads (`validate_study`).
DECIDED_BY = ("pilot", "log", "strategist", "rules-question", "data-analyst", "card-scout",
              "deck-doctor", "recon", "stacks", "engine", "context", "goldfish", "forge",
              "mana", "deckcheck")
#: The agents whose answers a study keeps. Each is spawned by its own type.
AGENTS = ("strategist", "rules-question", "data-analyst", "card-scout", "deck-doctor")
#: The charter's `model:` line, read once; a contribution records which model wrote it.
CHARTERS = ".claude/agents"

FACT_ID_RE = re.compile(r"\bS-[0-9a-f]{8}-\d{2,3}\b")
#: A Comprehensive Rules number as it appears in prose: `614.5`, `601.2f`, `903.8`.
#: First digit 1-9 so a zero-padded log id (`003`) is never read as a rule.
CR_RE = re.compile(r"(?<![\w.])([1-9]\d{2}(?:\.\d+[a-z]?)?)(?![\w.])")
STRATEGY_RE = re.compile(r"\bstrategy:[a-z0-9-]+(?:\.[a-z0-9-]+){0,2}\b")

#: Fact kinds that belong to ONE card even when two cards' texts coincide.
PER_CARD_KINDS = frozenset({"oracle", "coverage", "neighbours", "price", "card_read", "log",
                            "protected", "engine_stage", "context"})

#: The goldfish rows a study quotes from a branch's `net_change.json`, by name.
GOLDFISH_ROWS = ("damage @T10", "killed by T10", "board power @T6", "extra cards by T8",
                 "missed drop by T5", "mulliganed")


# ── paths and ids ─────────────────────────────────────────────────────────────

def studies_dir(slug):
    return deck_dir(slug) / STUDIES_DIR


def _kebab(text, limit=48):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:limit].rstrip("-")) or "study"


def path_of(slug, study_id):
    d = studies_dir(slug)
    hits = sorted(d.glob(f"{study_id}-*.json")) if d.is_dir() else []
    if not hits:
        raise SystemExit(f"{slug}: no study {study_id!r} under {d} — `manamap pilot study {slug} list`")
    return hits[0]


def list_studies(slug):
    d = studies_dir(slug)
    out = []
    for p in sorted(d.glob("[0-9][0-9][0-9]-*.json")) if d.is_dir() else []:
        doc = load_json(p)
        if isinstance(doc, dict):
            out.append(doc)
    return out


def next_id(slug):
    return f"{max((int(s['id']) for s in list_studies(slug)), default=0) + 1:03d}"


def read(slug, study_id):
    return load_json(path_of(slug, study_id)) or {}


def write(slug, doc):
    d = studies_dir(slug)
    d.mkdir(exist_ok=True)
    p = d / f"{doc['id']}-{doc.get('slug_name') or _kebab(doc.get('question'))}.json"
    doc.setdefault("slug_name", p.stem[4:])
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def fact_id(slug, study_id, sha, n):
    """`S-<sha8>-NN`: a hash of deck, study and the list studied, so two studies of
    one deck — or one study of two lists — never share a fact id."""
    key = f"{slug}:{study_id}:{sha or 'nolist'}"
    return f"S-{hashlib.sha256(key.encode()).hexdigest()[:8]}-{n:02d}"


def _today():
    return datetime.date.today().isoformat()


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ── the skeleton ──────────────────────────────────────────────────────────────

def _tier(x):
    return x if x in ("verified", "data", "coach") else "coach"


def skeleton(slug, doc):
    """Recompute `facts`, `goldfish_coverage` and the stamps on a study. Every fact
    is a pure read of a tracked file; the authored keys are not touched."""
    branch = doc.get("branch")
    sha = decklist_sha256(slug, branch) if (branch is None or _branch_exists(slug, branch)) else None
    facts, absent = [], []
    n = 0

    seen_texts = set()

    def add(kind, text, tier, source, **extra):
        nonlocal n
        # ONE FACT ONCE. A staging note, a stack or a recon finding that names both
        # the OUT and the IN card reaches the study through both dossiers, and the
        # second copy is the same fact; a per-card fact (its coverage, its price,
        # its oracle text) with the same words on two cards is two facts.
        key = (kind, text, extra.get("card") if kind in PER_CARD_KINDS else None)
        if key in seen_texts:
            return None
        seen_texts.add(key)
        n += 1
        f = {"id": fact_id(slug, doc["id"], sha, n), "kind": kind, "text": text,
             "tier": _tier(tier), "source": source}
        f.update({k: v for k, v in extra.items() if v is not None})
        facts.append(f)
        return f

    # the deck's record, from the one composer every surface reads
    try:
        from manamap.pilot import deck_info
        rec = deck_info.compose(slug, ladder=False).get("record") or {}
    except Exception as exc:                            # noqa: BLE001
        rec, _ = {}, absent.append({"source": "deck_info.record", "why": str(exc)[:120]})
    if rec.get("games"):
        causes = ", ".join(f"{k} {v}" for k, v in sorted((rec.get("cause_counts") or {}).items(),
                                                         key=lambda kv: (-kv[1], kv[0])))
        add("record", f"{rec['games']} game(s): {rec.get('win', 0)}W {rec.get('loss', 0)}L"
                      + (f" {rec['draw']}D" if rec.get("draw") else "")
                      + (f"; how they ended: {causes}" if causes else "")
                      + (f"; not yet debriefed: {', '.join(rec['undebriefed'])}" if rec.get("undebriefed") else ""),
            "data", "log.jsonl + log_causes.json", as_of=rec.get("last_played"),
            n=rec["games"], figure={"win": rec.get("win", 0), "loss": rec.get("loss", 0)})
    else:
        add("record", "no games logged", "data", "log.jsonl")

    # dated external evidence, deck-level
    dc = load_json(deck_dir(slug) / "deckcheck.json") or {}
    reading = (dc.get("readings") or [None])[-1]
    if reading:
        add("deckcheck", f"DeckCheck bracket {reading.get('bracket')} score {reading.get('score')}"
                         f"/{reading.get('max_score')} as of {reading.get('as_of')}"
                         + ("" if (reading.get("decklist_sha256") == decklist_sha256(slug))
                            else " (an older list)"),
            "data", "deckcheck.json", as_of=reading.get("as_of"), figure=reading.get("score"))
    recon = load_json(deck_dir(slug) / "deck_recon.json") or {}
    if recon.get("as_of"):
        add("recon", f"recon as of {recon['as_of']}: {str(recon.get('consensus') or '')[:260]}",
            "coach", "deck_recon.json", as_of=recon["as_of"])

    # every card the study names, through the dossier
    cards = list(doc.get("cards") or [])
    cov = {"named": 0, "seen": [], "partial": [], "dark": [], "invisible": [], "unknown": []}
    for card in cards:
        d = card_dossier.dossier(slug, card, branch if _branch_exists(slug, branch) else None)
        cov["named"] += 1
        st = d.get("coverage")
        cov[st if st in ("seen", "partial", "dark", "invisible") else "unknown"].append(d["canonical"])
        for f in d["facts"]:
            add(f["kind"], f["text"], f["tier"], f["source"], card=f.get("card"),
                **{k: v for k, v in f.items() if k not in ("kind", "text", "tier", "source", "card")})
        absent.extend({"card": d["canonical"], **a} for a in d["absent"])
    read_n = len(cov["seen"]) + len(cov["partial"])
    cov["share"] = round(read_n / cov["named"], 3) if cov["named"] else None

    # the ledger on this branch, deck-level
    if branch:
        try:
            from manamap.pilot import decisions
            for e in decisions.read(slug):
                if e.get("branch") == branch:
                    add("decision", f"decision {e['id']} {e['kind']} ({str(e.get('at'))[:10]})"
                                    + (f": {str(e.get('reason') or '')[:160]}" if e.get("reason") else ""),
                        "coach", "decisions.jsonl", as_of=str(e.get("at"))[:10])
        except SystemExit as bad:
            absent.append({"source": "decisions.jsonl", "why": str(bad)[:120]})

    # the instrument, LAST, labelled: the branch's goldfish report if one exists
    nc = load_json(deck_dir(slug, branch) / "net_change.json") if _branch_exists(slug, branch) else None
    if nc:
        current = nc.get("decklist_sha256") == sha
        rows = {str(r.get("measure")): r for r in (nc.get("table") or []) if isinstance(r, dict)}
        label = (f"goldfish, {len(cov['seen']) + len(cov['partial'])} of {cov['named']} named card(s) read"
                 + ("" if current else "; an OLDER list of this branch"))
        rec_ = nc.get("recommendation") or {}
        games = (nc.get("harness") or {}).get("iterations") if isinstance(nc.get("harness"), dict) else None
        add("goldfish_profile", f"{label}: {len(rec_.get('rose') or [])} rows better, "
                                f"{len(rec_.get('fell') or [])} worse, {len(rec_.get('no_call') or [])} no call"
                                + (f" over {games} paired games" if games else "") + ". "
                                + card_dossier.ASTERISK,
            "data", f"branches/{branch}/net_change.json", instrument="goldfish",
            current=current, coverage_share=cov["share"], n=games)
        for m in GOLDFISH_ROWS:
            r = rows.get(m)
            if not r or r.get("delta") is None:
                continue
            ci = r.get("ci95_diff") or [None, None]
            add("goldfish_row", f"{m}: {r.get('reads_as') or r.get('delta')}"
                                + (f" [{ci[0]}, {ci[1]}]" if ci[0] is not None else "")
                                + f" — {r.get('verdict') or ''}".rstrip(" —"),
                "data", f"branches/{branch}/net_change.json#{m}", instrument="goldfish",
                figure=r.get("delta"), ci95=ci if ci[0] is not None else None, measure=m,
                verdict=r.get("verdict"))
    elif branch and _branch_exists(slug, branch):
        absent.append({"source": "net_change.json", "why": "the instrument has not read this branch"})

    doc["facts"] = facts
    doc["goldfish_coverage"] = cov
    doc["absent"] = absent
    doc["decklist_sha256"] = sha
    doc["base_decklist_sha256"] = decklist_sha256(slug) if branch else None
    doc["_meta"] = {"v": 1, "built_at": _now(), "facts": len(facts)}
    return doc


def _branch_exists(slug, branch):
    if not branch:
        return False
    return (deck_dir(slug) / "branches" / branch).is_dir()


# ── the authored half ─────────────────────────────────────────────────────────

AUTHORED = ("question", "cards", "swap", "contributions", "call", "opened", "decided_at",
            "backfilled", "slug_name")


def with_authored(doc, previous):
    for k in AUTHORED:
        if k in previous:
            doc[k] = previous[k]
    return doc


def new(slug, question, branch=None, out=None, in_=None, cards=(), backfilled=False, opened=None):
    if not (question or "").strip():
        raise SystemExit("a study needs its question — the thing Jarvis was asked")
    if branch and not _branch_exists(slug, branch):
        raise SystemExit(f"{slug}: no branch {branch!r} — `manamap pilot deck-branch {slug} list`")
    names = list(dict.fromkeys([c for c in ([out, in_] + list(cards)) if c]))
    doc = {"id": next_id(slug), "slug": slug, "branch": branch, "question": question.strip(),
           "opened": opened or _today(), "decided_at": None,
           "cards": names, "swap": ({"out": out, "in": in_} if (out or in_) else None),
           "contributions": [], "call": {"decision": "open", "decided_by": [], "pilot_words": "", "reading": ""}}
    if backfilled:
        doc["backfilled"] = True
    doc["slug_name"] = _kebab(question)
    return skeleton(slug, doc)


def _charter_model(agent):
    import pathlib
    p = pathlib.Path(CHARTERS) / f"{agent}.md"
    try:
        for line in p.read_text(encoding="utf-8").splitlines()[:12]:
            if line.startswith("model:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def cites_in(text):
    """The citations a contribution carries, extracted from its text: fact ids,
    `strategy:<id>` sections and candidate CR numbers (form only; the rules DB is
    gitignored, so existence is the rules-question agent's job, not this file's)."""
    return {"facts": sorted(set(FACT_ID_RE.findall(text or ""))),
            "strategy": sorted(set(STRATEGY_RE.findall(text or ""))),
            "rules": sorted(set(CR_RE.findall(text or "")))}


def add_contribution(doc, agent, text, model=None, spawned_at=None, elapsed_s=None, slice_=None,
                     commands=(), prescription=None, source_file=None):
    if agent not in AGENTS:
        raise SystemExit(f"agent {agent!r} is not one of {AGENTS}")
    if not (text or "").strip():
        raise SystemExit("a contribution needs the agent's text (--from FILE or --text)")
    c = {"agent": agent, "model": model or _charter_model(agent), "spawned_at": spawned_at,
         "elapsed_s": elapsed_s, "slice": slice_, "text": text.strip(), "cites": cites_in(text),
         "added_at": _now()}
    if commands:
        c["commands"] = list(commands)
    if prescription:
        c["prescription"] = prescription
    if source_file:
        c["from"] = str(source_file)
    doc.setdefault("contributions", []).append({k: v for k, v in c.items() if v is not None})
    return c


def set_call(doc, decision, decided_by, pilot_words, reading=None):
    if decision not in DECISIONS:
        raise SystemExit(f"decision {decision!r} is not one of {DECISIONS}")
    by = [b.strip() for b in (decided_by or []) if b and b.strip()]
    bad = [b for b in by if b not in DECIDED_BY]
    if bad:
        raise SystemExit(f"decided_by {bad} not in {DECIDED_BY}")
    doc["call"] = {"decision": decision, "decided_by": by, "pilot_words": (pilot_words or "").strip(),
                   "reading": (reading or "").strip(), "at": _today()}
    return doc["call"]


def is_frozen(doc):
    return bool(doc.get("decided_at")) and (doc.get("call") or {}).get("decision", "open") != "open"


def install(slug, study_id):
    """Strict validate, stamp `decided_at`, freeze. The one step that makes a
    study a record."""
    from manamap.pilot import validate_study
    doc = read(slug, study_id)
    doc = with_authored(skeleton(slug, {k: v for k, v in doc.items()}), doc)
    errors, notes = validate_study.validate(doc, slug, strict=True)
    if errors:
        raise SystemExit(f"refusing to install study {study_id} — " + f"{len(errors)} error(s):\n  - "
                         + "\n  - ".join(errors))
    if (doc.get("call") or {}).get("decision", "open") == "open":
        raise SystemExit(f"study {study_id} is still open — `study {slug} call {study_id} --decision … "
                         f"--decided-by … --pilot-words \"…\"` first")
    doc["decided_at"] = doc.get("decided_at") or _today()
    return write(slug, doc), notes


# ── printing ──────────────────────────────────────────────────────────────────

_MARK = {"verified": "✓", "data": "◆", "coach": "★"}


def _line(doc):
    call = doc.get("call") or {}
    cov = doc.get("goldfish_coverage") or {}
    by = ", ".join(call.get("decided_by") or []) or "—"
    read_n = len(cov.get("seen") or []) + len(cov.get("partial") or [])
    return (f"{doc['id']}  {call.get('decision', 'open'):9} {by:28} {str(doc.get('decided_at') or doc.get('opened') or '')[:10]}  "
            f"{(doc.get('branch') or '—'):12} {doc.get('question', '')[:70]}"
            + (f"  · goldfish saw {read_n}/{cov.get('named')}" if cov.get("named") else ""))


def show(doc, facts_only=False):
    out = []
    call = doc.get("call") or {}
    if not facts_only:
        out.append(f"STUDY {doc['id']} — {doc['slug']}" + (f" @ {doc['branch']}" if doc.get("branch") else ""))
        out.append(f"  Q: {doc.get('question')}")
        if doc.get("swap"):
            out.append(f"  swap: {doc['swap'].get('out')} → {doc['swap'].get('in')}")
        out.append(f"  call: {call.get('decision', 'open').upper()}"
                   + (f" — decided by {', '.join(call.get('decided_by') or [])}" if call.get("decided_by") else "")
                   + (f" ({doc.get('decided_at')})" if doc.get("decided_at") else " (open)"))
        if call.get("pilot_words"):
            out.append(f"  pilot: “{call['pilot_words']}”")
        if call.get("reading"):
            out.append(f"  reading: {call['reading']}")
    cov = doc.get("goldfish_coverage") or {}
    if cov.get("named"):
        read_n = len(cov.get("seen") or []) + len(cov.get("partial") or [])
        out.append(f"  goldfish saw {read_n} of {cov['named']} named card(s)"
                   + (f"; dark: {', '.join(cov['dark'])}" if cov.get("dark") else "")
                   + (f"; invisible: {', '.join(cov['invisible'])}" if cov.get("invisible") else ""))
    out.append(f"  FACTS ({len(doc.get('facts') or [])}) — ✓ verified  ◆ data  ★ judgment")
    for f in doc.get("facts") or []:
        out.append(f"    {f['id']}  {_MARK.get(f['tier'], '?')} {f['kind']:14} {f['text'][:150]}")
    if doc.get("absent"):
        out.append("  absent: " + "; ".join(f"{a.get('card', '')} {a['source']} ({a['why']})".strip() for a in doc["absent"][:8]))
    if not facts_only:
        for c in doc.get("contributions") or []:
            head = f"  — {c['agent']}" + (f" ({c.get('model')})" if c.get("model") else "") \
                   + (f", {c['elapsed_s']} s" if c.get("elapsed_s") is not None else "")
            out.append(head)
            for ln in (c.get("text") or "").splitlines():
                out.append(f"      {ln}")
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args):
    slug = args.slug
    action = getattr(args, "action", None) or "list"
    ref = getattr(args, "ref", None)
    as_json = getattr(args, "as_json", False)

    if action == "list":
        docs = list_studies(slug)
        if as_json:
            print(json.dumps([{k: d.get(k) for k in ("id", "question", "branch", "opened", "decided_at", "call", "swap")}
                              for d in docs], indent=2, ensure_ascii=False))
            return
        if not docs:
            print(f"{slug}: no studies — `manamap pilot study {slug} new \"<question>\" [--branch B] [--out A --in B]`")
            return
        print(f"STUDIES — {slug} ({len(docs)})")
        for d in docs:
            print("  " + _line(d))
        return

    if action == "new":
        doc = new(slug, ref, branch=getattr(args, "branch", None), out=getattr(args, "swap_out", None),
                  in_=getattr(args, "swap_in", None), cards=getattr(args, "card", None) or (),
                  backfilled=getattr(args, "backfilled", False), opened=getattr(args, "opened", None))
        p = write(slug, doc)
        print(f"{slug}: study {doc['id']} opened — {len(doc['facts'])} fact(s) from the record -> {p}")
        print(f"  next: hand `study {slug} show {doc['id']} --facts` to each agent; `study {slug} add {doc['id']} "
              f"--agent <type> --from FILE`; then `call` and `install`")
        return

    if not ref:
        raise SystemExit(f"`study {slug} {action}` needs the study id (NNN)")
    doc = read(slug, ref)

    if action == "show":
        print(json.dumps(doc, indent=2, ensure_ascii=False) if as_json
              else show(doc, facts_only=getattr(args, "facts_only", False)))
        return

    if action == "facts":
        fresh = with_authored(skeleton(slug, {k: v for k, v in doc.items()}), doc)
        before = {(f["kind"], f["text"]) for f in doc.get("facts") or []}
        after = {(f["kind"], f["text"]) for f in fresh["facts"]}
        gained, lost = after - before, before - after
        if as_json:
            print(json.dumps(fresh, indent=2, ensure_ascii=False))
        else:
            print(f"{slug}: study {ref} — {len(fresh['facts'])} fact(s); +{len(gained)} new, -{len(lost)} gone")
            for k, t in sorted(gained)[:12]:
                print(f"  + {k:12} {t[:120]}")
            for k, t in sorted(lost)[:12]:
                print(f"  - {k:12} {t[:120]}")
        if getattr(args, "write", False):
            if is_frozen(doc) and not getattr(args, "force", False):
                raise SystemExit(f"study {ref} is decided ({doc.get('decided_at')}) and its facts are the record "
                                 f"the call rested on — `--force` to recompute them anyway")
            print(f"  wrote {write(slug, fresh)}")
        return

    if action == "add":
        if is_frozen(doc):
            raise SystemExit(f"study {ref} is decided — a late contribution belongs in a new study")
        text = getattr(args, "text", None)
        src = getattr(args, "source_file", None)
        if src:
            import pathlib
            text = pathlib.Path(src).read_text(encoding="utf-8")
        add_contribution(doc, getattr(args, "agent", None), text, model=getattr(args, "agent_model", None),
                         spawned_at=getattr(args, "spawned_at", None), elapsed_s=getattr(args, "elapsed", None),
                         slice_=getattr(args, "slice", None), commands=getattr(args, "commands_run", None) or (),
                         prescription=getattr(args, "prescription", None), source_file=src)
        p = write(slug, doc)
        c = doc["contributions"][-1]
        print(f"{slug}: study {ref} + {c['agent']} ({c.get('model') or '?'}) — cites {len(c['cites']['facts'])} fact(s), "
              f"{len(c['cites']['strategy'])} strategy, {len(c['cites']['rules'])} CR -> {p}")
        return

    if action == "call":
        by = []
        for chunk in (getattr(args, "decided_by", None) or []):
            by += [b.strip() for b in chunk.split(",")]
        set_call(doc, getattr(args, "decision", None), by, getattr(args, "pilot_words", None),
                 reading=getattr(args, "reading", None))
        from manamap.pilot import validate_study
        errors, notes = validate_study.validate(doc, slug)
        for n_ in notes:
            print(f"  ! {n_}")
        if errors:
            raise SystemExit("the call does not hold:\n  - " + "\n  - ".join(errors))
        p = write(slug, doc)
        print(f"{slug}: study {ref} — {doc['call']['decision'].upper()} decided by {', '.join(by)} -> {p}")
        print(f"  next: `manamap pilot study {slug} install {ref}` to stamp and freeze it")
        return

    if action == "install":
        p, notes = install(slug, ref)
        for n_ in notes:
            print(f"  ! {n_}")
        print(f"{slug}: study {ref} installed and frozen -> {p}")
        return

    raise SystemExit(f"unknown study action {action!r}")
