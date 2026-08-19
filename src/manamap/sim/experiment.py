"""Simulation: the controlled experiment — two versions of one deck, same table, one delta.

THE QUESTION IT ANSWERS. "Does this swap make the deck better?" was assembled by hand:
tag a version, run `simulate`, swap, commit, run again, compare two records by eye. This
is that assembly as one command and ONE artifact — A and B run against the same
opponents, the same games-per-arm, the same AI profiles and the same engine build, and
the artifact reports each figure for both arms with its interval and the difference,
plus the one sentence people skip: whether the intervals overlap at this N.

WHAT IS CONTROLLED, AND WHAT HONESTLY IS NOT. Same table, same N, same profile, same
Forge build, same seed set: controlled. **Same seeds do NOT pair games across arms** —
a changed list changes every shuffle, so game 3 of arm A and game 3 of arm B share
nothing but a starting number. The seeds buy replayability per arm, never a paired
test; the control is N, and the artifact says so in `assumptions`.

An ARM is a version ref (`V4`, a tag, a sha prefix — anything `deck_versions.resolve`
takes) or the literal `working` (the current `decklist.txt`, committed or not). Arms
run under their own Forge meta names (`mm-x-<slug>-a` / `-b`) and never touch the deck
directory — an experiment must be runnable on a version you are NOT holding.

The artifact accumulates under `data/decks/<slug>/experiments/` like a prescription:
it is a record of a question asked of the table, and a later decklist does not make an
old answer wrong. Logs sit beside it under `experiments/logs/<id>/` and are gitignored
(exactly regenerable: each arm's decklist text is IN the artifact).
"""

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from manamap.config import SIM_DEFAULT_GAMES, SIM_GAME_CLOCK_SECONDS
from manamap.pilot.common import deck_dir, load_json
from manamap.pilot import deck_versions as dv
from manamap.sim import parse as sim_parse
from manamap.sim.forge import (ASSUMPTIONS, FORGE_AI_CAVEAT, _java_version, _seat_label,
                               command, forge_jar, forge_version, install_deck,
                               install_named, seat_sha, split_games)

EXP_DIR = "experiments"
# The aggregate keys the delta reports, and where they live in a seat's analysis.
DELTA_KEYS = (
    ("win_rate", ("win_rate",)),
    ("eliminated_turn", ("eliminated_turn", "mean")),
    ("combat_damage_dealt_to_players", ("combat_damage_dealt_to_players", "mean")),
    ("combat_damage_taken", ("combat_damage_taken", "mean")),
    ("first_attack_turn", ("first_attack_turn", "mean")),
    ("token_damage_share", ("tokens", "token_damage_share", "mean")),
    ("tokens_observed", ("tokens", "tokens_observed", "mean")),
    ("token_resolutions", ("tokens", "token_resolutions", "mean")),
)


def resolve_arm(slug, ref):
    """A ref → {ref, label, decklist_text, decklist_sha256}. `working` is the file."""
    if str(ref).strip().lower() == "working":
        text = (deck_dir(slug) / "decklist.txt").read_text(encoding="utf-8")
        return {"ref": "working", "label": "working copy",
                "decklist_text": text,
                "decklist_sha256": hashlib.sha256(text.encode()).hexdigest()}
    v = dv.resolve(slug, ref)
    if v is None:
        raise SystemExit(f"{slug}: no version {ref!r} — `deck-version {slug} list` names "
                         f"them (V4, a tag, a sha prefix), or use `working`")
    text = dv.blob_at(slug, v)
    if text is None:
        raise SystemExit(f"{slug}: cannot read V{v['version']} from git")
    return {"ref": str(ref), "label": f"V{v['version']} ({v['first_date']}, {v['subject'][:50]})",
            "decklist_text": text, "decklist_sha256": v["decklist_sha256"]}


def experiment_id(slug, a, b, opponents, games, seed):
    digest = hashlib.sha256("\n".join([
        a["decklist_sha256"], b["decklist_sha256"],
        *(f"{o}:{seat_sha(o)}" for o in opponents)]).encode()).hexdigest()[:8]
    return f"{_safe(a['ref'])}-vs-{_safe(b['ref'])}-x-{'-'.join(opponents)}-n{games}-{digest}-s{seed}"


def _safe(ref):
    return str(ref).replace("/", "-").replace(" ", "-").lower()


def _dig(d, path):
    for k in path:
        d = (d or {}).get(k)
    return d


def delta(analysis_a, analysis_b, slug):
    """Per-figure a/b/diff, plus whether the win-rate intervals overlap."""
    sa = (analysis_a.get("seats") or {}).get(slug, {})
    sb = (analysis_b.get("seats") or {}).get(slug, {})
    out = {}
    for name, path in DELTA_KEYS:
        va, vb = _dig(sa, path), _dig(sb, path)
        out[name] = {"a": va, "b": vb,
                     "diff": (round(vb - va, 3) if isinstance(va, (int, float))
                              and isinstance(vb, (int, float)) else None)}
    ca, cb = sa.get("win_rate_ci95") or [None, None], sb.get("win_rate_ci95") or [None, None]
    overlap = None
    if None not in (*ca, *cb):
        overlap = not (ca[1] < cb[0] or cb[1] < ca[0])
    out["win_rate"]["ci95_a"], out["win_rate"]["ci95_b"] = ca, cb
    out["win_rate"]["intervals_overlap"] = overlap
    out["reading"] = (
        "the win-rate intervals OVERLAP at this N — the difference is noise until more games say otherwise"
        if overlap else
        "the win-rate intervals are disjoint — a real difference at this N" if overlap is not None
        else "no interval available")
    return out


def _run_arm(arm_letter, meta_name, opp_names, games, jobs, clock, seed, profiles,
             log_dir, jar):
    parts = split_games(games, jobs)
    seeds = [seed + i for i in range(len(parts))]
    cmds = [command([meta_name, *opp_names], g, clock, jar, seed=seeds[i], profiles=profiles)
            for i, g in enumerate(parts)]
    log_dir.mkdir(parents=True, exist_ok=True)

    def one(i_cmd):
        i, cmd = i_cmd
        log = log_dir / f"{arm_letter}-part-{i:02d}.log"
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  cwd=str(jar.parent), text=True)
        return log, proc.returncode

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        results = list(ex.map(one, enumerate(cmds)))
    texts = [log.read_text(encoding="utf-8", errors="replace") for log, _ in results]
    return texts, seeds, sum(1 for _, rc in results if rc)


def run(slug, ref_a, ref_b, opponents, games=SIM_DEFAULT_GAMES, jobs=None,
        clock=SIM_GAME_CLOCK_SECONDS, seed=None, profile=None, dry_run=False):
    if not opponents:
        raise SystemExit("experiment needs at least one opponent: --vs <slug> (repeatable)")
    import os
    a, b = resolve_arm(slug, ref_a), resolve_arm(slug, ref_b)
    if a["decklist_sha256"] == b["decklist_sha256"]:
        raise SystemExit(f"both arms are the same list ({a['decklist_sha256'][:12]}…) — "
                         f"an A/A tells you the noise floor, which is legitimate, but "
                         f"say so by passing different refs to different lists")
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    seed = seed if seed is not None else int(hashlib.sha256(
        (a["decklist_sha256"] + b["decklist_sha256"]).encode()).hexdigest()[:8], 16) % 2_000_000_000
    eid = experiment_id(slug, a, b, opponents, games, seed)
    out_dir = deck_dir(slug) / EXP_DIR
    path = out_dir / f"{eid}.json"
    if path.exists() and not dry_run:
        raise SystemExit(f"{slug}: {path.name} exists — the same arms, table and seed replay "
                         f"the same games. A new sample is a new --seed.")
    profiles = ([profile] + ["Default"] * len(opponents)) if profile else None
    if dry_run:
        return path, {"experiment_id": eid, "arms": {"a": a["label"], "b": b["label"]},
                      "seed": seed, "games_per_arm": games, "profiles": profiles}

    jar = forge_jar()
    names_a = install_named(f"mm-x-{slug}-a", a["decklist_text"])
    names_b = install_named(f"mm-x-{slug}-b", b["decklist_text"])
    opp_names = [install_deck(o) for o in opponents]
    log_dir = out_dir / "logs" / eid
    t0 = time.time()
    texts_a, seeds_a, bad_a = _run_arm("a", names_a, opp_names, games, jobs, clock, seed, profiles, log_dir, jar)
    texts_b, seeds_b, bad_b = _run_arm("b", names_b, opp_names, games, jobs, clock, seed, profiles, log_dir, jar)
    wall = round(time.time() - t0, 1)

    label_a = _seat_label([names_a, *opp_names]); label_a[f"Ai(1)-{names_a}"] = slug
    label_b = _seat_label([names_b, *opp_names]); label_b[f"Ai(1)-{names_b}"] = slug
    _, analysis_a = sim_parse.analyze_logs(texts_a, label_a)
    _, analysis_b = sim_parse.analyze_logs(texts_b, label_b)

    doc = {
        "experiment_id": eid, "slug": slug, "at": date.today().isoformat(),
        "engine": {"forge": forge_version(), "java": _java_version()},
        "question": f"{a['label']}  vs  {b['label']}, same table",
        "opponents": [{"slug": o, "decklist_sha256": seat_sha(o)} for o in opponents],
        "games_per_arm": int(games), "seed_base": seed, "seeds": seeds_a,
        "profiles": profiles, "clock_seconds": clock, "wall_seconds": wall,
        "nonzero_exit_jobs": bad_a + bad_b,
        "arms": {
            "a": {"ref": a["ref"], "label": a["label"], "decklist_sha256": a["decklist_sha256"],
                  "decklist_text": a["decklist_text"], "games": analysis_a.get("games"),
                  "analysis": analysis_a},
            "b": {"ref": b["ref"], "label": b["label"], "decklist_sha256": b["decklist_sha256"],
                  "decklist_text": b["decklist_text"], "games": analysis_b.get("games"),
                  "analysis": analysis_b},
        },
        "delta": delta(analysis_a, analysis_b, slug),
        "assumptions": [
            "SAME TABLE, NOT PAIRED GAMES: both arms ran the same opponents, N, profiles, "
            "engine build and seed set — but a changed list changes every shuffle, so seeds "
            "buy per-arm replayability, never a paired test. The control is N.",
            *ASSUMPTIONS[:1],           # SEEDED
            FORGE_AI_CAVEAT,
            "Both arms are flown by the same AI, so a difference the AI cannot exploit "
            "(a held-up trick, a political line) will not show here.",
        ],
        "logs": f"{EXP_DIR}/logs/{eid}/ (gitignored; regenerable — each arm's decklist is in this file)",
    }
    out_dir.mkdir(exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path, doc


def list_all(slug):
    base = deck_dir(slug) / EXP_DIR
    return [load_json(p) for p in sorted(base.glob("*.json"))] if base.is_dir() else []


def main(args):
    slug = args.slug
    if getattr(args, "list", False) or not (getattr(args, "a", None) and getattr(args, "b", None)):
        docs = list_all(slug)
        if not docs:
            print(f"{slug}: no experiments — `manamap pilot experiment {slug} --a V5 --b working "
                  f"--vs <opp> [--vs …] --games N`")
            return
        print(f"EXPERIMENTS — {slug} ({len(docs)})\n")
        for d in docs:
            w = d["delta"]["win_rate"]
            print(f"{d['experiment_id'][:64]}  {d['at']}  n={d['games_per_arm']}/arm")
            print(f"      {d['question']}")
            print(f"      win {w['a']} → {w['b']}  ({'overlap' if w['intervals_overlap'] else 'DISJOINT'})")
        return
    path, doc = run(slug, args.a, args.b, args.vs, games=args.games or SIM_DEFAULT_GAMES,
                    jobs=args.jobs, clock=args.clock or SIM_GAME_CLOCK_SECONDS,
                    seed=getattr(args, "seed", None), profile=getattr(args, "profile", None),
                    dry_run=getattr(args, "dry_run", False))
    if getattr(args, "dry_run", False):
        print(f"would run {doc['games_per_arm']} games/arm, seed {doc['seed']} → {path.name}")
        return
    d = doc["delta"]
    print(f"{slug}: experiment {doc['experiment_id'][:64]}")
    print(f"  A {doc['arms']['a']['label']}")
    print(f"  B {doc['arms']['b']['label']}")
    print(f"  {doc['games_per_arm']} games/arm vs {', '.join(o['slug'] for o in doc['opponents'])} "
          f"in {doc['wall_seconds']}s")
    w = d["win_rate"]
    print(f"  win rate   A {w['a']} {w['ci95_a']}   B {w['b']} {w['ci95_b']}   Δ {w['diff']}")
    for k, _ in DELTA_KEYS[1:]:
        r = d[k]
        if r["a"] is not None or r["b"] is not None:
            print(f"  {k:<34} A {r['a']}   B {r['b']}   Δ {r['diff']}")
    print(f"  → {d['reading']}")
    print(f"  → {path.relative_to(deck_dir(slug))}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot experiment <slug> --a <ref> --b <ref> --vs <opp> …`.")
