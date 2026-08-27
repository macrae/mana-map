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

from manamap.config import SIM_DECK_PREFIX, SIM_DEFAULT_GAMES, SIM_GAME_CLOCK_SECONDS
from manamap.pilot.common import deck_dir, load_json
from manamap.pilot import deck_versions as dv
from manamap.sim import parse as sim_parse
from manamap.sim import stats
from manamap.sim.forge import (ASSUMPTIONS, FORGE_AI_CAVEAT, _commanders_by_slug,
                               _java_version, _seat_label, command, commanders_from_text,
                               forge_jar, forge_version, install_deck, install_named,
                               seat_sha, split_games)

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
    # Per DEFENDER — 21 from one commander on one player is a whole archetype's only
    # win condition, and dealt_total cannot see it: 60 spread over three seats wins
    # nothing. Absent on both arms when the commander is unknown, so an arm that
    # cannot be scored reports None rather than 0.
    ("commander_damage_max_on_one_defender",
     ("commander_damage", "max_on_one_defender", "mean")),
    ("commander_damage_dealt_total", ("commander_damage", "dealt_total", "mean")),
    ("commander_damage_games_reaching_21", ("commander_damage", "games_reaching_21")),
)


def resolve_arm(slug, ref):
    """A ref → {ref, label, decklist_text, decklist_sha256}.

    Three kinds of ref, because there are three ways a list exists here:
    `working` is the file on disk, `V4`/a tag/a sha is a version out of git, and
    **`@<branch>` is a candidate list that is in neither** — designed, measurable,
    and deliberately not committed to `decklist.txt` because the cards are not
    all in the pilot's hands yet. Without this the most useful A/B in the system
    is unsayable: the branch you are considering against the deck you are
    playing, same pod, same seed.
    """
    ref_s = str(ref).strip()
    if ref_s.startswith("@"):
        from manamap.pilot.common import deck_dir as _dd
        branch = ref_s[1:]
        text = (_dd(slug, branch) / "decklist.txt").read_text(encoding="utf-8")
        return {"ref": ref_s, "label": f"branch {branch}",
                "decklist_text": text,
                "decklist_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if ref_s.lower() == "working":
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


# How to read a per-game row for each figure that is a MEAN. The aggregates
# already carry the mean; these give the raw distribution, which is what a Welch
# interval, a permutation test and a bootstrap all need and none of which can be
# recovered from a rounded `ci95` half-width.
PER_GAME = {
    "eliminated_turn": lambda p: p.get("eliminated_turn"),
    "combat_damage_dealt_to_players": lambda p: p.get("combat_damage_dealt_to_players"),
    "combat_damage_taken": lambda p: p.get("combat_damage_taken"),
    "first_attack_turn": lambda p: p.get("first_attack_turn"),
    "token_damage_share": lambda p: p.get("token_damage_share"),
    "tokens_observed": lambda p: p.get("tokens_observed"),
    "token_resolutions": lambda p: p.get("token_resolutions"),
    "commander_damage_max_on_one_defender": lambda p: p.get("commander_damage_max"),
    "commander_damage_dealt_total": lambda p: (
        sum((p.get("commander_damage_by_defender") or {}).values())
        if p.get("commander_damage_by_defender") is not None else None),
}

# Counts out of games, not means — so they get Newcombe rather than Welch.
PROPORTIONS = ("win_rate", "commander_damage_games_reaching_21")

# Figures whose sample is routinely mostly zeros with a long tail. A t interval on
# `0 0 0 0 0 0 0 0 0 0 31 178` is a true number describing no game, so these also
# report a bootstrap interval on the MEDIAN. Measured, not guessed: that sample is
# arm B's real commander damage from the kianne experiment.
SKEWED = ("commander_damage_max_on_one_defender", "commander_damage_dealt_total",
          "combat_damage_dealt_to_players")

# The one figure permitted a verdict. Everything else is descriptive: eleven
# figures at alpha=0.05 means roughly one interval in two experiments excludes
# zero by chance, and a reader who treats them all as tests will find a result
# every time. This is a pre-registration, not a limitation.
PRIMARY_ENDPOINT = "win_rate"


def _per_game_values(games, slug, name):
    """The arm's raw per-game values for one figure, or None if unavailable."""
    fn = PER_GAME.get(name)
    if fn is None or not games:
        return None
    out = []
    for g in games:
        row = (g.get("per_seat") or {}).get(slug)
        if row is None:
            continue
        out.append(fn(row))
    vals = [v for v in out if v is not None]
    return vals if vals else None


def delta(analysis_a, analysis_b, slug, games_a=None, games_b=None):
    """Per-figure a/b/diff with an interval ON THE DIFFERENCE, plus power.

    WHAT THIS REPLACED, AND WHY. The previous version compared eleven figures and
    tested one, by asking whether the two win-rate intervals OVERLAPPED, then
    reading an overlap as "the difference is noise until more games say
    otherwise". That is the overlap fallacy in the artifact's own voice:
    non-overlap does imply a difference, but overlap implies nothing at all,
    because two marginal intervals can overlap while the interval on their
    difference excludes zero. The other ten figures had their `ci95` blocks
    sitting one dict level away and unread.

    Every figure now carries `ci95_diff`, its N and the method that produced it,
    and no figure carries a bare `diff`. Only `win_rate` carries a verdict.

    And `power` answers the question the old artifact could not: not "was there a
    difference" but "what difference could this experiment have found at all". At
    twelve games an arm the answer is almost nothing, and saying so is the most
    useful sentence in the file.
    """
    sa = (analysis_a.get("seats") or {}).get(slug, {})
    sb = (analysis_b.get("seats") or {}).get(slug, {})
    n_a = analysis_a.get("games") or 0
    n_b = analysis_b.get("games") or 0
    out = {}

    for name, path in DELTA_KEYS:
        va, vb = _dig(sa, path), _dig(sb, path)
        row = {"a": va, "b": vb, "n_a": n_a, "n_b": n_b,
               "diff": (round(vb - va, 3) if isinstance(va, (int, float))
                        and isinstance(vb, (int, float)) else None)}

        if name in PROPORTIONS:
            k_a = sa.get("wins") if name == "win_rate" else _dig(
                sa, ("commander_damage", "games_reaching_21"))
            k_b = sb.get("wins") if name == "win_rate" else _dig(
                sb, ("commander_damage", "games_reaching_21"))
            if None not in (k_a, k_b) and n_a and n_b:
                d = stats.diff_proportions(int(k_a), n_a, int(k_b), n_b)
                row.update({"ci95_diff": d["ci95"], "excludes_zero": d["excludes_zero"],
                            "method": d["method"]})
            else:
                row["method"] = "not measured on both arms"
        else:
            xs = _per_game_values(games_a, slug, name)
            ys = _per_game_values(games_b, slug, name)
            d = stats.diff_means(xs, ys) if xs and ys else None
            if d:
                row.update({"ci95_diff": d["ci95"], "excludes_zero": d["excludes_zero"],
                            "df": d.get("df"), "method": d["method"],
                            "permutation_p": stats.permutation_p(xs, ys, seed=n_a)})
                if name in SKEWED:
                    m = stats.diff_medians(xs, ys, seed=n_a)
                    if m:
                        row["median_diff"] = m["diff"]
                        row["median_ci95_diff"] = m["ci95"]
            else:
                # An older artifact whose logs are gone re-derives to a bare diff
                # rather than failing. Saying so is the point: an absent interval
                # must never read as a narrow one.
                row["method"] = "no per-game values available; difference is unbounded"

        out[name] = row

    # Power, on the primary endpoint only.
    wins_a = sa.get("wins")
    if n_a and n_b and wins_a is not None:
        p_a = wins_a / n_a
        mde = stats.mde_proportion(p_a, n_a, n_b)
        out["power"] = {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "n_per_arm": n_a if n_a == n_b else [n_a, n_b],
            "alpha": 0.05, "target_power": 0.8,
            "baseline_rate_a": round(p_a, 4),
            **(mde or {"minimum_detectable_rate_b": None,
                       "note": "no rate for arm B would reach 80% power at this N"}),
            "games_per_arm_to_detect_0.10": stats.games_for_difference(p_a, 0.10),
            "method": ("exact enumeration of the two-binomial grid; the test is the "
                       "Newcombe score interval on the difference excluding zero"),
        }

    out["reading"] = _reading(out)
    return out


def _reading(out):
    """One sentence, and it must not commit the fallacy it replaced.

    Two clauses on purpose. The first says what the interval on the difference
    does; the second says what the experiment could have detected — because "we
    found nothing" and "we could not have found anything" are different
    statements and only one of them is about the deck.
    """
    w = out.get(PRIMARY_ENDPOINT) or {}
    ci = w.get("ci95_diff")
    power = out.get("power") or {}
    mde = power.get("minimum_detectable_difference")
    if not ci:
        return "no interval available for the win rate"
    if w.get("excludes_zero"):
        return (f"the 95% interval on the win-rate difference is [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                f"and EXCLUDES zero — a real difference at this N.")
    tail = ""
    if mde is not None:
        tail = (f" At {power.get('n_per_arm')} games per arm this experiment could only have "
                f"detected a difference of {mde:+.3f} or larger, so it is uninformative about "
                f"anything smaller — that is not evidence of no effect.")
    return (f"the 95% interval on the win-rate difference is [{ci[0]:+.3f}, {ci[1]:+.3f}] "
            f"and CONTAINS zero.{tail}")


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
    # Commander damage is per-defender and the parser cannot see a commander in a log,
    # so each arm is scored against ITS OWN list — the decklist text rides in the
    # artifact for exactly this reason. Without this the A/B, which is the tool for
    # judging a change to a commander-damage deck, was the one path blind to it.
    opp_cmd = _commanders_by_slug(opponents)
    def _cmd_map(meta_name, arm):
        m = {f"Ai(1)-{meta_name}": commanders_from_text(arm["decklist_text"])}
        for i, o in enumerate(opponents):
            if opp_cmd.get(o):
                m[f"Ai({i + 2})-{opp_names[i]}"] = opp_cmd[o]
        return {k: v for k, v in m.items() if v}
    # KEEP THE FACTS. `run` discarded them, so `delta` had only rounded aggregate
    # means to work with and could not compute an interval on a difference at all
    # — the raw distribution is what Welch, the permutation test and the bootstrap
    # all need. `forge.py` already does this for run records; this mirrors it.
    facts_a, analysis_a = sim_parse.analyze_logs(texts_a, label_a, _cmd_map(names_a, a))
    facts_b, analysis_b = sim_parse.analyze_logs(texts_b, label_b, _cmd_map(names_b, b))
    games_a = [sim_parse.compact(f, label_a) for f in facts_a]
    games_b = [sim_parse.compact(f, label_b) for f in facts_b]

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
                  "analysis": analysis_a, "games_detail": games_a},
            "b": {"ref": b["ref"], "label": b["label"], "decklist_sha256": b["decklist_sha256"],
                  "decklist_text": b["decklist_text"], "games": analysis_b.get("games"),
                  "analysis": analysis_b, "games_detail": games_b},
        },
        "delta": delta(analysis_a, analysis_b, slug, games_a, games_b),
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


def analyze(slug, experiment_id_or_path):
    """Re-derive both arms' analysis from their kept logs and rewrite the artifact.

    The same migration path `simulate --analyze` provides, and needed for the same
    reason: the parser gained commander damage after these logs were written, and an
    A/B whose figures cannot be re-derived is a claim nobody can check. Each arm is
    scored against ITS OWN decklist text, which rides in the artifact — so this needs
    the logs but never the deck directory, and an arm on a version you no longer hold
    still re-derives.
    """
    base = deck_dir(slug) / EXP_DIR
    name = experiment_id_or_path if str(experiment_id_or_path).endswith(".json") \
        else f"{experiment_id_or_path}.json"
    path = base / name
    if not path.exists():
        raise SystemExit(f"{slug}: no experiment {path.name} under {EXP_DIR}/")
    doc = load_json(path)
    log_dir = base / "logs" / path.stem
    opponents = [o["slug"] for o in doc["opponents"]]
    opp_names = [f"{SIM_DECK_PREFIX}{o}" for o in opponents]
    opp_cmd = _commanders_by_slug(opponents)
    for letter in ("a", "b"):
        logs = sorted(log_dir.glob(f"{letter}-part-*.log"))
        if not logs:
            raise SystemExit(f"{slug}: no {letter}-part-*.log under {log_dir} — an "
                             f"experiment's logs are gitignored and only exist where it ran")
        meta = f"mm-x-{slug}-{letter}"
        label = _seat_label([meta, *opp_names]); label[f"Ai(1)-{meta}"] = slug
        cmd = {f"Ai(1)-{meta}": commanders_from_text(doc["arms"][letter]["decklist_text"])}
        for i, o in enumerate(opponents):
            if opp_cmd.get(o):
                cmd[f"Ai({i + 2})-{opp_names[i]}"] = opp_cmd[o]
        texts = [l.read_text(encoding="utf-8", errors="replace") for l in logs]
        facts, analysis = sim_parse.analyze_logs(texts, label,
                                                 {k: v for k, v in cmd.items() if v})
        doc["arms"][letter]["analysis"] = analysis
        doc["arms"][letter]["games"] = analysis.get("games")
        doc["arms"][letter]["games_detail"] = [sim_parse.compact(f, label) for f in facts]
    doc["delta"] = delta(doc["arms"]["a"]["analysis"], doc["arms"]["b"]["analysis"], slug,
                         doc["arms"]["a"].get("games_detail"),
                         doc["arms"]["b"].get("games_detail"))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path, doc


def list_all(slug):
    base = deck_dir(slug) / EXP_DIR
    return [load_json(p) for p in sorted(base.glob("*.json"))] if base.is_dir() else []


def main(args):
    slug = args.slug
    if getattr(args, "analyze", None):
        path, doc = analyze(slug, args.analyze)
        d = doc["delta"]
        print(f"{slug}: re-derived both arms from logs → {path.name}")
        for k, _ in DELTA_KEYS:
            v = d[k]
            ci = v.get("ci95_diff")
            band = ("[%+.3f, %+.3f]" % (ci[0], ci[1])) if ci else "—"
            print(f"  {k:<40} A {v['a']}   B {v['b']}   Δ {v['diff']}   95% {band}")
        print(f"\n  {d.get('reading', '')}")
        return
    if getattr(args, "list", False) or not (getattr(args, "a", None) and getattr(args, "b", None)):
        docs = list_all(slug)
        if not docs:
            print(f"{slug}: no experiments — `manamap pilot experiment {slug} --a V5 --b working "
                  f"--vs <opp> [--vs …] --games N`")
            return
        print(f"EXPERIMENTS — {slug} ({len(docs)})\n")
        for d in docs:
            w = d["delta"]["win_rate"]
            ci = w.get("ci95_diff")
            verdict = ("DIFFERENT" if w.get("excludes_zero")
                       else "spans zero" if ci else "no interval")
            print(f"{d['experiment_id'][:64]}  {d['at']}  n={d['games_per_arm']}/arm")
            print(f"      {d['question']}")
            print(f"      win {w['a']} → {w['b']}   Δ95% "
                  f"{('[%+.3f, %+.3f]' % (ci[0], ci[1])) if ci else '—'}  ({verdict})")
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
