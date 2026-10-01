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
from manamap.sim import telemetry as _telemetry
from manamap.sim.forge import (ASSUMPTIONS, DEFAULT_PROFILE, FORGE_AI_CAVEAT,
                               card_overrides,
                               STANDARD_POD_PROFILE, TIMEOUT_FLOOR, TIMEOUT_SLACK,
                               _commanders_by_slug, _java_version, _profiles_for,
                               _seat_label, ai_profile_tag, clock_tag, telemetry_tag,
                               command, commanders_from_text, forge_jar,
                               forge_version, install_deck, install_named,
                               default_jobs, overrides_tag, profile_tag, seat_sha,
                               split_games, resolve_table as forge_resolve_table)
from manamap.sim import pods as _pods

#: The statuses a sequential record moves through. Derived from the looks it
#: holds; a record without `design` is a one-look experiment from before looks
#: existed and reads as `complete`.
STATUSES = ("running", "stopped_efficacy", "stopped_futility", "complete")
MAX_LOOKS = 4

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


def telemetry_for_run(plain=False):
    """The jar and the formatter fingerprint this experiment runs under. A named
    seam so a unit test can pin it, as the fixture pins `card_overrides`."""
    return _telemetry.jar_for_run(plain=plain)


def experiment_id(slug, a, b, opponents, games, seed,
                  profile=None, vs_profile=None, clock=None, overrides_sha=None,
                  ai_sha=None, looks=1, aa=False, profile_b=None, telemetry_sha=None):
    """The artifact's name, and — since it is also its identity — its receipt.

    THE PROFILES ARE IN IT FOR THE REASON `forge.profile_tag` GIVES: the digest
    is over decklists, opponents, games and seed, none of which move when the AI
    does, so two configurations would write the same path and the second would
    silently replace the first. `profile_tag` omits a suffix for Default, so
    every experiment already on disk keeps its id and still means what it said.

    AND THE REST OF THE HARNESS, as of 2026-09-29 — the clock, the card-script
    overrides and the per-deck AI profile — for the reason `forge.run_id` grew
    each of them: an axis missing from the id is a silent overwrite. Every tag
    is empty at its default, so every experiment on disk keeps its id. `-k{K}`
    marks a sequential design (its seed layout differs from a one-look run of
    the same N), `-aa` an A/A, and `-bme{P}` a policy arm — same list, our seat
    on another profile on arm B only.
    """
    digest = hashlib.sha256("\n".join([
        a["decklist_sha256"], b["decklist_sha256"],
        *(f"{o}:{seat_sha(o)}" for o in opponents)]).encode()).hexdigest()[:8]
    return (f"{_safe(a['ref'])}-vs-{_safe(b['ref'])}-x-{'-'.join(opponents)}"
            f"-n{games}-{digest}-s{seed}"
            + profile_tag(profile, vs_profile) + _clock_tag(clock)
            + overrides_tag(overrides_sha) + ai_profile_tag(ai_sha) + telemetry_tag(telemetry_sha)
            + (f"-bme{profile_b}" if profile_b else "")
            + ("-aa" if aa else "")
            + (f"-k{looks}" if looks and looks > 1 else ""))


def _clock_tag(clock):
    """Empty at the experiment's default clock, so every id on disk keeps its
    name; `forge.clock_tag` is frozen at the 300 s baseline run records were
    born under, and experiments were born at 600."""
    return "" if clock in (None, SIM_GAME_CLOCK_SECONDS) else f"-c{int(clock)}"


def _safe(ref):
    return str(ref).replace("/", "-").replace(" ", "-").lower()


def _dig(d, path):
    for k in path:
        d = (d or {}).get(k)
    return d


def _extra_draw_per_turn(hand):
    """Draw beyond the natural draw step, per own turn — or None without hand facts.

    The natural draw is one per own turn, except the first turn of the seat ON THE PLAY
    (`end_of_turn_size` carries turn 1 for that seat and no draw happened on it): the
    telemetry control game reads 6 library-to-hand moves over 7 own turns for the seat
    on the play and 7 over 7 for the draw — both are ZERO extra, not -0.14 and 0.
    Keys are ints in memory and strings on disk, so both spellings of turn 1 are read.
    """
    if not hand:
        return None
    sizes = hand.get("end_of_turn_size") or {}
    own = len(sizes)
    if not own or hand.get("library_to_hand") is None:
        return None
    on_the_play = 1 in sizes or "1" in sizes
    natural = own - (1 if on_the_play else 0)
    return round((hand["library_to_hand"] - natural) / own, 4)


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
    # THE DRAIN AXIS AND THE THREAT AXIS (2026-09-30)
    "drain_dealt": lambda p: p.get("drain_dealt"),
    "biggest_hit": lambda p: (p.get("biggest_hit") or {}).get("amount"),
    "evasive_damage_share": lambda p: (p.get("combat_damage_by_keyword") or {}).get("evasive_share"),
    "kills_by_ability": lambda p: (sum((p.get("kills_by_ability") or {}).values())
                                   if p.get("kills_by_ability") is not None else None),
    "life_gained": lambda p: (sum(v for v in (p.get("life_gained_by_source") or {}).values())
                              if p.get("life_gained_by_source") is not None else None),
    # THE DRAW AXIS (2026-09-30): the telemetry patch's hand facts, per own turn.
    "extra_draw_per_turn": lambda p: _extra_draw_per_turn(p.get("hand")),
    "empty_hand_turns": lambda p: (p.get("hand") or {}).get("empty_own_turns"),
}

# Counts out of games, not means — so they get Newcombe rather than Welch.
PROPORTIONS = ("win_rate", "commander_damage_games_reaching_21")

# Figures whose sample is routinely mostly zeros with a long tail. A t interval on
# `0 0 0 0 0 0 0 0 0 0 31 178` is a true number describing no game, so these also
# report a bootstrap interval on the MEDIAN. Measured, not guessed: that sample is
# arm B's real commander damage from the kianne experiment.
SKEWED = ("commander_damage_max_on_one_defender", "commander_damage_dealt_total",
          "combat_damage_dealt_to_players", "drain_dealt", "kills_by_ability", "life_gained",
          "extra_draw_per_turn", "empty_hand_turns")

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
    # WINS OVER DECIDED GAMES. The Newcombe interval on `win_rate` divided the
    # wins by every game PLAYED, clock-outs included, while the rate beside it
    # (`seats[].win_rate`) divides by decided games — two denominators for one
    # figure, the defect `forge.run` fixed for run records on 2026-09-04.
    dec_a = analysis_a.get("decided", n_a) or n_a
    dec_b = analysis_b.get("decided", n_b) or n_b
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
            den_a = dec_a if name == "win_rate" else n_a
            den_b = dec_b if name == "win_rate" else n_b
            if None not in (k_a, k_b) and den_a and den_b:
                d = stats.diff_proportions(int(k_a), den_a, int(k_b), den_b)
                row.update({"ci95_diff": d["ci95"], "excludes_zero": d["excludes_zero"],
                            "method": d["method"], "n_a": den_a, "n_b": den_b})
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
    if dec_a and dec_b and wins_a is not None:
        p_a = wins_a / dec_a
        mde = stats.mde_proportion(p_a, dec_a, dec_b)
        out["power"] = {
            "primary_endpoint": PRIMARY_ENDPOINT,
            "n_per_arm": dec_a if dec_a == dec_b else [dec_a, dec_b],
            "basis": "decided games per arm (clock-outs have no winner)",
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


def _run_wave(arm_letter, meta_name, opp_names, games, jobs, clock, seed_base,
              profile, vs_profile, log_dir, jar, first_job=0):
    """One wave of one arm: `jobs` JVMs, `games` split across them, SEATS
    ROTATED PER JOB exactly as `forge.run` rotates them.

    THE ARMS DID NOT ROTATE. `simulate` has rotated the `-d` order per job since
    the seat-bias finding — Forge gives turn 1 to the lowest-indexed seat that
    did not win the previous game, so seat 0 started 323 of 400 games in one
    run — and the experiment, the controlled instrument, kept our seat at index
    0 in every job of every arm. The job index is GLOBAL across waves
    (`first_job`), so a wave's jobs continue the rotation and the seeds rather
    than restarting both.
    """
    parts = split_games(games, jobs)
    seats = [meta_name, *opp_names]
    n = len(seats)
    job_ix = [first_job + i for i in range(len(parts))]
    orders = [seats[j % n:] + seats[:j % n] for j in job_ix]
    seeds = [seed_base + j for j in job_ix]
    cmds = [command(orders[i], g, clock, jar, seed=seeds[i],
                    profiles=_profiles_for(orders[i], meta_name, profile, vs_profile))
            for i, g in enumerate(parts)]
    log_dir.mkdir(parents=True, exist_ok=True)
    per_job_cap = int(clock * max(parts) * TIMEOUT_SLACK) + TIMEOUT_FLOOR

    def one(i_cmd):
        i, cmd = i_cmd
        log = log_dir / f"{arm_letter}-part-{job_ix[i]:02d}.log"
        with open(log, "w", encoding="utf-8") as f:
            # THE SAME CAP `forge.run` OBEYS, and for the same measured reason:
            # Forge's `-c` clock ends a game's accounting, not its AI thread, and
            # two tracked 20-game runs took 3.7 and 4.2 hours with 95% of the
            # wall claimed by no game at all. `_run_arm` had no timeout, so the
            # whole runaway class was unguarded on the flagship while being
            # fixed on `simulate`. The formula is imported, not retyped.
            try:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=str(jar.parent), text=True,
                                      timeout=per_job_cap)
                return log, proc.returncode
            except subprocess.TimeoutExpired:
                f.write(f"\n[manamap] job killed after {per_job_cap}s "
                        f"(clock {clock}s x {max(parts)} games x {TIMEOUT_SLACK} "
                        f"+ {TIMEOUT_FLOOR}s)\n")
                return log, 0

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        results = list(ex.map(one, enumerate(cmds)))
    texts = [log.read_text(encoding="utf-8", errors="replace") for log, _ in results]
    return {"texts": texts, "seeds": seeds, "orders": orders, "jobs": job_ix,
            "bad": sum(1 for _, rc in results if rc)}


def _label_for(meta_name, opp_names, slug):
    """Every seat index for our meta name maps to the slug — one entry per
    index, because the seats rotate. The old map set `Ai(1)` only, which under
    rotation would have scored a rotated arm zero (the bug `forge.py`'s own
    label map documents)."""
    label = _seat_label([meta_name, *opp_names])
    for k in range(1, len(opp_names) + 2):
        label[f"Ai({k})-{meta_name}"] = slug
    return label


def _score_arm(texts, meta_name, opp_names, slug, arm, opponents):
    """Texts -> (facts, analysis, games_detail) for one arm, scored against
    ITS OWN list. A seam, so a test can drive the sequential loop without
    Forge; the real thing is `parse.analyze_logs` under a rotation-safe map."""
    label = _label_for(meta_name, opp_names, slug)
    opp_cmd = _commanders_by_slug(opponents)
    n = len(opp_names) + 1
    cmd = {}
    mine = commanders_from_text(arm["decklist_text"])
    for k in range(1, n + 1):
        if mine:
            cmd[f"Ai({k})-{meta_name}"] = mine
        for i, o in enumerate(opponents):
            if opp_cmd.get(o):
                cmd[f"Ai({k})-{opp_names[i]}"] = opp_cmd[o]
    facts, analysis = sim_parse.analyze_logs(texts, label, cmd)
    return facts, analysis, [sim_parse.compact(f, label) for f in facts]


def _look(analysis_a, analysis_b, slug, z, until_mde=None):
    """One look at the primary endpoint: wins over DECIDED games per arm, the
    Newcombe interval at this look's boundary z, and the two decisions it can
    carry — efficacy (excludes zero at the boundary) and non-binding futility
    (the boundary interval already excludes the effect `until_mde` asked for,
    on the favourable side)."""
    sa = (analysis_a.get("seats") or {}).get(slug, {})
    sb = (analysis_b.get("seats") or {}).get(slug, {})
    n_a = analysis_a.get("decided", analysis_a.get("games") or 0) or 0
    n_b = analysis_b.get("decided", analysis_b.get("games") or 0) or 0
    k_a, k_b = sa.get("wins") or 0, sb.get("wins") or 0
    d = stats.diff_proportions(k_a, n_a, k_b, n_b, z=z) if n_a and n_b else None
    futile = bool(until_mde is not None and d and d["ci95"][1] < until_mde)
    return {"n_a": analysis_a.get("games"), "n_b": analysis_b.get("games"),
            "decided_a": n_a, "decided_b": n_b, "k_a": k_a, "k_b": k_b,
            "z_boundary": z,
            "ci95_diff_at_boundary": d["ci95"] if d else None,
            "excludes_zero": bool(d and d["excludes_zero"]),
            "futility_excluded": futile}


def run(slug, ref_a, ref_b, opponents, games=SIM_DEFAULT_GAMES, jobs=None,
        clock=SIM_GAME_CLOCK_SECONDS, seed=None, profile=None, dry_run=False,
        vs_profile=None, detect=None, anyway=False, looks=1, until_mde=None,
        aa=False, resume=False, boundary="obf", profile_b=None, pod_name=None):
    if not opponents:
        raise SystemExit("experiment needs at least one opponent: --vs <slug> (repeatable)")
    import os
    a, b = resolve_arm(slug, ref_a), resolve_arm(slug, ref_b)
    same_list = a["decklist_sha256"] == b["decklist_sha256"]
    # AN A/A IS SAID OUT LOUD OR REFUSED. The same list twice is the harness's
    # noise floor — legitimate, and the plan's standing check per harness
    # fingerprint — but only when the pilot says `--aa`, because the other way
    # it is a typo that spends ten hours measuring nothing. `--profile-b`
    # (our seat on another AI profile on arm B only) is the other honest way
    # two arms share a list: policy-on against policy-off, the A/B every
    # piloting change needs and nothing could express.
    if same_list and not (aa or profile_b):
        raise SystemExit(f"both arms are the same list ({a['decklist_sha256'][:12]}…) — "
                         f"an A/A tells you the noise floor, which is legitimate, but "
                         f"say so with --aa (two seeds, one list); or pass --profile-b "
                         f"to fly arm B on another AI profile")
    if aa and not same_list:
        raise SystemExit("--aa is one list twice; the two refs resolve to different lists")
    if profile_b and profile_b == (profile or DEFAULT_PROFILE):
        raise SystemExit(f"--profile-b {profile_b} is arm A's profile too; the arms "
                         f"would differ in nothing")
    try:
        looks = int(looks or 1)
        bounds = stats.look_boundaries(looks, boundary)
    except ValueError as exc:
        raise SystemExit(f"--looks: {exc}")
    # `forge.default_jobs()`, NOT `cpu_count - 1`. `simulate` was moved to
    # PERFORMANCE CORES on measurement and this was left behind. Re-derived
    # across every tracked run on 2026-09-04: 4 jobs truncate 3.4% of games
    # (14 of 408) and 7 jobs truncate 13.6% (120 of 880) — four-fold. A
    # truncated game has no winner and is EXCLUDED from the rate, so
    # oversubscribing does not merely run slower — it throws games away, and
    # an experiment throws them away from both arms.
    #
    # Measured here: the first real A/B ran at 7 jobs on a 4-performance-core
    # machine and took 7,295 seconds for 80 games.
    jobs = jobs or default_jobs()
    seed = seed if seed is not None else int(hashlib.sha256(
        (a["decklist_sha256"] + b["decklist_sha256"]).encode()).hexdigest()[:8], 16) % 2_000_000_000

    # THE POD RUNS THE SAME PILOT `simulate` GIVES IT. This built
    # `[profile] + ["Default"] * len(opponents)` and never read
    # `STANDARD_POD_PROFILE`, so the two commands measured DIFFERENT
    # POPULATIONS: `simulate` has run the pod on Experimental since 2026-08-30,
    # where it moved baylen-tokens 0.130 -> 0.190, and every experiment kept
    # running it on Default. A controlled A/B whose table is not the table the
    # deck is measured against is controlled against the wrong thing.
    #
    # The two arms still share one pod, which is what makes the delta a delta;
    # this only fixes WHICH pod. `--vs-profile Default` reproduces the old
    # population deliberately, and the id says which was used either way.
    profile = profile or DEFAULT_PROFILE
    vs_profile = vs_profile or STANDARD_POD_PROFILE
    profiles = [profile] + [vs_profile] * len(opponents)
    profiles_b = ([profile_b] + [vs_profile] * len(opponents)) if profile_b else profiles
    try:
        ov = card_overrides()          # the engine's fingerprint, or None
    except Exception:                  # noqa: BLE001 — a checkout with no Forge
        ov = None
    # THE LOG FORMATTER, the same way: the patched jar when it is installed and
    # registered, refused before launch when it is neither, and in the id either way.
    try:
        jar, tl = telemetry_for_run()
    except _telemetry.EngineMismatch as exc:
        raise SystemExit(f"{slug}: {exc}") from exc
    except Exception:                  # noqa: BLE001 — a checkout with no Forge
        jar, tl = None, None
    eid = experiment_id(slug, a, b, opponents, games, seed, profile, vs_profile,
                        clock=clock, overrides_sha=(ov or {}).get("sha"),
                        looks=looks, aa=aa, profile_b=profile_b,
                        telemetry_sha=(tl or {}).get("sha"))
    out_dir = deck_dir(slug) / EXP_DIR
    path = out_dir / f"{eid}.json"
    # THE LOOKS DIVIDE THE GAMES INTO WHOLE JOBS, or the design is refused. A
    # look is a set of complete, rotated jobs from both arms — never a partial
    # job, because a job's early games differ systematically from its late
    # ones (ca8e802e) and a look cut inside one would be the biased slice.
    games = int(games)
    if games % looks:
        raise SystemExit(f"--looks {looks} does not divide {games} games into equal waves")
    wave = games // looks
    # A wave smaller than the job count simply uses fewer JVMs — `split_games`
    # never makes an empty job — so a look is still whole jobs, just fewer.
    jobs = max(1, min(jobs, wave))
    schedule = [wave * (k + 1) for k in range(looks)]
    existing = load_json(path) if path.exists() else None
    if existing and not dry_run and not resume:
        raise SystemExit(f"{slug}: {path.name} exists — the same arms, table and seed replay "
                         f"the same games. A new sample is a new --seed"
                         + ("; an unfinished sequential run continues with --resume"
                            if existing.get("status") == "running" else "."))
    if resume:
        if not existing:
            raise SystemExit(f"{slug}: --resume names no record at {path.name}")
        if not (existing.get("design") or {}).get("looks"):
            raise SystemExit(f"{slug}: {path.name} was not pre-registered as sequential — "
                             f"a look added after the fact is optional stopping. Run it "
                             f"again with --looks K and a new --seed.")
        if existing["design"]["looks"] != looks or existing["design"].get("schedule") != schedule:
            raise SystemExit(f"{slug}: {path.name} was registered with --looks "
                             f"{existing['design']['looks']} over {existing['design'].get('schedule')}; "
                             f"the design cannot change mid-run")
        if existing.get("status") != "running":
            raise SystemExit(f"{slug}: {path.name} is {existing.get('status')} — nothing to resume")
    # THE ARITHMETIC, AT THE ONE MOMENT IT CAN STILL CHANGE THE DECISION.
    # `stats` has carried the power functions since the statistics went in and
    # nothing called them before a run — so a 100-per-arm A/B was launched
    # against a 0.244 baseline with a 0.34 chance of seeing a real ten-point
    # improvement. Four hours to be more likely to miss than to find.
    #
    # It PRINTS and does not refuse. An A/B that cannot resolve the effect the
    # pilot cares about is still a legitimate thing to run — as a noise floor,
    # as a smoke test, as the first half of a bigger sample — and a gate that
    # blocked it would be a validator firing on correct use.
    from manamap.sim import power as _power
    p_a, from_run = _power.baseline_rate(slug, opponents)
    for line in _power.preflight(p_a, int(games), detect=detect):
        print(line)
    if from_run:
        print(f"    baseline from {from_run}")
    print()
    # ...and it REFUSES only what the pilot asked it to see and it cannot:
    # `--detect X` without `--anyway` on a run whose power for X is under 0.8.
    if not dry_run:
        _power.refuse_if_underpowered(p_a, int(games), detect, anyway)

    if dry_run:
        return path, {"experiment_id": eid, "arms": {"a": a["label"], "b": b["label"]},
                      "seed": seed, "games_per_arm": games, "profiles": profiles,
                      "profiles_b": profiles_b,
                      "design": {"looks": looks, "boundary": boundary, "critical": bounds,
                                 "schedule": schedule, "until_mde": until_mde,
                                 "jobs_per_wave": jobs}}

    jar = jar or forge_jar()
    names_a = install_named(f"mm-x-{slug}-a", a["decklist_text"])
    names_b = install_named(f"mm-x-{slug}-b", b["decklist_text"])
    opp_names = [install_deck(o) for o in opponents]
    # THE SHA OF WHAT WAS PLAYED, SNAPSHOT AT LAUNCH — the same rule `forge.run`
    # keeps, and the same defect one door along. `install_deck` has just written
    # each opponent's .dck and the JVMs read those and nothing else; the record
    # is built hours later and called `seat_sha` there, so an opponent seat
    # edited mid-run would be recorded as a table nobody sat at.
    #
    # Latent rather than observed, unlike the run-record case: nobody has edited
    # an opponent during an experiment. It is fixed anyway because it is the
    # identical bug, and "no one has done it yet" is not a property of the code.
    opp_shas = {o: seat_sha(o) for o in opponents}
    log_dir = out_dir / "logs" / eid
    t0 = time.time()
    # THE WAVES. Each look is a whole set of rotated jobs from both arms, the
    # arms interleaved per wave so a kill leaves both at the same look; the
    # record is rewritten after every wave, which is the crash-resume unit —
    # a resumed run re-reads the logs of its completed waves and continues the
    # job index, so the seeds and rotations are the ones the design planned.
    # An A/A takes a second seed base for arm B; its games are then genuinely
    # different shuffles of one list, which is what a noise floor is.
    seed_b = seed + 100_000 if aa else seed
    prior = existing if resume else None
    done = len((prior or {}).get("looks") or [])
    texts = {"a": [], "b": []}
    seeds_all = {"a": [], "b": []}
    orders_all = {"a": [], "b": []}
    bad = {"a": 0, "b": 0}
    if done:
        for letter in ("a", "b"):
            for j in range(done * jobs):
                lp = log_dir / f"{letter}-part-{j:02d}.log"
                if not lp.exists():
                    raise SystemExit(f"{slug}: cannot resume — {lp.name} is missing; the "
                                     f"logs of a completed wave are what a resume reads")
                texts[letter].append(lp.read_text(encoding="utf-8", errors="replace"))
        for lk in prior["looks"]:
            for letter in ("a", "b"):
                seeds_all[letter] += lk["seeds"][letter]
                orders_all[letter] += lk["seat_orders"][letter]
        wall_before = prior.get("wall_seconds") or 0
    else:
        wall_before = 0
    looks_out = list((prior or {}).get("looks") or [])
    status = "running"
    analysis_a = analysis_b = None
    games_a = games_b = None
    for k in range(done, looks):
        got = {}
        for letter, meta, sb_ in (("a", names_a, seed), ("b", names_b, seed_b)):
            got[letter] = _run_wave(letter, meta, opp_names, wave, jobs, clock, sb_,
                                    profile if letter == "a" else (profile_b or profile),
                                    vs_profile, log_dir, jar, first_job=k * jobs)
            texts[letter] += got[letter]["texts"]
            seeds_all[letter] += got[letter]["seeds"]
            orders_all[letter] += got[letter]["orders"]
            bad[letter] += got[letter]["bad"]
        facts_a, analysis_a, games_a = _score_arm(texts["a"], names_a, opp_names, slug, a, opponents)
        facts_b, analysis_b, games_b = _score_arm(texts["b"], names_b, opp_names, slug, b, opponents)
        lk = _look(analysis_a, analysis_b, slug, bounds[k], until_mde)
        lk.update({"k": k + 1, "of": looks, "at": date.today().isoformat(),
                   "jobs": {"a": got["a"]["jobs"], "b": got["b"]["jobs"]},
                   "seeds": {"a": got["a"]["seeds"], "b": got["b"]["seeds"]},
                   "seat_orders": {"a": got["a"]["orders"], "b": got["b"]["orders"]},
                   "planned_n": schedule[k]})
        if lk["excludes_zero"]:
            lk["decision"] = "stop: efficacy"; status = "stopped_efficacy"
        elif lk["futility_excluded"]:
            lk["decision"] = "stop: futility"; status = "stopped_futility"
        elif k + 1 == looks:
            lk["decision"] = "complete"; status = "complete"
        else:
            lk["decision"] = "continue"
        looks_out.append(lk)
        wall = round(wall_before + time.time() - t0, 1)
        doc = _document(slug, eid, a, b, opponents, opp_shas, games, seed, seeds_all,
                        profiles, profiles_b, clock, wall, ov, bad, analysis_a, analysis_b,
                        games_a, games_b, looks, boundary, bounds, schedule, until_mde,
                        jobs, aa, looks_out, status, pod_name)
        doc["telemetry"] = tl        # the formatter both arms flew under, beside card_overrides
        out_dir.mkdir(exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        if status != "running":
            break
    return path, doc


def _document(slug, eid, a, b, opponents, opp_shas, games, seed, seeds_all, profiles,
              profiles_b, clock, wall, ov, bad, analysis_a, analysis_b, games_a, games_b,
              looks, boundary, bounds, schedule, until_mde, jobs, aa, looks_out, status,
              pod_name):
    d = delta(analysis_a, analysis_b, slug, games_a, games_b)
    last = looks_out[-1]
    if status in ("stopped_efficacy", "stopped_futility"):
        ci = last["ci95_diff_at_boundary"]
        d["reading"] = (f"STOPPED AT LOOK {last['k']} OF {looks} ({status.split('_')[1]}): "
                        f"the interval on the win-rate difference at that look's boundary "
                        f"z = {last['z_boundary']} is [{ci[0]:+.3f}, {ci[1]:+.3f}]. "
                        + ("It excludes zero at the boundary — a real difference under the "
                           "design's alpha; the 1.96 interval beside it is descriptive and "
                           "narrower than the design allows a reader to take it."
                           if status == "stopped_efficacy" else
                           f"It already excludes the {until_mde:+.3f} the run was asked to "
                           f"find, on the favourable side — non-binding futility, so the "
                           f"design's alpha is untouched and the answer is 'not that big'."))
    return {
        "experiment_id": eid, "slug": slug, "at": date.today().isoformat(),
        "engine": {"forge": forge_version(), "java": _java_version()},
        "question": (f"{a['label']}  vs  {b['label']}, same table"
                     + (" (A/A: one list, two seeds)" if aa else "")
                     + (f" (arm B on {profiles_b[0]})" if profiles_b != profiles else "")),
        "opponents": [{"slug": o, "decklist_sha256": opp_shas[o]} for o in opponents],
        # THE POD BLOCK `simulate` WRITES, so `net_change.forge` can bucket an
        # experiment's arms with the run records at the same table and harness.
        "pod": _pods.record_for(pod_name, opponents),
        "status": status,
        "design": {"looks": looks, "boundary": boundary, "alpha": 0.05,
                   "critical": bounds, "schedule": schedule, "until_mde": until_mde,
                   "jobs_per_wave": jobs, "primary": PRIMARY_ENDPOINT, "aa": aa,
                   "rule": ("each look tests the primary at its own boundary z on the "
                            "cumulative decided games of both arms; a look is whole "
                            "rotated jobs from both arms, never a partial job")},
        "looks": looks_out,
        "games_per_arm": int(games), "seed_base": seed, "seeds": seeds_all["a"],
        "seeds_b": seeds_all["b"],
        "profiles": profiles, **({"profiles_b": profiles_b} if profiles_b != profiles else {}),
        "clock_seconds": clock, "wall_seconds": wall,
        # THE HARNESS BOTH ARMS FLEW UNDER. An experiment is the one place in this repo
        # where two lists are compared under a CONTROLLED instrument, and it recorded
        # every part of that instrument — opponents, N, profiles, clock, engine build,
        # seeds — except the card scripts. So an A/B could be run half-overridden with
        # nothing on disk saying so, on the exact command whose whole purpose is that
        # the two arms differ in the list and nothing else.
        #
        # Both arms run in one invocation against one engine, so there is one value and
        # it belongs beside `profiles` rather than inside each arm.
        "card_overrides": ov,
        "nonzero_exit_jobs": bad["a"] + bad["b"],
        "arms": {
            "a": {"ref": a["ref"], "label": a["label"], "decklist_sha256": a["decklist_sha256"],
                  "decklist_text": a["decklist_text"], "games": analysis_a.get("games"),
                  "analysis": analysis_a, "games_detail": games_a},
            "b": {"ref": b["ref"], "label": b["label"], "decklist_sha256": b["decklist_sha256"],
                  "decklist_text": b["decklist_text"], "games": analysis_b.get("games"),
                  "analysis": analysis_b, "games_detail": games_b},
        },
        "delta": d,
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


def validate(doc):
    """Form-check a sequential record: the looks it holds against the design it
    registered. A record with no `design` predates looks and passes. Returns a
    list of errors."""
    errors = []
    dz = doc.get("design")
    if not dz:
        return errors
    looks = dz.get("looks")
    sched = dz.get("schedule") or []
    if looks not in stats.OBF_BOUNDARIES:
        errors.append(f"design.looks {looks!r} is not a tabulated design")
    if len(sched) != (looks or 0):
        errors.append(f"design.schedule has {len(sched)} entries for {looks} looks")
    got = doc.get("looks") or []
    status = doc.get("status")
    if status not in STATUSES:
        errors.append(f"status {status!r} is not one of {STATUSES}")
    if len(got) > (looks or 0):
        errors.append(f"{len(got)} looks recorded for a {looks}-look design")
    if status == "complete" and len(got) != looks:
        errors.append(f"status complete with {len(got)} of {looks} looks")
    if status in ("stopped_efficacy", "stopped_futility") and got:
        want = "stop: " + status.split("_")[1]
        if got[-1].get("decision") != want:
            errors.append(f"status {status} but the last look decided {got[-1].get('decision')!r}")
    try:
        bounds = stats.look_boundaries(looks, dz.get("boundary") or "obf")
    except ValueError:
        bounds = []
    for i, lk in enumerate(got):
        if lk.get("k") != i + 1:
            errors.append(f"looks[{i}] is numbered {lk.get('k')}")
        if bounds and lk.get("z_boundary") != bounds[i]:
            errors.append(f"looks[{i}] used z {lk.get('z_boundary')}; the design's is {bounds[i]}")
        if sched and lk.get("n_a") is not None and lk["n_a"] != sched[i]:
            errors.append(f"looks[{i}] has {lk['n_a']} games on arm A; the schedule says {sched[i]}")
    return errors


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


def _print_result(slug, doc, path):
    """The finished experiment, as the pilot reads it.

    A FUNCTION BECAUSE IT HAD TO BECOME TESTABLE. This was inline in `main` and
    raised `KeyError: 'ci95_a'` on every real run — after the artifact was
    written, so the measurement survived and only the exit code said anything.
    It lived because the tests exercised `--dry-run` and `--analyze` and the
    branch that actually runs had no seam at all.

    `ci95_diff`, NOT a marginal interval per arm. `delta()` has never emitted one
    and must not: two marginal intervals overlapping implies nothing, which is
    why `intervals_overlap` was DELETED rather than deprecated. The printer was
    asking for the keys the artifact exists to make impossible.
    """
    d = doc["delta"]
    print(f"{slug}: experiment {doc['experiment_id'][:64]}")
    print(f"  A {doc['arms']['a']['label']}")
    print(f"  B {doc['arms']['b']['label']}")
    print(f"  {doc['games_per_arm']} games/arm vs "
          f"{', '.join(o['slug'] for o in doc['opponents'])} "
          f"in {doc['wall_seconds']}s")
    for k, _ in DELTA_KEYS:
        r = d[k]
        if k == PRIMARY_ENDPOINT or r["a"] is not None or r["b"] is not None:
            label = "win rate" if k == PRIMARY_ENDPOINT else k
            print(f"  {label:<40} A {_num(r['a'])}   B {_num(r['b'])}   "
                  f"Δ {_num(r['diff'])}   95% {_band(r)}")
    print(f"  → {d['reading']}")
    print(f"  → {path.relative_to(deck_dir(slug))}")


def _band(row):
    """A figure's 95% interval ON THE DIFFERENCE, or an em dash.

    An absent interval must never read as a narrow one — `delta()` says so where
    it writes `"no per-game values available; difference is unbounded"`, and a
    printer that rendered `[0.000, 0.000]` there would undo it.
    """
    ci = row.get("ci95_diff")
    return ("[%+.3f, %+.3f]" % (ci[0], ci[1])) if ci else "—"


def _num(value):
    """A figure, or an em dash. `0` is a measurement and prints as one."""
    return "—" if value is None else value


def main(args):
    slug = args.slug
    if getattr(args, "analyze", None):
        path, doc = analyze(slug, args.analyze)
        d = doc["delta"]
        print(f"{slug}: re-derived both arms from logs → {path.name}")
        for k, _ in DELTA_KEYS:
            v = d[k]
            print(f"  {k:<40} A {_num(v['a'])}   B {_num(v['b'])}   "
                  f"Δ {_num(v['diff'])}   95% {_band(v)}")
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
            dz = d.get("design") or {}
            seq = (f"  look {len(d.get('looks') or [])}/{dz['looks']} {d.get('status')}"
                   if dz.get("looks", 1) > 1 else "")
            print(f"{d['experiment_id'][:64]}  {d['at']}  n={d['games_per_arm']}/arm{seq}")
            print(f"      {d['question']}")
            print(f"      win {w['a']} → {w['b']}   Δ95% "
                  f"{('[%+.3f, %+.3f]' % (ci[0], ci[1])) if ci else '—'}  ({verdict})")
        return
    # THE SAME RESOLVER `simulate` USES. These two commands must not disagree
    # about what a table is; they already disagreed about its AI profile, and
    # that made every controlled A/B controlled against the wrong pod.
    opponents, seat_profiles = forge_resolve_table(args)
    path, doc = run(slug, args.a, args.b, opponents, games=args.games or SIM_DEFAULT_GAMES,
                    jobs=args.jobs, clock=args.clock or SIM_GAME_CLOCK_SECONDS,
                    seed=getattr(args, "seed", None), profile=getattr(args, "profile", None),
                    vs_profile=getattr(args, "vs_profile", None) or seat_profiles,
                    dry_run=getattr(args, "dry_run", False),
                    detect=getattr(args, "detect", None),
                    anyway=getattr(args, "anyway", False),
                    looks=getattr(args, "looks", 1) or 1,
                    until_mde=getattr(args, "until_mde", None),
                    aa=getattr(args, "aa", False),
                    resume=getattr(args, "resume", False),
                    boundary=getattr(args, "boundary", None) or "obf",
                    profile_b=getattr(args, "profile_b", None),
                    pod_name=getattr(args, "pod", None))
    if getattr(args, "dry_run", False):
        dz = doc.get("design") or {}
        print(f"would run {doc['games_per_arm']} games/arm, seed {doc['seed']} → {path.name}")
        if dz.get("looks", 1) > 1:
            print(f"  {dz['looks']} looks ({dz['boundary']}) at {dz['schedule']} games/arm, "
                  f"boundary z {dz['critical']}"
                  + (f", futility below {dz['until_mde']:+.3f}" if dz.get("until_mde") is not None else ""))
        return
    _print_result(slug, doc, path)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot experiment <slug> --a <ref> --b <ref> --vs <opp> …`.")
