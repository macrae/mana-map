"""THE SIM DEBRIEF'S SKELETON: what a run says about how the deck was flown,
as findings with ids — computed, never authored.

THE CAPTAIN'S LOG STAYS HUMAN. `debrief`'s rule — "you may name nothing the
pilot and the deck did not" — is what makes the doctor trust a log entry, and a
Forge run has no pilot; so simulated games never enter `log.jsonl`. They get
their own record, `sim_findings.json`, under the same discipline the captain's
log uses: this module computes the deterministic skeleton, an agent writes
PROSE that may cite only finding ids, `merge-sim-findings` recomputes the
skeleton unconditionally and takes the prose, and `validate-sim-findings`
holds every number in the prose to a cited finding.

WHAT A FINDING IS. One figure with its interval and its N, a `source` naming
where in the record it came from, a `basis` when it rests on an inference, and
a deterministic sentence. Kinds, per run: `rate_vs_null` (the seat's rate
against the table's null, both with intervals), `pilot_quality` (the land-drop
ratio and its verdict, or the withheld verdict), `loss_decomposition` (who
eliminated us and how), `held` (`engine_casts.never_cast`, measured against
modelled), `first_attack`, `wipe_recovery`, `board_shape` (from `sim-boards`,
only where the logs are), `targeting` (the tracked `threat/targeting.json`),
and `piloting_delta` (a checker-passed line lifted from this run whose
proposed play differed from the AI's — D2). A finding never says what to do.
"""

import glob
import json
import pathlib

from manamap.pilot.common import deck_dir, load_json

ARTIFACT = "sim_findings.json"
SETTLED_BY = ("resolve-stack", "experiment", "campaign", "poh-proposal", "diagnose", "unsettled")
PROSE_KEYS = ("reading", "so_what", "open_questions")


def _fid(run_id, n):
    """`F-<sha8 of the run id>-NN`. A slice of the run id was the first cut and
    two runs ending in the same harness tag (`…-podExperimental-c600`) shared
    every finding id; a hash of the whole id does not."""
    import hashlib
    return f"F-{hashlib.sha256(run_id.encode()).hexdigest()[:8]}-{n:02d}"


def _round(x, nd=3):
    return round(x, nd) if isinstance(x, (int, float)) else x


def run_findings(slug, rec, targeting=None, boards=None):
    """Every finding one run supports from its record alone (plus the tracked
    targeting artifact and, where given, finder shapes)."""
    from manamap.sim import engine_casts as ec_mod
    from manamap.sim import pilot_quality
    from manamap.sim import power
    run_id = rec["run_id"]
    ours = slug
    a = rec.get("analysis") or {}
    me = (a.get("seats") or {}).get(ours) or {}
    out = []
    n = 0

    def add(kind, text, **fields):
        nonlocal n
        n += 1
        f = {"id": _fid(run_id, n), "kind": kind, "text": text}
        f.update({k: v for k, v in fields.items() if v is not None})
        out.append(f)
        return f

    # the rate, against the table's null
    pod = (rec.get("pod") or {}).get("name") if isinstance(rec.get("pod"), dict) else rec.get("pod")
    null, n_null = power.null_rate(pod)
    wins = (rec.get("summary") or {}).get("wins", {}).get(ours)
    decided = (rec.get("summary") or {}).get("decided")
    if me.get("win_rate") is not None and decided:
        ci = me.get("win_rate_ci95")
        txt = (f"{wins} of {decided} decided games ({me['win_rate']:.3f}, ci95 {ci})"
               + (f" against a null of {null:.3f} at {pod} — {me['win_rate'] / null:.0%} of par"
                  if null else " — no measured null for this table"))
        add("rate_vs_null", txt, figure=me["win_rate"], ci95=ci, n=decided,
            null=({"rate": null, "games": n_null, "pod": pod} if null else None),
            excluded=(rec.get("card_overrides") or {}).get("sha") and "this run is overridden and is not in the null" or None,
            source=f"analysis.seats[{ours}].win_rate; pods.calibration({pod})")

    # was our seat played like the rest of the table
    pq = pilot_quality.from_record(rec)
    if pq:
        lands = pq.get(pilot_quality.LANDS) or {}
        add("pilot_quality", pq.get("reading") or "", figure=lands.get("ratio"),
            verdict=pq.get("comparable"),
            n=((pq.get("per_seat") or {}).get(pq.get("seat")) or {}).get("games"),
            source="sim/pilot_quality.from_record (land drops per own turn, ours / pod)",
            basis="a ratio; the verdict is withheld inside the fleet's within-deck spread")

    # how we lost
    eb, eh = me.get("eliminated_by"), me.get("eliminated_how")
    if eb:
        total = sum(eb.values()) if isinstance(eb, dict) else None
        top = sorted(eb.items(), key=lambda kv: -kv[1]) if isinstance(eb, dict) else []
        txt = ("eliminated in " + (f"{total} of {a.get('games')} games" if total is not None else "some games")
               + (": " + ", ".join(f"{k} ×{v}" for k, v in top[:3]) if top else "")
               + (f"; how: {json.dumps(eh)}" if eh else ""))
        add("loss_decomposition", txt, eliminated_by=eb, eliminated_how=eh,
            n=a.get("games"), source=f"analysis.seats[{ours}].eliminated_by / eliminated_how",
            basis="the controller of the last damage source before the life line that crossed zero; "
                  "a drain kill reads through `eliminated_how`")

    # what the AI held
    try:
        engine = ec_mod.engine_set(slug.split("@")[0], slug.split("@")[1] if "@" in slug else None)
    except Exception:                                # noqa: BLE001 — no declaration
        engine = None
    ecr = ec_mod.from_record(rec, engine=engine)
    if ecr:
        never = ecr.get("never_cast") or []
        basis = ecr.get("never_cast_basis") or {}
        txt = (f"{len(never)} card(s) never cast over {ecr['games']} games "
               f"({basis.get('measured', 0)} seen discarded, {basis.get('modelled', 0)} inferred)"
               + (": " + ", ".join(x["card"] for x in never[:6]) if never else ""))
        add("held", txt, cards=[x["card"] for x in never],
            measured=[x["card"] for x in never if x["basis"] == "measured"],
            modelled=[x["card"] for x in never if x["basis"] == "modelled"],
            # the per-card counts, so a debrief can say "discarded four times"
            # and the validator can find the four
            counts={x["card"]: {k: x[k] for k in ("cast", "activated", "discarded")} for x in never},
            n=ecr["games"], played_share=(ecr.get("engine") or {}).get("played_share"),
            source="engine_casts (top-level) via sim/engine_casts.from_record",
            basis="measured = the log shows it discarded; modelled = never played across the "
                  "run's expected natural draws, an inference")
        # MEASURED, not inferred: under the telemetry patch the hand is on the log, so
        # "held" is a count of own turns the card sat in hand while castable on lands
        # alone. A plain record has none and gets no finding rather than a zero.
        hc = ecr.get("held_while_castable")
        if hc:
            add("held_castable",
                f"{len(hc)} card(s) held on >= {ec_mod.HELD_CASTABLE_TURNS} castable own turns and cast at most once "
                f"(MEASURED from the telemetry hand; lands only, a floor): "
                + ", ".join(f"{x['card']} x{x['castable_uncast']}" for x in hc[:6]),
                cards=[x["card"] for x in hc],
                counts={x["card"]: {k: x[k] for k in ("castable_uncast", "cast", "in_hand_games")} for x in hc},
                basis="own turns ending with the card in hand and lands >= its mana value; rocks and colours ignored")

    fa = me.get("first_attack_turn") or {}
    if isinstance(fa, dict) and fa.get("mean") is not None:
        add("first_attack", f"first attack on global turn {fa['mean']} (median {fa.get('median')}, "
                            f"n {fa.get('n')} of {a.get('games')} games — games with no attack are absent)",
            figure=fa["mean"], ci95=fa.get("ci95"), n=fa.get("n"),
            source=f"analysis.seats[{ours}].first_attack_turn", basis="conditional on attacking at all")

    wr = a.get("wipe_recovery") or {}
    if wr.get("available"):
        add("wipe_recovery",
            f"a wipe in {wr.get('games_with_a_wipe_rate')} of games (ci95 {wr.get('games_with_a_wipe_ci95')}); "
            f"damage on the wipe turn {(wr.get('damage_on_wipe') or {}).get('mean')}",
            figure=wr.get("games_with_a_wipe_rate"), ci95=wr.get("games_with_a_wipe_ci95"),
            n=wr.get("games"), source="analysis.wipe_recovery",
            basis="a heuristic over the log: 3+ permanents leaving across 2+ seats in one turn")

    if targeting:
        pol = targeting.get("forge_ai_targeting_policy") or {}
        best = max(pol.items(), key=lambda kv: kv[1].get("rate") or 0) if pol else None
        if best:
            name, row = best
            add("targeting",
                f"the pod attacks the seat with the {name.replace('_', ' ')} {row['rate']:.0%} of the "
                f"time (ci95 {row['ci95']}, {row['decisions']} decisions, permutation p {row.get('permutation_p')})",
                figure=row["rate"], ci95=row["ci95"], n=row["decisions"], hypothesis=name,
                source="threat/targeting.json (pooled over the deck's runs with logs)",
                basis="Forge's AI, not your table — opponent modelling, never an equilibrium")

    for shape in (boards or []):
        add("board_shape",
            f"{shape['criterion']}: {json.dumps(shape['shape'])} in {shape['games']} of {shape['of']} games "
            f"(ci95 {shape['ci95']}); exemplar game {shape['exemplar']['game']} turn {shape['exemplar']['turn']}",
            figure=shape.get("share"), ci95=shape["ci95"], n=shape["of"],
            criterion=shape["criterion"], shape=shape["shape"], exemplar=shape["exemplar"],
            source=f"sim-boards {run_id} --criterion {shape['criterion']} (logs on this machine)",
            basis="the board series' estimate; recurrence counts games with at least one hit")

    # a proven line lifted from THIS run whose proposed play differed from the AI's
    for st in _lifted_stacks(slug, run_id):
        d = (st.get("scenario") or {}).get("extras", {}).get("piloting_delta") or {}
        if d and (st.get("checker") or {}).get("verdict") == "pass":
            add("piloting_delta",
                f"stack {st.get('id')}: {d.get('kind')} — {d.get('mechanism')}",
                stack=st.get("id"), delta=d.get("kind"), mechanism=d.get("mechanism"),
                recurrence=((st.get("scenario") or {}).get("extras", {}).get("finder") or {}).get("recurrence"),
                source=f"stacks/{st.get('id')} (checker: pass)")
    return out


def _lifted_stacks(slug, run_id):
    base = deck_dir(slug.split("@")[0])
    out = []
    for p in sorted((base / "stacks").glob("[0-9][0-9][0-9]-*.json")) if (base / "stacks").is_dir() else []:
        d = load_json(p) or {}
        if ((d.get("scenario") or {}).get("source") or {}).get("run_id") == run_id:
            out.append(d)
    return out


def _version_of(slug, sha):
    try:
        from manamap.pilot import deck_versions
        for v in deck_versions.versions(slug.split("@")[0]):
            if sha in (v.get("decklist_sha256s") or []):
                return v["version"]
    except Exception:                                # noqa: BLE001
        return None
    return None


def skeleton(slug, run_ids=None, boards_for=None):
    """The whole artifact minus prose. `boards_for` is {run_id: [shapes]} from
    the finder, supplied by the caller because it needs the logs."""
    from manamap.sim.forge import list_runs
    from manamap.pilot.common import decklist_sha256
    base = deck_dir(slug)
    try:
        cur = decklist_sha256(slug)
    except FileNotFoundError:
        cur = None
    targeting = load_json(base / "threat" / "targeting.json")
    runs = {}
    for rec in list_runs(slug):
        rid = rec["run_id"]
        if run_ids and rid not in run_ids:
            continue
        ran = next((s.get("decklist_sha256") for s in rec.get("seats", []) if s.get("slug") == slug), None)
        pod = rec.get("pod")
        runs[rid] = {
            "run": {"id": rid, "at": rec.get("at"),
                    "pod": pod.get("name") if isinstance(pod, dict) else pod,
                    "games": rec.get("games_completed"),
                    "decided": (rec.get("summary") or {}).get("decided"),
                    "harness": {"card_overrides": (rec.get("card_overrides") or {}).get("sha"),
                                "telemetry": (rec.get("telemetry") or {}).get("sha"),
                                "profile": ((rec.get("profiles") or ["Default"])[0]) or "Default",
                                "clock": rec.get("clock_seconds"),
                                "forge": (rec.get("engine") or {}).get("forge")},
                    "decklist_sha256": ran, "version": _version_of(slug, ran),
                    "current": bool(cur and ran == cur)},
            "findings": run_findings(slug, rec, targeting=targeting,
                                     boards=(boards_for or {}).get(rid)),
            "prose": {},
        }
    return {"slug": slug, "artifact": ARTIFACT, "runs": runs,
            "limits": ["every figure is the record's, with its interval and N; a finding never says what to do",
                       "`board_shape` and `targeting` need the logs and exist only where a run was made",
                       "the captain's log is never written from here: a Forge game has no pilot"]}


def read(slug):
    return load_json(deck_dir(slug) / ARTIFACT) or {}


def write(slug, doc):
    p = deck_dir(slug) / ARTIFACT
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def with_prose(doc, previous):
    """Carry every run's prose forward from the tracked file where the run is
    still in the skeleton; a run that vanished takes its prose with it."""
    for rid, run in doc["runs"].items():
        old = ((previous.get("runs") or {}).get(rid) or {}).get("prose") or {}
        run["prose"] = {k: old[k] for k in PROSE_KEYS if k in old}
    return doc


def main(args):
    slug = args.slug
    run_ids = [args.run] if getattr(args, "run", None) else None
    boards_for = {}
    if getattr(args, "boards", False):
        from manamap.sim import boards as _boards
        from manamap.sim.forge import list_runs, _out_dir
        for rec in list_runs(slug):
            rid = rec["run_id"]
            if run_ids and rid not in run_ids:
                continue
            if not (_out_dir(slug) / "logs" / rid).is_dir():
                continue
            shapes = []
            for crit in ("held", "death", "widest"):
                found = _boards.find(slug, rid, crit, {"n": 4})
                for row in found["shapes"][:3]:
                    shapes.append(dict(row, criterion=crit))
            boards_for[rid] = shapes
    doc = with_prose(skeleton(slug, run_ids, boards_for), read(slug))
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(f"SIM FINDINGS — {slug} ({len(doc['runs'])} run(s))")
        for rid, run in doc["runs"].items():
            r = run["run"]
            print(f"\n  {rid[:64]}  {r['at']}  {r['games']} games at {r['pod']}"
                  + (f"  V{r['version']}" if r.get("version") is not None else "")
                  + ("" if r.get("current") else "  (NOT the current list)"))
            for f in run["findings"]:
                print(f"    {f['id']}  {f['kind']:18} {f['text'][:110]}")
            if run["prose"]:
                print(f"    prose: {(run['prose'].get('reading') or '')[:100]}")
    if getattr(args, "write", False):
        p = write(slug, doc)
        print(f"\n  wrote {p}")
