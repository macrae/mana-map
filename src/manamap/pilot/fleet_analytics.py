"""Pilot: the fleet compared — one tracked artifact of PROFILES, never a score.

`data/fleet/analytics.json` puts every live deck side by side: its bracket (the
pilot's declared target, floor as the fallback), the uniform-harness benchmark
figures with Wilson intervals, the diagnostic vitals, DeckCheck's Threat Index
with its history, the Forge table record, the pilot's own record by version,
the survey means, the price — each figure with its interval where one exists,
its N, its source file and the list sha it describes. Then the bands each
bracket group spans (min / median / max over its members), where each member
sits in them, and the comparisons a page draws: floor against DTI, the DTI
readings over time, the twelve DeckCheck grades as a grid, the kill / mana /
land curves from the benchmark's own run.

NO COMPOSITE NUMBER, AND THE VALIDATOR REFUSES ONE. `benchmark.py` says why
(`score: None` — "a formula picked before the distribution is a guess wearing a
number"), `diagnostic.py` says why again (speed spans 400x and ranks a combo deck
last for not attacking), and `calibrate.py` refuses a verdict below ten decks
with twenty games each. The rating that DOES exist (`rating.py`) is ★ coaching
over these ◆ inputs, lives per deck, and is labelled as such; this file carries
a `rating` slot per row and leaves it null.

ABSENT MEANS ABSENT. Every number here is a `fig()` or an `absent()`; a deck
with no benchmark has no benchmark figures, not zeros. A group with fewer than
`FLEET_MIN_GROUP` members has no bands, and its members' placements say so.

DETERMINISTIC. Sorted slugs, sorted keys, four decimals, no clock: `as_of` is
copied from the source file that carries one. Built twice, byte-identical, which
is what lets `regen` run it as the last stage and CI diff the result.

The survey block is AUTHORED (`tier: "authored"`): the pilot's own numbers, shown
with n / min / max beside the mean, never a band member — "a measure computed
from an authored file is not evidence".
"""

import json
import statistics

from manamap import config
from manamap.pilot import benchmark as benchmark_mod
from manamap.pilot import calibrate, dti
from manamap.pilot.common import deck_dir, deck_is_apart, load_json, measured_sha
from manamap.pilot.deck_notes import causes, feel_summary, read_log
from manamap.pilot.formats import for_deck
from manamap.sim.stats import wilson_bounds

ARTIFACT = config.FLEET_ANALYTICS_PATH
SCHEMA_VERSION = 1
MIN_GROUP = config.FLEET_MIN_GROUP
ROUND = 4

#: The measures a group is banded on: (name, where it lives on the row, higher
#: is better). Benchmark figures first — the one harness every Commander deck
#: shares — then the diagnostic vitals and the DTI. Nothing authored is here.
BAND_MEASURES = (
    ("kill_by_turn_6", ("benchmark", "kill_by_turn_6"), True),
    ("kill_by_turn_8", ("benchmark", "kill_by_turn_8"), True),
    ("never_kills_rate", ("benchmark", "never_kills_rate"), False),
    ("missed_land_drop_rate", ("benchmark", "missed_land_drop_rate"), False),
    ("mulligan_rate", ("benchmark", "mulligan_rate"), False),
    ("mana_at_turn_five", ("benchmark", "mana_at_turn_five"), True),
    ("answer_cards", ("benchmark", "answer_cards"), True),
    ("stall_two_in_a_row", ("diagnostic", "stall_two_in_a_row"), False),
    ("missed_land_drop_by_five", ("diagnostic", "missed_land_drop_by_five"), False),
    ("dti_score", ("dti", "score"), True),
)

COMPOSITE_KEYS = frozenset({"score", "rating", "index", "power", "grade", "rank"})

NO_COMPOSITE = ("This file publishes profiles, bands and placements; a composite score "
                "is refused at the ◆ tier (benchmark.score is None; calibrate.MIN_DECKS "
                f"is {calibrate.MIN_DECKS}). The rating is ★ coaching, per deck, in rating.json.")

LIMITS = [
    NO_COMPOSITE,
    "Benchmark figures are an unopposed ceiling under one frozen harness (no pod, "
    "no blockers, no removal); goldfish kill is one column, never the rule.",
    "Per-turn curves come from the benchmark's own run, never from a deck's "
    "goldfish_metrics.json: the deck's own file is measured under its own opt-in "
    "channels and two of them are not comparable.",
    "DeckCheck's Threat Index is an LLM's read of the list that moves between scans; "
    "it is dated external evidence, shown beside the floor, and it never gates.",
    "The Forge table record describes the AI's piloting of the list it played, not the "
    "deck's strength (the pilot's rule, 2026-10-18: not a power read until the "
    "harness work is done); `stale` marks a run on an older list.",
    "The survey is the pilot's own 1-5 scores, authored at log time; it is never a "
    "band member and a mean is shown only with its n, min and max.",
    f"A bracket group needs {MIN_GROUP} members for a band; below that the band is "
    "absent and the placement says so.",
    "The fleet is the SLEEVED decks: a bracket group bands and ranks sleeved Commander decks "
    "only; a brewing or bench deck keeps its row (`member: false`) and joins no group.",
]


# ── Figures ──────────────────────────────────────────────────────────────


def fig(value, *, source, sha=None, stale=None, n=None, ci95=None, as_of=None, basis=None):
    """One number, with everything a reader needs to weigh it. `value` is
    rounded to `ROUND` places when it is a float; the rest ride beside it only
    when known."""
    out = {"value": _r(value), "source": source}
    if sha is not None:
        out["decklist_sha256"] = sha
    if stale is not None:
        out["stale"] = bool(stale)
    if n is not None:
        out["n"] = n
    if ci95 is not None:
        out["ci95"] = [_r(ci95[0]), _r(ci95[1])]
    if as_of is not None:
        out["as_of"] = as_of
    if basis is not None:
        out["basis"] = basis
    return out


def absent(reason, *, source):
    return {"absent_because": reason, "source": source}


def is_fig(x):
    return isinstance(x, dict) and "value" in x and "source" in x


def is_absent(x):
    return isinstance(x, dict) and "absent_because" in x


def _r(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        return round(x, ROUND)
    return x


def _rate_fig(rate, n, *, source, sha, stale, basis=None):
    """A proportion with a Wilson interval from `k = rate·n`."""
    if rate is None or n is None:
        return absent("not in the source file", source=source)
    k = round(float(rate) * n)
    lo, hi = wilson_bounds(k, n)
    return fig(float(rate), source=source, sha=sha, stale=stale, n=n, ci95=(lo, hi),
               basis=basis or f"Wilson 95% from k = rate·n at n = {n}")


# ── Per-deck rows ────────────────────────────────────────────────────────


def live_slugs():
    """Every deck directory with a `cards.json`, sorted. Apart decks stay in
    (flagged), so the fleet shows what was built; the workbench sinks them."""
    if not config.DECKS_DIR.is_dir():
        return []
    return sorted(d.name for d in config.DECKS_DIR.iterdir()
                  if d.is_dir() and (d / "cards.json").exists())


def _group(bracket):
    target, floor = bracket.get("target"), bracket.get("floor")
    if target is not None:
        return f"B{target}", "target"
    if floor is not None:
        return f"B{floor}", "floor"
    return "unbracketed", None


def _benchmark_block(doc, sha, src="benchmark.json"):
    if not doc:
        return absent("no benchmark.json — `manamap pilot benchmark <slug> --write`", source=src)
    m = doc.get("metrics") or {}
    h = doc.get("harness") or {}
    n = h.get("iterations")
    bsha = doc.get("decklist_sha256")
    stale = (bsha != sha) if (bsha and sha) else None
    ms, sp, co, re = (m.get("mana_screw") or {}), (m.get("speed") or {}), \
        (m.get("consistency") or {}), (m.get("response") or {})
    out = {
        "missed_land_drop_rate": _rate_fig(
            ms.get("missed_land_drop_rate"), n, source=f"{src}#metrics.mana_screw.missed_land_drop_rate",
            sha=bsha, stale=stale,
            basis=f"a per-turn rate over turns 2-6; Wilson from k = rate·n at n = {n} is an approximation"),
        "mulligan_rate": _rate_fig(ms.get("mulligan_rate"), n,
                                   source=f"{src}#metrics.mana_screw.mulligan_rate", sha=bsha, stale=stale),
        "mana_at_turn_five": (fig(ms["mana_at_turn_five"], source=f"{src}#metrics.mana_screw.mana_at_turn_five",
                                  sha=bsha, stale=stale, n=n, basis="a mean; no interval is stored")
                              if ms.get("mana_at_turn_five") is not None
                              else absent("not in the source file", source=src)),
        "kill_by_turn_6": _rate_fig(sp.get("kill_by_turn_6"), n, source=f"{src}#metrics.speed.kill_by_turn_6",
                                    sha=bsha, stale=stale),
        "kill_by_turn_8": _rate_fig(sp.get("kill_by_turn_8"), n, source=f"{src}#metrics.speed.kill_by_turn_8",
                                    sha=bsha, stale=stale),
        "kill_by_turn_10": _rate_fig(sp.get("kill_by_turn_10"), n, source=f"{src}#metrics.speed.kill_by_turn_10",
                                     sha=bsha, stale=stale),
        "never_kills_rate": _rate_fig(sp.get("never_kills_rate"), n, source=f"{src}#metrics.speed.never_kills_rate",
                                      sha=bsha, stale=stale),
        "median_kill_turn": (fig(sp["median_kill_turn"], source=f"{src}#metrics.speed.median_kill_turn",
                                 sha=bsha, stale=stale, n=n)
                             if sp.get("median_kill_turn") is not None
                             else absent("not in the source file", source=src)),
        "mana_stdev_turn_five": (fig(co["mana_stdev_turn_five"], source=f"{src}#metrics.consistency.mana_stdev_turn_five",
                                     sha=bsha, stale=stale, n=n)
                                 if co.get("mana_stdev_turn_five") is not None
                                 else absent("not in the source file", source=src)),
        "answer_cards": (fig(re["answer_cards"], source=f"{src}#metrics.response.answer_cards", sha=bsha,
                             stale=stale, basis="CAPACITY counted from card_roles.json, not a measured response")
                         if re.get("answer_cards") is not None
                         else absent("not in the source file", source=src)),
        "classes_covered": (fig(re["classes_covered"], source=f"{src}#metrics.response.classes_covered",
                                sha=bsha, stale=stale,
                                basis=f"of {re.get('classes_possible')} answer classes; CAPACITY")
                            if re.get("classes_covered") is not None
                            else absent("not in the source file", source=src)),
        "harness_version": h.get("version"),
    }
    return out


def _diagnostic_block(doc, sha, src="diagnostic.json"):
    if not doc:
        return absent("no diagnostic.json — bootstrapped on sleeved decks by `regen`", source=src)
    dsha = doc.get("decklist_sha256")
    stale = (dsha != sha) if (dsha and sha) else None

    def cell(block, key):
        c = (doc.get(block) or {}).get(key)
        if not isinstance(c, dict) or c.get("rate") is None:
            return absent("not in the source file", source=f"{src}#{block}.{key}")
        return fig(c["rate"], source=f"{src}#{block}.{key}", sha=dsha, stale=stale,
                   n=c.get("n"), ci95=c.get("ci95"))

    return {"stall_two_in_a_row": cell("stall", "two_in_a_row"),
            "missed_land_drop_by_five": cell("mana", "missed_land_drop_by_five"),
            "mulliganed": cell("mana", "mulliganed"),
            "harness_version": (doc.get("harness") or {}).get("version")}


def _dti_block(doc, sha, bracket_report, src="deckcheck.json"):
    r = dti.latest(doc)
    if not r:
        return absent("no deckcheck.json — scan on DeckCheck, then `deckcheck <slug> --write`", source=src)
    stale = dti.stale_list(doc, sha)
    omni = (doc or {}).get("omni") or {}
    return {
        "score": fig(r.get("score"), source=f"{src}#readings[-1].score", sha=r.get("decklist_sha256"),
                     stale=stale, as_of=r.get("as_of"), n=len(doc.get("readings") or []),
                     basis=f"of {r.get('max_score', dti.MAX_SCORE)}; an LLM's read, dated"),
        "max_score": r.get("max_score"),
        "bracket": r.get("bracket"),
        "as_of": r.get("as_of"),
        "stale_list": stale,
        "readings_n": len(doc.get("readings") or []),
        "benchmarks": dict(r.get("benchmarks") or {}),
        "omni": {k: omni.get(k) for k in ("n", "median_score", "median_bracket", "benchmarks")},
        "floor_vs_dti": dti.compare_floor(r, bracket_report) if bracket_report else None,
    }


def _table_block(info, src="info.json#simulation"):
    sim = (info or {}).get("simulation") or {}
    if not sim or sim.get("win_rate") is None:
        return absent("no Forge run on file", source=src)
    return {"win_rate": fig(sim["win_rate"], source=src, stale=sim.get("stale"), n=sim.get("decided"),
                            ci95=sim.get("win_rate_ci95"),
                            basis="the AI's piloting at the standard table; decided games; not a power read"),
            "games": sim.get("games"), "ran_on_version": sim.get("ran_on_version"),
            "latest": sim.get("latest")}


def _record_block(slug, log, versions_doc):
    cause_of = causes(slug)
    rec = {"games": len(log),
           "win": sum(1 for e in log if e.get("result") == "win"),
           "loss": sum(1 for e in log if e.get("result") == "loss"),
           "draw": sum(1 for e in log if e.get("result") == "draw"),
           "last_played": max((e["at"][:10] for e in log), default=None),
           "cause_counts": {}}
    cs = [cause_of[e["id"]]["cause"] for e in log if e["id"] in cause_of]
    rec["cause_counts"] = {c: cs.count(c) for c in sorted(set(cs))}
    by_version, matched = [], set()
    vers = (versions_doc or {}).get("versions") or []
    tags = (versions_doc or {}).get("tags") or {}
    release_of = {}
    for name, t in tags.items():
        v = t.get("version") if isinstance(t, dict) else None
        if v is not None and v not in release_of:
            release_of[v] = name
    for v in vers:
        shas = set(v.get("decklist_sha256s") or [])
        games = [e for e in log if e.get("decklist_sha256") in shas]
        if not games:
            continue
        matched.update(e["id"] for e in games)
        by_version.append({
            "version": v.get("version"), "release": release_of.get(v.get("version")),
            "date": v.get("first_date") or v.get("date"),
            "games": len(games),
            "win": sum(1 for e in games if e.get("result") == "win"),
            "loss": sum(1 for e in games if e.get("result") == "loss"),
            "draw": sum(1 for e in games if e.get("result") == "draw"),
            "feel": feel_summary(games)})
    rec["by_version"] = by_version
    rec["unmatched_games"] = sorted(e["id"] for e in log if e["id"] not in matched)
    return rec


def _survey_block(log):
    s = feel_summary(log)
    if not s:
        return absent("no --feel scores logged (`deck-notes add … --feel power=4`)",
                      source="log.jsonl#feel")
    return {"tier": "authored", "source": "log.jsonl#feel",
            "games_surveyed": sum(1 for e in log if e.get("feel")), **s}


def _price_block(doc, sha, src="prices.json"):
    if not doc or doc.get("total_cents") is None:
        return absent("no prices.json — `manamap pilot prices <slug> --write`", source=src)
    psha = doc.get("decklist_sha256")
    return {"total_cents": fig(doc["total_cents"], source=f"{src}#total_cents", sha=psha,
                               stale=(psha != sha) if (psha and sha) else None,
                               as_of=doc.get("as_of"), basis=f"{doc.get('source')}; dated"),
            "as_of": doc.get("as_of"), "source": doc.get("source"),
            "missing": len(doc.get("missing") or [])}


def _own_goldfish(gm, targets):
    if not gm:
        return absent("no goldfish_metrics.json", source="goldfish_metrics.json")
    meta, met = gm.get("meta") or {}, gm.get("metrics") or {}
    flags = {k: v for k, v in (targets or {}).items() if k.startswith("model_")}
    return {"comparable": False,
            "why_not": "measured under the deck's own opt-in channels; see benchmark for the uniform figures",
            "model_version": meta.get("model_version"),
            "declared_flags": flags,
            "commander_cast_by_turn_6_rate": (met.get("commander") or {}).get("cast_by_turn_6_rate")}


def row(slug):
    base = deck_dir(slug)
    sha = measured_sha(slug)
    info = load_json(base / "info.json") or {}
    fmt = for_deck(slug)
    commander = bool(fmt.commanders)
    bracket_report = load_json(base / "bracket_report.json") or {}
    bracket = {"floor": bracket_report.get("floor"), "floor_name": bracket_report.get("floor_name"),
               "target": bracket_report.get("target")}
    group, basis = _group(bracket)
    log = read_log(slug)
    dti_doc = dti.load(slug)
    out = {
        "slug": slug,
        "format": None if commander else fmt.name,
        "stage": info.get("stage"),
        "apart": deck_is_apart(slug),
        "lifecycle": (info.get("lifecycle") or {}).get("status") if info.get("lifecycle") else None,
        "decklist_sha256": sha,
        "bracket": {**bracket, "group": group, "group_basis": basis,
                    "dti_bracket": (dti.latest(dti_doc) or {}).get("bracket") if dti_doc else None},
    }
    if not commander:
        nm = f"not modelled for {fmt.name}"
        out["benchmark"] = absent(nm, source="benchmark.json")
        out["diagnostic"] = absent(nm, source="diagnostic.json")
        out["own_goldfish"] = absent(nm, source="goldfish_metrics.json")
    else:
        out["benchmark"] = _benchmark_block(load_json(base / "benchmark.json"), sha)
        out["diagnostic"] = _diagnostic_block(load_json(base / "diagnostic.json"), sha)
        out["own_goldfish"] = _own_goldfish(load_json(base / "goldfish_metrics.json"),
                                            load_json(base / "goldfish_targets.json"))
    out["dti"] = _dti_block(dti_doc, sha, bracket_report)
    out["table"] = _table_block(info)
    out["record"] = _record_block(slug, log, load_json(base / "versions.json"))
    out["survey"] = _survey_block(log)
    out["price"] = _price_block(load_json(base / "prices.json"), sha)
    out["placements"] = {}
    out["rating"] = None
    return out


# ── Bands and placements ─────────────────────────────────────────────────


def is_member(r):
    """THE FLEET IS THE SLEEVED DECKS (the pilot's rule, 2026-10-19). A bracket
    group bands and ranks only decks that are sleeved, Commander, not apart and
    bracketed; a brewing or bench deck keeps its row (its figures are real) but
    joins no group, and the fleet page draws only members."""
    return (r.get("stage") == "sleeved" and not r.get("apart") and r.get("format") is None
            and r["bracket"].get("group") != "unbracketed")



def _measure_value(r, path):
    block = r.get(path[0])
    if not isinstance(block, dict) or is_absent(block):
        return None
    cell = block.get(path[1])
    return cell["value"] if is_fig(cell) and isinstance(cell["value"], (int, float)) \
        and not isinstance(cell["value"], bool) else None


def bands(rows, members):
    """Per measure: min / median / max / n over the members that carry it, or
    absent below `MIN_GROUP`."""
    out = {}
    for name, path, _hib in BAND_MEASURES:
        vals = [(_measure_value(r, path)) for r in rows if r["slug"] in members]
        vals = [v for v in vals if v is not None]
        if len(vals) < MIN_GROUP:
            out[name] = absent(f"fewer than {MIN_GROUP} decks carry it (n={len(vals)})",
                               source=f"{path[0]}.{path[1]}")
            continue
        out[name] = {"min": _r(min(vals)), "median": _r(statistics.median(vals)),
                     "max": _r(max(vals)), "n": len(vals)}
    return out


def _placements(r, group_bands, scope):
    from manamap.pilot.diagnostic import place_in
    out = {}
    for name, path, hib in BAND_MEASURES:
        v = _measure_value(r, path)
        band = group_bands.get(name)
        if v is None:
            continue
        if is_absent(band):
            n = band["absent_because"].split("n=")[-1].rstrip(")") if "n=" in band["absent_because"] else "?"
            out[name] = f"no band (n={n})"
            continue
        out[name] = place_in(v, band, higher_is_better=hib, scope=scope)
    return out


# ── Comparisons ──────────────────────────────────────────────────────────


def _spearman_or_refusal(pairs, name, what_it_would_take):
    n = len(pairs)
    if n < calibrate.MIN_DECKS:
        return {"verdict": "NOT ANSWERABLE", "n": n,
                "why": f"Spearman({name}) needs {calibrate.MIN_DECKS} decks carrying both; the fleet has {n}",
                "what_it_would_take": what_it_would_take}
    rho = calibrate.spearman([p[0] for p in pairs], [p[1] for p in pairs])
    crit = calibrate.critical_rho(n)
    return {"verdict": "measured", "n": n, "rho": _r(rho), "critical_rho": _r(crit),
            "significant": abs(rho) >= crit}


def _series(slug, key):
    doc = load_json(deck_dir(slug) / "benchmark.json") or {}
    s = (doc.get("series") or {}).get(key)
    return {str(k): _r(v) for k, v in s.items()} if isinstance(s, dict) else None


def comparisons(rows):
    fvd, hist, grades, kill, mana, land, prices = [], {}, {}, {}, {}, {}, []
    dti_vs_kill, dti_vs_table = [], []
    for r in rows:
        d = r["dti"]
        if not is_absent(d):
            fv = d.get("floor_vs_dti") or {}
            fvd.append({"slug": r["slug"], "floor": r["bracket"]["floor"], "target": r["bracket"]["target"],
                        "dti_bracket": d.get("bracket"), "dti_score": d["score"]["value"],
                        "relation": fv.get("relation"), "reading_as_of": d.get("as_of"),
                        "stale_list": d.get("stale_list")})
            doc = dti.load(r["slug"]) or {}
            hist[r["slug"]] = [{"as_of": x.get("as_of"), "score": x.get("score"), "bracket": x.get("bracket"),
                                "decklist_sha256": x.get("decklist_sha256"),
                                "current_list": x.get("decklist_sha256") == r["decklist_sha256"]}
                               for x in doc.get("readings") or []]
            grades[r["slug"]] = d.get("benchmarks") or {}
            k8 = _measure_value(r, ("benchmark", "kill_by_turn_8"))
            if k8 is not None:
                dti_vs_kill.append((d["score"]["value"], k8))
            wr = _measure_value(r, ("table", "win_rate"))
            if wr is not None and not (r["table"].get("win_rate") or {}).get("stale"):
                dti_vs_table.append((d["score"]["value"], wr))
        for key, bucket in (("kill_by_turn_rate", kill), ("mean_available_mana_by_turn", mana),
                            ("land_drop_hit_rate_by_turn", land)):
            s = _series(r["slug"], key) if not is_absent(r["benchmark"]) else None
            if s:
                bucket[r["slug"]] = s
        p = r["price"]
        if not is_absent(p):
            prices.append({"slug": r["slug"], "total_cents": p["total_cents"]["value"], "as_of": p["as_of"]})
    distribution = {}
    for bm in sorted(dti.BENCHMARKS):
        distribution[bm] = {g: sum(1 for row in grades.values() if row.get(bm) == g) for g in dti.GRADES}
    return {
        "floor_vs_dti": fvd,
        "dti_history": hist,
        "benchmark_grades": {"rows": grades, "distribution": distribution,
                             "groups": {k: list(v) for k, v in dti.BENCHMARK_GROUPS.items()},
                             "group_names": dict(dti.GROUP_NAMES)},
        "kill_curves": kill, "mana_curves": mana, "land_curves": land,
        "prices": prices,
        "spearman": {
            "dti_vs_kill_by_8": _spearman_or_refusal(
                dti_vs_kill, "DTI, benchmark kill by turn 8",
                "a DTI reading on every Commander deck, and more decks"),
            "dti_vs_table_win_rate": _spearman_or_refusal(
                dti_vs_table, "DTI, Forge win rate on the current list",
                "a Forge run on the current list of every deck with a DTI, and more decks"),
        },
    }


# ── The document ─────────────────────────────────────────────────────────


def build():
    rows = [row(s) for s in live_slugs()]
    members_of = {}
    for r in rows:
        if not is_member(r):
            continue
        members_of.setdefault(r["bracket"]["group"], []).append(r["slug"])
    groups = {g: {"members": m, "n": len(m), "bands": bands(rows, set(m))}
              for g, m in sorted(members_of.items())}
    fleet_members = {s for m in members_of.values() for s in m}
    fleet_bands = bands(rows, fleet_members)
    for r in rows:
        g = r["bracket"]["group"]
        r["member"] = r["slug"] in fleet_members
        if r["slug"] in fleet_members and g in groups:
            r["placements"] = {"group": _placements(r, groups[g]["bands"], "bracket"),
                               "fleet": _placements(r, fleet_bands, "fleet")}
        else:
            r["placements"] = {}
    limits = list(LIMITS)
    for g, spec in groups.items():
        if spec["n"] < MIN_GROUP:
            limits.append(f"{g} has {spec['n']} member(s): no bands, placements read 'no band'.")
    return {
        "schema_version": SCHEMA_VERSION,
        "decks": rows,
        "groups": groups,
        "fleet_bands": fleet_bands,
        "comparisons": comparisons(rows),
        "limits": limits,
        "meta": {
            "harness": dict(benchmark_mod.HARNESS),
            "min_group": MIN_GROUP,
            "calibrate_min_decks": calibrate.MIN_DECKS,
            "band_measures": [{"name": n, "path": f"{p[0]}.{p[1]}", "higher_is_better": h}
                              for n, p, h in BAND_MEASURES],
            "model_version": {r["slug"]: (r["own_goldfish"].get("model_version")
                                          if not is_absent(r["own_goldfish"]) else None)
                              for r in rows},
            "built_from": {r["slug"]: r["decklist_sha256"] for r in rows},
            "no_composite": NO_COMPOSITE,
        },
    }


def dumps(doc):
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write(doc=None):
    doc = doc or build()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(dumps(doc), encoding="utf-8")
    return ARTIFACT


def load():
    return load_json(ARTIFACT)


def _table(doc):
    lines = [f"FLEET — {len(doc['decks'])} decks, {len(doc['groups'])} bracket group(s)   (◆ profiles; no score)\n",
             f"  {'deck':18}{'group':>12}{'floor':>6}{'DTI':>12}  {'kill@8':<22}{'stall':<22}{'record':<10}{'survey'}"]
    for r in doc["decks"]:
        d = r["dti"]
        dti_s = "—" if is_absent(d) else f"{d['score']['value']}/{d['max_score']} B{d['bracket']}"
        k8 = _measure_value(r, ("benchmark", "kill_by_turn_8"))
        k8c = (r["benchmark"].get("kill_by_turn_8") or {}).get("ci95") if not is_absent(r["benchmark"]) else None
        k8_s = "—" if k8 is None else f"{k8:.3f} [{k8c[0]:.3f}, {k8c[1]:.3f}]"
        st = _measure_value(r, ("diagnostic", "stall_two_in_a_row"))
        stc = (r["diagnostic"].get("stall_two_in_a_row") or {}).get("ci95") if not is_absent(r["diagnostic"]) else None
        st_s = "—" if st is None else f"{st:.3f} [{stc[0]:.3f}, {stc[1]:.3f}]"
        rec = r["record"]
        rec_s = f"{rec['win']}W{rec['loss']}L/{rec['games']}"
        sv = r["survey"]
        sv_s = "—" if is_absent(sv) else f"n={sv['games_surveyed']}"
        lines.append(f"  {r['slug']:18}{r['bracket']['group']:>12}{str(r['bracket']['floor'] or '—'):>6}"
                     f"{dti_s:>12}  {k8_s:<22}{st_s:<22}{rec_s:<10}{sv_s}"
                     + ("   apart" if r["apart"] else ""))
    for g, spec in doc["groups"].items():
        lines.append(f"\n  {g}: {', '.join(spec['members'])}"
                     + ("" if spec["n"] >= MIN_GROUP else f"   (n={spec['n']}: no bands)"))
    sp = doc["comparisons"]["spearman"]["dti_vs_kill_by_8"]
    lines.append(f"\n  Spearman(DTI, kill by 8): {sp['verdict']}"
                 + (f" (n={sp['n']})" if sp["verdict"] != "measured" else f" rho={sp['rho']} n={sp['n']}"))
    return "\n".join(lines)


def main(args=None):
    doc = build()
    if getattr(args, "write", False):
        p = write(doc)
        print(f"wrote {p} — {len(doc['decks'])} decks, {len(doc['groups'])} group(s); "
              f"`manamap pilot validate-fleet-analytics` gates it")
        return
    if getattr(args, "as_json", False) or getattr(args, "json", False):
        print(dumps(doc), end="")
        return
    print(_table(doc))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fleet-analytics [--write]`.")
