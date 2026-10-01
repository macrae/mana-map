"""Pilot: `scan-candidates` — mine the corpus for a deck's optimisation candidates along
the DIMENSIONS the pilot wants to juice, and label every row with why it matched.

Built for edgar-vampires/drain-v1 (2026-09-30). The pilot's brief for the deck is a
list of dimensions — life DRAIN as the engine, the GAIN that feeds it, big evasive
THREATS, free OUTLETS, a SWEEPER, and DRAW that refills a hand emptied by turn 7 — and
`card-search` can reach each one only through a different door: `--role wincon:drain`
here, an oracle regex there, and never the mechanical tags (`lifegain`, `death_trigger`,
`evasion_*`) or the printed keywords and power that decide "big and scary". This is
one pass over the corpus with a NAMED predicate set per dimension, so the shortlist a
staging `--why` cites can point at a row and the row says which predicate found it.

THREE RULES, each one a lesson this bench has already paid for:

* **It retrieves and labels; it does not score fit.** The same contract as `close`:
  every ranking signal is stated per row (EDHREC rank, synergy edges into the 99 with
  their rule names, the commander page's synergy, the Forge AI flag) and the sort is
  EDHREC rank. A composite "fit" number would be an authored weight driving a headline.
* **A converter is FLAGGED, never ranked away and never silently dropped.** The deck is
  held at bracket 3 with zero two-card infinites, and the loss-to-gain cards (Exquisite
  Blood) are exactly the ones every published drain list runs beside Vito or Sanguine
  Bond. A candidate that forms a two-card infinite with a card in the (staged) 99 is
  marked `infinite_with` and sorted LAST — a reader sees the line and the reason.
* **Death-triggered draw splits on one word.** `nontoken` (Midnight Reaper, Grim
  Haruspex) misses every eminence token; Species Specialist and Liliana's Standard
  Bearer do not. No source makes the distinction, so the row does.

The output is DATED and tracked (`candidate_scan.json`), like `deck_recon.json`: it is
the evidence a staging decision cites, and the decklist moves under it, so the validator
WARNS rather than fails when a candidate has since joined the 99.
"""
import json
import re
from datetime import date

from manamap.pilot.card_pool import card_keywords, corpus_oracle, load_frame, load_pool
from manamap.pilot.card_search import UNRANKED, commander_identity, deck_names
from manamap.pilot.common import (
    deck_dir,
    decklist_sha256,
    expand_faces,
    load_card_roles,
    load_combo_details,
    load_json,
    load_synergy_graph,
    resolve_out_path,
)

ARTIFACT = "candidate_scan.json"
EDHREC_FILE = "edhrec_cards.json"
DIMENSIONS = ("drain", "gain", "threat", "outlet", "sweeper", "draw")
DEFAULT_LIMIT = 40

#: The keywords that make a body "scary" on the threat dimension — the ways it hurts
#: people that a blocker does not stop. Printed keywords only (`card_keywords`), so a
#: granted keyword does not count: the same floor `combat_damage_by_keyword` carries.
SCARY_KEYWORDS = ("Flying", "Trample", "Menace", "Deathtouch", "Lifelink", "Double strike")
THREAT_MIN_POWER = 4

#: THE PREDICATES, one home. Each is (id, regex) over the lowercased oracle text; a row
#: records the ids that fired, so "why is this here" is answered by the artifact.
_P = {
    # drain: an opponent loses life that is not damage
    "drain.each_or_target_loses": r"\b(each|target) (opponent|player)s? loses? (that much |\d+ |x |twice that much )?life\b",
    "drain.opponents_lose": r"\b(your )?opponents? (each )?loses? (that much |\d+ |x )?life\b",
    "drain.equal_to": r"loses? life equal to",
    "drain.death_trigger": r"whenever (a|another|one or more) (nontoken )?(other )?creatures?[^,\n]{0,30}?(dies|die|is put into a graveyard)",
    "drain.converter_gain_to_loss": r"whenever you gain life.{0,80}(loses?|lose) (that much |\d+ |x )?life",
    "drain.converter_loss_to_gain": r"whenever (an|each) opponent loses life.{0,80}(you )?gain (that much |\d+ |x )?life",
    # gain: the input of a gain/drain engine
    "gain.you_gain": r"\byou gain (\d+|x|that much|twice that much) life\b",
    "gain.payoff": r"whenever you gain life",
    # threat: an ability that hurts people
    "threat.activated_hurt": r"(\{[^}]*\}|remove [^:\n]{0,60}counters?[^:\n]{0,40}|sacrifice [^:\n]{0,40}): [^\n]{0,80}(loses? \d+ life|deals \d+ damage|destroy target|exile target|gets -)",
    # outlet: a sacrifice in the cost
    "outlet.free_sacrifice": r"(^|\n|, )sacrifice (a|another|an untapped|two) (creature|permanent|artifact)s?( or \w+)?(, |: )",
    "outlet.mill": r"\bmills?\b",
    # sweeper
    "sweeper.destroy_all": r"(destroy|exile) all (other )?(creatures|nonland permanents|permanents)",
    "sweeper.minus_all": r"(all|each) (other )?creatures? (gets?|your opponents control gets?) -\d+/-\d+",
    "sweeper.one_sided": r"(you don'?t control|your opponents? control|opponents? controls?)",
    # draw: by trigger class
    "draw.cards": r"\bdraws? (a |two |three |four |\d+ |x |that many )?cards?\b",
    "draw.put_into_hand": r"(reveal|exile|look at) the top (card|\w+ cards)[^\n]{0,120}put (it|that card|one of them|them) into your hand",
    "draw.death": r"whenever (a|another|one or more) (nontoken )?(other )?creatures?[^,\n]{0,30}?(dies|die)[^\n]{0,80}draw",
    "draw.died_this_turn": r"(died (under your control )?this turn[^\n]{0,80}draw|draw[^\n]{0,80}died (under your control )?this turn)",
    "draw.nontoken": r"nontoken",
    "draw.gain": r"whenever you gain life[^\n]{0,140}draw",
    "draw.typal": r"(vampire[^\n]{0,60}draw|draw[^\n]{0,60}vampire)",
    "draw.sacrifice": r"(sacrifice [^:\n]{0,60}: [^\n]{0,40}draw|additional cost[^\n]{0,60}sacrifice[^\n]{0,80}draw)",
    "draw.costs_life": r"((pay|lose) \d+ life[^\n]{0,60}draw|draw[^\n]{0,60}lose \d+ life)",
}
PREDICATES = {k: re.compile(v, re.IGNORECASE | re.DOTALL) for k, v in _P.items()}

#: Which predicates / roles / tags / keywords ADMIT a card to each dimension. Everything
#: else in `_P` is a FLAG recorded on an admitted row, never an admission.
ADMIT = {
    # A CONVERTER IS ADMITTED AND FLAGGED: Exquisite Blood's "whenever an opponent loses
    # life, you gain that much" names no drain itself, and leaving it out would hide the
    # one card every published drain list runs that bracket 3 forbids.
    "drain": {"oracle": ("drain.each_or_target_loses", "drain.opponents_lose", "drain.equal_to",
                         "drain.converter_gain_to_loss", "drain.converter_loss_to_gain"),
              "roles": ("wincon:drain",), "tags": ()},
    "gain": {"oracle": ("gain.you_gain",), "roles": (), "tags": ("lifegain",), "keywords": ("Lifelink",)},
    "threat": {"oracle": (), "roles": (), "tags": ()},            # power + keyword, below
    "outlet": {"oracle": ("outlet.free_sacrifice",), "roles": ("sac-outlet",), "tags": ()},
    "sweeper": {"oracle": ("sweeper.destroy_all", "sweeper.minus_all"), "roles": ("removal:sweeper",), "tags": ()},
    # Admitted by ROLE, by the `draw` mechanical tag, or by the oracle saying "draw a card"
    # / "put it into your hand": Midnight Reaper carries the tag and no draw role, and
    # Liliana's Standard Bearer ("draw cards equal to …") carries neither.
    "draw": {"oracle": ("draw.cards", "draw.put_into_hand"),
             "roles": ("draw:engine", "draw:burst", "draw:impulse", "draw:wheel", "draw:cantrip"),
             "tags": ("draw",)},
}

LIMITS = (
    "Retrieval, not judgement: rows are sorted by EDHREC rank (unranked last) with every "
    "signal stated per row; nothing here scores fit. A row flagged `infinite_with` sorts "
    "after every unflagged row in its dimension and is never dropped.",
    "Oracle predicates are regexes over printed text and read what the text says, not what "
    "the card does at a table: a drain keyed to a trigger the deck never fires is still a "
    "drain here. `matched` names the predicate ids so a reader can re-run the question.",
    "Keywords and power are PRINTED (`card_pool.card_keywords`); a granted keyword or a "
    "pumped body does not count, and a `*` power reads as unknown, not 0.",
    "`infinite_with` is read from combo_details two-card lines whose other card is in the "
    "99 (or the staged list under --against-branch); a three-card line is counted in "
    "`two_card_lines`/`bracket_max` only. Game Changers are dropped (bracket 3) and counted.",
    "`edhrec` per row is present only when edhrec_cards.json exists for the deck, dated by "
    "its own as_of; absent means not fetched, never zero.",
)


def _text(v):
    return "" if v is None or v != v else str(v)


def _tags(cell):
    return [t.strip() for t in _text(cell).split(",") if t.strip()]


def _frame_rows():
    """{name: {oracle, type_line, tags, cmc, mana_cost}} straight from the frame."""
    frame = load_frame()
    out = {}
    for name, tl, tags, cmc, mc in zip(frame["name"], frame["type_line"], frame["mechanical_tags"],
                                        frame["cmc"], frame["mana_cost"]):
        if name in out:
            continue
        out[name] = {"type_line": _text(tl), "tags": _tags(tags),
                     "cmc": 0.0 if cmc != cmc else float(cmc), "mana_cost": _text(mc)}
    return out


def staged_names(slug, branch):
    """The 99 the scan excludes against — the deck's, or a branch's decklist."""
    if not branch:
        return deck_names(slug)
    from manamap.pilot.fetch_deck import parse_decklist
    text = (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8")
    out = set()
    for e in parse_decklist(text):
        out |= expand_faces(e["name"])
    return out


def _edhrec(slug):
    doc = load_json(deck_dir(slug) / EDHREC_FILE)
    if not doc:
        return None, {}
    return doc.get("as_of"), doc.get("cards") or {}


def _two_card_lines(name, present, details):
    """(infinite_with, two_card_lines, bracket_max) for one candidate against `present`."""
    combos = details["combos"]
    inf, lines, brackets = [], 0, []
    for i in details["by_card"].get(name, []):
        c = combos[i]
        if len(c["cards"]) != 2:
            continue
        other = next((x for x in c["cards"] if x != name), None)
        if other is None or other not in present:
            continue
        lines += 1
        if c.get("bracket") is not None:
            brackets.append(c["bracket"])
        if any(str(p).lower().startswith("infinite") for p in c.get("produces", [])):
            inf.append(other)
    return sorted(set(inf)), lines, (max(brackets) if brackets else None)


def _admitted(dim, name, rec, oracle, roles, kw, fired):
    a = ADMIT[dim]
    hit = {"oracle": [p for p in a["oracle"] if p in fired],
           "roles": [r for r in a["roles"] if r in roles],
           "tags": [t for t in a["tags"] if t in rec["tags"]],
           "keywords": [k for k in a.get("keywords", ()) if k in kw["keywords"]]}
    if dim == "threat":
        front = rec["type_line"].split(" // ")[0]
        scary = [k for k in SCARY_KEYWORDS if k in kw["keywords"]]
        big = kw["power"] is not None and kw["power"] >= THREAT_MIN_POWER
        if "Creature" in front and big and (scary or "threat.activated_hurt" in fired):
            hit["keywords"] = scary
            if "threat.activated_hurt" in fired:
                hit["oracle"] = ["threat.activated_hurt"]
            return hit
        return None
    return hit if any(hit.values()) else None


def _flags(dim, rec, oracle, kw, fired, roles):
    f = {}
    if dim == "drain":
        f["death_drain"] = "death_trigger" in rec["tags"] or "drain.death_trigger" in fired
        conv = [k for k in ("drain.converter_gain_to_loss", "drain.converter_loss_to_gain") if k in fired]
        f["converter"] = conv[0].rsplit(".", 1)[1] if conv else None
    elif dim == "gain":
        f["lifelink"] = "Lifelink" in kw["keywords"]
        f["payoff"] = "gain.payoff" in fired
    elif dim == "threat":
        f["scary"] = len([k for k in SCARY_KEYWORDS if k in kw["keywords"]]) + (1 if "threat.activated_hurt" in fired else 0)
        f["activated_hurt"] = "threat.activated_hurt" in fired
        f["typal"] = "Vampire" in rec["type_line"]
        f["death_drain"] = "death_trigger" in rec["tags"] and ("lifegain" in rec["tags"] or "wincon:drain" in roles)
    elif dim == "outlet":
        f["free"] = "outlet.free_sacrifice" in fired
        f["mill"] = "outlet.mill" in fired
    elif dim == "sweeper":
        f["one_sided"] = "sweeper.one_sided" in fired
    elif dim == "draw":
        trig = [k for k in ("draw.death", "draw.died_this_turn", "draw.gain", "draw.typal", "draw.sacrifice") if k in fired]
        f["trigger"] = ("death" if trig and trig[0] in ("draw.death", "draw.died_this_turn")
                        else trig[0].rsplit(".", 1)[1] if trig else "plain")
        f["nontoken"] = ("draw.death" in fired) and ("draw.nontoken" in fired)
        f["costs_life"] = "draw.costs_life" in fired
    return f


def scan(slug, dimensions=DIMENSIONS, branch=None, limit=DEFAULT_LIMIT):
    pool = load_pool()
    if pool is None:
        raise SystemExit("cards.csv is absent — scan-candidates reads the corpus. Run `manamap extract` first.")
    rows = _frame_rows()
    oracle = corpus_oracle()
    keywords = card_keywords()
    roles_map = load_card_roles()
    details = load_combo_details()
    synergy = load_synergy_graph()
    from manamap.sim import forge_cards
    ident = commander_identity(slug)
    present = staged_names(slug, branch)
    edhrec_as_of, edhrec = _edhrec(slug)
    dims = list(dimensions)

    excluded = {"in_99": 0, "identity": 0, "illegal": 0, "game_changer": []}
    found = {d: [] for d in dims}
    for name, rec in pool.items():
        if expand_faces(name) & present:
            excluded["in_99"] += 1
            continue
        if not rec["legal"]:
            excluded["illegal"] += 1
            continue
        if not rec["color_identity"] <= ident:
            excluded["identity"] += 1
            continue
        fr = rows.get(name)
        if fr is None:
            continue
        text = oracle.get(name, "")
        low = text.lower()
        fired = {k for k, p in PREDICATES.items() if p.search(low)}
        roles = roles_map.get(name) or []
        kw = keywords.get(name) or {"keywords": [], "power": None, "toughness": None}
        for dim in dims:
            hit = _admitted(dim, name, fr, text, roles, kw, fired)
            if hit is None:
                continue
            if rec["game_changer"]:
                if name not in excluded["game_changer"]:
                    excluded["game_changer"].append(name)
                continue
            inf, lines, bmax = _two_card_lines(name, present, details)
            edges = [{"partner": e["partner"], "rules": e.get("synergies") or []}
                     for e in (synergy.get(name) or []) if expand_faces(e["partner"]) & present]
            row = {
                "name": name, "cmc": fr["cmc"], "mana_cost": fr["mana_cost"], "type_line": fr["type_line"],
                "keywords": kw["keywords"], "power": kw["power"], "toughness": kw["toughness"],
                "roles": sorted(roles), "tags": fr["tags"], "edhrec_rank": rec["edhrec_rank"],
                "matched": hit, "flags": _flags(dim, fr, text, kw, fired, roles),
                "synergy_into_99": edges,
                "combos": {"infinite_with": inf, "two_card_lines": lines, "bracket_max": bmax},
                "forge_ai_flag": forge_cards.ai_flag(name),
                "oracle_text": text,
            }
            if edhrec:
                e = edhrec.get(name) or edhrec.get(name.split(" // ")[0])
                if e is not None:
                    row["edhrec"] = e
            found[dim].append(row)

    def key(r):
        return (1 if r["combos"]["infinite_with"] else 0,
                r["edhrec_rank"] if r["edhrec_rank"] is not None else UNRANKED, r["name"])
    out_dims = {}
    for dim in dims:
        allrows = sorted(found[dim], key=key)
        out_dims[dim] = {
            "admits": {k: list(v) for k, v in ADMIT[dim].items()} | (
                {"power_at_least": THREAT_MIN_POWER, "scary_keywords": list(SCARY_KEYWORDS)} if dim == "threat" else {}),
            "matched": len(allrows),
            "flagged_infinite": sum(1 for r in allrows if r["combos"]["infinite_with"]),
            "candidates": allrows[:limit],
            "truncated": max(0, len(allrows) - limit),
        }
    return {
        "slug": slug, "as_of": date.today().isoformat(),
        "decklist_sha256": decklist_sha256(slug, branch), "against_branch": branch,
        "identity": sorted(ident), "limit": limit,
        "sources": {"edhrec_cards": edhrec_as_of, "combo_details": details.get("meta"),
                    "predicates": {k: v for k, v in _P.items()}},
        "dimensions": out_dims, "excluded": excluded, "limits": list(LIMITS),
    }


def format_report(doc):
    out = [f"CANDIDATE SCAN — {doc['slug']} ({doc['as_of']}) · identity {''.join(doc['identity'])}"
           + (f" · against branch {doc['against_branch']}" if doc.get("against_branch") else "")
           + f" · EDHREC page {'fetched ' + doc['sources']['edhrec_cards'] if doc['sources'].get('edhrec_cards') else 'ABSENT'}"]
    ex = doc["excluded"]
    out.append(f"  excluded: {ex['in_99']} in the 99, {ex['identity']} out of identity, {ex['illegal']} illegal, "
               f"{len(ex['game_changer'])} Game Changer(s) dropped" + (f" ({', '.join(ex['game_changer'][:6])}…)" if ex["game_changer"] else ""))
    for dim, d in doc["dimensions"].items():
        out.append("")
        out.append(f"  {dim.upper()} — {d['matched']} match(es), showing {len(d['candidates'])}"
                   + (f", {d['truncated']} more" if d["truncated"] else "")
                   + (f"; {d['flagged_infinite']} flagged infinite_with, sorted last" if d["flagged_infinite"] else ""))
        for r in d["candidates"]:
            rank = r["edhrec_rank"] if r["edhrec_rank"] is not None else "unranked"
            fl = ", ".join(f"{k}={v}" for k, v in r["flags"].items() if v not in (False, None, 0, "plain"))
            inf = f"  ⚠ INFINITE with {', '.join(r['combos']['infinite_with'])}" if r["combos"]["infinite_with"] else ""
            kw = f" [{', '.join(r['keywords'])}]" if r["keywords"] else ""
            pt = f" {r['power']}/{r['toughness']}" if r["power"] is not None else ""
            ed = f"  edhrec {rank}" + (f" · page synergy {r['edhrec'].get('synergy'):+.2f}" if r.get("edhrec") and r["edhrec"].get("synergy") is not None else "")
            ai = f"  AI:{r['forge_ai_flag']}" if r.get("forge_ai_flag") else ""
            out.append(f"    {r['mana_cost'] or '—':<10} {r['name']}{pt}{kw}{ed}{ai}{inf}")
            why = " · ".join(f"{k}:{','.join(v)}" for k, v in r["matched"].items() if v)
            out.append(f"      {why}" + (f"  flags: {fl}" if fl else "")
                       + (f"  synergy→99: {len(r['synergy_into_99'])}" if r["synergy_into_99"] else ""))
    return "\n".join(out)


def main(args):
    slug = args.slug
    names = list(getattr(args, "shortlist", None) or [])
    if names:
        doc = shortlist(slug, names, branch=getattr(args, "against_branch", None))
        if getattr(args, "as_json", False):
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        else:
            print(format_shortlist(doc))
        out = getattr(args, "out", None)
        if out:
            p = resolve_out_path(out, slug, "shortlist")
            p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  wrote {p}")
        return
    dim = getattr(args, "dimension", None) or "all"
    dims = DIMENSIONS if dim == "all" else (dim,)
    doc = scan(slug, dims, branch=getattr(args, "against_branch", None),
               limit=getattr(args, "limit", None) or DEFAULT_LIMIT)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(format_report(doc))
    if getattr(args, "write", False):
        if dim != "all":
            raise SystemExit("--write records the whole scan: run it without --dimension (or with all)")
        p = deck_dir(slug) / ARTIFACT
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n  wrote {p}")
    out = getattr(args, "out", None)
    if out:
        p = resolve_out_path(out, slug, "candidate-scan")
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot scan-candidates <slug>`.")


# ── THE SHORTLIST JOIN (Phase 3 of the drain-v1 plan) ────────────────────────────────────
#
# One row per candidate the pilot is weighing: what the scan says (dimensions, flags),
# what the prescription said (rank, closes, natural cut), which recon findings name it
# (dimension, confidence), the EDHREC page figures, `assess`'s read (job, gate, what the
# goldfish can price, the Forge AI flag) and a PREDICTED DIRECTION per Forge axis read off
# the flags — up / flat / unknown, never a number. A view, never tracked.

#: Which scan dimension predicts which Forge axis to move, and how. Stated here so the
#: shortlist's "predicted" column is a rule a reader can disagree with, not a judgement.
AXIS_PREDICTION = {
    "drain": {"forge.drain_dealt": "up"},
    "gain": {"forge.life_gained": "up"},
    "threat": {"forge.biggest_hit": "up", "forge.evasive_damage_share": "up"},
    "outlet": {"forge.drain_dealt": "up (via deaths)", "forge.kills_by_ability": "up (floor: AI must activate)"},
    "sweeper": {"forge.drain_dealt": "up (deaths on our terms)", "forge.combat_damage_dealt_to_players": "down (our board dies too)"},
    "draw": {"forge.extra_draw_per_turn": "up", "forge.empty_hand_turns": "down"},
}


def _prescription(slug):
    """The newest prescription's adds and cuts, by card."""
    import glob
    paths = sorted(glob.glob(str(deck_dir(slug) / "prescriptions" / "*.json")))
    if not paths:
        return {}, {}, None
    doc = load_json(__import__("pathlib").Path(paths[-1])) or {}
    adds = {a["card"]: dict(a, rank=i + 1) for i, a in enumerate(doc.get("add_candidates") or [])}
    cuts = {c["card"]: dict(c, rank=i + 1) for i, c in enumerate(doc.get("cut_candidates") or [])}
    return adds, cuts, doc.get("id")


def _recon(slug):
    doc = load_json(deck_dir(slug) / "deck_recon.json") or {}
    by = {}
    for i, f in enumerate(doc.get("findings") or []):
        for c in f.get("cards") or []:
            by.setdefault(c, []).append({"finding": i, "dimension": f.get("dimension"),
                                         "confidence": f.get("confidence"), "claim": f.get("claim", "")[:140]})
    return by, doc.get("as_of")


def shortlist(slug, names, branch=None):
    """Join every source the bench has on `names`. Returns the rows and the sources' dates."""
    from manamap.pilot import assess as _assess
    # A LIVE, UNCAPPED scan: the tracked file keeps forty rows per dimension and a card
    # can sit in the top forty of one dimension and past the cut of another (Twilight
    # Prophet is rank 1110 in draw), which read as "not a draw card" the first time.
    scan_doc = scan(slug, branch=branch, limit=5000)
    in_scan = {}
    for dim, block in scan_doc["dimensions"].items():
        for r in block["candidates"]:
            if r["name"] in names:
                in_scan.setdefault(r["name"], {})[dim] = r
    adds, cuts, rx_id = _prescription(slug)
    recon, recon_as_of = _recon(slug)
    _, edhrec = _edhrec(slug)
    try:
        arows = {r["card"]: r for r in (_assess.assess(slug, list(names), branch) or {}).get("cards") or []}
        assess_error = None
    except Exception as exc:                        # noqa: BLE001 - a view degrades, never dies
        arows, assess_error = {}, f"{exc.__class__.__name__}: {exc}"
    rows = []
    for name in names:
        dims = in_scan.get(name, {})
        any_row = next(iter(dims.values()), None)
        predicted = {}
        for dim in dims:
            for axis, direction in AXIS_PREDICTION[dim].items():
                predicted.setdefault(axis, direction)
        a = arows.get(name) or {}
        rows.append({
            "name": name,
            "dimensions": {d: {k: v for k, v in r["flags"].items() if v not in (False, None, 0, "plain")} for d, r in dims.items()},
            "matched": {d: r["matched"] for d, r in dims.items()},
            "edhrec_rank": any_row["edhrec_rank"] if any_row else None,
            "infinite_with": sorted({x for r in dims.values() for x in r["combos"]["infinite_with"]}),
            "forge_ai_flag": any_row["forge_ai_flag"] if any_row else a.get("forge_ai_flag"),
            "prescription": ({"as": "add", "rank": adds[name]["rank"], "closes": adds[name].get("closes"),
                              "natural_cut": adds[name].get("natural_cut")} if name in adds else
                             {"as": "cut", "rank": cuts[name]["rank"], "difficulty": cuts[name].get("difficulty")} if name in cuts else None),
            "recon": recon.get(name) or [],
            "edhrec": ({k: edhrec[name].get(k) for k in ("synergy", "num_decks", "from")} if name in edhrec else None),
            "assess": ({k: a.get(k) for k in ("job", "gate", "model_sees", "answers", "verdict", "mv")} if a else None),
            "predicted": predicted,
            "in_scan": bool(dims),
        })
    return {"slug": slug, "against_branch": branch, "sources": {"scan": scan_doc.get("as_of"), "prescription": rx_id,
                                                                 "recon": recon_as_of, "edhrec": scan_doc["sources"].get("edhrec_cards"),
                                                                 "assess": assess_error or "ok"},
            "rows": rows}


def format_shortlist(doc):
    out = [f"SHORTLIST — {doc['slug']}" + (f" against branch {doc['against_branch']}" if doc.get("against_branch") else "")
           + f" · scan {doc['sources']['scan']} · prescription {doc['sources']['prescription']} · recon {doc['sources']['recon']}"
           + f" · EDHREC {doc['sources']['edhrec'] or 'ABSENT'}"
           + ("" if doc["sources"].get("assess") == "ok" else f"\n  assess unavailable: {doc['sources'].get('assess')}")]
    for r in doc["rows"]:
        out.append("")
        head = f"  {r['name']}" + (f"  edhrec {r['edhrec_rank']}" if r["edhrec_rank"] is not None else "")
        if r["edhrec"]:
            head += f"  page synergy {r['edhrec']['synergy']:+.2f} · {r['edhrec']['num_decks']} decks ({r['edhrec']['from']})"
        if r["forge_ai_flag"]:
            head += f"  AI:{r['forge_ai_flag']}"
        if r["infinite_with"]:
            head += f"  ⚠ INFINITE with {', '.join(r['infinite_with'])}"
        out.append(head)
        if not r["in_scan"]:
            out.append("    scan: NOT ADMITTED by any dimension (in the 99, out of identity, a Game Changer, or matches no predicate)")
        for d, fl in r["dimensions"].items():
            out.append(f"    {d:8s} " + (", ".join(f"{k}={v}" for k, v in fl.items()) or "—")
                       + "   admitted by " + " · ".join(f"{k}:{','.join(v)}" for k, v in r["matched"][d].items() if v))
        if r["prescription"]:
            p = r["prescription"]
            out.append(f"    prescription: {p['as']} #{p['rank']}" + (f" — closes {p['closes']}" if p.get("closes") else "")
                       + (f"; natural cut {p['natural_cut']}" if p.get("natural_cut") else "") + (f"; {p['difficulty']}" if p.get("difficulty") else ""))
        for f in r["recon"]:
            out.append(f"    recon[{f['finding']}] {f['dimension']} · {f['confidence']}: {f['claim']}")
        if r["assess"]:
            a = r["assess"]
            out.append(f"    assess: job {a.get('job')} · gate {a.get('gate')} · model sees {a.get('model_sees') or '—'} · {a.get('verdict')}")
        if r["predicted"]:
            out.append("    predicted: " + "; ".join(f"{k} {v}" for k, v in r["predicted"].items()))
    return "\n".join(out)
