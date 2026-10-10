"""Pilot: `try` — a swap idea to an answer in under two minutes, one screen, nothing written.

THE LOOP WAS A DAY LONG. An idea became a branch, the branch became two Forge pod
runs at ~0.7 games a minute, the runs died to a sleeping laptop, and what came back
was INCONCLUSIVE at an MDE of 0.14 (sharknado/momentum-v1, edgar/draw-v1, 2026-10-04).
Worse, draw-v1 cut the pilot's favourite card and nobody read its OUT list before
eight hours of games. The pilot's ruling: a swap gets an answer in under two
minutes, from the terminal; Forge is a targeted probe, never the decision loop.

    manamap pilot try <slug> --out "A" --in "B" [--out C --in D ...]
                      [--branch B] [--each] [--stage NAME] [--json]

One screen, read top to bottom:

  1. THE SWAPS — every card leaving and arriving, with its roles, whether a declared
     target names it, what the goldfish can see of it (seen / DARK / invisible), and
     what Forge's AI did with it in runs already on disk. That last column is AI
     BEHAVIOUR and is never a reason to cut: Vish Kal was cut on "0 activations".
  2. REFUSALS AND WARNINGS — the pilot's keep list (`protected.json`), the paper
     refusals `stage` applies (singleton, size, commander, identity), and the
     colour sources before and after.
  3. THE GOLDFISH ROWS — the champion and the swapped list measured on the same
     harness `net-change` uses (`net_change.compare_readings`), each row with the
     interval on its own difference, Holm across the family, and a note naming any
     swapped card that row cannot see.
  4. ONE LINE — better / worse / trade / no call, how far to trust it, and the
     cheapest thing that would raise that trust.

`--stage NAME` writes the swaps to a branch only AFTER the screen is printed, so a
staged branch is always one somebody looked at.
"""
import hashlib
import json

from manamap import config
from manamap.pilot import deck_branch, check_in, protected
from manamap.pilot.common import deck_dir, deck_file, load_deck_cards, load_json

CACHE = config.DATA_DIR / "cache" / "try"


def _key(name):
    return str(name or "").split(" // ")[0].strip().lower()


# ── the swapped list, held in memory ─────────────────────────────────────

_SCRYFALL = {}


_NOT_A_CARD = {"art_series", "token", "double_faced_token", "emblem"}


def _scryfall_objects(names):
    """name-key -> the Scryfall oracle object, from the local bulk dump.

    THE CORPUS IS NOT A CARDS.JSON ENTRY. `cards.csv` flattens oracle text onto one
    line and the frame drops power, toughness and colours — and the goldfish's
    parsers read line breaks (an activation window stops at the next ability). A
    card built from it measured differently from the same card fetched: try's
    damage@T10 read 56.2 where net-change read 57.7 on the identical six swaps
    (2026-10-04). The bulk dump holds the very objects `fetch-deck` shapes."""
    from manamap.ingest.common import open_dump
    want = {_key(n) for n in names} - set(_SCRYFALL)
    if want:
        with open_dump(config.RAW_JSON_PATH) as fh:
            head = fh.read(1)
            fh.seek(0)
            rows = json.load(fh) if head == "[" else (json.loads(line) for line in fh if line.strip())
            for obj in rows:
                # NOT A CARD, though it carries the card's name: an art-series
                # card has no oracle text, so the first one in the dump made
                # Phyrexian Arena, Yawgmoth and Sorin screen as BLANK cards
                # (found by the edgar doctor, 2026-10-05). Tokens and emblems
                # share names the same way.
                if obj.get("layout") in _NOT_A_CARD:
                    continue
                k = _key(obj.get("name"))
                if k in want:
                    _SCRYFALL[k] = obj
                    want.discard(k)
                    if not want:
                        break
    return {_key(n): _SCRYFALL.get(_key(n)) for n in names}


def corpus_card(name):
    """A cards.json entry for `name`, shaped by `fetch_deck.shape_card` from the
    local Scryfall dump — the same projection `fetch-deck` applies, minus the
    printing lookup. None when the card is not in the dump."""
    from manamap.pilot.fetch_deck import shape_card
    obj = _scryfall_objects([name]).get(_key(name))
    return shape_card(obj, 1, False) if obj else None


def swap_ops(swaps):
    """`[(out, in), …]` as `deck_edit` ops."""
    return [{"op": "swap", "out": o, "in": i} for o, i in swaps]


def build_doc(slug, branch, p):
    """The cards doc for a `deck_edit.plan` result, held in memory.

    Every card the base `cards.json` holds keeps its record and gets the
    plan's copy count (0 drops it); a card it does not hold is shaped from the
    Scryfall dump (`corpus_card`) exactly as `fetch-deck` would shape it. Then
    the BASE LIST'S SLOTS (`diagnostic.align`), the order `net-change` gives a
    branch: the goldfish shuffles slots, so a new card in the replaced card's
    slot keeps every other game identical — and `try` and `net-change` agree.
    Raises SystemExit for a new name the dump does not hold.
    """
    from manamap.pilot import diagnostic
    from manamap.pilot.fetch_deck import parse_mainboard, parse_sideboard
    base = load_deck_cards(slug, branch)
    base_cards = base["cards"] if isinstance(base, dict) else base
    doc = dict(base) if isinstance(base, dict) else {"cards": base}

    def board(base_list, after_entries):
        want = {}
        for e in after_entries:
            want[_key(e["name"])] = want.get(_key(e["name"]), 0) + int(e.get("quantity") or 1)
        held = {_key(c["name"]) for c in base_list}
        new = [e["name"] for e in after_entries if _key(e["name"]) not in held]
        _scryfall_objects(new)              # one pass over the dump for every new card
        out = []
        for c in base_list:
            q = want.get(_key(c["name"]), 0)
            if q:
                out.append(dict(c, quantity=q))
        for e in after_entries:
            if _key(e["name"]) in held:
                continue
            rec = corpus_card(e["name"])
            if rec is None:
                raise SystemExit(f"{e['name']!r} is not in the corpus — check the spelling, "
                                 f"or `manamap pilot card-search --name {e['name']!r}`.")
            out.append(dict(rec, quantity=int(e.get("quantity") or 1)))
            held.add(_key(e["name"]))
        return out

    text = p["text_after"]
    cards = board(base_cards, parse_mainboard(text))
    doc["cards"] = diagnostic.align(base_cards, cards)
    side = parse_sideboard(text)
    if side or doc.get("sideboard"):
        doc["sideboard"] = board(doc.get("sideboard") or [], side)
        if not doc["sideboard"]:
            doc.pop("sideboard")
    return doc


def _rows(p):
    """The plan's ops as the swap table's rows: `{out, in}` per swap, a lone
    side for an add, a cut or a set."""
    rows = []
    for op in p["ops"]:
        if op["op"] == "swap":
            rows.append({"out": op["out"], "in": op["in"]})
        elif op["op"] == "cut":
            rows.append({"out": op["name"], "in": None})
        elif op["op"] == "add":
            rows.append({"out": None, "in": op["name"]})
        else:
            moved = (op["name"] in p["diff"]["in"], op["name"] in p["diff"]["out"])
            rows.append({"out": op["name"] if moved[1] else None,
                         "in": op["name"] if moved[0] else None})
    return rows


def apply_ops(slug, branch, ops):
    """`(doc, entries, rows, warnings, plan)` for any edit ops — through
    `deck_edit.plan`, THE ONE VALIDATOR, so `try`, the preview and `edit` can
    never disagree about what is legal. Raises SystemExit on any refusal (the
    keep list included) BEFORE anything is measured."""
    from manamap.pilot import deck_edit
    p = deck_edit.plan(slug, ops, branch=branch)
    if p["blocking"]:
        raise SystemExit("Refusing that swap set:\n  - " + "\n  - ".join(p["blocking"]))
    doc = build_doc(slug, branch, p)
    rows = _rows(p)
    # The IN side of each row names the record the doc carries (`A // B`), the
    # name every other artifact uses.
    by_key = {_key(c["name"]): c["name"] for c in doc["cards"]}
    for r in rows:
        if r.get("in"):
            r["in"] = by_key.get(_key(r["in"]), r["in"])
    entries = [e for e in p["entries_after"] if e.get("board", "main") == "main"]
    return doc, entries, rows, p["warnings"], p


def apply_swaps(slug, branch, swaps):
    """`(doc, entries, rows, warnings)` — the cards doc and decklist entries with
    every swap applied. A thin wrapper over `apply_ops` (and so `deck_edit.plan`)."""
    doc, entries, rows, warnings, _ = apply_ops(slug, branch, swap_ops(swaps))
    return doc, entries, rows, warnings


# ── the readings ─────────────────────────────────────────────────────────

def _champion_key(slug, branch, iterations, seed):
    from manamap.pilot import diagnostic, goldfish
    parts = [slug, branch or "", str(iterations), str(seed), json.dumps(diagnostic.HARNESS, sort_keys=True),
             goldfish.model_version()]
    for name in ("cards.json",):
        p = deck_dir(slug, branch) / name
        parts.append(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "")
    t = deck_file(slug, "goldfish_targets.json", branch)
    parts.append(hashlib.sha256(t.read_bytes()).hexdigest() if t.exists() else "")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def champion_reading(slug, branch, iterations, seed):
    """The base list's reading, cached on everything that could change it."""
    from manamap.pilot import diagnostic
    path = CACHE / f"{slug}-{_champion_key(slug, branch, iterations, seed)}.json"
    if path.exists():
        return json.loads(path.read_text()), True
    got = diagnostic.run(slug, branch=branch, iterations=iterations, seed=seed, quiet=True,
                         keep_games=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(got, default=str))
    return got, False


# ── the swap table ───────────────────────────────────────────────────────

def _forge_evidence(slug):
    """name-key -> by_card row from the latest Forge run of this deck, or {}."""
    try:
        from manamap.sim import forge
        runs = forge.list_runs(slug)
    except (Exception, SystemExit):              # noqa: BLE001 - no records, no column
        return {}, None
    for rec in reversed(runs or []):
        by = ((rec.get("engine_casts") or {}).get("by_card")) or {}
        if by:
            return {_key(k): v for k, v in by.items()}, rec.get("run_id") or rec.get("id")
    return {}, None


def _cast_proofs(slug):
    """name-key -> verdict from any branch's cast_proofs.json (PLAYED wins)."""
    out = {}
    for p in sorted((deck_dir(slug) / "branches").glob("*/cast_proofs.json")):
        doc = load_json(p) or {}
        for name, row in (doc.get("cards") or {}).items():
            v = (row or {}).get("verdict_word") or str((row or {}).get("verdict") or "").split(":")[0].strip()
            if v and out.get(_key(name)) != "PLAYED":
                out[_key(name)] = v
    return out


def card_rows(slug, branch, base_doc, swapped_doc, swaps):
    from manamap.pilot import model_coverage
    from manamap.pilot.common import load_card_roles
    roles = load_card_roles()
    flags, named = model_coverage.declaration(slug, branch)
    named_keys = {_key(n) for n in named}
    forge_by, forge_run = _forge_evidence(slug)
    proofs = _cast_proofs(slug)
    by_key = {}
    for c in (base_doc.get("cards") or []) + (swapped_doc.get("cards") or []):
        by_key.setdefault(_key(c["name"]), c)
    out = []
    for s in swaps:
        for side in ("out", "in"):
            name = s.get(side)
            if not name:
                continue
            card = by_key.get(_key(name)) or corpus_card(name) or {"name": name}
            st = model_coverage.card_state(card, flags, named)
            f = forge_by.get(_key(name))
            out.append({
                "side": side, "name": name,
                "roles": roles.get(name) or roles.get(card.get("name")) or [],
                "in_target": _key(name) in named_keys,
                "state": st["state"], "dark": st["dark_channels"],
                "forge": ({k: f.get(k) for k in ("cast", "activated", "castable_uncast")}
                          if f else None),
                "proof": proofs.get(_key(name)),
            })
    return out, forge_run


# ── the screen ───────────────────────────────────────────────────────────

def verdict(table, blind):
    better = [r["measure"] for r in table if r.get("verdict") == "better"]
    worse = [r["measure"] for r in table if r.get("verdict") == "worse"]
    call = ("trade" if better and worse else "better" if better else "worse" if worse else "no call")
    trust = ("high — every swapped card is visible to the model" if not blind else
             f"partial — the model cannot fully see {', '.join(sorted(blind))}, so its rows are "
             f"silent about {'it' if len(blind) == 1 else 'them'}")
    return call, better, worse, trust


def _dump_stamp():
    """The Scryfall dump's size and mtime: a new card's record comes from it,
    so a refreshed dump must not serve a figure measured on the old one."""
    try:
        st = config.RAW_JSON_PATH.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


def canonical_ops(p):
    """The edit as the cache sees it: the plan's copy diff, sorted. Two op
    sequences that produce the same list are the same candidate."""
    return json.dumps(p["diff"], sort_keys=True)


def candidate_key(slug, branch, iterations, seed, p):
    """THE PREVIEW CACHE KEY: everything that could move the candidate's rows —
    the champion's own key (cards.json bytes, the targets' bytes,
    `goldfish.model_version()`, the harness, iterations and seed), the
    canonical edit, and the dump the new cards are shaped from."""
    parts = [_champion_key(slug, branch, iterations, seed), canonical_ops(p), _dump_stamp()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def compare(slug, branch, doc, p, iterations, seed):
    """`(table, champion_cached, candidate_cached)` — the paired rows, the
    candidate's cached in `data/cache/try/` on `candidate_key`."""
    from manamap.pilot import diagnostic, net_change
    a, cached = champion_reading(slug, branch, iterations, seed)
    path = CACHE / f"{slug}-edit-{candidate_key(slug, branch, iterations, seed, p)}.json"
    if path.exists():
        return json.loads(path.read_text()), cached, True
    b = diagnostic.run_on(doc, slug, branch=branch, iterations=iterations, seed=seed,
                          quiet=True, keep_games=True)
    table = net_change.compare_readings(a, b)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, default=str))
    return json.loads(json.dumps(table, default=str)), cached, False


def authored_rate_notes(base_doc, doc):
    """A FIGURE RESTING ON AN AUTHORED RATE SAYS SO — one clause per rate."""
    from manamap.pilot import goldfish_profiles as _gp
    from manamap.config import (GEYSER_TAPPED_SHARE, OPPONENT_HAND, TITHE_PAY_RATE)
    both = (base_doc.get("cards") or []) + (doc.get("cards") or [])
    notes = []
    if any(_gp.treasure_profile(c)[1] == "opponent_draw_tax" for c in both):
        notes.append(f"Smothering Tithe's Treasure assumes opponents pay its tax "
                     f"{TITHE_PAY_RATE:.0%} of the time (an authored rate)")
    _rk = {(_gp.ritual_profile(c) or {}).get("kind") for c in both}
    if "opp_tapped_lands" in _rk:
        notes.append(f"Mana Geyser assumes {GEYSER_TAPPED_SHARE:.0%} of opponents' lands "
                     f"are still tapped on your turn (an authored rate)")
    if "opp_hand" in _rk:
        notes.append(f"Jeska's Will assumes an opponent holds {OPPONENT_HAND} cards, "
                     f"7 after a wheel (an authored rate)")
    return notes


#: Said beside every goldfish figure a preview shows (CLAUDE.md, the goldfish).
BOARD_QUALITY_CAVEAT = ("the goldfish has no blockers and no removal, so it says nothing "
                        "about board QUALITY — Forge is the probe for that")


def run(slug, swaps=None, branch=None, iterations=None, seed=None, each=False, ops=None):
    from manamap.pilot import diagnostic, mana_fit, net_change
    it = iterations or diagnostic.HARNESS["iterations"]
    sd = seed if seed is not None else diagnostic.HARNESS["seed"]
    ops = list(ops or []) + swap_ops(swaps or [])
    doc, entries, rows, warnings, p = apply_ops(slug, branch, ops)
    base_doc = load_deck_cards(slug, branch)
    table, cached, _cand_cached = compare(slug, branch, doc, p, it, sd)
    cards, forge_run = card_rows(slug, branch, base_doc, doc, rows)
    blind = {c["name"] for c in cards if c["state"] != "seen"}
    call, better, worse, trust = verdict(table, blind)
    # A FIGURE RESTING ON AN AUTHORED RATE SAYS SO. Tithe's Treasure is decided by
    # TITHE_PAY_RATE, which nothing here measures.
    for note in authored_rate_notes(base_doc, doc):
        trust += f"; {note}"
    per_swap = []
    if each and len(rows) > 1 and swaps:
        for (o, i) in swaps:
            d1, _, _, _ = apply_swaps(slug, branch, [(o, i)])
            r1 = diagnostic.run_on(d1, slug, branch=branch, iterations=it, seed=sd, quiet=True,
                                   keep_games=True)
            t1 = net_change.compare_readings(a, r1)
            per_swap.append({"out": o, "in": i,
                             "moved": [(r["measure"], r["verdict"], r["delta"]) for r in t1
                                       if r.get("verdict") != "noise"]})
    try:
        before = mana_fit.shortfall(slug, branch)["colours"]
        after = mana_fit.shortfall(slug, branch, deck_doc=doc)["colours"]
        mana = {c: {"before": before[c]["short"], "after": after[c]["short"],
                    "target": after[c]["target"]} for c in "WUBRG" if after[c]["target"]}
    except Exception as e:                       # noqa: BLE001 - mana report is a warning
        mana = {"error": str(e)}
    raise_trust = [f"`manamap pilot forge-cast-check {slug} --card \"{c['name']}\"` (~1 min) — "
                   f"the model cannot see it, Forge can say whether it is played"
                   for c in cards if c["side"] == "in" and c["state"] != "seen" and c["proof"] != "PLAYED"]
    return {"slug": slug, "branch": branch, "swaps": rows, "cards": cards,
            "forge_run": forge_run, "warnings": warnings, "mana": mana,
            "table": table, "per_swap": per_swap,
            "verdict": {"call": call, "better": better, "worse": worse, "trust": trust,
                        "raise": raise_trust},
            "harness": {"iterations": it, "seed": sd, "champion_cached": cached}}


# ── the preview: what an in-place edit would do, before it is applied ────

def _facts_by_key(names):
    from manamap.pilot import deck_edit
    return {_key(n): f for n, f in deck_edit.card_facts(names).items()}


def _curve(copies, facts):
    """mana value -> nonland copies (7 is 7+), from the corpus pool. A card the
    pool does not hold is counted under `unknown`, never guessed."""
    out = {str(i): 0 for i in range(8)}
    unknown = 0
    for name, k in copies.items():
        f = facts.get(_key(name))
        if not f:
            unknown += k
            continue
        if "Land" in (f.get("type_line") or "").split("//")[0]:
            continue
        out[str(min(int(float(f.get("cmc") or 0)), 7))] += k
    if unknown:
        out["unknown"] = unknown
    return out


def _roles(copies, facts):
    from manamap.pilot.common import load_card_roles
    try:
        roles = load_card_roles()
    except Exception:                               # noqa: BLE001 - no roles file, no figure
        return None
    c = {}
    for name, k in copies.items():
        full = (facts.get(_key(name)) or {}).get("name") or name
        for r in roles.get(full) or roles.get(name) or []:
            c[r] = c.get(r, 0) + k
    return c


def _combos(copies, commanders, facts):
    from manamap.pilot import deck_combos
    from manamap.pilot.common import load_combo_details
    details = load_combo_details()
    names = {(facts.get(_key(n)) or {}).get("name") or n for n in copies}
    cmd = {(facts.get(_key(n)) or {}).get("name") or n for n in commanders}
    return {r["id"]: r for r in deck_combos.included_combos(names, cmd, details, ranked=False)}


def _price(slug, branch, diff):
    """The price delta from the deck's DATED `prices.json` (and its branches',
    for cards only a branch priced) — never a live lookup. Absent without one."""
    base = deck_dir(slug)
    doc = load_json(deck_dir(slug, branch) / "prices.json") or load_json(base / "prices.json")
    if not doc or not isinstance(doc.get("cards"), dict):
        return {"absent": f"no prices.json — `manamap pilot prices {slug} --write` records one"}
    table = {_key(n): (r, doc.get("as_of")) for n, r in doc["cards"].items()}
    for p in sorted((base / "branches").glob("*/prices.json")):
        bdoc = load_json(p) or {}
        for n, r in (bdoc.get("cards") or {}).items():
            table.setdefault(_key(n), (r, bdoc.get("as_of")))
    out_c = in_c = 0
    unpriced, dates = [], {doc.get("as_of")}
    for side, sign in (("out", -1), ("in", 1)):
        for n, k in (diff.get(side) or {}).items():
            row, as_of = table.get(_key(n), (None, None))
            cents = (row or {}).get("nm_cents")
            if cents is None:
                unpriced.append(n)
                continue
            dates.add(as_of)
            if sign < 0:
                out_c += cents * k
            else:
                in_c += cents * k
    return {"as_of": doc.get("as_of"), "source": doc.get("source"),
            "dates": sorted(d for d in dates if d), "out_cents": out_c, "in_cents": in_c,
            "delta_cents": in_c - out_c, "unpriced": sorted(unpriced)}


def preview(slug, ops, branch=None, goldfish=True, iterations=None, seed=None):
    """A JSON-able reading of an edit before it is applied — Phase 3's tray.

    THE INSTANT TIER (no simulation): what blocks it, the warnings and the keep
    list hits; size and curve before and after; colour sources against target
    (`mana_fit.shortfall`); combos gained and lost (`deck_combos`); the role
    counts that moved; the price delta from the deck's dated `prices.json`.

    THE GOLDFISH TIER (Commander only, skipped when the edit is refused): the
    paired diagnostic, `net_change.compare_readings` rows with `ci95_diff` and
    Holm, the trust line and the board-quality caveat. Cached in
    `data/cache/try/` on `candidate_key`. A 60-card deck reads
    `{"absent": "not modelled for <format>"}`.
    """
    import time as _time
    from manamap.pilot import deck_edit, diagnostic, formats, mana_fit
    t0 = _time.perf_counter()
    spec = formats.for_deck(slug, branch)
    p = deck_edit.plan(slug, ops, branch=branch)
    out = {"slug": slug, "branch": branch, "format": p["format"], "base_sha": p["base_sha"],
           "ops": p["ops"], "diff": p["diff"], "size": p["size"],
           "blocking": p["blocking"], "warnings": p["warnings"],
           "keep_list_hits": p["keep_list_hits"]}
    before_entries = deck_branch._parsed(slug, branch)
    after_entries = [e for e in p["entries_after"] if e.get("board", "main") == "main"]
    cb, ca = {}, {}
    for e in before_entries:
        cb[e["name"]] = cb.get(e["name"], 0) + int(e.get("quantity") or 1)
    for e in after_entries:
        ca[e["name"]] = ca.get(e["name"], 0) + int(e.get("quantity") or 1)
    facts = _facts_by_key(set(cb) | set(ca))
    if facts:
        out["curve"] = {"before": _curve(cb, facts), "after": _curve(ca, facts)}
        rb, ra = _roles(cb, facts), _roles(ca, facts)
        if rb is not None:
            out["roles"] = {r: ra.get(r, 0) - rb.get(r, 0) for r in sorted(set(rb) | set(ra))
                            if ra.get(r, 0) != rb.get(r, 0)}
        else:
            out["roles"] = {"absent": "no card_roles.json on this machine"}
    else:
        out["curve"] = out["roles"] = {"absent": "no corpus on this machine (cards.csv)"}
    try:
        cmd_b = [e["name"] for e in before_entries if e.get("is_commander")]
        cmd_a = [e["name"] for e in after_entries if e.get("is_commander")]
        kb, ka = _combos(cb, cmd_b, facts), _combos(ca, cmd_a, facts)
        out["combos"] = {"gained": [ka[i] for i in sorted(set(ka) - set(kb))],
                         "lost": [kb[i] for i in sorted(set(kb) - set(ka))],
                         "before": len(kb), "after": len(ka)}
    except Exception as exc:                        # noqa: BLE001 - no combo file, no figure
        out["combos"] = {"absent": f"combos not read: {type(exc).__name__}: {exc}"}
    out["price"] = _price(slug, branch, p["diff"])
    doc = None
    if not p["blocking"]:
        try:
            doc = build_doc(slug, branch, p)
        except (Exception, SystemExit) as exc:      # noqa: BLE001 - reported, not raised
            out["colour_sources"] = {"absent": f"the new list could not be built: {exc}"}
    if doc is not None:
        try:
            before = mana_fit.shortfall(slug, branch)["colours"]
            after = mana_fit.shortfall(slug, branch, deck_doc=doc)["colours"]
            out["colour_sources"] = {c: {"before": before[c]["have"], "after": after[c]["have"],
                                         "target": after[c]["target"],
                                         "short_after": after[c]["short"]}
                                     for c in "WUBRG" if after[c]["target"] or before[c]["have"]
                                     or after[c]["have"]}
        except Exception as exc:                    # noqa: BLE001 - absent, with its reason
            out["colour_sources"] = {"absent": f"mana_fit: {type(exc).__name__}: {exc}"}
    elif "colour_sources" not in out:
        out["colour_sources"] = {"absent": "the edit is refused — nothing to measure"}
    out["timing"] = {"instant_ms": round((_time.perf_counter() - t0) * 1000)}

    if not spec.commanders:
        out["goldfish"] = {"absent": f"not modelled for {spec.name}"}
    elif p["blocking"] or doc is None:
        out["goldfish"] = {"absent": "the edit is refused — nothing to measure"}
    elif not goldfish:
        out["goldfish"] = {"absent": "not asked for (goldfish=False)"}
    else:
        t1 = _time.perf_counter()
        it = iterations or diagnostic.HARNESS["iterations"]
        sd = seed if seed is not None else diagnostic.HARNESS["seed"]
        base_doc = load_deck_cards(slug, branch)
        table, champ_cached, cand_cached = compare(slug, branch, doc, p, it, sd)
        cards, _forge_run = card_rows(slug, branch, base_doc, doc, _rows(p))
        blind = {c["name"] for c in cards if c["state"] != "seen"}
        call, better, worse, trust = verdict(table, blind)
        notes = authored_rate_notes(base_doc, doc)
        out["goldfish"] = {
            "table": table, "call": call, "better": better, "worse": worse,
            "trust": "; ".join([trust] + notes), "caveat": BOARD_QUALITY_CAVEAT,
            "cards": cards,
            "harness": {"iterations": it, "seed": sd, "champion_cached": champ_cached,
                        "candidate_cached": cand_cached},
            "ms": round((_time.perf_counter() - t1) * 1000)}
    return out


def render(r):
    L = []
    w = L.append
    where = r["slug"] + (f"@{r['branch']}" if r["branch"] else "")
    w(f"TRY — {where}: {len(r['swaps'])} swap(s), {r['harness']['iterations']:,} goldfish games per list"
      f"{' (champion cached)' if r['harness']['champion_cached'] else ''}")
    w("")
    w("  THE SWAPS   roles · declared target · what the goldfish sees · Forge AI on record")
    for c in r["cards"]:
        mark = "-" if c["side"] == "out" else "+"
        seen = c["state"].upper() if c["state"] != "seen" else "seen"
        if c["dark"]:
            seen += f" ({', '.join(c['dark'])} off)"
        f = c["forge"]
        forge = (f"AI cast {f.get('cast')}, activated {f.get('activated')}, held castable {f.get('castable_uncast')}"
                 if f else (f"cast-check {c['proof']}" if c["proof"] else "no Forge record"))
        w(f"    {mark} {c['name'][:34]:34} {', '.join(c['roles'])[:44] or 'no role':44} "
          f"{'TARGET' if c['in_target'] else '      '}  {seen}")
        w(f"      {'':34} {forge}")
    if r["forge_run"]:
        w("    (Forge columns are AI BEHAVIOUR from runs on disk — never a reason to cut a card.)")
    w("")
    w("  CHECKS")
    w("    keep list: clear" + (f" ({len(protected.names(r['slug']))} protected)" if protected.names(r["slug"]) else ""))
    for x in r["warnings"]:
        w(f"    warning: {x}")
    m = r["mana"]
    if "error" in m:
        w(f"    colour sources: unavailable ({m['error']})")
    else:
        moved = [f"{c} {v['before']:+d} -> {v['after']:+d}" for c, v in m.items() if v["before"] != v["after"]]
        short = [f"{c} {v['after']:+d}" for c, v in m.items() if v["after"] < 0]
        w("    colour sources vs target: " + (", ".join(moved) if moved else "unchanged")
          + (f"   SHORT after: {', '.join(short)}" if short else ""))
    w("")
    w("  GOLDFISH   champion -> swapped, interval on the difference, Holm across the rows")
    for row in r["table"]:
        ci = row.get("ci95_diff")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        w(f"    {row['measure'][:26]:26} {row['champion']:>8.3f} -> {row['branch']:>8.3f}  "
          f"{row['delta']:+.3f} {cis:22} {row['verdict'].upper() if row['verdict'] != 'noise' else 'no call'}")
    blind = [c for c in r["cards"] if c["state"] != "seen"]
    if blind:
        w("    these rows cannot see: " + "; ".join(
            f"{c['name']} ({c['state']}{': ' + ', '.join(c['dark']) + ' off' if c['dark'] else ''})"
            for c in blind))
    if r["per_swap"]:
        w("")
        w("  EACH SWAP ALONE")
        for s in r["per_swap"]:
            moved = ", ".join(f"{m} {v}" for m, v, _ in s["moved"]) or "nothing the run can resolve"
            w(f"    - {s['out'][:26]:26} + {s['in'][:26]:26} {moved}")
    v = r["verdict"]
    w("")
    w(f"  ==> {v['call'].upper()}"
      + (f": better on {', '.join(v['better'])}" if v["better"] else "")
      + (f"; worse on {', '.join(v['worse'])}" if v["worse"] else ""))
    w(f"      trust: {v['trust']}")
    for x in v["raise"]:
        w(f"      raise it: {x}")
    return "\n".join(L)


def stage(slug, name, swaps, why, base_branch=None):
    """Write the swaps to branch `name` — only ever called after the screen."""
    root = deck_branch.branch_root(slug) / name
    if not root.exists():
        text = (deck_dir(slug, base_branch) / "decklist.txt").read_text(encoding="utf-8")
        deck_branch.new(slug, name, text, why=why)
    for o, i in swaps:
        deck_branch.stage(slug, name, o, i, why=why)
    return root


def main(args):
    from manamap.pilot import deck_edit
    outs, ins = list(args.out or []), list(getattr(args, "in_") or [])
    adds, cuts = list(getattr(args, "add", None) or []), list(getattr(args, "cut", None) or [])
    sets = list(getattr(args, "set", None) or [])
    if len(outs) != len(ins) or not (outs or adds or cuts or sets):
        raise SystemExit("give the swaps as pairs: --out A --in B [--out C --in D ...], "
                         "and/or --add N, --cut N, --set NAME=COPIES")
    swaps = list(zip(outs, ins))
    ops = deck_edit.ops_from_args(add=adds, cut=cuts, set_=sets,
                                  side=getattr(args, "side", False))
    name = getattr(args, "stage", None)
    if name and ops:
        raise SystemExit("--stage writes SWAPS to a branch; an --add, --cut or --set changes "
                         "the list's size or counts — open a branch with the whole list "
                         "(`deck-branch new`) instead")
    from manamap import console
    n = len(swaps) + len(ops)
    with console.task(f"Trying {n} change(s) on {args.slug}", total=1, unit="run") as bar:
        r = run(args.slug, swaps, branch=getattr(args, "branch", None),
                iterations=getattr(args, "iterations", None), seed=getattr(args, "seed", None),
                each=getattr(args, "each", False), ops=ops)
        bar.advance(1)
    if getattr(args, "json", False):
        print(json.dumps(r, indent=1, default=str))
    else:
        print(render(r))
    if name:
        line = f"try: {r['verdict']['call']} — {r['verdict']['trust']}"
        stage(args.slug, name, swaps, line, base_branch=getattr(args, "branch", None))
        print(f"\nStaged {len(swaps)} swap(s) on {args.slug}/{name}. "
              f"`manamap pilot deck-branch {args.slug} commit {name} -m \"…\"` when it settles.")
