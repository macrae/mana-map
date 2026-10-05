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


def apply_swaps(slug, branch, swaps):
    """`(doc, entries, rows)` — the cards doc and decklist entries with every swap
    applied, through the SAME arithmetic and refusals `stage` uses. Raises
    SystemExit on anything `stage` would refuse, and on a protected cut."""
    protected.refuse(slug, [o for o, _ in swaps], "that swap")
    _scryfall_objects([i for _, i in swaps])     # one pass over the dump for every IN card
    entries = deck_branch._parsed(slug, branch)
    base = load_deck_cards(slug, branch)
    cards = [dict(c) for c in (base["cards"] if isinstance(base, dict) else base)]
    rows = []
    for out_name, in_name in swaps:
        entries, out_e, _in_e = deck_branch.swap_entries(slug, branch, entries, out_name, in_name)
        rec = corpus_card(in_name)
        if rec is None:
            raise SystemExit(f"{in_name!r} is not in the corpus — check the spelling, "
                             f"or `manamap pilot card-search --name {in_name!r}`.")
        # One COPY of the OUT card leaves cards.json (basics carry a quantity).
        for i, c in enumerate(cards):
            if _key(c["name"]) == _key(out_e["name"]):
                left = int(c.get("quantity") or 1) - 1
                if left > 0:
                    cards[i] = dict(c, quantity=left)
                else:
                    cards.pop(i)
                break
        hit = next((i for i, c in enumerate(cards) if _key(c["name"]) == _key(rec["name"])), None)
        if hit is not None:      # a basic already present: one more copy
            cards[hit] = dict(cards[hit], quantity=int(cards[hit].get("quantity") or 1) + 1)
        else:
            cards.append(rec)
        rows.append({"out": out_e["name"], "in": rec["name"]})
    checked = check_in.analyze(slug, check_in.render_decklist(entries))
    if checked["blocking"]:
        raise SystemExit("Refusing that swap set:\n  - " + "\n  - ".join(checked["blocking"]))
    # DECKLIST ORDER, as `fetch-deck` writes it. The goldfish shuffles the list it
    # is given, so on one seed a card appended at the end plays different games
    # than the same card in its decklist slot — and `try` would disagree with
    # `net-change` on the identical swaps for no reason but the order.
    order = {_key(e["name"]): i for i, e in enumerate(checked["entries"])}
    cards.sort(key=lambda c: (not c.get("is_commander"), order.get(_key(c["name"]), len(order))))
    doc = dict(base) if isinstance(base, dict) else {"cards": base}
    doc["cards"] = cards
    return doc, entries, rows, checked.get("warnings") or []


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
    got = diagnostic.run(slug, branch=branch, iterations=iterations, seed=seed, quiet=True)
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
            name = s[side]
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


def run(slug, swaps, branch=None, iterations=None, seed=None, each=False):
    from manamap.pilot import diagnostic, mana_fit, net_change
    it = iterations or diagnostic.HARNESS["iterations"]
    sd = seed if seed is not None else diagnostic.HARNESS["seed"]
    doc, entries, rows, warnings = apply_swaps(slug, branch, swaps)
    base_doc = load_deck_cards(slug, branch)
    a, cached = champion_reading(slug, branch, it, sd)
    b = diagnostic.run_on(doc, slug, branch=branch, iterations=it, seed=sd, quiet=True)
    table = net_change.compare_readings(a, b)
    cards, forge_run = card_rows(slug, branch, base_doc, doc, rows)
    blind = {c["name"] for c in cards if c["state"] != "seen"}
    call, better, worse, trust = verdict(table, blind)
    per_swap = []
    if each and len(rows) > 1:
        for (o, i) in swaps:
            d1, _, _, _ = apply_swaps(slug, branch, [(o, i)])
            r1 = diagnostic.run_on(d1, slug, branch=branch, iterations=it, seed=sd, quiet=True)
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
    outs, ins = list(args.out or []), list(getattr(args, "in_") or [])
    if not outs or len(outs) != len(ins):
        raise SystemExit("give the swaps as pairs: --out A --in B [--out C --in D ...]")
    swaps = list(zip(outs, ins))
    from manamap import console
    with console.task(f"Trying {len(swaps)} swap(s) on {args.slug}", total=1, unit="run") as bar:
        r = run(args.slug, swaps, branch=getattr(args, "branch", None),
                iterations=getattr(args, "iterations", None), seed=getattr(args, "seed", None),
                each=getattr(args, "each", False))
        bar.advance(1)
    if getattr(args, "json", False):
        print(json.dumps(r, indent=1, default=str))
    else:
        print(render(r))
    name = getattr(args, "stage", None)
    if name:
        line = f"try: {r['verdict']['call']} — {r['verdict']['trust']}"
        stage(args.slug, name, swaps, line, base_branch=getattr(args, "branch", None))
        print(f"\nStaged {len(swaps)} swap(s) on {args.slug}/{name}. "
              f"`manamap pilot deck-branch {args.slug} commit {name} -m \"…\"` when it settles.")
