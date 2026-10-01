"""`forge-cast-check` — PROVE that Forge's AI will cast (or activate) a card before a night is
spent measuring a branch that depends on it.

Built on 2026-10-01 after the drain-v1 branch arm: Toxic Deluge, unflagged and staged as the
deck's one sweeper, was drawn 28 times in 200 games and cast 0. The scan had carried the
Forge AI flag and "unflagged" was read as "castable", which it is not — `AI:RemoveDeck:All`
is a filter the AI applies to every ability of a card, but each API has its own AI class,
and that class can refuse the card for its own reasons: X priced before its cost is paid
(Vish Kal's -X/-X, Deluge's pay-X-life), a logic class with no branch for the shape (Altar
of Dementia, Teferi's Protection, Deflecting Swat), or a missing `IsCurse$` that makes a
-X/-X read as a pump of our own creatures (Vish Kal, and Deluge again). Every one of those
was found AFTER games had been spent on it.

THE SHELL. Our seat is the deck's own commander over `copies` of the card, the deck's own
cheap spells as filler (so the card has a board to interact with — a sweeper needs
creatures, an outlet needs fodder) and basics of the card's colours; the opponent is one
named seat (a pod seat or a deck). N short two-seat games under whatever jar `simulate`
would use, counted with the telemetry patch's hand facts: drawn, cast, activated, held
while castable, discarded. The verdict is a rate and a sentence — "cast 7 of 9 castable
games" or "HELD: castable in 11 games, cast 0" — never a yes/no, and the record of the
shell is a view (`--out`), because the branch's own record is where the claim is made.

It is not a measurement of the card's VALUE. It answers one question only: will the pilot
play it. A card that passes here can still be bad; a card that fails here is not yet in the
deck's hands at all, and measuring a list with it in is measuring a different list.
"""
import json
import pathlib
import re
import subprocess
from datetime import date

from manamap import config
from manamap.sim import forge, parse, telemetry

DEFAULT_COPIES = 4
DEFAULT_GAMES = 8
DEFAULT_CLOCK = 300
DEFAULT_VS = "giada-angels"
SHELL_LANDS = 36
BASIC = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}


def _deck(slug, branch=None):
    from manamap.pilot.common import load_deck_cards
    return load_deck_cards(slug, branch)


def shell_decklist(slug, card, copies=DEFAULT_COPIES, branch=None):
    """The shell's decklist text: the deck's commander, `copies` of the card, the deck's own
    cheapest nonland spells as filler, basics of the identity to 100. Returns (text, facts)."""
    from manamap.pilot.card_pool import load_pool
    doc = _deck(slug, branch)
    cards = doc.get("cards") or []
    cmdr = [c for c in cards if c.get("is_commander")]
    if not cmdr:
        raise SystemExit(f"{slug}: cards.json names no commander")
    pool = load_pool() or {}
    rec = pool.get(card) or pool.get(card.split(" // ")[0])
    if rec is None:
        raise SystemExit(f"{card!r} is not in the corpus — check the name")
    ident = {c for x in cards for c in (x.get("color_identity") or [])}
    if not rec["color_identity"] <= ident:
        raise SystemExit(f"{card} is outside {slug}'s identity {sorted(ident)}")
    filler = sorted((c for c in cards if not c.get("is_commander") and "Land" not in (c.get("type_line") or "")
                     and c["name"] != card and c["name"].split(" // ")[0] != card),
                    key=lambda c: (float(c.get("cmc") or 0), c["name"]))
    n_filler = 99 - copies - SHELL_LANDS
    filler = [c["name"] for c in filler[:n_filler]]
    colours = sorted(rec["color_identity"] or ident) or sorted(ident)
    basics = []
    for i in range(SHELL_LANDS):
        basics.append(BASIC[colours[i % len(colours)]])
    counts = {}
    for b in basics:
        counts[b] = counts.get(b, 0) + 1
    lines = ["Commander:", f"1 {cmdr[0]['name']}", "", "Deck:", f"{copies} {card}"]
    lines += [f"1 {n}" for n in filler]
    lines += [f"{n} {b}" for b, n in sorted(counts.items())]
    facts = {"commander": cmdr[0]["name"], "copies": copies, "filler": len(filler), "lands": SHELL_LANDS,
             "basics": counts, "cmc": float(rec["cmc"] or 0), "identity": sorted(ident)}
    return "\n".join(lines) + "\n", facts


#: THE FOUR CLASSES of "unflagged but never played", as predicates over the card's
#: Forge script — one home, each with its remedy. Every class here was found AFTER a night
#: of games before this existed (Vish Kal, Toxic Deluge, Altar of Dementia, Teferi's
#: Protection, Deflecting Swat, Bastion of Remembrance). A card with no matching class is
#: reported `unknown`, never guessed at: the remedy then is to read the API's AI class.
_ABILITY = re.compile(r"^A:(?P<prefix>(?:SP|AB)\$ (?P<api>\w+))(?P<rest>.*)$", re.M)
_SVAR_X = re.compile(r"^SVar:X:(.+)$", re.M)
_TYPES = re.compile(r"^Types:(.+)$", re.M)
_COST_RE = re.compile(r"\bCost\$ ([^|]+)")
#: APIs whose AI class refuses a shape unless a known `AILogic$` names it — the three the
#: patch set added logic for, keyed by the signature of the shape they refuse.
_NEEDS_LOGIC = {
    "Mill": ("Sac<", "SacOutlet (data/forge_patches/MillAi.java): a sacrifice-cost mill ability fires when a creature of ours is about to die anyway"),
    "Effect": (None, "Protection (data/forge_patches/EffectAi.java): cast against a wipe, removal on the commander, a lethal spell or lethal combat"),
    "ChangeTargets": (None, "Deflect (data/forge_patches/ChangeTargetsAi.java): redirect an opponent's targeted spell away from us"),
}


#: The AI classes the patch set has taught to price X from a cost, by API. Read against
#: the INSTALLED jar's registered classes, so a jar without the patch reports the defect.
_X_PATCH = {"Pump": "PumpAi.java"}


def _patched_api(api):
    src = _X_PATCH.get(api)
    if not src:
        return None
    try:
        fp = telemetry.installed()
    except Exception:                               # noqa: BLE001 - no jar, no patch
        return None
    return src if fp and any(c.get("source") == src for c in (fp.get("classes") or [])) else None


def diagnose(card, installed_copy=True):
    """What the card's Forge script says about WHY the AI would hold it: the API, the cost,
    the X SVar, IsCurse$, AILogic$, the RemoveDeck flag, the classes that match and the
    remedy each one has. Reads the INSTALLED script by default (overrides and hints
    included) — a hint already applied is not a defect."""
    from manamap.sim import forge_cards
    text = forge_cards.script(card, installed_copy=installed_copy)
    if text is None:
        return {"script": None, "classes": ["no-script"], "remedies": ["Forge has no script for this card — it cannot be played at all"]}
    abilities = [(m.group("prefix"), m.group("api"), m.group("rest")) for m in _ABILITY.finditer(text)]
    x = (_SVAR_X.search(text).group(1).strip() if _SVAR_X.search(text) else None)
    types = (_TYPES.search(text).group(1) if _TYPES.search(text) else "")
    flag = (re.search(r"^AI:RemoveDeck:(\w+)", text, re.M) or [None, None])[1] if re.search(r"^AI:RemoveDeck:", text, re.M) else None
    classes, remedies, notes = [], [], []
    if flag == "All":
        classes.append("removedeck-all")
        remedies.append("add its stem to data/forge_overrides/unflag.txt and `forge-install --generate`")
    for prefix, api, rest in abilities:
        cost = (_COST_RE.search(rest).group(1).strip() if _COST_RE.search(rest) else "")
        negative = bool(re.search(r"Num(Att|Def)\$ -", rest))
        x_from_cost = x is not None and ("CostCountersRemoved" in x or ("xPaid" in x and ("PayLife<X>" in cost or "SubCounter" in cost)))
        if re.search(r"Num(Att|Def)\$ [+-]?X", rest) and x_from_cost:
            # A JAVA PATCH IS NOT VISIBLE IN THE SCRIPT: PumpAi prices X from the cost since
            # 2026-09-30, so under a jar that carries that class the shape is handled.
            patched = _patched_api(api)
            if patched:
                notes.append(f"X priced from the cost by the {patched} patch in the installed jar")
            else:
                classes.append("x-priced-before-cost")
                remedies.append(f"an `ai` patch on {api}Ai pricing X from the cost before it is paid "
                                f"(PumpAi done 2026-09-30; {api}Ai next)")
        if negative and "IsCurse$" not in rest:
            classes.append("no-iscurse")
            remedies.append(f"ability_params {{\"IsCurse\": \"True\"}} on `{prefix}` (forge_hints.json)")
        need = _NEEDS_LOGIC.get(api)
        if need and "AILogic$" not in rest and (need[0] is None or need[0] in cost):
            classes.append("no-ai-logic")
            remedies.append(f"ai_logic hint on `{prefix}`: {need[1]}")
    if not abilities and re.search(r"\b(Enchantment|Artifact)\b", types) and "Creature" not in types:
        # a cheap do-nothing-now permanent: the AI casts it after every creature in hand
        classes.append("permanent-cast-priority")
        remedies.append("NO HINT EXISTS (read 2026-10-01): `AICastPreference` is a set of DON'T-CAST conditions only "
                        "(MustHaveInHand, MaxControlled[Globally|WithoutOppAuras], NumManaSources[NextTurn], "
                        "Never/AlwaysCastIfLife{Below,Above}, OnlyFromZone — 9 cards in the corpus use it), so it can "
                        "delay a cast and never hasten one. Either patch the AI's spell ORDERING (AiController, every "
                        "deck affected — not done) or accept the lateness and judge the card knowing it")
    return {"script": {"abilities": [p for p, _, _ in abilities], "apis": sorted({a for _, a, _ in abilities}),
                       "x": x, "types": types.strip(), "ai_flag": flag,
                       "is_curse": any("IsCurse$" in r for _, _, r in abilities) if abilities else None,
                       "ai_logic": [m.group(1) for m in re.finditer(r"AILogic\$ (\w+)", text)]},
            "classes": classes or ["unknown"], "notes": notes,
            "remedies": remedies or ["no known class matches — read the API's AI class (forge-ai/.../<Api>Ai.java)"]}


def harness():
    """The tuple a proof is valid under — the same axes `net_change.forge` buckets on."""
    from manamap.sim import forge_pilot as _fpl
    ov = (forge.card_overrides() or {}).get("sha")
    fp = telemetry.installed()
    return {"overrides": ov, "patches": (fp or {}).get("sha"), "forge": forge.forge_version(),
            "jar": pathlib.Path(telemetry.jar_for_run()[0]).name}


def same_harness(a, b):
    return bool(a) and bool(b) and all(a.get(k) == b.get(k) for k in ("overrides", "patches", "profile", "profile_sha"))


def count(text, card, ours_suffix, cmc):
    """What happened to `card` in our seat across the games in one log text."""
    games = parse.parse_games(text)
    out = {"games": len(games), "drawn_games": 0, "drawn": 0, "cast": 0, "activated": 0, "triggered": 0, "discarded": 0,
           "castable_uncast_turns": 0, "held_at_end_games": 0, "held_castable_games": 0, "timeouts": text.count("AI eval thread at timeout")}
    stem = re.escape(card)
    ours = re.escape(ours_suffix)
    out["cast"] = len(re.findall(ours + r" cast " + stem + r"\b", text))
    out["activated"] = len(re.findall(ours + r" activated " + stem + r"\b", text))
    out["triggered"] = len(re.findall(ours + r" triggered " + stem + r"\b", text))
    for g in games:
        seat = next((s for s in g["seats"] if s.endswith(ours_suffix)), None)
        if not seat:
            continue
        drawn = sum(1 for ev in g["events"] if ev.get("kind") == "zone" and ev.get("owner") == seat
                    and ev.get("card") == card and ev.get("to") == "Hand" and ev.get("from") == "Library")
        if drawn:
            out["drawn_games"] += 1
            out["drawn"] += drawn
        out["discarded"] += sum(1 for ev in g["events"] if ev.get("kind") == "zone" and ev.get("owner") == seat
                                and ev.get("card") == card and ev.get("from") == "Hand" and ev.get("to") == "Graveyard")
        hf = parse.hand_facts(g["events"], g["seats"], cmc={card: cmc}).get(seat) or {}
        row = (hf.get("cards") or {}).get(card) or {}
        cu = row.get("castable_uncast", 0)
        out["castable_uncast_turns"] += cu
        if cu:
            out["held_castable_games"] += 1
        if any(c["card"] == card for c in hf.get("hand_at_end", [])):
            out["held_at_end_games"] += 1
    return out


VERDICTS = ("PLAYED", "CAST-LATE", "HELD", "UNPLAYED", "NOT DRAWN")


def verdict(c):
    """One word a gate can read, then the sentence. CAST-LATE is Bastion's shape: played in
    fewer than half the games it was drawn while castable and uncast on at least two
    own turns per drawn game — not dead, but a floor on whatever it is for."""
    plays = c["cast"] + c["activated"]
    if not c["drawn_games"]:
        return "NOT DRAWN — the shell never put it in hand; raise --copies or --games"
    if plays == 0 and c["held_castable_games"]:
        return (f"HELD: in hand in {c['drawn_games']} game(s), castable and uncast on {c['castable_uncast_turns']} own turn(s) "
                f"across {c['held_castable_games']} game(s), cast 0 — the AI will not play it; a hint or a patch BEFORE a branch")
    if plays == 0:
        return f"UNPLAYED: never castable in {c['drawn_games']} drawn game(s) — the shell did not give it the mana; inconclusive"
    if plays * 2 < c["drawn_games"] and c["castable_uncast_turns"] >= 2 * c["drawn_games"]:
        return (f"CAST-LATE: cast {c['cast']} / activated {c['activated']} in {c['drawn_games']} drawn game(s) while castable and "
                f"uncast on {c['castable_uncast_turns']} own turn(s) — the AI plays it behind everything else; a FLOOR")
    return (f"PLAYED: cast {c['cast']} / activated {c['activated']} / triggered {c.get('triggered', 0)} across {c['drawn_games']} "
            f"drawn game(s); held castable and uncast on {c['castable_uncast_turns']} own turn(s)")


def verdict_word(c):
    return verdict(c).split(":")[0].split(" — ")[0].strip()


def run(slug, card, copies=DEFAULT_COPIES, games=DEFAULT_GAMES, vs=DEFAULT_VS, clock=DEFAULT_CLOCK,
        branch=None, seed=4343, profile=None):
    from manamap.sim import forge_cards
    text, facts = shell_decklist(slug, card, copies, branch)
    meta = forge.install_named(f"mm-castcheck-{forge.deck_meta_name(slug)}-{forge_cards.stem(card)[:24]}", text)
    opp = forge.install_deck(vs)
    jar, fp = telemetry.jar_for_run()
    profiles = None
    if profile is None:
        # THE DECK'S OWN PILOTING, the way `simulate` resolves it: a declared profile the
        # engine carries, else Forge's default — a cast check under a different pilot than
        # the branch arm would answer a different question.
        from manamap.sim import forge_pilot as _fpl
        _, fp_ = _fpl.declared_profile(slug, branch)
        if fp_ and _fpl.profile_agrees(slug, branch):
            profile = _fpl.profile_name(slug)
    if profile:
        profiles = [profile, forge.STANDARD_POD_PROFILE]
    argv = forge.command([meta, opp], games, clock, jar=jar, seed=seed, profiles=profiles)
    out = subprocess.run(argv, capture_output=True, text=True, cwd=str(forge.FORGE_HOME))
    log = out.stdout + out.stderr
    c = count(log, card, meta, facts["cmc"])
    from manamap.sim import forge_pilot as _fpl
    stamp = dict(harness(), profile=profile, profile_sha=_fpl.profile_content_sha(profile) if profile else None)
    return {"slug": slug, "branch": branch, "card": card, "as_of": date.today().isoformat(), "shell": facts,
            "vs": vs, "games": games, "clock": clock, "seed": seed, "jar": pathlib.Path(jar).name,
            "telemetry": (fp or {}).get("sha"), "profiles": profiles, "harness": stamp,
            "counts": c, "verdict": verdict(c), "verdict_word": verdict_word(c), "diagnosis": diagnose(card),
            "limits": ["A shell, not the deck: the card's VALUE is not measured here, only whether the AI plays it.",
                       "Castable is a lands-only floor (colours ignored), the same floor engine_casts carries.",
                       "Two seats, short clock: a card that needs a four-player board or a long game can read HELD here and play at the table."]}, log


PROOFS = "cast_proofs.json"


def adds(slug, branch):
    """Every card the branch ADDS: new names plus a name whose copy count rose."""
    from manamap.pilot import deck_branch
    d = deck_branch.diff(slug, branch)
    names = list(d.get("add") or [])
    names += [q["name"] for q in (d.get("quantity") or []) if (q.get("to") or 0) > (q.get("from") or 0)]
    return sorted(set(names))


def proofs_path(slug, branch):
    from manamap.pilot.common import deck_dir
    return deck_dir(slug, branch) / PROOFS


def read_proofs(slug, branch):
    from manamap.pilot.common import load_json
    try:
        return load_json(proofs_path(slug, branch))
    except FileNotFoundError:
        return None


def status(slug, branch, current):
    """What the gate reads: every add's standing under the harness `current` — PLAYED /
    CAST-LATE / HELD / UNPLAYED / NOT DRAWN / unproven (no row, or a row under another
    harness). Absent file -> every add unproven."""
    doc = read_proofs(slug, branch) or {}
    rows = doc.get("cards") or {}
    fresh = same_harness(doc.get("harness"), current)
    out = {"proven": [], "held": [], "late": [], "unproven": [], "as_of": doc.get("as_of"),
           "harness_matches": fresh if doc else None}
    for card in adds(slug, branch):
        r = rows.get(card)
        w = (r or {}).get("verdict_word")
        if not r or not fresh:
            out["unproven"].append(card)
        elif w == "PLAYED":
            out["proven"].append(card)
        elif w == "CAST-LATE":
            out["late"].append(card)
        else:
            out["held"].append(card)
    return out


def current_harness(slug, branch=None, profile=None):
    """The tuple `simulate` is about to stamp on a run of this seat — overrides, pilot
    profile (declared, or given), patch set — resolved the way `forge.run` resolves it."""
    from manamap.sim import forge_pilot as _fpl
    cur = dict(harness())
    if profile is None:
        _, fp_ = _fpl.declared_profile(slug, branch)
        if fp_ and _fpl.profile_agrees(slug, branch):
            profile = _fpl.profile_name(slug)
    cur.update(profile=profile, profile_sha=_fpl.profile_content_sha(profile) if profile else None)
    return cur


def gate(slug, branch, profile=None, anyway=False):
    """THE REFUSAL, before a branch arm: every add must be PLAYED under this harness.

    Prints the CAST PROOFS line; raises SystemExit naming the held or unproven adds and the
    command that fixes it, unless `anyway` — then the arm runs and the record says which
    slots are FLOORS. Returns the status dict the record carries."""
    cur = current_harness(slug, branch, profile)
    s = status(slug, branch, cur)
    s["harness"] = cur
    s["anyway"] = bool(anyway)
    n = len(s["proven"]) + len(s["held"]) + len(s["late"]) + len(s["unproven"])
    if n == 0:
        print("  CAST PROOFS   the branch adds nothing against the deck — nothing to prove")
        return s
    if not s["held"] and not s["late"] and not s["unproven"]:
        print(f"  CAST PROOFS   {len(s['proven'])}/{n} adds PLAYED under this harness ({s['as_of']})")
        return s
    lines = []
    if s["unproven"]:
        why = ("no cast_proofs.json" if s["harness_matches"] is None else
               "proofs taken under ANOTHER harness" if s["harness_matches"] is False else "no row")
        lines.append(f"{len(s['unproven'])} add(s) UNPROVEN under this harness ({why}): {', '.join(s['unproven'])}")
    doc = read_proofs(slug, branch) or {}
    for c in s["held"]:
        r = (doc.get("cards") or {}).get(c) or {}
        lines.append(f"{c} is HELD (drawn in {r.get('drawn_games', '?')} shell game(s), cast {r.get('cast', '?')}) — "
                     + "; ".join(r.get("remedies") or ["read the API's AI class"]))
    for c in s["late"]:
        r = (doc.get("cards") or {}).get(c) or {}
        lines.append(f"{c} is CAST-LATE (cast {r.get('cast', '?')} in {r.get('drawn_games', '?')} drawn game(s)) — "
                     + "; ".join(r.get("remedies") or ["a cast-priority hint"]))
    head = f"  CAST PROOFS   {len(s['proven'])}/{n} adds PLAYED under this harness"
    if anyway:
        print(head + " — RUNNING ANYWAY; the slots below are FLOORS on this arm:")
        for ln in lines:
            print(f"    · {ln}")
        return s
    print(head)
    raise SystemExit("REFUSED — a Forge arm on a list whose adds the AI does not play measures a different list:\n  · "
                     + "\n  · ".join(lines)
                     + f"\n  fix the card (a hint or a patch), drop it, or prove it: "
                       f"manamap pilot forge-cast-check {slug} --branch {branch} --adds --write"
                     + "\n  --anyway runs the arm with those slots recorded as FLOORS")


def run_many(slug, cards, branch=None, jobs=2, force=False, **kw):
    """One shell per card, `jobs` at a time; a card already PLAYED under the same harness
    in the existing proof file is kept rather than re-run unless `force`."""
    from concurrent.futures import ThreadPoolExecutor
    existing = read_proofs(slug, branch) if branch else None
    current = current_harness(slug, branch)
    kept, todo = {}, []
    if existing and same_harness(existing.get("harness"), current) and not force:
        for c in cards:
            r = (existing.get("cards") or {}).get(c)
            if r and r.get("verdict_word") == "PLAYED":
                kept[c] = r
                continue
            todo.append(c)
    else:
        todo = list(cards)
    docs = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for card, (doc, _log) in zip(todo, ex.map(lambda c: run(slug, c, branch=branch, **kw), todo)):
            docs[card] = doc
    return kept, docs, current


def write_proofs(slug, branch, kept, docs, current, shell=None):
    rows = dict(kept)
    for card, d in docs.items():
        rows[card] = {"as_of": d["as_of"], **d["counts"], "verdict": d["verdict"], "verdict_word": d["verdict_word"],
                      "classes": d["diagnosis"]["classes"], "remedies": d["diagnosis"]["remedies"],
                      "script": d["diagnosis"].get("script"), "shell": d["shell"], "vs": d["vs"], "games": d["games"]}
    out = {"slug": slug, "branch": branch, "as_of": date.today().isoformat(), "harness": current,
           "shell": shell or {"vs": DEFAULT_VS, "games": DEFAULT_GAMES, "copies": DEFAULT_COPIES, "clock": DEFAULT_CLOCK, "seed": 4343},
           "cards": dict(sorted(rows.items())),
           "limits": ["A shell, not the deck: the card's VALUE is not measured here, only whether the AI plays it.",
                      "Castable is a lands-only floor (colours ignored), the same floor engine_casts carries.",
                      "Two seats, short clock: a card that needs a four-player board or a long game can read HELD here and play at the table.",
                      "A proof is a measurement under one harness (overrides, profile, patch set); a changed harness voids it, like model_version."]}
    p = proofs_path(slug, branch)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p, out


def render_row(card, r):
    return (f"  {card:28s} {r['verdict_word']:9s} drawn {r['drawn_games']:2d}g  cast {r['cast']:2d}  act {r['activated']:2d}  "
            f"trig {r.get('triggered', 0):2d}  castable-uncast {r['castable_uncast_turns']:3d}t  "
            + (f"[{', '.join(r['classes'])}]" if r.get("classes") and r["classes"] != ["unknown"] else ""))


def main(args):
    if getattr(args, "adds", False):
        branch = getattr(args, "branch", None)
        if not branch:
            raise SystemExit("--adds proves a BRANCH's adds — `--branch <name>`")
        cards = adds(args.slug, branch)
        if not cards:
            raise SystemExit(f"{args.slug}@{branch} adds nothing against the deck — nothing to prove")
        kw = dict(copies=getattr(args, "copies", None) or DEFAULT_COPIES, games=getattr(args, "games", None) or DEFAULT_GAMES,
                  vs=getattr(args, "vs", None) or DEFAULT_VS, clock=getattr(args, "clock", None) or DEFAULT_CLOCK,
                  seed=getattr(args, "seed", None) or 4343)
        kept, docs, current = run_many(args.slug, cards, branch=branch, jobs=getattr(args, "jobs", None) or 2,
                                       force=getattr(args, "force", False), **kw)
        shell = {"vs": kw["vs"], "games": kw["games"], "copies": kw["copies"], "clock": kw["clock"], "seed": kw["seed"]}
        print(f"CAST PROOFS — {args.slug}@{branch}: {len(cards)} add(s), {len(docs)} shell(s) run, {len(kept)} kept from the file")
        rows = {**kept, **{c: {**d["counts"], "verdict_word": d["verdict_word"], "classes": d["diagnosis"]["classes"]} for c, d in docs.items()}}
        for c in cards:
            print(render_row(c, rows[c]))
        for c, d in docs.items():
            if d["verdict_word"] in ("HELD", "CAST-LATE"):
                for rem in d["diagnosis"]["remedies"]:
                    print(f"      -> {c}: {rem}")
        if getattr(args, "write", False):
            p, _ = write_proofs(args.slug, branch, kept, docs, current, shell)
            print(f"  wrote {p}")
        return
    if not getattr(args, "card", None):
        raise SystemExit("--card NAME, or --adds with --branch")
    doc, log = run(args.slug, args.card, copies=getattr(args, "copies", None) or DEFAULT_COPIES,
                   games=getattr(args, "games", None) or DEFAULT_GAMES, vs=getattr(args, "vs", None) or DEFAULT_VS,
                   clock=getattr(args, "clock", None) or DEFAULT_CLOCK, branch=getattr(args, "branch", None),
                   seed=getattr(args, "seed", None) or 4343)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        c = doc["counts"]
        print(f"CAST CHECK — {doc['card']} for {doc['slug']}" + (f"@{doc['branch']}" if doc["branch"] else "")
              + f" · {doc['shell']['copies']} copies, {doc['games']} games vs {doc['vs']}, {doc['jar']}")
        print(f"  drawn in {c['drawn_games']} game(s) ({c['drawn']} draws) · cast {c['cast']} · activated {c['activated']} · "
              f"discarded {c['discarded']} · castable-uncast turns {c['castable_uncast_turns']} · held at end {c['held_at_end_games']} · AI timeouts {c['timeouts']}")
        print(f"  {doc['verdict']}")
        if doc["verdict_word"] in ("HELD", "CAST-LATE", "UNPLAYED"):
            print(f"  script: {doc['diagnosis']['classes']}")
            for rem in doc["diagnosis"]["remedies"]:
                print(f"    -> {rem}")
    out = getattr(args, "out", None)
    if out:
        from manamap.pilot.common import resolve_out_path
        p = resolve_out_path(out, args.slug, f"cast-check-{forge.deck_meta_name(args.slug)}")
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (p.with_suffix(".log")).write_text(log, encoding="utf-8")
        print(f"  wrote {p} (+ .log)")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot forge-cast-check <slug> --card NAME`.")
