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


def count(text, card, ours_suffix, cmc):
    """What happened to `card` in our seat across the games in one log text."""
    games = parse.parse_games(text)
    out = {"games": len(games), "drawn_games": 0, "drawn": 0, "cast": 0, "activated": 0, "discarded": 0,
           "castable_uncast_turns": 0, "held_at_end_games": 0, "held_castable_games": 0, "timeouts": text.count("AI eval thread at timeout")}
    stem = re.escape(card)
    out["cast"] = len(re.findall(ours_suffix.replace("(", r"\(").replace(")", r"\)") + r" cast " + stem + r"\b", text))
    out["activated"] = len(re.findall(ours_suffix.replace("(", r"\(").replace(")", r"\)") + r" activated " + stem + r"\b", text))
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


def verdict(c):
    plays = c["cast"] + c["activated"]
    if not c["drawn_games"]:
        return "NOT DRAWN — the shell never put it in hand; raise --copies or --games"
    if plays == 0 and c["held_castable_games"]:
        return (f"HELD: in hand in {c['drawn_games']} game(s), castable and uncast on {c['castable_uncast_turns']} own turn(s) "
                f"across {c['held_castable_games']} game(s), cast 0 — the AI will not play it; a hint or a patch BEFORE a branch")
    if plays == 0:
        return f"UNPLAYED and never castable in {c['drawn_games']} drawn game(s) — the shell did not give it the mana; inconclusive"
    return (f"PLAYED: cast {c['cast']} / activated {c['activated']} across {c['drawn_games']} drawn game(s); "
            f"held castable and uncast on {c['castable_uncast_turns']} own turn(s)")


def run(slug, card, copies=DEFAULT_COPIES, games=DEFAULT_GAMES, vs=DEFAULT_VS, clock=DEFAULT_CLOCK,
        branch=None, seed=4343, profile=None):
    text, facts = shell_decklist(slug, card, copies, branch)
    meta = forge.install_named(f"mm-castcheck-{forge.deck_meta_name(slug)}", text)
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
    return {"slug": slug, "branch": branch, "card": card, "as_of": date.today().isoformat(), "shell": facts,
            "vs": vs, "games": games, "clock": clock, "seed": seed, "jar": pathlib.Path(jar).name,
            "telemetry": (fp or {}).get("sha"), "profiles": profiles, "counts": c, "verdict": verdict(c),
            "limits": ["A shell, not the deck: the card's VALUE is not measured here, only whether the AI plays it.",
                       "Castable is a lands-only floor (colours ignored), the same floor engine_casts carries.",
                       "Two seats, short clock: a card that needs a four-player board or a long game can read HELD here and play at the table."]}, log


def main(args):
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
    out = getattr(args, "out", None)
    if out:
        from manamap.pilot.common import resolve_out_path
        p = resolve_out_path(out, args.slug, f"cast-check-{forge.deck_meta_name(args.slug)}")
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (p.with_suffix(".log")).write_text(log, encoding="utf-8")
        print(f"  wrote {p} (+ .log)")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot forge-cast-check <slug> --card NAME`.")
