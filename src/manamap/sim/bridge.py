"""Simulation S4: the bridge — one game at one moment, lifted into a `game_state` v2
scenario the resolve loop can be handed.

A Forge run is a distribution; the citation contract works on ONE board. This module
connects them: it replays a game's events up to a cut — a global turn and a step, in
the Comprehensive Rules' own names — and reconstructs every seat as far as the log
allows, writing a v2 scenario (docs/pilot.md → Game state v2) whose `question` is empty
on purpose. The pilot poses the question; the resolver answers it; the checker audits
it. The ✓ tier on what the sample surfaced.

WHAT THE LOG CAN AND CANNOT GIVE, stated in the artifact. Forge's sim log is a stream of
events, not board snapshots, so reconstruction is inference over what was printed:

  life              exact — every change is a `Life:` line
  lands             exact — `Land:` lines carry name and id; tapped = a `Mana:` line since
                    the controller's last untap step
  cast permanents   good — a `cast X` paired with the next `Resolve Stack: X …`; creatures
                    print `X - Creature P / T`, other permanents print bare `X`; a spell that
                    resolved prints `X (id) - effect` and is not a permanent; a countered cast
                    never resolves and never enters
  removal           good by id, ambiguous by name — `Zone Change` carries the id once the
                    permanent has acted; before that, by name (two seats with one card → note)
  tokens            PARTIAL — a token is named on first use (attack/block/damage/death); one
                    that only sat is invisible; `extras.tokens_unobserved_resolutions` counts
                    the creation abilities that resolved per seat so the gap has a size
  tapped creatures  approximate — attacked since the controller's last untap; vigilance unknown
  hand              ESTIMATE — kept N + draw steps + "draws N" resolutions − lands − casts −
                    discards; written as {unknown: n, estimate: true}
  library, mana     not reconstructed; `open` is the untapped land count
  commander         zone from its last zone change or its resolution; casts counted

Everything marked approximate or estimate is ALSO written into `extras.reconstruction_notes`
so the resolver reads the limits in the artifact rather than here. Seeded runs make the
cut reproducible: `source` carries run, game, job, seed and game-in-job, and the same game
replays as `-n <game_in_job> -s <seed>`.
"""

import json
import re

from manamap.config import SIM_DIR
from manamap.pilot.common import deck_dir, load_json
from manamap.pilot import game_state
from manamap.sim import parse as sim_parse
from manamap.sim.forge import _seat_label, seat_dir

# Forge's phase labels → (CR phase, CR step). First-strike damage is part of the combat
# damage step (CR 510.4); Forge prints it as its own line.
FORGE_STEPS = {
    "Untap step": ("beginning", "untap"),
    "Upkeep step": ("beginning", "upkeep"),
    "Draw step": ("beginning", "draw"),
    "Main phase, precombat": ("precombat main", None),
    "Beginning of Combat Step": ("combat", "beginning of combat"),
    "Declare Attackers Step": ("combat", "declare attackers"),
    "Declare Blockers Step": ("combat", "declare blockers"),
    "First Strike Damage Step": ("combat", "combat damage"),
    "Combat Damage Step": ("combat", "combat damage"),
    "End of Combat Step": ("combat", "end of combat"),
    "Main phase, postcombat": ("postcombat main", None),
    "End step": ("ending", "end"),
    "Cleanup step": ("ending", "cleanup"),
}
_CR_TO_FORGE = {}
for forge_name, (ph, st) in FORGE_STEPS.items():
    _CR_TO_FORGE.setdefault((ph, st), forge_name)
DEFAULT_STEP = "precombat main"
_DRAWS = re.compile(r"\bdraws? (a|one|two|three|four|five|six|seven|\d+) cards?\b", re.I)
_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_CREATURE = re.compile(r"^(.+?) - Creature (\d+) ?/ ?(\d+)")
_SPELL = re.compile(r"^(.+?) \((\d+)\) - ")
_UNMORPH = re.compile(r"^(Ai\(\d+\)-[\w-]+) has unmorphed (.+)$")


def cut_matches(ev, turn, phase, step, start=True):
    """Is this phase event the cut point: turn T, the given CR phase/step?"""
    if ev.get("kind") != "phase" or ev.get("turn") != turn:
        return False
    f = FORGE_STEPS.get(ev.get("text"))
    return bool(f and f[0] == phase and f[1] == step)


def resolve_cut(step_text):
    """'declare blockers' / 'precombat main' / a Forge label → (phase, step)."""
    if step_text in FORGE_STEPS:
        return FORGE_STEPS[step_text]
    s = (step_text or DEFAULT_STEP).strip().lower()
    for (ph, st) in _CR_TO_FORGE:
        if s == (st or ph):
            return ph, st
    raise SystemExit(f"unknown step {step_text!r}; CR names: "
                     f"{sorted({st or ph for ph, st in _CR_TO_FORGE})}")


def _seat_state():
    return {"life": 40, "lands": {}, "perms": {}, "pending_casts": [], "tokens": {},
            "token_resolutions": 0, "kept": 7, "draw_steps": 0, "drawn": 0, "cast_n": 0,
            "lands_n": 0, "discards": 0, "graveyard": [], "commander_casts": 0,
            "commander_zone": None, "last_untap_turn": 0, "attacked_since_untap": set()}


def reconstruct(game, turn, phase, step, commanders):
    """Replay one parsed game up to the START of (turn, phase, step). Returns the v2
    `seats[]` plus bookkeeping for the scenario's `extras`."""
    seats = {s: _seat_state() for s in game["seats"]}
    for s, n in (game.get("mulligan") or {}).items():
        seats[s]["kept"] = n
    owner = dict(game["owner"])          # id -> seat, learned across the whole game
    notes = set()
    active, cur_phase, cur_step = None, None, None
    reached = False
    for ev in game["events"]:
        if cut_matches(ev, turn, phase, step):
            reached = True
            break
        if ev["turn"] > turn:
            break
        k = ev["kind"]
        if k == "phase":
            active = ev["seat"]
            cur_phase, cur_step = FORGE_STEPS.get(ev["text"], (None, None))
            st = seats.get(ev["seat"])
            if st is None:
                continue
            if ev["text"] == "Untap step":
                st["last_untap_turn"] = ev["turn"]
                st["attacked_since_untap"] = set()
                for land in st["lands"].values():
                    land["tapped"] = False
                for p in st["perms"].values():
                    p["tapped"] = False
                for t in st["tokens"].values():
                    t["tapped"] = False
            elif ev["text"] == "Draw step":
                st["draw_steps"] += 1
        elif k == "land":
            st = seats[ev["seat"]]
            st["lands"][ev["id"]] = {"name": ev["card"], "tapped": False, "entered_turn": ev["turn"]}
            st["lands_n"] += 1
        elif k == "mana":
            name, pid = ev["perm"]
            s = owner.get(pid)
            if s in seats and pid in seats[s]["lands"]:
                seats[s]["lands"][pid]["tapped"] = True
            elif s in seats and pid in seats[s]["perms"]:
                seats[s]["perms"][pid]["tapped"] = True
        elif k == "cast":
            st = seats[ev["seat"]]
            st["cast_n"] += 1
            st["pending_casts"].append(ev["what"])
            if commanders.get(ev["seat"]) and ev["what"] == commanders[ev["seat"]]:
                st["commander_casts"] += 1
        elif k == "resolve":
            text = ev["text"]
            mu = _UNMORPH.match(text)
            if mu and mu.group(1) in seats:
                # a face-down "Morph" becomes the card it always was
                for p in seats[mu.group(1)]["perms"].values():
                    if p["name"] == "Morph":
                        p["name"] = mu.group(2).strip(); p["pt"] = None; break
                continue
            if ev.get("creates_token") and ev["seat"] in seats:
                seats[ev["seat"]]["token_resolutions"] += 1
            m = _DRAWS.search(text)
            if m and ev["seat"] in seats:
                w = m.group(1).lower()
                seats[ev["seat"]]["drawn"] += _WORDS.get(w, int(w) if w.isdigit() else 1)
            # pair with a pending cast: creature "X - Creature P / T", permanent "X", spell "X (id) - …"
            for s, st in seats.items():
                for name in list(st["pending_casts"]):
                    mc = _CREATURE.match(text)
                    if mc and mc.group(1) == name:
                        st["perms"][f"name:{name}:{ev['turn']}"] = {
                            "name": name, "pt": f"{mc.group(2)}/{mc.group(3)}", "tapped": False,
                            "token": False, "entered_turn": ev["turn"], "id": None}
                        st["pending_casts"].remove(name)
                        if commanders.get(s) == name:
                            st["commander_zone"] = "battlefield"
                        break
                    if text == name:
                        st["perms"][f"name:{name}:{ev['turn']}"] = {
                            "name": name, "pt": None, "tapped": False, "token": False,
                            "entered_turn": ev["turn"], "id": None}
                        st["pending_casts"].remove(name)
                        if commanders.get(s) == name:
                            st["commander_zone"] = "battlefield"
                        break
                    if _SPELL.match(text) and _SPELL.match(text).group(1) == name:
                        st["pending_casts"].remove(name)          # an instant/sorcery resolved
                        break
        elif k == "attack":
            st = seats[ev["seat"]]
            for name, pid in ev["attackers"]:
                owner[pid] = ev["seat"]
                _bind(st, name, pid, ev["turn"])
                st["attacked_since_untap"].add(pid)
                if pid in st["perms"]:
                    st["perms"][pid]["tapped"] = True     # vigilance unknown → note
            notes.add("tapped creatures are those that attacked since their controller's last "
                      "untap step; vigilance is not visible in the log")
        elif k == "block":
            st = seats[ev["seat"]]
            for name, pid in ev["blockers"]:
                owner[pid] = ev["seat"]
                _bind(st, name, pid, ev["turn"])
        elif k == "damage":
            name, pid = ev["source"]
            s = owner.get(pid)
            if s in seats:
                _bind(seats[s], name, pid, ev["turn"])
        elif k == "life":
            if ev["seat"] in seats:
                seats[ev["seat"]]["life"] = ev["to"]
        elif k == "zone":
            name, pid, to, frm = ev["card"], ev["id"], ev["to"], ev["from"]
            if frm != "Battlefield":
                continue
            s = owner.get(pid)
            removed = False
            for st in seats.values():                 # by id, wherever it sits
                if st["lands"].pop(pid, None) is not None or st["perms"].pop(pid, None) is not None \
                        or st["tokens"].pop(pid, None) is not None:
                    removed = True
                    break
            if s in seats and commanders.get(s) == name:
                # Forge logs the exit zone BEFORE the command-zone replacement (CR 903.9a)
                # and the AI always takes it: the log later shows the commander recast.
                seats[s]["commander_zone"] = "command"
                notes.add("a commander's exit is logged as Graveyard/Exile before the "
                          "command-zone replacement; the bridge reads it as `command` "
                          "(the AI always takes the replacement, and later casts confirm)")
            if not removed:
                # never seen acting: remove by name, and say so if more than one seat had it
                holders = [x for x, st in seats.items()
                           if any(p["name"] == name for p in st["perms"].values())]
                if len(holders) > 1:
                    notes.add(f"'{name}' left the battlefield by name while more than one seat "
                              f"controlled one — removed from the first; check the board")
                for x in holders[:1]:
                    key = next(kk for kk, p in seats[x]["perms"].items() if p["name"] == name)
                    seats[x]["perms"].pop(key)
                    if commanders.get(x) == name:
                        seats[x]["commander_zone"] = "command"
            if s in seats and to == "Graveyard":
                seats[s]["graveyard"].append(name)
    if not reached:
        notes.add(f"the game did not reach turn {turn} {step or phase}: the state is the end "
                  f"of what was logged")
    return seats, sorted(notes), active, cur_phase, cur_step


def _bind(st, name, pid, turn):
    """Give a name-only permanent its id the first time it acts, or register a token."""
    if pid in st["perms"] or pid in st["lands"] or pid in st["tokens"]:
        return
    if name.endswith("Token"):
        st["tokens"][pid] = {"name": name, "tapped": False, "token": True, "id": pid,
                             "first_seen_turn": turn}
        return
    key = next((k for k, p in st["perms"].items() if p["name"] == name and p["id"] is None), None)
    if key:
        p = st["perms"].pop(key); p["id"] = pid; st["perms"][pid] = p


def seat_object(label, st, seat_id, deck_slug, commander, archetype):
    lands = sorted(st["lands"].values(), key=lambda l: (l["entered_turn"], l["name"]))
    perms = sorted(st["perms"].values(), key=lambda p: (p["entered_turn"], p["name"]))
    tokens = sorted(st["tokens"].values(), key=lambda t: (t["first_seen_turn"], t["name"]))
    board = []
    for p in perms:
        board.append({"name": p["name"], "controller": seat_id, "tapped": p["tapped"],
                      "summoning_sick": None, "pt": p["pt"], "token": False,
                      "annotations": []})
    for t in tokens:
        board.append({"name": t["name"], "controller": seat_id, "tapped": t["tapped"],
                      "summoning_sick": None, "pt": None, "token": True,
                      "annotations": ["observed acting in the log; tokens that only sat are not listed"]})
    for l in lands:
        board.append({"name": l["name"], "controller": seat_id, "tapped": l["tapped"],
                      "type": "Land", "token": False, "annotations": []})
    est = st["kept"] + st["draw_steps"] + st["drawn"] - st["lands_n"] - st["cast_n"] - st["discards"]
    return {"seat": seat_id, "label": label, "deck": deck_slug, "archetype": archetype,
            "commander": {"name": commander, "zone": st["commander_zone"] or "command",
                          "casts": st["commander_casts"]} if commander else None,
            "life": st["life"], "poison": 0,
            "hand": {"unknown": max(0, est), "estimate": True},
            "library": {"count": None},
            "graveyard": st["graveyard"], "exile": [],
            "mana": {"available": None, "open": sum(1 for l in lands if not l["tapped"]), "pool": "{0}"},
            "board": board}


def build_scenario(slug, rec, game_index, turn, step_text, game, label):
    phase, step = resolve_cut(step_text)
    outcome = rec["outcomes"][game_index - 1]
    seats_label = [s["forge_name"] for s in rec["seats"]]
    forge_labels = list(_seat_label(seats_label).keys())
    commanders, archetypes = {}, {}
    for fl, s in zip(forge_labels, rec["seats"]):
        d = seat_dir(s["slug"])
        from manamap.pilot.fetch_deck import parse_decklist
        entries = parse_decklist((d / "decklist.txt").read_text(encoding="utf-8"))
        commanders[fl] = next((e["name"] for e in entries if e.get("is_commander")), None)
        frame = load_json(d / "strategic_frame.json") or {}
        archetypes[fl] = frame.get("archetype")
    states, notes, active, cur_phase, cur_step = reconstruct(game, turn, phase, step, commanders)
    seat_ids = {fl: ("you" if i == 0 else f"seat-{i + 1}") for i, fl in enumerate(forge_labels)}
    seats_out = []
    for fl in forge_labels:
        if fl not in states:
            continue
        seats_out.append(seat_object(fl, states[fl], seat_ids[fl], label.get(fl, fl),
                                     commanders.get(fl), archetypes.get(fl)))
    # the active seat at the cut is whoever owns the turn; at the start of a step the
    # active player receives priority (CR 117.3a)
    active_id = seat_ids.get(active) if active else None
    notes = list(notes) + [
        "hand sizes are ESTIMATES: kept + draw steps + 'draws N' resolutions − lands − casts − discards",
        "library counts and mana symbols are not reconstructed; `mana.open` is the untapped land count",
        "a creature's summoning sickness is not reconstructed (null)",
    ]
    unobserved = {seat_ids[fl]: states[fl]["token_resolutions"] for fl in forge_labels if fl in states}
    title = f"sim {rec['run_id']} · game {game_index} · turn {turn} {step or phase}"
    return {
        "id": None, "slug": slug, "deck": slug, "title": title,
        "rules_version": None,
        "scenario": {
            "version": 2,
            "source": {"run_id": rec["run_id"], "game": game_index, "log": outcome.get("log"),
                       "seed": outcome.get("seed"), "game_in_job": outcome.get("game_in_job"),
                       "cut": {"turn": turn, "phase": phase, "step": step},
                       "replay": (f"-n {outcome.get('game_in_job')} -s {outcome.get('seed')}"
                                  if outcome.get("seed") else "not seeded — this run predates -s")},
            "turn": turn, "active_seat": active_id, "phase": phase, "step": step,
            "priority": active_id,
            "seats": seats_out,
            "stack": [], "actions": [],
            "extras": {"reconstruction_notes": notes,
                       "tokens_unobserved_resolutions": unobserved,
                       "outcome_of_this_game": {"winner": outcome.get("winner"),
                                                "round": outcome.get("round"),
                                                "global_turn": outcome.get("global_turn")}},
            "question": "",
        },
    }


def lift(slug, run_id, game_index, turn, step_text=None, to_stack=False):
    base = deck_dir(slug) / SIM_DIR
    rec_path = base / f"{run_id}.json"
    if not rec_path.exists():
        raise SystemExit(f"{slug}: no run {run_id!r} under {SIM_DIR}/")
    rec = load_json(rec_path)
    if not (1 <= game_index <= len(rec["outcomes"])):
        raise SystemExit(f"{slug}: run has {len(rec['outcomes'])} games; --game must be 1..{len(rec['outcomes'])}")
    outcome = rec["outcomes"][game_index - 1]
    log = base / "logs" / run_id / outcome["log"]
    if not log.exists():
        raise SystemExit(f"{log} is missing — logs are gitignored and exist only where the run "
                         f"was made; a seeded run replays with `{rec.get('seed_base') and 'simulate --force'}`")
    games = sim_parse.parse_games(log.read_text(encoding="utf-8", errors="replace"))
    gij = outcome.get("game_in_job") or 1 + sum(1 for o in rec["outcomes"][:game_index - 1]
                                                 if o.get("log") == outcome["log"])
    if gij > len(games):
        raise SystemExit(f"{log.name} holds {len(games)} game(s); wanted #{gij}")
    game = games[gij - 1]
    label = _seat_label([s["forge_name"] for s in rec["seats"]])
    doc = build_scenario(slug, rec, game_index, turn, step_text, game, label)
    phase, step = doc["scenario"]["phase"], doc["scenario"]["step"]
    if to_stack:
        stacks = deck_dir(slug) / "stacks"
        stacks.mkdir(exist_ok=True)
        nums = [int(p.name[:3]) for p in stacks.glob("[0-9][0-9][0-9]-*.json")]
        nnn = f"{max(nums, default=0) + 1:03d}"
        kebab = f"sim-g{game_index}-t{turn}-{(step or phase).replace(' ', '-')}"
        out = stacks / f"{nnn}-{kebab}.json"
        doc["id"] = nnn
    else:
        out_dir = base / "scenarios"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"{run_id}-g{game_index}-t{turn}-{(step or phase).replace(' ', '-')}.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return out, doc


def main(args):
    out, doc = lift(args.slug, args.run, args.game, args.turn, getattr(args, "step", None),
                    to_stack=getattr(args, "stack", False))
    sc = doc["scenario"]
    print(f"{args.slug}: lifted game {args.game} of {args.run} at turn {args.turn} "
          f"{sc['step'] or sc['phase']} → {out.relative_to(deck_dir(args.slug))}")
    for s in sc["seats"]:
        cz = s["commander"] or {}
        print(f"  {s['seat']:<7} {s['deck']:<18} life {s['life']:<3} board {len(s['board']):<3} "
              f"open {s['mana']['open']:<2} hand~{s['hand']['unknown']:<2} "
              f"cmdr {cz.get('zone', '—')} ×{cz.get('casts', 0)}")
    print(f"  notes: {len(sc['extras']['reconstruction_notes'])} · replay: {sc['source']['replay']}")
    print(f"  next: write `scenario.question` (one rules domain), add `stack`/`actions`, then "
          f"`validate-stack {args.slug} --scenario-only` and `/resolve-stack`")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot sim-scenario <slug> <run-id> --game G --turn T [--step S]`.")
