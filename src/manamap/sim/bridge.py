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
#: THE NUMBER WORDS FORGE ACTUALLY PRINTS, swept rather than assumed. This stopped
#: at "seven" and had no "an", so every larger count read as UNREADABLE — and the
#: large counts are the ones that matter: Krenko, Mob Boss prints "creates eight
#: 1/1 red Goblin creature tokens", and a doubler takes it to sixteen or thirty.
#: The biggest boards were therefore the ones most wrongly reported.
#: Sweep of one 100-game run: a 1114, two 120, X 39, three 17, eight 5, six 3,
#: four 3, eighteen 2, twelve 1, thirty 1, sixteen 1, fourteen 1.
#: `X` stays absent on purpose — it is reported as unreadable, never guessed.
_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
          "twenty": 20, "thirty": 30}
#: A CREATURE'S P/T IS NOT ALWAYS DIGITS. Forge prints a characteristic-defining
#: power/toughness verbatim: `Lord of Extinction - Creature * / *`. Demanding
#: `\d+` meant this pattern did not match, `text == name` did not match either,
#: and the cast STAYED IN `pending_casts` FOREVER — so the creature never reached
#: the board and the lift described a battlefield it was not on.
#:
#: FOUND BY A RULES-CHECKER ON A REAL BOARD, 2026-09-28, inside the
#: `/resolve-stack` loop. Stack 008 said seat-2 held "Ripples of Undeath and
#: three tapped lands"; the log had Splinterfright there too, and the kill the
#: artifact proved survived only because it happened to be tapped. ONE untapped
#: blocker turns that kill into a survival, so this gap is one-way HOSTILE to
#: any lethal claim — the dangerous direction.
#:
#: SWEEP, one 100-game run: 62 `* / *`, 17 `* / *+N`, 28 `* / * (X=N)` — about 107
#: resolutions over six distinct creatures (Boneyard Wurm, Lord of Extinction,
#: Mortivore, Old Stickfingers, Souls of the Lost, Splinterfright), every one an
#: opponent's.
#:
#: The P/T is kept as the LITERAL `*/*` rather than a number, because the log does
#: not carry the value. "A real creature whose size I cannot give you" is what a
#: resolver must reason about; a fabricated number would be worse than the
#: absence it replaces.
_CREATURE = re.compile(r"^(.+?) - Creature (\d+|\*) ?/ ?(\d+|\*)")
#: AN AURA ENTERS ATTACHED, and its resolution reads `Rancor (203) -  Attach to
#: Sythis (12)`. That matches `_SPELL` below, so the branch meaning "an instant or
#: sorcery resolved" DISCARDED it, and every Aura was absent from every lift.
#: Found the same day on stack 009: seat-4's Sphere of Safety taxes attackers {X}
#: where X counts its controller's enchantments, and an Aura this lift had dropped
#: made the real tax {3} against the {2} the artifact reasoned from.
#:
#: SWEEP, same run: Rancor 38, Whip Silk 26, All That Glitters 25, Overgrowth 24,
#: Ancestral Mask 23, Strength of the Harvest 17. EQUIPMENT IS NOT AFFECTED and
#: must not be: it enters as a bare name when cast, and its later equip ability is
#: not a cast, so it is no longer pending when it attaches.
_ATTACH = re.compile(r"^\s*Attach to (.+?)(?: \((\d+)\))?$")
#: A TOKEN EXISTS FROM THE MOMENT IT IS CREATED, not from the moment it acts.
#:
#: `_bind` registered a token the first time it attacked, blocked or dealt damage,
#: so a token that was made and left standing was ABSENT from every lifted board —
#: and the artifact said so in an annotation ("tokens that only sat are not
#: listed") as though that were a footnote rather than the whole board.
#:
#: MEASURED, 2026-09-28, one 100-game run over 804 of our precombat mains: the
#: lift listed 187 tokens where 1843 were alive (created minus died). **It saw
#: 10%.** On a Goblin token deck that is not a gap, it is the deck. Board width
#: beside the commander, as lifted against with tokens counted:
#:     >= 3 others   17.3%  ->  67.1%
#:     >= 6 others    2.0%  ->  36.5%
#:     >= 8 others    0.0%  ->  24.9%
#: A conclusion was drawn from the first column and it was wrong by ~18x.
#:
#: THE NAME MUST MATCH WHAT FORGE CALLS THE TOKEN WHEN IT ACTS, or binding an id
#: would add a second copy of a token already listed. Forge names them
#: "<Type> Token"; the creation text spells the type in Capitals and the colours
#: in lowercase, so the Capitalised words ARE the type — which keeps "Spirit
#: Cleric Token" whole. Sweep of one run: Goblin 1219, Treasure 978, Pegasus 474,
#: Angel 242, Cat 234, Human 207, Spawn 125, Spirit Cleric 111.
#: THE COUNT GROUP IS `\w+`, NOT A LIST OF THE COUNTS THIS CAN READ. Listing them
#: made `create X 1/1 red Goblin creature tokens` match NOTHING, so an unreadable
#: count was silently dropped instead of reported — the confident-zero shape this
#: file keeps paying for. Capture whatever word is there and let `_token_spec`
#: decide whether it knows it.
_TOKEN_MAKE = re.compile(
    r"create[sd]?\s+(\w+)\s+"
    r"((?:\d+/\d+)\s+)?([^.]{0,60}?)\btokens?\b", re.I)


def _token_spec(text):
    """`[(count, name, pt), ...]` for every token this resolution creates.

    A count this cannot read (X, "that many") yields None for the count, so the
    caller reports it instead of guessing — the same contract the goldfish keeps
    for an effect it cannot price.
    """
    out = []
    for m in _TOKEN_MAKE.finditer(text or ""):
        word = m.group(1).lower()
        n = _WORDS.get(word, int(word) if word.isdigit() else None)
        types = [w for w in (m.group(3) or "").split()
                 if w[:1].isupper() and w.lower() != "token"]
        name = (" ".join(types) + " Token") if types else "Token"
        out.append((n, name, (m.group(2) or "").strip() or None))
    return out
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
            "token_resolutions": 0, "token_serial": 0, "token_counts_unread": 0,
            "kept": 7, "draw_steps": 0, "drawn": 0, "cast_n": 0,
            "lands_n": 0, "discards": 0, "graveyard": [], "commander_casts": 0,
            "commander_zone": None, "last_untap_turn": 0, "attacked_since_untap": set()}


def reconstruct(game, turn, phase, step, commanders):
    """Replay one parsed game up to the START of (turn, phase, step). Returns the v2
    `seats[]` plus bookkeeping for the scenario's `extras`."""
    seats = {s: _seat_state() for s in game["seats"]}
    for s, n in (game.get("mulligan") or {}).items():
        seats[s]["kept"] = n
    owner = dict(game["owner"])          # id -> seat, learned across the whole game
    # WHO MAKES A TOKEN OF THIS NAME. `owner` is learned from lines that name a
    # controller outright, so a token that never attacked, blocked or dealt damage
    # NEVER ENTERED IT — and its death line carries an id but no seat. Measured on
    # one game: 11 of 18 token zone-changes were unattributable, so token deaths
    # were being applied to whichever seat happened to hold a matching one.
    #
    # THE SAME DISCIPLINE `parse.py` ALREADY USES FOR ITS NAME FALLBACK: this is an
    # INFERENCE where `owner` is a fact, it is consulted ONLY where `owner` is
    # silent, and a name two seats both make stays UNATTRIBUTED rather than
    # guessed. The creating resolution names its seat, which is what makes it
    # usable at all.
    token_makers = {}
    for _ev in game["events"]:
        if _ev.get("kind") == "resolve" and _ev.get("creates_token") and _ev.get("seat"):
            for _n, _nm, _pt in _token_spec(_ev.get("text") or ""):
                token_makers.setdefault(_nm, set()).add(_ev["seat"])
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
                st_t = seats[ev["seat"]]
                st_t["token_resolutions"] += 1
                # THE TOKENS THEMSELVES, at creation. Unbound (`id: None`) until
                # one of them acts and `_bind` gives it an id — which consumes a
                # placeholder rather than adding a second entry.
                for n, tname, tpt in _token_spec(text):
                    if n is None:
                        st_t["token_counts_unread"] += 1
                        notes.add("a token-creating resolution states a count this "
                                  "bridge cannot read (X, or 'that many'); those "
                                  "tokens are NOT on the board and the count is in "
                                  "`token_counts_unread`")
                        continue
                    for _i in range(n):
                        st_t["tokens"][f"tok:{tname}:{ev['turn']}:{_i}:"
                                       f"{st_t['token_serial']}"] = {
                            "name": tname, "pt": tpt, "tapped": False,
                            "token": True, "id": None,
                            "first_seen_turn": ev["turn"]}
                        st_t["token_serial"] += 1
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
                    ms = _SPELL.match(text)
                    if ms and ms.group(1) == name:
                        # AN AURA IS A PERMANENT, NOT A SPELL THAT WENT AWAY.
                        # Check the effect before assuming this resolution was an
                        # instant or sorcery; see `_ATTACH` above.
                        ma = _ATTACH.match(text[ms.end():])
                        if ma:
                            st["perms"][f"name:{name}:{ev['turn']}"] = {
                                "name": name, "pt": None, "tapped": False,
                                "token": False, "entered_turn": ev["turn"],
                                "id": ms.group(2), "attached_to": ma.group(1)}
                        st["pending_casts"].remove(name)
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
            if not removed and name.endswith("Token"):
                # A TOKEN THAT NEVER ACTED STILL DIES, and its zone change carries
                # an id that was never bound — so the by-id pop above missed it.
                # Attribute the death in three layers, most reliable first, and
                # leave it UNATTRIBUTED rather than guess:
                #   1. `owner` said so. A fact.
                #   2. exactly one seat ever MADE a token of this name.
                #   3. exactly one seat currently holds an unlisted one.
                # Removing from the wrong seat deletes an opponent's blocker,
                # which is the direction that flatters a kill — so a tie removes
                # from nobody and says so.
                if s in seats:
                    holders = [s]
                else:
                    # HOLDING ONE IS SHARPER THAN EVER MAKING ONE, so it is
                    # asked first. Ordered the other way round this was needlessly
                    # conservative: seat A makes two and both have died, seat B
                    # makes one and still holds it — both "make" them, so the
                    # maker layer calls it a tie when the holder layer knows it
                    # must be B's.
                    holders = [x for x in sorted(seats)
                               if any(t["id"] is None and t["name"] == name
                                      for t in seats[x]["tokens"].values())]
                    if len(holders) != 1:
                        holders = sorted(token_makers.get(name, set()) & set(seats))
                if len(holders) == 1:
                    x = holders[0]
                    key = next((k for k, t in seats[x]["tokens"].items()
                                if t["id"] is None and t["name"] == name), None)
                    if key is not None:
                        seats[x]["tokens"].pop(key)
                        removed = True
                elif len(holders) > 1:
                    notes.add(
                        f"a {name} left the battlefield and its controller is not "
                        f"in the log: {', '.join(holders)} all make one, so it was "
                        f"removed from NOBODY. Their token counts may each be one "
                        f"too high — absent attribution, never guessed")
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
    if any(str(p.get("pt") or "").count("*") for st in seats.values()
           for p in st["perms"].values()):
        notes.add("a creature on this board has a CHARACTERISTIC-DEFINING power "
                  "(`*/*`): the log prints no value and the graveyard lists here "
                  "are battlefield deaths only, so its size is UNKNOWN and cannot "
                  "be computed from this artifact. Reason about whether the answer "
                  "depends on it")
    if not reached:
        notes.add(f"the game did not reach turn {turn} {step or phase}: the state is the end "
                  f"of what was logged")
    return seats, sorted(notes), active, cur_phase, cur_step


def _bind(st, name, pid, turn):
    """Give a name-only permanent its id the first time it acts, or register a token."""
    if pid in st["perms"] or pid in st["lands"] or pid in st["tokens"]:
        return
    if name.endswith("Token"):
        # CONSUME A PLACEHOLDER rather than adding a second entry: the token was
        # already registered when it was created, and this is only the first time
        # it acted. Without this, every token that acts would be counted twice.
        key = next((k for k, t in st["tokens"].items()
                    if t["id"] is None and t["name"] == name), None)
        if key is not None:
            t = st["tokens"].pop(key)
            t["id"] = pid
            st["tokens"][pid] = t
            return
        st["tokens"][pid] = {"name": name, "tapped": False, "token": True, "id": pid,
                             "first_seen_turn": turn}
        return
    key = next((k for k, p in st["perms"].items() if p["name"] == name and p["id"] is None), None)
    if key:
        p = st["perms"].pop(key); p["id"] = pid; st["perms"][pid] = p
        return
    # A PERMANENT THAT ACTS IS ON THE BATTLEFIELD. That is ground truth, and it is
    # the only signal available for a permanent that ENTERED WITHOUT BEING CAST:
    # reanimation, a tutor straight to the battlefield, a blink returning. There
    # is no `pending_casts` entry to pair with and the resolution names no card —
    # `Rise of the Witch-king (309) - … returns up to one Permanent…card from
    # their graveyard to the battlefield` says WHO and not WHAT.
    #
    # FOUND BY A CHECKER ON THE RE-LIFTED STACK 010, after the `*/*` fix. Lord of
    # Extinction was reanimated on jarad's turn, attacked for 18 on turn 25 and was
    # exiled on turn 28 — so it was on the battlefield at the turn-27 cut and the
    # board still said seat-2 had two creatures when it had three. The kill the
    # artifact proved held only because that creature was ALSO tapped: luck, not
    # reasoning, which is exactly what this class of gap keeps buying.
    #
    # An earlier pass of this checker diagnosed "the bridge drops non-cast
    # permanents" and I narrowed it to `*/*` creatures. It was BOTH, and the wider
    # framing was the right one.
    st["perms"][pid] = {"name": name, "pt": None, "tapped": False, "token": False,
                        "entered_turn": turn, "id": pid,
                        "entered_without_a_cast": True}


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
                      "summoning_sick": None, "pt": t.get("pt"), "token": True,
                      "annotations": [
                          # THE ANNOTATION WAS THE BUG'S CONFESSION. It read
                          # "observed acting in the log; tokens that only sat are
                          # not listed" — describing a board missing 90% of its
                          # tokens as though that were a footnote. Tokens are now
                          # registered when CREATED, so the note says what is
                          # actually uncertain instead.
                          "created at the turn shown; its controller is inferred "
                          "where the log never named one (see notes)"
                          if t.get("id") is None else
                          "seen acting in the log, so its id is known"]})
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
            # A FLOOR, AND SAID SO. `parse.py` emits zone events only for
            # Battlefield -> Graveyard and Battlefield -> Exile, so a card MILLED
            # or DISCARDED into the graveyard never reaches this list. On a
            # graveyard deck that is most of it — and it is load-bearing, because a
            # characteristic-defining power counts cards there: Splinterfright is
            # `*/*` where * is the creature cards in its controller's graveyard,
            # so this list would make it 0/0 and already dead. Found by a resolver
            # on stack 010 the day `*/*` creatures became visible at all.
            "graveyard": st["graveyard"],
            "graveyard_is_a_floor": True,
            "graveyard_note": (
                "battlefield deaths only — a card milled or discarded into the "
                "graveyard is not logged as a zone change, so this list is a "
                "FLOOR. Do not compute a characteristic-defining power from it."),
            "exile": [],
            "mana": {"available": None, "open": sum(1 for l in lands if not l["tapped"]), "pool": "{0}"},
            "board": board}


def build_scenario(slug, rec, game_index, turn, step_text, game, label):
    phase, step = resolve_cut(step_text)
    outcome = rec["outcomes"][game_index - 1]
    # THE DECKS ROTATE THROUGH THE SEATS, so WHICH `Ai(k)` a deck was is a
    # property of THIS GAME, not of the record. `outcomes[i].seat_order` holds
    # the rotation that game was played under; a record from before rotation has
    # no such key and the record's own seat order is the answer.
    #
    # Two things broke without this. `_seat_label` is a CROSS PRODUCT — every
    # index for every deck — so zipping its keys against `rec["seats"]` paired
    # `Ai(2)-<deck 0>` with seat 1 and gave every seat the wrong decklist,
    # commander and archetype. And `seat_ids` called index 0 "you", which under
    # rotation is whichever deck happened to sit first — so a board lifted for
    # `/resolve-stack` could be argued from an opponent's side of the table.
    by_slug = {s["slug"]: s["forge_name"] for s in rec["seats"]}
    order = outcome.get("seat_order") or [s["slug"] for s in rec["seats"]]
    seat_slugs = [s for s in order if s in by_slug]
    forge_labels = [f"Ai({i + 1})-{by_slug[sl]}" for i, sl in enumerate(seat_slugs)]
    seats_in_order = [next(s for s in rec["seats"] if s["slug"] == sl) for sl in seat_slugs]
    commanders, archetypes = {}, {}
    for fl, s in zip(forge_labels, seats_in_order):
        d = seat_dir(s["slug"])
        from manamap.pilot.fetch_deck import parse_mainboard
        entries = parse_mainboard((d / "decklist.txt").read_text(encoding="utf-8"))
        commanders[fl] = next((e["name"] for e in entries if e.get("is_commander")), None)
        frame = load_json(d / "strategic_frame.json") or {}
        archetypes[fl] = frame.get("archetype")
    states, notes, active, cur_phase, cur_step = reconstruct(game, turn, phase, step, commanders)
    # "you" IS OUR DECK, wherever it sat this game — never seat index 0.
    seat_ids, n_other = {}, 0
    for i, (fl, sl) in enumerate(zip(forge_labels, seat_slugs)):
        if sl == slug:
            seat_ids[fl] = "you"
        else:
            n_other += 1
            seat_ids[fl] = f"seat-{n_other + 1}"
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


def lift(slug, run_id, game_index, turn, step_text=None, to_stack=False, extras=None):
    # A branch keeps its runs beside its own list, never in the deck's `sim/`.
    from manamap.sim.forge import _out_dir
    base = _out_dir(slug)
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
    # WHAT THE GATE WILL READ BACK. `lift_sha` is over the canonical board, so a
    # clone without the logs can still tell whether a re-lift moved it once it
    # has one; `bridge_sha` is over this module, the `model_version` idea — a
    # scenario lifted under an older bridge says so before anyone re-lifts.
    from manamap.pilot import validate_lift as _vl
    doc["scenario"]["source"]["lift_sha"] = _vl.lift_sha(doc)
    doc["scenario"]["source"]["bridge_sha"] = _vl.bridge_sha()
    if extras:
        # the finder's provenance (criterion, shape, recurrence) — what a
        # handbook proposal cites instead of "I picked this game"
        doc["scenario"]["extras"].update(extras)
    if to_stack:
        # A BRANCH'S RUN LIVES BESIDE ITS OWN LIST; ITS STACKS DO NOT.
        # `_out_dir` above already resolves "slug@branch" to the branch's sim/
        # directory, but `deck_dir` takes a plain slug — so lifting a board out
        # of a branch run died with "No deck directory for 'goblin-storm@zada-v1'"
        # after the run had already been found and parsed. Stacks are DECK-level
        # artifacts (the deck page and `build-index` both read
        # data/decks/<slug>/stacks/), so a scenario lifted from a branch belongs
        # with the deck, named by the game and turn it came from.
        stacks = deck_dir(slug.split("@", 1)[0]) / "stacks"
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
          f"{sc['step'] or sc['phase']} → "
          f"{out.relative_to(deck_dir(args.slug.split('@', 1)[0]))}")
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
