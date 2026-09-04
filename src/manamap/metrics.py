"""The metrics catalog — one definition per figure, and where it comes from.

PRD §14 makes the definitions BINDING: "the same definition is used by
simulation, by the builder's objective, and by the comparison report." B-3 adds
the part that makes a catalog more than a list — *"every metric has a written
definition AND THE LOG EVENTS IT DERIVES FROM."*

So this is not documentation. Every entry names the artifact and key path that
answers it, and `tests/test_metrics_catalog.py` checks those paths against real
tracked artifacts in both directions: a figure the catalog claims is published
must actually be there, and one it calls unavailable must NOT be.

## Three states, not two

The obvious shape is a boolean and it is wrong here, for the same reason the
paper lock needed a third state: "not present" and "cannot be present" are
different facts and only one of them is a gap in the code.

    PUBLISHED    every tracked artifact of its kind carries it
    OPT_IN       present only when the deck's declaration switches the channel
                 on, so it is in some artifacts and not others
    DERIVABLE    the data is in the record and nothing aggregates it yet
    UNAVAILABLE  no engine here can answer it, and `absent` says why

OPT_IN is the state a boolean hides and the fleet is full of. Measured across
the ten tracked goldfish artifacts: `mean_extra_cards_drawn_by_turn` is in ONE
of them and `combat.mean_kill_turn` in two, because both are behind a per-deck
`model_*` flag. That is why `benchmark` runs its own goldfish with UNIFORM flags
that override every declaration — ranking a fleet on figures half of it does not
have would compare a deck measured with a kill clock against nine measured
without.

An UNAVAILABLE entry is the most useful kind in this file. **Absent means
ABSENT, never zero** — a figure nobody measured must be a missing key with a
stated reason, because `0.0` is a measurement and a reader cannot tell them
apart.

## Two engines, and the split is not a preference

`docs/simulation.md` records the measurement: Forge emits exactly two zone
transitions — Battlefield→Graveyard and Battlefield→Exile — and across a
100-game pod run there are **zero `from Library` lines** and **zero
`to Battlefield` lines**. So everything about drawing, tutoring, recursion and
what ARRIVES on a board is unrecoverable from a Forge log, at any effort. The
goldfish simulates the library and the hand, so it is the only engine that can
answer the Mana and Card-flow groups at all.

That is why PRD Epic D's "fold the goldfish into Forge as the no-interaction
batch" is rejected in `docs/prd.md`'s intake notes: it would delete half of what
is below. The two engines answer different questions and both are load-bearing.
"""

#: Every tracked artifact of its kind carries this figure.
PUBLISHED = "published"
#: Behind a per-deck `model_*` flag: in some artifacts, absent from others.
OPT_IN = "opt_in"
#: The data is in the record; nothing aggregates it into this figure yet.
DERIVABLE = "derivable"
#: No engine here can answer it. `absent` says why, and the reason is measured.
UNAVAILABLE = "unavailable"

STATUSES = (PUBLISHED, OPT_IN, DERIVABLE, UNAVAILABLE)

#: PRD §14's seven groups, in its order.
GROUPS = ("mana", "card_flow", "board", "speed", "resilience",
          "interaction", "outcome")

#: The engines. `None` goes with UNAVAILABLE and with nothing else.
ENGINES = ("forge", "goldfish", "both", None)

#: THE MEASUREMENT THAT DECIDES HALF THIS FILE, kept as a constant so an entry
#: citing it cannot drift from the number. From a 100-game pod run.
FORGE_ZONE_LIMIT = (
    "Forge logs exactly two zone transitions — Battlefield to Graveyard and "
    "Battlefield to Exile. Measured on a 100-game pod run: ZERO `from Library` "
    "lines and ZERO `to Battlefield` lines. No parser change recovers this.")


def _m(group, definition, status, engine, source, absent=None, caveat=None):
    row = {"group": group, "definition": definition, "status": status,
           "engine": engine, "source": source}
    if absent:
        row["absent"] = absent
    if caveat:
        row["caveat"] = caveat
    return row


CATALOG = {

    # ── Mana ────────────────────────────────────────────────────────────────
    "missed land drops": _m(
        "mana",
        "Turns 1-8 where no land was played and one was available to play.",
        PUBLISHED, "goldfish",
        "diagnostic.json → mana.missed_land_drop_rate / missed_land_drop_by_five; "
        "goldfish_metrics.json → metrics.land_drop_hit_rate_by_turn",
        caveat="Forge cannot answer this. It logs `Land: seat played X`, so a "
               "turn WITHOUT a land drop is visible — but 'and one was available "
               "to play' is a fact about the hand, and the hand is never logged. "
               "A Forge version would count turns with no land played and quietly "
               "mean something else."),

    "mean available mana by turn": _m(
        "mana",
        "Untapped mana at start of main phase, per turn.",
        PUBLISHED, "goldfish",
        "goldfish_metrics.json → metrics.mean_available_mana_by_turn; "
        "diagnostic.json → mana.mean_mana_by_turn"),

    "color screw rate": _m(
        "mana",
        "Games where a castable card was held for missing a color, by color.",
        UNAVAILABLE, None,
        None,
        absent="Neither engine reports it per colour. The goldfish models "
               "colours (`model_colors`) and `mana_analysis` answers the "
               "adjacent question deterministically — how many sources each "
               "colour has against its Karsten target — but no artifact carries "
               "a per-colour count of games where a card was STUCK. Forge cannot: "
               "a card held in hand is never logged."),

    "keepable sevens": _m(
        "mana",
        "Share of opening sevens meeting the deck's stated keep rule.",
        PUBLISHED, "goldfish",
        "goldfish_metrics.json → metrics.opening_hand.keep_first_seven_rate",
        caveat="The keep rule is the goldfish's own — 2-5 lands in seven — not "
               "the deck's stated one. No deck states one anywhere a program can "
               "read, so this is a fixed rule applied uniformly rather than the "
               "per-deck gate PRD §14 describes."),

    "mulligan rate": _m(
        "mana",
        "Mean mulligans taken per game.",
        PUBLISHED, "both",
        "goldfish_metrics.json → metrics.opening_hand.mean_mulligans; "
        "sim/<run>.json → analysis.seats[].mulligans_taken and .mulligan_kept",
        caveat="FORGE GIVES THE FIRST MULLIGAN FREE. Measured across all 130 "
               "tracked logs and 5,056 seat-hands with zero exceptions: one "
               "mulligan still keeps SEVEN, so `kept = 7 - max(0, taken - 1)`. "
               "Under the London rule one mulligan keeps 7 and bottoms 1 for a "
               "hand of six. Both figures are reported because neither is "
               "derivable from the other under real rules."),

    # ── Card flow ───────────────────────────────────────────────────────────
    "cards drawn per game": _m(
        "card_flow",
        "Total draws beyond the natural draw step.",
        OPT_IN, "goldfish",
        "goldfish_metrics.json → metrics.mean_extra_cards_drawn_by_turn; "
        "diagnostic.json → steam.extra_cards_by_turn",
        caveat="Counts only draw the model can price — a card's own ETB draw, an "
               "instant or sorcery that draws, an upkeep trigger, and a trigger "
               "that draws when other creatures enter. Activated, X-based, "
               "sacrifice-gated and death-triggered draw are NOT counted, and the "
               "cards are named in the goldfish artifact. OPT-IN: it rides "
               "behind `model_draw` and is present in ONE of the ten tracked "
               "goldfish artifacts. " + FORGE_ZONE_LIMIT),

    "turns with empty hand": _m(
        "card_flow",
        "Count of turns ending with zero cards in hand.",
        PUBLISHED, "goldfish",
        "diagnostic.json → stall.cause.hand_empty (against stall.cause.stall_turns)",
        caveat="Measured as a STALL WITH AN EMPTY HAND rather than an empty hand "
               "as such: the goldfish records `hand_size_by_turn` alongside "
               "`stall_by_turn`, and `cause` splits a stall into hand_empty and "
               "mana_short. A turn with an empty hand that was not a stall — "
               "everything was cast — is not counted, which is the honest "
               "reading of the metric's intent. " + FORGE_ZONE_LIMIT),

    "draw-engine uptime": _m(
        "card_flow",
        "Share of turns with at least one active repeatable draw source.",
        UNAVAILABLE, None,
        None,
        absent="Neither engine tracks whether a draw source is ON the battlefield "
               "and active. The goldfish prices draw as it fires rather than "
               "modelling a permanent's uptime, and Forge never logs a permanent "
               "ARRIVING. " + FORGE_ZONE_LIMIT),

    # ── Board ───────────────────────────────────────────────────────────────
    "bodies by turn": _m(
        "board",
        "Creature count on board at end of each turn.",
        PUBLISHED, "goldfish",
        "goldfish_metrics.json → metrics.mean_bodies_by_turn",
        caveat="In all ten tracked goldfish artifacts — unlike the kill clock "
               "beside it, which is behind `model_combat` and is in two. "
               + FORGE_ZONE_LIMIT),

    "creature power distribution": _m(
        "board",
        "P25 / P50 / P75 / max power of creatures on board.",
        UNAVAILABLE, None,
        None,
        absent="No percentiles anywhere, and Forge cannot supply the board at "
               "all. `sim/threat.py` documents a deliberate REFUSAL of board "
               "power as a measure: counters, anthems, auras, equipment and token "
               "counts are invisible to the log, so a board-power ranking would "
               "be biased against exactly the decks that build one. The goldfish "
               "publishes `combat.mean_board_power_by_turn`, a mean and not a "
               "distribution — and a mean is not a result."),

    "tokens by type": _m(
        "board",
        "Created per game, split: creature, treasure, blood, clue, food, other.",
        UNAVAILABLE, None,
        None,
        absent="Forge token detection is a NAME SUFFIX — `\\bToken$` — so a token "
               "has an id and nothing else. No type, no colour, no power. The "
               "same gap makes `scenario-facts` file every lifted token under "
               "`other_permanents`. Totals ARE published "
               "(analysis.seats[].tokens); the split is not."),

    "counter frequency": _m(
        "board",
        "+1/+1 and -1/-1 counters placed per game, by source.",
        PUBLISHED, "forge",
        "sim/<run>.json → analysis.seats[].counter_events / mass_counter_events "
        "/ proliferate_events",
        caveat="EVENTS, not counters, and NOT split by kind or by source. The log "
               "says an ability that places counters resolved; it does not say "
               "how many, which kind, or on what. Attributed to the tagged seat "
               "where the line carries one and to the active seat otherwise."),

    "anthem-adjusted power": _m(
        "board",
        "Total board power with static pump effects applied.",
        UNAVAILABLE, None,
        None,
        absent="Continuous effects are not tracked by either engine. The bridge "
               "says so out loud — a Craterhoof pump must be AUTHORED into a "
               "lifted scenario, and `extras.reconstruction_notes` records it."),

    # ── Speed ───────────────────────────────────────────────────────────────
    "commander resolve turn": _m(
        "speed",
        "Turn the commander first resolves; share resolved by turn 5.",
        PUBLISHED, "goldfish",
        "goldfish_metrics.json → metrics.commander.mean_cast_turn / "
        "cast_by_turn_6_rate / cast_turn_histogram",
        caveat="By turn SIX is what is published, not by turn five; the "
               "histogram carries every turn, so the five figure is derivable "
               "from it without a new measurement."),

    "first payoff turn": _m(
        "speed",
        "Turn the deck's stated engine first produces its payoff.",
        PUBLISHED, "goldfish",
        "goldfish_metrics.json → metrics.targets[].by_turn_6_rate",
        caveat="'Stated' is the load-bearing word and the reason this figure may "
               "never GRADE anything: the targets come from the authored "
               "`goldfish_targets.json`, so the same hand writes the declaration "
               "and reads the verdict. `deck_branch.MEMBERSHIP_AXES` refuses "
               "these as branch objectives for exactly that reason — three "
               "defensible declarations of one list gave +0.007, -0.036 and "
               "+0.014 against the same 10,000 games."),

    "turn to lethal, goldfish": _m(
        "speed",
        "Unopposed kill turn, no interaction.",
        OPT_IN, "goldfish",
        "goldfish_metrics.json → metrics.combat.mean_kill_turn / "
        "median_kill_turn / kill_by_turn_rate",
        caveat="OPT-IN behind `model_combat`, and present in TWO of the ten "
               "tracked goldfish artifacts. THE GOLDFISH HAS NO BLOCKERS, so "
               "this is a clock and never a win "
               "rate. It is measured against ONE opponent at 40 life who never "
               "interacts. A go-wide deck reads far better here than in Forge: "
               "1/1 tokens do not connect against a real board, and a refactor "
               "the goldfish preferred lost 31/400 against 50/400."),

    "threat-to-lethal gap": _m(
        "speed",
        "Turns between board reading as lethal-capable and lethal landing.",
        UNAVAILABLE, None,
        None,
        absent="'Reading as lethal-capable' needs a board state per turn, which "
               "Forge cannot supply. " + FORGE_ZONE_LIMIT + " The captain's log "
               "asks for it directly — ur-dragon 003, 'I ramped fast enough to "
               "broadcast the win before I could actually present lethal' — so "
               "this is a real want with no route today."),

    # ── Resilience ──────────────────────────────────────────────────────────
    "post-wipe recovery": _m(
        "resilience",
        "Turns to return to pre-wipe board power.",
        UNAVAILABLE, None,
        None,
        absent="What IS published is `analysis.wipe_recovery` — damage dealt on "
               "the wipe turn and over the two turns after it. That is value on "
               "the way down, not board recovery, and the difference is not "
               "cosmetic. Board size before and after is impossible: Forge logs "
               "a permanent LEAVING the battlefield and never one arriving, so a "
               "reconstructed board series would be zeroes wearing the name of a "
               "measurement."),

    "value on creature death": _m(
        "resilience",
        "Cards drawn, damage dealt, and life gained triggered by own creatures "
        "dying.",
        UNAVAILABLE, None,
        None,
        absent="The three components exist separately in a run record — "
               "`creatures_lost`, `life_gained`, `noncombat_damage_dealt_to_players` "
               "— and nothing ties any of them to a death TRIGGER. The log records "
               "that an ability resolved, never what caused it. Cards drawn is "
               "unrecoverable outright. " + FORGE_ZONE_LIMIT),

    "commander uptime": _m(
        "resilience",
        "Share of turns after first resolve with the commander on battlefield.",
        UNAVAILABLE, None,
        None,
        absent="Forge logs the commander leaving the battlefield and never "
               "arriving, so time ON it cannot be reconstructed. `bridge.py` "
               "tracks a commander's zone for ONE lifted board and never across "
               "a run. Heliod's whole open question — losing the commander takes "
               "the engine with it — is unmeasured for this reason."),

    # ── Interaction ─────────────────────────────────────────────────────────
    "removal used vs held": _m(
        "interaction",
        "Interaction cast against interaction still in hand at loss.",
        PUBLISHED, "goldfish",
        "diagnostic.json → steam.interaction_in_hand_by_turn and "
        "steam.castable_given_in_hand_by_turn",
        caveat="The goldfish answers the half that matters and states it "
               "differently: of the games where an answer was IN HAND on a given "
               "turn, the share where the turn also ended with the mana to cast "
               "it. Conditional on purpose — the raw castable rate falls when you "
               "simply cut interaction, which answers a different question. Forge "
               "publishes `interaction_cast`, which is a floor at 59% attribution "
               "and is NOT a removal count: the log says a spell targeted "
               "something, never what it did."),

    "protection available at threat": _m(
        "interaction",
        "Share of turns holding a protection effect and the mana to cast it.",
        PUBLISHED, "goldfish",
        "diagnostic.json → steam.castable_given_in_hand_by_turn",
        caveat="The suite is removal + sweepers + protection together, the same "
               "set `deck-audit` counts — protection is not separated out. And it "
               "is a FLOOR: the model casts everything it can afford every turn, "
               "so a pilot choosing to hold up scores higher."),

    "opposing threats answered": _m(
        "interaction",
        "Opponent permanents removed per game.",
        UNAVAILABLE, None,
        None,
        absent="`interaction_cast` counts stack objects a seat aimed at another "
               "seat's permanent or face — once per opposing seat touched, at 59% "
               "attribution coverage. It cannot become this figure: a Swords, an "
               "edict, a drain and a pump spell aimed at an opponent's creature "
               "are the same event to the log, and only one of them removes "
               "anything."),

    # ── Outcome ─────────────────────────────────────────────────────────────
    "win rate": _m(
        "outcome",
        "Wins over games, reported with interval.",
        PUBLISHED, "forge",
        "sim/<run>.json → analysis.seats[].win_rate and .win_rate_ci95; "
        "summary.win_rate",
        caveat="The denominator is DECIDED games, not games: a clock-out is "
               "`truncated`, has no winner and is excluded. And Forge's AI pilots "
               "the deck — 'poor to ok in control, pretty bad for combo', its own "
               "words, quoted in every run record — so a control deck's rate is a "
               "LOWER BOUND on the pilot and a combo deck's is not a measurement."),

    "placement": _m(
        "outcome",
        "Finishing position distribution.",
        PUBLISHED, "forge",
        "sim/<run>.json → analysis.seats[].placement.by_position / "
        "mean_position, from each game's per-seat eliminated_turn",
        caveat="1 is the seat still standing; the rest are ordered by how late "
               "they were eliminated. A clock-out has no ordering at all and is "
               "EXCLUDED rather than filed as a draw at position 1, so "
               "`games_ranked` is below the game count."),

    "damage by source": _m(
        "outcome",
        "Dealt and taken, split: combat, direct, drain, ping, commander.",
        PUBLISHED, "forge",
        "sim/<run>.json → analysis.seats[].combat_damage_dealt_to_players / "
        "noncombat_damage_dealt_to_players / combat_damage_taken / "
        "damage_dealt_total / commander_damage",
        caveat="THREE OF THE FIVE SPLITS EXIST — combat, noncombat and commander. "
               "Drain and ping are not separable: drain is LIFE LOSS rather than "
               "damage and appears in no damage total at all, showing only in "
               "`life_by_turn` and `eliminated_how='life loss'`. Measured: vito "
               "won 9 of 20 games on 7.0 combat damage a game."),

    "seat effect": _m(
        "outcome",
        "Win rate by turn order position.",
        PUBLISHED, "forge",
        "sim/<run>.json → analysis.seats[].turn_order, from the `Turn: Turn 1 "
        "(seat)` line that names who actually went first",
        caveat="NOT `outcomes[].seat_order`, which is the `-d` order and is the "
               "wrong answer. Forge's `determineFirstTurnPlayer` gives the first "
               "turn, from game 2 of a job onward, to the lowest-indexed seat "
               "that DID NOT WIN the previous game — so the deck that loses most "
               "starts most, and a figure built from `-d` position reports "
               "backwards. Measured on the 400-game run: edgar-vampires started "
               "**81%** of games. `started_rate` therefore travels beside the "
               "split, because a seat starting far more than its share is being "
               "measured on its own losing streak rather than on turn order. "
               "Seat order ROTATES PER JVM JOB, not per game, because Forge's "
               "`Match` carries `lastOutcome` and picks the next first player "
               "from the previous game's loser. With four jobs and four seats "
               "every position is covered once; with fewer jobs than seats they "
               "are not, and games inside a job are a Markov chain rather than "
               "independent draws."),
}


#: PRD §2's table, which is the reason the catalog exists: six problems observed
#: at a real pod night, and the metric each one needs. Kept here rather than in
#: prose because it turns the catalog into an ANSWER — "can the bench measure the
#: thing that actually lost me the game" — and because a mapping can be checked.
#:
#: Every deck named is the deck the log names. `docs/prd.md` §2 is the source.
PROBLEMS = (
    ("ran out of gas late", ("edgar-vampires",),
     ("turns with empty hand", "draw-engine uptime")),
    ("3-5 lands behind curve", ("edgar-vampires", "ur-dragon"),
     ("missed land drops", "mean available mana by turn")),
    ("threat telegraphed before lethal", ("ur-dragon",),
     ("threat-to-lethal gap",)),
    ("kept a hand with no goblins", ("goblin-storm",),
     ("keepable sevens",)),
    ("board dies to wipes, no value", ("edgar-vampires", "gishath"),
     ("value on creature death", "post-wipe recovery")),
    ("lost commander, engine offline", ("heliod",),
     ("commander uptime", "protection available at threat")),
)


def answerable():
    """Which of PRD §2's observed problems the bench can measure today.

    `full` — every metric it needs exists. `partial` — some do. `none` — none.
    """
    out = []
    for problem, decks, needs in PROBLEMS:
        have = [n for n in needs if CATALOG[n]["status"] != UNAVAILABLE]
        state = ("full" if len(have) == len(needs)
                 else "none" if not have else "partial")
        out.append({"problem": problem, "decks": list(decks),
                    "needs": list(needs), "have": have, "state": state})
    return out



def by_group(group):
    """Every entry in one of PRD §14's groups, in catalog order."""
    return {k: v for k, v in CATALOG.items() if v["group"] == group}


def by_status(status):
    """Every entry in one of the three states."""
    return {k: v for k, v in CATALOG.items() if v["status"] == status}


def unavailable():
    """What no engine here can answer, each with its measured reason.

    The most useful view in this module: it is the honest answer to "does the
    simulation measure X", and every entry names why rather than saying no.
    """
    return by_status(UNAVAILABLE)


def coverage():
    """`{group: {status: count}}` — the catalog as a scoreboard."""
    out = {g: {s: 0 for s in STATUSES} for g in GROUPS}
    for row in CATALOG.values():
        out[row["group"]][row["status"]] += 1
    return out


# ── the CLI view ────────────────────────────────────────────────────────────

def format_catalog(group=None, status=None, verbose=False):
    """The catalog, as the pilot reads it. One line per figure, grouped."""
    lines = []
    cov = coverage()
    shown = 0
    for g in GROUPS:
        if group and g != group:
            continue
        rows = {k: v for k, v in by_group(g).items()
                if not status or v["status"] == status}
        if not rows:
            continue
        counts = "  ".join(f"{s} {cov[g][s]}" for s in STATUSES if cov[g][s])
        lines.append(f"\n{g.upper().replace('_', ' ')}   ({counts})")
        for name, row in rows.items():
            shown += 1
            mark = {PUBLISHED: "\u25c6", OPT_IN: "\u25c7",
                    DERIVABLE: "\u00b7", UNAVAILABLE: "\u2014"}[row["status"]]
            engine = row["engine"] or ""
            lines.append(f"  {mark} {name:<32} {engine:<9} {row['status']}")
            if verbose:
                lines.append(f"      {row['definition']}")
                if row["source"]:
                    lines.append(f"      from: {row['source']}")
                for key in ("caveat", "absent"):
                    if row.get(key):
                        lines.append(f"      {key}: {row[key]}")
                lines.append("")
    lines.append(f"\n  {shown} of {len(CATALOG)} shown.  "
                 "\u25c6 published  \u25c7 opt-in (a per-deck model flag)  "
                 "\u00b7 derivable  \u2014 unavailable")
    lines.append("  An UNAVAILABLE figure is absent, never zero, and says why. "
                 "Two engines answer")
    lines.append("  different questions: Forge logs no library and no board "
                 "arrival, so every")
    lines.append("  Mana and Card-flow figure is the goldfish's.")
    return "\n".join(lines)


def format_problems():
    """PRD §2's table with today's answer beside each row."""
    rows = answerable()
    mark = {"full": "\u2713", "partial": "~", "none": "\u2717"}
    lines = ["\nWHAT THE POD NIGHTS ASKED FOR (PRD \u00a72)\n"]
    for row in rows:
        lines.append(f"  {mark[row['state']]} {row['problem']:<34} "
                     f"{', '.join(row['decks'])}")
        for need in row["needs"]:
            status = CATALOG[need]["status"]
            flag = "  " if status != UNAVAILABLE else " \u2014"
            lines.append(f"     {flag} {need} ({status})")
    full = sum(1 for r in rows if r["state"] == "full")
    part = sum(1 for r in rows if r["state"] == "partial")
    lines.append(f"\n  {full} of {len(rows)} fully answerable, {part} partly. "
                 "`metrics --status unavailable` says why not.")
    return "\n".join(lines)


def main(args):
    import json as _json

    if getattr(args, "problems", False):
        if getattr(args, "as_json", False):
            print(_json.dumps(answerable(), indent=2, ensure_ascii=False))
        else:
            print(format_problems())
        return

    group = getattr(args, "group", None)
    status = getattr(args, "status", None)
    if getattr(args, "as_json", False):
        rows = {k: v for k, v in CATALOG.items()
                if (not group or v["group"] == group)
                and (not status or v["status"] == status)}
        print(_json.dumps({"catalog": rows, "coverage": coverage()},
                          indent=2, ensure_ascii=False))
        return
    print(format_catalog(group, status, verbose=getattr(args, "verbose", False)))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot metrics`.")
