"""The goldfish's CARD READERS — what a card is, from its oracle text.

MOVED OUT OF `goldfish.py` ON 2026-09-13, verbatim. 194 module-level names,
about 120 regexes, and not one of them touches a game state: every function
here is a pure function of a card, which is what makes the boundary real rather
than a place the file happened to be cut.

The simulator imports them; nothing here imports the simulator. That direction
is the whole point — `goldfish.py` was 5,830 lines and the classifiers were
two thirds of it, so a reader looking for the turn loop had to scroll past
every regex in the project first.

EVERY LESSON THESE CARRY IS IN `docs/gotchas-bench.md` and in the docstrings
below: the trigger pattern whose `.*` ate the subject noun, the condition scoped
to the wrong clause, the fetchland whose colours are a property of the DECK, the
`.*` in `token_bodies` that credits a free body. They were paid for one at a
time and they are why this file is long.
"""

import contextlib
import json
import pathlib
import random
import re
from manamap import console
from manamap.config import (
    GOLDFISH_ITERATIONS,
    GOLDFISH_MAX_MULLIGANS,
    GOLDFISH_MAX_TURN,
    GOLDFISH_MULLIGAN_MAX_LANDS,
    GOLDFISH_MULLIGAN_MIN_LANDS,
    GOLDFISH_OPPONENT_LIFE,
    GOLDFISH_POISON_TO_LOSE,
    GOLDFISH_SEED,
)
from manamap.pilot import manabase
from manamap.pilot.common import (
    deck_dir, deck_file, front_field, load_deck_cards)


_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "x": 0,
}

# Token types that are NOT creatures. `_TOKEN_RE` matches "create ... token"
# generically, so without this list a Treasure scores as a body — measured on
# ur-dragon, that was 37% of the reported turn-six board.
_NONCREATURE_TOKENS = ("treasure", "clue", "food", "blood", "gold", "powerstone",
                       "map", "incubator", "junk", "shard")

_TOKEN_RE = re.compile(r"create (\w+)(?: [\w/+-]+)* tokens?", re.IGNORECASE)
# A TAP-FOR-MANA ABILITY, WRITTEN THREE WAYS, AND THE FIRST CUT SAW ONE.
# The old pattern was `\{T\}: Add ((?:\{[WUBRGC0-9]\})+)` — an explicit symbol
# list and nothing else. So `{T}: Add {C}{C}` parsed and `{T}: Add one mana of
# any color` did not, which is **Arcane Signet, Birds of Paradise, Relic of
# Legends and Sanctum Weaver**: 71 of the fleet's 110 tap-for-mana cards, 65%,
# reading zero. ur-dragon's model could see 2 of its 11 non-land mana, and
# turn-seven mana came out ~19% low on every deck measured.
#
# It is NOT a stated assumption — the module's assumption list says rituals and
# cost reducers are unmodelled and says nothing about rocks, because nobody
# knew. A silent half-working regex is the most expensive kind: it produces a
# number, the number is plausible, and it is wrong by a fifth.
#
# `[^:\n]*` lets a cost precede the tap (`{1}, {T}: Add …`) while refusing to
# cross a colon or a line, so a `{T}` in one ability cannot bind to an `: Add`
# in another. The `{T}` requirement is load-bearing and stays: Phyrexian Altar
# is `Sacrifice a creature: Add one mana`, which is not free repeatable mana and
# must not be counted as a rock.
_TAP_ADD_RE = re.compile(r"\{T\}[^:\n]*: ?Add ([^.\n]+)", re.IGNORECASE)
#: A COST THAT CONSUMES THE PERMANENT IS NOT A RATE. Widening the pattern to
#: catch "Add one mana of any color" also caught **Jeweled Lotus**, whose
#: ability reads `{T}, Sacrifice this artifact: Add three mana` — the model
#: would have collected three mana from it every turn, forever. Same for
#: Kaleidostone, Lotus Bloom and Transmogrant Altar. `produced_mana` answers
#: "per turn, repeatably", so an ability that eats its own source, exiles or
#: discards is worth zero here and belongs to a one-shot channel that does not
#: exist. A mana cost in the activation ({1}, {T}: …) is fine — filters are
#: real rocks.
_CONSUMING_COST = re.compile(r"sacrifice|exile|discard", re.IGNORECASE)
#: Mana that can only be spent on some things. See the meta note in `run`.
_RESTRICTED_MANA_RE = re.compile(r"Spend this mana only", re.IGNORECASE)
#: Any card that draws beyond the draw step. The COUNT — 12 on edgar-vampires,
#: against a modelled 0 for the first year this file existed.
_DRAW_RE = re.compile(r"\bdraw (a|two|three|four|X|that many) card", re.I)

# ── Card draw ─────────────────────────────────────────────────────────────
#
# WHY THIS ARRIVED LATE AND WHAT IT IS FOR. `card_advantage` reported
# `{"cards_that_draw": 12, "modelled": 0}` and the loop drew exactly one card a
# turn whatever the list said, so two decks differing by twelve draw spells
# goldfished identically. The pilot's most-repeated table failure — "vampires on
# board, nothing in hand, no way to rebuild" — was the one thing the model was
# structurally incapable of seeing.
#
# THE SWEEP IS WHY THE TIERS ARE WHERE THEY ARE. 3,942 corpus cards draw. Sorted
# by how the text words it:
#
#     other, not modelled     2031   death/attack/discard-triggered, conditional
#     activated, not modelled 1001   "{1}{B}, {T}: Draw a card" — needs a policy
#     ETB, modelled            348   "When ~ enters, you draw a card"
#     X-based, not modelled     302   "draw X cards where X is..." — board-dependent
#     spell, modelled           187   an instant or sorcery that draws N
#     recurring, modelled        73   "at the beginning of your upkeep, draw"
#     arrival, modelled          33   "whenever a creature you control enters, draw"
#
# So 641 of 3,942 are priced and 3,301 are NAMED. That ratio is the honest state
# of it and `draw_not_modelled` carries the names per deck, the same contract
# `treasure_sources_not_modelled` and `combat_effects_not_modelled` already keep.
# Measured on edgar-vampires the ratio is brutal and is a FINDING rather than a
# defect: of its twelve, exactly ONE (Night's Whisper) is unconditional.
#
# THE ARRIVAL CHANNEL IS THE ONE THAT EARNS ITS KEEP. It rides the same single
# door onto the battlefield the ETB damage payoffs use, so "bodies convert into
# cards" — Welcoming Vampire, Caretaker's Talent, Tocasia's Welcome — becomes a
# measurable claim instead of a hope.
_ETB_DRAW_RE = re.compile(
    r"when(?:ever)? (?:this creature|this artifact|this enchantment|"
    r"[A-Z][\w' ,-]{2,30}) enters[^.]{0,60}?,? (?:you )?draw "
    r"(a|one|two|three|four|five) cards?", re.I)
_RECURRING_DRAW_RE = re.compile(
    r"at the beginning of your (?:upkeep|draw step|end step)[^.]{0,80}?,? "
    r"(?:you )?draw (a|one|two|three) cards?", re.I)
#: Anchored to a SENTENCE START and requiring "draw", never "draws" — otherwise
#: "target player draws a card" and "each player draws" score as your own draw.
# DRAW EQUAL TO THE GREATEST POWER (Rishkar's Expertise, Return of the
# Wildspeaker): seven corpus cards, none of them read until 2026-09-11 because
# `_DRAW_RE` wants a number word. Resolved at cast against the board.
_DRAW_GREATEST_POWER_RE = re.compile(
    r"draw cards equal to (?:the greatest power among (?:creatures|other creatures|"
    r"non-Human creatures) you control|(?:the|that creature's|its) power"
    r"(?: of target creature you control)?)", re.I)
# DRAW A CARD FOR EACH <TYPE> ON ENTRY (Earthshaker Dreadmaw): ten corpus
# cards. `_ETB_DRAW_RE` read the "draw a card" and priced it at ONE.
_ETB_DRAW_PER_TYPE_RE = re.compile(
    r"when (?:this creature|this artifact|this enchantment|[A-Z][\w' ,-]{2,30}) "
    r"enters, draw a card for each (?:other )?([A-Z][a-z]+) you control", re.I)
_SPELL_DRAW_RE = re.compile(
    r"(?:^|\.\s|^\s*)(?:you )?draw (a|one|two|three) cards?", re.I)
#: "Scry 2, then draw two cards" (Read the Bones) -- the draw is the second
#: clause of its sentence and the sentence-anchored pattern above never saw
#: it. Corpus sweep 2026-09-10: 34 instants and sorceries, ONE of which was
#: read; a card-advantage branch on ingris-infect added Read the Bones and
#: the draw axis did not move. Locked by test.
_SCRY_THEN_DRAW_RE = re.compile(
    r"(?:scry|surveil) \d+, then draw (a|one|two|three) cards?", re.I)
#: "At the beginning of your upkeep, reveal the top card of your library and
#: put that card into your hand" -- Dark Confidant, Dark Tutelage, Darkstar
#: Augur: a recurring draw that never says "draw". Three cards; locked.
_UPKEEP_REVEAL_RE = re.compile(
    r"at the beginning of your upkeep, reveal the top card of your library"
    r"(?: and)? (?:and )?put (?:that card|it) into your hand", re.I)
#: THE BODIES-INTO-CARDS FAMILY. 33 cards, and the qualifier between "you
#: control" and "enters" is load-bearing in BOTH directions: Welcoming Vampire
#: draws off a 1/1 token ("power 2 or less") and Garruk's Uprising must not
#: ("power 4 or greater"). Reading the trigger and ignoring its condition would
#: hand every token deck a draw engine it does not have.
_ARRIVAL_DRAW_RE = re.compile(
    r"whenever (?:this creature or )?(?:another |a |one or more )?"
    r"(?:other )?(?:nontoken )?([\w' ]{0,28}?)you control"
    r"([^.,]{0,44}?)enters?[^.]{0,60}?,\s*(?:you )?draw (a|one|two) cards?", re.I)
#: CAST-TRIGGERED DRAW — THE ENCHANTRESS CHANNEL, and it read as NOTHING.
#:
#: "Whenever you cast an enchantment spell, you may draw a card" matched no
#: pattern in this file: not `_ETB_DRAW_RE` (needs "enters"), not
#: `_RECURRING_DRAW_RE` (needs an upkeep), not `_ARRIVAL_DRAW_RE` (needs "you
#: control … enters"), and not `_SPELL_DRAW_RE`, which only runs on instants and
#: sorceries. So Mesa Enchantress — the single most-included card in the Zur,
#: Eternal Schemer meta at 72.6% of ~6,703 decks — was `unmodelled` on a list
#: with 44 enchantments, and adding it to a branch measured as a LOSS.
#:
#: The first diagnosis was wrong and is recorded because it was expensive: the
#: "you may" in its text looked like the culprit, since `_DRAW_CONDITIONAL_RE`
#: rejects it. Deleting "you may" from the oracle text and re-profiling still
#: read nothing. There was no channel at all.
#:
#: CORPUS SWEEP 2026-09-09 — 68 cards, and the TYPE GATE is load-bearing in both
#: directions: Beast Whisperer must not draw off an enchantment and Mesa
#: Enchantress must not draw off a creature. By gate:
#:
#:   19 (any)   11 creature   8 instant-or-sorcery   5 enchantment   4 artifact
#:    3 aura     3 noncreature  2 legendary  2 historic  and 12 one-offs
#:      (druid, doctor, hero, loud, kicked, multicolored, blue permanent,
#:       dragon-or-omen, adventure, eldrazi creature, spirit-or-arcane)
#:
#: ONLY FOUR GATES ARE MODELLED, and the rule is not "which can I write a regex
#: for" but WHICH ONES DOES THIS MODEL SEE EVERY CAST OF. A cast trigger fires
#: in this simulation at the two places a non-land permanent joins the
#: battlefield, so a gate whose spells are permanents is counted completely.
#: `(any)`, `noncreature` and `instant or sorcery` are NOT modelled even though
#: their regex is trivial, because this model casts few instants and sorceries
#: and would under-report those engines by an unknown amount — a wrong number
#: rather than an absent one. `legendary`, `historic` and the twelve tribal and
#: mechanic gates go to `unmodelled` because the type line does not settle them.
#: 23 of 68 modelled, 45 named and refused.
_CAST_DRAW_RE = re.compile(
    r"whenever you cast (?:a|an|another) ([\w' -]{0,28}?)spell[^.]{0,50}?,\s*"
    r"(?:you may )?draw (a|one|two|three) cards?", re.I)
#: gate -> the substring that must appear in the CAST card's type line.
_CAST_DRAW_GATES = {"enchantment": "Enchantment", "creature": "Creature",
                    "artifact": "Artifact", "aura": "Aura"}
_DRAW_POWER_MAX_RE = re.compile(r"power (\d+) or less", re.I)
_DRAW_POWER_MIN_RE = re.compile(r"power (\d+) or greater", re.I)
_DRAW_ONCE_RE = re.compile(r"once each turn", re.I)
#: A qualifier this model cannot evaluate. Named, never guessed: "with defender",
#: "of the chosen type", "named Gladewalker Ritualist", "with mana value 3 or
#: less". Firing on these would invent a draw engine; ignoring them silently
#: would hide one.
_DRAW_QUALIFIER_OK_RE = re.compile(
    r"^\s*(?:with power \d+ or (?:less|greater)\s*)?$", re.I)
#: A DRAW THIS MODEL CANNOT PROMISE. 39 of the 348 ETB-draw matches carry a
#: condition inside the trigger itself — "if you control an artifact", "if
#: you've cast two or more spells this turn", "you MAY draw" — and Selvala's is
#: not even your draw ("its controller may draw a card"). Reading the trigger
#: and ignoring its gate is the same defect the ETB life-loss channel above was
#: built to avoid, one clause further in. They go to `unmodelled`.
_DRAW_CONDITIONAL_RE = re.compile(r"\b(?:if|unless|you may|its controller)\b", re.I)
#: A COST THIS MODEL CANNOT PAY. 39 instants and sorceries word their draw as
#: "As an additional cost to cast this spell, sacrifice a creature. Draw two
#: cards" — Village Rites, Deadly Dispute, Altar's Reap, Costly Plunder. The
#: sentence-anchored spell pattern reads the second sentence and sees a free
#: draw-2. There is no sacrifice in this model and no discard, so the cost is
#: unpayable and the card is `unmodelled`, not free. Caught on the FIRST branch
#: measured with the draw model on: edgar-vampires' drain refactor adds both
#: Village Rites and Deadly Dispute, and they are precisely the cards whose
#: whole point is that they COST a body.
_DRAW_ADDITIONAL_COST_RE = re.compile(
    r"as an additional cost to cast", re.I)
_DRAW_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# ── Discard: wheels, loots, and the triggers that pay on a discard or a draw ──
#
# THE SIX WHEELS IN sharknado WERE INVISIBLE, NOT DARK. `_DRAW_RE` wants a
# written-out "draw N card"; Wheel of Fortune says "draws seven cards" and
# Windfall "draws cards equal to the greatest number", so both returned an
# all-zero profile with `unmodelled` still None -- the one value that means
# "nothing to see" -- and no casting loop ever selected them. The artifact read
# `cards_that_draw: 35, modelled: 2` on a deck whose plan is drawing. Corpus
# sweep 2026-09-10: 11 "each player discards their hand, then draws"; 24
# "discard your hand, then draw N".
_WHEEL_RE = re.compile(
    r"(?:each player |you |^|\. )(discards?|shuffles?) (?:all the cards in |the cards from )?(?:their|your) hand"
    r"(?: and graveyard)?(?: into (?:their|your) librar(?:y|ies))?, then draws? "
    r"(seven|six|five|four|three|two|a|one|X|that many|cards equal to the greatest number)",
    re.I)
#: The LOOT rider on a spell's own draw: "Draw two cards, then discard two
#: cards" (Faithless Looting). The X-path riders (`_X_DRAW_DISCARD_*`) do not
#: match it -- they know "a card" and "X cards" -- so Looting read +2 and
#: discarded nothing.
_DRAW_THEN_DISCARD_RE = re.compile(
    r"then discards? (a|one|two|three) cards?", re.I)
#: The triggers. 19 corpus cards say "whenever you discard a card", 11 "one or
#: more cards"; 48 say "whenever you draw a card"; 54 "your second card each
#: turn". Parsed, not declared: well past the "one card, no pattern" line that
#: makes an ability a per-deck declaration.
_EVENT_TRIGGER_RE = re.compile(
    r"whenever you (discard a card|discard one or more cards|draw a card|"
    r"draw your second card each turn)\b,?([^.\n]*(?:\.[^.\n]*)?)", re.I)
#: "deals N damage to each opponent" (Brallin), "deals N damage to any target"
#: (Niv-Mizzet, Irencrag -- one opponent, which is all this model tracks) and
#: "each opponent loses N life" (Psychosis Crawler -- life loss is damage
#: against one seat at 40, the convention the arrival channel already uses).
_EVENT_DAMAGE_RE = re.compile(
    r"deals (\d+) damage to (?:each opponent|any target|each opponent and each)"
    r"|each opponent loses (\d+) life", re.I)
_EVENT_COUNTER_RE = re.compile(r"put a \+1/\+1 counter on", re.I)
_EVENT_DRAW_RE = re.compile(r"\bdraw (a|two) cards?", re.I)
_EVENT_TOKEN_RE = re.compile(r"create an? (\d+)/(\d+) [^.]*?creature token", re.I)
#: Conditional and self-damaging: the lowest chooser does not wheel at all.
_WHEEL_EXCLUDED = frozenset({"Wheel of Misfortune"})
#: A wheel is cast only when the hand is this thin (nonland cards) or a discard
#: payoff is already on the battlefield. AUTHORED: without it the draw loop,
#: cheapest first, wheels away six good cards on turn three.
WHEEL_MIN_HAND = 3


def event_payoffs(card):
    """What this card pays on a discard or a draw, and the shape of each.

    Keyed like the drain pillar's `payoff_fixed`: a FLAT amount per EVENT.
    `second_draw_*` fires once per turn on the second draw. A payoff that
    DRAWS on a draw (Curiosity-class loops) is deliberately not read: a
    goldfish with no cap would draw its whole library and call it steam.
    """
    text = card.get("oracle_text", "") or ""
    out = {"per_discard_damage": 0, "per_discard_counter": 0,
           "per_discard_draw": 0, "per_discard_token_power": 0,
           "per_draw_damage": 0, "per_draw_counter": 0,
           "per_draw_token_power": 0,
           "second_draw_damage": 0, "second_draw_token_power": 0,
           "unmodelled": None}
    hit = False
    for m in _EVENT_TRIGGER_RE.finditer(text):
        hit = True
        kind, effect = m.group(1).lower(), m.group(2) or ""
        prefix = ("per_discard" if kind.startswith("discard")
                  else "second_draw" if "second" in kind else "per_draw")
        dmg = _EVENT_DAMAGE_RE.search(effect)
        if dmg:
            out[prefix + "_damage"] += int(dmg.group(1) or dmg.group(2))
        if prefix != "second_draw":
            if _EVENT_COUNTER_RE.search(effect):
                out[prefix + "_counter"] += 1
            # "you gain N life" is NOT read: this model has no own life total,
            # and a field nothing applies is the silent zero this file exists
            # to avoid.
            drw = _EVENT_DRAW_RE.search(effect)
            if drw and prefix == "per_discard":
                out["per_discard_draw"] += _DRAW_WORDS[drw.group(1).lower()]
        tok = _EVENT_TOKEN_RE.search(effect)
        if tok:
            out[prefix + "_token_power"] += int(tok.group(1))
    if hit and not any(v for k, v in out.items() if k != "unmodelled"):
        out["unmodelled"] = card.get("name")
    return out


def has_event_payoff(profile):
    return any(v for k, v in (profile or {}).items() if k != "unmodelled")


# ── Sacrifice and death ───────────────────────────────────────────────────
#
# THE HOLE THIS FILLS IS THE BIGGEST ONE THIS MODEL HAS EVER HAD. Nothing died
# here: no blockers, no removal, no sacrifice outlets. So every death-triggered
# card in a deck was priced at exactly zero, and on edgar-vampires' drain
# refactor that is TWENTY OF NINETY-NINE CARDS — Blood Artist, Zulaport
# Cutthroat, Cruel Celebrant, Bastion of Remembrance, Viscera Seer, Ashnod's
# Altar, Phyrexian Tower, Skullclamp, Woe Strider. The deck's entire stated
# engine, contributing nothing to any published figure.
#
# WHAT IS MODELLED: a FREE, repeatable sacrifice outlet converting creature
# TOKENS into whatever the death payoffs in play pay out — life loss (which is
# damage here, same as the arrival channel), a card, or a Treasure.
#
# WHAT IS NOT, AND WHY THE POLICY IS THE HARD PART. A real pilot sacrifices in
# response to a wipe or for lethal, and this model has neither. Any fixed rule
# is wrong somewhere: keep every token and the drain never fires; sacrifice
# every token and the board never grows. So the rule is stated rather than
# tuned, and the two extremes BRACKET the truth — a run without the flag is the
# floor, a run with it is the ceiling, and the deck's real value is between.
#
#   THE POLICY: after combat has swung, sacrifice creature TOKENS to a free
#   outlet while any death payoff is in play. Nontoken creatures are never
#   sacrificed — a pilot does not feed Blood Artist to the Altar.
#
# Sacrificing after `attackers` is snapshotted is what makes it a conversion of
# a token that has ALREADY attacked rather than a trade against this turn's
# swing.
_FREE_SAC_OUTLET_RE = re.compile(
    r"(?:^|[.\n] )sacrifice (?:a|another) "
    r"(?:creature|creature or artifact|artifact or creature)[^:.\n]{0,20}:",
    re.IGNORECASE)
#: A COSTED OUTLET IS NOT A FREE ONE. 180 corpus cards put mana or a tap symbol
#: in front of the colon — Phyrexian Tower, Indulgent Aristocrat, Acolyte of
#: Aclazotz — and a tap symbol also caps it at once a turn. Counting those as
#: free would hand this deck an engine it has to pay for; they are NAMED in
#: `meta.sacrifice` instead. 48 outlets in the corpus are genuinely free.
_COSTED_SAC_OUTLET_RE = re.compile(
    r"(?:\{[^}]+\}|\{T\})[^:.\n]{0,40}?sacrifice (?:a|an|another)[^:.\n]{0,30}:",
    re.IGNORECASE)
#: "Whenever [this creature or] another creature you control dies". The self-only
#: form ("When THIS creature dies") is deliberately not read: this model only
#: sacrifices tokens, and a token is never the card carrying the trigger.
_DEATH_TRIGGER_RE = re.compile(
    r"whenever (?:this creature or another|another|a|one or more)"
    r"[\w' ]{0,26}?(?:you control )?dies", re.IGNORECASE)
_DEATH_DRAIN_RE = re.compile(
    r"each opponent loses (\d+) life|target player loses (\d+) life",
    re.IGNORECASE)
_DEATH_DRAW_RE = re.compile(r"(?:you )?draw (a|one|two) cards?", re.IGNORECASE)
_DEATH_TREASURE_RE = re.compile(r"create a treasure token", re.IGNORECASE)
#: A runaway guard, the same shape as `ETB_CHAIN_LIMIT`. A death payoff that
#: makes a token is a loop, and a loop that terminates silently cannot be told
#: from one that never ran.
SAC_LIMIT_PER_TURN = 20


#: EMINENCE THAT MINTS A BODY, and the reason this file existed for a year
#: without it: `command_zone_reduction` reads the commander for COST REDUCTION
#: (The Ur-Dragon's eminence), and Edgar Markov's eminence does something else
#: entirely — "whenever you cast another Vampire spell, if Edgar is in the
#: command zone or on the battlefield, create a 1/1 black Vampire creature
#: token". It is live from turn one, it cannot be removed, and it is the deck's
#: whole token engine. Unmodelled, every one of those tokens was missing: the
#: bodies, the arrival-damage payoffs they fire, the arrival DRAW they fire, and
#: the fuel the sacrifice model eats. `deck-audit`'s engine brief describes it in
#: prose and the simulation could not see it.
#:
#: 92 corpus cards carry this shape and exactly ONE is a commander on this
#: bench, so implementing it moves edgar-vampires and no other deck.
_CAST_TOKEN_RE = re.compile(
    r"whenever you cast (?:another |a |an )?([\w' ]{0,20}?)spell[^.]{0,80}?"
    r"create (a|two|three) ([\w/+\- ]{0,40}?)creature token", re.IGNORECASE)
_PT_RE = re.compile(r"(\d+)/(\d+)")


#: A TOKEN DOUBLER, FOR CREATURE TOKENS. `treasure_doubler` has existed since
#: the Treasure model and its own comment calls the shape "Procession-style xN"
#: — but it was only ever applied to Treasures, so Anointed Procession, Parallel
#: Lives, Doubling Season and Mondrak doubled nothing that fights.
#:
#: On edgar-vampires that is THREE cards (Anointed Procession, Elspeth Storm
#: Slayer, Mondrak) and it compounds with the commander's eminence, which mints
#: a token on every other Vampire cast. `deck-audit`'s engine brief for this deck
#: says it in one line — "eminence mints a free body every time you cast a
#: Vampire and the doublers turn one mint into four" — and neither half of that
#: sentence was in the simulation.
#:
#: 11 corpus cards match; the six TRIPLERS ("three times that many") are left
#: alone rather than read as x2, and two conditional matches are excluded: Kaya,
#: Geist Hunter doubles only "until end of turn" off a -2, and Hosting Season is
#: gated on a calendar date.
_TOKEN_DOUBLER_RE = re.compile(
    r"if (?:an effect would create )?one or more tokens would be created"
    r"[^.]{0,80}?twice that many|if an effect would create one or more tokens"
    r"[^.]{0,80}?twice that many", re.IGNORECASE)
_TOKEN_DOUBLER_TEMPORARY_RE = re.compile(
    r"until end of turn|this turn|while it's", re.IGNORECASE)


def token_doubler(card):
    """Does this permanent double every token you create, for good?"""
    text = card.get("oracle_text", "") or ""
    m = _TOKEN_DOUBLER_RE.search(text)
    if not m:
        return False
    # The condition is scoped to the clause, the same lesson the ETB life-loss
    # channel records: a -2 that doubles "until end of turn" is not a doubler.
    window = text[max(0, m.start() - 90):m.end()]
    return not _TOKEN_DOUBLER_TEMPORARY_RE.search(window)


#: Gates that name a CARD TYPE rather than a creature subtype. Same four as
#: `_CAST_DRAW_GATES`, for the same reason: a cast trigger fires in this model at
#: the two doors a non-land PERMANENT joins the battlefield, so a permanent-type
#: gate is counted completely. A gate the type line does not settle stays a
#: SUBTYPE gate and keeps the old behaviour.
_CAST_TOKEN_TYPE_GATES = {"enchantment": "Enchantment", "creature": "Creature",
                          "artifact": "Artifact", "aura": "Aura"}
#: A TOKEN WHOSE SIZE THIS MODEL CANNOT PRICE. Hallowed Haunting mints a Spirit
#: whose power and toughness each equal "the number of SPIRITS you control" — a
#: self-referential snowball, since the only Spirits in the list are the ones it
#: has already made. Nothing here tracks a creature-type count, so the token is
#: priced at its FLOOR of 1/1 and the understatement is NAMED in the metrics
#: rather than guessed at. An absent figure beats a wrong one; a floor beats
#: both when the direction is known.
_CAST_TOKEN_SCALES_RE = re.compile(
    r"power and toughness are each equal to the number of", re.I)


def cast_token_profile(card):
    """A "cast an X spell -> make a token" trigger, or None.

    Returns `{subtype, gate_kind, bodies, power, scales}`.

    THIS WAS COMMANDER-ONLY AND SUBTYPE-ONLY, and that hid the enchantment
    archetype's entire payoff. `subtype` was matched against a cast card's
    `subtypes` — creature types like Vampire, for Edgar Markov's eminence — so a
    gate naming a CARD TYPE could never fire, and the profile was computed only
    for the commander, so a card in the 99 was never even asked.

    Sigil of the Empty Throne ("whenever you cast an enchantment spell, create a
    4/4 white Angel creature token with flying") parsed CORRECTLY here the whole
    time and was thrown away twice over. On a list of 44 enchantments that is
    a 4/4 flier per cast, worth nothing to the model.

    CORPUS SWEEP 2026-09-09: 93 cards in the family; 2 are gated on enchantment
    spells (Sigil, Hallowed Haunting) and 17 more on another permanent type.
    """
    text = card.get("oracle_text", "") or ""
    m = _CAST_TOKEN_RE.search(text)
    if not m:
        return None
    subtype = (m.group(1) or "").strip()
    if not subtype:
        return None
    pt = _PT_RE.search(m.group(3) or "")
    gate = _CAST_TOKEN_TYPE_GATES.get(subtype.lower())
    return {"subtype": gate or subtype,
            "gate_kind": "type" if gate else "subtype",
            "bodies": _DRAW_WORDS[m.group(2).lower()],
            "power": int(pt.group(1)) if pt else 1,
            "scales": bool(_CAST_TOKEN_SCALES_RE.search(text))}


def sac_outlet_profile(card):
    """Is this a FREE repeatable sacrifice outlet, a costed one, or neither."""
    text = card.get("oracle_text", "") or ""
    if _FREE_SAC_OUTLET_RE.search(text):
        return "free"
    if _COSTED_SAC_OUTLET_RE.search(text):
        return "costed"
    return None


#: "Whenever a creature an opponent controls dies, you gain N life" — the
#: Meathook's third ability, and the one that makes somebody else's removal
#: spell into our damage. Separate from `_DEATH_TRIGGER_RE`, which is about OUR
#: creatures dying.
_OPPONENT_DEATH_GAIN_RE = re.compile(
    r"whenever a creature an opponent controls dies, you gain (\d+) life", re.I)


def death_profile(card):
    """What fires when ANOTHER creature you control dies.

    `unreadable` marks a card that clearly has a death trigger whose effect this
    parser cannot price — surfaced in the metrics rather than silently zero, the
    same contract the Treasure and combat models keep.
    """
    text = card.get("oracle_text", "") or ""
    out = {"death_drain": 0, "death_draw": 0, "death_treasure": 0,
           "gain_on_opponent_death": 0, "unreadable": None}
    # THE OTHER HALF OF THE MEATHOOK. "Whenever a creature an OPPONENT controls
    # dies, you gain 1 life" is a separate trigger from the one above, and in a
    # deck that turns life gained into life lost it means every removal spell
    # anyone casts is damage from us. Parsed independently because the two
    # clauses can appear alone.
    m_opp = _OPPONENT_DEATH_GAIN_RE.search(text)
    if m_opp:
        out["gain_on_opponent_death"] = int(m_opp.group(1))
    m = _DEATH_TRIGGER_RE.search(text)
    if not m:
        return out
    clause = text[m.start():m.start() + 170]
    drain = _DEATH_DRAIN_RE.search(clause)
    if drain:
        out["death_drain"] = int(drain.group(1) or drain.group(2))
    draw = _DEATH_DRAW_RE.search(clause)
    if draw:
        out["death_draw"] = _DRAW_WORDS[draw.group(1).lower()]
    if _DEATH_TREASURE_RE.search(clause):
        out["death_treasure"] = 1
    if not any((out["death_drain"], out["death_draw"], out["death_treasure"],
                out["gain_on_opponent_death"])):
        out["unreadable"] = card.get("name")
    return out


def is_death_engine(prof):
    """One predicate, one home — the same lesson `is_etb_engine` records."""
    return bool(prof["death_drain"] or prof["death_draw"] or prof["death_treasure"]
                or prof["gain_on_opponent_death"])


#: X-SPELL DRAW — the class this model reads as NOTHING, on decks built out of it.
#:
#: Measured on heliod, a deck whose entire card-advantage plan is X spells:
#: of 28 instants and sorceries the model had a reason to cast FOUR, and the
#: 24 it could not see included Braingeyser, Stroke of Genius, Prosperity and
#: Skyscribing. `mean_extra_cards_drawn_by_turn` read 0.428 by turn eight on a
#: deck that draws for a living. `net_change`'s own caveat already named the
#: class — "X-based draw is unmodelled" — so this closes a gap the harness was
#: honest about rather than one it hid.
#:
#: CORPUS SWEEP: 33 X-cost instants and sorceries that draw X. Four shapes and
#: they all give the CASTER the cards, which is why one rule covers them:
#:     13  Draw X cards
#:      6  Target player draws X cards
#:      3  Each player draws X cards
#:     11  the same with a rider (lose X life, discard, mill)
#:
#: THE FIXED COST IS ALREADY `cmc`. Scryfall counts {X} as zero, so
#: Stroke of Genius at {X}{2}{U} has cmc 3 and Braingeyser at {X}{U}{U} has 2 —
#: the number to subtract before dividing. `x_draw_multiplier` is how many {X}
#: symbols the cost carries, because {X}{X} buys one card per TWO mana.
_X_DRAW_RE = re.compile(
    r"(?:you|target player|each player)?\s*draws? X cards?", re.I)
#: NET, NOT GROSS. Read the Runes draws X and discards X — a filter, not card
#: advantage, and crediting it X would have made the worst card in the family
#: read as the best. Pull from Tomorrow discards ONE, so it is X-1.
_X_DRAW_DISCARD_X_RE = re.compile(
    r"discards? X cards?|discard that many"
    # Read the Runes writes the same clause the long way round: "Draw X cards.
    # For each card drawn this way, discard a card unless you sacrifice a
    # permanent." Crediting it X would have made the worst card in the family
    # read as one of the best, at {X}{U} — the cheapest fixed cost of all 33.
    r"|for each card drawn this way, discard", re.I)
_X_DRAW_DISCARD_ONE_RE = re.compile(r"then discard a card", re.I)
#: "You may pay {2}{U} rather than pay this spell's mana cost" — under the
#: alternative cost X is zero, and the model has no way to choose.
_X_DRAW_ALT_COST_RE = re.compile(
    r"rather than pay this spell's mana cost", re.I)


def draw_profile(card):
    """How many cards this card draws, and through which channel.

    `unmodelled` is set when the card clearly draws but through a channel this
    model has no event for. Those are surfaced in `meta.draw_not_modelled`
    rather than silently scoring zero — the whole reason this function exists is
    that a silent zero was indistinguishable from a deck with no draw in it.
    """
    text = card.get("oracle_text", "") or ""
    type_line = card.get("type_line", "") or ""
    out = {"etb_draw": 0, "spell_draw": 0, "recurring_draw": 0,
           "spell_draw_greatest_power": False, "etb_draw_per_type": None,
           "arrival_draw": 0, "arrival_draw_once": False,
           "arrival_power_min": None, "arrival_power_max": None,
           "cast_draw": 0, "cast_draw_gate": None,
           "cast_draw_gate_mv": 0, "cast_draw_cost": 0,
           "x_draw_multiplier": 0, "x_draw_discard": 0,
           # A WHEEL: discard the hand, draw `wheel_draws` (-1 = as many as
           # were discarded). A LOOT: `spell_discard` cards leave hand after
           # the spell's own draw. Both are ACTED ON only under model_discard.
           "wheel_draws": 0, "wheel_shuffles": False, "spell_discard": 0,
           "unmodelled": None}
    _w = _WHEEL_RE.search(text)
    # A wheel with NO MANA COST (Wheel of Fate, suspend only) would be cast
    # for nothing by a loop that spends what a card costs; excluded.
    if _w and card.get("name") not in _WHEEL_EXCLUDED and str(card.get("mana_cost") or "") \
            and ("Instant" in type_line or "Sorcery" in type_line):
        _word = _w.group(2).lower()
        out["wheel_draws"] = (-1 if _word.startswith("cards equal") or _word in ("x", "that many")
                              else _DRAW_WORDS.get(_word, {"seven": 7, "six": 6}.get(_word, 7)))
        # A SHUFFLE WHEEL (Echo of Eons, Time Reversal, Molten Psyche) empties
        # the hand without a discard: no discard trigger fires.
        out["wheel_shuffles"] = _w.group(1).lower().startswith("shuffle")
    # BEFORE the `_DRAW_RE` guard, which wants a WRITTEN-OUT quantity ("draw
    # two cards") and does not recognise "draws X cards" — so every card in
    # this family returned here with an all-zero profile and, worse, with
    # `unmodelled` still None, which is the one value that means "nothing to
    # see". Braingeyser read as a card with no draw on it at all.
    _mc = str(card.get("mana_cost") or "")
    if (("Instant" in type_line or "Sorcery" in type_line)
            and "{X}" in _mc
            # A SPLIT CARD'S `cmc` IS BOTH HALVES (CR 202.3d), so the fixed part
            # this rule subtracts is wrong for it by the other half's cost.
            # Expansion // Explosion is the only one in the family; excluded
            # rather than guessed at.
            and "//" not in _mc
            and _X_DRAW_RE.search(text)
            and not _X_DRAW_DISCARD_X_RE.search(text)
            # X IS NOT ALWAYS BOUGHT WITH MANA. Skeletal Scrying exiles X cards
            # from a graveyard this model does not have, and Ingenious Mastery
            # has an alternative cost under which X is 0. Both would have been
            # credited the whole remaining pool. The additional-cost pattern is
            # the one `spell_draw` already uses, for the same reason.
            and not _DRAW_ADDITIONAL_COST_RE.search(text)
            and not _X_DRAW_ALT_COST_RE.search(text)):
        out["x_draw_multiplier"] = _mc.count("{X}")
        out["x_draw_discard"] = 1 if _X_DRAW_DISCARD_ONE_RE.search(text) else 0

    if _UPKEEP_REVEAL_RE.search(text):
        out["recurring_draw"] = 1
    if (("Instant" in type_line or "Sorcery" in type_line)
            and _DRAW_GREATEST_POWER_RE.search(text)
            and not _DRAW_ADDITIONAL_COST_RE.search(text)):
        out["spell_draw_greatest_power"] = True
    per_type = _ETB_DRAW_PER_TYPE_RE.search(text)
    if per_type:
        out["etb_draw_per_type"] = per_type.group(1)
    if (not _DRAW_RE.search(text) and not out["wheel_draws"]
            and not out["recurring_draw"] and not out["spell_draw_greatest_power"]):
        return out

    m = _ARRIVAL_DRAW_RE.search(text)
    if m and _DRAW_QUALIFIER_OK_RE.match(m.group(2) or ""):
        out["arrival_draw"] = _DRAW_WORDS[m.group(3).lower()]
        out["arrival_draw_once"] = bool(_DRAW_ONCE_RE.search(text))
        lo = _DRAW_POWER_MIN_RE.search(m.group(2) or "")
        hi = _DRAW_POWER_MAX_RE.search(m.group(2) or "")
        out["arrival_power_min"] = int(lo.group(1)) if lo else None
        out["arrival_power_max"] = int(hi.group(1)) if hi else None

    cast = _CAST_DRAW_RE.search(text)
    if cast:
        gate = _CAST_DRAW_GATES.get((cast.group(1) or "").strip().lower())
        if gate:
            out["cast_draw"] = _DRAW_WORDS[cast.group(2).lower()]
            out["cast_draw_gate"] = gate
    mv = _CAST_DRAW_MV_RE.search(text)
    if mv:
        out["cast_draw"] = 1
        out["cast_draw_gate_mv"] = int(mv.group(1))
    if _CAST_DRAW_PAY_RE.search(text):
        out["cast_draw"] = 1
        out["cast_draw_gate"] = "Creature"
        out["cast_draw_cost"] = 1

    etb = _ETB_DRAW_RE.search(text)
    if etb and not _DRAW_CONDITIONAL_RE.search(etb.group(0)) and not out["etb_draw_per_type"]:
        out["etb_draw"] = _DRAW_WORDS[etb.group(1).lower()]
    rec = _RECURRING_DRAW_RE.search(text)
    if rec and not _DRAW_CONDITIONAL_RE.search(rec.group(0)):
        out["recurring_draw"] = _DRAW_WORDS[rec.group(1).lower()]
    if ("Instant" in type_line or "Sorcery" in type_line) and not out["etb_draw"]:
        sp = _SPELL_DRAW_RE.search(text) or _SCRY_THEN_DRAW_RE.search(text)
        if sp and not _DRAW_ADDITIONAL_COST_RE.search(text):
            out["spell_draw"] = _DRAW_WORDS[sp.group(1).lower()]
            td = _DRAW_THEN_DISCARD_RE.search(text)
            if td:
                out["spell_discard"] = _DRAW_WORDS[td.group(1).lower()]

    if not any((out["etb_draw"], out["spell_draw"], out["recurring_draw"],
                out["arrival_draw"], out["cast_draw"],
                out["x_draw_multiplier"], out["wheel_draws"],
                out["spell_draw_greatest_power"], out["etb_draw_per_type"])):
        out["unmodelled"] = card.get("name")
    return out


#: Written-out quantities. `X` is board-dependent (Sanctum Weaver counts
#: enchantments, Selvala reads a power), so it takes the conservative 1 — the
#: same call `treasure_profile` makes for "for each" and "equal to".
_MANA_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "x": 1}

# Deliberately the SAME pattern as ROLE_PATTERNS["tutor:unrestricted"] in
# config.py. A second definition of "what is a tutor" would let the sim and the
# role histogram disagree about the same 99, which is the class of bug this repo
# has paid for before. Narrow tutors ("search your library for a LAND card") are
# excluded on purpose — they cannot fetch a missing combo half.
_TUTOR_RE = re.compile(r"search your library for a card", re.IGNORECASE)
#: A TYPED TUTOR IS STILL A TUTOR, and `_TUTOR_RE`'s literal "a card" matched
#: none of them. Sarkhan's Triumph ("a Dragon creature card"), Worldly Tutor ("a
#: creature card") and Enlightened Tutor ("an artifact or enchantment card") all
#: read as `tutor: False`, so ur-dragon's `*_assisted` figures were computed as
#: if it ran ZERO tutors while `model_assumptions` said tutors were modelled — a
#: number produced, plausible and wrong.
#:
#: THREE GUARDS, each bought by the corpus sweep. Widening naively took 114
#: cards to 881:
#:   `\A`          the tutor must be the SPELL'S OWN EFFECT, not an activated
#:                 ability, ETB or death trigger. Without it Birthing Pod, Academy
#:                 Rector and Amrou Scout become free wildcards, which they are
#:                 not — the model treats a tutor as "drawn and affordable".
#:   instant/sorcery  same reason, from the other side.
#:   not a land    482 of the 767 new matches were land fetches. Cultivate and
#:                 Farseek are RAMP, and pricing them belongs in the land-ramp
#:                 channel, not here. Counting them as wildcards would let a
#:                 basic-land search fill an engine component.
#: Net: 114 -> 165.
_TYPED_TUTOR_RE = re.compile(
    r"\Asearch your library for (?:a|an|up to \w+) [\w' -]{0,30}?card",
    re.IGNORECASE)
_TUTOR_LAND_RE = re.compile(
    r"\b(land|Plains|Island|Swamp|Mountain|Forest|basic)\b", re.IGNORECASE)


def is_tutor(card):
    """Does casting this card fetch a card the deck was missing?"""
    text = (card.get("oracle_text") or "").strip()
    if _TUTOR_RE.search(text):
        return True
    type_line = card.get("type_line") or ""
    if not ("Instant" in type_line or "Sorcery" in type_line):
        return False
    # A creature searched ONTO THE BATTLEFIELD is a tutor (Savage Order,
    # Natural Order); the typed pattern wants the search at the start of the
    # text and Savage Order opens with its additional cost.
    if _TUTOR_TO_BATTLEFIELD_RE.search(text):
        return True
    got = _TYPED_TUTOR_RE.match(text)
    if not got:
        return False
    return not _TUTOR_LAND_RE.search(text[:got.end() + 20])
# Vampiric Tutor and Insatiable Avarice fetch to the TOP of the library, not to
# hand: the card arrives on the next draw, so the wildcard lands a turn later.
# The printed wording is "put that card on top." with no "of your library", so
# match the bare phrase — an "on top of" pattern silently matches neither.
_TUTOR_TO_TOP_RE = re.compile(r"\bon top\b", re.IGNORECASE)
# Spree/modal tutors ("+ {2} — Search your library for a card") charge the mode
# cost ON TOP of the card's mana value. Insatiable Avarice is cmc 1 but cannot
# tutor for less than 3, and billing it at 1 would overstate how early the
# wildcard is live.
_TUTOR_MODE_COST_RE = re.compile(r"\+\s*\{(\d+)\}[^\n]{0,4}—[^\n]{0,40}search your library for a card",
                                 re.IGNORECASE)
# Diabolic Intent's additional cost. A tutor you cannot pay for is not a wildcard.
_TUTOR_SAC_RE = re.compile(r"as an additional cost.{0,40}sacrifice a creature",
                           re.IGNORECASE | re.DOTALL)


# ── Treasures ─────────────────────────────────────────────────────────────
#
# A Treasure is NOT a mana rock and modelling it as one is the whole trap: a
# rock produces every turn forever, a Treasure produces once and is gone. The
# stockpile below is spent only when lands and rocks come up short, which is
# both how it is played and what makes a hoard-counting payoff measurable.
#
# **Only triggers this simulation can honestly see are modelled.** There is no
# combat here and there are no opponents, so "whenever this creature deals
# combat damage to a player" (Old Gnawbone, Cavern-Hoard Dragon) and "whenever
# an opponent draws a card" (Smothering Tithe) produce NOTHING — and that is a
# finding rather than a shortcoming. Measured across the fleet, 16 of the 19
# Treasure sources in the nine decks are combat- or opponent-gated; a naive
# "create a Treasure token" match would hand eight decks free mana they never
# get, turning a deliberately conservative model optimistic. Unmodelled sources
# are NAMED in the output so a low number is legible instead of mysterious.
_TREASURE_RE = re.compile(r"creates?\s+(?:[\w\-]+\s+)*?Treasure tokens?", re.IGNORECASE)
_TREASURE_N_RE = re.compile(
    r"creates?\s+(a|an|one|two|three|four|five|X|\d+)\s+(?:[\w\-]+\s+)*?Treasure",
    re.IGNORECASE)
# Recurring, and free at the point of use.
_TRE_UPKEEP_RE   = re.compile(r"at the beginning of your upkeep", re.IGNORECASE)
_TRE_LANDFALL_RE = re.compile(r"whenever a land you control enters|landfall", re.IGNORECASE)
_TRE_CAST_RE     = re.compile(r"whenever you cast", re.IGNORECASE)
# A Saga adds a lore counter "after your draw step" every turn, so a Saga whose
# chapters make Treasures IS a recurring engine — The Misty Mountains Cold makes
# one on each of four chapters. Modelled as recurring and NOT sacrificed at IV:
# the chapter payout is the mana question, and the 6/6 Dragon it converts into is
# a body the bodies series would need to know about. Slightly generous after turn
# four on that one card, and stated here rather than hidden.
_TRE_SAGA_RE     = re.compile(r"Enchantment — Saga|add a lore counter", re.IGNORECASE)
# One-shot, on resolution.
_TRE_ETB_RE      = re.compile(r"when (?:this creature|this artifact|[A-Z][\w' ,-]{2,30}) enters",
                              re.IGNORECASE)
# TWO KINDS OF MULTIPLIER, AND CONFLATING THEM IS WRONG IN BOTH DIRECTIONS.
# Xorn and Jolene ADD one Treasure to every Treasure event; Anointed Procession,
# Parallel Lives, Doubling Season and Mondrak DOUBLE whatever the event makes.
# They coincide only when the event makes exactly one, which is why an additive
# stand-in for doubling reads almost right and is not.
#
# These are PUBLIC because `assess._MULTIPLIER` is the other reader of the same
# concept and the two had diverged silently: this module matched one wording and
# assess matched five, so the goldfish priced 2 of the 8 multipliers ur-dragon's
# treasure branch DECLARES and counted the other 6 as drawn-and-inert. That is
# the `front_field` defect one subsystem over — two halves of one idea drifting
# because nothing made them share a definition. They live here rather than in
# config because this module owns the Treasure model; config owns the frozen,
# model-facing vocabulary and adding to it invalidates a trained net.
TREASURE_BONUS_RE = re.compile(r"instead create those tokens plus an additional Treasure",
                               re.IGNORECASE)
#: "it creates twice that many of those tokens instead" and Mondrak's inversion
#: of the same sentence. Deliberately keyed on TOKENS — Panharmonicon doubles
#: ETB TRIGGERS and Academy Manufactor converts Clue/Food events into Treasure
#: ones; both are real multipliers for a deck and neither is this one, so they
#: stay blind and get NAMED rather than folded in where they would read as right.
TOKEN_DOUBLER_RE = re.compile(
    r"creates twice that many of those tokens|twice that many of those tokens are created|(?:twice|three times) that many (?:of those )?(?:creature )?tokens are created",
    re.IGNORECASE)
#: Kept so the old private name still resolves for anything reading it.
_TRE_EXTRA_RE = TREASURE_BONUS_RE


def treasure_profile(card):
    """How this card makes Treasures, and whether a goldfish can see it.

    Returns `(per_event, trigger)` where trigger is one of `upkeep`,
    `landfall`, `cast` (recurring), `etb` (once), or `unmodelled`.
    A card with no Treasure text returns `(0, None)`.
    """
    text = card.get("oracle_text") or ""
    if not _TREASURE_RE.search(text):
        return 0, None
    match = _TREASURE_N_RE.search(text)
    word = (match.group(1).lower() if match else "a")
    # "X Treasures" is opponent- or board-dependent every time it appears in
    # this corpus, so it is counted as one rather than guessed at.
    count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1) or 1
    saga = _TRE_SAGA_RE.search(card.get("type_line", "") or "") or _TRE_SAGA_RE.search(text)
    if saga:
        return count, "upkeep"
    for trigger, pattern in (("upkeep", _TRE_UPKEEP_RE),
                             ("landfall", _TRE_LANDFALL_RE),
                             ("cast", _TRE_CAST_RE),
                             ("etb", _TRE_ETB_RE)):
        if pattern.search(text):
            return count, trigger
    return count, "unmodelled"


# ── Combat ────────────────────────────────────────────────────────────────
#
# OPT-IN, for exactly the reason `model_treasures` is: switching combat on
# changes `mean_bodies_by_turn` for every deck that makes non-creature tokens
# (all nine of them), and those figures are quoted in published prose on five
# decks and in one `engine.json` carrying a critic verdict. A deck opts in when
# it is next re-baselined deliberately.
#
# The discipline is the same as the Treasure model's: model only what can be
# read honestly, and NAME what cannot. What this buys is the class of card the
# resource model priced at exactly zero — attack triggers (Savage Ventmaw's
# {R}{R}{R}{G}{G}{G}, Old Gnawbone's Treasures, Smaug's ping), additional combat
# phases (Scourge of the Throne, Aggravated Assault), and therefore the
# combat-gated Treasure sources that `treasure_profile` returns `unmodelled` for.
# On ur-dragon that was nine of fourteen sources and both halves of the deck's
# only verified win line.

# OWN haste is a KEYWORD -- "Haste", "Flying, haste" -- and not a mention. This
# read `\bhaste\b` anywhere in the text until 2026-09-11, so a creature that
# GRANTS haste without having it (Regisaur Alpha, Ogre Battledriver) or whose
# riot reminder text merely names it (Spider-Punk) attacked the turn it landed.
# A conditional own haste ("has haste as long as", Markov Crusader) is now read
# as none, which is the honest floor. Sweep locked in
# tests/test_pilot_goldfish_combat.py.
_KEYWORD_NOT_GRANTED = (r"(?<!have )(?<!has )(?<!gain )(?<!gains )(?<!or )(?<!with )"
                        r"(?<!lose )(?<!loses )(?<!without )(?<!and )")
_HASTE_RE = re.compile(_KEYWORD_NOT_GRANTED + r"\bhaste\b", re.IGNORECASE)
# A LAND-MANA BONUS: "whenever you tap a land for mana, add one mana" (Mirari's
# Wake, Zendikar Resurgent, Vorinclex, Nikya) -- four corpus cards, read as
# nothing until 2026-09-11 on a big-mana Dinosaur deck whose whole plan is
# eight mana on turn five.
_LAND_MANA_BONUS_RE = re.compile(
    r"whenever you tap a land for mana, add (?:an additional )?(one|two) "
    r"(?:additional )?mana", re.I)
# AN ATTACK TOKEN AS BIG AS YOUR BEST ATTACKER (Ghalta and Mavren): the one
# corpus card. `_TOKEN_PT_RE` cannot read "a tapped and attacking X/X".
def _land_mana_bonus(text):
    """Extra mana per land tapped: 1 for "one", 2 for "two", else 0."""
    m = _LAND_MANA_BONUS_RE.search(text or "")
    if not m:
        return 0
    return 2 if m.group(1).lower() == "two" else 1


_ATTACK_TOKEN_SCALES_RE = re.compile(
    r"create an? (?:tapped and attacking )?X/X[^.]*?where X is the greatest power",
    re.I)
_FLYING_KW_RE = re.compile(_KEYWORD_NOT_GRANTED + r"\bflying\b", re.IGNORECASE)
# TEAM HASTE, the grant the pilot's log asked for by name ("granting ur dragon
# haste is nice"). Four shapes, one profile key `team_haste` whose value is
# WHO gets it: "all" (Fervor, Temur Ascendancy, Urabrask, Concordant
# Crossroads), "nontoken" (Rhythm of the Wild's riot, read as always choosing
# haste -- the pilot's choice, stated), "flying" (Dragon Tempest's enters
# trigger) or a creature TYPE (Karrthus: Dragon; Goblin Chieftain: Goblin;
# Regisaur Alpha: Dinosaur). Not read, and named in COMBAT_ASSUMPTIONS: a
# grant from the graveyard (Anger), an activated grant (Crashing Drawbridge),
# a grant conditional on the board ("as long as"), a lowercase class
# ("outlaws"), and the two lands whose mana carries haste (Hall of the Bandit
# Lord, Arena of Glory) -- the goldfish taps no particular land.
_TEAM_HASTE_RE = re.compile(
    # "artifact creatures", "multicolored creatures", "equipped creatures": a
    # class this board has no key for, so the grant is left unread and NAMED
    # in the sweep rather than widened to the team.
    r"(?:all creatures|(?:[Oo]ther )?(?<!artifact )(?<!multicolored )(?<!equipped )"
    r"(?<!legendary )(?<!face-down )(?<!tapped )(?<!attacking )(?<!enchanted )"
    r"creatures you control) have "
    r"(?:[a-z]+(?:, [a-z]+)* and )?haste\b", re.IGNORECASE)
_TYPED_HASTE_RE = re.compile(
    r"(?:^|[.)]\s|haste\s)(?:[Oo]ther )?(nontoken |flying )?([A-Z][a-z]+?)s? (?:creatures? )?you control "
    r"(?:get \+\d/\+\d and )?have (?:[a-z]+(?:, [a-z]+)* and )?haste\b")
_ENTERS_HASTE_RE = re.compile(
    r"whenever (?:a|another) (?:nontoken )?creature you control( with flying)? enters, "
    r"(?:that creature|it) (?:gets \+\d/\+\d and )?gains haste", re.IGNORECASE)
_RIOT_TEAM_RE = re.compile(r"(?:nontoken )?creatures you control have riot", re.IGNORECASE)
_GRANT_CONDITIONAL_RE = re.compile(r"as long as|\bif\b|during (?:your|each)", re.IGNORECASE)


def _sentence_around(text, pos):
    """The sentence holding `pos`; a newline ends one as a period does, since
    cards.json keeps oracle newlines and cards.csv flattens them to spaces."""
    start = max(text.rfind(". ", 0, pos), text.rfind("\n", 0, pos))
    start = 0 if start < 0 else start + 1
    end = min((i for i in (text.find(".", pos), text.find("\n", pos)) if i >= 0), default=-1)
    return text[start:] if end < 0 else text[start:end]


def _static_grant(text, m):
    """Unconditional and not the effect of an activation or a loyalty ability
    (Barbarian Class's level 3, Ellywick's emblem) -- what a permanent does by
    being in play."""
    return (not _GRANT_CONDITIONAL_RE.search(_sentence_around(text, m.start()))
            and not _inside_activation(text, m.start()))


def team_haste_grant(text):
    """Who a card gives haste to: None, "all", "nontoken", "flying" or a type."""
    # TYPED FIRST: "Other Dragon creatures you control have haste" contains
    # "creatures you control have haste", so the team pattern alone read
    # Karrthus as a grant to everything.
    m = _TYPED_HASTE_RE.search(text)
    if m and _static_grant(text, m):
        if m.group(1):
            return m.group(1).strip().lower()
        word = m.group(2)
        if word == "Flying":
            return "flying"
        if word == "Nontoken":
            return "nontoken"
        if word in _corpus_creature_types():
            return word
        if word not in ("Other", "Creature", "All"):
            return None       # "Legendary creatures you control": a class this model has no key for
    m = _TEAM_HASTE_RE.search(text)
    if m and _static_grant(text, m):
        return "all"
    if _RIOT_TEAM_RE.search(text):
        return "nontoken"
    m = _ENTERS_HASTE_RE.search(text)
    if m:
        return "flying" if m.group(1) else "all"
    return None
# "create a 1/1 red Dragon creature token", "create two 2/2 ... tokens"
_TOKEN_PT_RE = re.compile(r"create (\w+) ([\dX]+)/([\dX]+)([^.]*?)tokens?", re.IGNORECASE)
_ATTACKS_RE = re.compile(
    r"whenever you attack\b|whenever (?:this creature|[A-Z][\w' ,-]{2,30}|one or more [\w ]+ you control) attacks",
    re.IGNORECASE)
_COMBAT_DMG_RE = re.compile(
    r"whenever (?:this creature|[A-Z][\w' ,-]{2,30}) deals combat damage to a player",
    re.IGNORECASE)
# POISON. Infect makes every point of damage a source deals to a player a poison
# counter instead (CR 702.90b) — combat or not — and ten counters lose the game
# (704.5c). Toxic N adds N counters when the creature deals COMBAT damage, on top
# of the damage. The keyword sits at the start of a line, alone or after a
# comma-separated keyword list ("Flying\nInfect (…)", "Deathtouch\nToxic 1");
# a GRANT ("has infect", "gains infect until end of turn") is lower-case mid-
# sentence and is deliberately not matched here — it is priced at nothing and
# named in the commit that added this, the way Vector Asp and Grafted
# Exoskeleton read. Corpus sweep 2026-09-10: 32 infect creatures, 32 toxic
# cards, 9 grants, and the counts are locked by a test.
# The keyword is CAPITALISED when it heads a line and lower-case after a comma
# in a keyword list; a grant is lower-case after "has" / "gains" / "have". The
# anchor accepts a space as well as a newline because `cards.csv` flattens
# oracle newlines to spaces while `cards.json` keeps them, and the first
# version of this pattern, anchored on the newline alone, missed Phyrexian
# Crusader and Skithiryx in the corpus while reading them in a deck.
_INFECT_KW_RE = re.compile(r"(?:(?:^|\n| )Infect\b|, infect\b)(?! until)")
_TOXIC_KW_RE = re.compile(r"(?:(?:^|\n| )Toxic|, toxic) (\d+)\b")
# A PING PER ATTACKER, dealt BY THE ATTACKER: "Whenever a creature you control
# attacks, that creature deals 1 damage to each opponent" (Ingris Stingerquill,
# Reality Fracture). Because the creature is the source, an infect attacker's
# ping is a poison counter to every opponent, before blocks. Zero cards in the
# 2026-08 corpus carry this shape; the first is a commander whose set is not
# yet released, so this is parsed from the deck's own cards.json rather than
# declared, and the sweep test asserts the corpus count so the day a second
# one prints it is read on purpose.
_ATTACK_PING_EACH_RE = re.compile(
    r"whenever a creature you control attacks, (?:that creature|it) deals (\d+) damage to each opponent",
    re.IGNORECASE)
_EXTRA_COMBAT_RE = re.compile(r"additional combat phase", re.IGNORECASE)
# An ACTIVATED extra combat (Aggravated Assault's {3}{R}{R}), as opposed to a
# triggered one (Scourge of the Throne). The cost decides whether it is a free
# repeat button or one you have to buy every turn.
# Sentence-crossing on PURPOSE. Aggravated Assault reads "{3}{R}{R}: Untap all
# creatures you control. After this main phase, there is an additional combat
# phase" — a `[^.]` bound stops at that period, the cost never binds, and the
# deck's only verified win line silently becomes unmodelled. Caught by test.
_ACTIVATED_COMBAT_RE = re.compile(
    r"((?:\{[WUBRGC0-9]\})+)\s*:.{0,160}?additional combat phase",
    re.IGNORECASE | re.DOTALL)
_DMG_EQUAL_TREASURE_RE = re.compile(
    r"deals damage equal to the number of Treasures", re.IGNORECASE)


# "{T}:", "{3}{R}:", "{1}, Sacrifice a creature:" — a cost that opens an
# activated ability. Bounded so a mana symbol inside an effect ("add {R}") is
# not read as one.
# A planeswalker's loyalty ability ("+1:", "−3:", "0:") is an activation too —
# Huatli, Poet of Unity's Dinosaur was the fleet's one instance.
_ACTIVATION_COST_RE = re.compile(r"(?:\{[^}]+\}[^:.\n]{0,60}|(?:^|\s)[+\-\u2212]?\d+):")


def _inside_activation(text, pos):
    """Is `pos` inside the effect of an activated ability?

    The ability is the stretch from the last sentence end (or line start)
    before `pos`; if it opens with a cost, the effect is bought, not triggered.
    Works on `cards.json` (real newlines) and on `cards.csv` (flattened to
    spaces) alike, because it bounds on the period rather than the line.
    """
    start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos)) + 1
    return bool(_ACTIVATION_COST_RE.search(text[start:pos]))


def _mana_pips(cost_string):
    """How much generic-equivalent mana a '{3}{R}{R}' style string costs."""
    total = 0
    for sym in re.findall(r"\{([WUBRGC0-9])\}", cost_string or ""):
        total += int(sym) if sym.isdigit() else 1
    return total


def _stat(value):
    """Power/toughness as an int; '*' and None become 0 (conservative)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def creature_body_count(card):
    """Bodies counting CREATURES only — a Treasure token is not a blocker.

    This is `body_count` with the non-creature tokens removed. It is reached
    only under `model_combat` so that a deck which has not opted in keeps the
    number it published.
    """
    text = card.get("oracle_text", "") or ""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for match in _TOKEN_RE.finditer(text):
        clause = text[match.start():match.end() + 60].lower()
        if any(k in clause for k in _NONCREATURE_TOKENS):
            continue
        word = match.group(1).lower()
        bodies += int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
    return bodies


#: THE ENTERS-THE-BATTLEFIELD PAYOFF, which this model had no channel for at
#: all. ETB was read for Treasure and nothing else (`_TRE_ETB_RE`), so a deck
#: whose stated win condition is "ETB and attack-trigger burn" had the ETB half
#: measured at ZERO: Terror of the Peaks and Scourge of Valkas read as vanilla
#: bodies and Dragon Tempest read as nothing whatever.
#:
#: The trigger, then four payloads read from a window after it — the same shape
#: `_ATTACKS_RE` and its window already use, so there is one idiom here and not
#: two.
#: The `(?!lands?\b)` is what the corpus sweep bought. Without it the lazy noun
#: run swallowed "land ", so every LANDFALL payoff — Omnath, Rampaging Baloths,
#: Titania, Zektar Shrine Expedition — read as a creature-entering payoff and
#: would have fired on each creature cast. A landfall trigger is a different
#: event and this channel must not claim it.
#: THE LOOKAHEAD BLOCKS THE WORD "land" AND ALSO EVERY LAND TYPE, and the
#: second half arrived 2026-08-28 with its own sweep. `(?!lands?\b)` was written
#: for "whenever a land you control enters" and let "whenever a MOUNTAIN you
#: control enters" straight through, so a landfall trigger named by basic type
#: read as a creature-arrival payoff. Fourteen corpus cards, and two of them
#: were actively scoring: Dread Presence billed 2 damage per CREATURE arrival
#: off a Swamp trigger, and Koth, Fire of Resistance — a PLANESWALKER — billed 4
#: off an emblem's Mountain trigger. Both surfaced in a candidate search for
#: this channel, which is how they were found.
#:
#: The sweep only NARROWS: 438 matches to 424, nothing newly matched, and all
#: fourteen read one by one as genuine landfall.
_ETB_TRIGGER_RE = re.compile(
    r"whenever (?:this creature or )?(?:another|a|one or more)\s+"
    r"(?:nontoken\s+)?(?!lands?\b|mountains?\b|swamps?\b|plains\b|islands?\b"
    r"|forests?\b|gates?\b|caves?\b|deserts?\b|towns?\b|spheres?\b)"
    r"[\w ]{0,24}?you control enters",
    re.IGNORECASE)
# THE SUBJECT OF AN ENTRY TRIGGER. `_ETB_TRIGGER_RE` throws the noun away, so
# Dragon Tempest ("whenever a Dragon you control enters") fired on a Bird of
# Paradise and Lathliss made a token for a mana dork. 28 of 72 entry engines
# in the corpus name a type (sweep 2026-09-11). The gate is read here and
# checked at the one door; TOKENS carry no type line in this model and pass
# the gate, which keeps Lathliss's Dragons firing Tempest (stated).
_ETB_SUBJECT_RE = re.compile(
    r"whenever (?:this creature or )?(?:another|a|an|one or more)\s+(?:nontoken\s+)?"
    r"([A-Z][a-z]+)\s+(?:creature\s+)?(?:you control\s+)?enters", re.IGNORECASE)
# MOLTEN ECHOES: a copy of every nontoken creature of the CHOSEN type entering.
_ETB_CHOSEN_TYPE_COPY_RE = re.compile(
    r"whenever a nontoken creature you control of the chosen type enters, "
    r"create a token that's a copy", re.IGNORECASE)
# CHANDRA'S IGNITION: a creature you control deals its power to each opponent.
_SPELL_DAMAGE_POWER_RE = re.compile(
    r"target creature you control deals damage equal to its power to each "
    r"(?:other creature and each )?opponent", re.IGNORECASE)
# SARKHAN'S UNSEALING: damage on casting a creature of at least this power.
_CAST_DAMAGE_POWER_RE = re.compile(
    r"whenever you cast a creature spell with power (\d)(?:, \d)*,? or \d, "
    r"[^.]*?deals (\d+) damage", re.IGNORECASE)
# UP THE BEANSTALK: draw on casting a spell of at least this mana value.
_CAST_DRAW_MV_RE = re.compile(
    r"whenever you cast a spell with mana value (\d) or greater, (?:you may )?draw a card",
    re.IGNORECASE)
# LIFECRAFTER'S BESTIARY: draw on a creature cast, for one mana.
_CAST_DRAW_PAY_RE = re.compile(
    r"whenever you cast a creature spell, you may pay \{[WUBRG]\}\. if you do, draw a card",
    re.IGNORECASE)
# SAVAGE ORDER, NATURAL ORDER: a creature searched straight onto the battlefield.
_TUTOR_TO_BATTLEFIELD_RE = re.compile(
    r"search your library for an? (?:([A-Z][a-z]+) )?creature card, put it onto the battlefield",
    re.IGNORECASE)
#: Terror of the Peaks — damage equal to the ENTERING creature's power.
_ETB_DMG_POWER_RE = re.compile(
    r"damage equal to (?:that creature'?s?|its) power", re.IGNORECASE)
#: A FIXED AMOUNT PER ARRIVAL — Impact Tremors, Purphoros, Warleader's Call.
#: All three read as nothing until this existed, and the tell was four
#: candidates returning byte-identical 55.44 alongside a control card the model
#: openly does not read.
#:
#: "each opponent" IS COUNTED ONCE. This model has one opponent at 40 life, so a
#: card that hits each of three seats is understated threefold here — the same
#: direction every other choice in this file takes, and stated rather than
#: silently corrected.
_ETB_DMG_FIXED_RE = re.compile(
    r"deals (\d+) damage to (?:each opponent|any target|that player)",
    re.IGNORECASE)
#: Scourge of Valkas and Dragon Tempest — X damage where X counts a board.
_ETB_DMG_COUNT_RE = re.compile(
    r"deals? X damage[^.\n]{0,60}?where X is the number of", re.IGNORECASE)
#: "another NONTOKEN Dragon you control enters" — Lathliss and Miirym both say
#: it, and it is what stops the board exploding: their own token copies do not
#: re-trigger them. Without this the first cut produced 67,000 damage by turn
#: six, because a copy made a copy made a copy. The rules already had the
#: brake; the model just had to read it.
_ETB_NONTOKEN_RE = re.compile(r"another nontoken", re.IGNORECASE)
#: "EACH OPPONENT LOSES N LIFE" IS THE SAME QUANTITY AS DAMAGE HERE, and until
#: this existed it read as ZERO. The gate was the literal word "damage": Impact
#: Tremors ("deals 1 damage to each opponent") was priced and Corpse Knight
#: ("each opponent loses 1 life") — the same event, the same 1, the same
#: per-arrival cadence — was worth nothing. Measured on edgar-vampires, where
#: the pilot's stated engine is exactly this: `combat_profile` read four of the
#: deck's payoffs and returned NOTHING READ for six, so `damage_8` scored the
#: combat plan the pilot wants to CUT and was blind to the drain plan they want
#: to DEEPEN. A branch aimed at that axis would have been graded on the wrong
#: half of the deck.
#:
#: The two are NOT the same in the rules — lifelink, damage prevention and
#: "whenever an opponent loses life" all tell them apart — and this model has
#: none of those. It has one opponent at 40 life and asks how fast it reaches 0.
#: The field is kept separate from `etb_damage_fixed` rather than folded into it
#: so that a later model which does care can tell the channels apart.
_ETB_LIFE_LOSS_RE = re.compile(
    r"each opponent loses (\d+) life", re.IGNORECASE)
#: A Saga chapter that drains only "this turn" is not a per-arrival engine.
#: One card in the sweep (Thunder of Unity) and it would have been over-read.
_ETB_THIS_TURN_RE = re.compile(r"\bthis turn\b", re.IGNORECASE)
#: WHERE THE CLAUSE THE TRIGGER INTRODUCES ENDS. The life-loss payload is read
#: from HERE and not from the 220-char window the damage payloads use, and the
#: corpus sweep is the whole argument for the asymmetry — the uniform fix is
#: worse in one direction or the other whichever one you pick:
#:
#:   life loss, 220-char window -> 12 matched, 2 of them WRONG. Elas il-Kor
#:     ("...enters, you gain 1 life. Whenever another creature you control DIES,
#:     each opponent loses 1 life") would have drained on every arrival, and
#:     Underworld Coinsmith's is an ACTIVATED ability behind {W}{B} and 1 life.
#:   life loss, clause-scoped -> 10 matched, and all ten read correctly card by
#:     card. Minus Thunder of Unity above: 9.
#:   damage, clause-scoped -> would LOSE 3 of 16 that are correct today.
#:     Crossbones ("...enters, put a +1/+1 counter on Crossbones. He deals 2
#:     damage to each opponent.") puts the payload in a SECOND SENTENCE, which
#:     is the shockland lesson exactly: the idiom spans the boundary.
#:
#: So the scope belongs to the payload, not to the module. This is the same
#: finding `manabase.enters_tapped_unconditionally` recorded when sentence
#: scoping flagged all ten shocklands.
_ETB_CLAUSE_END_RE = re.compile(
    r"(?:\.\s|\bwhenever\b|\bat the beginning\b|\bwhen )", re.IGNORECASE)
_ETB_CLAUSE_HEAD_RE = re.compile(r"enters[^,]{0,40},", re.IGNORECASE)
#: MIRKWOOD BATS, and in the whole 34,900-card corpus it is the only one. The
#: trigger is token CREATION rather than a permanent entering, so
#: `_ETB_TRIGGER_RE` never saw it — and it is a named member of edgar-vampires'
#: kill leg, priced by checker-passed stack 011. A channel for one card is worth
#: it when the card is load-bearing and the alternative is scoring it zero; the
#: count is stated here so nobody has to guess how wide it is.
#:
#: Bats says "create OR SACRIFICE a token" and this model has no sacrifice, so
#: only the creation half is read — an understatement, the same direction every
#: other choice in this file takes. It also fires on NONCREATURE tokens (a Blood
#: token counts) and the model only makes creature tokens, which understates it
#: again.
_TOKEN_CREATED_TRIGGER_RE = re.compile(
    r"whenever you create (?:or sacrifice )?(?:one or more |a |an |another )?"
    r"[\w ]{0,20}?tokens?", re.IGNORECASE)
_TOKEN_CLAUSE_HEAD_RE = re.compile(r"tokens?[^,]{0,40},", re.IGNORECASE)


def _etb_clause(text, start, head=_ETB_CLAUSE_HEAD_RE):
    """The clause one trigger introduces, from `start` to the next trigger.

    Skips past the trigger's own subject (up to the comma that ends it) so that
    a boundary word inside the CONDITION — "whenever another creature you
    control enters" — does not end the clause before the effect begins.
    """
    rest = text[start:]
    m = head.search(rest)
    end = _ETB_CLAUSE_END_RE.search(rest, m.end() if m else 0)
    return rest[:end.start()] if end else rest
#: A COPY EFFECT USUALLY CHARGES FOR ITSELF, and the first cut charged nothing.
#: Flameshadow Conjuring and Minion Reflector both say "you MAY PAY {R}" / "{2}"
#: per trigger; firing them free reported 130.91 damage at turn ten against a
#: 56.43 baseline, which is a plausible number and wrong twice over.
_ETB_OPTIONAL_COST_RE = re.compile(
    r"you may pay ((?:\{[WUBRGC0-9]\})+)", re.IGNORECASE)
#: AND A COPY IS LEGENDARY UNLESS THE CARD SAYS OTHERWISE. Miirym says "except
#: the token isn't legendary" and is played for exactly that; Flameshadow does
#: not, so a copy of any of the 12 legendary creatures in this deck dies to the
#: legend rule before it does anything. Modelling the copy without the rule
#: hands a five-colour legendary deck a doubled board it never gets.
_ETB_COPY_NONLEGENDARY_RE = re.compile(
    r"(?:except )?(?:the token |it )?(?:isn't|is not) legendary", re.IGNORECASE)
#: Miirym — a token that is a COPY of the creature that entered.
_ETB_COPY_RE = re.compile(r"token that'?s? a copy of", re.IGNORECASE)

#: A board that makes tokens that make damage that makes tokens terminates, but
#: only because this says so. Miirym's copy is itself a Dragon entering, which
#: fires Scourge and Tempest again — that is the deck working, not a bug, so the
#: guard has to be a stated depth rather than a silent one. Same shape as the
#: `phases > 20` runaway guard on extra combats.
ETB_CHAIN_LIMIT = 12

#: DAMAGE MULTIPLICATION, WHICH THIS MODEL COULD NOT SEE AT ALL. Three different
#: rules produce one measured effect, and every one of them was landing in
#: `combat_effects_not_modelled`:
#:
#:   Twinflame Tyrant  "If a source you control would deal damage to an opponent
#:                     … it deals double that damage instead" — a replacement
#:                     effect on EVERYTHING you deal.
#:   Atarka            "Whenever a Dragon you control attacks, it gains double
#:                     strike" — the team swings twice.
#:   Thrakkus          "double the power of each Dragon you control" — the team
#:                     hits twice as hard.
#:
#: They STACK MULTIPLICATIVELY by the real rules: double the power, swing twice,
#: then double the damage dealt is eight times, not four. So the model multiplies
#: rather than adds.
#:
#: THE APPROXIMATION, SAID OUT LOUD: a grant worded "each Dragon you control"
#: is treated as applying to the whole team. That is exact in a deck whose
#: attackers are Dragons and generous in one where they are not — so it is
#: recorded in `model_assumptions` rather than hidden, and a deck without the
#: flag is byte-identical.
#: `(?:\w+ )?` is Solphim, Mayhem Dominus: "would deal NONCOMBAT damage to an
#: opponent". One adjective was the whole difference between a doubler the model
#: prices and one it reads as a vanilla body.
_DAMAGE_DOUBLER_RE = re.compile(
    r"would deal (?:\w+ )?damage to[^.\n]{0,80}?(?:opponent|player|permanent)"
    r"[^.\n]{0,80}?deals? double that damage", re.IGNORECASE)
_TEAM_DOUBLE_STRIKE_RE = re.compile(
    r"(?:creatures?|dragons?)[^.\n]{0,60}?you control[^.\n]{0,60}?"
    r"(?:gains?|have|has) double strike", re.IGNORECASE)
_TEAM_POWER_DOUBLE_RE = re.compile(
    r"double the power of each[^.\n]{0,40}?you control", re.IGNORECASE)
#: The keyword on the card itself — its own damage counts twice, and nobody
#: else's. A different scope from the three above and kept separate for it.
_SELF_DOUBLE_STRIKE_RE = re.compile(r"(?:^|[\s,;(])double strike", re.IGNORECASE)


#: WHAT MAKES A CARD AN ETB ENGINE. Listed twice in the loop before this
#: existed — once for a cast creature, once for a cast noncreature — so adding a
#: payoff channel meant remembering both, and the two new drain channels are
#: exactly the change that would have been made in one place and not the other.
_ETB_ENGINE_FIELDS = ("etb_damage_self_power", "etb_damage_count",
                      "etb_damage_fixed", "etb_life_loss_fixed",
                      "token_created_life_loss", "etb_token_bodies", "etb_copy")


def is_etb_engine(combat):
    """Does this card fire on something arriving? One predicate, one home."""
    return any(combat[f] for f in _ETB_ENGINE_FIELDS)


# ── drain: the pillar the model could not see ───────────────────────────────
#
# THREE CARDS IN zur-enchantress CONVERT LIFE GAINED INTO DAMAGE DEALT, and
# until 2026-09-04 this model scored every one of them as its body and nothing
# else. Vito, Thorn of the Dusk Rose read as `power 1`. Sanctum of Stone Fangs
# fed NO channel at all. There was no drain metric anywhere in a metrics
# document — the only occurrence of the word was the LABEL of an assembly
# target, which measures whether the cards were DRAWN.
#
# The consequence was worse than a missing figure. `kill_by_turn_rate` was
# combat-only, so any change trading a body for a drain effect could ONLY ever
# measure as a loss, and the model kept reporting that the deck's declared third
# pillar was worthless.
#
# ONE TRACKED OPPONENT. `opponent_life` is a single pool, so "target opponent
# loses that much" credits the full amount and "each opponent loses N" credits N
# — the other seats' losses are real but do not help kill the one being tracked.
# That is a single-opponent clock, deliberately, and it is the same convention
# the combat half already uses.

#: Written-out quantities, as elsewhere in this module.
_LIFE_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
               "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

#: "whenever you gain life, ..." — the payoff half. The corpus sweep on
#: 2026-09-04 found exactly THREE distinct clause shapes across 12 cards, which
#: is why this is a narrow pattern rather than a general one:
#:     6 cards  whenever you gain life, each opponent loses 1 life.
#:     5 cards  whenever you gain life, target opponent loses that much life.
#:     1 card   whenever you gain life this turn, each opponent loses that much life.
_DRAIN_EQUAL_RE = re.compile(
    r"whenever you gain life[^.]*?, (?:target|each) opponent loses that much life", re.I)
_DRAIN_FIXED_RE = re.compile(
    r"whenever you gain life[^.]*?, each opponent loses (\w+) life", re.I)

#: Constellation and its plain-language twin. In a deck of forty enchantments
#: with a commander that puts one onto the battlefield on every attack, this is
#: the largest single drain source in the list and it was entirely unread.
#: TWO IDIOMS FOR THE SAME EVENT. Theros wrote "Constellation — Whenever this
#: creature or another enchantment you control enters"; Duskmourn writes
#: "Eerie — Whenever an enchantment you control enters and whenever you fully
#: unlock a Room". They are the same trigger and the first pattern read only the
#: older one, so Balemurk Leech — chosen precisely because it is a second Grim
#: Guardian at mana value 2 — scored as a 2-power body and drained nothing.
#:
#: Corpus sweep 2026-09-04: 23 cards use the Theros wording and 37 the plain
#: one, with 2 and 1 of them draining respectively. The clause between the
#: trigger and the effect is skipped with `[^.]*?` because the Room half sits
#: there, and it is bounded to the SENTENCE so a later unrelated clause cannot
#: be captured.
_ENCHANTMENT_ENTERS = (
    r"whenever (?:this (?:creature|enchantment) or another enchantment you control"
    r"|an(?:other)? enchantment(?: you control)?) enters[^.]*?, ")
_CONSTELLATION_DRAIN_RE = re.compile(
    _ENCHANTMENT_ENTERS + r"each opponent loses (\w+) life", re.I)
_CONSTELLATION_GAIN_RE = re.compile(
    _ENCHANTMENT_ENTERS + r"you gain (\w+) life", re.I)
#: A PERMANENT ANTHEM ON THE WHOLE TEAM. "Put X +1/+1 counters on each
#: creature you control, where X is the number of Shrines you control" — with
#: six Shrines out that is +6/+6 on every body, and the model scored Southern
#: Air Temple at ZERO, which made the entire Shrine package measure worse than
#: it is. Its second half puts one more counter on everything each time another
#: Shrine lands, so it COMPOUNDS with the count it reads.
#:
#: Corpus sweep 2026-09-05: 126 cards put +1/+1 counters on each creature you
#: control — 107 of them one at a time, 5 of them an X. Counters are permanent,
#: so this is modelled as a standing bonus to every creature rather than a
#: one-shot pump; a "until end of turn" effect is a DIFFERENT card and is not
#: matched here.
#: MASS ANIMATION — the free, static, board-wide version of the commander's
#: {1}{W}. Unread, it fed NO channel at all, so it was also never cast: the
#: sixth instance of that class.
#:
#: Parsed rather than declared because, unlike the commander's ability, this is
#: a card that may or may not be in any given 99 and the deck should not have to
#: announce it.
#:
#: CORPUS SWEEP 2026-09-09, and it corrected this comment. It used to read "ONE
#: CARD in the corpus — Starfield of Nyx". There are TWO, and the second is the
#: unconditional one:
#:
#:   Starfield of Nyx  "As long as you control five or more enchantments, each
#:                      other non-Aura enchantment YOU CONTROL is a creature…"
#:   Opalescence       "Each other non-Aura enchantment is a creature…"
#:
#: The broad pattern `enchantment…is a creature | enchantments…are creatures`
#: returns exactly those two across 34,890 cards. Bello, Bard of the Brambles
#: is deliberately NOT matched: it makes a flat 4/4 rather than a body whose
#: power is its mana value, which is a different effect and would need its own
#: field, and it is Gruul so no WUB list can run it.
#:
#: OPALESCENCE HAS NO CONDITION, so it is stored as threshold 1 rather than 0 —
#: `0` is the sentinel for "this card does not mass-animate" and the consumer
#: gates on `ench >= threshold`, which is always true once Opalescence itself is
#: on the battlefield. It is the strictly stronger card in a deck that runs 44
#: enchantments and was worth exactly nothing to this model.
#:
#: The one difference the model cannot see: Opalescence animates EVERY other
#: non-Aura enchantment, including opponents', where Starfield is "you control"
#: only. A goldfish has no opponents, so the two collapse here. At a real table
#: they do not, and Opalescence hands the pod bodies too.
_MASS_ANIMATE_RE = re.compile(
    r"as long as you control (\w+) or more enchantments, each other non-Aura "
    r"enchantment you control is a creature", re.I)
#: The unconditional form. Kept as its own pattern rather than folded into the
#: one above with an optional group, because the two differ in WHAT they animate
#: and a later model that cares will need to tell them apart.
_MASS_ANIMATE_ALWAYS_RE = re.compile(
    r"each other non-Aura enchantment is a creature", re.I)

_TEAM_COUNTER_ETB_RE = re.compile(
    r"(?:when|whenever)[^.]*?enters[^.]*?, put (a|X|one|two|three) \+1/\+1 "
    r"counters? on each creature you control", re.I)
_TEAM_COUNTER_ON_TYPE_RE = re.compile(
    r"whenever another ([A-Z][a-z]+) you control enters, put (a|X|one|two) "
    r"\+1/\+1 counters? on each creature you control", re.I)

_CONSTELLATION_TOKEN_RE = re.compile(
    _ENCHANTMENT_ENTERS + r"create a (\d+)/(\d+)[^.]*?creature token", re.I)

#: Recurring, on a phase this model has a turn for.
_RECURRING_GAIN_RE = re.compile(
    r"at the beginning of your (?:upkeep|first main phase|end step)[^.]*?,"
    r" you gain (\w+) life", re.I)
_RECURRING_DRAIN_RE = re.compile(
    r"at the beginning of your (?:upkeep|first main phase|end step)[^.]*?,"
    r" each opponent loses (\w+) life", re.I)

#: "each opponent loses X life and you gain X life" — the Shrine shape, which is
#: a drain AND a lifegain, so with a payoff on the battlefield it fires twice.
#: Matched on the SENTENCE, because Bastion of Remembrance uses the identical
#: wording on a DEATH trigger and nothing dies in this simulation — scoring it
#: as recurring would have invented a drain engine out of a card that cannot
#: fire here at all.
#: RECURRING ONLY. The first version matched the clause anywhere in a sentence,
#: so Northern Air Temple's ONE-SHOT ETB — "When Northern Air Temple enters,
#: each opponent loses X life and you gain X life" — was scored as a drain that
#: fires EVERY TURN. Caught 2026-09-05 by the poh-procedures agent, which read
#: the card and said the deck has no first-main-phase trigger except Sanctum of
#: Stone Fangs. It was right; the model had been inflating the drain figure
#: since the channel shipped, and the Shrine package's measured value rested
#: partly on it.
_SYMMETRIC_DRAIN_GAIN_RE = re.compile(
    r"(at the beginning of[^.]*?each opponent loses (\w+) life and you gain "
    r"(?:\w+) life[^.]*)\.", re.I)
#: The one-shot version of the same clause: pays once, on entry.
_SYMMETRIC_DRAIN_GAIN_ETB_RE = re.compile(
    r"(when [^.]*?enters[^.]*?, each opponent loses (\w+) life and you gain "
    r"(?:\w+) life[^.]*)\.", re.I)
#: And the per-type version: "whenever another Shrine you control enters, each
#: opponent loses 1 life and you gain 1 life" — the half that makes a Shrine
#: count compound.
_SYMMETRIC_DRAIN_GAIN_PER_TYPE_RE = re.compile(
    r"whenever another ([A-Z][a-z]+) you control enters, each opponent loses "
    r"(\w+) life and you gain (?:\w+) life", re.I)

#: Lifelink the card HAS, or grants to a creature it will keep — not a pump that
#: expires and not a token it makes. Corpus sweep 2026-09-04: a naive
#: `\blifelink\b` matches 737 cards; stripping these two forms and re-testing
#: keeps 607 and drops 130, every one of them a temporary grant ("gains lifelink
#: until end of turn") or a token-maker. The strip-then-test shape matters: a
#: scoped positive pattern dropped Behemoth Sledge ("has trample AND lifelink")
#: and Fear of Infinity ("Flying, lifelink"), both of which do have it.
#: A STATIC GRANT OF LIFELINK TO A WHOLE TYPE. Zur, Eternal Schemer says
#: "Enchantment creatures you control have deathtouch, lifelink, and hexproof",
#: which in a deck of twenty-odd enchantment creatures turns EVERY body into a
#: drain source — this list converts life gained into life lost three ways.
#: Read per-card lifelink only, the commander contributed nothing.
#:
#: Corpus sweep 2026-09-06: 7 cards grant lifelink to a named type this way.
#: Narrow, so the subject is taken as the WORD BEFORE "creatures" rather than a
#: general noun phrase — "Flying Enchantment creatures" must yield Enchantment.
_LIFELINK_GRANT_TYPE_RE = re.compile(
    r"(\w+) creatures you control have (?:[^.]*?\b)?lifelink", re.I)

_LIFELINK_NOT_SELF_RE = re.compile(r"gains? lifelink|token[^.]*?with lifelink", re.I)
_LIFELINK_RE = re.compile(r"\blifelink\b", re.I)

#: Per-creature-arrival gain (Daxos).
_ARRIVAL_GAIN_RE = re.compile(
    r"whenever another creature you control enters[^.]*?, you gain (\w+) life", re.I)


#: "where X is the number of <SUBJECT> you control" — the subjects this model
#: can actually COUNT on its battlefield, mapped to the word that identifies one
#: in a type line. A CLOSED SET on purpose.
#:
#: Corpus sweep 2026-09-04: 250 cards use the phrasing at all, but only 24 use
#: it to scale a drain or a gain, and their subjects split cleanly. These seven
#: are plain type or subtype counts. The rest — "colors among permanents",
#: "basic land types among lands", "different color pairs among permanents",
#: "creatures with defender", "artifact tokens" — are not counts of a type and
#: keep the conservative 1 rather than getting a wrong number. Land-based
#: subjects ("swamps", "nonbasic lands") are absent because this model tracks
#: lands as a COUNT and not as permanents with type lines.
_X_SUBJECTS = {
    "creatures": "Creature", "artifacts": "Artifact", "zombies": "Zombie",
    "shrines": "Shrine", "knights": "Knight", "auras": "Aura",
    "enchantments": "Enchantment",
}
_X_SCALES_RE = re.compile(
    r"where X is the number of ([A-Za-z' ]+?) you control", re.I)


def _life_amount(word):
    """`X` and `that many` are board-dependent, so they take the conservative 1
    — the same call `treasure_profile` makes for "for each" and "equal to". It
    UNDERSTATES a Shrine whose X is the Shrine count, and that is the direction
    to be wrong in."""
    if word is None:
        return 0
    return _LIFE_WORDS.get(word.strip().lower(), 1)


#: AN EFFECT THAT MAKES ATTACKING FREE. Narrow on purpose: a bare
#: "can't be blocked" matches 1306 cards in the corpus, most of them about
#: somebody else's creature or a conditional restriction. Scoped to an effect
#: attached to a creature I control, plus the one card that removes an
#: attacker from combat, it matches 57 — of which 40 are Esper-legal at mana
#: value 3, including every one of zur-enchantress's own six.
#:
#: This exists because the tutor rate was a CONSTANT once it was measured, so
#: no deck change could move it and "make the commander swing more often" was
#: not a question the model could answer. With an enabler on the battlefield
#: there is no combat cost to attacking, so the rate rises.
_ATTACK_ENABLER_RE = re.compile(
    r"enchanted creature[^.]*?can't be blocked"
    r"|enchanted creature[^.]*?protection from creatures"
    r"|equipped creature[^.]*?can't be blocked"
    r"|remove target attacking creature you control from combat", re.I)


def drain_profile(card):
    """How this card turns life into damage, and how it gains life to do it.

    `unmodelled` is set when a card clearly drains through a channel with no
    event here — death triggers above all, since nothing dies in this
    simulation — so the gap is surfaced rather than silently scoring zero.
    """
    text = card.get("oracle_text", "") or ""
    out = {"payoff_equal": False, "payoff_fixed": 0,
           "gain_recurring": 0, "gain_per_enchantment": 0, "gain_per_creature": 0,
           "drain_recurring": 0, "drain_per_enchantment": 0,
           "drain_etb": 0, "gain_etb": 0,
           "drain_per_type": 0, "gain_per_type": 0, "per_type": None,
           "lifelink": False, "grants_lifelink_to": None,
           "scales_with": None, "unmodelled": None}
    if not re.search(r"gain .*life|loses? .*life|lifelink", text, re.I):
        return out

    if _DRAIN_EQUAL_RE.search(text):
        out["payoff_equal"] = True
    m = _DRAIN_FIXED_RE.search(text)
    if m:
        out["payoff_fixed"] = _life_amount(m.group(1))

    # THE SHRINE SHAPE FIRST, because it sets BOTH halves off one trigger and
    # the individual patterns below would otherwise claim the drain and leave
    # the gain at zero — which is exactly what Sanctum of Stone Fangs did on the
    # first pass, scoring a drain-and-gain card as a drain only.
    m = _SYMMETRIC_DRAIN_GAIN_RE.search(text)
    if m and "dies" not in m.group(1).lower():
        n = _life_amount(m.group(2))
        out["drain_recurring"] = out["gain_recurring"] = n
    m = _SYMMETRIC_DRAIN_GAIN_ETB_RE.search(text)
    if m and "dies" not in m.group(1).lower():
        n = _life_amount(m.group(2))
        out["drain_etb"] = out["gain_etb"] = n
    m = _SYMMETRIC_DRAIN_GAIN_PER_TYPE_RE.search(text)
    if m:
        out["per_type"] = m.group(1)
        out["drain_per_type"] = out["gain_per_type"] = _life_amount(m.group(2))

    for rx, key in ((_CONSTELLATION_GAIN_RE, "gain_per_enchantment"),
                    (_ARRIVAL_GAIN_RE, "gain_per_creature"),
                    (_RECURRING_GAIN_RE, "gain_recurring")):
        m = rx.search(text)
        if m and not out[key]:
            out[key] = _life_amount(m.group(1))

    for rx, key in ((_CONSTELLATION_DRAIN_RE, "drain_per_enchantment"),
                    (_RECURRING_DRAIN_RE, "drain_recurring")):
        m = rx.search(text)
        if m and not out[key]:
            out[key] = _life_amount(m.group(1))

    out["lifelink"] = bool(_LIFELINK_RE.search(text)) and bool(
        _LIFELINK_RE.search(_LIFELINK_NOT_SELF_RE.sub("", text)))

    m = _LIFELINK_GRANT_TYPE_RE.search(text)
    if m:
        out["grants_lifelink_to"] = m.group(1).capitalize()

    # X IS A COUNT, AND SCORING IT AS 1 MAKES A SCALING CARD UNABLE TO SCALE.
    # Sanctum of Stone Fangs drains "X, where X is the number of Shrines you
    # control" — with the flat 1 the model could never show a second Shrine
    # doing anything, which is precisely the question the pilot asked of them.
    m = _X_SCALES_RE.search(text)
    if m:
        out["scales_with"] = _X_SUBJECTS.get(m.group(1).strip().lower())

    if not any((out["payoff_equal"], out["payoff_fixed"], out["gain_recurring"],
                out["gain_per_enchantment"], out["gain_per_creature"],
                out["drain_recurring"], out["drain_per_enchantment"],
                out["drain_etb"], out["drain_per_type"], out["lifelink"])):
        # A card that plainly drains but through an event this model has none
        # of. Death triggers are the big class: nothing dies here.
        if re.search(r"each opponent loses|target opponent loses", text, re.I):
            out["unmodelled"] = card.get("name")
    return out


def combat_profile(card):
    """What this card does once there is a combat step.

    Returns a dict the simulation reads directly. `unreadable` is set when the
    card clearly has a combat trigger whose EFFECT the parser cannot price —
    those are surfaced in the metrics rather than silently scoring zero.
    """
    text = card.get("oracle_text", "") or ""
    type_line = card.get("type_line", "") or ""
    is_creature = "Creature" in type_line

    profile = {
        "is_creature": is_creature,
        "power": _stat(card.get("power")) if is_creature else 0,
        "haste": bool(_HASTE_RE.search(text)),
        # Read for the flying-gated grant (Dragon Tempest) only.
        "flying": is_creature and bool(_FLYING_KW_RE.search(text)),
        "type_line": type_line,
        # Who this card gives haste to; see _TEAM_HASTE_RE.
        "team_haste": team_haste_grant(text),
        "token_power": 0,
        "token_bodies": 0,
        # A TOKEN PER ENCHANTMENT ENTERING, which in a Zur deck is a token per
        # attack. Read as a ONE-OFF before this, so Archon of Sun's Grace — the
        # best card on the constellation branch — was priced at a single 2/2
        # forever. Corpus sweep 2026-09-04: four cards in the whole corpus.
        "enchantment_token_power": 0,
        "enchantment_token_bodies": 0,
        # A standing +N/+N on every creature, and the type whose count sets N.
        "mass_animate_threshold": 0,
        "team_counters_etb": 0,
        "team_counters_scale_type": None,
        "team_counters_per_type": 0,
        "attack_mana": 0,
        "attack_treasure": 0,
        "attack_draw": 0,
        "attack_damage": 0,
        "attack_token_power": 0,
        "attack_token_bodies": 0,
        # The token's power is the best OTHER attacker's (Ghalta and Mavren).
        "attack_token_scales": False,
        "damage_scales_with_treasure": False,
        "extra_combat_free": False,
        "extra_combat_cost": None,
        # x2 per source, multiplied together across everything in play.
        "team_damage_multiplier": 1,
        "double_strike": False,
        # POISON, see the regexes above. `infect` turns this creature's damage
        # into counters; `toxic` adds N counters per connect; the ping is a
        # per-attacker trigger on some OTHER permanent, credited to the attacker.
        "infect": False,
        "toxic": 0,
        "attack_ping_per_attacker": 0,
        # The enters-the-battlefield family. Read for every card, acted on only
        # under model_combat, so a deck that does not opt in is byte-identical.
        "etb_damage_self_power": False,
        "etb_damage_count": False,
        "etb_damage_fixed": 0,
        # Life loss on arrival, and on token creation. Same quantity as damage
        # against one opponent at 40; kept apart so a model that grows lifelink
        # or damage prevention can tell them apart. See _ETB_LIFE_LOSS_RE.
        "etb_life_loss_fixed": 0,
        "token_created_life_loss": 0,
        "etb_token_power": 0,
        "etb_token_bodies": 0,
        "etb_copy": False,
        # None, a creature type, or "chosen" (the deck's chosen type).
        "etb_type_gate": None,
        # A spell: the biggest body's power dealt to the opponent at cast.
        "spell_damage_greatest_power": False,
        # A permanent: N damage on casting a creature of at least this power.
        "cast_damage": 0,
        "cast_damage_power_min": 0,
        "etb_copy_cost": 0,
        "etb_copy_keeps_legendary": True,
        "etb_nontoken_only": False,
        "unreadable": None,
    }

    tok = _TOKEN_CREATED_TRIGGER_RE.search(text)
    if tok:
        made = _etb_clause(text, tok.start(), head=_TOKEN_CLAUSE_HEAD_RE)
        drained = _ETB_LIFE_LOSS_RE.search(made)
        if drained and not _ETB_THIS_TURN_RE.search(made):
            profile["token_created_life_loss"] = int(drained.group(1))

    if _ETB_CHOSEN_TYPE_COPY_RE.search(text):
        profile["etb_copy"] = True
        profile["etb_nontoken_only"] = True
        profile["etb_type_gate"] = "chosen"
    if ("Instant" in type_line or "Sorcery" in type_line) and _SPELL_DAMAGE_POWER_RE.search(text):
        profile["spell_damage_greatest_power"] = True
    cd = _CAST_DAMAGE_POWER_RE.search(text)
    if cd:
        profile["cast_damage_power_min"] = int(cd.group(1))
        profile["cast_damage"] = int(cd.group(2))
    etb = _ETB_TRIGGER_RE.search(text)
    if etb:
        win = text[etb.start():etb.start() + 220]
        subj = _ETB_SUBJECT_RE.search(text)
        if subj and subj.group(1) in _corpus_creature_types():
            profile["etb_type_gate"] = subj.group(1)
        if _ETB_DMG_POWER_RE.search(win):
            profile["etb_damage_self_power"] = True
        if _ETB_DMG_COUNT_RE.search(win):
            profile["etb_damage_count"] = True
        fixed = _ETB_DMG_FIXED_RE.search(win)
        if fixed:
            profile["etb_damage_fixed"] = int(fixed.group(1))
        # READ FROM THE CLAUSE, NOT THE WINDOW — the sweep at _ETB_CLAUSE_END_RE
        # is the argument. A `deals N damage` payload two sentences downstream
        # still belongs to the trigger (Crossbones); a `loses N life` two
        # sentences downstream belongs to a DIFFERENT trigger (Elas il-Kor).
        drain = _ETB_LIFE_LOSS_RE.search(_etb_clause(text, etb.start()))
        if drain and not _ETB_THIS_TURN_RE.search(_etb_clause(text, etb.start())):
            profile["etb_life_loss_fixed"] = int(drain.group(1))
        profile["etb_nontoken_only"] = bool(_ETB_NONTOKEN_RE.search(win))
        if _ETB_COPY_RE.search(win):
            profile["etb_copy"] = True
            cost = _ETB_OPTIONAL_COST_RE.search(win)
            profile["etb_copy_cost"] = _mana_pips(cost.group(1)) if cost else 0
            profile["etb_copy_keeps_legendary"] = not _ETB_COPY_NONLEGENDARY_RE.search(win)
        else:
            for tok in _TOKEN_PT_RE.finditer(win):
                if any(k in (tok.group(4) or "").lower() for k in _NONCREATURE_TOKENS):
                    continue
                word = tok.group(1).lower()
                count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
                profile["etb_token_bodies"] += count
                profile["etb_token_power"] += count * _stat(tok.group(2))

    if (_DAMAGE_DOUBLER_RE.search(text) or _TEAM_DOUBLE_STRIKE_RE.search(text)
            or _TEAM_POWER_DOUBLE_RE.search(text)):
        profile["team_damage_multiplier"] = 2
    elif is_creature and _SELF_DOUBLE_STRIKE_RE.search(text):
        # Its OWN damage twice. `elif` because a card that grants the team
        # double strike and also has it would otherwise be counted twice for
        # its own body — the grant already covers it.
        profile["double_strike"] = True

    # Creature tokens this card makes, with their power.
    # AN ACTIVATION IS NOT A CAST TRIGGER. This loop read the whole text, so a
    # token behind "{T}:" (Bloodline Keeper), "{3}{R}:" (Den of the Bugbear) or
    # "{4}:" (Ingris's Cadet) was credited as a free body the turn the card was
    # cast — and cast again never, so the model paid once for a thing that
    # costs every time and the Forge AI mostly never buys. Measured 2026-09-10:
    # nine fleet cards across four decks. A token whose sentence sits inside an
    # activated ability's effect is skipped here; what an activation is worth
    # is a different model and it is not priced at zero by accident, it is
    # priced at zero and named.
    for match in _TOKEN_PT_RE.finditer(text):
        tail = (match.group(4) or "").lower()
        if any(k in tail for k in _NONCREATURE_TOKENS):
            continue
        if _inside_activation(text, match.start()):
            continue
        word = match.group(1).lower()
        count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
        profile["token_bodies"] += count
        profile["token_power"] += count * _stat(match.group(2))

    if is_creature:
        profile["infect"] = bool(_INFECT_KW_RE.search(text))
        _tox = _TOXIC_KW_RE.search(text)
        profile["toxic"] = int(_tox.group(1)) if _tox else 0
    _ping = _ATTACK_PING_EACH_RE.search(text)
    if _ping:
        profile["attack_ping_per_attacker"] = int(_ping.group(1))
    combat_trigger = _ATTACKS_RE.search(text) or _COMBAT_DMG_RE.search(text)
    if combat_trigger:
        window = text[combat_trigger.start():combat_trigger.start() + 220]
        # THE WINDOW STOPS AT THE NEXT ACTIVATED ABILITY. 220 characters from
        # the trigger ran into Ingris Stingerquill's "{4}: Create a 2/2 …
        # Cadet", which then read as a free token on every attack — an
        # activation the Forge AI prices at nothing, credited by this model
        # as a trigger. Only a cost that starts a new ability AFTER the
        # trigger cuts it, so Den of the Bugbear's trigger, which lives inside
        # its activation's granted text, is unaffected. `token_bodies` below
        # still reads the whole text and still credits nine fleet cards'
        # activation tokens on cast; that is a separate, measured change.
        window = re.split(r"(?:\n| )(?=\{[^}]+\}[^:\n]{0,60}:)", window, 1)[0]
        # `_TAP_ADD_RE` now captures the whole clause rather than a symbol run,
        # so route it through the one parser instead of counting pips here —
        # two readers of one pattern is the divergence this file has paid for.
        got = produced_mana(window, card.get("type_line"))
        if not got:
            plain = re.search(r"add ((?:\{[WUBRGC0-9]\})+)", window, re.IGNORECASE)
            got = _mana_pips(plain.group(1)) if plain else 0
        if got:
            profile["attack_mana"] = got
        if re.search(r"treasure token", window, re.IGNORECASE):
            n = _TREASURE_N_RE.search(window)
            word = (n.group(1).lower() if n else "a")
            # "for each" / "equal to" counts are board-dependent; one is the
            # conservative read, matching what `treasure_profile` does.
            profile["attack_treasure"] = int(word) if word.isdigit() else \
                _NUMBER_WORDS.get(word, 1) or 1
        drawn = re.search(r"draw (\w+) cards?", window, re.IGNORECASE)
        if drawn:
            word = drawn.group(1).lower()
            profile["attack_draw"] = int(word) if word.isdigit() else \
                _NUMBER_WORDS.get(word, 1)
        if _DMG_EQUAL_TREASURE_RE.search(window):
            profile["damage_scales_with_treasure"] = True
        # Direct damage on attack (Drakuseth). Only the FIRST "deals N damage"
        # is counted: the follow-on clauses ("and 3 damage to each of up to two
        # other targets") usually point at creatures, and this model has none to
        # point at, so crediting them to the opponent's face would invent reach.
        # A PER-ATTACKER PING IS NOT A FLAT TRIGGER. `_ATTACKS_RE` matches
        # "whenever a creature you control attacks" under IGNORECASE, and this
        # line then read Ingris Stingerquill's "that creature deals 1 damage"
        # as one damage per combat — on top of the per-attacker credit below —
        # so her trigger was counted twice and the wrong half was life.
        fixed = re.search(r"deals (\d+) damage", window, re.IGNORECASE)
        if fixed and not _ATTACK_PING_EACH_RE.search(window):
            profile["attack_damage"] = int(fixed.group(1))
        # Creature tokens made on attack (Utvara Hellkite). Counted ONCE per
        # combat even where the trigger is per-attacker, for the same reason.
        for tok in _TOKEN_PT_RE.finditer(window):
            if any(k in (tok.group(4) or "").lower() for k in _NONCREATURE_TOKENS):
                continue
            word = tok.group(1).lower()
            count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
            profile["attack_token_bodies"] += count
            profile["attack_token_power"] += count * _stat(tok.group(2))
        if _ATTACK_TOKEN_SCALES_RE.search(window):
            profile["attack_token_bodies"] += 1
            profile["attack_token_scales"] = True
        if not any((profile["attack_mana"], profile["attack_treasure"],
                    profile["attack_draw"], profile["damage_scales_with_treasure"],
                    profile["attack_damage"], profile["attack_token_bodies"],
                    # A multiplier IS priced now, so a card carrying one must
                    # not be reported as unreadable — that list is a promise
                    # about what the figures leave out.
                    profile["team_damage_multiplier"] > 1,
                    profile["double_strike"],
                    profile["attack_ping_per_attacker"],
                    # Checked against the FULL text, not the window: Scourge of
                    # the Throne's reminder clause pushes "additional combat
                    # phase" past 220 characters, and flagging a card whose
                    # effect IS modelled makes the not-modelled list a liar.
                    _EXTRA_COMBAT_RE.search(text))):
            profile["unreadable"] = card.get("name")

    if _EXTRA_COMBAT_RE.search(text):
        activated = _ACTIVATED_COMBAT_RE.search(text)
        if activated:
            profile["extra_combat_cost"] = _mana_pips(activated.group(1))
        elif combat_trigger:
            profile["extra_combat_free"] = True
        else:
            # AN EXTRA COMBAT THIS MODEL CANNOT PLACE, AND IT WAS SILENT.
            # Neither activated (no mana cost binds) nor triggered on an attack:
            # a one-shot spell ("After this main phase, there is an additional
            # combat phase"), or a permanent keyed on being BLOCKED, on exert,
            # on landfall, or on a loyalty ability. The model has no channel for
            # any of those, which is a boundary rather than a bug — but it fell
            # through both branches and set nothing, so the card contributed
            # nothing to the clock AND appeared in no not-modelled list.
            #
            # Corpus-wide that is 32 cards; on this fleet it is ONE
            # (goblin-storm's Great Train Heist). Naming it is what keeps a low
            # kill figure legible, the same contract
            # `treasure_sources_not_modelled` keeps.
            profile["unreadable"] = card.get("name")

    # ONE TOKEN PER ENCHANTMENT, not one ever. Bounded to the sentence so a
    # later unrelated "create a token" clause cannot be captured.
    m = _CONSTELLATION_TOKEN_RE.search(text)
    if m:
        profile["enchantment_token_power"] = int(m.group(1))
        profile["enchantment_token_bodies"] = 1

    m = _MASS_ANIMATE_RE.search(text)
    if m:
        profile["mass_animate_threshold"] = _LIFE_WORDS.get(m.group(1).lower(), 5)
    elif _MASS_ANIMATE_ALWAYS_RE.search(text):
        profile["mass_animate_threshold"] = 1

    m = _TEAM_COUNTER_ETB_RE.search(text)
    if m:
        word = m.group(1).lower()
        profile["team_counters_etb"] = (
            -1 if word == "x" else {"a": 1, "one": 1, "two": 2, "three": 3}[word])
    m = _TEAM_COUNTER_ON_TYPE_RE.search(text)
    if m:
        profile["team_counters_scale_type"] = m.group(1)
        word = m.group(2).lower()
        profile["team_counters_per_type"] = {"a": 1, "one": 1, "two": 2}.get(word, 1)

    return profile


#: THE MODEL WAS COLOURLESS, AND `mana_analysis` HAS ALWAYS SAID IT MATTERED.
#: `spend()` took a scalar, so a five-colour Ur-Dragon and a mono-green Radagast
#: with the same land count had identical curves. Measured against the closed
#: form one module over, the BINDING colour is available on curve 56%-99% of the
#: time across the fleet (median 82%; ur-dragon's black is 56%) — and the
#: simulation was casting at 100%. That is the largest single accuracy gap in
#: the resource model, and it ran OPPOSITE to the rock blindness above, so the
#: two partly cancelled and the total stayed plausible while both halves were
#: wrong.
#: A PIP, FOR CASTING. `manabase.count_pips` answers a different question and
#: answers it correctly: it half-charges a hybrid to each side, which is right
#: for SIZING a base (a {W/U} spell really is castable off either, so charging
#: both a full pip over-builds). For CASTABILITY a hybrid is one pip payable two
#: ways, and half a pip is not a thing you can pay. Same split as `bodies` vs
#: `creature_bodies` two functions down — one concept, two questions, and
#: forcing them into one reader is how this file has been bitten before.
_CAST_PIP_RE = re.compile(r"\{([^}]+)\}")


#: A ROOM IS TWO CARDS UNDER ONE TYPE LINE, AND SCRYFALL'S `cmc` IS BOTH OF THEM.
#:
#: CR 202.3d takes a split card's mana value from the combined costs of its
#: halves, so `Bottomless Pool // Locker Room` reports `cmc` 6.0 — while the door
#: you actually cast costs `{U}`. `classify` spent straight from that field, so
#: the model charged SIX MANA FOR A ONE-MANA ENCHANTMENT and every Room in a
#: branch sat in hand. All 30 Rooms in the corpus share one exact type line, so
#: detection is unambiguous rather than a heuristic.
#:
#: THE CHEAPER DOOR IS CAST, NOT THE FRONT ONE, and the sweep is what settles it:
#: the front door is cheaper-or-equal on 24 of 30, but `Defiled Crypt // Cadaver
#: Lab` is `{3}{B} // {B}` — front 4, back 1 — so a front-door rule would be a
#: fourfold error on six cards. The PIPS come from the same door for the same
#: reason; `front_field` cannot be reused here because it always answers with the
#: left half, which is the wrong half exactly when the cost is.
_ROOM_TYPE_RE = re.compile(
    r"Enchantment\s+—\s+Room\s*//\s*Enchantment\s+—\s+Room", re.I)

#: EERIE IS NOT CONSTELLATION AND THE DIFFERENCE IS THE WHOLE POINT OF A ROOM.
#: "Constellation — whenever … an enchantment you control enters" fires once, when
#: the Room is cast. Eerie reads "whenever an enchantment you control enters AND
#: WHENEVER YOU FULLY UNLOCK A ROOM", so it fires a SECOND time for no card. 17
#: cards in the corpus carry the clause, four of them already in zur-enchantress,
#: where it has never once been able to trigger because the deck owns no Rooms.
#: Crediting an unlock to every per-enchantment payoff would over-pay the five
#: plain constellation cards, which is why this is a separate flag.
_EERIE_RE = re.compile(r"fully unlock a Room", re.I)

_GENERIC_PIP_RE = re.compile(r"\{(\d+)\}")


def mana_value(mana_cost):
    """Mana value of ONE cost string — generic plus every coloured symbol."""
    generic = sum(int(n) for n in _GENERIC_PIP_RE.findall(mana_cost or ""))
    return generic + len(cast_pips(mana_cost))


def room_profile(card):
    """Door costs for a Room, or None for every other card in the game.

    `entry` is the door the model casts and `unlock` the one it may open later
    as a special action (CR 709.5e: sorcery speed, own main phase, empty stack).
    `full_mv` is what the permanent's mana value becomes once BOTH doors are
    unlocked — CR 709.5 says a permanent does not have the mana cost of a locked
    half, so a half-open Room's mana value is just the open door, which is
    already what `cmc` carries. That is what `model_commander_animate` reads to
    set base power, so an 8-mana Room is an 8/8 only after it is fully open.
    """
    if not _ROOM_TYPE_RE.search(card.get("type_line") or ""):
        return None
    halves = (card.get("mana_cost") or "").split(" // ")
    if len(halves) != 2:
        return None
    costs = [mana_value(h) for h in halves]
    entry = 0 if costs[0] <= costs[1] else 1
    # SCRYFALL CONCATENATES BOTH HALVES INTO `oracle_text`, so every text-derived
    # profile read a LOCKED door's rules text. `Unholy Annex // Ritual Chamber`
    # was charged 3 for its {2}{B} front half AND handed the 6/6 Demon printed on
    # the {3}{B}{B} back half at the same instant. Splitting the text on the same
    # separator as the cost, in the same order, is what lets each door be paid
    # for and credited separately.
    faces = [t.strip() for t in
             (card.get("oracle_text") or "").split(" // ")]
    if len(faces) != 2:
        faces = ["", ""]
    return {
        "entry_cost": costs[entry],
        "entry_pips": cast_pips(halves[entry]),
        "entry_text": faces[entry],
        "unlock_cost": costs[1 - entry],
        "unlock_pips": cast_pips(halves[1 - entry]),
        "unlock_text": faces[1 - entry],
        "full_mv": costs[0] + costs[1],
    }


def cast_pips(mana_cost):
    """One entry per coloured pip: the set of colours that can pay it."""
    out = []
    for symbol in _CAST_PIP_RE.findall(mana_cost or ""):
        inner = symbol.upper()
        # {2/W} is payable with two generic OR one white; a goldfish that has
        # the mana always has the two, so it never constrains a colour.
        if any(ch.isdigit() for ch in inner):
            continue
        colours = frozenset(c for c in inner.split("/") if c in "WUBRG")
        if colours:
            out.append(colours)
    return out


#: A DORK WHOSE OUTPUT IS THE BOARD, NOT A FIXED LIST.
#:
#: `_TAP_ADD_RE` wants `{T}: Add <symbols>`. Bloom Tender and Faeburrow Elder
#: say `{T}: For each color among permanents you control, add one mana of that
#: color` — so the two best dorks a five-colour deck can run read as producing
#: NOTHING, while the conditional rocks they replace counted as five sources
#: each. That is the same silent-half-working shape as the 65% of mana rocks
#: this model could not see.
#:
#: The corpus sweep is why this is two lines rather than a family: of 34,084
#: cards, exactly FIVE have a `{T}` mana ability the old regex misses, and only
#: these two are this shape. The other three are correctly excluded — Charmed
#: Pendant pays with a mill, Idol of False Gods makes a token that sacrifices
#: itself (the Jeweled Lotus rule: a cost that consumes the source is not a
#: rate), and Rainbow Dash is an acorn card.
_SCALING_COLOR_MANA_RE = re.compile(
    r"\{T\}[^:\n]*: ?[^.\n]*for each colou?r among permanents you control,? "
    r"add one mana of that colou?r", re.IGNORECASE)


def can_pay(pips, sources, wildcards=0):
    """Can these coloured pips be paid from these sources?

    `pips` is a list of colour-sets (one per pip), `sources` one colour-set per
    untapped producer, `wildcards` the Treasures, which make any colour.
    GREEDY, MOST-CONSTRAINED PIP FIRST, and each pip takes the source with the
    FEWEST colours that can pay it — the standard assignment heuristic, exact at
    these sizes (a Commander cost is a handful of pips against a dozen sources).
    """
    if not pips:
        return True
    used = [False] * len(sources)
    def supply(pip):
        return sum(1 for c in sources if c & pip)
    for pip in sorted(pips, key=supply):
        best = -1
        for i, c in enumerate(sources):
            if used[i] or not (c & pip):
                continue
            if best < 0 or len(c) < len(sources[best]):
                best = i
        if best >= 0:
            used[best] = True
        elif wildcards > 0:
            wildcards -= 1
        else:
            return False
    return True


#: A GRANTED ABILITY BELONGS TO WHOEVER RECEIVED IT, and until 2026-08-31 this
#: function counted every one of them as the card's own. 145 corpus cards, 8 of
#: them sleeved across five decks and five of those in kinnan.
#:
#: THE OBVIOUS FIX IS WRONG AND THE SWEEP IS WHAT SAYS SO. Stripping quoted text
#: zeroes five cards that are correct today: **Citanul Hierophants** grants
#: `{T}: Add {G}` to "creatures you control" and IS a creature, so it does tap
#: for green; likewise **Gemhide Sliver** ("All Slivers"), **Enduring Vitality**
#: and **Inga and Esika**. **Dryad Arbor**'s ability sits in reminder text about
#: itself — *"it's affected by summoning sickness, and it has "{T}: Add {G}.""*.
#:
#: So the question is not "is it quoted" but **is this card a member of the class
#: it grants to**. Sorted by what introduces the grant, over all 34,890 cards:
#:
#:     23  "creates a Powerstone token. (It's an artifact with …)"  the TOKEN taps
#:     13  "Enchanted land has …" / "Enchanted creature has …"      the HOST taps
#:     21  "<Noun> you control have …"                             self IFF a <Noun>
#:      4  "…and it has …"                                          the card itself
#:      6  "Target land gains …" / "lands you control gain …"       temporary, elsewhere
#:
#: DEFAULT TO NOT-SELF when the phrasing is unrecognised. Overcounting mana tells
#: the model it can cast things it cannot, which is the failure that produced
#: this bug; undercounting only makes a deck look slower than it is.
#: Checked against the NEAR window — the clause that introduces the grant. These
#: must NOT be looked for further back: Sachi, Daughter of Seshiro opens with
#: *"OTHER Snake creatures you control get +0/+1"* and then grants to "Shamans
#: you control", which she is.
_GRANTED_AWAY = re.compile(r"\bopponent|\bother\b|\btarget\b", re.IGNORECASE)
#: `…, and it has "{T}: Add {G}."` — the card talking about ITSELF, which only
#: Dryad Arbor and Jasconian Isle do, both in parenthetical reminder text.
#:
#: A bare `it has$` was too loose and four cards proved it. In **Jiang Yanggu**
#: (*"Each creature you control with a +1/+1 counter ON IT HAS …"*) the "it" is
#: the recipient; in **Llanowar Mentor** and **The Bus Runner** it is a TOKEN
#: created one sentence earlier; in **Nature's Embrace** it is the enchanted
#: permanent. Requiring `and` or the start of the clause rejects all four, and
#: they then fall through to the class test, which cannot match a two-letter
#: noun and so returns foreign.
#:
#: A SECOND, WIDER WINDOW LOOKING FOR `token|create|enchanted|equipped|emblem`
#: WAS WRITTEN FOR THOSE FOUR CARDS AND THEN DELETED: swept across all 34,890
#: cards it changed **zero** readings, because the fallthrough above already
#: covers them. A guard that cannot fail is not a guard, it is a claim that
#: something is being checked.
_GRANT_TO_SELF = re.compile(r"(?:^|\band)\s+it has\s*$", re.IGNORECASE)
#: `Creatures you control have "…"`, `All Slivers have "…"`, `Basic lands you
#: control have "…"`. The noun phrase is the class the ability is granted to.
#: `have vigilance and "…"` — Inga and Esika grants keywords ALONGSIDE the mana
#: ability, so the verb is not the last thing before the quote. The optional tail
#: absorbs those, and refuses to cross a sentence so it cannot reach back into an
#: unrelated clause.
_GRANT_CLASS = re.compile(
    r"(?:^|[.;]\s*|\bAll\s+)(?P<noun>[A-Za-z][A-Za-z' ]{2,40}?)"
    r"(?:\s+you control)?\s+(?:have|has|gain|gains)"
    r"(?:\s+[A-Za-z,' ]{1,60}?\s+and)?\s*$", re.IGNORECASE)
#: Plurals the naive rule gets wrong. `Elves` is the one that matters — Thranduil
#: is an Elf Noble — though `Other` already excludes that card on its own.
_IRREGULAR = {"elves": "elf", "dwarves": "dwarf", "thieves": "thief",
              "wolves": "wolf", "leaves": "leaf"}


def _singular(word):
    low = word.lower()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", "")):
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            return low[: -len(suffix)] + replacement
    return low


def _grant_is_to_self(text, quote_start, type_line):
    """Does a quoted ability starting at `quote_start` belong to THIS card?"""
    # THE WINDOW STOPS AT THE CLAUSE THAT INTRODUCES THE GRANT. Reading a flat
    # 90 characters crossed sentences, and Sachi, Daughter of Seshiro paid for
    # it: her first line is *"OTHER Snake creatures you control get +0/+1"* and
    # her second grants `{T}: Add {G}{G}` to "Shamans you control" — which she
    # is. The stray "Other" from the line above marked her own ability foreign.
    #
    # NOT `.rstrip("and")` either — that strips any of those CHARACTERS, so a
    # window ending in "command" becomes "comm". `_GRANT_CLASS` matches the
    # trailing `and` as a word instead.
    window = text[max(0, quote_start - 90):quote_start]
    for boundary in ("\n", ". ", "; ", "• "):
        if boundary in window:
            window = window.rsplit(boundary, 1)[1]
    window = window.replace("\n", " ").rstrip()
    if _GRANT_TO_SELF.search(window):
        return True
    if _GRANTED_AWAY.search(window):
        return False
    match = _GRANT_CLASS.search(window)
    if not match:
        return False
    line = (type_line or "").lower()
    words = [w for w in match.group("noun").split() if w.lower() != "basic"]
    # "Creature tokens you control" is caught by _GRANTED_AWAY above; what is
    # left is a plain class name, and one word of it matching the type line is
    # enough — `Basic lands` -> land, `All Slivers` -> sliver.
    return any(_singular(w) in line for w in words)


def _strip_foreign_grants(oracle_text, type_line):
    """Remove quoted abilities this card granted to something else."""
    text = oracle_text or ""
    out, at = [], 0
    for match in re.finditer(r'"[^"]*"', text):
        if not _grant_is_to_self(text, match.start(), type_line):
            out.append(text[at:match.start()])
            at = match.end()
    out.append(text[at:])
    return "".join(out)


def produced_mana(oracle_text, type_line=""):
    """Mana a persistent '{T}: Add ...' producer yields per turn (0 if none).

    `type_line` decides whether a QUOTED ability is this card's own. It defaults
    to empty, which reads every granted ability as foreign — the conservative
    direction, and byte-identical to the old behaviour for the 34,745 cards that
    grant nothing.
    """
    oracle_text = _strip_foreign_grants(oracle_text, type_line)
    match = _TAP_ADD_RE.search(oracle_text or "")
    if not match:
        return 0
    # The activation cost is everything from `{T}` back to the clause start.
    cost = (oracle_text or "")[max(0, match.start() - 40):match.start()
                               + match.group(0).index(":")]
    if _CONSUMING_COST.search(cost):
        return 0
    body = match.group(1)
    # ALTERNATIVES ARE A CHOICE, NOT A SUM. `Add {R}, {G}, or {W}` is ONE mana
    # and counting the symbols gives three; `Add {U} or {C}{U}` is two, not
    # three. So take the LARGEST CONSECUTIVE RUN rather than the total — which
    # is also exactly what the old narrow pattern did by accident, since
    # `(?:\{..\})+` stopped at the first comma. Widening the match without
    # this reintroduced the bug as an overcount on every dual-choice rock.
    runs = [len(re.findall(r"\{[WUBRGC0-9]\}", r))
            for r in re.findall(r"(?:\{[WUBRGC0-9]\})+", body)]
    if runs:
        return max(runs)
    word = re.match(r"\s*(one|two|three|four|five|X)\b", body, re.IGNORECASE)
    return _MANA_WORDS[word.group(1).lower()] if word else 0


def body_count(card):
    """Bodies this card contributes when cast: itself (if creature) + tokens."""
    bodies = 1 if "Creature" in card.get("type_line", "") else 0
    for word in _TOKEN_RE.findall(card.get("oracle_text", "") or ""):
        bodies += _NUMBER_WORDS.get(word.lower(), 1 if not word.isdigit() else int(word))
    return bodies


#: STATIC COST REDUCTION — `<Type> spells you cast cost {N} less to cast`.
#:
#: THE MODEL SAID "cost reducers are not modeled (conservative)" AND FOR THIS
#: FLEET THAT IS NOT CONSERVATIVE, IT IS WRONG ABOUT THE THESIS. The Ur-Dragon's
#: eminence takes {1} off every Dragon spell from the COMMAND ZONE — always on,
#: from turn one, unremovable — which is 22 of its 24 creatures and takes the
#: mean Dragon from 5.73 to 4.73. Four more reducers sit in the 99 and read as
#: vanilla bodies. A mv5 Dragon with eminence and one Dragonlord's Servant costs
#: 3 in paper and 5 in the model, so every figure about when a threat lands was
#: measured in a world where the commander's ability does not exist.
#:
#: Same class as the two already in the record: the model could not see 65% of
#: the fleet's mana rocks, and the mana model was colourless.
#:
#: The subtype capture is CASE-SENSITIVE and deliberate: `Dragon` is a creature
#: type and `dragon` in flavour prose is not.
_COST_REDUCTION_RE = re.compile(
    r"(?P<other>other )?(?P<what>[A-Z][a-z]+) spells"
    r"(?: you cast)?(?P<chosen> of the chosen type)? cost "
    r"\{(?P<amount>\d)\} less to cast")

#: What a reduction applies to when the card says "the chosen type". A real
#: player names the type they built around, so the model resolves it to the
#: deck's most common creature subtype — stated, because it is a choice the
#: model is making on the pilot's behalf.
CHOSEN_TYPE = "\x00chosen"


#: A REDUCTION THAT SCALES IS NOT A RATE, and the corpus sweep is what caught it.
#: `Creature spells you cast cost {1} less to cast FOR EACH …` — Hamza counts
#: +1/+1 counters, Animar counts counters on itself and starts at zero, and
#: Rakdos counts life the opponents lost this turn, which in a solitaire model
#: is zero forever. The regex stops at "to cast" and would have reported a flat
#: 1 for all three: a plausible number that is wrong, which is the Jeweled Lotus
#: failure exactly. Refused, and named as unmodelled rather than counted.
_SCALING_REDUCTION_RE = re.compile(r"\A for each\b")

#: WHAT A REDUCTION CAN APPLY TO, checked against the corpus's own creature
#: types rather than a word list. The sweep found the regex capturing
#: `Noncreature` (7 cards), `Artifact` (7), `Equipment` (5), `Enchantment` (4)
#: and six colour words (14) — none of them creature subtypes, all of them
#: silently matching nothing once tested against a card's subtypes. A silent
#: half-working matcher is the most expensive kind, so these are REFUSED here
#: instead of quietly reducing nothing. Artifact and noncreature reduction is
#: real and simply not modelled; saying so is the difference.
_ALL_CREATURES = ("Creature",)


def cost_reduction(card, creature_types=None):
    """`(amount, applies_to, excludes_self)` for a static reducer, else None.

    `applies_to` is a creature subtype, `CHOSEN_TYPE`, or None for every
    creature spell. GENERIC ONLY — a reduction can never pay a coloured pip,
    which is why the caller floors the result at the pip count rather than at
    zero.
    """
    text = card.get("oracle_text") or ""
    got = _COST_REDUCTION_RE.search(text)
    if not got:
        return None
    if _SCALING_REDUCTION_RE.match(text[got.end():]):
        return None
    what = got.group("what")
    if got.group("chosen"):
        applies = CHOSEN_TYPE
    elif what in _ALL_CREATURES:
        applies = None
    elif creature_types is not None and what not in creature_types:
        return None
    elif creature_types is None and what in _NOT_A_CREATURE_TYPE:
        return None
    else:
        applies = what
    return (int(got.group("amount")), applies, bool(got.group("other")))


def _corpus_creature_types():
    """The corpus's own creature types, memoised.

    `analysis.common.creature_types` is the one scan, shared with `assess` and
    `power_creep`, so the triage that WARNS a pilot and the model that PRICES
    their deck cannot drift on what a tribe is. Returns None when there is no
    corpus, and `cost_reduction` then falls back to its own refusal list — a
    unit test with no data behind it still rejects the right words.
    """
    global _CREATURE_TYPES_CACHE
    if _CREATURE_TYPES_CACHE is _UNSET:
        try:
            from manamap.analysis import common as _acommon
            from manamap.pilot import card_pool
            _CREATURE_TYPES_CACHE = _acommon.creature_types(card_pool.load_frame())
        except Exception:                      # pragma: no cover - defensive
            _CREATURE_TYPES_CACHE = None
    return _CREATURE_TYPES_CACHE or None


_UNSET = object()
_CREATURE_TYPES_CACHE = _UNSET


def reduced_cost(card, reductions, chosen=None):
    """What this spell costs with these static reducers in play.

    GENERIC ONLY, floored at the coloured pip count. A cost reduction can never
    pay a coloured pip — `{4}{W}{U}{B}{R}{G}` with three reducers out is still
    five mana, not two — and flooring at zero instead would have made a
    five-colour commander look castable off two lands.
    """
    total = 0
    for amount, applies, excludes_self in reductions:
        # `is_commander` is NOT a key `classify` emits — the commander is never
        # in the library — so it stays a `.get`. Everything else is subscripted,
        # because `test_every_signal_the_model_sets_is_read_by_something` counts
        # a subscript as the proof a flag is acted on and a `.get` would let a
        # key look read while nothing used it.
        if excludes_self and card.get("is_commander"):
            continue
        if applies is None:
            if card["is_creature"]:
                total += amount
        elif applies is CHOSEN_TYPE:
            if chosen and chosen in card["subtypes"]:
                total += amount
        elif applies in card["subtypes"]:
            total += amount
    return max(len(card["pips"]), int(card["cmc"]) - total)


#: The fallback when no corpus is loaded — the classes the sweep actually found,
#: so a unit test with no corpus behind it still refuses the right words.
_NOT_A_CREATURE_TYPE = frozenset({
    "Noncreature", "Artifact", "Equipment", "Enchantment", "Aura", "Vehicle",
    "Lair", "Lesson", "Arcane", "Historic", "Legendary", "Commander",
    "Planeswalker", "Multicolored", "Colorless",
    "White", "Blue", "Black", "Red", "Green",
})


#: "Changeling (This card is every creature type.)" — and it is every type in
#: EVERY ZONE, so a changeling SPELL on the stack is a Dragon spell and takes a
#: Dragon's discount. 61 of them are legal here and the type line says
#: `Shapeshifter`, so reading it literally makes every one of them invisible to
#: eminence, to Lathliss and Miirym's "another Dragon enters", and to a tribal
#: cost reducer. Universal Automaton is a {1} Dragon that this deck casts for
#: nothing; the model would have priced it at one.
_CHANGELING_RE = re.compile(r"\bchangeling\b|is every creature type", re.IGNORECASE)


def subtypes_of(type_line, oracle_text=None):
    """The subtypes after the em dash. `Legendary Creature — Dragon` -> {Dragon}.

    A CHANGELING IS EVERY CREATURE TYPE, which is a rules fact rather than a
    heuristic, so it answers the corpus's whole type list. Falls back to the
    literal type line when no corpus is loaded — a unit test with no data behind
    it still gets the printed types.
    """
    if oracle_text and _CHANGELING_RE.search(oracle_text):
        every = _corpus_creature_types()
        if every:
            return frozenset(every)
    if "\u2014" not in (type_line or ""):
        return frozenset()
    tail = type_line.split("\u2014", 1)[1].split("//")[0]
    return frozenset(w for w in tail.split() if w[:1].isupper())


def chosen_type_for(cards):
    """The creature subtype a player would name. Most common wins; ties by name.

    Deterministic, because everything in this model must be: a tie broken by
    dict order would make two identical decks measure differently.
    """
    counts = {}
    for c in cards:
        if "Creature" not in (c.get("type_line") or ""):
            continue
        # A CHANGELING IS EVERY TYPE, so it must not vote on which type the deck
        # is built around — it would add one to all 383 of them and the argmax
        # would become alphabetical noise.
        if _CHANGELING_RE.search(c.get("oracle_text") or ""):
            continue
        for t in subtypes_of(c.get("type_line")):
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


#: "As long as your devotion to white is less than five, Heliod isn't a creature."
#: The Theros gods, and the reason a board-power figure can be confidently wrong:
#: a God on the battlefield below its threshold is an ENCHANTMENT and nothing
#: else — it cannot attack, cannot block and has no power. Counting it as a body
#: on arrival overstates the board by its printed power for as long as the
#: threshold is unmet, which on Thassa in this list is essentially the whole game.
#:
#: Only 3 of the 23 enchantment creatures in bodies-v3 carry this clause; the
#: other 20 are creatures the moment they land. It is a narrow gate and a
#: load-bearing one.
_DEVOTION_GATE_RE = re.compile(
    r"as long as your devotion to ([\w\s]+?) is less than (\w+)", re.I)

_DEVOTION_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                   "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_DEVOTION_COLOURS = {"white": "W", "blue": "U", "black": "B",
                     "red": "R", "green": "G"}


def devotion_gate(card):
    """`{"colors": frozenset, "threshold": int}` for a God, else None.

    DEVOTION COUNTS MANA SYMBOLS, not permanents: each `{W}` in the mana cost of
    a permanent you control is one devotion to white, and a hybrid `{W/U}` counts
    for BOTH. `classify` already stores `pips` as one frozenset per coloured
    symbol, which is exactly that — so nothing new has to be parsed.
    """
    m = _DEVOTION_GATE_RE.search(card.get("oracle_text") or "")
    if not m:
        return None
    words = m.group(1).lower().replace(" and ", " ").split()
    colours = frozenset(_DEVOTION_COLOURS[w] for w in words
                        if w in _DEVOTION_COLOURS)
    raw = m.group(2).lower()
    threshold = _DEVOTION_WORDS.get(raw, int(raw) if raw.isdigit() else 0)
    if not colours or not threshold:
        return None
    return {"colors": colours, "threshold": threshold}


def devotion_of(pips_lists, colours):
    """Devotion to `colours` from every permanent's pip list on the battlefield."""
    return sum(1 for pips in pips_lists for pip in pips if pip & colours)


