"""Pilot: goldfish simulation — seeded Monte Carlo resource-development metrics.

Tier-2 (data-derived) evidence for the bench. The model simulates
resource development, NOT full games; its assumptions are stated in the output
artifact and rendered in the manual. Deterministic: same seed and deck produce
byte-identical metrics.

Model assumptions (v1):
- Multiplayer Commander: every player draws on each of their turns, turn 1 included.
- Mulligan rule: keep a 7 with 2-5 lands; otherwise redraw a fresh 7 (up to 2
  redraws), keeping the last hand regardless. No bottoming.
- One land played per turn when available.
- Persistent mana producers ("{T}: Add ...") are cast greedily after the
  commander and contribute their mana starting the following turn.
- The commander is cast on the first turn it is affordable (highest priority).
- Bodies-by-turn casts creature/token cards greedily by cost with leftover
  mana, counting the card itself (if a creature) plus tokens parsed from
  "create ... token" text. Crude by design: no interactions, no haste math.
- Cost reducers, rituals, and card draw beyond one per turn are NOT modeled;
  estimates are therefore conservative for decks that use them.
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
    GOLDFISH_SEED,
)
from manamap.pilot import manabase
from manamap.pilot.common import (
    deck_dir, deck_file, front_field, load_deck_cards)

#: THE MODEL'S OWN IDENTITY, SO STALENESS BECOMES DECIDABLE.
#:
#: Every artifact here stamped the DECK (`decklist_sha256`), the seed, the
#: iteration count and the turn limit — and nothing identified the model that
#: produced the figures. So a number computed today and one computed before a
#: model fix were indistinguishable, and when the fleet was regenerated after
#: the mana-rock and colour fixes it left **39 stale figures in authored prose
#: across four decks** with `validate-diagnosis`, `validate-strategic-frame` and
#: `validate-tutor-guide` all passing. The decklist sha had not moved, so
#: nothing could tell.
#:
#: A sha over THIS FILE's bytes. The same trick `tests/conftest.py:unchanged()`
#: and `pilot/agent_cache.py` already use, and deliberately not a hand-kept
#: integer: a version somebody has to remember to bump is one that will not be.
#: It is coarse on purpose — a comment edit bumps it, which costs a regeneration
#: nobody needed. The alternative is a curated list of "model-facing" lines,
#: which is exactly the judgement call that goes wrong silently.
def model_version():
    """First 12 hex of a sha256 over the simulator's source."""
    import hashlib
    return hashlib.sha256(
        pathlib.Path(__file__).read_bytes()).hexdigest()[:12]


MODEL_ASSUMPTIONS = [
    "Simulates resource development, not full games (no interaction, no removal).",
    "Draw every turn including turn 1 (multiplayer Commander).",
    "Mulligan: keep 7-card hands with 2-5 lands; up to 2 fresh redraws, keep the last.",
    "One land drop per turn when available.",
    # MEASURED, NOT ASSUMED. A `candidates` sweep of twelve lands against
    # ur-dragon returned exactly TWO distinct readings: 45.304 for every
    # five-colour land and 44.027 for every restricted one. Grand Coliseum,
    # which always enters tapped, read identically to Forbidden Orchard,
    # which never does — the byte-identical tell this repo already uses to
    # catch a flag nothing acts on. The loop plays the FIRST land in hand
    # and credits its colours the same turn, so there is no tapped state to
    # act on. Modelling it would slow every deck's early turns and restate
    # every published figure on the fleet, so it is named here rather than
    # changed quietly.
    "LANDS ENTER UNTAPPED, ALWAYS, and the one in hand longest is the one "
    "played. A tapland costs nothing here and no land is ever chosen over "
    "another, so this model CANNOT rank two lands that make the same "
    "colours — `mana-analysis` and `mana-fit` are what answer a land "
    "question, and they are deterministic for exactly this reason.",
    "Mana rocks ('{T}: Add') contribute from the turn after they are cast.",
    "Commander cast on first affordable turn (highest spending priority).",
    "Bodies count = creatures cast + tokens parsed from 'create ... token' text.",
    "Target assembly counts cards DRAWN by a turn (cast cards still count).",
    "CARD DRAW IS ONE A TURN UNLESS `model_draw` IS SET. With the flag on, four "
    "channels are modelled — a card's own ETB draw, an instant or sorcery that "
    "draws, an upkeep trigger, and an ARRIVAL trigger that draws when other "
    "creatures or tokens enter (Welcoming Vampire, Caretaker's Talent), which "
    "rides the same single door the ETB damage payoffs use. Activated draw "
    "(needs a spending policy this model has no opinion on), X-based draw "
    "(board-dependent), and death- or attack-triggered draw (no deaths, and "
    "attacks only under `model_combat`) are NOT modelled and the cards are "
    "named in `meta.card_advantage.draw_not_modelled`. A conditional trigger — "
    "'if you control an artifact', 'you may draw' — is treated as unmodelled "
    "rather than assumed to be on: 39 of the 348 ETB-draw cards in the corpus "
    "carry one.",
    "AN ARRIVAL-DRAW ENGINE NEVER SEES ITS OWN ARRIVAL. Welcoming Vampire is a "
    "2/3 that draws 'whenever one or more OTHER creatures you control with "
    "power 2 or less enter'; registered before its own entry it passed its own "
    "gate and drew a card it does not draw, overstating edgar-vampires by 22% "
    "at turn eight. The deferral applies to every card and not only the ones "
    "worded 'other', which understates Tocasia's Welcome and two relatives by "
    "one draw on the turn they land.",
    "ARRIVAL DRAW IS SMALLER WITHOUT `model_combat`, and legitimately so. The "
    "channel rides the one door onto the battlefield, which `model_combat` also "
    "uses to spawn token copies and ETB-payoff tokens — those are real "
    "additional arrivals that do not exist in the resource-only model. Measured "
    "on edgar-vampires: 1.264 extra cards by turn ten with both flags against "
    "1.106 with draw alone. Before 2026-08-28 the gap was 1.264 against 0.323, "
    "because every call to that door sat inside `if model_combat:` and a "
    "draw-only deck silently lost three quarters of its arrival draws.",
    "HELD-UP INTERACTION is reported as two series and both are FLOORS. This "
    "model casts everything it can afford every turn, so `interaction_castable` "
    "is what a pilot developing at full speed has left over, not what a pilot "
    "choosing to hold up two mana would have. A low figure is therefore a real "
    "finding and a high one is unambiguous good news.",
    "Tutors are modeled as wildcards: a CAST tutor that fetches to hand or the "
    "top — including a TYPED one (Worldly Tutor, Sarkhan's Triumph), which the "
    "old literal 'a card' match missed entirely — fills ONE missing any_of "
    "group. Consumed once, mana paid, and a tutor that puts the card on top of "
    "the library costs a turn. An ACTIVATED, ETB or death-triggered tutor does "
    "not count, because this model only knows 'drawn and affordable'; nor does "
    "a land fetch, which is ramp and has no channel here yet. Reported as the "
    "*_assisted figures; the unassisted figures beside them exclude tutors "
    "entirely.",
    "ENTERS-THE-BATTLEFIELD PAYOFFS are modelled: damage equal to the entering "
    "creature's power (Terror of the Peaks), X damage where X counts your board "
    "(Scourge of Valkas, Dragon Tempest), a token (Lathliss) and a token COPY "
    "of what entered (Miirym). Every arrival fires them — cast, token or copy — "
    "and a copy is itself an arrival, so they compound, which is the deck "
    "working rather than a bug. A payoff worded 'another NONTOKEN' does not "
    "re-trigger on its own tokens, which is the brake the rules already had. A "
    "chain is capped at 12 deep. X counts the whole board rather than one "
    "subtype: exact where the creatures are Dragons, generous otherwise.",
    "The COMMANDER joins the battlefield when cast, attacks, and fires its own "
    "triggers. It used to be a mana sink and a flag — cast, then dropped — so a "
    "10/10 flier contributed no power and never swung. Commander tax, death and "
    "recasting are still not modelled: it is cast once and it stays.",
    # "CONSERVATIVE" IS TRUE OF SPEED AND FALSE OF COMPARABILITY, and the old
    # one-word claim hid the difference. Drawing one card a turn understates
    # every deck's speed, which is safe; but the size of the understatement
    # scales with how much draw a deck RUNS — heliod 16 cards, gishath 7 — so
    # the bias is not neutral ACROSS decks, and `benchmark` ranks decks.
    #
    # Modelling it was measured and refused: of 146 draw cards on this fleet,
    # **5 are unconditional**. The other 141 are triggers, activations and
    # costs — Rhystic Study needs opponents, Yawgmoth needs a sac outlet and a
    # creature. A model that saw the 5 would cover 3.4% of the axis while
    # letting this list claim draw was modelled, which is worse than the
    # refusal.
    "STATIC cost reduction IS modeled: a commander's eminence from the command "
    "zone (live from turn one, unremovable) and typed reducers once they are on "
    "the battlefield. It pays GENERIC only and is floored at the coloured pip "
    "count, because a discount can never pay a pip. A reduction that SCALES "
    "with a board state (Animar, Rakdos, Hamza) is refused rather than counted "
    "flat, and cost reduction on artifacts, noncreature spells or a colour is "
    "not modeled at all.",
    "Rituals are not modeled (conservative).",
    "Extra card draw is NOT modeled: one card per turn, always. That "
    "understates every deck, but by an amount proportional to how much draw it "
    "runs — so it is conservative WITHIN a deck and not neutral BETWEEN them. "
    "`meta.card_advantage` reports how much of each list this hides. Card "
    "advantage is measured nowhere in this suite.",
]

# Appended only for a deck that opts in. Stating "Treasures are modelled" on a
# deck where they are not is worse than saying nothing — and keeping them out
# keeps every non-opted deck's artifact byte-identical, which is the whole point
# of the flag.
TREASURE_ASSUMPTIONS = [
    "Treasures are a one-shot STOCKPILE, not a mana rock: spent only when lands "
    "and rocks fall short, and gone once broken. Reported separately from "
    "mean_available_mana_by_turn, which still means repeatable mana per turn.",
    "Only Treasure triggers a goldfish can see are modeled — upkeep and landfall "
    "are RECURRING, Saga chapters likewise; a `cast` or enters-the-battlefield "
    "trigger pays out ONCE, when its own source resolves. "
    "Combat- and opponent-gated sources produce NOTHING here, because this model "
    "has no combat and no opponents; they are named in "
    "meta.treasure_sources_not_modelled so a low hoard figure is legible.",
]

# Appended only for a deck that opts in, same contract as TREASURE_ASSUMPTIONS.
COMBAT_ASSUMPTIONS = [
    "COMBAT: one opponent at 40 life who does nothing — no blockers, no removal, "
    "no interaction. This is a goldfish in the literal sense, so `kill_turn` is "
    "the turn an UNOPPOSED board would finish ONE seat, not a win rate.",
    "Attacks with every creature that is not summoning-sick; haste is read from "
    "the ORACLE TEXT (this line said 'type line' and was simply wrong). There is nothing to block, so nothing is ever held back — in "
    "a real four-player game you would keep blockers, which makes this an "
    "optimistic clock and a pessimistic board.",
    "Attack triggers, combat-damage triggers and additional combat phases are "
    "modelled, which is what makes Treasure sources gated on combat produce here "
    "when they produce nothing without this flag. Effects the parser cannot read "
    "are named in meta.combat_effects_not_modelled.",
    "DAMAGE MULTIPLICATION is modelled and stacks MULTIPLICATIVELY as the rules "
    "do: doubling the power, swinging twice and doubling the damage dealt is "
    "eight times, not four. Three wordings are read — a replacement effect on "
    "damage you deal (Twinflame Tyrant), a granted double strike (Atarka) and a "
    "power doubling (Thrakkus). A grant worded 'each Dragon you control' is "
    "treated as applying to the WHOLE team: exact in a deck whose attackers are "
    "Dragons, generous in one where they are not. A creature's own double strike "
    "multiplies only itself. Board power is unaffected — a double-striker is not "
    "a bigger creature.",
    "Bodies count CREATURES only under this flag. Without it a Treasure token "
    "scores as a creature, which inflates `mean_bodies_by_turn` for any deck "
    "that makes non-creature tokens.",
    "LIFE LOSS ON ARRIVAL COUNTS AS DAMAGE. 'Each opponent loses 1 life' "
    "(Corpse Knight) and 'deals 1 damage to each opponent' (Impact Tremors) are "
    "the same event, the same cadence and the same number against one opponent "
    "at 40 life, and only the second was priced until 2026-08-28. They are NOT "
    "the same in the rules — lifelink, damage prevention and 'whenever an "
    "opponent loses life' all separate them — and this model has none of those. "
    "Nine corpus cards read through this channel, plus Mirkwood Bats through a "
    "token-creation channel of its own.",
    "DEATH-TRIGGERED DRAIN IS NOT MODELLED AND IS THE LARGEST KNOWN GAP HERE. "
    "Blood Artist, Cruel Celebrant, Zulaport Cutthroat, Bastion of Remembrance "
    "and Elas il-Kor all key on a creature DYING, and nothing dies in this "
    "model: no blockers, no removal, no sacrifice outlets. Their contribution "
    "is ABSENT, not zero, and a deck whose kill runs through them is understated "
    "here by however much that line is worth. `simulate` against a real pod is "
    "where that number lives.",
]

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
_SPELL_DRAW_RE = re.compile(
    r"(?:^|\.\s|^\s*)(?:you )?draw (a|one|two|three) cards?", re.I)
#: THE BODIES-INTO-CARDS FAMILY. 33 cards, and the qualifier between "you
#: control" and "enters" is load-bearing in BOTH directions: Welcoming Vampire
#: draws off a 1/1 token ("power 2 or less") and Garruk's Uprising must not
#: ("power 4 or greater"). Reading the trigger and ignoring its condition would
#: hand every token deck a draw engine it does not have.
_ARRIVAL_DRAW_RE = re.compile(
    r"whenever (?:this creature or )?(?:another |a |one or more )?"
    r"(?:other )?(?:nontoken )?([\w' ]{0,28}?)you control"
    r"([^.,]{0,44}?)enters?[^.]{0,60}?,\s*(?:you )?draw (a|one|two) cards?", re.I)
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


def cast_token_profile(card):
    """The commander's "cast an X spell -> make a token" trigger, or None.

    Returns `{subtype, bodies, power}`. `subtype` is the spell type that
    triggers it, matched against a cast card's own subtypes; an empty subtype
    means any spell and is left unmodelled rather than fired on everything.
    """
    text = card.get("oracle_text", "") or ""
    m = _CAST_TOKEN_RE.search(text)
    if not m:
        return None
    subtype = (m.group(1) or "").strip()
    if not subtype:
        return None
    pt = _PT_RE.search(m.group(3) or "")
    return {"subtype": subtype,
            "bodies": _DRAW_WORDS[m.group(2).lower()],
            "power": int(pt.group(1)) if pt else 1}


def sac_outlet_profile(card):
    """Is this a FREE repeatable sacrifice outlet, a costed one, or neither."""
    text = card.get("oracle_text", "") or ""
    if _FREE_SAC_OUTLET_RE.search(text):
        return "free"
    if _COSTED_SAC_OUTLET_RE.search(text):
        return "costed"
    return None


def death_profile(card):
    """What fires when ANOTHER creature you control dies.

    `unreadable` marks a card that clearly has a death trigger whose effect this
    parser cannot price — surfaced in the metrics rather than silently zero, the
    same contract the Treasure and combat models keep.
    """
    text = card.get("oracle_text", "") or ""
    out = {"death_drain": 0, "death_draw": 0, "death_treasure": 0,
           "unreadable": None}
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
    if not any((out["death_drain"], out["death_draw"], out["death_treasure"])):
        out["unreadable"] = card.get("name")
    return out


def is_death_engine(prof):
    """One predicate, one home — the same lesson `is_etb_engine` records."""
    return bool(prof["death_drain"] or prof["death_draw"] or prof["death_treasure"])


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
           "arrival_draw": 0, "arrival_draw_once": False,
           "arrival_power_min": None, "arrival_power_max": None,
           "unmodelled": None}
    if not _DRAW_RE.search(text):
        return out

    m = _ARRIVAL_DRAW_RE.search(text)
    if m and _DRAW_QUALIFIER_OK_RE.match(m.group(2) or ""):
        out["arrival_draw"] = _DRAW_WORDS[m.group(3).lower()]
        out["arrival_draw_once"] = bool(_DRAW_ONCE_RE.search(text))
        lo = _DRAW_POWER_MIN_RE.search(m.group(2) or "")
        hi = _DRAW_POWER_MAX_RE.search(m.group(2) or "")
        out["arrival_power_min"] = int(lo.group(1)) if lo else None
        out["arrival_power_max"] = int(hi.group(1)) if hi else None

    etb = _ETB_DRAW_RE.search(text)
    if etb and not _DRAW_CONDITIONAL_RE.search(etb.group(0)):
        out["etb_draw"] = _DRAW_WORDS[etb.group(1).lower()]
    rec = _RECURRING_DRAW_RE.search(text)
    if rec and not _DRAW_CONDITIONAL_RE.search(rec.group(0)):
        out["recurring_draw"] = _DRAW_WORDS[rec.group(1).lower()]
    if ("Instant" in type_line or "Sorcery" in type_line) and not out["etb_draw"]:
        sp = _SPELL_DRAW_RE.search(text)
        if sp and not _DRAW_ADDITIONAL_COST_RE.search(text):
            out["spell_draw"] = _DRAW_WORDS[sp.group(1).lower()]

    if not any((out["etb_draw"], out["spell_draw"], out["recurring_draw"],
                out["arrival_draw"])):
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

_HASTE_RE = re.compile(r"\bhaste\b", re.IGNORECASE)
# "create a 1/1 red Dragon creature token", "create two 2/2 ... tokens"
_TOKEN_PT_RE = re.compile(r"create (\w+) ([\dX]+)/([\dX]+)([^.]*?)tokens?", re.IGNORECASE)
_ATTACKS_RE = re.compile(
    r"whenever you attack\b|whenever (?:this creature|[A-Z][\w' ,-]{2,30}|one or more [\w ]+ you control) attacks",
    re.IGNORECASE)
_COMBAT_DMG_RE = re.compile(
    r"whenever (?:this creature|[A-Z][\w' ,-]{2,30}) deals combat damage to a player",
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
_CONSTELLATION_DRAIN_RE = re.compile(
    r"whenever this (?:creature|enchantment) or another enchantment you control enters,"
    r" each opponent loses (\w+) life", re.I)
_CONSTELLATION_GAIN_RE = re.compile(
    r"whenever this (?:creature|enchantment) or another enchantment you control enters,"
    r" you gain (\w+) life", re.I)

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
_SYMMETRIC_DRAIN_GAIN_RE = re.compile(
    r"([^.]*?each opponent loses (\w+) life and you gain (?:\w+) life[^.]*)\.", re.I)

#: Lifelink the card HAS, or grants to a creature it will keep — not a pump that
#: expires and not a token it makes. Corpus sweep 2026-09-04: a naive
#: `\blifelink\b` matches 737 cards; stripping these two forms and re-testing
#: keeps 607 and drops 130, every one of them a temporary grant ("gains lifelink
#: until end of turn") or a token-maker. The strip-then-test shape matters: a
#: scoped positive pattern dropped Behemoth Sledge ("has trample AND lifelink")
#: and Fear of Infinity ("Flying, lifelink"), both of which do have it.
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
           "lifelink": False, "scales_with": None, "unmodelled": None}
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
                out["lifelink"])):
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
        "token_power": 0,
        "token_bodies": 0,
        "attack_mana": 0,
        "attack_treasure": 0,
        "attack_draw": 0,
        "attack_damage": 0,
        "attack_token_power": 0,
        "attack_token_bodies": 0,
        "damage_scales_with_treasure": False,
        "extra_combat_free": False,
        "extra_combat_cost": None,
        # x2 per source, multiplied together across everything in play.
        "team_damage_multiplier": 1,
        "double_strike": False,
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

    etb = _ETB_TRIGGER_RE.search(text)
    if etb:
        win = text[etb.start():etb.start() + 220]
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
    for match in _TOKEN_PT_RE.finditer(text):
        tail = (match.group(4) or "").lower()
        if any(k in tail for k in _NONCREATURE_TOKENS):
            continue
        word = match.group(1).lower()
        count = int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 1)
        profile["token_bodies"] += count
        profile["token_power"] += count * _stat(match.group(2))

    combat_trigger = _ATTACKS_RE.search(text) or _COMBAT_DMG_RE.search(text)
    if combat_trigger:
        window = text[combat_trigger.start():combat_trigger.start() + 220]
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
        fixed = re.search(r"deals (\d+) damage", window, re.IGNORECASE)
        if fixed:
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
        if not any((profile["attack_mana"], profile["attack_treasure"],
                    profile["attack_draw"], profile["damage_scales_with_treasure"],
                    profile["attack_damage"], profile["attack_token_bodies"],
                    # A multiplier IS priced now, so a card carrying one must
                    # not be reported as unreadable — that list is a promise
                    # about what the figures leave out.
                    profile["team_damage_multiplier"] > 1,
                    profile["double_strike"],
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


def classify(card, pool=None):
    """Return a compact sim-card dict for one physical copy.

    `pool` is the deck's lands, and it exists for ONE card class: a fetchland's
    colours are a property of the deck, not of the card (`manabase.land_colors`).
    Without it every fetch is a colourless land that never produces anything.
    """
    type_line = card.get("type_line", "")
    text = card.get("oracle_text") or ""
    is_land = "Land" in type_line and "Creature" not in type_line.split("//")[0]
    is_tutor_card = bool(not is_land and is_tutor(card))
    mode_cost = _TUTOR_MODE_COST_RE.search(text) if is_tutor_card else None
    return {
        "name": card["name"],
        "is_land": is_land,
        "cmc": int(card.get("cmc") or 0),
        # A GOD IS NOT A CREATURE BELOW ITS DEVOTION THRESHOLD. None when the
        # card carries no such clause, which is 20 of the 23 enchantment
        # creatures in the list this was written for.
        "devotion_gate": devotion_gate(card),
        # CARRIED FOR THE COMMANDER'S ATTACK TUTOR, which filters on the printed
        # type. Nothing else in this model reads a type line at simulation time —
        # every other question is answered here, at classify time — so this key
        # exists for one caller and says so. Without it the filter matched the
        # empty string and the tutor silently never fired.
        "type_line": type_line,
        # What it actually costs to USE the tutor mode, which is what decides
        # when the wildcard comes online.
        "tutor_cmc": int(card.get("cmc") or 0) + (int(mode_cost.group(1)) if mode_cost else 0),
        # A SCALING DORK PRODUCES AT LEAST ONE. Without this it never reaches
        # the rock loop at all, which is how it came to read as zero.
        "produces": 0 if "Land" in type_line else (
            produced_mana(card.get("oracle_text"), type_line)
            or (1 if _SCALING_COLOR_MANA_RE.search(text) else 0)),
        "bodies": 0 if "Land" in type_line else body_count(card),
        # Creature-only body count and the combat profile ride along always;
        # they are READ only under `model_combat`, so a non-opted deck is
        # byte-identical and this stays a pure widening of the sim card.
        "creature_bodies": 0 if "Land" in type_line else creature_body_count(card),
        "combat": combat_profile(card),
        "drain": drain_profile(card),
        "draw": draw_profile(card),
        "death": death_profile(card),
        "token_doubler": token_doubler(card),
        "sac_outlet": sac_outlet_profile(card),
        "tutor": is_tutor_card,
        # A top-of-library tutor delivers on the next draw step, not this turn.
        "tutor_delay": 1 if is_tutor_card and _TUTOR_TO_TOP_RE.search(text) else 0,
        "tutor_needs_body": bool(is_tutor_card and _TUTOR_SAC_RE.search(text)),
        "treasure_n": 0 if is_land else treasure_profile(card)[0],
        "treasure_trigger": None if is_land else treasure_profile(card)[1],
        # Xorn makes no Treasure of its own; it adds one to every event.
        # WHAT IT PRODUCES and WHAT IT COSTS, in colours. Both ride along
        # always and are READ only under `model_colors`, so the colourless path
        # stays byte-identical — the `creature_bodies` rule. `land_colors` is
        # `manabase`'s and is deliberately restriction-aware: Haven of the
        # Spirit Dragon taps for {C} in a Vampire deck, and counting it as five
        # sources is how a mana base comes out looking fine and cannot cast its
        # spells.
        # `pool` resolves a fetchland against what it can actually go and get.
        # A non-land is unaffected: `fetch_profile` gates on the type line.
        "colors": frozenset(manabase.land_colors(card, pool=pool)),
        "pips": cast_pips(front_field(card, "mana_cost") or ""),
        "treasure_bonus": bool(TREASURE_BONUS_RE.search(text)),
        # Anointed Procession et al. make none either, and DOUBLE every event.
        # Rides on the card always and is read only under `model_treasures`, so
        # a non-opted deck stays byte-identical — the `creature_bodies` rule.
        "treasure_doubler": bool(TOKEN_DOUBLER_RE.search(text)),
        # Static cost reduction, and what this card IS so a reduction can be
        # tested against it. Both ride along always and are read only when a
        # reducer is actually in play, so a deck with none stays byte-identical.
        "reduces": cost_reduction(card, _corpus_creature_types()),
        # Priced at CAST TIME from the colours actually in play, so it enters
        # the rock loop (`produces > 0`) and its real output is computed there.
        "scales_with_colors": bool(_SCALING_COLOR_MANA_RE.search(text)),
        "subtypes": subtypes_of(type_line, text),
        "is_creature": "Creature" in type_line,
    }


def build_library(doc):
    """Expand the main deck (minus commanders) into per-copy sim cards."""
    library = []
    commanders = []
    # The fetch pool is every land in the list, commander included — a fetch
    # searches the LIBRARY, and what it may find does not depend on which zone
    # the search was started from.
    pool = [c for c in doc["cards"] if "Land" in str(c.get("type_line") or "")]
    for card in doc["cards"]:
        if card.get("is_commander"):
            commanders.append(card)
            continue
        library.extend([classify(card, pool=pool)] * card.get("quantity", 1))
    return library, commanders


def keepable(hand):
    lands = sum(1 for c in hand if c["is_land"])
    return GOLDFISH_MULLIGAN_MIN_LANDS <= lands <= GOLDFISH_MULLIGAN_MAX_LANDS


def _target_met(target, names_in_hand, commander_cast, tutors=0):
    """Is this target assembled, allowing `tutors` wildcards to fill holes?

    `tutors` is applied per target independently — each target is a separate
    counterfactual ("could this have been assembled by now"), exactly as the
    unassisted metric already treats them. It is NOT a shared pool drained
    across targets, which would make one target's rate depend on the order the
    others happen to be listed in.
    """
    if target.get("commander") and not commander_cast:
        return False
    # THE HOTTEST LINE IN THE SIMULATION. Profiled on edgar at 2,000 games,
    # this function and its two generator expressions were 0.797s of a 2.088s
    # loop — 38% — over 290,127 calls, because the `any_of` scan is rebuilt in
    # Python on every call for every unmet need on every turn of every game.
    #
    # The `any_of` list is CONSTANT for the whole run, so the set is built once
    # per need and cached on it. `names_in_hand` is already a set (`seen`), so
    # `isdisjoint` is a C-level intersection test. Same predicate, same result:
    # a need is unmet exactly when none of its names has been seen.
    #
    # Cached on the need dict rather than threaded through the signature
    # because the raw need dicts never reach the artifact — `target_stats`
    # takes only `target["label"]` — so there is nothing for a private key to
    # leak into.
    unmet = 0
    for need in target["need"]:
        names = need.get("_any_of_set")
        if names is None:
            names = need["_any_of_set"] = frozenset(need["any_of"])
        if names.isdisjoint(names_in_hand):
            unmet += 1
            # Counting past the tutor budget cannot change the answer.
            if unmet > tutors:
                return False
    return True


def simulate_once(rng, library, commander_cmc, targets, max_turn,
                  model_treasures=False, model_combat=False, model_draw=False,
                  model_sacrifice=False, model_drain=False,
                  model_colors=False, commander_pips=None,
                  command_zone_reduction=(), chosen_type=None,
                  commander_subtypes=frozenset(), commander_combat=None,
                  commander_cast_token=None, interaction_names=frozenset(),
                  attack_tutor=None):
    """One goldfish iteration. Returns a per-iteration result dict."""
    deck = library[:]
    rng.shuffle(deck)

    hand = deck[:7]
    deck = deck[7:]
    # Captured BEFORE the mulligan loop rebinds `hand`. The two populations
    # answer different questions and must not be conflated: the first seven is
    # what the keep rule is applied to, the kept hand is what you actually play.
    # Reporting the kept hand as "opening" made the distribution nearly
    # invariant to deck composition — every deck looks ~99% healthy at 2-5
    # lands, because that is the keep rule restating itself.
    first_seven_lands = sum(1 for c in hand if c["is_land"])

    attack_tutor_fired = 0
    # DEVOTION, and the gods waiting on it. `battlefield_pips` is one pip list
    # per nonland permanent in play; `pending_gods` holds the gods that have
    # RESOLVED but are not creatures yet. A god below its threshold is an
    # enchantment: no power, no attack, no block. It still contributes its own
    # pips to devotion, which is what lets a second god switch a first one on.
    battlefield_pips = []
    pending_gods = []
    mulligans = 0
    while not keepable(hand) and mulligans < GOLDFISH_MAX_MULLIGANS:
        mulligans += 1
        deck = library[:]
        rng.shuffle(deck)
        hand = deck[:7]
        deck = deck[7:]

    kept_hand_lands = sum(1 for c in hand if c["is_land"])
    # A STRICTER KEEP TEST, REPORTED AND NEVER ENFORCED. `keepable` asks only
    # "2-5 lands", which is the rule this model mulligans by and changing it
    # would restate every figure on every deck. This asks the question the
    # pilot's log actually raises about a two-land keep going fifth: COULD THIS
    # HAND HAVE DONE SOMETHING BY TURN THREE — is there a nonland card whose
    # cost the lands in this hand can reach by then?
    #
    # Lands in hand are capped at 3 because you get three land drops by turn 3,
    # and no draws are assumed: it is a property of the SEVEN CARDS KEPT, not a
    # forecast. That makes it a floor, like every other figure here.
    _reach = min(kept_hand_lands, 3)
    keep_can_act_by_t3 = any(
        (not c["is_land"]) and c["cmc"] <= _reach for c in hand)
    seen = {c["name"] for c in hand}

    lands_in_play = 0
    rock_production = 0
    # EMINENCE IS ON FROM TURN ONE. It works from the command zone, so it is
    # live before the commander is ever cast and cannot be removed — which is
    # exactly why leaving it unmodelled mispriced the whole deck rather than
    # just its late game. Reducers cast from hand are appended as they land.
    reductions = list(command_zone_reduction)
    team_damage_multiplier = 1
    etb_engines = []              # payoffs in play that fire on a creature entering
    # DRAW. `draw_engines` holds permanents that keep drawing — an upkeep
    # trigger or an arrival trigger; `drawn_extra_by_turn` is the series the
    # whole model was built for, and it is CUMULATIVE EXTRA cards: the one-a-turn
    # draw step is not in it, because every deck gets that and a series that
    # includes it hides the difference it exists to show.
    # INTERACTION HELD UP. Two series, because they answer different halves of
    # the same failure and the pilot's own diagnosis turned on which half it was:
    # "Deflecting Swat and Teferi's Protection sat in my hand uncast for the
    # entire game ... my mana was spent casting vampires, so there was never
    # open mana to hold up."
    #   _in_hand    — you drew it. A draw problem if this is low.
    #   _castable   — you drew it AND the turn ended with enough mana unspent to
    #                 cast it. A MANA problem if this is low while the first is
    #                 high, which is the structural claim the log makes.
    #
    # MEASURED AT THE END OF THE MAIN PHASE, which is the moment the decision is
    # actually made: you commit your mana to the board or you keep it up, and
    # you choose before combat. Extra combat phases are paid for BELOW this
    # point and attack triggers add mana below it too, so neither is counted —
    # on a deck with Aggravated Assault the float measured here is larger than
    # what survives the turn. edgar-vampires has no extra-combat effect, so the
    # figure is exact for it.
    #
    # AND IT IS A FLOOR AGAINST THE SPENDING POLICY: the model casts everything
    # it can afford, cheapest-first, every turn. A real pilot holding up two
    # mana would score higher. So a LOW figure says "this deck cannot afford to
    # hold up interaction while developing at full speed", which is the question
    # asked, and a high one is unambiguous good news.
    interaction_in_hand_by_turn = []
    interaction_castable_by_turn = []
    # SACRIFICE. `death_engines` are the payoffs in play; `free_sac_outlet` is
    # whether anything can convert a token for nothing. Both are needed — a
    # drain with no outlet never fires and an outlet with no payoff is a worse
    # board.
    # TOKEN DOUBLERS, MULTIPLICATIVE. Two doublers is x4 and not x3 — each
    # replaces the other's output, exactly as `treasure_multiplier` documents
    # for the Treasure side. Three doublers is x8, which is what this deck's
    # engine brief means by "the doublers turn one mint into four".
    token_multiplier = 1
    death_engines = []
    free_sac_outlet = False
    sacrifices = 0
    sacrifices_by_turn = []
    sac_cap_hits = 0
    draw_engines = []
    drawn_extra = 0
    drawn_extra_by_turn = []
    arrival_draw_used = set()     # ids of `once each turn` engines, per turn
    etb_damage = 0                # noncombat damage dealt this turn by those
    etb_chain_hits = 0            # times the chain guard stopped a cascade
    bodies_cum_bump = [0]         # tokens spawned by an ETB payoff, counted once
    commander_card = {"is_commander": True, "is_creature": True,
                      "subtypes": commander_subtypes, "cmc": commander_cmc,
                      "pips": commander_pips or ()}
    # One colour-set per untapped producer, parallel to the two counts above.
    # Only maintained under `model_colors`; empty otherwise, so `can_pay` is
    # never reached and the colourless arithmetic is untouched.
    sources = []
    treasures = 0                 # a STOCKPILE: each one is spendable once
    treasure_engines = []         # (per_event, trigger) for modelled sources in play
    treasure_bonus = 0            # Xorn-style +N per creation event
    # Procession-style xN. Multiplicative and applied AFTER the additive bonus,
    # which is the order a player would choose: replacement effects on one event
    # are ordered by the affected player, and (n + 1) x 2 beats n x 2 + 1. A
    # goldfish assumes the pilot takes the better line.
    treasure_multiplier = 1
    treasures_by_turn = []
    treasure_online_by_turn = []
    commander_turn = None
    land_hits = []
    # STALL: a turn on which NOTHING IN HAND COULD BE CAST AT ALL.
    #
    # THE OBVIOUS DEFINITION IS WRONG HERE AND WRONG BY A LOT. "A turn on which
    # nothing was cast" measures the MODEL, not the deck: this is a resource
    # simulation, so it casts rocks, tutors, extra-combat permanents and bodies
    # and never casts a wipe, a counterspell or a targeted removal spell. Scored
    # that way ur-dragon shows 6.4 dead turns in 10 while its hand grows to
    # eleven cards — which is a description of what the model declines to
    # represent, not of the deck stalling.
    #
    # So the question asked is CASTABILITY: with the mana this turn produced,
    # was there any nonland card in hand you could legally have cast? That needs
    # only mana value and available mana, so it is true of cards the model would
    # never pick up, and it is the honest reading of the PRD's "no legal play".
    #
    # A LAND DROP IS NOT A PLAY. A turn spent playing a land and casting nothing
    # is exactly the turn this measures.
    stall_by_turn = []          # True on a turn with nothing castable
    hand_size_by_turn = []      # a stall with cards left is a mana problem;
                                # a stall with an empty hand is a draw problem
    mana_by_turn = []
    bodies_cum = 0
    bodies_by_turn = []
    # Combat state. `battlefield` holds one entry per creature that can attack:
    # (power, turn_it_arrived, has_haste, own_damage_multiplier) — the fourth is
    # 2 for a double-striker and 1 otherwise, and is SEPARATE from the board-wide
    # `team_damage_multiplier` because the two have different scopes and stack.
    # Kept only under model_combat so the resource-only path allocates nothing.
    battlefield = []
    combat_engines = []           # per-attack triggers of creatures in play
    extra_combat_free = 0         # Scourge-style, one additional phase
    extra_combat_costs = []       # Aggravated Assault-style, buy each time
    opponent_life = GOLDFISH_OPPONENT_LIFE
    kill_turn = None
    # THE DRAIN PILLAR. `drain_permanents` is every profile on the battlefield
    # that gains life, drains, or pays off on gaining; `lifelink_power` is the
    # power of lifelink CREATURES, accumulated and never removed because nothing
    # dies here. Arrival counters are reset each turn.
    drain_permanents = []
    lifelink_power = 0
    drain_by_turn = []
    # Type lines of nonland permanents on the battlefield, so a card whose X is
    # "the number of Shrines you control" can be given the real count.
    battlefield_types = []
    damage_by_turn = []
    board_power_by_turn = []
    target_turns = [None] * len(targets)
    target_turns_unassisted = [None] * len(targets)
    tutor_ready_turns = []

    for turn in range(1, max_turn + 1):
        enchantments_entered = 0
        creatures_entered_this_turn = 0
        if deck:
            drawn = deck.pop(0)
            hand.append(drawn)
            seen.add(drawn["name"])

        land_index = next((i for i, c in enumerate(hand) if c["is_land"]), None)
        if land_index is not None:
            played = hand.pop(land_index)
            lands_in_play += 1
            if model_colors:
                sources.append(played["colors"])
            land_hits.append(True)
        else:
            land_hits.append(False)

        # Recurring Treasure engines already in play fire before you spend.
        # `landfall` only pays out on a turn a land actually entered, which is
        # what makes Tireless Provisioner worth less than an upkeep trigger.
        for per_event, trigger in treasure_engines:
            if trigger == "landfall" and not land_hits[-1]:
                continue
            if trigger in ("upkeep", "landfall"):
                treasures += (per_event + treasure_bonus) * treasure_multiplier

        etb_damage = 0
        bodies_cum_bump[0] = 0
        arrival_draw_used = set()

        def draw_n(n):
            """Take n off the top. The deck running out is a real outcome and
            is not an error: a goldfish that decks itself has answered the
            question about steam more loudly than any rate could."""
            nonlocal drawn_extra
            for _ in range(int(n)):
                if not deck:
                    return
                got = deck.pop(0)
                hand.append(got)
                seen.add(got["name"])
                drawn_extra += 1

        # Recurring draw engines already in play fire in the upkeep, BEFORE the
        # mana is spent, so a card drawn this way is castable this turn.
        if model_draw:
            for eng in draw_engines:
                if eng["recurring_draw"]:
                    draw_n(eng["recurring_draw"])

        pool = lands_in_play + rock_production
        # Reported WITHOUT the stockpile, so this series keeps meaning exactly
        # what it has always meant: repeatable mana per turn. Treasures are a
        # one-shot reserve and get their own series.
        mana_by_turn.append(pool)
        treasures_by_turn.append(treasures)
        treasure_online_by_turn.append(bool(treasure_engines))

        def creature_entered(power, arrived, haste=False, mult=1, depth=0,
                             is_token=False, is_legendary=False):
            """ONE DOOR ONTO THE BATTLEFIELD, so every payoff fires every time.

            Casting a creature, a token being made and a copy being made are the
            same event to Terror of the Peaks — the model used to have three
            separate `battlefield.append` sites and no payoff at any of them.

            IT RECURSES ON PURPOSE. Miirym's copy is a Dragon entering, which
            fires Scourge and Tempest again and raises X for the next one; that
            compounding IS the deck. `ETB_CHAIN_LIMIT` stops it and the depth is
            reported, because a loop that terminates silently cannot be told
            from one that never ran.
            """
            nonlocal etb_damage, etb_chain_hits
            battlefield.append((power, arrived, haste, mult, is_token))
            # BODIES INTO CARDS, on the same door the damage payoffs use.
            # The power condition is honoured in both directions: Welcoming
            # Vampire ("power 2 or less") draws off a 1/1 token, Garruk's
            # Uprising ("power 4 or greater") must not.
            if model_draw:
                for i, eng in enumerate(draw_engines):
                    n = eng["arrival_draw"]
                    if not n:
                        continue
                    lo, hi = eng["arrival_power_min"], eng["arrival_power_max"]
                    if lo is not None and power < lo:
                        continue
                    if hi is not None and power > hi:
                        continue
                    if eng["arrival_draw_once"]:
                        if i in arrival_draw_used:
                            continue
                        arrival_draw_used.add(i)
                    draw_n(n)
            if not model_combat or depth >= ETB_CHAIN_LIMIT:
                if depth >= ETB_CHAIN_LIMIT:
                    etb_chain_hits += 1
                return
            spawned = []
            for eng in etb_engines:
                # "another NONTOKEN Dragon you control enters" — a token copy
                # does not re-trigger the thing that made it. This is the brake
                # the rules already had, and without it the board compounds
                # without bound.
                if is_token and eng["etb_nontoken_only"]:
                    continue
                if eng["etb_damage_self_power"]:
                    etb_damage += power
                if eng["etb_damage_fixed"]:
                    etb_damage += eng["etb_damage_fixed"]
                if eng["etb_life_loss_fixed"]:
                    # Corpse Knight. Same event, same cadence and same number as
                    # Impact Tremors above; it read as zero until the payload
                    # regex learned the second way of wording it.
                    etb_damage += eng["etb_life_loss_fixed"]
                if is_token and eng["token_created_life_loss"]:
                    # Mirkwood Bats, and only on CREATURE tokens, because those
                    # are the only tokens this model makes. Bats also fires on a
                    # Blood token and on a sacrifice, and neither exists here.
                    etb_damage += eng["token_created_life_loss"]
                if eng["etb_damage_count"]:
                    # X is "the number of Dragons you control". The board is
                    # counted whole rather than by subtype — exact in a deck
                    # whose creatures are Dragons, generous otherwise, and
                    # stated in model_assumptions.
                    etb_damage += len(battlefield)
                if eng["etb_copy"]:
                    # A COPY IS LEGENDARY UNLESS THE CARD STRIPS IT. Miirym says
                    # "except the token isn't legendary" and is played for
                    # exactly that; Flameshadow Conjuring does not, so a copy of
                    # a legendary creature dies to the legend rule before it
                    # does anything — and 12 of this deck's creatures are
                    # legendary.
                    if is_legendary and eng["etb_copy_keeps_legendary"]:
                        continue
                    # And it usually charges. "You may pay {R}" is a cost, not a
                    # formality: firing it free reported 130.91 damage at turn
                    # ten against a 56.43 baseline.
                    if eng["etb_copy_cost"] and not spend(eng["etb_copy_cost"]):
                        continue
                    spawned.append((power, haste, mult))
                elif eng["etb_token_bodies"]:
                    each = eng["etb_token_power"] // max(eng["etb_token_bodies"], 1)
                    # DOUBLED HERE TOO. A payoff that makes a token on arrival
                    # is a token-creation event like any other, and a doubler
                    # that missed this site would double the printed token
                    # makers and not the engine's own.
                    for _ in range(eng["etb_token_bodies"] * token_multiplier):
                        spawned.append((each, False, 1))
            for pw, hs, mt in spawned:
                bodies_cum_bump[0] += 1
                creature_entered(pw, arrived, hs, mt, depth + 1, is_token=True)

        def spend(cost, pips=None):
            """Pay from lands and rocks first, then break Treasures.

            Under `model_colors` the colour requirement is checked too, against
            the sources actually in play, with Treasures as wildcards. Refusing
            here is the whole point: it is the turn the deck has the mana and
            not the colour, which every figure in this model used to ignore.
            """
            nonlocal pool, treasures
            if pool + treasures < cost:
                return False
            if model_colors and pips and not can_pay(pips, sources, treasures):
                return False
            if cost <= pool:
                pool -= cost
            else:
                treasures -= cost - pool
                pool = 0
            return True

        # A REDUCER ON THE BATTLEFIELD DOES CUT THE COMMANDER'S COST. Eminence
        # says "OTHER Dragon spells", so it never pays for itself — but
        # Dragonlord's Servant takes {1} off The Ur-Dragon like any other Dragon
        # spell, and a nine-drop commander is exactly where that matters.
        if commander_turn is None and spend(
                reduced_cost(commander_card, reductions, chosen_type),
                commander_pips):
            commander_turn = turn
            # The commander is a nonland permanent and its own pips count
            # toward devotion — Zur is {1}{W}{U}{B}, one each of three colours.
            battlefield_pips.append(commander_pips or [])
            battlefield_types.append(commander_card.get("type_line") or "")
            # THE COMMANDER USED TO BE CAST AND THEN DROPPED. It set this flag,
            # spent the mana, and never joined the battlefield — so a 10/10
            # flier contributed no power, never attacked, and fired none of its
            # own triggers. On The Ur-Dragon that is an entire stated win
            # condition (commander damage) measured as zero, and it is also why
            # Hellkite Courser, whose whole text is "put a commander onto the
            # battlefield", read as a vanilla body.
            #
            # Commander tax, death and recasting stay out of scope and stay
            # named: it is cast once, it stays, which is the same generous
            # direction the rest of this model takes.
            if model_combat and commander_combat and commander_combat["is_creature"]:
                creature_entered(commander_combat["power"], turn,
                                 commander_combat["haste"],
                                 2 if commander_combat["double_strike"] else 1,
                                 is_legendary=True)
                if commander_combat["team_damage_multiplier"] > 1:
                    team_damage_multiplier *= commander_combat["team_damage_multiplier"]
                if any((commander_combat["attack_mana"],
                        commander_combat["attack_damage"],
                        commander_combat["attack_treasure"],
                        commander_combat["attack_draw"],
                        commander_combat["attack_token_bodies"])):
                    combat_engines.append(commander_combat)
                if commander_combat["extra_combat_free"]:
                    extra_combat_free += 1
                if commander_combat["extra_combat_cost"] is not None:
                    extra_combat_costs.append(commander_combat["extra_combat_cost"])

        # THE COMMANDER'S OWN ATTACK TUTOR — an APPROXIMATION, declared per deck.
        #
        # Zur reads "whenever Zur attacks, search your library for an enchantment
        # card with mana value 3 or less, put it ONTO THE BATTLEFIELD". That is
        # the deck's entire engine and this model could not see it: every figure
        # for that deck counted only cards it DREW, so the axis the win condition
        # rides was measured as nothing. Leaving it unmodelled is not neutral —
        # it is a systematic understatement of exactly one deck's plan.
        #
        # WHAT IS MODELLED, AND HOW GENEROUSLY. From the turn AFTER the commander
        # lands (it must survive to attack), the best matching card is pulled
        # from the library and enough mana is added to the pool to pay for it, so
        # the normal casting loop resolves it through the ordinary path and every
        # arrival, ETB and body channel fires exactly as it would for a cast card.
        #
        # THIS IS OPTIMISTIC AND THE DIRECTION IS KNOWN. A goldfish has no
        # blockers, so the commander always attacks; at a real table it attacks
        # when the pilot judges it safe, and Forge's AI would not attack at all
        # (measured: Zur attacked in 17-47% of games where a human attacks every
        # turn, because the trigger fires on ATTACK and not on connect). So this
        # is a CEILING on the engine, not an estimate of it — and the ceiling is
        # the useful bound, because before this the floor was zero and nothing
        # else was available.
        #
        # "Best" is the highest mana value that fits the filter, which is the
        # crude part: a real pilot fetches for the board, not for the curve. It
        # is stated rather than hidden, and it is the same rule for every list
        # being compared.
        if attack_tutor and commander_turn is not None and turn > commander_turn:
            match = None
            for i, cand in enumerate(deck):
                if cand["cmc"] > attack_tutor["max_mv"]:
                    continue
                tl = cand["type_line"] or ""
                if attack_tutor["type"] not in tl:
                    continue
                if match is None or cand["cmc"] > deck[match]["cmc"]:
                    match = i
            if match is not None:
                fetched = deck.pop(match)
                hand.append(fetched)
                pool += reduced_cost(fetched, reductions, chosen_type)
                attack_tutor_fired += 1

        # A COST REDUCER IS NEITHER A ROCK, A TUTOR NOR A BODY — the third card
        # to fall through this hole, after Aggravated Assault and Primal Vigor.
        # Urza's Incubator and Herald's Horn are artifacts with `produces` 0 and
        # `bodies` 0, so every existing loop skips them and they would sit in
        # hand for ten turns while being the deck's stated curve fixer. Cast
        # BEFORE anything else affordable, because a reducer's whole value is
        # what it makes the rest of the turn cost.
        for card in sorted((c for c in hand if c["reduces"] and c["bodies"] == 0
                            and c["produces"] == 0),
                           key=lambda c: c["cmc"]):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                reductions.append(card["reduces"])
                hand.remove(card)

        # AN ETB PAYOFF THAT IS NOT A BODY falls through every other loop —
        # Dragon Tempest is an enchantment with `bodies` 0 and `produces` 0, so
        # it sat in hand for ten turns while being half of the deck's stated win
        # condition. Fourth card to find this hole, after Aggravated Assault,
        # Primal Vigor and the cost reducers. Cast early: its whole value is
        # what the creatures behind it are worth.
        if model_combat:
            for card in sorted((c for c in hand
                                if c["bodies"] == 0 and c["produces"] == 0
                                and not c["is_land"] and not c["tutor"]
                                and any((c["combat"]["etb_damage_self_power"],
                                         c["combat"]["etb_damage_count"],
                                         c["combat"]["etb_damage_fixed"],
                                         c["combat"]["etb_token_bodies"],
                                         c["combat"]["etb_copy"],
                                         # A DAMAGE DOUBLER THAT IS NOT A BODY
                                         # fell through here too — Gratuitous
                                         # Violence and Dictate of the Twin Gods
                                         # are enchantments, read correctly and
                                         # never cast.
                                         c["combat"]["team_damage_multiplier"] > 1))),
                               key=lambda c: c["cmc"]):
                if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                    if is_etb_engine(card["combat"]):
                        etb_engines.append(card["combat"])
                    if card["combat"]["team_damage_multiplier"] > 1:
                        team_damage_multiplier *= card["combat"]["team_damage_multiplier"]
                    hand.remove(card)

        # Cast rocks cheapest-first; they produce starting next turn.
        for card in sorted((c for c in hand if c["produces"] > 0), key=lambda c: c["cmc"]):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                if card["reduces"]:
                    reductions.append(card["reduces"])
                made, colors = card["produces"], card["colors"] or frozenset()
                if card["scales_with_colors"]:
                    # SNAPSHOT AT CAST, and deliberately the conservative end of
                    # the range: it counts the colours on the board the turn it
                    # resolves and never grows, so a Faeburrow Elder cast on two
                    # colours and living to see five is UNDERSTATED. Understating
                    # is recoverable; overstating is how a mana base comes out
                    # looking fine and cannot cast its spells.
                    colors = frozenset().union(*sources) if sources else frozenset()
                    made = max(1, min(len(colors), 5))
                rock_production += made
                if model_colors:
                    # A rock adds as many sources as it makes mana.
                    sources.extend([colors] * made)
                hand.remove(card)

        # Cast tutors before bodies: a tutor is a setup spell, and it competes
        # for the same mana. Previously tutors had bodies=0 and produces=0, so
        # they were never cast at all and their mana silently went to creatures.
        for card in sorted((c for c in hand if c["tutor"]), key=lambda c: c["tutor_cmc"]):
            if card["tutor_needs_body"] and bodies_cum < 1:
                continue
            if not spend(card["tutor_cmc"], card["pips"]):
                continue
            hand.remove(card)
            tutor_ready_turns.append(turn + card["tutor_delay"])

        # A permanent that grants an ADDITIONAL COMBAT PHASE is neither a rock,
        # a tutor nor a body, so the resource-only loop never cast it at all --
        # Aggravated Assault sat in hand for ten turns while being the deck's
        # only verified win line. Bought before creatures, because it is the
        # thing the creatures are for.
        if model_combat:
            for card in sorted((c for c in hand
                                if not c["is_land"] and c["bodies"] == 0
                                and (c["combat"]["extra_combat_cost"] is not None
                                     or c["combat"]["extra_combat_free"])),
                               key=lambda c: c["cmc"]):
                if spend(card["cmc"], card["pips"]):
                    hand.remove(card)
                    if card["combat"]["extra_combat_cost"] is not None:
                        extra_combat_costs.append(card["combat"]["extra_combat_cost"])
                    else:
                        extra_combat_free += 1

        # A TREASURE MULTIPLIER IS NEITHER A ROCK, A TUTOR NOR A BODY — the
        # same hole Aggravated Assault fell through two loops up. `bodies` is
        # the model's proxy for "is this worth casting", and it happens to be 1
        # for Anointed Procession, Parallel Lives and Doubling Season (their
        # text reads as token creation) and 0 for Primal Vigor, which is the
        # identical card. So Primal Vigor sat in hand for ten turns while
        # carrying the flag that says it changes what the deck produces, and a
        # candidate sweep read it as byte-identical to a card that does nothing.
        # A flag the model set is a claim the model must act on.
        if model_treasures:
            for card in sorted((c for c in hand if not c["is_land"]
                                and c["bodies"] == 0 and c["produces"] == 0
                                and not c["tutor"]
                                and (c["treasure_doubler"] or c["treasure_bonus"])),
                               key=lambda c: c["cmc"]):
                if not spend(card["cmc"], card["pips"]):
                    continue
                hand.remove(card)
                if card["treasure_doubler"]:
                    treasure_multiplier *= 2
                if card["treasure_bonus"]:
                    treasure_bonus += 1

        # A DRAW SPELL IS NEITHER A ROCK, A TUTOR, A BODY NOR AN ETB PAYOFF —
        # the fifth card to fall through every loop here, after Aggravated
        # Assault, Primal Vigor, the cost reducers and Dragon Tempest. Night's
        # Whisper is a sorcery with `bodies` 0 and `produces` 0, so it sat in
        # hand for ten turns while being the only unconditional card advantage
        # in edgar-vampires. Cast BEFORE bodies, cheapest first: a cantrip you
        # cast first can find the body, and one you cast last cannot.
        if model_draw:
            for card in sorted((c for c in hand if not c["is_land"]
                                and c["bodies"] == 0 and c["produces"] == 0
                                and not c["tutor"]
                                and any((c["draw"]["spell_draw"],
                                         c["draw"]["etb_draw"],
                                         c["draw"]["recurring_draw"],
                                         c["draw"]["arrival_draw"]))),
                               key=lambda c: reduced_cost(c, reductions, chosen_type)):
                if not spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                    continue
                hand.remove(card)
                draw_n(card["draw"]["spell_draw"] + card["draw"]["etb_draw"])
                if card["draw"]["recurring_draw"] or card["draw"]["arrival_draw"]:
                    draw_engines.append(card["draw"])

        # A PERMANENT THAT ONLY DRAINS WAS NEVER CAST AT ALL.
        #
        # Every casting loop above this one selects on a channel: cards that
        # draw, cards that ramp, cards that make Treasure, and below, cards with
        # bodies. A card with none of those — Sanctum of Stone Fangs, Northern
        # Air Temple: not creatures, no draw, no mana — matched no loop and sat
        # in hand for ten turns.
        #
        # That is why zur-enchantress's two Shrines measured as EXACTLY nothing
        # when they were added: not because two Shrines are weak, but because
        # the model never put them on the battlefield. A synthetic library of
        # twenty of them drained 0.
        #
        # Cheapest-first, like every other loop here, and guarded by the flag so
        # a deck that has not opted in is byte-identical.
        if model_drain:
            for card in sorted(
                    (c for c in hand
                     if c["bodies"] <= 0 and not c["is_land"]
                     and any(c["drain"][k] for k in (
                         "payoff_equal", "payoff_fixed", "gain_recurring",
                         "gain_per_enchantment", "gain_per_creature",
                         "drain_recurring", "drain_per_enchantment"))),
                    key=lambda c: reduced_cost(c, reductions, chosen_type)):
                if not spend(reduced_cost(card, reductions, chosen_type),
                             card["pips"]):
                    continue
                hand.remove(card)
                battlefield_pips.append(card["pips"])
                battlefield_types.append(card.get("type_line") or "")
                if "Enchantment" in (card.get("type_line") or ""):
                    enchantments_entered += 1
                drain_permanents.append(card["drain"])

        # Spend what's left on bodies, cheapest-first.
        for card in sorted((c for c in hand if c["bodies"] > 0),
                           key=lambda c: reduced_cost(c, reductions, chosen_type)):
            if spend(reduced_cost(card, reductions, chosen_type), card["pips"]):
                bodies_cum += card["creature_bodies"] if model_combat else card["bodies"]
                hand.remove(card)
                # A BODY THAT ALSO DRAWS is the shape the pilot's Edgar refactor
                # is built on: "a vampire that draws is better than a sorcery
                # that draws — same effect, plus a body, plus an eminence
                # trigger".
                #
                # THE ENGINE IS REGISTERED AFTER ITS OWN ARRIVAL, and the first
                # cut of this registered it before. Welcoming Vampire is a 2/3
                # that draws "whenever one or more OTHER creatures you control
                # with power 2 or less enter" — its own power is 2, so it passed
                # its own gate and drew a card it does not draw. Deferred, and
                # appended below once `creature_entered` has fired.
                #
                # It is deferred for EVERY card and not only the ones worded
                # "other", which understates Tocasia's Welcome and its two
                # relatives by one draw apiece on the turn they land. That is
                # the direction every other choice in this file takes.
                pending_draw_engine = None
                if model_draw:
                    draw_n(card["draw"]["etb_draw"])
                    if (card["draw"]["recurring_draw"]
                            or card["draw"]["arrival_draw"]):
                        pending_draw_engine = card["draw"]
                # Dragonlord's Servant and Dragonspeaker Shaman are bodies that
                # also reduce; from here on they pay for every Dragon behind them.
                if card["reduces"]:
                    reductions.append(card["reduces"])
                combat = card["combat"]
                # THE DOOR OPENS FOR EITHER MODEL, and it used to open only for
                # combat. `creature_entered` is where the arrival-draw channel
                # lives (Welcoming Vampire, Caretaker's Talent, Tocasia's
                # Welcome), and every call to it sat inside `if model_combat:` —
                # so a deck opting into `model_draw` ALONE lost three quarters
                # of its arrival draws and reported the smaller number without
                # saying anything. Measured on edgar-vampires: 1.264 extra cards
                # by turn ten with both flags, 0.323 with draw alone.
                #
                # A deck with `model_combat` on is byte-identical either way —
                # the disjunction is already true — and a deck with neither flag
                # never reaches here at all.
                arrivals_matter = model_combat or model_draw
                if not card["is_land"]:
                    battlefield_pips.append(card["pips"])
                    battlefield_types.append(card.get("type_line") or "")
                    if "Enchantment" in (card.get("type_line") or ""):
                        enchantments_entered += 1
                    if any(card["drain"][k] for k in
                           ("payoff_equal", "payoff_fixed", "gain_recurring",
                            "gain_per_enchantment", "gain_per_creature",
                            "drain_recurring", "drain_per_enchantment")):
                        drain_permanents.append(card["drain"])
                    if card["drain"]["lifelink"] and combat["is_creature"]:
                        lifelink_power += combat["power"]
                if model_combat:
                    # REGISTERED BEFORE IT ENTERS, and that is correct for the
                    # printed wording: Scourge of Valkas says "whenever THIS
                    # CREATURE or another Dragon you control enters", so it does
                    # see itself. Terror of the Peaks says "another", and its
                    # own entry deals nothing because the damage is the
                    # ENTERING creature's power and it is not another creature.
                    if is_etb_engine(combat):
                        etb_engines.append(combat)
                if arrivals_matter and combat["is_creature"]:
                    # A GOD RESOLVES AS AN ENCHANTMENT. It is held here and
                    # joins the battlefield on the turn its devotion is met,
                    # which may be this turn (its own pips count) or never.
                    if card["devotion_gate"]:
                        pending_gods.append((card, combat))
                    else:
                        creature_entered(
                            combat["power"], turn, combat["haste"],
                            2 if combat["double_strike"] else 1,
                            is_legendary="Legendary" in (card.get("type_line") or ""))
                        creatures_entered_this_turn += 1
                # EMINENCE MINTS ITS TOKEN ON THE CAST, from the command zone,
                # whether or not the commander has ever been cast. "Another"
                # is why the commander's own arrival does not trigger it — it
                # is not in this loop.
                if (arrivals_matter and commander_cast_token
                        and commander_cast_token["subtype"] in card["subtypes"]):
                    minted = commander_cast_token["bodies"] * token_multiplier
                    for _ in range(minted):
                        creature_entered(commander_cast_token["power"], turn,
                                         False, 1, is_token=True)
                    bodies_cum += minted
                # Creature tokens arrive with summoning sickness too, and they
                # arrive on the turn their maker resolved.
                if arrivals_matter and combat["token_bodies"]:
                    each = combat["token_power"] // max(combat["token_bodies"], 1)
                    for _ in range(combat["token_bodies"] * token_multiplier):
                        creature_entered(each, turn, False, 1, is_token=True)
                if model_combat:
                    if combat["team_damage_multiplier"] > 1:
                        team_damage_multiplier *= combat["team_damage_multiplier"]
                    if combat["extra_combat_free"]:
                        extra_combat_free += 1
                    if combat["extra_combat_cost"] is not None:
                        extra_combat_costs.append(combat["extra_combat_cost"])
                    if any((combat["attack_mana"], combat["attack_treasure"],
                            combat["attack_draw"], combat["damage_scales_with_treasure"],
                            combat["attack_damage"], combat["attack_token_bodies"])):
                        combat_engines.append(combat)
                if pending_draw_engine is not None:
                    draw_engines.append(pending_draw_engine)
                if card["token_doubler"]:
                    token_multiplier *= 2
                if model_sacrifice:
                    if is_death_engine(card["death"]):
                        death_engines.append(card["death"])
                    if card["sac_outlet"] == "free":
                        free_sac_outlet = True
                # Casting it turns its Treasure engine on for later turns, and
                # an ETB or cast trigger pays out immediately.
                if not model_treasures:
                    pass
                elif card["treasure_bonus"]:
                    treasure_bonus += 1
                if model_treasures and card["treasure_doubler"]:
                    # Two doublers is x4, not x3 — each replaces the other's
                    # output, which is why this compounds rather than sums.
                    treasure_multiplier *= 2
                if not model_treasures:
                    pass
                elif card["treasure_trigger"] in ("upkeep", "landfall"):
                    treasure_engines.append((card["treasure_n"], card["treasure_trigger"]))
                elif card["treasure_trigger"] in ("etb", "cast"):
                    treasures += ((card["treasure_n"] + treasure_bonus)
                                  * treasure_multiplier)
        bodies_cum += bodies_cum_bump[0]
        bodies_by_turn.append(bodies_cum)
        drawn_extra_by_turn.append(drawn_extra)
        sacrifices_by_turn.append(sacrifices)
        held = [c for c in hand if c["name"] in interaction_names]
        interaction_in_hand_by_turn.append(bool(held))
        interaction_castable_by_turn.append(
            any(c["cmc"] <= pool + treasures for c in held))

        # ── Combat step ────────────────────────────────────────────────────
        # Nothing blocks, so every creature that can attack does. Each combat
        # phase fires the attack triggers again, which is what makes an
        # additional combat phase worth more than its own power.
        if model_combat:
            # DOUBLE STRIKE IS PER CREATURE; the team multiplier is per board.
            # Kept apart because they have different scopes and stack: a
            # double-striker under Twinflame Tyrant deals its power four times.
            attackers = [p * mult for p, arrived, haste, mult, _tok in battlefield
                         if haste or arrived < turn]
            swing = sum(attackers)
            phases = 1 + extra_combat_free
            # Buy as many extra combats as the leftover mana allows, cheapest
            # first. `pool` is what survived the main phase.
            for cost in sorted(extra_combat_costs):
                while pool + treasures >= cost:
                    if not spend(cost):
                        break
                    phases += 1
                    if phases > 20:       # runaway guard; an infinite is a win
                        break
                if phases > 20:
                    break

            dealt = 0
            for _ in range(phases):
                if not attackers:
                    break
                bonus = treasures if any(
                    e["damage_scales_with_treasure"] for e in combat_engines) else 0
                dealt += swing + bonus
                for engine in combat_engines:
                    pool += engine["attack_mana"]
                    dealt += engine["attack_damage"]
                    if model_treasures:
                        treasures += engine["attack_treasure"]
                    for _ in range(engine["attack_draw"]):
                        if deck:
                            extra = deck.pop(0)
                            hand.append(extra)
                            seen.add(extra["name"])
                    # Tokens made mid-combat are summoning-sick, so they swell
                    # the board for NEXT turn rather than this swing.
                    if engine["attack_token_bodies"]:
                        each = (engine["attack_token_power"]
                                // max(engine["attack_token_bodies"], 1))
                        for _ in range(engine["attack_token_bodies"]):
                            creature_entered(each, turn, False, 1, is_token=True)
                        bodies_cum += engine["attack_token_bodies"]
            # THE REPLACEMENT EFFECT APPLIES LAST, to everything this deck
            # dealt — combat swings and the attack triggers alike, because
            # Twinflame Tyrant says "a source you control" and an attack
            # trigger is one.
            # ETB damage is NONCOMBAT and already happened this main phase, so
            # it is added before the multiplier rather than per combat phase —
            # Twinflame Tyrant says "a source you control", and a Terror trigger
            # is one, but it fires once per creature and not once per swing.
            # SACRIFICE, AFTER THE SWING IS SNAPSHOTTED. `attackers` was taken
            # above, so a token converted here has already attacked — this is a
            # conversion, not a trade against this turn's combat.
            if model_sacrifice and free_sac_outlet and death_engines:
                kept, n_sac = [], 0
                for entry in battlefield:
                    is_tok = entry[4]
                    if not is_tok or n_sac >= SAC_LIMIT_PER_TURN:
                        kept.append(entry)
                        continue
                    n_sac += 1
                    for eng in death_engines:
                        # Life loss is damage here for the same reason the
                        # arrival channel says so: one opponent, 40 life.
                        etb_damage += eng["death_drain"]
                        draw_n(eng["death_draw"])
                        if model_treasures:
                            treasures += eng["death_treasure"]
                if n_sac >= SAC_LIMIT_PER_TURN:
                    sac_cap_hits += 1
                battlefield[:] = kept
                sacrifices += n_sac
            dealt += etb_damage
            dealt *= team_damage_multiplier
            opponent_life -= dealt
            damage_by_turn.append(dealt)
            # BOARD POWER IS ACTUAL POWER. A double-striker is not a bigger
            # creature, so the multiplier belongs to the damage series and
            # never to this one.
            # A GOD SWITCHES ON when devotion reaches its threshold, and it is
            # checked AFTER the turn's permanents have resolved because they are
            # what moves devotion. It arrives with summoning sickness like any
            # other creature: `creature_entered` stamps this turn.
            if pending_gods:
                still = []
                for card, combat in pending_gods:
                    gate = card["devotion_gate"]
                    if devotion_of(battlefield_pips, gate["colors"]) >= gate["threshold"]:
                        creature_entered(
                            combat["power"], turn, combat["haste"],
                            2 if combat["double_strike"] else 1, is_legendary=True)
                    else:
                        still.append((card, combat))
                pending_gods[:] = still

            board_power_by_turn.append(sum(p for p, _, _, _, _ in battlefield))
            if kill_turn is None and opponent_life <= 0:
                kill_turn = turn

        # ── the drain pillar ────────────────────────────────────────────────
        #
        # Runs OUTSIDE the combat block, because a drain deck kills without ever
        # attacking and gating this on `model_combat` would reproduce the bug it
        # exists to fix.
        #
        # EVENTS AND TOTAL ARE TRACKED SEPARATELY, and they have to be: "target
        # opponent loses that much life" (Vito) scales with the AMOUNT gained,
        # while "each opponent loses 1 life" (Marauding Blight-Priest) fires once
        # per GAIN EVENT whatever the amount. Aggregating a turn into one event
        # would understate the second by however many times you gained.
        if model_drain:
            def _x_for(d):
                """X, as a real count of what the card names. 1 when the subject
                is one this model cannot count — never 0, because the card does
                do something."""
                if not d["scales_with"]:
                    return 1
                return max(1, sum(1 for tl in battlefield_types
                                  if d["scales_with"] in tl))

            gain_total = gain_events = 0
            for d in drain_permanents:
                if d["gain_recurring"]:
                    gain_total += d["gain_recurring"] * _x_for(d); gain_events += 1
                if d["gain_per_enchantment"] and enchantments_entered:
                    gain_total += d["gain_per_enchantment"] * enchantments_entered
                    gain_events += enchantments_entered
                if d["gain_per_creature"] and creatures_entered_this_turn:
                    gain_total += d["gain_per_creature"] * creatures_entered_this_turn
                    gain_events += creatures_entered_this_turn
            # Lifelink gains what those creatures DEALT, so it is capped by the
            # damage actually dealt this turn — one event, because the combat
            # model resolves a swing as a single number.
            if lifelink_power and damage_by_turn:
                linked = min(lifelink_power, damage_by_turn[-1])
                if linked > 0:
                    gain_total += linked; gain_events += 1

            drained = 0
            for d in drain_permanents:
                drained += d["drain_recurring"] * _x_for(d)
                drained += d["drain_per_enchantment"] * enchantments_entered
                if d["payoff_equal"]:
                    drained += gain_total
                drained += d["payoff_fixed"] * gain_events
            drain_by_turn.append(drained)
            if drained:
                opponent_life -= drained
                if kill_turn is None and opponent_life <= 0:
                    kill_turn = turn

        # Measured against the turn's FULL mana — lands, rocks and the
        # Treasure stockpile — because a Treasure you are holding is mana you
        # could have spent. `pool` has already been drawn down by the main
        # phase, so this asks what was reachable at the START of it.
        available = lands_in_play + rock_production + treasures
        stall_by_turn.append(not any(
            (not c["is_land"]) and c["cmc"] <= available for c in hand))
        hand_size_by_turn.append(len(hand))

        tutors = sum(1 for t in tutor_ready_turns if t <= turn)
        for i, target in enumerate(targets):
            commander_cast = commander_turn is not None
            if target_turns[i] is None and _target_met(target, seen, commander_cast, tutors):
                target_turns[i] = turn
            if target_turns_unassisted[i] is None and _target_met(target, seen, commander_cast):
                target_turns_unassisted[i] = turn

    return {
        "first_seven_lands": first_seven_lands,
        "kept_hand_lands": kept_hand_lands,
        "keep_can_act_by_t3": keep_can_act_by_t3,
        "mulligans": mulligans,
        "attack_tutor_fired": attack_tutor_fired,
        "land_hits": land_hits,
        "stall_by_turn": stall_by_turn,
        "hand_size_by_turn": hand_size_by_turn,
        "mana_by_turn": mana_by_turn,
        "commander_turn": commander_turn,
        "bodies_by_turn": bodies_by_turn,
        "drawn_extra_by_turn": drawn_extra_by_turn,
        "sacrifices_by_turn": sacrifices_by_turn,
        "sac_cap_hits": sac_cap_hits,
        "interaction_in_hand_by_turn": interaction_in_hand_by_turn,
        "interaction_castable_by_turn": interaction_castable_by_turn,
        "target_turns": target_turns,
        "target_turns_unassisted": target_turns_unassisted,
        "treasures_by_turn": treasures_by_turn,
        "treasure_online_by_turn": treasure_online_by_turn,
        "damage_by_turn": damage_by_turn,
        "board_power_by_turn": board_power_by_turn,
        "kill_turn": kill_turn,
        "drain_by_turn": drain_by_turn,
    }


def _round(x):
    return round(x, 3)


def aggregate(results, targets, max_turn, model_treasures=False,
              model_combat=False, model_draw=False, model_sacrifice=False,
              model_drain=False):
    n = len(results)
    turns = list(range(1, max_turn + 1))

    commander_turns = [r["commander_turn"] for r in results]
    cast_counts = {}
    for t in commander_turns:
        key = str(t) if t is not None else "not_by_max_turn"
        cast_counts[key] = cast_counts.get(key, 0) + 1
    cast_values = sorted(t for t in commander_turns if t is not None)

    target_stats = []
    for i, target in enumerate(targets):
        def _rates(key):
            got = sorted(r[key][i] for r in results if r[key][i] is not None)
            return {
                "assembled_rate": _round(len(got) / n),
                "mean_turn": _round(sum(got) / len(got)) if got else None,
                "by_turn_6_rate": _round(sum(1 for t in got if t <= 6) / n),
            }
        # The unassisted figures keep the historical key names, so every
        # existing consumer and every published figure still means what it
        # meant. Tutor-assisted estimates sit beside them under _assisted.
        assisted = _rates("target_turns")
        stats = {"label": target["label"], **_rates("target_turns_unassisted")}
        stats.update({
            "assembled_rate_assisted": assisted["assembled_rate"],
            "mean_turn_assisted": assisted["mean_turn"],
            "by_turn_6_rate_assisted": assisted["by_turn_6_rate"],
        })
        target_stats.append(stats)

    def _histogram(key):
        counts = {}
        for r in results:
            bucket = str(r[key])
            counts[bucket] = counts.get(bucket, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))

    return {
        "iterations": n,
        "opening_hand": {
            # Two distributions, deliberately both reported. `first_seven` is
            # the deck's real land distribution and moves when you change the
            # mana base. `kept_hand` is that distribution after the keep rule
            # has filtered it, so it sits near 100% inside the keep window for
            # every deck — informative about the mulligan rule, useless as a
            # fitness signal. The single `land_histogram` key this replaces
            # carried the second while being read as the first.
            "first_seven_land_histogram": _histogram("first_seven_lands"),
            "kept_hand_land_histogram": _histogram("kept_hand_lands"),
            "keep_first_seven_rate": _round(sum(1 for r in results if r["mulligans"] == 0) / n),
            # THE STRICTER THRESHOLD, beside the loose one it does not replace.
            "keep_can_act_by_t3_rate": _round(
                sum(1 for r in results if r["keep_can_act_by_t3"]) / n),
            "mean_mulligans": _round(sum(r["mulligans"] for r in results) / n),
        },
        # THE COMMANDER'S ATTACK TUTOR. It is a CEILING (see the model note
        # where it fires): a goldfish commander always attacks, a real one
        # attacks when it is safe to.
        #
        # ABSENT MEANS ABSENT. A deck that declares no tutor gets no block at
        # all rather than one reading `declared: false` — the same rule the
        # commander-damage block keeps, and it also keeps every other deck's
        # artifact free of a key about a thing it does not have.
        **({"attack_tutor": {
            # DERIVED from the rows rather than passed in: a result carries
            # `attack_tutor_fired`, and a deck that never declared one has the
            # key at 0 in every game. `declared` is therefore "it fired at least
            # once somewhere", which is the honest thing this function can see.
            "declared": any(r["attack_tutor_fired"] for r in results),
            "mean_fired": _round(sum(r["attack_tutor_fired"] for r in results) / n),
            "games_it_fired": sum(1 for r in results if r["attack_tutor_fired"]),
            "basis": ("fires once a turn from the turn after the commander lands, "
                      "pulling the highest-mana-value match from the library onto "
                      "the battlefield. A goldfish has no blockers so the "
                      "commander always attacks — this is the engine's CEILING, "
                      "not an estimate of it."),
        }} if any(r["attack_tutor_fired"] for r in results) else {}),
        "land_drop_hit_rate_by_turn": {
            str(t): _round(sum(1 for r in results if r["land_hits"][t - 1]) / n) for t in turns
        },
        "mean_available_mana_by_turn": {
            str(t): _round(sum(r["mana_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        "commander": {
            "cast_turn_histogram": cast_counts,
            "mean_cast_turn": _round(sum(cast_values) / len(cast_values)) if cast_values else None,
            "median_cast_turn": cast_values[len(cast_values) // 2] if cast_values else None,
            "cast_by_turn_6_rate": _round(sum(1 for t in cast_values if t <= 6) / n),
        },
        # THE SERIES THIS MODEL EXISTED WITHOUT FOR A YEAR. Cumulative cards
        # drawn BEYOND the one-a-turn draw step, so it is zero for a deck with
        # no readable draw and the difference between two lists is the whole
        # of it. Absent, not zero, when the deck has not opted in.
        # HOW MANY TOKENS THE POLICY ATE. Reported so the reader can see the
        # size of the assumption rather than only its consequence: a run showing
        # 30 sacrifices by turn ten is a very different claim from one showing 3.
        **({"mean_sacrifices_by_turn": {
            str(t): _round(sum(r["sacrifices_by_turn"][t - 1] for r in results) / n)
            for t in turns},
            "sac_cap_hit_rate": _round(
                sum(1 for r in results if r.get("sac_cap_hits")) / n)}
           if model_sacrifice else {}),
        **({"drain": {
            "mean_drain_by_turn": {
                str(i): _round(sum(r["drain_by_turn"][i - 1] for r in results) / n)
                for i in turns if all(len(r["drain_by_turn"]) >= i for r in results)},
            "mean_cumulative_drain_by_turn": {
                str(i): _round(sum(sum(r["drain_by_turn"][:i]) for r in results) / n)
                for i in turns if all(len(r["drain_by_turn"]) >= i for r in results)},
            "share_of_games_draining_by_turn_10": _round(
                sum(1 for r in results if sum(r["drain_by_turn"])) / n),
           }} if model_drain else {}),
        **({"mean_extra_cards_drawn_by_turn": {
            str(t): _round(sum(r["drawn_extra_by_turn"][t - 1] for r in results) / n)
            for t in turns}} if model_draw else {}),
        # HELD-UP INTERACTION, both halves. A low `castable` against a high
        # `in_hand` is a MANA problem and not a drawing problem, which is the
        # distinction the pilot's own diagnosis turned on.
        "interaction_in_hand_by_turn": {
            str(t): _round(sum(r["interaction_in_hand_by_turn"][t - 1] for r in results) / n)
            for t in turns},
        "interaction_castable_by_turn": {
            str(t): _round(sum(r["interaction_castable_by_turn"][t - 1] for r in results) / n)
            for t in turns},
        "mean_bodies_by_turn": {
            str(t): _round(sum(r["bodies_by_turn"][t - 1] for r in results) / n) for t in turns
        },
        # A Treasure is a one-shot reserve, so it is reported SEPARATELY from
        # `mean_available_mana_by_turn` rather than folded into it. Folding it in
        # would have changed what that series has always meant — repeatable mana
        # per turn — and it is quoted in published prose across the fleet.
        **({} if not model_treasures else {"treasure": {
            "mean_treasures_in_hoard_by_turn": {
                str(t): _round(sum(r["treasures_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "engine_online_rate_by_turn": {
                str(t): _round(sum(1 for r in results if r["treasure_online_by_turn"][t - 1]) / n)
                for t in turns
            },
        }}),
        # The clock. `kill_turn` is the turn an UNOPPOSED board would finish one
        # 40-life seat — deliberately not a win rate, because nothing here models
        # blockers, removal or three opponents. Reported beside board power so a
        # fast clock built out of 2/2s is legible as such.
        **({} if not model_combat else {"combat": (lambda kills: {
            "mean_board_power_by_turn": {
                str(t): _round(sum(r["board_power_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "mean_damage_by_turn": {
                str(t): _round(sum(r["damage_by_turn"][t - 1] for r in results) / n)
                for t in turns
            },
            "kill_turn_histogram": dict(sorted(
                ((str(k), sum(1 for r in results if r["kill_turn"] == k))
                 for k in {r["kill_turn"] for r in results} if k is not None),
                key=lambda kv: int(kv[0]))),
            "mean_kill_turn": _round(sum(kills) / len(kills)) if kills else None,
            "median_kill_turn": kills[len(kills) // 2] if kills else None,
            "kill_by_turn_rate": {
                str(t): _round(sum(1 for k in kills if k <= t) / n) for t in turns
            },
            "no_kill_by_max_turn_rate": _round((n - len(kills)) / n),
        })(sorted(r["kill_turn"] for r in results if r["kill_turn"] is not None))}),
        "targets": target_stats,
    }


class _Silent:
    """A progress sink for a run inside a sweep, which draws its own."""

    def advance(self, n=1):
        pass


def run(slug, iterations=None, seed=None, max_turn=None,
        model_treasures=None, model_combat=None, model_draw=None,
        model_sacrifice=None, with_results=False, branch=None,
        doc=None, quiet=False, targets_override=None, model_colors=None):
    """Run the goldfish simulation for a deck. Returns the metrics document.

    `model_treasures` and `model_combat` default to None, meaning READ THE
    DECK'S DECLARATION — the opt-in described below, and the behaviour every
    tracked `goldfish_metrics.json` was produced under. An explicit bool
    OVERRIDES it, which exists for one caller: the benchmark.

    A benchmark cannot read per-deck flags. Exactly one deck of twelve opts into
    combat today, so aggregating the fleet's own metrics would rank a deck that
    was measured with a kill clock against eleven that were not — the
    "uncontrolled output cannot be aggregated" failure the PRD names. The
    benchmark therefore runs its OWN uniform configuration and never writes to
    the deck's tracked file.
    """
    iterations = iterations or GOLDFISH_ITERATIONS
    seed = GOLDFISH_SEED if seed is None else seed
    max_turn = max_turn or GOLDFISH_MAX_TURN

    # `doc` lets a caller measure a list that is not on disk — one card
    # substituted, to find out what that card actually does. It changes nothing
    # about the model; it only skips the read.
    doc = doc if doc is not None else load_deck_cards(slug, branch)
    library, commanders = build_library(doc)
    if not commanders:
        raise SystemExit(f"No commander flagged in {slug}/cards.json")
    commander_cmc = int(commanders[0].get("cmc") or 0)

    # A branch inherits the deck's ENGINE DECLARATION unless it writes its own:
    # nobody authors a second targets file to try a candidate list, and measuring
    # a branch against no declaration would report a different deck rather than a
    # different list.
    targets_path = deck_file(slug, "goldfish_targets.json", branch)
    targets = []
    # OPT-IN, and for the same reason `OPTIONAL_DEPARTMENTS` existed: a model
    # that changes every deck's numbers at once cannot be landed on one deck
    # first. Measured before choosing this — turning it on fleet-wide moves
    # three decks' published figures, and gishath's `mean_cast_turn` alone
    # (7.969 -> 7.912) is quoted SIXTEEN times across seven tracked artifacts
    # including agent-authored prose and an `engine.json` carrying a critic
    # verdict. Silently invalidating that is the "confident and wrong" failure
    # this project exists to avoid, so a deck opts in when it is next
    # re-baselined deliberately.
    #
    # With the model off the treasure keys are ABSENT rather than zeroed, so the
    # six unaffected decks stay byte-identical and nothing needs regenerating.
    # Remove this flag once every deck has been re-baselined; a permanently
    # optional model is one nobody committed to.
    declared_treasures = False
    declared_combat = False
    declared_draw = False
    declared_sacrifice = False
    # Bound before the branch: a deck with no declaration still has colours,
    # and reading it only inside the `if` made every declaration-less deck
    # (which is the benchmark's whole fleet path) raise UnboundLocalError.
    targets_doc = {}
    if targets_path.exists():
        with open(targets_path) as f:
            targets_doc = json.load(f)
        # AN OVERRIDE MUST REACH THE SIMULATION, NOT JUST THE REPORT. The first
        # cut passed a modified declaration to the reporting layer while this
        # loop still read the file — so `target_turns` was indexed by the FILE's
        # targets and the override changed nothing. The tell was eight different
        # candidates returning the identical 0.501.
        targets = targets_override if targets_override is not None else targets_doc["targets"]
        declared_treasures = bool(targets_doc.get("model_treasures"))
        declared_combat = bool(targets_doc.get("model_combat"))
        declared_draw = bool(targets_doc.get("model_draw"))
        declared_sacrifice = bool(targets_doc.get("model_sacrifice"))
        declared_drain = bool(targets_doc.get("model_drain"))
        # A target member not in the deck can never be drawn — it silently
        # deflates the assembly rate (a target naming a card ur-dragon had moved
        # out once cost it a wrong "cost reducer drawn" figure). Warn loudly; the
        # fix is authored, so this stays a warning rather than a hard error.
        main_names = {c.get("name") for c in doc.get("cards", [])}
        for target in targets:
            for group in target.get("need", []):
                ghosts = [n for n in group.get("any_of", []) if n not in main_names]
                if ghosts:
                    if not quiet:
                        print(f"  WARNING target '{target.get('label', '?')}' names "
                          f"cards not in the maindeck (can never be drawn): "
                          f"{', '.join(ghosts)}")

    # Name every Treasure source the model CANNOT see. Silence here would make a
    # low hoard figure look like a modelling bug instead of a fact about the
    # deck — and the fact is usually load-bearing: ur-dragon's four Treasure
    # makers are all combat-triggered, so a goldfish that never attacks reports
    # zero and is right to.
    # An explicit argument OVERRIDES the declaration; None means read it. The
    # benchmark is the one caller that overrides, because uniform conditions are
    # what makes decks comparable at all.
    model_treasures = declared_treasures if model_treasures is None else bool(model_treasures)
    # THE COMMANDER'S ATTACK TUTOR, declared per deck because it is one
    # commander's text rather than a rule of the format. Shape:
    #     "model_commander_attack_tutor": {"type": "Enchantment", "max_mv": 3}
    # `type` is matched as a SUBSTRING of the type line, so "Enchantment" also
    # matches an enchantment CREATURE — which is the point for Zur, whose whole
    # plan is fetching bodies. Absent means absent: no other deck's figures move.
    attack_tutor = targets_doc.get("model_commander_attack_tutor") or None
    if attack_tutor:
        attack_tutor = {"type": str(attack_tutor.get("type") or "Enchantment"),
                        "max_mv": int(attack_tutor.get("max_mv", 3))}
    model_combat = declared_combat if model_combat is None else bool(model_combat)
    model_draw = declared_draw if model_draw is None else bool(model_draw)
    model_sacrifice = (declared_sacrifice if model_sacrifice is None
                       else bool(model_sacrifice))
    model_drain = declared_drain
    # LOUD, NOT SILENT. The drain half of this model is DAMAGE, and damage only
    # exists under `model_combat`. A deck that sets one flag and not the other
    # would otherwise get the draw and the Treasures and silently lose the
    # payoff it turned the flag on for — which is exactly the defect the arrival
    # channel shipped with and had to be found by measurement.
    if model_sacrifice and not model_combat and not quiet:
        print("  WARNING model_sacrifice is set without model_combat: death "
              "DRAIN is damage and there is no damage series without combat, "
              "so only the draw and Treasure halves will be read.")
    # COLOUR IS NOT OPTIONAL THE WAY TREASURES ARE. Every deck has colours and a
    # colourless mana model is simply wrong; the flag exists so the change can
    # be measured against the old behaviour and against `mana_analysis`'s
    # closed form, not so a deck can decline to have colours.
    declared_colors = bool((targets_doc or {}).get("model_colors", True))
    model_colors = declared_colors if model_colors is None else bool(model_colors)
    commander_pips = cast_pips(
        front_field(commanders[0], "mana_cost") or "") if commanders else []

    # THE COMMAND ZONE IS A SOURCE OF STATIC EFFECTS, and this is the first one
    # the model reads. Eminence is live from turn one whether or not the
    # commander is ever cast, and it cannot be answered — the single most
    # load-bearing fact about a deck built on it.
    # The commander's own combat profile — it is a creature like any other and
    # was the only one the loop never put on the battlefield.
    commander_combat = combat_profile(commanders[0]) if commanders else None
    creature_types = _corpus_creature_types()
    chosen_type = chosen_type_for(doc["cards"])
    command_zone_reduction = []
    commander_subtypes = frozenset()
    commander_cast_token = None
    for c in commanders:
        got = cost_reduction(c, creature_types)
        if got:
            command_zone_reduction.append(got)
        commander_subtypes |= subtypes_of(c.get("type_line") or "",
                                          c.get("oracle_text") or "")
        commander_cast_token = commander_cast_token or cast_token_profile(c)

    # A CARD IS BLIND ONLY IF EVERY CHANNEL IS BLIND. This list was built from
    # `treasure_profile` alone while the model has three ways to see a Treasure:
    # the trigger table, `treasure_bonus` (an adder — Xorn, Jolene),
    # `treasure_doubler` (Procession, Mondrak) and `combat.attack_treasure` (Goldspan, Old Gnawbone, Ragavan) once
    # `model_combat` is on. Reported from one channel it named nineteen sources
    # on ur-dragon's treasure branch as invisible when six of them were being
    # simulated — and the whole point of the list is that a low hoard figure
    # should be LEGIBLE, so over-reporting it is the same failure as omitting it.
    def _blind(c):
        # Computed from the card, not read off it: these are raw cards.json
        # entries, and the derived fields only exist on a built library entry.
        if treasure_profile(c)[1] != "unmodelled":
            return False
        text = c.get("oracle_text") or ""
        if TREASURE_BONUS_RE.search(text):
            return False                       # an adder: Xorn, Jolene
        if TOKEN_DOUBLER_RE.search(text):
            return False                       # a doubler: Procession, Mondrak
        if model_combat and combat_profile(c).get("attack_treasure"):
            return False                       # Goldspan, Old Gnawbone, Ragavan
        return True

    unmodelled = sorted({
        c["name"] for c in doc.get("cards", [])
        if not c.get("is_commander") and _blind(c)
    })
    # HOW MUCH OF THIS LIST THE DRAW ASSUMPTION HIDES. Not a list of names:
    # unlike a Treasure blind spot there is no figure here to make legible —
    # card advantage is measured nowhere — so what a reader needs is the SIZE
    # of the gap, which is what makes two decks' figures comparable or not.
    draw_cards = sum(
        c.get("quantity", 1) for c in doc.get("cards", [])
        if not c.get("is_commander") and _DRAW_RE.search(c.get("oracle_text") or ""))
    # WHAT COUNTS AS INTERACTION IS DECK_AUDIT'S QUESTION AND IT ALREADY OWNS
    # THE ANSWER. `SUITE_ROLES` is removal + sweepers + protection + stax — the
    # wider "interactive suite" rather than the removal count, because the two
    # cards the pilot's log names are Deflecting Swat and Teferi's Protection
    # and both are protection. Imported lazily: deck_audit reaches for goldfish
    # figures and a module-level import would close the loop.
    interaction_names = frozenset()
    draw_unmodelled = []
    try:
        from manamap.pilot.common import load_card_roles
        from manamap.pilot.deck_audit import SUITE_ROLES
        roles = load_card_roles()
        suite = set(SUITE_ROLES)
        interaction_names = frozenset(
            c["name"] for c in doc.get("cards", [])
            if not c.get("is_commander") and suite & set(roles.get(c["name"], [])))
    except Exception:
        # A missing card_roles.json is not a reason to lose the whole run; the
        # series simply reports against an empty set and says so below.
        interaction_names = frozenset()
    for c in doc.get("cards", []):
        if c.get("is_commander"):
            continue
        d = draw_profile(dict(c, oracle_text=c.get("oracle_text") or ""))
        if d["unmodelled"]:
            draw_unmodelled.append(d["unmodelled"])
    restricted = sorted({
        f"{c['name']} ({produced_mana(c.get('oracle_text'), c.get('type_line'))})"
        for c in doc.get("cards", [])
        if "Land" not in (c.get("type_line") or "")
        and produced_mana(c.get("oracle_text"), c.get("type_line"))
        and _RESTRICTED_MANA_RE.search(c.get("oracle_text") or "")
    })
    if not model_treasures:
        visible = sorted({
            c["name"] for c in doc.get("cards", [])
            if not c.get("is_commander")
            and treasure_profile(c)[1] in ("upkeep", "landfall", "cast", "etb")
        })
        if visible:
            print(f"  WARNING {slug} has {len(visible)} Treasure source(s) this model "
                  f"CAN simulate and `model_treasures` is not set in "
                  f"goldfish_targets.json, so they are ignored: {', '.join(visible)}")

    # Same contract as the Treasure warning above, one layer out: a combat
    # trigger whose EFFECT the parser cannot price scores zero, and a zero
    # nobody is told about reads as a fact about the deck.
    combat_unreadable = sorted({
        c["combat"]["unreadable"] for c in library if c["combat"]["unreadable"]
    }) if model_combat else []

    # THE DRAIN FIGURE IS A FLOOR AND HAS TO SAY SO. A death drain has no event
    # here (nothing dies), so Bastion of Remembrance and The Meathook Massacre
    # contribute zero — and a reader with no list of names cannot tell a deck
    # whose drain is small from one whose drain is unread. Same contract as
    # `draw_not_modelled` and `combat_effects_not_modelled`.
    drain_unmodelled = sorted({
        c["drain"]["unmodelled"] for c in library if c["drain"]["unmodelled"]
    }) if model_drain else []

    rng = random.Random(seed)
    # The loop is a list comprehension no longer, because 10,000 silent
    # simulations look identical to a hang. The comprehension is otherwise
    # unchanged — same rng, same order, same seed, so the RESULT is
    # bit-identical and `tests/test_pilot_goldfish.py`'s determinism assertions
    # hold. Progress is drawn on stderr and nothing here reads it back.
    results = []
    # A sweep runs this dozens of times; a progress bar per run is noise, and the
    # sweep draws its own.
    ctx = (contextlib.nullcontext(_Silent()) if quiet
           else console.task(f"Goldfishing {slug}", total=iterations, unit="sims"))
    with ctx as t:
        for _ in range(iterations):
            results.append(
                simulate_once(rng, library, commander_cmc, targets, max_turn,
                              commander_combat=commander_combat,
                              command_zone_reduction=command_zone_reduction,
                              chosen_type=chosen_type,
                              commander_subtypes=commander_subtypes,
                              commander_cast_token=commander_cast_token,
                              attack_tutor=attack_tutor,
                              model_treasures=model_treasures,
                              model_combat=model_combat,
                              model_draw=model_draw,
                              model_sacrifice=model_sacrifice,
                              model_drain=model_drain,
                              interaction_names=interaction_names,
                              model_colors=model_colors,
                              commander_pips=commander_pips))
            t.advance()

    return {
        "meta": {
            "deck": slug,
            "decklist_sha256": doc.get("decklist_sha256"),
            "seed": seed,
            "model_version": model_version(),
            "iterations": iterations,
            "max_turn": max_turn,
            "commander": commanders[0]["name"],
            "commander_cmc": commander_cmc,
            "model_assumptions": MODEL_ASSUMPTIONS + (
                TREASURE_ASSUMPTIONS if model_treasures else []) + (
                COMBAT_ASSUMPTIONS if model_combat else []),
            # RESTRICTED MANA IS COUNTED AS FREE, AND THE READER SHOULD KNOW.
            # `spend()` is a scalar, so it cannot represent "only to cast
            # Dragon spells". Delighted Halfling's legendary-only mana is very
            # nearly free in a Commander deck; Throne of Eldraine's four is not.
            # Same contract as the Treasure blind spots: the assumption is
            # NAMED rather than silently made or silently dropped.
            "card_advantage": {
                "cards_that_draw": draw_cards,
                "modelled": (draw_cards - len(draw_unmodelled)) if model_draw else 0,
                "why": ("ETB, spell, upkeep and arrival draw are modelled; "
                        "activated, X-based, death- and attack-triggered draw "
                        "are not, and the cards are named in "
                        "`draw_not_modelled`. Read the two numbers together: a "
                        "deck whose count is twelve and whose modelled figure "
                        "is one has almost no UNCONDITIONAL card advantage, "
                        "which is a finding about the deck."
                        if model_draw else
                        "one card per turn, always — see model_assumptions. "
                        "The understatement is proportional to this count, so "
                        "two decks with different counts are not directly "
                        "comparable on any speed figure."),
                **({"draw_not_modelled": sorted(draw_unmodelled)}
                   if model_draw and draw_unmodelled else {}),
            },
            # WHAT THE HELD-UP SERIES WAS MEASURED AGAINST. An empty set would
            # make both series read a flat zero, which is indistinguishable from
            # a deck that runs no interaction.
            "interaction_suite_counted": sorted(interaction_names),
            **({"restricted_mana_counted_as_free": restricted} if restricted else {}),
            **({"treasure_sources_not_modelled": unmodelled} if model_treasures else {}),
            **({"combat_effects_not_modelled": combat_unreadable}
               if model_combat and combat_unreadable else {}),
            **({"drain_not_modelled": drain_unmodelled}
               if model_drain and drain_unmodelled else {}),
        },
        "metrics": aggregate(results, targets, max_turn, model_treasures,
                             model_combat, model_draw, model_sacrifice,
                             model_drain),
        # OPT-IN, and default off so the returned document is byte-identical
        # to every tracked `goldfish_metrics.json`. Two tests compare `run`'s
        # output against the artifact directly, and they caught this the first
        # time it was unconditional — which is exactly what they are for.
        #
        # The benchmark needs a SPREAD and `aggregate` reports means, so it asks
        # for the rows rather than the shared artifact growing a stdev key to
        # serve one caller.
        **({"_results": results} if with_results else {}),
    }


def _coverage_preflight(slug, branch):
    """Say what this model cannot see BEFORE it spends ten thousand games.

    Every expensive fidelity surprise on this bench — eminence, the token
    doublers, the fetchlands — was found after the run. Imported lazily because
    `model_coverage` imports this module.
    """
    try:
        from manamap.pilot import model_coverage

        line = model_coverage.headline(model_coverage.analyze(slug, branch))
    except Exception:                              # noqa: BLE001 - never block
        return
    if line:
        print(f"  {line}")


def main(args):
    branch = getattr(args, "branch", None)
    _coverage_preflight(args.slug, branch)
    # BRANCHED WRITE, UN-BRANCHED READ — the mirror of the defect
    # `resolve_out_path` documents, and it silently filed the CHAMPION's
    # measurement under the branch's name for as long as branches have existed.
    # On ur-dragon's treasure branch that understated the turn-10 hoard 5.29 ->
    # 1.32, a factor of four, in a file whose own `meta.decklist_sha256` said
    # which list it had really measured. Nothing read that field.
    doc = run(args.slug, branch=branch)
    out = deck_dir(args.slug, branch) / "goldfish_metrics.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    commander = doc["meta"]["commander"]
    stats = doc["metrics"]["commander"]
    print(
        f"Wrote {out}\n  {commander}: mean cast turn {stats['mean_cast_turn']}, "
        f"cast by turn 6 in {stats['cast_by_turn_6_rate']:.0%} of games"
    )
    for target in doc["metrics"]["targets"]:
        print(f"  {target['label']}: by turn 6 in {target['by_turn_6_rate']:.0%} of games")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot goldfish <slug>`.")
