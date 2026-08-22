"""Pilot: is this deck any good, and what is stopping it?

Five commands already measure a deck and none of them joins the answers.
`deck-facts` reports composition, `mana-analysis` castability, `goldfish` speed,
`bracket-check` power. Ask "is my card
draw enough" and nothing answers; ask "what is missing that would make the engine
go" and nothing even represents the question.

This module is the join. Two blocks:

  **axes** — one measurement per row of `config.DECK_AXIS_TARGETS`, each against a
  target the strategy corpus can
  actually support, carried with the VERBATIM quote that supports it. The point is
  not the arithmetic (every figure here already existed somewhere); it is that an
  agent reading this can cite the number instead of inventing one. That is the
  failure `DECK_ROLE_BUDGET` was built to have — one flat uncited budget handed to
  every deck, its own comment calling it "PROVISIONAL", the pool brief printing
  its shortfalls as "Context, not evidence."

  **engine** — `goldfish_targets.json` is already a machine-readable declaration
  of what the deck is trying to assemble, and nothing has ever read it as one. Its
  `need: [{any_of: [...]}]` groups ARE the engine's components, and the size of a
  group is that component's redundancy. Price it through the hypergeometric, set
  it beside the rate the simulation measured, and the thinnest group is the single
  point of failure. "What would activate the engine" then has a deterministic
  answer: which cards in the pool would join that group.

**Computed on demand, never committed.** Same rule as `deck-facts` and
`artist_credits`, and for a sharper reason here: this file embeds goldfish and
bracket figures, so a committed copy would be a second source of truth that goes
stale the moment the decklist moves.

**Deterministic — it never reads the clock.** `deck_recon.json`'s `as_of` is
reported verbatim and the SKILL judges its age against RECON_MAX_AGE_DAYS. An
audit whose output changed with the date could not be diffed.

Composes primitives that already exist; no new analysis lives here:

    composition     pilot/deck_facts.py     analyze
    mana            mana_analysis.json      (artifact)
    speed           goldfish_metrics.json   (artifact)
    power           bracket_report.json     (artifact)
    odds            pilot/manabase.py       hypergeometric_at_least, cards_seen
    the pool        pilot/card_pool.py      load_pool
"""

import json
import re
from collections import Counter

from manamap.config import (
    CARD_ROLES_PATH,
    DECK_ARCHETYPE_BUDGETS,
    DECK_ARCHETYPE_BUDGET_CITATION,
    DECK_AXIS_TARGETS,
    ENGINE_MAX_CLOSERS,
    ENGINE_REDUNDANCY_CITATION,
    ENGINE_THIN_GROUP,
)
from manamap.pilot import deck_facts as deck_facts_mod
from manamap.pilot.common import (
    front_face,
    resolve_out_path,
    deck_dir,
    expand_copies,
    is_land,
    load_card_roles,
    load_combo_details,
    load_deck_cards,
    load_json,
)
from manamap.pilot.manabase import (
    DECK_SIZE_AFTER_COMMANDER, cards_seen, enters_tapped,
    enters_tapped_unconditionally, hypergeometric_at_least,
)

# Which classifier roles each axis counts. Deliberately NOT DECK_ROLE_GROUPS:
# that dict partitions the 99 for a slot filler (every card lands in exactly one
# group, unmapped roles fall to flex). An audit asks a different question — "how
# many cards can answer a creature" — and one card legitimately answers several
# axes. So these overlap on purpose, and a card is counted once per axis it
# qualifies for.
AXIS_ROLES = {
    "ramp": ("ramp:rock", "ramp:dork", "ramp:ritual", "ramp:land",
             "ramp:cost-reduction", "ramp:treasure"),
    "card-advantage": ("draw:engine", "draw:burst", "draw:impulse", "draw:wheel",
                       "draw:cantrip"),
    "interaction": ("removal:spot", "removal:damage", "removal:edict", "removal:fight",
                    "removal:debuff", "removal:bounce", "removal:tax", "counterspell",
                    "hate:graveyard"),
    "sweepers": ("removal:sweeper",),
    "protection": ("protection:self", "protection:granted", "protection:fog",
                   "protection:redirect"),
    "threat-density": ("wincon:alt", "wincon:drain", "wincon:combat"),
    "tutors": ("tutor:unrestricted", "tutor:narrow"),
}

# The wider "interactive suite" the corpus prices at 15-20 cards: removal plus
# everything else that answers something. Reported beside the removal count
# because Walser's two numbers are different numbers.
SUITE_ROLES = AXIS_ROLES["interaction"] + AXIS_ROLES["sweepers"] + AXIS_ROLES["protection"] + ("stax",)

# Breadth: the five permanent classes the corpus says to cover before doubling
# any. Heuristics over oracle text, and deliberately CONSERVATIVE — a counterspell
# is not counted as an answer to any class, because the same section warns that a
# counter answers a lock piece "only on the way in". Under-counting breadth
# reports a gap that may not exist; over-counting hides one that does.
# Values are TUPLES because the clause can run either way round: "put a -1/-1
# counter on target creature" is verb-first, "All creatures get -X/-X" is not,
# and a verb-first-only pattern read Toxic Deluge as answering nothing.
_CLASS_PATTERNS = {
    "creature": (
        re.compile(r"(destroy|exile|sacrifices?|fights?|deals? \d+ damage to|"
                   r"gets? -[\dX]+/-[\dX]+|-1/-1 counter)[^.]{0,90}creature",
                   re.I | re.S),
        re.compile(r"creatures?[^.]{0,50}(gets? -[\dX]+/-[\dX]+|"
                   r"gets? -[\dX]+/-[\dX]+ until)", re.I | re.S),
    ),
    "artifact": (re.compile(r"(destroy|exile)[^.]{0,90}artifact", re.I | re.S),),
    "enchantment": (re.compile(r"(destroy|exile)[^.]{0,90}enchantment", re.I | re.S),),
    # `land` must be the OBJECT of the verb. Two clause shapes put the word within
    # 90 characters of a destroy/exile while meaning the opposite, and a bare
    # proximity match scored all eleven of them as land removal:
    #   a search redirects to a different object — Winds of Abandon exiles CREATURES
    #     and its victims "search their library for a basic land", i.e. it ramps the
    #     opponents it is credited with answering (also Sword of Hearth and Home,
    #     Surveyor's Scope);
    #   an exclusion SPARES lands — "Destroy all other permanents except for lands"
    #     (Elspeth Tirel, Scourglass, Elesh Norn, Eye of Singularity), and
    #     "other than a basic land card" is graveyard hate (Haunting Echoes, Kotose).
    # Tempering the gap on both shapes drops 268 corpus matches to 257 and loses no
    # real land destruction: Ghost Quarter, Sowing Salt and Mwonvuli Acid-Moss all
    # keep, because their search clause follows the kill rather than redirecting it.
    "land": (re.compile(
        r"(destroy|exile)"
        r"(?:(?!search(es)?\b[^.]{0,40}?\bfor\b|\bother than\b|\bexcept\b)[^.]){0,90}"
        r"\bland", re.I | re.S),),
    # Someone ELSE'S graveyard. A bare `exile ... graveyard` matches Necropotence
    # exiling its own discards, delve, escape and flashback — self-exile is the
    # commonest use of the word in Magic and none of it is graveyard hate.
    "graveyard": (re.compile(
        r"exile[^.]{0,90}(all graveyards|graveyards|each (opponent|player)'s "
        r"graveyard|target (opponent|player)'s graveyard|target card from a "
        r"graveyard|from an opponent's graveyard)", re.I | re.S),),
}
PERMANENT_CLASSES = tuple(_CLASS_PATTERNS)

# Catch-alls, which the same section says "earn their premium" — and which the
# per-class patterns cannot see, because they never name a type. Cyclonic Rift,
# Teferi and Archmage's Charm all answer three classes each and every one of
# them read as answering nothing.
_CATCHALL_VERB = r"(destroy|exile|return|puts?|gain control of)"
_CATCHALL_NONLAND = re.compile(
    _CATCHALL_VERB + r"[^.]{0,70}nonland permanent", re.I | re.S)
_CATCHALL_ANY = re.compile(
    _CATCHALL_VERB + r"[^.]{0,70}(target|each|all|any|another) permanent",
    re.I | re.S)
_NONLAND_CLASSES = ("creature", "artifact", "enchantment")

# A catch-all gated on a mana-value FLOOR cannot reach a land, because a land has no
# mana cost and so is always mana value 0. Despark ("exile target permanent with mana
# value 4 or greater") reads as answering every class including land, and it answers
# four of five. Four cards in the corpus are gated this way; all four floor at 3+.
_CATCHALL_MV_FLOOR = re.compile(r"mana value ([1-9]\d*) or greater", re.I)

# ...and a catch-all that names lands in an EXCLUSION spares them. "Destroy all
# permanents except for artifacts and lands" matches "(destroy) all permanent" and was
# credited with the land class by the clause that protects it. Four cards: Scourglass,
# Eye of Singularity, World-Bottling Kit, Cyclone Summoner.
_CATCHALL_LAND_SPARED = re.compile(r"(except|other than)[^.]{0,60}\bland", re.I)

# Which archetype an authored frame is talking about. The frame's `archetype` is
# free prose, so this is a keyword read, and it says which word it matched rather
# than asserting it got the deck right.
_ARCHETYPE_HINTS = (
    ("voltron", ("voltron",)),
    ("combo", ("combo", "cedh")),
    ("control", ("control", "attrition", "stax", "prison")),
    ("aggro", ("aggro", "aggressive", "beatdown", "voltron-lite")),
)


def _count_copies(cards, roles, wanted):
    """Copies carrying ANY of `wanted`, and the distinct names behind them.

    Copies, not entries: the shuffler sees eleven Swamps, and comparing entry
    counts against copy-denominated targets is the bug that once briefed a
    scout that a 36-land deck was twenty-one lands short.
    """
    want = set(wanted)
    names, copies = set(), 0
    for card in expand_copies(cards):
        if want & set(roles.get(card["name"], [])):
            copies += 1
            names.add(card["name"])
    return copies, sorted(names)


def _verdict(value, low, high):
    if low is not None and value < low:
        return "under"
    if high is not None and value > high:
        return "over"
    if low is None and high is None:
        return "informational"
    return "at"


def _axis(name, value, unit, cards, how, low=None, high=None, extra=None,
          target_override=None):
    """One axis row: what was measured, what the corpus says, and the gap."""
    spec = DECK_AXIS_TARGETS[name]
    low = spec.get("low") if low is None else low
    high = spec.get("high") if high is None else high
    if target_override:
        low, high = target_override
    target = {"low": low, "high": high,
              "source": spec["source"], "quote": spec["quote"]}
    if spec.get("formula"):
        target["formula"] = spec["formula"]
    verdict = _verdict(value, low, high)
    row = {
        "axis": name,
        "measured": {"value": value, "unit": unit, "cards": cards, "how": how},
        "target": target,
        "verdict": verdict,
        "gap": (round(low - value, 4) if verdict == "under"
                else round(high - value, 4) if verdict == "over" else 0),
    }
    if extra:
        row["measured"].update(extra)
    return row


# ── Archetype ────────────────────────────────────────────────────────────

def detect_archetype(slug, override=None):
    """Which archetype's budget applies, and how we decided.

    Never guessed from the cards: either the caller says so, or the strategic
    frame does, or the base targets apply and the audit says they did. A budget
    silently attributed to the wrong archetype is worse than no budget.
    """
    if override:
        return {"archetype": override, "source": "--archetype"}
    frame = load_json(deck_dir(slug) / "strategic_frame.json", default=None)
    if frame:
        text = str(frame.get("archetype", "")).lower()
        for name, hints in _ARCHETYPE_HINTS:
            hit = next((h for h in hints if h in text), None)
            if hit:
                return {"archetype": name, "source": "strategic_frame.json",
                        "matched": hit}
    return {"archetype": None, "source": "none — base targets apply"}


def archetype_overrides(archetype):
    """{axis: (low, high)} for this archetype, plus the citation that backs it."""
    if not archetype or archetype not in DECK_ARCHETYPE_BUDGETS:
        return {}, None
    return dict(DECK_ARCHETYPE_BUDGETS[archetype]), dict(DECK_ARCHETYPE_BUDGET_CITATION)


# ── Axes ─────────────────────────────────────────────────────────────────

def _interaction_breadth(cards, roles):
    """Which of the five permanent classes this deck can answer at all."""
    interactive = set(SUITE_ROLES)
    answered = {cls: [] for cls in PERMANENT_CLASSES}
    for card in cards:
        card_roles = set(roles.get(card["name"], []))
        if not (interactive & card_roles):
            continue
        text = str(card.get("oracle_text", "") or "")
        hits = {cls for cls, patterns in _CLASS_PATTERNS.items()
                if any(p.search(text) for p in patterns)}
        if "hate:graveyard" in card_roles:
            hits.add("graveyard")
        if _CATCHALL_NONLAND.search(text):
            hits.update(_NONLAND_CLASSES)
        if _CATCHALL_ANY.search(text):
            hits.update(_NONLAND_CLASSES)
            if not (_CATCHALL_MV_FLOOR.search(text)
                    or _CATCHALL_LAND_SPARED.search(text)):
                hits.add("land")
        for cls in hits:
            if card["name"] not in answered[cls]:
                answered[cls].append(card["name"])
    return {cls: sorted(names) for cls, names in answered.items()}


def build_axes(slug, cards, roles, facts, mana, goldfish, bracket, overrides):
    """One row per `DECK_AXIS_TARGETS` entry, plus the conditional ones
    (`colour-sources`, `consistency`, `power`) when their artifact exists. Every
    measurement traces to an artifact or to `roles`.

    The count is deliberately NOT stated here. Both this docstring and the module
    header said "thirteen" while `DECK_AXIS_TARGETS` held sixteen — the same
    transcribed-registry-count failure `test_docs_section_count` exists to catch,
    living inside the module that owns the registry."""
    axes = []
    commanders = [c for c in cards if c.get("is_commander")]
    identity = sorted({col for c in commanders for col in (c.get("color_identity") or [])})
    commander_cmc = int(commanders[0].get("cmc") or 0) if commanders else 0
    land_copies = facts["counts"]["land_copies"]

    # 1. Land count, against the format's conventional band.
    low, high = overrides.get("mana-base", (None, None))
    axes.append(_axis(
        "mana-base", land_copies, "copies", [],
        "land copies from deck-facts (copies, not entries)",
        low=low, high=high, extra={"land_entries": facts["counts"]["lands"]}))

    # 1b. Mana SOURCES — the instruction the section actually opens with.
    persistent = ("ramp:rock", "ramp:dork", "ramp:land")
    ramp_sources, ramp_names = _count_copies(cards, roles, persistent)
    burgess = 31 + len(identity) + commander_cmc
    axes.append(_axis(
        "mana-sources", land_copies + ramp_sources, "copies", ramp_names,
        f"land copies plus persistent producers ({', '.join(persistent)}); "
        f"target 31 + {len(identity)} colour(s) + commander mana value "
        f"{commander_cmc} = {burgess}",
        low=burgess, high=None,
        extra={"burgess_target": burgess, "colour_identity": identity,
               "commander_mana_value": commander_cmc,
               "persistent_producers": ramp_sources,
               "note": "rituals and Treasure makers are one-shot and are not "
                       "counted as sources"}))

    # 2. Taplands — the ones that ALWAYS enter tapped.
    #
    # `mana_analysis.lands.enters_tapped` is a substring count and a superset:
    # it sweeps in shocklands (a choice at two life) and every "enters tapped
    # unless…" land, including the "unless you have two or more opponents"
    # cycle, which in Commander is always true. Reading that as tempo cost
    # overstates it by a factor of five on a three-colour deck.
    lands = [c for c in expand_copies(cards) if is_land(c)]
    always = [c["name"] for c in lands if enters_tapped_unconditionally(c)]
    conditional = [c["name"] for c in lands
                   if enters_tapped(c) and not enters_tapped_unconditionally(c)]
    axes.append(_axis(
        "taplands", len(always), "copies", sorted(set(always)),
        "land copies that enter tapped with no condition and no pay-life escape",
        extra={"conditional_or_shock": len(conditional),
               "conditional_cards": sorted(set(conditional)),
               "note": "mana_analysis.lands.enters_tapped is a substring superset "
                       "and counts both groups together"}))

    # 3. Colour sources — per colour, against the deck's own computed target.
    if mana:
        targets = mana.get("source_targets") or {}
        totals = (mana.get("sources") or {}).get("total") or {}
        short = {c: targets[c] - totals.get(c, 0)
                 for c in targets if totals.get(c, 0) < targets[c]}
        # The value is the WORST colour's surplus, so one starved colour cannot
        # hide behind four comfortable ones — which is how a five-colour pile
        # reads as fine while its splash is uncastable.
        surplus = min((totals.get(c, 0) - targets[c] for c in targets), default=0)
        axes.append(_axis(
            "colour-sources", surplus, "sources above target (worst colour)", [],
            "mana_analysis.sources.total against source_targets (Karsten, 90%)",
            low=0, high=None,
            extra={"per_colour": {c: {"have": totals.get(c, 0), "target": targets[c]}
                                  for c in sorted(targets)},
                   "short": short}))

    # 4-8. Role-counted axes.
    for name in ("ramp", "card-advantage", "interaction", "sweepers",
                 "protection", "threat-density"):
        copies, names = _count_copies(cards, roles, AXIS_ROLES[name])
        low, high = overrides.get(name, (None, None))
        probed = _probe_uncounted(name, cards, roles, names)
        axes.append(_axis(
            name, copies, "copies", names,
            f"copies carrying any of {', '.join(AXIS_ROLES[name])}",
            low=low, high=high,
            extra={"possible_uncounted": probed} if probed else None))
        if name == "interaction":
            suite, _ = _count_copies(cards, roles, SUITE_ROLES)
            axes[-1]["measured"]["interactive_suite_copies"] = suite

    # 8b. Creature count — the aggro budget's actual unit.
    creature_copies = sum(1 for c in expand_copies(cards)
                          if "Creature" in front_face(c.get("type_line", "")))
    low, high = overrides.get("creatures", (None, None))
    axes.append(_axis("creatures", creature_copies, "copies", [],
                      "front-face creature copies", low=low, high=high))

    # 9. Breadth.
    breadth = _interaction_breadth(cards, roles)
    covered = [cls for cls, names in breadth.items() if names]
    axes.append(_axis(
        "interaction-breadth", len(covered), "classes answered", sorted(covered),
        "oracle-text heuristic over interactive cards; counterspells deliberately "
        "excluded — a counter answers a permanent only on the way in",
        extra={"by_class": {cls: len(names) for cls, names in breadth.items()},
               "uncovered": sorted(cls for cls in PERMANENT_CLASSES if not breadth[cls])}))

    # 10. Consistency — the turn-3 land drop, the one bound the corpus states.
    if goldfish:
        hit3 = (goldfish.get("metrics", {}).get("land_drop_hit_rate_by_turn") or {}).get("3")
        if hit3 is not None:
            keep = (goldfish.get("metrics", {}).get("opening_hand") or {}).get(
                "keep_first_seven_rate")
            axes.append(_axis(
                "consistency", hit3, "rate", [],
                "goldfish land_drop_hit_rate_by_turn['3']",
                extra={"keep_first_seven_rate": keep}))

    # 11. Curve — reported against the measured Commander shape, not scored.
    curve = facts["curve"]
    nonland = sum(curve.values())
    mode = max(curve, key=lambda k: (curve[k], -int(k))) if curve else None
    axes.append(_axis(
        "curve", int(mode) if mode is not None else 0, "modal mana value", [],
        "deck-facts curve (nonland entries)",
        extra={"two_drops": curve.get("2", 0), "three_drops": curve.get("3", 0),
               "mana_value_8_plus": sum(v for k, v in curve.items() if int(k) >= 8),
               "nonland_entries": nonland}))

    # 12. Power — a floor, reported, never scored.
    if bracket:
        axes.append(_axis(
            "power", bracket.get("floor") or 0, "bracket floor", [],
            "bracket_report.json",
            extra={"floor_name": bracket.get("floor_name"),
                   "game_changers": bracket.get("game_changers", []),
                   "drivers": [d.get("signal") for d in bracket.get("drivers", [])]}))

    # 13. Tutors — bounded only inside the combo archetype; otherwise reported.
    tutor_copies, tutor_names = _count_copies(cards, roles, AXIS_ROLES["tutors"])
    low, high = overrides.get("tutors", (None, None))
    axes.append(_axis("tutors", tutor_copies, "copies", tutor_names,
                      "copies carrying tutor:unrestricted or tutor:narrow",
                      low=low, high=high))
    return axes


# ── Engine activation ────────────────────────────────────────────────────

# What the taxonomy misses, named rather than silently absorbed.
#
# `card_roles.json` files Yawgmoth, Thran Physician as `removal:debuff` — and his
# ability reads "Pay 1 life, Sacrifice another creature: Put a -1/-1 counter on up
# to one target creature and draw a card." He IS that deck's draw engine, and the
# card-advantage axis counted five without him. An agent reading "UNDER" there
# would report a weakness the deck's own verified stacks refute.
#
# These probes never change a count — the count stays traceable to the taxonomy.
# They name cards the axis's function appears on anyway, so the reader can decide.
_AXIS_PROBES = {
    "card-advantage": re.compile(r"draws? (a card|\w+ cards?|cards)", re.I),
    "ramp": re.compile(r"add (\{[wubrgcx\d]\}|one mana|two mana|three mana|\w+ mana)",
                       re.I),
    # `ward\b` without a leading boundary matched "toward" on Gray Merchant of
    # Asphodel, and `regenerate` matched Snuff Out's "It can't be regenerated" —
    # a clause that removes protection rather than granting it.
    "protection": re.compile(
        r"(hexproof|indestructible|protection from|\bward\b|can't be the target|"
        r"\bshroud\b|phases? out)", re.I),
}

# The ramp probe reads "add {B}" off every land in the deck, and a land that taps
# for mana is the mana base, not acceleration.
_PROBE_SKIPS_LANDS = frozenset({"ramp"})

_ROLE_TO_AXIS = {role: axis for axis, members in AXIS_ROLES.items() for role in members}


def _probe_uncounted(axis, cards, roles, counted_names):
    """Cards whose oracle text shows the axis's function but whose roles do not.

    Reported, never counted. A floor that says why it is a floor.
    """
    pattern = _AXIS_PROBES.get(axis)
    if pattern is None:
        return []
    counted = set(counted_names)
    skip_lands = axis in _PROBE_SKIPS_LANDS
    out = []
    for card in cards:
        if card["name"] in counted:
            continue
        if skip_lands and is_land(card):
            continue
        if pattern.search(str(card.get("oracle_text", "") or "")):
            out.append(card["name"])
    return sorted(out)


def _component(group, deck_names, roles):
    """One `any_of` group, priced.

    `size` counts only members actually in the 99 — goldfish already warns about
    ghosts, and a component's redundancy is what the shuffler can find, not what
    the target file remembers.
    """
    members = list(group.get("any_of") or [])
    present = [n for n in members if n in deck_names]
    ghosts = [n for n in members if n not in deck_names]
    size = len(present)
    role_counts = Counter(r for n in present for r in roles.get(n, []))
    # Two members can do the same job under different role names — Sol Ring is
    # ramp:rock and Dark Ritual is ramp:ritual, and a group holding both is a
    # ramp group whatever the raw counter says. The axis rollup is the second,
    # coarser signature; the raw role wins when it exists because it is tighter.
    axis_counts = Counter()
    for name in present:
        for axis in {_ROLE_TO_AXIS[r] for r in roles.get(name, [])
                     if r in _ROLE_TO_AXIS}:
            axis_counts[axis] += 1
    return {
        "cards": present,
        "not_in_deck": ghosts,
        "size": size,
        "odds": {
            "opening_seven": round(hypergeometric_at_least(
                size, cards_seen(1), 1, DECK_SIZE_AFTER_COMMANDER), 3),
            "by_turn_three": round(hypergeometric_at_least(
                size, cards_seen(3), 1, DECK_SIZE_AFTER_COMMANDER), 3),
        },
        "shared_roles": [r for r, n in role_counts.most_common(3) if n > 1],
        "shared_axes": [a for a, n in axis_counts.most_common(3) if n > 1],
        "modal_role": role_counts.most_common(1)[0][0] if role_counts else None,
        "thin": size <= ENGINE_THIN_GROUP,
    }


def _closers(component, siblings, deck_names, identity, pool, roles, details):
    """Pool cards that would join this component — two routes, both mechanical.

    Route one is the role signature: a group of sacrifice outlets takes another
    sacrifice outlet. Route two is the combo graph: a card that completes a line
    with the OTHER components of the same target is a piece of this engine even
    when the taxonomy has no word for it. Neither is a recommendation — every
    combo line stays "needs a stack scenario" until a checker says otherwise.

    **The role route needs a SHARED role, not a modal one.** Run it off one
    card's roles and a component holding only Blowfly Infestation returns
    Massacre Wurm and Dismember — the roles describe the card, not the job the
    group does. A role held by two or more members is the only evidence that the
    group has a role at all, so a singleton component gets the combo route only.
    """
    if pool is None:
        return {"available": False,
                "reason": "requires a pipeline run (data/cards.csv is absent)"}

    def eligible(name):
        info = pool.get(name)
        return (info is not None and name not in deck_names and info["legal"]
                and info["color_identity"] <= identity)

    by_role = []
    shared_roles = component.get("shared_roles") or []
    shared_axes = component.get("shared_axes") or []
    if shared_roles:
        signature, kind = shared_roles[0], "role"
        wanted = {signature}
    elif shared_axes:
        signature, kind = shared_axes[0], "axis"
        wanted = set(AXIS_ROLES[signature])
    else:
        signature, kind, wanted = None, None, set()
    if signature:
        for name, info in pool.items():
            if eligible(name) and wanted & set(roles.get(name, [])):
                by_role.append({"name": name, "signature": signature,
                                "cmc": info["cmc"], "type_line": info["type_line"],
                                "edhrec_rank": info["edhrec_rank"],
                                "game_changer": info["game_changer"]})
        by_role.sort(key=lambda e: (e["edhrec_rank"] or 10**9, e["name"]))

    by_line = []
    others = {n for sib in siblings for n in sib["cards"]}
    if others and details:
        for name, indices in (details.get("by_card") or {}).items():
            if not eligible(name):
                continue
            for idx in indices:
                combo = details["combos"][idx]
                rest = set(combo["cards"]) - {name}
                if not rest or not rest <= deck_names or not (rest & others):
                    continue
                by_line.append({
                    "name": name, "cards": combo["cards"],
                    "produces": combo.get("produces", []),
                    "popularity": combo.get("popularity"),
                    "status": "needs a stack scenario",
                })
                break
        by_line.sort(key=lambda e: (-(e["popularity"] or 0), e["name"]))

    return {
        "available": True,
        "role_signature": signature,
        "role_signature_kind": kind,
        "by_role": by_role[:ENGINE_MAX_CLOSERS],
        "by_combo_line": by_line[:ENGINE_MAX_CLOSERS],
        "by_role_total": len(by_role),
        "by_combo_line_total": len(by_line),
        "note": (None if signature else
                 "No role and no axis is shared by two or more members, so this "
                 "component has no signature and only the combo route ran. That "
                 "is the normal case for a named combo half."),
    }


def engine_activation(slug, cards, roles, identity, load_pool_fn):
    """The deck's own engine declaration, priced and measured.

    `goldfish_targets.json` is hand-authored and already says what the deck is
    trying to assemble; `goldfish_metrics.json` already says how often it does.
    Joining them by label is the whole trick.
    """
    targets_doc = load_json(deck_dir(slug) / "goldfish_targets.json", default=None)
    if not targets_doc:
        return {"available": False,
                "reason": "no goldfish_targets.json — the deck has not declared "
                          "what it is trying to assemble"}
    metrics = load_json(deck_dir(slug) / "goldfish_metrics.json", default=None) or {}
    measured = {t["label"]: t for t in (metrics.get("metrics", {}).get("targets") or [])}
    deck_names = {c["name"] for c in cards}

    pool = details = None
    out = []
    for target in targets_doc.get("targets", []):
        comps = [_component(g, deck_names, roles) for g in target.get("need", [])]
        if not comps:
            continue
        thinnest = min(range(len(comps)), key=lambda i: comps[i]["size"])
        comps[thinnest]["thinnest"] = True
        for i, comp in enumerate(comps):
            comp.setdefault("thinnest", False)
            if not comp["thin"]:
                continue
            if pool is None:
                pool = load_pool_fn()
                try:
                    details = load_combo_details()
                except FileNotFoundError:
                    details = None
            comp["closers"] = _closers(
                comp, [c for j, c in enumerate(comps) if j != i],
                deck_names, identity, pool, roles, details)
        row = {
            "label": target["label"],
            "components": comps,
            "thinnest_size": comps[thinnest]["size"],
            "commander_gated": bool(target.get("commander")),
        }
        m = measured.get(target["label"])
        if m:
            row["measured"] = {k: m[k] for k in
                               ("assembled_rate", "assembled_rate_assisted",
                                "by_turn_6_rate", "by_turn_6_rate_assisted",
                                "mean_turn") if k in m}
        out.append(row)

    single_points = [
        {"target": r["label"],
         "component": next(c for c in r["components"] if c["thinnest"]),
         "measured": r.get("measured")}
        for r in out if r["thinnest_size"] <= 1
    ]
    return {
        "available": True,
        "redundancy_citation": dict(ENGINE_REDUNDANCY_CITATION),
        "targets": out,
        "single_points_of_failure": single_points,
    }


# ── Freshness ────────────────────────────────────────────────────────────

def freshness(slug, doc):
    """Which derived artifacts were computed against the current decklist.

    A diagnosis of stale numbers is worse than none, and every artifact here
    stamps the decklist it was built from — so this is a comparison, not a guess.
    """
    current = doc.get("decklist_sha256")
    out = {"decklist_sha256": current, "artifacts": {}}
    checks = {
        "goldfish_metrics.json": ("meta", "decklist_sha256"),
        "mana_analysis.json": (None, "decklist_sha256"),
    }
    for filename, (block, key) in checks.items():
        art = load_json(deck_dir(slug) / filename, default=None)
        if art is None:
            out["artifacts"][filename] = {"present": False}
            continue
        stamped = (art.get(block) or {}).get(key) if block else art.get(key)
        out["artifacts"][filename] = {
            "present": True, "decklist_sha256": stamped,
            "current": stamped == current,
        }
    # bracket_report.json carries no stamp; report its presence so the caller
    # knows the power axis is answerable at all.
    out["artifacts"]["bracket_report.json"] = {
        "present": (deck_dir(slug) / "bracket_report.json").exists(),
        "decklist_sha256": None, "current": None,
    }
    recon = load_json(deck_dir(slug) / "deck_recon.json", default=None)
    out["recon"] = ({"present": True, "as_of": recon.get("as_of")} if recon
                    else {"present": False, "as_of": None})
    return out


# ── Notes ────────────────────────────────────────────────────────────────

def build_notes(audit):
    notes = []
    stale = [name for name, a in audit["freshness"]["artifacts"].items()
             if a.get("present") and a.get("current") is False]
    if stale:
        notes.append(
            "Computed against a DIFFERENT decklist than cards.json: "
            + ", ".join(sorted(stale))
            + ". Every figure sourced from those is stale — re-run them before "
              "quoting anything here."
        )
    missing = [name for name, a in audit["freshness"]["artifacts"].items()
               if not a.get("present")]
    if missing:
        notes.append(
            "Absent, so the axes that read them are omitted rather than defaulted: "
            + ", ".join(sorted(missing)) + "."
        )
    if audit["archetype"]["archetype"] is None:
        notes.append(
            "No archetype was established, so the BASE targets apply and no "
            "archetype override was used. strategy:deckbuilding.ratios is explicit "
            "that templates are counts of functions rather than cards and that the "
            "counts should come from the deck's actual failure modes — read every "
            "target below as a starting point, not a verdict."
        )
    under = [a["axis"] for a in audit["axes"] if a["verdict"] == "under"]
    over = [a["axis"] for a in audit["axes"] if a["verdict"] == "over"]
    if under:
        notes.append("Under target: " + ", ".join(under) + ".")
    if over:
        notes.append("Over target: " + ", ".join(over) + ".")
    breadth = next((a for a in audit["axes"] if a["axis"] == "interaction-breadth"), None)
    if breadth and breadth["measured"].get("uncovered"):
        notes.append(
            "Permanent class(es) with no answer detected: "
            + ", ".join(breadth["measured"]["uncovered"])
            + ". The detection is an oracle-text heuristic and counterspells are "
              "excluded from it, so this is a floor on breadth — read the cards "
              "before calling a class uncovered."
        )
    engine = audit["engine"]
    if engine.get("available"):
        spf = engine["single_points_of_failure"]
        if spf:
            notes.append(
                f"{len(spf)} declared target(s) rest on a component with one card. "
                "A named card is 7.1% of an opening seven; that is the ceiling on "
                "those lines, not a defect in the simulation."
            )
        notes.append(
            "Each target is an AND of ORs and the schema cannot express the UNION "
            "of several independent kills — a deck with four kills has no single "
            "assembled_rate here, and averaging them would be inventing a number "
            "the simulation never measured."
        )
    else:
        notes.append(
            "No engine block: " + engine.get("reason", "unavailable")
            + ". Authoring goldfish_targets.json is what makes the engine "
              "questions answerable at all."
        )
    probed = {a["axis"]: a["measured"]["possible_uncounted"]
              for a in audit["axes"]
              if a["measured"].get("possible_uncounted")
              and a["verdict"] == "under"}
    if probed:
        notes.append(
            "Under target, but the oracle text shows the function on card(s) the "
            "taxonomy did not classify that way — check these before calling the "
            "axis a weakness: "
            + "; ".join(f"{axis} — {', '.join(names[:6])}"
                        + (" …" if len(names) > 6 else "")
                        for axis, names in sorted(probed.items()))
            + ". They are NOT included in the count."
        )
    notes.append(
        "Roles come from card_roles.json, which classifies "
        f"{audit['roles']['coverage'] * 100:.1f}% of this deck. An axis count is a "
        "floor: a card the taxonomy has no pattern for is invisible to it, not "
        "absent from the deck."
    )
    return notes


# ── Entry point ──────────────────────────────────────────────────────────

def analyze(slug, archetype=None, load_pool_fn=None):
    if load_pool_fn is None:
        from manamap.pilot.card_pool import load_pool as load_pool_fn

    doc = load_deck_cards(slug)
    cards = doc.get("cards", [])
    facts = deck_facts_mod.analyze(slug)
    roles = load_card_roles() if CARD_ROLES_PATH.exists() else {}
    mana = load_json(deck_dir(slug) / "mana_analysis.json", default=None)
    goldfish = load_json(deck_dir(slug) / "goldfish_metrics.json", default=None)
    bracket = load_json(deck_dir(slug) / "bracket_report.json", default=None)

    detected = detect_archetype(slug, archetype)
    overrides, citation = archetype_overrides(detected["archetype"])
    detected["overrides"] = {k: list(v) for k, v in overrides.items()}
    if citation:
        detected["citation"] = citation

    identity = {col for c in cards if c.get("is_commander")
                for col in (c.get("color_identity") or [])}

    audit = {
        "slug": slug,
        "commander": facts["commander"],
        "archetype": detected,
        "freshness": freshness(slug, doc),
        "counts": facts["counts"],
        "roles": {"coverage": facts["roles"].get("coverage", 0.0),
                  "no_role": facts["roles"].get("no_role", [])},
        "axes": build_axes(slug, cards, roles, facts, mana, goldfish, bracket, overrides),
        "engine": engine_activation(slug, cards, roles, identity, load_pool_fn),
    }
    audit["notes"] = build_notes(audit)
    return audit


def format_report(audit):
    lines = [
        f"{audit['slug']}: deck audit — archetype "
        f"{audit['archetype']['archetype'] or 'unestablished'} "
        f"({audit['archetype']['source']})"
    ]
    for axis in audit["axes"]:
        m, t = axis["measured"], axis["target"]
        bound = ""
        if t:
            lo, hi = t.get("low"), t.get("high")
            if lo is not None and hi is not None:
                bound = f" (target {lo}-{hi})"
            elif lo is not None:
                bound = f" (target >= {lo})"
            elif hi is not None:
                bound = f" (target <= {hi})"
        flag = {"under": "UNDER", "over": "OVER "}.get(axis["verdict"], "     ")
        lines.append(f"  {flag} {axis['axis']:<20} {m['value']!s:>7} {m['unit']}{bound}")
    engine = audit["engine"]
    if engine.get("available"):
        lines.append("  engine (declared targets, thinnest component first):")
        for row in sorted(engine["targets"], key=lambda r: r["thinnest_size"]):
            rate = (row.get("measured") or {}).get("assembled_rate")
            thin = next(c for c in row["components"] if c["thinnest"])
            lines.append(
                f"    [{thin['size']} card(s), {thin['odds']['opening_seven']:.0%} "
                f"in an opener] {row['label'][:64]}"
                + (f" — measured {rate:.1%}" if rate is not None else "")
            )
            closers = thin.get("closers") or {}
            if closers.get("available"):
                picks = [c["name"] for c in closers["by_role"][:3]]
                picks += [c["name"] for c in closers["by_combo_line"][:3]]
                if picks:
                    lines.append(f"        closers: {', '.join(dict.fromkeys(picks))}")
    for note in audit["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(args):
    audit = analyze(args.slug, archetype=getattr(args, "archetype", None))
    if getattr(args, "as_json", False):
        print(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(audit))
    out = getattr(args, "out", None)
    if out:
        path = resolve_out_path(out, args.slug, "deck-audit")
        with open(path, "w") as f:
            json.dump(audit, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-audit <slug>`.")
