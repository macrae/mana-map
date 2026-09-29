"""The FORGE half of a pilot policy: 121 AI knobs, their gates, and the compiled profile.

WHY THIS IS NOT IN `pilot_policy.py`, AND IT IS NOT TIDINESS. `pilot_policy.py` is in
`goldfish._MODEL_FILES`, added 2026-09-28 because the policy layer decides whether a card is
cast at all and a change to it must stale every derived figure. `model_version` is a sha over
the WHOLE FILE, coarse on purpose — "a comment edit bumps it, which costs a regeneration
nobody needed".

Putting 121 Forge constants in there made every knob edit stale all 37 goldfish artifacts.
It happened immediately: adding them moved `model_version` c42ed8ea2ec3 -> e41cebc10cf0 and
reddened ~25 fleet tests within the hour, for constants the goldfish does not read and cannot
read — they are AI properties of an external engine.

So the split follows what actually consumes what. The goldfish reads `CHANNELS`, `VERBS`,
`COUNTERS` and `hold_thresholds`; Forge reads these. A knob edit now stales Forge runs, which
it should, and leaves the goldfish alone, which it should.

WHAT THESE KNOBS ARE. `res/ai/*.ai` are plain KEY=VALUE files;
`AiProfileUtil.getAvailableProfiles()` is `new File(res/ai).list()`, so any file dropped there
becomes a name `-a` accepts; and `loadProfile()` falls back to `AiProps.getDefault()` per key,
so a PARTIAL file holding only the keys we change is legal. That is Forge's own supported lever
for piloting, and unlike a card-script override it needs no engine patching at all.
"""

import hashlib
import json

from manamap.pilot.pilot_policy import PolicyError, VERBS

#: Every `AiProps` key Forge 2.0.14 exposes, with its DEFAULT and type, read out of
#: `res/ai/Default.ai`. Tracked here rather than read at validation time so a policy
#: can be checked on a machine with no Forge install — and re-derived from the engine
#: by a test that fails on drift, which is how `pod_behaviour.POD` keeps a measured
#: constant honest.
FORGE_AI_DEFAULTS = {
    "MULLIGAN_THRESHOLD": (4, "int"),
    "PLAY_AGGRO": (False, "bool"),
    "CHANCE_TO_ATTACK_INTO_TRADE": (0, "int"),
    "ATTACK_INTO_TRADE_WHEN_TAPPED_OUT": (False, "bool"),
    "RANDOMLY_ATKTRADE_ONLY_ON_LOWER_LIFE_PRESSURE": (True, "bool"),
    "CHANCE_TO_ATKTRADE_WHEN_OPP_HAS_MANA": (30, "int"),
    "TRY_TO_AVOID_ATTACKING_INTO_CERTAIN_BLOCK": (True, "bool"),
    "USE_BERSERK_AGGRESSIVELY": (True, "bool"),
    "TRY_TO_HOLD_COMBAT_TRICKS_UNTIL_BLOCK": (True, "bool"),
    "CHANCE_TO_HOLD_COMBAT_TRICKS_UNTIL_BLOCK": (65, "int"),
    "ENABLE_RANDOM_FAVORABLE_TRADES_ON_BLOCK": (True, "bool"),
    "RANDOMLY_TRADE_EVEN_WHEN_HAVE_LESS_CREATS": (False, "bool"),
    "MAX_DIFF_IN_CREATURE_COUNT_TO_TRADE": (1, "int"),
    "ALSO_TRADE_WHEN_HAVE_A_REPLACEMENT_CREAT": (True, "bool"),
    "MAX_DIFF_IN_CREATURE_COUNT_TO_TRADE_WITH_REPL": (1, "int"),
    "MIN_CHANCE_TO_RANDOMLY_TRADE_ON_BLOCK": (30, "int"),
    "MAX_CHANCE_TO_RANDOMLY_TRADE_ON_BLOCK": (70, "int"),
    "CHANCE_DECREASE_TO_TRADE_VS_EMBALM": (30, "int"),
    "CHANCE_TO_TRADE_TO_SAVE_PLANESWALKER": (70, "int"),
    "CHANCE_TO_TRADE_DOWN_TO_SAVE_PLANESWALKER": (0, "int"),
    "THRESHOLD_NONTOKEN_CHUMP_TO_SAVE_PLANESWALKER": (120, "int"),
    "THRESHOLD_TOKEN_CHUMP_TO_SAVE_PLANESWALKER": (135, "int"),
    "CHUMP_TO_SAVE_PLANESWALKER_ONLY_ON_LETHAL": (True, "bool"),
    "AVOID_TARGETING_CREATS_THAT_WILL_DIE": (True, "bool"),
    "DONT_EVAL_KILLSPELLS_ON_STACK_WITH_PERMISSION": (True, "bool"),
    "AI_IN_DANGER_THRESHOLD": (4, "int"),
    "AI_IN_DANGER_MAX_THRESHOLD": (4, "int"),
    "CHEAT_WITH_MANA_ON_SHUFFLE": (True, "bool"),
    "HOLD_LAND_DROP_FOR_MAIN2_IF_UNUSED": (100, "int"),
    "HOLD_LAND_DROP_ONLY_IF_HAVE_OTHER_PERMS": (True, "bool"),
    "DEFAULT_MAX_PLANAR_DIE_ROLLS_PER_TURN": (1, "int"),
    "DEFAULT_MIN_TURN_TO_ROLL_PLANAR_DIE": (3, "int"),
    "DEFAULT_PLANAR_DIE_ROLL_CHANCE": (50, "int"),
    "PLANAR_DIE_ROLL_HESITATION_CHANCE": (10, "int"),
    "MOVE_EQUIPMENT_TO_BETTER_CREATURES": ('from_useless_only', "str"),
    "MOVE_EQUIPMENT_CREATURE_EVAL_THRESHOLD": (60, "int"),
    "PRIORITIZE_MOVE_EQUIPMENT_IF_USELESS": (True, "bool"),
    "SAC_TO_REATTACH_TARGET_EVAL_THRESHOLD": (400, "int"),
    "PREDICT_SPELLS_FOR_MAIN2": (True, "bool"),
    "RESERVE_MANA_FOR_MAIN2_CHANCE": (100, "int"),
    "ACTIVELY_PROTECT_VS_CURSE_AURAS": (True, "bool"),
    "ACTIVELY_DESTROY_ARTS_AND_NONAURA_ENCHS": (True, "bool"),
    "ACTIVELY_DESTROY_IMMEDIATELY_UNBLOCKABLE": (True, "bool"),
    "DESTROY_IMMEDIATELY_UNBLOCKABLE_THRESHOLD": (2, "int"),
    "DESTROY_IMMEDIATELY_UNBLOCKABLE_ONLY_IN_DNGR": (True, "bool"),
    "DESTROY_IMMEDIATELY_UNBLOCKABLE_LIFE_IN_DNGR": (5, "int"),
    "CHANCE_TO_CHAIN_TWO_DAMAGE_SPELLS": (90, "int"),
    "HOLD_X_DAMAGE_SPELLS_FOR_MORE_DAMAGE_CHANCE": (100, "int"),
    "HOLD_X_DAMAGE_SPELLS_THRESHOLD": (5, "int"),
    "MIN_SPELL_CMC_TO_COUNTER": (0, "int"),
    "CHANCE_TO_COUNTER_CMC_1": (30, "int"),
    "CHANCE_TO_COUNTER_CMC_2": (75, "int"),
    "CHANCE_TO_COUNTER_CMC_3": (100, "int"),
    "ALWAYS_COUNTER_OTHER_COUNTERSPELLS": (True, "bool"),
    "ALWAYS_COUNTER_DAMAGE_SPELLS": (True, "bool"),
    "ALWAYS_COUNTER_CMC_0_MANA_MAKING_PERMS": (True, "bool"),
    "ALWAYS_COUNTER_REMOVAL_SPELLS": (True, "bool"),
    "ALWAYS_COUNTER_PUMP_SPELLS": (True, "bool"),
    "ALWAYS_COUNTER_AURAS": (True, "bool"),
    "ALWAYS_COUNTER_SPELLS_FROM_NAMED_CARDS": ('None', "str"),
    "CHANCE_TO_COPY_OWN_SPELL_WHILE_ON_STACK": (30, "int"),
    "ALWAYS_COPY_SPELL_IF_CMC_DIFF": (2, "int"),
    "PRIORITY_REDUCTION_FOR_STORM_SPELLS": (9, "int"),
    "MIN_COUNT_FOR_STORM_SPELLS": (1, "int"),
    "STRIPMINE_MIN_LANDS_IN_HAND_TO_ACTIVATE": (1, "int"),
    "STRIPMINE_MIN_LANDS_FOR_NO_TIMING_CHECK": (9999, "int"),
    "STRIPMINE_MIN_LANDS_OTB_FOR_NO_TEMPO_CHECK": (6, "int"),
    "STRIPMINE_MAX_LANDS_TO_ATTEMPT_MANALOCKING": (3, "int"),
    "STRIPMINE_HIGH_PRIORITY_ON_SKIPPED_LANDDROP": (True, "bool"),
    "TOKEN_GENERATION_ABILITY_CHANCE": (80, "int"),
    "TOKEN_GENERATION_ALWAYS_IF_FROM_PLANESWALKER": (True, "bool"),
    "TOKEN_GENERATION_ALWAYS_IF_OPP_ATTACKS": (True, "bool"),
    "FLASH_ENABLE_ADVANCED_LOGIC": (True, "bool"),
    "FLASH_CHANCE_TO_OBEY_AMBUSHAI": (100, "int"),
    "FLASH_CHANCE_TO_CAST_DUE_TO_ETB_EFFECTS": (100, "int"),
    "FLASH_CHANCE_TO_CAST_FOR_ETB_BEFORE_MAIN1": (10, "int"),
    "FLASH_CHANCE_TO_RESPOND_TO_STACK_WITH_ETB": (0, "int"),
    "FLASH_CHANCE_TO_CAST_AS_VALUABLE_BLOCKER": (100, "int"),
    "FLASH_USE_BUFF_AURAS_AS_COMBAT_TRICKS": (True, "bool"),
    "FLASH_BUFF_AURA_CHANCE_TO_CAST_EARLY": (1, "int"),
    "FLASH_BUFF_AURA_CHANCE_CAST_AT_EOT": (5, "int"),
    "FLASH_BUFF_AURA_CHANCE_TO_RESPOND_TO_STACK": (100, "int"),
    "SCRY_NUM_LANDS_TO_STILL_NEED_MORE": (4, "int"),
    "SCRY_NUM_LANDS_TO_NOT_NEED_MORE": (7, "int"),
    "SCRY_NUM_CREATURES_TO_NOT_NEED_SUBPAR_ONES": (4, "int"),
    "SCRY_EVALTHR_CREATCOUNT_TO_SCRY_AWAY_LOWCMC": (3, "int"),
    "SCRY_EVALTHR_TO_SCRY_AWAY_LOWCMC_CREATURE": (160, "int"),
    "SCRY_EVALTHR_CMC_THRESHOLD": (3, "int"),
    "SCRY_IMMEDIATELY_UNCASTABLE_TO_BOTTOM": (True, "bool"),
    "SCRY_IMMEDIATELY_UNCASTABLE_CMC_DIFF": (1, "int"),
    "SURVEIL_NUM_CARDS_IN_LIBRARY_TO_BAIL": (10, "int"),
    "SURVEIL_LIFEPERC_AFTER_PAYING_LIFE": (60, "int"),
    "COMBAT_ASSAULT_ATTACK_EVASION_PREDICTION": (True, "bool"),
    "COMBAT_ATTRITION_ATTACK_EVASION_PREDICTION": (True, "bool"),
    "CONSERVATIVE_ENERGY_PAYMENT_ONLY_IN_COMBAT": (True, "bool"),
    "CONSERVATIVE_ENERGY_PAYMENT_ONLY_DEFENSIVELY": (False, "bool"),
    "BOUNCE_ALL_TO_HAND_CREAT_EVAL_DIFF": (200, "int"),
    "BOUNCE_ALL_ELSEWHERE_CREAT_EVAL_DIFF": (200, "int"),
    "BOUNCE_ALL_TO_HAND_NONCREAT_EVAL_DIFF": (3, "int"),
    "BOUNCE_ALL_ELSEWHERE_NONCREAT_EVAL_DIFF": (3, "int"),
    "BLINK_RELOAD_PLANESWALKER_CHANCE": (30, "int"),
    "BLINK_RELOAD_PLANESWALKER_MAX_LOYALTY": (2, "int"),
    "BLINK_RELOAD_PLANESWALKER_LOYALTY_DIFF": (2, "int"),
    "INTUITION_ALTERNATIVE_LOGIC": (True, "bool"),
    "TRY_TO_PRESERVE_BUYBACK_SPELLS": (True, "bool"),
    "EXPLORE_MAX_CMC_DIFF_TO_PUT_IN_GRAVEYARD": (2, "int"),
    "EXPLORE_NUM_LANDS_TO_STILL_NEED_MORE": (2, "int"),
    "MOMIR_BASIC_LAND_STRATEGY": ('default', "str"),
    "MOJHOSTO_NUM_LANDS_TO_ACTIVATE_JHOIRA": (4, "int"),
    "MOJHOSTO_CHANCE_TO_PREFER_JHOIRA_OVER_MOMIR": (50, "int"),
    "MOJHOSTO_CHANCE_TO_USE_JHOIRA_COPY_INSTANT": (20, "int"),
    "SACRIFICE_DEFAULT_PREF_ENABLE": (False, "bool"),
    "SACRIFICE_DEFAULT_PREF_MIN_CMC": (0, "int"),
    "SACRIFICE_DEFAULT_PREF_MAX_CMC": (2, "int"),
    "SACRIFICE_DEFAULT_PREF_ALLOW_TOKENS": (True, "bool"),
    "SACRIFICE_DEFAULT_PREF_MAX_CREATURE_EVAL": (135, "int"),
    "SIDEBOARDING_IN_LIMITED_FORMATS": (False, "bool"),
    "SIDEBOARDING_CHANCE_PER_CARD": (50, "int"),
    "SIDEBOARDING_CHANCE_ON_WIN": (0, "int"),
    "SIDEBOARDING_SHARED_TYPE_ONLY": (False, "bool"),
    "SIDEBOARDING_PLANESWALKER_EQ_CREATURE": (False, "bool"),
}


#: MASTER TOGGLES, and the keys each one gates — Forge's own comments say so, verbatim:
#: "Master toggle for the following options", "If disabled, the following three options do
#: nothing", "If it is disabled, the following related options have no effect".
#:
#: THIS IS THE BLOCKER AN ADVERSARIAL READ OF THE PLAN CAUGHT. `SACRIFICE_DEFAULT_PREF_ENABLE`
#: is **false** in `Default.ai`, so a policy declaring `SACRIFICE_DEFAULT_PREF_MAX_CMC=3` on a
#: Default base does NOTHING — the A/B returns no difference, the loop records UNPROVEN, and
#: the conclusion drawn is "the sacrifice knob does not help" about a knob that was never on.
#: That is CLAUDE.md's "A CARD THE MODEL CANNOT READ LOOKS EXACTLY LIKE A CARD THAT DOES NOT
#: HELP" one layer up, and it is worth a validation error rather than a comment.
#:
#: `PLAY_AGGRO` is deliberately NOT here. Its family is not a clean gate: Forge says
#: `CHANCE_TO_ATTACK_INTO_TRADE` "works even if not playing all-out aggro, e.g. PLAY_AGGRO
#: disabled", while `ATTACK_INTO_TRADE_WHEN_TAPPED_OUT` is "ignored if PLAY_AGGRO is globally
#: enabled" — an INVERSE relation. Encoding it as a master would refuse a correct policy.
FORGE_AI_MASTERS = {
    "SACRIFICE_DEFAULT_PREF_ENABLE": (
        "SACRIFICE_DEFAULT_PREF_MIN_CMC", "SACRIFICE_DEFAULT_PREF_MAX_CMC",
        "SACRIFICE_DEFAULT_PREF_ALLOW_TOKENS",
        "SACRIFICE_DEFAULT_PREF_MAX_CREATURE_EVAL"),
    "ACTIVELY_DESTROY_IMMEDIATELY_UNBLOCKABLE": (
        "DESTROY_IMMEDIATELY_UNBLOCKABLE_THRESHOLD",
        "DESTROY_IMMEDIATELY_UNBLOCKABLE_ONLY_IN_DNGR",
        "DESTROY_IMMEDIATELY_UNBLOCKABLE_LIFE_IN_DNGR"),
    "ENABLE_RANDOM_FAVORABLE_TRADES_ON_BLOCK": (
        "RANDOMLY_TRADE_EVEN_WHEN_HAVE_LESS_CREATS",
        "MAX_DIFF_IN_CREATURE_COUNT_TO_TRADE",
        "ALSO_TRADE_WHEN_HAVE_A_REPLACEMENT_CREAT",
        "MAX_DIFF_IN_CREATURE_COUNT_TO_TRADE_WITH_REPL"),
    "FLASH_ENABLE_ADVANCED_LOGIC": (
        "FLASH_CHANCE_TO_OBEY_AMBUSHAI",
        "FLASH_CHANCE_TO_CAST_DUE_TO_ETB_EFFECTS",
        "FLASH_CHANCE_TO_CAST_FOR_ETB_BEFORE_MAIN1",
        "FLASH_CHANCE_TO_RESPOND_TO_STACK_WITH_ETB",
        "FLASH_CHANCE_TO_CAST_AS_VALUABLE_BLOCKER",
        "FLASH_USE_BUFF_AURAS_AS_COMBAT_TRICKS"),
}

#: Which master gates a given key.
FORGE_AI_GATED_BY = {child: master
                     for master, children in FORGE_AI_MASTERS.items()
                     for child in children}


def validate_forge(doc):
    """Refuse a `forge` section Forge would ignore. Returns the section, or `{}`.

    THE DANGEROUS ERROR HERE IS NOT AN INVALID KEY, it is a key Forge accepts and does
    nothing with. A policy declaring `SACRIFICE_DEFAULT_PREF_MAX_CMC: 3` reads as a
    deliberate piloting choice, compiles into a legal `.ai` file, runs 400 games and returns
    no difference — because `SACRIFICE_DEFAULT_PREF_ENABLE` is FALSE in `Default.ai` and the
    whole family is inert. The loop would then record UNPROVEN and the conclusion drawn is
    "the sacrifice knob does not help", about a knob that was never on.

    So a gated key requires its master, and the error says which.
    """
    forge = doc.get("forge")
    if forge is None:
        return {}
    if not isinstance(forge, dict):
        raise PolicyError(f"`forge` is {type(forge).__name__}, not a mapping of "
                          f"AiProps keys to values")
    for key, value in sorted(forge.items()):
        if key not in FORGE_AI_DEFAULTS:
            raise PolicyError(
                f"forge.{key} is not an AiProps key Forge reads. A policy cannot invent a "
                f"knob; `Default.ai` exposes {len(FORGE_AI_DEFAULTS)} of them.")
        default, kind = FORGE_AI_DEFAULTS[key]
        if kind == "bool" and not isinstance(value, bool):
            raise PolicyError(f"forge.{key} is a flag; give true or false, not {value!r}")
        if kind == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise PolicyError(f"forge.{key} is a number; got {value!r}")
        if value == default:
            raise PolicyError(
                f"forge.{key} is already {default!r} in Default.ai, so declaring it changes "
                f"nothing. A policy that restates a default is a rule nobody can measure — "
                f"drop it, or set a different value.")
        master = FORGE_AI_GATED_BY.get(key)
        if master is None:
            continue
        master_default = FORGE_AI_DEFAULTS[master][0]
        if forge.get(master, master_default) is not True:
            raise PolicyError(
                f"forge.{key} is gated by {master}, which is "
                f"{forge.get(master, master_default)!r} — so Forge would read this key and "
                f"do nothing with it. Set {master}: true in the same policy, or drop "
                f"{key}. (A knob that was never switched on measures as a knob that does "
                f"not help.)")
    return forge


def compile_profile(doc, name=None):
    """The `.ai` text for this policy: ONLY the keys it changes.

    A partial profile is legal — `AiProfileUtil.loadProfile` falls back to
    `AiProps.getDefault()` per key (verified in the 2.0.14 bytecode) — so the file IS the
    diff. That matters for reading a run later: the profile beside a record says what was
    changed and nothing else, where a full 121-key copy would bury four deltas in 117
    restatements.
    """
    forge = validate_forge(doc)
    lines = [
        "# GENERATED by `manamap pilot forge-install` from pilot_policy.json.",
        "# Do not edit: the policy is the source, and an edit here is invisible to every",
        "# record, because `card_overrides`/`ai_profile` fingerprint the POLICY.",
        "#",
        "# Partial on purpose. Forge falls back to AiProps.getDefault() per key, so this",
        "# file is the DIFF from Default.ai and reads as the piloting decision it is.",
    ]
    if name:
        lines.append(f"# profile: {name}")
    for key, value in sorted(forge.items()):
        why = next((r.get("why") for r in (doc.get("rules") or [])
                    if key in (r.get("forge") or {})), None)
        if why:
            lines.append(f"# {why}")
        # THE DEFAULT, BESIDE THE NEW VALUE, so the DIRECTION of the change is visible
        # where the change is. Writing the first trial policy I set
        # `MIN_COUNT_FOR_STORM_SPELLS: 2` under a `why` that said "lowering it" — the
        # default is 1, so the value RAISED the threshold and would have made the AI hold
        # storm spells longer, the exact opposite of the stated reason. Nothing mechanical
        # can check that a reason matches a direction; printing both makes a human able to.
        default = FORGE_AI_DEFAULTS[key][0]
        fmt = (lambda x: "true" if x is True else "false" if x is False else x)
        lines.append(f"#   Default.ai has {fmt(default)}; this sets {fmt(value)}")
        lines.append(f"{key}={fmt(value)}")
    return "\n".join(lines) + "\n"


def forge_fingerprint(doc):
    """A stamp over the compiled profile, for the run id and the record.

    OVER THE COMPILED DELTAS, not the profile NAME. `forge.profile_tag` carries the name
    only, so iterating knob VALUES under one `mm-<slug>` name would write the same run id
    every time and the second run would be refused as an existing measurement — which is
    precisely the collision `profile_tag` and `clock_tag` were each written to prevent,
    left open for the one axis this project adds.
    """
    forge = doc.get("forge") or {}
    if not forge:
        return None
    blob = json.dumps(sorted(forge.items()), separators=(",", ":"))
    return {"sha": hashlib.sha256(blob.encode()).hexdigest()[:12],
            "keys": sorted(forge), "n": len(forge)}
