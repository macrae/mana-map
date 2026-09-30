"""THE PILOT'S DECISIONS, AS DATA — the half of the manual the simulator could not read.

The goldfish has always made piloting decisions. They were written in Python, one
per channel, by whoever added that channel:

    "Cast LAST in the main phase and only with attackers already out"
    "Most expensive pump first"
    "AN UNTAPPER IS HELD, NOT CAST ON CURVE"

Every one of those is a line from a pilot's manual, hardcoded and generic. Meanwhile
the Pilot's Operating Handbook carries the pilot's OWN version of the same thing, in
prose, under *normal procedures* — and the two have never been connected. A policy
the pilot writes could not change a figure, and a policy the model followed could
not be read by a person.

This module is the join: a per-deck `pilot_policy.json` the simulator CONSULTS.

WHY THIS AND NOT AN AGENT PLAYING THE GAMES. An LLM deciding 10,000 games is neither
seeded nor affordable, and nothing here could advance a game state legally anyway —
the goldfish is a stochastic model, `validate_stack` adjudicates one frozen board,
and Forge's sim mode has no external-seat hook (`-a` sets AI profiles and stops
there). A DECLARED policy keeps the ◆ tier: same seed, same figure, re-derivable,
and every rule is a measurable A/B instead of advice. Agents do what they are good
at — authoring a rule and attacking it — not executing it.

ABSENT MEANS ABSENT. A deck with no policy file is byte-identical to one before this
module existed, the same contract every `model_*` channel keeps. A policy can only
ever make the model play a deck DIFFERENTLY, never make a figure appear.

THE FIRST RULE IS ONE THE CITATION LOOP PROVED, not one somebody liked the sound of.
Stack 011 established, with CR citations and an adversarial check, that Zada copies
NOTHING with no other creature out, and that no single card fixes it because the
trigger resolves before the spell that caused it (603.3/603.3b). So a token-copy
spell cast into an empty board is two mana for one token, and the play is to hold it.
`hold_until.other_creatures` is that sentence as a policy.
"""

import hashlib
import json

from manamap.pilot.common import deck_file, load_json

#: What a rule may key on. A `when.channel` names a profile field the simulator
#: already computes, so a policy can never invent a card property.
CHANNELS = ("spell_token_copy",)

#: What a rule may DO. One verb for now, deliberately: a hold threshold is the
#: decision the evidence supports, and a vocabulary that grows one proven verb at a
#: time cannot outrun what has been measured.
#: `forge`: the rule EXPLAINS AiProps keys the document's `forge` section sets — each
#: key it names must be there with the same value, so the reason and the knob cannot
#: drift apart. Such a rule needs no goldfish channel: the goldfish never reads it.
VERBS = ("hold_until", "forge")

#: What `hold_until` may count.
COUNTERS = ("other_creatures",)


class PolicyError(ValueError):
    """A policy that cannot be followed, refused loudly at load."""


def load(slug, branch=None):
    """The deck's policy, or `{}` when it has none.

    Read through `deck_file`, so a BRANCH inherits the deck's policy exactly as it
    inherits `goldfish_targets.json`. Two candidate 99s are only comparable if the
    same hand pilots both — a branch with its own policy would be the "two models,
    not two lists" defect that cost this project a day on 2026-09-27.
    """
    # A SLUG WITH NO DECK DIRECTORY HAS NO POLICY, and `deck_dir` RAISES rather
    # than returning nothing. Every synthetic slug in the suite — `fake`, `x` —
    # reaches here through `goldfish.run`, so without this the policy layer broke
    # five tests that have nothing to do with policy.
    #
    # THE SAME PATTERN WAS FIXED IN `net_change.forge` EARLIER THE SAME DAY and
    # reintroduced here in new code within the hour. A module that reads an
    # authored per-deck file must tolerate a deck that does not exist.
    try:
        path = deck_file(slug, "pilot_policy.json", branch)
    except FileNotFoundError:
        return {}
    doc = load_json(path) or {}
    if doc:
        validate(doc)
    return doc


def validate(doc):
    """Refuse a policy the simulator cannot follow, naming what is wrong."""
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        raise PolicyError("a policy with no `rules` list changes nothing — delete "
                          "the file or write a rule")
    seen = set()
    for i, r in enumerate(rules):
        rid = r.get("id")
        if not rid:
            raise PolicyError(f"rules[{i}] has no `id`; a rule nobody can name "
                              f"cannot be cited in a record or turned off")
        if rid in seen:
            raise PolicyError(f"duplicate rule id {rid!r}")
        seen.add(rid)
        if not r.get("why"):
            raise PolicyError(
                f"{rid}: no `why`. A piloting rule is a CLAIM about how the deck "
                f"is flown; one with no reason cannot be argued with, and this "
                f"file exists to make the manual falsifiable")
        verbs = [v for v in VERBS if v in r]
        if len(verbs) != 1:
            raise PolicyError(f"{rid}: exactly one of {list(VERBS)} is required")
        if "forge" in r:
            knobs = r["forge"]
            if not isinstance(knobs, dict) or not knobs:
                raise PolicyError(f"{rid}: `forge` is a mapping of the AiProps keys this rule explains")
            for k, v in knobs.items():
                if (doc.get("forge") or {}).get(k, object()) != v:
                    raise PolicyError(f"{rid}: forge.{k} = {v!r} is not what the document's `forge` "
                                      f"section sets — a rule explains a knob, it does not set a second one")
            continue
        chan = (r.get("when") or {}).get("channel")
        if chan not in CHANNELS:
            raise PolicyError(
                f"{rid}: when.channel {chan!r} is not a channel the simulator "
                f"computes; pick one of {list(CHANNELS)}")
        for counter, value in (r.get("hold_until") or {}).items():
            if counter not in COUNTERS:
                raise PolicyError(f"{rid}: hold_until.{counter} is not countable; "
                                  f"pick one of {list(COUNTERS)}")
            if not isinstance(value, int) or value < 0:
                raise PolicyError(f"{rid}: hold_until.{counter} must be a "
                                  f"non-negative integer, not {value!r}")
    # The Forge half lives in `forge_ai`, which the goldfish never reads —
    # see that module's docstring for why the split is load-bearing.
    from manamap.pilot import forge_ai

    forge_ai.validate_forge(doc)
    return doc


def hold_thresholds(doc):
    """`{channel: {counter: n}}` — the thresholds the turn loop enforces.

    Flattened here rather than in the turn loop so the simulator reads a plain
    mapping and never parses a policy: one home for the shape.
    """
    out = {}
    for r in (doc.get("rules") or []):
        if "hold_until" not in r:
            continue                      # a `forge` rule: the goldfish never reads it
        chan = r["when"]["channel"]
        for counter, n in (r.get("hold_until") or {}).items():
            prev = out.setdefault(chan, {}).get(counter)
            # The STRICTER threshold wins, so two rules can never make the model
            # play looser than either of them asked for.
            out[chan][counter] = n if prev is None else max(prev, n)
    return out


def render(doc):
    """The policy as the pilot wrote it, for a record or a report."""
    if not doc:
        return ["no policy — the simulator uses its own built-in heuristics"]
    lines = []
    for r in doc.get("rules") or []:
        if "forge" in r:
            what = ", ".join(f"{k}={v}" for k, v in sorted(r["forge"].items()))
            lines.append(f"{r['id']}: Forge AI knobs {what}")
        else:
            hold = r.get("hold_until") or {}
            what = ", ".join(f"{k} >= {v}" for k, v in sorted(hold.items()))
            lines.append(f"{r['id']}: hold a {r['when']['channel']} until {what}")
        lines.append(f"    {r['why']}")
    return lines


def fingerprint(doc):
    """A stamp for the record: which policy, and which rules were live.

    Over the RULES rather than the file bytes, so reformatting the JSON or editing a
    `why` does not stale a figure it cannot have changed — the opposite call from
    `model_version`, which is coarse on purpose because a comment there sits beside code
    that runs. Here the prose genuinely is prose and the thresholds genuinely are the
    model, so the stamp follows the thresholds.

    `rules` lists the ids so a reader can see WHICH rule was on without opening the file,
    and a rule turned off changes the sha.
    """
    rules = doc.get("rules") or []
    live = [{"id": r.get("id"),
             "channel": (r.get("when") or {}).get("channel"),
             **{v: r[v] for v in VERBS if v in r}}
            for r in rules]
    blob = json.dumps(live, sort_keys=True, separators=(",", ":"))
    return {"sha": hashlib.sha256(blob.encode()).hexdigest()[:12],
            "rules": sorted(r["id"] for r in rules if r.get("id"))}
