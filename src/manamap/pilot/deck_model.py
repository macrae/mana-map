"""ONE COMPOSED MODEL, and two presentations over it.

The dossier is what is on the desk; the handbook is what is in the cockpit. They
have different jobs and they read the same facts, and until now they read them
SEPARATELY: `deck_info.compose()` emitted roll-ups, and `deck-view.js` went
around it to fetch fifteen raw artifacts itself — `log.jsonl`,
`captains_log.json`, `stacks/*`, `sim/*`, `versions.json`, `log_causes.json` —
rendering each at whatever length it happened to be.

That is why four dossier panels are three to four thousand pixels tall. Nothing
decides what matters, so everything renders at full length; and a handbook built
the same way would be a SECOND reader of the same artifacts with its own opinion
about what matters, and the two would drift.

So: this module owns the vocabulary, `deck_info.compose()` builds the model, and
NO RENDERER READS A RAW ARTIFACT. What a surface may show, it shows because the
model marked it.

THE THREE THINGS EVERY FACT CARRIES
-----------------------------------
**tier** — the evidence contract, unchanged: ✓ verified / ◆ data / ★ coach.
**weight** — how much room it has earned. This is the new one, and it is what
fixes the dossier's size by construction rather than by a renderer remembering
to collapse something.
**definition** — what the number MEANS, travelling with the number. CLAUDE.md's
oldest standing rule: "Every figure carries its definition, in the report that
prints it." A number a reader has to look up elsewhere gets guessed at, and the
guesses go one way — a mean read as a rate, a clock read as a win rate.

ABSENT IS NOT ZERO, AND THE SHAPE ENFORCES IT. `figure()` refuses to build a
measurement out of nothing; a fact nobody measured is `absent(reason)`, which
renders as a stated absence. `0.0` is a measurement and a reader cannot tell it
from one.
"""

#: How much room a fact has earned on a surface. The renderers agree on this and
#: nothing else about layout.
#:
#:   headline  the two or three things a reader came for. Always visible, on
#:             every surface, and a panel may carry very few.
#:   body      the substance. Visible on the dossier; the section text in the
#:             handbook.
#:   detail    the evidence. Behind a disclosure on the dossier; an appendix in
#:             the handbook. Never inline.
#:
#: A renderer may show LESS than the weight allows (a phone may fold `body`); it
#: may never show more. `detail` inline is the failure this vocabulary exists to
#: stop.
WEIGHTS = ("headline", "body", "detail")

#: The evidence ladder, unchanged from the manual and the dossier.
TIERS = ("verified", "data", "coach")

#: Per-panel caps on how many headline facts may be claimed. Measured against the
#: fleet before being written down: the dossier's cover sheet carries four today
#: and reads well; the goldfish panel claims none and renders 3,833 px, which is
#: the failure in the other direction.
MAX_HEADLINES = 4


class ModelError(ValueError):
    """A model built wrong. Raised at compose time, never rendered."""


def figure(value, *, tier, definition, weight="body", unit=None, ci95=None,
           n=None, source=None):
    """One measured fact, with everything a reader needs to judge it.

    `definition` is REQUIRED and is not decoration. `net_change.METRICS` exists
    because a reader met "17.42" with no unit and read a mean as a rate; the
    registry there is asserted against its rendered rows in both directions.
    This makes the same rule structural for every figure on both surfaces.

    `ci95` is not optional in spirit: a rate without its interval is the repo's
    most-repeated sin, and two marginal intervals overlapping implies nothing at
    all. It is allowed to be None only because some figures are counts.
    """
    if tier not in TIERS:
        raise ModelError(f"tier {tier!r} not in {TIERS}")
    if weight not in WEIGHTS:
        raise ModelError(f"weight {weight!r} not in {WEIGHTS}")
    if value is None:
        raise ModelError(
            f"figure({definition!r}) has no value — use absent(reason) so the "
            f"surface can state the absence. A missing measurement must never "
            f"reach a reader as a number.")
    if not definition:
        raise ModelError("every figure carries its definition")
    out = {"value": value, "tier": tier, "weight": weight,
           "definition": definition}
    for k, v in (("unit", unit), ("ci95", ci95), ("n", n), ("source", source)):
        if v is not None:
            out[k] = v
    return out


def absent(reason, *, weight="body"):
    """A fact nobody measured, and WHY.

    Renders as a stated absence on both surfaces. The alternative — omitting the
    row, or worse filling it with 0.0 — is how a deck reads as complete because
    nothing said what was missing. `regen.BOOTSTRAP` exists because a deck was
    skipped forever in silence.
    """
    if not reason:
        raise ModelError("an absence must state its reason")
    return {"absent_because": reason, "weight": weight}


def is_absent(fact):
    return isinstance(fact, dict) and "absent_because" in fact


def block(title, facts, *, tier=None, source=None, definition=None):
    """A group of facts a surface renders together.

    The dossier makes it a panel; the handbook makes it a numbered subsection.
    Neither decides what goes in it.
    """
    if not isinstance(facts, dict):
        raise ModelError(f"block({title!r}) needs a dict of facts")
    heads = [k for k, v in facts.items()
             if not is_absent(v) and v.get("weight") == "headline"]
    if len(heads) > MAX_HEADLINES:
        raise ModelError(
            f"block({title!r}) claims {len(heads)} headline facts "
            f"({', '.join(heads)}) — the cap is {MAX_HEADLINES}. A panel where "
            f"everything is a headline has no headline, which is how the "
            f"goldfish panel came to be 3,833 pixels tall.")
    out = {"title": title, "facts": facts}
    for k, v in (("tier", tier), ("source", source), ("definition", definition)):
        if v is not None:
            out[k] = v
    return out


def visible(facts, weight):
    """The facts a surface may show at this weight or above.

    `visible(facts, "body")` gives headline + body; `visible(facts, "headline")`
    gives only the headline. A renderer calls this instead of deciding for
    itself, which is what keeps the two surfaces agreeing about what matters.
    """
    if weight not in WEIGHTS:
        raise ModelError(f"weight {weight!r} not in {WEIGHTS}")
    allowed = set(WEIGHTS[:WEIGHTS.index(weight) + 1])
    return {k: v for k, v in facts.items() if v.get("weight") in allowed}


def headline_of(blk):
    """The one line a reader gets if they read nothing else."""
    for k, v in (blk.get("facts") or {}).items():
        if not is_absent(v) and v.get("weight") == "headline":
            return k, v
    return None, None
