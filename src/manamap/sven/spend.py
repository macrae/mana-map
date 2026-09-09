"""What Sven has cost, and the ceiling he cannot spend past.

THE ACCOUNT HAS NO AUTO-RELOAD. When the balance is gone it is gone, and the
answer to "where did it go" has to be better than "the console says so, three
days later". So every turn's usage is recorded the moment it comes back from the
API, and a running total is checked BEFORE the next call rather than after.

TWO THINGS THIS IS NOT.

It is not a bill. These are locally-computed estimates from a price table
written down on a date, and prices change without asking this file. The
authority on what was actually charged is console.anthropic.com, always. Every
figure here is labelled `estimated` for the same reason a measured rate in this
repo carries its interval — a number that looks authoritative and is not is
worse than no number.

It is not a substitute for the console's own limits. A local ceiling protects
against a loop in THIS code; it cannot protect against anything that does not
route through it. Set a spend limit in the console as well.

WHY A CEILING AT ALL, when a Sven turn is a fraction of a cent: the risk was
never the typical turn. It is the atypical one — a tool result that balloons, an
escalation that fires every time, a loop that retries. Those are exactly the
failures that happen while nobody is watching, which is when a ceiling is the
only thing in the room.
"""

import json
import os
import time
from pathlib import Path

#: USD per million tokens, checked 2026-09-08. A price table in source is stale
#: the moment a vendor changes one, so this is deliberately easy to find and the
#: reports it feeds all say `estimated`. If these drift the estimate drifts with
#: them, and the console remains the truth.
PRICES = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "claude-sonnet-5": {"in": 3.00, "out": 15.00},
    "claude-opus-5": {"in": 15.00, "out": 75.00},
}

#: Machine-local, outside the repo. Under `~/.mana-map/` beside `forge/`, which
#: `config.FORGE_HOME` already uses — a spend log is not a project artifact and
#: has no business being near anything git tracks.
LEDGER = Path(os.environ.get(
    "MANAMAP_SVEN_LEDGER", Path.home() / ".mana-map" / "sven" / "spend.jsonl"))

#: Refuse to start a turn once the ledger passes this. Deliberately small
#: relative to a real balance: the point is to catch a runaway early, and a
#: ceiling set near the balance would only fire once the damage was done.
#: Raise it with `MANAMAP_SVEN_BUDGET_USD`, or 0 to disable.
DEFAULT_BUDGET_USD = 10.0


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a call, never after. An overspend you learn about
    afterwards is one you already paid for."""


def cost_of(model, usage):
    """Estimated USD for one turn, or None for a model with no price on file.

    None rather than 0.0, and the distinction is the whole habit of this repo: a
    zero is a measurement, and a reader cannot tell it from one. An unpriced
    model must read as "not counted", never as "free".
    """
    price = PRICES.get(model)
    if not price or not usage:
        return None
    return round((usage.get("in", 0) / 1e6) * price["in"]
                 + (usage.get("out", 0) / 1e6) * price["out"], 6)


def record(model, usage, question=None):
    """Append one turn to the ledger. Returns its estimated cost, or None.

    Best-effort on the write: a ledger failure must not fail a turn the pilot
    has already paid for. It is logged, not swallowed silently.
    """
    cost = cost_of(model, usage)
    row = {"at": time.time(), "model": model, "usage": usage,
           "estimated_usd": cost,
           "question": (question or "")[:120]}
    try:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as exc:                         # noqa: BLE001
        from manamap import console
        console.err(f"  sven: could not write the spend ledger ({exc})")
    return cost


def total(since=None):
    """`{turns, estimated_usd, unpriced, by_model}` over the whole ledger.

    `unpriced` is counted separately and never folded into the total, so a turn
    on a model with no price on file cannot quietly read as a free one.
    """
    out = {"turns": 0, "estimated_usd": 0.0, "unpriced": 0, "by_model": {}}
    try:
        lines = LEDGER.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if since and row.get("at", 0) < since:
            continue
        out["turns"] += 1
        cost = row.get("estimated_usd")
        model = row.get("model", "?")
        bucket = out["by_model"].setdefault(model, {"turns": 0, "estimated_usd": 0.0})
        bucket["turns"] += 1
        if cost is None:
            out["unpriced"] += 1
            continue
        out["estimated_usd"] = round(out["estimated_usd"] + cost, 6)
        bucket["estimated_usd"] = round(bucket["estimated_usd"] + cost, 6)
    return out


def budget():
    """The ceiling in force. 0 or negative disables it."""
    raw = os.environ.get("MANAMAP_SVEN_BUDGET_USD")
    if raw is None:
        return DEFAULT_BUDGET_USD
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_BUDGET_USD


def check():
    """Raise if the ledger has already passed the ceiling. Call before a turn."""
    cap = budget()
    if cap <= 0:
        return
    spent = total()["estimated_usd"]
    if spent >= cap:
        raise BudgetExceeded(
            f"Sven has spent an estimated ${spent:.2f}, which is at or past the "
            f"${cap:.2f} ceiling.\n"
            f"  This is a LOCAL estimate — console.anthropic.com is the truth.\n"
            f"  Raise it:   MANAMAP_SVEN_BUDGET_USD=25 mm ask \"...\"\n"
            f"  Or review:  mm ask --spend")


def report():
    """The ledger, rendered. Says `estimated` everywhere, on purpose."""
    got = total()
    lines = [f"SVEN SPEND — {got['turns']} turn(s), "
             f"estimated ${got['estimated_usd']:.4f}",
             f"  ceiling ${budget():.2f}   ledger {LEDGER}"]
    for model, bucket in sorted(got["by_model"].items()):
        lines.append(f"  {model:32s}{bucket['turns']:>5} turn(s)  "
                     f"${bucket['estimated_usd']:.4f}")
    if got["unpriced"]:
        lines.append(f"  {got['unpriced']} turn(s) on a model with no price on "
                     f"file — counted, not costed")
    lines.append("")
    lines.append("  ESTIMATED, from a price table checked 2026-09-08. The "
                 "console is the authority on")
    lines.append("  what was actually charged, and a local ceiling cannot "
                 "protect against anything")
    lines.append("  that does not route through this code — set a limit there "
                 "as well.")
    return "\n".join(lines)
