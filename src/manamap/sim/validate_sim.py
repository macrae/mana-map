"""Simulation: form-check a run record, and prove its analysis is what its logs say.

A run record is tier ◆ SEEDED (runs made with `-s`, replayable byte for byte) or
◆ SAMPLED (the first runs, made before the seed flag was known). Either way the parser
over the logs is deterministic, so the one thing a validator CAN prove is that the tracked
`analysis` block is exactly what `sim.parse` derives from the logs — where the logs exist.
Where they do not (a fresh clone; they are gitignored), the record is form-checked: the
keys a consumer relies on, counts that agree with each other, every seat's decklist sha
present, and the SEEDED/SAMPLED assumption that matches whether seeds are recorded. A record that cannot be re-derived is still evidence
of a run; it just cannot be re-proven here, and the OK line says so.
"""

import re

from manamap.config import SIM_DIR
from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.sim import parse as sim_parse
from manamap.sim.forge import _seat_label, record_commanders

REQUIRED = {"run_id", "slug", "at", "engine", "seats", "games_requested", "games_completed",
            "summary", "outcomes", "analysis", "assumptions"}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _card_overrides_errors(rec):
    """The harness a record CLAIMS, checked for internal sense.

    `card_overrides` is deliberately NOT in `REQUIRED`: a plain run has no such key and
    every record written before 2026-09-28 lacks it, so demanding it would redden history
    to no purpose. What is checked is that a record carrying one is coherent.

    The cross-check against the live engine is NOT done here and that is deliberate too.
    A validator runs on a checkout that may have no Forge install at all, and a record is
    a statement about the machine that MADE it — asking this machine to confirm it would
    fail on CI and on any clone, which is a check firing on correct data. `forge.run`
    does the engine comparison at the only moment it is meaningful: before the games.
    """
    got = rec.get("card_overrides")
    if got is None:
        return []
    errors = []
    if not isinstance(got, dict):
        return [f"card_overrides is {type(got).__name__}, not an object"]
    for key in ("sha", "n", "cards"):
        if key not in got:
            errors.append(f"card_overrides has no {key!r} — a harness stamp that cannot "
                          f"be compared is not provenance")
    sha, n, cards = got.get("sha"), got.get("n"), got.get("cards")
    if sha is not None and not re.fullmatch(r"[0-9a-f]{12}", str(sha)):
        errors.append(f"card_overrides.sha {sha!r} is not a 12-hex digest")
    if isinstance(cards, list) and isinstance(n, int) and len(cards) != n:
        errors.append(f"card_overrides lists {len(cards)} card(s) and claims n={n}")
    # A record stamped while the engine DISAGREED describes neither arm. `forge.run`
    # refuses to start in that state, so a record carrying it predates the guard or was
    # written by hand.
    if got.get("agrees") is False:
        errors.append(
            "card_overrides.agrees is false — this run was made while the engine carried "
            "neither Forge's own card scripts nor the ones the repo declares, so its rate "
            "describes neither arm. `simulate` refuses to start in that state now.")
    return errors


def _telemetry_errors(rec):
    """The log formatter a record CLAIMS, checked for internal sense — the
    `card_overrides` rule: absent or None is a plain run and fine; a block must be
    comparable (a 12-hex class sha, the class entry, a line count that is a count)."""
    got = rec.get("telemetry")
    if got is None:
        return []
    if not isinstance(got, dict):
        return [f"telemetry is {type(got).__name__}, not an object"]
    errors = []
    for key in ("sha", "class"):
        if key not in got:
            errors.append(f"telemetry has no {key!r} — a formatter stamp that cannot be "
                          f"compared is not provenance")
    sha = got.get("sha")
    if sha is not None and not re.fullmatch(r"[0-9a-f]{12}", str(sha)):
        errors.append(f"telemetry.sha {sha!r} is not a 12-hex digest")
    lines = got.get("lines")
    if lines is not None and (not isinstance(lines, int) or lines < 0):
        errors.append(f"telemetry.lines {lines!r} is not a count")
    if lines == 0:
        errors.append("telemetry.lines is 0 — a run stamped as patched whose logs carry no "
                      "new zone line was not played under the patched formatter")
    return errors


def _cast_proofs_errors(rec):
    """The gate's block on a branch record (2026-10-01): absent or None on a deck seat
    and on every earlier record; where present, the four lists and the rule that an
    unproven add only ran under --anyway."""
    cp = rec.get("cast_proofs")
    if cp is None:
        return []
    want = {"proven", "held", "late", "unproven", "as_of", "harness_matches", "harness", "anyway"}
    if not isinstance(cp, dict) or not want <= set(cp):
        return ["cast_proofs: wrong shape"]
    errors = []
    for k in ("proven", "held", "late", "unproven"):
        if not isinstance(cp[k], list) or any(not isinstance(x, str) for x in cp[k]):
            errors.append(f"cast_proofs.{k}: not a list of card names")
    if (cp["held"] or cp["late"] or cp["unproven"]) and not cp["anyway"]:
        errors.append("cast_proofs: the gate refuses an unproven add unless --anyway, and this record says it ran")
    return errors


def validate(rec, slug, logs_text=None):
    errors = []
    errors += _card_overrides_errors(rec)
    errors += _telemetry_errors(rec)
    missing = REQUIRED - set(rec)
    if missing:
        return [f"missing keys {sorted(missing)}"]
    if rec["slug"] != slug:
        errors.append(f"slug {rec['slug']!r} != {slug!r}")
    seats = rec.get("seats") or []
    if not seats or seats[0].get("slug") != slug:
        errors.append("seats[0] must be this deck")
    for i, s in enumerate(seats):
        if not _SHA.match(str(s.get("decklist_sha256") or "")):
            errors.append(f"seats[{i}] ({s.get('slug')}): decklist_sha256 is not a sha256")
    n = rec["games_completed"]
    if n != len(rec["outcomes"]):
        errors.append(f"games_completed {n} != {len(rec['outcomes'])} outcomes")
    if rec["analysis"].get("games") != n:
        errors.append(f"analysis.games {rec['analysis'].get('games')} != games_completed {n}")
    wins = rec["summary"].get("wins") or {}
    # EVERY GAME IS WON, DRAWN, OR UNFINISHED. The third term is new: a game the
    # `-c` clock stopped has no winner, and is excluded from the win rate rather
    # than awarded to a survivor.
    #
    # This invariant is why the truncation bug survived so long. It held
    # perfectly while the parser handed every clock-out to the highest-numbered
    # surviving seat — the books balanced because the wins were REASSIGNED, not
    # lost. An accounting check cannot see a misattribution that conserves the
    # total, which is worth remembering the next time one of these reads green.
    trunc = rec["summary"].get("truncated", 0)
    if sum(wins.values()) + rec["summary"].get("draws", 0) + trunc != n:
        errors.append(
            f"summary.wins {wins} + draws {rec['summary'].get('draws')} + "
            f"truncated {trunc} != {n}")
    decided = rec["summary"].get("decided")
    draws = rec["summary"].get("draws", 0)
    # A decided game has a WINNER: neither a clock-out nor a draw. The check
    # used to accept draws in the count, which is how `summary` and
    # `analysis.seats` disagreed on one denominator for six records.
    if decided is not None and decided != n - trunc - draws:
        errors.append(f"summary.decided {decided} != games_completed {n} - "
                      f"truncated {trunc} - draws {draws}")
    # `summary.wins` is keyed by slug (`zur-enchantress@drain-v2`) and
    # `analysis.seats` by Forge's meta name (`zur-enchantress-drain-v2`), so a
    # BRANCH record failed this comparison on the spelling of its own name.
    from manamap.sim.forge import deck_meta_name
    a_wins = {k: v.get("wins") for k, v in (rec["analysis"].get("seats") or {}).items()}
    if any(a_wins.get(deck_meta_name(k)) != v for k, v in wins.items()):
        errors.append(f"analysis per-seat wins {a_wins} disagree with summary.wins {wins}")
    for s, d in (rec["analysis"].get("seats") or {}).items():
        lo, hi = (d.get("win_rate_ci95") or [None, None])
        if lo is not None and not (0 <= lo <= (d.get("win_rate") or 0) <= hi <= 1):
            errors.append(f"analysis.seats[{s}]: win_rate {d.get('win_rate')} outside ci95 [{lo}, {hi}]")
    seeded = bool(rec.get("seeds"))
    want = "SEEDED" if seeded else "SAMPLED"
    if not any(want in str(x) for x in rec["assumptions"]):
        errors.append(f"assumptions must state {want} — the record "
                      f"{'carries seeds' if seeded else 'has no seeds'}")
    if seeded and len(rec["seeds"]) != rec.get("jobs"):
        errors.append(f"{len(rec['seeds'])} seeds for {rec.get('jobs')} jobs")
    # ENGINE CASTS, checked only where PRESENT. A record written before the
    # block existed is "not measured", not wrong -- the `record_commanders`
    # precedent -- so an absent key is never an error here.
    ec = rec.get("engine_casts")
    if ec is not None:
        if not isinstance(ec, dict) or set(ec) != {"seat", "games", "turns", "kept_hand_mean", "by_card"}:
            errors.append("engine_casts: wrong shape")
        else:
            if ec["games"] != n:
                errors.append(f"engine_casts.games {ec['games']} != games_completed {n}")
            if n and not (isinstance(ec["turns"], int) and ec["turns"] > 0):
                errors.append(f"engine_casts.turns {ec['turns']!r} is not a positive count")
            # The three counted keys always; the three MEASURED hand keys only under the
            # telemetry patch (`parse.engine_casts`), and then together or not at all.
            base, hand = {"cast", "activated", "discarded"}, {"in_hand_games", "turns_in_hand", "castable_uncast", "turns_on_battlefield"}
            bad = [k for k, v in ec["by_card"].items()
                   if not (set(v) == base or base < set(v) <= base | hand)
                   or any(not isinstance(x, int) or x < 0 for x in v.values())]
            if bad:
                errors.append(f"engine_casts.by_card: malformed rows for {bad[:3]}")
    # THE CAST PROOFS, checked only where PRESENT and not None — the same rule.
    errors += _cast_proofs_errors(rec)
    # THE BOARD SERIES, checked only where PRESENT — the `engine_casts` rule.
    bser = rec.get("board_series")
    if bser is not None:
        from manamap.sim import board_series as _bs
        errors += _bs.validate(bser, n)
    if logs_text:
        label = _seat_label([s["forge_name"] for s in seats])
        from manamap.sim.forge import cmc_map, keyword_map, played_doc
        doc = played_doc(slug, rec)   # the list these games PLAYED, not today's
        facts, derived = sim_parse.analyze_logs(logs_text, label, record_commanders(rec),
                                                cmc_map(slug, doc), keyword_map(slug, doc))
        if derived != rec["analysis"]:
            keys = [k for k in set(derived) | set(rec["analysis"]) if derived.get(k) != rec["analysis"].get(k)]
            errors.append(f"analysis does not match what the logs derive (differs at {sorted(keys)}) — "
                          f"`simulate {slug} --analyze {rec['run_id']}` rewrites it")
        if ec is not None:
            from manamap.sim.forge import deck_meta_name
            if sim_parse.engine_casts(facts, label, deck_meta_name(slug)) != ec:
                errors.append(f"engine_casts does not match what the logs derive — "
                              f"`simulate {slug} --analyze {rec['run_id']}` rewrites it")
        if bser is not None:
            from manamap.sim import board_series as _bs
            from manamap.sim.forge import deck_meta_name
            if _bs.from_logs(logs_text, record_commanders(rec), deck_meta_name(slug)) != bser:
                errors.append(f"board_series does not match what the logs derive — "
                              f"`simulate {slug} --analyze {rec['run_id']}` rewrites it")
    return errors


def main(args):
    slug = args.slug
    # A branch keeps its runs beside its own list, never in the deck's `sim/`.
    from manamap.sim.forge import _out_dir
    base = _out_dir(slug)
    paths = sorted(base.glob("*.json")) if base.is_dir() else []
    if not paths:
        print(f"OK   {slug} — no simulation runs")
        return
    errors, reproven = [], 0
    for p in paths:
        rec = load_json(p)
        logs = sorted((base / "logs" / p.stem).glob("part-*.log"))
        texts = [l.read_text(encoding="utf-8", errors="replace") for l in logs] or None
        errs = validate(rec, slug, texts)
        errors += [f"{p.name}: {e}" for e in errs]
        reproven += bool(texts)
    report_errors(f"{slug} simulation runs", errors,
                  ok_line=f"OK   {slug} — {len(paths)} run(s), {reproven} re-derived from logs"
                          f"{'' if reproven == len(paths) else ' (the rest form-checked; logs are gitignored)'}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-sim <slug>`.")
