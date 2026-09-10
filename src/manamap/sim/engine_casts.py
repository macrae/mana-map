"""Did the AI ever play the deck's engine? Read from the record, at print time.

THE OTHER HALF OF THE PILOTING GATE. `pilot_quality` asks whether our seat
made its land drops like the rest of the table; this asks whether the cards
the deck's PLAN depends on were ever cast. On 2026-09-10 sharknado sat at
standard-v3 for 60 games and its seat cast Wheel of Fortune once and Windfall
never -- while DISCARDING Windfall three times and Faithless Looting six. The
0.132 that run produced was a floor on a deck the AI could not pilot, and the
only thing that said so was a cast count parsed by hand afterwards.

WHAT IS MEASURED AND WHAT IS MODELLED, kept apart on purpose:

  * `record["engine_casts"]` is MEASURED: per-card cast / activated / discarded
    for our seat, own turns, kept hand. Written by `forge.run`, re-derived by
    `--analyze`, proven by `validate-sim`.
  * "held and never cast" is MEASURED: a card discarded from hand was in hand.
    The log says so. It is a FLOOR on held -- a card drawn and never discarded
    leaves no trace at all.
  * expected natural draws is a MODEL floor: games x (kept hand + own turns) /
    library, natural draws only. Wheel of Misfortune alone resolved 17 times in
    that run, so real draws were far higher; the figure is labelled.
  * the ENGINE SET comes from an AUTHORED file (`goldfish_targets.json`, plus
    `engine.json` where it exists) and is therefore read HERE, at print time,
    and never stored in the record -- the rule `net_change` deleted the engine
    lift for. A scaffolded targets file is role axes, not components
    (`docs/gotchas-bench.md`, the RAMP-drawn lesson), so on a scaffold the
    set-level figure is None with the reason and the per-card table stands.

THE SHARP LINE IS PER CARD AND SET-INDEPENDENT: zero plays, and either held
at least `NEVER_CAST_DISCARDS` times or expected at least `NEVER_CAST_EXPECTED`
natural draws. The set-level ratio is context; the list is the finding.
"""

from manamap.pilot.common import deck_file, load_deck_cards, load_json

#: Below this many games the counts are one table's variance; the rates are
#: reported and the verdict withheld, exactly as `pilot_quality.MIN_GAMES`.
MIN_GAMES = 8
#: A card discarded this many times with zero plays was in hand and passed
#: over on at least that many occasions -- measured, not modelled.
NEVER_CAST_DISCARDS = 3
#: A card with zero plays whose expected natural draws over the run reach
#: this is one the seat almost certainly saw and never played. Stated, not
#: fitted: at 60 games and ~16 cards seen a game a singleton expects ~10.
NEVER_CAST_EXPECTED = 8.0
#: Target labels that are the MANA half of a declaration, not the engine.
_MANA_LABELS = ("ramp", "mana", "land")
_ENGINE_STAGES_EXCLUDED = ("mana", "protection")


def _front(name):
    return name.split(" // ")[0].strip()


def engine_set(slug, branch=None):
    """The cards the deck's plan depends on, from its own declarations.

    Returns `{cards, scaffolded, sources}` or None when nothing is declared.
    `cards` are deck names (the `A // B` form) present in the 99.
    """
    try:
        doc = load_deck_cards(slug, branch)
    except FileNotFoundError:
        return None
    deck_names = {c["name"] for c in doc["cards"]}
    targets = load_json(deck_file(slug, "goldfish_targets.json", branch), default=None)
    engine = load_json(deck_file(slug, "engine.json", branch), default=None)
    if not targets and not engine:
        return None
    picked, sources = set(), []
    if targets:
        sources.append("goldfish_targets.json")
        for t in targets.get("targets") or []:
            label = (t.get("label") or "").lower()
            if any(w in label for w in _MANA_LABELS):
                continue
            for group in t.get("need") or []:
                picked.update(n for n in (group.get("any_of") or []) if n in deck_names)
    if engine:
        sources.append("engine.json")
        for stage in engine.get("stages") or []:
            if stage.get("stage") in _ENGINE_STAGES_EXCLUDED:
                continue
            picked.update(n for n in (stage.get("cards") or []) if n in deck_names)
    return {"cards": sorted(picked), "scaffolded": bool((targets or {}).get("scaffolded")),
            "sources": sources}


def nonland_names(doc):
    """The names this reading is over: a LAND is played, never cast, so it can
    only ever read as 'never cast' -- Swamp was the first entry on the list
    until this existed."""
    return {c["name"] for c in doc["cards"]
            if "Land" not in (c.get("type_line") or "").split(" // ")[0]}


def from_record(rec, deck_names=None, engine=None):
    """The reading. `deck_names` (the 99, `A // B` form) and `engine`
    (`engine_set(...)`) are optional context; without them the table is over
    whatever the seat played or discarded, and the set-level figure is None.

    Returns None when the record carries no `engine_casts` block: NOT MEASURED,
    which a record made before 2026-09-10 is.
    """
    ec = rec.get("engine_casts")
    if not ec or not ec.get("games"):
        return None
    games, turns = ec["games"], ec["turns"]
    kept = ec.get("kept_hand_mean") or 7.0
    seats = rec.get("seats") or []
    n_cmd = len((seats[0].get("commander") or [None])) if seats else 1
    library = 100 - max(1, n_cmd)
    # Expected NATURAL draws of one singleton over the run: per game,
    # min(1, cards seen / library), cards seen = kept hand + own turns.
    seen_per_game = kept + (turns / games if games else 0)
    expected = round(games * min(1.0, seen_per_game / library), 1)

    # Cast lines carry the FRONT FACE ("cast Commit"); discard lines carry the
    # full name ("discards Commit // Memory (12)"). Merge on the face, or the
    # second row silently replaces the first and a card cast twelve times
    # reads as never cast.
    by_face = {}
    for name, row in ec["by_card"].items():
        acc = by_face.setdefault(_front(name), {"cast": 0, "activated": 0, "discarded": 0})
        for k in acc:
            acc[k] += row.get(k, 0)
    names = sorted(deck_names) if deck_names else sorted(by_face)
    rows = {}
    for name in names:
        row = by_face.get(_front(name), {"cast": 0, "activated": 0, "discarded": 0})
        plays = row["cast"] + row["activated"]
        rows[name] = {"cast": row["cast"], "activated": row["activated"],
                      "discarded": row["discarded"], "plays": plays,
                      "expected_natural_draws": expected}
    never = [n for n, r in rows.items()
             if r["plays"] == 0 and (r["discarded"] >= NEVER_CAST_DISCARDS
                                     or expected >= NEVER_CAST_EXPECTED)]
    never.sort(key=lambda n: (-rows[n]["discarded"], n))
    out = {"seat": ec["seat"], "games": games, "own_turns": turns,
           "expected_natural_draws_per_card": expected,
           "expected_is": "a MODEL floor: natural draws only, games x (kept hand + own turns) / library",
           "held_is": "a MEASURED floor: a discarded card was in hand; an undiscarded one leaves no trace",
           "never_cast": [{"card": n, **{k: rows[n][k] for k in ("cast", "activated", "discarded")}}
                          for n in never],
           "by_card": rows}
    if engine:
        eng_rows = [rows[n] for n in engine["cards"] if n in rows]
        out["engine"] = {"cards": engine["cards"], "sources": engine["sources"],
                         "scaffolded": engine["scaffolded"],
                         "never_cast": [n for n in never if n in set(engine["cards"])]}
        if engine["scaffolded"]:
            out["engine"]["played_share"] = None
            out["engine"]["why"] = ("goldfish_targets.json is a scaffold: role axes, not "
                                    "components, so a set-level share would grade the "
                                    "declaration's shape. Per-card table stands.")
        elif eng_rows:
            exp = sum(r["expected_natural_draws"] for r in eng_rows)
            out["engine"]["played_share"] = round(
                sum(r["plays"] for r in eng_rows) / exp, 3) if exp else None
    else:
        out["engine"] = None
    if games < MIN_GAMES:
        out["covered"] = None
        out["reading"] = (f"only {games} game(s) — too few to say whether the AI ever "
                          f"played the engine. The counts are reported; the verdict is withheld.")
        return out
    if out["engine"] is not None:
        bad = out["engine"]["never_cast"]
        out["covered"] = not bad
        out["reading"] = (
            "the AI played the deck's declared engine; a win rate from this run is about the deck."
            if not bad else
            f"THE AI NEVER CAST PART OF THE ENGINE: {', '.join(bad[:6])}"
            f"{' …' if len(bad) > 6 else ''}. A win rate from this run is a FLOOR on a "
            f"deck the AI could not pilot; read the observations, not the outcome.")
    else:
        out["covered"] = None
        out["reading"] = (
            f"no engine declaration to judge against; {len(never)} card(s) held and never cast"
            + (f": {', '.join(never[:6])}" if never else "") + ".")
    return out


def render(q):
    if not q:
        return []
    lines = [f"  ENGINE CASTS (did the AI play the deck's plan? {q['games']} games, "
             f"{q['own_turns']} own turns, a singleton expects ~{q['expected_natural_draws_per_card']} natural draws)"]
    if q["never_cast"]:
        lines.append("    held and never cast: " + ", ".join(
            f"{r['card']} (discarded x{r['discarded']})" for r in q["never_cast"][:8]))
    eng = q.get("engine")
    if eng:
        share = eng.get("played_share")
        lines.append(f"    engine set: {len(eng['cards'])} cards from {' + '.join(eng['sources'])}"
                     + (f", played {share:.0%} of expected natural draws" if share is not None
                        else " (scaffolded declaration — no set-level share)"))
    lines.append("    " + ("ENGINE PLAYED" if q["covered"] else "NOT ENOUGH GAMES"
                           if q["covered"] is None and q["games"] < MIN_GAMES else
                           "NO DECLARATION" if q["covered"] is None else "ENGINE NEVER CAST"))
    lines.append("    " + q["reading"])
    return lines
