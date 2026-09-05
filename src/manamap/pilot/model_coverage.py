"""Pilot: what the goldfish CANNOT see in this deck — BEFORE you run it.

THE MOST EXPENSIVE FAILURE ON THIS BENCH IS A MODEL THAT CANNOT READ THE DECK,
and every instance of it was discovered the same way: after the games.

  * Edgar Markov's EMINENCE mints a token on every other Vampire cast. It was
    not modelled at all, understating bodies at turn ten by 50%, and the deck's
    whole axis was invisible. `deck-audit` had described it in prose the entire
    time.
  * The token doublers doubled Treasures and nothing else, though the code's own
    comment said "Procession-style".
  * `land_colors` read every fetchland as producing NOTHING, so `mana-fit`
    scored a strict improvement as a five-colour regression.

Each cost a full re-run, and in the eminence case a 400-game Forge arm.

Nothing warned first, and the reason is structural: the `*_not_modelled` lists
(`draw_not_modelled`, `treasure_sources_not_modelled`,
`combat_effects_not_modelled`) are only populated for a channel the deck has
ALREADY OPTED INTO. A deck that models nothing reports nothing missing. Measured
across the fleet: SEVEN OF ELEVEN decks set no model flag at all, and every one
of them is silently scored by a model that cannot see combat, draw, sacrifice or
Treasures.

So this command answers the question in the other direction — not "what did the
channel miss" but "what would this deck need, and is it switched on".

THREE STATES PER CARD, and the middle one is the point:

  seen       a channel reads it, and that channel is ON
  DARK       a channel would read it, and that channel is OFF  <- the trap
  invisible  no channel reads it at all; the model knows only its mana cost

`invisible` is not a defect on its own — a removal spell in a resource model is
correctly invisible. A large DARK count is the thing that invalidates a run.
"""

import json

from manamap.pilot.common import deck_file, expand_copies, load_deck_cards


#: channel -> the `goldfish_targets.json` flag that switches it on. None means
#: the channel is always read, so a card it sees is never DARK.
CHANNELS = {
    "mana": None,
    "bodies": None,
    "tutor": None,
    "target": None,
    "treasure": "model_treasures",
    "combat": "model_combat",
    "draw": "model_draw",
    "sacrifice": "model_sacrifice",
    "drain": "model_drain",
    "attack_enabler": "model_commander_attack_tutor",
}

#: `model_colors` defaults to TRUE where the other four default to False, so it
#: is not a channel in the same sense — a deck cannot forget to turn it on.
DEFAULT_ON = ("model_colors",)


#: The trigger values `goldfish`'s turn loop actually branches on. Anything
#: else — the "unmodelled" sentinel above all — produces no Treasure whatever
#: the flag says, so it cannot be DARK on this channel. Kept beside CHANNELS
#: rather than imported from goldfish so that adding a trigger there without
#: teaching this module about it fails a test instead of going quiet.
_MODELLED_TREASURE_TRIGGERS = frozenset({"upkeep", "landfall", "etb", "cast"})

def _nonzero(profile, keys):
    return any((profile or {}).get(k) for k in keys)


def channels_for(profile):
    """Every channel this card's text would feed, flags ignored."""
    found = set()
    if profile["is_land"] or profile["produces"] or profile["reduces"] \
            or profile["scales_with_colors"]:
        found.add("mana")
    if profile["creature_bodies"] or profile["bodies"]:
        found.add("bodies")
    if profile["tutor"]:
        found.add("tutor")
    # ONLY A TRIGGER THE TURN LOOP ACTS ON. `treasure_trigger` is a string, and
    # one of its values is the sentinel "unmodelled" — which is TRUTHY, so a
    # card the parser explicitly gave up on was counted as feeding this channel.
    # The consequence is the opposite of the one this module exists to prevent:
    # it told the pilot to set `model_treasures` on cards that produce nothing
    # either way, and inflated the DARK count — the figure this file calls "the
    # thing that invalidates a run".
    #
    # Measured across the fleet on 2026-09-04: 21 cards in 7 decks, including
    # BOTH of zur-enchantress's, where flipping the flag returned byte-identical
    # figures and a hoard of 0.0 at every turn. Goldspan Dragon and Old Gnawbone
    # are in the same state on ur-dragon, whose nonzero hoard comes from its
    # OTHER treasure sources entirely.
    #
    # Such a card is INVISIBLE — no channel reads it — not DARK.
    if profile["treasure_trigger"] in _MODELLED_TREASURE_TRIGGERS \
            or profile["treasure_bonus"] or profile["treasure_doubler"] \
            or profile["token_doubler"]:
        found.add("treasure")
    if _nonzero(profile.get("combat"), (
            "etb_damage_self_power", "etb_damage_count", "etb_damage_fixed",
            "etb_life_loss_fixed", "token_created_life_loss", "etb_token_bodies",
            "etb_copy", "attack_treasure", "power", "cast_token_bodies")):
        found.add("combat")
    if _nonzero(profile.get("draw"), (
            "etb_draw", "spell_draw", "recurring_draw", "arrival_draw")):
        found.add("draw")
    if profile["sac_outlet"] or _nonzero(profile.get("death"), (
            "death_drain", "death_draw", "death_treasure")):
        found.add("sacrifice")
    # Named keys, not the truthiness of the dict — `drain` carries an
    # "unmodelled" sentinel like `draw` does, and reading the dict as a whole
    # would repeat the treasure bug fixed in dfe72f9 one channel over.
    if profile.get("attack_enabler"):
        found.add("attack_enabler")
    if _nonzero(profile.get("drain"), (
            "payoff_equal", "payoff_fixed", "gain_recurring",
            "gain_per_enchantment", "gain_per_creature",
            "drain_recurring", "drain_per_enchantment", "lifelink")):
        found.add("drain")
    return found


#: What each casting loop in `goldfish.simulate_once`'s main phase selects on.
#: MIRRORS the loops; it does not replace them, and the fleet assertion in
#: `tests/test_pilot_model_coverage.py` is what keeps the mirror honest.
#:
#: This exists because a card can be READ correctly and never PLAYED. Every
#: loop selects on a channel — draws, ramps, makes Treasure, has a body — and a
#: card matching none of them sits in hand for ten turns while its profile says
#: exactly what it would have done. `goldfish.py` already carries a comment
#: about patching that once for damage doublers ("read correctly and never
#: cast"); the fleet sweep on 2026-09-04 found two more classes.
def never_cast(profile, flags):
    """True when no casting loop would ever select this card.

    NOT a defect on its own: a counterspell should never be cast in a goldfish,
    and most of the 26% of the fleet this matches is exactly that. The defect is
    a card that is never cast AND feeds a channel that is switched ON — the
    model understands it, was told to look, and never puts it on the table.
    """
    if profile["is_land"] or profile["bodies"] > 0 or profile["produces"] > 0 \
            or profile["tutor"] or profile["reduces"]:
        return False
    cb = profile.get("combat") or {}
    if flags.get("model_combat"):
        if any((cb.get("etb_damage_self_power"), cb.get("etb_damage_count"),
                cb.get("etb_damage_fixed"), cb.get("etb_token_bodies"),
                cb.get("etb_copy"), (cb.get("team_damage_multiplier") or 0) > 1)):
            return False
        if cb.get("extra_combat_cost") is not None or cb.get("extra_combat_free"):
            return False
    if flags.get("model_treasures") and (profile["treasure_doubler"]
                                         or profile["treasure_bonus"]):
        return False
    if flags.get("model_draw") and _nonzero(profile.get("draw"), (
            "spell_draw", "etb_draw", "recurring_draw", "arrival_draw")):
        return False
    if flags.get("model_drain") and (_nonzero(profile.get("drain"), (
            "payoff_equal", "payoff_fixed", "gain_recurring",
            "gain_per_enchantment", "gain_per_creature",
            "drain_recurring", "drain_per_enchantment"))
            or (profile.get("drain") or {}).get("lifelink")):
        return False
    if flags.get("model_sacrifice") and profile.get("sac_outlet"):
        return False
    # DEATH ENGINES. Added when `model_deaths` shipped and NOT at the same time
    # as goldfish's own casting predicate — the fleet test caught the drift
    # within the hour, which is the whole reason it exists. A mirror that is
    # only checked by hand is a mirror that is wrong.
    if flags.get("model_deaths") and _nonzero(profile.get("death"), (
            "death_drain", "gain_on_opponent_death")):
        return False
    # An attack enabler is only meaningful where a commander attack trigger is
    # declared; there it is the card that lets the engine start at all.
    if flags.get("model_commander_attack_tutor") and profile.get("attack_enabler"):
        return False
    return True


def silent_losses(slug, branch=None):
    """`[(name, [channels])]` — cards that feed an ACTIVE channel and are never
    cast. This list must be EMPTY; anything in it is a measured effect the model
    computes and then never gets the chance to apply."""
    import json as _json

    from manamap.pilot import goldfish
    from manamap.pilot.common import deck_dir

    path = deck_dir(slug, branch) / "goldfish_targets.json"
    if not path.exists():
        return []
    flags = _json.loads(path.read_text(encoding="utf-8"))
    active = {ch for ch, flag in CHANNELS.items()
              if flag is None or flags.get(flag)}
    out = []
    for card in load_deck_cards(slug, branch).get("cards", []):
        p = goldfish.classify(card)
        if not never_cast(p, flags):
            continue
        feeds = channels_for(p) & active
        if feeds:
            out.append((card["name"], sorted(feeds)))
    return out


def analyze(slug, branch=None):
    """`{flags, cards, counts}` — what this deck needs and what is switched on."""
    from manamap.pilot import goldfish

    doc = load_deck_cards(slug, branch=branch)
    cards = expand_copies(doc["cards"] if isinstance(doc, dict) else doc)
    # `deck_file`, NOT `deck_dir(slug, branch)` — READS FALL BACK. A branch has
    # its own measurements but not its own AUTHORED files, and `goldfish.run`
    # resolves the declaration exactly this way. Reading the branch directory
    # directly reported ur-dragon@landbase-v1 as 30 cards dark when the deck's
    # declaration (which the simulation actually used) makes it 2 — a coverage
    # report that disagrees with the model it reports on is worse than none.
    targets_doc = {}
    path = deck_file(slug, "goldfish_targets.json", branch)
    if path.exists():
        targets_doc = json.loads(path.read_text(encoding="utf-8")) or {}

    flags = {flag: bool(targets_doc.get(flag))
             for flag in sorted({f for f in CHANNELS.values() if f})}
    named = {name
             for target in (targets_doc.get("targets") or [])
             for need in (target.get("need") or [])
             for name in (need.get("any_of") or [])}

    seen_names, rows = set(), []
    for card in cards:
        if card["name"] in seen_names:
            continue
        seen_names.add(card["name"])
        possible = channels_for(goldfish.classify(card))
        if card["name"] in named:
            possible.add("target")
        active = {c for c in possible
                  if CHANNELS[c] is None or flags.get(CHANNELS[c])}
        # DARK IS PER-CHANNEL, NOT PER-CARD, and getting this wrong hid the
        # whole problem on the first pass: nearly every creature feeds `bodies`,
        # which is always on, so "seen if anything is active" reported gishath —
        # a deck that opts into NOTHING — as 0 dark. A card is dark when
        # ANYTHING about it is not being read, whatever else is.
        rows.append({
            "name": card["name"],
            "possible": sorted(possible),
            "active": sorted(active),
            "dark_channels": sorted(possible - active),
            "state": ("dark" if possible - active
                      else "seen" if active else "invisible"),
        })

    rows.sort(key=lambda r: (r["state"] != "dark", r["name"]))
    counts = {state: sum(1 for r in rows if r["state"] == state)
              for state in ("seen", "dark", "invisible")}
    # Which OFF channel is costing the most cards — the thing to switch on first.
    by_channel = {}
    for row in rows:
        if row["state"] != "dark":
            continue
        for channel in row["possible"]:
            if CHANNELS[channel] and not flags.get(CHANNELS[channel]):
                by_channel.setdefault(channel, []).append(row["name"])
    return {"slug": slug, "branch": branch, "flags": flags,
            "cards": rows, "counts": counts, "dark_by_channel": by_channel,
            "distinct": len(rows)}


def headline(report):
    """One line, for a preflight. Empty string when nothing is dark."""
    dark = report["counts"]["dark"]
    if not dark:
        return ""
    worst = sorted(report["dark_by_channel"].items(),
                   key=lambda kv: (-len(kv[1]), kv[0]))
    parts = ", ".join(f"{len(names)} {channel}" for channel, names in worst)
    where = report["slug"] + (f" --branch {report['branch']}"
                              if report["branch"] else "")
    return (f"MODEL COVERAGE — {dark} of {report['distinct']} distinct cards are "
            f"DARK ({parts}). `model-coverage {where}` names them.")


def _print(report):
    print(f"MODEL COVERAGE — {report['slug']}"
          + (f" @ {report['branch']}" if report["branch"] else "")
          + f"   {report['distinct']} distinct card(s)")
    print("\n  CHANNELS")
    for flag, on in report["flags"].items():
        print(f"    {flag:22} {'ON' if on else 'off'}")
    for flag in DEFAULT_ON:
        print(f"    {flag:22} ON  (defaults on; not something a deck forgets)")

    c = report["counts"]
    print(f"\n  {'seen':10} {c['seen']:>4}   every channel it feeds is switched on")
    print(f"  {'DARK':10} {c['dark']:>4}   it feeds a channel that is OFF — whatever else is on")
    print(f"  {'invisible':10} {c['invisible']:>4}   no channel reads it; only its mana cost is known")

    if report["dark_by_channel"]:
        print("\n  DARK, BY CHANNEL — this is what would change if you switched it on")
        for channel, names in sorted(report["dark_by_channel"].items(),
                                     key=lambda kv: (-len(kv[1]), kv[0])):
            flag = CHANNELS[channel]
            print(f"\n    {channel}  ({len(names)} card(s)) — set \"{flag}\": true "
                  f"in goldfish_targets.json")
            for name in names[:12]:
                print(f"      {name}")
            if len(names) > 12:
                print(f"      … {len(names) - 12} more")
    else:
        print("\n  Nothing is dark: every channel this deck would use is switched on.")

    print("\n  A CARD IS NOT A DEFECT FOR BEING INVISIBLE. The goldfish models "
          "resource\n  development, so removal and interaction are correctly "
          "invisible to it. A large\n  DARK count is the figure that invalidates "
          "a run — the deck has the cards, and\n  the model was told not to look.")


def main(args):
    report = analyze(args.slug, branch=getattr(args, "branch", None))
    if getattr(args, "as_json", False):
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print(report)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot model-coverage <slug>`.")
