"""Pilot: `scan-candidates` — mine the corpus for a deck's optimisation candidates along
the DIMENSIONS the pilot wants to juice, and label every row with why it matched.

Built for edgar-vampires/drain-v1 (2026-09-30). The pilot's brief for the deck is a
list of dimensions — life DRAIN as the engine, the GAIN that feeds it, big evasive
THREATS, free OUTLETS, a SWEEPER, and DRAW that refills a hand emptied by turn 7 — and
`card-search` can reach each one only through a different door: `--role wincon:drain`
here, an oracle regex there, and never the mechanical tags (`lifegain`, `death_trigger`,
`evasion_*`) or the printed keywords and power that decide "big and scary". This is
one pass over the corpus with a NAMED predicate set per dimension, so the shortlist a
staging `--why` cites can point at a row and the row says which predicate found it.

THREE RULES, each one a lesson this bench has already paid for:

* **It retrieves and labels; it does not score fit.** The same contract as `close`:
  every ranking signal is stated per row (EDHREC rank, synergy edges into the 99 with
  their rule names, the commander page's synergy, the Forge AI flag) and the sort is
  EDHREC rank. A composite "fit" number would be an authored weight driving a headline.
* **A converter is FLAGGED, never ranked away and never silently dropped.** The deck is
  held at bracket 3 with zero two-card infinites, and the loss-to-gain cards (Exquisite
  Blood) are exactly the ones every published drain list runs beside Vito or Sanguine
  Bond. A candidate that forms a two-card infinite with a card in the (staged) 99 is
  marked `infinite_with` and sorted LAST — a reader sees the line and the reason.
* **Death-triggered draw splits on one word.** `nontoken` (Midnight Reaper, Grim
  Haruspex) misses every eminence token; Species Specialist and Liliana's Standard
  Bearer do not. No source makes the distinction, so the row does.

The output is DATED and tracked (`candidate_scan.json`), like `deck_recon.json`: it is
the evidence a staging decision cites, and the decklist moves under it, so the validator
WARNS rather than fails when a candidate has since joined the 99.
"""
import json
import re
from datetime import date

from manamap.pilot.card_pool import card_keywords, corpus_oracle, load_frame, load_pool
from manamap.pilot.card_search import UNRANKED, commander_identity, deck_names
from manamap.pilot.deck_combos import is_infinite
from manamap.pilot.common import (
    deck_dir,
    decklist_sha256,
    expand_faces,
    load_card_roles,
    load_combo_details,
    load_json,
    load_synergy_graph,
    resolve_out_path,
)

ARTIFACT = "candidate_scan.json"
EDHREC_FILE = "edhrec_cards.json"
DIMENSIONS = ("drain", "gain", "threat", "outlet", "sweeper", "draw")
DEFAULT_LIMIT = 40

#: The keywords that make a body "scary" on the threat dimension — the ways it hurts
#: people that a blocker does not stop. Printed keywords only (`card_keywords`), so a
#: granted keyword does not count: the same floor `combat_damage_by_keyword` carries.
SCARY_KEYWORDS = ("Flying", "Trample", "Menace", "Deathtouch", "Lifelink", "Double strike")
THREAT_MIN_POWER = 4

#: THE PREDICATES, one home. Each is (id, regex) over the lowercased oracle text; a row
#: records the ids that fired, so "why is this here" is answered by the artifact.
_P = {
    # drain: an opponent loses life that is not damage
    "drain.each_or_target_loses": r"\b(each|target) (opponent|player)s? loses? (that much |\d+ |x |twice that much )?life\b",
    "drain.opponents_lose": r"\b(your )?opponents? (each )?loses? (that much |\d+ |x )?life\b",
    "drain.equal_to": r"loses? life equal to",
    "drain.death_trigger": r"whenever (a|another|one or more) (nontoken )?(other )?creatures?[^,\n]{0,30}?(dies|die|is put into a graveyard)",
    "drain.converter_gain_to_loss": r"whenever you gain life.{0,80}(loses?|lose) (that much |\d+ |x )?life",
    "drain.converter_loss_to_gain": r"whenever (an|each) opponent loses life.{0,80}(you )?gain (that much |\d+ |x )?life",
    # gain: the input of a gain/drain engine
    "gain.you_gain": r"\byou gain (\d+|x|that much|twice that much) life\b",
    "gain.payoff": r"whenever you gain life",
    # threat: an ability that hurts people
    "threat.activated_hurt": r"(\{[^}]*\}|remove [^:\n]{0,60}counters?[^:\n]{0,40}|sacrifice [^:\n]{0,40}): [^\n]{0,80}(loses? \d+ life|deals \d+ damage|destroy target|exile target|gets -)",
    # outlet: a sacrifice in the cost
    "outlet.free_sacrifice": r"(^|\n|, )sacrifice (a|another|an untapped|two) (creature|permanent|artifact)s?( or \w+)?(, |: )",
    "outlet.mill": r"\bmills?\b",
    # sweeper
    "sweeper.destroy_all": r"(destroy|exile) all (other )?(creatures|nonland permanents|permanents)",
    "sweeper.minus_all": r"(all|each) (other )?creatures? (gets?|your opponents control gets?) -\d+/-\d+",
    "sweeper.one_sided": r"(you don'?t control|your opponents? control|opponents? controls?)",
    # draw: by trigger class
    "draw.cards": r"\bdraws? (a |two |three |four |\d+ |x |that many )?cards?\b",
    "draw.put_into_hand": r"(reveal|exile|look at) the top (card|\w+ cards)[^\n]{0,120}put (it|that card|one of them|them) into your hand",
    "draw.death": r"whenever (a|another|one or more) (nontoken )?(other )?creatures?[^,\n]{0,30}?(dies|die)[^\n]{0,80}draw",
    "draw.died_this_turn": r"(died (under your control )?this turn[^\n]{0,80}draw|draw[^\n]{0,80}died (under your control )?this turn)",
    "draw.nontoken": r"nontoken",
    "draw.gain": r"whenever you gain life[^\n]{0,140}draw",
    "draw.typal": r"(vampire[^\n]{0,60}draw|draw[^\n]{0,60}vampire)",
    "draw.sacrifice": r"(sacrifice [^:\n]{0,60}: [^\n]{0,40}draw|additional cost[^\n]{0,60}sacrifice[^\n]{0,80}draw)",
    "draw.costs_life": r"((pay|lose) \d+ life[^\n]{0,60}draw|draw[^\n]{0,60}lose \d+ life)",
}
PREDICATES = {k: re.compile(v, re.IGNORECASE | re.DOTALL) for k, v in _P.items()}

#: WHICH EVENT DOES THIS CARD KEY ON. `function_overlap` below says two cards READ
#: alike; this says whether they FIRE on the same thing, which is the question a deck
#: actually asks. Measured need, 2026-10-02: Corpse Knight ("whenever another creature
#: you control enters"), Blood Artist ("whenever a creature dies"), Vito ("whenever you
#: gain life") and Bloodthirsty Conqueror ("whenever an opponent loses life") sit at
#: 0.96-0.98 cosine to one another in the ability space, and in a deck minting 8.04
#: tokens a game and losing 9.64 creatures they are four DIFFERENT multipliers rather
#: than four copies of one card. Similarity alone would have called them redundant.
#:
#: THE SELF/OTHER SPLIT IS LOAD-BEARING, and it is the distinction that already cost a
#: slot: Vampire Socialite reads "WHEN THIS CREATURE ENTERS, if an opponent lost life
#: this turn, put a +1/+1 counter on each other Vampire" — a ONE-SHOT on itself — and was
#: cut as though it were the same shape as Cordial Vampire's "whenever this creature or
#: another creature dies", which re-applies all game. One fires once. The other never
#: stops. No regex can be trusted to tell a reader which card is which unless it keeps
#: that split, so `enters.self` and `dies.self` are reported separately and never merged
#: into `enters.other` / `dies.other`.
_E = {
    # THE SUBJECT NOUN SITS WHERE THE WILDCARD IS, and a first pass got this wrong in
    # the most expensive possible way: "Whenever THIS CREATURE OR ANOTHER CREATURE dies"
    # is the printed wording of Blood Artist, Zulaport Cutthroat and Cordial Vampire, and
    # a pattern anchored on "whenever (another|a|each)" read all three as `static` —
    # every aristocrat in the format, silently classed as having no trigger at all. The
    # subject may therefore be anything up to the first clause break, and the compound
    # "this creature or another creature" must fire BOTH ids, because the card really
    # does cover its own death and the deck cares which.
    "enters.other":   r"whenever [^.\n]{0,60}?\b(another|a|one or more|each)\b[^,.\n]{0,40}?\b(creature|permanent|token|vampire|artifact)s?\b[^,.\n]{0,36}?\benters?\b",
    "enters.self":    r"when(ever)? this (creature|card|permanent|artifact|enchantment|land|token)\b[^,.\n]{0,60}?\benters?\b",
    "dies.other":     r"whenever [^.\n]{0,60}?\b(another|a|one or more|each)\b[^,.\n]{0,40}?\b(creature|permanent|token|vampire)s?\b[^,.\n]{0,36}?\b(dies|die)\b",
    "dies.self":      r"when(ever)? this (creature|card|permanent|token)\b[^,.\n]{0,60}?\b(dies|die)\b",
    # THE TOKEN EVENTS, found by the sweep and not by design. Mirkwood Bats reads
    # "Whenever you create or sacrifice a token, each opponent loses 1 life" and the first
    # pass classed it `no_trigger` — on a deck whose commander mints 8.04 tokens a game,
    # which is the single event most worth seeing. A card can key on the token rather than
    # on the creature, and "creature enters" does not cover it.
    "token_created":  r"whenever you create[^,.\n]{0,40}?\btokens?\b",
    "sacrifice_event": r"whenever (you|an opponent|a player)[^,.\n]{0,34}?sacrifices?\b",
    "gain_life":      r"whenever you gain life\b",
    "opp_loses_life": r"whenever (an|each|one or more) opponents? loses? life\b",
    "attacks":        r"whenever [^,\n]{0,48}?attacks\b",
    "cast":           r"whenever you cast\b",
    "combat_damage":  r"deals combat damage to (a|target) player\b",
    "upkeep":         r"at the beginning of (your|each) ([\w' ]{0,16})?upkeep\b",
    "end_step":       r"at the beginning of (your|each) ([\w' ]{0,16})?end step\b",
    "main_phase":     r"at the beginning of (your|each) (precombat |postcombat |first )?main phase\b",
    "sacrifice_cost": r"sacrifice (a|another|an|two|three|x) [^:\n]{0,40}:",
    "activated":      r"(\{[^}]{1,14}\})[^:\n]{0,40}:",
}
EVENTS = {k: re.compile(v, re.IGNORECASE | re.DOTALL) for k, v in _E.items()}


def trigger_events(text):
    """The events a card's abilities key on, as sorted ids; `['no_trigger']` when none fire.

    A card may key on several and ALL are reported — Zulaport Cutthroat is both
    `dies.other` and `dies.self`, because its text is "whenever this creature or
    another creature you control dies". Reporting both is the point: the reader
    sees that it also covers its own death, which a single id would hide.
    """
    low = (text or "").lower()
    hits = sorted(k for k, rx in EVENTS.items() if rx.search(low))
    return hits or ["no_trigger"]


#: THE ABILITY SPACE, loaded once per process. `embeddings_ability.npy` is the FUNCTION
#: space and is the only legitimate source of similarity here — the layout space knows
#: colour and type and would answer "what else is a black two-drop".
_ABILITY = {}


def _ability_space():
    """(unit-normalised ability embeddings, name -> row). Empty when the artifact is absent.

    Absent means absent: on a fresh clone with no pipeline run there is no ability
    space, and the scan must still produce rows rather than fail. Callers check for
    the empty dict and omit `function_overlap` from the row instead of writing a 0.0,
    which a reader could not tell from a measured dissimilarity.
    """
    if _ABILITY:
        return _ABILITY
    import numpy as np
    from manamap import config
    path = config.DATA_DIR / "embeddings_ability.npy"
    if not path.exists():
        _ABILITY["index"] = {}
        return _ABILITY
    frame = load_frame()
    emb = np.load(path)
    names = list(frame["name"])
    if len(names) != emb.shape[0]:
        # INDEX ALIGNMENT IS THE PIPELINE'S LOUDEST INVARIANT: projection[i] ==
        # cards.csv[i] == embeddings[i]. A mismatch means a partial regeneration, and
        # guessing which rows moved would silently mislabel every neighbour. Degrade.
        _ABILITY["index"] = {}
        return _ABILITY
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    _ABILITY["matrix"] = emb / (norms + 1e-9)
    _ABILITY["index"] = {n: i for i, n in enumerate(names)}
    return _ABILITY


def function_overlap(name, present):
    """The card in the 99 this candidate most resembles IN FUNCTION, and how closely.

    Answers "do I already own this card's job?" — the column the scanner lacked while
    four consecutive Edgar branches each bought another copy of a drain body the deck
    already had seven of (Cruel Celebrant 0.981, Sanctum Seeker 0.980, Bloodthirsty
    Conqueror 0.975, Vito 0.971, Blood Artist 0.970, Sanguine Bond 0.963, Bloodletter
    0.945 against that cluster's centre).

    It is a SIGNAL, not a verdict, and `trigger_events` is why: a high cosine to a card
    you own means "reads alike", and two cards that read alike can fire on different
    events and stack. Returns None when the ability space or either name is unavailable.
    """
    space = _ability_space()
    idx = space.get("index") or {}
    if not idx or name not in idx:
        return None
    mine = [p for p in present if p in idx]
    if not mine:
        return None
    import numpy as np
    m = space["matrix"]
    sims = m[[idx[p] for p in mine]] @ m[idx[name]]
    best = int(np.argmax(sims))
    return {"nearest_in_99": mine[best], "cosine": round(float(sims[best]), 4),
            "space": "embeddings_ability.npy"}


#: Which predicates / roles / tags / keywords ADMIT a card to each dimension. Everything
#: else in `_P` is a FLAG recorded on an admitted row, never an admission.
ADMIT = {
    # A CONVERTER IS ADMITTED AND FLAGGED: Exquisite Blood's "whenever an opponent loses
    # life, you gain that much" names no drain itself, and leaving it out would hide the
    # one card every published drain list runs that bracket 3 forbids.
    "drain": {"oracle": ("drain.each_or_target_loses", "drain.opponents_lose", "drain.equal_to",
                         "drain.converter_gain_to_loss", "drain.converter_loss_to_gain"),
              "roles": ("wincon:drain",), "tags": ()},
    "gain": {"oracle": ("gain.you_gain",), "roles": (), "tags": ("lifegain",), "keywords": ("Lifelink",)},
    "threat": {"oracle": (), "roles": (), "tags": ()},            # power + keyword, below
    "outlet": {"oracle": ("outlet.free_sacrifice",), "roles": ("sac-outlet",), "tags": ()},
    "sweeper": {"oracle": ("sweeper.destroy_all", "sweeper.minus_all"), "roles": ("removal:sweeper",), "tags": ()},
    # Admitted by ROLE, by the `draw` mechanical tag, or by the oracle saying "draw a card"
    # / "put it into your hand": Midnight Reaper carries the tag and no draw role, and
    # Liliana's Standard Bearer ("draw cards equal to …") carries neither.
    "draw": {"oracle": ("draw.cards", "draw.put_into_hand"),
             "roles": ("draw:engine", "draw:burst", "draw:impulse", "draw:wheel", "draw:cantrip"),
             "tags": ("draw",)},
}

LIMITS = (
    "Retrieval, not judgement: rows are sorted by EDHREC rank (unranked last) with every "
    "signal stated per row; nothing here scores fit. A row flagged `infinite_with` sorts "
    "after every unflagged row in its dimension and is never dropped.",
    "Oracle predicates are regexes over printed text and read what the text says, not what "
    "the card does at a table: a drain keyed to a trigger the deck never fires is still a "
    "drain here. `matched` names the predicate ids so a reader can re-run the question.",
    "Keywords and power are PRINTED (`card_pool.card_keywords`); a granted keyword or a "
    "pumped body does not count, and a `*` power reads as unknown, not 0.",
    "`infinite_with` is read from combo_details two-card lines whose other card is in the "
    "99 (or the staged list under --against-branch); a three-card line is counted in "
    "`two_card_lines`/`bracket_max` only. Game Changers are dropped (bracket 3) and counted.",
    "`edhrec` per row is present only when edhrec_cards.json exists for the deck, dated by "
    "its own as_of; absent means not fetched, never zero.",
    "`function_overlap` is the 99's NEAREST card in the ability space with its cosine — "
    "'do I already own this job?'. It is a signal and never a verdict, and it is read "
    "TOGETHER with `trigger_events`: Corpse Knight sits at 0.974 to a drain body this deck "
    "already runs and fires on creatures ENTERING while that one fires on deaths, which in "
    "a deck minting 8.04 tokens and losing 9.64 creatures a game is a second multiplier "
    "rather than a duplicate. High cosine plus the SAME event is redundancy; high cosine "
    "plus a different event is a new line on an old theme. Absent when the ability space "
    "is missing or the card is not in it — never 0.0, which would read as measured.",
    "`trigger_events` reads PRINTED text, keeps the self/other split ("
    "`enters.self` is a one-shot on the card itself, `enters.other` re-applies), and "
    "reports every event a card keys on rather than picking one. It says what the card "
    "WOULD fire on, not that the deck fires it: an `enters.other` drain in a deck that "
    "makes no tokens is still classed `enters.other` here.",
)


def _text(v):
    return "" if v is None or v != v else str(v)


def _tags(cell):
    return [t.strip() for t in _text(cell).split(",") if t.strip()]


def _frame_rows():
    """{name: {oracle, type_line, tags, cmc, mana_cost}} straight from the frame."""
    frame = load_frame()
    out = {}
    for name, tl, tags, cmc, mc in zip(frame["name"], frame["type_line"], frame["mechanical_tags"],
                                        frame["cmc"], frame["mana_cost"]):
        if name in out:
            continue
        out[name] = {"type_line": _text(tl), "tags": _tags(tags),
                     "cmc": 0.0 if cmc != cmc else float(cmc), "mana_cost": _text(mc)}
    return out


def staged_names(slug, branch):
    """The 99 the scan excludes against — the deck's, or a branch's decklist."""
    if not branch:
        return deck_names(slug)
    from manamap.pilot.fetch_deck import parse_decklist
    text = (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8")
    out = set()
    for e in parse_decklist(text):
        out |= expand_faces(e["name"])
    return out


def _edhrec(slug):
    doc = load_json(deck_dir(slug) / EDHREC_FILE)
    if not doc:
        return None, {}
    return doc.get("as_of"), doc.get("cards") or {}


def _two_card_lines(name, present, details):
    """(infinite_with, two_card_lines, bracket_max) for one candidate against `present`."""
    combos = details["combos"]
    inf, lines, brackets = [], 0, []
    for i in details["by_card"].get(name, []):
        c = combos[i]
        if len(c["cards"]) != 2:
            continue
        other = next((x for x in c["cards"] if x != name), None)
        if other is None or other not in present:
            continue
        lines += 1
        if c.get("bracket") is not None:
            brackets.append(c["bracket"])
        if is_infinite(c):
            inf.append(other)
    return sorted(set(inf)), lines, (max(brackets) if brackets else None)


def _admitted(dim, name, rec, oracle, roles, kw, fired):
    a = ADMIT[dim]
    hit = {"oracle": [p for p in a["oracle"] if p in fired],
           "roles": [r for r in a["roles"] if r in roles],
           "tags": [t for t in a["tags"] if t in rec["tags"]],
           "keywords": [k for k in a.get("keywords", ()) if k in kw["keywords"]]}
    if dim == "threat":
        front = rec["type_line"].split(" // ")[0]
        scary = [k for k in SCARY_KEYWORDS if k in kw["keywords"]]
        big = kw["power"] is not None and kw["power"] >= THREAT_MIN_POWER
        if "Creature" in front and big and (scary or "threat.activated_hurt" in fired):
            hit["keywords"] = scary
            if "threat.activated_hurt" in fired:
                hit["oracle"] = ["threat.activated_hurt"]
            return hit
        return None
    return hit if any(hit.values()) else None


def _flags(dim, rec, oracle, kw, fired, roles):
    f = {}
    if dim == "drain":
        f["death_drain"] = "death_trigger" in rec["tags"] or "drain.death_trigger" in fired
        conv = [k for k in ("drain.converter_gain_to_loss", "drain.converter_loss_to_gain") if k in fired]
        f["converter"] = conv[0].rsplit(".", 1)[1] if conv else None
    elif dim == "gain":
        f["lifelink"] = "Lifelink" in kw["keywords"]
        f["payoff"] = "gain.payoff" in fired
    elif dim == "threat":
        f["scary"] = len([k for k in SCARY_KEYWORDS if k in kw["keywords"]]) + (1 if "threat.activated_hurt" in fired else 0)
        f["activated_hurt"] = "threat.activated_hurt" in fired
        f["typal"] = "Vampire" in rec["type_line"]
        f["death_drain"] = "death_trigger" in rec["tags"] and ("lifegain" in rec["tags"] or "wincon:drain" in roles)
    elif dim == "outlet":
        f["free"] = "outlet.free_sacrifice" in fired
        f["mill"] = "outlet.mill" in fired
    elif dim == "sweeper":
        f["one_sided"] = "sweeper.one_sided" in fired
    elif dim == "draw":
        trig = [k for k in ("draw.death", "draw.died_this_turn", "draw.gain", "draw.typal", "draw.sacrifice") if k in fired]
        f["trigger"] = ("death" if trig and trig[0] in ("draw.death", "draw.died_this_turn")
                        else trig[0].rsplit(".", 1)[1] if trig else "plain")
        f["nontoken"] = ("draw.death" in fired) and ("draw.nontoken" in fired)
        f["costs_life"] = "draw.costs_life" in fired
    return f


def scan(slug, dimensions=DIMENSIONS, branch=None, limit=DEFAULT_LIMIT):
    pool = load_pool()
    if pool is None:
        raise SystemExit("cards.csv is absent — scan-candidates reads the corpus. Run `manamap extract` first.")
    rows = _frame_rows()
    oracle = corpus_oracle()
    keywords = card_keywords()
    roles_map = load_card_roles()
    details = load_combo_details()
    synergy = load_synergy_graph()
    from manamap.sim import forge_cards
    ident = commander_identity(slug)
    present = staged_names(slug, branch)
    edhrec_as_of, edhrec = _edhrec(slug)
    dims = list(dimensions)

    excluded = {"in_99": 0, "identity": 0, "illegal": 0, "game_changer": []}
    found = {d: [] for d in dims}
    for name, rec in pool.items():
        if expand_faces(name) & present:
            excluded["in_99"] += 1
            continue
        if not rec["legal"]:
            excluded["illegal"] += 1
            continue
        if not rec["color_identity"] <= ident:
            excluded["identity"] += 1
            continue
        fr = rows.get(name)
        if fr is None:
            continue
        text = oracle.get(name, "")
        low = text.lower()
        fired = {k for k, p in PREDICATES.items() if p.search(low)}
        roles = roles_map.get(name) or []
        kw = keywords.get(name) or {"keywords": [], "power": None, "toughness": None}
        for dim in dims:
            hit = _admitted(dim, name, fr, text, roles, kw, fired)
            if hit is None:
                continue
            if rec["game_changer"]:
                if name not in excluded["game_changer"]:
                    excluded["game_changer"].append(name)
                continue
            inf, lines, bmax = _two_card_lines(name, present, details)
            edges = [{"partner": e["partner"], "rules": e.get("synergies") or []}
                     for e in (synergy.get(name) or []) if expand_faces(e["partner"]) & present]
            row = {
                "name": name, "cmc": fr["cmc"], "mana_cost": fr["mana_cost"], "type_line": fr["type_line"],
                "keywords": kw["keywords"], "power": kw["power"], "toughness": kw["toughness"],
                "roles": sorted(roles), "tags": fr["tags"], "edhrec_rank": rec["edhrec_rank"],
                "matched": hit, "flags": _flags(dim, fr, text, kw, fired, roles),
                "synergy_into_99": edges,
                "combos": {"infinite_with": inf, "two_card_lines": lines, "bracket_max": bmax},
                "forge_ai_flag": forge_cards.ai_flag(name),
                # WHICH EVENT, and DO I ALREADY OWN THIS JOB. The pair is deliberate:
                # similarity alone called Corpse Knight and Blood Artist the same card at
                # 0.97, and they fire on entries and deaths — two events this deck produces
                # 8.04 and 9.64 times a game, so both stack rather than duplicating.
                "trigger_events": trigger_events(text),
                "function_overlap": function_overlap(name, present),
                "oracle_text": text,
            }
            if edhrec:
                e = edhrec.get(name) or edhrec.get(name.split(" // ")[0])
                if e is not None:
                    row["edhrec"] = e
            found[dim].append(row)

    def key(r):
        return (1 if r["combos"]["infinite_with"] else 0,
                r["edhrec_rank"] if r["edhrec_rank"] is not None else UNRANKED, r["name"])
    out_dims = {}
    for dim in dims:
        allrows = sorted(found[dim], key=key)
        out_dims[dim] = {
            "admits": {k: list(v) for k, v in ADMIT[dim].items()} | (
                {"power_at_least": THREAT_MIN_POWER, "scary_keywords": list(SCARY_KEYWORDS)} if dim == "threat" else {}),
            "matched": len(allrows),
            "flagged_infinite": sum(1 for r in allrows if r["combos"]["infinite_with"]),
            "candidates": allrows[:limit],
            "truncated": max(0, len(allrows) - limit),
        }
    doc = {
        "slug": slug, "as_of": date.today().isoformat(),
        "decklist_sha256": decklist_sha256(slug, branch), "against_branch": branch,
        "identity": sorted(ident), "limit": limit,
        "sources": {"edhrec_cards": edhrec_as_of, "combo_details": details.get("meta"),
                    "predicates": {k: v for k, v in _P.items()},
                    # The event patterns travel with the artifact for the same reason the
                    # admission predicates do: "why is this row classed enters.other" has
                    # to be answerable from the file, not from whatever the code says today.
                    "events": {k: v for k, v in _E.items()},
                    "ability_space": "embeddings_ability.npy"},
        "dimensions": out_dims, "excluded": excluded, "limits": list(LIMITS),
    }
    return _with_oracle(doc, present)


def _with_oracle(doc, present):
    """Attach the 99's oracle text for the report's SAME-EVENT comparison only.

    Underscore-prefixed because it is a rendering aid, not evidence: `main` strips it
    before the artifact is written, so cast_proofs-style key creep cannot reach the
    tracked file and the validator never has to know about it.
    """
    doc["_oracle"] = {n: corpus_oracle().get(n, "") for n in present}
    return doc


def format_report(doc):
    out = [f"CANDIDATE SCAN — {doc['slug']} ({doc['as_of']}) · identity {''.join(doc['identity'])}"
           + (f" · against branch {doc['against_branch']}" if doc.get("against_branch") else "")
           + f" · EDHREC page {'fetched ' + doc['sources']['edhrec_cards'] if doc['sources'].get('edhrec_cards') else 'ABSENT'}"]
    ex = doc["excluded"]
    out.append(f"  excluded: {ex['in_99']} in the 99, {ex['identity']} out of identity, {ex['illegal']} illegal, "
               f"{len(ex['game_changer'])} Game Changer(s) dropped" + (f" ({', '.join(ex['game_changer'][:6])}…)" if ex["game_changer"] else ""))
    for dim, d in doc["dimensions"].items():
        out.append("")
        out.append(f"  {dim.upper()} — {d['matched']} match(es), showing {len(d['candidates'])}"
                   + (f", {d['truncated']} more" if d["truncated"] else "")
                   + (f"; {d['flagged_infinite']} flagged infinite_with, sorted last" if d["flagged_infinite"] else ""))
        for r in d["candidates"]:
            rank = r["edhrec_rank"] if r["edhrec_rank"] is not None else "unranked"
            fl = ", ".join(f"{k}={v}" for k, v in r["flags"].items() if v not in (False, None, 0, "plain"))
            inf = f"  ⚠ INFINITE with {', '.join(r['combos']['infinite_with'])}" if r["combos"]["infinite_with"] else ""
            kw = f" [{', '.join(r['keywords'])}]" if r["keywords"] else ""
            pt = f" {r['power']}/{r['toughness']}" if r["power"] is not None else ""
            ed = f"  edhrec {rank}" + (f" · page synergy {r['edhrec'].get('synergy'):+.2f}" if r.get("edhrec") and r["edhrec"].get("synergy") is not None else "")
            ai = f"  AI:{r['forge_ai_flag']}" if r.get("forge_ai_flag") else ""
            out.append(f"    {r['mana_cost'] or '—':<10} {r['name']}{pt}{kw}{ed}{ai}{inf}")
            why = " · ".join(f"{k}:{','.join(v)}" for k, v in r["matched"].items() if v)
            out.append(f"      {why}" + (f"  flags: {fl}" if fl else "")
                       + (f"  synergy→99: {len(r['synergy_into_99'])}" if r["synergy_into_99"] else ""))
            # THE TWO READ TOGETHER OR NOT AT ALL. A cosine without the event says
            # "you own this card" about a second multiplier; an event without the
            # cosine hides the eighth copy. So one line carries both, and it says
            # SAME EVENT when the overlap is close AND keyed to the same thing —
            # which is the only combination that means redundancy.
            ov = r.get("function_overlap")
            ev = r.get("trigger_events") or []
            if ov or ev:
                bits = []
                if ev:
                    bits.append("fires on " + ",".join(ev))
                if ov:
                    near = ov["nearest_in_99"]
                    bits.append(f"nearest in 99: {near} {ov['cosine']:.2f}")
                    if ov["cosine"] >= 0.95:
                        # `no_trigger` IS THE ABSENCE OF AN EVENT, NOT AN EVENT, and
                        # intersecting it with itself called Feed the Swarm a duplicate of
                        # Anguished Unmaking because both are instants. Two one-shots that
                        # read alike may still both be worth running, so the verdict is
                        # withheld and the cosine is left to speak.
                        mine = set(ev) - {"no_trigger"}
                        theirs = set(trigger_events(doc.get("_oracle", {}).get(near, ""))) - {"no_trigger"}
                        if mine and theirs:
                            bits.append("SAME EVENT — likely a duplicate" if (mine & theirs)
                                        else "different event — stacks rather than duplicates")
                        else:
                            bits.append("neither keys on an event — judge on the text")
                out.append("      " + " · ".join(bits))
    return "\n".join(out)


def main(args):
    slug = args.slug
    names = list(getattr(args, "shortlist", None) or [])
    if names:
        doc = shortlist(slug, names, branch=getattr(args, "against_branch", None))
        if getattr(args, "as_json", False):
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        else:
            print(format_shortlist(doc))
        out = getattr(args, "out", None)
        if out:
            p = resolve_out_path(out, slug, "shortlist")
            p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  wrote {p}")
        return
    dim = getattr(args, "dimension", None) or "all"
    dims = DIMENSIONS if dim == "all" else (dim,)
    doc = scan(slug, dims, branch=getattr(args, "against_branch", None),
               limit=getattr(args, "limit", None) or DEFAULT_LIMIT)
    if getattr(args, "as_json", False):
        print(json.dumps({k: v for k, v in doc.items() if not k.startswith("_")},
                         indent=2, ensure_ascii=False))
    else:
        print(format_report(doc))
    # The report is rendered; the rendering aid must not reach a file. Popped HERE rather
    # than inside each writer, so a future third write path cannot miss it.
    doc.pop("_oracle", None)
    if getattr(args, "write", False):
        if dim != "all":
            raise SystemExit("--write records the whole scan: run it without --dimension (or with all)")
        p = deck_dir(slug) / ARTIFACT
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n  wrote {p}")
    out = getattr(args, "out", None)
    if out:
        p = resolve_out_path(out, slug, "candidate-scan")
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot scan-candidates <slug>`.")


# ── THE SHORTLIST JOIN (Phase 3 of the drain-v1 plan) ────────────────────────────────────
#
# One row per candidate the pilot is weighing: what the scan says (dimensions, flags),
# what the prescription said (rank, closes, natural cut), which recon findings name it
# (dimension, confidence), the EDHREC page figures, `assess`'s read (job, gate, what the
# goldfish can price, the Forge AI flag) and a PREDICTED DIRECTION per Forge axis read off
# the flags — up / flat / unknown, never a number. A view, never tracked.

#: Which scan dimension predicts which Forge axis to move, and how. Stated here so the
#: shortlist's "predicted" column is a rule a reader can disagree with, not a judgement.
AXIS_PREDICTION = {
    "drain": {"forge.drain_dealt": "up"},
    "gain": {"forge.life_gained": "up"},
    "threat": {"forge.biggest_hit": "up", "forge.evasive_damage_share": "up"},
    "outlet": {"forge.drain_dealt": "up (via deaths)", "forge.kills_by_ability": "up (floor: AI must activate)"},
    "sweeper": {"forge.drain_dealt": "up (deaths on our terms)", "forge.combat_damage_dealt_to_players": "down (our board dies too)"},
    "draw": {"forge.extra_draw_per_turn": "up", "forge.empty_hand_turns": "down"},
}


def _prescription(slug):
    """The newest prescription's adds and cuts, by card."""
    import glob
    paths = sorted(glob.glob(str(deck_dir(slug) / "prescriptions" / "*.json")))
    if not paths:
        return {}, {}, None
    doc = load_json(__import__("pathlib").Path(paths[-1])) or {}
    adds = {a["card"]: dict(a, rank=i + 1) for i, a in enumerate(doc.get("add_candidates") or [])}
    cuts = {c["card"]: dict(c, rank=i + 1) for i, c in enumerate(doc.get("cut_candidates") or [])}
    return adds, cuts, doc.get("id")


def _recon(slug):
    doc = load_json(deck_dir(slug) / "deck_recon.json") or {}
    by = {}
    for i, f in enumerate(doc.get("findings") or []):
        for c in f.get("cards") or []:
            by.setdefault(c, []).append({"finding": i, "dimension": f.get("dimension"),
                                         "confidence": f.get("confidence"), "claim": f.get("claim", "")[:140]})
    return by, doc.get("as_of")


def shortlist(slug, names, branch=None):
    """Join every source the bench has on `names`. Returns the rows and the sources' dates."""
    from manamap.pilot import assess as _assess
    # A LIVE, UNCAPPED scan: the tracked file keeps forty rows per dimension and a card
    # can sit in the top forty of one dimension and past the cut of another (Twilight
    # Prophet is rank 1110 in draw), which read as "not a draw card" the first time.
    scan_doc = scan(slug, branch=branch, limit=5000)
    in_scan = {}
    for dim, block in scan_doc["dimensions"].items():
        for r in block["candidates"]:
            if r["name"] in names:
                in_scan.setdefault(r["name"], {})[dim] = r
    adds, cuts, rx_id = _prescription(slug)
    recon, recon_as_of = _recon(slug)
    _, edhrec = _edhrec(slug)
    try:
        arows = {r["card"]: r for r in (_assess.assess(slug, list(names), branch) or {}).get("cards") or []}
        assess_error = None
    except Exception as exc:                        # noqa: BLE001 - a view degrades, never dies
        arows, assess_error = {}, f"{exc.__class__.__name__}: {exc}"
    rows = []
    for name in names:
        dims = in_scan.get(name, {})
        any_row = next(iter(dims.values()), None)
        predicted = {}
        for dim in dims:
            for axis, direction in AXIS_PREDICTION[dim].items():
                predicted.setdefault(axis, direction)
        a = arows.get(name) or {}
        rows.append({
            "name": name,
            "dimensions": {d: {k: v for k, v in r["flags"].items() if v not in (False, None, 0, "plain")} for d, r in dims.items()},
            "matched": {d: r["matched"] for d, r in dims.items()},
            "edhrec_rank": any_row["edhrec_rank"] if any_row else None,
            "infinite_with": sorted({x for r in dims.values() for x in r["combos"]["infinite_with"]}),
            "forge_ai_flag": any_row["forge_ai_flag"] if any_row else a.get("forge_ai_flag"),
            "prescription": ({"as": "add", "rank": adds[name]["rank"], "closes": adds[name].get("closes"),
                              "natural_cut": adds[name].get("natural_cut")} if name in adds else
                             {"as": "cut", "rank": cuts[name]["rank"], "difficulty": cuts[name].get("difficulty")} if name in cuts else None),
            "recon": recon.get(name) or [],
            "edhrec": ({k: edhrec[name].get(k) for k in ("synergy", "num_decks", "from")} if name in edhrec else None),
            "assess": ({k: a.get(k) for k in ("job", "gate", "model_sees", "answers", "verdict", "mv")} if a else None),
            "predicted": predicted,
            "in_scan": bool(dims),
        })
    return {"slug": slug, "against_branch": branch, "sources": {"scan": scan_doc.get("as_of"), "prescription": rx_id,
                                                                 "recon": recon_as_of, "edhrec": scan_doc["sources"].get("edhrec_cards"),
                                                                 "assess": assess_error or "ok"},
            "rows": rows}


def format_shortlist(doc):
    out = [f"SHORTLIST — {doc['slug']}" + (f" against branch {doc['against_branch']}" if doc.get("against_branch") else "")
           + f" · scan {doc['sources']['scan']} · prescription {doc['sources']['prescription']} · recon {doc['sources']['recon']}"
           + f" · EDHREC {doc['sources']['edhrec'] or 'ABSENT'}"
           + ("" if doc["sources"].get("assess") == "ok" else f"\n  assess unavailable: {doc['sources'].get('assess')}")]
    for r in doc["rows"]:
        out.append("")
        head = f"  {r['name']}" + (f"  edhrec {r['edhrec_rank']}" if r["edhrec_rank"] is not None else "")
        if r["edhrec"]:
            head += f"  page synergy {r['edhrec']['synergy']:+.2f} · {r['edhrec']['num_decks']} decks ({r['edhrec']['from']})"
        if r["forge_ai_flag"]:
            head += f"  AI:{r['forge_ai_flag']}"
        if r["infinite_with"]:
            head += f"  ⚠ INFINITE with {', '.join(r['infinite_with'])}"
        out.append(head)
        if not r["in_scan"]:
            out.append("    scan: NOT ADMITTED by any dimension (in the 99, out of identity, a Game Changer, or matches no predicate)")
        for d, fl in r["dimensions"].items():
            out.append(f"    {d:8s} " + (", ".join(f"{k}={v}" for k, v in fl.items()) or "—")
                       + "   admitted by " + " · ".join(f"{k}:{','.join(v)}" for k, v in r["matched"][d].items() if v))
        if r["prescription"]:
            p = r["prescription"]
            out.append(f"    prescription: {p['as']} #{p['rank']}" + (f" — closes {p['closes']}" if p.get("closes") else "")
                       + (f"; natural cut {p['natural_cut']}" if p.get("natural_cut") else "") + (f"; {p['difficulty']}" if p.get("difficulty") else ""))
        for f in r["recon"]:
            out.append(f"    recon[{f['finding']}] {f['dimension']} · {f['confidence']}: {f['claim']}")
        if r["assess"]:
            a = r["assess"]
            out.append(f"    assess: job {a.get('job')} · gate {a.get('gate')} · model sees {a.get('model_sees') or '—'} · {a.get('verdict')}")
        if r["predicted"]:
            out.append("    predicted: " + "; ".join(f"{k} {v}" for k, v in r["predicted"].items()))
    return "\n".join(out)
