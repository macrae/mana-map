"""EDHREC: what the world actually plays, cached on disk.

Two endpoints, both public JSON, and between them they answer the two questions
the bench cannot answer from its own corpus:

- **`/pages/commanders/<identity>.json`** — the top ~100 commanders for a colour
  identity, in EDHREC's popularity order. This is PRD §6.1 step 4.
- **`/pages/average-decks/<slug>.json`** — a representative decklist for one
  commander. `sim/opponents.py` has used this since S3 to build pod seats; this
  module is that fetcher generalised, and `opponents` should be moved onto it
  rather than keeping two.

**Everything is cached to disk and nothing re-fetches without being asked.**
Two separate reasons, and the second is the one that bites:

1. Politeness. An eval that re-ranks 60 commanders is 60 requests, and it gets
   run on every change to the embedding.
2. **Reproducibility.** EDHREC rankings move — §6.1 step 4 says explicitly not
   to freeze them for the *product*, and that is right for the product. It is
   exactly wrong for an *eval*: a benchmark whose ground truth shifts under you
   cannot tell a model change from a metagame change. So the eval reads a
   frozen snapshot with a fetch date on it, and refreshing that snapshot is a
   deliberate act with its own commit.

The colour-identity path takes a two-letter guild code (`wu`) and answers with a
redirect to a guild name (`/commanders/azorius`), which is followed once and
cached under the code the caller asked for.
"""

import json
import time
import urllib.error
import urllib.request

from manamap.config import DATA_DIR

BASE = "https://json.edhrec.com/pages"
CACHE_DIR = DATA_DIR / "edhrec"

#: One request a second. Nothing here is latency-sensitive — the eval runs in a
#: batch and the product path reads the cache — so there is no reason to lean on
#: somebody else's server.
DELAY_SECONDS = 1.0

#: Named, because an unidentified scraper is the kind that gets blocked, and
#: being blockable-on-purpose is better than being anonymous.
USER_AGENT = "mana-map/1.0 (personal deck research; one request per second)"

_last_request = [0.0]


def _get(url):
    """One request, rate-limited. Raises on anything but success."""
    wait = DELAY_SECONDS - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _cached(key, fetch):
    """Read `key` from the cache, or fetch and store it.

    The cache is keyed by the CALLER's key, not the resolved URL, so a redirect
    does not produce two entries for one question.
    """
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    doc = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def _follow(doc, url):
    """EDHREC answers a colour code with a redirect to a guild name."""
    if isinstance(doc, dict) and doc.get("redirect") and len(doc) == 1:
        return _get(f"{BASE}{doc['redirect']}.json")
    return doc


def _cardviews(doc):
    """The card rows out of EDHREC's nested container, or []."""
    lists = (((doc or {}).get("container") or {}).get("json_dict") or {}).get("cardlists") or []
    return [c for group in lists for c in (group.get("cardviews") or [])]


#: Three-colour identities, and ONLY three-colour, need their name.
#:
#: Probed rather than assumed: `w` redirects to mono-white, `rg` to gruul,
#: `wubr` to yore-tiller and `wubrg` to five-color — every arity takes a colour
#: code and answers with a redirect. `wur` returns **403**. So the shards and
#: wedges are the one gap, and they are hardcoded here because they are
#: EDHREC's vocabulary rather than Magic's; a caller should be able to ask in
#: colours and never learn the word "Jeskai".
_WEDGES = {
    frozenset("wub"): "esper",   frozenset("ubr"): "grixis",
    frozenset("brg"): "jund",    frozenset("rgw"): "naya",
    frozenset("gwu"): "bant",    frozenset("wbg"): "abzan",
    frozenset("wur"): "jeskai",  frozenset("ubg"): "sultai",
    frozenset("wbr"): "mardu",   frozenset("urg"): "temur",
}


def identity_segment(identity):
    """The URL segment EDHREC will answer for a colour code."""
    letters = frozenset(identity)
    return _WEDGES.get(letters, identity) if len(letters) == 3 else identity


def top_commanders(identity, limit=100):
    """The most-played commanders for a colour identity, EDHREC's order.

    `identity` is a lowercase colour code — `w`, `wu`, `wubrg`, or `colorless`.
    Order is popularity and is preserved: it is the only ranking signal here,
    and re-sorting it would discard the one thing the endpoint is for.
    """
    key = f"commanders-{identity}"
    seg = identity_segment(identity)
    doc = _cached(key, lambda: _follow(_get(f"{BASE}/commanders/{seg}.json"),
                                       f"{BASE}/commanders/{seg}.json"))
    return [c["name"] for c in _cardviews(doc)][:limit]


def themes(commander):
    """How this commander is ACTUALLY built, with a deck count each. PRD §7.2.

    EDHREC's `taglinks` panel, which is the archetype list the PRD asks an agent
    to research — except it is already data. Zur comes back Enchantress 1201,
    Auras 736, Stax 542, Control 529, Combo 380, Voltron 361, which is precisely
    the split §7.2 names as its worked example.

    **The count is a PLAY RATE and nothing more.** §7.2 requires styles be shown
    side by side without ranking, and permits play rates as data. EDHREC's order
    is preserved because re-sorting it would be inventing a ranking; the count
    travels so a reader can see what the order means, and nothing here calls one
    of them better.
    """
    from manamap.sim.opponents import edhrec_slug

    slug = edhrec_slug(commander)
    doc = _cached(f"themes-{slug}", lambda: _get(f"{BASE}/commanders/{slug}.json"))
    panel = ((doc.get("panels") or {}).get("taglinks")) or []
    return [{"slug": t["slug"], "name": t.get("value") or t["slug"],
             "decks": int(t.get("count") or 0)}
            for t in panel if t.get("slug")]


def average_deck(commander, theme=None):
    """One commander's representative decklist: `{commander, cards: [(name, qty)]}`.

    With `theme`, the representative deck for THAT STYLE — the path is
    `average-decks/<commander>/<theme>`, and it is what makes a per-archetype
    role template derivable rather than authored.

    A representative list, never a specific one — the same caveat
    `sim/opponents.py` writes into every `source.json` it produces.
    """
    from manamap.sim.opponents import edhrec_slug          # one slug rule, not two

    slug = edhrec_slug(commander)
    path = f"{slug}/{theme}" if theme else slug
    key = f"average-{slug}" + (f"-{theme}" if theme else "")
    doc = _cached(key, lambda: _get(f"{BASE}/average-decks/{path}.json"))
    deck = doc.get("deck") or {}
    cards = [(name, int(qty))
             for rows in (deck.get("cards") or {}).values()
             for name, qty in rows]
    return {"slug": slug, "theme": theme,
            "commander": (deck.get("commander") or [None])[0], "cards": cards}


def cache_state():
    """What is on disk, for a command that wants to report it before fetching."""
    if not CACHE_DIR.is_dir():
        return {"entries": 0, "dir": str(CACHE_DIR)}
    files = sorted(CACHE_DIR.glob("*.json"))
    return {"entries": len(files), "dir": str(CACHE_DIR),
            "bytes": sum(f.stat().st_size for f in files)}
