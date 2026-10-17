"""Pilot: read a deck's DeckCheck copy back — is it the list that matters?

`manamap pilot deckcheck <slug>`            the copy vs our list, dated
`manamap pilot deckcheck <slug> --text`     the copy as `decklist.txt` text (stdout only)
`manamap pilot deckcheck <slug> --json`
`manamap pilot deckcheck <slug> --url <deckcheck url>`   an unlinked deck, or another copy

DECKCHECK HAS A READ-ONLY PUBLIC API AND NO WRITE (docs/integrations.md, 2026-10-17):
`GET https://deckcheck.co/api/external/deck?deck_id=<id>` returns the deck's boards and
its Threat Index as JSON, no key needed, rate-limited by IP. So publishing stays a
paste (`deck-export --format moxfield`, then `deck-link <slug> deckcheck <url>`), and
this command is the OTHER direction: it reads the copy back and says whether it is the
sleeved list, naming the cards that differ. The offline claim (`deck_link.drift`) is
about what was PASTED; this is the confirmation, and it is dated by the cache record
it came from, so a cached answer never reads as fresher than it is.

THE PULL PATH NEVER FETCHES INSIDE `check-in`. `--text` prints exactly what
`check_in.render_decklist` makes of the copy, so

    manamap pilot deckcheck <slug> --text | manamap pilot check-in <slug> --from -

puts every `check_in.analyze` refusal in front of a list that came from the other site.

FAILS OPEN, WRITES NOTHING. `net.Offline` (no network and no cached answer), a private
deck (403) and a missing one (404) each print one line and exit 1; the only thing this
module ever writes is `net`'s cache under `data/cache/deckcheck/`. Zero LLM calls.
"""

import json
import os
import sys

import requests

from manamap import config, net
from manamap.pilot import check_in, deck_link
from manamap.pilot.common import deck_dir, decklist_sha256
from manamap.pilot.fetch_deck import parse_mainboard

SERVICE = "deckcheck"
KEYCHAIN_SERVICE = "manamap-deckcheck"

#: The boards DeckCheck returns and where each lands in `decklist.txt`. Anything
#: else it ever adds (a maybeboard) is a sideboard here, and `check_in.analyze`
#: drops a sideboard on a Commander deck with a warning — never silently main.
BOARDS = {"commanders": ("main", True), "mainboard": ("main", False)}


class Private(Exception):
    """The deck is private on DeckCheck (403): nothing to read, nothing wrong here."""


def headers():
    """`X-Api-Key` when the pilot set one; `{}` otherwise.

    Read from the environment directly rather than `net.load_token`, which prints
    a hint every time the variable is unset — right for a token a feed needs,
    noise for one this endpoint does not (docs/integrations.md has the recipe)."""
    key = os.environ.get(config.DECKCHECK_TOKEN_ENV)
    return {config.DECKCHECK_TOKEN_HEADER: key} if key else {}


def _request(deck_id=None, deck_url=None):
    url = config.DECKCHECK_API_BASE + config.DECKCHECK_DECK_PATH
    params = {"deck_id": deck_id} if deck_id else {"deck_url": deck_url}
    return url, params


def fetch(deck_id=None, deck_url=None, session=None):
    """The deck document, through `net` (retried, cached for `DECKCHECK_TTL_S`)."""
    if not (deck_id or deck_url):
        raise ValueError("fetch needs a deck_id or a deck_url")
    url, params = _request(deck_id, deck_url)
    try:
        return net.get_json(url, params=params, headers=headers(), service=SERVICE,
                            ttl_s=config.DECKCHECK_TTL_S, session=session)
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        what = deck_id or deck_url
        if status == 403:
            raise Private(f"DeckCheck deck {what} is private — set it to public or "
                          f"unlisted on DeckCheck to read it back") from exc
        if status == 404:
            raise SystemExit(f"deckcheck: no deck {what} on DeckCheck (404) — "
                             f"re-check the link with `deck-link <slug> list`") from exc
        raise


def read_at(deck_id=None, deck_url=None):
    """When the answer was actually fetched — the cache record's `fetched_at`,
    or None. A cached answer is dated by its fetch, never by the read."""
    url, params = _request(deck_id, deck_url)
    path = net.cache_path(SERVICE, net.cache_key("GET", url, params=params,
                                                 headers=headers()))
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fetched_at")
    except (FileNotFoundError, ValueError, AttributeError):
        return None


def _cards(board):
    """DeckCheck keys board cards by `deckCardId`; a list is tolerated too."""
    cards = (board or {}).get("cards") or {}
    return list(cards.values()) if isinstance(cards, dict) else list(cards)


def project(doc):
    """The document's boards as `fetch_deck.parse_decklist` entries, so
    `check_in.render_decklist` is the one writer of the list's form."""
    entries = []
    for name, board in ((doc or {}).get("boards") or {}).items():
        where, is_cmdr = BOARDS.get(name, ("side", False))
        for e in _cards(board):
            card = e.get("card") or {}
            if not card.get("name"):
                continue
            entry = {"name": card["name"], "quantity": int(e.get("quantity") or 1),
                     "board": where, "is_commander": is_cmdr, "foil": bool(e.get("isFoil"))}
            if card.get("setCode") and card.get("collectorNumber"):
                entry["set"] = str(card["setCode"]).lower()
                entry["collector_number"] = str(card["collectorNumber"])
            entries.append(entry)
    return entries


def _key(entries):
    """Content identity: (name, commander?) -> copies — `deck_versions._version_key`'s
    rule, applied to entries rather than to a blob."""
    out = {}
    for e in entries:
        if e.get("board", "main") != "main":
            continue
        k = (e["name"], bool(e.get("is_commander")))
        out[k] = out.get(k, 0) + int(e.get("quantity") or 1)
    return out


def _version_matching(slug, text):
    """The newest committed version whose list IS this text, or None (no repo,
    no match). Git, swallowed whole — absent means absent."""
    try:
        from manamap.pilot import deck_versions
        want = deck_versions._version_key(text)
        for v in reversed(deck_versions.versions(slug)):
            if deck_versions._version_key(deck_versions.blob_at(slug, v)) == want:
                return v["version"]
    except Exception:                                   # noqa: BLE001
        return None
    return None


def compare(slug, entries, text=None):
    """Their copy against our lists. `pull` / `add` are what checking THEIR list
    in would do to OUR working list (the words `check_in.analyze` uses): `pull`
    is in ours and not in theirs, `add` the reverse, in copies."""
    path = deck_dir(slug) / "decklist.txt"
    ours = parse_mainboard(path.read_text(encoding="utf-8")) if path.exists() else []
    old, new = check_in._copies(ours), check_in._copies(e for e in entries
                                                        if e.get("board", "main") == "main")
    pull = {n: old[n] - new.get(n, 0) for n in old if old[n] > new.get(n, 0)}
    add = {n: new[n] - old.get(n, 0) for n in new if new[n] > old.get(n, 0)}
    text = text or check_in.render_decklist(entries)
    version = _version_matching(slug, text)
    sleeved = None
    try:
        from manamap.pilot import deck_versions
        lock = deck_versions.paper(slug)
        if lock and lock.get("version"):
            sleeved = version == lock["version"] if version is not None else False
    except Exception:                                   # noqa: BLE001
        sleeved = None
    return {"pull": pull, "add": add, "version": version,
            "matches": {"working": not pull and not add, "sleeved": sleeved},
            "working_sha256": decklist_sha256(slug)}


def report(slug, url=None, session=None):
    """One read of the copy, compared and dated."""
    link = deck_link.load(slug).get(SERVICE) or {}
    if url:
        deck_id = deck_link.parse_url(SERVICE, url)
    elif link.get("id"):
        deck_id, url = link["id"], link.get("url")
    else:
        raise SystemExit(f"{slug} has no DeckCheck link — `manamap pilot deck-link {slug} "
                         f"deckcheck <url>` after pasting it, or pass --url")
    doc = fetch(deck_id=deck_id, session=session)
    entries = project(doc)
    text = check_in.render_decklist(entries)
    cmp = compare(slug, entries, text)
    in_sync = cmp["matches"]["sleeved"] if cmp["matches"]["sleeved"] is not None \
        else cmp["matches"]["working"]
    return {"service": SERVICE, "slug": slug, "id": deck_id,
            "url": doc.get("publicUrl") or url,
            "read_at": read_at(deck_id=deck_id),
            "source_updated_at": doc.get("updatedAtUtc"),
            "source_name": doc.get("name"), "visibility": doc.get("visibility"),
            "cards": sum(e["quantity"] for e in entries if e.get("board", "main") == "main"),
            "pull": cmp["pull"], "add": cmp["add"], "matches": cmp["matches"],
            "version": cmp["version"], "in_sync": in_sync,
            "linked_version": link.get("version"), "private": False, "text": text}


def format_report(r):
    lines = [f"DECKCHECK — {r['slug']}: {r['url']}",
             f"  read {r['read_at'] or 'now'} · their copy last edited "
             f"{r['source_updated_at'] or 'unknown'} · {r['cards']} cards"
             + (f" · {r['visibility']}" if r.get("visibility") else "")]
    v = r["version"]
    which = (f"V{v}" if v else "no committed version")
    m = r["matches"]
    if m["sleeved"] is True:
        lines.append(f"  IN SYNC with the sleeved list ({which})")
    elif m["sleeved"] is False:
        lines.append(f"  BEHIND the sleeved list — their copy is {which}")
    elif m["working"]:
        lines.append(f"  matches the working list ({which}); the deck is not locked in paper")
    else:
        lines.append(f"  differs from the working list ({which})")
    for name, n in sorted(r["pull"].items()):
        lines.append(f"    - {n} {name}    (here, not on DeckCheck)")
    for name, n in sorted(r["add"].items()):
        lines.append(f"    + {n} {name}    (on DeckCheck, not here)")
    if r["pull"] or r["add"]:
        lines.append(f"  to bring DeckCheck level: `manamap pilot deck-export {r['slug']} "
                     f"--format moxfield`, paste, then `deck-link {r['slug']} deckcheck <url>`")
        lines.append(f"  to take THEIR list instead: `manamap pilot deckcheck {r['slug']} "
                     f"--text | manamap pilot check-in {r['slug']} --from -`")
    return "\n".join(lines)


def main(args):
    try:
        r = report(args.slug, url=getattr(args, "url", None))
    except net.Offline as exc:
        print(f"  deckcheck: the network could not be had — {exc}")
        sys.exit(1)
    except Private as exc:
        print(f"  deckcheck: {exc}")
        sys.exit(1)
    if getattr(args, "text", False):
        sys.stdout.write(r["text"])
        return
    if getattr(args, "as_json", False):
        print(json.dumps(r, indent=2, sort_keys=True))
    else:
        print(format_report(r))
