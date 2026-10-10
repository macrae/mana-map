"""Pilot: a deck's (or a branch's) card prices as DATED EVIDENCE — `prices.json`.

A price is the one figure on this bench that nobody here measures and everybody
quotes: an agent that says "about $4" from memory is guessing, and a live lookup in
the middle of an answer is a number with no date on it. So prices live in one
artifact beside the list, stamped `as_of` and `source`, written only by this
command, and every reader — `net-change`'s bill, the deck page's cover, the Build
panel, the card-scout and Jarvis — quotes THAT file, with its date, or says there
is none. Absent means absent, never zero (`docs/gotchas-bench.md`).

Two sources, one shape:

- **Mana Pool** (`source: "manapool"`, the default): the PUBLIC singles price
  feed (`config.MANAPOOL_PRICES_PATH`, verified against the OpenAPI spec
  2026-10-09 — no token needed), keyed by `scryfall_id`, with NM / LP+ / foil
  cents and a listing URL; the feed's own timestamp is kept as `feed_as_of`. A
  token, when `MANAPOOL_TOKEN` and `MANAPOOL_EMAIL` are set, rides along in the
  headers and changes nothing. A 400/404 on the feed prints one line and falls
  through to Scryfall rather than failing the command.
- **Scryfall** (`source: "scryfall"`): the same `/cards/collection` POST
  `fetch-deck` makes, read for `prices.usd` / `prices.usd_foil` — the fallback,
  and the one that resolves a `scryfall_id` for a cards.json written before that
  field existed.

Both go through `manamap.net`, so the feed is cached six hours and the collection
answer a day (`data/cache/manapool/`, `data/cache/scryfall/`), the unit tier runs
offline, and `net.Offline` is an operating condition this command reports in one
line and exits 1 on, writing nothing. NOT a lifecycle stage and NOT a regen stage:
a gate that fails when the network is down is a gate that gets switched off.
"""

import json
import os
import sys
import time
from datetime import date
from urllib.parse import urlsplit

import requests

from manamap import config, net
from manamap.pilot.common import deck_dir, load_deck_cards

ARTIFACT = "prices.json"
SOURCES = ("manapool", "scryfall")
#: The only hosts a `url` in the artifact may point at. A token can never end up
#: in a tracked file through a URL, because the validator refuses any other host.
ALLOWED_HOSTS = frozenset({"manapool.com", "www.manapool.com",
                           "scryfall.com", "api.scryfall.com"})
TOKEN_ENV = "MANAPOOL_TOKEN"
EMAIL_ENV = "MANAPOOL_EMAIL"
KEYCHAIN_SERVICE = "manamap-manapool"


# ── the two sources ──────────────────────────────────────────────────────

def manapool_headers():
    """The two auth headers, or None when either half is missing.

    The token comes through `net.load_token` (environment, with the Keychain
    recipe on stderr once per process); the email is plain environment, since
    it is not a secret. The price feed is public, so neither is required; one
    without the other sends neither.
    """
    token = net.load_token(TOKEN_ENV, keychain_service=KEYCHAIN_SERVICE)
    email = os.environ.get(EMAIL_ENV)
    if not token or not email:
        return None
    return {config.MANAPOOL_TOKEN_HEADER: token, config.MANAPOOL_EMAIL_HEADER: email}


def _feed_rows(doc):
    """The feed's rows, whatever envelope the (unverified) endpoint wraps them in."""
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for key in ("data", "prices", "singles", "results"):
            if isinstance(doc.get(key), list):
                return doc[key]
    return []


def manapool_feed(session=None, meta=None):
    """The singles price feed as `{scryfall_id: row}`, or None.

    None means "no Mana Pool source": the feed path answered 400/404. The caller
    falls through to Scryfall and the printed line says why. Any other HTTP
    failure, and `net.Offline`, propagate: a 503 after four retries is not a
    wrong path. `meta`, when a dict, receives the feed's own `as_of`.
    """
    headers = manapool_headers() or {}
    url = config.MANAPOOL_API_BASE + config.MANAPOOL_PRICES_PATH
    try:
        doc = net.get_json(url, headers=headers, service="manapool",
                           ttl_s=config.MANAPOOL_FEED_TTL_S, session=session)
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (400, 404):
            print(f"  Mana Pool answered {status} on {url} — check "
                  f"config.py (MANAPOOL_PRICES_PATH) against the API docs; using "
                  f"Scryfall prices instead.")
            return None
        raise
    if isinstance(meta, dict) and isinstance(doc, dict):
        meta["as_of"] = (doc.get("meta") or {}).get("as_of")
    feed = {}
    for row in _feed_rows(doc):
        sid = (row or {}).get("scryfall_id")
        if sid:
            feed[str(sid)] = row
    return feed


def _identifier(card):
    """The Scryfall identifier for one cards.json entry: id, then printing, then name."""
    if card.get("scryfall_id"):
        return {"id": card["scryfall_id"]}
    if card.get("set") and card.get("collector_number"):
        return {"set": card["set"], "collector_number": str(card["collector_number"])}
    return {"name": card["name"]}


def scryfall_prices(cards, session=None):
    """`/cards/collection` for every card, read for its prices.

    Returns one dict keyed THREE ways for the same values — by Scryfall id, by
    `(set, collector_number)` and by lowercased name — so a card can be found
    from whichever of the three its cards.json entry carries. Each value is
    `{usd, usd_foil, id, set, collector_number, name, uri}`; the dollar figures
    are Scryfall's strings or None. Cached a day under `data/cache/scryfall/`.
    """
    identifiers = [_identifier(c) for c in cards]
    out = {}
    for start in range(0, len(identifiers), config.SCRYFALL_BATCH_SIZE):
        batch = identifiers[start:start + config.SCRYFALL_BATCH_SIZE]
        if start > 0:
            time.sleep(config.SCRYFALL_REQUEST_DELAY_S)
        doc = net.post_json(config.SCRYFALL_COLLECTION_URL, {"identifiers": batch},
                            service="scryfall", ttl_s=config.SCRYFALL_PRICES_TTL_S,
                            session=session)
        for sc in doc.get("data") or []:
            prices = sc.get("prices") or {}
            row = {"usd": prices.get("usd"), "usd_foil": prices.get("usd_foil"),
                   "id": sc.get("id"), "set": sc.get("set"),
                   "collector_number": sc.get("collector_number"),
                   "name": sc.get("name"), "uri": sc.get("scryfall_uri")}
            if row["id"]:
                out[row["id"]] = row
            if row["set"] and row["collector_number"]:
                out[(row["set"], str(row["collector_number"]))] = row
            if row["name"]:
                out[row["name"].lower()] = row
    return out


# ── the artifact ─────────────────────────────────────────────────────────

def _cents(value):
    """Dollars-as-string (Scryfall) or cents-as-number (Mana Pool) -> int cents or None."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _dollars_to_cents(value):
    if value is None or value == "":
        return None
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return None


def _safe_url(url):
    """A URL on an allowed host, or None — never a token-bearing query string."""
    if not url or not isinstance(url, str):
        return None
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.query:
        return None
    return url


def _printing(set_code, number):
    if set_code and number:
        return f"({str(set_code).upper()}) {number}"
    return None


def _scryfall_row(card, sc):
    sid = card.get("scryfall_id") or (sc or {}).get("id")
    if not sc or (sc.get("usd") is None and sc.get("usd_foil") is None):
        return None, sid
    return {
        "scryfall_id": sid,
        "printing": _printing(sc.get("set") or card.get("set"),
                              sc.get("collector_number") or card.get("collector_number")),
        "nm_cents": _dollars_to_cents(sc.get("usd")),
        "lp_cents": None,
        "foil_cents": _dollars_to_cents(sc.get("usd_foil")),
        "url": _safe_url(sc.get("uri")),
        "quantity": int(card.get("quantity") or 1),
        "note": None if sc.get("usd") is not None else "foil listing only",
    }, sid


def _row_nm(row):
    """A feed row's NM cents, falling back to its plain `price_cents`."""
    nm = _cents(row.get("price_cents_nm"))
    return _cents(row.get("price_cents")) if nm is None else nm


def _manapool_row(card, sid, feed, by_name):
    row = feed.get(sid) if sid else None
    note = None
    if row is None:
        # The deck's exact printing is not listed; the cheapest listing of the
        # NAME is still a price, said as such.
        row = by_name.get(card["name"].lower())
        if row is None:
            return None
        note = "cheapest listed printing, not the deck's"
    nm = _row_nm(row)
    if "available_quantity" in row and (row.get("available_quantity") or 0) == 0:
        note = (note + "; " if note else "") + "out of stock"
    return {
        "scryfall_id": row.get("scryfall_id") or sid,
        "printing": _printing(row.get("set_code") or card.get("set"),
                              row.get("number") or card.get("collector_number")),
        "nm_cents": nm,
        "lp_cents": _cents(row.get("price_cents_lp_plus")),
        "foil_cents": _cents(row.get("price_cents_nm_foil")),
        "url": _safe_url(row.get("url")),
        "quantity": int(card.get("quantity") or 1),
        "note": note,
    }


def build(slug, branch=None, source="auto", session=None):
    """The artifact for a deck or a branch — see the module docstring for the shape.

    `source`: `auto` tries Mana Pool's public feed and falls through to
    Scryfall on a 400/404; `manapool` insists (and raises on one); `scryfall`
    never touches the feed. The Scryfall collection is fetched whenever any card lacks a
    `scryfall_id` (every cards.json written before 2026-10-09) or when it is the
    price source — one cached POST either way.
    """
    if source not in SOURCES + ("auto",):
        raise ValueError(f"source must be one of auto, manapool, scryfall — not {source!r}")
    doc = load_deck_cards(slug, branch)
    cards = doc.get("cards") or []

    feed, feed_meta = None, {}
    if source in ("auto", "manapool"):
        feed = manapool_feed(session=session, meta=feed_meta)
        if feed is None and source == "manapool":
            raise SystemExit(
                "no Mana Pool source: the price feed answered 400/404 (see the line "
                "above); run with --source scryfall.")
    used = "manapool" if feed is not None else "scryfall"

    need_scryfall = used == "scryfall" or any(not c.get("scryfall_id") for c in cards)
    sc_map = scryfall_prices(cards, session=session) if need_scryfall else {}

    by_name = {}
    if feed is not None:
        for row in feed.values():
            name = str(row.get("name") or "").lower()
            nm = _row_nm(row)
            if not name or nm is None:
                continue
            if name not in by_name or nm < _row_nm(by_name[name]):
                by_name[name] = row

    out_cards, missing = {}, []
    for card in cards:
        sc = (sc_map.get(card.get("scryfall_id"))
              or sc_map.get((card.get("set"), str(card.get("collector_number"))))
              or sc_map.get(card["name"].lower()))
        if used == "manapool":
            sid = card.get("scryfall_id") or (sc or {}).get("id")
            row = _manapool_row(card, sid, feed, by_name)
        else:
            row, _sid = _scryfall_row(card, sc)
        if row is None:
            missing.append(card["name"])
        else:
            out_cards[card["name"]] = row

    priced = [r for r in out_cards.values() if r["nm_cents"] is not None]
    extra = {"feed_as_of": feed_meta["as_of"]} if used == "manapool" and feed_meta.get("as_of") else {}
    return {
        "slug": slug,
        "branch": branch,
        "as_of": date.today().isoformat(),
        "source": used,
        "currency": "USD",
        "decklist_sha256": doc.get("decklist_sha256"),
        "cards": out_cards,
        # Every copy, at NM: what buying the list as sleeved would cost.
        "total_cents": sum(r["nm_cents"] * r["quantity"] for r in priced),
        # ONE copy of each priced card — the "singles" figure, where eleven
        # Islands count once.
        "total_nm_cents": sum(r["nm_cents"] for r in priced),
        "missing": sorted(missing),
        # When the SOURCE's figures were taken, beside the day we read them.
        **extra,
    }


def _dollars(cents):
    return "—" if cents is None else f"${cents / 100:,.2f}"


def print_report(doc):
    rows = sorted(((r.get("nm_cents") or 0) * r.get("quantity", 1), name, r)
                  for name, r in doc["cards"].items())
    print(f"{doc['slug']}{'@' + doc['branch'] if doc.get('branch') else ''} — prices "
          f"from {doc['source']} as of {doc['as_of']} ({doc['currency']})"
          + (f"; feed taken {doc['feed_as_of']}" if doc.get("feed_as_of") else ""))
    print(f"  {len(doc['cards'])} priced, {len(doc['missing'])} missing; total at NM "
          f"{_dollars(doc['total_cents'])} (every copy), {_dollars(doc['total_nm_cents'])} "
          f"(one of each)")
    print("  top 10 by NM price:")
    for _line, name, r in sorted(rows, key=lambda t: -(t[2].get("nm_cents") or 0))[:10]:
        qty = f" x{r['quantity']}" if r.get("quantity", 1) > 1 else ""
        note = f"   ({r['note']})" if r.get("note") else ""
        print(f"    {name[:36]:38} {_dollars(r.get('nm_cents')):>10}{qty:4}  "
              f"{r.get('printing') or '':12}{note}")
    if doc["missing"]:
        print(f"  no listing ({len(doc['missing'])}): " + ", ".join(doc["missing"][:12])
              + (" …" if len(doc["missing"]) > 12 else ""))


def main(args):
    branch = getattr(args, "branch", None)
    source = getattr(args, "source", None) or "auto"
    try:
        doc = build(args.slug, branch, source=source)
    except net.Offline as exc:
        print(f"  prices: the network could not be had — {exc}")
        sys.exit(1)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, sort_keys=True))
    else:
        print_report(doc)
    if getattr(args, "write", False):
        path = deck_dir(args.slug, branch) / ARTIFACT
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  wrote {path}")
