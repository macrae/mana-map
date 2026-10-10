"""Pilot: form-check `prices.json` — a deck's card prices, dated.

FORM ONLY, NEVER FRESHNESS. Prices are dated evidence like `deck_recon.json` and
`edhrec_cards.json`: the gate asks whether the file is well-formed and talks about
the list beside it — every name in the deck's (or branch's) cards.json, cents that
are non-negative integers or null, the totals that re-add, `missing` disjoint from
`cards`, every URL on a host that cannot carry a token, the source in the closed
vocabulary and `as_of` an ISO date. Whether a figure is CURRENT is `as_of`'s job and
the reader's; a gate that failed on a stale price would fail on every deck the
morning after it was written, and a gate that fails when the network is down is a
gate that gets switched off (`registry.py`, `validate-brief --themes`).
"""
from datetime import date, datetime
from urllib.parse import urlsplit

from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.pilot.prices import ALLOWED_HOSTS, ARTIFACT, SOURCES

REQUIRED = ("slug", "branch", "as_of", "source", "currency", "decklist_sha256",
            "cards", "total_cents", "total_nm_cents", "missing")
ROW_KEYS = ("scryfall_id", "printing", "nm_cents", "lp_cents", "foil_cents",
            "url", "quantity", "note")
CENTS_KEYS = ("nm_cents", "lp_cents", "foil_cents")


def _is_cents(v):
    return v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0)


def validate(slug, branch, doc, deck_names=None):
    """Errors for `doc`; `deck_names` is the set of names in the list it prices
    (None skips the membership check — the caller says why)."""
    errors = []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing required key {k!r}")
    if errors:
        return errors
    if doc["slug"] != slug:
        errors.append(f"slug is {doc['slug']!r} but the artifact lives in {slug}/")
    if (doc.get("branch") or None) != (branch or None):
        errors.append(f"branch is {doc.get('branch')!r} but the artifact lives in "
                      f"{'branches/' + branch if branch else 'the deck root'}")
    try:
        date.fromisoformat(str(doc["as_of"]))
    except (TypeError, ValueError):
        errors.append(f"as_of {doc['as_of']!r} is not an ISO date")
    if doc["source"] not in SOURCES:
        errors.append(f"source {doc['source']!r} is not one of {', '.join(SOURCES)}")
    if doc["currency"] != "USD":
        errors.append(f"currency {doc['currency']!r} is not USD")
    cards = doc["cards"]
    missing = doc["missing"]
    if not isinstance(cards, dict):
        errors.append("cards is not a dict keyed by card name")
        cards = {}
    if not isinstance(missing, list):
        errors.append("missing is not a list")
        missing = []
    overlap = sorted(set(missing) & set(cards))
    if overlap:
        errors.append(f"missing and cards overlap: {', '.join(overlap)}")
    if deck_names is not None:
        for name in list(cards) + list(missing):
            if name not in deck_names:
                errors.append(f"{name!r} is not in the list's cards.json")
    total, total_nm = 0, 0
    for name, row in cards.items():
        if not isinstance(row, dict):
            errors.append(f"cards[{name!r}] is not a dict")
            continue
        for k in ROW_KEYS:
            if k not in row:
                errors.append(f"cards[{name!r}] lacks {k!r}")
        for k in CENTS_KEYS:
            if not _is_cents(row.get(k)):
                errors.append(f"cards[{name!r}].{k} {row.get(k)!r} is not a non-negative "
                              f"integer or null")
        qty = row.get("quantity")
        if not (isinstance(qty, int) and not isinstance(qty, bool) and qty >= 1):
            errors.append(f"cards[{name!r}].quantity {qty!r} is not a positive integer")
            qty = 0
        url = row.get("url")
        if url is not None:
            parts = urlsplit(str(url))
            if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
                errors.append(f"cards[{name!r}].url {url!r} is not on an allowed host "
                              f"({', '.join(sorted(ALLOWED_HOSTS))})")
            elif parts.query:
                errors.append(f"cards[{name!r}].url carries a query string — a token "
                              f"could ride there")
        nm = row.get("nm_cents")
        if _is_cents(nm) and nm is not None:
            total += nm * qty
            total_nm += nm
    if doc["total_cents"] != total:
        errors.append(f"total_cents {doc['total_cents']!r} != sum of nm_cents x quantity "
                      f"({total})")
    if doc["total_nm_cents"] != total_nm:
        errors.append(f"total_nm_cents {doc['total_nm_cents']!r} != sum of nm_cents "
                      f"({total_nm})")
    if "feed_as_of" in doc:
        # Optional: the Mana Pool feed's own timestamp (ISO, `Z`-suffixed).
        stamp = str(doc["feed_as_of"] or "")
        try:
            datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"feed_as_of {doc['feed_as_of']!r} is not an ISO timestamp")
        if doc["source"] != "manapool":
            errors.append("feed_as_of is the Mana Pool feed's stamp, but source is "
                          f"{doc['source']!r}")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    branch = getattr(args, "branch", None)
    path = deck_dir(slug, branch) / ARTIFACT
    where = f"{slug}@{branch}" if branch else slug
    if not path.is_file():
        print(f"{where}: no {ARTIFACT} — prices not fetched for this list (absent means absent)")
        return
    doc = load_json(path) or {}
    cards_path = deck_dir(slug, branch) / "cards.json"
    deck_names = None
    if cards_path.is_file():
        deck_names = {c["name"] for c in (load_json(cards_path) or {}).get("cards", [])}
    errors = validate(slug, branch, doc, deck_names)
    if deck_names is None:
        errors.append(f"no cards.json beside {ARTIFACT} — nothing to check the names against")
    report_errors(f"{where} — {ARTIFACT}", errors)
    print(f"OK   {where} — {ARTIFACT} from {doc.get('source')} as of {doc.get('as_of')}: "
          f"{len(doc.get('cards') or {})} priced, {len(doc.get('missing') or [])} missing, "
          f"${(doc.get('total_cents') or 0) / 100:,.2f} at NM")
