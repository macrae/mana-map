"""Pilot: form-check `links.json` — where a deck lives on other sites.

FORM ONLY, NEVER REACHABILITY. Each key is a service `deck_link.SERVICES` knows; each
value carries `url`, `id` and `as_of` (and an optional string `note`, and the
optional "what I pasted" stamp `decklist_sha256` / `version`); the URL is https on
the service's host allow-list and in its deck-URL form; the id matches
`[A-Za-z0-9_-]+` and is the id the URL carries; `as_of` is an ISO date. Whether the
deck behind the URL still exists is not checked here and must not be: Moxfield 403s
every server-side request and DeckCheck's read API is a separate, optional command
(docs/integrations.md), and a gate that needs the network is a gate that gets
switched off.
"""
import re
from datetime import date
from urllib.parse import urlsplit

from manamap.pilot.common import deck_dir, load_json, report_errors
from manamap.pilot.deck_link import ARTIFACT, ID_RE, SERVICES

REQUIRED = ("url", "id", "as_of")
#: `decklist_sha256` / `version` are the "what I pasted" stamp (2026-10-17):
#: optional, because a link recorded before the stamp existed, or on an
#: uncommitted list, carries none — and a missing stamp is reported by `drift`
#: as "re-link", never invented.
ALLOWED = set(REQUIRED) | {"note", "decklist_sha256", "version"}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def validate(doc):
    errors = []
    if not isinstance(doc, dict) or not doc:
        return [f"{ARTIFACT} is not a non-empty object keyed by service"]
    for service, entry in doc.items():
        spec = SERVICES.get(service)
        if spec is None:
            errors.append(f"unknown service {service!r} (known: {', '.join(sorted(SERVICES))})")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{service} is not an object")
            continue
        for k in REQUIRED:
            if k not in entry:
                errors.append(f"{service} lacks {k!r}")
        for k in sorted(set(entry) - ALLOWED):
            errors.append(f"{service} carries an unknown key {k!r}")
        if "note" in entry and not isinstance(entry["note"], str):
            errors.append(f"{service}.note is not a string")
        url, deck_id = entry.get("url"), entry.get("id")
        if url is not None:
            parts = urlsplit(str(url))
            if parts.scheme != "https" or parts.hostname not in spec["hosts"]:
                errors.append(f"{service}.url {url!r} is not https on "
                              f"{', '.join(sorted(spec['hosts']))}")
            else:
                m = spec["url_re"].match(str(url))
                if not m:
                    errors.append(f"{service}.url {url!r} is not a {service} deck URL "
                                  f"(e.g. {spec['example']})")
                elif deck_id is not None and m.group(1) != deck_id:
                    errors.append(f"{service}.id {deck_id!r} is not the id the url "
                                  f"carries ({m.group(1)!r})")
        if deck_id is not None and not ID_RE.match(str(deck_id)):
            errors.append(f"{service}.id {deck_id!r} does not match [A-Za-z0-9_-]+")
        if "as_of" in entry:
            try:
                date.fromisoformat(str(entry["as_of"]))
            except (TypeError, ValueError):
                errors.append(f"{service}.as_of {entry['as_of']!r} is not an ISO date")
        if "decklist_sha256" in entry and not SHA_RE.match(str(entry["decklist_sha256"])):
            errors.append(f"{service}.decklist_sha256 {entry['decklist_sha256']!r} "
                          f"is not a sha256")
        if "version" in entry:
            v = entry["version"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                errors.append(f"{service}.version {v!r} is not a positive integer")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / ARTIFACT
    if not path.is_file():
        print(f"{slug}: no {ARTIFACT} — not published anywhere (absent means absent)")
        return
    doc = load_json(path)
    report_errors(f"{slug} — {ARTIFACT}", validate(doc))
    print(f"OK   {slug} — {ARTIFACT}: {', '.join(sorted(doc))}")
