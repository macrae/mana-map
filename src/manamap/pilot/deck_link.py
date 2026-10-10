"""Where a deck lives on other sites — recorded by hand, in one file.

`manamap pilot deck-link <slug> moxfield <url> [--note "…"]`
`manamap pilot deck-link <slug> moxfield --remove`
`manamap pilot deck-link <slug> list`

ONE HOME: `data/decks/<slug>/links.json`, `{"moxfield": {"url", "id", "as_of",
"note"?}}`, keyed by service so another site is a new key, not a new file. The deck
manifest carries it to the pages (`deck_manifest.gather_entries`, `links`), which is
how the workbench card and the deck header link out.

BY HAND, BECAUSE THERE IS NO OTHER WAY. Moxfield has no API that would tell us the
URL a pasted deck landed at, and its Cloudflare front 403s every server-side request
(docs/integrations.md, "Moxfield"). So the pilot exports (`deck-export --format
moxfield`), pastes, saves, and records the URL here. This command never fetches the
URL — it checks only its FORM: the host on the allow-list, a deck id parsed out of
the path, `as_of` today. Whether the deck behind it still exists, or still matches
the list, is the reader's to find out by opening it; `as_of` says when it was true.

NOT A LIFECYCLE STAGE. Optional and hand-written, like `protected.json`: absent means
the deck has not been published anywhere, which is a fact, not a todo. Removing the
last link deletes the file rather than leaving `{}` behind.
"""

import json
import re
from datetime import date

from manamap.pilot.common import deck_dir, load_json

ARTIFACT = "links.json"

#: service -> its URL form. The id group is the one thing a link must carry;
#: `validate_links` holds every stored link to the same pattern.
SERVICES = {
    "moxfield": {
        "hosts": frozenset({"moxfield.com", "www.moxfield.com"}),
        "url_re": re.compile(r"^https://(?:www\.)?moxfield\.com/decks/([A-Za-z0-9_-]+)/?$"),
        "example": "https://moxfield.com/decks/AbC123xyz",
    },
}

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def path(slug):
    return deck_dir(slug) / ARTIFACT


def load(slug):
    """The links, or {} when the deck has none (no file)."""
    doc = load_json(path(slug), None)
    return doc if isinstance(doc, dict) else {}


def parse_url(service, url):
    """The deck id in `url`, or SystemExit naming the form a link must have."""
    spec = SERVICES.get(service)
    if spec is None:
        raise SystemExit(f"deck-link: unknown service {service!r} — one of "
                         f"{', '.join(sorted(SERVICES))}")
    m = spec["url_re"].match(str(url or "").strip())
    if not m:
        raise SystemExit(f"deck-link: {url!r} is not a {service} deck URL — it must be "
                         f"https://moxfield.com/decks/<id> (or www.), the id letters, "
                         f"digits, '_' or '-', no query string; e.g. {spec['example']}")
    return m.group(1)


def _write(slug, doc):
    p = path(slug)
    if not doc:
        if p.exists():
            p.unlink()
        return
    p.write_text(json.dumps(dict(sorted(doc.items())), indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")


def set_link(slug, service, url, note=None, today=None):
    """Record `url` for `service`; returns the stored entry."""
    deck_id = parse_url(service, url)
    entry = {"url": str(url).strip().rstrip("/"), "id": deck_id,
             "as_of": (today or date.today()).isoformat()}
    if note:
        entry["note"] = str(note)
    doc = load(slug)
    doc[service] = entry
    _write(slug, doc)
    return entry


def remove_link(slug, service):
    """Drop `service`'s link; True when there was one. The file goes with the last."""
    if service not in SERVICES:
        raise SystemExit(f"deck-link: unknown service {service!r} — one of "
                         f"{', '.join(sorted(SERVICES))}")
    doc = load(slug)
    had = doc.pop(service, None) is not None
    _write(slug, doc)
    return had


def main(args):
    slug = args.slug
    action = args.action
    if action == "list":
        doc = load(slug)
        if not doc:
            print(f"{slug}: no links — publish with `manamap pilot deck-export {slug} "
                  f"--format moxfield`, paste it into Moxfield, then "
                  f"`manamap pilot deck-link {slug} moxfield <url>`")
            return
        for service, e in sorted(doc.items()):
            note = f"  — {e['note']}" if e.get("note") else ""
            print(f"{service:10s} {e.get('url')}  (as of {e.get('as_of')}){note}")
        return
    if getattr(args, "remove", False):
        if remove_link(slug, action):
            print(f"{slug}: removed the {action} link")
        else:
            print(f"{slug}: no {action} link to remove")
        return
    if not getattr(args, "url", None):
        raise SystemExit(f"deck-link {slug} {action} needs a URL (or --remove)")
    e = set_link(slug, action, args.url, note=getattr(args, "note", None))
    print(f"{slug}: {action} -> {e['url']} (id {e['id']}, as of {e['as_of']})")
    print(f"  WROTE data/decks/{slug}/{ARTIFACT}; `manamap pilot build-index` puts it "
          f"on the pages")
