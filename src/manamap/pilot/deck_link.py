"""Where a deck lives on other sites — recorded by hand, in one file.

`manamap pilot deck-link <slug> moxfield|deckcheck <url> [--note "…"]`
`manamap pilot deck-link <slug> moxfield|deckcheck --remove`
`manamap pilot deck-link <slug> list`

ONE HOME: `data/decks/<slug>/links.json`, `{"<service>": {"url", "id", "as_of",
"note"?, "decklist_sha256"?, "version"?}}`, keyed by service so another site is a
new key, not a new file. The deck manifest carries it to the pages
(`deck_manifest.gather_entries`, `links`), which is how the workbench card and the
deck header link out.

BY HAND, BECAUSE NEITHER SITE TAKES A PUSH. Moxfield has no API and its Cloudflare
front 403s every server-side request; DeckCheck's API is read-only (both in
docs/integrations.md). So the pilot exports (`deck-export --format moxfield`),
pastes, saves, and records the URL here. This command never fetches the URL — it
checks only its FORM: the host on the allow-list, a deck id parsed out of the path,
`as_of` today — and STAMPS the list as pasted (its sha, and the committed version it
is), which is what `drift` reads back to say whether the copy over there is behind
the sleeved list. Whether the deck behind the URL still exists is the reader's to
find out by opening it (or, for DeckCheck, `manamap pilot deckcheck <slug>`).

NOT A LIFECYCLE STAGE. Optional and hand-written, like `protected.json`: absent means
the deck has not been published anywhere, which is a fact, not a todo. Removing the
last link deletes the file rather than leaving `{}` behind.
"""

import json
import re
from datetime import date

from manamap.pilot.common import deck_dir, decklist_sha256, load_json

ARTIFACT = "links.json"

#: service -> its URL form. The id group is the one thing a link must carry;
#: `validate_links` holds every stored link to the same pattern.
SERVICES = {
    "moxfield": {
        "label": "Moxfield",
        "hosts": frozenset({"moxfield.com", "www.moxfield.com"}),
        "url_re": re.compile(r"^https://(?:www\.)?moxfield\.com/decks/([A-Za-z0-9_-]+)/?$"),
        "example": "https://moxfield.com/decks/AbC123xyz",
        "form": "https://moxfield.com/decks/<id>",
    },
    # DeckCheck (2026-10-17): a read-only public API (`pilot/deckcheck.py`) and
    # still a paste to publish, so the link is recorded by hand exactly as
    # Moxfield's is. Both `/deck/<id>` and `/decks/<id>` are accepted until the
    # live `publicUrl` form is verified against a real deck (docs/integrations.md).
    "deckcheck": {
        "label": "DeckCheck",
        "hosts": frozenset({"deckcheck.co", "www.deckcheck.co"}),
        "url_re": re.compile(r"^https://(?:www\.)?deckcheck\.co/decks?/([A-Za-z0-9_-]+)/?$"),
        "example": "https://deckcheck.co/deck/AbC123xyz",
        "form": "https://deckcheck.co/deck/<id>",
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
                         f"{spec['form']} (or www.), the id letters, "
                         f"digits, '_' or '-', no query string; e.g. {spec['example']}")
    return m.group(1)


def _version_of(slug, sha):
    """The committed version number whose list has byte-sha `sha`, or None.

    A git walk, done lazily and swallowed whole: a deck outside a repo, or an
    uncommitted list, simply has no version to stamp — absent means absent."""
    if not sha:
        return None
    try:
        from manamap.pilot import deck_versions
        for v in deck_versions.versions(slug):
            if sha in (v.get("decklist_sha256s") or ()):
                return v["version"]
    except Exception:                                   # noqa: BLE001 - no repo, no git
        return None
    return None


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
    # THE CLAIM "THIS IS WHAT I PASTED" (2026-10-17): the list's sha as it
    # stands, and the committed version that list is, when it is one. `drift`
    # reads these back to say whether the copy on the other site is behind the
    # sleeved list — by VERSION, so a re-render is never drift.
    sha = decklist_sha256(slug)
    if sha:
        entry["decklist_sha256"] = sha
        v = _version_of(slug, sha)
        if v:
            entry["version"] = v
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
                  f"--format moxfield`, paste it on Moxfield or DeckCheck, then "
                  f"`manamap pilot deck-link {slug} moxfield|deckcheck <url>`")
            return
        for service, e in sorted(doc.items()):
            note = f"  — {e['note']}" if e.get("note") else ""
            stamp = (f"  V{e['version']}" if e.get("version")
                     else "  (no version stamp)" if not e.get("decklist_sha256") else "")
            print(f"{service:10s} {e.get('url')}  (as of {e.get('as_of')}){stamp}{note}")
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
    if e.get("version"):
        print(f"  stamped V{e['version']} {e['decklist_sha256'][:12]} — the list as pasted")
    elif e.get("decklist_sha256"):
        print(f"  stamped {e['decklist_sha256'][:12]}, no version: the list is uncommitted "
              f"— commit, then re-link to stamp one")
    print(f"  WROTE data/decks/{slug}/{ARTIFACT}; `manamap pilot deck-info {slug} --write` "
          f"and `manamap pilot build-index` put it on the pages")
