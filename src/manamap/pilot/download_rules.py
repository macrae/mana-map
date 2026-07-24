"""Pilot: download the Magic Comprehensive Rules TXT (idempotent).

Mirrors the sidecar pattern of ingest/download.py. The CR file has no catalog
API, so the sidecar is keyed on the effective date parsed from the file header
plus a content hash. The source TXT is historically windows-1252 encoded; we
decode it explicitly and store normalized UTF-8.
"""

import hashlib
import json
import re

import requests

from manamap.config import CR_RULES_META_PATH, CR_RULES_PATH, CR_RULES_URL, RULES_DIR, USER_AGENT

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

EFFECTIVE_DATE_RE = re.compile(r"effective as of (\w+ \d{1,2}, \d{4})", re.IGNORECASE)


def parse_effective_date(text):
    """Extract the 'effective as of <date>' stamp from the CR header, or None."""
    match = EFFECTIVE_DATE_RE.search(text[:4000])
    return match.group(1) if match else None


def is_up_to_date(sha256):
    """True if the sidecar records the same content hash."""
    if not CR_RULES_META_PATH.exists():
        return False
    with open(CR_RULES_META_PATH) as f:
        meta = json.load(f)
    return meta.get("sha256") == sha256


def save_meta(effective_date, sha256):
    with open(CR_RULES_META_PATH, "w") as f:
        json.dump(
            {"effective_date": effective_date, "sha256": sha256, "source_url": CR_RULES_URL},
            f,
            indent=2,
        )


def main(args=None):
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Comprehensive Rules from {CR_RULES_URL} ...")
    resp = SESSION.get(CR_RULES_URL, timeout=60)
    if resp.status_code == 404:
        raise SystemExit(
            "CR_RULES_URL is stale (404). Get the current TXT link from "
            "https://magic.wizards.com/en/rules and update CR_RULES_URL in "
            "src/manamap/config.py."
        )
    resp.raise_for_status()

    # Recent CR files are UTF-8; older ones were windows-1252. Try in order.
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("cp1252")
    # Normalize line endings; keep everything else verbatim.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if CR_RULES_PATH.exists() and is_up_to_date(sha256):
        print("  Already up to date — skipping write.")
        return

    effective_date = parse_effective_date(text)
    CR_RULES_PATH.write_text(text, encoding="utf-8")
    save_meta(effective_date, sha256)
    size_mb = CR_RULES_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {CR_RULES_PATH} ({size_mb:.1f} MB, effective {effective_date})")


if __name__ == "__main__":
    main()
