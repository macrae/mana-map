"""Step 7: download every combo variant from Commander Spellbook.

Two routes to the same dump, `combos_raw.json.gz`:

- **Bulk (default, since 2026-10-08)**: one gzipped file, `COMBOS_BULK_URL`, shaped
  `{timestamp, version, variants, aliases}`. It is streamed to disk byte for byte
  — the server's gzip IS our gzip — and step 8 (`process_combos.raw_variants`)
  reads `variants` out of it. The response's `ETag` / `Last-Modified` go into the
  sidecar `.combos-meta.json`, and a re-run HEADs the URL and skips the download
  while they match. Offline, the existing dump is kept (fail open) — a missing
  network must not turn a pipeline run into a missing artifact.
- **Paged (`--paged`)**: the old walk over `COMBOS_API_URL`, ~2.5 min, written as a
  bare list. Step 8 reads both shapes, so a repo holding an old paged dump keeps
  working with no migration.

`.combos-meta.json` is `{etag, last_modified, timestamp, version, count,
downloaded_at}`. The pre-2026-10 file carried `count` alone; a sidecar missing the
validators reads as "unknown", which is a refresh, never a silent skip.

Needs the network. `python -m manamap.ingest.download_combos [--force] [--paged]`;
`manamap download-combos` (and `manamap run`) take the defaults.
"""

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone

import requests

from manamap.ingest.common import dump_exists, dump_paths, dump_size_mb, open_dump
from manamap.config import (
    COMBOS_API_URL,
    COMBOS_BULK_URL,
    COMBOS_META_PATH,
    COMBOS_RAW_PATH,
    DATA_DIR,
    USER_AGENT,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

PAGE_LIMIT = 100
REQUEST_DELAY = 0.2  # 200ms between requests
HEAD_TIMEOUT = 20
STREAM_TIMEOUT = 120
CHUNK_BYTES = 1 << 20
GZIP_MAGIC = b"\x1f\x8b"

META_KEYS = ("etag", "last_modified", "timestamp", "version", "count", "downloaded_at")
#: The failures that mean "could not ask", as opposed to "asked and it changed".
NETWORK_ERRORS = (requests.ConnectionError, requests.Timeout, requests.HTTPError)


# ── the sidecar ──


def load_meta(path=None):
    """The sidecar as a dict, every `META_KEYS` key present (None when unknown).

    A missing or unreadable file is an empty record, not an error: the caller's
    question is "what do we know", and the answer may be nothing. `path` defaults
    to `COMBOS_META_PATH` at call time, so a test can point the module elsewhere.
    """
    path = COMBOS_META_PATH if path is None else path
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {key: raw.get(key) for key in META_KEYS}


def save_meta(meta, path=None):
    """Write the sidecar, every `META_KEYS` key, nothing else."""
    path = COMBOS_META_PATH if path is None else path
    record = {key: meta.get(key) for key in META_KEYS}
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def remote_signature(session=SESSION, url=None):
    """`(etag, last_modified)` of the bulk file, from one HEAD. Raises on failure."""
    url = COMBOS_BULK_URL if url is None else url
    resp = session.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT)
    resp.raise_for_status()
    return resp.headers.get("ETag"), resp.headers.get("Last-Modified")


def _validators_match(local, remote):
    """Compare on ETag when both sides carry one, else on Last-Modified.

    Neither on both sides is "cannot tell", and cannot-tell refreshes.
    """
    local_etag, local_lm = local
    remote_etag, remote_lm = remote
    if local_etag and remote_etag:
        return local_etag == remote_etag
    if local_lm and remote_lm:
        return local_lm == remote_lm
    return False


def is_up_to_date(session=SESSION):
    """Is the dump on disk the one the server would send?

    False when the dump or the sidecar is missing, when the sidecar carries no
    validator (the old `{count}` shape), or when the server's ETag / Last-Modified
    moved. True when they match — and True when the server cannot be reached,
    with a one-line warning: an offline run keeps what it has.
    """
    if not COMBOS_META_PATH.exists() or not dump_exists(COMBOS_RAW_PATH):
        return False
    meta = load_meta()
    local = (meta.get("etag"), meta.get("last_modified"))
    if local == (None, None):
        return False
    try:
        remote = remote_signature(session)
    except NETWORK_ERRORS as exc:
        print(f"  WARNING: could not check {COMBOS_BULK_URL} "
              f"({type(exc).__name__}); keeping the existing dump.")
        return True
    return _validators_match(local, remote)


# ── bulk ──


def _write_gzip_stream(resp, gz_path):
    """Stream the response body to `gz_path` as gzip, whatever the wire carried.

    The file on the server is gzip, so the bytes normally land as-is. A proxy or
    CDN that advertises `Content-Encoding: gzip` makes `requests` inflate them in
    flight; the first chunk's magic decides, and an inflated body is re-gzipped
    on the way down so the path's suffix is never a lie.
    """
    chunks = resp.iter_content(CHUNK_BYTES)
    first = b""
    for first in chunks:
        if first:
            break
    tmp = gz_path.with_name(gz_path.name + ".part")
    opener = open if first.startswith(GZIP_MAGIC) else gzip.open
    total = 0
    with opener(tmp, "wb") as f:
        if first:
            f.write(first)
            total += len(first)
        for chunk in chunks:
            if chunk:
                f.write(chunk)
                total += len(chunk)
    tmp.replace(gz_path)
    return total


def download_bulk(session=SESSION, url=None):
    """Fetch the bulk file to `COMBOS_RAW_PATH` and write the sidecar.

    Returns the meta record written. The dump is parsed once after landing, for
    its `timestamp`, `version` and variant count — step 8 parses it again, which
    is the price of a sidecar that states what is actually on disk.
    """
    from manamap.ingest.process_combos import raw_variants  # the reader owns the shape

    url = COMBOS_BULK_URL if url is None else url
    resp = session.get(url, stream=True, timeout=STREAM_TIMEOUT)
    resp.raise_for_status()
    gz, legacy = dump_paths(COMBOS_RAW_PATH)
    if legacy.exists():
        legacy.unlink()  # the same rule `open_dump` keeps: the two never both linger
    _write_gzip_stream(resp, gz)

    with open_dump(COMBOS_RAW_PATH, "rt") as f:
        doc = json.load(f)
    variants = raw_variants(doc)
    header = doc if isinstance(doc, dict) else {}
    meta = {
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified"),
        "timestamp": header.get("timestamp"),
        "version": header.get("version"),
        "count": len(variants),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return save_meta(meta)


# ── paged (the fallback) ──


def download_all_combos(session=SESSION):
    """Paginate through all combo variants following 'next' links."""
    all_results = []
    url = COMBOS_API_URL
    params = {"format": "json", "limit": PAGE_LIMIT}
    page = 0

    while url:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        all_results.extend(results)

        page += 1
        print(f"\r  Page {page}: {len(all_results):,} combos downloaded", end="", flush=True)

        url = data.get("next")
        # After the first request, 'next' is a full URL with params baked in
        params = None
        time.sleep(REQUEST_DELAY)

    print()
    return all_results


def download_paged(session=SESSION):
    """The old route: walk the API and write a bare list. No validators to keep."""
    combos = download_all_combos(session)
    with open_dump(COMBOS_RAW_PATH, "wt") as f:
        json.dump(combos, f, separators=(",", ":"))
    return save_meta({
        "count": len(combos),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


# ── entry ──


def build_parser():
    parser = argparse.ArgumentParser(
        prog="manamap download-combos",
        description="Step 7: download Commander Spellbook's combo variants.")
    parser.add_argument("--force", action="store_true",
                        help="download even when the sidecar says the dump is current")
    parser.add_argument("--paged", action="store_true",
                        help="walk the paged API instead of the bulk file (~2.5 min)")
    return parser


def main(argv=None):
    """`argv` is the flag list; the pipeline calls `main()` and gets the defaults."""
    args = build_parser().parse_args([] if argv is None else list(argv))
    DATA_DIR.mkdir(exist_ok=True)

    if not args.force and is_up_to_date():
        print("  combos_raw.json.gz is current — skipping download.")
        print("  (--force, or delete the dump, to re-download.)")
        return

    if args.paged:
        print("Downloading combo variants from Commander Spellbook (paged API)...")
        meta = download_paged()
    else:
        print(f"Downloading combo variants from {COMBOS_BULK_URL} ...")
        meta = download_bulk()
        print(f"  bulk timestamp {meta['timestamp']}, version {meta['version']}")

    print(f"  Saved {meta['count']:,} combos to {COMBOS_RAW_PATH}")
    print(f"  File size: {dump_size_mb(COMBOS_RAW_PATH):.1f} MB")
    print("  Download complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
