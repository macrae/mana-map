"""Step 1: Download Scryfall Oracle Cards bulk data."""

import json

import requests

from manamap.ingest.common import dump_exists, dump_paths, open_dump
from manamap.config import (
    BULK_DATA_TYPE,
    BULK_DATA_URL,
    DATA_DIR,
    DOWNLOAD_META_PATH,
    RAW_JSON_PATH,
    USER_AGENT,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


def get_bulk_data_info(bulk_type=BULK_DATA_TYPE):
    """Fetch a bulk entry's download URI and updated_at from Scryfall.

    `bulk_type` defaults to the corpus (`oracle_cards`); `pilot/download_rulings`
    passes `rulings`. The defaults on every function here keep pipeline step 1
    byte-identical — the parametrisation exists so the rulings downloader can
    reuse the catalog walk, the streaming write and the sidecar rather than
    copy them.

    Scryfall migrated bulk data in August 2026: catalog entries now expose only
    `jsonl_download_uri` (a gzipped JSONL file, one card per line) — the old
    `download_uri` single-JSON-array form is gone. Prefer the legacy key if it
    ever returns; otherwise take the JSONL one. `extract` sniffs the on-disk
    format, so either shape parses downstream.
    """
    resp = SESSION.get(BULK_DATA_URL)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == bulk_type:
            uri = entry.get("download_uri") or entry.get("jsonl_download_uri")
            if not uri:
                raise ValueError(
                    f"Bulk entry '{bulk_type}' has neither download_uri nor "
                    f"jsonl_download_uri — Scryfall changed the schema again: {sorted(entry)}"
                )
            return uri, entry["updated_at"]
    raise ValueError(f"No bulk data entry found for type '{bulk_type}'")


def is_up_to_date(updated_at, meta_path=DOWNLOAD_META_PATH):
    """Check sidecar metadata to see if we already have this version."""
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("updated_at") == updated_at


def download_file(url, path=RAW_JSON_PATH):
    """Stream-download a file with progress reporting.

    Scryfall's JSONL bulk files arrive ALREADY gzipped (`*.jsonl.gz`) — those
    bytes are written verbatim to the canonical dump path, because routing them
    through `open_dump`'s write mode would gzip a gzip. A plain-JSON URL (the
    pre-2026-08 form) still streams through `open_dump`, which compresses it
    locally. Either way exactly one dump file exists afterwards: the verbatim
    path replicates `open_dump`'s delete-the-sibling rule.
    """
    resp = SESSION.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    already_gzipped = url.endswith(".gz")
    if already_gzipped:
        gz, legacy = dump_paths(path)
        if legacy.exists():
            legacy.unlink()
        sink = open(gz, "wb")
    else:
        sink = open_dump(path, "wb")

    with sink as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloading: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  Downloading: {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def save_meta(updated_at, download_uri, meta_path=DOWNLOAD_META_PATH, **extra):
    """Write sidecar metadata after successful download."""
    meta = {"updated_at": updated_at, "download_uri": download_uri, **extra}
    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching bulk data catalog...")
    download_uri, updated_at = get_bulk_data_info()
    print(f"  Latest update: {updated_at}")

    if dump_exists(RAW_JSON_PATH) and is_up_to_date(updated_at):
        print("  Already up to date — skipping download.")
        return

    print(f"  Downloading oracle cards from Scryfall...")
    download_file(download_uri)
    save_meta(updated_at, download_uri)
    print("  Download complete.")


if __name__ == "__main__":
    main()
