"""Step 1: Download Scryfall Oracle Cards bulk data."""

import json
import sys

import requests

from config import (
    BULK_DATA_TYPE,
    BULK_DATA_URL,
    DATA_DIR,
    DOWNLOAD_META_PATH,
    RAW_JSON_PATH,
    USER_AGENT,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT


def get_bulk_data_info():
    """Fetch the oracle_cards download URI and updated_at from Scryfall."""
    resp = SESSION.get(BULK_DATA_URL)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == BULK_DATA_TYPE:
            return entry["download_uri"], entry["updated_at"]
    raise ValueError(f"No bulk data entry found for type '{BULK_DATA_TYPE}'")


def is_up_to_date(updated_at):
    """Check sidecar metadata to see if we already have this version."""
    if not DOWNLOAD_META_PATH.exists():
        return False
    meta = json.loads(DOWNLOAD_META_PATH.read_text())
    return meta.get("updated_at") == updated_at


def download_file(url):
    """Stream-download a file with progress reporting."""
    resp = SESSION.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(RAW_JSON_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Downloading: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  Downloading: {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()


def save_meta(updated_at, download_uri):
    """Write sidecar metadata after successful download."""
    meta = {"updated_at": updated_at, "download_uri": download_uri}
    DOWNLOAD_META_PATH.write_text(json.dumps(meta, indent=2))


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching bulk data catalog...")
    download_uri, updated_at = get_bulk_data_info()
    print(f"  Latest update: {updated_at}")

    if RAW_JSON_PATH.exists() and is_up_to_date(updated_at):
        print("  Already up to date — skipping download.")
        return

    print(f"  Downloading oracle cards from Scryfall...")
    download_file(download_uri)
    save_meta(updated_at, download_uri)
    print("  Download complete.")


if __name__ == "__main__":
    main()
