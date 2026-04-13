"""Step 7: Download Commander Spellbook combo data."""

import json
import time

import requests

from config import (
    COMBOS_API_URL,
    COMBOS_META_PATH,
    COMBOS_RAW_PATH,
    DATA_DIR,
    USER_AGENT,
)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

PAGE_LIMIT = 100
REQUEST_DELAY = 0.2  # 200ms between requests


def is_up_to_date():
    """Check sidecar metadata to skip re-download."""
    if not COMBOS_META_PATH.exists() or not COMBOS_RAW_PATH.exists():
        return False
    return True


def download_all_combos():
    """Paginate through all combo variants following 'next' links."""
    all_results = []
    url = COMBOS_API_URL
    params = {"format": "json", "limit": PAGE_LIMIT}
    page = 0

    while url:
        resp = SESSION.get(url, params=params)
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


def save_meta(count):
    """Write sidecar metadata after successful download."""
    meta = {"count": count}
    COMBOS_META_PATH.write_text(json.dumps(meta, indent=2))


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if is_up_to_date():
        print("  combos_raw.json already exists — skipping download.")
        print("  (Delete data/combos_raw.json to force re-download.)")
        return

    print("Downloading combo variants from Commander Spellbook...")
    combos = download_all_combos()

    with open(COMBOS_RAW_PATH, "w") as f:
        json.dump(combos, f, separators=(",", ":"))
    print(f"  Saved {len(combos):,} combos to {COMBOS_RAW_PATH}")

    size_mb = COMBOS_RAW_PATH.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    save_meta(len(combos))
    print("  Download complete.")


if __name__ == "__main__":
    main()
