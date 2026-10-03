"""Step 1: Download Scryfall Oracle Cards bulk data."""

import json

import requests

from manamap.ingest.common import dump_exists, dump_paths, open_dump
from manamap.config import (
    BULK_DATA_TYPE,
    BULK_DATA_URL,
    BULK_PRINTINGS_TYPE,
    DATA_DIR,
    DOWNLOAD_META_PATH,
    FIRST_PRINTINGS_PATH,
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


def oracle_id_of(card):
    """A printing's oracle id. Reversible cards carry it on their faces only."""
    oid = card.get("oracle_id")
    if not oid:
        faces = card.get("card_faces") or []
        oid = faces[0].get("oracle_id") if faces else None
    return oid


def reduce_first_printings(cards):
    """{oracle_id: earliest released_at} over an iterable of printings.

    PAPER printings decide where a card has any: an MTGO-only cube or an Arena
    remaster can predate the cardboard, and "first released" here means the card
    a pilot could first have sleeved. A card with only digital printings (Alchemy)
    falls back to its earliest digital one rather than going absent. ISO dates
    compare correctly as strings, so `min` needs no parsing.
    """
    paper, digital = {}, {}
    for card in cards:
        oid, date = oracle_id_of(card), card.get("released_at")
        if not oid or not date:
            continue
        bucket = digital if card.get("digital") else paper
        if date < bucket.get(oid, "9999"):
            bucket[oid] = date
    return {**digital, **paper}


def stream_jsonl_gz(url):
    """Yield each card object from a gzipped JSONL URL without touching disk."""
    import gzip
    import io

    resp = SESSION.get(url, stream=True)
    resp.raise_for_status()
    resp.raw.decode_content = False
    with gzip.GzipFile(fileobj=resp.raw) as gz:
        for n, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8"), 1):
            if line.strip():
                yield json.loads(line)
            if n % 20000 == 0:
                print(f"\r  Printings read: {n:,}", end="", flush=True)
    print()


def refresh_first_printings(path=FIRST_PRINTINGS_PATH):
    """Step 1's second half: the earliest printing of every card, by oracle id."""
    uri, updated_at = get_bulk_data_info(BULK_PRINTINGS_TYPE)
    if path.exists() and json.loads(path.read_text()).get("updated_at") == updated_at:
        print("  First printings up to date — skipping.")
        return
    print("  Streaming every printing (default_cards) for first-release dates...")
    first = reduce_first_printings(stream_jsonl_gz(uri))
    path.write_text(json.dumps(
        {"updated_at": updated_at, "source": uri, "first_released_at": first},
        separators=(",", ":"),
    ))
    print(f"  {len(first):,} oracle ids -> {path.name}")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching bulk data catalog...")
    download_uri, updated_at = get_bulk_data_info()
    print(f"  Latest update: {updated_at}")

    if dump_exists(RAW_JSON_PATH) and is_up_to_date(updated_at):
        print("  Already up to date — skipping download.")
    else:
        print(f"  Downloading oracle cards from Scryfall...")
        download_file(download_uri)
        save_meta(updated_at, download_uri)
        print("  Download complete.")

    # Independent of the oracle sidecar: an up-to-date dump from before this
    # existed must still get its first-release dates.
    refresh_first_printings()


if __name__ == "__main__":
    main()
