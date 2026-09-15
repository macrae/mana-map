"""Pilot: download Scryfall's card-rulings dump (idempotent on CONTENT).

The official WotC / Gatherer rulings for every card, as Scryfall serves them:
a gzipped JSONL bulk file, one ruling per line, keyed by `oracle_id`. It is an
INPUT to the resolve-stack loop — a ruling tells the resolver and the checker
which Comprehensive Rules rule to look for — and never a citation.

Reuses `ingest/download.py`'s catalog walk, streaming write and sidecar rather
than copying them. Two idempotence gates, because Scryfall re-stamps the bulk
file's `updated_at` EVERY DAY whether or not a ruling changed:

  1. the catalog `updated_at` matches the sidecar -> nothing is fetched;
  2. it does not, but the sha over the DECOMPRESSED content matches -> the
     file is not rewritten (only the sidecar's `updated_at` moves).

The second is `download_rules.py`'s discipline — skip the WRITE, not merely the
parse — and it is what keeps the daemon's `mtime_memo` of the parsed rulings
warm and every `rulings:scenario` cache digest stable across a no-op refresh.
"""

import gzip
import hashlib
import json
import os

from manamap import config
from manamap.ingest.download import download_file, get_bulk_data_info, save_meta


def content_sha256(path):
    """(sha256 over the decompressed bytes, line count) of a `*.jsonl.gz`.

    Streamed, never materialised: 5 MB compressed is ~30 MB of JSON.
    """
    digest = hashlib.sha256()
    lines = 0
    with gzip.open(path, "rb") as f:
        for line in f:
            digest.update(line)
            lines += 1
    return digest.hexdigest(), lines


def read_meta():
    """The sidecar, or {} when there is none."""
    if not config.RULINGS_META_PATH.exists():
        return {}
    return json.loads(config.RULINGS_META_PATH.read_text())


def main(args=None):
    force = bool(getattr(args, "force", False))
    config.RULINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching bulk data catalog...")
    download_uri, updated_at = get_bulk_data_info(config.BULK_RULINGS_TYPE)
    print(f"  Latest update: {updated_at}")

    meta = read_meta()
    if (not force and config.RULINGS_PATH.exists()
            and meta.get("updated_at") == updated_at):
        print("  Already up to date — skipping download.")
        return

    tmp = config.RULINGS_DIR / "rulings.new.jsonl.gz"
    print("  Downloading card rulings from Scryfall...")
    download_file(download_uri, path=tmp)
    sha, count = content_sha256(tmp)

    if (not force and config.RULINGS_PATH.exists()
            and meta.get("content_sha256") == sha):
        # Scryfall re-cut the file; nothing in it changed. Do not touch the
        # dump — its mtime is what every reader memoises on — but record the
        # new stamp so tomorrow's run stops at gate 1.
        tmp.unlink()
        save_meta(updated_at, download_uri, meta_path=config.RULINGS_META_PATH,
                  content_sha256=sha, count=count)
        print(f"  Content unchanged ({count:,} rulings) — skipping write.")
        return

    os.replace(tmp, config.RULINGS_PATH)
    save_meta(updated_at, download_uri, meta_path=config.RULINGS_META_PATH,
              content_sha256=sha, count=count)
    size_mb = config.RULINGS_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {config.RULINGS_PATH} ({size_mb:.1f} MB, {count:,} rulings)")


if __name__ == "__main__":
    main()
