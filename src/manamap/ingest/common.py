"""Shared ingest helpers.

The two raw dumps are by far the largest things this project puts on disk —
`oracle-cards.json` at 172 MB and `combos_raw.json` at 428 MB, together 600 MB
of the 871 MB in `data/`. Both are pure pipeline INPUT: each is written by one
downloader, read by one extractor, and never touched again. Neither is tracked.

They are also JSON, which means they compress about 8.5x — measured on this
repo's actual files, 600 MB becomes roughly 70 MB. Storing them compressed is
the single largest disk saving available here and it costs one function, because
`json.load` does not care what kind of file object it is handed.

Two deliberate properties:

  **Reading falls back to the uncompressed sibling.** A repo that already has
  `combos_raw.json` keeps working with no migration step and no re-download; the
  gzip path simply wins when both exist.

  **Writing removes the stale sibling.** Otherwise a re-download leaves both on
  disk and the change costs space instead of saving it.

RAM is deliberately unaffected. `json.load` still materialises the whole document
either way — gzip streams the decompression, so peak memory is the same. This is
a disk optimisation and claiming otherwise would be wrong.
"""

import gzip


def dump_paths(path):
    """(gzip path, legacy uncompressed path) for a configured `*.json.gz` path."""
    if path.suffix == ".gz":
        return path, path.with_suffix("")      # foo.json.gz -> foo.json
    return path.with_suffix(path.suffix + ".gz"), path


def resolve_dump(path):
    """The dump that actually exists, gzip preferred. Returns None if neither does."""
    gz, legacy = dump_paths(path)
    if gz.exists():
        return gz
    if legacy.exists():
        return legacy
    return None


def dump_exists(path):
    return resolve_dump(path) is not None


def open_dump(path, mode="rt"):
    """Open a raw dump, transparently handling gzip.

    On read, whichever of the two forms exists is opened. On write, always gzip,
    and the uncompressed sibling is removed so the two cannot both linger.
    """
    if "w" in mode:
        gz, legacy = dump_paths(path)
        if legacy.exists():
            legacy.unlink()
        return gzip.open(gz, mode)
    found = resolve_dump(path)
    if found is None:
        raise FileNotFoundError(f"{path} not found (nor its uncompressed sibling)")
    if found.suffix == ".gz":
        return gzip.open(found, mode)
    return open(found, mode.replace("t", "") or "r")


def dump_size_mb(path):
    """Size on disk of whichever form exists, or 0.0."""
    found = resolve_dump(path)
    return found.stat().st_size / (1024 * 1024) if found else 0.0
