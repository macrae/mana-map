"""Scryfall card rulings: the download, the loader, the bridge, the lookup.

Rulings are INPUT to the resolve loop and never a citation, so nothing here
touches the citation contract — `test_pilot_validate_stack.py` proving the ~54
committed artifacts still pass is the test that the contract did not move.

Everything is offline: a hand-built catalog, a hand-built gz body, config paths
patched BY STRING on `manamap.config` (the `decks_root()` docstring in
`common.py` says why patching an importing module's copy does not work).
"""

import argparse
import gzip
import json

import pytest

from manamap.ingest import download as ingest_download
from manamap.pilot import download_rulings


# ── fixtures ─────────────────────────────────────────────────────────────

RULING_LINES = [
    {"object": "ruling", "oracle_id": "aaaa-1", "source": "wotc",
     "published_at": "2020-04-17", "comment": "Brallin's middle ability gives it only one counter."},
    {"object": "ruling", "oracle_id": "aaaa-1", "source": "wotc",
     "published_at": "2019-01-01", "comment": "An earlier ruling, which sorts first by date."},
    {"object": "ruling", "oracle_id": "aaaa-1", "source": "scryfall",
     "published_at": "2021-06-01", "comment": "An editorial note from Scryfall, not WotC."},
    {"object": "ruling", "oracle_id": "bbbb-2", "source": "wotc",
     "published_at": "2018-05-05", "comment": "A ruling on a different card."},
]


def gz_body(lines=RULING_LINES):
    return gzip.compress("".join(json.dumps(l) + "\n" for l in lines).encode("utf-8"))


def catalog(updated_at, with_rulings=True):
    entries = [{"type": "oracle_cards", "updated_at": "2026-09-01T00:00:00Z",
                "jsonl_download_uri": "https://data.scryfall.io/oracle/x.jsonl.gz"}]
    if with_rulings:
        # ONLY jsonl_download_uri — that is the live catalog's shape since 2026-08.
        entries.append({"type": "rulings", "updated_at": updated_at,
                        "jsonl_download_uri": f"https://data.scryfall.io/rulings/rulings-{updated_at}.jsonl.gz"})
    return {"data": entries}


class FakeGet:
    """SESSION.get stand-in: the catalog for the bulk-data URL, a gz body for anything else."""

    def __init__(self, updated_at, body, with_rulings=True):
        self.updated_at, self.body, self.with_rulings = updated_at, body, with_rulings
        self.calls = []

    def __call__(self, url, stream=False, **kw):
        self.calls.append(url)
        get = self

        class Resp:
            headers = {"content-length": str(len(get.body))}

            def raise_for_status(self):
                pass

            def json(self):
                return catalog(get.updated_at, get.with_rulings)

            def iter_content(self, chunk_size):
                yield get.body

        return Resp()


@pytest.fixture
def rulings_dir(tmp_path, monkeypatch):
    d = tmp_path / "rulings"
    monkeypatch.setattr("manamap.config.RULINGS_DIR", d)
    monkeypatch.setattr("manamap.config.RULINGS_PATH", d / "rulings.jsonl.gz")
    monkeypatch.setattr("manamap.config.RULINGS_META_PATH", d / ".rulings-meta.json")
    return d


def run_download(monkeypatch, updated_at, body, force=False, with_rulings=True):
    fake = FakeGet(updated_at, body, with_rulings)
    monkeypatch.setattr(ingest_download.SESSION, "get", fake)
    download_rulings.main(argparse.Namespace(force=force))
    return fake


# ── the download ─────────────────────────────────────────────────────────

def test_first_run_writes_the_dump_and_a_sidecar_with_a_content_sha(rulings_dir, monkeypatch):
    from manamap import config
    fake = run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    assert config.RULINGS_PATH.exists()
    assert len(fake.calls) == 2, "the catalog, then the file"
    meta = json.loads(config.RULINGS_META_PATH.read_text())
    assert meta["updated_at"] == "2026-09-15T09:00:00Z"
    assert meta["download_uri"].endswith(".jsonl.gz")
    assert meta["count"] == len(RULING_LINES)
    assert meta["content_sha256"] == download_rulings.content_sha256(config.RULINGS_PATH)[0]
    assert not (rulings_dir / "rulings.new.jsonl.gz").exists(), "the temp file is gone"


def test_same_catalog_stamp_fetches_only_the_catalog(rulings_dir, monkeypatch):
    run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    fake = run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    assert fake.calls == [ingest_download.BULK_DATA_URL]


def test_new_stamp_with_identical_content_does_not_rewrite_the_dump(rulings_dir, monkeypatch):
    """Scryfall re-cuts the file daily. A rewrite would move the mtime every
    reader memoises on and MISS every stack routine for a change nobody can see."""
    from manamap import config
    run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    before = config.RULINGS_PATH.stat().st_mtime_ns
    fake = run_download(monkeypatch, "2026-09-16T09:00:00Z", gz_body())
    assert len(fake.calls) == 2, "it had to download to know the content matched"
    assert config.RULINGS_PATH.stat().st_mtime_ns == before
    meta = json.loads(config.RULINGS_META_PATH.read_text())
    assert meta["updated_at"] == "2026-09-16T09:00:00Z", "the stamp moves so tomorrow stops at gate 1"
    assert not (rulings_dir / "rulings.new.jsonl.gz").exists()


def test_changed_content_replaces_the_dump(rulings_dir, monkeypatch):
    from manamap import config
    run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    old_sha = json.loads(config.RULINGS_META_PATH.read_text())["content_sha256"]
    new_lines = RULING_LINES + [{"object": "ruling", "oracle_id": "cccc-3", "source": "wotc",
                                 "published_at": "2026-09-16", "comment": "A brand-new ruling."}]
    run_download(monkeypatch, "2026-09-16T09:00:00Z", gz_body(new_lines))
    meta = json.loads(config.RULINGS_META_PATH.read_text())
    assert meta["content_sha256"] != old_sha
    assert meta["count"] == len(new_lines)
    with gzip.open(config.RULINGS_PATH, "rt") as f:
        assert sum(1 for _ in f) == len(new_lines)


def test_force_redownloads_and_rewrites(rulings_dir, monkeypatch):
    from manamap import config
    run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body())
    before = config.RULINGS_PATH.stat().st_mtime_ns
    import os
    os.utime(config.RULINGS_PATH, ns=(before - 10**9, before - 10**9))
    fake = run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body(), force=True)
    assert len(fake.calls) == 2
    assert config.RULINGS_PATH.stat().st_mtime_ns != before - 10**9


def test_a_catalog_without_a_rulings_entry_fails_naming_it(rulings_dir, monkeypatch):
    with pytest.raises(ValueError, match="rulings"):
        run_download(monkeypatch, "2026-09-15T09:00:00Z", gz_body(), with_rulings=False)


def test_the_corpus_download_defaults_are_untouched():
    """Pipeline step 1 must be byte-identical: every new parameter has the old
    value as its default, so a call with no arguments means what it always did."""
    import inspect
    from manamap.config import BULK_DATA_TYPE, DOWNLOAD_META_PATH, RAW_JSON_PATH
    sig = inspect.signature
    assert sig(ingest_download.get_bulk_data_info).parameters["bulk_type"].default == BULK_DATA_TYPE
    assert sig(ingest_download.is_up_to_date).parameters["meta_path"].default == DOWNLOAD_META_PATH
    assert sig(ingest_download.download_file).parameters["path"].default == RAW_JSON_PATH
    assert sig(ingest_download.save_meta).parameters["meta_path"].default == DOWNLOAD_META_PATH


def test_download_rulings_is_registered_and_not_read_only():
    """It writes, so it must not be reachable through the daemon or the MCP tool."""
    from manamap.pilot.registry import PILOT_STEPS
    from manamap import serve
    names = [n for n, _, _ in PILOT_STEPS]
    assert "download-rulings" in names
    assert "download-rulings" not in serve.CLI_READONLY
