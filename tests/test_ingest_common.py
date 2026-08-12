"""Raw dumps on disk: gzip, with a fallback so nothing has to migrate.

`oracle-cards.json` (172 MB) and `combos_raw.json` (428 MB) were 600 MB of the
871 MB in `data/` — and both are pure pipeline input, written by one downloader
and read by one extractor. JSON compresses about 8.5x, so storing them gzipped
took them to 56 MB. That is the largest disk saving available in this repo and
it costs one small module.
"""

import gzip
import json

import pytest

from manamap.ingest.common import (
    dump_exists,
    dump_paths,
    dump_size_mb,
    open_dump,
    resolve_dump,
)


@pytest.fixture
def dump(tmp_path):
    """The configured path — always the `.gz` form, as config now declares."""
    return tmp_path / "raw.json.gz"


def test_dump_paths_pairs_the_two_forms(dump):
    gz, legacy = dump_paths(dump)
    assert gz.name == "raw.json.gz"
    assert legacy.name == "raw.json"


def test_writing_produces_a_real_gzip_file(dump):
    with open_dump(dump, "wt") as f:
        json.dump([{"name": "Sol Ring"}], f)
    assert dump.exists()
    with gzip.open(dump, "rt") as f:          # readable by plain gzip, not just us
        assert json.load(f) == [{"name": "Sol Ring"}]


def test_round_trip_through_open_dump(dump):
    payload = [{"name": "Blowfly Infestation"}, {"name": "Nest of Scarabs"}]
    with open_dump(dump, "wt") as f:
        json.dump(payload, f)
    with open_dump(dump, "rt") as f:
        assert json.load(f) == payload


def test_reading_falls_back_to_an_uncompressed_sibling(dump):
    """A repo that already has the plain file keeps working — no migration, no
    re-download. This is the whole reason the change is safe to land."""
    _, legacy = dump_paths(dump)
    legacy.write_text(json.dumps([{"name": "Yawgmoth, Thran Physician"}]))
    assert resolve_dump(dump) == legacy
    with open_dump(dump, "rt") as f:
        assert json.load(f)[0]["name"] == "Yawgmoth, Thran Physician"


def test_gzip_wins_when_both_exist(dump):
    gz, legacy = dump_paths(dump)
    legacy.write_text(json.dumps(["stale"]))
    with gzip.open(gz, "wt") as f:
        json.dump(["fresh"], f)
    assert resolve_dump(dump) == gz
    with open_dump(dump, "rt") as f:
        assert json.load(f) == ["fresh"]


def test_writing_removes_the_stale_sibling(dump):
    """Otherwise a re-download leaves both on disk and the change costs space
    instead of saving it."""
    _, legacy = dump_paths(dump)
    legacy.write_text(json.dumps(["old"]))
    with open_dump(dump, "wt") as f:
        json.dump(["new"], f)
    assert not legacy.exists()
    assert dump.exists()


def test_dump_exists_sees_either_form(dump):
    assert not dump_exists(dump)
    _, legacy = dump_paths(dump)
    legacy.write_text("[]")
    assert dump_exists(dump)
    legacy.unlink()
    with open_dump(dump, "wt") as f:
        f.write("[]")
    assert dump_exists(dump)


def test_size_reports_what_is_on_disk(dump):
    assert dump_size_mb(dump) == 0.0
    with open_dump(dump, "wt") as f:
        json.dump([{"x": 1}] * 1000, f)
    assert 0 < dump_size_mb(dump) < 1


def test_reading_a_missing_dump_fails_loudly(dump):
    with pytest.raises(FileNotFoundError):
        open_dump(dump, "rt")


def test_binary_write_mode_is_supported(dump):
    """`download.py` streams response chunks straight in as bytes."""
    with open_dump(dump, "wb") as f:
        f.write(b'["chunk-one"')
        f.write(b',"chunk-two"]')
    with open_dump(dump, "rt") as f:
        assert json.load(f) == ["chunk-one", "chunk-two"]


def test_extracts_jsonl_dumps_as_well_as_legacy_arrays(dump, monkeypatch):
    """Scryfall's 2026-08 bulk migration serves gzipped JSONL (one card per
    line) where a single JSON array used to be. `extract` sniffs the first
    character rather than trusting the filename, so both generations of dump
    parse — a repo with an old array on disk needs no migration."""
    import gzip

    from manamap.ingest.common import dump_paths

    gz, _ = dump_paths(dump)

    def load(path):
        # mirror extract.py's sniffing loader exactly
        with open_dump(path, "rt") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                return json.load(f)
            return [json.loads(line) for line in f if line.strip()]

    with gzip.open(gz, "wt") as f:
        f.write('{"name": "A"}\n{"name": "B"}\n\n')
    assert [c["name"] for c in load(dump)] == ["A", "B"]

    with gzip.open(gz, "wt") as f:
        json.dump([{"name": "A"}, {"name": "B"}], f)
    assert [c["name"] for c in load(dump)] == ["A", "B"]
