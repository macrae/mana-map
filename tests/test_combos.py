"""Tests for the combo steps: download_combos.py (stubbed session) and process_combos.py."""

import gzip
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import requests

from manamap.ingest import download_combos as dc
from manamap.ingest.process_combos import (
    bracket_summary,
    build_card_index,
    build_combo_graph,
    build_combo_index,
    extract_bracket,
    extract_card_names,
    extract_color_identity,
    extract_produces,
    is_infinite,
    load_known_cards,
    load_source_meta,
    raw_variants,
    status_counts,
)


# ── Fixtures ──


def make_combo(card_names, identity="", produces=None, bracket_tag=None,
               mana_value_needed=None, popularity=None):
    """Helper to build a combo variant dict matching Commander Spellbook format."""
    uses = [{"card": {"name": name}} for name in card_names]
    prods = [{"feature": {"name": p}} for p in (produces or [])]
    return {
        "uses": uses,
        "identity": identity,
        "produces": prods,
        "bracketTag": bracket_tag,
        "manaValueNeeded": mana_value_needed,
        "popularity": popularity,
    }


# ── extract_card_names ──


def test_extract_card_names_basic():
    combo = make_combo(["Sol Ring", "Dramatic Reversal", "Isochron Scepter"])
    assert extract_card_names(combo) == ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"]


def test_extract_card_names_empty():
    assert extract_card_names({}) == []
    assert extract_card_names({"uses": []}) == []


def test_extract_card_names_missing_card_field():
    combo = {"uses": [{"card": {}}, {"card": {"name": "Lightning Bolt"}}]}
    assert extract_card_names(combo) == ["Lightning Bolt"]


def test_extract_card_names_strips_whitespace():
    combo = make_combo(["  Sol Ring  ", "Lightning Bolt"])
    names = extract_card_names(combo)
    assert names == ["Sol Ring", "Lightning Bolt"]


# ── extract_color_identity ──


def test_extract_color_identity():
    assert extract_color_identity({"identity": "wub"}) == "WUB"
    assert extract_color_identity({"identity": "r"}) == "R"
    assert extract_color_identity({}) == ""


# ── extract_produces ──


def test_extract_produces():
    combo = make_combo(["A", "B"], produces=["Infinite mana", "Infinite storm count"])
    assert extract_produces(combo) == ["Infinite mana", "Infinite storm count"]


def test_extract_produces_empty():
    assert extract_produces({}) == []
    assert extract_produces({"produces": []}) == []


# ── load_known_cards ──


def test_load_known_cards():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame({"name": ["Sol Ring", "Lightning Bolt", "Counterspell"]})
        df.to_csv(f.name, index=False)
        cards = load_known_cards(Path(f.name))
    assert cards == {"Sol Ring", "Lightning Bolt", "Counterspell"}


# ── build_combo_graph ──


def test_build_combo_graph_basic():
    known = {"Sol Ring", "Dramatic Reversal", "Isochron Scepter"}
    combos = [
        make_combo(
            ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"],
            identity="u",
            produces=["Infinite colorless mana"],
        )
    ]
    partners, combo_list = build_combo_graph(combos, known)

    # Each card should partner with the other two
    assert set(partners["Sol Ring"]) == {"Dramatic Reversal", "Isochron Scepter"}
    assert set(partners["Dramatic Reversal"]) == {"Sol Ring", "Isochron Scepter"}
    assert set(partners["Isochron Scepter"]) == {"Sol Ring", "Dramatic Reversal"}

    assert len(combo_list) == 1
    assert combo_list[0]["cards"] == ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"]
    assert combo_list[0]["ci"] == "U"
    assert combo_list[0]["produces"] == ["Infinite colorless mana"]


def test_build_combo_graph_filters_unknown_cards():
    known = {"Sol Ring", "Lightning Bolt"}
    combos = [
        make_combo(["Sol Ring", "Unknown Card That Doesnt Exist"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)
    assert len(partners) == 0
    assert len(combo_list) == 0


def test_build_combo_graph_skips_single_card_combos():
    known = {"Sol Ring"}
    combos = [make_combo(["Sol Ring"])]
    partners, combo_list = build_combo_graph(combos, known)
    assert len(partners) == 0
    assert len(combo_list) == 0


def test_build_combo_graph_multiple_combos():
    known = {"A", "B", "C", "D"}
    combos = [
        make_combo(["A", "B"], produces=["Effect 1"]),
        make_combo(["C", "D"], produces=["Effect 2"]),
        make_combo(["A", "C"], produces=["Effect 3"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)

    assert len(combo_list) == 3
    # A partners with B and C
    assert set(partners["A"]) == {"B", "C"}
    # B only partners with A
    assert set(partners["B"]) == {"A"}
    # C partners with A and D
    assert set(partners["C"]) == {"A", "D"}


def test_build_combo_graph_deduplicates_partners():
    known = {"A", "B", "C"}
    combos = [
        make_combo(["A", "B"], produces=["Effect 1"]),
        make_combo(["A", "B", "C"], produces=["Effect 2"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)

    # A-B partnership appears in both combos but should be deduplicated
    assert "B" in partners["A"]
    assert partners["A"].count("B") == 1  # sorted list, each entry once


def test_build_combo_graph_partners_are_sorted():
    known = {"Z", "M", "A"}
    combos = [make_combo(["Z", "M", "A"])]
    partners, _ = build_combo_graph(combos, known)

    assert partners["Z"] == ["A", "M"]
    assert partners["M"] == ["A", "Z"]
    assert partners["A"] == ["M", "Z"]


def test_combo_graph_json_serializable():
    """Ensure the output can be serialized to JSON."""
    known = {"Sol Ring", "Dramatic Reversal"}
    combos = [make_combo(["Sol Ring", "Dramatic Reversal"], identity="u", produces=["Infinite mana"])]
    partners, combo_list = build_combo_graph(combos, known)

    graph = {"partners": partners, "combos": combo_list}
    output = json.dumps(graph, separators=(",", ":"))
    parsed = json.loads(output)
    assert "partners" in parsed


# ── extract_bracket ──


@pytest.mark.parametrize("tag,expected", [
    ("E", 1), ("C", 2), ("O", 2), ("P", 3), ("S", 3), ("R", 4),
])
def test_extract_bracket_maps_spellbook_letters(tag, expected):
    bracket, banned = extract_bracket({"bracketTag": tag})
    assert bracket == expected
    assert banned is False


def test_extract_bracket_flags_banned():
    bracket, banned = extract_bracket({"bracketTag": "B"})
    assert bracket is None
    assert banned is True


def test_extract_bracket_unknown_letter_is_none_not_one():
    """An unrecognized tag must not read as bracket 1 — that under-reports a floor."""
    bracket, banned = extract_bracket({"bracketTag": "X"})
    assert bracket is None
    assert banned is False


def test_extract_bracket_missing_tag():
    assert extract_bracket({}) == (None, False)


# ── enriched combo records ──


def test_build_combo_graph_carries_bracket_fields():
    known = {"A", "B"}
    combos = [make_combo(["A", "B"], bracket_tag="R", mana_value_needed=4, popularity=1200)]
    _, combo_list = build_combo_graph(combos, known)

    assert combo_list[0]["bracket"] == 4
    assert combo_list[0]["mana_value_needed"] == 4
    assert combo_list[0]["popularity"] == 1200
    assert "banned" not in combo_list[0]


def test_build_combo_graph_keeps_banned_combos_flagged():
    """Format-agnostic by design: banned combos are flagged, never dropped."""
    known = {"A", "B"}
    combos = [make_combo(["A", "B"], bracket_tag="B")]
    partners, combo_list = build_combo_graph(combos, known)

    assert len(combo_list) == 1
    assert combo_list[0]["banned"] is True
    assert combo_list[0]["bracket"] is None
    assert set(partners["A"]) == {"B"}


# ── build_card_index ──


def test_build_card_index_maps_names_to_combo_indices():
    known = {"A", "B", "C"}
    combos = [make_combo(["A", "B"]), make_combo(["B", "C"])]
    _, combo_list = build_combo_graph(combos, known)
    index = build_card_index(combo_list)

    assert index["A"] == [0]
    assert index["B"] == [0, 1]
    assert index["C"] == [1]


def test_build_card_index_deduplicates_repeated_names():
    index = build_card_index([{"cards": ["A", "A", "B"]}])
    assert index["A"] == [0]


def test_build_card_index_empty():
    assert build_card_index([]) == {}


# ── bracket_summary ──


def test_bracket_summary_counts_by_bracket_and_banned():
    known = {"A", "B", "C", "D"}
    combos = [
        make_combo(["A", "B"], bracket_tag="E"),
        make_combo(["C", "D"], bracket_tag="E"),
        make_combo(["A", "C"], bracket_tag="R"),
        make_combo(["B", "D"], bracket_tag="B"),
    ]
    _, combo_list = build_combo_graph(combos, known)

    assert bracket_summary(combo_list) == {"1": 2, "4": 1, "banned": 1}


# ── ids, source meta, the two dump shapes ──


def test_build_combo_graph_carries_spellbook_id():
    known = {"A", "B"}
    combo = make_combo(["A", "B"])
    combo["id"] = "513-5034--46"
    _, combo_list = build_combo_graph([combo], known)
    assert combo_list[0]["id"] == "513-5034--46"


def test_raw_variants_reads_both_dump_shapes():
    variants = [make_combo(["A", "B"])]
    assert raw_variants(variants) is variants
    assert raw_variants({"timestamp": "t", "version": 3, "variants": variants, "aliases": []}) is variants
    with pytest.raises(ValueError):
        raw_variants({"timestamp": "t"})
    with pytest.raises(ValueError):
        raw_variants("nonsense")


def test_load_source_meta_absent_or_old_shape_is_none_values(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_source_meta(missing) == {"timestamp": None, "version": None}
    old = tmp_path / "old.json"
    old.write_text('{"count": 83261}')
    assert load_source_meta(old) == {"timestamp": None, "version": None}
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    assert load_source_meta(broken) == {"timestamp": None, "version": None}
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"timestamp": "2026-10-08T00:00:00Z", "version": 7, "etag": "x"}))
    assert load_source_meta(new) == {"timestamp": "2026-10-08T00:00:00Z", "version": 7}


def test_status_counts_reports_without_filtering():
    combos = [make_combo(["A", "B"]), make_combo(["A", "B"]), make_combo(["A", "B"])]
    combos[0]["status"] = "OK"
    combos[1]["status"] = "OK"
    combos[2]["status"] = "D"
    assert status_counts(combos) == {"OK": 2, "D": 1}
    # Status never decides membership: the graph keeps all three.
    _, combo_list = build_combo_graph(combos, {"A", "B"})
    assert len(combo_list) == 3


def test_is_infinite_agrees_with_the_bracket_engine():
    """The copied one-liner must answer exactly as `pilot.bracket.is_infinite`."""
    from manamap.pilot.bracket import is_infinite as bracket_is_infinite

    cases = [
        {"produces": ["Infinite colorless mana"]},
        {"produces": ["Near-infinite damage"]},
        {"produces": ["Win the game", "infinite storm count"]},
        {"produces": []},
        {},
    ]
    for record in cases:
        assert is_infinite(record) == bracket_is_infinite(record), record


# ── build_combo_index ──


def indexed(card_specs):
    """(name list, popularity, produces, bracket_tag, id) rows -> processed records."""
    combos = []
    known = set()
    for names, popularity, produces, tag, cid in card_specs:
        combo = make_combo(names, produces=produces, bracket_tag=tag, popularity=popularity,
                           mana_value_needed=len(names))
        combo["id"] = cid
        combos.append(combo)
        known.update(names)
    _, combo_list = build_combo_graph(combos, known)
    return combo_list


def test_build_combo_index_caps_top_and_keeps_true_totals():
    # A and B share all twenty combos; with per_card=5 the fifteen least popular
    # are in nobody's top, so they stay out of the rows while the totals count them.
    specs = [(["A", "B"], 100 - i, ["Infinite mana"] if i % 2 == 0 else ["Win"], "E", f"c{i:02d}")
             for i in range(20)]
    combo_list = indexed(specs)
    index = build_combo_index(combo_list, per_card=5)

    for name in ("A", "B"):
        entry = index["by_card"][name]
        assert entry["n"] == 20
        assert entry["inf"] == 10
        assert len(entry["top"]) == 5
        assert [index["combos"][row][0] for row in entry["top"]] == ["c00", "c01", "c02", "c03", "c04"]
    assert index["meta"] == {"source_timestamp": None, "per_card": 5, "combos": 5, "indexed": 2}
    assert len(index["combos"]) == 5


def test_build_combo_index_rows_hold_every_combo_some_top_names():
    # Twenty partners, each in one combo with A: A's top is capped at five, but
    # every partner's own top names its one combo, so all twenty rows are kept.
    specs = [(["A", f"P{i:02d}"], 100 - i, ["Win"], "E", f"c{i:02d}") for i in range(20)]
    index = build_combo_index(indexed(specs), per_card=5)
    assert len(index["by_card"]["A"]["top"]) == 5
    assert index["by_card"]["A"]["n"] == 20
    assert len(index["combos"]) == 20
    for i in range(20):
        entry = index["by_card"][f"P{i:02d}"]
        assert entry == {"n": 1, "inf": 0, "top": [i]}  # rows are id-sorted: c00..c19


def test_build_combo_index_top_is_popularity_desc_then_id_asc():
    specs = [
        (["A", "B"], 5, ["Win"], "E", "z-later"),
        (["A", "B"], 9, ["Win"], "E", "m"),
        (["A", "B"], 5, ["Win"], "E", "a-first"),
        (["A", "B"], None, ["Win"], "E", "none-pop"),
    ]
    index = build_combo_index(indexed(specs), per_card=3)
    top_ids = [index["combos"][row][0] for row in index["by_card"]["A"]["top"]]
    assert top_ids == ["m", "a-first", "z-later"]
    assert index["by_card"]["B"]["top"] == index["by_card"]["A"]["top"]
    # The rows themselves are sorted by id, independent of popularity; the
    # unreferenced fourth combo is counted in `n` and absent from the rows.
    assert [row[0] for row in index["combos"]] == ["a-first", "m", "z-later"]
    assert index["by_card"]["A"]["n"] == 4


def test_build_combo_index_is_deterministic_under_input_order():
    specs = [(["A", f"P{i}"], i % 4, ["Win"], "E", f"id{i}") for i in range(12)]
    forward = build_combo_index(indexed(specs), per_card=4)
    backward = build_combo_index(indexed(list(reversed(specs))), per_card=4)
    assert json.dumps(forward, sort_keys=True) == json.dumps(backward, sort_keys=True)


def test_build_combo_index_banned_keeps_null_bracket_and_rows_resolve():
    specs = [
        (["A", "B"], 10, ["Infinite damage"], "B", "banned-1"),
        (["A", "C"], 3, ["Win"], "R", "ruthless-1"),
    ]
    combo_list = indexed(specs)
    index = build_combo_index(combo_list, per_card=12)
    rows = index["combos"]
    by_id = {row[0]: row for row in rows}
    assert by_id["banned-1"] == ["banned-1", ["A", "B"], 1, None, 2]
    assert by_id["ruthless-1"] == ["ruthless-1", ["A", "C"], 0, 4, 2]
    for name, entry in index["by_card"].items():
        assert len(entry["top"]) <= 12
        for row in entry["top"]:
            assert name in rows[row][1]
    # Every index id resolves in the details list it was cut from.
    detail_ids = {record["id"] for record in combo_list}
    assert {row[0] for row in rows} <= detail_ids


def test_build_combo_index_dedupes_a_repeated_name_within_one_combo():
    combo_list = [{"id": "x", "cards": ["A", "A", "B"], "produces": [], "bracket": 1,
                   "mana_value_needed": 0, "popularity": 1}]
    index = build_combo_index(combo_list)
    assert index["by_card"]["A"] == {"n": 1, "inf": 0, "top": [0]}


def test_build_combo_index_empty():
    index = build_combo_index([], per_card=3)
    assert index == {"meta": {"source_timestamp": None, "per_card": 3, "combos": 0, "indexed": 0},
                     "combos": [], "by_card": {}}


def test_combo_index_is_compact_json():
    index = build_combo_index(indexed([(["A", "B"], 1, ["Win"], "E", "i")]))
    text = json.dumps(index, separators=(",", ":"))
    assert " " not in text.replace('"A"', "").replace('"B"', "")


# ── download_combos: the sidecar and the bulk route ──


class StubResponse:
    def __init__(self, headers=None, body=b"", status=200):
        self.headers = dict(headers or {})
        self._body = body
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]


class StubSession:
    """A `requests.Session` stand-in: one HEAD answer, one GET body, or one error."""

    def __init__(self, headers=None, body=b"", error=None, status=200):
        self.headers = dict(headers or {})
        self.body = body
        self.error = error
        self.status = status
        self.calls = []

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url))
        if self.error:
            raise self.error
        return StubResponse(self.headers, status=self.status)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url))
        if self.error:
            raise self.error
        return StubResponse(self.headers, self.body, status=self.status)


@pytest.fixture
def combos_dir(tmp_path, monkeypatch):
    """Point the downloader at a scratch data dir and seed a dump + a bulk-shape sidecar."""
    raw = tmp_path / "combos_raw.json.gz"
    meta = tmp_path / ".combos-meta.json"
    monkeypatch.setattr(dc, "COMBOS_RAW_PATH", raw)
    monkeypatch.setattr(dc, "COMBOS_META_PATH", meta)
    monkeypatch.setattr(dc, "DATA_DIR", tmp_path)
    with gzip.open(raw, "wt") as f:
        json.dump({"timestamp": "2026-10-01T00:00:00Z", "version": 1, "variants": [], "aliases": []}, f)
    dc.save_meta({"etag": '"abc"', "last_modified": "Wed, 01 Oct 2026 00:00:00 GMT",
                  "timestamp": "2026-10-01T00:00:00Z", "version": 1, "count": 0,
                  "downloaded_at": "2026-10-01T00:00:01+00:00"})
    return tmp_path


def test_is_up_to_date_false_without_dump_or_meta(combos_dir):
    session = StubSession({"ETag": '"abc"'})
    (combos_dir / "combos_raw.json.gz").unlink()
    assert dc.is_up_to_date(session) is False
    assert session.calls == []  # nothing to compare, nothing asked


def test_is_up_to_date_old_shape_meta_refreshes(combos_dir):
    (combos_dir / ".combos-meta.json").write_text('{"count": 83261}')
    session = StubSession({"ETag": '"abc"'})
    assert dc.is_up_to_date(session) is False
    assert session.calls == []


def test_is_up_to_date_same_etag(combos_dir):
    session = StubSession({"ETag": '"abc"', "Last-Modified": "Thu, 08 Oct 2026 00:00:00 GMT"})
    assert dc.is_up_to_date(session) is True
    assert session.calls == [("HEAD", dc.COMBOS_BULK_URL)]


def test_is_up_to_date_changed_etag_is_stale(combos_dir):
    session = StubSession({"ETag": '"def"'})
    assert dc.is_up_to_date(session) is False


def test_is_up_to_date_falls_back_to_last_modified(combos_dir):
    same = StubSession({"Last-Modified": "Wed, 01 Oct 2026 00:00:00 GMT"})
    assert dc.is_up_to_date(same) is True
    moved = StubSession({"Last-Modified": "Thu, 08 Oct 2026 00:00:00 GMT"})
    assert dc.is_up_to_date(moved) is False
    silent = StubSession({})  # no validator at all: cannot tell, so refresh
    assert dc.is_up_to_date(silent) is False


@pytest.mark.parametrize("error", [
    requests.ConnectionError("dns"),
    requests.Timeout("slow"),
    requests.HTTPError("503"),
])
def test_is_up_to_date_offline_keeps_the_dump_with_a_warning(combos_dir, capsys, error):
    session = StubSession(error=error)
    assert dc.is_up_to_date(session) is True
    out = capsys.readouterr().out
    assert "WARNING" in out and type(error).__name__ in out
    assert out.count("\n") == 1


def test_download_bulk_writes_dump_and_meta(combos_dir):
    variants = [{"id": "1-2", "uses": [], "status": "OK"}, {"id": "3-4", "uses": [], "status": "OK"}]
    doc = {"timestamp": "2026-10-08T12:00:00Z", "version": 9, "variants": variants, "aliases": []}
    body = gzip.compress(json.dumps(doc).encode())
    session = StubSession({"ETag": '"new"', "Last-Modified": "Thu, 08 Oct 2026 12:00:00 GMT"}, body=body)
    legacy = combos_dir / "combos_raw.json"
    legacy.write_text("[]")  # the stale uncompressed sibling must not linger

    meta = dc.download_bulk(session)

    assert session.calls == [("GET", dc.COMBOS_BULK_URL)]
    assert not legacy.exists()
    with gzip.open(combos_dir / "combos_raw.json.gz", "rt") as f:
        assert json.load(f) == doc
    on_disk = json.loads((combos_dir / ".combos-meta.json").read_text())
    assert on_disk == meta
    assert set(on_disk) == set(dc.META_KEYS)
    assert on_disk["etag"] == '"new"'
    assert on_disk["last_modified"] == "Thu, 08 Oct 2026 12:00:00 GMT"
    assert on_disk["timestamp"] == "2026-10-08T12:00:00Z"
    assert on_disk["version"] == 9
    assert on_disk["count"] == 2
    assert on_disk["downloaded_at"]
    # And the round trip: the file just written is now current.
    assert dc.is_up_to_date(StubSession({"ETag": '"new"'})) is True


def test_download_bulk_regzips_an_inflated_body(combos_dir):
    """A CDN that inflates in flight still leaves a real .gz on disk."""
    doc = {"timestamp": "t", "version": 1, "variants": [{"id": "a", "uses": []}], "aliases": []}
    session = StubSession({"ETag": '"x"'}, body=json.dumps(doc).encode())
    meta = dc.download_bulk(session)
    raw = combos_dir / "combos_raw.json.gz"
    assert raw.read_bytes()[:2] == dc.GZIP_MAGIC
    with gzip.open(raw, "rt") as f:
        assert json.load(f) == doc
    assert meta["count"] == 1


def test_download_bulk_http_error_leaves_the_old_dump(combos_dir):
    session = StubSession({}, status=503)
    before = (combos_dir / "combos_raw.json.gz").read_bytes()
    with pytest.raises(requests.HTTPError):
        dc.download_bulk(session)
    assert (combos_dir / "combos_raw.json.gz").read_bytes() == before
    assert not list(combos_dir.glob("*.part"))


def test_main_skips_when_current_and_force_overrides(combos_dir, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(dc, "is_up_to_date", lambda session=None: True)
    monkeypatch.setattr(dc, "download_bulk", lambda: calls.append("bulk") or
                        {"count": 0, "timestamp": None, "version": None})
    monkeypatch.setattr(dc, "download_paged", lambda: calls.append("paged") or {"count": 0})
    dc.main()
    assert calls == []
    assert "skipping" in capsys.readouterr().out
    dc.main(["--force"])
    assert calls == ["bulk"]
    dc.main(["--force", "--paged"])
    assert calls == ["bulk", "paged"]


def test_main_default_argv_is_empty_not_sys_argv(combos_dir, monkeypatch):
    """The pipeline calls `main()` under `manamap run …`; sys.argv must not leak in."""
    monkeypatch.setattr(sys, "argv", ["manamap", "run", "--from", "download-combos"])
    monkeypatch.setattr(dc, "is_up_to_date", lambda session=None: True)
    dc.main()  # would SystemExit(2) on the unknown args if argv leaked
