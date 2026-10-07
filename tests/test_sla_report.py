"""`sla-report` reads the band's log and says when the running band is stale."""
import json

from manamap.pilot import sla_report as sr


def test_summary_counts_runs_medians_and_misses(tmp_path):
    log = tmp_path / "sla-log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"type": "data-analyst", "elapsed_s": 12, "sla_s": 30, "missed": False, "at": "a"},
        {"type": "data-analyst", "elapsed_s": 41, "sla_s": 30, "missed": True, "at": "b"},
        {"type": "context-keeper", "elapsed_s": 47, "sla_s": 120, "missed": False, "at": "c"},
    ]) + "\nnot json\n")
    out = {s["agent"]: s for s in sr.summary(sr.rows(log))}
    assert out["data-analyst"]["runs"] == 2 and out["data-analyst"]["missed"] == 1
    assert out["data-analyst"]["median_s"] == 26.5 and out["context-keeper"]["missed"] == 0


def test_a_stale_cached_band_is_named(tmp_path):
    """THE BUG: the band ran a cached 0.1.0 while the repo had the SLA code."""
    repo = tmp_path / "repo"
    (repo / "hooks").mkdir(parents=True)
    (repo / "hooks" / "register.tsx").write_text("new")
    cache = tmp_path / "cache" / "0.1.0"
    (cache / "hooks").mkdir(parents=True)
    (cache / "hooks" / "register.tsx").write_text("old")
    installed = tmp_path / "installed.json"
    installed.write_text(json.dumps({"plugins": {"job-band@mana-map": [
        {"installPath": str(cache), "version": "0.1.0"}]}}))
    assert "not the repo's" in sr.band_drift(installed, repo)
    (cache / "hooks" / "register.tsx").write_text("new")
    assert sr.band_drift(installed, repo) is None
    assert sr.band_drift(tmp_path / "absent.json", repo) is None


def test_first_response_latency_reads_median_p90_and_misses(tmp_path):
    log = tmp_path / "latency-log.jsonl"
    log.write_text("\n".join(json.dumps({"first_response_s": s}) for s in (0.8, 1.1, 1.4, 3.2)) + "\n")
    lat = sr.latency(log)
    assert lat["n"] == 4 and lat["median_s"] == 1.25 and lat["over"] == 1 and lat["p90_s"] == 3.2
    assert sr.latency(tmp_path / "none.jsonl") is None
