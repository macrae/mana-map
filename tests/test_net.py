"""`manamap.net`: the one HTTP client — its retry loop, its JSON cache, the
offline switch and the token hint. Every test here drives a fake session; the
real network is never patched, because it is never reached.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import requests

from manamap import config, net


# ── a scripted session ───────────────────────────────────────────────────

class Resp:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code = status
        self.headers = {}
        self._body = body
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._body


class Session:
    """Answers `script` in order; an exception instance in it is raised."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.closed = 0

    def _next(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        if not self.script:
            raise AssertionError(f"the session was asked for more than its script: {method} {url}")
        answer = self.script.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get(self, url, **kwargs):
        return self._next("GET", url, kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, kwargs)

    def close(self):
        self.closed += 1


@pytest.fixture
def online(monkeypatch):
    """The switch off and the schedule real: the retry tests record it."""
    monkeypatch.delenv(net.OFFLINE_ENV, raising=False)
    monkeypatch.setattr(config, "NET_BACKOFF_S", 1.0)
    monkeypatch.setattr(config, "NET_MAX_RETRIES", 4)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path / "cache"


URL = "https://example.invalid/v1/thing"


# ── the retry loop ───────────────────────────────────────────────────────

def test_a_503_is_retried_and_the_200_behind_it_is_returned(online):
    session = Session(Resp(503), Resp(200, {"ok": 1}))
    slept = []
    resp = net.request("get", URL, session=session, sleep=slept.append)
    assert resp.status_code == 200 and resp.json() == {"ok": 1}
    assert [m for m, _u, _k in session.calls] == ["GET", "GET"]
    assert slept == [1.0], "linear backoff: the first wait is one backoff unit"


def test_a_429_is_honoured_not_raised(online):
    session = Session(Resp(429), Resp(429), Resp(200, {"ok": 2}))
    slept = []
    resp = net.request("get", URL, session=session, sleep=slept.append)
    assert resp.status_code == 200
    assert len(session.calls) == 3
    assert slept == [1.0, 2.0], "the backoff grows with the attempt"


def test_a_connection_error_is_retried_then_raised_as_offline(online):
    session = Session(
        requests.ConnectionError("aborted"), requests.ConnectionError("aborted"),
        requests.Timeout("slow"), requests.ConnectionError("aborted"))
    slept = []
    with pytest.raises(net.Offline) as exc:
        net.request("post", URL, session=session, json={"a": 1}, sleep=slept.append)
    message = str(exc.value)
    assert "POST" in message and URL in message
    assert "4 attempt" in message and "nothing was written" in message
    assert "again" in message, "it must say what to do"
    assert len(session.calls) == 4
    assert slept == [1.0, 2.0, 3.0], "no sleep after the last attempt"
    assert session.closed == 3, "a dead keep-alive socket is dropped before each retry"


def test_a_dropped_connection_then_a_200_is_a_success(online):
    session = Session(requests.ConnectionError("aborted"), Resp(200, {"data": []}))
    assert net.post_json(URL, {"identifiers": []}, session=session, timeout=7) == {"data": []}
    assert session.calls[-1] == ("POST", URL, {"json": {"identifiers": []}, "timeout": 7})


def test_exhausting_the_retries_on_a_status_raises_for_that_status(online):
    session = Session(Resp(503), Resp(503), Resp(503), Resp(503))
    with pytest.raises(requests.HTTPError):
        net.get_json(URL, session=session)
    assert len(session.calls) == 4


def test_a_response_without_a_status_code_counts_as_success(online):
    """The rulings test's fake response carries no `status_code`; the loop must
    not demand one of a seam that predates it."""
    class Bare:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    session = Session(Bare())
    assert net.get_json(URL, session=session) == {"data": []}


# ── the cache ────────────────────────────────────────────────────────────

def test_a_cache_hit_within_the_ttl_makes_no_call(online, cache_dir):
    first = Session(Resp(200, {"n": 1}))
    assert net.get_json(URL, service="svc", ttl_s=3600, session=first) == {"n": 1}
    entries = list((cache_dir / "svc").glob("*.json"))
    assert len(entries) == 1
    record = json.loads(entries[0].read_text())
    assert set(record) == {"fetched_at", "url", "body"} and record["url"] == URL

    second = Session()  # anything asked of it is an assertion error
    assert net.get_json(URL, service="svc", ttl_s=3600, session=second) == {"n": 1}
    assert second.calls == []


def test_an_expired_entry_is_refetched_and_rewritten(online, cache_dir):
    key = net.cache_key("GET", URL)
    path = net.cache_path("svc", key)
    path.parent.mkdir(parents=True)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat(timespec="seconds")
    path.write_text(json.dumps({"fetched_at": stale, "url": URL, "body": {"n": "old"}}))

    session = Session(Resp(200, {"n": "new"}))
    assert net.get_json(URL, service="svc", ttl_s=60, session=session) == {"n": "new"}
    assert len(session.calls) == 1
    assert json.loads(path.read_text())["body"] == {"n": "new"}


def test_no_service_means_no_cache(online, cache_dir):
    session = Session(Resp(200, {"n": 1}), Resp(200, {"n": 2}))
    assert net.get_json(URL, session=session) == {"n": 1}
    assert net.get_json(URL, session=session) == {"n": 2}
    assert not cache_dir.exists()


def test_a_miss_offline_raises_before_any_call(cache_dir, monkeypatch):
    monkeypatch.setenv(net.OFFLINE_ENV, "1")
    session = Session(Resp(200, {"n": 1}))
    with pytest.raises(net.Offline) as exc:
        net.get_json(URL, service="svc", ttl_s=60, session=session)
    message = str(exc.value)
    assert net.OFFLINE_ENV in message and URL in message and "nothing was written" in message
    assert session.calls == [], "offline is decided before the session is asked"
    assert not (cache_dir / "svc").exists()


def test_a_hit_offline_is_still_served(cache_dir, monkeypatch):
    monkeypatch.delenv(net.OFFLINE_ENV, raising=False)
    net.get_json(URL, service="svc", ttl_s=60, session=Session(Resp(200, {"n": 1})))
    monkeypatch.setenv(net.OFFLINE_ENV, "1")
    assert net.get_json(URL, service="svc", ttl_s=60, session=Session()) == {"n": 1}


def test_the_real_session_refuses_offline_before_touching_a_socket(monkeypatch):
    """The switch sits in `SESSION.request`, under every direct `SESSION.get`."""
    monkeypatch.setenv(net.OFFLINE_ENV, "1")
    with pytest.raises(net.Offline) as exc:
        net.SESSION.get("http://127.0.0.1:9/never")
    assert "GET http://127.0.0.1:9/never" in str(exc.value)


def test_the_unit_tier_is_offline():
    """What `conftest._unit_tier_runs_offline` promises: this test did not set it."""
    assert os.environ.get(net.OFFLINE_ENV) == "1"
    assert config.NET_BACKOFF_S == 0.0


def test_the_cache_key_ignores_header_values_and_keeps_their_names():
    a = net.cache_key("GET", URL, headers={"Authorization": "Bearer one"})
    b = net.cache_key("GET", URL, headers={"Authorization": "Bearer two"})
    assert a == b, "a token must never reach the key"
    assert a != net.cache_key("GET", URL), "but WHICH headers vary is part of it"
    assert a != net.cache_key("GET", URL, headers={"X-Other": "Bearer one"})
    assert len(a) == 24 and int(a, 16) >= 0


def test_the_cache_key_orders_params_and_separates_methods():
    assert net.cache_key("GET", URL, params={"a": 1, "b": 2}) == \
        net.cache_key("GET", URL, params={"b": 2, "a": 1})
    assert net.cache_key("GET", URL, params={"a": 1}) != net.cache_key("GET", URL, params={"a": 2})
    assert net.cache_key("GET", URL) != net.cache_key("POST", URL)
    assert net.cache_key("POST", URL, body={"x": 1}) != net.cache_key("POST", URL, body={"x": 2})


def test_purge_removes_the_old_and_the_unreadable_and_keeps_the_young(cache_dir):
    root = cache_dir / "svc"
    root.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=3)).isoformat(timespec="seconds")
    young = (now - timedelta(seconds=5)).isoformat(timespec="seconds")
    (root / "old.json").write_text(json.dumps({"fetched_at": old, "url": URL, "body": 1}))
    (root / "young.json").write_text(json.dumps({"fetched_at": young, "url": URL, "body": 2}))
    (root / "junk.json").write_text("not json")
    assert net.purge("svc", older_than_s=86400) == 2
    assert sorted(p.name for p in root.glob("*.json")) == ["young.json"]
    assert net.purge("absent", older_than_s=0) == 0


# ── tokens ───────────────────────────────────────────────────────────────

def test_a_token_in_the_environment_wins_and_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(net, "_HINTED", set())
    monkeypatch.setenv("MM_TEST_TOKEN", "secret")
    assert net.load_token("MM_TEST_TOKEN", keychain_service="manamap-test") == "secret"
    assert capsys.readouterr().err == ""


def test_a_missing_token_is_none_with_the_keychain_hint_printed_once(monkeypatch, capsys):
    monkeypatch.setattr(net, "_HINTED", set())
    monkeypatch.delenv("MM_TEST_TOKEN", raising=False)
    assert net.load_token("MM_TEST_TOKEN", keychain_service="manamap-test") is None
    assert net.load_token("MM_TEST_TOKEN", keychain_service="manamap-test") is None
    out = capsys.readouterr()
    assert out.out == "", "the hint goes to stderr, never into a command's output"
    assert out.err.count("MM_TEST_TOKEN is not set") == 1, "once per process"
    assert "security find-generic-password -a $USER -s manamap-test -w" in out.err
    assert "security add-generic-password -a $USER -s manamap-test -w" in out.err


# ── the session every caller shares ──────────────────────────────────────

def test_every_migrated_module_holds_the_one_session():
    from manamap.ingest import download, download_combos
    from manamap.pilot import download_rules, fetch_deck

    for module in (download, download_combos, download_rules, fetch_deck):
        assert module.SESSION is net.SESSION, module.__name__
    assert net.SESSION.headers["User-Agent"] == config.USER_AGENT
