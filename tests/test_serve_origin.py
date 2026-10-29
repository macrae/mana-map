"""The origin gate on `manamap serve`'s /api — against a REAL server.

The API is unauthenticated and local, so the browser's same-origin policy is the
only wall between a page on the internet and `deck/delete`. A `text/plain` POST
crosses origins without a preflight, and `do_POST` used to parse any body as
JSON; these tests hold the three checks that close it (foreign Origin, foreign
Host, non-JSON POST) and the two callers that must still get through (the CLI
daemon client, which sends no Origin, and a page served by this server).
"""
import http.client
import json
import threading

import pytest

from manamap import serve


@pytest.fixture
def server():
    httpd = serve.serve(port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        # skip_host so a test can send a forged Host; otherwise send the real one.
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        headers = dict(headers or {})
        headers.setdefault("Host", f"127.0.0.1:{port}")
        blob = body.encode() if isinstance(body, str) else body
        if blob is not None:
            headers["Content-Length"] = str(len(blob))
        for k, v in headers.items():
            if v is not None:
                conn.putheader(k, v)
        conn.endheaders(blob)
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"{}")
    finally:
        conn.close()


JSON = {"Content-Type": "application/json"}


def test_a_foreign_origin_is_refused(server):
    status, doc = _request(server, "POST", "/api/health", "{}",
                           dict(JSON, Origin="https://evil.example"))
    assert status == 403
    assert "evil.example" in doc["error"]


def test_a_foreign_origin_cannot_read_by_get_either(server):
    status, _ = _request(server, "GET", "/api/health",
                         headers={"Origin": "https://evil.example"})
    assert status == 403


def test_a_text_plain_post_is_refused_before_anything_runs(server):
    """The simple-request CSRF: no preflight, so it reaches the server. It must
    be refused before the body is parsed, let alone dispatched. `health`, not
    `deck/delete`: if this gate ever regresses the test must fail, not delete."""
    status, doc = _request(server, "POST", "/api/health", "{}",
                           {"Content-Type": "text/plain"})
    assert status == 415
    assert "application/json" in doc["error"]


def test_a_rebound_host_is_refused(server):
    """DNS rebinding: same-origin to the browser, but the Host is not ours."""
    status, doc = _request(server, "POST", "/api/health", "{}",
                           dict(JSON, Host=f"evil.example:{server}"))
    assert status == 403
    assert "evil.example" in doc["error"]


def test_the_cli_daemon_client_still_gets_through(server):
    """`cli._daemon_run` sends JSON, no Origin, Host 127.0.0.1:<port>."""
    status, doc = _request(server, "POST", "/api/health", "{}", JSON)
    assert status == 200 and doc["ok"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_a_page_served_from_here_still_gets_through(server, host):
    status, doc = _request(server, "POST", "/api/health", "{}",
                           dict(JSON, Host=f"{host}:{server}",
                                Origin=f"http://{host}:{server}"))
    assert status == 200 and doc["ok"]
    status, doc = _request(server, "GET", "/api/health",
                           headers={"Host": f"{host}:{server}",
                                    "Origin": f"http://{host}:{server}"})
    assert status == 200 and doc["ok"]


def test_json_with_a_charset_is_still_json(server):
    status, _ = _request(server, "POST", "/api/health", "{}",
                         {"Content-Type": "application/json; charset=utf-8"})
    assert status == 200


def _headers(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path, headers=headers or {})
        r = conn.getresponse()
        r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}
    finally:
        conn.close()


@pytest.fixture
def static_server(tmp_path):
    """serve over a tmp root holding one file, so this reads no tracked data."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "m.bin").write_bytes(b"\0" * 64)
    httpd = serve.serve(port=0, root=tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_static_files_revalidate_and_the_api_is_never_cached(static_server):
    """With no Cache-Control a browser applies HEURISTIC freshness (10% of the file's
    age) and reuses a file without asking — a `cards.json` a regen just rewrote under
    the same `?v=` included. `no-cache` keeps the bytes and asks first; an unchanged
    file answers 304 with no body, which is what makes a navigation back to Build cheap
    (2026-10-28). Re-introducing the bug: drop `end_headers`' header and the first
    assertion fails."""
    port = static_server
    status, h = _headers(port, "/data/m.bin?v=13")
    assert status == 200 and h.get("cache-control") == "no-cache", h
    assert "last-modified" in h
    status, h2 = _headers(port, "/data/m.bin?v=13", {"If-Modified-Since": h["last-modified"]})
    assert status == 304 and h2.get("cache-control") == "no-cache", (status, h2)
    status, h3 = _headers(port, "/api/health", {"Host": f"127.0.0.1:{port}"})
    assert status == 200 and h3.get("cache-control") == "no-store", h3
