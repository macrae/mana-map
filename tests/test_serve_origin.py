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
