"""One HTTP client for the whole package (docs/integrations.md).

Until 2026-10-09 four modules each built their own `requests.Session`, and only
one of them — `pilot/fetch_deck.py` — retried a 429, a 5xx or a dropped socket.
A transient Scryfall outage therefore aborted a corpus download that a deck fetch
would have survived. This module is that retry loop, lifted, plus the two things
the loop alone could not give:

- **A JSON disk cache** per service (`data/cache/<service>/<key>.json`) for the
  catalog-shaped answers that do not change between runs, each entry served while
  younger than the caller's `ttl_s`.
- **An offline switch.** `MANAMAP_NET_OFFLINE=1` makes every request that is not
  answered from the cache raise `Offline` — and the unit tier sets it, so a test
  that forgot to patch its seam fails with a sentence instead of reaching out.

Nothing here is a gate. A command that needs the network asks for it, says so when
it cannot have it, and writes nothing on the way out — `registry.py`'s own words:
"a gate that fails when the network is down is a gate that gets switched off".

The seams tests use: `SESSION` is one object, so `monkeypatch.setattr(net.SESSION,
"post", fake)` intercepts every caller; `session=` on each function takes a
stand-in outright. Every function looks the method up on the session AT CALL TIME,
which is what makes both work.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

from manamap import config

OFFLINE_ENV = "MANAMAP_NET_OFFLINE"
DEFAULT_TIMEOUT = 60
#: The transport failures the loop retries. Not `RequestException`: an HTTPError
#: from `raise_for_status` or an invalid URL is not something waiting fixes.
TRANSPORT_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class Offline(Exception):
    """The network could not be had: unreachable after every retry, or
    `MANAMAP_NET_OFFLINE=1` and the answer was not in the cache. The message
    says what was wanted; nothing was written either way."""


def offline():
    """Is `MANAMAP_NET_OFFLINE` set to anything but empty or `0`?"""
    return os.environ.get(OFFLINE_ENV, "") not in ("", "0")


class _Session(requests.Session):
    """`requests.Session` with the offline switch at its one choke point.

    `Session.get`/`post`/`head` all go through `request`, so a caller that holds
    `SESSION` and calls it directly — a streaming download, say — is still
    refused offline. A test that monkeypatches `SESSION.post` never reaches here,
    which is the point: the patch IS the declaration that no network is wanted.
    """

    def request(self, method, url, *args, **kwargs):
        if offline():
            raise Offline(
                f"{OFFLINE_ENV}=1: wanted {method.upper()} {url} and it is not in "
                f"the cache; nothing was written.")
        return super().request(method, url, *args, **kwargs)


SESSION = _Session()
SESSION.headers["User-Agent"] = config.USER_AGENT


# ── the cache ────────────────────────────────────────────────────────────

def cache_root():
    """`data/cache/`, read from `config` at call time so a test can move it."""
    return config.DATA_DIR / "cache"


def cache_key(method, url, *, params=None, body=None, headers=None):
    """24 hex chars over the method, the URL, the sorted params or body, and the
    NAMES of any headers — never their values, which is where a token lives."""
    record = {
        "method": method.upper(),
        "url": url,
        "params": sorted((str(k), str(v)) for k, v in (params or {}).items()),
        "body": json.dumps(body, sort_keys=True, separators=(",", ":"), default=str),
        "headers": sorted(str(k).lower() for k in (headers or {})),
    }
    blob = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def cache_path(service, key):
    return cache_root() / service / f"{key}.json"


def _now():
    return datetime.now(timezone.utc)


def _age_s(record):
    """Seconds since the record's `fetched_at`; None when it cannot be read."""
    try:
        fetched = datetime.fromisoformat(record["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (_now() - fetched).total_seconds()


def _read_cache(path, ttl_s):
    """The cached body, or None on a miss — absent, unreadable or older than `ttl_s`."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    age = _age_s(record) if isinstance(record, dict) else None
    if age is None or age >= ttl_s or "body" not in record:
        return None
    return record["body"]


def _write_cache(path, url, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"fetched_at": _now().isoformat(timespec="seconds"), "url": url, "body": body}
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def purge(service, older_than_s):
    """Delete a service's entries fetched more than `older_than_s` ago, and any
    that cannot be read. Returns how many went. An absent directory is zero."""
    root = cache_root() / service
    if not root.is_dir():
        return 0
    removed = 0
    for path in sorted(root.glob("*.json")):
        try:
            age = _age_s(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            age = None
        if age is None or age > older_than_s:
            path.unlink()
            removed += 1
    return removed


# ── the retry loop ───────────────────────────────────────────────────────

def _retryable(resp):
    status = getattr(resp, "status_code", 200)
    return status == 429 or status >= 500


def request(method, url, *, session=None, timeout=DEFAULT_TIMEOUT, retries=None,
            backoff_s=None, sleep=None, **kwargs):
    """`session.<method>(url, …)` with the retry loop. Returns the response —
    NOT raised for status; the caller decides what a 404 means.

    Retries a 429, any 5xx, and a dropped connection or a timeout, with linear
    backoff (`NET_BACKOFF_S` x attempt). The transport half is not a status
    code: a keep-alive socket closed between requests raises inside the call,
    with no response to inspect, so it is caught here — and the session is
    closed first, because the dead socket stays in the pool and the retry would
    reuse it. Exhausting the retries on a transport error raises `Offline`;
    exhausting them on a status leaves the last response for `raise_for_status`.

    `retries`, `backoff_s` and `sleep` default to `config` and `time.sleep` AT
    CALL TIME, so a test can zero the wait or record the schedule.
    """
    session = SESSION if session is None else session
    retries = config.NET_MAX_RETRIES if retries is None else retries
    backoff_s = config.NET_BACKOFF_S if backoff_s is None else backoff_s
    sleep = time.sleep if sleep is None else sleep
    call = getattr(session, method.lower())
    args = {k: v for k, v in kwargs.items() if v is not None and v is not False}
    args["timeout"] = timeout
    resp = None
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            resp = call(url, **args)
        except TRANSPORT_ERRORS as exc:
            if last:
                raise Offline(
                    f"{method.upper()} {url} did not answer after {retries} "
                    f"attempt(s) ({exc.__class__.__name__}); nothing was written. "
                    f"Check the connection and run the same command again.") from exc
            close = getattr(session, "close", None)
            if callable(close):
                close()
            sleep(backoff_s * (attempt + 1))
            continue
        if not _retryable(resp):
            break
        if not last:
            sleep(backoff_s * (attempt + 1))
    return resp


def _cached_json(method, url, *, params, body, headers, service, ttl_s, fetch):
    """The cache around a JSON call: hit -> body; miss offline -> `Offline`;
    miss -> `fetch()`, written through. No `service`/`ttl_s` means no cache."""
    caching = service is not None and ttl_s is not None
    path = None
    if caching:
        path = cache_path(service, cache_key(method, url, params=params, body=body,
                                             headers=headers))
        hit = _read_cache(path, ttl_s)
        if hit is not None:
            return hit
        if offline():
            raise Offline(
                f"{OFFLINE_ENV}=1: wanted {method.upper()} {url} ({service}) and the "
                f"cache has no entry younger than {ttl_s} s; nothing was written.")
    doc = fetch()
    if caching:
        _write_cache(path, url, doc)
    return doc


def get_json(url, *, params=None, headers=None, service=None, ttl_s=None,
             session=None, timeout=DEFAULT_TIMEOUT):
    """GET a JSON document, retried, cached under `service` for `ttl_s` seconds."""
    def fetch():
        resp = request("get", url, session=session, timeout=timeout,
                       params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    return _cached_json("GET", url, params=params, body=None, headers=headers,
                        service=service, ttl_s=ttl_s, fetch=fetch)


def post_json(url, json, *, headers=None, service=None, ttl_s=None,
              session=None, timeout=DEFAULT_TIMEOUT):
    """POST a JSON body and parse the JSON answer, retried, cached like `get_json`."""
    body = json

    def fetch():
        resp = request("post", url, session=session, timeout=timeout,
                       json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()
    return _cached_json("POST", url, params=None, body=body, headers=headers,
                        service=service, ttl_s=ttl_s, fetch=fetch)


def get_stream(url, *, params=None, headers=None, session=None, timeout=DEFAULT_TIMEOUT):
    """GET with `stream=True`, retried, raised for status. Returns the response
    so the caller can `iter_content` a bulk file to disk. No JSON cache."""
    resp = request("get", url, session=session, timeout=timeout,
                   params=params, headers=headers, stream=True)
    resp.raise_for_status()
    return resp


def get_bytes(url, *, params=None, headers=None, session=None, timeout=DEFAULT_TIMEOUT):
    """GET a body as bytes, retried, raised for status. No JSON cache."""
    resp = request("get", url, session=session, timeout=timeout,
                   params=params, headers=headers)
    resp.raise_for_status()
    return resp.content


# ── tokens ───────────────────────────────────────────────────────────────

_HINTED = set()


def load_token(env_name, *, keychain_service):
    """`os.environ[env_name]`, or None with a one-line-per-process hint on stderr
    saying where to put it — the same Keychain recipe `sven/llm.py` prints, so
    every token this package reads is stored the same way."""
    value = os.environ.get(env_name)
    if value:
        return value
    if env_name not in _HINTED:
        _HINTED.add(env_name)
        print(
            f"  {env_name} is not set. Store it in the Keychain and export it in ~/.zshrc:\n"
            f"    security add-generic-password -a $USER -s {keychain_service} -w\n"
            f"    export {env_name}=$(security find-generic-password "
            f"-a $USER -s {keychain_service} -w)",
            file=sys.stderr)
    return None
