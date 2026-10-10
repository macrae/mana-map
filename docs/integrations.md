# Integrations — the one HTTP client (2026-10-09)

Everything in this package that reaches an outside service goes through
`src/manamap/net.py`: Scryfall (the corpus, a deck's cards, the rulings, a proxy
sheet's images), Commander Spellbook (the combos), Wizards (the Comprehensive
Rules). Until this page's date each of those modules built its own
`requests.Session`, and only the deck fetch retried anything — so a transient 503
aborted a corpus download that a deck fetch would have survived. One client, one
retry loop, one place to switch the network off.

Two things are deliberately NOT on it and stay on `urllib`: `ingest/edhrec.py` and
`sim/edhrec.py`, whose rate-limiting and User-Agent strings are their own; and
`sven/*`, which speaks to the Anthropic SDK, not to HTTP.

## The client

| call | what it does |
|---|---|
| `net.SESSION` | the one `requests.Session`, `User-Agent: config.USER_AGENT`. Every migrated module re-exports it as its own `SESSION`, so a test's `monkeypatch.setattr(fetch_deck.SESSION, "post", fake)` keeps intercepting |
| `net.request(method, url, …)` | the retry loop, returning the response un-raised — for the caller that must read a 404 itself (`download_rules`) |
| `net.get_json` / `net.post_json` | retried, `raise_for_status`, parsed; cached when `service` and `ttl_s` are both given |
| `net.get_stream` / `net.get_bytes` | retried, raised; the response to stream a bulk file to disk, or the body as bytes. Never cached here — a bulk dump has its own sidecar, a proxy image its own PNG cache |
| `net.load_token(env, keychain_service=…)` | the token from the environment, or None with the Keychain hint on stderr once per process |
| `net.cache_path`, `net.purge(service, older_than_s)` | where an entry lives; drop a service's old or unreadable entries |

**The retry loop** retries a 429, any 5xx, a dropped connection and a timeout, with
linear backoff (`config.NET_BACKOFF_S` x attempt, `config.NET_MAX_RETRIES` attempts;
the `SCRYFALL_*` names are aliases of the same values). The transport half is not a
status code — a keep-alive socket closed between requests raises inside the call with
no response to inspect — so it is caught, and the session is closed first, because the
dead socket stays in the pool and the retry would reuse it (`docs/gotchas-bench.md` has
the live failure). Exhausting the retries on a transport error raises `net.Offline`;
on a status it leaves the last response for `raise_for_status`. Every function takes
`session=` for a stand-in, and the method is looked up on it at call time. The
combos downloader's HEAD probe is the one call that does not retry: its failure fails
open (the dump on disk is kept), so four backoffs would only delay the pipeline.

## The cache: `data/cache/<service>/`

A JSON answer that does not change between runs is written to
`data/cache/<service>/<key>.json` as `{"fetched_at", "url", "body"}` and served while
younger than the caller's `ttl_s`. The key is 24 hex characters of a sha256 over the
method, the URL, the sorted params or body, and the NAMES of any headers passed —
never their values, which is where a token would be. `data/*` is gitignored, so the
cache never travels; `net.purge("<service>", older_than_s)` prunes one service, and
deleting the directory is always safe. No TTL is set here: each caller names its own
service and how long its answer stays good, so the table of TTLs is the callers.
`data/cache/proxies/` is the proxy sheet's PNG cache, keyed by URL, and predates this
client (`docs/data-artifacts.md`).

## `MANAMAP_NET_OFFLINE=1`

With the switch set, any request not answered from the cache raises `net.Offline`,
whose message names the method and URL and says that nothing was written. The switch
sits in `SESSION.request`, under every `get`/`post`/`head`, so a module that holds the
session and calls it directly is refused too. **The unit tier sets it** for every test
(`tests/conftest.py`, `_unit_tier_runs_offline`), with the backoff zeroed, so a test
that forgot to patch its seam fails with a sentence instead of reaching Scryfall; a
patched `SESSION.post` or a `session=` stand-in never reaches the switch, which is the
point. The regression and integration tiers are not forced offline.

## Tokens

A service that needs a token reads it with `net.load_token("<ENV>", keychain_service=
"<name>")`: the environment wins, and when it is empty the recipe `sven/llm.py` prints
is printed once per process on stderr:

```bash
security add-generic-password -a $USER -s <name> -w        # store it, once
export <ENV>=$(security find-generic-password -a $USER -s <name> -w)   # in ~/.zshrc
```

The Keychain keeps the token off disk in plaintext; a gitignored `.env` also works. A
token is never part of a cache key and never written into a cache entry.

## Mana Pool (prices; `pilot/prices.py`, 2026-10-09)

**The price feed is public — no token.** Verified 2026-10-09 against the official
OpenAPI spec (`https://manapool.com/api/docs/v1/openapi.json`, linked from
`manapool.com/api/docs/v1`): `GET https://manapool.com/api/v1/prices/singles` carries no
security and answered 200 without one. It is every in-stock single, ~104k rows and ~52 MB,
`{meta: {as_of, base_url}, data: [...]}`; each row has `scryfall_id`, `set_code`,
`number`, `price_cents_nm` / `_lp_plus` / `_nm_foil` (and etched, market), `url` and
`available_quantity`. `manamap pilot prices <slug>` reads it by default and keeps the
feed's `meta.as_of` as `feed_as_of` beside the day it was read. A 400/404 on the feed
prints a line and falls through to Scryfall (`/cards/collection`'s `prices.usd` /
`usd_foil`, `source: "scryfall"`), never a traceback; `--source scryfall` asks for that
directly.

The token (`X-ManaPool-Access-Token` + `X-ManaPool-Email`) is needed only for accounts,
orders, `POST /deck` validation and seller inventory — none of which the bench calls. If
both halves are set they ride along on the feed request and change nothing:

```bash
security add-generic-password -a $USER -s manamap-manapool -w                   # the token, once
export MANAPOOL_TOKEN=$(security find-generic-password -a $USER -s manamap-manapool -w)
export MANAPOOL_EMAIL=you@example.com                                           # in ~/.zshrc
```

Paths live in `config.py`: `MANAPOOL_API_BASE`, `MANAPOOL_PRICES_PATH` (`prices/singles`,
GET) and `MANAPOOL_CARD_INFO_PATH` (`card_info`, POST, unused so far). TTLs: the feed is
cached six hours (`MANAPOOL_FEED_TTL_S`, service `manapool`), the collection answer a day
(`SCRYFALL_PRICES_TTL_S`, service `scryfall`). A token is never part of a cache key and
never written into the artifact: `validate-prices` refuses any `url` off `manapool.com` /
`scryfall.com` or carrying a query string. One `@pytest.mark.network` test reads the live
feed.

## Moxfield (export + paste; 2026-10-09) — nothing on this client

Moxfield is the one service the bench publishes to, and nothing in Python talks to it.
There is no public API and no write API; the endpoints its own site calls sit behind
Cloudflare, which answers server and datacenter traffic with a 403 whatever the headers
say, so a fetch from `net.py` — or from `serve.py` — fails before Moxfield sees it. The
sanctioned path to programmatic access is to email support@moxfield.com and ask for a
custom User-Agent to be allow-listed; we have not asked, and nothing here assumes we will.
So the flow is the pilot's own browser both ways: `manamap pilot deck-export <slug>
--format moxfield` prints text Moxfield's import box takes as it stands, the pilot pastes
and saves it, and `deck-link <slug> moxfield <url>` records where it landed in
`links.json`, form-checked by `validate-links` and never fetched. Into the repo, `check-in
--from <url>` refuses any URL with one sentence — export on Moxfield, copy, `check-in
--from -`, paste. `docs/pilot.md` ("Moxfield") has the commands and the three export forms.

## Nothing here is a gate

A command that needs the network asks for it, says so when it cannot have it, and
writes nothing on the way out. `registry.py` put the rule in a `--help` string, on
`validate-brief --themes`, which resolves a theme against EDHREC only when asked: "a
gate that fails when the network is down is a gate that gets switched off". The combos
downloader keeps its dump when the HEAD fails; `fetch-deck` leaves the deck unchanged
and says "run the same command again"; the test suite's one real network leg is the
weekly `corpus-gates` CI job, never a push. An `Offline` reaching a user is an
operating condition, not a bug, and its message is written to be read as one.
