"""`manamap pilot page-state` — what Sean has open in the browser, for Jarvis.

PRD v2 Step 7. "Is this card any good here?" names neither the card nor the deck;
the page Sean is looking at does. Every page served by `manamap serve` reports a
small snapshot of itself — which page, which deck, the Atlas mode, the focused card,
the selection, the filters — through serve's POST `page/state`, and this module
keeps the latest one per browser tab in `.progress/page/state.json`.

  * **Latest only, per tab.** No history: a page reports when its snapshot changes,
    and a tab not heard from in `KEEP_S` is dropped on the next write.
  * **Local only.** The deployed site has no `/api`, so it reports nothing; a plain
    `python -m http.server` reports nothing either. An absent file is an answer:
    nothing is open, or serve is not running.
  * **Never a result.** It is a pointer for resolving "this", gitignored with the
    job band's files (a subdirectory, so the band never reads it as a job), and
    nothing measured or tracked reads it.

`record(tab, state)` is serve's writer; `read()` returns the tabs, most recently
focused first; `main` prints them.

**A pending ASK (2026-10-28).** "Send to Jarvis" on a page posts its snapshot with
an `ask` — `{text, page, deck, cards, branch}`, stamped `at` here — and copies
`/jarvis <text>` to the clipboard for Sean to paste. The ask rides on the tab's
record and keeps its staleness rules (gone with the tab after `KEEP_S`, flagged
STALE past `FRESH_S`). There is ONE pending ask: a newer one, from any tab,
replaces it; a later snapshot without one carries it over; `--clear-ask`
(`clear_ask()`) is how Jarvis marks it taken. The page never calls an LLM, and
neither does this: it is a note on the fridge, not a request.
"""

import contextlib
import json
import os
import threading
import time

try:                      # POSIX: the CLI's --clear-ask and serve's writes are two processes
    import fcntl
except ImportError:       # pragma: no cover - Windows: the in-process lock is all there is
    fcntl = None

from manamap import progress

#: None = `progress.DIR/page/state.json`, resolved per call so a redirected
#: `progress.DIR` (the test suite's) is honoured; tests may set it outright.
PATH = None
#: A tab not heard from in this long is gone (closed, or the machine slept).
KEEP_S = 6 * 3600
#: Older than this and the CLI says so: Sean may have moved on without the page
#: changing (a snapshot is only re-sent when it differs).
FRESH_S = 15 * 60
#: Fields a page may report, and the cap on each list. Anything else is dropped,
#: so a page cannot grow this file into a second store.
#: `panel` is Build's side-panel tab (change / candidates / deck / numbers, 2026-10-28).
#: `ask` is a pending "Send to Jarvis" question (`ASK_FIELDS`, cleaned by `_clean_ask`).
FIELDS = {"page": str, "url": str, "title": str, "deck": str, "branch": str, "mode": str,
          "focus": str, "selected": list, "library": list, "filters": dict, "view": str,
          "panel": str, "visible": bool, "focused": bool, "ask": dict}
MAX_LIST = 40
MAX_TEXT = 300
#: What an ask may carry. `at` is stamped by `record`, never taken from the page.
ASK_FIELDS = {"text": str, "page": str, "deck": str, "branch": str, "cards": list}
#: A question is a sentence or two; longer is a paste that belongs in the terminal.
MAX_ASK = 1000

_lock = threading.Lock()


def _path(path=None):
    return path or PATH or progress.DIR / "page" / "state.json"


def _clean_ask(v):
    """An ask's own fields, capped; None when it carries no question."""
    if not isinstance(v, dict):
        return None
    text = " ".join(str(v.get("text") or "").split())[:MAX_ASK]
    if not text:
        return None
    out = {"text": text}
    for key, kind in ASK_FIELDS.items():
        x = v.get(key)
        if key == "text" or x in (None, "", []):
            continue
        if kind is str:
            out[key] = str(x)[:MAX_TEXT]
        elif isinstance(x, (list, tuple)):
            out[key] = [str(c)[:MAX_TEXT] for c in x[:MAX_LIST]]
    return out


def _clean(state):
    out = {}
    for key, kind in FIELDS.items():
        v = (state or {}).get(key)
        if v is None or v == "" or v == [] or v == {}:
            continue
        if key == "ask":
            ask = _clean_ask(v)
            if ask:
                out[key] = ask
        elif kind is str:
            out[key] = str(v)[:MAX_TEXT]
        elif kind is bool:
            out[key] = bool(v)
        elif kind is list and isinstance(v, (list, tuple)):
            out[key] = [str(x)[:MAX_TEXT] for x in v[:MAX_LIST]]
        elif kind is dict and isinstance(v, dict):
            out[key] = {str(k)[:60]: (v2 if isinstance(v2, (bool, int, float)) else str(v2)[:MAX_TEXT])
                        for k, v2 in list(v.items())[:MAX_LIST]}
    return out


def _load(path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc.get("tabs") or {} if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def _locked(path):
    """One writer at a time: serve's threads (`_lock`) AND the CLI's `--clear-ask`,
    which is another process — without the file lock a snapshot read before the
    clear and written after it would put the answered question back."""
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            yield
            return
        with open(path.with_suffix(".lock"), "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _save(path, tabs):
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"tabs": tabs}, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def record(tab, state, now=None, path=None):
    """Keep `state` as tab `tab`'s latest; drop tabs gone quiet. Returns what was kept.

    A state carrying an `ask` makes it THE pending ask (any other tab's is dropped);
    one without carries this tab's pending ask over, so the next ordinary snapshot
    does not erase a question nobody has answered yet."""
    if not tab or not isinstance(state, dict):
        raise ValueError("page/state needs a tab id and a state object")
    path = _path(path)
    now = time.time() if now is None else now
    tab = str(tab)[:64]
    row = _clean(state)
    row["at"] = round(now, 1)
    with _locked(path):
        tabs = {k: v for k, v in _load(path).items() if now - (v.get("at") or 0) < KEEP_S}
        if row.get("focused"):
            row["focused_at"] = row["at"]
        elif tab in tabs and tabs[tab].get("focused_at"):
            row["focused_at"] = tabs[tab]["focused_at"]
        if "ask" in row:
            row["ask"]["at"] = row["at"]
            for other in tabs.values():
                other.pop("ask", None)
        elif tab in tabs and tabs[tab].get("ask"):
            row["ask"] = tabs[tab]["ask"]
        tabs[tab] = row
        _save(path, tabs)
    return row


def pending_ask(tabs):
    """The one pending ask among `read()`'s tabs, with its tab and age; or None."""
    now = time.time()
    for t in tabs:
        a = t.get("ask")
        if a and a.get("text"):
            return dict(a, tab=t["tab"], age_s=round(now - (a.get("at") or 0)))
    return None


def clear_ask(path=None):
    """Mark the pending ask taken (remove it). Returns the ask cleared, or None."""
    path = _path(path)
    if not path.exists():
        return None
    with _locked(path):
        tabs = _load(path)
        gone = None
        for t in tabs.values():
            a = t.pop("ask", None)
            if a:
                gone = a
        if gone:
            _save(path, tabs)
    return gone


def read(now=None, path=None):
    """The open tabs, the one Sean focused most recently first, each with its age."""
    now = time.time() if now is None else now
    tabs = [dict(v, tab=k, age_s=round(now - (v.get("at") or 0)))
            for k, v in _load(_path(path)).items() if now - (v.get("at") or 0) < KEEP_S]
    return sorted(tabs, key=lambda t: (-(t.get("focused_at") or 0), -(t.get("at") or 0)))


def _ago(s):
    return f"{s}s" if s < 90 else f"{s // 60}m" if s < 5400 else f"{s // 3600}h"


def line(t):
    """One tab in a sentence Jarvis can quote."""
    bits = [t.get("page") or "?"]
    for key, label in (("mode", "mode"), ("deck", "deck"), ("branch", "branch"), ("view", "view"),
                       ("panel", "tab")):
        if t.get(key):
            bits.append(f"{label} {t[key]}")
    if t.get("focus"):
        bits.append(f"focus {t['focus']}")
    if t.get("selected"):
        bits.append(f"selected {', '.join(t['selected'][:8])}"
                    + (f" (+{len(t['selected']) - 8})" if len(t["selected"]) > 8 else ""))
    if t.get("library"):
        bits.append(f"library {len(t['library'])} cards")
    if t.get("filters"):
        bits.append("filters " + ", ".join(f"{k}={v}" for k, v in t["filters"].items()))
    stale = " — STALE, may have moved on" if t["age_s"] > FRESH_S else ""
    return f"{' · '.join(bits)}  ({_ago(t['age_s'])} ago{stale})"


def ask_line(a):
    """The pending ask on its own line, then where it came from."""
    where = [a.get("page") or "?"]
    for key, label in (("deck", "deck"), ("branch", "branch")):
        if a.get(key):
            where.append(f"{label} {a[key]}")
    if a.get("cards"):
        where.append("cards " + ", ".join(a["cards"][:8])
                     + (f" (+{len(a['cards']) - 8})" if len(a["cards"]) > 8 else ""))
    stale = " — STALE, may have moved on" if a["age_s"] > FRESH_S else ""
    return (f"ASK      \"{a['text']}\"\n"
            f"         from {' · '.join(where)}  ({_ago(a['age_s'])} ago{stale})"
            f" — answer it, then `manamap pilot page-state --clear-ask`")


def main(args):
    if getattr(args, "clear_ask", False):
        gone = clear_ask()
        print(f"cleared the pending ask: \"{gone['text']}\"" if gone else "no pending ask")
        return
    tabs = read()
    ask = pending_ask(tabs)
    if getattr(args, "as_json", False):
        print(json.dumps({"tabs": tabs, "ask": ask}, indent=1))
        return
    if ask:
        print(ask_line(ask))
    if not tabs:
        print("nothing open: no page has reported (is `manamap serve` running, and the page "
              "opened through it?)")
        return
    first, rest = tabs[0], tabs[1:]
    print(f"{'FOCUSED' if first.get('focused_at') else 'LATEST ':7}  {line(first)}")
    if first.get("url"):
        print(f"         {first['url']}")
    for t in rest:
        print(f"  also   {line(t)}")
