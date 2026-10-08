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
"""

import json
import os
import threading
import time

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
FIELDS = {"page": str, "url": str, "title": str, "deck": str, "branch": str, "mode": str,
          "focus": str, "selected": list, "library": list, "filters": dict, "view": str,
          "visible": bool, "focused": bool}
MAX_LIST = 40
MAX_TEXT = 300

_lock = threading.Lock()


def _path(path=None):
    return path or PATH or progress.DIR / "page" / "state.json"


def _clean(state):
    out = {}
    for key, kind in FIELDS.items():
        v = (state or {}).get(key)
        if v is None or v == "" or v == [] or v == {}:
            continue
        if kind is str:
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


def record(tab, state, now=None, path=None):
    """Keep `state` as tab `tab`'s latest; drop tabs gone quiet. Returns what was kept."""
    if not tab or not isinstance(state, dict):
        raise ValueError("page/state needs a tab id and a state object")
    path = _path(path)
    now = time.time() if now is None else now
    row = _clean(state)
    row["at"] = round(now, 1)
    with _lock:
        tabs = {k: v for k, v in _load(path).items() if now - (v.get("at") or 0) < KEEP_S}
        if row.get("focused"):
            row["focused_at"] = row["at"]
        elif tab in tabs and tabs[tab].get("focused_at"):
            row["focused_at"] = tabs[tab]["focused_at"]
        tabs[str(tab)[:64]] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"tabs": tabs}, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    return row


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
    for key, label in (("mode", "mode"), ("deck", "deck"), ("branch", "branch"), ("view", "view")):
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


def main(args):
    tabs = read()
    if getattr(args, "as_json", False):
        print(json.dumps({"tabs": tabs}, indent=1))
        return
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
