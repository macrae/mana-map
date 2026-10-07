"""Pilot: the gate on `watchlist.json` — candidate sets Sean is reviewing.

Every card is a real, Commander-legal card inside the commander's colour identity;
set ids are unique; `pays`, `axis` and `verdict` are from their closed vocabularies;
a `Q…` source names a real queue item. A candidate that has since JOINED the 99 is
a warning, not an error: a watch list is dated, and the list moves under it.
"""
from manamap.pilot import watchlist as wl
from manamap.pilot.common import load_json, report_errors


def validate(slug, doc, warnings=None):
    from manamap.pilot import queue
    from manamap.pilot.card_pool import load_pool
    from manamap.pilot.card_search import commander_identity, deck_names

    errors = []
    if doc.get("slug") != slug:
        errors.append(f"slug is {doc.get('slug')!r} but the file lives in {slug}/")
    sets = doc.get("sets")
    if not isinstance(sets, list):
        return errors + ["`sets` must be a list"]
    pool = load_pool()
    try:
        ident = set(commander_identity(slug))
        present = {n.lower() for n in deck_names(slug)}
    except FileNotFoundError:
        ident, present = None, set()
    items = queue.items(queue.read())
    ids = set()
    for i, s in enumerate(sets):
        where = f"sets[{i}]"
        sid = s.get("id")
        if not sid or not str(sid).replace("-", "").isalnum():
            errors.append(f"{where}: id must be kebab-case")
        if sid in ids:
            errors.append(f"{where}: id {sid!r} appears twice")
        ids.add(sid)
        if not str(s.get("title") or "").strip():
            errors.append(f"{where}: needs a title")
        src = s.get("source")
        if src and str(src).startswith("Q") and src not in items:
            errors.append(f"{where}: source {src!r} is not a queue item")
        seen = set()
        for j, c in enumerate(s.get("cards") or []):
            at = f"{where}.cards[{j}] ({c.get('name')})"
            name = c.get("name")
            if not name or name.lower() in seen:
                errors.append(f"{at}: missing or repeated")
                continue
            seen.add(name.lower())
            if c.get("pays") not in wl.PAYS:
                errors.append(f"{at}: pays is one of {', '.join(wl.PAYS)}")
            if c.get("axis") not in wl.AXES:
                errors.append(f"{at}: axis is one of {', '.join(wl.AXES)}")
            if c.get("verdict") not in wl.VERDICTS:
                errors.append(f"{at}: verdict is one of {', '.join(wl.VERDICTS)}")
            if not str(c.get("why") or "").strip():
                errors.append(f"{at}: needs a why")
            if pool is not None:
                rec = pool.get(name)
                if rec is None:
                    errors.append(f"{at}: not in the corpus")
                    continue
                if not rec["legal"]:
                    errors.append(f"{at}: not Commander-legal")
                if ident is not None and not rec["color_identity"] <= ident:
                    errors.append(f"{at}: outside the commander's colour identity")
            if name.lower() in present and warnings is not None:
                warnings.append(f"{at}: now in the 99")
        if not s.get("cards"):
            errors.append(f"{where}: holds no cards")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    p = wl.path(slug)
    if not p.exists():
        print(f"{slug}: no {wl.ARTIFACT} — nothing on watch (absent means absent)")
        return
    doc = load_json(p) or {}
    warnings = []
    report_errors(f"{slug} — {wl.ARTIFACT}", validate(slug, doc, warnings))
    n = sum(len(s.get("cards") or []) for s in doc["sets"])
    print(f"OK   {slug} — {wl.ARTIFACT}: {len(doc['sets'])} set(s), {n} card(s)"
          + "".join(f"\n  · {w}" for w in warnings))
