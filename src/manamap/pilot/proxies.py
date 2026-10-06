"""Pilot: a print-ready proxy sheet for the cards still in the mail.

Codified 2026-10-06 from a sheet built by hand for edgar's mardu-combo-v1 and
sharknado's momentum-v1: a branch's adds, minus what the pilot already has in
hand, laid out 3x3 at TRUE card size (63 x 88 mm at 300 dpi) with crop marks,
one PDF, opened. The pilot sleeves them in front of a basic until the order
arrives.

    manamap pilot proxies edgar-vampires@mardu-combo-v1 sharknado@momentum-v1 \\
        --have "Mana Geyser" --have "Demonic Tutor" --dry-run

TWO THINGS THE FIRST SHEET HAD TO WORK AROUND, both answered here:

  PRINT RESOLUTION. cards.json keeps Scryfall's `normal` image, 488 x 680 —
  blurry at card size. `print_url` swaps it for the `png` render (745 x 1040,
  ~300 dpi at 63 x 88 mm), the same card from the same CDN.

  NO JPEG IN THIS PILLOW. `Image.save("x.pdf")` needs Pillow's JPEG encoder,
  which this build lacks (KeyError: 'JPEG'). Pages go to PDF through `img2pdf`
  as lossless PNGs instead, which also keeps the text crisp.

A print sheet is not a deck artifact: it lands on the Desktop (or `--dest`),
never in data/decks/, and images are cached under data/cache/proxies/
(gitignored). Display only — nothing reads it back.
"""

import hashlib
import io
import math
import re
import subprocess
import time
from pathlib import Path

from manamap import config

DPI = 300
CARD_MM = (63.0, 88.0)
CARD_PX = (round(CARD_MM[0] / 25.4 * DPI), round(CARD_MM[1] / 25.4 * DPI))   # 744 x 1039
PAPER_IN = {"letter": (8.5, 11.0), "a4": (210 / 25.4, 297 / 25.4)}
PER_PAGE = 9
CROP_MARK_PX, CROP_GAP_PX = 60, 10
CACHE = config.DATA_DIR / "cache" / "proxies"


# ── which cards ──────────────────────────────────────────────────────────

def _key(name):
    return str(name or "").split(" // ")[0].strip().lower()


def faces(entry):
    """One proxy per printed face: a double-faced card's back is a second card
    to cut out, so a transform or modal DFC yields two; anything else one."""
    fs = [f for f in (entry.get("card_faces") or []) if f.get("image")]
    if len(fs) >= 2:
        return [{"name": f.get("name") or entry["name"], "image": f["image"],
                 "face": "front" if i == 0 else "back", "card": entry["name"]}
                for i, f in enumerate(fs)]
    image = entry.get("image") or (fs[0]["image"] if fs else None)
    return [{"name": entry["name"], "image": image, "face": "front", "card": entry["name"]}]


def _split_target(target):
    if "@" not in target:
        raise SystemExit(f"{target!r}: name the branch — `<slug>@<branch>` "
                         "(the proxies are a branch's adds)")
    slug, branch = target.split("@", 1)
    return slug, branch


def resolve(targets=(), have=(), names=()):
    """`(proxies, report)`: the cards to print, in order, and what was dropped.

    `targets` are `slug@branch`; their ADDS are taken from `deck_branch.diff`
    (names, never basics — a basic whose count moved is in `quantity`, not
    `add`) and each card's entry from that branch's cards.json. `names` are
    one-off cards from the local corpus. `have` removes cards the pilot holds;
    a `have` that matches nothing is REPORTED, so a typo cannot silently print
    the card it was meant to skip."""
    from manamap.pilot import deck_branch
    from manamap.pilot.common import load_deck_cards

    wanted, seen = [], set()
    for target in targets:
        slug, branch = _split_target(target)
        adds = deck_branch.diff(slug, branch)["add"]
        doc = load_deck_cards(slug, branch)
        by_key = {_key(c["name"]): c for c in doc["cards"]}
        for name in adds:
            entry = by_key.get(_key(name))
            if entry is None:
                raise SystemExit(f"{slug}@{branch}: {name!r} is an add but is not in the "
                                 f"branch's cards.json — `manamap pilot fetch-deck {slug} "
                                 f"--branch {branch}` first")
            if _key(name) not in seen:
                seen.add(_key(name))
                wanted.append((f"{slug}@{branch}", entry))
    if names:
        from manamap.pilot.try_swap import corpus_card
        for name in names:
            entry = corpus_card(name)
            if entry is None:
                raise SystemExit(f"{name!r} is not in the local card corpus")
            if _key(name) not in seen:
                seen.add(_key(name))
                wanted.append(("--card", entry))

    have_keys = {_key(h): h for h in have if str(h).strip()}
    matched = {k for k in have_keys if any(_key(e["name"]) == k for _w, e in wanted)}
    proxies = [dict(p, source=where)
               for where, entry in wanted if _key(entry["name"]) not in have_keys
               for p in faces(entry)]
    missing = [p["name"] for p in proxies if not p["image"]]
    if missing:
        raise SystemExit(f"no image for: {', '.join(missing)}")
    return proxies, {"kept_back": sorted(have_keys[k] for k in matched),
                     "unmatched_have": sorted(have_keys[k] for k in set(have_keys) - matched)}


# ── images ───────────────────────────────────────────────────────────────

def print_url(url):
    """Scryfall's `normal` JPEG -> its `png` render (745 x 1040), the size that
    prints sharp at 63 x 88 mm. Same card, same CDN path, other size folder."""
    if "/png/" in url:
        return url
    url = url.split("?", 1)[0]
    url = re.sub(r"/(normal|large|small)/", "/png/", url, count=1)
    return re.sub(r"\.(jpg|jpeg)$", ".png", url)


def _requests_get(url):
    import requests
    r = requests.get(url, headers={"User-Agent": config.USER_AGENT, "Accept": "image/png"},
                     timeout=30)
    r.raise_for_status()
    return r.content


def fetch(url, cache_dir=None, get=_requests_get):
    """The print image's bytes, cached on disk by URL. A cache hit costs no
    request; a miss sleeps Scryfall's asked-for delay after fetching."""
    cache_dir = Path(cache_dir or CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".png")
    if path.exists():
        return path.read_bytes()
    data = get(url)
    path.write_bytes(data)
    time.sleep(config.SCRYFALL_REQUEST_DELAY_S)
    return data


# ── layout ───────────────────────────────────────────────────────────────

def layout(n, paper="letter"):
    """`(page_px, pages, slots)`: page size in pixels, page count, and each
    card's `(page, x, y)` top-left — a 3 x 3 grid centred on the page."""
    if paper not in PAPER_IN:
        raise SystemExit(f"--paper must be one of {', '.join(PAPER_IN)}")
    pw, ph = (round(v * DPI) for v in PAPER_IN[paper])
    cw, ch = CARD_PX
    x0, y0 = (pw - 3 * cw) // 2, (ph - 3 * ch) // 2
    if x0 < 0 or y0 < 0:
        raise SystemExit(f"a 3 x 3 grid of cards does not fit on {paper}")
    slots = [(i // PER_PAGE, x0 + (i % PER_PAGE % 3) * cw, y0 + (i % PER_PAGE // 3) * ch)
             for i in range(n)]
    return (pw, ph), math.ceil(n / PER_PAGE), slots


def render(images, paper="letter"):
    """PIL pages: each card image (bytes) at true size, its transparent rounded
    corners on white, crop marks on every grid line OUTSIDE the grid so a cut
    never shows a line on a card."""
    from PIL import Image, ImageDraw

    (pw, ph), npages, slots = layout(len(images), paper)
    cw, ch = CARD_PX
    pages = [Image.new("RGB", (pw, ph), "white") for _ in range(npages)]
    for data, (p, x, y) in zip(images, slots):
        card = Image.open(io.BytesIO(data)).convert("RGBA").resize((cw, ch), Image.LANCZOS)
        white = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
        white.alpha_composite(card)
        pages[p].paste(white.convert("RGB"), (x, y))
    x0, y0 = (pw - 3 * cw) // 2, (ph - 3 * ch) // 2
    for page in pages:
        d = ImageDraw.Draw(page)
        for i in range(4):
            gx, gy = x0 + i * cw, y0 + i * ch
            for a, b in (((gx, y0 - CROP_GAP_PX - CROP_MARK_PX), (gx, y0 - CROP_GAP_PX)),
                         ((gx, y0 + 3 * ch + CROP_GAP_PX), (gx, y0 + 3 * ch + CROP_GAP_PX + CROP_MARK_PX)),
                         ((x0 - CROP_GAP_PX - CROP_MARK_PX, gy), (x0 - CROP_GAP_PX, gy)),
                         ((x0 + 3 * cw + CROP_GAP_PX, gy), (x0 + 3 * cw + CROP_GAP_PX + CROP_MARK_PX, gy))):
                d.line([a, b], fill="black", width=2)
    return pages


def write_pdf(pages, dest, paper="letter"):
    """Lossless PNG pages into one PDF at the paper's exact size (img2pdf)."""
    import img2pdf

    blobs = []
    for page in pages:
        buf = io.BytesIO()
        page.save(buf, format="PNG", dpi=(DPI, DPI))
        blobs.append(buf.getvalue())
    w, h = PAPER_IN[paper]
    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img2pdf.convert(
        blobs, layout_fun=img2pdf.get_layout_fun((img2pdf.in_to_pt(w), img2pdf.in_to_pt(h)))))
    return dest


# ── the command ──────────────────────────────────────────────────────────

def default_dest(targets, names=()):
    slugs = []
    for t in targets:
        s = t.split("@", 1)[0]
        if s not in slugs:
            slugs.append(s)
    stem = "-".join(slugs) if slugs else "cards"
    return Path.home() / "Desktop" / f"{stem}-proxies.pdf"


def _read_have(args):
    have = list(getattr(args, "have", None) or [])
    path = getattr(args, "have_file", None)
    if path:
        for line in Path(path).expanduser().read_text().splitlines():
            line = re.sub(r"^\s*\d+\s*x?\s+", "", line.strip())     # "1 Card" / "1x Card"
            if line and not line.startswith("#"):
                have.append(line)
    return have


def main(args, get=_requests_get, opener=("open",)):
    targets = list(getattr(args, "targets", None) or [])
    names = list(getattr(args, "card", None) or [])
    if not targets and not names:
        raise SystemExit("name at least one `<slug>@<branch>` or `--card NAME`")
    paper = getattr(args, "paper", None) or "letter"
    proxies, report = resolve(targets, _read_have(args), names)

    print(f"PROXIES — {len(proxies)} to print "
          f"({math.ceil(len(proxies) / PER_PAGE) if proxies else 0} page(s), {paper})")
    for p in proxies:
        back = "  (back face)" if p["face"] == "back" else ""
        print(f"  {p['name']:40} {p['source']}{back}")
    if report["kept_back"]:
        print(f"  not printed, you have them: {len(report['kept_back'])}")
    if report["unmatched_have"]:
        print("  WARNING — these --have names match no card being printed (typo?): "
              + ", ".join(report["unmatched_have"]))
    if not proxies or getattr(args, "dry_run", False):
        if getattr(args, "dry_run", False):
            print("  --dry-run: nothing downloaded or written")
        return

    images = [fetch(print_url(p["image"]), get=get) for p in proxies]
    dest = write_pdf(render(images, paper), getattr(args, "dest", None)
                     or default_dest(targets, names), paper)
    print(f"  wrote {dest}")
    print("  PRINT AT 100% / ACTUAL SIZE (never 'fit to page'); cut on the crop marks; "
          "sleeve each in front of a basic land")
    if not getattr(args, "no_open", False) and opener:
        subprocess.run([*opener, str(dest)], check=False)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot proxies`")
