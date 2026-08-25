/* Discovery — the front door.
 *
 * The map used to be where you arrived: 34,322 points and a request that you already
 * know where to look. This lands you on ONE card instead, and the graph grows from
 * whatever you click.
 *
 * Two artifacts make that cheap enough to feel instant:
 *
 *   viz_index.json   0.56 MB gz   name/type/colour/rarity/cmc/roles per card
 *   neighbours.bin   1.27 MB gz   12 similar + 10 synergy + 5 obsoleted-by row ids
 *
 * 1.83 MB to be USABLE, against the 18.4 MB it used to take to reach a first branch
 * (12.9 MB projection, then 16.8 MB of incompressible float32 embeddings on the first
 * click). Measured, and worth stating precisely: the page also fetches the projection
 * (2.90 gz) and region labels (0.06 gz) unconditionally behind the landing, so 4.80 MB
 * crosses the wire in total — discovery is simply usable long before that lands.
 * The point is not the megabytes — it is that branching becomes **synchronous**. An
 * await inside a click is what makes a graph feel laggy instead of physical.
 *
 * THE RULE, which the artifact's own docstring states and a Python test enforces:
 * the neighbour lists are stored pre-sorted and must never be re-sorted here. Their
 * similarities are uint8-quantised for edge length only. This embedding is a narrow
 * cone — median pairwise cosine 0.714 — so sorting by a lossy value reorders the
 * top-10 for roughly two thirds of cards, and it would look like a model regression
 * rather than a precision artefact.
 */
window.Discovery = (function () {
  'use strict';

  const MAGIC = 'MMNB';
  let index = null;      // [{n,s,c,r,m,g?}] positionally aligned with cards.csv
  let table = null;      // decoded neighbours.bin
  let indexPromise = null, tablePromise = null;
  // URLs are injected rather than read from MM.DATA. `window.MM` is only assigned at the
  // very END of mana-map.js's IIFE, and the boot code that starts discovery runs inside
  // that IIFE — so reading MM here threw, which aborted the IIFE, which meant MM was
  // never exported at all and every later module failed too. One missing global, four
  // broken files.
  let urls = { vizIndex: '../data/viz_index.json', neighbours: '../data/neighbours.bin',
               deckIndex: '../data/decks/index.json', deckBase: '../data/decks/',
               // Local-only scratch: `data/reference/` is gitignored, so a
               // deployed page never has one and says so rather than failing quietly.
               referenceBase: '../data/reference/' };

  function configure(u) { urls = Object.assign({}, urls, u || {}); }

  // Regions big enough to be a graph and small enough to stay under the node cap. Loaded
  // once, lazily, and only used to render the seed list — a failure here just means the
  // list is absent, never that discovery breaks.
  let regionSeeds = [];
  function loadRegionSeeds() {
    if (regionSeeds.length || !window.MM || !MM.getRegionData) return;
    MM.getRegionData().then(function (rd) {
      if (!rd || !rd.membership) return;
      regionSeeds = rd.regions
        .filter(function (r) { return r.level === 1 && r.count >= 60 && r.count <= 500; })
        .sort(function (a, b) { return b.count - a.count; })
        .slice(0, 8);
      if (isReady()) render();
    }).catch(function () { /* the graph works without them */ });
  }

  // ── loading ────────────────────────────────────────────────────────────

  function loadIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(urls.vizIndex)
      .then(r => { if (!r.ok) throw new Error('viz_index ' + r.status); return r.json(); })
      .then(rows => {
        index = rows;
        /* Hand Session the card index and let it restore the saved library.
         *
         * HERE, and not at load: `session.js` runs before `mana-map.js` and has
         * no name resolver of its own — reaching for `MM.*` at its top level
         * would run inside that IIFE before `window.MM` exists, which is the
         * boot-order failure that once broke four files from one mistake. This
         * is the first moment a name CAN become a row, so it is the moment the
         * library comes back. */
        if (window.Session && Session.useCards) {
          watchLibrary();
          const report = Session.useCards({
            nameOf: function (row) { return index[row] ? index[row].n : null; },
            rowOf: rowByName,
            // The corpus's own identity. Names are stable so this does not
            // invalidate a save; it EXPLAINS a shortfall instead of causing one.
            fingerprint: function () { return String(index.length); },
          });
          if (report && report.missing.length && window.MM) {
            MM.setStatus(report.restored + ' cards restored to your library — '
              + report.missing.length + ' no longer in the corpus: '
              + report.missing.slice(0, 3).join(', '));
          }
        }
        return rows;
      });
    return indexPromise;
  }

  function decode(buf) {
    const dv = new DataView(buf);
    let magic = '';
    for (let i = 0; i < 4; i++) magic += String.fromCharCode(dv.getUint8(i));
    if (magic !== MAGIC) throw new Error('neighbours.bin: bad magic ' + JSON.stringify(magic));

    const n = dv.getUint32(8, true);
    const ks = dv.getUint16(12, true);
    const ky = dv.getUint16(14, true);
    const ko = dv.getUint16(16, true);
    const lo = dv.getFloat32(52, true);
    const hi = dv.getFloat32(56, true);
    const vocabLen = dv.getUint32(60, true);

    // Every Uint16Array view below is 2-aligned because the header is 64 bytes and the
    // uint16 blocks are contiguous after it. A misaligned view throws here, at load,
    // nowhere near whatever changed upstream — hence the writer's block ordering.
    let off = 64;
    const simIdx = new Uint16Array(buf, off, n * ks); off += n * ks * 2;
    const synIdx = new Uint16Array(buf, off, n * ky); off += n * ky * 2;
    const obsIdx = new Uint16Array(buf, off, n * ko); off += n * ko * 2;
    const simVal = new Uint8Array(buf, off, n * ks); off += n * ks;
    const synReason = new Uint8Array(buf, off, n * ky); off += n * ky;
    const counts = new Uint8Array(buf, off, n * 3); off += n * 3;
    // The reason codebook rides in the file rather than being a third fetch.
    const reasons = vocabLen
      ? JSON.parse(new TextDecoder().decode(new Uint8Array(buf, off, vocabLen)))
      : [];

    return { n, ks, ky, ko, lo, hi, simIdx, synIdx, obsIdx, simVal, synReason, counts,
             reasons, NONE: 0xFFFF, NO_REASON: 0xFF };
  }

  function loadNeighbours() {
    if (tablePromise) return tablePromise;
    tablePromise = fetch(urls.neighbours)
      .then(r => { if (!r.ok) throw new Error('neighbours ' + r.status); return r.arrayBuffer(); })
      .then(buf => { table = decode(buf); return table; });
    return tablePromise;
  }

  function ready() { return Promise.all([loadIndex(), loadNeighbours(), loadManifest()]); }
  function isReady() { return !!(index && table); }

  // ── the card record ────────────────────────────────────────────────────

  /* The slim record, which is all the landing needs. `MM.cardRecord` prefers the full
   * projection row when it has arrived and falls back to this — so discovery paints
   * immediately and gets richer, rather than waiting. */
  function record(row) {
    return index ? index[row] : null;
  }

  // ── neighbours, synchronously ──────────────────────────────────────────

  /* [{row, sim}] for one relation, already in the right order. `sim` is decoded from
   * uint8 for edge length; do not sort by it. */
  function neighbours(row, relation) {
    if (!table) return [];
    const out = [];
    if (relation === 'synergy') {
      const base = row * table.ky;
      for (let i = 0; i < table.ky; i++) {
        const r = table.synIdx[base + i];
        if (r === table.NONE) continue;
        // The reason is what makes a synergy edge worth drawing: it says WHY these two
        // cards are connected, not merely that a rule fired. These strings were already
        // being computed and thrown away — the old Find Synergies wrote them into a
        // Plotly trace and then set hoverinfo:'none', so nobody ever saw one.
        const code = table.synReason[base + i];
        out.push({ row: r, sim: 0.55, relation: 'synergy',
                   reason: code === table.NO_REASON ? null : (table.reasons[code] || null) });
      }
    } else if (relation === 'obsolete') {
      const base = row * table.ko;
      for (let i = 0; i < table.ko; i++) {
        const r = table.obsIdx[base + i];
        if (r !== table.NONE) out.push({ row: r, sim: 0.9, relation: 'obsolete' });
      }
    } else {
      const base = row * table.ks;
      const span = table.hi - table.lo;
      for (let i = 0; i < table.ks; i++) {
        const r = table.simIdx[base + i];
        if (r === table.NONE) continue;
        out.push({ row: r, sim: table.simVal[base + i] / 255 * span + table.lo,
                   relation: 'similar' });
      }
    }
    return out;
  }

  /* What this card actually has. Precomputed, so the UI can state it BEFORE a click
   * rather than discovering emptiness after one — 23.6% of cards have nothing but
   * similar, and a button that does nothing reads as broken rather than as a fact
   * about the card. */
  function counts(row) {
    if (!table) return { similar: 0, synergy: 0, obsolete: 0 };
    const b = row * 3;
    return {
      similar: table.counts[b],
      synergy: table.counts[b + 1],
      obsolete: table.counts[b + 2],
    };
  }

  // ── picking a card ─────────────────────────────────────────────────────

  const COLORS = ['W', 'U', 'B', 'R', 'G', 'Multicolor', 'Colorless'];
  const CMC_BANDS = { any: null, low: [0, 2], mid: [3, 5], high: [6, 99] };
  let filters = { supertype: '', color: '', cmc: 'any' };

  function matches(rec, f) {
    if (f.supertype && rec.s !== f.supertype) return false;
    if (f.color && rec.c !== f.color) return false;
    const band = CMC_BANDS[f.cmc];
    if (band && (rec.m < band[0] || rec.m > band[1])) return false;
    return true;
  }

  /* A uniform draw over 34,322 rows lands on an obscure card essentially every time —
   * as the entire first impression. Weight toward cards that have somewhere to go and
   * something to say, using data we already have rather than editorial judgement. */
  function weight(row, rec) {
    const c = counts(row);
    let w = 1;
    if (c.synergy > 0) w += 2;
    if (rec.g && rec.g.length) w += 1;
    if (rec.r === 'rare' || rec.r === 'mythic') w += 2;
    else if (rec.r === 'uncommon') w += 1;
    return w;
  }

  /* Seeded so a landing is reproducible. Without this every browser test starts from a
   * random state, and `?card=` / `?seed=` could not exist. */
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return function () {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }

  function pick(f, seed) {
    if (!index) return -1;
    const use = f || filters;
    const rand = rng(seed || (Date.now() & 0x7fffffff));
    const pool = [];
    let total = 0;
    for (let i = 0; i < index.length; i++) {
      if (!matches(index[i], use)) continue;
      const w = weight(i, index[i]);
      total += w;
      pool.push({ row: i, acc: total });
    }
    if (!pool.length) return -1;
    const target = rand() * total;
    let lo = 0, hi = pool.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (pool[mid].acc < target) lo = mid + 1; else hi = mid;
    }
    return pool[lo].row;
  }

  let nameMap = null;

  function ensureNameMap() {
    if (nameMap || !index) return nameMap;
    // Built once and reused. A linear scan is 34,322 comparisons per lookup, which is
    // fine for one card and 3.4M for a 100-card import.
    nameMap = new Map();
    for (let i = 0; i < index.length; i++) {
      const key = index[i].n.toLowerCase();
      if (!nameMap.has(key)) nameMap.set(key, i);   // first printing wins, as everywhere
    }
    return nameMap;
  }

  function rowByName(name) {
    const m = ensureNameMap();
    if (!m) return -1;
    const want = String(name).trim().toLowerCase();
    if (m.has(want)) return m.get(want);
    // Decklists often carry only the front face of a double-faced card. cards.csv keys
    // the full "A // B" form, so fall back to a front-face match before giving up.
    for (const [key, row] of m) {
      const cut = key.indexOf(' // ');
      if (cut > 0 && key.slice(0, cut) === want) return row;
    }
    return -1;
  }

  function poolSize(f) {
    if (!index) return 0;
    const use = f || filters;
    let n = 0;
    for (let i = 0; i < index.length; i++) if (matches(index[i], use)) n++;
    return n;
  }

  function setFilter(key, value) {
    filters[key] = value;
    return poolSize();
  }

  function getFilters() { return Object.assign({}, filters); }

  // ── the landing ────────────────────────────────────────────────────────

  let current = -1;

  function panel() { return document.getElementById('deckPanel'); }
  function inner() { return document.getElementById('deckInner'); }

  function optionsFor(values, selected, anyLabel) {
    return ['<option value="">' + anyLabel + '</option>'].concat(
      values.map(v => '<option value="' + v + '"' +
        (v === selected ? ' selected' : '') + '>' + v + '</option>')
    ).join('');
  }

  function render() {
    const el = inner();
    if (!el) return;
    panel().classList.add('open');
    loadRegionSeeds();

    // Read once, at the top: the seed controls need it ABOVE the graph section
    // that computes it, and "is there a walk to add to" must be the same answer
    // in both places within one render.
    const graphN0 = window.Force ? Force.nodeCount : 0;

    const rec = current >= 0 ? index[current] : null;
    const c = current >= 0 ? counts(current) : { similar: 0, synergy: 0, obsolete: 0 };
    const types = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact',
                   'Land', 'Planeswalker', 'Battle'];

    let html = '<div class="deck-header"><h2>Discover</h2>' +
      '<button class="lens-btn lens-btn-inline" onclick="Discovery.reroll()">Feeling lucky ↻</button>' +
      '</div>';

    /* THE CARD LEADS.
     *
     * It used to render NINTH, under a deck picker, a paste box, an export
     * button, a Clear, a seed box and three filter rows — sixteen equal-weight
     * buttons above the one thing the front door is about. The landing is a
     * card and its relations; everything else is a way of choosing a different
     * card, which is a smaller question and now looks like one.
     *
     * GUARDED, because it moved above the `!rec` early return that used to
     * protect it — the filters can exclude everything, and then there is no
     * card to lead with.
     */
    if (rec) {
      html += '<div class="lens-title">' + rec.n + '</div>';
    // The relation buttons used to be built here. They now live in the shared card HTML
    // (MM.buildRelationHtml) so every panel that shows a card gets the same control —
    // which is what makes deleting the old Find Similar / Find Synergies pair an
    // unification rather than the removal of a feature.
    // The Keep button is part of the shared card HTML now (MM.buildRelationHtml), so
    // every panel that shows a card can put it in the library.
      html += MM.buildCardDetailHtml(MM.cardRecord(current), current);
    }

    /* Everything below is a way of choosing a DIFFERENT card, which is a
     * smaller question than the card itself — so it is one collapsed block
     * rather than nine buttons competing with the thing you came for. */
    html += '<details class="discover-more"><summary>Start somewhere else</summary>';

    html += '<div class="discover-decks">' +
      '<select id="dcDeck" onchange="Discovery.onDeckPick(this.value)">' +
      '<option value="">Load one of my decks…</option>' +
      (manifest || []).map(d => '<option value="' + d.slug + '">' + d.deck_name +
        ' — ' + d.commander + '</option>').join('') +
      '</select></div>';

    /* THE LIBRARY'S CONTROLS LIVE IN THE DRAWER, not here. They used to sit
     * inside this collapsed `<details>` — so the count, Export and an
     * unconfirmed Clear were all hidden behind a disclosure labelled "Start
     * somewhere else", which is a destructive control mislabelled by its
     * container. What is left is a count and the one action that is genuinely
     * about the WALK rather than about the library. The count reads the same
     * array the strip does, so the two cannot disagree. */
    html += '<div class="discover-tray">' +
      '<button class="lens-btn" onclick="Discovery.toggleImport()">Paste a decklist</button>' +
      '<span class="discover-traycount">' + Session.library.size + ' in library</span>' +
      (Session.library.size
        ? '<button class="lens-btn" onclick="Shell.toggle()">Show them</button>' +
          '<button class="lens-btn" onclick="Discovery.exportBrief()">Export brief</button>'
        : '') +
      '</div>';
    /* START FROM CARDS YOU NAME.
     *
     * Two buttons, not one, because there are two different acts here and this
     * repo has already paid for a control that meant different things in
     * different states. "Start here" REPLACES the graph — a fresh walk, which
     * is the explicit request that makes replacement legitimate. "Add to walk"
     * GROWS it, and only appears once there is a walk to add to. A single
     * button that silently switched between them would destroy a graph the
     * pilot spent ten minutes building, with no way to tell in advance. */
    html += '<div class="discover-tray">' +
      '<button class="lens-btn" onclick="Discovery.toggleSeed()">Start from cards I name</button>' +
      '</div>';
    html += '<div id="dcSeedWrap" style="display:none">' +
      '<textarea id="dcSeed" class="discover-import" rows="3" ' +
      'placeholder="Zur, the Enchanter&#10;Sol Ring&#10;&#10;or: 1) zur, 2) sol ring"></textarea>' +
      '<div class="discover-tray">' +
        '<button class="lens-btn" onclick="Discovery.onSeedCards(true)">Start here</button>' +
        (graphN0 ? '<button class="lens-btn" onclick="Discovery.onSeedCards(false)">Add to walk</button>' : '') +
      '</div>' +
      '<p class="lens-note">One card per line. Commas stay inside a name — ' +
      '"Zur, the Enchanter" is one card — so number the items to put several on one line.</p></div>';

    html += '<div id="dcImportWrap" style="display:none">' +
      '<textarea id="dcImport" class="discover-import" rows="5" ' +
      'placeholder="1 Sol Ring&#10;1 Edgar Markov *CMDR*&#10;&#10;Moxfield exports work as-is."></textarea>' +
      '<button class="lens-btn" onclick="Discovery.onImport()">Load deck as a graph</button>' +
      '<p class="lens-note">Resolved against the card index, not the published decks — ' +
      'any list works, it does not have to be one of the seven.</p></div>';

    // The graph you are holding, and the controls that act on it. Ported from the walk
    // panel when The Walk was deleted: it was the same force engine with different chrome,
    // but its panel was the only home for Fit, Reheat, New walk, the scoreboard and — the
    // one that was an actual defect — the truncation notice. Discovery has always
    // truncated a >500-card import to MAX_NODES and never said so.
    const graphN = window.Force ? Force.nodeCount : 0;
    // The graph controls act on the walk you HAVE, so they close the "start
    // somewhere else" block rather than hiding inside it — and they only exist
    // once there is more than the landing card to act on.
    html += '</details>';

    if (graphN > 1) {
      const cut = Force.truncatedFrom;
      html += '<div class="deck-section">' +
        '<div class="lens-stats">' +
          '<div class="lens-stat"><div class="lens-stat-n">' + graphN + '</div><div class="lens-stat-l">cards</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + Force.linkCount + '</div><div class="lens-stat-l">links</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + Force.trailLength + '</div><div class="lens-stat-l">visited</div></div>' +
          '<div class="lens-stat"><div class="lens-stat-n">' + Force.MAX_NODES + '</div><div class="lens-stat-l">cap</div></div>' +
        '</div>' +
        (cut ? '<div class="lens-note">seeded with ' + Force.MAX_NODES + ' of ' + cut +
               ' — the rest were dropped to keep the graph readable</div>' : '') +
        '<div class="discover-graphctl">' +
          '<button class="lens-btn" onclick="Force.fit()">Fit</button>' +
          '<button class="lens-btn" onclick="Force.reheat()">Reheat</button>' +
          '<button class="lens-btn" onclick="Discovery.newGraph()">Start over ↺</button>' +
        '</div>' +
        '<div class="lens-note">Drag a card to fling it · click to branch · scroll to zoom</div>' +
        '</div>';
    }

    // Seed from a region. This lived in the walk's empty state and was the ONLY
    // region -> graph path in the app (`Drill.enterRegion` re-embeds a region into the
    // atlas; it never seeds the graph). Deleting The Walk without this would have deleted
    // the capability.
    if (regionSeeds.length) {
      html += '<details class="discover-more"><summary>Or start from a region</summary>' +
        regionSeeds.map(function (r) {
          return '<div class="lens-cand" onclick="Force.walkRegion(' +
            JSON.stringify(r.id).replace(/"/g, '&quot;') + ')">' +
            '<span class="lens-cand-name">' + MM.escHtml(r.short || r.label) + '</span>' +
            '<span class="lens-chip">' + r.count + '</span></div>';
        }).join('') + '</div>';
    }

    html += '</details>';

    html += '<div class="discover-filters">' +
      '<select id="dcType" onchange="Discovery.onFilter(\'supertype\', this.value)">' +
        optionsFor(types, filters.supertype, 'Any type') + '</select>' +
      '<select id="dcColor" onchange="Discovery.onFilter(\'color\', this.value)">' +
        optionsFor(COLORS, filters.color, 'Any colour') + '</select>' +
      '<select id="dcCmc" onchange="Discovery.onFilter(\'cmc\', this.value)">' +
        '<option value="any">Any cost</option>' +
        '<option value="low"' + (filters.cmc === 'low' ? ' selected' : '') + '>0–2</option>' +
        '<option value="mid"' + (filters.cmc === 'mid' ? ' selected' : '') + '>3–5</option>' +
        '<option value="high"' + (filters.cmc === 'high' ? ' selected' : '') + '>6+</option>' +
      '</select></div>';

    if (!rec) {
      html += '<p class="lens-empty">No card matches those filters.</p>';
      el.innerHTML = html;
      return;
    }


    if (window.Force && Force.trailLength > 1) {
      html += '<div class="deck-section"><div class="deck-section-title">Where you have been ' +
        '<span>' + Force.trailLength + '</span></div>' +
        Force.trailNames().slice().reverse().map(function (n) {
          return '<div class="lens-cand"><span class="lens-cand-name">' + MM.escHtml(n) + '</span></div>';
        }).join('') +
        '<button class="lens-btn" onclick="Force.clearTrail()">Clear the trail</button></div>';
    }
    el.innerHTML = html;
  }

  /* Start over: empty the graph and land somewhere new. The walk panel called
   * `Force.newWalk()` (loud), which rendered the deck/region picker; here the picker is
   * already in the panel above, so this reseeds on a fresh card instead of leaving you
   * looking at an empty canvas. */
  function newGraph() {
    if (window.Force) Force.newWalk(true);
    show(pick());
  }

  function relBtn(rel, label, n) {
    const dead = n === 0;
    return '<button class="lens-btn discover-rel' + (dead ? ' is-empty' : '') + '"' +
      (dead ? ' disabled' : ' onclick="Discovery.walk(\'' + rel + '\')"') +
      '>' + label + ' <span class="discover-count">' + n + '</span></button>';
  }

  /* Show a card. This is the landing: one card, its art, and what it connects to. */
  /* Show a card: the landing IS the graph, seeded with one node. That is what makes the
   * card float in space rather than sit in a panel beside an empty canvas, and it means
   * picking a relation grows what is already on screen instead of swapping views. */
  function show(row) {
    current = row;
    Session.setFocus(row);
    if (row >= 0 && window.Force) {
      // The landing is the one seed that is unconditionally a REPLACE: it is
      // what "Feeling lucky" means, and there is nothing to lose on boot.
      seedFromRows([row], index[row].n, {});
    }
    render();
    if (row >= 0) {
      MM.setStatus(index[row].n + ' — pick a relation, or hit Feeling lucky for another card.');
    }
  }

  function land(params) {
    const wanted = params && params.get('card');
    const seed = params && params.get('seed');

    /* `?cards=Zur+the+Enchanter,Sol+Ring` — a walk you can link to.
     *
     * Query strings are the one place a comma IS a conventional separator, and
     * `URLSearchParams` has already decoded any `%2C` a real name carried — so
     * by the time the value is read, the two are indistinguishable. Rather than
     * guess, the parameter is handed to the SAME reader the textarea uses: a
     * bare comma stays inside a name, and numbering separates items. So the
     * shareable form of "Zur, the Enchanter + Sol Ring" is
     * `?cards=1) Zur, the Enchanter, 2) Sol Ring`, which is ugly and correct,
     * and `?cards=Sol Ring,Lightning Bolt` — no commas in either name — is
     * simply one unresolvable string that reports itself as unresolved instead
     * of half-working.
     *
     * A newline works too, and is the unambiguous form: `%0A` between names. */
    const many = params && params.get('cards');
    if (many) {
      const out = parseSeedNames(many);
      if (out.rows.length) {
        // One seed, not `show()` then a reseed: seeding twice draws the first
        // card, throws it away and draws the set, which reads as a flicker on
        // the very first frame the pilot sees.
        const label = out.rows.length === 1 ? index[out.rows[0]].n
          : index[out.rows[0]].n + ' +' + (out.rows.length - 1);
        Promise.resolve(seedFromRows(out.rows, label, {})).then(function () {
          MM.setStatus(label + (out.missing.length
            ? ' — no match for ' + out.missing.slice(0, 3).join(', ')
            : ' — pick a relation to grow the graph.'));
        });
        return;
      }
    }

    /* `?ref=<slug>` — a REFERENCE deck, opened from commander search. §6.1 §9.
     *
     * Someone else's list, fetched by the CLI because the page cannot reach
     * EDHREC, and scratch rather than yours: it seeds the graph so you can walk
     * it and Keep what you want, and it is NOT loaded as a deck. That
     * distinction is the whole point of step 10 — `opts.deck` would ring a
     * commander, ink the cards as yours and put all eighty in your library,
     * which is the opposite of harvesting a few out of somebody else's brew.
     *
     * Absent file = absent feature, said out loud. `data/reference/` is
     * gitignored and local-only, so a deployed page will never have one, and
     * "nothing happened" is the wrong way to learn that. */
    const ref = params && params.get('ref');
    if (ref) {
      loadReference(ref);
      return;
    }

    let row = -1;
    if (wanted) row = rowByName(wanted);
    if (row < 0) row = pick(null, seed ? parseInt(seed, 10) : 0);
    show(row);
  }

  function loadReference(slug) {
    const url = (urls.referenceBase || '../data/reference/') + slug + '.json';
    fetch(url)
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then(doc => {
        const out = parseSeedNames((doc.cards || []).join('\n'));
        if (!out.rows.length) {
          MM.setStatus('reference deck ' + slug + ' resolved no cards');
          return;
        }
        const label = (doc.commander || slug) + ' — reference';
        return Promise.resolve(seedFromRows(out.rows, label, {})).then(function () {
          referenceNote = {
            commander: doc.commander || slug,
            cards: out.rows.length,
            missing: out.missing.length,
          };
          render();
          MM.setStatus(label + ': ' + out.rows.length + ' cards — Keep the ones you '
            + 'want and they go to your library.');
        });
      })
      .catch(function (e) {
        MM.setStatus('no reference deck "' + slug + '" — run `manamap pilot '
          + 'commander-search … --open N` to write one (local only). [' + e.message + ']');
      });
  }

  let referenceNote = null;

  /* The graph says which card is selected; the panel follows. Distinct from `show()`,
   * which RESEEDS the graph — clicking a node must open that card without throwing away
   * the walk you just built to reach it. */
  /* Note the card WITHOUT claiming the panel. The graph engine calls this on every branch
   * and pin, and it used to render Discovery's panel unconditionally — which meant Build,
   * which seeds the same engine, had its roles and curve repainted with Discover's landing
   * controls on every reheat. Who draws the panel is `Force.renderPanel`'s decision, and
   * it asks the mode. */
  function setCurrent(row) {
    if (row < 0) return;
    current = row;
    Session.setFocus(row);
  }

  function focus(row) {
    if (row < 0 || row === current) { render(); return; }
    setCurrent(row);
    render();
  }

  function reroll() { show(pick()); }

  function onFilter(key, value) {
    setFilter(key, value);
    show(pick());
  }

  /* Hand off to the graph. The landing card becomes the single seed — the thing
   * `Force.enter` used to refuse. */
  /* Hand off to the walk chrome. The graph is already seeded and on screen, so this is
   * a change of panel plus one branch — not a mode switch that rebuilds anything. */

  // ── the library ────────────────────────────────────────────────────────

  /* The library is Session's; these are the mode's wrappers around it. The graph
   * is where you are LOOKING, the library is what you are KEEPING, and it is the
   * thing `brief()` exports. `toggleLibrary` and `clearLibrary` are NOT
   * delegates — they add the repaint, because the panel belongs to the mode and
   * not to Session.
   *
   * It moved to Session, and it is called a library rather than a tray because in
   * Magic your library IS your deck: the cards you gather while brewing are the
   * deck you are gathering. Its own comment here used to say it was "not another
   * one of those" four set-of-cards ideas — true of its purpose and not of its
   * storage, since it was a fifth array all the same. */
  function inLibrary(row) { return Session.library.has(row); }

  /* SUBSCRIBED, NOT CALLED. Every panel showing a library count repaints itself
   * when Session says the library moved, so a writer does not have to know which
   * panels exist. `MM.keep` wrote straight to Session and repainted nothing in
   * this mode, which is exactly the count that went stale. */
  let subscribed = false;

  function watchLibrary() {
    if (subscribed || !window.Session || !Session.on) return;
    subscribed = true;
    Session.on(function (what) {
      if (what !== 'library') return;
      if (MM.mode === 'discover' && isReady()) render();
      if (window.Force && Force.isActive()) Force.renderPanel();
    });
  }

  function toggleLibrary(row) { Session.library.toggle(row); }

  function clearLibrary() { Session.library.clear(); }

  /* Straight from Session: `list` holds only the names this corpus could place,
   * and a name it could not is still a card the pilot kept. */
  function libraryNames() { return Session.library.names; }

  /* The hand-off to the pilot loop. There is no backend and this plan does not add one:
   * the manuals are 6-10 serially dependent LLM subagents costing ~330k-1.7M tokens, and
   * a static page on Pages cannot run Python. So the library produces a BRIEF — the thing a
   * human pastes into Claude Code, where that loop already works. */
  /* THE BRIEF IS THE SCHEMA `build-deck` ALREADY READS, not a description of it.
   *
   * `pilot/build_deck.py:load_brief` wants `{slug, commander, bracket}` with optional
   * `must_include` / `must_exclude`. What this used to emit — `{generated_by, card_count,
   * cards, commander_candidates, next_step}` — was none of that, so every export had to be
   * hand-translated before the loop could run, and the browser's own answers (which card
   * IS the commander, how many copies, what you brought versus what you found) were thrown
   * away and re-derived.
   *
   * Two rules from the Python side, honoured here rather than guessed at:
   *   - Colour identity is DERIVED from the commander, never authored. It rides in the
   *     provenance block as information, and the builder ignores it.
   *   - Budget is unsupported: prices are stripped from the card data. Saying so beats
   *     approximating it.
   *
   * `must_include` is the LIBRARY — cards you deliberately kept — not the whole pool. The
   * pool is context for the analyst; the library is a claim that these belong in the 99.
   */
  const BRACKET_DEFAULT = 3;      // mirrors config.py:BRACKET_DEFAULT
  const COMMANDER_SLOTS = 99;

  function slugify(name) {
    return String(name || 'untitled').toLowerCase()
      .replace(/\s*\/\/.*$/, '')            // DFCs: the front face names the deck
      .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'untitled';
  }

  function brief() {
    const cmdRow = Session.commander;
    const cmd = cmdRow >= 0 && index[cmdRow] ? index[cmdRow].n : null;

    // What you brought vs what you found. The graph knows; the agent should not have to
    // guess from a flat name list which cards were your idea.
    const onGraph = window.Force ? Force.rows() : [];
    const pool = onGraph.map(function (r) {
      const rec = index[r];
      return rec ? { name: rec.n, source: Session.library.has(r) ? 'kept' : 'found' } : null;
    }).filter(Boolean);

    const must = Session.library.list.filter(function (r) { return r !== cmdRow; })
                                  .map(function (r) { return index[r].n; });

    const ci = [];
    if (cmdRow >= 0) {
      const full = MM.cardRecord(cmdRow);
      if (full && full.ci) String(full.ci).split(',').forEach(function (c) {
        c = c.trim(); if (c) ci.push(c);
      });
    }

    const doc = {
      // ── what build_deck.py reads ──
      slug: slugify(cmd || 'untitled'),
      commander: cmd,
      bracket: BRACKET_DEFAULT,
      must_include: must.slice(0, COMMANDER_SLOTS),
      must_exclude: [],

      // ── what the agents can use, and the builder ignores ──
      _manamap: {
        generated_by: 'manamap Build',
        commander_row: cmdRow,
        colour_identity: ci,          // DERIVED, informational — build_deck derives its own
        budget: 'unsupported — prices are stripped from the card data',
        kept: must.length,
        pool_size: pool.length,
        pool: pool,
      },
    };

    if (!cmd) {
      doc._manamap.blocked = 'No commander set. `build-deck` requires one — open a '
        + 'legendary creature and choose "Set as commander".';
      // Keep the old heuristic as a suggestion, clearly labelled as a guess.
      doc._manamap.commander_candidates = Session.library.list
        .filter(function (r) { return index[r] && index[r].s === 'Creature'; })
        .slice(0, 8).map(function (r) { return index[r].n; });
    }
    if (must.length > COMMANDER_SLOTS) {
      doc._manamap.truncated_must_include = must.length;
    }
    // Loading a deck puts all 99 in the library, so a rebuild can arrive with `must_include`
    // pinning almost every slot and the builder left with nothing to decide. Say so rather
    // than letting the loop discover it: the fix is to Clear the library and keep only what
    // you actually insist on.
    if (must.length >= COMMANDER_SLOTS - 10) {
      doc._manamap.note = must.length + ' of ' + COMMANDER_SLOTS + ' slots are pinned by '
        + 'must_include, leaving ' + Math.max(0, COMMANDER_SLOTS - must.length)
        + ' for the builder. Clear the library and keep only what you insist on if you want '
        + 'it to actually build.';
    }

    doc.next_step = cmd
      ? 'Save as data/decks/' + doc.slug + '/brief.json, then run /build-deck in Claude '
        + 'Code. Check `bracket` first — the browser cannot know your target. The pilot '
        + 'subsystem is 6-10 serial subagent spawns and cannot run in a browser.'
      : 'Set a commander, then export again — build-deck cannot start without one.';
    return doc;
  }

  function exportBrief() {
    const doc = brief();
    const text = JSON.stringify(doc, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (doc.slug || 'manamap') + '-brief.json';
    a.click();
    URL.revokeObjectURL(a.href);
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
    MM.setStatus(doc.commander
      ? doc.must_include.length + ' cards for ' + doc.commander + ' — save as data/decks/'
        + doc.slug + '/brief.json and run /build-deck'
      : 'Exported, but no commander is set — build-deck needs one.');
    return doc;
  }

  // ── import ─────────────────────────────────────────────────────────────

  /* Paste a Moxfield export, get your deck as a graph. Resolution is against
   * `viz_index.json`, NOT `data/decks/index.json` — Build's deck picker refuses any
   * slug it does not already know, and an imported deck has no slug and never will. */
  /* Seed the graph from a set of rows, or GROW it from them.
   *
   * The three-step dance below — newWalk, enter, then pin/current/render — is
   * repeated at four call sites (the landing card, a pasted list, a loaded
   * deck, and a browse selection). Written a fifth time by hand for named-card
   * seeding it would have got the ONE rule that matters wrong, because two of
   * those four already did:
   *
   *   GROWING MUST NEVER BE ABLE TO DELETE.
   *
   * `Force.enter([row])` REBUILDS — it replaces the graph with that one card —
   * while `Force.enter(null)` restores what is there. So `replace: false` never
   * calls enter with a seed at all; it adopts each row into the graph that
   * exists, which is what `MM.relate` had to be taught after merely looking at
   * the atlas and coming back wiped a walk. */
  function seedFromRows(rows, label, opts) {
    const o = opts || {};
    if (!rows || !rows.length || !window.Force) return Promise.resolve(null);

    // GROW. Adopt places each card at the graph's centre of mass and links it to
    // what it already belongs beside — nothing existing is touched.
    if (o.replace === false && Force.nodeCount) {
      for (const r of rows) if (!Force.hasRow(r)) Force.adopt(r);
      setCurrent(rows[0]);
      render();
      return Promise.resolve({ rows: rows.length, grew: true });
    }

    // Commander first so it becomes the pinned node, and — for a named-card
    // walk — so the FIRST card the pilot typed is the one the walk opens on.
    const cmdr = typeof o.commander === 'number' ? o.commander : -1;
    const seeds = cmdr >= 0 ? [cmdr].concat(rows.filter(r => r !== cmdr)) : rows;
    if (cmdr >= 0) Session.setCommander(cmdr);
    if (o.library) for (const r of rows) Session.library.add(r);

    Force.newWalk(true);
    return Promise.resolve(
      Force.enter(seeds, label, { chrome: 'discovery', deck: o.deck || undefined })
    ).then(function () {
      if (cmdr >= 0) Force.pinCard(cmdr);
      setCurrent(seeds[0]);
      render();
      return { rows: rows.length, grew: false };
    });
  }

  /* ── START A WALK FROM CARDS YOU NAME ─────────────────────────────────
   *
   * "Zur the Enchanter", or "1) zur, 2) sol ring" — the pilot's own two
   * examples, and they need different splitting rules.
   *
   * A COMMA CANNOT BE A SEPARATOR. 3,222 of 34,890 card names contain one
   * (9.2%), and they are overwhelmingly the legendary creatures somebody would
   * actually seed a walk with: Zur, the Enchanter. Miirym, Sentinel Wyrm.
   * Splitting on commas turns the commonest input into two cards that do not
   * exist, and it fails SILENTLY — "Zur" resolves to nothing and the graph
   * comes up short with no explanation.
   *
   * So the ENUMERATION is the separator, not the comma. A `1)` / `2.` / `3:`
   * marker splits an item only where an item can begin — at the start of the
   * text, after a newline, or after a comma or semicolon. That last clause is
   * the whole trick: it is what makes "1) zur, 2) sol ring" two cards while
   * leaving "Zur, the Enchanter" one.
   *
   * Requiring a boundary also protects the 8 corpus names that carry a marker
   * INSIDE them — "Vault 87: Forced Evolution" and its five Fallout siblings.
   * The `87:` sits after a space, which is not an item boundary, so the name
   * survives; a naive /\d+[).:]/ would have cut it in half.
   *
   * Then hand the normalised, one-per-line text to `Decklist.parse` — never a
   * second name reader. That parser is fixture-locked in parity with
   * `pilot/fetch_deck.py`, and it already handles the quantity prefix, `*CMDR*`
   * and printing suffixes. Normalising IN FRONT of it costs nothing and forking
   * it would put a third decklist parser in a repo that has twice paid for
   * having two. */
  const ITEM_MARKER = /(^|[\n,;])\s*\d+\s*[).:]\s*/g;

  function parseSeedNames(text) {
    const normalised = String(text || '')
      .replace(ITEM_MARKER, '$1\n')   // keep the boundary char's own line break
      .split(/[\n;]/)
      .map(function (line) {
        // A trailing comma or the pilot's own trailing "…" is punctuation from
        // the list, not part of the name.
        return line.trim().replace(/[,;]+$/, '').replace(/\.{2,}$|…$/, '').trim();
      })
      .filter(Boolean)
      .join('\n');
    if (!normalised) return { rows: [], missing: [], entries: [] };

    const entries = window.Decklist ? Decklist.parse(normalised)
                                    : normalised.split('\n').map(n => ({ name: n }));
    const rows = [];
    const missing = [];
    for (const e of entries) {
      const row = rowByName(e.name);
      // Order is the pilot's, and it is meaningful: the FIRST card named is the
      // one the walk opens on. Dedupe without reordering.
      if (row < 0) missing.push(e.name);
      else if (rows.indexOf(row) === -1) rows.push(row);
    }
    return { rows: rows, missing: missing, entries: entries };
  }

  function importText(text) {
    if (!index) return { resolved: 0, missing: [], total: 0 };
    const entries = Decklist.parse(text);
    const rows = [];
    const missing = [];
    let commanderRow = -1;
    for (const e of entries) {
      const row = rowByName(e.name);
      if (row < 0) { missing.push(e.name); continue; }
      if (rows.indexOf(row) === -1) rows.push(row);
      if (e.is_commander && commanderRow < 0) commanderRow = row;
    }
    if (!rows.length) {
      MM.setStatus('Nothing in that list resolved to a card.');
      return { resolved: 0, missing: missing, total: entries.length };
    }

    for (const r of rows) Session.library.add(r);

    if (window.Force) {
      // Pass `opts.deck`, exactly as `loadDeck` does. Without it a PASTED list — which is
      // how a bulk pool arrives — got a pinned commander and none of the visual language:
      // no gold ring, no deck ink, no warm deck edges, every card washed out as if you had
      // wandered into it. The cards you brought must look different from the cards you find.
      // ONE ANSWER to "who is the commander". The graph learned it from `opts.deck` and
      // the brief reads Session, so writing only the graph meant an imported deck named
      // its commander and the exported brief came out with `commander: null` — which
      // `build-deck` refuses.
      seedFromRows(rows, 'Imported pool', {
        commander: commanderRow,
        deck: { rows: new Set(rows), commander: commanderRow },
      });
    }
    current = commanderRow >= 0 ? commanderRow : rows[0];
    MM.setStatus(rows.length + ' of ' + entries.length + ' cards placed'
      + (missing.length ? ' — ' + missing.length + ' unresolved' : ''));
    return { resolved: rows.length, missing: missing, total: entries.length,
             commander: commanderRow };
  }

  // ── the checked-in decks ───────────────────────────────────────────────

  /* The seven published decks, loaded by slug from the tracked manifest. Distinct from a
   * pasted import in one way that matters: these have a KNOWN commander, so it can be
   * ringed and centred rather than guessed from a `*CMDR*` marker. */
  let manifest = null;

  function loadManifest() {
    if (manifest) return Promise.resolve(manifest);
    return fetch(urls.deckIndex + '?v=' + ((window.MM && MM.DATA_VERSION) || 0))
      .then(r => (r.ok ? r.json() : { decks: [] }))
      .then(doc => { manifest = doc.decks || []; return manifest; })
      .catch(() => { manifest = []; return manifest; });
  }

  function loadDeck(slug) {
    const entry = (manifest || []).find(d => d.slug === slug);
    return fetch(urls.deckBase + slug + '/cards.json?v=' + ((window.MM && MM.DATA_VERSION) || 0))
      .then(r => { if (!r.ok) throw new Error(slug + ' ' + r.status); return r.json(); })
      .then(doc => {
        const rows = [];
        const missing = [];
        let cmdr = -1;
        for (const card of doc.cards) {
          const row = rowByName(card.name);
          if (row < 0) { missing.push(card.name); continue; }
          if (rows.indexOf(row) === -1) rows.push(row);
          if (card.is_commander && cmdr < 0) cmdr = row;
        }
        if (cmdr < 0 && entry && entry.commander) cmdr = rowByName(entry.commander);
        if (!rows.length) { MM.setStatus('Could not resolve any of ' + slug); return null; }

        // Session is the one answer to "who is the commander" — the ring, the colour
        // identity and the exported brief all read it from there. `seedFromRows`
        // writes it, along with the library, the pin and the reseed.
        return seedFromRows(rows, (entry && entry.deck_name) || slug, {
          commander: cmdr,
          library: true,
          deck: { rows: new Set(rows), commander: cmdr },
        }).then(() => {
          MM.setStatus(((entry && entry.deck_name) || slug) + ' — ' + rows.length +
            ' cards' + (cmdr >= 0 ? ', commander ringed' : '') +
            '. Click any card to explore outward.');
          return { rows: rows.length, commander: cmdr, missing: missing };
        });
      })
      .catch(err => { MM.setStatus('Could not load ' + slug + ': ' + err.message); return null; });
  }

  function onDeckPick(slug) {
    if (!slug) return;
    loadManifest().then(() => loadDeck(slug));
  }

  function onImport() {
    const box = document.getElementById('dcImport');
    if (!box) return;
    const text = box.value.trim();
    if (!text) { MM.setStatus('Paste a decklist first.'); return; }
    importText(text);
  }

  function toggleImport() {
    const box = document.getElementById('dcImportWrap');
    if (box) box.style.display = box.style.display === 'none' ? '' : 'none';
  }

  function toggleSeed() {
    const box = document.getElementById('dcSeedWrap');
    if (!box) return;
    const open = box.style.display === 'none';
    box.style.display = open ? '' : 'none';
    if (open) { const t = document.getElementById('dcSeed'); if (t) t.focus(); }
  }

  /* Resolve what the pilot typed and either start a walk from it or grow the
   * one that is there.
   *
   * UNRESOLVED NAMES ARE REPORTED, never silently dropped — the same contract
   * `importText` keeps with its `missing[]`. A typo that quietly produces a
   * one-card walk instead of two is indistinguishable from the feature not
   * working, and the pilot has no way to tell which happened. */
  function onSeedCards(replace) {
    const box = document.getElementById('dcSeed');
    const text = box ? box.value : '';
    const out = parseSeedNames(text);

    if (!out.rows.length) {
      MM.setStatus(out.missing.length
        ? 'No card matched: ' + out.missing.slice(0, 3).join(', ') +
          (out.missing.length > 3 ? ' (+' + (out.missing.length - 3) + ')' : '')
        : 'Type a card name to start from.');
      return;
    }

    const label = out.rows.length === 1
      ? index[out.rows[0]].n
      : index[out.rows[0]].n + ' +' + (out.rows.length - 1);

    Promise.resolve(seedFromRows(out.rows, label, { replace: replace })).then(function () {
      const note = out.missing.length
        ? ' — no match for ' + out.missing.slice(0, 3).join(', ') +
          (out.missing.length > 3 ? ' (+' + (out.missing.length - 3) + ')' : '')
        : '';
      MM.setStatus(out.rows.length + (replace ? ' card' : ' card added')
        + (out.rows.length === 1 ? '' : 's') + note
        + ' — click a relation to grow the graph.');
    });
  }

  function enter() {
    // Called before the index exists on a cold boot — render the chrome now and let the
    // boot promise land the card. Calling land() here would pick from a null index.
    if (!isReady()) { render(); MM.setStatus('Finding you a card…'); return; }
    if (current < 0) { land(new URLSearchParams(location.search)); return; }
    render();
    if (!window.Force) return;
    // RESTORE, do not reseed. `Force.enter` only picks up where you left off when it is
    // handed no seed (`force.js:555`); passing an explicit `[current]` takes the rebuild
    // path and replaces the whole graph with that one card. So a round trip to the atlas
    // and back silently cost you the entire walk — which is the same "growing must never
    // delete" bug as `relate`'s, one level further out, and the reason the fix there alone
    // did not hold.
    if (Force.nodeCount) Force.enter(null, null, { chrome: 'discovery' });
    else Force.enter([current], index[current].n, { chrome: 'discovery' });
  }
  function exitMode() {
    const p = panel();
    if (p) p.classList.remove('open');
    // Clear the content, not just the class. A closed panel is 1px wide rather than
    // absent, so stale relation buttons stayed in the DOM and a querySelector could
    // pick the invisible copy over the live one.
    const el = inner();
    if (el && !(window.Force && Force.isActive())) el.innerHTML = '';
  }

  return {
    configure, ready, isReady, loadIndex, loadNeighbours, decode,
    enter, exit: exitMode, land, show, focus, setCurrent, reroll, onFilter, render, newGraph,
    loadManifest, loadDeck, onDeckPick,
    get decks() { return manifest || []; },
    library: { get list() { return Session.library.list; }, has: inLibrary, toggle: toggleLibrary,
            clear: clearLibrary, names: libraryNames },
    brief, exportBrief, importText, onImport, toggleImport, rowByName,
    seedFromRows, parseSeedNames, onSeedCards, toggleSeed,
    get current() { return current; },
    record, neighbours, counts,
    pick, rowByName, poolSize, setFilter, getFilters,
    COLORS, CMC_BANDS,
    get index() { return index; },
    get table() { return table; },
  };
})();
