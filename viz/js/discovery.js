/* Discovery — the front door.
 *
 * The map used to be where you arrived: 34,322 points and a request that you already
 * know where to look. This lands you on ONE card instead, and the graph grows from
 * whatever you click.
 *
 * Two artifacts make that cheap enough to feel instant:
 *
 *   viz_index.json   0.56 MB gz   name/type/colour/rarity/cmc/roles per card
 *   neighbours.bin   1.70 MB gz   12 similar + 10 synergy + 5 obsoleted-by row ids
 *
 * 2.26 MB against the 18.4 MB it used to take to reach a first branch (12.9 MB
 * projection, then 16.8 MB of incompressible float32 embeddings on the first click).
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
               deckIndex: '../data/decks/index.json', deckBase: '../data/decks/' };

  function configure(u) { urls = Object.assign({}, urls, u || {}); }

  // ── loading ────────────────────────────────────────────────────────────

  function loadIndex() {
    if (indexPromise) return indexPromise;
    indexPromise = fetch(urls.vizIndex)
      .then(r => { if (!r.ok) throw new Error('viz_index ' + r.status); return r.json(); })
      .then(rows => { index = rows; return rows; });
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

    const rec = current >= 0 ? index[current] : null;
    const c = current >= 0 ? counts(current) : { similar: 0, synergy: 0, obsolete: 0 };
    const types = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact',
                   'Land', 'Planeswalker', 'Battle'];

    let html = '<div class="deck-header"><h2>Discover</h2>' +
      '<button class="lens-btn lens-btn-inline" onclick="Discovery.reroll()">Feeling lucky ↻</button>' +
      '</div>';

    html += '<div class="discover-decks">' +
      '<select id="dcDeck" onchange="Discovery.onDeckPick(this.value)">' +
      '<option value="">Load one of my decks…</option>' +
      (manifest || []).map(d => '<option value="' + d.slug + '">' + d.deck_name +
        ' — ' + d.commander + '</option>').join('') +
      '</select></div>';

    html += '<div class="discover-tray">' +
      '<button class="lens-btn" onclick="Discovery.toggleImport()">Paste a decklist</button>' +
      '<span class="discover-traycount">' + tray.length + ' in tray</span>' +
      (tray.length ? '<button class="lens-btn" onclick="Discovery.exportBrief()">Export brief</button>' +
                     '<button class="lens-btn" onclick="Discovery.tray.clear()">Clear</button>' : '') +
      '</div>';
    html += '<div id="dcImportWrap" style="display:none">' +
      '<textarea id="dcImport" class="discover-import" rows="5" ' +
      'placeholder="1 Sol Ring&#10;1 Edgar Markov *CMDR*&#10;&#10;Moxfield exports work as-is."></textarea>' +
      '<button class="lens-btn" onclick="Discovery.onImport()">Load deck as a graph</button>' +
      '<p class="lens-note">Resolved against the card index, not the published decks — ' +
      'any list works, it does not have to be one of the seven.</p></div>';

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

    html += '<div class="lens-title">' + rec.n + '</div>';
    // State what this card HAS before anything is clicked. 23.6% of cards have nothing
    // but similar, and a button that turns out to do nothing reads as broken rather
    // than as a fact about the card.
    html += '<div class="discover-relations">' +
      relBtn('similar', 'Similar', c.similar) +
      relBtn('synergy', 'Synergy', c.synergy) +
      relBtn('obsolete', 'Outclassed by', c.obsolete) +
      '</div>';
    if (c.synergy) {
      html += '<p class="lens-note">Synergy is a rule-based list of ten, not a ranking — ' +
              'the first one is not "the best".</p>';
    }
    html += '<button class="lens-btn discover-keep" onclick="Discovery.tray.toggle(' +
      current + ')">' + (inTray(current) ? '✓ In tray' : '+ Keep this card') + '</button>';
    html += MM.buildCardDetailHtml(MM.cardRecord(current));
    el.innerHTML = html;
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
    if (row >= 0 && window.Force) {
      Force.newWalk(true);
      Force.enter([row], index[row].n, { chrome: 'discovery' });
    }
    render();
    if (row >= 0) {
      MM.setStatus(index[row].n + ' — pick a relation, or hit Feeling lucky for another card.');
    }
  }

  function land(params) {
    const wanted = params && params.get('card');
    const seed = params && params.get('seed');
    let row = -1;
    if (wanted) row = rowByName(wanted);
    if (row < 0) row = pick(null, seed ? parseInt(seed, 10) : 0);
    show(row);
  }

  /* The graph says which card is selected; the panel follows. Distinct from `show()`,
   * which RESEEDS the graph — clicking a node must open that card without throwing away
   * the walk you just built to reach it. */
  function focus(row) {
    if (row < 0 || row === current) { render(); return; }
    current = row;
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
  function walk(relation) {
    if (current < 0) return;
    const row = current;
    document.getElementById('modeSelect').value = 'force';
    MM.setMode('force');
    Force.branchByRow(row, relation || 'similar');
  }

  // ── the tray ───────────────────────────────────────────────────────────

  /* A deliberately light selected-set, separate from the graph. The graph is where you
   * are looking; the tray is what you are keeping. Four other "set of cards" ideas
   * already exist in this codebase (selectedCards, browseSet, deck-builder seeds,
   * Deck Lens) — this is not another one of those, it is the thing you export. */
  const tray = [];

  function inTray(row) { return tray.indexOf(row) !== -1; }

  function toggleTray(row) {
    const at = tray.indexOf(row);
    if (at === -1) tray.push(row); else tray.splice(at, 1);
    render();
    if (window.Force && Force.isActive()) Force.renderPanel();
  }

  function clearTray() { tray.length = 0; render(); }

  function trayNames() { return tray.map(r => index[r].n); }

  /* The hand-off to the pilot loop. There is no backend and this plan does not add one:
   * the manuals are 6-10 serially dependent LLM subagents costing ~330k-1.7M tokens, and
   * a static page on Pages cannot run Python. So the tray produces a BRIEF — the thing a
   * human pastes into Claude Code, where that loop already works. */
  function brief() {
    const cards = trayNames();
    const commanderish = tray.filter(r => (index[r].g || []).length &&
                                          index[r].s === 'Creature');
    return {
      generated_by: 'manamap discovery tray',
      card_count: cards.length,
      cards: cards,
      commander_candidates: commanderish.slice(0, 8).map(r => index[r].n),
      next_step: 'Run the build loop in Claude Code: /build-deck with these cards as the '
               + 'pool. The pilot subsystem is 6-10 serial subagent spawns and cannot run '
               + 'in a browser.',
    };
  }

  function exportBrief() {
    const doc = brief();
    const text = JSON.stringify(doc, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'manamap-brief.json';
    a.click();
    URL.revokeObjectURL(a.href);
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
    MM.setStatus(doc.card_count + ' cards exported — paste the brief into Claude Code.');
    return doc;
  }

  // ── import ─────────────────────────────────────────────────────────────

  /* Paste a Moxfield export, get your deck as a graph. Resolution is against viz_index,
   * NOT data/decks/index.json — Deck Lens refuses any slug it does not already know
   * (deck-map.js), and an imported deck has no slug and never will. */
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

    for (const r of rows) if (!inTray(r)) tray.push(r);

    if (window.Force) {
      Force.newWalk(true);
      // Commander first so it becomes the pinned node — "where does my deck live" starts
      // from the card the deck is built around.
      const seeds = commanderRow >= 0
        ? [commanderRow].concat(rows.filter(r => r !== commanderRow))
        : rows;
      Promise.resolve(Force.enter(seeds, 'Imported deck', { chrome: 'discovery' }))
        .then(function () {
          if (commanderRow >= 0) Force.pinCard(commanderRow);
          render();
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
    return fetch(urls.deckIndex)
      .then(r => (r.ok ? r.json() : { decks: [] }))
      .then(doc => { manifest = doc.decks || []; return manifest; })
      .catch(() => { manifest = []; return manifest; });
  }

  function loadDeck(slug) {
    const entry = (manifest || []).find(d => d.slug === slug);
    return fetch(urls.deckBase + slug + '/cards.json')
      .then(r => { if (!r.ok) throw new Error(slug + ' ' + r.status); return r.json(); })
      .then(doc => {
        const rows = [];
        const missing = [];
        let cmdr = -1;
        for (const card of doc.cards) {
          if (card.is_sideboard) continue;          // the 100, not the maybeboard
          const row = rowByName(card.name);
          if (row < 0) { missing.push(card.name); continue; }
          if (rows.indexOf(row) === -1) rows.push(row);
          if (card.is_commander && cmdr < 0) cmdr = row;
        }
        if (cmdr < 0 && entry && entry.commander) cmdr = rowByName(entry.commander);
        if (!rows.length) { MM.setStatus('Could not resolve any of ' + slug); return null; }

        for (const r of rows) if (!inTray(r)) tray.push(r);

        const seeds = cmdr >= 0 ? [cmdr].concat(rows.filter(r => r !== cmdr)) : rows;
        const deck = { rows: new Set(rows), commander: cmdr };
        Force.newWalk(true);
        return Promise.resolve(
          Force.enter(seeds, (entry && entry.deck_name) || slug,
                      { chrome: 'discovery', deck: deck })
        ).then(() => {
          if (cmdr >= 0) Force.pinCard(cmdr);
          current = cmdr >= 0 ? cmdr : rows[0];
          render();
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

  function enter() {
    // Called before the index exists on a cold boot — render the chrome now and let the
    // boot promise land the card. Calling land() here would pick from a null index.
    if (!isReady()) { render(); MM.setStatus('Finding you a card…'); return; }
    if (current < 0) land(new URLSearchParams(location.search));
    else { render(); if (window.Force) Force.enter([current], index[current].n, { chrome: 'discovery' }); }
  }
  function exitMode() { const p = panel(); if (p) p.classList.remove('open'); }

  return {
    configure, ready, isReady, loadIndex, loadNeighbours, decode,
    enter, exit: exitMode, land, show, focus, reroll, walk, onFilter, render,
    loadManifest, loadDeck, onDeckPick,
    get decks() { return manifest || []; },
    tray: { get list() { return tray.slice(); }, has: inTray, toggle: toggleTray,
            clear: clearTray, names: trayNames },
    brief, exportBrief, importText, onImport, toggleImport, rowByName,
    get current() { return current; },
    record, neighbours, counts,
    pick, rowByName, poolSize, setFilter, getFilters,
    COLORS, CMC_BANDS,
    get index() { return index; },
    get table() { return table; },
  };
})();
