/* build.js — one mode for looking at a set of cards and changing it.
 *
 * This is Deck Lens and Build Deck merged. They were halves of one activity: you load a
 * set of cards, you look at it, you change it, you ship it. Splitting that across two
 * modes meant a published deck could be inspected but not edited, and a deck under
 * construction had no roles, no curve and no verified lines.
 *
 * WHAT CAME FROM DECK LENS (and is the reason this file starts from it): `card_roles.json`
 * in the browser, `FAMILY_PRIORITY`/`FAMILY_COLOR`, the role histogram, the Short List, the
 * verified-lines list, the copies-vs-dots discipline, `dimsAll()`'s scalar-opacity fast
 * path (~100 ms of a 133 ms render), and the `?deck=<slug>` deep link that the dossier and
 * every published manual emit. That last one is an inbound contract in shipped HTML.
 *
 * WHAT CAME FROM THE BUILDER: colour identity, format legality, the mana curve and colour
 * distribution, the mana-base generator.
 *
 * WHAT WAS DELETED WITH IT, deliberately:
 *   - The six-factor recommender. Measured against `config.DECK_BUILD_WEIGHTS`, two of its
 *     six factors had no counterpart on the Python side (JS had keyword-Jaccard, Python
 *     has castability), three of the four shared weights differed, and two shared factors
 *     used different curves. Python's `synergy_affinity` docstring says outright that the
 *     synergy graph "is a retrieval aid and not a scoring function — you cannot ask it
 *     'how well does X fit deck D'", which is exactly what the JS did. It cost ~50 MB of
 *     lazy downloads (embeddings + synergy + combo) to be wrong differently from the
 *     pipeline. Suggestions now come from the same precomputed relations Discover uses,
 *     and real evaluation comes from the sub-agent routine via the exported brief.
 *   - `localStorage['manamap-deck']`. It stored raw positional row indices with no schema
 *     version, so a Scryfall refresh that reorders cards.csv silently reinterpreted a
 *     saved deck as a different set of cards. Restore did no range check either, so
 *     entering Build before the 2.9 MB projection landed threw.
 *
 * Contract with mana-map.js:
 *   getOverlayTraces()  -> layers drawn above the base scatter
 *   getDimmedIndices()  -> Set of rows to draw dim, or null (see dimsAll)
 *   enter() / exit()    -> called by MM.setMode
 *
 * Every index here is a row index into MM.allData, which is projection_2d.json, which is
 * cards.csv row order. Deck cards are matched by name and all seven decks match exactly —
 * see tests/test_viz_deck_lens.py, which fails if a deck ever names a card the map lacks.
 */
(function () {
  'use strict';

  const DECK_BASE = '../data/decks/';
  const MANIFEST_URL = DECK_BASE + 'index.json';
  const ROLES_URL = '../data/card_roles.json';

  // A stack scenario names its cards in prose. Match deck card names against that prose
  // rather than trusting a schema field, because the scenario block has no card list —
  // board entries carry annotations ("Vampire Nighthawk (lifelink, 2/3)") and filler
  // ("lands"). Guard rails: skip basics, skip short names, cap the pairs we draw.
  const MIN_EDGE_NAME_LEN = 6;
  const MAX_EDGE_CARDS = 4;
  const BASIC_LANDS = new Set(['Plains', 'Island', 'Swamp', 'Mountain', 'Forest', 'Wastes']);

  // One card carries several roles; the lens paints it with the most decision-relevant
  // one. Order matters: threat:body sits on 19,032 of 34,322 cards, so it must lose every
  // tie or the map turns into one colour.
  const FAMILY_PRIORITY = [
    'wincon', 'doubler', 'tutor', 'counterspell', 'stax', 'hate', 'removal', 'ramp',
    'draw', 'recursion', 'protection', 'sac-outlet', 'payoff', 'land', 'value', 'buff',
    'utility', 'sac-cost', 'threat',
  ];
  const FAMILY_COLOR = {
    wincon: '#FFD700', doubler: '#22D3EE', tutor: '#E879F9', counterspell: '#38BDF8',
    stax: '#94A3B8', hate: '#CBD5E1', removal: '#EF4444', ramp: '#22C55E',
    draw: '#3B82F6', recursion: '#A78BFA', protection: '#FDE68A', 'sac-outlet': '#FB923C',
    payoff: '#EC4899', land: '#A16207', value: '#14B8A6', buff: '#84CC16',
    utility: '#64748B', 'sac-cost': '#78716C', threat: '#F59E0B', unclassified: '#6B7280',
  };

  // ── State ──
  let manifest = null;       // index.json {decks:[...]}
  let rolesByName = null;    // card_roles.json .roles
  let nameToIdx = null;      // built once from MM.allData
  let active = null;         // the loaded deck, see buildActive()
  let loading = false;

  let showEdges = true;
  let showCandidates = true;
  let showSideboard = false;
  let dimOthers = true;
  // Off-limits highlighting is opt-in: it costs the per-point opacity array.
  let showIllegal = false;
  let format = 'commander';

  // ── Helpers ──

  function esc(s) { return MM.escHtml(String(s == null ? '' : s)); }

  function buildNameIndex() {
    if (nameToIdx) return;
    nameToIdx = new Map();
    const all = MM.allData;
    // First writer wins: cards.csv is oracle-deduplicated, so a repeat would be a bug,
    // but preferring the earlier row keeps this deterministic if one ever appears.
    for (let i = 0; i < all.length; i++) {
      if (!nameToIdx.has(all[i].n)) nameToIdx.set(all[i].n, i);
    }
  }

  // ROLE_PATTERNS does not classify every card, and three of Edgar's lands (Urborg,
  // Takenuma, Reflecting Pool) are the honest example: they do a job no role regex
  // names. Fall back to the map's own supertype for lands only — a Land is a land
  // whatever else is true of it. Everything else stays unclassified rather than being
  // assigned a role the roles file never claimed.
  function primaryFamily(name, idx) {
    const roles = (rolesByName && rolesByName[name]) || [];
    if (roles.length) {
      const families = new Set(roles.map(r => r.split(':')[0]));
      for (const f of FAMILY_PRIORITY) if (families.has(f)) return f;
    }
    if (idx !== null && idx !== undefined && MM.allData[idx] && MM.allData[idx].s === 'Land') return 'land';
    return 'unclassified';
  }

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + ' -> ' + res.status);
    return res.json();
  }

  // A card entry is in the maindeck unless it says otherwise. Copies matter for counts
  // (see CLAUDE.md: count copies, not decklist entries) but a point on the map is a
  // position, so eleven Islands are one dot and the quantity rides along in the panel.
  // ── Colour identity and legality, from the builder ──────────────────────
  //
  // Cheap, correct, and the only part of the old builder that was not competing with the
  // Python. `d.ci` and `d.f` are packed by export/reduce.py; nothing here needs a fetch.

  function parseColorIdentity(ciStr) {
    if (!ciStr) return new Set();
    return new Set(String(ciStr).split(',').map(function (x) { return x.trim(); }).filter(Boolean));
  }

  /* A Commander deck's colour identity is its COMMANDER's, not the union of what is in
   * it — that is the rule, and it is why an off-colour card is a violation rather than a
   * widening. With no commander set there is no restriction at all. */
  function deckColorIdentity() {
    const ci = new Set();
    if (!active) return ci;
    const cmdIdx = active.commanderName && nameToIdx ? nameToIdx.get(active.commanderName) : null;
    if (cmdIdx != null && MM.allData[cmdIdx]) {
      parseColorIdentity(MM.allData[cmdIdx].ci).forEach(function (c) { ci.add(c); });
    }
    return ci;
  }

  function isColorIdentitySubset(cardCI, deckCI) {
    if (!deckCI.size) return true;
    for (const c of parseColorIdentity(cardCI)) if (!deckCI.has(c)) return false;
    return true;
  }

  function isLegalInFormat(d, format) {
    if (!d || !d.f) return false;
    return d.f.split(',').includes(format);
  }

  // ── Curve and colour, from the builder ──────────────────────────────────

  function renderManaCurve(indices) {
    const buckets = [0, 0, 0, 0, 0, 0, 0];
    for (const idx of indices) {
      const d = MM.allData[idx];
      if (!d || d.s === 'Land') continue;
      buckets[Math.min(Math.floor(d.m || 0), 6)]++;
    }
    const max = Math.max.apply(null, buckets.concat([1]));
    let html = '<div class="deck-section"><div class="deck-section-title">Mana curve</div>' +
               '<div class="mana-curve">';
    for (let i = 0; i < buckets.length; i++) {
      const pct = (buckets[i] / max) * 100;
      html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">' +
        '<div style="height:40px;width:100%;display:flex;align-items:flex-end;">' +
        '<div style="width:100%;background:#c4a747;border-radius:2px 2px 0 0;height:' + pct +
        '%;min-height:' + (buckets[i] > 0 ? '2px' : '0') + ';"></div></div>' +
        '<div class="curve-label">' + (i === 6 ? '6+' : i) + '</div></div>';
    }
    return html + '</div></div>';
  }

  function renderColorDist(indices) {
    const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 };
    let total = 0;
    for (const idx of indices) {
      const d = MM.allData[idx];
      if (!d || d.s === 'Land') continue;
      for (const tok of String(d.mc || '').match(/\{([^}]+)\}/g) || []) {
        const inner = tok.slice(1, -1);
        if ('WUBRG'.includes(inner)) { pips[inner]++; total++; }
      }
    }
    if (!total) return '';
    let html = '<div class="deck-section"><div class="deck-section-title">Colour load</div>' +
               '<div class="color-dist">';
    for (const c of 'WUBRG') {
      if (!pips[c]) continue;
      html += '<div class="color-dist-item"><span class="mana-sym mana-' + c +
        '" style="width:14px;height:14px;font-size:0.5rem;">' + c + '</span>' +
        Math.round(pips[c] / total * 100) + '%</div>';
    }
    return html + '</div></div>';
  }

  function isMain(card) { return !card.is_sideboard; }
  function qty(card) { return parseInt(card.quantity, 10) || 1; }

  // ── Loading ──

  async function ensureShared() {
    buildNameIndex();
    if (!manifest) manifest = await getJSON(MANIFEST_URL);
    if (!rolesByName) {
      const doc = await getJSON(ROLES_URL);
      rolesByName = doc.roles || {};
    }
  }

  async function loadDeck(slug) {
    const entry = manifest.decks.find(d => d.slug === slug);
    if (!entry) throw new Error('unknown deck: ' + slug);

    const deckDoc = await getJSON(DECK_BASE + slug + '/cards.json');

    // considering.json is optional in principle; every published deck has one today.
    let considering = null;
    try { considering = await getJSON(DECK_BASE + slug + '/considering.json'); } catch (e) { /* absent */ }

    // The manifest lists checker-passed stacks only, which is the whole point of it —
    // a browser can list neither data/decks/ nor stacks/.
    const stacks = [];
    for (const file of (entry.stack_files || [])) {
      try { stacks.push(await getJSON(DECK_BASE + slug + '/stacks/' + file)); } catch (e) { /* skip */ }
    }
    return buildActive(entry, deckDoc, considering, stacks);
  }

  function buildActive(entry, deckDoc, considering, stacks) {
    const cards = deckDoc.cards || [];
    const commanderName = (deckDoc.commander && deckDoc.commander.name) || entry.commander;

    function toSlot(card) {
      const idx = nameToIdx.has(card.name) ? nameToIdx.get(card.name) : null;
      return {
        name: card.name,
        idx,
        qty: qty(card),
        family: primaryFamily(card.name, idx),
        roles: (rolesByName && rolesByName[card.name]) || [],
        isCommander: card.name === commanderName,
      };
    }

    const main = cards.filter(isMain).map(toSlot);
    const side = cards.filter(c => !isMain(c)).map(toSlot);
    const unmapped = main.concat(side).filter(s => s.idx === null).map(s => s.name);

    // The Short List: ten cards the pilot might sleeve. Pool picks are the interesting
    // ones on a map — they are, literally, elsewhere.
    const mainNames = new Set(main.map(s => s.name));
    const candidates = ((considering && considering.ten) || [])
      .map(t => ({
        name: t.card,
        idx: nameToIdx.has(t.card) ? nameToIdx.get(t.card) : null,
        source: t.source || 'pool',
        role: t.role || '',
      }))
      .filter(c => c.idx !== null && !mainNames.has(c.name));

    return {
      slug: entry.slug,
      entry,
      commanderName,
      main,
      side,
      candidates,
      unmapped,
      edges: buildEdges(stacks, main),
      copies: main.reduce((n, s) => n + s.qty, 0),
    };
  }

  // Each verified stack becomes a small clique between the deck cards it names, so a
  // rules-verified line reads as geometry: which corners of the deck actually talk to
  // each other.
  function buildEdges(stacks, main) {
    const byName = new Map();
    for (const s of main) if (s.idx !== null) byName.set(s.name, s);

    // Longest first so "Sanguine Bond" cannot shadow a longer name containing it.
    const searchable = Array.from(byName.keys())
      .filter(n => n.length >= MIN_EDGE_NAME_LEN && !BASIC_LANDS.has(n))
      .sort((a, b) => b.length - a.length);

    const edges = [];
    for (const stack of stacks) {
      const hay = (stack.title || '') + ' ' + JSON.stringify(stack.scenario || {});
      const hit = [];
      for (const name of searchable) {
        if (hay.indexOf(name) !== -1) {
          hit.push(name);
          if (hit.length >= MAX_EDGE_CARDS) break;
        }
      }
      // A stack that names fewer than two deck cards draws no line — a scenario can be
      // about one card. Keep it in the list anyway: the panel's count must agree with
      // the manifest's `verified`, or the lens quietly loses a verified line.
      const pairs = [];
      for (let i = 0; i < hit.length; i++) {
        for (let j = i + 1; j < hit.length; j++) pairs.push([byName.get(hit[i]), byName.get(hit[j])]);
      }
      edges.push({ id: stack.id || '', title: stack.title || '', cards: hit, pairs });
    }
    return edges;
  }

  // ── Overlay ──

  function drawnEdges() { return active ? active.edges.filter(e => e.pairs.length) : []; }

  function getOverlayTraces() {
    if (!active) return [];
    const all = MM.allData;
    const traces = [];

    // Edges first so markers draw on top of them.
    if (showEdges && active.edges.length) {
      const ex = [], ey = [];
      for (const edge of active.edges) {
        for (const [a, b] of edge.pairs) {
          ex.push(all[a.idx].x, all[b.idx].x, null);
          ey.push(all[a.idx].y, all[b.idx].y, null);
        }
      }
      if (ex.length) {
        traces.push({
          type: 'scattergl',
          mode: 'lines',
          name: 'Verified lines (' + drawnEdges().length + ')',
          x: ex,
          y: ey,
          hoverinfo: 'none',
          line: { color: '#4CAF50', width: 1.2 },
          opacity: 0.55,
          _isDeckOverlay: true,
        });
      }
    }

    // One trace per role family, so the legend doubles as the deck's role budget.
    const byFamily = new Map();
    for (const slot of active.main) {
      if (slot.idx === null || slot.isCommander) continue;
      if (!byFamily.has(slot.family)) byFamily.set(slot.family, []);
      byFamily.get(slot.family).push(slot);
    }
    const ordered = FAMILY_PRIORITY.concat(['unclassified']).filter(f => byFamily.has(f));
    for (const family of ordered) {
      const slots = byFamily.get(family);
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: family + ' (' + slots.length + ')',
        x: slots.map(s => all[s.idx].x),
        y: slots.map(s => all[s.idx].y),
        customdata: slots.map(s => s.idx),
        hoverinfo: 'none',
        marker: {
          size: 9, opacity: 1, color: FAMILY_COLOR[family] || '#6B7280',
          line: { color: '#0d0d1a', width: 1 },
        },
        _isDeckOverlay: true,
      });
    }

    if (showSideboard) {
      const bench = active.side.filter(s => s.idx !== null);
      if (bench.length) {
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: 'Sideboard (' + bench.length + ')',
          x: bench.map(s => all[s.idx].x),
          y: bench.map(s => all[s.idx].y),
          customdata: bench.map(s => s.idx),
          hoverinfo: 'none',
          marker: {
            size: 7, opacity: 0.9, color: 'rgba(0,0,0,0)',
            line: { color: '#D4AF4A', width: 1.5 },
          },
          _isDeckOverlay: true,
        });
      }
    }

    if (showCandidates && active.candidates.length) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Short List (' + active.candidates.length + ')',
        x: active.candidates.map(c => all[c.idx].x),
        y: active.candidates.map(c => all[c.idx].y),
        customdata: active.candidates.map(c => c.idx),
        hoverinfo: 'none',
        marker: {
          size: 11, opacity: 1, color: 'rgba(0,0,0,0)', symbol: 'circle',
          line: { color: '#4A7BFF', width: 2 },
        },
        _isDeckOverlay: true,
      });
    }

    // The commander last, so nothing covers it.
    const cmd = active.main.find(s => s.isCommander && s.idx !== null);
    if (cmd) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Commander',
        x: [all[cmd.idx].x],
        y: [all[cmd.idx].y],
        customdata: [cmd.idx],
        hoverinfo: 'none',
        marker: { size: 18, opacity: 1, color: '#c4a747', symbol: 'star', line: { color: '#fff', width: 1.5 } },
        _isDeckOverlay: true,
      });
    }

    return traces;
  }

  // The Lens dims the whole world and redraws the deck as overlay traces on top, so
  // render() can use one scalar opacity instead of a 34,000-entry per-point array.
  // Returning the index Set at all was the expensive half of Deck Lens mode.
  // `dimsAll` says "one scalar opacity for everything", which lets mana-map.js skip
  // building a 34,322-entry per-point array — measured at ~100 ms of a 133 ms render.
  // `getDimmedIndices` is the per-point path, and it is only worth paying for when a
  // GENUINE SUBSET is dim: showing what you may not legally play.
  function dimsAll() { return !!(active && dimOthers && !showIllegal); }

  function getDimmedIndices() {
    if (!active || !showIllegal) return null;
    // Everything you could not legally put in this deck: wrong format, or outside the
    // commander's colour identity. The builder computed this; the Lens never did, so a
    // published deck could not show you what was off-limits.
    const ci = deckColorIdentity();
    const out = new Set();
    for (let i = 0; i < MM.allData.length; i++) {
      const d = MM.allData[i];
      if (!isLegalInFormat(d, format) || !isColorIdentitySubset(d.ci, ci)) out.add(i);
    }
    return out;
  }

  // ── Panel ──

  function panelEl() { return document.getElementById('deckInner'); }

  function familyCounts() {
    const counts = new Map();
    for (const slot of active.main) {
      counts.set(slot.family, (counts.get(slot.family) || 0) + slot.qty);
    }
    return FAMILY_PRIORITY.concat(['unclassified'])
      .filter(f => counts.has(f))
      .map(f => ({ family: f, n: counts.get(f) }))
      .sort((a, b) => b.n - a.n);
  }

  function renderPanel() {
    const el = panelEl();
    if (!el) return;

    const picker = manifest ? manifest.decks.map(d =>
      '<option value="' + esc(d.slug) + '"' + (active && active.slug === d.slug ? ' selected' : '') + '>' +
      'Vol. ' + String(d.volume).padStart(3, '0') + ' — ' + esc(d.deck_name) + '</option>').join('') : '';

    let html =
      '<div class="deck-header">' +
        '<h2>Deck Lens</h2>' +
        '<button class="detail-close" onclick="Build.close()" title="Close">×</button>' +
      '</div>' +
      '<div class="deck-section">' +
        '<div class="deck-format-row">' +
          '<label for="deckLensSelect">Deck</label>' +
          '<select id="deckLensSelect" onchange="Build.select(this.value)">' +
            '<option value="">Choose a deck…</option>' + picker +
          '</select>' +
        '</div>' +
      '</div>';

    if (loading) {
      html += '<div class="deck-section"><div class="deck-empty">Loading artifacts…</div></div>';
      el.innerHTML = html;
      return;
    }

    if (!active) {
      html +=
        '<div class="deck-section">' +
          '<div class="deck-empty">Pick a published deck to light up its 99 on the map. ' +
          'Everything else dims, so the deck’s footprint in card space becomes visible.</div>' +
        '</div>';
      el.innerHTML = html;
      return;
    }

    const e = active.entry;
    html +=
      '<div class="deck-section">' +
        '<div class="lens-title">' + esc(e.deck_name) + '</div>' +
        '<div class="lens-sub">' + esc(e.commander) + '</div>' +
        '<div class="lens-coverline">“' + esc(e.coverline) + '”</div>' +
        '<div class="lens-links">' +
          '<a href="../manuals/' + esc(active.slug) + '.html">Read the issue →</a>' +
          '<a href="deck.html?deck=' + esc(active.slug) + '">Dossier →</a>' +
        '</div>' +
      '</div>' +

      '<div class="deck-section">' +
        '<div class="lens-stats">' +
          statBox(active.copies, 'cards') +
          statBox(e.verified, 'verified') +
          statBox(active.candidates.length, 'short list') +
          statBox(active.side.length, 'bench') +
        '</div>' +
        '<button class="lens-btn" onclick="Build.zoomToDeck()">Zoom to the deck</button>' +
      '</div>' +

      '<div class="deck-section">' +
        '<div class="deck-section-title">Show</div>' +
        toggleRow('dimOthers', dimOthers, 'Dim the other cards') +
        toggleRow('showEdges', showEdges, 'Verified lines (' + drawnEdges().length + ' drawn)') +
        toggleRow('showCandidates', showCandidates, 'Short List (' + active.candidates.length + ')') +
        toggleRow('showSideboard', showSideboard, 'Sideboard (' + active.side.length + ')') +
        toggleRow('showIllegal', showIllegal, 'Grey out what you cannot play') +
      '</div>';

    // Curve and colour load, from the builder half. These are about the deck as a
    // machine — what it costs and what it demands — where the role budget is about what
    // the cards DO. Both are cheap: `d.m` and `d.mc` are already in the projection row.
    const mainIdx = active.main.map(function (x) { return x.idx; })
                               .filter(function (i) { return i !== null; });
    html += renderManaCurve(mainIdx) + renderColorDist(mainIdx);

    // Role budget, which is also the map legend.
    const counts = familyCounts();
    const max = counts.length ? counts[0].n : 1;
    html +=
      '<div class="deck-section">' +
        '<div class="deck-section-title">Role budget <span>' + active.copies + ' copies</span></div>' +
        // Bars count copies (eleven Islands are eleven sources); the map legend counts
        // dots, and a dot is a distinct card. Saying so beats letting the two numbers
        // disagree in silence — that is the mistake the land audit already made once.
        '<div class="lens-note">Bars count copies · map dots count distinct cards</div>' +
        counts.map(c =>
          '<div class="lens-bar-row">' +
            '<span class="lens-swatch" style="background:' + FAMILY_COLOR[c.family] + '"></span>' +
            '<span class="lens-bar-label">' + esc(c.family) + '</span>' +
            '<span class="lens-bar-track"><span class="lens-bar-fill" style="width:' +
              Math.round((c.n / max) * 100) + '%;background:' + FAMILY_COLOR[c.family] + '"></span></span>' +
            '<span class="lens-bar-n">' + c.n + '</span>' +
          '</div>').join('') +
      '</div>';

    if (active.edges.length) {
      html +=
        '<div class="deck-section">' +
          '<div class="deck-section-title">Verified lines <span>✓ rules-verified</span></div>' +
          active.edges.map((edge, i) =>
            '<div class="lens-line' + (edge.pairs.length ? '' : ' lens-line-nodraw') +
              '" onclick="Build.focusLine(' + i + ')">' +
              '<div class="lens-line-title">' + esc(edge.title) + '</div>' +
              '<div class="lens-line-cards">' +
                (edge.cards.length ? edge.cards.map(esc).join(' · ') : 'no deck card named — no line drawn') +
              '</div>' +
            '</div>').join('') +
        '</div>';
    }

    if (active.candidates.length) {
      html +=
        '<div class="deck-section">' +
          '<div class="deck-section-title">The Short List <span>◆ data-derived</span></div>' +
          active.candidates.map(c =>
            '<div class="lens-cand" onclick="MM.selectByName(' + JSON.stringify(c.name).replace(/"/g, '&quot;') + ')">' +
              '<span class="lens-cand-name">' + esc(c.name) + '</span>' +
              '<span class="lens-chip lens-chip-' + esc(c.source) + '">' + esc(c.source) + '</span>' +
            '</div>').join('') +
        '</div>';
    }

    if (active.unmapped.length) {
      html +=
        '<div class="deck-section">' +
          '<div class="deck-section-title">Not on this map</div>' +
          '<div class="deck-empty">' + active.unmapped.map(esc).join(', ') + '</div>' +
        '</div>';
    }

    el.innerHTML = html;
  }

  function statBox(n, label) {
    return '<div class="lens-stat"><div class="lens-stat-n">' + esc(n) + '</div>' +
           '<div class="lens-stat-l">' + esc(label) + '</div></div>';
  }

  function toggleRow(key, on, label) {
    return '<label class="lens-toggle">' +
      '<input type="checkbox"' + (on ? ' checked' : '') +
      ' onchange="Build.toggle(\'' + key + '\', this.checked)"> ' + esc(label) + '</label>';
  }

  // ── Actions ──

  async function select(slug) {
    if (!slug) { active = null; renderPanel(); MM.render(); return; }
    loading = true;
    renderPanel();
    try {
      active = await loadDeck(slug);
      MM.setStatus(active.entry.deck_name + ' — ' + active.copies + ' cards lit, ' +
                   active.entry.verified + ' verified line(s)');
    } catch (err) {
      active = null;
      MM.setStatus('Could not load deck: ' + err.message);
    }
    loading = false;
    renderPanel();
    MM.render();
  }

  function toggle(key, value) {
    if (key === 'showIllegal') { showIllegal = !!value; renderPanel(); MM.render(); return; }
    if (key === 'dimOthers') dimOthers = value;
    else if (key === 'showEdges') showEdges = value;
    else if (key === 'showCandidates') showCandidates = value;
    else if (key === 'showSideboard') showSideboard = value;
    renderPanel();
    MM.render();
  }

  // Frame the deck. Plotly.relayout fires plotly_relayout, which mana-map.js debounces
  // into a region-label refresh — that is the wanted behaviour here, not a loop.
  function zoomToDeck() {
    if (!active) return;
    const all = MM.allData;
    const pts = active.main.filter(s => s.idx !== null).map(s => all[s.idx]);
    if (!pts.length) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const pad = 2;
    setCamera([Math.min(...xs) - pad, Math.max(...xs) + pad],
              [Math.min(...ys) - pad, Math.max(...ys) + pad]);
  }

  // The canvas wants a zoom transform. This used to branch — Plotly took an axis-range
  // relayout — and the branch went with the renderer.
  function setCamera(xr, yr) {
    if (MM.mapRenderer) MM.mapRenderer.setCamera({ x: xr, y: yr }, { animate: true });
  }

  // The panel is a 0.25s CSS width transition, so the map's box is only final after it
  // ends — hence the 260ms. The canvas also has a ResizeObserver, but that is throttled
  // in a background tab, so the explicit call is what makes this reliable.
  function resizeMap() {
    setTimeout(() => { if (MM.mapRenderer) MM.mapRenderer.resize(); }, 260);
  }

  function focusLine(i) {
    if (!active || !active.edges[i]) return;
    const edge = active.edges[i];
    const all = MM.allData;
    const pts = edge.cards
      .map(n => active.main.find(s => s.name === n))
      .filter(s => s && s.idx !== null)
      .map(s => all[s.idx]);
    if (!pts.length) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const pad = 4;
    setCamera([Math.min(...xs) - pad, Math.max(...xs) + pad],
              [Math.min(...ys) - pad, Math.max(...ys) + pad]);
    MM.setStatus(edge.title);
  }

  async function enter() {
    const panel = document.getElementById('deckPanel');
    panel.classList.add('open');
    resizeMap();
    loading = true;
    renderPanel();
    try {
      await ensureShared();
    } catch (err) {
      MM.setStatus('Deck Lens unavailable: ' + err.message);
    }
    loading = false;
    renderPanel();
    // ?deck=<slug> deep-links from the dossier and the magazine's Back Page.
    const wanted = new URLSearchParams(window.location.search).get('deck');
    if (wanted && manifest && manifest.decks.some(d => d.slug === wanted)) {
      await select(wanted);
    } else {
      MM.render();
    }
  }

  function exit() {
    const panel = document.getElementById('deckPanel');
    panel.classList.remove('open');
    resizeMap();
  }

  function close() {
    document.getElementById('modeSelect').value = 'explore';
    MM.setMode('explore');
  }

  // ── Adding cards to the loaded set ──────────────────────────────────────
  //
  // The builder had `addSeed` writing into `deckState.seeds`, one of eight "set of cards"
  // containers. That container is gone with the scorer it fed. A card you want goes into
  // the TRAY — which is Session's, is shared with Discover and Explore, and is the thing
  // the brief exports to the sub-agent routine.

  function addCard(row) {
    if (typeof row !== 'number' || row < 0) return;
    if (!Session.tray.has(row)) Session.tray.toggle(row);
    if (active) renderPanel();
    MM.setStatus((MM.cardRecord(row) || {}).n + ' added — ' + Session.tray.size +
                 ' in the tray · Export brief hands them to the build loop');
  }

  /* Is this card already accounted for — in the loaded deck, or in the tray? Read by the
   * card panel's "+ Deck" button so it can say so rather than offering a no-op. */
  function isInDeck(row) {
    if (Session.tray.has(row)) return true;
    if (!active || !nameToIdx) return false;
    const d = MM.allData[row];
    if (!d) return false;
    return active.main.some(function (slot) { return slot.name === d.n; });
  }

  window.Build = {
    addCard,
    isInDeck,
    enter,
    exit,
    close,
    select,
    toggle,
    zoomToDeck,
    focusLine,
    getOverlayTraces,
    getDimmedIndices,
    dimsAll,
  };
})();
