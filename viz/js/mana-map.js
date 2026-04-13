/**
 * mana-map.js — Explore mode: map rendering, search, toggles, card viewer panel
 * with multi-card selection, pinch zoom.
 * Exposes shared state and helpers on window.MM for deck-builder.js.
 */
(function () {
  // ── Palettes ──
  const COLOR_PALETTE = { W: '#F0E68C', U: '#4A90D9', B: '#8B5CF6', R: '#DC2626', G: '#22C55E', Colorless: '#9CA3AF', Multicolor: '#D4A017' };
  const SUPERTYPE_PALETTE = { Creature: '#22C55E', Instant: '#4A90D9', Sorcery: '#8B5CF6', Enchantment: '#EC4899', Artifact: '#9CA3AF', Land: '#92400E', Planeswalker: '#F59E0B', Battle: '#DC2626', Unknown: '#555' };
  const RARITY_PALETTE = { common: '#9CA3AF', uncommon: '#C0C0C0', rare: '#C4A747', mythic: '#EA580C', bonus: '#A855F7', special: '#F472B6' };

  const ALL_FORMATS = ['standard', 'modern', 'legacy', 'vintage', 'commander', 'pioneer', 'pauper', 'historic'];
  const SUPERTYPES = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Land', 'Planeswalker', 'Battle', 'Unknown'];

  let allData = [];
  let activeSupertypes = new Set(SUPERTYPES);
  let currentColorBy = 'color';
  let searchTerm = '';
  let searchTimeout = null;
  let plotInitialized = false;
  let similarTrace = null;
  let currentMode = 'explore';
  let embeddings = null; // Float32Array, loaded lazily for Find Similar
  const EMBED_DIM = 128;
  let currentMap = 'default'; // 'default' or 'ability'
  const projectionCache = {}; // { default: [...], ability: [...] }
  const embeddingsCache = {}; // { default: Float32Array, ability: Float32Array }
  const MAP_CONFIGS = {
    default: { projection: '../data/projection_2d.json', embeddings: '../data/embeddings.bin' },
    ability: { projection: '../data/projection_2d_ability.json', embeddings: '../data/embeddings_ability.bin' },
  };

  // ── Multi-select state ──
  const MAX_SELECTED = 8;
  let selectedCards = [];   // Array of { idx, data }, max 8
  let topCardIndex = 0;     // Which card is "on top" in the viewer

  function getSelectedCard() {
    return selectedCards[topCardIndex]?.data ?? null;
  }

  // ── Helpers ──

  function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function buildHoverText(d) {
    let lines = ['<b>' + escHtml(d.n) + '</b>'];
    if (d.t) lines.push(escHtml(d.t));
    let statsLine = '';
    if (d.mc) statsLine += escHtml(d.mc);
    if (d.p != null && d.th != null) {
      statsLine += (statsLine ? '  \u2022  ' : '') + d.p + '/' + d.th;
    } else if (d.l != null) {
      statsLine += (statsLine ? '  \u2022  ' : '') + 'Loyalty: ' + d.l;
    } else if (d.d != null) {
      statsLine += (statsLine ? '  \u2022  ' : '') + 'Defense: ' + d.d;
    }
    if (statsLine) lines.push(statsLine);
    if (d.o) {
      let preview = d.o.length > 120 ? d.o.substring(0, 117) + '...' : d.o;
      lines.push('<i>' + escHtml(preview) + '</i>');
    }
    return lines.join('<br>');
  }

  function buildHoverTextMinimal(d) {
    let line = '<b>' + escHtml(d.n) + '</b>';
    let parts = [];
    if (d.s) parts.push(escHtml(d.s));
    if (d.mc) parts.push(escHtml(d.mc));
    if (parts.length) line += '<br>' + parts.join(' \u00b7 ');
    return line;
  }

  function renderManaSymbols(manaCost) {
    if (!manaCost) return '';
    const tokens = manaCost.match(/\{[^}]+\}/g);
    if (!tokens) return escHtml(manaCost);
    return tokens.map(tok => {
      const inner = tok.slice(1, -1);
      if ('WUBRG'.includes(inner)) {
        return '<span class="mana-sym mana-' + inner + '">' + inner + '</span>';
      }
      if (inner === 'C') {
        return '<span class="mana-sym mana-C">C</span>';
      }
      if (inner.includes('/')) {
        return '<span class="mana-sym mana-num" style="width:auto;padding:0 4px;border-radius:10px;">' + escHtml(inner) + '</span>';
      }
      return '<span class="mana-sym mana-num">' + escHtml(inner) + '</span>';
    }).join('');
  }

  // ── Selection Functions ──

  function addToSelection(idx) {
    // Don't add duplicates — if already selected, bring to top
    const existing = selectedCards.findIndex(c => c.idx === idx);
    if (existing !== -1) {
      topCardIndex = existing;
      updateViewerPanel();
      updateSelectionHighlight();
      return;
    }

    // Enforce max — drop oldest
    if (selectedCards.length >= MAX_SELECTED) {
      selectedCards.shift();
      if (topCardIndex > 0) topCardIndex--;
    }

    selectedCards.push({ idx, data: allData[idx] });
    topCardIndex = selectedCards.length - 1;
    updateViewerPanel();
    updateSelectionHighlight();
  }

  function removeFromSelection(idx) {
    const pos = selectedCards.findIndex(c => c.idx === idx);
    if (pos === -1) return;

    selectedCards.splice(pos, 1);

    if (selectedCards.length === 0) {
      topCardIndex = 0;
      closeViewerPanel();
      return;
    }

    // Adjust topCardIndex
    if (topCardIndex >= selectedCards.length) {
      topCardIndex = selectedCards.length - 1;
    } else if (pos < topCardIndex) {
      topCardIndex--;
    }

    updateViewerPanel();
    updateSelectionHighlight();
  }

  function clearSelection() {
    selectedCards = [];
    topCardIndex = 0;
    closeViewerPanel();
    clearSimilarTrace();
    updateSelectionHighlight();
  }

  function bringToTop(stackIndex) {
    if (stackIndex < 0 || stackIndex >= selectedCards.length) return;
    topCardIndex = stackIndex;
    updateViewerPanel();
    updateSelectionHighlight();
  }

  // ── Viewer Panel ──

  function buildCardDetailHtml(d) {
    let html = '';

    const imgUrl = 'https://api.scryfall.com/cards/named?exact='
      + encodeURIComponent(d.n) + '&format=image&version=normal';
    html += '<div class="detail-card-image">';
    html += '<img src="' + imgUrl + '" alt="' + escHtml(d.n) + '"';
    html += ' loading="lazy" onerror="this.onerror=null;this.parentElement.style.minHeight=\'auto\';this.parentElement.innerHTML=\'<div class=\\\'detail-image-fallback\\\'>Image not available</div>\'">';
    html += '</div>';

    if (d.t) html += '<div class="detail-type">' + escHtml(d.t) + '</div>';

    html += buildObsolescenceHtml(d.n);

    if (d.o) {
      html += '<div class="detail-section">';
      html += '<div class="detail-section-title">Oracle Text</div>';
      html += '<div class="detail-oracle">' + escHtml(d.o).replace(/ \/\/ /g, '<br><br>') + '</div>';
      html += '</div>';
    }

    if (d.k) {
      html += '<div class="detail-section">';
      html += '<div class="detail-section-title">Keywords</div>';
      html += '<div class="detail-keywords">';
      d.k.split(', ').forEach(kw => {
        html += '<span class="keyword-badge">' + escHtml(kw) + '</span>';
      });
      html += '</div></div>';
    }

    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">Details</div>';
    html += '<div class="detail-meta">';
    if (d.ci) html += '<span>Color Identity: ' + escHtml(d.ci) + '</span>';
    html += '<span>CMC: ' + d.m + '</span>';
    if (d.er != null) html += '<span>EDHREC Rank: #' + d.er.toLocaleString() + '</span>';
    html += '</div></div>';

    html += '<div class="detail-section">';
    html += '<div class="detail-section-title">Format Legality</div>';
    html += '<div class="detail-formats">';
    const legalSet = d.f ? new Set(d.f.split(',')) : new Set();
    ALL_FORMATS.forEach(fmt => {
      const isLegal = legalSet.has(fmt);
      html += '<span class="format-badge' + (isLegal ? ' legal' : '') + '">' + fmt + '</span>';
    });
    html += '</div></div>';

    html += '<div class="detail-section">';
    html += '<button class="btn-similar" onclick="MM.findSimilar()">Find Similar Cards</button>';
    html += '<button class="btn-synergy" onclick="MM.findSynergies()">Find Synergies</button>';
    html += '</div>';

    return html;
  }

  function updateViewerPanel() {
    if (selectedCards.length === 0) {
      closeViewerPanel();
      return;
    }

    const panel = document.getElementById('detailPanel');
    const inner = document.getElementById('detailInner');
    const topCard = selectedCards[topCardIndex];
    const d = topCard.data;

    // Header with card name, count, in-deck badge, close
    let html = '<div class="viewer-header">';
    html += '<h2>' + escHtml(d.n) + '</h2>';
    if (selectedCards.length > 1) {
      html += '<span class="viewer-count">' + (topCardIndex + 1) + '/' + selectedCards.length + '</span>';
    }
    // In-deck badge or add-to-deck button
    if (typeof window.DeckBuilder !== 'undefined' && window.DeckBuilder.isInDeck) {
      if (window.DeckBuilder.isInDeck(topCard.idx)) {
        html += '<span class="in-deck-badge">\u2713 In Deck</span>';
      } else {
        html += '<button class="btn-add-deck" onclick="DeckBuilder.addSeed(' + topCard.idx + '); MM.render(); MM.bringToTop(' + topCardIndex + ')">+ Deck</button>';
      }
    }
    html += '<button class="detail-close" onclick="MM.closeDetail()" title="Close (ESC)">\u00d7</button>';
    // Quickstats line: mana + P/T + rarity
    html += '<div class="viewer-quickstats">';
    if (d.mc) html += renderManaSymbols(d.mc);
    if (d.p != null && d.th != null) {
      html += '<span class="stat-divider">\u00b7</span><strong>' + escHtml(d.p) + '/' + escHtml(d.th) + '</strong>';
    } else if (d.l != null) {
      html += '<span class="stat-divider">\u00b7</span><strong>Loyalty: ' + escHtml(String(d.l)) + '</strong>';
    } else if (d.d != null) {
      html += '<span class="stat-divider">\u00b7</span><strong>Defense: ' + escHtml(String(d.d)) + '</strong>';
    }
    if (d.r) {
      const rarityClass = (d.r === 'mythic' || d.r === 'rare' || d.r === 'uncommon' || d.r === 'common') ? d.r : '';
      html += '<span class="stat-divider">\u00b7</span><span class="rarity-pill ' + rarityClass + '">' + escHtml(d.r) + '</span>';
    }
    html += '</div>';
    html += '</div>';

    // Top card full detail
    html += buildCardDetailHtml(d);

    // Ensure obsolescence data is loaded, then patch it in
    if (!obsolescenceIndex) {
      loadObsolescenceIndex().then(ok => {
        if (ok && selectedCards.length > 0) {
          const obsHtml = buildObsolescenceHtml(selectedCards[topCardIndex].data.n);
          const placeholder = inner.querySelector('.obsolescence-placeholder');
          if (placeholder && obsHtml) placeholder.outerHTML = obsHtml;
        }
      });
    }

    // Stack tabs for all selected cards
    if (selectedCards.length > 1) {
      html += '<div class="stack-tabs">';
      for (let i = 0; i < selectedCards.length; i++) {
        const isActive = (i === topCardIndex);
        const card = selectedCards[i];
        const cd = card.data;
        html += '<div class="stack-tab' + (isActive ? ' active' : '') + '" onclick="MM.bringToTop(' + i + ')">';
        html += '<span class="stack-tab-name">' + escHtml(cd.n) + '</span>';
        html += '<span class="stack-tab-mana">' + renderManaSymbols(cd.mc) + '</span>';
        if (cd.p != null && cd.th != null) html += '<span class="stack-tab-stats">' + escHtml(cd.p) + '/' + escHtml(cd.th) + '</span>';
        html += '<span class="stack-tab-type">' + escHtml(cd.s) + '</span>';
        if (cd.m > 0) html += '<span class="stack-tab-cmc">' + cd.m + '</span>';
        html += '<button class="stack-tab-remove" onclick="event.stopPropagation(); MM.removeFromSelection(' + card.idx + ')">\u00d7</button>';
        html += '</div>';
      }
      html += '</div>';
      html += '<div class="keyboard-hint">\u2190 \u2192 navigate \u00b7 1-8 jump \u00b7 Del remove \u00b7 Esc clear all \u00b7 / search</div>';
    } else {
      html += '<div class="keyboard-hint">Shift+click to multi-select \u00b7 Esc clear \u00b7 / search</div>';
    }

    inner.innerHTML = html;
    panel.classList.add('open');
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  function closeViewerPanel() {
    document.getElementById('detailPanel').classList.remove('open');
    clearSimilarTrace();
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  // ── Selection Highlight on Plot ──

  function updateSelectionHighlight() {
    const plotDiv = document.getElementById('plot');
    if (!plotDiv || !plotDiv.data) return;

    // Remove existing selection traces
    const toDelete = [];
    for (let i = plotDiv.data.length - 1; i >= 0; i--) {
      if (plotDiv.data[i]._isSelection) {
        toDelete.push(i);
      }
    }
    if (toDelete.length) {
      Plotly.deleteTraces('plot', toDelete);
    }

    if (selectedCards.length === 0) return;

    // Build selection highlight trace
    const topIdx = selectedCards[topCardIndex]?.idx;
    const otherCards = selectedCards.filter((_, i) => i !== topCardIndex);

    const traces = [];

    // Other selected cards (dimmer gold)
    if (otherCards.length > 0) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Selected',
        x: otherCards.map(c => allData[c.idx].x),
        y: otherCards.map(c => allData[c.idx].y),
        text: otherCards.map(c => buildHoverTextMinimal(allData[c.idx])),
        customdata: otherCards.map(c => c.idx),
        hoverinfo: 'none',
        marker: { size: 12, opacity: 1, color: '#8B7730', symbol: 'circle', line: { color: '#fff', width: 1.5 } },
        _isSelection: true,
      });
    }

    // Top card (bright gold)
    if (topIdx != null) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Active',
        x: [allData[topIdx].x],
        y: [allData[topIdx].y],
        text: [buildHoverTextMinimal(allData[topIdx])],
        customdata: [topIdx],
        hoverinfo: 'none',
        marker: { size: 12, opacity: 1, color: '#c4a747', symbol: 'circle', line: { color: '#fff', width: 2 } },
        _isSelection: true,
      });
    }

    if (traces.length) {
      Plotly.addTraces('plot', traces);
    }
  }

  // ── Detail Panel (legacy wrapper for backward compat) ──

  function showDetailPanel(d) {
    clearSelection();
    const idx = allData.indexOf(d);
    if (idx !== -1) {
      addToSelection(idx);
    }
  }

  // ── Find Similar Cards ──

  function clearSimilarTrace() {
    if (similarTrace) {
      const plotDiv = document.getElementById('plot');
      const numTraces = plotDiv.data.length;
      const toDelete = [];
      for (let i = numTraces - 1; i >= 0; i--) {
        if (plotDiv.data[i]._isSimilar || plotDiv.data[i]._isReference) {
          toDelete.push(i);
        }
      }
      if (toDelete.length) {
        Plotly.deleteTraces('plot', toDelete);
      }
      similarTrace = null;
    }
  }

  async function loadEmbeddings() {
    if (embeddings) return true;
    // Check cache first
    if (embeddingsCache[currentMap]) {
      embeddings = embeddingsCache[currentMap];
      return true;
    }
    try {
      const config = MAP_CONFIGS[currentMap];
      const r = await fetch(config.embeddings);
      if (!r.ok) return false;
      const buf = await r.arrayBuffer();
      embeddings = new Float32Array(buf);
      embeddingsCache[currentMap] = embeddings;
      return true;
    } catch (e) {
      return false;
    }
  }

  function cosineSimilarity(idxA, idxB) {
    const offA = idxA * EMBED_DIM;
    const offB = idxB * EMBED_DIM;
    let dot = 0, normA = 0, normB = 0;
    for (let j = 0; j < EMBED_DIM; j++) {
      const a = embeddings[offA + j], b = embeddings[offB + j];
      dot += a * b;
      normA += a * a;
      normB += b * b;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  async function findSimilarCards() {
    const ref = getSelectedCard();
    if (!ref) return;
    clearSimilarTrace();

    const refIdx = selectedCards[topCardIndex].idx;

    // Load embeddings on first use
    if (!embeddings) {
      setStatus('Loading embeddings...');
      const ok = await loadEmbeddings();
      if (!ok) {
        setStatus('Could not load embeddings.bin \u2014 run export_embeddings.py');
        return;
      }
    }

    // Cosine similarity in 128D embedding space
    const scored = [];
    for (let i = 0; i < allData.length; i++) {
      if (i === refIdx) continue;
      if (!activeSupertypes.has(allData[i].s)) continue;
      scored.push({ i, sim: cosineSimilarity(refIdx, i) });
    }
    scored.sort((a, b) => b.sim - a.sim);
    const nearest = scored.slice(0, 20);

    const simTrace = {
      type: 'scattergl',
      mode: 'markers',
      name: 'Similar (20)',
      x: nearest.map(n => allData[n.i].x),
      y: nearest.map(n => allData[n.i].y),
      text: nearest.map(n => buildHoverTextMinimal(allData[n.i])),
      customdata: nearest.map(n => n.i),
      hoverinfo: 'none',
      marker: { size: 9, opacity: 1, color: '#FFA500', symbol: 'diamond', line: { color: '#EA580C', width: 1.5 } },
      _isSimilar: true,
    };

    const refTrace = {
      type: 'scattergl',
      mode: 'markers',
      name: ref.n,
      x: [ref.x],
      y: [ref.y],
      text: [buildHoverTextMinimal(ref)],
      customdata: [refIdx],
      hoverinfo: 'none',
      marker: { size: 14, opacity: 1, color: '#c4a747', symbol: 'star', line: { color: '#fff', width: 1.5 } },
      _isReference: true,
    };

    Plotly.addTraces('plot', [simTrace, refTrace]);
    similarTrace = true;

    // Populate card viewer with top results (reference card on top + up to 7 similar)
    selectedCards = [{ idx: refIdx, data: ref }];
    const fillCount = Math.min(nearest.length, MAX_SELECTED - 1);
    for (let j = 0; j < fillCount; j++) {
      selectedCards.push({ idx: nearest[j].i, data: allData[nearest[j].i] });
    }
    topCardIndex = 0;
    updateViewerPanel();
    updateSelectionHighlight();

    setStatus(`20 similar cards to "${ref.n}" highlighted (128D cosine similarity)`);
  }

  // ── Obsolescence Loading ──

  async function loadObsolescenceIndex() {
    if (obsolescenceIndex) return true;
    try {
      const r = await fetch('../data/obsolescence_index.json');
      if (!r.ok) return false;
      obsolescenceIndex = await r.json();
      return true;
    } catch (e) {
      return false;
    }
  }

  function buildObsolescenceHtml(cardName) {
    if (!obsolescenceIndex) return '<span class="obsolescence-placeholder"></span>';
    if (!obsolescenceIndex[cardName]) return '';
    const data = obsolescenceIndex[cardName];
    if (!data.obsoleted_by || data.obsoleted_by.length === 0) return '';

    let html = '<div class="obsolescence-section">';
    html += '<div class="obsolescence-title">Obsoleted By</div>';
    for (const rep of data.obsoleted_by.slice(0, 3)) {
      html += '<div class="obsolescence-item">';
      html += '<span class="obsolescence-name clickable" onclick="MM.selectByName(\'' + escHtml(rep.name).replace(/'/g, "\\'") + '\')">' + escHtml(rep.name) + '</span>';
      html += '<div class="obsolescence-advantages">';
      for (const adv of rep.advantages) {
        html += '<span class="obsolescence-badge">' + escHtml(adv) + '</span>';
      }
      html += '</div>';
      html += '</div>';
    }
    html += '</div>';
    return html;
  }

  // Eagerly load obsolescence data in background
  setTimeout(loadObsolescenceIndex, 2000);

  // ── Map switching ──

  async function loadProjection(mapName) {
    if (projectionCache[mapName]) {
      applyProjection(projectionCache[mapName]);
      return;
    }

    const config = MAP_CONFIGS[mapName];
    if (!config) return;

    setStatus('Loading ' + mapName + ' map...');
    try {
      const r = await fetch(config.projection);
      if (!r.ok) throw new Error('Projection file not found \u2014 run pipeline');
      const data = await r.json();
      projectionCache[mapName] = data;
      applyProjection(data);
    } catch (e) {
      setStatus('Error loading ' + mapName + ' map: ' + e.message);
    }
  }

  function applyProjection(data) {
    // Update x/y on allData from the new projection
    for (let i = 0; i < allData.length && i < data.length; i++) {
      allData[i].x = data[i].x;
      allData[i].y = data[i].y;
    }
    // Clear embeddings cache for current map (will reload on Find Similar)
    embeddings = embeddingsCache[currentMap] || null;
    plotInitialized = false; // Force full re-render with new coordinates
    render();
    setStatus(`${allData.length.toLocaleString()} cards loaded \u2014 ${currentMap === 'ability' ? 'Abilities' : 'Color + Type'} map`);
  }

  async function switchMap(mapName) {
    if (mapName === currentMap) return;
    currentMap = mapName;
    embeddings = embeddingsCache[currentMap] || null;
    await loadProjection(mapName);
    // Re-apply selection highlight after map switch (positions changed)
    updateSelectionHighlight();
  }

  // ── Find Synergies ──

  let synergyGraph = null; // lazy-loaded
  let obsolescenceIndex = null; // lazy-loaded

  async function loadSynergyGraph() {
    if (synergyGraph) return true;
    try {
      const r = await fetch('../data/synergy_graph.json');
      if (!r.ok) return false;
      synergyGraph = await r.json();
      return true;
    } catch (e) {
      return false;
    }
  }

  async function findSynergyCards() {
    const ref = getSelectedCard();
    if (!ref) return;
    clearSimilarTrace();

    const refIdx = selectedCards[topCardIndex].idx;

    // Lazy-load synergy graph
    if (!synergyGraph) {
      setStatus('Loading synergy data...');
      const ok = await loadSynergyGraph();
      if (!ok) {
        setStatus('Could not load synergy_graph.json \u2014 run synergy.py');
        return;
      }
    }

    const partners = synergyGraph[ref.n];
    if (!partners || partners.length === 0) {
      setStatus(`No synergy partners found for "${ref.n}"`);
      return;
    }

    // Build name-to-index map
    const nameToIdx = {};
    allData.forEach((d, i) => { nameToIdx[d.n] = i; });

    const synPoints = [];
    for (const p of partners) {
      const idx = nameToIdx[p.partner];
      if (idx == null) continue;
      if (!activeSupertypes.has(allData[idx].s)) continue;
      synPoints.push({ idx, score: p.score, synergies: p.synergies });
    }

    if (synPoints.length === 0) {
      setStatus(`No visible synergy partners for "${ref.n}"`);
      return;
    }

    const synTrace = {
      type: 'scattergl',
      mode: 'markers',
      name: 'Synergies (' + synPoints.length + ')',
      x: synPoints.map(p => allData[p.idx].x),
      y: synPoints.map(p => allData[p.idx].y),
      text: synPoints.map(p => {
        const d = allData[p.idx];
        return '<b>' + escHtml(d.n) + '</b><br>' + p.synergies.join(', ');
      }),
      customdata: synPoints.map(p => p.idx),
      hoverinfo: 'none',
      marker: { size: 10, opacity: 1, color: '#E040FB', symbol: 'diamond', line: { color: '#9C27B0', width: 1.5 } },
      _isSimilar: true,
    };

    const refTrace = {
      type: 'scattergl',
      mode: 'markers',
      name: ref.n,
      x: [ref.x],
      y: [ref.y],
      text: [buildHoverTextMinimal(ref)],
      customdata: [refIdx],
      hoverinfo: 'none',
      marker: { size: 14, opacity: 1, color: '#c4a747', symbol: 'star', line: { color: '#fff', width: 1.5 } },
      _isReference: true,
    };

    Plotly.addTraces('plot', [synTrace, refTrace]);
    similarTrace = true;

    // Populate card viewer with top results (reference card on top + up to 7 synergy partners)
    selectedCards = [{ idx: refIdx, data: ref }];
    const fillCount = Math.min(synPoints.length, MAX_SELECTED - 1);
    for (let j = 0; j < fillCount; j++) {
      selectedCards.push({ idx: synPoints[j].idx, data: allData[synPoints[j].idx] });
    }
    topCardIndex = 0;
    updateViewerPanel();
    updateSelectionHighlight();

    const labels = synPoints.slice(0, 3).map(p =>
      allData[p.idx].n + ' (' + p.synergies[0] + ')'
    ).join(', ');
    setStatus(`${synPoints.length} synergy partners for "${ref.n}" \u2014 ${labels}...`);
  }

  // ── Load data ──
  fetch('../data/projection_2d.json')
    .then(r => r.json())
    .then(data => {
      allData = data;
      projectionCache['default'] = data;
      initToggles();
      render();
      setStatus(`${allData.length.toLocaleString()} cards loaded`);
    })
    .catch(err => setStatus('Error loading data: ' + err.message));

  // ── Supertype toggle buttons ──
  function initToggles() {
    const container = document.getElementById('toggles');
    SUPERTYPES.forEach(st => {
      const btn = document.createElement('button');
      btn.className = 'toggle-btn active';
      btn.textContent = st;
      btn.dataset.supertype = st;
      btn.addEventListener('click', () => {
        if (activeSupertypes.has(st)) {
          activeSupertypes.delete(st);
          btn.classList.remove('active');
        } else {
          activeSupertypes.add(st);
          btn.classList.add('active');
        }
        render();
      });
      container.appendChild(btn);
    });
  }

  // ── Event listeners ──
  document.getElementById('colorBy').addEventListener('change', e => {
    currentColorBy = e.target.value;
    render();
  });

  document.getElementById('mapSelect').addEventListener('change', e => {
    switchMap(e.target.value);
  });

  document.getElementById('search').addEventListener('input', e => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      searchTerm = e.target.value.trim().toLowerCase();
      render();
    }, 300);
  });

  // ── Keyboard handlers ──
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (currentMode === 'build') {
        if (typeof window.DeckBuilder !== 'undefined') {
          window.DeckBuilder.handleEscape();
        }
      } else {
        clearSelection();
      }
      return;
    }

    // Stack navigation: only in explore mode, not typing in an input
    if (currentMode !== 'explore') return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    // / to focus search
    if (e.key === '/') {
      e.preventDefault();
      const searchInput = document.getElementById('search');
      if (searchInput) searchInput.focus();
      return;
    }

    if (selectedCards.length === 0) return;

    if ((e.key === 'ArrowLeft' || e.key === 'ArrowUp') && selectedCards.length > 1) {
      e.preventDefault();
      const newIdx = topCardIndex <= 0 ? selectedCards.length - 1 : topCardIndex - 1;
      bringToTop(newIdx);
    } else if ((e.key === 'ArrowRight' || e.key === 'ArrowDown') && selectedCards.length > 1) {
      e.preventDefault();
      const newIdx = topCardIndex >= selectedCards.length - 1 ? 0 : topCardIndex + 1;
      bringToTop(newIdx);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      removeFromSelection(selectedCards[topCardIndex].idx);
    } else if (e.key >= '1' && e.key <= '8') {
      const n = parseInt(e.key) - 1;
      if (n < selectedCards.length) {
        e.preventDefault();
        bringToTop(n);
      }
    }
  });

  // ── Shift+Drag Box Select ──
  let shiftHeld = false;

  document.addEventListener('keydown', e => {
    if (e.key === 'Shift' && !shiftHeld && currentMode === 'explore') {
      shiftHeld = true;
      const plotDiv = document.getElementById('plot');
      if (plotDiv && plotDiv._fullLayout) {
        Plotly.relayout('plot', { dragmode: 'select' });
      }
      // Show shift-mode hint
      let hint = document.getElementById('shiftHint');
      if (!hint) {
        hint = document.createElement('div');
        hint.id = 'shiftHint';
        hint.className = 'shift-hint';
        hint.textContent = '\u21e7 Multi-select';
        plotDiv.style.position = 'relative';
        plotDiv.appendChild(hint);
      }
      hint.style.display = '';
    }
  });

  document.addEventListener('keyup', e => {
    if (e.key === 'Shift' && shiftHeld) {
      shiftHeld = false;
      const plotDiv = document.getElementById('plot');
      if (plotDiv && plotDiv._fullLayout) {
        Plotly.relayout('plot', { dragmode: 'pan' });
      }
      // Hide shift-mode hint
      const hint = document.getElementById('shiftHint');
      if (hint) hint.style.display = 'none';
    }
  });

  // ── Mode Toggle ──
  document.getElementById('modeSelect').addEventListener('change', e => {
    setMode(e.target.value);
  });

  function setMode(mode) {
    currentMode = mode;
    if (mode === 'build') {
      clearSelection();
      document.getElementById('detailPanel').style.display = 'none';
      if (typeof window.DeckBuilder !== 'undefined') {
        window.DeckBuilder.enter();
      }
    } else {
      document.getElementById('detailPanel').style.display = '';
      if (typeof window.DeckBuilder !== 'undefined') {
        window.DeckBuilder.exit();
      }
    }
    render();
  }

  // ── Mobile pinch-to-zoom ──
  (function () {
    const plotEl = document.getElementById('plot');
    let startDist = null;
    let startMidX = null;
    let startMidY = null;
    let startXRange = null;
    let startYRange = null;

    function touchDist(t) {
      const dx = t[1].clientX - t[0].clientX;
      const dy = t[1].clientY - t[0].clientY;
      return Math.sqrt(dx * dx + dy * dy);
    }

    function getAxRanges() {
      const xa = plotEl._fullLayout.xaxis;
      const ya = plotEl._fullLayout.yaxis;
      return { x: xa.range.slice(), y: ya.range.slice() };
    }

    function plotFraction(clientX, clientY) {
      const rect = plotEl.getBoundingClientRect();
      const fl = plotEl._fullLayout;
      const plotLeft = rect.left + fl.margin.l;
      const plotTop = rect.top + fl.margin.t;
      const plotWidth = fl.width - fl.margin.l - fl.margin.r;
      const plotHeight = fl.height - fl.margin.t - fl.margin.b;
      return {
        fx: (clientX - plotLeft) / plotWidth,
        fy: (clientY - plotTop) / plotHeight,
      };
    }

    plotEl.addEventListener('touchstart', function (e) {
      if (e.touches.length === 2) {
        e.preventDefault();
        const t = e.touches;
        startDist = touchDist(t);
        startMidX = (t[0].clientX + t[1].clientX) / 2;
        startMidY = (t[0].clientY + t[1].clientY) / 2;
        const ranges = getAxRanges();
        startXRange = ranges.x;
        startYRange = ranges.y;
      }
    }, { passive: false });

    plotEl.addEventListener('touchmove', function (e) {
      if (e.touches.length === 2 && startDist) {
        e.preventDefault();
        const t = e.touches;
        const curDist = touchDist(t);
        const scale = startDist / curDist;

        const { fx, fy } = plotFraction(startMidX, startMidY);

        const xLen = startXRange[1] - startXRange[0];
        const yLen = startYRange[1] - startYRange[0];
        const newXLen = xLen * scale;
        const newYLen = yLen * scale;

        const anchorX = startXRange[0] + fx * xLen;
        const anchorY = startYRange[1] - fy * yLen;

        const newXRange = [anchorX - fx * newXLen, anchorX + (1 - fx) * newXLen];
        const newYRange = [anchorY - (1 - fy) * newYLen, anchorY + fy * newYLen];

        Plotly.relayout(plotEl, {
          'xaxis.range': newXRange,
          'yaxis.range': newYRange,
        });
      }
    }, { passive: false });

    plotEl.addEventListener('touchend', function () {
      startDist = null;
    });
  })();

  // ── Get category key and palette for current color mode ──
  function getCategoryInfo(d) {
    if (currentColorBy === 'color') return { key: d.c, palette: COLOR_PALETTE };
    if (currentColorBy === 'supertype') return { key: d.s, palette: SUPERTYPE_PALETTE };
    return { key: d.r, palette: RARITY_PALETTE };
  }

  // ── Render plot ──
  function render() {
    clearSimilarTrace();

    // Get overlay traces from deck builder (if active)
    let overlayTraces = [];
    let dimmedIndices = null;
    if (currentMode === 'build' && typeof window.DeckBuilder !== 'undefined') {
      overlayTraces = window.DeckBuilder.getOverlayTraces();
      dimmedIndices = window.DeckBuilder.getDimmedIndices();
    }

    const filtered = allData.filter(d => activeSupertypes.has(d.s));

    // Group by category (iterate with index to avoid O(n) indexOf)
    const groups = {};
    for (let i = 0; i < allData.length; i++) {
      const d = allData[i];
      if (!activeSupertypes.has(d.s)) continue;
      const { key } = getCategoryInfo(d);
      if (!groups[key]) groups[key] = { x: [], y: [], text: [], customdata: [], key };
      groups[key].x.push(d.x);
      groups[key].y.push(d.y);
      groups[key].text.push(buildHoverTextMinimal(d));
      groups[key].customdata.push(i);
    }

    const palette = currentColorBy === 'color' ? COLOR_PALETTE
      : currentColorBy === 'supertype' ? SUPERTYPE_PALETTE
      : RARITY_PALETTE;

    // Build traces with optional per-point opacity for dimming
    const traces = Object.values(groups).map(g => {
      let opacity;
      if (dimmedIndices) {
        opacity = g.customdata.map(idx => dimmedIndices.has(idx) ? 0.08 : 0.7);
      } else {
        opacity = 0.7;
      }
      return {
        type: 'scattergl',
        mode: 'markers',
        name: g.key,
        x: g.x,
        y: g.y,
        text: g.text,
        customdata: g.customdata,
        hoverinfo: 'none',
        marker: { size: 3, opacity, color: palette[g.key] || '#666' },
      };
    });

    // Search highlight trace (index-tracking to avoid O(n²) indexOf)
    if (searchTerm.length >= 2) {
      const term = searchTerm;
      let matches = [];
      let isOracleSearch = false;
      let oracleTotal = 0;

      // Tier 1: exact name match
      for (let i = 0; i < allData.length; i++) {
        const d = allData[i];
        if (activeSupertypes.has(d.s) && d.n.toLowerCase() === term) matches.push({i, d});
      }
      // Tier 2: name starts with
      if (!matches.length) {
        for (let i = 0; i < allData.length; i++) {
          const d = allData[i];
          if (activeSupertypes.has(d.s) && d.n.toLowerCase().startsWith(term)) matches.push({i, d});
        }
      }
      // Tier 3: name includes
      if (!matches.length) {
        for (let i = 0; i < allData.length; i++) {
          const d = allData[i];
          if (activeSupertypes.has(d.s) && d.n.toLowerCase().includes(term)) matches.push({i, d});
        }
      }
      // Tier 4: oracle text includes (capped at 200)
      if (!matches.length) {
        const oracleMatches = [];
        for (let i = 0; i < allData.length; i++) {
          const d = allData[i];
          if (activeSupertypes.has(d.s) && d.o && d.o.toLowerCase().includes(term)) oracleMatches.push({i, d});
        }
        oracleTotal = oracleMatches.length;
        matches = oracleMatches.slice(0, 200);
        isOracleSearch = matches.length > 0;
      }

      if (matches.length) {
        const displayCount = isOracleSearch && oracleTotal > matches.length
          ? matches.length + ' of ' + oracleTotal.toLocaleString()
          : String(matches.length);
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: `Search (${displayCount})`,
          x: matches.map(m => m.d.x),
          y: matches.map(m => m.d.y),
          text: matches.map(m => buildHoverTextMinimal(m.d)),
          customdata: matches.map(m => m.i),
          hoverinfo: 'none',
          marker: { size: 8, opacity: 1, color: '#fff', symbol: 'diamond', line: { color: '#EA580C', width: 2 } },
        });
        const suffix = isOracleSearch ? ' (oracle text)' : '';
        setStatus(`${displayCount} result${matches.length === 1 ? '' : 's'} for "${searchTerm}"${suffix} \u2014 ${filtered.length.toLocaleString()} cards shown`);
      } else {
        setStatus(`No results for "${searchTerm}" \u2014 ${filtered.length.toLocaleString()} cards shown`);
      }
    } else if (currentMode === 'explore') {
      setStatus(`${filtered.length.toLocaleString()} cards shown`);
    }

    // Add overlay traces from deck builder
    traces.push(...overlayTraces);

    const layout = {
      paper_bgcolor: '#1a1a2e',
      plot_bgcolor: '#1a1a2e',
      font: { color: '#e0e0e0' },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: { visible: false, scaleanchor: 'y' },
      yaxis: { visible: false },
      dragmode: shiftHeld ? 'select' : 'pan',
      legend: { bgcolor: 'rgba(22,33,62,0.85)', bordercolor: '#3a3a5a', borderwidth: 1, font: { size: 11 } },
      hovermode: 'closest',
    };

    const config = { scrollZoom: true, displayModeBar: false, responsive: true };

    if (!plotInitialized) {
      Plotly.newPlot('plot', traces, layout, config);

      // Click handler
      document.getElementById('plot').on('plotly_click', function (eventData) {
        if (eventData.points && eventData.points.length > 0) {
          const pt = eventData.points[0];
          const idx = pt.customdata;
          if (idx != null && allData[idx]) {
            if (currentMode === 'build') {
              // In build mode, clicking adds to seeds
              if (typeof window.DeckBuilder !== 'undefined') {
                window.DeckBuilder.addSeed(idx);
              }
            } else if (eventData.event && eventData.event.shiftKey) {
              // Shift+Click: toggle in/out of selection
              const existing = selectedCards.findIndex(c => c.idx === idx);
              if (existing !== -1) {
                removeFromSelection(idx);
              } else {
                addToSelection(idx);
              }
            } else {
              // Regular click: replace selection
              clearSelection();
              addToSelection(idx);
            }
          }
        }
      });

      // Box select handler (Shift+Drag)
      document.getElementById('plot').on('plotly_selected', function (eventData) {
        if (!eventData || !eventData.points || eventData.points.length <= 1) return;
        if (!shiftHeld) return;

        // Collect indices from selected points
        const indices = [];
        for (const pt of eventData.points) {
          if (pt.customdata != null && allData[pt.customdata]) {
            indices.push(pt.customdata);
          }
          if (indices.length >= MAX_SELECTED) break;
        }

        if (indices.length === 0) return;

        // Replace selection with box-selected cards
        selectedCards = indices.map(idx => ({ idx, data: allData[idx] }));
        topCardIndex = 0;
        updateViewerPanel();
        updateSelectionHighlight();

        const total = eventData.points.length;
        if (total > MAX_SELECTED) {
          setStatus(`Selected ${MAX_SELECTED} of ${total} cards (max ${MAX_SELECTED})`);
        } else {
          setStatus(`Selected ${indices.length} card${indices.length === 1 ? '' : 's'}`);
        }
      });

      plotInitialized = true;
    } else {
      Plotly.react('plot', traces, layout, config);
    }

    // Re-apply selection highlight after render
    updateSelectionHighlight();
  }

  function setStatus(msg) { document.getElementById('status').textContent = msg; }

  function selectByName(name) {
    const idx = allData.findIndex(d => d.n === name);
    if (idx !== -1) addToSelection(idx);
  }

  // ── Expose shared state/functions on window.MM ──
  window.MM = {
    get allData() { return allData; },
    get currentMode() { return currentMode; },
    get currentMap() { return currentMap; },
    get activeSupertypes() { return activeSupertypes; },
    get selectedCard() { return getSelectedCard(); },
    get selectedCards() { return selectedCards; },
    escHtml,
    buildHoverText,
    buildHoverTextMinimal,
    renderManaSymbols,
    showDetailPanel,
    closeDetail: clearSelection,
    addToSelection,
    removeFromSelection,
    clearSelection,
    bringToTop,
    selectByName,
    findSimilar: findSimilarCards,
    findSynergies: findSynergyCards,
    render,
    setStatus,
    setMode,
    ALL_FORMATS,
    SUPERTYPES,
    MAP_CONFIGS,
  };
})();
