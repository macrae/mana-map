/**
 * deck-builder.js — Deck builder: state management, UI, recommendation algorithm,
 * mana base generation, and export.
 */
(function () {

  // ── Format Rules ──
  const FORMAT_RULES = {
    standard:  { deckSize: 60, maxCopies: 4, sideboard: 15 },
    modern:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
    legacy:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
    vintage:   { deckSize: 60, maxCopies: 4, sideboard: 15 },
    commander: { deckSize: 100, maxCopies: 1, sideboard: 0 },
    pioneer:   { deckSize: 60, maxCopies: 4, sideboard: 15 },
    pauper:    { deckSize: 60, maxCopies: 4, sideboard: 15 },
    historic:  { deckSize: 60, maxCopies: 4, sideboard: 15 },
  };

  const DEFAULT_DISTRIBUTIONS = {
    commander: { Creature: 30, Instant: 8, Sorcery: 7, Enchantment: 8, Artifact: 10, Planeswalker: 1, Land: 36, Battle: 0 },
    default60: { Creature: 20, Instant: 6, Sorcery: 4, Enchantment: 3, Artifact: 3, Planeswalker: 0, Land: 24, Battle: 0 },
  };

  const BASIC_LANDS = new Set(['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']);
  const EMBED_DIM = 128;
  let synergyGraph = null; // lazy-loaded for synergy scoring

  // ── Deck State ──
  let deckState = null;

  function initDeckState() {
    return {
      format: 'commander',
      commander: null,
      colorIdentity: new Set(),
      seeds: [],
      accepted: [],
      rejected: new Set(),
      recommendations: [],
      distribution: { ...DEFAULT_DISTRIBUTIONS.commander },
      landSlots: [],
      embeddings: null,
      comboGraph: null,
      nameToIndex: null,
    };
  }

  function getDistDefault(format) {
    if (format === 'commander') return { ...DEFAULT_DISTRIBUTIONS.commander };
    return { ...DEFAULT_DISTRIBUTIONS.default60 };
  }

  // ── Data Loading ──

  async function loadBuildData() {
    const panel = document.getElementById('deckPanel');
    const inner = document.getElementById('deckInner');
    inner.innerHTML = '<div class="deck-loading"><div class="spinner"></div>Loading deck builder data...</div>';
    panel.classList.add('open');
    setTimeout(() => Plotly.Plots.resize('plot'), 260);

    try {
      const embUrl = (MM.MAP_CONFIGS && MM.currentMap) ? MM.MAP_CONFIGS[MM.currentMap].embeddings : '../data/embeddings.bin';
      const [embBuf, comboData, synergyData] = await Promise.all([
        fetch(embUrl).then(r => {
          if (!r.ok) throw new Error('embeddings.bin not found — run export_embeddings.py');
          return r.arrayBuffer();
        }),
        fetch('../data/combo_graph.json').then(r => {
          if (!r.ok) throw new Error('combo_graph.json not found — run process_combos.py');
          return r.json();
        }),
        fetch('../data/synergy_graph.json').then(r => {
          if (!r.ok) return null;
          return r.json();
        }).catch(() => null),
      ]);

      deckState.embeddings = new Float32Array(embBuf);
      deckState.comboGraph = comboData;
      synergyGraph = synergyData;

      // Build name → index map
      deckState.nameToIndex = new Map();
      MM.allData.forEach((d, i) => {
        deckState.nameToIndex.set(d.n, i);
      });

      renderDeckPanel();
    } catch (err) {
      inner.innerHTML = '<div class="deck-loading" style="color:#DC2626;">Error: ' + MM.escHtml(err.message) + '</div>';
    }
  }

  // ── Color Identity Helpers ──

  function parseColorIdentity(ciStr) {
    if (!ciStr) return new Set();
    return new Set(ciStr.split(',').map(s => s.trim()).filter(Boolean));
  }

  function computeDeckColorIdentity() {
    const ci = new Set();
    if (deckState.commander != null) {
      const d = MM.allData[deckState.commander];
      parseColorIdentity(d.ci).forEach(c => ci.add(c));
    }
    // For non-commander, union of seed color identities
    if (deckState.format !== 'commander') {
      for (const idx of deckState.seeds) {
        const d = MM.allData[idx];
        parseColorIdentity(d.ci).forEach(c => ci.add(c));
      }
      for (const idx of deckState.accepted) {
        const d = MM.allData[idx];
        parseColorIdentity(d.ci).forEach(c => ci.add(c));
      }
    }
    deckState.colorIdentity = ci;
  }

  function isColorIdentitySubset(cardCI, deckCI) {
    if (deckCI.size === 0) return true; // no restriction
    const cardColors = parseColorIdentity(cardCI);
    for (const c of cardColors) {
      if (!deckCI.has(c)) return false;
    }
    return true;
  }

  // ── Format Legality ──

  function isLegalInFormat(d, format) {
    if (!d.f) return false;
    return d.f.split(',').includes(format);
  }

  // ── Seed Management ──

  function addSeed(idx) {
    if (!deckState) return;
    const d = MM.allData[idx];
    if (!d) return;

    // Don't add duplicates
    if (deckState.seeds.includes(idx) || deckState.accepted.includes(idx)) return;

    // For commander, check singleton
    if (deckState.format === 'commander' && !BASIC_LANDS.has(d.n)) {
      const allCards = getAllDeckIndices();
      if (allCards.includes(idx)) return;
    }

    deckState.seeds.push(idx);
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function removeSeed(idx) {
    if (!deckState) return;
    deckState.seeds = deckState.seeds.filter(i => i !== idx);
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function setCommander(idx) {
    if (!deckState) return;
    deckState.commander = idx;
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function clearCommander() {
    if (!deckState) return;
    deckState.commander = null;
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  // ── Card Counting ──

  function getAllDeckIndices() {
    if (!deckState) return [];
    const all = [...deckState.seeds, ...deckState.accepted, ...deckState.landSlots];
    if (deckState.commander != null) all.unshift(deckState.commander);
    return all;
  }

  function countByType() {
    const counts = { Creature: 0, Instant: 0, Sorcery: 0, Enchantment: 0, Artifact: 0, Planeswalker: 0, Land: 0, Battle: 0 };
    for (const idx of getAllDeckIndices()) {
      const d = MM.allData[idx];
      if (d && counts[d.s] !== undefined) counts[d.s]++;
    }
    return counts;
  }

  function getDeckSize() {
    return getAllDeckIndices().length;
  }

  // ── Scoring & Recommendations (PR 3) ──

  function computeCentroid(indices) {
    const emb = deckState.embeddings;
    if (!emb || indices.length === 0) return null;
    const c = new Float32Array(EMBED_DIM);
    for (const idx of indices) {
      const off = idx * EMBED_DIM;
      for (let j = 0; j < EMBED_DIM; j++) c[j] += emb[off + j];
    }
    let norm = 0;
    for (let j = 0; j < EMBED_DIM; j++) norm += c[j] * c[j];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let j = 0; j < EMBED_DIM; j++) c[j] /= norm;
    return c;
  }

  function embeddingSim(idx, centroid) {
    const emb = deckState.embeddings;
    const off = idx * EMBED_DIM;
    let dot = 0;
    for (let j = 0; j < EMBED_DIM; j++) dot += emb[off + j] * centroid[j];
    return (dot + 1) / 2; // map [-1,1] to [0,1]
  }

  function comboBonus(cardName) {
    const graph = deckState.comboGraph;
    if (!graph || !graph.partners[cardName]) return 0;
    const partners = graph.partners[cardName];
    const deckNames = new Set(getAllDeckIndices().map(i => MM.allData[i].n));
    for (const p of partners) {
      if (deckNames.has(p)) return 1;
    }
    return 0;
  }

  function getComboPartners(cardName) {
    const graph = deckState.comboGraph;
    if (!graph || !graph.partners[cardName]) return [];
    const partners = graph.partners[cardName];
    const deckNames = new Set(getAllDeckIndices().map(i => MM.allData[i].n));
    return partners.filter(p => deckNames.has(p));
  }

  function edhrecScore(d) {
    if (d.er == null) return 0.3; // neutral default
    // Find max EDHREC rank across all cards (cached)
    if (!deckState._maxEdhrecRank) {
      let max = 0;
      for (const card of MM.allData) {
        if (card.er != null && card.er > max) max = card.er;
      }
      deckState._maxEdhrecRank = max || 1;
    }
    return 1 - (d.er / deckState._maxEdhrecRank);
  }

  function synergyBonus(cardName) {
    if (!synergyGraph || !synergyGraph[cardName]) return { score: 0, labels: [] };
    const partners = synergyGraph[cardName];
    const deckNames = new Set(getAllDeckIndices().map(i => MM.allData[i].n));
    let matchCount = 0;
    const labels = [];
    for (const p of partners) {
      if (deckNames.has(p.partner)) {
        matchCount += p.score;
        for (const label of p.synergies) {
          if (!labels.includes(label)) labels.push(label);
        }
      }
    }
    // Normalize: 5+ synergy matches = max score
    const normalized = Math.min(matchCount / 5, 1);
    return { score: normalized, labels };
  }

  function keywordOverlap(d) {
    if (!d.k) return 0;
    // Build deck keyword set
    const deckKw = new Set();
    for (const idx of getAllDeckIndices()) {
      const card = MM.allData[idx];
      if (card.k) card.k.split(', ').forEach(kw => deckKw.add(kw));
    }
    if (deckKw.size === 0) return 0;

    const cardKw = new Set(d.k.split(', '));
    let intersection = 0;
    for (const kw of cardKw) {
      if (deckKw.has(kw)) intersection++;
    }
    const union = new Set([...deckKw, ...cardKw]).size;
    return union > 0 ? intersection / union : 0;
  }

  function generateRecommendations() {
    if (!deckState || !deckState.embeddings) return;

    const deckIndices = getAllDeckIndices();
    if (deckIndices.length === 0) return;

    const format = deckState.format;
    const rules = FORMAT_RULES[format];
    const typeCounts = countByType();
    const deckSet = new Set(deckIndices);
    const centroid = computeCentroid(deckIndices);
    if (!centroid) return;

    computeDeckColorIdentity();

    // Score all candidates
    const candidates = [];
    for (let i = 0; i < MM.allData.length; i++) {
      const d = MM.allData[i];

      // Filter: already in deck
      if (deckSet.has(i)) continue;
      // Filter: rejected
      if (deckState.rejected.has(i)) continue;
      // Filter: format legality
      if (!isLegalInFormat(d, format)) continue;
      // Filter: color identity
      if (deckState.colorIdentity.size > 0 && !isColorIdentitySubset(d.ci, deckState.colorIdentity)) continue;
      // Filter: skip lands (handled by mana base generator)
      if (d.s === 'Land') continue;
      // Filter: type distribution full (skip only when target > 0 and met; target=0 means uncapped)
      if (deckState.distribution[d.s] > 0 && typeCounts[d.s] >= deckState.distribution[d.s]) continue;
      // Filter: copy limit (commander = singleton)
      if (rules.maxCopies === 1 && !BASIC_LANDS.has(d.n) && deckSet.has(i)) continue;

      const sim = embeddingSim(i, centroid);
      const combo = comboBonus(d.n);
      const synergy = synergyBonus(d.n);
      const edhrec = edhrecScore(d);
      const kwOverlap = keywordOverlap(d);

      const score = 0.40 * sim + 0.20 * combo + 0.20 * synergy.score + 0.10 * edhrec + 0.10 * kwOverlap;

      candidates.push({
        index: i,
        score,
        simScore: sim,
        comboScore: combo,
        synergyScore: synergy.score,
        synergyLabels: synergy.labels,
        edhrecScore: edhrec,
        kwScore: kwOverlap,
        comboPartners: combo > 0 ? getComboPartners(d.n) : [],
      });
    }

    candidates.sort((a, b) => b.score - a.score);
    deckState.recommendations = candidates.slice(0, 20);

    renderDeckPanel();
    MM.render();
    saveDeckState();

    MM.setStatus(`${getDeckSize()}/${rules.deckSize} cards · ${deckState.recommendations.length} recommendations · Build mode`);
  }

  function acceptRecommendation(idx) {
    if (!deckState) return;
    deckState.accepted.push(idx);
    deckState.recommendations = deckState.recommendations.filter(r => r.index !== idx);
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function rejectRecommendation(idx) {
    if (!deckState) return;
    deckState.rejected.add(idx);
    deckState.recommendations = deckState.recommendations.filter(r => r.index !== idx);
    renderDeckPanel();
    saveDeckState();
  }

  function acceptAllRecommendations() {
    if (!deckState) return;
    for (const rec of deckState.recommendations) {
      deckState.accepted.push(rec.index);
    }
    deckState.recommendations = [];
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function removeAccepted(idx) {
    if (!deckState) return;
    deckState.accepted = deckState.accepted.filter(i => i !== idx);
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  // ── Mana Base Generation (PR 4) ──

  function countPips() {
    const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 };
    const allIndices = getAllDeckIndices();
    for (const idx of allIndices) {
      const d = MM.allData[idx];
      if (d.s === 'Land') continue;
      const mc = d.mc || '';
      const tokens = mc.match(/\{([^}]+)\}/g) || [];
      for (const tok of tokens) {
        const inner = tok.slice(1, -1);
        if ('WUBRG'.includes(inner)) pips[inner]++;
        if (inner.includes('/')) {
          inner.split('/').forEach(p => { if ('WUBRG'.includes(p)) pips[p] += 0.5; });
        }
      }
    }
    return pips;
  }

  function detectLandColors(d) {
    const colors = new Set();
    const typeLine = (d.t || '').toLowerCase();
    const oracleText = (d.o || '').toLowerCase();

    // Basic land subtypes
    if (typeLine.includes('plains') || oracleText.includes('add {w}')) colors.add('W');
    if (typeLine.includes('island') || oracleText.includes('add {u}')) colors.add('U');
    if (typeLine.includes('swamp') || oracleText.includes('add {b}')) colors.add('B');
    if (typeLine.includes('mountain') || oracleText.includes('add {r}')) colors.add('R');
    if (typeLine.includes('forest') || oracleText.includes('add {g}')) colors.add('G');

    // Additional patterns like "add one mana of any color"
    if (oracleText.includes('any color')) {
      'WUBRG'.split('').forEach(c => colors.add(c));
    }

    // Fallback to color identity
    if (colors.size === 0 && d.ci) {
      parseColorIdentity(d.ci).forEach(c => colors.add(c));
    }

    return colors;
  }

  function entersTapped(d) {
    const o = (d.o || '').toLowerCase();
    return o.includes('enters the battlefield tapped') || o.includes('enters tapped');
  }

  function hasBasicSubtype(d) {
    const t = (d.t || '').toLowerCase();
    return t.includes('plains') || t.includes('island') || t.includes('swamp') || t.includes('mountain') || t.includes('forest');
  }

  function generateManaBase() {
    if (!deckState) return;

    // Remove existing land slots
    deckState.landSlots = [];

    const pips = countPips();
    const totalPips = Object.values(pips).reduce((a, b) => a + b, 0);
    const totalLandSlots = deckState.distribution.Land || 24;

    if (totalPips === 0 || totalLandSlots === 0) {
      renderDeckPanel();
      return;
    }

    const format = deckState.format;
    const rules = FORMAT_RULES[format];
    const utilitySlots = Math.floor(totalLandSlots * 0.10);
    const coloredSlots = totalLandSlots - utilitySlots;

    // Allocate slots per color
    const colorSlots = {};
    let allocated = 0;
    for (const c of 'WUBRG') {
      if (pips[c] > 0) {
        colorSlots[c] = Math.round(coloredSlots * pips[c] / totalPips);
        allocated += colorSlots[c];
      } else {
        colorSlots[c] = 0;
      }
    }
    // Adjust rounding
    while (allocated > coloredSlots) {
      const maxC = Object.entries(colorSlots).sort((a, b) => b[1] - a[1])[0][0];
      colorSlots[maxC]--;
      allocated--;
    }
    while (allocated < coloredSlots) {
      const maxPipC = Object.entries(pips).sort((a, b) => b[1] - a[1])[0][0];
      colorSlots[maxPipC]++;
      allocated++;
    }

    // Find all legal lands
    const deckSet = new Set(getAllDeckIndices());
    const candidateLands = [];
    for (let i = 0; i < MM.allData.length; i++) {
      const d = MM.allData[i];
      if (d.s !== 'Land') continue;
      if (BASIC_LANDS.has(d.n)) continue; // handle basics separately
      if (!isLegalInFormat(d, format)) continue;
      if (deckState.colorIdentity.size > 0 && !isColorIdentitySubset(d.ci, deckState.colorIdentity)) continue;
      if (deckSet.has(i)) continue;
      if (rules.maxCopies === 1 && deckState.landSlots.includes(i)) continue;

      const colors = detectLandColors(d);
      if (colors.size === 0) continue;

      const score = colors.size * 10
        + (hasBasicSubtype(d) ? 5 : 0)
        + edhrecScore(d) * 3
        - (entersTapped(d) ? 4 : 0);

      candidateLands.push({ index: i, colors, score });
    }

    candidateLands.sort((a, b) => b.score - a.score);

    // Commander extras: Command Tower
    if (format === 'commander' && deckState.colorIdentity.size >= 2) {
      const ctIdx = deckState.nameToIndex ? deckState.nameToIndex.get('Command Tower') : null;
      if (ctIdx != null && !deckSet.has(ctIdx) && isLegalInFormat(MM.allData[ctIdx], format)) {
        deckState.landSlots.push(ctIdx);
        deckSet.add(ctIdx);
      }
    }

    // Greedy set cover
    const remaining = { ...colorSlots };
    const usedLands = new Set(deckState.landSlots);

    for (const land of candidateLands) {
      if (deckState.landSlots.length >= totalLandSlots) break;
      if (usedLands.has(land.index)) continue;

      // Check if this land covers a needed color
      let coversNeeded = false;
      for (const c of land.colors) {
        if (remaining[c] > 0) { coversNeeded = true; break; }
      }
      // Also pick utility lands
      if (!coversNeeded && deckState.landSlots.length < totalLandSlots - Object.values(remaining).reduce((a, b) => a + b, 0)) {
        coversNeeded = true; // utility slot
      }
      if (!coversNeeded) continue;

      deckState.landSlots.push(land.index);
      usedLands.add(land.index);
      if (rules.maxCopies === 1) deckSet.add(land.index);

      for (const c of land.colors) {
        if (remaining[c] > 0) remaining[c]--;
      }
    }

    // Fill remaining with basics
    const basicMap = { W: 'Plains', U: 'Island', B: 'Swamp', R: 'Mountain', G: 'Forest' };
    for (const c of 'WUBRG') {
      while (remaining[c] > 0 && deckState.landSlots.length < totalLandSlots) {
        const basicName = basicMap[c];
        const basicIdx = deckState.nameToIndex ? deckState.nameToIndex.get(basicName) : null;
        if (basicIdx != null) {
          deckState.landSlots.push(basicIdx);
          remaining[c]--;
        } else {
          break;
        }
      }
    }

    // Fill any remaining slots with basics of most-needed color
    while (deckState.landSlots.length < totalLandSlots) {
      const maxPipC = Object.entries(pips).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])[0];
      if (!maxPipC) break;
      const basicIdx = deckState.nameToIndex ? deckState.nameToIndex.get(basicMap[maxPipC[0]]) : null;
      if (basicIdx != null) {
        deckState.landSlots.push(basicIdx);
      } else {
        break;
      }
    }

    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  // ── Export ──

  function exportDeck() {
    if (!deckState) return;
    const allIndices = getAllDeckIndices();
    const nameCounts = {};
    for (const idx of allIndices) {
      const name = MM.allData[idx].n;
      nameCounts[name] = (nameCounts[name] || 0) + 1;
    }

    const lines = [];
    // Commander first
    if (deckState.commander != null) {
      const name = MM.allData[deckState.commander].n;
      lines.push('1 ' + name);
      delete nameCounts[name]; // already counted
    }

    for (const [name, count] of Object.entries(nameCounts).sort((a, b) => a[0].localeCompare(b[0]))) {
      lines.push(count + ' ' + name);
    }

    const text = lines.join('\n');
    navigator.clipboard.writeText(text).then(() => {
      MM.setStatus('Deck list copied to clipboard!');
    }).catch(() => {
      // Fallback: show in a prompt
      prompt('Deck list (copy below):', text);
    });
  }

  // ── Persistence ──

  function saveDeckState() {
    if (!deckState) return;
    const save = {
      format: deckState.format,
      commander: deckState.commander,
      seeds: deckState.seeds,
      accepted: deckState.accepted,
      rejected: [...deckState.rejected],
      distribution: deckState.distribution,
      landSlots: deckState.landSlots,
    };
    try {
      localStorage.setItem('manamap-deck', JSON.stringify(save));
    } catch (e) { /* ignore */ }
  }

  function loadDeckState() {
    try {
      const raw = localStorage.getItem('manamap-deck');
      if (!raw) return null;
      const save = JSON.parse(raw);
      return save;
    } catch (e) { return null; }
  }

  function restoreDeckState() {
    const save = loadDeckState();
    if (!save) return;
    deckState.format = save.format || 'commander';
    deckState.commander = save.commander;
    deckState.seeds = save.seeds || [];
    deckState.accepted = save.accepted || [];
    deckState.rejected = new Set(save.rejected || []);
    deckState.distribution = save.distribution || getDistDefault(deckState.format);
    deckState.landSlots = save.landSlots || [];
    computeDeckColorIdentity();
  }

  // ── Overlay Traces for Plot ──

  function getOverlayTraces() {
    if (!deckState) return [];
    const traces = [];
    const allData = MM.allData;

    // Seed cards — gold stars
    if (deckState.seeds.length > 0) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Seeds',
        x: deckState.seeds.map(i => allData[i].x),
        y: deckState.seeds.map(i => allData[i].y),
        text: deckState.seeds.map(i => MM.buildHoverTextMinimal(allData[i])),
        customdata: deckState.seeds,
        hoverinfo: 'none',
        marker: { size: 12, opacity: 1, color: '#c4a747', symbol: 'star', line: { color: '#fff', width: 1.5 } },
        _isDeckOverlay: true,
      });
    }

    // Commander — large gold star
    if (deckState.commander != null) {
      const ci = deckState.commander;
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Commander',
        x: [allData[ci].x],
        y: [allData[ci].y],
        text: [MM.buildHoverTextMinimal(allData[ci])],
        customdata: [ci],
        hoverinfo: 'none',
        marker: { size: 16, opacity: 1, color: '#c4a747', symbol: 'star', line: { color: '#FFD700', width: 2 } },
        _isDeckOverlay: true,
      });
    }

    // Accepted cards — green circles
    if (deckState.accepted.length > 0) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Accepted',
        x: deckState.accepted.map(i => allData[i].x),
        y: deckState.accepted.map(i => allData[i].y),
        text: deckState.accepted.map(i => MM.buildHoverTextMinimal(allData[i])),
        customdata: deckState.accepted.slice(),
        hoverinfo: 'none',
        marker: { size: 8, opacity: 1, color: '#4ade80', symbol: 'circle', line: { color: '#22C55E', width: 1.5 } },
        _isDeckOverlay: true,
      });
    }

    // Recommendation cards — orange diamonds
    if (deckState.recommendations.length > 0) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Recommendations',
        x: deckState.recommendations.map(r => allData[r.index].x),
        y: deckState.recommendations.map(r => allData[r.index].y),
        text: deckState.recommendations.map(r => MM.buildHoverTextMinimal(allData[r.index])),
        customdata: deckState.recommendations.map(r => r.index),
        hoverinfo: 'none',
        marker: { size: 9, opacity: 1, color: '#FFA500', symbol: 'diamond', line: { color: '#EA580C', width: 1.5 } },
        _isDeckOverlay: true,
      });
    }

    // Land slots — blue squares
    if (deckState.landSlots.length > 0) {
      const uniqueLands = [...new Set(deckState.landSlots)];
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Lands',
        x: uniqueLands.map(i => allData[i].x),
        y: uniqueLands.map(i => allData[i].y),
        text: uniqueLands.map(i => MM.buildHoverTextMinimal(allData[i])),
        customdata: uniqueLands,
        hoverinfo: 'none',
        marker: { size: 7, opacity: 1, color: '#60A5FA', symbol: 'square', line: { color: '#3B82F6', width: 1 } },
        _isDeckOverlay: true,
      });
    }

    return traces;
  }

  function getDimmedIndices() {
    if (!deckState) return null;
    const format = deckState.format;
    const dimmed = new Set();

    for (let i = 0; i < MM.allData.length; i++) {
      const d = MM.allData[i];
      // Dim format-illegal cards
      if (!isLegalInFormat(d, format)) {
        dimmed.add(i);
        continue;
      }
      // Dim color identity violations (Commander)
      if (deckState.colorIdentity.size > 0 && !isColorIdentitySubset(d.ci, deckState.colorIdentity)) {
        dimmed.add(i);
      }
    }
    return dimmed;
  }

  // ── Commander Autocomplete ──

  let commanderTimeout = null;

  function setupCommanderAutocomplete() {
    const input = document.getElementById('commanderInput');
    const listEl = document.getElementById('commanderAutocomplete');
    if (!input || !listEl) return;

    input.addEventListener('input', () => {
      clearTimeout(commanderTimeout);
      commanderTimeout = setTimeout(() => {
        const term = input.value.trim().toLowerCase();
        if (term.length < 2) { listEl.innerHTML = ''; listEl.style.display = 'none'; return; }

        // Filter to legendary creatures legal in commander
        const matches = [];
        for (let i = 0; i < MM.allData.length; i++) {
          const d = MM.allData[i];
          if (!d.t || !d.t.toLowerCase().includes('legendary')) continue;
          if (!d.t.toLowerCase().includes('creature')) continue;
          if (!isLegalInFormat(d, 'commander')) continue;
          if (d.n.toLowerCase().includes(term)) matches.push(i);
          if (matches.length >= 10) break;
        }

        if (matches.length === 0) {
          listEl.innerHTML = '';
          listEl.style.display = 'none';
          return;
        }

        listEl.innerHTML = matches.map(i => {
          const d = MM.allData[i];
          return '<div class="autocomplete-item" data-idx="' + i + '">' + MM.escHtml(d.n) + '</div>';
        }).join('');
        listEl.style.display = 'block';

        listEl.querySelectorAll('.autocomplete-item').forEach(el => {
          el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx);
            setCommander(idx);
            input.value = MM.allData[idx].n;
            listEl.innerHTML = '';
            listEl.style.display = 'none';
          });
        });
      }, 200);
    });

    // Close on click outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.deck-commander-row')) {
        listEl.style.display = 'none';
      }
    });
  }

  // ── Render Deck Panel ──

  function renderDeckPanel() {
    if (!deckState) return;
    const inner = document.getElementById('deckInner');
    const format = deckState.format;
    const rules = FORMAT_RULES[format];
    const typeCounts = countByType();
    const deckSize = getDeckSize();
    let html = '';

    // Header
    html += '<div class="deck-header">';
    html += '<h2>Deck Builder</h2>';
    html += '<button class="detail-close" onclick="DeckBuilder.close()" title="Close">\u00d7</button>';
    html += '</div>';

    // Format selector
    html += '<div class="deck-section">';
    html += '<div class="deck-format-row">';
    html += '<label>Format:</label>';
    html += '<select id="formatSelect" onchange="DeckBuilder.changeFormat(this.value)">';
    for (const f of Object.keys(FORMAT_RULES)) {
      html += '<option value="' + f + '"' + (f === format ? ' selected' : '') + '>' + f.charAt(0).toUpperCase() + f.slice(1) + '</option>';
    }
    html += '</select>';
    html += '</div>';

    // Commander input (only for Commander format)
    if (format === 'commander') {
      html += '<div class="deck-commander-row">';
      html += '<label style="min-width:80px;font-size:0.85rem;color:#aaa;">Commander:</label>';
      const cmdName = deckState.commander != null ? MM.allData[deckState.commander].n : '';
      html += '<input id="commanderInput" type="text" placeholder="Search legendary creatures..." value="' + MM.escHtml(cmdName) + '" autocomplete="off">';
      if (deckState.commander != null) {
        html += '<button class="btn-sm" onclick="DeckBuilder.clearCommander()">Clear</button>';
      }
      html += '<div id="commanderAutocomplete" class="autocomplete-list" style="display:none;"></div>';
      html += '</div>';
    }
    html += '</div>';

    // Seeds section
    html += '<div class="deck-section">';
    html += '<div class="deck-section-title">';
    html += '<span>Seed Cards (' + deckState.seeds.length + ')</span>';
    if (deckState.seeds.length > 0) {
      html += '<button class="btn-sm" onclick="DeckBuilder.clearSeeds()">Clear</button>';
    }
    html += '</div>';

    if (deckState.seeds.length === 0) {
      html += '<div style="color:#666;font-size:0.8rem;padding:8px 0;">Click cards on the map to add seeds</div>';
    } else {
      html += '<ul class="seed-list">';
      for (const idx of deckState.seeds) {
        const d = MM.allData[idx];
        const legal = isLegalInFormat(d, format);
        html += '<li class="seed-item' + (!legal ? ' format-warning' : '') + '">';
        html += '<span class="seed-name">' + MM.escHtml(d.n) + '</span>';
        html += '<span class="seed-mana">' + MM.renderManaSymbols(d.mc) + '</span>';
        html += '<button class="seed-remove" onclick="DeckBuilder.removeSeed(' + idx + ')" title="Remove">\u00d7</button>';
        html += '</li>';
      }
      html += '</ul>';
    }
    html += '</div>';

    // Type distribution
    html += '<div class="deck-section">';
    html += '<div class="deck-section-title">';
    html += '<span>Type Distribution</span>';
    html += '<button class="btn-sm" onclick="DeckBuilder.resetDistribution()">Defaults</button>';
    html += '</div>';

    const distTypes = ['Creature', 'Instant', 'Sorcery', 'Enchantment', 'Artifact', 'Planeswalker', 'Land', 'Battle'];
    for (const type of distTypes) {
      const target = deckState.distribution[type] || 0;
      const current = typeCounts[type] || 0;
      const pct = target > 0 ? Math.min(100, (current / target) * 100) : 0;
      const fillClass = current >= target ? (current > target ? 'over' : 'full') : '';

      html += '<div class="dist-row">';
      html += '<span class="dist-label">' + type + '</span>';
      html += '<input class="dist-input" type="number" min="0" value="' + target + '" onchange="DeckBuilder.setDistribution(\'' + type + '\', parseInt(this.value) || 0)">';
      html += '<div class="dist-bar"><div class="dist-bar-fill ' + fillClass + '" style="width:' + pct + '%"></div></div>';
      html += '<span style="font-size:0.7rem;color:#888;width:30px;text-align:right;">' + current + '</span>';
      html += '</div>';
    }

    const distTotal = Object.values(deckState.distribution).reduce((a, b) => a + b, 0);
    html += '<div class="dist-total">Total: ' + deckSize + ' / ' + rules.deckSize + ' · Target: ' + distTotal + ' · Remaining: ' + Math.max(0, rules.deckSize - deckSize) + '</div>';
    html += '</div>';

    // Generate recommendations button
    html += '<div class="deck-section">';
    const canGenerate = deckState.seeds.length > 0 || deckState.accepted.length > 0 || deckState.commander != null;
    html += '<button class="btn-primary" onclick="DeckBuilder.generate()"' + (!canGenerate ? ' disabled' : '') + '>Generate Recommendations</button>';
    html += '</div>';

    // Recommendations
    if (deckState.recommendations.length > 0) {
      html += '<div class="deck-section">';
      html += '<div class="deck-section-title">';
      html += '<span>Recommendations (' + deckState.recommendations.length + ')</span>';
      html += '<button class="btn-sm" onclick="DeckBuilder.acceptAll()">Accept All</button>';
      html += '</div>';
      html += '<ul class="rec-list">';
      for (const rec of deckState.recommendations) {
        const d = MM.allData[rec.index];
        html += '<li class="rec-item">';

        // Header (always visible)
        html += '<div class="rec-header">';
        html += '<div class="rec-name-row">';
        html += '<span class="rec-name">' + MM.escHtml(d.n) + '</span>';
        html += '<span class="rec-mana">' + MM.renderManaSymbols(d.mc) + '</span>';
        html += '</div>';

        html += '<div class="rec-type">' + MM.escHtml(d.s);
        if (d.p != null && d.th != null) html += ' ' + MM.escHtml(d.p) + '/' + MM.escHtml(d.th);
        else if (d.l != null) html += ' [' + d.l + ']';
        html += '</div>';

        html += '<div class="rec-actions">';
        html += '<span class="rec-score">' + rec.score.toFixed(2) + '</span>';
        html += '<button class="rec-accept" onclick="DeckBuilder.accept(' + rec.index + ')" title="Accept">&#10003;</button>';
        html += '<button class="rec-reject" onclick="DeckBuilder.reject(' + rec.index + ')" title="Reject">&#10007;</button>';
        html += '<button class="rec-expand" onclick="DeckBuilder.toggleRecExpand(this)" title="Expand">&#9660;</button>';
        html += '</div>';
        html += '</div>';

        // Body (hidden by default, shown on expand)
        html += '<div class="rec-body">';

        // Score breakdown chips
        html += '<div class="rec-scores">';
        html += '<span class="rec-score-chip">Similarity ' + rec.simScore.toFixed(2) + '</span>';
        if (rec.comboScore > 0) html += '<span class="rec-score-chip chip-combo">Combo</span>';
        if (rec.synergyScore > 0) html += '<span class="rec-score-chip chip-synergy">Synergy ' + rec.synergyScore.toFixed(2) + '</span>';
        html += '<span class="rec-score-chip">Pop. ' + rec.edhrecScore.toFixed(2) + '</span>';
        if (rec.kwScore > 0) html += '<span class="rec-score-chip">KW ' + rec.kwScore.toFixed(2) + '</span>';
        html += '</div>';

        // Synergy labels
        if (rec.synergyLabels && rec.synergyLabels.length > 0) {
          html += '<div class="rec-synergy">Synergizes: ' + rec.synergyLabels.map(l => MM.escHtml(l)).join(', ') + '</div>';
        }

        // Combo partners
        if (rec.comboPartners.length > 0) {
          html += '<div class="rec-combo">Combos with: ' + rec.comboPartners.map(p => MM.escHtml(p)).join(', ') + '</div>';
        }

        // Keywords
        if (d.k) {
          html += '<div class="rec-keywords">';
          d.k.split(', ').forEach(kw => {
            html += '<span class="keyword-badge">' + MM.escHtml(kw) + '</span>';
          });
          html += '</div>';
        }

        // Oracle text
        if (d.o) {
          html += '<div class="rec-oracle">' + MM.escHtml(d.o) + '</div>';
        }

        html += '</div>'; // .rec-body
        html += '</li>';
      }
      html += '</ul>';
      html += '</div>';
    }

    // Deck list
    const allCards = getAllDeckIndices();
    if (allCards.length > 0) {
      html += '<div class="deck-section">';
      html += '<div class="deck-section-title">';
      html += '<span>Deck List (' + deckSize + '/' + rules.deckSize + ')</span>';
      html += '<div>';
      html += '<button class="btn-sm" onclick="DeckBuilder.exportDeck()" style="margin-right:4px;">Export</button>';
      html += '<button class="btn-sm btn-danger" onclick="DeckBuilder.resetDeck()">Reset</button>';
      html += '</div>';
      html += '</div>';

      // Group by type
      const groups = {};
      for (const idx of allCards) {
        const d = MM.allData[idx];
        const type = d.s || 'Unknown';
        if (!groups[type]) groups[type] = [];
        groups[type].push(idx);
      }

      for (const type of distTypes) {
        if (!groups[type] || groups[type].length === 0) continue;
        const target = deckState.distribution[type] || 0;
        const count = groups[type].length;
        const isComplete = count >= target;

        html += '<div class="deck-group">';
        html += '<div class="deck-group-header">';
        html += '<span>' + type + 's</span>';
        html += '<span class="group-progress' + (isComplete ? ' complete' : '') + '">' + count + '/' + target + (isComplete ? ' &#10003;' : '') + '</span>';
        html += '</div>';

        // Count occurrences for display
        const nameCount = {};
        for (const idx of groups[type]) {
          const name = MM.allData[idx].n;
          if (!nameCount[name]) nameCount[name] = { count: 0, idx };
          nameCount[name].count++;
        }

        for (const [name, info] of Object.entries(nameCount).sort((a, b) => a[0].localeCompare(b[0]))) {
          const cd = MM.allData[info.idx];
          html += '<div class="deck-card">';
          html += '<span class="deck-card-qty">' + (info.count > 1 ? info.count + 'x' : '') + '</span>';
          html += '<span class="deck-card-name">' + MM.escHtml(name) + '</span>';
          html += '<span class="deck-card-mana">' + MM.renderManaSymbols(cd.mc) + '</span>';
          if (cd.p != null && cd.th != null) {
            html += '<span class="deck-card-stats">' + MM.escHtml(cd.p) + '/' + MM.escHtml(cd.th) + '</span>';
          } else if (cd.l != null) {
            html += '<span class="deck-card-stats">[' + cd.l + ']</span>';
          }
          // Only show remove for seeds/accepted, not auto-generated lands
          const isSeed = deckState.seeds.includes(info.idx);
          const isAccepted = deckState.accepted.includes(info.idx);
          if (isSeed) {
            html += '<button class="deck-card-remove" onclick="DeckBuilder.removeSeed(' + info.idx + ')">&#10007;</button>';
          } else if (isAccepted) {
            html += '<button class="deck-card-remove" onclick="DeckBuilder.removeAccepted(' + info.idx + ')">&#10007;</button>';
          }
          html += '</div>';
        }
        html += '</div>';
      }

      // Mana base button (only if no lands yet)
      if (deckState.landSlots.length === 0 && (deckState.seeds.length + deckState.accepted.length) > 0) {
        html += '<button class="btn-primary" onclick="DeckBuilder.generateManaBase()" style="margin-top:8px;">Auto-Generate Mana Base</button>';
      } else if (deckState.landSlots.length > 0) {
        html += '<button class="btn-sm" onclick="DeckBuilder.clearLands()" style="margin-top:8px;">Regenerate Mana Base</button>';
      }

      // Mana curve
      if (allCards.length > 5) {
        html += renderManaCurve(allCards);
      }

      // Color distribution
      if (allCards.length > 0) {
        html += renderColorDist(allCards);
      }
    }

    inner.innerHTML = html;

    // Setup commander autocomplete after DOM update
    if (format === 'commander') {
      setTimeout(setupCommanderAutocomplete, 0);
    }
  }

  function renderManaCurve(indices) {
    const buckets = [0, 0, 0, 0, 0, 0, 0]; // 0,1,2,3,4,5,6+
    for (const idx of indices) {
      const d = MM.allData[idx];
      if (d.s === 'Land') continue;
      const cmc = Math.floor(d.m || 0);
      const bucket = Math.min(cmc, 6);
      buckets[bucket]++;
    }
    const max = Math.max(...buckets, 1);

    let html = '<div style="margin-top:12px;">';
    html += '<div class="deck-section-title"><span>Mana Curve</span></div>';
    html += '<div class="mana-curve">';
    for (let i = 0; i < buckets.length; i++) {
      const pct = (buckets[i] / max) * 100;
      html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">';
      html += '<div style="height:40px;width:100%;display:flex;align-items:flex-end;">';
      html += '<div style="width:100%;background:#c4a747;border-radius:2px 2px 0 0;height:' + pct + '%;min-height:' + (buckets[i] > 0 ? '2px' : '0') + ';"></div>';
      html += '</div>';
      html += '<div class="curve-label">' + (i === 6 ? '6+' : i) + '</div>';
      html += '</div>';
    }
    html += '</div>';
    html += '</div>';
    return html;
  }

  function renderColorDist(indices) {
    const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 };
    let total = 0;
    for (const idx of indices) {
      const d = MM.allData[idx];
      if (d.s === 'Land') continue;
      const mc = d.mc || '';
      const tokens = mc.match(/\{([^}]+)\}/g) || [];
      for (const tok of tokens) {
        const inner = tok.slice(1, -1);
        if ('WUBRG'.includes(inner)) { pips[inner]++; total++; }
      }
    }
    if (total === 0) return '';

    const colorNames = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };
    const colorCSS = { W: '#F9FAF4', U: '#0E68AB', B: '#6B3FA0', R: '#D3202A', G: '#00733E' };

    let html = '<div style="margin-top:8px;">';
    html += '<div class="deck-section-title"><span>Color Distribution</span></div>';
    html += '<div class="color-dist">';
    for (const c of 'WUBRG') {
      if (pips[c] === 0) continue;
      const pct = Math.round(pips[c] / total * 100);
      html += '<div class="color-dist-item"><span class="mana-sym mana-' + c + '" style="width:14px;height:14px;font-size:0.5rem;">' + c + '</span>' + pct + '%</div>';
    }
    html += '</div></div>';
    return html;
  }

  // ── Mode Management ──

  function enterBuildMode() {
    deckState = initDeckState();
    restoreDeckState();
    loadBuildData();
  }

  function exitBuildMode() {
    const panel = document.getElementById('deckPanel');
    panel.classList.remove('open');
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  function closeBuildMode() {
    if (deckState && getAllDeckIndices().length > 0) {
      if (!confirm('Leave Build mode? Your deck is saved and will be here when you return.')) return;
    }
    document.getElementById('modeSelect').value = 'explore';
    MM.setMode('explore');
  }

  function handleEscape() {
    closeBuildMode();
  }

  function changeFormat(format) {
    if (!deckState) return;
    deckState.format = format;
    deckState.distribution = getDistDefault(format);
    deckState.recommendations = [];
    deckState.landSlots = [];
    if (format !== 'commander') {
      deckState.commander = null;
    }
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function clearSeeds() {
    if (!deckState) return;
    deckState.seeds = [];
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function resetDistribution() {
    if (!deckState) return;
    deckState.distribution = getDistDefault(deckState.format);
    renderDeckPanel();
    saveDeckState();
  }

  function setDistribution(type, value) {
    if (!deckState) return;
    deckState.distribution[type] = value;
    renderDeckPanel();
    saveDeckState();
  }

  function resetDeck() {
    if (!confirm('Reset deck? This will remove all cards.')) return;
    deckState.seeds = [];
    deckState.accepted = [];
    deckState.rejected = new Set();
    deckState.recommendations = [];
    deckState.landSlots = [];
    deckState.commander = null;
    deckState.distribution = getDistDefault(deckState.format);
    deckState._maxEdhrecRank = null;
    computeDeckColorIdentity();
    renderDeckPanel();
    MM.render();
    saveDeckState();
  }

  function clearLands() {
    if (!deckState) return;
    deckState.landSlots = [];
    generateManaBase();
  }

  // ── Expand/Collapse Recommendation Cards ──

  function toggleRecExpand(btn) {
    const recItem = btn.closest('.rec-item');
    if (!recItem) return;
    recItem.classList.toggle('expanded');
    btn.innerHTML = recItem.classList.contains('expanded') ? '&#9650;' : '&#9660;';
  }

  function isInDeck(idx) {
    if (!deckState) return false;
    return deckState.seeds.includes(idx) || deckState.accepted.includes(idx) ||
           deckState.landSlots.includes(idx) || deckState.commander === idx;
  }

  // ── Expose API ──
  window.DeckBuilder = {
    enter: enterBuildMode,
    exit: exitBuildMode,
    close: closeBuildMode,
    handleEscape,
    addSeed,
    removeSeed,
    clearSeeds,
    setCommander,
    clearCommander,
    changeFormat,
    resetDistribution,
    setDistribution,
    generate: generateRecommendations,
    accept: acceptRecommendation,
    reject: rejectRecommendation,
    acceptAll: acceptAllRecommendations,
    removeAccepted,
    generateManaBase: generateManaBase,
    clearLands,
    exportDeck,
    resetDeck,
    toggleRecExpand,
    getOverlayTraces,
    getDimmedIndices,
    isInDeck,
  };
})();
