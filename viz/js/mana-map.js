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
  // How near a click must land to a region label to count as clicking it. Labels are
  // 11–16px text, so this is roughly "on the word or just beside it".
  const REGION_CLICK_RADIUS_PX = 44;

  // Phase 2 of the migration: ?renderer=canvas draws the map with viz/js/render/canvas.js
  // instead of Plotly. Both paths are live so they can be compared on identical data —
  // the layer format IS the trace format, so render() builds one structure either way.
  const USE_CANVAS = new URLSearchParams(window.location.search).get('renderer') === 'canvas';
  let mapCanvas = null;

  let allData = [];
  let activeSupertypes = new Set(SUPERTYPES);
  let currentColorBy = 'color';
  let searchTerm = '';
  let searchTimeout = null;
  let plotInitialized = false;
  let similarTrace = null;
  let currentMode = 'explore';
  let embeddings = null; // Float32Array, loaded lazily for Find Similar
  const EMBED_DIM = 128; // mirrors FINAL_EMBEDDING_DIM in config.py
  let currentMap = 'default'; // 'default' or 'ability'
  const projectionCache = {}; // { default: [...], ability: [...] }
  const embeddingsCache = {}; // { default: Float32Array, ability: Float32Array }
  // All data files the viz fetches, relative to viz/index.html. The server
  // must be rooted at the repo top so '../data/' resolves (GitHub Pages layout).
  const DATA_BASE = '../data/';
  // Bump when a data artifact's SCHEMA changes — a new key, a renamed field, a changed
  // shape. Not needed for content refreshes (a re-run pipeline with the same fields),
  // where serving a slightly stale copy is harmless.
  //
  // Learned the hard way: `membership` was added to regions_*.json and every browser
  // that had ever loaded the map kept serving its cached copy, so drill-by-region found
  // no membership and disabled itself. It failed politely, which is exactly what makes
  // this class of bug expensive — the code was right and the bytes were old.
  const DATA_VERSION = 2;
  const v = url => url + '?v=' + DATA_VERSION;
  const DATA = {
    projection: v(DATA_BASE + 'projection_2d.json'),
    projectionAbility: v(DATA_BASE + 'projection_2d_ability.json'),
    embeddings: v(DATA_BASE + 'embeddings.bin'),
    embeddingsAbility: v(DATA_BASE + 'embeddings_ability.bin'),
    regionsDefault: v(DATA_BASE + 'regions_default.json'),
    regionsAbility: v(DATA_BASE + 'regions_ability.json'),
    obsolescence: v(DATA_BASE + 'obsolescence_index.json'),
    synergyGraph: v(DATA_BASE + 'synergy_graph.json'),
    comboGraph: v(DATA_BASE + 'combo_graph.json'),
  };
  const MAP_CONFIGS = {
    default: { projection: DATA.projection, embeddings: DATA.embeddings, regions: DATA.regionsDefault },
    ability: { projection: DATA.projectionAbility, embeddings: DATA.embeddingsAbility, regions: DATA.regionsAbility },
  };

  // ── Region/Topo state ──
  let regionDataCache = {};
  let showContours = false;
  let showRegionLabels = true;
  let regionDebounceTimer = null;

  // ── Multi-select state ──
  const MAX_SELECTED = 8;
  let selectedCards = [];   // Array of { idx, data }, max 8
  let topCardIndex = 0;     // Which card is "on top" in the viewer

  // Browse mode: a selection too big for the accordion. Holds the WHOLE set — no cap —
  // because only the card you are looking at is ever fetched, so the cost is one Scryfall
  // request per arrow press rather than one per card in the box.
  //
  // It exists because the old handler truncated a box-select to the first 8 points
  // `plotly_selected` happened to return, and that order is grouped by trace: colour
  // groups in palette order, then cards.csv row order within each. Box a mixed cluster
  // and you got eight green cards in Scryfall dump order — not a sample of your
  // selection, an artifact of how the traces were built.
  let browseSet = null;     // { indices: [...ordered], pos, label }

  function getSelectedCard() {
    return selectedCards[topCardIndex]?.data ?? null;
  }

  // ── Helpers ──

  function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // Currently uncalled by design. Every trace on this plot sets `hoverinfo: 'none'`,
  // and feeding this to `trace.text` anyway cost ~34,000 escHtml calls per render for
  // strings nothing displayed. Kept (and exported) for when hover is turned on — but
  // call it from the hover callback for the one point under the cursor, never in bulk.
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
    // Picking a single card is an exit from browse mode: you have stopped surveying a
    // set and started looking at one thing. Keeping both would leave two different
    // "current card" markers on the plot.
    browseSet = null;

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
    browseSet = null;
    closeViewerPanel();
    clearSimilarTrace();
    updateSelectionHighlight();
  }

  function bringToTop(stackIndex) {
    if (stackIndex < 0 || stackIndex >= selectedCards.length) return;
    topCardIndex = stackIndex;
    updateViewerPanel();   // reveals the opened row
    updateSelectionHighlight();
  }

  // ── Viewer Panel ──

  // ── Browse mode ──

  // Order a selection by distance from its own centroid in the 128-d embedding space,
  // furthest first — so you start on the least typical card in the box and walk inward
  // to the most representative. Cosine, because the rows are L2-normalised at export, so
  // the dot product IS the cosine and the centroid only needs renormalising once.
  //
  // 128-d rather than the 2D positions: screen distance is the projection's compromise,
  // and the whole point of an ordering is to say something the picture does not already.
  function orderByCentroidDistance(rows) {
    if (!embeddings) return rows.slice();
    const dim = EMBED_DIM;
    const centroid = new Float64Array(dim);
    for (const r of rows) {
      const o = r * dim;
      for (let i = 0; i < dim; i++) centroid[i] += embeddings[o + i];
    }
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += centroid[i] * centroid[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i++) centroid[i] /= norm;

    return rows
      .map(r => {
        const o = r * dim;
        let dot = 0;
        for (let i = 0; i < dim; i++) dot += embeddings[o + i] * centroid[i];
        return { r, d: 1 - dot };
      })
      .sort((a, b) => b.d - a.d)
      .map(x => x.r);
  }

  // Walking outward from one card. Reuses `browseSet` wholesale — the counter, the arrows,
  // `moveBrowseMarker`'s single-restyle fast path and the image preloader all come free —
  // and adds one field, `anchor`, so the panel can say whose neighbourhood you are in.
  //
  // Ordering is NEAREST-first, the opposite of a plain browse (furthest-from-centroid).
  // Both are defensible and they mean opposite things, so the panel states which is which.
  const NEIGHBOURHOOD_K = 24;

  async function enterNeighbourhood(row, initialStep) {
    if (!allData[row]) return;
    setStatus('Finding neighbours of ' + allData[row].n + '…');
    const near = await nearestTo(row, NEIGHBOURHOOD_K, {});
    if (!near.length) { setStatus('Embeddings unavailable — cannot walk the neighbourhood.'); return; }

    browseSet = {
      indices: [row].concat(near.map(x => x.i)),
      sims: [1].concat(near.map(x => x.sim)),
      pos: 0,
      label: allData[row].n,
      anchor: row,
    };
    if (initialStep) {
      const len = browseSet.indices.length;
      browseSet.pos = ((initialStep % len) + len) % len;
    }
    selectedCards = [];                    // browse and the 8-stack never coexist
    topCardIndex = 0;
    updateViewerPanel();
    updateSelectionHighlight();
    setStatus(allData[row].n + ' — ' + near.length + ' nearest, ← → to walk them, Enter to re-anchor');
  }

  async function enterBrowse(rowIndices, label) {
    const rows = Array.from(new Set(rowIndices)).filter(i => allData[i]);
    if (rows.length === 0) return;
    setStatus(`Ordering ${rows.length.toLocaleString()} cards…`);
    await loadEmbeddings();          // no-op after the first call
    browseSet = { indices: orderByCentroidDistance(rows), pos: 0, label: label || 'Selection',
                  anchor: null, sims: null };
    selectedCards = [];              // browse replaces the 8-card stack, never coexists
    topCardIndex = 0;
    updateViewerPanel();
    updateSelectionHighlight();
    setStatus(`${rows.length.toLocaleString()} cards — ordered furthest to nearest from the selection's centre`);
  }

  function browseCard() {
    if (!browseSet) return null;
    return allData[browseSet.indices[browseSet.pos]];
  }

  // ── Hover card ──────────────────────────────────────────────────────────
  //
  // A floating image at the cursor. Verified in the browser before building it:
  // `plotly_hover` DOES fire on traces with `hoverinfo: 'none'` — 'none' suppresses the
  // label, 'skip' suppresses the event — so this needs no `text` arrays and reintroduces
  // none of the per-point work that made Plotly's own hover cost 37 ms a render.
  //
  // The magazine's card preview (design.py `.card-pop`) is pure CSS, anchored to a static
  // inline element. A point in a WebGL scatter is not an element, so only the look
  // transfers, not the mechanism.
  const HOVER_DELAY_MS = 180;
  let hoverTimer = null;
  let hoverRow = null;
  let popupEl = null;

  function ensurePopup() {
    if (popupEl) return popupEl;
    popupEl = document.createElement('div');
    popupEl.className = 'card-popup';
    popupEl.style.display = 'none';
    document.getElementById('plot').appendChild(popupEl);
    return popupEl;
  }

  // clientX/clientY, because the two callers measure in different spaces: Plotly hands back
  // an event on the graph div, the canvas hands back a raw MouseEvent.
  function showCardPopup(row, clientX, clientY) {
    const d = allData[row];
    if (!d) return;
    clearTimeout(hoverTimer);
    if (hoverRow === row && popupEl && popupEl.style.display !== 'none') {
      positionPopup(clientX, clientY);
      return;
    }
    hoverTimer = setTimeout(function () {
      hoverRow = row;
      const el = ensurePopup();
      el.innerHTML =
        '<img src="' + cardImageUrl(d.n) + '" alt="' + escHtml(d.n) + '"' +
        ' onerror="this.onerror=null;this.parentElement.classList.add(\'card-popup-failed\');' +
        'this.parentElement.textContent=' + JSON.stringify(d.n).replace(/"/g, '&quot;') + '">';
      el.style.display = 'block';
      positionPopup(clientX, clientY);
    }, HOVER_DELAY_MS);
  }

  // Flip rather than clip. The panel side is where the cursor usually is, so a popup that
  // always opened right would spend most of its life half off-screen.
  function positionPopup(clientX, clientY) {
    if (!popupEl) return;
    const host = document.getElementById('plot');
    const r = host.getBoundingClientRect();
    const w = popupEl.offsetWidth || 230;
    const h = popupEl.offsetHeight || 320;
    let x = clientX - r.left + 18;
    let y = clientY - r.top - h / 2;
    if (x + w > r.width - 8) x = clientX - r.left - w - 18;
    if (x < 8) x = 8;
    if (y < 8) y = 8;
    if (y + h > r.height - 8) y = Math.max(8, r.height - h - 8);
    popupEl.style.left = Math.round(x) + 'px';
    popupEl.style.top = Math.round(y) + 'px';
  }

  function hideCardPopup() {
    clearTimeout(hoverTimer);
    hoverRow = null;
    if (popupEl) popupEl.style.display = 'none';
  }

  function cardImageUrl(name) {
    return 'https://api.scryfall.com/cards/named?exact='
      + encodeURIComponent(name) + '&format=image&version=normal';
  }

  // Warm the browser cache for the cards either side of the open one. Each image is a
  // Scryfall round-trip, so without this every arrow press shows a beat of empty grey
  // before the card appears — which is most of what made the old panel feel slow to
  // browse. Neighbours only: preloading all eight would be eight requests for the seven
  // the reader may never look at.
  function preloadNeighbourImages() {
    const n = browseSet ? browseSet.indices.length : selectedCards.length;
    if (n < 2) return;
    const at = browseSet ? browseSet.pos : topCardIndex;
    for (const delta of [-1, 1]) {
      const i = ((at + delta) % n + n) % n;
      const d = browseSet ? allData[browseSet.indices[i]] : selectedCards[i].data;
      if (d) new Image().src = cardImageUrl(d.n);
    }
  }

  function buildCardDetailHtml(d) {
    let html = '';

    // No `loading="lazy"`: the only card image we ever render is the open one, and it is
    // scrolled into view the moment it appears — deferring it just adds a beat of grey.
    html += '<div class="detail-card-image">';
    html += '<img src="' + cardImageUrl(d.n) + '" alt="' + escHtml(d.n) + '"';
    html += ' onerror="this.onerror=null;this.parentElement.style.minHeight=\'auto\';this.parentElement.innerHTML=\'<div class=\\\'detail-image-fallback\\\'>Image not available</div>\'">';
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
    if (browseSet) { renderBrowsePanel(); return; }
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
      // The arrows live in the sticky header so they stay reachable no matter how far
      // down the list you have scrolled. Same action as ← / →.
      html += '<span class="viewer-nav">';
      html += '<button class="viewer-arrow" onclick="MM.cyclePrev()" title="Previous (←)">‹</button>';
      html += '<span class="viewer-count">' + (topCardIndex + 1) + '/' + selectedCards.length + '</span>';
      html += '<button class="viewer-arrow" onclick="MM.cycleNext()" title="Next (→)">›</button>';
      html += '</span>';
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

    // One card: no list to navigate, so the detail is the panel.
    //
    // More than one: the LIST is the structure and the card opens inside the row you
    // clicked. The old layout put the detail on top and the list underneath, which meant
    // choosing a different card scrolled you away from the thing you were choosing, and
    // then you scrolled back to look at it. The accordion keeps the point of interaction
    // and the thing it reveals in the same place.
    if (selectedCards.length > 1) {
      html += '<div class="accordion">';
      for (let i = 0; i < selectedCards.length; i++) {
        const isActive = (i === topCardIndex);
        const card = selectedCards[i];
        const cd = card.data;
        html += '<div class="acc-row' + (isActive ? ' active' : '') + '">';
        html += '<div class="acc-head" onclick="MM.bringToTop(' + i + ')">';
        html += '<span class="acc-caret">' + (isActive ? '\u25be' : '\u25b8') + '</span>';
        html += '<span class="acc-name">' + escHtml(cd.n) + '</span>';
        html += '<span class="acc-mana">' + renderManaSymbols(cd.mc) + '</span>';
        if (cd.p != null && cd.th != null) html += '<span class="acc-stats">' + escHtml(cd.p) + '/' + escHtml(cd.th) + '</span>';
        html += '<span class="acc-type">' + escHtml(cd.s) + '</span>';
        html += '<button class="acc-remove" onclick="event.stopPropagation(); MM.removeFromSelection(' + card.idx + ')" title="Remove">\u00d7</button>';
        html += '</div>';
        if (isActive) {
          html += '<div class="acc-body">' + buildCardDetailHtml(cd) + '</div>';
        }
        html += '</div>';
      }
      html += '</div>';
      html += '<div class="keyboard-hint">\u2190 \u2192 navigate \u00b7 1-8 jump \u00b7 Del remove \u00b7 Esc clear all \u00b7 / search</div>';
    } else {
      html += buildCardDetailHtml(d);
      html += '<div class="keyboard-hint">Shift+click to multi-select \u00b7 Esc clear \u00b7 / search</div>';
    }

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

    inner.innerHTML = html;
    panel.classList.add('open');
    // Reveal whichever row is open, on every path that changes it — clicking a row,
    // the arrows, the arrow keys, a number key, removing a card, or selecting a new one
    // from the map. This function is only called when the selection actually changes,
    // so it never fights a scroll the reader started themselves.
    scrollActiveRowIntoView();
    preloadNeighbourImages();
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  // No list — a list of 400 names is not navigation, it is a wall. The arrows are the
  // whole interface, the plot shows you where you are, and the order carries the meaning
  // a list would have had to.
  function renderBrowsePanel() {
    const panel = document.getElementById('detailPanel');
    const inner = document.getElementById('detailInner');
    const d = browseCard();
    if (!d) { closeViewerPanel(); return; }
    const n = browseSet.indices.length;

    let html = '<div class="viewer-header">';
    html += '<h2>' + escHtml(d.n) + '</h2>';
    html += '<span class="viewer-nav">';
    html += '<button class="viewer-arrow" onclick="MM.cyclePrev()" title="Previous (←)">‹</button>';
    html += '<span class="viewer-count">' + (browseSet.pos + 1) + ' / ' + n.toLocaleString() + '</span>';
    html += '<button class="viewer-arrow" onclick="MM.cycleNext()" title="Next (→)">›</button>';
    html += '</span>';
    if (browseSet.anchor != null && browseSet.pos !== 0) {
      html += '<div class="browse-anchor">near <strong>' +
        escHtml(allData[browseSet.anchor].n) + '</strong></div>';
    }
    html += '<button class="detail-close" onclick="MM.closeDetail()" title="Close (ESC)">×</button>';
    html += '<div class="viewer-quickstats">';
    if (d.mc) html += renderManaSymbols(d.mc);
    if (d.p != null && d.th != null) {
      html += '<span class="stat-divider">·</span><strong>' + escHtml(d.p) + '/' + escHtml(d.th) + '</strong>';
    }
    if (d.r) {
      const rc = ['mythic', 'rare', 'uncommon', 'common'].includes(d.r) ? d.r : '';
      html += '<span class="stat-divider">·</span><span class="rarity-pill ' + rc + '">' + escHtml(d.r) + '</span>';
    }
    html += '</div></div>';

    // Say what the order is. An unexplained sequence through 400 cards is just a shuffle
    // with extra steps, and the ordering is the only thing making this browsable.
    const nb = browseSet.anchor != null;
    html += '<div class="browse-order">';
    html += '<span class="browse-order-bar"><span style="width:' +
      ((browseSet.pos / Math.max(n - 1, 1)) * 100).toFixed(1) + '%"></span></span>';
    html += '<span class="browse-order-label">' + (nb
      ? (browseSet.pos === 0
          ? 'the anchor · ← → walks its ' + (n - 1) + ' nearest · Enter re-anchors here'
          : 'nearest → furthest from ' + escHtml(allData[browseSet.anchor].n) +
            (browseSet.sims ? ' · cosine ' + browseSet.sims[browseSet.pos].toFixed(3) : '') +
            ' · Enter re-anchors here')
      : 'least typical → most typical · 128-dim distance from the selection’s centre') +
      '</span>';
    html += '</div>';

    html += buildCardDetailHtml(d);
    html += '<div class="keyboard-hint">← → browse · Esc clear · click a point to leave browse mode</div>';

    inner.innerHTML = html;
    panel.classList.add('open');
    inner.scrollTop = 0;
    preloadNeighbourImages();
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  // Put the open row's header just under the sticky masthead, so the card it just
  // revealed is on screen. Without this the accordion still scrolls you away from your
  // own click once the list is longer than the panel.
  function scrollActiveRowIntoView() {
    const inner = document.getElementById('detailInner');
    if (!inner) return;
    const row = inner.querySelector('.acc-row.active');
    if (!row) return;
    const header = inner.querySelector('.viewer-header');
    const offset = header ? header.offsetHeight : 0;
    inner.scrollTop = Math.max(0, row.offsetTop - offset - 8);
  }

  // Shared by the header arrows and the arrow keys so the two can never disagree.
  // Wraps in both directions \u2014 with at most 8 cards, running off the end and stopping
  // is more annoying than looping.
  function cycleSelection(delta) {
    if (browseSet) {
      const n = browseSet.indices.length;
      if (n < 2) return;
      browseSet.pos = ((browseSet.pos + delta) % n + n) % n;
      updateViewerPanel();
      // Fast path: nudge the marker. Falls back to a full rebuild only if the trace is
      // missing (first render, or a mode change tore it down).
      if (!moveBrowseMarker()) updateSelectionHighlight();
      return;
    }
    // One card selected: arrows used to be a no-op. Seed its neighbourhood and step into
    // it in the direction pressed, so the first press already moves.
    if (selectedCards.length === 1) { enterNeighbourhood(selectedCards[0].idx, delta); return; }
    if (selectedCards.length < 2) return;
    const n = selectedCards.length;
    bringToTop(((topCardIndex + delta) % n + n) % n);
  }

  function closeViewerPanel() {
    document.getElementById('detailPanel').classList.remove('open');
    clearSimilarTrace();
    setTimeout(() => Plotly.Plots.resize('plot'), 260);
  }

  // ── Selection Highlight on Plot ──

  // Where a card is depends on which layout is showing. Drilling replaces the coordinate
  // system, so a highlight drawn at `allData[i].x` while a local layout is on screen is a
  // gold ring pointing at nothing — the exact ambiguity the drill breadcrumb exists to
  // prevent. Returns null for a card with no position in the current system; callers must
  // drop it rather than falling back to a world coordinate.
  function cardPosition(idx) {
    const drilling = typeof window.Drill !== 'undefined' && window.Drill.isActive();
    if (!drilling) return [allData[idx].x, allData[idx].y];
    return window.Drill.localPosition(idx);
  }

  // Move only the marker. Rebuilding the whole highlight on every arrow press meant a
  // deleteTraces + addTraces of the entire selection — 197 ms per step on a 3,434-card
  // browse, which is a visible stutter on a keypress. One restyle of a single-point
  // trace instead. Returns false if the trace is not there, so the caller can fall back
  // to a full rebuild.
  function moveBrowseMarker() {
    const gd = document.getElementById('plot');
    if (!gd || !gd.data || !browseSet) return false;
    let ti = -1;
    for (let i = 0; i < gd.data.length; i++) {
      if (gd.data[i]._isBrowseCurrent) { ti = i; break; }
    }
    if (ti === -1) return false;
    const cur = browseSet.indices[browseSet.pos];
    const p = cardPosition(cur);
    if (!p) return false;
    Plotly.restyle('plot', { x: [[p[0]]], y: [[p[1]]], customdata: [[cur]] }, [ti]);
    return true;
  }

  // Identity of whatever the current _isSelection traces are drawing. A browse selection
  // can be tens of thousands of points, and render() calls this at the end of every pass
  // — so without a check, panning, filtering, toggling Topo or opening a panel each did a
  // deleteTraces + addTraces of the whole set. Nothing about the set changed; only the
  // marker moves, and moveBrowseMarker() handles that in one restyle.
  let _highlightKey = null;

  function browseHighlightKey() {
    if (!browseSet) return null;
    const ix = browseSet.indices;
    const drilling = typeof window.Drill !== 'undefined' && window.Drill.isActive();
    return 'b:' + ix.length + ':' + ix[0] + ':' + ix[ix.length - 1] +
           ':' + (browseSet.anchor == null ? '-' : browseSet.anchor) +
           ':' + (drilling ? 'local' : 'world');
  }

  // Pure: build the highlight traces without touching the plot. render() folds these
  // into its single Plotly.react, so a re-render no longer wipes them and then adds them
  // back — which, with a 15,000-card browse selection, was a full trace rebuild on every
  // pan, filter, Topo toggle and panel open.
  function buildSelectionTraces() {
    if (selectedCards.length === 0 && !browseSet) return [];

    const posOf = cardPosition;

    // Browse mode paints the whole set small and the card you are on large with a white
    // ring, so the arrows have a visible position on the map. No animation: one restyle
    // per press, nothing to tear down, and it reads at any zoom.
    if (browseSet) {
      const rows = browseSet.indices.filter(i => posOf(i));
      const cur = browseSet.indices[browseSet.pos];
      const curPos = posOf(cur);
      const traces = [];
      if (rows.length) {
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: 'Selection (' + browseSet.indices.length.toLocaleString() + ')',
          x: rows.map(i => posOf(i)[0]),
          y: rows.map(i => posOf(i)[1]),
          customdata: rows.slice(),
          hoverinfo: 'none',
          marker: { size: 5, opacity: 0.85, color: '#8B7730' },
          _isSelection: true,
        });
      }
      // The anchor keeps a distinct marker from the card you have walked to — otherwise
      // there is nothing on the map saying where the neighbourhood is centred.
      if (browseSet.anchor != null && browseSet.anchor !== cur) {
        const ap = posOf(browseSet.anchor);
        if (ap) {
          traces.push({
            type: 'scattergl',
            mode: 'markers',
            name: 'Anchor',
            x: [ap[0]],
            y: [ap[1]],
            customdata: [browseSet.anchor],
            hoverinfo: 'none',
            marker: { size: 14, opacity: 1, color: 'rgba(0,0,0,0)', symbol: 'circle',
                      line: { color: '#4A7BFF', width: 2.5 } },
            _isSelection: true,
          });
        }
      }
      if (curPos) {
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: 'Browsing',
          x: [curPos[0]],
          y: [curPos[1]],
          customdata: [cur],
          hoverinfo: 'none',
          marker: { size: 16, opacity: 1, color: '#c4a747', line: { color: '#fff', width: 2.5 } },
          _isSelection: true,
          _isBrowseCurrent: true,
        });
      }
      return traces;
    }

    // Build selection highlight trace
    const topIdx = selectedCards[topCardIndex]?.idx;
    const otherCards = selectedCards
      .filter((_, i) => i !== topCardIndex)
      .filter(c => posOf(c.idx));

    const traces = [];

    // Other selected cards (dimmer gold)
    if (otherCards.length > 0) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Selected',
        x: otherCards.map(c => posOf(c.idx)[0]),
        y: otherCards.map(c => posOf(c.idx)[1]),
        customdata: otherCards.map(c => c.idx),
        hoverinfo: 'none',
        marker: { size: 12, opacity: 1, color: '#8B7730', symbol: 'circle', line: { color: '#fff', width: 1.5 } },
        _isSelection: true,
      });
    }

    // Top card (bright gold)
    const topPos = topIdx != null ? posOf(topIdx) : null;
    if (topPos) {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Active',
        x: [topPos[0]],
        y: [topPos[1]],
        customdata: [topIdx],
        hoverinfo: 'none',
        marker: { size: 12, opacity: 1, color: '#c4a747', symbol: 'circle', line: { color: '#fff', width: 2 } },
        _isSelection: true,
      });
    }

    return traces;
  }

  // Out-of-render updates (selecting a card, removing one, cycling the stack). Uses the
  // marker fast path when only the browse position moved, otherwise swaps the traces in
  // place — still much cheaper than a full render() for the <=8 case.
  function updateSelectionHighlight() {
    const plotDiv = document.getElementById('plot');
    if (!plotDiv || !plotDiv.data) return;

    if (browseSet) {
      const key = browseHighlightKey();
      if (key === _highlightKey && plotDiv.data.some(t => t._isSelection) && moveBrowseMarker()) return;
      _highlightKey = key;
    } else {
      _highlightKey = null;
    }

    const toDelete = [];
    for (let i = plotDiv.data.length - 1; i >= 0; i--) {
      if (plotDiv.data[i]._isSelection) toDelete.push(i);
    }
    if (toDelete.length) Plotly.deleteTraces('plot', toDelete);

    const traces = buildSelectionTraces();
    if (traces.length) Plotly.addTraces('plot', traces);
  }

  // ── Detail Panel (legacy wrapper for backward compat) ──

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

  // THE k-nearest primitive. There were two before this — `cosineSimilarity` plus the sort
  // inside `findSimilarCards`, and `force.js:nearestInCorpus` — and the neighbourhood walk
  // would have been a third. Rows are L2-normalised at export, so the dot product IS the
  // cosine and no norms are needed. Returns `{i, sim}` nearest-first.
  //
  // `respectFilters` defaults to true: if you have hidden Lands, a neighbourhood should not
  // walk you into one. `force.js` passes false, because a graph you are branching through
  // should not silently change shape when a toolbar toggle flips.
  async function nearestTo(row, k, opts) {
    const o = opts || {};
    if (!(await loadEmbeddings())) return [];
    const dim = EMBED_DIM;
    const base = row * dim;
    const exclude = o.exclude || null;
    const respectFilters = o.respectFilters !== false;
    const best = [];                       // ascending by sim; best[0] is the weakest kept
    for (let j = 0; j < allData.length; j++) {
      if (j === row) continue;
      if (exclude && exclude.has(j)) continue;
      if (respectFilters && !activeSupertypes.has(allData[j].s)) continue;
      const oj = j * dim;
      let dot = 0;
      for (let i = 0; i < dim; i++) dot += embeddings[base + i] * embeddings[oj + i];
      if (best.length < k) {
        best.push({ i: j, sim: dot });
        if (best.length === k) best.sort((a, b) => a.sim - b.sim);
      } else if (dot > best[0].sim) {
        best[0] = { i: j, sim: dot };
        best.sort((a, b) => a.sim - b.sim);
      }
    }
    return best.sort((a, b) => b.sim - a.sim);
  }

  function cosineSimilarity(idxA, idxB) {
    // Rows of embeddings.bin are L2-normalized at export, so the cosine IS the
    // dot product — computing the norms was 3x the float work for a constant 1.
    const offA = idxA * EMBED_DIM;
    const offB = idxB * EMBED_DIM;
    let dot = 0;
    for (let j = 0; j < EMBED_DIM; j++) {
      dot += embeddings[offA + j] * embeddings[offB + j];
    }
    return dot;
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

    // A neighbourhood scattered as 21 dots across a 34K-point map shows you where its
    // members live but nothing about how they relate. Offer the drill, which lays them
    // out against each other instead.
    if (typeof window.Drill !== 'undefined') {
      window.Drill.offer([refIdx].concat(nearest.map(m => m.i)), `Near ${ref.n}`);
    }
  }

  // ── Obsolescence Loading ──

  async function loadObsolescenceIndex() {
    if (obsolescenceIndex) return true;
    try {
      const r = await fetch(DATA.obsolescence);
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
    // Pre-load region data so it's ready when render() builds annotations
    if (showRegionLabels) await loadRegionData(mapName);
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
      const r = await fetch(DATA.synergyGraph);
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

    // Same offer as Find Similar, and more interesting here: synergy is rule-based and
    // complementary, so its partners are scattered by construction. Their embedding
    // layout says whether the rules found one coherent group or several.
    if (typeof window.Drill !== 'undefined') {
      window.Drill.offer([refIdx].concat(synPoints.map(p => p.idx)), `Synergies of ${ref.n}`);
    }
  }

  // ── Region loading and rendering ──

  async function loadRegionData(mapName) {
    if (regionDataCache[mapName]) return regionDataCache[mapName];
    const config = MAP_CONFIGS[mapName];
    if (!config || !config.regions) return null;
    try {
      const r = await fetch(config.regions);
      if (!r.ok) return null;
      const data = await r.json();
      regionDataCache[mapName] = data;
      return data;
    } catch (e) {
      return null;
    }
  }

  function buildContourTrace(filtered) {
    return {
      type: 'histogram2dcontour',
      x: filtered.map(d => d.x),
      y: filtered.map(d => d.y),
      ncontours: 15,
      showscale: false,
      hoverinfo: 'skip',
      colorscale: [
        [0, 'rgba(0,0,0,0)'],
        [0.2, 'rgba(90,60,140,0.08)'],
        [0.4, 'rgba(90,80,160,0.15)'],
        [0.6, 'rgba(100,90,180,0.22)'],
        [0.8, 'rgba(120,100,200,0.28)'],
        [1, 'rgba(140,120,220,0.35)'],
      ],
      contours: { coloring: 'heatmap' },
      line: { width: 0.5, color: 'rgba(140,120,220,0.25)' },
      _isContour: true,
    };
  }

  function getRegionAnnotations(visibleSpan) {
    const regionData = regionDataCache[currentMap];
    if (!regionData || !showRegionLabels) return [];

    const annotations = [];
    for (const region of regionData.regions) {
      let opacity = 0;
      let fontSize = 11;

      if (region.level === 0) {
        if (visibleSpan > 25) opacity = 1;
        else if (visibleSpan > 15) opacity = (visibleSpan - 15) / 10;
        fontSize = 16;
      } else {
        if (region.span < visibleSpan * 0.05) continue;
        if (visibleSpan < 20) opacity = 1;
        else if (visibleSpan < 30) opacity = (30 - visibleSpan) / 10;
        fontSize = 11;
      }

      if (opacity <= 0) continue;

      annotations.push({
        x: region.cx,
        y: region.cy,
        text: region.level === 0 ? region.label : region.short,
        showarrow: false,
        xref: 'x',
        yref: 'y',
        font: {
          family: 'system-ui, -apple-system, sans-serif',
          size: fontSize,
          color: region.level === 0
            ? 'rgba(196,167,71,' + opacity.toFixed(2) + ')'
            : 'rgba(200,200,200,' + opacity.toFixed(2) + ')',
        },
      });
    }
    return annotations;
  }

  // Called only from plotly_relayout (zoom/pan) — uses Plotly.update with annotations only
  let _labelUpdateInFlight = false;
  function refreshLabelsOnZoom() {
    if (_labelUpdateInFlight) return;
    if (USE_CANVAS) {
      // Region labels become DOM in Phase 3. Until then the canvas path simply has none,
      // which is visible and honest rather than a half-drawn annotation layer.
      return;
    }
    const plotEl = document.getElementById('plot');
    if (!plotEl || !plotEl._fullLayout) return;
    const xRange = plotEl._fullLayout.xaxis.range;
    const visibleSpan = Math.abs(xRange[1] - xRange[0]);
    _labelUpdateInFlight = true;
    Plotly.relayout('plot', { annotations: getRegionAnnotations(visibleSpan) })
      .finally(() => { _labelUpdateInFlight = false; });
  }

  // ── Load data ──
  fetch(MAP_CONFIGS.default.projection)
    .then(r => r.json())
    .then(data => {
      allData = data;
      projectionCache['default'] = data;
      initToggles();
      render();
      refreshDrillButton();
      setStatus(`${allData.length.toLocaleString()} cards loaded`);
      // ?deck=<slug> is the map's first inbound deep link — the dossier and the
      // magazine's Back Page both use it. Honour it by entering the Lens, not by
      // dropping the reader on an unfiltered map with a query string they can't see.
      const wantedDeck = new URLSearchParams(window.location.search).get('deck');
      if (wantedDeck) {
        document.getElementById('modeSelect').value = 'deck';
        setMode('deck');
      }
      // Load region data in background, then re-render with labels
      loadRegionData('default').then(data => {
        if (data) render();
      });
    })
    .catch(err => setStatus('Error loading data: ' + err.message));

  // ── Supertype toggle buttons ──
  // The Drill button used to say only "Drill ⤓". With no filters that meant "re-map all
  // 34,322 cards", which the cap then truncated to an arbitrary 2,000 — an incoherent
  // cross-section of the whole universe that flew in from everywhere and settled into a
  // multicoloured pile. The button now states the size of what it would drill and goes
  // inert when that is over the cap, so you can see whether pressing it will do anything.
  function refreshDrillButton() {
    const btn = document.getElementById('drillFiltered');
    if (!btn || typeof window.Drill === 'undefined') return;
    let n = 0;
    for (let i = 0; i < allData.length; i++) if (activeSupertypes.has(allData[i].s)) n++;
    const cap = window.Drill.MAX_DRILL;
    const tooMany = n > cap;
    btn.textContent = 'Drill ' + n.toLocaleString() + ' ⤓';
    btn.classList.toggle('is-disabled', tooMany);
    btn.title = tooMany
      ? n.toLocaleString() + ' cards is too many to re-map — filter below ' +
        cap.toLocaleString() + ', or box-select, or click a region label'
      : 'Re-map these ' + n.toLocaleString() + ' cards from their 128-dim embeddings';
  }

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
        refreshDrillButton();
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

  // ── Topo toggle handlers ──
  document.getElementById('toggleContours').addEventListener('click', function () {
    showContours = !showContours;
    this.classList.toggle('active', showContours);
    render();
  });

  document.getElementById('toggleLabels').addEventListener('click', function () {
    showRegionLabels = !showRegionLabels;
    this.classList.toggle('active', showRegionLabels);
    if (showRegionLabels && !regionDataCache[currentMap]) {
      loadRegionData(currentMap).then(() => render());
    } else {
      render();
    }
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
      // Drill wins the key: it is the deepest state on screen, and surfacing back up
      // one level is what Escape should mean while a local layout is showing.
      if (typeof window.Drill !== 'undefined' && window.Drill.isActive()) {
        window.Drill.back();
      } else if (currentMode === 'build') {
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

    // Arrows come FIRST and are gated on their own terms. They used to sit behind
    // `selectedCards.length === 0`, and browse mode sets `selectedCards = []` — so the
    // arrow KEYS were dead in browse mode and only the on-screen ‹ › buttons worked, while
    // the panel's own hint said "← → browse". `cycleSelection` guards its own bounds, so
    // the `> 1` conditions that also blocked the single-card case are gone too.
    const back = e.key === 'ArrowLeft' || e.key === 'ArrowUp';
    const fwd = e.key === 'ArrowRight' || e.key === 'ArrowDown';
    if (back || fwd) {
      if (!browseSet && selectedCards.length === 0) return;
      e.preventDefault();
      cycleSelection(back ? -1 : 1);
      return;
    }

    // Enter re-anchors the neighbourhood to whatever you have walked to.
    if (e.key === 'Enter' && browseSet && browseSet.anchor != null) {
      e.preventDefault();
      enterNeighbourhood(browseSet.indices[browseSet.pos]);
      return;
    }

    if (selectedCards.length === 0) return;

    if (e.key === 'Delete' || e.key === 'Backspace') {
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

  // Build and Deck Lens share one side panel (#deckPanel), so entering either must exit
  // the other. Explore keeps the detail panel; Build hides it because its own panel needs
  // the width, and the Lens keeps it because clicking a lit deck card to read it is the
  // whole interaction.
  function setMode(mode) {
    currentMode = mode;
    hideCardPopup();
    const detail = document.getElementById('detailPanel');

    if (mode !== 'build' && typeof window.DeckBuilder !== 'undefined') window.DeckBuilder.exit();
    if (mode !== 'deck' && typeof window.DeckMap !== 'undefined') window.DeckMap.exit();
    if (mode !== 'force' && typeof window.Force !== 'undefined') window.Force.exit();

    // The Walk replaces the plot rather than overlaying it: it is a different renderer
    // (canvas) drawing a different thing (a graph, not a projection), so the Plotly
    // surface is hidden outright while it runs.
    const plotEl = document.getElementById('plot');
    plotEl.classList.toggle('force-mode', mode === 'force');

    if (mode === 'build') {
      clearSelection();
      detail.style.display = 'none';
      if (typeof window.DeckBuilder !== 'undefined') window.DeckBuilder.enter();
    } else if (mode === 'deck') {
      detail.style.display = '';
      if (typeof window.DeckMap !== 'undefined') window.DeckMap.enter();
    } else if (mode === 'force') {
      detail.style.display = 'none';
      if (typeof window.Force !== 'undefined') window.Force.enter();
      return;                     // no Plotly render — the canvas owns the surface
    } else {
      detail.style.display = '';
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

    // Get overlay traces from whichever mode owns the side panel. Both implement the
    // same two-method contract; see docs/viz.md.
    let overlayTraces = [];
    let dimmedIndices = null;
    const overlay = currentMode === 'build' ? window.DeckBuilder
      : currentMode === 'deck' ? window.DeckMap
      : null;
    if (overlay) {
      overlayTraces = overlay.getOverlayTraces();
      dimmedIndices = overlay.getDimmedIndices();
    }

    // Drill is orthogonal to mode: it replaces the world's *coordinates* rather than
    // painting over them, so the 34K base traces are hidden outright while it is active.
    // Dimming would leave two coordinate systems on screen at once, which is exactly the
    // ambiguity the breadcrumb exists to prevent.
    const drilling = typeof window.Drill !== 'undefined' && window.Drill.hidesWorld();
    if (drilling) overlayTraces = overlayTraces.concat(window.Drill.getOverlayTraces());

    const filtered = allData.filter(d => activeSupertypes.has(d.s));

    // Contour trace (prepended before scatter so it renders beneath). While drilling it
    // re-bins over the local layout — histogram2dcontour auto-bins to whatever extent it
    // is handed, so levels are relative to the current selection and are NOT comparable
    // across drills.
    const contourTraces = [];
    const contourSource = drilling ? window.Drill.getContourSource() : filtered;
    if (showContours && contourSource && contourSource.length > 0) {
      contourTraces.push(buildContourTrace(contourSource));
    }

    // Group by category (iterate with index to avoid O(n) indexOf)
    // No `text` — every trace on this plot sets `hoverinfo: 'none'` and nothing reads
    // `trace.text`, so building hover strings here was ~34,000 calls into escHtml (four
    // chained global regexes each, three fields per card) on EVERY render, producing
    // ~275,000 regex operations whose output was thrown away. Measured at 37 ms of the
    // 90 ms render. If hover is ever turned on, add the text back deliberately — and
    // build it in the hover callback, not for all 34K points up front.
    const groups = {};
    for (let i = 0; i < allData.length; i++) {
      const d = allData[i];
      if (!activeSupertypes.has(d.s)) continue;
      const { key } = getCategoryInfo(d);
      if (!groups[key]) groups[key] = { x: [], y: [], customdata: [], key };
      groups[key].x.push(d.x);
      groups[key].y.push(d.y);
      groups[key].customdata.push(i);
    }

    const palette = currentColorBy === 'color' ? COLOR_PALETTE
      : currentColorBy === 'supertype' ? SUPERTYPE_PALETTE
      : RARITY_PALETTE;

    // Build traces with optional per-point opacity for dimming. `visible: false` keeps
    // the trace (and its legend entry order) while drilling instead of rebuilding the
    // whole plot on the way in and out.
    // Per-point opacity is expensive: a 34,000-entry array per group, plus Plotly's
    // per-point path through the WebGL renderer. Measured at ~100 ms of a 133 ms Deck
    // Lens render, and memoising the Set did not touch it because the Set was never the
    // cost.
    //
    // The Lens dims *everything* and redraws its 99 as overlay traces on top, so a scalar
    // opacity is equivalent and ~free. The deck builder dims a genuine subset (format
    // illegal, colour-identity violations) with nothing drawn over it, so it still needs
    // the per-point array — `dimsAll()` is how a mode says which it is.
    const dimsAll = !!(overlay && overlay.dimsAll && overlay.dimsAll());
    const traces = Object.values(groups).map(g => {
      let opacity;
      if (dimsAll) {
        opacity = 0.08;
      } else if (dimmedIndices) {
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
        customdata: g.customdata,
        hoverinfo: 'none',
        visible: drilling ? false : true,
        marker: { size: 3, opacity, color: palette[g.key] || '#666' },
      };
    });

    // Search highlight trace (index-tracking to avoid O(n²) indexOf). Suppressed while
    // drilling: it plots world coordinates, and a diamond at a world position on top of
    // a local layout would be pointing at nothing.
    if (searchTerm.length >= 2 && !drilling) {
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
          customdata: matches.map(m => m.i),
          hoverinfo: 'none',
          marker: { size: 8, opacity: 1, color: '#fff', symbol: 'diamond', line: { color: '#EA580C', width: 2 } },
        });
        const suffix = isOracleSearch ? ' (oracle text)' : '';
        setStatus(`${displayCount} result${matches.length === 1 ? '' : 's'} for "${searchTerm}"${suffix} \u2014 ${filtered.length.toLocaleString()} cards shown`);
      } else {
        setStatus(`No results for "${searchTerm}" \u2014 ${filtered.length.toLocaleString()} cards shown`);
      }
    } else if (drilling) {
      // The world count is a lie while drilling — those cards are not on screen, and
      // their positions would not mean the same thing if they were.
      const n = window.Drill.getContourSource().length;
      setStatus(`${n.toLocaleString()} cards · local layout from the 128-dim embeddings`);
    } else if (currentMode === 'explore') {
      setStatus(`${filtered.length.toLocaleString()} cards shown`);
    }

    // Add overlay traces from deck builder
    traces.push(...overlayTraces);
    // ...and the selection highlight, so one react draws everything. Previously react
    // replaced the trace list (dropping the highlight) and updateSelectionHighlight then
    // added it straight back — an extra addTraces of the whole selection per render.
    traces.push(...buildSelectionTraces());
    _highlightKey = browseHighlightKey();

    // Prepend contour traces
    const allTraces = [...contourTraces, ...traces];

    // Read the live camera before rebuilding the layout — for the region
    // annotations, and to put it back.
    //
    // `Plotly.react` replaces layout wholesale, so a layout with no explicit
    // range silently resets the viewport to autorange. That made filtering and
    // zooming mutually destructive: every supertype toggle, colour-by change,
    // search keystroke and deck-panel mutation threw the camera away, and this
    // very block read a span that the next statement destroyed. Measured before
    // the fix: zoom to a span of 20.5, call render(), get 116.6.
    //
    // Gated on `plotInitialized` so the two cases that *should* autorange still
    // do — the first render, and a map switch (`applyProjection` clears the flag
    // because the coordinates themselves changed).
    const plotEl = document.getElementById('plot');
    let visibleSpan = 70; // default: full extent, shows L0 labels
    let keepX = null;
    let keepY = null;
    if (USE_CANVAS && mapCanvas) {
      const cam = mapCanvas.getCamera();
      if (cam) visibleSpan = Math.abs(cam.x[1] - cam.x[0]);
    } else if (plotInitialized && plotEl && plotEl._fullLayout) {
      const xr = plotEl._fullLayout.xaxis.range;
      const yr = plotEl._fullLayout.yaxis.range;
      visibleSpan = Math.abs(xr[1] - xr[0]);
      keepX = xr.slice();
      keepY = yr.slice();
    }

    const layout = {
      paper_bgcolor: '#1a1a2e',
      plot_bgcolor: '#1a1a2e',
      font: { color: '#e0e0e0' },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: keepX ? { visible: false, scaleanchor: 'y', range: keepX }
        : { visible: false, scaleanchor: 'y' },
      yaxis: keepY ? { visible: false, range: keepY } : { visible: false },
      dragmode: shiftHeld ? 'select' : 'pan',
      legend: { bgcolor: 'rgba(22,33,62,0.85)', bordercolor: '#3a3a5a', borderwidth: 1, font: { size: 11 } },
      hovermode: 'closest',
      // Region labels are anchored to world centroids, so they are meaningless over a
      // local layout — and their absence is part of how a drill reads as somewhere else.
      annotations: drilling ? [] : getRegionAnnotations(visibleSpan),
    };

    const config = { scrollZoom: true, displayModeBar: false, responsive: true };

    if (USE_CANVAS) {
      if (!mapCanvas) {
        mapCanvas = window.MapCanvas.create().init(document.getElementById('plot'));
        mapCanvas.on('click', function (ev) {
          if (ev.shiftKey) {
            const at = selectedCards.findIndex(c => c.idx === ev.row);
            if (at !== -1) removeFromSelection(ev.row); else addToSelection(ev.row);
          } else {
            clearSelection();
            addToSelection(ev.row);
          }
        });
        mapCanvas.on('hover', function (ev) { showCardPopup(ev.row, ev.clientX, ev.clientY); });
        mapCanvas.on('unhover', hideCardPopup);
        // The label crossfade keys on the visible span, exactly as it did off
        // `_fullLayout.xaxis.range` — getCamera() reports in data units for that reason.
        mapCanvas.on('camera', function () {
          clearTimeout(regionDebounceTimer);
          regionDebounceTimer = setTimeout(refreshLabelsOnZoom, 150);
        });
        plotInitialized = true;
      }
      mapCanvas.setLayers(allTraces);
      return;
    }

    if (!plotInitialized) {
      Plotly.newPlot('plot', allTraces, layout, config);

      // Region labels are Plotly annotations, and annotations are not clickable — so
      // hit-test raw clicks against their anchors. Runs on the capture-free bubble
      // phase after plotly_click, and bails if a point was hit, so clicking a card
      // still selects the card.
      document.getElementById('plot').addEventListener('click', function (ev) {
        if (typeof window.Drill === 'undefined' || window.Drill.isActive()) return;
        if (currentMode === 'build') return;
        const gd = document.getElementById('plot');
        const fl = gd && gd._fullLayout;
        if (!fl || !fl.annotations || !fl.annotations.length) return;
        const regionData = regionDataCache[currentMap];
        if (!regionData) return;

        const rect = gd.getBoundingClientRect();
        const px = ev.clientX - rect.left;
        const py = ev.clientY - rect.top;

        let best = null;
        let bestDist = REGION_CLICK_RADIUS_PX;
        for (const ann of fl.annotations) {
          const ax = fl.xaxis.d2p(ann.x) + fl._size.l;
          const ay = fl.yaxis.d2p(ann.y) + fl._size.t;
          const dist = Math.hypot(ax - px, ay - py);
          if (dist < bestDist) { bestDist = dist; best = ann; }
        }
        if (!best) return;

        // Annotations carry the display text; L0 shows `label`, L1 shows `short`.
        const region = regionData.regions.find(
          r => (r.level === 0 ? r.label : r.short) === best.text);
        if (!region) return;
        ev.stopPropagation();
        window.Drill.enterRegion(region.id);
      });

      // Hover → card image. `hoverinfo: 'none'` keeps the label off; the event still fires.
      document.getElementById('plot').on('plotly_hover', function (eventData) {
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const pt = eventData.points[0];
        if (pt.customdata == null || !allData[pt.customdata]) return;
        const ev = eventData.event;
        if (ev) showCardPopup(pt.customdata, ev.clientX, ev.clientY);
      });
      document.getElementById('plot').on('plotly_unhover', hideCardPopup);

      // Zoom listener for region label crossfade
      document.getElementById('plot').on('plotly_relayout', function () {
        if (_labelUpdateInFlight) return; // ignore relayouts we caused
        clearTimeout(regionDebounceTimer);
        regionDebounceTimer = setTimeout(refreshLabelsOnZoom, 150);
      });

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

        // One pass over the points, then one destination. This used to build the 8-card
        // stack, render the whole panel and rebuild the plot highlight — and then, for a
        // big box, throw all of it away and do it again as browse. Two full renders per
        // selection, one of them purely wasted.
        const all = [];
        for (const pt of eventData.points) {
          if (pt.customdata != null && allData[pt.customdata]) all.push(pt.customdata);
        }
        if (all.length === 0) return;

        if (all.length > MAX_SELECTED) {
          // Too many for the accordion: take the WHOLE set into browse rather than an
          // arbitrary 8, and offer the same set to drill.
          enterBrowse(all, 'Selection');
          if (typeof window.Drill !== 'undefined') window.Drill.offer(all, 'Selection');
          return;
        }

        selectedCards = all.map(idx => ({ idx, data: allData[idx] }));
        topCardIndex = 0;
        updateViewerPanel();
        updateSelectionHighlight();
        setStatus(`Selected ${all.length} card${all.length === 1 ? '' : 's'}`);
      });

      // Keep the Plotly SVG in sync with the container: the side panels resize
      // #plot via CSS transitions, and one-shot timers can fire mid-transition
      // (or while large deck-builder fetches block the main thread), leaving a
      // stale full-width SVG painted over the open panel.
      if (window.ResizeObserver) {
        let resizeDebounce = null;
        new ResizeObserver(function () {
          clearTimeout(resizeDebounce);
          resizeDebounce = setTimeout(function () {
            const el = document.getElementById('plot');
            if (USE_CANVAS) { if (mapCanvas) mapCanvas.resize(); return; }
            if (el._fullLayout && Math.abs(el._fullLayout.width - el.getBoundingClientRect().width) > 1) {
              Plotly.Plots.resize('plot');
            }
          }, 120);
        }).observe(document.getElementById('plot'));
      }

      plotInitialized = true;
    } else {
      Plotly.react('plot', allTraces, layout, config);
      // react() can restore a stale cached width (e.g. full-window) after the
      // container shrank for the deck panel, painting the plot over the panel.
      const plotEl = document.getElementById('plot');
      if (plotEl._fullLayout && Math.abs(plotEl._fullLayout.width - plotEl.getBoundingClientRect().width) > 1) {
        Plotly.Plots.resize('plot');
      }
    }

  }

  function setStatus(msg) { document.getElementById('status').textContent = msg; }

  function selectByName(name) {
    const idx = allData.findIndex(d => d.n === name);
    if (idx !== -1) addToSelection(idx);
  }

  // ── Expose shared state/functions on window.MM ──
  // Only members with a live caller (deck-builder.js, generated onclick
  // handlers, or index.html) are exported — see docs/viz.md for the contract.
  window.MM = {
    get allData() { return allData; },
    get currentMap() { return currentMap; },
    escHtml,
    buildHoverTextMinimal,
    renderManaSymbols,
    closeDetail: clearSelection,
    removeFromSelection,
    bringToTop,
    cyclePrev: () => cycleSelection(-1),
    cycleNext: () => cycleSelection(1),
    enterBrowse,
    enterNeighbourhood,
    // The active canvas renderer, or null under Plotly. Phase 3 needs `dataToPixel` to
    // place region labels as DOM; the browser tests need it to aim a click at a card
    // rather than at a guessed pixel.
    get mapRenderer() { return mapCanvas; },
    nearestTo,
    buildCardDetailHtml,
    showCardPopup,
    hideCardPopup,
    get browseSet() { return browseSet; },
    // The row indices currently selected, whichever container holds them. The Walk seeds
    // from this so every existing way of picking cards feeds it for free.
    selectedRows() {
      if (browseSet) return browseSet.indices.slice();
      return selectedCards.map(c => c.idx);
    },
    selectByName,
    findSimilar: findSimilarCards,
    findSynergies: findSynergyCards,
    render,
    setStatus,
    setMode,
    MAP_CONFIGS,
    DATA,
    EMBED_DIM,
    get obsolescence() { return obsolescenceIndex; },
    // Shared big-data loaders: the deck builder awaits these instead of
    // re-downloading its own copies (embeddings 17.5 MB, synergy 27.8 MB).
    async getEmbeddings() {
      const ok = await loadEmbeddings();
      return ok ? embeddings : null;
    },
    async getSynergyGraph() {
      const ok = await loadSynergyGraph();
      return ok ? synergyGraph : null;
    },
    // ── Drill support ──
    // The colour a card would be painted under the current colour-by, so a local
    // layout stays readable against the world the reader just left.
    categoryColor(d) {
      const { key, palette } = getCategoryInfo(d);
      return palette[key] || '#666';
    },
    async getRegionData() { return loadRegionData(currentMap); },
    passesFilters(d) { return activeSupertypes.has(d.s); },
    filterLabel() {
      const on = Array.from(activeSupertypes);
      return on.length >= SUPERTYPES.length ? 'Everything' : on.join(' + ');
    },
  };
})();
