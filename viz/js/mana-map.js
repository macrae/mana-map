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

  // The map is drawn by viz/js/render/canvas.js. Plotly is gone: it was kept alongside
  // through the port so both could be compared on identical data, and the layer format
  // deliberately WAS the trace format so there would be no adapter to delete at the end.
  // There wasn't — `render()` still builds one structure, it just has one consumer now.
  let mapCanvas = null;

  let allData = [];
  let activeSupertypes = new Set(SUPERTYPES);
  let currentColorBy = 'color';
  let searchTerm = '';
  let searchTimeout = null;
  let plotInitialized = false;
  let currentMode = 'discover';
  let embeddings = null; // Float32Array, loaded lazily for Find Similar
  const EMBED_DIM = 128; // mirrors FINAL_EMBEDDING_DIM in config.py
  let currentMap = 'default'; // 'default' or 'ability'
  const projectionCache = {}; // { default: [...], ability: [...] }
  const embeddingsCache = {}; // { function: Float32Array } — one space, not one per map
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
  // Bumped to 3 when the embeddings were retrained. The rule used to be "bump on schema
  // change, not content refresh" — which is wrong for a change that alters what the bytes
  // MEAN. `embeddings_ability.bin` kept its exact shape and every value changed, so a
  // cached copy still parsed, still rendered, and silently answered "similar" out of the
  // old collapsed space. Verified in a browser: the page returned the pre-retrain
  // neighbours for Doubling Season while a cache-busted fetch of the same URL returned
  // the new ones. Bump whenever a consumer would draw a different conclusion from the
  // bytes, not only when the parser would.
  const DATA_VERSION = 4;
  const v = url => url + '?v=' + DATA_VERSION;
  // Exported because the deck manifest and per-deck artifacts are fetched by
  // build.js and discovery.js, which had NO cache-busting at all — adding a key to
  // `index.json` served the old copy and every verified line silently drew nothing.
  // Same class as the `membership` incident: a schema change, politely stale.
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
    // The discovery front door — small enough to land on before anything else arrives.
    vizIndex: v(DATA_BASE + 'viz_index.json'),
    neighbours: v(DATA_BASE + 'neighbours.bin'),
  };
  const MAP_CONFIGS = {
    default: { projection: DATA.projection, embeddings: DATA.embeddings, regions: DATA.regionsDefault },
    ability: { projection: DATA.projectionAbility, embeddings: DATA.embeddingsAbility, regions: DATA.regionsAbility },
  };

  // Similarity is NOT the displayed map. The default map is laid out by colour and type,
  // which is a good picture and a terrible answer to "what is like this card" — measured,
  // that space used 3.05 of its 128 dimensions and scored 0.044 recall@10 against known
  // functional equivalents, which is why Doubling Season's neighbours came back as
  // arbitrary green enchantments. Find Similar, the walk and drill all ask a question
  // about function, so they all read the function space regardless of which projection is
  // on screen. `MAP_CONFIGS[*].embeddings` survives only because each projection is still
  // built from its own space.
  const SIMILARITY_EMBEDDINGS = DATA.embeddingsAbility;

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

  // TWO JOBS, TWO FUNCTIONS. These used to be one, and the overload was a real bug:
  // every plain click runs "replace the selection" first, so clicking a point while a
  // region was focused ran the Escape chain instead — clearing the focus and refitting
  // the camera, i.e. the map zoomed out from under you as you selected a card. The
  // `orientation` branch had done the same thing for longer and less visibly.
  //
  // `clearSelection` now only clears the selection. Peeling belongs to the key.
  function clearSelection() {
    selectedCards = [];
    topCardIndex = 0;
    browseSet = null;
    closeViewerPanel();
    updateSelectionHighlight();
  }

  // Escape peels ONE layer at a time, outermost first: a focused region, then the
  // orientation lens, then the selection. Each press does exactly one visible thing.
  function escapeOnce() {
    if (regionFocus) { clearRegionFocus(); return; }
    if (orientation) { clearOrientation(); return; }
    clearSelection();
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

  // ── Explore as an orientation lens ──────────────────────────────────────
  //
  // Explore stopped being a workspace. Entering it from a graph lights up the cards you
  // are actually holding and dims the other 34,000, so the atlas answers the one question
  // the graph structurally cannot: WHERE this sits. `force.js` says so in its own header —
  // it encodes adjacency, not absolute position.
  // LIVE, not a snapshot. This used to hold `{rows: Set, label, anchor}` copied out of
  // `Force.rows()` at the moment you entered Explore — so the atlas showed a photograph
  // of your walk, and anything you did afterwards was invisible until you left and came
  // back. That is a large part of why Explore felt inert next to Discover.
  //
  // Now it holds only whether the lens is ON; membership is read from Session on every
  // render, and Session reads it from wherever the graph actually lives.
  let orientation = null;   // { label } | null

  function orientationRows() {
    if (!orientation) return [];
    return Session.rows().filter(i => allData[i]);
  }

  function orientTo(rows, label) {
    orientation = { label: label || 'your graph' };
    if (!orientationRows().length) { orientation = null; return false; }
    render();          // render() writes the status line for this mode
    return true;
  }

  function clearOrientation() {
    if (!orientation) return;
    orientation = null;
    render();
    setStatus(allData.length.toLocaleString() + ' cards shown');
  }

  // ── Zoom to a region ────────────────────────────────────────────────────
  //
  // Clicking a cluster label frames that cluster and shows only its cards. Position is
  // preserved: these are the same points at the same world coordinates, just closer.
  //
  // This used to run DRILL, which is a different thing wearing the same gesture. Drill
  // re-embeds the subset from the 128-d vectors with stress majorization, so the points
  // fly out of their world positions over 90 frames and land somewhere new — informative
  // when you *want* local structure, disorienting when you clicked a label expecting to
  // look closer. It also left the map uninteractable afterwards, because the drill
  // animation pushes new coordinates through `updateLayerBy` while the quadtree still
  // holds the world positions it was built from, so every hit-test was against where the
  // cards used to be.
  //
  // Drill is still reachable from the toolbar and from box-select, where asking for a
  // re-layout is explicit. A label click is a camera move.
  let regionFocus = null;   // { id, label, rows: Set }

  async function focusRegion(regionId) {
    const data = await loadRegionData(currentMap);
    if (!data || !data.membership) {
      setStatus('This map has no region membership — re-run `manamap cluster-regions`.');
      return;
    }
    const m = /^l(\d)_(\d+)$/.exec(regionId);
    if (!m) return;
    const labels = data.membership['l' + m[1]];
    const cid = parseInt(m[2], 10);
    if (!labels) return;
    const rows = new Set();
    for (let i = 0; i < labels.length; i++) if (labels[i] === cid) rows.add(i);
    if (!rows.size) { setStatus('That region has no cards on this map.'); return; }

    const region = data.regions.find(r => r.id === regionId);
    regionFocus = { id: regionId, label: (region && region.label) || regionId, rows: rows };
    render();

    // Frame it from the members' real extent rather than the stored w/h, so the camera
    // agrees with what is actually drawn after the supertype filters have had their say.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of rows) {
      const d = allData[i];
      if (!d) continue;
      if (d.x < x0) x0 = d.x;
      if (d.x > x1) x1 = d.x;
      if (d.y < y0) y0 = d.y;
      if (d.y > y1) y1 = d.y;
    }
    if (isFinite(x0) && mapCanvas) {
      const padX = Math.max((x1 - x0) * 0.12, 0.5);
      const padY = Math.max((y1 - y0) * 0.12, 0.5);
      mapCanvas.setCamera({ x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] },
                          { animate: true });
    }
    setStatus(rows.size.toLocaleString() + ' cards in ' + regionFocus.label +
              ' · Esc for the whole map');
  }

  function clearRegionFocus() {
    if (!regionFocus) return false;
    regionFocus = null;
    render();
    if (mapCanvas) mapCanvas.fitToData();
    setStatus(allData.length.toLocaleString() + ' cards shown');
    return true;
  }

  // WHICH RELATIONS EARN AN ARC ON WHICH MAP — measured, not chosen.
  //
  // Median edge length as a multiple of a random pair on the same map:
  //
  //                     default (colour/type)     ability (function)
  //   outclassed-by     7.4u   0.29x              0.82u  0.04x
  //   similar          15.2u   0.60x              0.27u  0.01x
  //   synergy          24.0u   0.95x             19.3u   1.04x
  //
  // Three consequences, each of which decides something:
  //
  // 1. On the DEFAULT map, similar and outclassed-by are real structure — long enough to
  //    see, short enough to mean something. This is where the constellation earns its keep.
  // 2. On the ABILITY map those same relations are already stacked (0.27u apart, 97% of
  //    them inside 5% of the atlas). An arc there is a single pixel pretending to be
  //    information. Drill already exists and is the honest answer to "these are all on top
  //    of each other".
  // 3. SYNERGY is indistinguishable from random on BOTH maps, and that is correct rather
  //    than broken: synergy is complementary, so partners belong in different regions by
  //    construction (blink finds an ETB creature). It is orthogonal to every 2-D
  //    projection we have, so it is NEVER drawn as an atlas arc — no amount of curving or
  //    fading makes a random-length line informative. Its partners light up in place and
  //    the affordance is the graph, one click away, where adjacency IS the geometry.
  const MAP_ARC_RELATIONS = {
    default: { similar: true, obsolete: true, deck: true, synergy: false },
    ability: { similar: false, obsolete: false, deck: true, synergy: false },
  };

  function arcsAllowedOn(map) { return MAP_ARC_RELATIONS[map] || MAP_ARC_RELATIONS.default; }

  // Same two-method contract as Deck Lens and the deck builder — see docs/viz.md.
  const OrientationOverlay = {
    getOverlayTraces() {
      if (!orientation) return [];
      const rows = orientationRows();
      const out = [{
        type: 'scattergl', mode: 'markers',
        name: 'On your graph (' + rows.length + ')',
        x: rows.map(i => allData[i].x), y: rows.map(i => allData[i].y),
        customdata: rows,
        marker: { size: 8, color: '#c4a747', line: { color: '#fff', width: 0.7 } },
        hoverinfo: 'none', _isOrientation: true,
      }];
      // The constellation's edges, drawn where those cards actually live. This is the
      // thing the atlas could never do: a relation you can SEE reaching across the map.
      const allowed = arcsAllowedOn(currentMap);
      const edges = [];
      for (const l of Session.links()) {
        if (!allowed[l.rel]) continue;
        const a = allData[l.a], b = allData[l.b];
        if (!a || !b) continue;
        edges.push({ source: [a.x, a.y], target: [b.x, b.y], rel: l.rel,
                     reason: l.reason, d: l.d });
      }
      if (edges.length) {
        // Edges first so the markers draw on top of them.
        out.unshift({
          mode: 'edges', name: 'relations', edges: edges,
          // A straight line between two distant cards reads as a claim about the space
          // between them; a shallow arc reads as a connection.
          curve: 0.12, line: { width: 1.3 }, opacity: 0.85, _isOrientation: true,
        });
      }

      const anchor = Session.focus;
      if (anchor >= 0 && allData[anchor]) {
        const a = allData[anchor];
        out.push({
          type: 'scattergl', mode: 'markers', name: 'Where you are',
          x: [a.x], y: [a.y], customdata: [anchor],
          marker: { size: 16, color: '#fff', symbol: 'star',
                    line: { color: '#c4a747', width: 1.5 } },
          hoverinfo: 'none', _isOrientation: true,
        });
      }
      return out;
    },
    getDimmedIndices() { return null; },
    dimsAll() { return !!orientation; },
  };

  // THE relation entry point, and it does the SAME THING everywhere.
  //
  // It used to fork: graph modes grew the graph, Explore opened a linear browse set,
  // on the reasoning that a scatter plot cannot grow. True, but it made one button mean
  // two things — and the fix is not to teach the scatter plot to grow, it is to let the
  // click carry you to where growing happens. Explore is a lens now: you go there to see
  // where things sit, then click to start walking from one.
  //
  // Replaces `findSimilarCards` / `findSynergyCards`, which were broken four ways: silent
  // no-ops in Discover and the browse panel (both clear `selectedCards`, the only card
  // identity those functions had), the *wrong card* in The Walk drawn onto a hidden Plotly
  // surface, and an outright throw under `?renderer=canvas` where `#plot` has no `.data`.
  function relate(row, relation) {
    const rel = relation || 'similar';
    if (!window.Discovery || !Discovery.isReady() || !window.Force) return;

    // FROM EXPLORE: grow in place. The card and its relations join the constellation and
    // the edges are drawn where those cards actually live, so you see reach and position
    // at once — the one thing the graph structurally cannot show you.
    //
    // This used to switch modes and carry you into the walk. That was better than the
    // fork before it (Explore opened a linear browse set) but it still meant the atlas
    // could only ever hand you off, never respond. Growing here is what makes Explore a
    // place you can work rather than a place you pass through.
    if (currentMode === 'explore') {
      const before = Session.size();
      Session.grow(row, rel);
      if (!orientation) orientTo(null, 'your walk');
      render();
      const added = Session.size() - before;
      const name = (cardRecord(row) || {}).n || 'that card';
      if (!arcsAllowedOn(currentMap)[rel]) {
        // Synergy is ~random in world space on both maps, and similarity is already
        // stacked on the ability map — so say what happened and where to see it, rather
        // than drawing a line that means nothing. See MAP_ARC_RELATIONS.
        const why = rel === 'synergy'
          ? 'synergy partners sit all over the map — see them in the graph'
          : 'these sit on top of each other here — drill in, or see the graph';
        setStatus(name + ': ' + added + ' added · ' + why);
      } else {
        setStatus(name + ': ' + added + ' added by ' + rel + ' · ' +
                  Session.size() + ' on your graph');
      }
      return;
    }

    // Graph modes: seed ONLY when there is nothing to lose. `Discovery.show` calls
    // `Force.newWalk(true)`, which empties the graph, so calling it for any card not
    // already on the walk destroyed however much you had built. With a graph in hand the
    // card is adopted into it instead. Growing must never be able to delete.
    if (Force.nodeCount === 0) Discovery.show(row);
    else Discovery.setCurrent(row);   // note the card; the panel owner draws it
    Force.branchByRow(row, rel);
    // Whoever owns the panel in this mode repaints it. Calling `Discovery.focus` here
    // rendered Discover's landing controls over Build's roles and curve on every branch.
    if (window.Force) Force.renderPanel();
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
    // cardRecord, not allData: on the discovery landing the projection has not arrived
    // yet, and hovering would silently do nothing.
    const d = cardRecord(row);
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
      // Reposition once the image has real dimensions. The CSS aspect-ratio means the box
      // is already the right size, but a failed load collapses it to the name text, and
      // that box wants clamping too.
      const img = el.querySelector('img');
      if (img) img.addEventListener('load', function () {
        positionPopup(clientX, clientY);
      }, { once: true });
    }, HOVER_DELAY_MS);
  }

  // Flip rather than clip. The panel side is where the cursor usually is, so a popup that
  // always opened right would spend most of its life half off-screen.
  function positionPopup(clientX, clientY) {
    if (!popupEl) return;
    const host = document.getElementById('plot');
    const r = host.getBoundingClientRect();
    const w = popupEl.offsetWidth || 230;
    // The card is 230px wide at a 488:680 ratio, so ~321px tall. Measuring is preferred,
    // but this is positioned the instant the <img> is inserted — before the network has
    // returned anything — and an unloaded image used to measure ~0, which meant the
    // bottom clamp below did nothing and a card hovered near the foot of the page ran
    // straight off it. The CSS reserves the box; this is the belt to that braces.
    const h = Math.max(popupEl.offsetHeight, 321);
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

  // The record for a row, whichever half of the data has arrived. Discovery boots on
  // viz_index (0.56 MB) and the projection lands behind it, so this is what lets the
  // landing paint immediately and get richer rather than waiting for 2.9 MB.
  // `buildCardDetailHtml` is already field-by-field optional — only `.n` is required —
  // so a slim record renders a real card, just without the local oracle text that the
  // Scryfall image is showing anyway.
  function cardRecord(row) {
    if (allData.length && allData[row]) return allData[row];
    return (window.Discovery && Discovery.record(row)) || null;
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

  // `row` is required for the relation buttons: the old pair took no argument at all and
  // leaned on `selectedCards`, which is exactly why they did nothing in three of the five
  // panels that render this HTML.
  function buildCardDetailHtml(d, row) {
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

    html += buildRelationHtml(row);

    return html;
  }

  /* ONE card header, for both panels that draw one.
   *
   * There were two, and they had drifted: the browse panel lost the loyalty and defense
   * branches (so a planeswalker showed no loyalty while browsing but did while selected)
   * and the in-deck badge. Neither omission was a decision — they were copies that
   * stopped being copied.
   *
   * `nav` is the prev/next block, which differs in what it counts (stack position vs
   * browse position), and `extra` is whatever the caller needs between the nav and the
   * close button. Everything else is the same card, so it is written once.
   */
  function cardHeaderHtml(d, row, nav, extra) {
    let html = '<div class="viewer-header">';
    html += '<h2>' + escHtml(d.n) + '</h2>';
    if (nav) {
      html += '<span class="viewer-nav">' +
        '<button class="viewer-arrow" onclick="MM.cyclePrev()" title="Previous (\u2190)">\u2039</button>' +
        '<span class="viewer-count">' + nav + '</span>' +
        '<button class="viewer-arrow" onclick="MM.cycleNext()" title="Next (\u2192)">\u203a</button>' +
        '</span>';
    }
    if (extra) html += extra;
    if (typeof row === 'number' && typeof window.Build !== 'undefined' && Build.isInDeck) {
      if (Build.isInDeck(row)) {
        html += '<span class="in-deck-badge">\u2713 In Deck</span>';
      } else {
        html += '<button class="btn-add-deck" onclick="Build.addCard(' + row +
                '); MM.render()">+ Deck</button>';
      }
    }
    html += '<button class="detail-close" onclick="MM.closeDetail()" title="Close (ESC)">\u00d7</button>';
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
      const rc = ['mythic', 'rare', 'uncommon', 'common'].indexOf(d.r) !== -1 ? d.r : '';
      html += '<span class="stat-divider">\u00b7</span><span class="rarity-pill ' + rc + '">' +
              escHtml(d.r) + '</span>';
    }
    html += '</div></div>';
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

    let html = cardHeaderHtml(d, topCard.idx,
      selectedCards.length > 1 ? (topCardIndex + 1) + '/' + selectedCards.length : null);

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
          html += '<div class="acc-body">' + buildCardDetailHtml(cd, card.idx) + '</div>';
        }
        html += '</div>';
      }
      html += '</div>';
      html += '<div class="keyboard-hint">\u2190 \u2192 navigate \u00b7 1-8 jump \u00b7 Del remove \u00b7 Esc clear all \u00b7 / search</div>';
    } else {
      html += buildCardDetailHtml(d, selectedCards[topCardIndex].idx);
      html += '<div class="keyboard-hint">Shift+click to multi-select \u00b7 Esc clear \u00b7 / search</div>';
    }

    inner.innerHTML = html;
    panel.classList.add('open');
    // Reveal whichever row is open, on every path that changes it — clicking a row,
    // the arrows, the arrow keys, a number key, removing a card, or selecting a new one
    // from the map. This function is only called when the selection actually changes,
    // so it never fights a scroll the reader started themselves.
    scrollActiveRowIntoView();
    preloadNeighbourImages();
    setTimeout(() => { if (mapCanvas) mapCanvas.resize(); }, 260);
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

    let html = cardHeaderHtml(d, browseSet.indices[browseSet.pos],
      (browseSet.pos + 1) + ' / ' + n.toLocaleString(),
      (browseSet.anchor != null && browseSet.pos !== 0)
        ? '<div class="browse-anchor">near <strong>' +
          escHtml(allData[browseSet.anchor].n) + '</strong></div>'
        : '');

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

    // A lassoed set can become a graph. This was `Force.seedFrom()`, reachable only by
    // entering The Walk with a selection live — so when that mode was deleted the
    // capability went quiet rather than away: `seedFrom` still worked and nothing called
    // it. Growing from what you just boxed is the same gesture as growing from a card.
    html += '<button class="lens-btn" onclick="MM.growFromBrowse()">Grow a graph from these ' +
            n.toLocaleString() + '</button>';
    html += buildCardDetailHtml(d, browseSet.indices[browseSet.pos]);
    html += '<div class="keyboard-hint">← → browse · Esc clear · click a point to leave browse mode</div>';

    inner.innerHTML = html;
    panel.classList.add('open');
    inner.scrollTop = 0;
    preloadNeighbourImages();
    setTimeout(() => { if (mapCanvas) mapCanvas.resize(); }, 260);
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
    setTimeout(() => { if (mapCanvas) mapCanvas.resize(); }, 260);
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
    if (!mapCanvas || !browseSet) return false;
    const cur = browseSet.indices[browseSet.pos];
    const p = cardPosition(cur);
    if (!p) return false;
    // `updateLayerBy` is the canvas's `Plotly.restyle`: it matches one layer by flag and
    // moves its points without rebuilding the other 34,322.
    return mapCanvas.updateLayerBy('_isBrowseCurrent',
      { x: [p[0]], y: [p[1]], customdata: [cur] });
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
    if (!mapCanvas) return;

    // The one-marker fast path survives; the add/delete-traces path does not, and did not
    // work here anyway. This function opened by reading `plotDiv.data` — Plotly's trace
    // array, which the canvas host does not have — so under `?renderer=canvas` it returned
    // at the first line and selecting a card never repainted the highlight at all. A full
    // `render()` is the honest replacement: 15 ms on canvas against the 30 ms this was
    // written to avoid, and selection is a user gesture, not a per-frame cost.
    if (browseSet) {
      const key = browseHighlightKey();
      if (key === _highlightKey && moveBrowseMarker()) return;
      _highlightKey = key;
    } else {
      _highlightKey = null;
    }
    render();
  }

  // One space, fetched once. This used to key on `currentMap` and re-fetch on every map
  // toggle, so the same card had different "nearest" answers depending on which picture
  // you happened to be looking at.
  async function loadEmbeddings() {
    if (embeddings) return true;
    if (embeddingsCache.function) {
      embeddings = embeddingsCache.function;
      return true;
    }
    try {
      const r = await fetch(SIMILARITY_EMBEDDINGS);
      if (!r.ok) return false;
      embeddings = new Float32Array(await r.arrayBuffer());
      embeddingsCache.function = embeddings;
      return true;
    } catch (e) {
      return false;
    }
  }

  // THE k-nearest primitive, and now genuinely the only one. The header used to claim
  // this had replaced `findSimilarCards`' hand-rolled scan; it had not — that scan was
  // still there, sorting all 34,322 rows to take 20, with different filter semantics.
  // Both are gone. Rows are L2-normalised at export, so the dot product IS the cosine
  // and no norms are needed. Returns `{i, sim}` nearest-first.
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
    // Exclude by NAME, not just by row. cards.csv carries 51 duplicate names (Un-set
    // reprints and the like), so self-exclusion alone let a card return its own twin at
    // cosine 1.0 as its most similar card — a true statement and a useless answer.
    const selfName = allData[row] && allData[row].n;
    const best = [];                       // ascending by sim; best[0] is the weakest kept
    for (let j = 0; j < allData.length; j++) {
      if (j === row) continue;
      if (allData[j].n === selfName) continue;
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

  // Fires the fetch itself and fills every placeholder on the page when it lands.
  //
  // The load used to be triggered in exactly one place — inside `updateViewerPanel` —
  // and patched only that panel's open card. Every other renderer of card detail (the
  // browse panel, The Walk, Discover) drew a placeholder that nothing ever filled, so
  // "Obsoleted By" and its advantage badges were permanently invisible in three of the
  // five places a card can appear.
  let obsolescencePending = false;

  function ensureObsolescenceIndex() {
    if (obsolescenceIndex || obsolescencePending) return;
    obsolescencePending = true;
    loadObsolescenceIndex().then(function (ok) {
      obsolescencePending = false;
      if (ok) patchObsolescencePlaceholders();
    });
  }

  function patchObsolescencePlaceholders() {
    const slots = document.querySelectorAll('.obsolescence-placeholder[data-card]');
    for (const el of slots) {
      const html = buildObsolescenceHtml(el.getAttribute('data-card'));
      el.outerHTML = html || '';
    }
  }

  // The relation controls, in the shared card HTML so every panel gets the same thing.
  //
  // Counts are precomputed and stated BEFORE the click — 23.6% of cards have nothing but
  // similar, and a control that turns out to do nothing reads as broken rather than as a
  // fact about the card.
  function buildRelationHtml(row) {
    if (typeof row !== 'number' || row < 0) return '';
    if (!window.Discovery || !Discovery.isReady()) return '';
    const c = Discovery.counts(row);
    const btn = (rel, label, n) =>
      '<button class="lens-btn discover-rel' + (n ? '' : ' is-empty') + '"'
      + (n ? ' onclick="MM.relate(' + row + ',\'' + rel + '\')"' : ' disabled')
      + '>' + label + ' <span class="discover-count">' + n + '</span></button>';
    let html = '<div class="discover-relations">'
      + btn('similar', 'Similar', c.similar)
      + btn('synergy', 'Synergy', c.synergy)
      + btn('obsolete', 'Outclassed by', c.obsolete)
      + '</div>';
    if (c.synergy) {
      html += '<p class="lens-note">Synergy is a rule-based list of ten, not a ranking — '
            + 'partners are ordered by how played they are.</p>';
    }
    // The tray follows the card, not the mode. Keeping something you found in the atlas
    // is the same act as keeping something you walked to, so the control lives here
    // rather than only in the Discover panel.
    const kept = Session.tray.has(row);
    html += '<button class="lens-btn discover-keep" onclick="MM.keep(' + row + ')">'
          + (kept ? '✓ In tray' : '+ Keep this card') + '</button>';
    // One card is the commander, and everything reads it from Session: the gold ring on
    // the graph, the colour identity that decides what is legal, the exported brief.
    // Offered on legendary creatures only — the rule, not a preference.
    const rec = cardRecord(row);
    const legendary = rec && /legendary/i.test(rec.t || '') && /creature/i.test(rec.t || '');
    if (legendary) {
      const isCmd = Session.commander === row;
      html += '<button class="lens-btn discover-keep" onclick="MM.setCommander(' +
        (isCmd ? -1 : row) + ')">' +
        (isCmd ? '★ Commander' : 'Set as commander') + '</button>';
    }
    return html;
  }

  // Toggle a card in the tray from any panel, then repaint whichever one is showing.
  /* Designate the commander. Writes Session, then asks the graph to re-ink so the ring
   * moves — `Force.setCommander` is a redraw, not a reseed, because changing your mind
   * about the commander must not cost you the graph. */
  function setCommander(row) {
    Session.setCommander(row);
    if (window.Force && Force.setCommander) Force.setCommander(Session.commander);
    if (window.Build && Build.onCommanderChange) Build.onCommanderChange();
    updateViewerPanel();
    const rec = cardRecord(Session.commander);
    setStatus(Session.commander >= 0
      ? (rec ? rec.n : 'That card') + ' is your commander — colour identity follows it'
      : 'Commander cleared.');
  }

  /* Seed the graph from the current browse set and go where growing happens. Capped by
   * `Force.enter` itself (MAX_NODES, announced in the panel's truncation notice). */
  function growFromBrowse() {
    if (!browseSet || !browseSet.indices.length || !window.Force) return;
    // Capture BEFORE switching modes: `setMode` clears the browse set, so reading
    // `browseSet.label` in the callback throws on null. The set is the input to this
    // function, not state it can rely on afterwards.
    const rows = browseSet.indices.slice();
    const label = browseSet.label || 'Selection';
    const sel = document.getElementById('modeSelect');
    if (sel) sel.value = 'discover';
    setMode('discover');
    Force.newWalk(true);
    Promise.resolve(Force.enter(rows, label, { chrome: 'discovery' }))
      .then(function () {
        if (window.Discovery) { Discovery.setCurrent(rows[0]); Discovery.render(); }
        setStatus(rows.length.toLocaleString() + ' cards from ' + label +
                  ' — click any card to grow outward.');
      });
  }

  function keep(row) {
    if (!window.Discovery || !Discovery.isReady()) return;
    Session.tray.toggle(row);
    if (currentMode === 'explore' || currentMode === 'build') {
      updateViewerPanel();
    }
  }

  function buildObsolescenceHtml(cardName) {
    if (!obsolescenceIndex) {
      ensureObsolescenceIndex();
      return '<span class="obsolescence-placeholder" data-card="'
        + escHtml(cardName) + '"></span>';
    }
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
    // Embeddings are deliberately untouched: the projection changed, the space did not.
    // These used to be re-keyed per map, which both dropped a 17 MB array that was still
    // correct and made "similar" mean something different on each picture.
    // The coordinates themselves changed, so this is the one case that must forget the
    // camera rather than preserve it — holding the old range would frame the wrong part
    // of a different map. Plotly expressed this by clearing `plotInitialized` so the next
    // `react` autoranged; the canvas needs it said out loud, because its camera is a
    // persistent d3 transform that survives `setLayers` by design.
    plotInitialized = false;
    render();
    if (mapCanvas) mapCanvas.fitToData();
    setMapStatus();
  }

  async function switchMap(mapName) {
    if (mapName === currentMap) return;
    currentMap = mapName;
    // Pre-load region data so it's ready when render() builds annotations
    if (showRegionLabels) await loadRegionData(mapName);
    await loadProjection(mapName);
    // Re-apply selection highlight after map switch (positions changed).
    updateSelectionHighlight();
    // ...and restate which map you are on, because the highlight now goes through
    // `render()`, which writes its own status line and would otherwise leave you looking
    // at the Abilities map being told how many cards are shown. Under Plotly this was an
    // addTraces/deleteTraces pair that touched no status.
    setMapStatus();
  }

  function setMapStatus() {
    setStatus(`${allData.length.toLocaleString()} cards loaded — ` +
              `${currentMap === 'ability' ? 'Abilities' : 'Color + Type'} map`);
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

  // `getRegionAnnotations` lived here: the identical span→opacity/size curve that
  // `refreshCanvasLabels` computes below, differing only in output field names, built
  // into a Plotly `annotations` array. Under the canvas it was still computed on every
  // render and thrown away unread at the renderer fork. Deleted with `refreshLabelsOnZoom`
  // and the `_labelUpdateInFlight` re-entry guard, which existed only because
  // `Plotly.relayout` fires `plotly_relayout` and would otherwise loop on itself.
  // The same opacity/size curve `getRegionAnnotations` computes, handed to the canvas
  // renderer as DOM instead of Plotly annotations — so the crossfade is a CSS transition
  // rather than an rgba() alpha rebuilt on a 150 ms debounce, and each label is a real
  // button rather than something a 30-line d2p hit-test has to find.
  function refreshCanvasLabels() {
    if (!mapCanvas) return;
    const cam = mapCanvas.getCamera();
    const span = cam ? Math.abs(cam.x[1] - cam.x[0]) : 70;
    const data = regionDataCache[currentMap];
    if (!data || !showRegionLabels) { mapCanvas.setAnnotations([]); return; }
    const out = [];
    for (const region of data.regions) {
      let opacity = 0, size = 11;
      if (region.level === 0) {
        if (span > 25) opacity = 1;
        else if (span > 15) opacity = (span - 15) / 10;
        size = 16;
      } else {
        if (region.span < span * 0.05) continue;
        if (span < 20) opacity = 1;
        else if (span < 30) opacity = (30 - span) / 10;
      }
      if (opacity <= 0) continue;
      out.push({
        x: region.cx, y: region.cy, id: region.id, size: size,
        text: region.level === 0 ? region.label : region.short,
        colour: region.level === 0
          ? 'rgba(196,167,71,' + opacity.toFixed(2) + ')'
          : 'rgba(200,200,200,' + opacity.toFixed(2) + ')',
      });
    }
    mapCanvas.setAnnotations(out);
  }

  // Plotly's legend, rebuilt as ours — which is most of what "control and polish" meant.
  function renderCanvasLegend(traces) {
    let el = document.getElementById('mapLegend');
    if (!el) {
      el = document.createElement('div');
      el.id = 'mapLegend';
      el.className = 'map-legend';
      document.getElementById('plot').appendChild(el);
    }
    el.innerHTML = traces
      .filter(tr => tr.name && tr.visible !== false &&
              tr.mode !== 'lines' && tr.mode !== 'edges')
      .map(tr => {
        const m = tr.marker || {};
        const c = Array.isArray(m.color) ? '#8a8a8a' : (m.color || '#666');
        return '<div class="map-legend-row"><span class="map-legend-dot" style="background:' +
          (c === 'rgba(0,0,0,0)' ? 'transparent;border:2px solid ' +
            ((m.line && m.line.color) || '#888') : c) +
          '"></span>' + escHtml(tr.name) + '</div>';
      }).join('');
  }

  // ── Load data ──
  //
  // Two tracks. Discovery is usable on 1.83 MB (viz_index 0.56 + neighbours 1.27, gz) and is the front
  // door; the 2.9 MB projection loads *behind* it and upgrades every record in place —
  // `MM.cardRecord` prefers the full row when it exists and falls back to the slim one.
  // Landing used to mean waiting for the projection before a single pixel appeared.
  const params = new URLSearchParams(window.location.search);
  const wantedDeck = params.get('deck');
  // ?mode=explore deep-links straight to the atlas. Discovery is the front door now, so
  // anything that wants the 34,322-point map — a bookmark, a browser test about
  // rendering — has to ask for it rather than assume it is what boot produces.
  const wantedMode = params.get('mode');
  if (wantedMode) currentMode = wantedMode;

  Discovery.configure({ vizIndex: DATA.vizIndex, neighbours: DATA.neighbours });
  // Apply the mode chrome BEFORE the data arrives. `currentMode` being 'discover' is not
  // enough on its own — setMode is what hides the Plotly surface and gives the force
  // canvas a size, and without it the canvas measured 0x0, so the landing card's
  // transform resolved to (0,0) and it drew half off-screen behind the toolbar.
  //
  // queueMicrotask, not a direct call: every line in this file runs INSIDE the IIFE whose
  // return value becomes `window.MM`, so the global does not exist yet. Discovery touches
  // MM.setStatus, and calling it here threw — which aborted the IIFE, so MM was never
  // exported and deck-builder.js failed at its own top level too. One ordering mistake,
  // four broken files, twice. A microtask runs after the assignment completes.
  if (!wantedDeck) queueMicrotask(function () {
    const sel = document.getElementById('modeSelect');
    if (sel) sel.value = currentMode;
    // Only discovery needs its chrome applied before data arrives — it is the one mode
    // that renders from viz_index alone. Calling setMode('explore') here would run
    // render() against an empty allData and initialise Plotly on nothing; the data-load
    // path below already renders explore once there is something to draw.
    if (currentMode === 'discover') setMode('discover');
  });
  Discovery.ready()
    .then(() => {
      if (!wantedDeck && currentMode === 'discover') Discovery.land(params);
    })
    .catch(err => setStatus('Discovery unavailable: ' + err.message));

  fetch(MAP_CONFIGS.default.projection)
    .then(r => r.json())
    .then(data => {
      allData = data;
      projectionCache['default'] = data;
      initToggles();
      refreshDrillButton();
      // Only paint the scatter if that is what the user is looking at. Rendering 34,322
      // points behind a landing card is work nobody asked for.
      if (currentMode === 'explore') {
        render();
        setStatus(`${allData.length.toLocaleString()} cards loaded`);
      }
      // ?deck=<slug> is the map's first inbound deep link — the dossier and the
      // magazine's Back Page both use it. Honour it by entering the Lens, not by
      // dropping the reader on an unfiltered map with a query string they can't see.
      if (wantedDeck) {
        // `?deck=<slug>` is an inbound contract: the dossier and every published manual
        // link to it. Deck Lens and Build Deck are one mode now, so it lands in Build.
        document.getElementById('modeSelect').value = 'build';
        setMode('build');
      }
      // Load region data in background, then re-render with labels
      loadRegionData('default').then(data => {
        if (data && currentMode === 'explore') render();
      });
    })
    .catch(err => setStatus('Error loading data: ' + err.message));

  // ── Supertype toggle buttons ──
  // The Drill button used to say only "Drill ⤓". With no filters that meant "re-map all
  // 34,322 cards", which the cap then truncated to an arbitrary 2,000 — an incoherent
  // cross-section of the whole universe that flew in from everywhere and settled into a
  // multicoloured pile. The button now states the size of what it would drill and goes
  // inert when that is over the cap, so you can see whether pressing it will do anything.
  // Shift arms the marquee on canvas; on Plotly it flips dragmode. Same gesture either way.
  function setCanvasSelectMode(on) { if (mapCanvas) mapCanvas.setSelectMode(on); }

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
        if (typeof window.Build !== 'undefined' && window.Build.handleEscape) {
          window.Build.handleEscape();
        }
      } else {
        escapeOnce();
      }
      return;
    }

    // NO BLANKET MODE GATE. This was `if (currentMode !== 'explore') return;`, which
    // killed every key below in every other mode — including `/` for search, which is in
    // the toolbar and visible from everywhere. It was also redundant: each branch already
    // refuses when its own data is absent (`browseSet`, `selectedCards`), so the gate was
    // doing nothing except making three modes feel keyboard-dead. That asymmetry is a real
    // part of why the modes felt like different products.
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

    // Search is in the toolbar and reachable from every mode, so its shortcut is too.
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

  /* Is the 34K atlas the surface under the cursor? Explore always; Build only in its map
   * view, since its graph view hands the canvas to force.js. */
  function mapSurfaceShowing() {
    if (currentMode === 'explore') return true;
    return currentMode === 'build' && window.Build && Build.view === 'map';
  }

  document.addEventListener('keydown', e => {
    // Box-select needs the atlas under the cursor, which is Explore and Build's map view.
    // The graph modes have their own drag (fling a node), so arming a marquee there would
    // fight it.
    if (e.key === 'Shift' && !shiftHeld && mapSurfaceShowing()) {
      shiftHeld = true;
      setCanvasSelectMode(true);          // arms the marquee
      const plotDiv = document.getElementById('plot');
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
      setCanvasSelectMode(false);
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
    const sel = document.getElementById('modeSelect');
    if (sel && sel.value !== mode) sel.value = mode;
    hideCardPopup();
    const detail = document.getElementById('detailPanel');

    if (mode !== 'discover' && typeof window.Discovery !== 'undefined') window.Discovery.exit();
    if (mode !== 'build' && typeof window.Build !== 'undefined') window.Build.exit();
    // Discover owns the graph surface, so entering it must not tear the graph down.
    // (`currentMode` is already the DESTINATION here, which is what lets `Build.exit`
    // tell "leaving for Explore" — a lens, keep the graph — from "leaving for Discover" —
    // a different workspace, hand the canvas back.)
    if (mode !== 'discover' && typeof window.Force !== 'undefined') {
      window.Force.exit();
    }

    // The Walk replaces the plot rather than overlaying it: it is a different renderer
    // (canvas) drawing a different thing (a graph, not a projection), so the Plotly
    // surface is hidden outright while it runs.
    // Discovery has no plot of its own — it is a card and a panel, with the canvas
    // waiting behind. Hiding the Plotly surface keeps the landing from being a card
    // pasted over 34,322 points the visitor did not ask for.
    const plotEl = document.getElementById('plot');
    // The class is still called `force-mode` — it names the force CANVAS, not the
    // deleted mode — and Discover is the only mode that shows it.
    plotEl.classList.toggle('force-mode', mode === 'discover');

    // Leaving a graph for the atlas is a question about position, so answer it: carry the
    // graph's cards across and light them up. Entering explore any other way clears it.
    if (mode === 'explore') {
      // NOT gated on Force.isActive(): the exit above already flipped it false, and
      // `exit()` deliberately keeps the nodes so the walk can be resumed. The graph you
      // just left is exactly the thing you want to locate.
      // Membership is read live from Session now, so this only decides whether the lens
      // is on. The anchor comes from Session.focus rather than being frozen here.
      if (Session.size()) orientTo(null, 'your walk');
      else clearOrientation();
    } else if (mode !== 'discover') {
      orientation = null;
    }

    if (mode === 'discover') {
      clearSelection();
      detail.style.display = 'none';
      if (typeof window.Discovery !== 'undefined') window.Discovery.enter();
      return;
    }

    if (mode === 'build') {
      // Build KEEPS the detail panel. Deck Lens did and the builder did not, and the Lens
      // was right: clicking a lit card to read it is the whole interaction. The builder
      // hid it because its own panel carried the card, which is why adding a card there
      // felt like filing rather than looking at anything.
      detail.style.display = '';
      if (typeof window.Build !== 'undefined') window.Build.enter();
    } else {
      detail.style.display = '';
    }
    render();
  }

  // Mobile pinch-to-zoom used to live here: ~80 lines of touchstart/touchmove that
  // computed an anchor fraction and pushed axis ranges through `Plotly.relayout`,
  // written because Plotly's scattergl has no native pinch. `d3.zoom` handles touch
  // itself, so the canvas gets pinch for free and the hand-rolled version is gone.

  // ── Get category key and palette for current color mode ──
  function getCategoryInfo(d) {
    if (currentColorBy === 'color') return { key: d.c, palette: COLOR_PALETTE };
    if (currentColorBy === 'supertype') return { key: d.s, palette: SUPERTYPE_PALETTE };
    return { key: d.r, palette: RARITY_PALETTE };
  }

  // Canvas wiring, done once. This lived inside `render()` behind the renderer fork;
  // with one renderer it is plain initialisation and belongs at the top level.
  function initMapCanvas() {
      mapCanvas = window.MapCanvas.create().init(document.getElementById('plot'));
      mapCanvas.on('click', function (ev) {
        // Region labels are real DOM buttons on this renderer and emit `regionId` with a
        // null `row`. This handler read only `ev.row`, so clicking a region label ran
        // `addToSelection(null)` and threw inside `updateViewerPanel` — a dead control
        // that read as a rendering bug. The Plotly path routed the same click to
        // `Drill.enterRegion` through a 30-line d2p hit-test against annotation anchors;
        // a real button hands it over for free.
        if (ev.regionId != null) {
          // Zoom and filter, not re-embed. See `focusRegion`.
          if (currentMode !== 'build' &&
              !(typeof window.Drill !== 'undefined' && window.Drill.isActive())) {
            focusRegion(ev.regionId);
          }
          return;
        }
        if (ev.row == null || !allData[ev.row]) return;
        if (currentMode === 'build') {
          if (typeof window.Build !== 'undefined') window.Build.addCard(ev.row);
          return;
        }
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
        regionDebounceTimer = setTimeout(refreshCanvasLabels, 150);
      });
      // Box-select, on a quadtree instead of Plotly's hit test: 4.5 ms against 138 ms.
      mapCanvas.on('select', function (ev) {
        const rows = ev.rows || [];
        if (!rows.length) return;
        if (rows.length > MAX_SELECTED) {
          enterBrowse(rows, 'Selection');
          if (typeof window.Drill !== 'undefined') window.Drill.offer(rows, 'Selection');
        } else {
          selectedCards = rows.map(idx => ({ idx, data: allData[idx] }));
          topCardIndex = 0;
          updateViewerPanel();
          updateSelectionHighlight();
          setStatus(`Selected ${rows.length} card${rows.length === 1 ? '' : 's'}`);
        }
      });
      plotInitialized = true;

    // The side panels resize #plot through a CSS transition, so a one-shot timer can
    // fire mid-transition and leave a stale-width canvas painted over the open panel.
    // The observer is the reliable version; Plotly needed four scattered 260 ms timers
    // for the same job and they went with it.
    if (window.ResizeObserver) {
      let resizeDebounce = null;
      new ResizeObserver(function () {
        clearTimeout(resizeDebounce);
        resizeDebounce = setTimeout(function () { if (mapCanvas) mapCanvas.resize(); }, 120);
      }).observe(document.getElementById('plot'));
    }
  }

  // ── Render plot ──
  function render() {

    // Get overlay traces from whichever mode owns the side panel. Both implement the
    // same two-method contract; see docs/viz.md.
    let overlayTraces = [];
    let dimmedIndices = null;
    const overlay = currentMode === 'build' ? window.Build
      : (currentMode === 'explore' && orientation) ? OrientationOverlay
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

    // A focused region is a filter, which is what makes "show me this cluster" mean the
    // same thing as every other way of narrowing the map.
    // One predicate, used by BOTH the group loop below and the contour source. They used
    // to disagree: `filtered` fed contours and the status count while the group loop
    // re-tested `activeSupertypes` against `allData` itself, so adding a filter here
    // silently did nothing to what was drawn.
    const visible = (d, i) => activeSupertypes.has(d.s) &&
                              (!regionFocus || regionFocus.rows.has(i));
    const filtered = allData.filter(visible);

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
      if (!visible(d, i)) continue;
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
    } else if (currentMode === 'explore' && orientation) {
      // The bare card count is the wrong answer while the lens is on — you came here to
      // see where YOUR cards are, not to be told how many exist.
      setStatus(orientationRows().length + ' cards from ' + orientation.label +
                ' — highlighted in the full map · Esc to see everything');
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

    // The canvas owns the camera, so nothing here has to preserve it. Under Plotly this
    // block read `_fullLayout.xaxis.range` and wrote it back into a freshly built layout,
    // because `react()` replaced layout wholesale and would otherwise silently autorange —
    // filtering and zooming were mutually destructive without it (zoom to a span of 20.5,
    // call render(), get 116.6). The canvas never rebuilds a layout, so the hazard left
    // with the renderer rather than being ported.
    if (!mapCanvas) initMapCanvas();
    mapCanvas.setLayers(allTraces);
    mapCanvas.setContours(showContours);
    refreshCanvasLabels();
    renderCanvasLegend(allTraces);
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
    cardRecord,
    cardImageUrl,
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
    get mode() { return currentMode; },
    selectByName,
    growFromBrowse,
    setCommander,
    focusRegion,
    clearRegionFocus,
    get regionFocus() { return regionFocus; },
    relate,
    keep,
    orientTo,
    clearOrientation,
    get orientation() { return orientation; },
    render,
    setStatus,
    setMode,
    MAP_CONFIGS,
    DATA,
    DATA_VERSION,
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
