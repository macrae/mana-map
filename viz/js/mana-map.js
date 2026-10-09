/**
 * mana-map.js — Explore mode: the 34K atlas, search, overlays, and the card
 * viewer panel with multi-card selection.
 *
 * Exposes shared state and helpers on `window.MM`, which `discovery.js`,
 * `drill.js`, `force.js` and `build.js` all read. `deck-view.js` does NOT —
 * `deck.html` loads it alone, so `MM` does not exist there and it carries its
 * own `esc`. Anything that runs during this file's boot executes INSIDE the
 * IIFE, before `window.MM` is assigned; touching `MM.*` there aborts the IIFE
 * and every later file fails at its own top level.
 */
(function () {
  // ── Palettes ──
  const COLOR_PALETTE = { W: '#F0E68C', U: '#4A90D9', B: '#8B5CF6', R: '#DC2626', G: '#22C55E', Colorless: '#9CA3AF', Multicolor: '#D4A017' };
  /* Supertype colours, with SATURATION RUNNING INVERSE TO FREQUENCY.
   *
   * Measured over the 34,322-card corpus: Creature 55.5%, Instant 11.2%, Sorcery 10.6%,
   * Enchantment 10.4%, Artifact 7.5%, Land 3.4%, Planeswalker 1.0% (329 cards),
   * Unknown 0.3%, Battle 0.1% (39 cards). A palette that ignores that spread gives the
   * majority class the loudest ink and buries the 39 cards you would actually hunt for.
   *
   * The previous palette also collided head-on with COLOR_PALETTE — Creature was `#22C55E`,
   * the exact green that means "green card" in the other colour mode, on 55% of the points.
   * The supertype map looked like a broken colour-identity map.
   *
   * Three families, so the legend teaches itself: SPELLS are a chroma sweep (azure ->
   * violet -> rose), PERMANENTS are material tones (bone, steel, earth), and the two rare
   * types are hot accents that stay findable at 3px among 34,000 neighbours.
   */
  const SUPERTYPE_PALETTE = {
    Creature: '#A8977F',      // 55% of the map — muted bone, the substrate
    Instant: '#3FA9F5',       // azure: reactive, at instant speed
    Sorcery: '#D1497F',       // rose-magenta: the deliberate half of the pair
    Enchantment: '#9B7BE8',   // Nyx violet — Theros starfield
    Artifact: '#9FB0BC',      // Mirrodin steel; neutral, NOT cyan, to stay clear of Instant
    Land: '#8C6B34',          // earth ochre, darkened so it cannot read as Planeswalker gold
    Planeswalker: '#FFC94A',  // 329 cards: the spark, bright enough to spot
    Battle: '#FF4A34',        // 39 cards: the hottest thing on the map
    Unknown: '#4A5058',       // recedes
  };
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
  let currentColorBy = 'supertype';   // see SUPERTYPE_PALETTE for why this is the default
  let searchTerm = '';
  let searchTimeout = null;
  let plotInitialized = false;
  let currentMode = 'discover';
  let embeddings = null; // Float32Array, loaded lazily for Find Similar
  const EMBED_DIM = 128; // mirrors FINAL_EMBEDDING_DIM in config.py
  // The ABILITY map is the default. It is the space that answers "what does this card
  // DO" — the same embedding Find Similar, the walk and drill all read regardless of
  // which map is displayed. Colour+Type is a projection of information already on the
  // card face; abilities is the one you cannot get by reading the card.
  let currentMap = 'ability'; // 'ability' or 'default'
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
  // 6: yawgmoth-swarm's 99 changed by twenty cards and gained two verified lines. The
  // deck artifacts are cache-busted through this constant, so a deck edit that does not
  // bump it serves the OLD 99 from cache — silently, and looking exactly like a render
  // bug: cut cards keep drawing and the panel counts a sideboard that no longer exists.
  // 9: 2026-09-01 a SECOND similarity space: the toggle changes which .bin answers 'what is
  // like this card' — same shape, different meaning.
  // 10: 2026-10-02 Reality Fracture: 34,955 cards and EVERY space retrained, so every row
  // index and every neighbour means something new.
  const DATA_VERSION = 11;  // 2026-10-09 card_flags.json (bans, Game Changers) and combo_index.json join the registry; the panel now says BANNED
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
    // `combo_graph.json` was registered here and fetched by NOTHING. The deck
    // builder that read it is gone, and a registered URL nobody requests is a
    // trap: it reads as a live dependency, so the 4.5 MB artifact behind it
    // looks load-bearing to anyone deciding what may be deleted. Combo prose
    // now comes from the stack artifacts, which are per-deck and already
    // fetched. The FILE stays — `config.py` uses it as the invalidation proxy
    // for `combo_details.json`, which several Python consumers do read.
    // The discovery front door — small enough to land on before anything else arrives.
    vizIndex: v(DATA_BASE + 'viz_index.json'),
    neighbours: v(DATA_BASE + 'neighbours.bin'),
    // The SECOND similarity space. Same four artifacts as the function space,
    // built by `manamap <step> --space cardbert`. `viz_index.json` is NOT
    // duplicated — it carries no embedding-derived field, so one copy serves
    // every space and cannot disagree with itself.
    projectionCardbert: v(DATA_BASE + 'projection_2d_cardbert.json'),
    embeddingsCardbert: v(DATA_BASE + 'embeddings_cardbert.bin'),
    regionsCardbert: v(DATA_BASE + 'regions_cardbert.json'),
    neighboursCardbert: v(DATA_BASE + 'neighbours_cardbert.bin'),
    // Lazy: only fetched when the Role grouping is selected. 0.39 MB gzipped against a
    // 1.83 MB discovery boot is not something to spend before someone asks for it.
    cardRoles: v(DATA_BASE + 'card_roles.json'),
    // Lazy: the set picker's labels and newest-first order, fetched the first time
    // Explore is reached. ~15 KB, and the filter works without it — the codes
    // themselves ride in `viz_index.json` as `e`.
    sets: v(DATA_BASE + 'sets.json'),
    // Fetched at boot beside `viz_index.json`: 12 KB, and it is what lets the panel say
    // BANNED rather than "not legal" — the projection's `f` lists LEGAL formats only, so
    // the two states were indistinguishable until this file existed. Also the WotC Game
    // Changer list, which `quickStatsHtml` pins on every card that carries it.
    cardFlags: v(DATA_BASE + 'card_flags.json'),
    // Lazy, like `cardRoles`: ~1 MB gzipped, fetched by the first card panel that opens
    // and resolved once (`comboIndex()`). The per-deck `combos.json` is Build's and is
    // fetched with the deck; this is the corpus-wide "what does this card go infinite
    // with", capped at `meta.per_card` lines per card.
    comboIndex: v(DATA_BASE + 'combo_index.json'),
  };
  const MAP_CONFIGS = {
    default: { projection: DATA.projection, embeddings: DATA.embeddings, regions: DATA.regionsDefault },
    ability: { projection: DATA.projectionAbility, embeddings: DATA.embeddingsAbility, regions: DATA.regionsAbility },
    cardbert: { projection: DATA.projectionCardbert, embeddings: DATA.embeddingsCardbert, regions: DATA.regionsCardbert },
  };

  // THE SIMILARITY SPACES, mirroring `src/manamap/spaces.py`. Only 128-d spaces
  // appear here: `EMBED_DIM` below is a hardcoded 128 and the .bin is
  // HEADERLESS, so a 384-d file would parse as plausible garbage rather than
  // fail. The `map` field is which projection this space laid out, so switching
  // space can move the picture with it.
  //
  // WHAT THE CHOICE COSTS, measured (`manamap eval-embeddings`, intervals on the
  // difference, 95%): cardbert LOSES functional similarity at every candidate
  // pool size — -0.205 at pool 100, which is the size Find Similar actually
  // ranks against — and WINS theme/tribe at every size, +0.094 at pool 100. It
  // also separates hard negatives 2.8x better (0.0377 against 0.0133). It is a
  // trade, so it is offered rather than defaulted to.
  const SPACES = {
    function: {
      label: 'function',
      embeddings: DATA.embeddingsAbility,
      neighbours: DATA.neighbours,
      map: 'ability',
      note: 'what a card DOES. Trained on role and tag positives.',
    },
    cardbert: {
      label: 'cardbert',
      embeddings: DATA.embeddingsCardbert,
      neighbours: DATA.neighboursCardbert,
      map: 'cardbert',
      note: 'masked-field imputation. Better at tribe, worse at function.',
    },
  };
  let currentSpace = 'function';

  // Similarity is NOT the displayed map. The default map is laid out by colour and type,
  // which is a good picture and a terrible answer to "what is like this card" — measured,
  // that space used 3.05 of its 128 dimensions and scored 0.044 recall@10 against known
  // functional equivalents, which is why Doubling Season's neighbours came back as
  // arbitrary green enchantments. Find Similar, the walk and drill all ask a question
  // about function, so they all read the function space regardless of which projection is
  // on screen. `MAP_CONFIGS[*].embeddings` survives only because each projection is still
  // built from its own space.
  // Was a `const` pointing at one file. It is a LOOKUP now, because the space is
  // selectable — but the rule above is unchanged: similarity never follows the
  // displayed MAP, it follows the chosen SPACE. Switching to the colour+type map
  // still asks the function space for neighbours.
  const similarityEmbeddings = () => SPACES[currentSpace].embeddings;

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

  // Escape peels ONE layer at a time, outermost first: a focused region, then a clicked
  // legend group, then the orientation lens, then the selection. Each press does exactly
  // one visible thing.
  function escapeOnce() {
    // The query is peeled first because it is the narrowing the pilot most
    // recently asked for, and because leaving it on while clearing a region
    // would look like Escape did nothing.
    if (queryFocus) { clearQueryFocus(); return; }
    // The printing highlight next: it is a toolbar narrowing like the query,
    // and leaving it lit while Escape peeled a region would look like the key
    // did nothing.
    if (printFocus) { clearPrintFocus(); return; }
    if (regionFocus) { clearRegionFocus(); return; }
    // The WHOLE legend selection, in one press. "Each press does exactly one
    // visible thing" — and "the legend filter is gone" is one thing, where
    // popping keys one at a time would be several presses for one intent.
    if (legendKeys.size) { clearLegendFocus(); return; }
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

  /* Order a set as a WALK: start near the middle, then always step to the
   * nearest card not yet visited. Browsing the result moves you through the
   * neighbourhood rather than through an arbitrary list, which is the whole
   * ergonomic point — consecutive cards are alike, so a filtered set reads as a
   * tour instead of a shuffle.
   *
   * MATERIALISED UP FRONT, AND CAPPED, and the cap is why. A greedy tour is
   * O(n²·d): measured against the real 128-d matrix that is 24M multiply-adds
   * for 433 cards and 2,960M for 4,809. Computing each step lazily instead is
   * ~55k — trivially fast — but `preloadNeighbourImages` reads `indices[pos+1]`
   * to warm the next card's art, and its own comment says image latency is
   * "most of what made the old panel feel slow to browse". A lazily-ordered
   * walk would have nothing to preload. So the tour is built once, inside the
   * async `enterBrowse` that already says "Ordering N cards…", and above the
   * cap the existing centroid order stands — with the panel SAYING which,
   * because a label that lies about the sequence is worse than no walk.
   *
   * THE CAP IS MEASURED IN THE BROWSER, not inferred from the multiply-add
   * count: 250 cards 83ms, 1,000 113ms, 1,500 238ms, 2,000 413ms. It is set to
   * `Drill.MAX_DRILL` so one sentence covers both — if you can re-map a set,
   * you can walk it — and 413ms sits inside the async "Ordering N cards…" the
   * caller already shows. A first guess of 1,000 was half of what the machine
   * will actually do.
   */
  const WALK_MAX = 2000;

  function orderByNearestWalk(rows) {
    if (!embeddings || rows.length < 3) return null;
    if (rows.length > WALK_MAX) return null;
    const dim = EMBED_DIM;

    const dot = (a, b) => {
      const oa = a * dim, ob = b * dim;
      let v = 0;
      for (let i = 0; i < dim; i++) v += embeddings[oa + i] * embeddings[ob + i];
      return v;
    };

    // Start from the most typical card — nearest the set's own centroid — so the
    // walk opens in the middle of the cluster and works outward, rather than
    // opening on an outlier and spending its first steps crossing the space.
    const centroid = new Float64Array(dim);
    for (const r of rows) {
      const o = r * dim;
      for (let i = 0; i < dim; i++) centroid[i] += embeddings[o + i];
    }
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += centroid[i] * centroid[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < dim; i++) centroid[i] /= norm;

    let start = rows[0], bestSim = -Infinity;
    for (const r of rows) {
      const o = r * dim;
      let v = 0;
      for (let i = 0; i < dim; i++) v += embeddings[o + i] * centroid[i];
      if (v > bestSim) { bestSim = v; start = r; }
    }

    const remaining = new Set(rows);
    remaining.delete(start);
    const out = [start];
    const sims = [1];
    let cur = start;
    while (remaining.size) {
      let next = null, best = -Infinity;
      for (const r of remaining) {
        const v = dot(cur, r);
        if (v > best) { best = v; next = r; }
      }
      remaining.delete(next);
      out.push(next);
      // The cosine to the PREVIOUS card, not to a fixed anchor — a different
      // quantity in the same slot, which is why the panel labels it.
      sims.push(best);
      cur = next;
    }
    return { indices: out, sims: sims };
  }

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
  /* What the search box has narrowed the atlas to. Same SHAPE as `regionFocus` on
   * purpose — a row Set — so `spotlightFor` composes the two without a second
   * dimming path. `field` is which tier hit, and it is shown, because "treasure"
   * matching 413 cards through ORACLE TEXT and "flying" matching 3,361 through
   * KEYWORD are different facts and a bare count cannot tell them apart. */
  let queryFocus = null;    // { rows: Set, field, total, shown, term } | null

  /* WHAT A FILTER MEANS, and the reason it is not the old tier cascade.
   *
   * The cascade stopped at the first tier that matched anything, which is right
   * for FINDING A CARD and wrong for narrowing the atlas. Measured: "treasure"
   * returned ONE card, because a card is named exactly "Treasure" and the exact
   * tier fires first — while 390 cards carry the treasure KEYWORD and 413
   * mention it in oracle text. A pilot asking to see treasure cards got one.
   *
   * So: an exact name match still wins alone, because typing a card's whole name
   * is unambiguous. Everything else UNIONS across the fields, and the status
   * names the two that contributed most — the counts overlap, so they are
   * reported rather than summed.
   */
  const SEARCH_FIELDS = [
    ['NAME', (d) => d.n],
    ['TYPE', (d) => d.t],
    ['KEYWORD', (d) => d.k],
    ['ORACLE TEXT', (d) => d.o],
  ];

  function computeQuery(term) {
    if (!term || term.length < 2) return null;
    // Oracle, type and keyword live only in the projection. Without it the honest
    // answer is "not yet", not an empty result set that looks like "none".
    if (!allData.length) return { rows: new Set(), fields: [], total: 0, term, pending: true };

    /* NO EXACT-NAME SHORTCUT, and it was tried. "An exact card name is
     * unambiguous intent" sounds right and is wrong here: a card is named
     * exactly "Treasure", so asking to see treasure cards returned ONE while
     * 390 carry the keyword and 413 mention it. The shortcut also buys nothing —
     * measured, a full card name unions to 1 ("Craterhoof Behemoth",
     * "Counterspell") or 3 ("Sol Ring"), because almost nothing else says it.
     * One rule, no special case, no surprising minimum. */
    const rows = new Set();
    const counts = [];
    for (const [label, get] of SEARCH_FIELDS) {
      let n = 0;
      for (let i = 0; i < allData.length; i++) {
        const d = allData[i];
        if (!activeSupertypes.has(d.s)) continue;
        const hay = get(d);
        if (hay && hay.toLowerCase().includes(term)) { rows.add(i); n++; }
      }
      if (n) counts.push([label, n]);
    }
    counts.sort((a, b) => b[1] - a[1]);
    return { rows, fields: counts, total: rows.size, term };
  }
  /* A legend row you clicked. Deliberately a GROUP key and not a row set: the traces are
   * already partitioned by category, so "light up Planeswalkers" is one scalar per group
   * rather than a 34,322-entry array — the same distinction `dimsAll()` exists to make.
   * It composes with `regionFocus` through `spotlight()`, which is the single place that
   * decides whether a point is lit. */
  /* A SET, because selecting one colour and selecting two are the same gesture.
   * Empty means everything, so there is no "clear" state to explain — you turn
   * the last one off and the map is whole again.
   *
   * Still GROUP KEYS and not a row set, which is the point the original comment
   * was making: the traces are already partitioned by category, so this stays
   * one comparison per group and never touches the 34,890-entry opacity array.
   * A Set of keys costs exactly what one key cost.
   *
   * It is a FILTER now, not only a spotlight: `narrowedTo` reads it, so `Drill`
   * and its count follow the selection. What it is NOT is a hide — non-selected
   * groups recede and stay drawn, the same choice the search filter makes and
   * for the reason the region focus records ("a region only means something
   * against its neighbours"). */
  let legendKeys = new Set();

  function clearLegendFocus() {
    if (!legendKeys.size) return;
    legendKeys.clear();
    render();
    refreshDrillButton();
  }

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
    // `level` rides along so the label pass can ask "is this region INSIDE what is
    // focused?" — which is what makes zooming into a country reveal its states.
    regionFocus = { id: regionId, label: (region && region.label) || regionId, rows: rows,
                    level: region ? region.level : 0 };
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

  /* ── The printing highlight: a SET and a FIRST-PRINTED range ─────────────
   *
   * "Show me Reality Fracture" and "show me everything first printed since
   * 2024" are both a row Set — the same SHAPE as `regionFocus` and
   * `queryFocus` — so they join `spotlightFor` and `narrowedTo` rather than
   * growing a third dimming path. It is a HIGHLIGHT, never a hide: a new set's
   * shape in the embedding space only means something against the whole
   * corpus, which is the region focus's argument one more time.
   *
   * TWO DIFFERENT DATES, and confusing them is a confidently wrong filter.
   * `e` is the set of the CORPUS printing; `f` is the date the card FIRST
   * existed across every printing. The corpus printing of Sol Ring is a 2026
   * Commander product, so "printed after 2026-01-01" on the printing date
   * lights Sol Ring. The date bounds read `f` and only `f`.
   *
   * A CARD WITH NO `f` MATCHES NO DATE BOUND. Not 0, not the epoch, not "today":
   * an unknown date is absent, and treating it as any value puts it on one side
   * of every bound, which reads as a measurement. The count of cards excluded
   * for that reason is carried and shown, so the exclusion is visible.
   *
   * Both read `Discovery.index` — `viz_index.json`, positionally aligned with
   * every projection — because no projection carries set or date. That also
   * means switching maps keeps the highlight: the rows do not move. */
  let printFocus = null;    // { set, after, before, rows: Set, undated } | null
  let printControlsReady = null;   // the one-shot promise behind the picker

  const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

  function computePrintFocus(set, after, before) {
    const index = (window.Discovery && Discovery.index) || null;
    if (!index) return null;
    if (after && !ISO_DATE.test(after)) after = '';
    if (before && !ISO_DATE.test(before)) before = '';
    if (!set && !after && !before) return null;
    const rows = new Set();
    let undated = 0;
    for (let i = 0; i < index.length; i++) {
      const rec = index[i];
      if (!rec) continue;
      if (set && rec.e !== set) continue;
      if (after || before) {
        // ISO dates compare correctly as strings, which is why the contract
        // insists on YYYY-MM-DD rather than anything a locale might produce.
        const f = rec.f;
        if (typeof f !== 'string' || !f) { undated++; continue; }
        if (after && f < after) continue;
        if (before && f > before) continue;
      }
      rows.add(i);
    }
    return { set, after, before, rows, undated };
  }

  function printLabel(pf) {
    const bits = [];
    if (pf.set) {
      const cat = setCatalogue && setCatalogue.byCode[pf.set];
      bits.push('in ' + (cat ? cat.name + ' (' + pf.set.toUpperCase() + ')'
                             : pf.set.toUpperCase()));
    }
    if (pf.after && pf.before) bits.push('first printed ' + pf.after + ' to ' + pf.before);
    else if (pf.after) bits.push('first printed on or after ' + pf.after);
    else if (pf.before) bits.push('first printed on or before ' + pf.before);
    return bits.join(', ');
  }

  function applyPrintFocus() {
    // A DISABLED control contributes nothing. An input can still hold a value
    // while disabled, and on an index with no `f` a stray date would narrow the
    // atlas to zero cards through a control that says it is off.
    const read = function (id) {
      const el = document.getElementById(id);
      return el && !el.disabled ? el.value || '' : '';
    };
    printFocus = computePrintFocus(read('setSelect'), read('firstAfter'), read('firstBefore'));
    syncPrintClear();
    render();
    refreshDrillButton();
  }

  /* Clearing restores normal rendering AND the controls, or the toolbar keeps
   * claiming a filter the map has dropped. */
  function clearPrintFocus() {
    if (!printFocus) return false;
    printFocus = null;
    for (const id of ['setSelect', 'firstAfter', 'firstBefore']) {
      const el = document.getElementById(id);
      if (el) el.value = '';
    }
    syncPrintClear();
    render();
    refreshDrillButton();
    return true;
  }

  function syncPrintClear() {
    const b = document.getElementById('printClear');
    if (b) b.hidden = !printFocus;
  }

  /* `sets.json` turns a code into "Reality Fracture (FRA) · 461" and puts the
   * picker newest-first. It is a small sidecar, fetched the first time Explore
   * is reached rather than in the discovery boot — and if it fails, the picker
   * still works on raw codes tallied from the index, which is the one thing
   * the filter actually needs. */
  let setCatalogue = null;  // { order: [code], byCode: {code: {name, released_at, count}} }

  function preparePrintControls() {
    if (printControlsReady) return printControlsReady;
    const index = (window.Discovery && Discovery.index) || null;
    if (!index) return Promise.resolve(false);   // not yet; the next call retries
    const select = document.getElementById('setSelect');
    const after = document.getElementById('firstAfter');
    const before = document.getElementById('firstBefore');
    if (!select) return Promise.resolve(false);

    // WHICH HALF THE INDEX CAN ANSWER. An index written before the set filter
    // shipped carries neither key, and a control that silently matches nothing
    // is indistinguishable from a set with no cards — so it is disabled and
    // says which artifact is behind.
    let hasSet = false, hasDate = false;
    const tally = Object.create(null);
    for (let i = 0; i < index.length; i++) {
      const rec = index[i];
      if (!rec) continue;
      if (typeof rec.e === 'string' && rec.e) {
        hasSet = true;
        tally[rec.e] = (tally[rec.e] || 0) + 1;
      }
      if (!hasDate && typeof rec.f === 'string' && rec.f) hasDate = true;
    }
    const stale = 'viz_index.json predates this filter — re-run `manamap viz-index`';
    select.disabled = !hasSet;
    if (!hasSet) select.title = stale;
    for (const el of [after, before]) {
      if (!el) continue;
      el.disabled = !hasDate;
      if (!hasDate) el.title = stale;
    }
    if (!hasSet) { printControlsReady = Promise.resolve(false); return printControlsReady; }

    const fill = function (cat) {
      // Codes the catalogue knows, in its newest-first order, then any the
      // index carries that it does not — never dropped, labelled raw.
      const order = cat ? cat.order.filter(c => tally[c]) : [];
      const known = new Set(order);
      Object.keys(tally).sort().forEach(c => { if (!known.has(c)) order.push(c); });
      const keep = select.value;
      select.innerHTML = '<option value="">any set</option>' + order.map(function (c) {
        const meta = cat && cat.byCode[c];
        const label = (meta ? meta.name + ' (' + c.toUpperCase() + ')' : c.toUpperCase())
          + ' · ' + tally[c].toLocaleString();
        return '<option value="' + escHtml(c) + '">' + escHtml(label) + '</option>';
      }).join('');
      select.value = keep;
    };
    fill(null);
    printControlsReady = fetch(DATA.sets)
      .then(r => { if (!r.ok) throw new Error('sets ' + r.status); return r.json(); })
      .then(function (obj) {
        // Key order IS the newest-first order the exporter wrote; it is
        // re-sorted here anyway, because "the JSON kept its key order" is a
        // property of one serialiser, not of the format.
        const order = Object.keys(obj).sort(function (a, b) {
          const x = (obj[a] && obj[a].released_at) || '', y = (obj[b] && obj[b].released_at) || '';
          return x < y ? 1 : x > y ? -1 : (a < b ? -1 : 1);
        });
        setCatalogue = { order: order, byCode: obj };
        fill(setCatalogue);
        if (printFocus) render();   // the status line can name the set now
        return true;
      })
      .catch(function () { return true; });   // raw codes stand
    return printControlsReady;
  }

  function initPrintControls() {
    const select = document.getElementById('setSelect');
    if (!select) return;
    select.addEventListener('change', applyPrintFocus);
    for (const id of ['firstAfter', 'firstBefore']) {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', applyPrintFocus);
    }
    const clear = document.getElementById('printClear');
    if (clear) clear.addEventListener('click', clearPrintFocus);
  }
  initPrintControls();

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
    const walk = orderByNearestWalk(rows);
    browseSet = walk
      ? { indices: walk.indices, pos: 0, label: label || 'Selection',
          anchor: null, sims: walk.sims, order: 'walk' }
      : { indices: orderByCentroidDistance(rows), pos: 0, label: label || 'Selection',
          anchor: null, sims: null, order: 'centroid' };
    selectedCards = [];              // browse replaces the 8-card stack, never coexists
    topCardIndex = 0;
    updateViewerPanel();
    updateSelectionHighlight();
    setStatus(rows.length.toLocaleString() + ' cards \u00b7 '
      + (browseSet.order === 'walk'
          ? '\u2190 \u2192 walks them nearest-to-nearest'
          : 'ordered furthest to nearest from the centre')
      + ' \u00b7 Esc to clear');
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

  /* ONE IMAGE-URL BUILDER, and it lives in the shell because the shell is the
   * one module present on all three surfaces — the drawer draws cards on two
   * pages that never load this file. The local form is the fallback for the
   * boot window before `shell.js` has run. */
  function cardImageUrl(name, version) {
    if (window.Shell && Shell.cardImageUrl) return Shell.cardImageUrl(name, version || 'normal');
    return 'https://api.scryfall.com/cards/named?exact='
      + encodeURIComponent(name) + '&format=image&version=' + (version || 'normal');
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

  /* THE CARD BODY, TEXT FIRST. One builder for every panel that shows a card (the
   * selected stack, the browse panel, Discover's landing and Build's selected card).
   *
   * Audit 2026-10-08 at 1440x900: the image came first at 443px, so on a long card the
   * oracle started at the fold — and the image only repeats the name, cost, type and
   * text the panel prints anyway. Keywords repeated the oracle, CMC repeated the cost in
   * the header, every format got a badge, and obsolescence was shown twice ("Compare
   * with" and "Outclassed by" read the same index). The order is now the markup's, in
   * every panel — `#deckInner` no longer reorders it with flex `order`:
   *
   *   [title/stats when the panel has no header] · deck context (Build only) ·
   *   type + oracle (DFC faces labelled) · relations, the outclassed-by comparison and
   *   Keep · the image (240px, click to enlarge, flip for a DFC) · EDHREC, Commander
   *   legality and identity, with every other format in a closed <details>.
   *
   * `row` is required for the relation buttons: the old pair took no argument at all and
   * leaned on `selectedCards`, which is exactly why they did nothing in three of the five
   * panels that render this HTML. `opts.title` / `opts.stats` are for the two `#deckInner`
   * panels, which draw no `.viewer-header`: Build names the card here, Discover already
   * has its `.lens-title`. */
  function buildCardDetailHtml(d, row, opts) {
    opts = opts || {};
    let html = '';
    if (opts.title || opts.stats) {
      html += '<div class="detail-head">' +
        (opts.title ? '<div class="detail-name">' + escHtml(d.n) + '</div>' : '') +
        '<div class="viewer-quickstats">' + quickStatsHtml(d) + '</div></div>';
    }
    html += deckContextHtml(d, row);
    html += cardTextHtml(d);
    // After the rules text, before the relations: what the card goes infinite with is a
    // fact about the card, read before "what is like it".
    html += comboLinesHtml(d, row);
    html += buildRelationHtml(row);
    // Not inside buildRelationHtml: that returns nothing until the neighbour table is in,
    // and the comparison must not wait on it. It still renders directly under the
    // relation buttons, so obsolescence has ONE place in the panel.
    html += buildObsolescenceHtml(d.n);
    html += cardActionsHtml(row);
    html += cardImageHtml(d);
    html += cardFactsHtml(d, { format: opts.format });
    return html;
  }

  /* Type and oracle. A multi-face card (`A // B`, 891 in the corpus) gets one block per
   * face, labelled — they used to be joined with an unlabelled <br><br>. Faces are
   * aligned by ` // ` in name, type and oracle; 17 cards have a face with no rules text,
   * so the oracle has fewer parts than the name and cannot be assigned to a face — those
   * keep their face headings and print the text once, unassigned, rather than guess. */
  function cardTextHtml(d) {
    const faces = String(d.n || '').split(' // ');
    if (faces.length < 2) {
      let html = d.t ? '<div class="detail-type">' + escHtml(d.t) + '</div>' : '';
      if (d.o) {
        html += '<div class="detail-section detail-text">' +
          '<div class="detail-section-title">Oracle Text</div>' +
          '<div class="detail-oracle">' + escHtml(d.o) + '</div></div>';
      }
      return html;
    }
    const types = String(d.t || '').split(' // ');
    const texts = d.o ? String(d.o).split(' // ') : [];
    const costs = String(d.mc || '').split(' // ');
    const aligned = texts.length === faces.length;
    let html = '<div class="detail-section detail-text detail-faces">' +
      '<div class="detail-section-title">Oracle Text</div>';
    faces.forEach(function (face, i) {
      const label = i === 0 ? 'Front' : i === 1 ? 'Back' : 'Face ' + (i + 1);
      html += '<div class="detail-face">' +
        '<div class="detail-face-label">' + label + ' — ' + escHtml(face) +
        (costs.length === faces.length && costs[i]
          ? ' <span class="detail-face-cost">' + renderManaSymbols(costs[i]) + '</span>' : '') +
        '</div>';
      if (types[i]) html += '<div class="detail-type">' + escHtml(types[i]) + '</div>';
      if (aligned && texts[i]) html += '<div class="detail-oracle">' + escHtml(texts[i]) + '</div>';
      html += '</div>';
    });
    if (!aligned && d.o) {
      html += '<div class="detail-oracle">' + escHtml(d.o) + '</div>' +
        '<div class="lens-note">One face has no rules text; the export does not say which.</div>';
    }
    return html + '</div>';
  }

  /* The image, AFTER the text and the actions: 240px, click to see it full width. A
   * multi-face card gets a flip: Scryfall's `named` endpoint takes `face=back` for a card
   * with a printed back. The projection carries no `layout`, so whether one exists is
   * learned by asking — a split, adventure or flip card prints both halves on ONE face,
   * Scryfall refuses `face=back`, and `cardImageError` puts the front back and says so.
   * Never lazy-loaded: the only card image we render is the open one. */
  function cardImageHtml(d) {
    const faces = String(d.n || '').split(' // ');
    const front = cardImageUrl(d.n);
    let html = '<div class="detail-card-image">';
    html += '<img src="' + escHtml(front) + '" alt="' + escHtml(d.n) + '"' +
      ' data-front="' + escHtml(front) + '"' +
      (faces.length > 1 ? ' data-retry="' + escHtml(cardImageUrl(faces[0])) + '"' : '') +
      ' title="Click to enlarge" onclick="MM.toggleCardImage(this)" onerror="MM.cardImageError(this)">';
    if (faces.length > 1) {
      html += '<button class="lens-btn detail-flip" data-back="' +
        escHtml(cardImageUrl(faces[0]) + '&face=back') + '" onclick="MM.flipCard(this)"' +
        ' title="Show the other face">⇄ Back — ' + escHtml(faces[1]) + '</button>';
    }
    return html + '</div>';
  }

  function toggleCardImage(img) {
    const box = img && img.closest('.detail-card-image');
    if (!box) return;
    const full = box.classList.toggle('is-full');
    img.title = full ? 'Click to shrink' : 'Click to enlarge';
  }

  function flipCard(btn) {
    const box = btn && btn.closest('.detail-card-image');
    const img = box && box.querySelector('img');
    if (!img) return;
    const name = img.getAttribute('alt') || '';
    const faces = name.split(' // ');
    const toBack = img.getAttribute('data-face') !== 'back';
    img.setAttribute('data-face', toBack ? 'back' : 'front');
    img.src = toBack ? btn.getAttribute('data-back') : img.getAttribute('data-front');
    btn.textContent = '⇄ ' + (toBack ? 'Front — ' + faces[0] : 'Back — ' + (faces[1] || ''));
  }

  /* Three failures, three answers. The back face refused: this card has no separate
   * back image, so show the front and retire the flip. The full `A // B` name 404s for
   * some DFCs (see `Shell.cardImageUrl`): retry the front face once. Anything else is a
   * missing image, said in words. */
  function cardImageError(img) {
    const box = img && img.closest('.detail-card-image');
    if (!box) return;
    const flip = box.querySelector('.detail-flip');
    if (img.getAttribute('data-face') === 'back') {
      img.setAttribute('data-face', 'front');
      img.src = img.getAttribute('data-front');
      if (flip) {
        flip.disabled = true;
        flip.textContent = 'both faces are on this image';
        flip.title = 'Scryfall has no separate back image for this card';
      }
      return;
    }
    const retry = img.getAttribute('data-retry');
    if (retry && img.getAttribute('data-retried') !== '1') {
      img.setAttribute('data-retried', '1');
      img.setAttribute('data-front', retry);
      img.src = retry;
      return;
    }
    img.onerror = null;
    box.innerHTML = '<div class="detail-image-fallback">Image not available</div>';
  }

  /* ── Card flags: bans per format and the Game Changer list ──────────────
   *
   * `card_flags.json` is `{as_of, game_changers:[names], banned:{fmt:[names]}, counts}`,
   * 12 KB, fetched at boot (`loadCardFlags`, below the projection fetch). Held as Sets
   * keyed by the full `A // B` name, which is `viz_index`'s `n`. A failed fetch leaves
   * the sets EMPTY and `as_of` null — the feature is silently off and every card reads as
   * it did before the file existed, which is the right failure for a 12 KB nicety. */
  const EMPTY_FLAGS = { gc: new Set(), banned: {}, as_of: null };
  let cardFlagsDoc = EMPTY_FLAGS;

  function cardFlags() { return cardFlagsDoc; }

  function loadCardFlags() {
    return fetch(DATA.cardFlags)
      .then(r => (r.ok ? r.json() : null))
      .then(function (doc) {
        if (!doc) return;
        const banned = {};
        Object.keys(doc.banned || {}).forEach(f => { banned[f] = new Set(doc.banned[f] || []); });
        cardFlagsDoc = { gc: new Set(doc.game_changers || []), banned, as_of: doc.as_of || null };
      })
      .catch(function () { /* feature off */ });
  }

  /* ── The combo index, lazy ────────────────────────────────────────────────
   *
   * `combo_index.json`: `{meta:{per_card, combos, indexed}, combos:[[id, [names], infinite,
   * bracket|null, mana_value_needed], …], by_card:{name:{n, inf, top:[indices]}}}`. ~1 MB
   * gzipped, so it waits for the first panel that needs it and is fetched ONCE: the
   * promise is kept, not the fact of having asked, so a second panel opening during the
   * fetch shares it. Resolves to null on any failure — the block simply never appears. */
  let comboIndexPromise = null;
  let comboIndexDoc = null;

  function comboIndex() {
    if (!comboIndexPromise) {
      comboIndexPromise = fetch(DATA.comboIndex)
        .then(r => (r.ok ? r.json() : null))
        .then(function (doc) { comboIndexDoc = doc || null; return comboIndexDoc; })
        .catch(function () { return null; });
    }
    return comboIndexPromise;
  }

  /* WHAT THIS CARD GOES INFINITE WITH — two sources, one block, deck first.
   *
   * (a) The open deck's `combos.json` (Build fetches it with the deck): every `included`
   * or `near` entry naming this card, labelled "in this deck" or "one card short: X".
   * Synchronous, so the deck's own lines never wait on the 1 MB index. (b) The corpus
   * index's `by_card[name].top`, deduped against (a) by combo id, appended when the
   * promise resolves. THE PANEL MAY HAVE MOVED ON by then: the slot is found by
   * `data-card` in the live DOM, so a panel now showing another card gets nothing
   * appended and a re-render of the same card gets its fresh slot filled.
   *
   * A partner chip is `.is-in-deck` when `Build.hasCard` says so; outside Build nothing
   * is in any deck and the block says why instead of lighting nothing. Renders NOTHING —
   * no header — for a card with no lines in either source. */
  function comboLinesHtml(d, row) {
    const name = d && d.n;
    if (!name) return '';
    const deckDoc = (window.Build && Build.deckCombos) ? Build.deckCombos() : null;
    const deckLines = [];
    const seen = new Set();
    const forRows = (rows, kind) => (rows || []).forEach(function (e) {
      const cards = e.cards || [];
      if (cards.indexOf(name) === -1 || seen.has(e.id)) return;
      seen.add(e.id);
      deckLines.push(comboLineHtml(name, {
        id: e.id, cards, infinite: !!e.infinite, bracket: e.bracket,
        banned: !!e.banned || e.bracket == null,
        assumes: !!e.assumes_other_commander,
        kind, missing: [].concat(e.missing || []),
      }));
    });
    if (deckDoc) { forRows(deckDoc.included, 'deck'); forRows(deckDoc.near, 'near'); }

    let indexHtml;
    if (comboIndexDoc) {
      indexHtml = comboIndexLinesHtml(name, comboIndexDoc, seen);
    } else {
      // Not loaded yet (or never will be): leave a slot and ask for the index.
      indexHtml = '<span class="combo-index-slot" data-card="' + escHtml(name) + '"></span>';
      comboIndex().then(function (doc) {
        if (!doc) return;
        document.querySelectorAll('.combo-index-slot[data-card]').forEach(function (slot) {
          const n = slot.getAttribute('data-card');
          const block = slot.closest('.combo-block');
          const ids = new Set(block ? Array.from(block.querySelectorAll('.combo-line[data-id]'))
            .map(el => el.getAttribute('data-id')) : []);
          const html = comboIndexLinesHtml(n, doc, ids);
          if (block) { slot.outerHTML = html; return; }
          // A placeholder with no block: the deck had nothing, so the block is born here —
          // or not at all, when the index has nothing either.
          slot.outerHTML = html ? comboBlockHtml(n, '', html) : '';
        });
      });
    }
    if (!deckLines.length && !indexHtml) return '';
    if (!deckLines.length && comboIndexDoc == null) return indexHtml;   // the bare slot
    return comboBlockHtml(name, deckLines.join(''), indexHtml);
  }

  function comboBlockHtml(name, deckHtml, indexHtml) {
    const inBuild = currentMode === 'build' && window.Build && Build.deckSlug;
    return '<div class="combo-block" data-card="' + escHtml(name) + '">' +
      '<div class="detail-section-title combo-title" title="' + (inBuild
        ? 'Partners you run are lit'
        : 'open a deck in Build to see which partners you run') + '">Combos' +
      (inBuild ? '' : ' <span class="combo-hint">open a deck in Build to see which partners you run</span>') +
      '</div>' + deckHtml + indexHtml + '</div>';
  }

  /* The index's lines for one card, skipping ids the deck already drew, plus the count
   * line. '' when the index does not know the card. */
  function comboIndexLinesHtml(name, doc, skipIds) {
    const entry = doc && doc.by_card && doc.by_card[name];
    if (!entry) return '';
    let html = '';
    let shown = 0;
    (entry.top || []).forEach(function (i) {
      const c = doc.combos[i];
      if (!c || skipIds.has(c[0])) return;
      shown += 1;
      html += comboLineHtml(name, { id: c[0], cards: c[1], infinite: !!c[2], bracket: c[3],
                                    banned: c[3] == null, kind: 'index' });
    });
    const total = entry.n || 0;
    html += '<div class="combo-count">in ' + total.toLocaleString() + ' known combo' +
      (total === 1 ? '' : 's') + ' (' + (entry.inf || 0).toLocaleString() + ' infinite) — ' +
      (shown + skipIds.size) + ' shown</div>';
    return html;
  }

  function comboLineHtml(name, c) {
    const has = (window.Build && Build.hasCard) ? n => Build.hasCard(n) : () => false;
    const partners = (c.cards || []).filter(n => n !== name);
    let html = '<div class="combo-line is-' + c.kind + '" data-id="' + escHtml(String(c.id)) + '">';
    if (c.kind === 'deck') html += '<span class="combo-tag">in this deck</span>';
    if (c.kind === 'near') {
      html += '<span class="combo-tag combo-near">one card short: ' +
        escHtml(c.missing.join(', ') || '?') + '</span>';
    }
    html += partners.map(n => '<span class="combo-partner' + (has(n) ? ' is-in-deck' : '') +
      '" title="' + escHtml(n) + '">' + escHtml(n) + '</span>').join('');
    if (c.infinite) html += '<span class="inf-badge" title="infinite">∞</span>';
    if (c.banned) {
      html += '<span class="bracket-pill banned" title="uses a banned card">banned</span>';
    } else if (c.bracket != null) {
      html += '<span class="bracket-pill" title="Spellbook bracket ' + escHtml(String(c.bracket)) +
        '">B' + escHtml(String(c.bracket)) + '</span>';
    }
    if (c.assumes) html += '<span class="combo-note">assumes its own commander</span>';
    html += '<a class="combo-link" href="https://commanderspellbook.com/combo/' +
      encodeURIComponent(String(c.id)) + '/" target="_blank" rel="noopener" title="Commander Spellbook">↗</a>';
    return html + '</div>';
  }

  /* EDHREC, the deck's format and identity on one line; every other format collapsed.
   *
   * Three states per format, from two sources. `card_flags.json` says BANNED — it is
   * keyed by name and lands at boot, so a ban shows even on the slim `viz_index` record
   * Discover paints from before the projection arrives. `d.f` says LEGAL — but only in
   * the projection record; in the slim record `f` is the first-printed DATE, and reading
   * a date as a format list would call every card "not legal". So the legal / not-legal
   * half waits for the projection's record (`x` is its own field) while the ban does
   * not. "Not legal" is the quiet grey remainder: not on the ban list and not on the
   * legal list, which for a Commander deck is mostly cards from the wrong era.
   *
   * `opts.format` is the deck's format (Build passes `Build.deckFormat()`); the atlas
   * and Discover default to commander. */
  function cardFactsHtml(d, opts) {
    opts = opts || {};
    const fmt = opts.format || 'commander';
    const full = d.x != null;
    const flags = cardFlags();
    const legal = full ? new Set(d.f ? String(d.f).split(',') : []) : null;
    const stateOf = f => (flags.banned[f] && flags.banned[f].has(d.n)) ? 'banned'
                       : !legal ? null : legal.has(f) ? 'legal' : 'not';
    const label = f => f.charAt(0).toUpperCase() + f.slice(1);
    const parts = [];
    if (d.er != null) parts.push('EDHREC #' + Number(d.er).toLocaleString());
    const lead = stateOf(fmt);
    if (lead === 'banned') {
      parts.push('<span class="legal-banned" title="On the ' + escHtml(label(fmt)) +
        ' ban list' + (flags.as_of ? ' as of ' + escHtml(flags.as_of) : '') + '">' +
        escHtml(label(fmt)) + ': BANNED</span>');
    } else if (lead === 'legal') {
      parts.push('<span class="legal-yes">' + escHtml(label(fmt)) + ': legal</span>');
    } else if (lead === 'not') {
      parts.push('<span class="legal-not">' + escHtml(label(fmt)) + ': not legal</span>');
    }
    let formats = '';
    if (legal) {
      formats = '<details class="detail-formats-more"><summary>Other formats</summary>' +
        '<div class="detail-formats">' + ALL_FORMATS.filter(f => f !== fmt).map(function (f) {
          const s = stateOf(f);
          return '<span class="format-badge' + (s === 'legal' ? ' legal' : s === 'banned' ? ' banned' : '') +
            '" title="' + escHtml(label(f)) + ': ' + (s === 'not' ? 'not legal' : s) + '">' +
            escHtml(f) + '</span>';
        }).join('') + '</div></details>';
    }
    if (d.ci) parts.push('Identity ' + escHtml(String(d.ci).replace(/,\s*/g, '')));
    if (!parts.length && !formats) return '';
    return '<div class="detail-facts">' + parts.join(' · ') + '</div>' + formats;
  }

  /* THE CARD IN THE DECK BUILD HAS OPEN — Build only, and only with a deck loaded.
   *
   * Before this, Build with a deck loaded said nothing about the deck in the card panel
   * except "+ Deck" / "In Deck" (which also counts the library). The facts come from
   * `Build.cardContext`, which reads the same role table, commander identity and watch
   * set the rest of Build draws from; Watch / Pass go through `Build.markFromPanel`,
   * which is the review grid's own guarded write queue — there is no second write path. */
  function deckContextHtml(d, row) {
    if (currentMode !== 'build' || !window.Build || !Build.cardContext) return '';
    if (typeof row !== 'number' || row < 0) return '';
    const c = Build.cardContext(row);
    if (!c) return '';
    let html = '<div class="deck-ctx" data-row="' + row + '" data-card="' + escHtml(d.n) + '">';
    html += '<div class="deck-ctx-head"><span class="deck-ctx-deck" title="' + escHtml(c.deckName) +
      '">In ' + escHtml(c.slug) + '</span><span class="deck-ctx-status' +
      (c.inDeck ? ' is-in' : '') + '">' +
      (c.isCommander ? '★ Commander' : c.inDeck ? 'In the 99' + (c.qty > 1 ? ' ×' + c.qty : '')
                                               : 'Not in the 99') + '</span></div>';
    html += '<div class="deck-ctx-row"><span class="deck-ctx-k">roles</span>' +
      (c.roles.length
        ? c.roles.map(r => '<span class="deck-ctx-role' + (r.primary ? ' is-primary' : '') + '"' +
            (r.primary ? ' title="counted in the role budget as ' + escHtml(r.family) + '"' : '') +
            '><span class="lens-swatch" style="background:' + escHtml(r.colour) + '"></span>' +
            escHtml(r.family) + (r.sub ? '<span class="deck-ctx-sub">: ' + escHtml(r.sub) + '</span>' : '') +
            '</span>').join('')
        : '<span class="deck-ctx-none">' + escHtml(c.family === 'land' ? 'land' : 'no role pattern matches') +
          '</span>') + '</div>';
    const col = c.colour;
    html += '<div class="deck-ctx-row"><span class="deck-ctx-k">colour identity</span>' +
      (!col.checked ? '<span class="deck-ctx-none">no commander to check against</span>'
        : col.off.length
          ? '<span class="deck-ctx-bad">off-colour: ' + escHtml(col.off.join('')) + ' outside ' +
            escHtml(col.deck.join('')) + '</span>'
          : '<span class="deck-ctx-ok">fits ' + escHtml(col.deck.join('')) + '</span>') + '</div>';
    const gcRow = gameChangerRowHtml(c);
    if (gcRow) html += gcRow;
    const w = c.watch;
    if (w) {
      const nameArg = escHtml(JSON.stringify(d.n));
      const verdict = w.verdict === 'watching' ? '★ watching' : w.verdict === 'pass' ? 'passed' : 'unreviewed';
      html += '<div class="deck-ctx-watch v-' + escHtml(w.verdict) + '">' +
        '<div class="deck-ctx-row"><span class="deck-ctx-k" title="' + escHtml(w.set) + '">on watch</span>' +
          '<span class="deck-ctx-axis">' + escHtml(w.axis) + '</span>' +
          (w.pays ? '<span class="deck-ctx-pays">' + escHtml(w.pays) + '</span>' : '') +
          '<span class="deck-ctx-verdict">' + (w.pending ? 'saving…' : verdict) + '</span></div>' +
        '<p class="deck-ctx-why">' + escHtml(w.why) + '</p>' +
        (w.note ? '<p class="deck-ctx-note">' + escHtml(w.note) + '</p>' : '');
      if (c.canWrite) {
        const act = (v, label) => '<button class="lens-btn deck-ctx-act' + (w.verdict === v ? ' is-on' : '') +
          '" data-verdict="' + v + '" onclick="Build.markFromPanel(' + nameArg + ',\'' + v + '\')">' +
          label + '</button>';
        html += '<div class="deck-ctx-actions">' + act('watching', 'Watch') + act('pass', 'Pass') +
          (w.verdict !== 'unreviewed' ? act('unreviewed', 'Undo') : '') + '</div>';
      } else {
        html += '<div class="lens-note">Read-only here — run <code>manamap serve</code> to mark cards</div>';
      }
      html += '</div>';
    }
    return html + '</div>';
  }

  /* The Game Changer row of the deck block. Nothing for a card that is not one; for a
   * GC, where the deck stands against its bracket's allowance — the count and floor are
   * `bracket_report.json`'s (Build reads it), the limit per bracket is Build's mirror of
   * `config.BRACKETS`. With no report there is still the fact that it IS one. */
  function gameChangerRowHtml(c) {
    if (!c.gameChanger) return '';
    const b = c.bracket;
    const ord = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th'
      : n % 10 === 1 ? 'st' : n % 10 === 2 ? 'nd' : n % 10 === 3 ? 'rd' : 'th');
    let text;
    if (!b) {
      text = 'no bracket report for this deck';
    } else if (c.inDeck) {
      text = b.gcLimit == null
        ? 'this deck runs ' + b.gcCount + '; no limit at bracket ' + b.floor
        : 'this deck runs ' + b.gcCount + ' of ' + b.gcLimit + ' allowed at bracket ' + b.floor;
    } else if (b.gcLimit == null) {
      text = 'no limit at bracket ' + b.floor;
    } else if (b.gcCount + 1 > b.gcLimit) {
      text = 'would be the ' + ord(b.gcCount + 1) + ': floor moves to ' + b.floorIfAdded;
    } else {
      text = 'would be the ' + ord(b.gcCount + 1) + ' of ' + b.gcLimit + ' allowed at bracket ' + b.floor;
    }
    return '<div class="deck-ctx-row deck-ctx-gc"><span class="deck-ctx-k">bracket</span>' +
      gcPillHtml(c.name) + '<span class="deck-ctx-gc-text">Game Changer · ' + escHtml(text) + '</span></div>';
  }

  /* Repaint every deck-context block for one card, wherever it is drawn. Build calls
   * this when a mark is queued and when it lands, so the panel and the grid's tile can
   * never disagree about a verdict — and `#detailInner`, which Build does not own, is
   * patched in place rather than re-rendered (a re-render would reset its scroll). */
  function refreshDeckContext(name) {
    document.querySelectorAll('.deck-ctx[data-row]').forEach(function (el) {
      if (name && el.getAttribute('data-card') !== name) return;
      const row = Number(el.getAttribute('data-row'));
      const d = cardRecord(row);
      const html = d ? deckContextHtml(d, row) : '';
      el.outerHTML = html;
    });
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
    html += '<div class="viewer-quickstats">' + quickStatsHtml(d) + '</div>';
    html += '</div>';
    return html;
  }

  /* Cost \u00b7 P/T (or loyalty, or defense) \u00b7 rarity. The header's line, and the
   * `#deckInner` panels' too, so the two cannot drift the way the two headers did. */
  function quickStatsHtml(d) {
    // Joined, not prefixed: a land or a slim record has no cost, and a line that opens
    // on its own divider reads as something missing.
    const parts = [];
    if (d.mc) parts.push(renderManaSymbols(d.mc));
    if (d.p != null && d.th != null) {
      parts.push('<strong>' + escHtml(d.p) + '/' + escHtml(d.th) + '</strong>');
    } else if (d.l != null) {
      parts.push('<strong>Loyalty: ' + escHtml(String(d.l)) + '</strong>');
    } else if (d.d != null) {
      parts.push('<strong>Defense: ' + escHtml(String(d.d)) + '</strong>');
    }
    if (d.r) {
      const rc = ['mythic', 'rare', 'uncommon', 'common'].indexOf(d.r) !== -1 ? d.r : '';
      parts.push('<span class="rarity-pill ' + rc + '">' + escHtml(d.r) + '</span>');
    }
    // The Game Changer pill rides this line because this line is drawn everywhere a
    // card is named: the atlas header, Discover's stats and Build's selected card.
    const gc = gcPillHtml(d.n);
    if (gc) parts.push(gc);
    return parts.join('<span class="stat-divider">\u00b7</span>');
  }

  /* The WotC Game Changer pill, or '' \u2014 Build's review tiles draw the same one. */
  function gcPillHtml(name) {
    if (!cardFlags().gc.has(name)) return '';
    return '<span class="gc-pill" title="WotC Game Changer \u2014 bracket 3 allows three">GC</span>';
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
    // THREE ORDERINGS, THREE LABELS. This was a two-way branch, and the comment
    // above it says the label is "the only thing making this browsable" — so a
    // third ordering with no third branch would have the panel state, in
    // confident prose, a sequence the cards are not in.
    html += '<span class="browse-order-label">' + (nb
      ? (browseSet.pos === 0
          ? 'the anchor · ← → walks its ' + (n - 1) + ' nearest · Enter re-anchors here'
          : 'nearest → furthest from ' + escHtml(allData[browseSet.anchor].n) +
            (browseSet.sims ? ' · cosine ' + browseSet.sims[browseSet.pos].toFixed(3) : '') +
            ' · Enter re-anchors here')
      : browseSet.order === 'walk'
        ? 'a walk through the set · each step is the nearest card not yet seen' +
          (browseSet.sims && browseSet.pos > 0
            ? ' · cosine ' + browseSet.sims[browseSet.pos].toFixed(3) + ' from the last'
            : ' · starting at the most typical')
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
      keepBrowseCardInView();
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
  /* Pan only when the card has walked off screen.
   *
   * Re-centring every step makes the atlas move constantly under a reader who is
   * trying to hold a mental picture of where the set sits; never moving loses
   * the marker entirely once the walk leaves the viewport. Proximity ordering is
   * what makes "only when needed" cheap — consecutive cards are near each other,
   * so most steps need no pan at all.
   */
  function keepBrowseCardInView() {
    if (!mapCanvas || !browseSet) return;
    const cam = mapCanvas.getCamera();
    if (!cam) return;
    const p = cardPosition(browseSet.indices[browseSet.pos]);
    if (!p) return;
    /* NORMALISE THE BOUNDS. `getCamera().y` comes back DESCENDING — screen y
     * grows downward, so a fitted view reports something like [8.2, -40.1].
     * Comparing against them unordered makes every point fail the test, so the
     * first version panned on every single step: the "always re-centre"
     * behaviour, arrived at by accident. Caught by the test asserting the
     * opposite. */
    const x0 = Math.min(cam.x[0], cam.x[1]), x1 = Math.max(cam.x[0], cam.x[1]);
    const y0 = Math.min(cam.y[0], cam.y[1]), y1 = Math.max(cam.y[0], cam.y[1]);
    const w = x1 - x0, h = y1 - y0;
    // A margin, so a card sitting exactly on the edge counts as off screen —
    // technically visible and practically invisible are different things.
    const mx = w * 0.08, my = h * 0.08;
    if (p[0] > x0 + mx && p[0] < x1 - mx && p[1] > y0 + my && p[1] < y1 - my) return;
    // Keep the camera's own y orientation rather than imposing ascending order,
    // or the map flips vertically on the first pan.
    const flip = cam.y[0] > cam.y[1];
    mapCanvas.setCamera({
      x: [p[0] - w / 2, p[0] + w / 2],
      y: flip ? [p[1] + h / 2, p[1] - h / 2] : [p[1] - h / 2, p[1] + h / 2],
    }, { animate: true });
  }

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
    // The ORDER rides in the key. Without it two different orderings of the same
    // set with the same endpoints hash identically, and the full-set trace is
    // skipped on a change that really did move every point.
    return 'b:' + ix.length + ':' + ix[0] + ':' + ix[ix.length - 1] +
           ':' + (browseSet.order || '-') +
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
    // THE CACHE KEY IS THE SPACE. It was the literal string 'function' for one
    // space; with two, a fixed key hands the toggle the previous space's matrix
    // out of cache and every "different" neighbour is the same neighbour. The
    // bare `embeddings` variable is the hot path, so it has to be invalidated on
    // switch too — see `setSpace`.
    if (embeddings) return true;
    const key = currentSpace;
    if (embeddingsCache[key]) {
      embeddings = embeddingsCache[key];
      return true;
    }
    try {
      const r = await fetch(similarityEmbeddings());
      if (!r.ok) return false;
      const buf = await r.arrayBuffer();
      // The .bin is headerless: nothing in it says how many rows or dims it
      // holds, so a truncated or wrong-dimension file parses fine and every
      // offset is silently wrong. This is the only place that can notice.
      if (allData && allData.length && buf.byteLength !== allData.length * EMBED_DIM * 4) {
        console.error('[MM] ' + key + ' embeddings are ' + buf.byteLength +
          ' bytes, expected ' + (allData.length * EMBED_DIM * 4) +
          ' (' + allData.length + ' cards x ' + EMBED_DIM + ' dims x 4). Refusing it.');
        return false;
      }
      embeddings = new Float32Array(buf);
      embeddingsCache[key] = embeddings;
      return true;
    } catch (e) {
      return false;
    }
  }

  // Switch similarity space. Everything downstream reads through
  // `getEmbeddings()` / `nearestTo()` / `Discovery.neighbours()`, so this is the
  // whole of it — plus the two caches that would otherwise answer out of the old
  // space, which is the failure that makes a toggle look cosmetic.
  async function setSpace(name) {
    if (!SPACES[name] || name === currentSpace) return false;
    currentSpace = name;
    embeddings = null;                       // the hot path, not just the cache
    Discovery.configure({ vizIndex: DATA.vizIndex, neighbours: SPACES[name].neighbours });
    Discovery.resetNeighbours();
    // AND RE-FETCH. Clearing alone leaves `Discovery.isReady()` false forever —
    // nothing re-requests on its own, so the panel goes permanently empty rather
    // than answering out of the old space. Caught by
    // `test_switching_space_changes_the_answer`, which waits for ready.
    await Discovery.loadNeighbours();

    // AND MOVE THE PICTURE WITH IT.
    //
    // Each space carries the projection it laid out, and for one commit NOTHING
    // READ THAT FIELD — the repo's own "a flag the model sets is a claim the
    // model must ACT ON", in JavaScript. The toggle changed every answer and
    // moved nothing on screen, so it read as broken: the atlas stayed on the
    // `ability` projection while cardbert quietly supplied the neighbours.
    //
    // This does NOT reintroduce the defect the split exists to prevent. That one
    // is the MAP driving SIMILARITY — pick the colour+type picture and get
    // same-colour "neighbours". The dependency here runs the other way: the
    // chosen space moves its own picture, and `switchMap` still never touches
    // which space answers.
    const wantedMap = SPACES[name].map;
    if (wantedMap && MAP_CONFIGS[wantedMap] && wantedMap !== currentMap) {
      const sel = document.getElementById('mapSelect');
      if (sel) sel.value = wantedMap;
      await switchMap(wantedMap);
    }
    return true;
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
    const btn = (rel, label, n, title) =>
      '<button class="lens-btn discover-rel' + (n ? '' : ' is-empty') + '"'
      + (n ? ' onclick="MM.relate(' + row + ',\'' + rel + '\')"' : ' disabled')
      + (title ? ' title="' + title + '"' : '')
      + '>' + label + ' <span class="discover-count">' + n + '</span></button>';
    // The synergy caveat rides on its button: as a paragraph under the row it cost a
    // line on every card, for a sentence that is about one of the three buttons.
    return '<div class="discover-relations">'
      + btn('similar', 'Similar', c.similar)
      + btn('synergy', 'Synergy', c.synergy, 'Synergy is a rule-based list of ten, not a '
            + 'ranking — partners are ordered by how played they are.')
      + btn('obsolete', 'Outclassed by', c.obsolete)
      + '</div>';
  }

  /* Keep and Set as commander: what you DO with the card, on one row under its
   * relations. Split from `buildRelationHtml` so the outclassed-by comparison can sit
   * directly under the button it explains. */
  function cardActionsHtml(row) {
    if (typeof row !== 'number' || row < 0) return '';
    if (!window.Discovery || !Discovery.isReady()) return '';
    // The library follows the card, not the mode. Keeping something you found in the atlas
    // is the same act as keeping something you walked to, so the control lives here
    // rather than only in the Discover panel.
    const kept = Session.library.has(row);
    let html = '<div class="detail-actions">';
    html += '<button class="lens-btn discover-keep" onclick="MM.keep(' + row + ')">'
          + (kept ? '✓ In library' : '+ Keep this card') + '</button>';
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
    return html + '</div>';
  }

  // Toggle a card in the library from any panel, then repaint whichever one is showing.
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
    // Through Discovery's shared seed path, which owns newWalk-then-enter and,
    // more importantly, owns the replace-versus-adopt decision. A box-select is
    // unambiguously a REPLACE — you drew a new set and asked to walk it.
    Promise.resolve(Discovery.seedFromRows(rows, label, {}))
      .then(function () {
        setStatus(rows.length.toLocaleString() + ' cards from ' + label +
                  ' — click any card to grow outward.');
      });
  }

  /* THE REPAINT IS A SUBSCRIPTION, NOT A LIST OF MODES THIS FUNCTION KNOWS ABOUT.
   *
   * This used to repaint only in `explore` and `build`. In `discover` it wrote to
   * Session and told nobody, so the strip — which Session notifies — went to 2
   * while the panel's own count sat at whatever it last rendered. Measured, with
   * no reload involved: strip `empty -> 1 -> 2` against a tray frozen at `2`.
   * That is the "it says two cards, only shows one" report.
   *
   * Adding `discover` to the list would have fixed today and left the fourth
   * panel to rediscover it. Every surface that shows a count now subscribes to
   * Session and repaints itself — the same argument `Force.renderPanel` makes
   * about asking `MM.mode` rather than assuming who owns the panel. */
  function keep(row) {
    if (!window.Discovery || !Discovery.isReady()) return;
    Session.library.toggle(row);
  }

  function buildObsolescenceHtml(cardName) {
    if (!obsolescenceIndex) {
      ensureObsolescenceIndex();
      return '<span class="obsolescence-placeholder" data-card="'
        + escHtml(cardName) + '"></span>';
    }
    if (!obsolescenceIndex[cardName]) return '';
    const data = obsolescenceIndex[cardName];
    // `compare_with` since the 2026-08 repair; `obsoleted_by` is the pre-repair
    // key, read so an older index still renders rather than silently showing
    // nothing.
    const rows = (data.compare_with || data.obsoleted_by || [])
      .slice().sort(function (x, y) {
        return (y.strength || 0) - (x.strength || 0);
      });
    if (rows.length === 0) return '';

    // NOT "Obsoleted By". That was a VERDICT about all contexts, published to
    // users on data where 36.5% of pairs failed a purely mechanical check —
    // costs counted as advantages, restrictions unread, illegal cards offered.
    // The data supports a COMPARISON and the pilot supplies the context.
    //
    // ONE PLACE. This was a separate "Compare with" box above the oracle while the
    // "Outclassed by" button below read the same index — two answers to one question,
    // 115px apart. It is now the button's own detail: collapsed under the relation row,
    // carrying what only it shows (the strength, gains and costs of each).
    let html = '<details class="obsolescence-section">';
    html += '<summary class="obsolescence-title">Compare the ' + rows.length +
            ' that outclass it</summary>';
    for (const rep of rows) {
      html += '<div class="obsolescence-item">';
      html += '<span class="obsolescence-name clickable" onclick="MM.selectByName(\'' + escHtml(rep.name).replace(/'/g, "\\'") + '\')">' + escHtml(rep.name) + '</span>';
      // THE STRENGTH LEADS. 0.0 is "these two cards merely sort near each
      // other"; 1.0 is "strictly better, cheaper, no strings". The index stopped
      // claiming a verdict because it could not support one — it publishes the
      // degree and the reader supplies the context a corpus cannot.
      if (typeof rep.strength === 'number') {
        const band = rep.strength >= 0.65 ? 'strong'
                   : rep.strength >= 0.4 ? 'mild' : 'weak';
        html += '<span class="obsolescence-strength ' + band + '" title="' +
                'How strongly this card outclasses the one you are looking at. ' +
                '0 = not at all, 1 = strictly better and cheaper.">' +
                rep.strength.toFixed(2) + '</span>';
      }
      html += '<div class="obsolescence-advantages">';
      // BOTH SIDES, and the costs are marked. A one-sided list is how "discard
      // a card for hexproof" read as an upgrade over unconditional hexproof.
      for (const gain of (rep.gains || rep.advantages || [])) {
        html += '<span class="obsolescence-badge">+ ' + escHtml(gain) + '</span>';
      }
      for (const cost of (rep.costs || [])) {
        html += '<span class="obsolescence-badge cost">\u2212 ' + escHtml(cost) + '</span>';
      }
      for (const n of (rep.narrows || [])) {
        html += '<span class="obsolescence-badge cost">narrower: ' + escHtml(n) + '</span>';
      }
      if (rep.played_more === false) {
        html += '<span class="obsolescence-badge rank">played less</span>';
      }
      html += '</div>';
      html += '</div>';
    }
    html += '</details>';
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
      // Not copied, and that is not an oversight: this array is freshly parsed
      // and never becomes `allData`, so nothing mutates it. The BOOT path is the
      // one that aliased.
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

  // Changing the MAP never changes the SPACE. That asymmetry is the whole point
  // of the split: the projection is a picture, similarity is a question, and
  // reading neighbours out of the colour+type space returned arbitrary
  // same-colour cards (3.05 of 128 effective dimensions, 0.044 recall@10).
  async function switchMap(mapName) {
    if (mapName === currentMap) return;
    currentMap = mapName;
    // Pre-load region data so it's ready when render() builds annotations
    if (showRegionLabels) await loadRegionData(mapName);
    await loadProjection(mapName);
    // Re-apply selection highlight after map switch (positions changed). This is also
    // what re-runs `render()`, and therefore what installs layers holding the NEW
    // coordinates — which is why the reindex below has to come after it.
    updateSelectionHighlight();

    // THE POSITIONS ALL MOVED, SO THE QUADTREE IS NOW WRONG.
    //
    // `applyProjection` mutates x/y on the existing `allData` in place. The tree's
    // signature is layer lengths plus endpoint ids — all identical across a map switch,
    // same 34,322 cards in the same groups in the same order — so it never rebuilds and
    // keeps answering with where the cards were on the OTHER map. Every hover and click
    // on the Abilities map missed, which read as "the card images are broken" because the
    // popup simply never opened.
    //
    // ORDER IS THE WHOLE FIX. `buildTree` copies coordinates out of the LAYER arrays, and
    // `render()` rebuilds those arrays from scratch. Reindexing before the render rebuilds
    // the tree from the outgoing layers, and then `setLayers` computes that same unchanged
    // signature and skips its own rebuild — so the stale positions survive a call whose
    // entire purpose was to remove them. It looks fixed and measures broken.
    //
    // Deliberately NOT solved by making `treeSignature()` position-aware: drill mutates
    // coordinates in place for 90 frames with the endpoints unchanged, so that would
    // rebuild the tree every frame at 23.5 ms a go — the exact cost the cheap signature
    // exists to avoid.
    if (mapCanvas) mapCanvas.reindex();

    // ...and restate which map you are on, because the highlight now goes through
    // `render()`, which writes its own status line and would otherwise leave you looking
    // at the Abilities map being told how many cards are shown. Under Plotly this was an
    // addTraces/deleteTraces pair that touched no status.
    setMapStatus();
  }

  /* Changing the grouping changes a LANGUAGE, not just the map's colours.
   *
   * Every surface that aggregates cards and reports them back — the legend, and Build's
   * segmented curve and role bars — has to repaint together, or the swatch beside a bar
   * means one thing while the identical swatch on the atlas means another. Repainting
   * only the map is what made the curve keep answering in supertypes after the overlay
   * had been switched to roles.
   */
  function regroup() {
    render();
    if (window.Build && typeof Build.renderPanel === 'function') Build.renderPanel();
    /* AND THE GRAPH, which is a third surface this function did not know about.
     * `render()` draws the world map and is inert while the force canvas is up,
     * so in Build's graph view — and in Discover, which is always a graph — the
     * colour control changed the legend and nothing else. The node colour is
     * baked at construction, so the canvas has to be told. */
    if (window.Force && Force.isActive && Force.isActive() && Force.recolour) {
      Force.recolour();
    }
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
  /* Region parentage, indexed once per map. `regions_*.json` already carries `parent` on
   * every entry, so the hierarchy needs no new data — only a lookup. */
  const regionIndexCache = {};
  function regionIndex() {
    if (regionIndexCache[currentMap]) return regionIndexCache[currentMap];
    const data = regionDataCache[currentMap];
    if (!data) return null;
    const byId = {};
    for (const r of data.regions) byId[r.id] = r;
    regionIndexCache[currentMap] = byId;
    return byId;
  }

  /* THE TELESCOPE: which names are on screen is a question about DEPTH, not only zoom.
   *
   * Before this, every level answered from absolute camera span alone. Two consequences.
   * Neighbourhoods (L2) needed span < 6 while their own spans are ~0.6, so in practice
   * they never appeared — 168 of the 227 names on the ability map were unreachable.
   * And focusing a region framed it without naming a single thing inside it, so clicking
   * into a country told you less than standing outside it did.
   *
   * Now the span bands still decide the unfocused case, but a focused region promotes its
   * OWN descendants: its children are always named, its grandchildren once the camera is
   * close enough to tell them apart. Everything outside keeps a faint L0 label so you can
   * still see which country you left — the same reason focusing mutes points instead of
   * hiding them.
   */
  function refreshCanvasLabels() {
    if (!mapCanvas) return;
    const cam = mapCanvas.getCamera();
    const span = cam ? Math.abs(cam.x[1] - cam.x[0]) : 70;
    const data = regionDataCache[currentMap];
    if (!data || !showRegionLabels) { mapCanvas.setAnnotations([]); return; }
    const byId = regionIndex();
    const focusId = regionFocus ? regionFocus.id : null;

    // 0 = the focused region, 1 = child, 2 = grandchild, -1 = elsewhere, null = no focus.
    function depthFromFocus(region) {
      if (!focusId) return null;
      if (region.id === focusId) return 0;
      if (region.parent === focusId) return 1;
      const parent = byId && byId[region.parent];
      if (parent && parent.parent === focusId) return 2;
      return -1;
    }

    const out = [];
    for (const region of data.regions) {
      const depth = depthFromFocus(region);
      let opacity = 0;
      const size = region.level === 0 ? 16 : region.level === 1 ? 11 : 9;

      if (depth === null) {
        // Nothing focused: the plain span bands. L2 fades in far earlier than it used to
        // — its own span is a fraction of a unit, so gating on `region.span` at 4% of the
        // camera meant a neighbourhood had to fill the screen to be allowed a name.
        if (region.level === 0) {
          if (span > 25) opacity = 1;
          else if (span > 15) opacity = (span - 15) / 10;
        } else if (region.level === 1) {
          if (region.span < span * 0.05) continue;
          if (span < 20) opacity = 1;
          else if (span < 30) opacity = (30 - span) / 10;
        } else {
          if (span < 9) opacity = 1;
          else if (span < 16) opacity = (16 - span) / 7;
        }
      } else if (depth === 0) {
        // Where you are. Named, but quieter than its children — it is the title of the
        // view, not a thing to click into again.
        opacity = 0.55;
      } else if (depth === 1) {
        opacity = 1;                                   // what is inside: always named
      } else if (depth === 2) {
        if (span < 10) opacity = 0.9;                  // one level further in
        else if (span < 20) opacity = (20 - span) / 10 * 0.9;
      } else if (region.level === 0) {
        opacity = 0.28;                                // context: the countries you left
      }

      /* An invisible label is worse than no label: placement is greedy and sorted
       * big-first, so a country name fading through 0.03 alpha still claims the largest
       * collision box on screen and suppresses the readable neighbourhood name underneath
       * it. Cut the band off where the text stops being legible rather than where it
       * reaches zero. Measured at span 15.3, where the L0 band sits at 0.03. */
      if (opacity < 0.09) continue;
      out.push({
        x: region.cx, y: region.cy, id: region.id, size: size, level: region.level,
        // The outline has to fade WITH the text. A fixed-alpha dark ring under a 0.28
        // label is more opaque than the label itself, so the faint context names rendered
        // as dark smudges — legible only as "something is wrong there". Scaled here rather
        // than in CSS because CSS cannot see the per-label opacity.
        outline: opacity,
        text: region.level === 0 ? region.label : region.short,
        colour: region.level === 0
          ? 'rgba(196,167,71,' + opacity.toFixed(2) + ')'
          : region.level === 1
            ? 'rgba(232,236,244,' + opacity.toFixed(2) + ')'
            : 'rgba(198,210,232,' + opacity.toFixed(2) + ')',
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
        const on = legendKeys.has(tr.name);
        return '<div class="map-legend-row' + (on ? ' is-active' : '') +
          '" role="button" tabindex="0" data-key="' + escHtml(tr.name) +
          '"><span class="map-legend-dot" style="background:' +
          (c === 'rgba(0,0,0,0)' ? 'transparent;border:2px solid ' +
            ((m.line && m.line.color) || '#888') : c) +
          '"></span>' + escHtml(tr.name) + '</div>';
      }).join('');

    // Bound once on the container, not per row: the rows are replaced on every render,
    // so per-row listeners would be re-attached 34,000-point-render after render.
    if (!el._legendBound) {
      el._legendBound = true;
      el.addEventListener('click', function (ev) {
        const row = ev.target.closest('.map-legend-row');
        if (!row) return;
        const key = row.getAttribute('data-key');
        // ACCUMULATE. Click adds, click again removes, empty means everything.
        if (legendKeys.has(key)) legendKeys.delete(key); else legendKeys.add(key);
        render();
        // It narrows what Drill would re-map, so the button's count has to move
        // with it — the "three predicates agree" contract the search filter set.
        refreshDrillButton();
      });
    }
  }

  // ── Load data ──
  //
  // Two tracks. Discovery is usable on 1.83 MB (viz_index 0.56 + neighbours 1.27, gz) and is the front
  // door; the 2.9 MB projection loads *behind* it and upgrades every record in place —
  // `MM.cardRecord` prefers the full row when it exists and falls back to the slim one.
  // Landing used to mean waiting for the projection before a single pixel appeared.
  const params = new URLSearchParams(window.location.search);
  const wantedDeck = params.get('deck');
  // `?draft=<slug>` is the Workbench's "pick this up" link. Same shape as
  // `?deck=`, different destination: a draft has no 99 to light, so it reopens
  // the new-deck form rather than loading a deck that does not exist yet.
  const wantedDraft = params.get('draft');
  // ?mode=explore deep-links straight to the atlas. Discovery is the front door now, so
  // anything that wants the 34,322-point map — a bookmark, a browser test about
  // rendering — has to ask for it rather than assume it is what boot produces.
  const wantedMode = params.get('mode');
  if (wantedMode) currentMode = wantedMode;

  Discovery.configure({ vizIndex: DATA.vizIndex, neighbours: SPACES[currentSpace].neighbours });
  // Apply the mode chrome BEFORE the data arrives. `currentMode` being 'discover' is not
  // enough on its own — setMode is what hides the Plotly surface and gives the force
  // canvas a size, and without it the canvas measured 0x0, so the landing card's
  // transform resolved to (0,0) and it drew half off-screen behind the toolbar.
  //
  // queueMicrotask, not a direct call: every line in this file runs INSIDE the IIFE whose
  // return value becomes `window.MM`, so the global does not exist yet. Discovery touches
  // MM.setStatus, and calling it here threw — which aborted the IIFE, so MM was never
  // exported and every later file failed at its own top level too. One ordering
  // mistake, four broken files, twice. A microtask runs after the assignment completes.
  /* THE VIEWER PANEL SUBSCRIBES TOO, and forgetting it was the first cost of
   * this change: `keep` used to repaint explore and build directly, so removing
   * that call while wiring subscribers for Discover and Build left the Keep
   * button in Explore reading "+ Keep this card" over a card that was already
   * in the library. Every panel that draws library state listens; none of them
   * is a special case the writer has to know about.
   *
   * Registered in the microtask for the reason above it — subscribing is safe
   * anywhere, but the callback reads `MM.mode`, and the IIFE has not returned. */
  queueMicrotask(function () {
    if (!window.Session || !Session.on) return;
    Session.on(function (what) {
      if (what !== 'library') return;
      if (currentMode === 'explore' && selectedCards.length) updateViewerPanel();
    });
  });

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
      // Booted straight into the atlas (`?mode=explore`): no `setMode` call
      // reaches the picker on that path, so it is prepared here.
      if (currentMode === 'explore') preparePrintControls();
    })
    .catch(err => setStatus('Discovery unavailable: ' + err.message));

  // Bans and Game Changers ride beside `viz_index.json`: 12 KB, and the landing card
  // may be banned. Nothing here touches `MM.*` — the loader writes a module variable.
  loadCardFlags();

  // Boot the map the app actually opens on. This was hardcoded to `default`, so flipping
  // the default to Abilities would have left `currentMap` saying one thing while
  // `allData` held the other map's coordinates — every position wrong and nothing to
  // indicate it.
  fetch(MAP_CONFIGS[currentMap].projection)
    .then(r => r.json())
    .then(data => {
      allData = data;
      // THE CACHE MUST NOT ALIAS `allData`. `applyProjection` writes
      // `allData[i].x = data[i].x`, so caching the same objects meant switching
      // away from the boot map OVERWROTE that map's own cached coordinates —
      // and switching back re-applied them, leaving every card where the other
      // map had put it. `currentMap` said "ability" while the points sat on the
      // cardbert (or colour+type) layout, which is the exact class the comment
      // above this fetch was written about.
      //
      // A COORDINATE-ONLY SNAPSHOT, not a deep clone: `applyProjection` reads
      // nothing but x and y, and copying all 34,890 full records to guard two
      // numbers each would cost megabytes for nothing.
      projectionCache[currentMap] = data.map(d => ({ x: d.x, y: d.y }));
      const sel = document.getElementById('mapSelect');
      if (sel) sel.value = currentMap;
      // Same for the colour mode. Both selects are pinned from the JS defaults rather than
      // left to option order: markup order and a `let` default are two places deciding one
      // thing, and they drift the moment someone reorders the list for readability.
      const colourSel = document.getElementById('colorBy');
      if (colourSel) colourSel.value = currentColorBy;
      // Explore does not get `setMode` at boot — only Discover has its chrome
      // applied before the data lands, deliberately — so the per-mode option
      // list is synced here too, or arriving on ?mode=explore shows an option
      // that mode does not offer.
      syncColourOptions(currentMode);
      initToggles();
      refreshDrillButton();
      // Only paint the scatter if that is what the user is looking at. Rendering 34,322
      // points behind a landing card is work nobody asked for.
      if (currentMode === 'explore') {
        render();
        setStatus(`${allData.length.toLocaleString()} cards loaded`);
      }
      // THE LANDING UPGRADES IN PLACE. Discover painted its card from the slim
      // viz_index record, which has no type, oracle, cost or legality — and the card
      // panel is text-first now, so until something re-rendered, the landing showed a
      // name, the relations and an image. `cardRecord` prefers this row from here on.
      // Not while the pilot is typing into the panel (the seed or paste box): a
      // re-render would throw the text away.
      const typing = document.activeElement && document.activeElement.closest &&
        document.activeElement.closest('#deckInner') &&
        /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
      if (currentMode === 'discover' && window.Discovery && Discovery.isReady() &&
          Discovery.current >= 0 && !typing) {
        Discovery.render();
      }
      // ?deck=<slug> is the map's first inbound deep link — the dossier and the
      // magazine's Back Page both use it. Honour it by entering the Lens, not by
      // dropping the reader on an unfiltered map with a query string they can't see.
      if (wantedDeck || wantedDraft) {
        // `?deck=<slug>` is an inbound contract: the dossier and every published manual
        // link to it. Deck Lens and Build Deck are one mode now, so it lands in Build.
        document.getElementById('modeSelect').value = 'build';
        setMode('build');
        if (wantedDraft && window.Build && Build.resumeDraft) {
          // After `setMode`, which is what loads the manifest the draft is
          // looked up in.
          setTimeout(function () { Build.resumeDraft(wantedDraft); }, 300);
        }
      }
      // Load region data in background, then re-render with labels. `currentMap`, not a
      // hardcoded 'default' — the second place the boot map was written as a literal, and
      // like the projection fetch above it failed silently: `regions_ability.json` was
      // never requested, `refreshCanvasLabels` found no data, and the map simply had no
      // names on it. Nothing errors when the answer is an empty list.
      loadRegionData(currentMap).then(data => {
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

  /* WHAT THE PILOT IS NARROWED TO — which is NOT what is drawn, deliberately.
   *
   * `visible` in `render()` stays supertypes-only, so a search leaves every card
   * on screen and merely dims the ones that did not match: a spotlight keeps
   * context, and "where do treasure cards live" is unanswerable if the rest of
   * the atlas disappears.
   *
   * This is the other question — what set has the pilot actually singled out —
   * and it is what `Drill ⤓` re-maps and what the button counts. The two
   * disagreeing is normal here and was a BUG when it happened by accident
   * (`filtered` fed contours while the group loop re-tested `activeSupertypes`,
   * so a new filter silently drew nothing). Named apart so the next reader does
   * not helpfully collapse them back together.
   */
  function narrowedTo(d, i) {
    if (!activeSupertypes.has(d.s)) return false;
    // The legend narrows too. It cannot contradict the supertype toggles, and
    // the reason is structural rather than lucky: the legend is built from the
    // TRACE LIST, so it only ever lists groups that survived `visible`. A
    // hidden group has no row to click.
    if (legendKeys.size && !legendKeys.has(grouping().keyOf(d))) return false;
    // An EMPTY printing set narrows to nothing — unlike the query, whose empty
    // set means "no term yet". A set and a date range that match no card is an
    // answer, and Drill offering to re-map the whole corpus would contradict it.
    if (printFocus && !printFocus.rows.has(i)) return false;
    if (queryFocus && queryFocus.rows.size) return queryFocus.rows.has(i);
    return true;
  }

  function refreshDrillButton() {
    const btn = document.getElementById('drillFiltered');
    if (!btn || typeof window.Drill === 'undefined') return;
    let n = 0;
    // Follows `passesFilters`, never `visible` — see the note there. A query
    // narrows what Drill would re-map even though every card stays drawn.
    for (let i = 0; i < allData.length; i++) if (narrowedTo(allData[i], i)) n++;
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
    // A SELECTION DOES NOT SURVIVE ITS KEY SPACE. "Red" means nothing once the
    // map is coloured by rarity, and as a filter it would select zero cards
    // rather than merely lighting none — the map going blank with no active row
    // and nothing to click to undo it. (It was already a live bug as a
    // spotlight: everything dimmed to 9% with no row marked.)
    legendKeys.clear();
    // A grouping whose data is not in the boot payload loads here, once. Selecting Role
    // before `card_roles.json` lands would otherwise colour all 34,322 cards
    // 'unclassified' and look like the roles file was wrong rather than absent.
    const g = grouping();
    if (g.ensure) {
      setStatus('Loading ' + g.label.toLowerCase() + ' data…');
      g.ensure().then(ok => {
        if (!ok) setStatus('Could not load ' + g.label.toLowerCase() + ' data');
        else setMapStatus();
        regroup();
      });
    } else {
      regroup();
    }
    render();
  });

  document.getElementById('mapSelect').addEventListener('change', e => {
    switchMap(e.target.value);
  });

  document.getElementById('spaceSelect').addEventListener('change', async e => {
    const name = e.target.value;
    const info = SPACES[name];
    if (!(await setSpace(name))) return;
    setStatus('Similarity: ' + info.label + ' — ' + info.note);
    // A GROWN GRAPH IS AN ANSWER FROM THE OLD SPACE. The nodes were chosen by
    // neighbours that no longer apply, so the walk is left standing rather than
    // silently re-rooted: the user asked to change the question, not to lose
    // their work. What is dropped is the CACHED matrix, not the session.
    if (currentMode === 'explore') {
      await loadEmbeddings();
      render();
      return;
    }
    /* BUILD IS THE CASE THE GUARD ABOVE FORGOT. The reasoning behind leaving a
     * grown graph standing is that its NODES were chosen by the old space, and
     * re-rooting would throw away a walk the user built. Build's node set is not
     * grown — it IS the deck, fixed by the decklist — and only the EDGES come
     * from the space. So there is nothing to protect and everything to redraw,
     * and the control did nothing at all here: same dots, same lines, new label.
     * `Build.reseedGraph` carries the explored cards across, so the walk-shaped
     * half of the worry does not apply either. */
    if (currentMode === 'build' && window.Build && Build.reseedGraph) {
      await loadEmbeddings();
      if (Build.reseedGraph()) {
        setStatus('Similarity: ' + info.label + ' — graph re-linked in the new space');
      }
    }
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

  /* Ambient motion is a preference, so the button reports the renderer's state rather than
   * assuming it: `prefers-reduced-motion` means the map boots still, and a control that
   * showed "on" while nothing moved would read as a broken toggle rather than an honoured
   * system setting. The renderer owns the default; this only ever flips it. */
  // Probe for the local API once at boot. Every agent affordance is gated on
  // the answer, and a page that cannot reach one says so rather than offering a
  // button that does nothing.
  if (window.Api) {
    Api.probe().then(function (ready) {
      if (ready && window.Build && Build.renderPanel && MM.mode === 'build') {
        Build.renderPanel();
        // The review grid's buttons are gated on the API too.
        if (Build.renderGrid) Build.renderGrid();
      }
    });
  }

  for (const tab of document.querySelectorAll('.mode-tab')) {
    tab.addEventListener('click', function () {
      const sel = document.getElementById('modeSelect');
      const mode = this.getAttribute('data-mode');
      if (sel) sel.value = mode;          // keep the one source of truth first
      setMode(mode);
    });
  }

  document.getElementById('toggleMotion').addEventListener('click', function () {
    if (!mapCanvas) return;
    mapCanvas.setMotion(!mapCanvas.motion);
    syncMotionButton();
  });

  /* A function declaration, not `MM.something = ...`: this file's whole top level runs
   * INSIDE the IIFE, before `window.MM` is exported, so touching `MM` here throws, aborts
   * the module, and takes discovery, drill, force and build down with it — one ordering
   * mistake, four broken files. Hoisted, so `initMapCanvas` can call it from above. */
  function syncMotionButton() {
    const b = document.getElementById('toggleMotion');
    if (b && mapCanvas) b.classList.toggle('active', mapCanvas.motion);
  }

  /* ── The search box, which does two different things ────────────────────
   *
   * IN DISCOVER IT WAS DOING NOTHING AT ALL, and that is what this fixes. Typing
   * set `searchTerm` and called the MAP's `render()` — but Discover hides the
   * atlas canvas (`#plot.force-mode`), so the white diamonds it drew landed on a
   * surface nobody could see and the only observable effect was a status line.
   * There was nothing to click.
   *
   * So in Discover the box becomes a PICKER: names ranked out of `Discovery.index`,
   * click one to land on it. In Explore it keeps narrowing the atlas, which is a
   * different question asked with the same control — but the two never both
   * render, and each says which it is.
   */
  function renderSearchResults(term) {
    const box = document.getElementById('searchResults');
    if (!box) return;
    if (currentMode !== 'discover' || !window.Discovery || !Discovery.isReady()
        || term.length < 2) {
      box.hidden = true;
      box.innerHTML = '';
      return;
    }
    const hits = Discovery.searchByName(term, 8);
    if (!hits.length) {
      box.hidden = false;
      box.innerHTML = '<p class="search-none">no card matches that</p>';
      return;
    }
    // "Is there a walk to add to" is read ONCE, so both the row and its `+` agree
    // within a single render — the same reason `Discovery.render` hoists `graphN0`.
    const hasWalk = window.Force ? Force.nodeCount > 0 : false;
    box.hidden = false;
    box.innerHTML = hits.map(function (h) {
      const arg = JSON.stringify(h.row);
      return '<div class="search-hit">'
        + '<button class="search-pick" onclick="MM.searchPick(' + arg + ')">'
        +   escHtml(h.name) + '</button>'
        + (hasWalk
            ? '<button class="search-add" title="Add to the walk instead of starting over"'
              + ' onclick="MM.searchAdd(' + arg + ')">+</button>'
            : '')
        + '</div>';
    }).join('');
  }

  /* Frame the matches from where they actually are.
   *
   * Copied in shape from `focusRegion`, and for its stated reason: the extent
   * comes from the members' real coordinates rather than anything stored, so the
   * camera agrees with what is drawn after the supertype filters have had their
   * say. Skipped for a very large match set — framing 3,361 cards scattered
   * across the atlas is the whole atlas, and animating to it is a jolt that
   * changes nothing. */
  const FRAME_QUERY_MAX = 1500;

  /* Where the camera was before a search moved it. Saved ONCE per search
   * session, not per keystroke — saving on every input would capture the
   * already-framed view and Escape would restore you to the last query's
   * framing rather than to where you were reading. */
  let cameraBeforeQuery = null;

  function frameQuery() {
    if (!queryFocus || !queryFocus.rows.size || !mapCanvas) return;
    if (queryFocus.rows.size > FRAME_QUERY_MAX) return;
    if (!cameraBeforeQuery) cameraBeforeQuery = mapCanvas.getCamera();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of queryFocus.rows) {
      const d = allData[i];
      if (!d) continue;
      if (d.x < x0) x0 = d.x;
      if (d.x > x1) x1 = d.x;
      if (d.y < y0) y0 = d.y;
      if (d.y > y1) y1 = d.y;
    }
    if (!isFinite(x0)) return;
    const padX = Math.max((x1 - x0) * 0.12, 0.5);
    const padY = Math.max((y1 - y0) * 0.12, 0.5);
    mapCanvas.setCamera({ x: [x0 - padX, x1 + padX], y: [y0 - padY, y1 + padY] },
                        { animate: true });
  }

  /* Peeling a query PUTS THE CAMERA BACK. A search frames its matches, so
   * clearing one without restoring leaves the pilot zoomed into a corner with
   * the whole atlas shown again and no idea why — the narrowing is gone but its
   * side effect is not, which reads as the map having jumped on its own. */
  function clearQueryFocus() {
    queryFocus = null;
    searchTerm = '';
    const input = document.getElementById('search');
    if (input) input.value = '';
    closeSearchResults();
    render();
    refreshDrillButton();
    if (cameraBeforeQuery && mapCanvas) {
      mapCanvas.setCamera(cameraBeforeQuery, { animate: true });
    }
    cameraBeforeQuery = null;
  }

  function closeSearchResults() {
    const box = document.getElementById('searchResults');
    if (box) { box.hidden = true; box.innerHTML = ''; }
  }

  /* TWO CONTROLS, TWO ACTS. Clicking the name REPLACES the walk with that card —
   * the explicit request that makes replacement legitimate. `+` GROWS it. A single
   * control that switched between them silently is the bug `seedFromRows`' header
   * records having shipped twice. */
  function searchPick(row) {
    if (window.Discovery) Discovery.startHere(row);
    const input = document.getElementById('search');
    if (input) input.value = '';
    searchTerm = '';
    closeSearchResults();
  }

  function searchAdd(row) {
    if (window.Discovery) Discovery.addToWalk(row);
  }

  document.getElementById('search').addEventListener('input', e => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      searchTerm = e.target.value.trim().toLowerCase();
      renderSearchResults(searchTerm);
      // Explore narrows the atlas; Discover has no atlas on screen to narrow.
      if (currentMode === 'discover') return;
      queryFocus = computeQuery(searchTerm);
      render();
      refreshDrillButton();
      frameQuery();
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
      /* A FILTER IS A SET, AND A SET IS BROWSABLE. Arrows used to give up here
       * when nothing was selected — so after narrowing the atlas to 433 cards
       * the only way to read one was to find its dot and click it. Entering
       * browse over the matches costs no new control: `browseSet` already is an
       * ordered list plus a cursor, and the arrows already drive it. */
      if (!browseSet && selectedCards.length === 0) {
        if (currentMode === 'explore' && queryFocus && queryFocus.rows.size > 1) {
          e.preventDefault();
          enterBrowse(Array.from(queryFocus.rows), '"' + queryFocus.term + '"')
            .then(function () { if (back) cycleSelection(-1); });
          return;
        }
        return;
      }
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
  /* Show only the controls this mode uses.
   *
   * The toolbar carried the same 17 controls in every mode, so Discover — a
   * graph of ONE CARD — showed nine type filters, a density-contour toggle and
   * "Color by", none of which act on anything there. Roughly half the surface
   * was inert at any moment, which is most of why the atlas felt heavy.
   *
   * `data-modes` on a group names where it belongs; no attribute means always.
   * Driven from markup rather than a list here, because a list in this file and
   * the controls in the HTML are two places to remember, and the one that gets
   * forgotten is the one nobody sees fail. */
  function syncToolbar(mode) {
    for (const el of document.querySelectorAll('.toolbar [data-modes]')) {
      el.hidden = !el.getAttribute('data-modes').split(/\s+/).includes(mode);
    }
    for (const tab of document.querySelectorAll('.mode-tab')) {
      const on = tab.getAttribute('data-mode') === mode;
      tab.classList.toggle('is-on', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
    }
  }

  /* WHICH GROUPINGS THE PICKER OFFERS, per mode.
   *
   * Role colouring is Build's language, not the atlas's: on 34,890 cards it paints
   * mostly `unclassified`, and it costs a 0.39 MB lazy fetch to say so. Explore does
   * not offer it.
   *
   * REMOVED FROM THE PICKER, NEVER FROM `MM.GROUPINGS`. `build.js` reads
   * `GROUPINGS.role.order` and `.palette` through `familyPriority()`/`familyColour()`
   * at six call sites, and deleting the key is a `TypeError` at every one — the role
   * bars, their swatches, the segmented curve and the deck's map overlay.
   *
   * And the option cannot simply be deleted either: clicking a role bar in Build calls
   * `MM.focusGroup(key, 'role')`, which does `select.value = 'role'`. With no such
   * option that assignment SILENTLY NO-OPS and the select then disagrees with
   * `currentColorBy` — "two controls disagreeing about one value is how a legend ends
   * up lying", which is the failure `focusGroup`'s own comment warns about. So the
   * option is present in Build and absent in Explore, and entering Explore while it is
   * selected falls back rather than leaving a select showing a value it no longer has.
   */
  const MODE_GROUPINGS = { explore: ['supertype', 'color', 'rarity'] };

  function syncColourOptions(mode) {
    const sel = document.getElementById('colorBy');
    if (!sel) return;
    const allowed = MODE_GROUPINGS[mode];
    for (const opt of sel.options) {
      opt.hidden = !!(allowed && allowed.indexOf(opt.value) === -1);
      opt.disabled = opt.hidden;
    }
    if (allowed && allowed.indexOf(currentColorBy) === -1) {
      currentColorBy = allowed[0];
      sel.value = currentColorBy;
      regroup();
    }
  }

  function setMode(mode) {
    currentMode = mode;
    // `modeSelect` stays the ONE answer to "which mode is current" — it is
    // hidden now, but `?mode=` applies through it and the browser suite reads
    // it. The tabs are a view of it, never a second source of truth.
    const sel = document.getElementById('modeSelect');
    if (sel && sel.value !== mode) sel.value = mode;
    syncToolbar(mode);
    syncColourOptions(mode);
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

    // ARRIVING IN EXPLORE SHOWS THE WHOLE MAP, EVERY TIME.
    //
    // This used to auto-orient: `if (Session.size()) orientTo(null, 'your walk')` — walk a
    // few cards in Discover, switch to Explore, and the atlas opened with 97% of itself
    // dimmed to 8% alpha and the camera somewhere else. The intent was good (locate what
    // you hold) but as an ENTRY state it means the atlas almost never gets to be the atlas,
    // and the dimming reads as a rendering fault rather than as a lens.
    //
    // The lens is not gone — `orientTo` still runs when you ask for it from inside Explore
    // (clicking a card, following a relation). What changed is that arriving is not asking.
    if (mode === 'explore') {
      // The set picker's labels, the first time the atlas is reached. Not
      // awaited: the picker is usable on raw codes the moment the index is in.
      preparePrintControls();
      clearOrientation();
      regionFocus = null;
      legendKeys.clear();
      clearSelection();
      if (mapCanvas) mapCanvas.fitToData();
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
  /* ── The grouping registry ────────────────────────────────────────────
   *
   * ONE definition of "how are cards grouped, in what order, and what colour is each
   * group" — because a colour only carries meaning if it means the same thing on every
   * surface that reports it. Before this there were two taxonomies with no relationship:
   * the map coloured by `COLOR_PALETTE`/`SUPERTYPE_PALETTE`/`RARITY_PALETTE` here, and
   * Build's role-budget bars coloured by a `FAMILY_COLOR` table of its own. Same screen,
   * same cards, two unrelated colour languages and two legends that could never agree.
   *
   * A grouping is `{label, keyOf(d), palette, order, ensure?}`. `order` is authoritative
   * for legend and stack order; `ensure` is for groupings whose data is not in the boot
   * payload. Roles are the only one today: `card_roles.json` is 0.39 MB gzipped against a
   * 1.83 MB discovery boot, so it loads when the grouping is SELECTED and never before.
   */
  const ROLE_ORDER = [
    'wincon', 'doubler', 'tutor', 'counterspell', 'stax', 'hate', 'removal', 'ramp',
    'draw', 'recursion', 'protection', 'sac-outlet', 'payoff', 'land', 'value', 'buff',
    'utility', 'sac-cost', 'threat', 'unclassified',
  ];
  const ROLE_PALETTE = {
    wincon: '#FFD700', doubler: '#22D3EE', tutor: '#E879F9', counterspell: '#38BDF8',
    stax: '#94A3B8', hate: '#CBD5E1', removal: '#EF4444', ramp: '#22C55E',
    draw: '#3B82F6', recursion: '#A78BFA', protection: '#FDE68A', 'sac-outlet': '#FB923C',
    payoff: '#EC4899', land: '#A16207', value: '#14B8A6', buff: '#84CC16',
    utility: '#64748B', 'sac-cost': '#78716C', threat: '#F59E0B', unclassified: '#6B7280',
  };

  let rolesByName = null;   // card_roles.json .roles — lazy, see GROUPINGS.role.ensure

  // Lands fall back to their supertype: ROLE_PATTERNS does not classify every card, and a
  // Land is a land whatever else is true of it. Everything else stays unclassified rather
  // than being given a role the roles file never claimed.
  function roleOf(d) {
    const roles = (rolesByName && rolesByName[d.n]) || [];
    if (roles.length) {
      const families = new Set(roles.map(r => r.split(':')[0]));
      for (const f of ROLE_ORDER) if (families.has(f)) return f;
    }
    return d.s === 'Land' ? 'land' : 'unclassified';
  }

  const GROUPINGS = {
    supertype: {
      label: 'Supertype', palette: SUPERTYPE_PALETTE,
      order: SUPERTYPES, keyOf: function (d) { return d.s; },
    },
    color: {
      label: 'Primary Color', palette: COLOR_PALETTE,
      order: ['W', 'U', 'B', 'R', 'G', 'Multicolor', 'Colorless'],
      keyOf: function (d) { return d.c; },
    },
    rarity: {
      label: 'Rarity', palette: RARITY_PALETTE,
      order: ['common', 'uncommon', 'rare', 'mythic', 'special', 'bonus'],
      keyOf: function (d) { return d.r; },
    },
    role: {
      label: 'Role', palette: ROLE_PALETTE, order: ROLE_ORDER, keyOf: roleOf,
      ensure: async function () {
        if (rolesByName) return true;
        try {
          const r = await fetch(DATA.cardRoles);
          rolesByName = (await r.json()).roles || {};
          return true;
        } catch (e) {
          rolesByName = null;
          return false;
        }
      },
    },
  };

  function grouping() { return GROUPINGS[currentColorBy] || GROUPINGS.supertype; }

  function getCategoryInfo(d) {
    const g = grouping();
    return { key: g.keyOf(d), palette: g.palette };
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
      // The renderer decides the default (it reads `prefers-reduced-motion`); the button
      // reports it. Sync once the renderer exists, never guess from the markup.
      syncMotionButton();

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
    // A focused region no longer HIDES the rest of the map. It used to: `visible`
    // excluded every non-member, so clicking a region left you staring at a cluster with
    // no idea where it sat. Orientation is the whole point of the atlas — a region only
    // means something against its neighbours — so non-members stay drawn and recede.
    const visible = (d) => activeSupertypes.has(d.s);
    const filtered = allData.filter(visible);
    // Contours and the status count still speak for the focused region only, so
    // "1,234 cards in Swolesville" keeps meaning the region rather than the map.
    const focused = regionFocus
      ? allData.filter((d, i) => visible(d) && regionFocus.rows.has(i))
      : filtered;

    // Contour trace (prepended before scatter so it renders beneath). While drilling it
    // re-bins over the local layout — histogram2dcontour auto-bins to whatever extent it
    // is handed, so levels are relative to the current selection and are NOT comparable
    // across drills.
    const contourTraces = [];
    const contourSource = drilling ? window.Drill.getContourSource() : focused;
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

    const palette = grouping().palette;

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
    /* ONE spotlight, however many things are pointing it.
     *
     * A focused region and a clicked legend row are the same gesture — "show me this,
     * keep the rest for context" — and they compose: hold both and you get that region's
     * Planeswalkers. Deciding it in two places is how the atlas ends up with two
     * disagreeing answers to whether a point is lit, which this file has been bitten by
     * before (`visible()` vs `filtered`).
     *
     * A region is per-point and costs the 34K array; a legend row is per-GROUP and costs
     * one comparison, so the group case is hoisted out and never touches the array path.
     */
    const LIT = 0.95, UNLIT = 0.09;
    function spotlightFor(g) {
      const groupLit = !legendKeys.size || legendKeys.has(g.key);
      // Two per-point sources now — a focused region and a search — and a point is
      // lit only if it survives both. Composing here rather than adding a second
      // dimming path is the whole reason this function exists.
      const rowSets = [];
      if (regionFocus) rowSets.push(regionFocus.rows);
      if (queryFocus && queryFocus.rows.size) rowSets.push(queryFocus.rows);
      // Pushed even when EMPTY: a printing filter that matches nothing dims
      // everything, which is what it means. (The query's empty set means "no
      // term yet", which is why it alone is guarded on size.)
      if (printFocus) rowSets.push(printFocus.rows);
      if (!rowSets.length) return groupLit ? LIT : UNLIT;       // scalar, free
      if (!groupLit) return UNLIT;                              // scalar, free
      return g.customdata.map(function (idx) {
        for (const rows of rowSets) if (!rows.has(idx)) return UNLIT;
        return LIT;
      });
    }
    /* Registry order, not hash order. `groups` is keyed by category, so `Object.values`
     * hands back whatever order the cards happened to arrive in — which made the legend
     * shuffle between renders and bear no relation to the order the same groups appear in
     * anywhere else. The registry's `order` is the one answer, and every surface that
     * reports these groups sorts by it. Unknown keys sort last rather than being dropped. */
    const groupOrder = grouping().order || [];
    const orderedGroups = Object.values(groups).sort(
      (a, b) => (groupOrder.indexOf(a.key) + 1 || 999) - (groupOrder.indexOf(b.key) + 1 || 999));
    const traces = orderedGroups.map(g => {
      let opacity;
      if (dimsAll) {
        opacity = 0.08;
      } else if (regionFocus || legendKeys.size || printFocus ||
                 (queryFocus && queryFocus.rows.size)) {
        // A spotlight, not a filter. Everything stays on screen at a low alpha so you can
        // still see WHERE the lit set sits — the question the atlas exists to answer, and
        // the one that hiding everything else destroyed.
        opacity = spotlightFor(g);
      } else if (dimmedIndices) {
        opacity = g.customdata.map(idx => dimmedIndices.has(idx) ? 0.08 : 0.85);
      } else {
        opacity = 0.85;
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
        // `glow` opts this layer into the renderer's zoom-responsive halo. Only the base
        // scatter takes it: the overlays (search, selection, deck) are already at full
        // alpha and a halo on them would read as a second, wrong highlight.
        marker: { size: 3, opacity, color: palette[g.key] || '#666', glow: true },
      };
    });

    // Search highlight trace (index-tracking to avoid O(n²) indexOf). Suppressed while
    // drilling: it plots world coordinates, and a diamond at a world position on top of
    // a local layout would be pointing at nothing.
    if (queryFocus && !drilling) {
      /* THE MATCHES ARE LIT, NOT OVERLAID. This used to push a trace of white
       * diamonds over 34,890 points left at full opacity — findable only if you
       * already knew where to look, and no help at all in answering "where do
       * treasure cards live". The spotlight above does the narrowing now; this
       * only says what happened.
       *
       * The FIELD is named because a count on its own is not explicable: 413 for
       * "treasure" is oracle text, 3,361 for "flying" is a keyword, and a pilot
       * seeing only the number cannot tell which question was answered. */
      if (queryFocus.pending) {
        setStatus('Loading the card data to search \u2014 "' + searchTerm + '" in a moment');
      } else if (queryFocus.total) {
        // The two biggest contributors, named. They OVERLAP, so they are listed
        // rather than summed — 413 oracle and 390 keyword is 433 cards, and a
        // reader shown only "433" cannot tell which question was answered.
        const by = queryFocus.fields.slice(0, 2)
          .map(function (f) { return f[0] + ' ' + f[1].toLocaleString(); }).join(', ');
        setStatus(queryFocus.total.toLocaleString() + ' card'
          + (queryFocus.total === 1 ? '' : 's') + ' match "' + searchTerm + '"'
          + ' \u00b7 ' + by + ' \u00b7 Esc to clear');
      } else {
        setStatus('No cards match "' + searchTerm + '" \u2014 '
          + filtered.length.toLocaleString() + ' shown');
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
    } else if (currentMode === 'explore' && printFocus) {
      // The count is of what is LIT — `narrowedTo`, so the supertype toggles and
      // a legend selection compose into it — and the undated exclusion is said
      // out loud, because a date bound silently dropping cards with no date is
      // the zero-for-absent failure wearing a filter's clothes.
      let n = 0;
      for (let i = 0; i < allData.length; i++) if (narrowedTo(allData[i], i)) n++;
      const undated = (printFocus.after || printFocus.before) && printFocus.undated
        ? ' \u00b7 ' + printFocus.undated.toLocaleString() + ' with no first-printed date excluded'
        : '';
      const reprints = printFocus.set
        ? ' \u00b7 a set means the corpus printing, so its reprints count' : '';
      setStatus(n.toLocaleString() + ' card' + (n === 1 ? '' : 's') + ' highlighted '
        + printLabel(printFocus) + (legendKeys.size ? ' (' + Array.from(legendKeys).join(' + ') + ')' : '')
        + undated + reprints + ' \u00b7 Esc to clear');
    } else if (currentMode === 'explore' && legendKeys.size) {
      // Counts what the legend NARROWED to, not what is drawn. Saying "34,890
      // cards shown" over a map where all but 2,431 have receded is the
      // dishonesty the three-predicates rule exists to prevent.
      let n = 0;
      for (let i = 0; i < allData.length; i++) if (narrowedTo(allData[i], i)) n++;
      setStatus(n.toLocaleString() + ' ' + Array.from(legendKeys).join(' + ')
        + ' card' + (n === 1 ? '' : 's') + ' \u00b7 Esc to clear');
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

  /* OPEN A CARD, FROM ANYWHERE, WITHOUT DESTROYING A WALK.
   *
   * The library drawer needed "show me this card" and there was no safe way to
   * say it. `selectByName` is a MAP selection — it needs the 12.9 MB projection
   * and does nothing visible in Discover, where the panel belongs to Discovery.
   * `Discovery.show` would have worked and is exactly wrong: it calls
   * `newWalk(true)`, so opening a card you kept would silently delete the graph
   * you kept it from. That is the "growing must never be able to delete" rule,
   * and this is a new door onto the same trap.
   *
   * So it routes by mode, for the reason `MM.relate` and `applyLine` already do:
   * `Discovery.focus` notes the card and repaints Discovery's panel, and calling
   * it from Build would repaint Build's roles and curve with Discover's landing
   * controls. Nothing here touches the graph's membership at all. */
  function openCard(nameOrRow) {
    if (!window.Discovery || !Discovery.isReady()) return false;
    const row = typeof nameOrRow === 'number'
      ? nameOrRow : Discovery.rowByName(nameOrRow);
    if (row < 0) return false;
    if (currentMode === 'discover') { Discovery.focus(row); return true; }
    Discovery.setCurrent(row);
    // The map lights the card too when the projection has landed — but only as
    // an addition, never as the thing that makes this work.
    if (allData.length) selectByName((cardRecord(row) || {}).n);
    if (currentMode === 'build' && window.Build) Build.renderPanel();
    else updateViewerPanel();
    return true;
  }

  // ── Expose shared state/functions on window.MM ──
  // Only members with a live caller (the other eight scripts, generated onclick
  // handlers, or index.html) are exported — see docs/viz.md for the contract.
  window.MM = {
    get allData() { return allData; },
    get currentMap() { return currentMap; },
    escHtml,
    openCard,
    searchPick,
    searchAdd,
    buildHoverTextMinimal,
    renderManaSymbols,
    closeDetail: clearSelection,
    escapeOnce: escapeOnce,
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
    // The card body's own inline handlers (image enlarge, DFC flip, image retry) and
    // the repaint Build calls when a watch verdict is queued or lands.
    toggleCardImage,
    flipCard,
    cardImageError,
    refreshDeckContext,
    // Bans and Game Changers (`{gc: Set, banned: {fmt: Set}, as_of}`, empty sets until
    // `card_flags.json` lands or if it never does), the pill Build's tiles share, and
    // the lazy combo index — a promise, resolved once, null on failure.
    cardFlags,
    gcPillHtml,
    comboIndex,
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
    // The registry, exported so aggregate views (Build's curve and role bars) colour by
    // the SAME definition the map and its legend use, rather than keeping a second table.
    GROUPINGS: GROUPINGS,
    get grouping() { return currentColorBy; },
    groupKey: function (d) { return grouping().keyOf(d); },
    groupColour: function (d) { const g = grouping(); return g.palette[g.keyOf(d)] || '#666'; },
    get regionFocus() { return regionFocus; },
    /* The printing highlight. `setPrintFocus` drives the CONTROLS and then
     * applies, so the toolbar and the map cannot disagree about what is lit. */
    get printFocus() {
      return printFocus && { set: printFocus.set, after: printFocus.after,
                             before: printFocus.before, count: printFocus.rows.size,
                             undated: printFocus.undated };
    },
    printRows: function () { return printFocus ? Array.from(printFocus.rows) : null; },
    setPrintFocus: function (set, after, before) {
      return Promise.resolve(preparePrintControls()).then(function () {
        const pairs = [['setSelect', set], ['firstAfter', after], ['firstBefore', before]];
        for (const [id, val] of pairs) {
          const el = document.getElementById(id);
          if (el) el.value = val || '';
        }
        applyPrintFocus();
        return MM.printFocus;
      });
    },
    clearPrintFocus: clearPrintFocus,
    /* Kept as `{key}` for the FIRST selection so `build.js` and the browser
     * suite read what they always read; `legendGroups` is the honest plural. */
    get legendFocus() {
      const first = legendKeys.values().next();
      return first.done ? null : { key: first.value };
    },
    get legendGroups() { return Array.from(legendKeys); },
    /* ONE spotlight, reachable from more than the legend.
     *
     * `legendFocus` was private with a getter and no setter, so the legend was
     * the only control that could point it. Build's mana curve and role bars
     * report the SAME groups in the SAME colours from the SAME registry — that
     * is the whole argument for `MM.GROUPINGS` — and a bar you can read but
     * cannot click is a legend that forgot it was one.
     *
     * Passing `groupingName` switches the overlay first, because a role bar is
     * always about roles: clicking "ramp" while the map is coloured by
     * supertype must mean "colour by role AND light up ramp", not "light up a
     * supertype called ramp", which is nothing. Toggles off when re-clicked,
     * exactly like the legend row.
     */
    focusGroup: function (key, groupingName) {
      var toggle = function () {
        // REPLACE, not accumulate. A role bar in Build says "show me this one",
        // and it is mutually exclusive with a line spotlight there; only the
        // legend accumulates. Keeping this single-select is what leaves
        // `build.js` and its tests untouched.
        const had = legendKeys.has(key) && legendKeys.size === 1;
        legendKeys.clear();
        if (!had) legendKeys.add(key);
        regroup();
        return legendKeys.size ? { key: key } : null;
      };
      if (!groupingName || !GROUPINGS[groupingName] || groupingName === currentColorBy) {
        return Promise.resolve(toggle());
      }
      // Switching the overlay is not a field assignment. `role` is the one
      // grouping whose data is NOT in the boot payload — `card_roles.json` is
      // 0.39 MB gz and loads on selection — so skipping `ensure()` would
      // colour all 34,890 cards 'unclassified' and read as a broken roles file
      // rather than an absent one. The select is kept in sync because it is
      // the other control for the same state, and two controls disagreeing
      // about one value is how a legend ends up lying.
      currentColorBy = groupingName;
      var sel = document.getElementById('colorBy');
      if (sel) sel.value = groupingName;
      var g = grouping();
      if (!g.ensure) return Promise.resolve(toggle());
      setStatus('Loading ' + g.label.toLowerCase() + ' data…');
      return Promise.resolve(g.ensure()).then(function (ok) {
        if (!ok) { setStatus('Could not load ' + g.label.toLowerCase() + ' data'); return null; }
        setMapStatus();
        return toggle();
      });
    },
    clearGroupFocus: function () { clearLegendFocus(); },
    relate,
    keep,
    orientTo,
    clearOrientation,
    get orientation() { return orientation; },
    render,
    setStatus,
    setMode,
    MAP_CONFIGS,
    SPACES,
    get space() { return currentSpace; },
    setSpace,
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
    passesFilters(d, i) { return narrowedTo(d, i); },
    filterLabel() {
      const on = Array.from(activeSupertypes);
      const types = on.length >= SUPERTYPES.length ? 'Everything' : on.join(' + ');
      if (queryFocus && queryFocus.total) return '"' + queryFocus.term + '"';
      // The breadcrumb has to name what was drilled, or a legend-filtered drill
      // is labelled "Everything" while showing one colour.
      if (legendKeys.size) return Array.from(legendKeys).join(' + ');
      return types;
    },
  };

  /* WHAT THIS TAB HAS OPEN, for Jarvis (PRD v2 Step 7; `page-state.js`).
   *
   * Registered HERE, after `window.MM` exists, and never earlier: everything above
   * runs inside the IIFE that defines MM (see the queueMicrotask note at boot). The
   * describer only READS — closure state and the getters Build and Session already
   * export — and is polled every 1.5 s, so it must stay cheap: the selection is
   * sliced to 40 BEFORE names are looked up (a browse set can hold 2,000 rows).
   *
   * FOCUS, the one card "this card" means, in order:
   *   1. Build's review-grid tile, while the keyboard is in the grid (j/k move it);
   *   2. the map viewer's card — the browse cursor, else the top of the 8-stack;
   *   3. Session's focus — the landing card, or the node pinned in the graph. */
  if (window.PageState) PageState.register(function () {
    const nameAt = function (row) {
      const r = (typeof row === 'number' && row >= 0) ? cardRecord(row) : null;
      return r ? r.n : null;
    };
    const build = currentMode === 'build' && window.Build ? window.Build : null;
    const s = { mode: currentMode };

    const ae = document.activeElement;
    const inGrid = !!(build && ae && ae.closest && ae.closest('#candGrid'));
    const viewerRow = browseSet ? browseSet.indices[browseSet.pos]
      : (selectedCards[topCardIndex] || selectedCards[0] || {}).idx;
    s.focus = (inGrid && build.gridCard) || nameAt(viewerRow)
      || (window.Session ? nameAt(Session.focus) : null) || undefined;

    const rows = browseSet ? browseSet.indices.slice(0, 40) : selectedCards.map(c => c.idx);
    s.selected = rows.map(nameAt).filter(Boolean);
    if (window.Session && Session.library) s.library = Session.library.names;

    if (build) {
      if (build.deckSlug) s.deck = build.deckSlug;
      s.view = build.view;
    } else if (currentMode === 'explore') {
      s.view = currentMap;
    }

    const f = {};
    if (searchTerm && currentMode !== 'discover') f.search = searchTerm;
    if (regionFocus) f.region = regionFocus.label;
    if (legendKeys.size) { f.groups = Array.from(legendKeys).join(' + '); f.colour_by = currentColorBy; }
    if (activeSupertypes.size < SUPERTYPES.length) f.types = Array.from(activeSupertypes).join(' + ');
    if (printFocus) {
      if (printFocus.set) f.set = printFocus.set;
      if (printFocus.after) f.first_after = printFocus.after;
      if (printFocus.before) f.first_before = printFocus.before;
    }
    if (browseSet && browseSet.label) f.browse = browseSet.label + ' (' + browseSet.indices.length + ')';
    if (window.Drill && Drill.isActive && Drill.isActive()) f.drill = true;
    if (build && build.watchSetId) { f.watch_set = build.watchSetId; f.grid = build.gridFilter; }
    s.filters = f;
    return s;
  });
})();
