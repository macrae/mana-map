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

  // The scenario DOES have a card list now: `build_index.py:line_cards` derives it from the
  // ordered stack, the hand and the graveyard, and the manifest carries it per stack file.
  // `MIN_EDGE_NAME_LEN` and `BASIC_LANDS` went with the prose matching they existed to make
  // survivable — you no longer need to skip basics and short names when you are not
  // searching a haystack. The cap stays: a clique of five is unreadable regardless.
  const MAX_EDGE_CARDS = 4;

  // One card carries several roles; the lens paints it with the most decision-relevant
  // one. Order matters: threat:body sits on 19,032 of 34,322 cards, so it must lose every
  // tie or the map turns into one colour.
  /* The role taxonomy now lives in ONE place — `MM.GROUPINGS.role` — because it is a
   * colour language, and a colour language that is defined twice is two languages. These
   * two names are the local aliases so the rest of this file reads unchanged; the tables
   * themselves are the map's, so a role is the same colour on the atlas, in this panel's
   * bars, and in the curve below. */
  // LAZY, not top-level constants. Reading `MM.GROUPINGS` while this file is being
  // evaluated would make Build depend on script ORDER in index.html, and this repo has
  // already paid for that once: anything that touches `MM.*` before mana-map.js has
  // exported it throws at module scope, which aborts the IIFE and takes every later file
  // with it. Functions defer the read to first use, by which time boot is long done.
  function roleGroup() { return MM.GROUPINGS.role; }
  function familyPriority() { return roleGroup().order.filter(f => f !== 'unclassified'); }
  function familyColour(f) { return roleGroup().palette[f] || '#6B7280'; }

  // ── State ──
  let manifest = null;       // index.json {decks:[...]}
  let rolesByName = null;    // card_roles.json .roles
  let nameToIdx = null;      // built once from MM.allData
  let active = null;         // the loaded deck, see buildActive()
  let loading = false;

  let showEdges = true;
  let showCandidates = true;
  let dimOthers = true;
  // Build opens on the GRAPH. The map is still one click away and is what Deck Lens
  // always was — position, roles and verified lines drawn where the cards actually live.
  // The graph answers the other question: what is next to what, and what could join.
  let view = 'graph';
  // Which verified line is under the spotlight, as an index into `active.edges`, or -1.
  // Lives here rather than in the DOM because `renderPanel` replaces the panel's
  // innerHTML wholesale, so any selection state has to be re-emitted on every render.
  let focusedLine = -1;
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
      for (const f of familyPriority()) if (families.has(f)) return f;
    }
    if (idx !== null && idx !== undefined && MM.allData[idx] && MM.allData[idx].s === 'Land') return 'land';
    return 'unclassified';
  }

  /* Every deck artifact goes through here, and every one of them is now cache-busted.
   * They were not: `index.json`, `cards.json`, `considering.json` and each stack were
   * fetched at bare URLs, so adding `stack_cards` to the manifest served the old copy and
   * every verified line drew nothing while reporting six of them. The failure was silent
   * and looked like a logic bug — I spent four probes on it before checking the bytes. */
  function bust(url) {
    const version = (window.MM && MM.DATA_VERSION) || 0;
    return url + (url.indexOf('?') === -1 ? '?v=' : '&v=') + version;
  }

  async function getJSON(url) {
    const res = await fetch(bust(url));
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
    const byBucket = [{}, {}, {}, {}, {}, {}, {}];
    for (const idx of indices) {
      const d = MM.allData[idx];
      if (!d || d.s === 'Land') continue;
      const b = Math.min(Math.floor(d.m || 0), 6);
      buckets[b]++;
      const key = MM.groupKey(d);
      byBucket[b][key] = (byBucket[b][key] || 0) + 1;
    }
    // Registry order, not insertion order, so the stacks read consistently across
    // buckets and match the legend top-to-bottom.
    const order = MM.GROUPINGS[MM.grouping].order;
    const palette = MM.GROUPINGS[MM.grouping].palette;
    const segments = byBucket.map(counts => {
      const keys = Object.keys(counts).sort(
        (a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
      return keys.map(k => ({ key: k, n: counts[k], colour: palette[k] || '#6B7280' }));
    });
    const max = Math.max.apply(null, buckets.concat([1]));
    let html = '<div class="deck-section"><div class="deck-section-title">Mana curve</div>' +
               '<div class="mana-curve">';
    for (let i = 0; i < buckets.length; i++) {
      const pct = (buckets[i] / max) * 100;
      // Segmented by the CURRENT overlay, in the registry's order, using the registry's
      // colours — so the curve answers "what is this deck made of at each cost" in the
      // same visual language as the map and the legend, and switching the overlay
      // recolours every surface at once.
      const seg = segments[i];
      let stack = '';
      for (const s of seg) {
        // A bar you can read but cannot click is a legend that forgot it was
        // one. These segments report the same groups, in the same order, in
        // the same colours as the map legend — from the same registry — so
        // they get the same gesture.
        stack += '<div class="curve-seg' +
          (focusedGroup && focusedGroup.key === s.key ? ' is-on' : '') +
          '" data-group="' + esc(s.key) + '"' +
          ' title="' + esc(s.key) + ' \u00d7 ' + s.n + ' \u2014 click to spotlight"' +
          ' style="width:100%;background:' + s.colour + ';height:' +
          ((s.n / (buckets[i] || 1)) * 100) + '%;"></div>';
      }
      html += '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">' +
        '<div style="height:40px;width:100%;display:flex;align-items:flex-end;">' +
        '<div style="width:100%;display:flex;flex-direction:column;justify-content:flex-end;' +
        'border-radius:2px 2px 0 0;overflow:hidden;height:' + pct +
        '%;min-height:' + (buckets[i] > 0 ? '2px' : '0') + ';">' + stack + '</div></div>' +
        '<div class="curve-label">' + (i === 6 ? '6+' : i) + '</div></div>';
    }
    html += '</div>';
    // A stacked bar without a key is decoration. Only the groups actually present, in
    // registry order, using registry colours — the same swatches the map legend shows,
    // which is the whole point of grouping through one definition.
    const present = [];
    for (const seg of segments) for (const s of seg) if (!present.some(x => x.key === s.key)) present.push(s);
    present.sort((a, b) => (order.indexOf(a.key) + 1 || 99) - (order.indexOf(b.key) + 1 || 99));
    if (present.length > 1) {
      html += '<div class="curve-key">' + present.map(s =>
        '<span class="curve-key-item' +
        (focusedGroup && focusedGroup.key === s.key ? ' is-on' : '') +
        '" data-group="' + esc(s.key) + '">' +
        '<span class="lens-swatch" style="background:' +
        s.colour + '"></span>' + esc(s.key) + '</span>').join('') + '</div>';
    }
    return html + '</div>';
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
    // A line index means nothing against a different deck's `edges` array.
    focusedLine = -1;
    if (window.Force && Force.clearLine) Force.clearLine();

    const deckDoc = await getJSON(DECK_BASE + slug + '/cards.json');

    // considering.json is optional in principle; every published deck has one today.
    let considering = null;
    try { considering = await getJSON(DECK_BASE + slug + '/considering.json'); } catch (e) { /* absent */ }

    /* `engine.json` is the ONLY place direction lives.
     *
     * Every edge on the graph is undirected by construction — a verified line
     * becomes a clique over the cards its stack names, and a clique has no
     * arrows. The engine model is the one artifact that says which way a
     * resource moves: each line is `from -> to` between two of eight stages,
     * with a `carries` noun for what travels. That is a modelled assertion an
     * engineer wrote and `engine-critic` attacked, not something derived here.
     *
     * Gated on `has.engine` so a deck without a model costs no failed fetch —
     * the flag is in the manifest precisely because a browser cannot stat. */
    let engine = null;
    if (entry.has && entry.has.engine) {
      try { engine = await getJSON(DECK_BASE + slug + '/engine.json'); } catch (e) { /* absent */ }
    }

    /* The authored intro to each verified line.
     *
     * `manual_prose.json`'s `combo_lines` is keyed by STACK ID and is the most
     * authoritative prose a line has: a person wrote it against the passing
     * resolution, and `config.py` gates the key on `stacks:passing` so it
     * cannot be written about a line nobody checked. The compact page already
     * renders it; the graph threw it away and showed a title.
     *
     * Gated on `has.manual_prose` for the same reason `engine` is: a browser
     * cannot stat, and a 404 per deck load is a cost paid for nothing. */
    let prose = null;
    if (entry.has && entry.has.manual_prose) {
      try { prose = await getJSON(DECK_BASE + slug + '/manual_prose.json'); } catch (e) { /* absent */ }
    }

    // The manifest lists checker-passed stacks only, which is the whole point of it —
    // a browser can list neither data/decks/ nor stacks/.
    const stacks = [];
    for (const file of (entry.stack_files || [])) {
      try {
        const doc = await getJSON(DECK_BASE + slug + '/stacks/' + file);
        // Keep the filename: it is the key `stack_cards` is stored under, since that is
        // what the manifest hands the browser.
        doc.__file = file;
        stacks.push(doc);
      } catch (e) { /* skip */ }
    }
    return buildActive(entry, deckDoc, considering, stacks, engine, prose);
  }

  function buildActive(entry, deckDoc, considering, stacks, engine, prose) {
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

    const main = cards.map(toSlot);
    const unmapped = main.filter(s => s.idx === null).map(s => s.name);

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
      candidates,
      unmapped,
      edges: buildEdges(stacks, main, entry, engine, prose),
      engine: engine || null,
      copies: main.reduce((n, s) => n + s.qty, 0),
    };
  }

  /* What a verified line SAYS, in the order the evidence ranks.
   *
   * Three things could speak for a line and they are not interchangeable, so
   * each keeps its own slot and its own attribution rather than being merged
   * into one blob of text:
   *
   *   `note`     the authored intro from `manual_prose.combo_lines[<id>]` —
   *              a person's argument for why this line matters, gated on the
   *              stack passing. 47 of the fleet's 60 stacks have one.
   *   `answer`   the resolution's direct reply to the scenario's question.
   *              Only 8 stacks ask a question sharp enough to have one, which
   *              is why it cannot be the primary source.
   *   `summary`  `resolution.final_state.summary` — what the board looked like
   *              when the dust settled. Present on ALL 60, so it is the floor:
   *              every drawn line has at least this much to say.
   *
   * Every one of these was already fetched. `buildEdges` kept `title` and
   * dropped the rest, so the panel could name a line and never explain it.
   *
   * Nothing here is derived, summarised or truncated. A checker read these
   * words; re-wording them in the browser would put a ✓ over prose no checker
   * saw, which is the same mistake as editing a resolution to fix a reference. */
  function linePr(stack, authored) {
    const res = stack.resolution || {};
    const fs = res.final_state || {};
    const out = {
      note: authored[stack.id] || null,
      answer: typeof res.answer === 'string' ? res.answer : null,
      summary: typeof fs.summary === 'string' ? fs.summary : null,
      steps: Array.isArray(res.steps) ? res.steps.length : 0,
    };
    return (out.note || out.answer || out.summary) ? out : null;
  }

  // Each verified stack becomes a small clique between the deck cards it names, so a
  // rules-verified line reads as geometry: which corners of the deck actually talk to
  // each other.
  function buildEdges(stacks, main, entry, engine, prose) {
    const byName = new Map();
    for (const s of main) if (s.idx !== null) byName.set(s.name, s);
    const derived = (entry && entry.stack_cards) || {};
    const authored = (prose && prose.combo_lines) || {};

    const edges = [];
    for (const stack of stacks) {
      // The cards the line is MADE OF, derived in Python from the scenario's structured
      // fields (the ordered stack, the hand, the graveyard) and carried in the manifest.
      //
      // This used to substring-match every deck card name against the whole scenario blob,
      // `board` included. `board` is where a line is cast, not what it is made of, so
      // heliod's Approach-of-the-Second-Sun line drew "verified" edges to Ancient Tomb and
      // Howling Mine — lands that happened to be on the table — while Swan Song, the actual
      // interaction, was cut by a 4-card cap that truncated in NAME-LENGTH order.
      const named = derived[stack.__file] || [];
      const hit = [];
      for (const name of named) {
        if (byName.has(name) && hit.indexOf(name) === -1) hit.push(name);
        if (hit.length >= MAX_EDGE_CARDS) break;
      }

      // A stack that names fewer than two deck cards draws no line — a scenario can be
      // about one card. Keep it in the list anyway: the panel's count must agree with
      // the manifest's `verified`, or the lens quietly loses a verified line.
      const pairs = [];
      for (let i = 0; i < hit.length; i++) {
        for (let j = i + 1; j < hit.length; j++) pairs.push([byName.get(hit[i]), byName.get(hit[j])]);
      }
      const flow = engineFlow(engine, stack.id || '');
      edges.push({ id: stack.id || '', title: stack.title || '', cards: hit, pairs,
                   prose: linePr(stack, authored),
                   carries: flow ? flow.carries : null,
                   from: flow ? flow.from : null, to: flow ? flow.to : null,
                   stageOf: flow ? flow.stageOf : null });
    }
    return edges;
  }

  /* What `engine.json` says this stack's line MOVES, and which way.
   *
   * A verified line becomes a clique over the cards its stack names, and a
   * clique has no direction — `{source, target}` is whichever order the pair
   * was built in. The engine model is the one artifact that knows: each line
   * is `from -> to` across two of eight stages, carrying a named resource.
   *
   * Returns the two stage names, the noun, and a card -> stage lookup, so an
   * edge can be ORIENTED rather than merely labelled: a pair is drawn
   * `a -> b` when a sits in the `from` stage and b in the `to` stage. A pair
   * that does not span the two stages gets no arrowhead, because for that pair
   * the direction genuinely is not known. */
  function engineFlow(engine, stackId) {
    if (!engine || !stackId || !Array.isArray(engine.lines)) return null;
    /* ALL the lines citing this stack, not the first.
     *
     * A stack can carry more than one arrow: ur-dragon's 002 is cited twice,
     * once for `bodies` and once for `triggers`, and taking `.find()` would
     * have silently dropped half of what that board proves. If they agree on
     * direction the nouns are joined; if they disagree, no arrowhead is drawn,
     * because a pair pointing two ways is a pair whose direction is not a
     * fact. */
    const lines = engine.lines.filter(function (l) {
      return l && l.from && l.to && String(l.verified_by || '') === String(stackId);
    });
    if (!lines.length) return null;
    const line = lines[0];
    if (lines.some(function (l) { return l.from !== line.from || l.to !== line.to; })) return null;
    const carries = lines.map(function (l) { return l.carries; })
                         .filter(Boolean)
                         .filter(function (c, i, a) { return a.indexOf(c) === i; })
                         .join(' · ');
    const stageOf = new Map();
    for (const st of (engine.stages || [])) {
      for (const c of (st.cards || [])) {
        const name = typeof c === 'string' ? c : (c && (c.card || c.name));
        if (name) stageOf.set(name, st.stage);
      }
    }
    return { from: line.from, to: line.to, carries: carries || null, stageOf: stageOf };
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
    const ordered = familyPriority().concat(['unclassified']).filter(f => byFamily.has(f));
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
          size: 9, opacity: 1, color: familyColour(family),
          line: { color: '#0d0d1a', width: 1 },
        },
        _isDeckOverlay: true,
      });
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
    return familyPriority().concat(['unclassified'])
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
        '<h2>Build</h2>' +
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
        // The magazine is not a product any more, so nothing invites a pilot
        // into it. This link survived the removal on the other three surfaces
        // because it lives in a panel that only renders with a deck loaded —
        // a grep for the string found it, a click-through never would have.
        // `has.page` gates the manual for the same reason it does everywhere
        // else: a link that 404s is worse than a link that is not there.
        '<div class="lens-links">' +
          ((e.has && e.has.page)
            ? '<a href="../manuals/p/' + esc(active.slug) + '.html">Pilot\'s Manual →</a>'
            : '') +
          '<a href="deck.html?deck=' + esc(active.slug) + '">Dossier →</a>' +
        '</div>' +
      '</div>' +

      '<div class="deck-section">' +
        '<div class="lens-stats">' +
          statBox(active.copies, 'cards') +
          statBox(e.verified, 'verified') +
          statBox(active.candidates.length, 'short list') +
        '</div>' +
        '<button class="lens-btn" onclick="Build.fitDeck()">Zoom to the deck</button>' +
      '</div>' +

      '<div class="deck-section">' +
        '<div class="deck-section-title">View</div>' +
        '<div class="discover-graphctl">' +
          '<button class="lens-btn' + (view === 'graph' ? ' is-on' : '') +
            '" onclick="Build.setView(\'graph\')">Graph</button>' +
          '<button class="lens-btn' + (view === 'map' ? ' is-on' : '') +
            '" onclick="Build.setView(\'map\')">Map</button>' +
        '</div>' +
        '<div class="lens-note">The graph shows what is next to what · the map shows where it sits</div>' +
      '</div>' +

      '<div class="deck-section">' +
        '<div class="deck-section-title">Show</div>' +
        toggleRow('dimOthers', dimOthers, 'Dim the other cards') +
        toggleRow('showEdges', showEdges, 'Verified lines (' + drawnEdges().length + ' drawn)') +
        toggleRow('showCandidates', showCandidates, 'Short List (' + active.candidates.length + ')') +
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
          // `data-role` rather than `data-group`: a role bar is ALWAYS about
          // roles, so clicking it must switch the overlay to `role` as well as
          // spotlight the family. Clicking "ramp" while the map is coloured by
          // supertype would otherwise mean "light up a supertype called ramp",
          // which is nothing at all.
          '<div class="lens-bar-row' +
          (focusedGroup && focusedGroup.key === c.family ? ' is-on' : '') +
          '" data-role="' + esc(c.family) + '"' +
          ' title="' + esc(c.family) + ' \u2014 click to spotlight">' +
            '<span class="lens-swatch" style="background:' + familyColour(c.family) + '"></span>' +
            '<span class="lens-bar-label">' + esc(c.family) + '</span>' +
            '<span class="lens-bar-track"><span class="lens-bar-fill" style="width:' +
              Math.round((c.n / max) * 100) + '%;background:' + familyColour(c.family) + '"></span></span>' +
            '<span class="lens-bar-n">' + c.n + '</span>' +
          '</div>').join('') +
      '</div>';

    /* THE SELECTED CARD.
     *
     * Build's panel showed roles, a curve and the verified lines, and never once showed a
     * CARD — so single-clicking a node in the deck graph had nowhere to put what you
     * selected. `MM.buildCardDetailHtml` is the shared block Discovery already renders,
     * and it carries the relation controls and Keep, so a card behaves identically in
     * both modes rather than being a second, poorer card view.
     *
     * The row comes from Session, which owns "which card am I looking at" — the same
     * value `Discovery.setCurrent` writes when the graph selects one. */
    const selected = window.Session ? Session.focus : -1;
    if (selected >= 0 && MM.buildCardDetailHtml) {
      html += '<div class="deck-section">' +
        MM.buildCardDetailHtml(MM.cardRecord(selected), selected) + '</div>';
    }

    if (active.edges.length) {
      html +=
        '<div class="deck-section">' +
          '<div class="deck-section-title">Verified lines <span>✓ rules-verified</span></div>' +
          active.edges.map((edge, i) =>
            '<div class="lens-line' + (edge.pairs.length ? '' : ' lens-line-nodraw') +
              (i === focusedLine ? ' is-on' : '') +
          '" onclick="Build.focusLine(' + i + ')">' +
              '<div class="lens-line-title">' + esc(edge.title) + '</div>' +
              '<div class="lens-line-cards">' +
                (edge.cards.length ? edge.cards.map(esc).join(' · ') : 'no deck card named — no line drawn') +
              '</div>' +
              (i === focusedLine ? lineProseHtml(edge) : '') +
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

    // Bound ONCE on the container, not per element: `renderPanel` replaces the
    // whole innerHTML on every repaint, so per-row listeners would be re-bound
    // on each render and leak the ones they replaced. Same pattern the map
    // legend uses, for the same reason.
    if (!el._groupBound) {
      el._groupBound = true;
      el.addEventListener('click', function (ev) {
        const role = ev.target.closest && ev.target.closest('[data-role]');
        if (role) { focusGroup(role.getAttribute('data-role'), 'role'); return; }
        const grp = ev.target.closest && ev.target.closest('[data-group]');
        if (grp) { focusGroup(grp.getAttribute('data-group')); }
      });
    }
    bindProseFade(el);

    /* A block SHORTER than the cap never fires a scroll event, so `is-end` would
     * never be set and the fade would dim the last lines of prose that has
     * nothing after it. Rare but real — goblin-storm's line 002 is 877
     * characters — and a fade over the end of a complete text is a lie about
     * there being more. Settled on render rather than on scroll, because that
     * is the only moment the two heights are both known and the reader has not
     * touched anything yet. */
    const pr = el.querySelector('.lens-line-prose');
    if (pr) pr.classList.toggle('is-end', pr.scrollHeight <= pr.clientHeight + 2);
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
    applyView();
  }

  function toggle(key, value) {
    if (key === 'showIllegal') { showIllegal = !!value; renderPanel(); MM.render(); return; }
    if (key === 'dimOthers') dimOthers = value;
    else if (key === 'showEdges') showEdges = value;
    else if (key === 'showCandidates') showCandidates = value;
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

  /* The rows a verified line is made of. One resolver for both surfaces — the map wants
   * their world positions, the graph wants their row ids, and deriving each separately is
   * how the two drift. */
  function lineRows(edge) {
    if (!edge) return [];
    return edge.cards
      .map(n => active.main.find(s => s.name === n))
      .filter(s => s && s.idx !== null)
      .map(s => s.idx);
  }

  /* The deck's lines in the shape `Force.enter` injects: pairs of ROWS. Only lines that
   * actually name two deck cards produce edges — a scenario can be about a single card,
   * and those stay in the sidebar list but draw nothing. */
  /* Orient a pair against the line's two stages. Null when the pair does not
   * span them — a clique includes pairs that sit wholly inside one stage, and
   * an arrow there would be array order wearing a claim. */
  function orient(edge, a, b) {
    if (!edge.stageOf || !edge.from || !edge.to) return null;
    const sa = edge.stageOf.get(a.name), sb = edge.stageOf.get(b.name);
    if (sa === edge.from && sb === edge.to) return [a.idx, b.idx];
    if (sb === edge.from && sa === edge.to) return [b.idx, a.idx];
    return null;
  }

  function graphLines() {
    if (!active) return [];
    return active.edges.map((edge, i) => ({
      id: edge.id || String(i),
      title: edge.title || '',
      pairs: edge.pairs.map(pr => [pr[0].idx, pr[1].idx]),
      // Only the pairs that actually span the engine line's two stages, in
      // `from -> to` order. Everything else on the clique stays undirected.
      directed: edge.stageOf
        ? edge.pairs.map(pr => orient(edge, pr[0], pr[1])).filter(Boolean)
        : [],
      carries: edge.carries || null,
      from: edge.from || null,
      to: edge.to || null,
    })).filter(l => l.pairs.length);
  }

  /* The open line's prose, under the row it belongs to.
   *
   * ONLY under the focused row. The fleet's summaries run a median of 838
   * characters and up to 4,337; printing all of them at once turns a scannable
   * list of four to eleven lines into a wall, and the list's job is choosing
   * which line to open. Clicking already spotlights the cards — this is the
   * same click saying what you are looking at.
   *
   * Each source is LABELLED. An authored intro and a resolver's final state are
   * different kinds of claim, and running them together as one paragraph would
   * make the second look as considered as the first. The step count and the
   * pointer to the manual carry the rest: the full resolution with its
   * citations lives there, and this panel must not become a second, worse copy
   * of it — the same rule that keeps the theatre from reprinting Judge's Desk.
   *
   * The cards named in the prose are NOT linked. Every one is already a lit
   * node three inches to the left, and a second affordance for the same card
   * in the same glance is the interaction bug this repo fixed in the atlas. */
  function lineProseHtml(edge) {
    const p = edge.prose;
    if (!p) return '';
    let h = '<div class="lens-line-prose">';
    if (p.note) h += '<p>' + esc(p.note) + '</p>';
    if (p.answer) h += '<p><b>The answer.</b> ' + esc(p.answer) + '</p>';
    if (p.summary) h += '<p><b>Where it ends.</b> ' + esc(p.summary) + '</p>';
    const bits = [];
    if (p.steps) bits.push(p.steps + ' resolved step' + (p.steps === 1 ? '' : 's'));
    if (edge.carries) bits.push('carries ' + esc(edge.carries));
    if (bits.length) h += '<div class="lens-line-meta">' + bits.join(' · ') + '</div>';
    h += '</div>';
    return h;
  }

  /* The fade means "there is more", so it must come off at the bottom. Bound on
   * the container by delegation rather than per-row, because the panel's HTML is
   * rebuilt wholesale on every render and a listener on a row does not survive
   * that — the same reason the group and line clicks are delegated. */
  let _proseBound = false;
  function bindProseFade(host) {
    if (_proseBound || !host) return;
    _proseBound = true;
    host.addEventListener('scroll', function (e) {
      const el = e.target;
      if (!el.classList || !el.classList.contains('lens-line-prose')) return;
      const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 2;
      el.classList.toggle('is-end', atEnd);
    }, true); // capture: scroll does not bubble
  }

  /* Click a verified line to put it under a spotlight.
   *
   * This only ever moved the MAP camera, and Build defaults to the GRAPH — where the map
   * canvas is `display:none`. So the click panned a hidden canvas and changed a status
   * string, which is why it read as doing nothing. Clicking the same line again clears it.
   */
  /* ── SPOTLIGHT A GROUP ──────────────────────────────────────────────────
   *
   * The curve segments and the role bars already report the same groups, in
   * the same order, in the same colours as the map legend — all three read
   * `MM.GROUPINGS`, which is the entire argument for that registry existing.
   * What they could not do is be CLICKED, so the panel was a readout beside a
   * picture rather than a control over it.
   *
   * Routed exactly like `applyLine`, and for the same reason it had to be:
   * Build defaults to the GRAPH, and a handler that only moved the map spent
   * its time restyling a `display:none` canvas — which reads as the click
   * doing nothing. The two surfaces answer at their own scale: the graph
   * spotlights this DECK's cards in the group, the map spotlights the group
   * across the atlas and composes with the deck lens, which is what the legend
   * has always done.
   */
  let focusedGroup = null;

  function groupRows(key, groupingName) {
    if (!active || !window.MM || !MM.cardRecord) return [];
    const want = groupingName || MM.grouping;
    const g = MM.GROUPINGS && MM.GROUPINGS[want];
    if (!g) return [];
    return active.main.map(function (x) { return x.idx; })
      .filter(function (i) {
        if (i === null) return false;
        const rec = MM.cardRecord(i);
        return rec && g.keyOf(rec) === key;
      });
  }

  function focusGroup(key, groupingName) {
    if (!key) return;
    const same = focusedGroup && focusedGroup.key === key;
    focusedGroup = same ? null : { key: key, grouping: groupingName || null };
    // A group and a line are two answers to "show me"; holding both at once
    // means neither is legible, so taking one puts the other down.
    if (focusedGroup && focusedLine !== -1) { focusedLine = -1; applyLine(null); }

    const apply = function () {
      if (view === 'graph') {
        if (window.Force && Force.setGroup) {
          Force.setGroup(focusedGroup ? groupRows(key, groupingName) : null, key);
        }
      } else if (MM.focusGroup) {
        MM.focusGroup(key, groupingName);
      }
      renderPanel();
      MM.setStatus(focusedGroup
        ? key + ' — ' + groupRows(key, groupingName).length + ' in this deck'
        : (active && active.entry ? active.entry.deck_name : ''));
    };

    // Switching to `role` lazy-loads card_roles.json, so the rows cannot be
    // computed until it lands — see MM.focusGroup's own note on `ensure`.
    if (groupingName && MM.focusGroup && groupingName !== MM.grouping) {
      Promise.resolve(MM.focusGroup(key, groupingName)).then(function () {
        if (view === 'graph' && window.Force && Force.setGroup) {
          Force.setGroup(focusedGroup ? groupRows(key, groupingName) : null, key);
        }
        renderPanel();
      });
      return;
    }
    apply();
  }

  function focusLine(i) {
    if (!active || !active.edges[i]) return;
    focusedLine = (focusedLine === i) ? -1 : i;
    if (focusedLine !== -1 && focusedGroup) {
      focusedGroup = null;
      if (window.Force && Force.setGroup) Force.setGroup(null);
      if (MM.clearGroupFocus) MM.clearGroupFocus();
    }
    const edge = focusedLine === -1 ? null : active.edges[i];
    applyLine(edge);
    renderPanel();
    MM.setStatus(edge ? edge.title : (active.entry.deck_name || ''));
  }

  function applyLine(edge) {
    const rows = lineRows(edge);
    if (view === 'graph') {
      if (window.Force && Force.setLine) {
        Force.setLine(rows, edge ? (edge.id || String(focusedLine)) : null, { fit: !!edge });
      }
      return;
    }
    // Map view keeps the behaviour it always had: frame the line's cards on the atlas.
    if (!edge) return;
    const all = MM.allData;
    const pts = rows.map(r => all[r]).filter(Boolean);
    if (!pts.length) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const pad = 4;
    setCamera([Math.min(...xs) - pad, Math.max(...xs) + pad],
              [Math.min(...ys) - pad, Math.max(...ys) + pad]);
  }

  /* Drop the spotlight without touching the panel's other state. Called when the view
   * changes, when a different deck loads, when Build is left, and from the Escape chain. */
  /* Escape, in Build.
   *
   * `mana-map.js` has called `Build.handleEscape()` in build mode since the mode existed
   * — and it was never defined, so the call was a silent no-op behind its own `&&` guard.
   * Escape simply did nothing here. Same shape as `Build.onCommanderChange`, which was
   * also called-but-undefined and also invisible for exactly that reason.
   *
   * Peel outermost-first, then hand the rest to the shared chain. */
  function handleEscape() {
    // Escape means "get me back out". Dropping the spotlight without moving the camera
    // left you zoomed into two cards with no way back except a manual pan, so it also
    // reframes whatever surface you are on.
    if (focusedLine !== -1) { clearLine(); fitDeck(); return; }
    if (window.MM && MM.escapeOnce) MM.escapeOnce();
  }

  /* Frame the whole deck on whichever surface is showing. The graph's extent is emergent
   * and can only be measured; the map's is the deck's world positions. */
  function fitDeck() {
    if (view === 'graph') {
      if (window.Force && Force.fit) Force.fit();
      return;
    }
    zoomToDeck();
  }

  function clearLine() {
    if (focusedLine === -1) return;
    focusedLine = -1;
    if (window.Force && Force.clearLine) Force.clearLine();
    renderPanel();
  }

  /* Seed the force graph from the loaded deck — the same call Discovery makes, with
   * `opts.deck`, which is where the visual language lives: deck cards at full colour and
   * a larger radius with a white rim, the commander double-ringed in gold, warm heavy
   * deck edges, commander-first labels. Cards you branch to are washed out, so what you
   * brought stays legible against what you found. */
  function seedGraph() {
    if (!active || !window.Force) return;
    const rows = active.main.map(function (x) { return x.idx; })
                            .filter(function (i) { return i !== null; });
    if (!rows.length) return;
    // RESEED ONLY WHEN THERE IS NOTHING TO LOSE. `Force.enter` with an explicit seed takes
    // the rebuild path, so calling it unconditionally threw away everything you had
    // branched to — flipping to the map and back cost six explored cards, measured. Same
    // hazard the relation buttons had. If the graph already holds this deck, leave it.
    if (Force.nodeCount && rows.every(function (r) { return Force.hasRow(r); })) {
      Force.enter(null, null, { chrome: 'discovery' });   // restore, do not rebuild
      return;
    }
    const cmd = active.commanderName && nameToIdx ? nameToIdx.get(active.commanderName) : null;
    const cmdIdx = typeof cmd === 'number' ? cmd : -1;
    const seeds = cmdIdx >= 0
      ? [cmdIdx].concat(rows.filter(function (r) { return r !== cmdIdx; }))
      : rows;
    if (cmdIdx >= 0) Session.setCommander(cmdIdx);   // one answer, read by the brief
    /* Build seeds the SAME engine and deliberately does NOT go through
     * `Discovery.seedFromRows`. That helper ends by calling Discovery's own
     * `render()`, which would repaint Build's roles, curve and verified lines
     * with Discover's landing controls — the exact defect `Force.renderPanel`
     * was taught to avoid by asking `MM.mode`. The panel belongs to the mode;
     * the seed helper belongs to the mode that owns the panel it draws. */
    Promise.resolve(Force.enter(seeds, active.entry.deck_name,
      { chrome: 'discovery',
        deck: { rows: new Set(rows), commander: cmdIdx, lines: graphLines() } }))
      .then(function () { if (cmdIdx >= 0) Force.pinCard(cmdIdx); });
  }

  function applyView() {
    const plot = document.getElementById('plot');
    if (!plot) return;
    // Same class the graph mode uses — it names the force CANVAS, not a mode.
    plot.classList.toggle('force-mode', view === 'graph' && !!active);
    if (view === 'graph' && active) seedGraph();
    else { if (window.Force) Force.exit(); MM.render(); }
  }

  function setView(v) {
    if (v === view) return;
    // The spotlight is a property of one surface's rendering, and the other surface has
    // no idea it exists. Drop it rather than leave a highlighted row pointing at nothing.
    clearLine();
    view = v;
    applyView();
    renderPanel();
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
      MM.setStatus('Build unavailable: ' + err.message);
    }
    loading = false;
    renderPanel();
    // ?deck=<slug> deep-links from the dossier and the magazine's Back Page.
    const wanted = new URLSearchParams(window.location.search).get('deck');
    if (wanted && manifest && manifest.decks.some(d => d.slug === wanted)) {
      await select(wanted);
    } else {
      applyView();
    }
  }

  function exit() {
    const plot = document.getElementById('plot');
    if (plot) plot.classList.remove('force-mode');
    // `Force.newWalk` below clears the graph's half of this; the index is ours.
    focusedLine = -1;
    // BUILD OWNS ITS GRAPH, so leaving takes it with you. Everywhere else the rule is
    // "growing must never be able to delete" — a walk you grew is yours and survives a
    // round trip through Explore. A loaded DECK is not that: it is Build's subject, and
    // leaving it behind meant Discover opened on someone else's 97-card deck with the
    // landing card buried in it. Two workspaces sharing one canvas have to hand it back.
    //
    // Explore is exempt: it is a LENS on whatever graph is current, not a workspace, and
    // clearing on the way there would empty the very thing it exists to show you.
    if (window.Force && active && window.MM && MM.mode !== 'explore') Force.newWalk(true);
    if (window.Force) Force.exit();
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
  // the LIBRARY — which is Session's, is shared with Discover and Explore, and is the thing
  // the brief exports to the sub-agent routine.

  function addCard(row) {
    if (typeof row !== 'number' || row < 0) return;
    if (!Session.library.has(row)) Session.library.toggle(row);
    if (active) renderPanel();
    MM.setStatus((MM.cardRecord(row) || {}).n + ' added — ' + Session.library.size +
                 ' in the library · Export brief hands them to the build loop');
  }

  /* Is this card already accounted for — in the loaded deck, or in the library? Read by the
   * card panel's "+ Deck" button so it can say so rather than offering a no-op. */
  function isInDeck(row) {
    if (Session.library.has(row)) return true;
    if (!active || !nameToIdx) return false;
    const d = MM.allData[row];
    if (!d) return false;
    return active.main.some(function (slot) { return slot.name === d.n; });
  }

  /* Naming a commander changes what is legal, so the panel's legality dimming and its
   * colour-identity read have to follow. `MM.setCommander` calls this behind a guard, and
   * the guard was hiding the fact that it did not exist. */
  function onCommanderChange() {
    if (!active) return;
    renderPanel();
    if (showIllegal) MM.render();
  }

  window.Build = {
    renderPanel,
    onCommanderChange,
    addCard,
    isInDeck,
    setView,
    get view() { return view; },
    enter,
    exit,
    close,
    select,
    toggle,
    zoomToDeck,
    fitDeck,
    focusLine,
    focusGroup,
    get focusedGroup() { return focusedGroup; },
    clearLine,
    handleEscape,
    get activeLine() { return focusedLine; },
    // A read-only probe for the browser suite: the prose a line was BUILT with,
    // so a test can compare what the panel printed against what the artifact
    // says without re-fetching and re-implementing the source ranking.
    __lineProse: i => (active && active.edges[i] ? active.edges[i].prose : null),
    getOverlayTraces,
    getDimmedIndices,
    dimsAll,
  };
})();
