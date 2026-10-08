/* context-md.js — the Deck Context (data/decks/<slug>/CONTEXT.md) as HTML.
 *
 * PRD v2 Step 3: the deck page renders the one living document per deck. This is a
 * SMALL, ESCAPED renderer for the markdown subset the Context Keeper and
 * `deck_context.py` write — headings (# ## ###), paragraphs, `- ` lists (one level
 * of nesting), **bold**, _italic_, `code`, [text](url) — and nothing else. Every
 * byte of text is escaped before any markup is added back, so a card name or a
 * pilot note can never inject HTML. No markdown library: the subset is fixed by
 * the writer, and a general parser would accept things the gate never checked.
 *
 * Card links (`…/viz/index.html?cards=<Name>`) become `a.cardref` with the
 * deck page's hover art (`data-card`, `img.card-pop[data-src]`, wired by
 * deck-view.js's one delegated listener). Every other link to the deployed site
 * is made RELATIVE, so the page works the same served locally or deployed.
 *
 * `ContextMD.render(text, opts)` -> html. `ContextMD.wireFilters(root, cards)`
 * adds the Cards-by-role filter (name, colour, type, cost) over `cards`, a
 * `{name: {colors, type_line, cmc}}` map from the deck's cards.json.
 */
(function () {
  var SITE = 'https://manamap.seanmacrae.com/viz/';
  var CARD_PREFIX = SITE + 'index.html?cards=';

  function esc(v) {
    return String(v === undefined || v === null ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
  }

  function cardName(url) {
    if (url.indexOf(CARD_PREFIX) !== 0) return null;
    try { return decodeURIComponent(url.slice(CARD_PREFIX.length).replace(/\+/g, ' ')); }
    catch (e) { return null; }
  }

  function cardRef(name, label, opts) {
    var img = opts.cardImageUrl ? opts.cardImageUrl(name) : null;
    var pop = img ? '<img class="card-pop" data-src="' + esc(img) + '" alt="' + esc(name) + '">' : '';
    return '<a class="cardref" data-card="' + esc(name) + '" href="index.html?cards=' +
      encodeURIComponent(name) + '" tabindex="0">' + esc(label) + pop + '</a>';
  }

  function safeHref(url) {
    if (url.indexOf(SITE) === 0) return url.slice(SITE.length);          // relative
    return /^https?:\/\//.test(url) ? url : null;                          // no javascript:
  }

  /* Inline markup over ONE line. Links are cut out first (their text and URL are
   * escaped separately), then the rest is escaped and bold/italic/code added. */
  function inline(line, opts) {
    var out = '', re = /\[([^\]]+)\]\(([^)\s]+)\)/g, last = 0, m;
    while ((m = re.exec(line))) {
      out += plain(line.slice(last, m.index));
      var name = cardName(m[2]);
      if (name) out += cardRef(name, m[1], opts);
      else {
        var href = safeHref(m[2]);
        out += href ? '<a href="' + esc(href) + '">' + plain(m[1]) + '</a>' : plain(m[1]);
      }
      last = re.lastIndex;
    }
    return out + plain(line.slice(last));
  }

  function plain(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])_([^_\s][^_]*?)_(?=[\s.,;:)!?]|$)/g, '$1<em>$2</em>');
  }

  function slugify(h) {
    return h.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  }

  function render(text, opts) {
    opts = opts || {};
    var lines = String(text || '').replace(/<!--[\s\S]*?-->/g, '').split('\n');
    var html = [], para = [], list = [];      // list: [{depth, html}]

    function flushPara() {
      if (para.length) html.push('<p>' + para.map(function (l) { return inline(l, opts); }).join(' ') + '</p>');
      para = [];
    }
    function flushList() {
      if (!list.length) return;
      var out = '<ul>', depth = 0;
      list.forEach(function (it, i) {
        if (it.depth > depth) { out += '<ul>'; depth = it.depth; }
        else if (it.depth < depth) { out += '</li></ul></li>'; depth = it.depth; }
        else if (i) out += '</li>';
        out += '<li>' + it.html;
      });
      out += '</li>' + (depth ? '</ul></li>' : '') + '</ul>';
      html.push(out);
      list = [];
    }

    lines.forEach(function (raw) {
      var line = raw.replace(/\s+$/, '');
      var h = /^(#{1,3}) (.*)$/.exec(line);
      var li = /^( *)- (.*)$/.exec(line);
      if (!line.trim()) { flushPara(); flushList(); return; }
      if (h) {
        flushPara(); flushList();
        var lvl = h[1].length;
        if (lvl === 1 && opts.skipTitle) return;
        html.push('<h' + (lvl + 1) + ' class="ctx-h" data-section="' + esc(slugify(h[2])) + '">' +
                  inline(h[2], opts) + '</h' + (lvl + 1) + '>');
        return;
      }
      if (li) {
        flushPara();
        list.push({ depth: li[1].length >= 2 ? 1 : 0, html: inline(li[2], opts) });
        return;
      }
      if (list.length && /^ {2,}\S/.test(raw)) {           // a wrapped list item
        list[list.length - 1].html += ' ' + inline(line.trim(), opts);
        return;
      }
      flushList();
      para.push(line);
    });
    flushPara(); flushList();
    return html.join('\n');
  }

  /* THE CARDS-BY-ROLE FILTER. Works on whatever `render` produced: it finds the
   * "cards by role" heading and every list item and card-naming paragraph after it,
   * up to the next section heading. An item stays when ANY card in it matches; within a
   * kept item, the cards that do not match are dimmed rather than removed, so the
   * sentence still reads. */
  var COLOURS = [['W', 'white'], ['U', 'blue'], ['B', 'black'], ['R', 'red'], ['G', 'green'], ['C', 'colourless']];
  var TYPES = ['Creature', 'Instant', 'Sorcery', 'Artifact', 'Enchantment', 'Planeswalker', 'Land', 'Battle'];

  function matches(card, f) {
    if (!card) return !f.colour && !f.type && f.cost === '' && !f.name;
    if (f.name && card.name.toLowerCase().indexOf(f.name) < 0) return false;
    var cols = card.colors || [];
    if (f.colour === 'C' ? cols.length : (f.colour && cols.indexOf(f.colour) < 0)) return false;
    if (f.type && String(card.type_line || '').indexOf(f.type) < 0) return false;
    if (f.cost !== '') {
      var c = Math.round(card.cmc || 0);
      if (f.cost === '6' ? c < 6 : c !== +f.cost) return false;
    }
    return true;
  }

  function wireFilters(root, cards) {
    var head = root.querySelector('[data-section="cards-by-role"]');
    if (!head) return null;
    // `##` renders as h3 and `###` (a role) as h4, so the section runs to the next h3.
    var items = [], n = head.nextElementSibling;
    while (n && n.tagName !== 'H3') {
      // A role is a list, or — as the Keeper writes lands — a paragraph of names.
      if (n.tagName === 'UL') items = items.concat([].slice.call(n.children));
      else if (n.tagName === 'P' && n.querySelector('a.cardref')) items.push(n);
      n = n.nextElementSibling;
    }
    var by = {};
    Object.keys(cards || {}).forEach(function (k) { by[k.toLowerCase()] = cards[k]; });
    function cardOf(a) {
      var nm = a.getAttribute('data-card') || '';
      var c = by[nm.toLowerCase()] || by[nm.split(' // ')[0].toLowerCase()];
      return c ? Object.assign({ name: nm }, c) : { name: nm };
    }
    var bar = document.createElement('div');
    bar.className = 'ctx-filter';
    bar.innerHTML =
      '<input type="search" placeholder="Filter cards by name" aria-label="Filter cards by name" data-f="name">' +
      '<select aria-label="Colour" data-f="colour"><option value="">any colour</option>' +
        COLOURS.map(function (c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join('') + '</select>' +
      '<select aria-label="Type" data-f="type"><option value="">any type</option>' +
        TYPES.map(function (t) { return '<option>' + t + '</option>'; }).join('') + '</select>' +
      '<select aria-label="Mana value" data-f="cost"><option value="">any cost</option>' +
        [0, 1, 2, 3, 4, 5].map(function (c) { return '<option>' + c + '</option>'; }).join('') +
        '<option value="6">6+</option></select>' +
      '<span class="ctx-count" aria-live="polite"></span>';
    head.parentNode.insertBefore(bar, head.nextSibling);
    var count = bar.querySelector('.ctx-count');

    function apply() {
      var f = { name: bar.querySelector('[data-f=name]').value.trim().toLowerCase(),
                colour: bar.querySelector('[data-f=colour]').value,
                type: bar.querySelector('[data-f=type]').value,
                cost: bar.querySelector('[data-f=cost]').value };
      var active = f.name || f.colour || f.type || f.cost !== '';
      var shown = 0, hits = {};
      items.forEach(function (li) {
        var refs = [].slice.call(li.querySelectorAll('a.cardref'));
        var any = false;
        refs.forEach(function (a) {
          var ok = !active || matches(cardOf(a), f);
          a.classList.toggle('ctx-dim', active && !ok);
          if (ok && active) hits[a.getAttribute('data-card')] = 1;
          any = any || ok;
        });
        li.hidden = active && !any;
        if (!li.hidden) shown++;
      });
      count.textContent = active ? Object.keys(hits).length + ' card(s) match' : '';
    }
    bar.addEventListener('input', apply);
    bar.addEventListener('change', apply);
    return { apply: apply, items: items };
  }

  window.ContextMD = { render: render, wireFilters: wireFilters, _inline: inline };
})();
