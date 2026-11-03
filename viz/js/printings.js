/* The printing strip: every printing Scryfall knows of a card, the deck's one ringed.
 * ONE COPY, so Build's card panel and the deck page's art gallery cannot disagree about
 * what a foil toggle means. HTML only; each page wires its own click to its own write. */
window.Printings = (function () {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function price(p) {
    return [p.prices_usd ? '$' + p.prices_usd : '',
            p.prices_usd_foil ? 'foil $' + p.prices_usd_foil : ''].filter(Boolean).join(' / ');
  }

  /* `current`: `{set, collector_number, foil}`. `opts.attrs(p)`: extra attributes for one
   * printing's button, which is how a page hangs its click on it. `opts.prices` prints
   * the price under the thumb, for a page that has room. */
  function stripHtml(list, current, opts) {
    opts = opts || {};
    current = current || {};
    return '<label class="printing-foil"><input type="checkbox"' + (current.foil ? ' checked' : '') +
        '> foil</label>' +
      list.map(function (p) {
        const cur = p.set === current.set && String(p.collector_number) === String(current.collector_number);
        const cap = '(' + String(p.set || '').toUpperCase() + ') ' + esc(String(p.collector_number || ''));
        return '<button class="printing-thumb' + (cur ? ' is-current' : '') + '" type="button"' +
          ' data-set="' + esc(p.set || '') + '" data-cn="' + esc(String(p.collector_number || '')) + '"' +
          ' data-finishes="' + esc((p.finishes || []).join(',')) + '"' +
          ' title="' + esc([p.set_name, p.artist ? 'art by ' + p.artist : '', p.released_at, price(p)]
                           .filter(Boolean).join(' · ')) + '"' +
          (opts.attrs ? ' ' + opts.attrs(p) : '') + '>' +
          (p.image ? '<img src="' + esc(p.image) + '" alt="" loading="lazy">' : '<span class="printing-noart"></span>') +
          '<span class="printing-thumb-cap">' + cap + '</span>' +
          (opts.prices && price(p) ? '<span class="printing-thumb-cap">' + esc(price(p)) + '</span>' : '') +
          (opts.prices && p.artist ? '<span class="printing-thumb-cap">' + esc(p.artist) + '</span>' : '') +
          '</button>';
      }).join('');
  }

  /* What a pick means for foil. A printing has finishes it was released in, and the box
   * cannot make one up: foil-only is foil, never-foil is not. Unknown finishes: the box. */
  function foilFor(btn, checked) {
    const raw = btn.getAttribute('data-finishes') || '';
    const finishes = raw ? raw.split(',') : [];
    if (!finishes.length) return !!checked;
    if (finishes.indexOf('nonfoil') < 0) return true;
    if (finishes.indexOf('foil') < 0 && finishes.indexOf('etched') < 0) return false;
    return !!checked;
  }

  return { stripHtml: stripHtml, foilFor: foilFor };
})();
