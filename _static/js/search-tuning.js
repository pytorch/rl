/*
 * Relevance tuning for Sphinx's built-in search.
 *
 * searchtools.js declares `var Scorer` only when it is still undefined, so
 * defining it from a script that loads earlier replaces the stock scorer
 * wholesale. Nothing else in Sphinx is patched.
 *
 * Two things go wrong on a large API reference:
 *
 *  1. The Python domain gives every module search priority 0, worth +15, while
 *     classes and functions get priority 1 for +5. On projects where module
 *     pages are stubs, every module outranks the class the user wanted.
 *
 *  2. performSearch merges three passes into one list and sorts on the raw
 *     score, but the passes are not on a common scale: the title/index pass is
 *     a 0-100 percentage while object and fulltext hits are small additive
 *     scores. Any title match therefore beats any object match.
 */
(function () {
  "use strict";

  var cachedSearch = null;
  var cachedQuery = null;

  function currentQuery() {
    if (window.location.search !== cachedSearch) {
      cachedSearch = window.location.search;
      var fromUrl = new URLSearchParams(cachedSearch).get("q");
      var box = document.querySelector("input[name='q']");
      cachedQuery = (fromUrl || (box && box.value) || "").toLowerCase().trim();
    }
    return cachedQuery;
  }

  // The dotted symbol a result points at. autodoc/autosummary emit one page per
  // symbol as `generated/<dotted.path>`, which is the only symbol signal a
  // title-pass result carries -- its display text is just the section heading.
  function symbolOf(docname, title, descr) {
    var generated = /(?:^|\/)generated\/(.+)$/.exec(String(docname));
    if (generated) return generated[1];
    if (descr != null) return String(title);
    return null;
  }

  window.Scorer = {
    objNameMatch: 11,
    objPartialMatch: 6,
    // Priority 0 is modules. Do not treat them as the most important results.
    objPrio: { 0: -5, 1: 5, 2: -5 },
    objPrioDefault: 0,
    title: 15,
    partialTitle: 7,
    term: 5,
    partialTerm: 2,

    score: function (result) {
      var docname = result[0];
      var title = result[1];
      var descr = result[3];
      var score = result[4];

      var query = currentQuery();
      if (!query) return score;

      // Compress the top of the title pass's percentage range. Monotonic, so
      // ordering within that pass is preserved while it stops dominating.
      var tuned = descr == null && score > 56 ? 56 + (score - 56) * 0.3 : score;

      var symbol = symbolOf(docname, title, descr);
      if (!symbol) return tuned;

      var lower = symbol.toLowerCase();
      var parts = lower.split(".");
      var leaf = parts[parts.length - 1];

      if (lower === query) tuned += 60;
      else if (lower.endsWith("." + query)) tuned += 45;
      else if (leaf === query) tuned += 25;
      else if (leaf.indexOf(query) === 0) tuned += 10;

      tuned -= 3 * Math.max(0, parts.length - 3);
      if (descr != null && /Python module/.test(descr)) tuned -= 10;
      return tuned;
    },
  };
})();
