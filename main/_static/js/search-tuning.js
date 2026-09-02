/*
 * Relevance tuning for Sphinx's built-in search.
 *
 * searchtools.js declares `var Scorer` only when it is still undefined, so
 * defining it from a script that loads earlier replaces the stock scorer
 * wholesale. Nothing else in Sphinx is patched.
 *
 * Three things go wrong on a large API reference:
 *
 *  1. The Python domain gives every module search priority 0, worth +15, while
 *     classes and functions get priority 1 for +5. On projects where module
 *     pages are stubs, every module outranks the class the user wanted.
 *
 *  2. performSearch merges three passes into one list and sorts on the raw
 *     score, but the passes are not on a common scale: the title/index pass is
 *     a 0-100 percentage while object and fulltext hits are small additive
 *     scores. Any title match therefore beats any object match.
 *
 *  3. Those same three passes each emit their own row for one page, and the
 *     built-in dedup keys on [docname, title, anchor, descr, filename], which
 *     differs per pass. So a single symbol page is listed repeatedly, each row
 *     labelled with whatever heading matched -- eight rows reading "Linear".
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

      // Collapse the per-pass rows for one symbol page into a single entry.
      // searchtools.js applies Scorer.score to each result *before* it dedupes,
      // and hands score() the live result array, so rewriting the fields the
      // dedup keys on makes its existing dedup do the work. Relabelling to the
      // page's own symbol also replaces the bare matched heading ("Linear")
      // with something that distinguishes the row from its siblings.
      //
      // This depends on score() running ahead of the dedup in Search.query. If
      // a future Sphinx reorders that, results would duplicate again as they do
      // today -- it degrades to the current behaviour rather than breaking.
      var generated = /(?:^|\/)generated\/(.+)$/.exec(String(docname));
      if (generated) {
        result[1] = generated[1];
        result[2] = "";
        result[3] = null;
      }

      return tuned;
    },
  };
})();
