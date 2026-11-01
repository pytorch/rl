// Correct the tutorial "View on GitHub" link.
//
// pytorch_sphinx_theme2 renders the call-to-action header (Run in Google
// Colab / Download Notebook / View on GitHub) as server-side anchors and
// fills their hrefs from JS. Its default GitHub path is
// ``<section>_source/<name>.py`` (the pytorch/tutorials layout). TorchRL keeps
// its tutorial sources under ``tutorials/sphinx-tutorials/`` instead, so we
// rewrite that one href. We listen on ``load`` (which fires after every
// ``DOMContentLoaded`` handler) so this runs after the theme has set the href.
window.addEventListener("load", function () {
  var typeEl = document.getElementById("tutorial-type");
  var githubLink = document.getElementById("github-link");
  if (!typeEl || !githubLink) {
    return;
  }
  var name = typeEl.textContent.trim().split("/").pop();
  if (name) {
    githubLink.setAttribute(
      "href",
      "https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/" +
        name +
        ".py"
    );
  }
});
