# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import os.path
import sys
import warnings

import pytorch_sphinx_theme
import torchrl

# Suppress warnings - TODO
# suppress_warnings = [ 'misc.highlighting_failure' ]
warnings.filterwarnings("ignore", category=UserWarning)

project = "torchrl"
copyright = "2022, Meta"
author = "Torch Contributors"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "main (" + str(torchrl.__version__) + " )"
# The full version, including alpha/beta/rc tags.
# TODO: verify this works as expected
release = "main"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.aafig",
    "myst_parser",
]

sphinx_gallery_conf = {
    "examples_dirs": "reference/generated/tutorials/",  # path to your example scripts
    "gallery_dirs": "tutorials",  # path to where to save gallery generated output
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("torchrl",),
    "filename_pattern": "reference/generated/tutorials/",  # files to parse
    "notebook_images": "reference/generated/tutorials/media/",  # images to parse
    "download_all_examples": True,
}

napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    "pytorch_project": "torchrl",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
    "analytics_id": "UA-117752657-2",
}
html_css_files = [
    "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css",
    "css/custom.css",
]

# Output file base name for HTML help builder.
htmlhelp_basename = "PyTorchRLdoc"

autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {}


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "torchvision", "torchrl Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "torchrl",
        "torchrl Documentation",
        author,
        "torchrl",
        "TorchRL doc.",
        "Miscellaneous",
    ),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


aafig_default_options = {"scale": 1.5, "aspect": 1.0, "proportional": True}

# -- Generate knowledge base references -----------------------------------
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
from content_generation import (
    generate_knowledge_base_references,
    generate_tutorial_references,
)

generate_knowledge_base_references("../../knowledge_base")
generate_tutorial_references("../../tutorials/sphinx-tutorials/", "tutorial")
generate_tutorial_references("../../tutorials/src/", "src")
generate_tutorial_references("../../tutorials/media/", "media")
