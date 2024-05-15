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
import os
import sys

if not os.path.exists("i6_core"):
    os.mkdir("i6_core")
    for fn in sorted(os.listdir("..")):
        print(fn)
        if fn.endswith(".py"):
            os.symlink("../../%s" % fn, "i6_core/%s" % fn)
            continue
        if os.path.isdir(os.path.join("..", fn)) and os.path.exists(
            os.path.join("..", fn, "__init__.py")
        ):
            os.symlink("../../%s" % fn, "i6_core/%s" % fn)
            continue

sys.path.insert(0, os.path.abspath("."))

import generateapi

generateapi.generate()

# -- Project information -----------------------------------------------------

project = "i6_core"
copyright = "2021, %s contributors" % project


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",  # link other projects
]

intersphinx_mapping = {
    "sisyphus": ("https://sisyphus-workflow-manager.readthedocs.io/en/latest/", None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# other options
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
if os.environ.get("READTHEDOCS") != "True":
    try:
        import sphinx_rtd_theme
    except ImportError:
        pass  # assume we have sphinx >= 1.3
    else:
        html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
