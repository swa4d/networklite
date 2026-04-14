"""Sphinx configuration for ChemSim documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# ── Project ───────────────────────────────────────────────────────────────────
project   = "ChemSim"
copyright = "2025, ChemSim Authors"
author    = "ChemSim Authors"
release   = "0.1.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",         # Copy button on code blocks (optional)
]

autosummary_generate  = True
autodoc_typehints     = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
add_module_names      = False

intersphinx_mapping = {
    "python":  ("https://docs.python.org/3/",  None),
    "numpy":   ("https://numpy.org/doc/stable/", None),
    "scipy":   ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url":      "https://github.com/yourname/chemsim",
    "show_nav_level":  2,
    "navigation_depth": 3,
}
html_static_path = ["_static"]

# ── Source ────────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path   = ["_templates"]
