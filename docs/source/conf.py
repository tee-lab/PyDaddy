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
sys.path.insert(0, os.path.abspath('../..'))

#import sphinx_bootstrap_theme
#import sphinx_theme

import pydaddy
#autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance']
#autodoc_mock_imports = ["pydaddy.sde.SDE"]
#autodoc_mock_import = ["sklearn"]
autodoc_default_options = {"members": True, "undoc-members": False, "private-members": False, 'show-inheritance': False, 'inherited-members': False}


# -- Project information -----------------------------------------------------

project = 'pydaddy'
copyright = '2022, TEE-Lab'
author = 'Ashwin Karichannavar, Arshed Nabeel'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 
				'sphinx.ext.napoleon',
				'sphinx.ext.coverage', 
				'sphinx.ext.todo', 
				'sphinx.ext.mathjax', 
				'sphinxcontrib.contentui', 
				'sphinx.ext.autosectionlabel',
				'myst_nb'
			]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")


source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README_necsim.rst', 'README.md',
					'*/cmake-*/*','**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'bootstrap'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
#html_theme = 'furo'
#html_theme = 'insegel'
#html_theme = 'stanford_theme'
#html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]


html_logo = 'logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# needed for plotly, see https://myst-nb.readthedocs.io/en/latest/render/interactive.html#plotly
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]

#nbsphinx_prolog = r"""
#.. raw:: html
#
#    <script src='https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js'></script>
#    <script>require=requirejs;</script>
#
#
#"""

nb_number_source_lines = True
nb_merge_streams = True
nb_execution_timeout = -1
nb_execution_mode = "cache"
