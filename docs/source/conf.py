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



import pydaddy
#autodoc_default_flags = ['members', 'undoc-members', 'private-members', 'special-members', 'inherited-members', 'show-inheritance']
#autodoc_mock_imports = ["pydaddy.sde.SDE"]
autodoc_mock_import = ["sklearn"]
autodoc_default_options = {"members": True, "undoc-members": True, "private-members": True, 'show-inheritance': False, 'inherited-members':False}


# -- Project information -----------------------------------------------------

project = 'pydaddy'
copyright = '2021, TEE-Lab'
author = 'Ashwin Karichannavar'

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
			]


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

html_logo = 'logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
