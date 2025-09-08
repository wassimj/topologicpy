# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'topologicpy'
copyright = '2025, Wassim Jabi'
author = 'Wassim Jabi'

from topologicpy.version import __version__
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    'sphinx_mdinclude',
    "nbsphinx_link"
]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
