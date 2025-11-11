# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

os.environ['HOME'] = ''
os.environ['STORE'] = ''

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ViqueiraQRNN'
copyright = '2025, Daniel Expósito, José Daniel Viqueira'
author = 'Daniel Expósito, José Daniel Viqueira'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    #'sphinx_mdinclude',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    #"autodocsumm",
    #"sphinx_toolbox.more_autosummary"
]

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_mock_imports = [
    "os",
    "sys",
    "math",
    "numpy",
    "matplotlib",
    "typing",
    "cunqa",
    "subprocess",
    "random",
    "inspect",
    "functools"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = "_static/logo_cesga_blanco.png"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'flyout_display': 'hidden',
    'version_selector': True,
    'language_selector': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}

napoleon_google_docstring = True
napoleon_preprocess_types = True
napoleon_numpy_docstring = False

