# -*- coding: utf-8 -*-
#
# Documentation Configuration File
#
# This file contains the most commonly used options for documentation setup.

# -- Path Setup --------------------------------------------------------------

import os
import sys

# Append the documentation source directory to system path for module discovery
sys.path.insert(0, os.path.abspath('../GauOptX'))
print('=== Path Added ===')
print(os.path.abspath('../GauOptX'))

# -- Project Metadata --------------------------------------------------------

project_name = 'GauOptX'
copyright_year = '2024-2025'
author_name = 'Eva Kaushik'

# Version details
short_version = '1.0'  
full_version = '1.0.1-alpha'   

# -- General Configuration ---------------------------------------------------

# Define necessary Sphinx extensions for documentation
extensions = [
    'sphinx.ext.autodoc',  # Automatic documentation generation
    'sphinx.ext.mathjax',  # Support for rendering math equations
]

# Paths for custom templates
template_dirs = ['_templates']

# File suffix for documentation sources
source_file_suffix = '.rst'  # Other options: '.md' for Markdown

# Main documentation entry point
root_doc = 'index'

# Language settings (set dynamically if needed)
content_language = 'en'

# Exclude specific files and directories from documentation processing
excluded_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Syntax highlighting style
highlighting_style = 'sphinx'

# -- HTML Output Settings ----------------------------------------------------

# Select theme for HTML-based documentation
html_theme_style = 'alabaster'

# Paths for custom static files like stylesheets and images
html_static_dirs = ['_static']

# -- HTML Help Output --------------------------------------------------------

# Define base name for HTML help builder output
html_help_basename = 'GauOptXdoc'

# -- LaTeX Output Settings ---------------------------------------------------

latex_configuration = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'figure_align': 'htbp',
}

# Group documentation into LaTeX files
latex_docs = [
    (root_doc, 'GauOptX.tex', 'GauOptX Documentation',
     'GauOptX Team', 'manual'),
]

# -- Manual Page Output ------------------------------------------------------

# Define manual pages
manual_pages = [
    (root_doc, 'gauoptx', 'GauOptX Documentation',
     [author_name], 1)
]

# -- Texinfo Output Configuration --------------------------------------------

# Generate Texinfo documentation files
texinfo_docs = [
    (root_doc, 'GauOptX', 'GauOptX Documentation',
     author_name, 'GauOptX', 'A structured overview of the project.',
     'Miscellaneous'),
]
