import importlib

project = 'classtrees'
copyright = '2026, Jakub Sawicki'
author = 'Jakub Sawicki'

release = importlib.metadata.version("classtrees")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["classtrees._tree_module"]

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath("../../../src"))

# sphinx-build -b html docs/source docs/
