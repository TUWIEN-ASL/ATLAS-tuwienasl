# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atlas_gui'
copyright = '2026, TUWIEN-ASL'
author = 'Sergej Stanovcic, Daniel Sliwowski'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy-style docstrings
    'sphinx_autodoc_typehints',  # For type hints in docs
    'sphinx.ext.viewcode'
]
autodoc_mock_imports = [                                                     
    'cv2',                                                                   
    'numpy',                                                                 
    'PyQt5',                                                               
    'matplotlib',
    'pyqtgraph',
    'h5py',                                                                  
    'tensorflow',
    'tensorflow_datasets',                                                   
    'yaml',                                                                  
    'rosbags',
    'apache_beam',                                                           
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Autodoc configuration ---------------------------------------------------
# These settings control how class members are displayed

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# Remove module names from class/function signatures
add_module_names = False

# Control how type hints are displayed
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Control signature formatting
autodoc_class_signature = 'mixed'
autodoc_preserve_defaults = True

# This might help with the class name prefixing
def setup(app):
    app.connect('autodoc-process-signature', process_signature)

def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Remove class name from method signatures."""
    if what in ('method', 'attribute') and signature:
        # Remove class name prefix from method signatures
        if '.' in name:
            short_name = name.split('.')[-1]
            return (signature, return_annotation)
    return None

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# Control what appears in TOC - this should fix the repetitive class names
toc_object_entries_show_parents = 'hide'