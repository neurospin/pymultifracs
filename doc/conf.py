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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'PyMultiFracs'
copyright = '2020-2023, M. Dumeur, P. Ciuciu, V. van Wassenhove, P. Abry'
author = 'M. Dumeur, P. Ciuciu, V. van Wassenhove, P. Abry'

# The full version, including alpha/beta/rc tags
release = '0.2'


# -- General configuration ---------------------------------------------------

needs_sphinx = "2.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.intersphinx',
              # 'sphinx.ext.linkcode',
              'numpydoc',
            #   'sphinx_autodoc_typehints',
              # 'sphinx_gallery.notebook',
            #   'sphinx_bootstrap_theme',
            #   'nbsphinx',
              'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

autosummary_generate = True
autodoc_default_options = {'members': True, 'exclude-members': '__init__',
                           'inherited-members': True}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/neurospin/pymultifracs/",
            icon="fa-brands fa-square-github",
        ),
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "article_header_start": [],  # disable breadcrumbs
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc"],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_css_files = [
#     "style.css",
# ]

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {
#     "index": ["sidebar-quicklinks.html"],
# }

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False


# def setup(app):
#     app.add_stylesheet("style.css")  # also can be a full URL
#     app.add_stylesheet("font-awesome.css")
#     app.add_stylesheet("font-source-code-pro.css")
#     app.add_stylesheet("font-source-sans-pro.css")


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    # "numba": ("https://numba.readthedocs.io/en/latest", None),
    # "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    # "statsmodels": ("https://www.statsmodels.org/dev", None),
}

# numpydoc

numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

# nbsphinx

highlight_language = 'none'
html_scaled_image_link = False
html_sourcelink_suffix = ''
nbsphinx_kernel_name = 'megfractal'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
