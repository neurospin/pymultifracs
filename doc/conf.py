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

import numpy as np
import pymultifracs
import sphinx
import numpydoc

from sphinx.ext.autosummary.generate import AutosummaryRenderer


# -- Project information -----------------------------------------------------

project = 'PyMultiFracs'
copyright = '2020-2025, M. Dumeur, P. Ciuciu, V. van Wassenhove, P. Abry'
author = 'M. Dumeur, O. D. Domingues, P. Ciuciu, V. van Wassenhove, P. Abry'

# The full version, including alpha/beta/rc tags
from pymultifracs._version import __version__
release = __version__

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
              'sphinx_gallery.gen_gallery',
              'numpydoc',
            #   'sphinx_autodoc_typehints',
              # 'sphinx_gallery.notebook',
            #   'sphinx_bootstrap_theme',
            #   'nbsphinx',
            #   'myst_nb',
              'sphinx.ext.mathjax']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'exclude-members': '__init__',
    'inherited-members': True,
}

autodoc_typehints = 'none'

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
    # 'sklearn': ('https://scikit-learn.org/stable', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'pywavelets': ('https://pywavelets.readthedocs.io/en/latest/', None),
    # "numba": ("https://numba.readthedocs.io/en/latest", None),
    # "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    # "statsmodels": ("https://www.statsmodels.org/dev", None),
}

# numpydoc

numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

numpydoc_xref_ignore = {
    # words
    "and",
    "between",
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    # shapes
    "n_vertices",
    "n_faces",
    "n_channels",
    "m",
    "n",
    "n_events",
    "n_colors",
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_dipoles",
    "n_ica_components",
    "n_pos",
    "n_node_names",
    "n_tapers",
    "n_channelnals",
    "n_step",
    "n_freqs",
    "wsize",
    "Tx",
    "M",
    "N",
    "p",
    "q",
    "r",
    "n_observations",
    "n_regressors",
    "n_cols",
    "n_frequencies",
    "n_tests",
    "n_samples",
    "n_permutations",
    "nchan",
    "n_points",
    "n_features",
    "n_parts",
    "n_features_new",
    "n_components",
    "n_labels",
    "n_events_in",
    "n_splits",
    "n_scores",
    "n_outputs",
    "n_trials",
    "n_estimators",
    "n_tasks",
    "n_slices",
    "n_hpi",
    "n_fids",
    "n_elp",
    "n_pts",
    "n_tris",
    "n_nodes",
    "n_nonzero",
    "n_events_out",
    "n_segments",
    "n_orient_inv",
    "n_orient_fwd",
    "n_orient",
    "n_dipoles_lcmv",
    "n_dipoles_fwd",
    "n_picks_ref",
    "n_coords",
    "n_meg",
    "n_good_meg",
    "n_moments",
    "n_patterns",
    "n_new_events",
    "n_j",
    "n_scaling_ranges",
    # "n_rep",
    "n_realisations",
    # sklearn subclasses
    "mapping",
    "to",
    "any",
    # unlinkable
    "CoregistrationUI",
    "mne_qt_browser.figure.MNEQtBrowser",
    # pooch, since its website is unreliable and users will rarely need the links
    "pooch.Unzip",
    "pooch.Untar",
    "pooch.HTTPDownloader",
}

numpydoc_validate = True
numpydoc_validation_checks = {"all"}

examples_dirs = ["../tutorials", "../examples"]
gallery_dirs = ["auto_tutorials", "auto_examples"]
# os.environ["_MNE_BUILDING_DOC"] = "true"

scrapers = (
    "matplotlib",
    # "pyvista", # check if pyvista installed
)

compress_images = ("images", "thumbnails")
#let's make things easier on Windows users
#(on Linux and macOS it's easy enough to require this)
if sys.platform.startswith("win"):
    try:
        import subprocess
        subprocess.check_call(["optipng", "--version"])
    except Exception:
        compress_images = ()

# sphinx_gallery_parallel = int(os.getenv("MNE_DOC_BUILD_N_JOBS", "1"))

sphinx_gallery_conf = {
    "doc_module": ("pymultifracs",),
    "reference_url": dict(pymultifracs=None),
    "examples_dirs": examples_dirs,
    # "subsection_order": ExplicitOrder(
    #     [
    #         "../examples/io/",
    #         "../examples/simulation/",
    #         "../tutorials/io/",
    #         "../tutorials/misc/",
    #     ]
    # ),
    "gallery_dirs": gallery_dirs,
    # "default_thumb_file": os.path.join("_static", "mne_helmet.png"),
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    # "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": True,
    "reset_modules": (
        "matplotlib",
        "seaborn",
    ),  # called w/each script
    "reset_modules_order": "both",
    "image_scrapers": scrapers,
    # "show_memory": sys.platform == "linux" and sphinx_gallery_parallel == 1,
    "show_memory": False,
    "line_numbers": False,  # messes with style
    "within_subsection_order": "FileNameSortKey",
    # "capture_repr": ("_repr_html_", "__str__"),
    # "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    # "matplotlib_animations": True,
    "compress_images": compress_images,
    "filename_pattern": "^((?!sgskip).)*$",
    # "exclude_implicit_doc": {
    #     r"mne\.io\.read_raw_fif",
    #     r"mne\.io\.Raw",
    #     r"mne\.Epochs",
    #     r"mne.datasets.*",
    # },
    # "show_api_usage": "unused",
    "api_usage_ignore": (
        "("
        ".*__.*__|"  # built-ins
        # ".*Base.*|.*Array.*|mne.Vector.*|mne.Mixed.*|mne.Vol.*|"  # inherited
        # "mne.coreg.Coregistration.*|"  # GUI
        # common
        ".*utils.*|.*verbose()|.*copy()|.*update()|.*save()|"
        ".*get_data()|"
        # mixins
        # ".*add_channels()|.*add_reference_channels()|"
        # ".*anonymize()|.*apply_baseline()|.*apply_function()|"
        # ".*apply_hilbert()|.*as_type()|.*decimate()|"
        # ".*drop()|.*drop_channels()|.*drop_log_stats()|"
        # ".*export()|.*get_channel_types()|"
        # ".*get_montage()|.*interpolate_bads()|.*next()|"
        # ".*pick()|.*pick_channels()|.*pick_types()|"
        # ".*plot_sensors()|.*rename_channels()|"
        # ".*reorder_channels()|.*savgol_filter()|"
        # ".*set_eeg_reference()|.*set_channel_types()|"
        # ".*set_meas_date()|.*set_montage()|.*shift_time()|"
        # ".*time_as_index()|.*to_data_frame()|"
        # dictionary inherited
        # ".*clear()|.*fromkeys()|.*get()|.*items()|"
        # ".*keys()|.*pop()|.*popitem()|.*setdefault()|"
        # ".*values()|"
        # sklearn inherited
        # ".*apply()|.*decision_function()|.*fit()|"
        # ".*fit_transform()|.*get_params()|.*predict()|"
        # ".*predict_proba()|.*set_params()|.*transform()|"
        # I/O, also related to mixins
        # ".*.remove.*|.*.write.*)"
    ),
    "copyfile_regex": r".*index\.rst",  # allow custom index.rst files
    "parallel": -2,
}
# assert is_serializable(sphinx_gallery_conf)


# nbsphinx

highlight_language = 'none'
html_scaled_image_link = False
html_sourcelink_suffix = ''
nbsphinx_kernel_name = 'MFA'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 300}",
]

# myst-nb

nb_execution_mode = "off"

#%% Adjusting the displayed name of functions
# https://stackoverflow.com/a/72658470

# def smart_fullname(fullname):
#     parts = fullname.split(".")
#     return ".".join(parts[1:])


# def fixed_init(self, app, template_dir=None):
#     AutosummaryRenderer.__old_init__(self, app, template_dir)
#     self.env.filters["smart_fullname"] = smart_fullname


# AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
# AutosummaryRenderer.__init__ = fixed_init

