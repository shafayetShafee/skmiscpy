# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "skmiscpy"
copyright = "2024, Shafayet Khan Shafee"
author = "Shafayet Khan Shafee"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_immaterial",
]
autoapi_dirs = ["../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

myst_enable_extensions = ["dollarmath", "linkify"]


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_js_files = [
    "js/custom_copy.js",
]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_show_sphinx = False
html_theme = "sphinx_immaterial"


# material theme options (see theme.conf for more information)
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://skmiscpy.readthedocs.io",
    "repo_url": "https://github.com/shafayetShafee/skmiscpy",
    "repo_name": "skmiscpy",
    "edit_uri": "blob/main/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        "header.autohide",
        "navigation.top",
        "navigation.tracking",
        "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "teal",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "deep-orange",
            "accent": "lime",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    # BEGIN: version_dropdown
    "version_dropdown": True,
    "version_info": [
        {
            "version": "https://skmiscpy.readthedocs.io/en/stable/",
            "title": "stable",
            "aliases": [],
        },
        {
            "version": "https://skmiscpy.readthedocs.io/en/latest/",
            "title": "latest",
            "aliases": [],
        },
    ],
    # END: version_dropdown
    "toc_title_is_page_title": True,
    # BEGIN: social icons
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/shafayetShafee/skmiscpy",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/skmiscpy/",
        },
    ],
    # END: social icons
}
