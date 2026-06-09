# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the z4ai documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "z4ai"
author = "Hyukjin Kwon"
copyright = "2026, the z4ai authors"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Optional presentation niceties: copy buttons on code blocks (sphinx-copybutton)
# and the card/grid layout used on the landing page (sphinx-design).  Added only
# when importable, so a minimal docs build without them still succeeds; the
# GitHub Pages workflow installs them via docs/requirements.txt.
for _opt_ext in ("sphinx_copybutton", "sphinx_design"):
    try:
        __import__(_opt_ext)
        extensions.append(_opt_ext)
    except ImportError:  # pragma: no cover - presentation-only
        pass

# Strip the shell/REPL prompts and output lines when copying code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Optional / heavy deps that the package imports lazily - mock so the docs build
# never needs them installed.
autodoc_mock_imports = ["torch", "zipnn", "brotli"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "z4ai"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.png"

# Brand palette lifted from the logo: a deep-space navy card with a neon
# cyan -> violet -> magenta accent.  Light mode uses a readable violet; dark mode
# uses the logo's neon cyan/violet directly.
_NEON_CYAN = "#22e3ff"
_NEON_VIOLET = "#6a7bff"
_LIGHT_PRIMARY = "#5a3fd6"
_LIGHT_CONTENT = "#6a2fce"

html_theme_options = {
    "source_repository": "https://github.com/z4ai/z4ai",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,  # the logo already carries the wordmark
    "light_css_variables": {
        "color-brand-primary": _LIGHT_PRIMARY,
        "color-brand-content": _LIGHT_CONTENT,
        "font-stack--monospace": (
            "'JetBrains Mono','SF Mono',ui-monospace,'Menlo',monospace"
        ),
    },
    "dark_css_variables": {
        "color-brand-primary": _NEON_CYAN,
        "color-brand-content": _NEON_VIOLET,
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/z4ai/z4ai",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16" width="1em" height="1em">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49'
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68"
                "-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66."
                "07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59"
                ".82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32"
                "-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1"
                ".16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46."
                '55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
