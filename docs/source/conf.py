import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LarND-Sim-Jax'
copyright = '2025, Pierre Granger'
author = 'Pierre Granger'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'rtds_action']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# The name of your GitHub repository
rtds_action_github_repo = "pgranger23/larnd-sim-jax"

# The path where the artifact should be extracted
# Note: this is relative to the conf.py file!
rtds_action_path = "source/debug-plots"

# The "prefix" used in the `upload-artifact` step of the action
rtds_action_artifact_prefix = "simulation-plots-"

# A GitHub personal access token is required
rtds_action_github_token = os.environ["GITHUB_TOKEN"]
