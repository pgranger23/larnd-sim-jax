import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('.'))

from artifacts import download_artifact

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

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Download artifacts from GitHub Actions

download_artifact(
    prefix="simulation-plots-",
    path="debug-plots",
    repo="pgranger23/larnd-sim-jax",
    token=os.environ["GITHUB_TOKEN"],
    raise_error=True,
    retries=3,
)

download_artifact(
    prefix="scan-plots-",
    path="debug-plots",
    repo="pgranger23/larnd-sim-jax",
    token=os.environ["GITHUB_TOKEN"],
    raise_error=True,
    retries=3,
)

download_artifact(
    prefix="scan-lut-plots-",
    path="scan-lut-plots",
    repo="pgranger23/larnd-sim-jax",
    token=os.environ["GITHUB_TOKEN"],
    raise_error=True,
    retries=3,
)

download_artifact(
    prefix="fit-plots-",
    path="fit-plots",
    repo="pgranger23/larnd-sim-jax",
    token=os.environ["GITHUB_TOKEN"],
    raise_error=True,
    retries=3,
)

download_artifact(
    prefix="fit-lut-plots-",
    path="fit-lut-plots",
    repo="pgranger23/larnd-sim-jax",
    token=os.environ["GITHUB_TOKEN"],
    raise_error=True,
    retries=3,
)
