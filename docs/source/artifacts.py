# -*- coding: utf-8 -*-

import subprocess
import time
from io import BytesIO
from zipfile import ZipFile

import requests
from sphinx.util import logging
from sphinx.errors import ExtensionError

logger = logging.getLogger(__name__)

def download_artifact(prefix, path, repo, token, raise_error, retries=3):
    """Downloads and extracts a GitHub Actions artifact for the current commit.

    This function attempts to find a GitHub Actions artifact matching a specific
    name constructed from the provided prefix and the current git commit hash.
    If found, it downloads the artifact (which is expected to be a zip archive)
    and extracts its contents to the specified local path. It includes a retry
    mechanism to handle potential delays in artifact availability.

    Args:
        prefix (str): The prefix string used to construct the expected artifact
            name. The final name will be ``f"{prefix}{git_hash}"``.
        path (str): The local directory path where the artifact's contents
            should be extracted.
        repo (str): The GitHub repository identifier in the format
            "owner/repository".
        token (str): A GitHub personal access token (PAT) with permissions to
            read Actions artifacts.
        raise_error (bool): If True, raise an Exception if the artifact cannot
            be found after all retries. If False, log a warning and return.
        retries (int, optional): The number of times to retry downloading if the
            artifact is not found initially. Defaults to 3.

    Raises:
        ExtensionError: If 'repo' or 'token' arguments are missing.
        Exception: If 'raise_error' is True and the artifact cannot be found
            after all retries.

    Returns:
        None: The function performs side effects (downloading and extracting)
              and does not explicitly return a value upon success.
    """
    
    if repo is None:
        raise ExtensionError(
            "rtds_action: missing required argument 'rtds_action_github_repo'"
        )

    if token is None:
        raise ExtensionError(
            "rtds_action: missing required argument 'rtds_action_github_token'"
        )

    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("ascii")
        )
    except subprocess.CalledProcessError:
        logger.warning("rtds_action: can't get git hash")
        return

    r = requests.get(
        f"https://api.github.com/repos/{repo}/actions/artifacts",
        params=dict(per_page=100),
        headers={"Authorization": f"token {token}"},
    )
    if r.status_code != 200:
        logger.warning(f"Can't list files ({r.status_code})")
        return

    expected_name = f"{prefix}{git_hash}"
    result = r.json()
    for artifact in result.get("artifacts", []):
        if artifact["name"] == expected_name:
            logger.info(artifact)
            r = requests.get(
                artifact["archive_download_url"],
                headers={"Authorization": f"token {token}"},
            )

            if r.status_code != 200:
                logger.warning(f"Can't download artifact ({r.status_code})")
                return

            with ZipFile(BytesIO(r.content)) as f:
                f.extractall(path=path)

            return

    logger.warning(
        f"rtds_action: can't find expected artifact '{expected_name}' "
        f"at https://api.github.com/repos/{repo}/actions/artifacts"
    )
    if retries > 0:
        logger.warning("Trying again")
        time.sleep(10)
        download_artifact(prefix, path, repo, token, raise_error, retries - 1)
    else:
        if raise_error:
            raise Exception(
                f"rtds_action: can't find expected artifact '{expected_name}' "
                f"at https://api.github.com/repos/{repo}/actions/artifacts"
            )