"""Model file downloader with progress indicator."""

import os
import requests

from config import DOWNLOAD_TIMEOUT_SEC, DOWNLOAD_CHUNK_SIZE
from src.exceptions import ModelDownloadError
from src.logger import get_logger

log = get_logger(__name__)


def download_model(url: str, dest: str) -> None:
    """Download a model file if it doesn't already exist."""
    if os.path.isfile(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log.info("Downloading %s ...", os.path.basename(dest))
    try:
        resp = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ModelDownloadError(url, str(e)) from e
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                log.debug("  %.1f / %.1f MB (%d%%)", downloaded / 1e6, total / 1e6, pct)
    log.info("Download complete: %s", os.path.basename(dest))
