"""ArcFace ONNX model download and extraction."""

import os
import zipfile

from config import ARCFACE_MODEL_URL, ARCFACE_MODEL_PATH, ARCFACE_ZIP_PATH
from src.utils import download_model
from src.exceptions import ModelExtractionError
from src.logger import get_logger

log = get_logger(__name__)


def ensure_arcface_model() -> str:
    """Download & extract the ArcFace ONNX model if needed."""
    if os.path.isfile(ARCFACE_MODEL_PATH):
        return ARCFACE_MODEL_PATH

    download_model(ARCFACE_MODEL_URL, ARCFACE_ZIP_PATH)

    log.info("Extracting w600k_mbf.onnx ...")
    try:
        with zipfile.ZipFile(ARCFACE_ZIP_PATH, "r") as zf:
            for name in zf.namelist():
                if name.endswith("w600k_mbf.onnx"):
                    data = zf.read(name)
                    with open(ARCFACE_MODEL_PATH, "wb") as f:
                        f.write(data)
                    break
            else:
                raise ModelExtractionError(ARCFACE_ZIP_PATH, "w600k_mbf.onnx")
    except zipfile.BadZipFile as e:
        raise ModelExtractionError(ARCFACE_ZIP_PATH, "w600k_mbf.onnx") from e

    try:
        os.remove(ARCFACE_ZIP_PATH)
    except OSError:
        pass
    log.info("ArcFace model extraction complete.")
    return ARCFACE_MODEL_PATH
