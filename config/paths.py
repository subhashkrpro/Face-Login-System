"""
Centralized path configuration.

All directory and file paths are derived from ROOT_DIR.
Directories are auto-created on import.
"""

import os

# ── Root ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Output directories ───────────────────────────────────────────────────
IMG_DIR = os.path.join(ROOT_DIR, "img")
ENHANCED_DIR = os.path.join(IMG_DIR, "enhanced")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")

# ── Model file paths ─────────────────────────────────────────────────────
BLAZEFACE_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
FACE_LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
REALESRGAN_MODEL_PATH = os.path.join(MODEL_DIR, "realesr-general-x4v3.pth")
GFPGAN_MODEL_PATH = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")
OPENVINO_SR_MODEL_XML = os.path.join(MODEL_DIR, "sr_model.xml")
OPENVINO_SR_MODEL_BIN = os.path.join(MODEL_DIR, "sr_model.bin")
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, "w600k_mbf.onnx")
ARCFACE_ZIP_PATH = os.path.join(MODEL_DIR, "buffalo_sc.zip")
FACE_DB_PATH = os.path.join(EMBEDDINGS_DIR, "face_db.json")
FACE_FRAMES_DB_PATH = os.path.join(EMBEDDINGS_DIR, "face_frames.json")
MESH_DB_PATH = os.path.join(EMBEDDINGS_DIR, "mesh_db.json")
HNSW_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "hnsw_index.bin")
HNSW_LABELS_PATH = os.path.join(EMBEDDINGS_DIR, "hnsw_labels.json")

# ── Auto-create directories ──────────────────────────────────────────────
for _dir in (IMG_DIR, ENHANCED_DIR, MODEL_DIR, EMBEDDINGS_DIR):
    os.makedirs(_dir, exist_ok=True)
