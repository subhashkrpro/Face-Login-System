"""Model download URLs and architecture parameters."""

# ── BlazeFace (face detection) ────────────────────────────────────────────
BLAZEFACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)

# ── FaceLandmarker (478 3D landmarks) ─────────────────────────────────────
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

# ── Real-ESRGAN (super resolution) ───────────────────────────────────────
REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)

# SRVGGNetCompact architecture params (must match .pth weights)
SR_NUM_IN_CH = 3
SR_NUM_OUT_CH = 3
SR_NUM_FEAT = 64
SR_NUM_CONV = 32
SR_UPSCALE = 4
SR_TILE_SIZE = 128
SR_TILE_PAD = 10

# ── ArcFace (face embedding) ─────────────────────────────────────────────
ARCFACE_MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/"
    "v0.7/buffalo_sc.zip"
)
ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBEDDING_DIM = 512
ARCFACE_ONNX_THREADS = 4
