"""Default thresholds and tunable parameters."""

# ── Face detection ────────────────────────────────────────────────────────
DETECTOR_BACKEND = "mediapipe"         # "mediapipe" (fast, reuses FaceLandmarker) | "blazeface" (legacy TFLite)
MIN_DETECTION_CONFIDENCE = 0.3
MIN_FACE_BBOX_PX = 10                 # Ignore detections smaller than this
FACE_BBOX_PADDING = 0.30              # Expand crop by 40% on each side (hair, ears, chin)

# ── CLAHE (contrast enhancement) ─────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ── Low-light adaptive gamma correction ──────────────────────────────────
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60   # Mean L below this triggers gamma boost
LOW_LIGHT_TARGET_BRIGHTNESS = 110     # Target mean L after gamma correction
LOW_LIGHT_MAX_GAMMA = 0.3             # Minimum gamma (lower = more aggressive)

# ── Super resolution ─────────────────────────────────────────────────────
DEFAULT_SR_BACKEND = "realesrgan"      # "realesrgan" | "openvino" | "gfpgan"

# ── Face recognition ─────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.45          # Cosine similarity for ArcFace match
MESH_WEIGHT = 0.15                    # Mesh embedding weight in combined score
                                      # final = (1 - MESH_WEIGHT) * arcface_sim + MESH_WEIGHT * mesh_sim
DUPLICATE_THRESHOLD = 0.55            # Reject enrollment if face matches existing (higher = stricter)
VERIFY_TOP_K = 3                      # Stage-1 candidates to verify against individual frames
MAX_FACES = 1

# ── HNSW index ────────────────────────────────────────────────────────────
HNSW_SPACE = "cosine"                 # Distance metric: "cosine" | "l2" | "ip"
HNSW_EF_CONSTRUCTION = 200            # Build-time accuracy (higher = slower build, better recall)
HNSW_M = 16                           # Max connections per node (16 is standard)
HNSW_EF_SEARCH = 50                   # Query-time accuracy (higher = slower search, better recall)
HNSW_MAX_ELEMENTS = 10000             # Pre-allocated capacity

# ── Face mesh ─────────────────────────────────────────────────────────────
MESH_MIN_DETECTION_CONFIDENCE = 0.5
MESH_MIN_TRACKING_CONFIDENCE = 0.5

# ── Anti-spoofing (3D depth liveness) ─────────────────────────────────────
SPOOF_ENABLED = True                  # Enable 3D-depth anti-spoofing check
SPOOF_Z_RANGE_THRESHOLD = 0.03        # min z-range (max−min) for a real face
SPOOF_Z_STD_THRESHOLD = 0.008         # min z-std for a real face

# ── Anti-spoofing (LBP texture analysis) ──────────────────────────────────
SPOOF_LBP_ENABLED = False              # Enable LBP texture spoof detection
SPOOF_LBP_HIST_VARIANCE_THRESHOLD = 0.00005   # min LBP histogram variance for real skin
SPOOF_LBP_HF_ENERGY_THRESHOLD = 0.02          # min high-frequency energy ratio for real face

# ── Anti-spoofing (recapture / photo-of-face detection) ───────────────────
# These features detect photos displayed on screens or printed photos by
# analysing noise distribution and gradient structure that differ between
# a live 3D face and a flat recaptured image.
SPOOF_NOISE_KURTOSIS_THRESHOLD = 10.0          # min noise kurtosis (real ≈25-35, photo ≈7-18)
SPOOF_BLOCK_GRAD_VAR_THRESHOLD = 700.0         # min block gradient variance (real ≈1000+, photo ≈450)
SPOOF_LOCAL_NOISE_VAR_THRESHOLD = 1.5           # min local noise level variance (real ≈3-8, photo ≈0.5-1.2)

# ── Liveness detection (blink) ────────────────────────────────────────────
LIVENESS_ENABLED = False                # Require a blink to prove liveness
EAR_BLINK_THRESHOLD = 0.21            # EAR below this = eyes closed
BLINK_CONSEC_FRAMES = 2               # Consecutive closed-eye frames for a blink
LIVENESS_TIMEOUT_SEC = 10.0           # Max seconds to wait for a blink

# ── Downloader ────────────────────────────────────────────────────────────
DOWNLOAD_TIMEOUT_SEC = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024     # 1 MB
