
## 6. Configuration Reference

All configuration is centralized in the `config/` package. Values are validated at startup by `validate_config()`.

### 6.1 Face Detection (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTOR_BACKEND` | `"mediapipe"` | Detector: `"mediapipe"` (fast) or `"blazeface"` (legacy) |
| `MIN_DETECTION_CONFIDENCE` | `0.3` | Minimum face detection confidence (0.0–1.0) |
| `MIN_FACE_BBOX_PX` | `10` | Ignore detections smaller than this (pixels) |
| `FACE_BBOX_PADDING` | `0.30` | Expand crop by 30% on each side (hair, ears, chin) |

### 6.2 CLAHE (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLAHE_CLIP_LIMIT` | `2.0` | Contrast limiting threshold |
| `CLAHE_TILE_GRID` | `(8, 8)` | Tile grid size for local histogram |

### 6.3 Low-Light Gamma (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOW_LIGHT_BRIGHTNESS_THRESHOLD` | `60` | Mean L below this triggers gamma boost (1–255) |
| `LOW_LIGHT_TARGET_BRIGHTNESS` | `110` | Target mean L after correction (1–255) |
| `LOW_LIGHT_MAX_GAMMA` | `0.3` | Minimum gamma (lower = more aggressive, 0.05–1.0) |

### 6.4 Super Resolution (`config/defaults.py`, `config/models.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_SR_BACKEND` | `"realesrgan"` | Backend: `"realesrgan"`, `"openvino"`, `"gfpgan"` |
| `SR_UPSCALE` | `4` | Upscale factor |
| `SR_TILE_SIZE` | `128` | Tile size for large image processing |
| `SR_TILE_PAD` | `10` | Tile overlap padding |
| `SR_NUM_FEAT` | `64` | SRVGGNetCompact feature channels |
| `SR_NUM_CONV` | `32` | SRVGGNetCompact conv layers |

### 6.5 Face Recognition (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RECOGNITION_THRESHOLD` | `0.45` | Minimum combined similarity for a match (0.0–1.0) |
| `MESH_WEIGHT` | `0.15` | Weight of mesh score: `final = 0.85×ArcFace + 0.15×Mesh` |
| `DUPLICATE_THRESHOLD` | `0.55` | Reject enrollment if face matches existing (0.0–1.0) |
| `VERIFY_TOP_K` | `3` | Stage-1 HNSW candidates to verify in Stage-2 |
| `MAX_FACES` | `1` | Max faces to detect per frame |

### 6.6 HNSW Index (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HNSW_SPACE` | `"cosine"` | Distance metric: `"cosine"`, `"l2"`, `"ip"` |
| `HNSW_EF_CONSTRUCTION` | `200` | Build-time accuracy (higher = better recall) |
| `HNSW_M` | `16` | Max connections per node |
| `HNSW_EF_SEARCH` | `50` | Query-time accuracy (higher = better recall) |
| `HNSW_MAX_ELEMENTS` | `10000` | Pre-allocated capacity |

### 6.7 ArcFace (`config/models.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARCFACE_INPUT_SIZE` | `112` | Input face size (pixels) |
| `ARCFACE_EMBEDDING_DIM` | `512` | Output embedding dimension |
| `ARCFACE_ONNX_THREADS` | `4` | ONNX Runtime intra-op threads |

### 6.8 Face Mesh (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MESH_MIN_DETECTION_CONFIDENCE` | `0.5` | FaceLandmarker detection confidence |
| `MESH_MIN_TRACKING_CONFIDENCE` | `0.5` | FaceLandmarker tracking confidence |

### 6.9 Anti-Spoofing — Z-Depth (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPOOF_ENABLED` | `True` | Enable 3D z-depth anti-spoofing |
| `SPOOF_Z_RANGE_THRESHOLD` | `0.03` | Minimum z-range (max−min) for real face |
| `SPOOF_Z_STD_THRESHOLD` | `0.008` | Minimum z-std for real face |

### 6.10 Anti-Spoofing — LBP Texture (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPOOF_LBP_ENABLED` | `True` | Enable LBP texture spoof detection |
| `SPOOF_LBP_HIST_VARIANCE_THRESHOLD` | `0.00003` | Minimum LBP histogram variance |
| `SPOOF_LBP_HF_ENERGY_THRESHOLD` | `0.01` | Minimum high-frequency energy ratio |

### 6.11 Liveness Detection — Blink (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LIVENESS_ENABLED` | `False` | Enable blink-based liveness check |
| `EAR_BLINK_THRESHOLD` | `0.21` | EAR below this = eyes closed (0.0–1.0) |
| `BLINK_CONSEC_FRAMES` | `2` | Consecutive closed-eye frames for a blink |
| `LIVENESS_TIMEOUT_SEC` | `10.0` | Max seconds to wait for a blink |

### 6.12 Camera (`config/camera.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_SOURCE` | `0` | Camera index |
| `CAMERA_BACKEND` | `cv2.CAP_DSHOW` | OpenCV backend (DirectShow for Windows) |
| `CAMERA_CODEC` | `"MJPG"` | Video codec |
| `CAMERA_WIDTH` | `640` | Capture width |
| `CAMERA_HEIGHT` | `480` | Capture height |
| `CAMERA_FPS` | `60` | Target FPS |
| `CAMERA_BUFFER_SIZE` | `1` | Internal OpenCV buffer |
| `CAMERA_WARMUP_SEC` | `1.0` | Auto-exposure settling time |
| `CAPTURE_TIMEOUT_SEC` | `5.0` | Max wait for buffered frames |
| `DEFAULT_CAPTURE_FRAMES` | `3` | Default frames to capture |
| `DEFAULT_BUFFER_SIZE` | `3` | Pre-buffer deque size |

### 6.13 Guided Enrollment (`config/camera.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENROLL_FRAMES` | `5` | Frames to capture during guided enrollment |
| `ENROLL_POSE_HOLD_SEC` | `0.8` | Hold pose duration before capture |
| `ENROLL_POSE_YAW_THRESH` | `12.0` | Degrees yaw for left/right pose |
| `ENROLL_POSE_PITCH_THRESH` | `10.0` | Degrees pitch for up/down pose |
| `ENROLL_BLUR_THRESH` | `30.0` | Laplacian blur rejection threshold |

### 6.14 File Paths (`config/paths.py`)

| Path | Location | Description |
|------|----------|-------------|
| `ROOT_DIR` | `D:\LOGIN` | Project root |
| `IMG_DIR` | `img/` | Output image directory |
| `ENHANCED_DIR` | `img/enhanced/` | Enhanced face images |
| `MODEL_DIR` | `models/` | Downloaded model weights |
| `EMBEDDINGS_DIR` | `embeddings/` | Enrolled face data |
| `FACE_DB_PATH` | `embeddings/face_db.json` | Averaged ArcFace embeddings |
| `FACE_FRAMES_DB_PATH` | `embeddings/face_frames.json` | Per-frame ArcFace embeddings |
| `MESH_DB_PATH` | `embeddings/mesh_db.json` | Averaged mesh embeddings |
| `HNSW_INDEX_PATH` | `embeddings/hnsw_index.bin` | HNSW binary index |
| `HNSW_LABELS_PATH` | `embeddings/hnsw_labels.json` | HNSW label mapping |

### 6.15 Downloads (`config/defaults.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DOWNLOAD_TIMEOUT_SEC` | `60` | HTTP download timeout |
| `DOWNLOAD_CHUNK_SIZE` | `1048576` | Download chunk size (1 MB) |

---
