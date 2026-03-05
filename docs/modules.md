
## 7. Module Documentation

### 7.1 `main.py` — Entry Point

**Purpose:** CLI dispatcher, argument parsing, logging setup, error handling.

#### Function: `main()`

Parses CLI arguments, sets up logging, validates config, dispatches to command handler.

**Exit Codes:**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Config error or unexpected error |
| 2 | Camera error |
| 3 | Model error |
| 4 | Face processing error |
| 130 | Keyboard interrupt (Ctrl+C) |

---

### 7.2 `src/capture/stream.py` — Camera Capture

#### Class: `FastStream`

Threaded webcam capture with configurable pre-buffering. A background thread continuously reads frames into a deque.

**Constructor:**

```python
FastStream(src=None, buffer_size=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `src` | `CAMERA_SOURCE` (0) | Camera index or path |
| `buffer_size` | `DEFAULT_BUFFER_SIZE` (3) | Pre-buffer deque capacity |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `self` | Start background capture thread |
| `read()` | `np.ndarray \| None` | Get latest frame (thread-safe copy) |
| `capture_frames(timeout=None)` | `list[np.ndarray]` | Wait for buffer_size frames, return copies |
| `stop()` | `None` | Stop thread and release camera |

**Raises:** `CameraOpenError` if camera cannot be opened, `CameraTimeoutError` if frames not delivered in time.

---

### 7.3 `src/detection/` — Face Detection

#### Factory: `create_detector(backend=None, min_confidence=None, max_faces=None)`

Creates a face detector by backend name. Returns either `MediaPipeDetector` or `BlazeFaceDetector`.

#### Class: `MediaPipeDetector` *(primary)*

Uses MediaPipe FaceLandmarker — detects faces AND extracts 478 3D landmarks in a single pass. This avoids the need for a separate FaceMesh init (~9s saved).

**Constructor:**

```python
MediaPipeDetector(min_confidence=None, max_faces=None)
```

**Method: `detect(frame: np.ndarray) -> list[dict]`**

Returns a list of face dicts:

```python
{
    "bbox": (x, y, w, h),          # padded bounding box in pixels
    "confidence": float,            # face presence confidence
    "crop": np.ndarray,             # BGR face crop
    "keypoints": [                  # for ArcFace alignment
        (left_eye_x, left_eye_y),   # centroid of 4 contour points
        (right_eye_x, right_eye_y), # centroid of 4 contour points
        (nose_x, nose_y),           # nose tip (landmark 1)
    ],
    "landmarks": np.ndarray,        # (478, 3) float32 — for mesh/liveness reuse
}
```

**Eye Keypoint Calculation:**

Eye centers are computed as the centroid of 4 contour landmarks per eye:
- Left: landmarks 33 (outer), 133 (inner), 159 (upper), 145 (lower)
- Right: landmarks 263 (outer), 362 (inner), 386 (upper), 374 (lower)

This matches BlazeFace's eye-center keypoints for consistent ArcFace alignment.

#### Class: `BlazeFaceDetector` *(legacy)*

Standalone BlazeFace TFLite detector. ~8s init, separate from FaceMesh. Used when `DETECTOR_BACKEND="blazeface"`.

Same `detect()` return format as `MediaPipeDetector` but without the `"landmarks"` key.

---

### 7.4 `src/enhancement/` — Image Enhancement

#### Class: `FaceEnhancer`

Full pipeline: frame → detect → crop → gamma → CLAHE → SR 4× → save.

**Constructor:**

```python
FaceEnhancer(min_confidence=None, sr_backend=None, clip_limit=None,
             debug_saver=None, **sr_kwargs)
```

Uses `ThreadPoolExecutor` to load detector and SR backend in parallel.

**Method: `process_frame(frame, frame_id=0) -> list[str]`**

Detects faces, enhances each crop, saves to `img/enhanced/`. Returns list of saved file paths.

Pipeline per face:
1. Raw crop extraction
2. `auto_brighten()` — adaptive gamma correction
3. `apply_clahe()` — CLAHE contrast enhancement
4. `sr.upscale()` — 4× super resolution
5. Save to disk

#### Function: `apply_clahe(img, clip_limit=None, tile_grid_size=None) -> np.ndarray`

Applies CLAHE to a BGR image via LAB L-channel (preserves color).

#### Function: `auto_brighten(img) -> np.ndarray`

Brightens image if mean brightness < `LOW_LIGHT_BRIGHTNESS_THRESHOLD`. Uses gamma correction via LUT.

**Internal functions:**
- `_estimate_brightness(img)` — Mean L-channel brightness (0–255)
- `_compute_gamma(mean_brightness)` — `gamma = log(target/255) / log(current/255)`, clamped
- `apply_gamma(img, gamma)` — Apply gamma via LUT (fast)

---

### 7.5 `src/enhancement/sr/` — Super Resolution Backends

#### Factory: `create_sr_backend(backend=None, **kwargs)`

Creates an SR backend. Accepts aliases:
- `"realesrgan"`, `"real-esrgan"`, `"pytorch"`, `"pth"` → `RealESRGAN_SR`
- `"openvino"`, `"ov"`, `"npu"` → `OpenVINO_SR`
- `"gfpgan"`, `"gfp"` → `GFPGAN_SR`

#### Class: `RealESRGAN_SR`

Real-ESRGAN 4× via PyTorch. Uses `SRVGGNetCompact` architecture.

| Method | Description |
|--------|-------------|
| `upscale(img)` | Upscale image 4×, auto-tiles large images |
| `_infer(img)` | Single tile inference (BGR → tensor → model → BGR) |
| `_upscale_tiled(img)` | Overlapping tile processing for large images |

#### Class: `SRVGGNetCompact` (`nn.Module`)

Compact VGG-style SR network matching `realesr-general-x4v3.pth` weights.

Architecture: `Conv2d → [PReLU → Conv2d] × 32 → PixelShuffle(4)` with residual connection.

#### Class: `OpenVINO_SR`

SR via OpenVINO IR model (.xml + .bin). Auto-selects best device: NPU → GPU → CPU.

#### Class: `GFPGAN_SR`

GFPGAN face restoration + super resolution. Uses local `GFPGANv1Clean` architecture (no external `gfpgan`/`basicsr` packages needed). Operates at 512×512 internally.

#### Class: `GFPGANv1Clean` (`nn.Module`)

GFP-GAN v1 Clean architecture. Components:
- **U-Net encoder** — `ResBlock` chain with bilinear downsampling (512 → 4×4)
- **Bottleneck** — Maps to W+ style code via `final_linear`
- **U-Net decoder** — `ResBlock` chain with skip connections
- **SFT modulation** — Spatial Feature Transform (scale + shift conditions)
- **StyleGAN2 decoder** — `StyleGAN2GeneratorClean` with SFT conditions

#### Class: `StyleGAN2GeneratorClean` (`nn.Module`)

StyleGAN2 generator (clean, no custom CUDA ops). Components:
- `NormStyleCode` — Normalize style to unit sphere
- `EqualLinear` — Equalized learning rate linear layer
- `ModulatedConv2d` — Style-modulated convolution
- `StyleConv` — ModulatedConv2d + noise injection + activation
- `ToRGB` — Feature → RGB via 1×1 modulated conv
- `ConstantInput` — Learned 4×4 constant input

---

### 7.6 `src/recognition/recognizer.py` — Face Recognizer

#### Class: `FaceRecognizer`

Main orchestrator for enrollment and recognition. Uses three-stage recognition with anti-spoofing.

**Constructor:**

```python
FaceRecognizer(threshold=None, max_faces=None, min_detection_confidence=None,
               debug_saver=None, recognition_debug=None)
```

Uses `ThreadPoolExecutor` to load detector (~9s) and ArcFace (~200ms) in parallel. HNSW index and databases load on main thread during wait.

**Key Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `detector` | `MediaPipeDetector` | Face detector |
| `embedder` | `ArcFaceEmbedder` | ArcFace embedding extractor |
| `index` | `HNSWIndex` | HNSW vector index |
| `db` | `dict` | Averaged ArcFace embeddings |
| `frames_db` | `dict` | Individual frame embeddings |
| `mesh_db` | `dict` | Averaged mesh embeddings |
| `spoof_enabled` | `bool` | Z-depth anti-spoofing |
| `lbp_enabled` | `bool` | LBP texture anti-spoofing |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `enroll(name, frame)` | `bool` | Enroll from single frame |
| `enroll_multi(name, frames)` | `bool` | Enroll from multiple frames |
| `recognize(frame)` | `list[dict]` | Three-stage recognition |
| `list_enrolled()` | `list[str]` | List enrolled names |
| `delete(name)` | `bool` | Delete enrolled face |
| `close()` | `None` | Release resources |

**Internal Methods:**

| Method | Description |
|--------|-------------|
| `_get_embedding(frame, face, ...)` | ArcFace embedding (aligned if keypoints available) |
| `_get_mesh_embedding(frame, face)` | 1434-dim mesh embedding from landmarks |
| `_check_spoof(frame, face)` | Anti-spoof: z-depth + LBP (for enroll) |
| `_check_spoof_from_landmarks(landmarks, crop)` | Anti-spoof from pre-extracted data (for recognize) |
| `_verify_against_frames(embedding, name, mesh_emb)` | Stage 2+3: frame verification + mesh fusion |
| `_load_db()` | Load JSON databases from disk |
| `_save_db()` | Persist all three databases |
| `_sync_json_to_hnsw()` | Sync JSON entries to HNSW index |

**Three-Stage Recognition:**

```
Stage 1: HNSW search → top-K candidates (fast, averaged ArcFace)
Stage 2: Verify each candidate against individual frame embeddings
Stage 3: Combine ArcFace score with 3D mesh geometric similarity

Combined = (1 - MESH_WEIGHT) × ArcFace_sim + MESH_WEIGHT × Mesh_sim
         = 0.85 × ArcFace_sim + 0.15 × Mesh_sim
```

**Recognition Return Format:**

```python
[{
    "name": str,        # matched name, "unknown", or "spoof"
    "similarity": float, # combined score (0.0–1.0)
    "bbox": (x, y, w, h),
    "spoof_info": dict,  # only if spoofed
}]
```

---

### 7.7 `src/recognition/embedding/` — ArcFace Embeddings

#### Class: `ArcFaceEmbedder`

ArcFace 512-dim face embedding via InsightFace MobileFaceNet (ONNX Runtime).

**Constructor:**

```python
ArcFaceEmbedder(model_path=None)
```

Auto-downloads and extracts `w600k_mbf.onnx` from InsightFace buffalo_sc.zip (~12 MB).

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_embedding(face)` | `(512,) float64` | Embedding from face crop (any size → 112×112) |
| `get_embedding_aligned(frame, keypoints)` | `(512,) float64` | Align face first, then embed |
| `_preprocess(face)` | `(1,3,112,112) float32` | BGR → RGB → normalize to [-1,1] → NCHW |

#### Function: `align_face(frame, keypoints, output_size=None) -> np.ndarray`

Aligns a face using 3 keypoints (left eye, right eye, nose) via similarity transform (`cv2.estimateAffinePartial2D`). Target landmarks are the standard ArcFace 112×112 alignment points.

#### Function: `ensure_arcface_model() -> str`

Downloads `buffalo_sc.zip`, extracts `w600k_mbf.onnx`, cleans up zip. Returns path.

---

### 7.8 `src/recognition/enrollment/` — Enrollment

#### Class: `GuidedEnrollment`

Phone-style guided enrollment with oval UI and head-pose diversity.

**Constructor:**

```python
GuidedEnrollment(num_frames=None, landmarker=None)
```

Pass a pre-loaded `landmarker` to avoid duplicate TFLite init (~9s saved).

**Method: `capture(name) -> list[np.ndarray]`**

Opens webcam with oval guide overlay. Two phases:
1. **Center** — User looks straight, hold for 0.5s
2. **Circle** — User moves head around, captures at diverse angles (minimum 25° gap between captures)

Returns list of BGR frames. Empty if cancelled (ESC key).

**Head Pose Estimation:** Uses nose-tip offset from face center (landmarks 1, 33, 263, 152, 10). Normalized to [-1, 1] range.

**Quality Checks:**
- Blur rejection via Laplacian variance (threshold: 12.0)
- Minimum 0.35s between captures
- Minimum 25° angle separation between captures

#### Class: `DuplicateChecker`

Prevents enrolling faces that already exist in the database.

**Constructor:**

```python
DuplicateChecker(index: HNSWIndex, threshold=None)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `is_duplicate(embedding, exclude_name=None)` | `str \| None` | Returns matching name if duplicate |
| `audit()` | `list[dict]` | Compare all enrolled faces pairwise |

---

### 7.9 `src/mesh/` — 3D Face Mesh & Anti-Spoofing

#### Class: `FaceMesh`

Extracts 478 3D face landmarks from BGR images using MediaPipe FaceLandmarker.

Each landmark: `(x, y, z)` normalized [0,1], z = relative depth.

**Method: `extract(frame) -> list[np.ndarray]`**

Returns list of `(478, 3) float32` arrays, one per face.

#### Function: `mesh_to_embedding(landmarks) -> np.ndarray`

Converts 478×3 landmarks to 1434-dim L2-normalized embedding:
1. Center mesh (translate centroid to origin)
2. Scale to unit norm (size-invariant)
3. Flatten to 1D (1434-dim)
4. L2-normalize

#### Function: `check_liveness(landmarks) -> dict` *(Z-Depth)*

Analyses z-depth statistics to detect flat/photo spoofing.

**Returns:**

```python
{
    "is_live": bool,     # True if face appears 3D
    "z_range": float,    # max(z) − min(z)
    "z_std": float,      # standard deviation of z
    "reason": str,       # human-readable explanation
}
```

| Metric | Real Face | Flat Photo |
|--------|-----------|------------|
| z_range | 0.04 – 0.15+ | 0.00 – 0.02 |
| z_std | 0.01+ | < 0.008 |

#### Function: `check_texture(crop) -> dict` *(LBP)*

Analyses face crop texture via Local Binary Patterns to detect photo/screen replay.

**Process:**
1. Resize crop to 128×128 (consistent comparison)
2. Compute LBP (8-connected, 256 codes)
3. Build normalized histogram → compute variance
4. Compute Laplacian high-frequency energy ratio

**Returns:**

```python
{
    "is_live": bool,       # True if texture looks like real skin
    "hist_var": float,     # LBP histogram variance
    "hf_energy": float,    # High-frequency energy ratio
    "reason": str,         # human-readable explanation
}
```

| Metric | Real Face | Recaptured |
|--------|-----------|------------|
| hist_var | ~0.00007 | < 0.00003 |
| hf_energy | ~0.03 | < 0.01 |

---

### 7.10 `src/liveness/` — Blink-Based Liveness

#### Class: `BlinkDetector`

Detects eye blinks using EAR (Eye Aspect Ratio) from MediaPipe FaceLandmarker.

**A blink =** EAR drops below threshold for N consecutive frames, then recovers.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `update(frame)` | `(ear, blinked)` | Feed frame, check for blink |
| `reset()` | `None` | Reset blink counter |
| `close()` | `None` | Release landmarker |
| `blink_count` | `int` | Total blinks detected (property) |

#### Function: `compute_ear(landmarks, indices) -> float`

Eye Aspect Ratio: `EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)`

**Landmark indices:**
- Right eye: (33, 160, 158, 133, 153, 144)
- Left eye: (362, 385, 387, 263, 380, 373)

#### Function: `check_liveness(stream, timeout=None) -> (bool, np.ndarray | None)`

Read frames from stream and wait for a blink within timeout. Returns `(is_live, frame)`.

---

### 7.11 `src/indexing/hnsw_index.py` — Vector Index

#### Class: `HNSWIndex`

HNSW approximate nearest-neighbor index for face embeddings using `hnswlib`.

**Constructor:**

```python
HNSWIndex(index_path=None, labels_path=None)
```

Loads existing index from disk or creates empty. Auto-persists on every modification.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add(name, embedding)` | `None` | Insert/update a face (marks old entry deleted) |
| `search(embedding, k=1)` | `list[dict]` | Find k nearest: `[{"name", "distance", "similarity"}]` |
| `delete(name)` | `bool` | Remove face (mark deleted in hnswlib) |
| `list_names()` | `list[str]` | All enrolled names |
| `get_embedding(name)` | `np.ndarray \| None` | Retrieve stored embedding |
| `count` | `int` | Number of active faces (property) |

**Search Return Format:**

```python
[{
    "name": str,
    "distance": float,     # hnswlib cosine distance (0–2)
    "similarity": float,   # 1 - distance
}]
```

---

### 7.12 `src/utils/` — Utilities

#### Function: `download_model(url, dest) -> None`

Downloads a model file if it doesn't exist. Shows progress in debug mode.

**Raises:** `ModelDownloadError` on HTTP failure.

#### Function: `cosine_similarity(a, b) -> float`

Cosine similarity between two embedding vectors: `dot(a, b) / (||a|| × ||b||)`. Returns 0.0 if either vector has near-zero norm.

---

### 7.13 `src/logger/` — Logging

#### Function: `setup_logging(verbose=False, quiet=False) -> None`

Configure the root `login` logger. Called once at startup.

| Mode | Level | Format |
|------|-------|--------|
| Default | INFO | `[I] message` |
| Verbose | DEBUG | `[D] module: message` |
| Quiet | WARNING | `[W] message` |

#### Function: `get_logger(name) -> logging.Logger`

Returns a child logger under the `login` namespace: `login.<name>`.

---

### 7.14 `src/exceptions/` — Error Hierarchy

```
BaseException
├── ValueError
│   └── ConfigValidationError(param, value, reason)
├── RuntimeError
│   └── CameraError
│       ├── CameraOpenError(source)
│       └── CameraTimeoutError(expected_frames, timeout)
└── Exception
    ├── ModelError
    │   ├── ModelDownloadError(url, reason)
    │   ├── ModelNotFoundError(path, hint)  [also FileNotFoundError]
    │   └── ModelExtractionError(archive, expected)
    └── FaceError
        ├── NoFaceDetectedError(context)
        └── DuplicateFaceError(name, existing_name, similarity)
```

---