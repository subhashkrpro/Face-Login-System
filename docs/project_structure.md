## 5. Project Structure

```
D:\LOGIN/
├── main.py                          # CLI entry point & dispatcher
├── pyproject.toml                   # Project metadata & dependencies
├── README.md                        # Basic readme
├── .gitignore                       # Git ignore rules
├── .python-version                  # Python version pin
│
├── config/                          # Configuration package
│   ├── __init__.py                  # Central re-exports of all config
│   ├── defaults.py                  # Tunable thresholds & parameters
│   ├── paths.py                     # File & directory paths
│   ├── models.py                    # Model URLs & architecture params
│   ├── camera.py                    # Camera hardware settings
│   └── validation.py               # Startup config validation
│
├── cli/                             # CLI command handlers
│   ├── __init__.py                  # Package marker
│   ├── enhance.py                   # cmd_enhance() — capture + enhance
│   ├── enroll.py                    # cmd_enroll() — guided enrollment
│   ├── recognize.py                 # cmd_recognize() — face recognition
│   ├── list_faces.py                # cmd_list() — list enrolled
│   ├── delete.py                    # cmd_delete() — delete enrolled
│   └── audit.py                     # cmd_audit() — duplicate scan
│
├── src/                             # Core source package
│   ├── __init__.py                  # Package marker
│   │
│   ├── capture/                     # Camera capture
│   │   ├── __init__.py              # Exports: FastStream
│   │   └── stream.py               # FastStream — threaded webcam
│   │
│   ├── detection/                   # Face detection
│   │   ├── __init__.py              # Factory: create_detector()
│   │   ├── detector.py              # BlazeFaceDetector (legacy)
│   │   └── mediapipe_detector.py    # MediaPipeDetector (primary)
│   │
│   ├── enhancement/                 # Image enhancement
│   │   ├── __init__.py              # Exports: FaceEnhancer, apply_clahe, etc.
│   │   ├── enhancer.py              # FaceEnhancer — full pipeline
│   │   ├── clahe.py                 # CLAHE contrast enhancement
│   │   ├── low_light.py             # Adaptive gamma correction
│   │   └── sr/                      # Super Resolution backends
│   │       ├── __init__.py          # Exports: create_sr_backend
│   │       ├── sr_factory.py        # Factory function
│   │       ├── sr_model.py          # SRVGGNetCompact neural network
│   │       ├── sr_realesrgan.py     # Real-ESRGAN PyTorch backend
│   │       ├── sr_openvino.py       # OpenVINO IR backend
│   │       ├── sr_gfpgan.py         # GFPGAN face restoration backend
│   │       ├── arch_gfpgan.py       # GFPGANv1Clean architecture
│   │       └── arch_stylegan2.py    # StyleGAN2 generator (GFPGAN decoder)
│   │
│   ├── recognition/                 # Face recognition
│   │   ├── __init__.py              # Exports: FaceRecognizer, ArcFaceEmbedder
│   │   ├── recognizer.py            # FaceRecognizer — main orchestrator
│   │   ├── embedding/               # Embedding extraction
│   │   │   ├── __init__.py          # Exports: ArcFaceEmbedder, align_face
│   │   │   ├── arcface.py           # ArcFaceEmbedder — ONNX inference
│   │   │   ├── align.py             # Face alignment for ArcFace
│   │   │   └── model_loader.py      # ArcFace model download & extraction
│   │   └── enrollment/              # Enrollment tools
│   │       ├── __init__.py          # Exports: GuidedEnrollment, DuplicateChecker
│   │       ├── guided_enroll.py     # GuidedEnrollment — oval UI capture
│   │       └── duplicate_checker.py # DuplicateChecker — prevents re-enrollment
│   │
│   ├── mesh/                        # 3D face mesh & anti-spoofing
│   │   ├── __init__.py              # Exports: FaceMesh, mesh_to_embedding, etc.
│   │   ├── face_mesh.py             # FaceMesh — 478 landmark extraction
│   │   ├── mesh_embedding.py        # mesh_to_embedding() — 1434-dim vector
│   │   ├── liveness.py              # Z-depth anti-spoofing
│   │   └── texture_liveness.py      # LBP texture anti-spoofing
│   │
│   ├── liveness/                    # Blink-based liveness detection
│   │   ├── __init__.py              # Exports: check_liveness, BlinkDetector
│   │   ├── blink_detector.py        # BlinkDetector — EAR-based blink detection
│   │   ├── ear.py                   # compute_ear() — Eye Aspect Ratio
│   │   └── liveness_check.py        # check_liveness() — blink check wrapper
│   │
│   ├── indexing/                    # Vector indexing
│   │   ├── __init__.py              # Exports: HNSWIndex
│   │   └── hnsw_index.py            # HNSWIndex — hnswlib wrapper
│   │
│   ├── utils/                       # Shared utilities
│   │   ├── __init__.py              # Exports: download_model, cosine_similarity
│   │   ├── downloader.py            # Model file downloader
│   │   └── similarity.py            # Cosine similarity function
│   │
│   ├── logger/                      # Logging infrastructure
│   │   ├── __init__.py              # Exports: get_logger, setup_logging
│   │   └── setup.py                 # Logger configuration
│   │
│   └── exceptions/                  # Custom exception hierarchy
│       ├── __init__.py              # Central re-exports
│       ├── camera.py                # CameraError, CameraOpenError, CameraTimeoutError
│       ├── model.py                 # ModelError, ModelDownloadError, etc.
│       ├── face.py                  # FaceError, NoFaceDetectedError, etc.
│       └── config.py                # ConfigValidationError
│
├── debug/                           # Debug output package
│   ├── __init__.py                  # Exports: DebugSaver, RecognitionDebugSaver
│   ├── debug_saver.py               # DebugSaver — 8-stage pipeline images
│   ├── recognition_debug.py         # RecognitionDebugSaver — 10-stage images
│   └── recognition/                 # Recognition debug session folders
│       └── session_*/               # Timestamped session outputs
│
├── models/                          # Downloaded model weights
│   ├── face_landmarker.task         # MediaPipe FaceLandmarker
│   ├── blaze_face_short_range.tflite# BlazeFace detector
│   ├── w600k_mbf.onnx               # ArcFace MobileFaceNet
│   ├── realesr-general-x4v3.pth     # Real-ESRGAN
│   └── GFPGANv1.4.pth               # GFPGAN face restoration
│
├── embeddings/                      # Enrolled face data
│   ├── face_db.json                 # Averaged ArcFace embeddings
│   ├── face_frames.json             # Individual frame ArcFace embeddings
│   ├── mesh_db.json                 # Averaged 3D mesh embeddings
│   ├── hnsw_index.bin               # HNSW binary index
│   └── hnsw_labels.json             # HNSW label ↔ name mapping
│
└── img/                             # Output images
    └── enhanced/                    # Enhanced face images
```

---