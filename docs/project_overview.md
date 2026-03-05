## 1. Project Overview

LOGIN is a complete face detection, enhancement, and recognition pipeline built from scratch using MediaPipe, ArcFace, and 3D mesh geometry. It provides:

- **Face Detection** — MediaPipe FaceLandmarker (478 3D landmarks in one pass)
- **Face Enhancement** — Low-light gamma correction → CLAHE → Super Resolution (4×)
- **Face Recognition** — ArcFace 512-dim embeddings + 3D mesh 1434-dim geometric embeddings
- **Anti-Spoofing** — Dual-layer: 3D z-depth analysis + LBP texture analysis
- **Liveness Detection** — Optional blink-based EAR (Eye Aspect Ratio) check
- **Guided Enrollment** — Phone-style oval UI with head-pose diversity
- **Database Management** — HNSW vector index, duplicate detection, audit

### Key Metrics

| Metric | Value |
|--------|-------|
| Init time (recognize) | ~7.5 seconds |
| Per-frame detection | ~4 ms |
| ArcFace embedding | ~15 ms |
| Recognition threshold | 0.45 (combined score) |
| Embedding dimension | 512 (ArcFace) + 1434 (mesh) |
| HNSW search complexity | O(log n) |
