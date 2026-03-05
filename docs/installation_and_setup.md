## 3. Installation & Setup

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- Webcam (DirectShow on Windows)

### Install

```bash
uv sync
```

### Dependencies (from pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| `mediapipe` | ≥ 0.10.32 | Face detection + 478 landmarks |
| `onnxruntime` | ≥ 1.24.2 | ArcFace inference |
| `opencv-python` | ≥ 4.13.0 | Image processing |
| `opencv-contrib-python` | ≥ 4.13.0 | Extended OpenCV |
| `hnswlib` | ≥ 0.8.0 | HNSW vector index |
| `torch` | ≥ 2.10.0 | Real-ESRGAN / GFPGAN inference |
| `torchvision` | ≥ 0.25.0 | Torch vision utilities |
| `openvino` | ≥ 2026.0.0 | OpenVINO SR backend |
| `requests` | ≥ 2.32.5 | Model downloads |

### Models (Auto-Downloaded)

Models are auto-downloaded to `models/` on first use:

| Model | File | Size | Source |
|-------|------|------|--------|
| FaceLandmarker | `face_landmarker.task` | ~4 MB | MediaPipe |
| BlazeFace | `blaze_face_short_range.tflite` | ~200 KB | MediaPipe |
| ArcFace MobileFaceNet | `w600k_mbf.onnx` | ~12 MB | InsightFace |
| Real-ESRGAN | `realesr-general-x4v3.pth` | ~5 MB | xinntao |
| GFPGAN v1.4 | `GFPGANv1.4.pth` | ~350 MB | TencentARC |

---