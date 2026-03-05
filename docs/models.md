## 12. Models Reference

### 12.1 MediaPipe FaceLandmarker

- **File:** `models/face_landmarker.task`
- **Format:** TFLite Task bundle
- **Output:** 478 3D landmarks per face (x, y, z normalized [0,1])
- **Used by:** `MediaPipeDetector`, `FaceMesh`, `BlinkDetector`, `GuidedEnrollment`
- **Source:** `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`

### 12.2 BlazeFace

- **File:** `models/blaze_face_short_range.tflite`
- **Format:** TFLite
- **Output:** Face bounding boxes + 6 keypoints
- **Used by:** `BlazeFaceDetector` (legacy backend)
- **Source:** `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite`

### 12.3 ArcFace MobileFaceNet (w600k_mbf)

- **File:** `models/w600k_mbf.onnx`
- **Format:** ONNX
- **Input:** 112×112×3 RGB, normalized to [-1, 1]
- **Output:** 512-dim L2-normalized embedding
- **Used by:** `ArcFaceEmbedder`
- **Source:** `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip` (extracted)
- **Training data:** WebFace 600K identities

### 12.4 Real-ESRGAN (SRVGGNetCompact)

- **File:** `models/realesr-general-x4v3.pth`
- **Format:** PyTorch state dict
- **Architecture:** SRVGGNetCompact (64 features, 32 conv layers, PixelShuffle ×4)
- **Input:** Any size BGR
- **Output:** 4× upscaled BGR
- **Source:** `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth`

### 12.5 GFPGAN v1.4

- **File:** `models/GFPGANv1.4.pth`
- **Format:** PyTorch state dict
- **Architecture:** GFPGANv1Clean (U-Net encoder + StyleGAN2 decoder + SFT modulation)
- **Input:** Any size BGR (resized to 512×512 internally)
- **Output:** Restored + upscaled face
- **Source:** `https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth`

---