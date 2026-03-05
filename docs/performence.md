## 14. Performance Notes

### Optimization Techniques

1. **Parallel model loading:** Detector (~9s) and ArcFace (~200ms) load via `ThreadPoolExecutor` — ArcFace is hidden behind detector init.

2. **Camera-model overlap:** In recognize mode, camera starts BEFORE model loading. Warmup time runs in parallel with init.

3. **Shared FaceLandmarker:** In enroll mode, `FaceRecognizer.detector.landmarker` is shared with `GuidedEnrollment` — saves ~9s duplicate TFLite load.

4. **Parallel SR init:** `FaceEnhancer` loads detector and SR backend concurrently.

5. **Landmark reuse:** `MediaPipeDetector` returns landmarks with each face dict. These are reused for:
   - Mesh embedding (no separate FaceMesh extraction)
   - Z-depth anti-spoofing
   - ArcFace alignment keypoints

6. **HNSW O(log n) search:** Approximate nearest neighbor instead of brute-force.

### Typical Timings (Windows, CPU-only)

| Operation | Time |
|-----------|------|
| FaceRecognizer init | ~7.5s |
| Face detection (per frame) | ~4ms |
| ArcFace embedding | ~15ms |
| HNSW search (1000 faces) | < 1ms |
| Full recognize pipeline | ~50ms (after init) |
| Guided enrollment capture | ~10–20s (user-dependent) |
| Real-ESRGAN upscale (face crop) | ~200ms |
| GFPGAN restoration | ~500ms |

---