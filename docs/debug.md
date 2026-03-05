## 10. Debug System

Enable debug mode with the `--debug` flag. Two debug savers produce timestamped session folders.

### 10.1 `DebugSaver` — Enhancement Pipeline (8 Stages)

**Output:** `debug/session_YYYYMMDD_HHMMSS/`

| Stage | File | Content |
|-------|------|---------|
| 1 | `frame0_1_raw.png` | Original camera frame |
| 2 | `frame0_2_detected.png` | Full frame with bbox + keypoints |
| 3 | `frame0_face0_3_crop.png` | Raw face crop |
| 4 | `frame0_face0_4_clahe.png` | After CLAHE |
| 5 | `frame0_face0_5_sr.png` | After super-resolution |
| 6 | `frame0_face0_6_mesh.png` | 3D mesh overlay |
| 7 | `frame0_face0_7_aligned.png` | 112×112 ArcFace-aligned |
| 8 | `frame0_face0_8_result.png` | Name + similarity annotation |

### 10.2 `RecognitionDebugSaver` — Recognition Pipeline (10 Stages)

**Output:** `debug/recognition/session_YYYYMMDD_HHMMSS/`

| Stage | File | Content |
|-------|------|---------|
| 01 | `frame0_01_raw.png` | Original camera frame |
| 02 | `frame0_02_detected.png` | All bboxes + keypoints |
| 03 | `frame0_face0_03_crop.png` | Raw face crop |
| 04 | `frame0_face0_04_spoof.png` | Z-depth heatmap (blue=close, red=far) |
| 05 | `frame0_face0_05_gamma.png` | After gamma correction |
| 06 | `frame0_face0_06_clahe.png` | After CLAHE |
| 07 | `frame0_face0_07_mesh.png` | 3D mesh overlay |
| 08 | `frame0_face0_08_aligned.png` | 112×112 ArcFace-aligned |
| 09 | `frame0_face0_09_candidates.png` | HNSW top-K with scores |
| 10 | `frame0_face0_10_result.png` | Final result (green/red/magenta) |

**Result colors:**
- Green = matched identity
- Red = unknown face
- Magenta = spoof detected

---