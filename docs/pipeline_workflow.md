
## 8. Pipeline Workflows

### 8.1 Enhancement Pipeline

```
Camera Frame
    │
    ▼
┌─────────────┐
│  Detect Face │ ← MediaPipeDetector.detect()
│  (bbox, crop)│
└─────┬───────┘
      │ for each face
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────┐
│  Raw Crop   │ ──▶ │auto_brighten│ ──▶ │ apply_clahe │ ──▶ │sr.upscale│
│             │     │(gamma < 1)  │     │(LAB L-chan) │     │   (4×)   │
└─────────────┘     └─────────────┘     └─────────────┘     └────┬─────┘
                                                                  │
                                                                  ▼
                                                          img/enhanced/
```

### 8.2 Enrollment Pipeline

```
GuidedEnrollment.capture()
    │ (5 diverse frames)
    ▼
┌──────────────────────────────────────────────────┐
│  For each frame:                                  │
│                                                   │
│  1. detect(frame) → face + landmarks              │
│  2. _check_spoof(frame, face)                     │
│     ├── check_liveness(landmarks) → z-depth gate  │
│     └── check_texture(crop) → LBP gate            │
│  3. _get_embedding(frame, face) → 512-dim         │
│  4. _get_mesh_embedding(frame, face) → 1434-dim   │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Post-processing:                                 │
│                                                   │
│  1. Average ArcFace embeddings → re-normalize     │
│  2. Duplicate check (HNSW search, threshold 0.55) │
│  3. Average mesh embeddings → re-normalize        │
│  4. Save to face_db.json + face_frames.json       │
│     + mesh_db.json + hnsw_index.bin               │
└──────────────────────────────────────────────────┘
```

### 8.3 Recognition Pipeline

```
Camera Frame
    │
    ▼
┌──────────────────────┐
│  detect(frame)        │ → face dict + 478 landmarks
└─────────┬────────────┘
          │
    ┌─────┴─────┐
    │Anti-Spoof │
    │  Gate     │
    ├───────────┤
    │ Z-Depth   │ check_liveness(landmarks)
    │ z_range ≥ 0.03 AND z_std ≥ 0.008
    ├───────────┤
    │ LBP Tex.  │ check_texture(crop)
    │ hist_var ≥ 0.00003 AND hf_energy ≥ 0.01
    └─────┬─────┘
          │ PASS
          ▼
┌──────────────────────┐
│ ArcFace Embed        │ align_face() → get_embedding()
│ (512-dim, L2-norm)   │
├──────────────────────┤
│ Mesh Embed           │ mesh_to_embedding(landmarks)
│ (1434-dim, L2-norm)  │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Stage 1: HNSW Search │ search(embedding, k=3)
│ → Top-3 candidates   │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Stage 2: Frame Verify│ max cosine_sim vs stored per-frame embeddings
│                      │
│ Stage 3: Mesh Fusion │ cosine_sim(query_mesh, stored_mesh)
│                      │
│ Combined =           │
│  0.85 × ArcFace     │
│ +0.15 × Mesh        │
└─────────┬────────────┘
          │
          ▼
    combined ≥ 0.45 ?
    ├── YES → "Matched: <name> (similarity: X.XXX)"
    └── NO  → "Unknown face (best similarity: X.XXX)"
```

---