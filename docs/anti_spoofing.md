## 9. Anti-Spoofing System

The system uses a **dual-layer anti-spoofing** pipeline. Both layers must pass for a face to be accepted.

### Layer 1: 3D Z-Depth Analysis

**Module:** `src/mesh/liveness.py`

Analyzes the z-coordinates of 478 face landmarks. A real face has significant depth variation (nose protrudes, eyes are recessed). A flat photo has nearly co-planar landmarks.

| Check | Threshold | Real Face | Photo/Screen |
|-------|-----------|-----------|--------------|
| z_range (max−min) | ≥ 0.03 | 0.04 – 0.15+ | 0.00 – 0.02 |
| z_std | ≥ 0.008 | 0.01+ | < 0.008 |

**Limitations:**
- Video replay can bypass (shows depth from original recording)
- 3D printed masks can bypass
- Monocular depth estimation inherently imprecise

### Layer 2: LBP Texture Analysis

**Module:** `src/mesh/texture_liveness.py`

Analyzes face crop texture using Local Binary Patterns (LBP) and high-frequency energy. Real skin has diverse micro-texture; recaptured images (printed or screen) have different texture characteristics.

**LBP Computation:**
1. Resize crop to 128×128
2. For each pixel, compare with 8 neighbors (clockwise)
3. Generate 8-bit code (0–255)
4. Build normalized 256-bin histogram
5. Compute histogram variance

**High-Frequency Energy:**
1. Apply Laplacian filter
2. Compute mean squared energy
3. Normalize by total image energy

| Check | Threshold | Real Face | Recaptured |
|-------|-----------|-----------|------------|
| hist_var | ≥ 0.00003 | ~0.00007 | Low |
| hf_energy | ≥ 0.01 | ~0.03 | Low |

### Anti-Spoof Pipeline Order

```
Z-Depth Check (fast, uses existing landmarks)
    │
    ├── FAIL → reject immediately (no LBP check needed)
    │
    └── PASS → LBP Texture Check (uses crop image)
                │
                ├── FAIL → reject
                └── PASS → proceed to recognition
```

### Optional: Blink-Based Liveness

When `--liveness` is enabled:
- Uses Eye Aspect Ratio (EAR) to detect blinks
- Requires user to blink within 10 seconds
- Separate from z-depth/LBP (runs BEFORE face embedding)
- Disabled by default (`LIVENESS_ENABLED = False`)

---