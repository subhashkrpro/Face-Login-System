
## 4. CLI Commands Reference

### General Syntax

```bash
uv run main.py [global-flags] <command> [command-flags]
```

### Global Flags

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Show DEBUG-level messages (detailed timing, internal state) |
| `-q`, `--quiet` | Show only WARNING/ERROR messages |
| `--debug` | Save intermediate pipeline images to `debug/` folder |

### Commands

#### `enhance` — Capture + Detect + Enhance Faces

Captures frames from webcam, detects faces, applies low-light correction → CLAHE → super-resolution, and saves enhanced images.

```bash
uv run main.py enhance
uv run main.py --debug enhance                    # save debug images
uv run main.py enhance --backend gfpgan            # use GFPGAN for SR
uv run main.py enhance --backend openvino --device GPU
uv run main.py enhance --frames 5 --confidence 0.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `realesrgan` | SR backend: `realesrgan`, `openvino`, `gfpgan` |
| `--device` | auto | OpenVINO device: `NPU`, `GPU`, `CPU` |
| `--confidence` | `0.3` | Minimum face detection confidence |
| `--frames` | `3` | Number of frames to capture |

**Output:** Enhanced face images saved to `img/enhanced/`

---

#### `enroll` — Register a Face

Guided enrollment with oval UI. User moves head in a circle for pose diversity.

```bash
uv run main.py enroll --name Alice
uv run main.py enroll --name Bob --frames 7
uv run main.py --debug enroll --name Alice
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | *required* | Name to register |
| `--frames` | `5` | Number of diverse frames to capture |

**Process:**
1. Opens webcam with oval guide
2. Phase 1: Capture center (look straight)
3. Phase 2: Capture circle (move head around, captures at diverse angles)
4. Anti-spoofing check per frame (z-depth + LBP texture)
5. Duplicate detection against existing database
6. Stores averaged ArcFace + mesh embeddings

---

#### `recognize` — Identify a Face

Captures one frame and identifies the person against enrolled database.

```bash
uv run main.py recognize
uv run main.py recognize --threshold 0.5
uv run main.py recognize --liveness           # force blink check
uv run main.py recognize --no-liveness         # skip blink check
uv run main.py --debug recognize
```

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.45` | Minimum combined similarity for a match |
| `--liveness` | from config (`False`) | Force blink-based liveness check on |
| `--no-liveness` | — | Skip blink-based liveness check |

**Process:**
1. Camera warmup overlaps model loading (parallel)
2. Optional blink-based liveness check
3. Detect face + 478 landmarks
4. Anti-spoofing: z-depth → LBP texture
5. ArcFace embedding extraction
6. HNSW top-K candidate search
7. Stage-2 verification against individual frame embeddings
8. Stage-3 mesh geometric similarity fusion
9. Combined score threshold check

**Exit codes:** 0=success, 2=camera error, 3=model error, 4=face error

---

#### `list` — List Enrolled Faces

```bash
uv run main.py list
```

Shows all enrolled face names and count. No additional flags.

---

#### `delete` — Remove an Enrolled Face

```bash
uv run main.py delete --name Alice
```

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | *required* | Name to delete |

Removes from HNSW index, face_db.json, face_frames.json, and mesh_db.json.

---

#### `audit` — Scan for Duplicate Faces

```bash
uv run main.py audit
uv run main.py audit --threshold 0.6
```

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `0.55` | Similarity above which pairs are flagged as duplicates |

Compares every enrolled face against every other. Reports duplicate pairs sorted by similarity.

---
