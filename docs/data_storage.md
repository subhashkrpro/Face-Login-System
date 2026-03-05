
## 11. Data Storage

### 11.1 `embeddings/face_db.json`

Averaged ArcFace embeddings per enrolled person.

```json
{
  "Alice": [0.0123, -0.0456, ...],   // 512 float values
  "Bob": [0.0789, -0.0012, ...]
}
```

### 11.2 `embeddings/face_frames.json`

Individual frame ArcFace embeddings (for Stage 2 verification).

```json
{
  "Alice": [
    [0.0123, ...],    // frame 1 embedding (512)
    [0.0124, ...],    // frame 2 embedding (512)
    ...               // up to ENROLL_FRAMES embeddings
  ]
}
```

### 11.3 `embeddings/mesh_db.json`

Averaged 3D mesh geometric embeddings.

```json
{
  "Alice": [0.00123, -0.00456, ...],  // 1434 float values
}
```

### 11.4 `embeddings/hnsw_index.bin`

Binary HNSW index file (hnswlib format). Contains indexed ArcFace embeddings for fast search.

### 11.5 `embeddings/hnsw_labels.json`

HNSW label ↔ name mapping.

```json
{
  "name_to_id": {"Alice": 0, "Bob": 1},
  "id_to_name": {"0": "Alice", "1": "Bob"},
  "next_id": 2
}
```

---