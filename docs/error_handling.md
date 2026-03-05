## 13. Error Handling

### Exit Codes

| Code | Error Type | Description |
|------|-----------|-------------|
| 0 | — | Success |
| 1 | `ConfigValidationError` / General | Bad config value or unexpected error |
| 2 | `CameraError` | Camera cannot open or times out |
| 3 | `ModelError` | Model download/load/extraction failure |
| 4 | `FaceError` | No face detected or duplicate face |
| 130 | `KeyboardInterrupt` | User pressed Ctrl+C |

### Error Recovery

- **Camera failures:** Release and exit with code 2
- **Model download failures:** Retry manually, models are cached after first download
- **Corrupted databases:** Auto-detected on load, fresh database created with warning
- **HNSW index corruption:** Automatically rebuilt from JSON database via `_sync_json_to_hnsw()`

---