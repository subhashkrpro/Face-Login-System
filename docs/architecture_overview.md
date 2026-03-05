## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│              CLI dispatcher + error handling            │
├────────┬────────┬──────────┬──────┬────────┬────────────┤
│enhance │ enroll │recognize │ list │ delete │   audit    │
│ (cli/) │ (cli/) │  (cli/)  │(cli/)│ (cli/) │   (cli/)   │
├────────┴────────┴──────────┴──────┴────────┴────────────┤
│                    src/ (core modules)                  │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ capture/ │detection/│enhance-  │recognize/│   mesh/     │
│          │          │  ment/   │          │             │
│FastStream│MediaPipe │FaceEn-   │FaceRecog-│FaceMesh     │
│(threaded │Detector  │hancer    │nizer     │mesh_embed   │
│ webcam)  │BlazeFace │CLAHE     │ArcFace   │z-depth      │
│          │Detector  │LowLight  │Alignment │LBP texture  │
│          │          │SR(3 be-  │HNSW Index│             │
│          │          │  ends)   │Duplicate │             │
│          │          │          │GuidedEnr │             │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│  liveness/  │  indexing/   │  utils/  │  logger/  │excep│
│  BlinkDet   │  HNSWIndex   │download  │  setup    │tions│
│  EAR calc   │              │cosine_sim│  logging  │     │
├─────────────┴──────────────┴──────────┴───────────┴─────┤
│                     config/ package                     │
│  paths.py │ camera.py │ models.py │ defaults.py │valid. │
├─────────────────────────────────────────────────────────┤
│                     debug/ package                      │
│         DebugSaver (8 stages) │ RecognitionDebugSaver   │
│                               │ (10 stages)             │
└─────────────────────────────────────────────────────────┘
```