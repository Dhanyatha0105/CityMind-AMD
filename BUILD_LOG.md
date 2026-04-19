# 🔧 CityMind — BUILD LOG & ARCHITECTURE TRACKER

> **Purpose:** Track what's built, layer interactions, potential errors, and integration status.  
> **Last Updated:** 28 Feb 2026, Hour 18

---

## ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────┐
│  LAYER 6: VISUALIZATION (Streamlit + Plotly + PDF)   │
│  Inputs: All layer outputs                           │
│  Status: ✅ COMPLETE                                 │
├──────────────────────────────────────────────────────┤
│  LAYER 5: MULTI-AGENT INTELLIGENCE (GAIA + RAG)     │
│  Inputs: L2 defects, L3 3D model, L4 twin data      │
│  Status: ✅ COMPLETE                                 │
├──────────────────────────────────────────────────────┤
│  LAYER 4: DIGITAL TWIN ENGINE                        │
│  Inputs: L3 point cloud + L2 semantic labels         │
│  Status: ✅ COMPLETE                                 │
├──────────────────────────────────────────────────────┤
│  LAYER 3: 3D RECONSTRUCTION (COLMAP + synthetic)     │
│  Inputs: L1 keyframes + L2 depth maps               │
│  Status: ✅ COMPLETE                                 │
├──────────────────────────────────────────────────────┤
│  LAYER 2: EDGE PERCEPTION (Depth + Detection)        │
│  Inputs: L1 keyframes                                │
│  Status: ✅ COMPLETE                                 │
├──────────────────────────────────────────────────────┤
│  LAYER 1: VIDEO INGESTION (OpenCV + FFmpeg)          │
│  Inputs: Raw video file (MP4)                        │
│  Status: ✅ COMPLETE                                 │
└──────────────────────────────────────────────────────┘
```

## DATA FLOW — WHAT CONNECTS TO WHAT

```
Video (MP4) 
  → L1: frame_extractor → keyframes/ (PNG images)
  → L1: quality_filter → filtered_keyframes/ (clean images)
  → L1: metadata → metadata.json (GPS, timestamps)
  
filtered_keyframes/
  → L2: depth_estimation → depth_maps/ (numpy arrays + visualizations)
  → L2: object_detection → detections.json (bounding boxes, labels, scores)
  → L2: defect_detection → defects.json (defect boxes, types, severity 0-10)
  
filtered_keyframes/ + depth_maps/
  → L3: colmap_pipeline → sparse_cloud.ply + cameras.json
  → L3: gaussian_splatting → dense_model/ (optional, stretch)
  
sparse_cloud.ply + detections.json + defects.json
  → L3: semantic_projection → labeled_cloud.ply (each point has labels)
  
labeled_cloud.ply + defects.json
  → L4: twin_engine → digital_twin.json (structured twin data)
  → L4: deviation_analysis → deviation_heatmap.json

digital_twin.json + defects.json + detections.json
  → L5: orchestrator → runs 4 agents in sequence:
    → inspector_agent → classified_defects.json (severity classification)
    → compliance_agent → compliance_report.json (code violations via RAG)
    → safety_agent → risk_assessment.json (risk score 0-100)
    → report_agent → inspection_report.md + inspection_report.pdf

ALL OUTPUTS
  → L6: app.py (Streamlit) → Interactive dashboard
    → Upload video tab
    → Defect detection results tab (images with overlays)
    → 3D viewer tab (Plotly scatter/mesh)
    → Agent reports tab (findings, compliance, risk)
    → Download PDF report
```

## LAYER INTERACTION MATRIX

| Producer → Consumer | L1 | L2 | L3 | L4 | L5 | L6 |
|---------------------|----|----|----|----|----|----|
| **L1** (Ingestion)  | —  | ✅ keyframes | ✅ keyframes | ❌ | ❌ | ✅ video info |
| **L2** (Perception) | ❌ | —  | ✅ depth maps | ✅ labels | ✅ defects | ✅ detections |
| **L3** (Reconstruction) | ❌ | ❌ | — | ✅ point cloud | ✅ 3D data | ✅ 3D model |
| **L4** (Digital Twin) | ❌ | ❌ | ❌ | — | ✅ twin data | ✅ twin viz |
| **L5** (Agents) | ❌ | ❌ | ❌ | ❌ | — | ✅ reports |

## POTENTIAL ERRORS & WATCH ITEMS

### Known Risk: COLMAP Installation
- **Risk:** COLMAP may not be installed via brew on macOS ARM
- **Mitigation:** Use `brew install colmap` or fall back to pre-computed point cloud
- **Status:** ⬜ TO CHECK

### Known Risk: Large Model Downloads
- **Risk:** Depth Anything v2 and YOLO models are large (~200MB-1GB)
- **Mitigation:** Start downloads early, use smaller model variants
- **Status:** ⬜ TO CHECK

### Known Risk: ONNX Runtime on macOS
- **Risk:** Vitis AI EP not available on macOS (it's for Ryzen AI)
- **Mitigation:** Use CPU/CoreML EP for demo. Show Vitis AI config as docs.
- **Status:** ⬜ EXPECTED — document in README that NPU deployment targets Windows Ryzen AI

### Known Risk: LLM for Agents
- **Risk:** Need local LLM or API for agent reasoning
- **Mitigation:** Use OpenAI API (fastest) or local Ollama. Config to support both.
- **Status:** ⬜ TO DECIDE

### Known Risk: Memory Usage
- **Risk:** 3DGS + depth models + YOLO running together may exceed RAM
- **Mitigation:** Process sequentially, not in parallel. Clear models between layers.
- **Status:** ⬜ MONITOR

## BUILD LOG

### Entry 1 — Hour 0 (28 Feb 2026)
- Created project structure
- Created tracking documents
- Next: Layer 1 Video Ingestion

### Entry 2 — Hour 1-2
- ✅ Layer 1 complete: frame_extractor.py, quality_filter.py, metadata.py
- Keyframe extraction with SSIM dedup and blur filtering

### Entry 3 — Hour 2-6
- ✅ Layer 2 complete: depth_estimation.py, object_detection.py, defect_detection.py, npu_inference.py
- Depth Anything v2, YOLOv8, defect detection with heuristic fallbacks
- ONNX Runtime + Vitis AI EP NPU wrapper

### Entry 4 — Hour 6-10
- ✅ Layer 3 complete: colmap_pipeline.py, semantic_projection.py, point_cloud_utils.py
- COLMAP SfM with synthetic fallback
- 2D→3D semantic label projection
- PLY read/write, synthetic building generation

### Entry 5 — Hour 10-12
- ✅ Layer 4 complete: twin_engine.py
- Health Index scoring (0-100, A-F grade)
- Zone-based analysis, defect aggregation

### Entry 6 — Hour 12-16
- ✅ Layer 5 complete: orchestrator.py, prompts/, RAG subsystem (ingest.py, retriever.py)
- 4-agent pipeline: Inspector, Compliance, Safety, Report
- RAG with FAISS + sentence-transformers for building code retrieval
- Sample building codes (ACI, ASCE, IBC)
- LLM backends: OpenAI, Ollama, deterministic fallback

### Entry 7 — Hour 16-18
- ✅ Layer 6 complete: app.py, viewer_3d.py, charts.py, report_pdf.py
- Full Streamlit dashboard with 7 tabs
- Interactive 3D Plotly point cloud viewer
- Defect distribution, severity, risk gauge, zone health charts
- PDF report generation with FPDF2
- ✅ pipeline.py — End-to-end pipeline orchestrator
- ✅ Training scripts: export_onnx.py, quantize_vitis.py
- ✅ vaip_config.json — NPU configuration
- ✅ Documentation: README.md, architecture.md, amd_tech_integration.md, setup_guide.md
- ✅ LICENSE (MIT)
- ✅ Demo script: scripts/run_demo.py

### CURRENT STATUS: ALL 6 LAYERS COMPLETE ✅
- All core modules implemented
- Dashboard ready for demo
- Documentation complete
- Remaining: Testing, performance optimization

---
*This file is updated as each layer is built.*
