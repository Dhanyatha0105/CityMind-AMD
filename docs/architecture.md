# 🏗️ CityMind — Detailed System Architecture

## Overview

CityMind is a 6-layer AI pipeline that transforms infrastructure video/images into an intelligent 3D digital twin with automated defect analysis, building code compliance checking, and risk assessment. The system is designed for deployment on AMD edge AI hardware.

---

## Layer 1: Video Ingestion

**Purpose:** Extract high-quality keyframes from infrastructure inspection video.

**Components:**
- `frame_extractor.py` — FFmpeg/OpenCV-based keyframe extraction
  - Scene change detection (SSIM-based)
  - Configurable FPS sampling
  - Handles MP4, AVI, MOV formats
  
- `quality_filter.py` — Multi-criteria quality assessment
  - **Blur Detection:** Laplacian variance (threshold: 100)
  - **SSIM Deduplication:** Removes frames with SSIM > 0.95
  - **Brightness/contrast check:** Rejects over/under-exposed frames
  
- `metadata.py` — Metadata extraction
  - Video duration, resolution, codec info
  - GPS coordinates (if available in EXIF)
  - Timestamp extraction for temporal tracking

**Input:** Video file or image directory  
**Output:** Directory of quality-filtered keyframes + metadata JSON

---

## Layer 2: Edge Perception

**Purpose:** Run AI models on frames for depth, structural elements, and defects.

**AMD Technology:**
- AMD Ryzen AI NPU (XDNA) for INT8 inference
- ONNX Runtime + Vitis AI Execution Provider
- AMD CVML Library for depth estimation pipeline

**Components:**
- `depth_estimation.py` — Depth Anything v2
  - HuggingFace Transformers pipeline
  - ONNX Runtime backend for NPU acceleration
  - Monocular depth → metric depth estimation
  
- `object_detection.py` — YOLOv8/RT-DETR
  - Structural element detection: beam, column, wall, slab, etc.
  - Maps COCO classes to structural categories
  - Confidence threshold: 0.35
  
- `defect_detection.py` — Defect Detection
  - YOLOv8 fine-tuned for: crack, spalling, corrosion, exposed rebar
  - Heuristic fallback: edge detection + color analysis
  - Severity scoring (0-10) based on area, confidence, type
  
- `npu_inference.py` — AMD NPU Wrapper
  - ONNX Runtime session with Vitis AI EP
  - Fallback: CPU EP → CUDA EP → Vitis AI EP
  - vaip_config.json for NPU graph compilation

**Input:** Keyframe images  
**Output:** Per-frame depth maps, structural detections, defect detections

---

## Layer 3: 3D Reconstruction

**Purpose:** Build 3D point cloud from 2D frames.

**AMD Technology:**
- AMD Ryzen AI Max+ iGPU for GPU-accelerated reconstruction
- Vulkan backend for AMD GPU optimization

**Components:**
- `colmap_pipeline.py` — COLMAP Structure-from-Motion
  - Feature extraction (SIFT)
  - Feature matching (exhaustive/sequential)
  - Sparse reconstruction → dense (optional)
  - Synthetic fallback when COLMAP unavailable
  
- `semantic_projection.py` — 2D→3D Label Projection
  - Projects detection bounding boxes to 3D points
  - Assigns structural labels and defect markers
  - Generates colored semantic point cloud
  
- `point_cloud_utils.py` — Point Cloud Utilities
  - PLY read/write (plyfile + Open3D)
  - Voxel downsampling for visualization
  - Synthetic building generation for demos
  - Height-based and label-based coloring

**Input:** Quality keyframes + perception results  
**Output:** Labeled 3D point cloud (.ply) + statistics

---

## Layer 4: Digital Twin Engine

**Purpose:** Create intelligent digital twin with health metrics.

**Components:**
- `twin_engine.py` — Digital Twin Engine
  - Aggregates defects across all frames (deduplication)
  - Aggregates structural element detections
  - **Structural Health Index (SHI):** 0-100 score with A-F grade
    - Based on defect count, severity, confidence
    - Critical defects have amplified impact
  - **Zone Analysis:** Foundation, Lower, Mid, Upper, Roof
  - Building code reference mapping per defect type

**Health Index Grading:**
| Score | Grade | Status | Action |
|-------|-------|--------|--------|
| 90-100 | A | Excellent | Routine monitoring |
| 75-89 | B | Good | Preventive maintenance (6 months) |
| 60-74 | C | Fair | Detailed inspection (3 months) |
| 40-59 | D | Poor | Immediate engineering assessment |
| 0-39 | F | Critical | Restrict access, emergency response |

**Input:** Point cloud + all perception results  
**Output:** Digital twin JSON with complete analysis

---

## Layer 5: Multi-Agent Intelligence

**Purpose:** AI-driven inspection analysis using 4 specialized agents.

**AMD Technology:**
- AMD GAIA Framework for agent orchestration
- Lemonade (TurnkeyML) for LLM optimization
- Ryzen AI Software v1.7 for local LLM inference

**Agent Pipeline (Sequential):**

### Agent 1: Inspector Agent 🔍
- Validates and classifies detected defects
- Determines true positive probability
- Identifies patterns (systemic vs. isolated)
- Recommends urgency level

### Agent 2: Compliance Agent 📋
- RAG-based building code retrieval (ACI, ASCE, IBC)
- Cross-references defects against code requirements
- Identifies violations and severity
- Recommends remediation standards

### Agent 3: Safety Agent 🛡️
- Computes overall risk score (0-100)
- Risk breakdown: structural, compliance, pattern, environmental
- Occupancy recommendation (NORMAL/RESTRICTED/EVACUATE)
- Priority action items

### Agent 4: Report Agent 📄
- Generates comprehensive inspection report
- Includes all agent findings
- Professional formatting with tables
- AMD technology methodology section

**RAG Subsystem:**
- `ingest.py` — PDF/TXT → chunks → FAISS embeddings
- `retriever.py` — Semantic search for relevant code sections
- Default building codes: ACI 318-19, ACI 562-19, ACI 222R-19, ASCE 7-22, IBC 2021

**LLM Backends:**
1. OpenAI (GPT-4o-mini) — highest quality
2. Ollama (Llama 3.1:8b) — local, privacy-preserving
3. Deterministic — template-based fallback, no API needed

**Input:** Digital twin data + RAG context  
**Output:** Agent results JSON + inspection report markdown

---

## Layer 6: Visualization & Dashboard

**Purpose:** Interactive web dashboard for inspection results.

**Components:**
- `app.py` — Streamlit main application
  - Video upload / demo mode
  - Real-time pipeline progress tracking
  - 7 tabbed result views
  
- `viewer_3d.py` — 3D Point Cloud Viewer
  - Plotly Scatter3D interactive visualization
  - Semantic coloring by label
  - Defect highlighting
  - Camera controls and hover info
  
- `charts.py` — Analytics Charts
  - Defect distribution (donut chart)
  - Severity histogram (stacked by category)
  - Risk gauge indicator
  - Zone health bar chart
  - Pipeline performance comparison (AMD vs CPU)
  
- `report_pdf.py` — Report Generation
  - FPDF2-based PDF generation
  - Cover page with executive summary
  - Defect table with color coding
  - Compliance & risk sections
  - AMD technology appendix

**Input:** Pipeline results  
**Output:** Interactive dashboard + downloadable reports

---

## Data Flow

```
Video/Images
    ↓
[Layer 1] Frame Extraction + Quality Filter
    ↓
Quality Keyframes
    ↓
[Layer 2] Depth + Detection + Defects (AMD NPU)
    ↓
Per-frame Analysis
    ↓
[Layer 3] COLMAP SfM → 3D Point Cloud (AMD iGPU)
    ↓
Labeled 3D Model
    ↓
[Layer 4] Digital Twin Engine → Health Index
    ↓
Digital Twin JSON
    ↓
[Layer 5] Inspector → Compliance (RAG) → Safety → Report (AMD GAIA)
    ↓
Inspection Results
    ↓
[Layer 6] Streamlit Dashboard + PDF Report
```

---

*CityMind v1.0 — Powered by AMD Ryzen AI*
