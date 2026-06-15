# 🏗️ CityMind — AI-Powered Infrastructure Digital Twin

**Live demo:** https://citymind-amd.vercel.app — _Demo: dashboard served with precomputed inspection data &amp; defect overlays (no live CV/NPU pipeline)._

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Ryzen_AI_Powered-ED1C24?style=for-the-badge&logo=amd" alt="AMD Ryzen AI"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/AMD_Ryzen_AI-NPU_Accelerated-ED1C24?style=for-the-badge" alt="Ryzen AI"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

> **Multi-Agent AI Platform for Automated Infrastructure Inspection** — Turn drone/smartphone video into an intelligent 3D digital twin with AI-driven defect detection, building code compliance checking, and structural risk assessment. Powered by the AMD edge AI ecosystem.

---

## 🎯 Problem Statement

Infrastructure deterioration is a **$2.6 trillion global crisis** (ASCE 2025). Manual inspection is:
- ⏱️ **Slow** — weeks per structure
- 💰 **Expensive** — $3,000-$10,000 per bridge inspection
- 🔴 **Dangerous** — inspectors in confined spaces, at heights
- 📋 **Subjective** — 40% disagreement between inspectors on severity

**CityMind** automates this with a 6-layer AI pipeline that runs on AMD edge hardware.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CityMind Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📹 Layer 1: Video Ingestion                                       │
│  ├── FFmpeg/OpenCV keyframe extraction                              │
│  ├── Quality filtering (Laplacian blur, SSIM dedup)                │
│  └── GPS/metadata extraction                                       │
│                                                                     │
│  🔍 Layer 2: Edge Perception (AMD Ryzen AI NPU)                   │
│  ├── Depth Anything v2 (monocular depth estimation)                │
│  ├── YOLOv8/RT-DETR (structural element detection)                │
│  ├── Defect detection (crack/corrosion/spalling)                   │
│  └── ONNX Runtime + Vitis AI EP (INT8 NPU acceleration)           │
│                                                                     │
│  🗺️ Layer 3: 3D Reconstruction (AMD Ryzen AI Max+ iGPU)          │
│  ├── COLMAP Structure-from-Motion pipeline                         │
│  ├── Semantic 2D→3D label projection                               │
│  └── Colored/labeled point cloud generation                        │
│                                                                     │
│  🏢 Layer 4: Digital Twin Engine                                   │
│  ├── Structural Health Index (SHI: 0-100, A-F grade)               │
│  ├── Zone-based analysis (Foundation → Roof)                       │
│  └── Building code reference mapping                               │
│                                                                     │
│  🤖 Layer 5: Multi-Agent Intelligence (AMD GAIA Framework)        │
│  ├── Inspector Agent — Defect validation & classification          │
│  ├── Compliance Agent — Building code RAG (ACI/ASCE/IBC)          │
│  ├── Safety Agent — Risk scoring (0-100)                           │
│  └── Report Agent — Professional inspection report                 │
│                                                                     │
│  📊 Layer 6: Visualization & Dashboard (Streamlit)                 │
│  ├── Interactive 3D point cloud viewer (Plotly)                    │
│  ├── Risk/defect analytics charts                                  │
│  └── PDF/Markdown report export                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔴 AMD Technology Integration (12 Technologies)

| # | AMD Technology | Layer | Role in CityMind |
|---|---------------|-------|-----------------|
| 1 | **AMD Instinct MI300X** | Training | Model training with 192GB HBM3 |
| 2 | **ROCm 7.2** | Training | PyTorch backend for training on AMD GPUs |
| 3 | **PyTorch (ROCm) + TunableOp** | Training | Kernel auto-tuning for AMD hardware |
| 4 | **Vitis AI Quantizer** | Optimization | FP32→INT8 model quantization for NPU |
| 5 | **ONNX Runtime + Vitis AI EP** | Deployment | NPU execution provider for ONNX models |
| 6 | **AMD Ryzen AI NPU (XDNA)** | Edge Inference | INT8 inference for detection models |
| 7 | **AMD Ryzen AI CVML Library** | Perception | Depth estimation optimization |
| 8 | **AMD Ryzen AI Max+ iGPU** | Reconstruction | GPU-accelerated 3D processing |
| 9 | **AMD GAIA Framework** | Intelligence | Multi-agent orchestration pattern |
| 10 | **Lemonade (TurnkeyML)** | Intelligence | LLM optimization for local inference |
| 11 | **Ryzen AI Software v1.7** | Runtime | Local AI runtime with 16K context |
| 12 | **Genesis Simulation Engine** | Simulation | Digital twin physics (stretch) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) AMD Ryzen AI hardware for NPU acceleration
- (Optional) OpenAI API key for LLM-powered agents

### Installation

```bash
# Clone the repository
git clone https://github.com/Dhanyatha0105/CityMind.git
cd CityMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env with your API keys
```

### Run the Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run citymind/visualization/app.py
```

### Run the CLI Pipeline

```bash
# Run with synthetic demo data
python -m citymind.pipeline --demo

# Run with video input
python -m citymind.pipeline --video path/to/inspection_video.mp4

# Run with images
python -m citymind.pipeline --images path/to/frames/

# Use OpenAI for agent intelligence
python -m citymind.pipeline --demo --llm openai --api-key sk-...
```

---

## 📁 Project Structure

```
CityMind/
├── citymind/
│   ├── __init__.py
│   ├── config.py                    # Central configuration
│   ├── pipeline.py                  # End-to-end pipeline orchestrator
│   ├── ingestion/                   # Layer 1: Video ingestion
│   │   ├── frame_extractor.py       # Keyframe extraction
│   │   ├── quality_filter.py        # Blur/SSIM quality filtering
│   │   └── metadata.py              # GPS/timestamp extraction
│   ├── perception/                  # Layer 2: Edge perception (AMD NPU)
│   │   ├── depth_estimation.py      # Depth Anything v2
│   │   ├── object_detection.py      # YOLOv8 structural detection
│   │   ├── defect_detection.py      # Crack/corrosion/spalling
│   │   └── npu_inference.py         # ONNX RT + Vitis AI EP wrapper
│   ├── reconstruction/              # Layer 3: 3D reconstruction
│   │   ├── colmap_pipeline.py       # COLMAP SfM pipeline
│   │   ├── semantic_projection.py   # 2D→3D label projection
│   │   └── point_cloud_utils.py     # PLY read/write, utilities
│   ├── digital_twin/                # Layer 4: Digital twin engine
│   │   ├── twin_engine.py           # Health index, zone analysis
│   │   └── deviation_analysis.py    # Cloud-to-cloud deviation & temporal tracking
│   ├── agents/                      # Layer 5: Multi-agent intelligence
│   │   ├── orchestrator.py          # 4-agent pipeline (AMD GAIA)
│   │   └── prompts/                 # Agent prompt templates
│   ├── rag/                         # RAG subsystem
│   │   ├── ingest.py                # PDF→chunks→FAISS
│   │   └── retriever.py             # Semantic code retrieval
│   └── visualization/               # Layer 6: Dashboard
│       ├── app.py                   # Streamlit main application
│       ├── viewer_3d.py             # 3D point cloud viewer
│       ├── charts.py                # Analytics charts
│       └── report_pdf.py            # PDF report generation
├── training/                        # Model training & export
│   ├── export_onnx.py              # PyTorch → ONNX export
│   └── quantize_vitis.py           # Vitis AI INT8 quantization
├── scripts/                         # Utility scripts
│   ├── benchmark.py                # Performance benchmarking (all layers)
│   ├── generate_demo_data.py       # Rich demo data generator
│   └── run_demo.py                 # Quick demo runner
├── models/                          # Model files & configs
│   └── vaip_config.json            # Vitis AI EP configuration
├── data/                            # Data directory
│   ├── sample_videos/              # Sample inspection videos
│   └── building_codes/             # Building code documents (RAG)
├── docs/                            # Documentation
│   ├── architecture.md             # Detailed architecture
│   ├── amd_tech_integration.md     # AMD technology mapping
│   └── setup_guide.md             # Detailed setup guide
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package configuration
├── .env.example                     # Environment template
└── README.md                        # This file
```

---

## 🧪 Testing & Benchmarks

```bash
# Run the full test suite (44 tests)
pytest tests/ -v

# Run performance benchmarks across all 6 layers + stretch goals
python scripts/benchmark.py --iterations 3

# Generate rich demo data for dashboard screenshots
python scripts/generate_demo_data.py
```

### Stretch Goals Implemented

| Feature | Module | Description |
|---------|--------|-------------|
| **Deviation Analysis** | `digital_twin/deviation_analysis.py` | Cloud-to-cloud and cloud-to-plane deviation, temporal tracking, heatmap visualization |
| **Performance Benchmarks** | `scripts/benchmark.py` | Per-layer latency/throughput profiling with AMD NPU speedup projections |
| **Rich Demo Data** | `scripts/generate_demo_data.py` | Synthetic defect overlays, point clouds, agent results for polished demos |

---

## 🎬 Demo

> Launch the interactive Streamlit dashboard to see CityMind in action:

```bash
streamlit run citymind/visualization/app.py
```

The dashboard provides:
- 📹 Video upload or demo mode
- 🗺️ Interactive 3D digital twin viewer
- ⚠️ AI-detected defect analysis with severity scoring
- 🤖 Multi-agent inspection intelligence (GAIA)
- 📋 Building code compliance checking (RAG)
- 🛡️ Structural risk assessment
- 📄 Professional inspection report generation
- 📊 Analytics with AMD performance benchmarks

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👨‍💻 Author

**Dhanyatha**  
[GitHub](https://github.com/Dhanyatha0105)  

---

<p align="center">
  <b>CityMind v1.0</b> — Turning infrastructure inspection from manual to intelligent.<br/>
  Powered by <span style="color: #ED1C24;">AMD</span> Edge AI
</p>
