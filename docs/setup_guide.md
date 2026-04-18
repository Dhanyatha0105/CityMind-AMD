# 🛠️ CityMind — Setup Guide

## Prerequisites

- **Python 3.10+** (tested with 3.10, 3.11)
- **pip** package manager
- **Git** for version control
- (Optional) **COLMAP** for real 3D reconstruction
- (Optional) **AMD Ryzen AI hardware** for NPU acceleration
- (Optional) **OpenAI API key** for LLM-powered agents

---

## Quick Setup

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Dhanyatha0105/CityMind.git
cd CityMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# LLM Configuration (optional)
LLM_PROVIDER=deterministic          # or "openai" or "ollama"
OPENAI_API_KEY=sk-your-key-here     # if using OpenAI
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# AMD NPU (only on Ryzen AI hardware)
USE_NPU=false
VITIS_AI_EP=false

# Processing
MAX_FRAMES=30
OUTPUT_DIR=output
```

### 3. Launch Dashboard

```bash
streamlit run citymind/visualization/app.py
```

Open http://localhost:8501 in your browser.

### 4. Run CLI Pipeline

```bash
# Synthetic demo (no dependencies needed)
python -m citymind.pipeline --demo

# With video input
python -m citymind.pipeline --video path/to/video.mp4

# With LLM agents
python -m citymind.pipeline --demo --llm openai --api-key sk-...
```

---

## Dependency Installation

### Core (Required)
```bash
pip install numpy opencv-python Pillow ffmpeg-python python-dotenv pyyaml
```

### Deep Learning (Layer 2)
```bash
pip install torch torchvision
pip install transformers        # Depth Anything v2
pip install ultralytics         # YOLOv8
pip install onnxruntime         # ONNX inference
```

### 3D Reconstruction (Layer 3)
```bash
pip install open3d plyfile
# Optional: Install COLMAP from https://colmap.github.io/install.html
```

### RAG & Agents (Layer 5)
```bash
pip install openai              # OpenAI LLM backend
pip install sentence-transformers  # Embeddings
pip install faiss-cpu           # Vector search
pip install pypdf               # PDF parsing
```

### Visualization (Layer 6)
```bash
pip install streamlit plotly altair fpdf2 jinja2
```

---

## AMD NPU Setup (Ryzen AI Hardware)

### 1. Install Ryzen AI Software

Download from: https://www.amd.com/en/products/software/ryzen-ai-software.html

### 2. Install ONNX Runtime with Vitis AI EP

```bash
pip install onnxruntime-vitisai
```

### 3. Configure NPU

```bash
# Enable NPU in .env
USE_NPU=true
VITIS_AI_EP=true
VAIP_CONFIG_PATH=models/vaip_config.json
```

### 4. Export and Quantize Models

```bash
# Export to ONNX
python training/export_onnx.py --model all

# Quantize to INT8
python training/quantize_vitis.py --input models/yolov8m_structural.onnx
python training/quantize_vitis.py --create-config
```

---

## Adding Building Codes

Place PDF or TXT building code documents in `data/building_codes/`:

```
data/building_codes/
├── ACI_318-19.pdf
├── ACI_562-19.pdf
├── ASCE_7-22.pdf
└── local_building_code.txt
```

The RAG system automatically ingests these on pipeline startup.

---

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### COLMAP not found
- The system falls back to synthetic point cloud generation
- Install COLMAP: `brew install colmap` (macOS) or build from source

### Dashboard won't start
```bash
pip install streamlit --upgrade
streamlit run citymind/visualization/app.py --server.port 8502
```

### NPU not detected
- Ensure AMD Ryzen AI Software is installed
- Check: `python -c "import onnxruntime; print(ort.get_available_providers())"`
- Should include `VitisAIExecutionProvider`

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific layer test
python -m pytest tests/test_layer1_ingestion.py -v
```

---

*CityMind v1.0 — Powered by AMD Ryzen AI*
