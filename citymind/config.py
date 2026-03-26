"""
CityMind Configuration — Central config for all layers.
Loads from .env file and provides defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
BUILDING_CODES_DIR = DATA_DIR / "building_codes"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "frames").mkdir(exist_ok=True)
(OUTPUT_DIR / "depth_maps").mkdir(exist_ok=True)
(OUTPUT_DIR / "detections").mkdir(exist_ok=True)
(OUTPUT_DIR / "defects").mkdir(exist_ok=True)
(OUTPUT_DIR / "reconstruction").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)
(OUTPUT_DIR / "digital_twin").mkdir(exist_ok=True)

# ── Layer 1: Ingestion ────────────────────────────────────────
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "50"))
FRAME_EXTRACTION_FPS = int(os.getenv("FRAME_EXTRACTION_FPS", "3"))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "100.0"))
SSIM_THRESHOLD = 0.95  # frames more similar than this are duplicates

# ── Layer 2: Perception ───────────────────────────────────────
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
DETECTION_MODEL = os.getenv("DETECTION_MODEL", "yolov8m.pt")
DEFECT_MODEL = os.getenv("DEFECT_MODEL", "yolov8m.pt")
DETECTION_CONFIDENCE = 0.35
DEFECT_CONFIDENCE = 0.25

# AMD NPU Settings
USE_NPU = os.getenv("USE_NPU", "false").lower() == "true"
VITIS_AI_EP = os.getenv("VITIS_AI_EP", "false").lower() == "true"
VAIP_CONFIG_PATH = os.getenv("VAIP_CONFIG_PATH", str(MODELS_DIR / "vaip_config.json"))

# ── Layer 3: Reconstruction ───────────────────────────────────
COLMAP_PATH = os.getenv("COLMAP_PATH", "colmap")
USE_GPU_RECONSTRUCTION = os.getenv("USE_GPU_RECONSTRUCTION", "false").lower() == "true"

# ── Layer 5: Agents / LLM ────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Embedding model for RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Layer 6: Visualization ────────────────────────────────────
REPORT_FORMAT = os.getenv("REPORT_FORMAT", "pdf")
STREAMLIT_PORT = 8501

# ── Defect Categories ─────────────────────────────────────────
DEFECT_TYPES = {
    "crack": {"color": "#FF0000", "severity_base": 6},
    "spalling": {"color": "#FF6600", "severity_base": 7},
    "corrosion": {"color": "#FF9900", "severity_base": 5},
    "delamination": {"color": "#FFCC00", "severity_base": 4},
    "staining": {"color": "#FFFF00", "severity_base": 2},
    "displacement": {"color": "#FF3366", "severity_base": 8},
    "exposed_rebar": {"color": "#CC0000", "severity_base": 9},
    "water_damage": {"color": "#0066FF", "severity_base": 5},
}

STRUCTURAL_ELEMENTS = [
    "beam", "column", "wall", "slab", "foundation",
    "rebar", "scaffolding", "railing", "joint", "connection"
]

# ── AMD Technology Stack ──────────────────────────────────────
AMD_TECH_STACK = {
    "training": [
        "AMD Instinct MI300X",
        "ROCm 7.2",
        "PyTorch (ROCm) + TunableOp",
    ],
    "optimization": [
        "Vitis AI Quantizer (INT8/INT4)",
        "ONNX Runtime + Vitis AI EP",
    ],
    "edge_inference": [
        "Ryzen AI NPU (XDNA)",
        "AMD Ryzen AI CVML Library",
        "Ryzen AI Max+ iGPU",
    ],
    "intelligence": [
        "AMD GAIA Framework",
        "Lemonade (TurnkeyML)",
        "Ryzen AI Software v1.7",
    ],
    "simulation": [
        "Genesis Simulation Engine",
    ],
}
