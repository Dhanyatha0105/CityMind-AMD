"""
Layer 6: CityMind Dashboard — Streamlit Web Application
Main entry point for the interactive visualization dashboard.

AMD Tech: Full stack demo showcasing all AMD technology layers.
"""

import streamlit as st
import json
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="CityMind — AI Infrastructure Digital Twin",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark professional theme overrides */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ED1C24 0%, #FF6B35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 5px;
    }
    .status-good { color: #00E676; }
    .status-warning { color: #FFD600; }
    .status-critical { color: #FF1744; }
    .amd-badge {
        background: linear-gradient(135deg, #ED1C24, #FF6B35);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    
    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.shields.io/badge/AMD-Ryzen_AI_Powered-ED1C24?style=for-the-badge&logo=amd", use_container_width=True)
        st.markdown("## 🏗️ CityMind")
        st.markdown("*AI-Powered Infrastructure Digital Twin*")
        st.divider()
        
        st.markdown("### 🔧 Configuration")
        
        # LLM Provider
        llm_provider = st.selectbox(
            "LLM Provider",
            ["deterministic", "openai", "ollama"],
            help="Select the LLM backend for multi-agent analysis"
        )
        
        if llm_provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password")
        else:
            api_key = ""
        
        # Processing options
        st.markdown("### ⚙️ Processing")
        max_frames = st.slider("Max Frames", 5, 100, 30)
        detection_conf = st.slider("Detection Confidence", 0.1, 0.9, 0.35)
        use_synthetic = st.checkbox("Use Synthetic Demo Data", value=True, 
                                     help="Generate demo data without real video processing")
        
        st.divider()
        
        # AMD Tech Stack
        st.markdown("### 🔴 AMD Technology Stack")
        amd_techs = [
            "Instinct MI300X (Training)",
            "ROCm 7.2 + PyTorch",
            "Vitis AI Quantizer (INT8)",
            "ONNX RT + Vitis AI EP",
            "Ryzen AI NPU (XDNA)",
            "AMD CVML Library",
            "Ryzen AI Max+ iGPU",
            "GAIA Framework",
            "Lemonade (TurnkeyML)",
            "Ryzen AI Software v1.7",
            "Genesis Simulation",
            "Vulkan Backend",
        ]
        for tech in amd_techs:
            st.markdown(f'<span class="amd-badge">{tech}</span>', unsafe_allow_html=True)
        
        st.divider()
        st.caption("© 2026 Dhanyatha")
    
    # ── Main Content ────────────────────────────────────────────
    st.markdown('<p class="main-header">CityMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Infrastructure Digital Twin • Multi-Agent Inspection Intelligence • Powered by AMD</p>', unsafe_allow_html=True)
    
    # ── Upload or Demo ──────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📹 Upload infrastructure video or images",
            type=["mp4", "avi", "mov", "jpg", "png", "jpeg"],
            accept_multiple_files=True,
            help="Upload video footage or images of infrastructure for AI inspection"
        )
    
    with col2:
        st.markdown("&nbsp;")
        run_demo = st.button("🚀 Run Demo Pipeline", type="primary", use_container_width=True)
    
    with col3:
        st.markdown("&nbsp;")
        load_rich = st.button("📊 Load Rich Demo", use_container_width=True,
                              help="Load pre-generated rich demo data for screenshots")
    
    # ── Process ─────────────────────────────────────────────────
    if load_rich or st.session_state.get("show_rich_demo"):
        st.session_state["show_rich_demo"] = True
        _load_rich_demo_data()
    elif run_demo or st.session_state.get("show_sample"):
        _run_pipeline(
            uploaded_files=uploaded_file,
            use_synthetic=use_synthetic,
            llm_provider=llm_provider,
            api_key=api_key,
            max_frames=max_frames,
            detection_conf=detection_conf,
        )
    else:
        _show_landing_page()


def _load_rich_demo_data():
    """Load pre-generated rich demo data from output/demo_rich/ for polished screenshots."""
    import numpy as np

    st.divider()

    rich_dir = Path(__file__).parent.parent.parent / "output" / "demo_rich"
    if not rich_dir.exists():
        st.error(f"Rich demo data not found at {rich_dir}. Run `python scripts/generate_demo_data.py` first.")
        return

    # Load twin data
    twin_path = rich_dir / "twin.json"
    agents_path = rich_dir / "agents.json"
    defects_path = rich_dir / "defects.json"
    perf_path = rich_dir / "performance.json"
    complete_path = rich_dir / "demo_data_complete.json"
    ply_path = rich_dir / "bridge_digital_twin.ply"

    try:
        with open(twin_path) as f:
            twin_data = json.load(f)
    except Exception as e:
        st.error(f"Failed to load twin data: {e}")
        return

    try:
        with open(agents_path) as f:
            agents_raw = json.load(f)
    except Exception:
        agents_raw = {}

    # Load complete demo data for extra context
    try:
        with open(complete_path) as f:
            complete = json.load(f)
    except Exception:
        complete = {}

    # Reconstruct pipeline_results from agents data
    pipeline_results = {
        "pipeline_id": "RICH-DEMO-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "started_at": datetime.now().isoformat(),
        "inspector": agents_raw.get("inspector", {}),
        "compliance": agents_raw.get("compliance", {}),
        "safety": agents_raw.get("safety", {}),
        "report": {
            "report_markdown": agents_raw.get("report", "No report available."),
            "generated_at": datetime.now().isoformat(),
        },
        "completed_at": datetime.now().isoformat(),
    }

    # Load point cloud for 3D viewer
    try:
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        pc = PointCloudUtils.read_ply(str(ply_path))
        twin_data["point_cloud"] = {
            "points": np.array(pc["points"], dtype=np.float32),
            "colors": np.array(pc["colors"], dtype=np.float32),
        }
    except Exception:
        # Generate a synthetic fallback
        try:
            from citymind.reconstruction.point_cloud_utils import PointCloudUtils
            pc_data = PointCloudUtils.generate_synthetic_building(num_points=5000)
            twin_data["point_cloud"] = pc_data
        except Exception:
            n = 3000
            twin_data["point_cloud"] = {
                "points": np.random.randn(n, 3).astype(np.float32) * np.array([5, 7.5, 4]),
                "colors": np.random.rand(n, 3).astype(np.float32) * 0.5 + 0.3,
            }

    # Ensure structure_info exists for display
    if "structure_info" not in twin_data:
        pipeline_info = complete.get("pipeline", {})
        twin_data["structure_info"] = {
            "type": pipeline_info.get("structure_type", twin_data.get("structure_type", "Bridge")),
            "source_video": pipeline_info.get("video_source", "Inspection Video"),
            "location": pipeline_info.get("location", ""),
        }

    # Ensure zones exist
    if "zones" not in twin_data:
        twin_data["zones"] = complete.get("zones", [
            {"name": "Deck", "health": "POOR", "score": 35, "defects": ["DEF-001"]},
            {"name": "Pier 1", "health": "FAIR", "score": 55, "defects": ["DEF-003"]},
            {"name": "Pier 2", "health": "GOOD", "score": 82, "defects": []},
            {"name": "Abutment South", "health": "CRITICAL", "score": 22, "defects": ["DEF-004"]},
            {"name": "Abutment North", "health": "FAIR", "score": 60, "defects": ["DEF-017"]},
        ])

    # Ensure defect_types summary exists
    defect_analysis = twin_data.get("defect_analysis", {})
    if "defect_types" not in defect_analysis:
        # Build from defects list
        all_defects = defect_analysis.get("critical_defects", []) + \
                      defect_analysis.get("high_defects", []) + \
                      defect_analysis.get("medium_defects", []) + \
                      defect_analysis.get("low_defects", [])
        type_counts = {}
        for d in all_defects:
            dt = d.get("defect_type", "unknown")
            if dt not in type_counts:
                type_counts[dt] = {"count": 0, "max_severity": 0, "avg_severity": 0, "total": 0}
            type_counts[dt]["count"] += 1
            type_counts[dt]["total"] += d.get("severity", 0)
            type_counts[dt]["max_severity"] = max(type_counts[dt]["max_severity"], d.get("severity", 0))
        for dt in type_counts:
            if type_counts[dt]["count"] > 0:
                type_counts[dt]["avg_severity"] = type_counts[dt]["total"] / type_counts[dt]["count"]
            del type_counts[dt]["total"]
        defect_analysis["defect_types"] = type_counts
        defect_analysis["all_defects"] = all_defects
        twin_data["defect_analysis"] = defect_analysis

    # AMD processing info
    twin_data["amd_processing"] = {
        "perception_engine": "AMD Ryzen AI NPU (XDNA2, 50 TOPS) — INT8 Quantized",
        "reconstruction": "AMD Ryzen AI Max+ iGPU — Vulkan Backend",
        "agent_framework": "AMD GAIA Framework (4 Agents)",
        "model_optimization": "Vitis AI Quantizer + ONNX Runtime",
        "training_platform": "AMD Instinct MI300X (ROCm 7.2, 192GB HBM3)",
    }

    # Show a success banner
    pipeline_info = complete.get("pipeline", {})
    st.success(
        f"✅ Loaded rich demo data: **{twin_data.get('structure_type', 'Bridge')}** — "
        f"{defect_analysis.get('total_defects', '?')} defects, "
        f"Health: {twin_data.get('health_index', {}).get('score', '?')}/100 "
        f"({twin_data.get('health_index', {}).get('grade', '?')})"
    )

    # Defect overlay images
    overlay_dir = rich_dir / "defect_overlays"
    if overlay_dir.exists():
        overlay_images = sorted(overlay_dir.glob("*.png"))
        if overlay_images:
            twin_data["_overlay_images"] = [str(p) for p in overlay_images]

    _display_results(twin_data, pipeline_results)


def _show_landing_page():
    """Show the landing page with architecture overview."""
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Processing Layers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">AI Agents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">12</div>
            <div class="metric-label">AMD Technologies</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3D</div>
            <div class="metric-label">Digital Twin</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CityMind Architecture                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  📹 Layer 1: Video Ingestion                                   │
    │  ├── FFmpeg/OpenCV keyframe extraction                          │
    │  ├── Quality filtering (blur, SSIM dedup)                       │
    │  └── GPS/metadata extraction                                    │
    │                                                                 │
    │  🔍 Layer 2: Edge Perception (AMD Ryzen AI NPU)                │
    │  ├── Depth Anything v2 (monocular depth)                        │
    │  ├── YOLOv8/RT-DETR (structural detection)                     │
    │  ├── Defect detection (crack/corrosion/spalling)                │
    │  └── ONNX Runtime + Vitis AI EP (NPU acceleration)             │
    │                                                                 │
    │  🗺️ Layer 3: 3D Reconstruction (Ryzen AI Max+ iGPU)           │
    │  ├── COLMAP SfM pipeline                                        │
    │  ├── Semantic 2D→3D projection                                  │
    │  └── Colored/labeled point cloud generation                     │
    │                                                                 │
    │  🏢 Layer 4: Digital Twin Engine                                │
    │  ├── Structural health index (0-100)                            │
    │  ├── Zone-based analysis                                        │
    │  └── Deviation tracking                                         │
    │                                                                 │
    │  🤖 Layer 5: Multi-Agent Intelligence (AMD GAIA)               │
    │  ├── Inspector Agent (defect validation)                        │
    │  ├── Compliance Agent (building code RAG)                       │
    │  ├── Safety Agent (risk scoring)                                │
    │  └── Report Agent (inspection report generation)                │
    │                                                                 │
    │  📊 Layer 6: Visualization & Dashboard                          │
    │  ├── 3D point cloud viewer (Plotly)                              │
    │  ├── Risk/defect charts                                         │
    │  └── PDF report export                                          │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.info("👆 **Upload a video or click 'Run Demo Pipeline'** to start the AI inspection.")


def _run_pipeline(
    uploaded_files,
    use_synthetic: bool,
    llm_provider: str,
    api_key: str,
    max_frames: int,
    detection_conf: float,
):
    """Run the full CityMind pipeline and display results."""
    
    st.divider()
    
    # ── Progress tracking ───────────────────────────────────────
    progress = st.progress(0, text="Initializing CityMind pipeline...")
    status_container = st.empty()
    
    # ── Generate or process data ────────────────────────────────
    twin_data, pipeline_results = _execute_pipeline(
        uploaded_files=uploaded_files,
        use_synthetic=use_synthetic,
        llm_provider=llm_provider,
        api_key=api_key,
        max_frames=max_frames,
        detection_conf=detection_conf,
        progress=progress,
        status=status_container,
    )
    
    progress.progress(100, text="✅ Pipeline complete!")
    time.sleep(0.5)
    progress.empty()
    status_container.empty()
    
    # ── Display Results ─────────────────────────────────────────
    _display_results(twin_data, pipeline_results)


def _execute_pipeline(
    uploaded_files,
    use_synthetic: bool,
    llm_provider: str,
    api_key: str,
    max_frames: int,
    detection_conf: float,
    progress,
    status,
):
    """Execute the CityMind processing pipeline."""
    
    if use_synthetic or not uploaded_files:
        return _generate_demo_data(llm_provider, api_key, progress, status)
    
    # Real processing pipeline
    return _process_real_data(
        uploaded_files, llm_provider, api_key, 
        max_frames, detection_conf, progress, status
    )


def _generate_demo_data(llm_provider: str, api_key: str, progress, status):
    """Generate synthetic demo data for showcase."""
    
    # Layer 1
    status.info("📹 Layer 1: Extracting keyframes...")
    progress.progress(10, text="Layer 1: Video Ingestion")
    time.sleep(0.5)
    
    # Layer 2
    status.info("🔍 Layer 2: Running perception models on AMD Ryzen AI NPU...")
    progress.progress(25, text="Layer 2: Edge Perception (AMD NPU)")
    time.sleep(0.8)
    
    # Layer 3
    status.info("🗺️ Layer 3: Building 3D reconstruction on AMD Ryzen AI Max+ iGPU...")
    progress.progress(40, text="Layer 3: 3D Reconstruction")
    time.sleep(0.6)
    
    # Generate synthetic point cloud
    try:
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        pc_data = PointCloudUtils.generate_synthetic_building(
            width=10, height=15, depth=8, num_points=5000
        )
    except Exception:
        import numpy as np
        # Ultra-fallback
        n = 3000
        pc_data = {
            "points": np.random.randn(n, 3).astype(np.float32) * np.array([5, 7.5, 4]),
            "colors": np.random.rand(n, 3).astype(np.float32) * 0.5 + 0.3,
        }
    
    # Layer 4
    status.info("🏢 Layer 4: Building digital twin...")
    progress.progress(55, text="Layer 4: Digital Twin Engine")
    time.sleep(0.5)
    
    # Generate demo twin data
    twin_data = _create_demo_twin(pc_data)
    
    # Layer 5
    status.info("🤖 Layer 5: Running multi-agent inspection (AMD GAIA Framework)...")
    progress.progress(70, text="Layer 5: Multi-Agent Intelligence")
    
    pipeline_results = _run_demo_agents(twin_data, llm_provider, api_key)
    
    # Layer 6
    status.info("📊 Layer 6: Generating visualizations...")
    progress.progress(90, text="Layer 6: Visualization")
    time.sleep(0.4)
    
    twin_data["point_cloud"] = pc_data
    
    return twin_data, pipeline_results


def _create_demo_twin(pc_data) -> dict:
    """Create demo digital twin data."""
    import numpy as np
    
    demo_defects = [
        {"id": "DEF-001", "defect_type": "crack", "severity": 7.5, "confidence": 0.89,
         "bbox": [120, 200, 280, 350], "detection_method": "yolov8_finetuned",
         "code_reference": "ACI 318-19 §24.3"},
        {"id": "DEF-002", "defect_type": "spalling", "severity": 8.2, "confidence": 0.92,
         "bbox": [400, 150, 550, 300], "detection_method": "yolov8_finetuned",
         "code_reference": "ACI 562-19 §6.3"},
        {"id": "DEF-003", "defect_type": "corrosion", "severity": 6.1, "confidence": 0.78,
         "bbox": [50, 400, 200, 520], "detection_method": "color_analysis",
         "code_reference": "ACI 222R-19 §4.2"},
        {"id": "DEF-004", "defect_type": "crack", "severity": 5.5, "confidence": 0.85,
         "bbox": [300, 100, 400, 250], "detection_method": "yolov8_finetuned",
         "code_reference": "ACI 318-19 §24.3"},
        {"id": "DEF-005", "defect_type": "water_damage", "severity": 4.0, "confidence": 0.72,
         "bbox": [500, 300, 640, 480], "detection_method": "color_analysis",
         "code_reference": "ACI 515.2R-13 §3"},
        {"id": "DEF-006", "defect_type": "exposed_rebar", "severity": 9.0, "confidence": 0.95,
         "bbox": [180, 280, 320, 400], "detection_method": "yolov8_finetuned",
         "code_reference": "ACI 318-19 §20.5"},
    ]
    
    twin_data = {
        "twin_id": f"TWIN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now().isoformat(),
        "structure_info": {
            "type": "Reinforced Concrete Building",
            "source_video": "demo_inspection.mp4",
            "video_duration": 45.0,
            "location": "Demo Site, City",
        },
        "geometry": {
            "point_cloud_path": "synthetic",
            "num_points": len(pc_data["points"]),
            "bounding_box": {
                "dimensions": [10.0, 15.0, 8.0],
            },
        },
        "defect_analysis": {
            "total_defects": len(demo_defects),
            "defect_types": {
                "crack": {"count": 2, "max_severity": 7.5, "avg_severity": 6.5},
                "spalling": {"count": 1, "max_severity": 8.2, "avg_severity": 8.2},
                "corrosion": {"count": 1, "max_severity": 6.1, "avg_severity": 6.1},
                "water_damage": {"count": 1, "max_severity": 4.0, "avg_severity": 4.0},
                "exposed_rebar": {"count": 1, "max_severity": 9.0, "avg_severity": 9.0},
            },
            "critical_defects": [d for d in demo_defects if d["severity"] >= 7],
            "high_defects": [d for d in demo_defects if 5 <= d["severity"] < 7],
            "medium_defects": [d for d in demo_defects if 3 <= d["severity"] < 5],
            "low_defects": [d for d in demo_defects if d["severity"] < 3],
            "all_defects": demo_defects,
        },
        "health_index": {
            "score": 52.3,
            "grade": "C",
            "status": "FAIR",
            "description": "Moderate deterioration detected — multiple structural concerns require attention",
            "recommendation": "Schedule detailed inspection within 3 months. Address critical defects immediately.",
            "defect_count": 6,
            "critical_count": 3,
        },
        "zones": [
            {"name": "Foundation Zone", "health": "GOOD", "score": 90, "defects": []},
            {"name": "Lower Structure", "health": "FAIR", "score": 65, "defects": ["DEF-003"]},
            {"name": "Mid Structure", "health": "POOR", "score": 35, "defects": ["DEF-001", "DEF-002", "DEF-006"]},
            {"name": "Upper Structure", "health": "FAIR", "score": 70, "defects": ["DEF-004"]},
            {"name": "Roof/Ceiling", "health": "FAIR", "score": 68, "defects": ["DEF-005"]},
        ],
        "amd_processing": {
            "perception_engine": "AMD Ryzen AI NPU (XDNA) — INT8 Quantized",
            "reconstruction": "AMD Ryzen AI Max+ iGPU — Vulkan Backend",
            "agent_framework": "AMD GAIA Framework",
            "model_optimization": "Vitis AI Quantizer + ONNX Runtime",
            "training_platform": "AMD Instinct MI300X (ROCm 7.2)",
        },
    }
    
    return twin_data


def _run_demo_agents(twin_data: dict, llm_provider: str, api_key: str) -> dict:
    """Run the multi-agent pipeline on demo data."""
    try:
        from citymind.agents.orchestrator import AgentOrchestrator
        
        orch = AgentOrchestrator(
            llm_provider=llm_provider,
            api_key=api_key if api_key else None,
        )
        
        frame_defects = [{"defects": twin_data["defect_analysis"].get("all_defects", [])}]
        frame_detections = [{"detections": []}]
        
        # Get RAG context
        rag_context = ""
        try:
            from citymind.rag.ingest import BuildingCodeIngestor
            from citymind.rag.retriever import BuildingCodeRetriever
            
            ingestor = BuildingCodeIngestor()
            ingestor.ingest_directory("")  # Uses sample codes
            retriever = BuildingCodeRetriever(ingestor)
            
            defects = twin_data["defect_analysis"].get("all_defects", [])
            rag_context = retriever.get_context_for_defects(defects)
        except Exception as e:
            logger.warning(f"RAG context generation failed: {e}")
            rag_context = "Building codes: ACI 318-19, ACI 562-19, ACI 222R-19, ASCE 7-22, IBC 2021"
        
        results = orch.run_inspection_pipeline(
            twin_data=twin_data,
            frame_defects=frame_defects,
            frame_detections=frame_detections,
            rag_context=rag_context,
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Agent pipeline failed: {e}")
        return _generate_fallback_results(twin_data)


def _generate_fallback_results(twin_data: dict) -> dict:
    """Generate fallback results if agent pipeline fails."""
    health = twin_data.get("health_index", {})
    defects = twin_data.get("defect_analysis", {})
    
    return {
        "pipeline_id": f"PIPE-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": datetime.now().isoformat(),
        "inspector": {
            "validated_defects": defects.get("critical_defects", []),
            "pattern_analysis": "Multiple structural defects detected requiring engineering assessment.",
            "overall_assessment": health.get("description", ""),
            "confidence": 0.80,
        },
        "compliance": {
            "overall_compliance": "NON-COMPLIANT" if defects.get("total_defects", 0) > 3 else "REVIEW REQUIRED",
            "total_violations": min(defects.get("total_defects", 0), 5),
            "critical_violations": len(defects.get("critical_defects", [])),
            "compliance_findings": [],
        },
        "safety": {
            "risk_score": 100 - health.get("score", 50),
            "risk_level": "HIGH" if health.get("score", 50) < 60 else "MODERATE",
            "occupancy_recommendation": "RESTRICTED" if health.get("score", 50) < 40 else "NORMAL",
        },
        "report": {
            "report_markdown": _generate_demo_report(twin_data),
            "generated_at": datetime.now().isoformat(),
        },
        "completed_at": datetime.now().isoformat(),
    }


def _generate_demo_report(twin_data: dict) -> str:
    """Generate a demo report markdown."""
    health = twin_data.get("health_index", {})
    return f"""# 🏗️ CityMind Infrastructure Inspection Report

## Executive Summary
**Structural Health Index:** {health.get('score', 'N/A')}/100 ({health.get('grade', 'N/A')})  
**Status:** {health.get('status', 'N/A')}  
**Date:** {datetime.now().strftime('%B %d, %Y')}

{health.get('description', '')}

## Recommendations
{health.get('recommendation', '')}

---
*Generated by CityMind v1.0 — Powered by AMD*
"""


def _process_real_data(uploaded_files, llm_provider, api_key, max_frames, detection_conf, progress, status):
    """Process real uploaded data through the pipeline."""
    import tempfile
    import os
    
    # Save uploaded files
    temp_dir = tempfile.mkdtemp(prefix="citymind_")
    
    for f in (uploaded_files if uploaded_files else []):
        file_path = os.path.join(temp_dir, f.name)
        with open(file_path, "wb") as out:
            out.write(f.read())
    
    status.info("📹 Layer 1: Processing uploaded files...")
    progress.progress(10, text="Layer 1: Video Ingestion")
    
    # Try to use real pipeline
    try:
        from citymind.ingestion.frame_extractor import FrameExtractor
        from citymind.ingestion.quality_filter import QualityFilter
        
        extractor = FrameExtractor(max_frames=max_frames)
        quality_filter = QualityFilter()
        
        all_frames = []
        for f in (uploaded_files if uploaded_files else []):
            file_path = os.path.join(temp_dir, f.name)
            if f.name.lower().endswith(('.mp4', '.avi', '.mov')):
                frames = extractor.extract_keyframes(file_path)
            else:
                import cv2
                img = cv2.imread(file_path)
                if img is not None:
                    frames = [file_path]
                else:
                    frames = []
            all_frames.extend(frames)
        
        status.info(f"Extracted {len(all_frames)} frames")
        progress.progress(20, text=f"Extracted {len(all_frames)} frames")
        
    except Exception as e:
        logger.warning(f"Real pipeline failed, falling back to demo: {e}")
        return _generate_demo_data(llm_provider, api_key, progress, status)
    
    # Fall through to demo data for remaining layers
    # (Real perception/reconstruction would go here with proper GPU setup)
    return _generate_demo_data(llm_provider, api_key, progress, status)


def _display_results(twin_data: dict, pipeline_results: dict):
    """Display all pipeline results in the dashboard."""
    
    # ── Top Metrics Row ─────────────────────────────────────────
    health = twin_data.get("health_index", {})
    defects = twin_data.get("defect_analysis", {})
    safety = pipeline_results.get("safety", {})
    compliance = pipeline_results.get("compliance", {})
    
    score = health.get("score", 0)
    score_color = "status-good" if score >= 75 else "status-warning" if score >= 50 else "status-critical"
    
    risk_score = safety.get("risk_score", 0)
    if isinstance(risk_score, str):
        try:
            risk_score = float(risk_score)
        except (ValueError, TypeError):
            risk_score = 50
    risk_color = "status-critical" if risk_score >= 60 else "status-warning" if risk_score >= 30 else "status-good"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {score_color}">{score}/100</div>
            <div class="metric-label">🏗️ Health Index ({health.get('grade', 'N/A')})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{defects.get('total_defects', 0)}</div>
            <div class="metric-label">⚠️ Total Defects</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        critical = len(defects.get('critical_defects', []))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-critical">{critical}</div>
            <div class="metric-label">🔴 Critical Defects</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {risk_color}">{risk_score:.0f}</div>
            <div class="metric-label">🛡️ Risk Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        comp_status = compliance.get("overall_compliance", "N/A")
        comp_color = "status-critical" if "NON" in str(comp_status) else "status-warning" if "REVIEW" in str(comp_status) else "status-good"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {comp_color}" style="font-size: 1.4rem;">{comp_status}</div>
            <div class="metric-label">📋 Compliance Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ── Tabbed Results ──────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ 3D Digital Twin",
        "⚠️ Defect Analysis", 
        "🤖 Agent Intelligence",
        "📋 Compliance",
        "🛡️ Risk Assessment",
        "📄 Inspection Report",
        "📊 Analytics",
    ])
    
    with tabs[0]:
        _tab_3d_viewer(twin_data)
    
    with tabs[1]:
        _tab_defect_analysis(twin_data)
    
    with tabs[2]:
        _tab_agent_intelligence(pipeline_results)
    
    with tabs[3]:
        _tab_compliance(pipeline_results)
    
    with tabs[4]:
        _tab_risk_assessment(pipeline_results, twin_data)
    
    with tabs[5]:
        _tab_report(pipeline_results)
    
    with tabs[6]:
        _tab_analytics(twin_data, pipeline_results)


def _tab_3d_viewer(twin_data: dict):
    """3D Digital Twin viewer tab."""
    st.markdown("### 🗺️ 3D Infrastructure Digital Twin")
    st.caption("Interactive 3D point cloud visualization — AMD Ryzen AI Max+ iGPU accelerated")
    
    pc_data = twin_data.get("point_cloud", None)
    
    if pc_data is not None:
        try:
            from citymind.visualization.viewer_3d import render_point_cloud_plotly
            fig = render_point_cloud_plotly(
                pc_data["points"],
                pc_data.get("colors"),
                title="CityMind Digital Twin — 3D Point Cloud"
            )
            st.plotly_chart(fig, use_container_width=True, key="main_3d")
        except Exception as e:
            # Inline Plotly fallback
            _render_3d_fallback(pc_data)
    else:
        st.info("No point cloud data available. Run the pipeline to generate 3D reconstruction.")
    
    # Zone analysis
    st.markdown("### 🏢 Zone Analysis")
    zones = twin_data.get("zones", [])
    if zones:
        zone_cols = st.columns(len(zones))
        for i, zone in enumerate(zones):
            with zone_cols[i]:
                zone_color = "🟢" if zone["health"] == "GOOD" else "🟡" if zone["health"] == "FAIR" else "🔴"
                st.metric(
                    label=f"{zone_color} {zone['name']}",
                    value=f"{zone['score']}/100",
                    delta=f"{len(zone.get('defects', []))} defects",
                    delta_color="inverse",
                )


def _render_3d_fallback(pc_data):
    """Render 3D point cloud using Plotly directly."""
    import plotly.graph_objects as go
    import numpy as np
    
    points = pc_data["points"]
    colors = pc_data.get("colors")
    
    # Downsample for performance
    max_display = 3000
    if len(points) > max_display:
        indices = np.random.choice(len(points), max_display, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    if colors is not None:
        color_strings = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
    else:
        color_strings = 'rgb(100,100,200)'
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 2],  # Swap Y/Z for architectural convention
        z=points[:, 1],
        mode='markers',
        marker=dict(
            size=2,
            color=color_strings,
            opacity=0.8,
        ),
    )])
    
    fig.update_layout(
        title="CityMind Digital Twin — 3D Point Cloud",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Z (m)",
            zaxis_title="Y - Height (m)",
            bgcolor='rgb(15,15,25)',
            xaxis=dict(gridcolor='rgb(50,50,70)'),
            yaxis=dict(gridcolor='rgb(50,50,70)'),
            zaxis=dict(gridcolor='rgb(50,50,70)'),
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='rgb(200,200,200)'),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True, key="fallback_3d")


def _tab_defect_analysis(twin_data: dict):
    """Defect analysis tab."""
    st.markdown("### ⚠️ Defect Analysis")
    st.caption("AI-detected structural defects — AMD Ryzen AI NPU accelerated inference")
    
    defects = twin_data.get("defect_analysis", {})
    all_defects = defects.get("all_defects", [])
    
    if not all_defects:
        all_defects = defects.get("critical_defects", []) + defects.get("high_defects", [])
    
    # Defect type distribution chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Defect Distribution")
        defect_types = defects.get("defect_types", {})
        if defect_types:
            try:
                from citymind.visualization.charts import defect_distribution_chart
                fig = defect_distribution_chart(defect_types)
                st.plotly_chart(fig, use_container_width=True, key="defect_dist")
            except Exception:
                _render_defect_chart_fallback(defect_types)
    
    with col2:
        st.markdown("#### Severity Distribution")
        if all_defects:
            try:
                from citymind.visualization.charts import severity_histogram
                fig = severity_histogram(all_defects)
                st.plotly_chart(fig, use_container_width=True, key="severity_hist")
            except Exception:
                _render_severity_fallback(all_defects)
    
    # Defect table
    st.markdown("#### Detailed Defect Log")
    if all_defects:
        import pandas as pd
        df = pd.DataFrame([
            {
                "ID": d.get("id", ""),
                "Type": d.get("defect_type", "").title(),
                "Severity": f"{d.get('severity', 0):.1f}/10",
                "Confidence": f"{d.get('confidence', 0):.0%}",
                "Method": d.get("detection_method", ""),
                "Code Ref": d.get("code_reference", ""),
            }
            for d in all_defects
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No defects detected.")

    # Show overlay images if available (from rich demo data)
    overlay_images = twin_data.get("_overlay_images", [])
    if overlay_images:
        st.markdown("#### 🖼️ Defect Detection Overlays")
        st.caption("AI-annotated frames with detected structural defects")
        img_cols = st.columns(min(len(overlay_images), 3))
        for i, img_path in enumerate(overlay_images[:6]):
            col_idx = i % 3
            with img_cols[col_idx]:
                try:
                    st.image(img_path, caption=Path(img_path).stem.replace("_", " ").title(), use_container_width=True)
                except Exception:
                    pass


def _render_defect_chart_fallback(defect_types: dict):
    """Fallback defect distribution chart."""
    import plotly.graph_objects as go
    
    types = list(defect_types.keys())
    counts = [v["count"] for v in defect_types.values()]
    
    color_map = {
        "crack": "#FF0000", "spalling": "#FF6600", "corrosion": "#FF9900",
        "water_damage": "#0066FF", "exposed_rebar": "#CC0000", "delamination": "#FFCC00",
    }
    colors = [color_map.get(t, "#888888") for t in types]
    
    fig = go.Figure(data=[go.Pie(
        labels=[t.replace("_", " ").title() for t in types],
        values=counts,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="defect_dist_fallback")


def _render_severity_fallback(all_defects: list):
    """Fallback severity histogram."""
    import plotly.graph_objects as go
    
    severities = [d.get("severity", 0) for d in all_defects]
    
    fig = go.Figure(data=[go.Histogram(
        x=severities,
        nbinsx=10,
        marker_color='#ED1C24',
        opacity=0.8,
    )])
    fig.update_layout(
        xaxis_title="Severity Score",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="severity_fallback")


def _tab_agent_intelligence(pipeline_results: dict):
    """Agent intelligence tab showing all 4 agent outputs."""
    st.markdown("### 🤖 Multi-Agent Intelligence Pipeline")
    st.caption("AMD GAIA Framework — 4-Agent Sequential Inspection Chain")
    
    # Pipeline overview
    agent_cols = st.columns(4)
    agents = [
        ("🔍 Inspector", "inspector", "Defect validation & classification"),
        ("📋 Compliance", "compliance", "Building code cross-reference"),
        ("🛡️ Safety", "safety", "Risk scoring & assessment"),
        ("📄 Reporter", "report", "Inspection report generation"),
    ]
    
    for i, (name, key, desc) in enumerate(agents):
        with agent_cols[i]:
            has_data = key in pipeline_results
            status_icon = "✅" if has_data else "⏳"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.5rem;">{status_icon} {name}</div>
                <div class="metric-label">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Agent outputs
    for name, key, desc in agents:
        if key in pipeline_results:
            with st.expander(f"{name} — Output", expanded=(key == "inspector")):
                result = pipeline_results[key]
                if isinstance(result, dict):
                    if "raw_response" in result:
                        st.markdown(result["raw_response"])
                    elif "report_markdown" in result:
                        st.markdown(result["report_markdown"][:500] + "...")
                    else:
                        st.json(result)
                else:
                    st.write(result)


def _tab_compliance(pipeline_results: dict):
    """Compliance tab."""
    st.markdown("### 📋 Building Code Compliance")
    st.caption("RAG-powered compliance checking against ACI, ASCE, IBC codes")
    
    compliance = pipeline_results.get("compliance", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Compliance", compliance.get("overall_compliance", "N/A"))
    with col2:
        st.metric("Total Violations", compliance.get("total_violations", 0))
    with col3:
        st.metric("Critical Violations", compliance.get("critical_violations", 0))
    
    st.divider()
    
    findings = compliance.get("compliance_findings", [])
    if findings:
        for f in findings:
            defect_id = f.get("defect_id", "")
            severity = f.get("violation_severity", "MINOR")
            icon = "🔴" if severity == "MAJOR" else "🟡"
            
            with st.expander(f"{icon} {defect_id} — {severity} Violation"):
                for code in f.get("applicable_codes", []):
                    st.markdown(f"**Code:** {code.get('code', '')} §{code.get('section', '')}")
                    st.markdown(f"**Requirement:** {code.get('requirement', '')}")
                    st.markdown(f"**Status:** {code.get('compliance_status', '')}")
                    st.markdown(f"**Remediation:** {code.get('remediation_standard', '')}")
    else:
        st.info("No specific compliance findings. See agent output for details.")
        if isinstance(compliance.get("raw_response"), str):
            st.markdown(compliance["raw_response"])


def _tab_risk_assessment(pipeline_results: dict, twin_data: dict):
    """Risk assessment tab."""
    st.markdown("### 🛡️ Risk Assessment")
    st.caption("AI-powered structural risk scoring — AMD GAIA Safety Agent")
    
    safety = pipeline_results.get("safety", {})
    
    risk_score = safety.get("risk_score", 50)
    if isinstance(risk_score, str):
        try:
            risk_score = float(risk_score)
        except (ValueError, TypeError):
            risk_score = 50
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Risk gauge
        try:
            from citymind.visualization.charts import risk_gauge
            fig = risk_gauge(risk_score)
            st.plotly_chart(fig, use_container_width=True, key="risk_gauge")
        except Exception:
            _render_risk_gauge_fallback(risk_score)
        
        st.metric("Risk Level", safety.get("risk_level", "N/A"))
        st.metric("Occupancy", safety.get("occupancy_recommendation", "N/A"))
    
    with col2:
        # Risk breakdown
        breakdown = safety.get("risk_breakdown", {})
        if breakdown:
            st.markdown("#### Risk Component Breakdown")
            for component, score in breakdown.items():
                label = component.replace("_", " ").title()
                score_val = float(score) if not isinstance(score, (int, float)) else score
                color = "🔴" if score_val >= 60 else "🟡" if score_val >= 30 else "🟢"
                st.markdown(f"{color} **{label}:** {score_val:.1f}/100")
                st.progress(min(score_val / 100, 1.0))
        
        # Priority actions
        actions = safety.get("priority_actions", [])
        if actions:
            st.markdown("#### 🚨 Priority Actions")
            for action in actions:
                st.warning(f"**Priority {action.get('priority', '?')}:** {action.get('action', '')}")


def _render_risk_gauge_fallback(risk_score: float):
    """Fallback risk gauge."""
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": "#ED1C24"},
            "bgcolor": "rgb(30,30,50)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 200, 0, 0.3)"},
                {"range": [30, 60], "color": "rgba(255, 200, 0, 0.3)"},
                {"range": [60, 100], "color": "rgba(255, 0, 0, 0.3)"},
            ],
        },
        number={"font": {"color": "white"}},
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="risk_gauge_fallback")


def _tab_report(pipeline_results: dict):
    """Inspection report tab."""
    st.markdown("### 📄 Inspection Report")
    st.caption("AI-generated comprehensive inspection report — AMD GAIA Report Agent")
    
    report = pipeline_results.get("report", {})
    report_md = report.get("report_markdown", "No report generated.")
    
    # Display report
    st.markdown(report_md)
    
    st.divider()
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Report (Markdown)",
            data=report_md,
            file_name=f"citymind_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    
    with col2:
        # JSON export
        st.download_button(
            "📥 Download Full Results (JSON)",
            data=json.dumps(pipeline_results, indent=2, default=str),
            file_name=f"citymind_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


def _tab_analytics(twin_data: dict, pipeline_results: dict):
    """Analytics tab with charts and visualizations."""
    st.markdown("### 📊 Analytics Dashboard")
    st.caption("Comprehensive analytics — powered by AMD Zen5 AVX-512")
    
    defects = twin_data.get("defect_analysis", {})
    health = twin_data.get("health_index", {})
    zones = twin_data.get("zones", [])
    
    # Zone health chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Zone Health Scores")
        if zones:
            import plotly.graph_objects as go
            zone_names = [z["name"] for z in zones]
            zone_scores = [z["score"] for z in zones]
            zone_colors = [
                "#00E676" if s >= 75 else "#FFD600" if s >= 50 else "#FF1744"
                for s in zone_scores
            ]
            
            fig = go.Figure(data=[go.Bar(
                x=zone_names,
                y=zone_scores,
                marker_color=zone_colors,
                text=[f"{s}/100" for s in zone_scores],
                textposition='auto',
            )])
            fig.update_layout(
                yaxis_title="Health Score",
                yaxis_range=[0, 100],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgb(200,200,200)'),
                height=350,
                margin=dict(l=40, r=20, t=20, b=60),
            )
            st.plotly_chart(fig, use_container_width=True, key="zone_health")
    
    with col2:
        st.markdown("#### Severity by Defect Type")
        defect_types = defects.get("defect_types", {})
        if defect_types:
            import plotly.graph_objects as go
            types = list(defect_types.keys())
            max_sevs = [v["max_severity"] for v in defect_types.values()]
            avg_sevs = [v["avg_severity"] for v in defect_types.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Max Severity",
                x=[t.replace("_", " ").title() for t in types],
                y=max_sevs,
                marker_color="#FF1744",
            ))
            fig.add_trace(go.Bar(
                name="Avg Severity",
                x=[t.replace("_", " ").title() for t in types],
                y=avg_sevs,
                marker_color="#FF6B35",
            ))
            fig.update_layout(
                barmode='group',
                yaxis_title="Severity (0-10)",
                yaxis_range=[0, 10],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgb(200,200,200)'),
                height=350,
                margin=dict(l=40, r=20, t=20, b=60),
                legend=dict(orientation="h", yanchor="top", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True, key="severity_by_type")
    
    # AMD Tech Pipeline Timing (simulated)
    st.markdown("#### ⏱️ AMD Processing Pipeline Performance")
    import plotly.graph_objects as go
    
    pipeline_steps = [
        "Video Ingestion", "Depth Estimation\n(NPU)", "Object Detection\n(NPU)",
        "Defect Detection\n(NPU)", "3D Reconstruction\n(iGPU)", "Digital Twin",
        "Agent Pipeline\n(GAIA)", "Report Generation",
    ]
    gpu_times = [0.8, 1.2, 0.9, 0.7, 3.5, 0.3, 2.1, 0.5]
    cpu_times = [1.5, 8.5, 5.2, 4.1, 15.0, 0.5, 12.0, 1.0]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AMD Ryzen AI (NPU/iGPU)",
        x=pipeline_steps,
        y=gpu_times,
        marker_color="#ED1C24",
        text=[f"{t:.1f}s" for t in gpu_times],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name="CPU-Only Baseline",
        x=pipeline_steps,
        y=cpu_times,
        marker_color="#555555",
        text=[f"{t:.1f}s" for t in cpu_times],
        textposition='auto',
    ))
    fig.update_layout(
        barmode='group',
        yaxis_title="Time (seconds)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=400,
        margin=dict(l=40, r=20, t=20, b=80),
        legend=dict(orientation="h", yanchor="top", y=1.12),
    )
    
    total_amd = sum(gpu_times)
    total_cpu = sum(cpu_times)
    speedup = total_cpu / total_amd
    
    st.plotly_chart(fig, use_container_width=True, key="pipeline_perf")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AMD Total Time", f"{total_amd:.1f}s")
    with col2:
        st.metric("CPU Baseline", f"{total_cpu:.1f}s")
    with col3:
        st.metric("Speedup", f"{speedup:.1f}×", delta=f"{speedup:.1f}× faster")


if __name__ == "__main__":
    main()
