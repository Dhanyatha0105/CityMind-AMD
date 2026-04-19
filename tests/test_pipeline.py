"""
CityMind Test Suite — Unit & Integration Tests
Tests all 6 layers of the pipeline.
"""

import sys
import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════
# Layer 1: Ingestion Tests
# ═══════════════════════════════════════════════════════════════

class TestQualityFilter:
    """Test quality filtering module."""
    
    def test_assess_quality_sharp_image(self, tmp_path):
        import cv2
        from citymind.ingestion.quality_filter import QualityFilter
        qf = QualityFilter()
        
        # Create a sharp image with texture
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = str(tmp_path / "sharp.jpg")
        cv2.imwrite(img_path, img)
        
        result = qf.assess_quality(img_path)
        assert "quality_score" in result
        assert result["blur"]["score"] > 0
    
    def test_assess_quality_blurry_image(self, tmp_path):
        import cv2
        from citymind.ingestion.quality_filter import QualityFilter
        qf = QualityFilter()
        
        # Create a uniform (very blurry) image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        img_path = str(tmp_path / "blurry.jpg")
        cv2.imwrite(img_path, img)
        
        result = qf.assess_quality(img_path)
        assert result["blur"]["score"] < 10
        assert result["blur"]["pass"] == False
    
    def test_assess_nonexistent_file(self):
        from citymind.ingestion.quality_filter import QualityFilter
        qf = QualityFilter()
        
        result = qf.assess_quality("/nonexistent/image.jpg")
        assert result["pass"] is False
    
    def test_filter_frames(self, tmp_path):
        import cv2
        from citymind.ingestion.quality_filter import QualityFilter
        qf = QualityFilter(blur_threshold=10.0)
        
        # Create several test images
        paths = []
        for i in range(3):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            p = str(tmp_path / f"frame_{i}.jpg")
            cv2.imwrite(p, img)
            paths.append(p)
        
        result = qf.filter_frames(paths)
        assert "passed" in result
        assert "failed" in result
        assert result["total"] == 3


class TestMetadata:
    """Test metadata extraction."""
    
    def test_metadata_extractor_init(self):
        from citymind.ingestion.metadata import MetadataExtractor
        me = MetadataExtractor()
        assert me is not None
    
    def test_extract_from_nonexistent_file(self):
        from citymind.ingestion.metadata import MetadataExtractor
        me = MetadataExtractor()
        result = me.extract_video_metadata("/nonexistent/video.mp4")
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════
# Layer 2: Perception Tests (mocked heavy models)
# ═══════════════════════════════════════════════════════════════

class TestNPUInference:
    """Test NPU inference wrapper."""
    
    def test_npu_wrapper_init(self):
        from citymind.perception.npu_inference import NPUInferenceEngine
        engine = NPUInferenceEngine()
        assert engine is not None
        assert isinstance(engine._available_providers, list)


# ═══════════════════════════════════════════════════════════════
# Layer 3: Reconstruction Tests
# ═══════════════════════════════════════════════════════════════

class TestPointCloudUtils:
    """Test point cloud utilities."""
    
    def test_generate_synthetic_building(self):
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        result = PointCloudUtils.generate_synthetic_building()
        
        assert "points" in result
        assert "colors" in result
        assert result["points"].shape[1] == 3
        assert result["colors"].shape[1] == 3
        assert len(result["points"]) > 100
    
    def test_write_and_read_ply(self, tmp_path):
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        
        points = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        
        ply_path = str(tmp_path / "test.ply")
        PointCloudUtils.write_ply(ply_path, points, colors)
        
        assert Path(ply_path).exists()
        assert Path(ply_path).stat().st_size > 0
        
        # Read back
        data = PointCloudUtils.read_ply(ply_path)
        assert data["points"].shape == (100, 3)
    
    def test_downsample(self):
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        
        points = np.random.rand(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        
        result = PointCloudUtils.downsample(points, colors, voxel_size=0.1)
        assert "points" in result
        assert len(result["points"]) <= 1000
    
    def test_read_nonexistent_ply(self):
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        result = PointCloudUtils.read_ply("/nonexistent/file.ply")
        assert result["points"].shape[1] == 3
        assert len(result["points"]) == 0


class TestSemanticProjection:
    """Test semantic projection."""
    
    def test_project_labels_empty(self):
        from citymind.reconstruction.semantic_projection import SemanticProjector
        projector = SemanticProjector()
        
        result = projector.project_labels(
            point_cloud_path="",
            frame_detections=[],
            frame_defects=[],
        )
        assert isinstance(result, dict)


class TestColmapPipeline:
    """Test COLMAP pipeline (synthetic fallback)."""
    
    def test_synthetic_fallback(self, tmp_path):
        from citymind.reconstruction.colmap_pipeline import ColmapPipeline
        
        pipeline = ColmapPipeline(output_dir=str(tmp_path))
        result = pipeline.reconstruct(str(tmp_path), str(tmp_path / "output"))
        
        assert "point_cloud_path" in result
        assert result.get("method") in ["colmap_sfm", "synthetic_feature_based", "synthetic_fallback"]


# ═══════════════════════════════════════════════════════════════
# Layer 4: Digital Twin Tests
# ═══════════════════════════════════════════════════════════════

class TestDigitalTwinEngine:
    """Test digital twin engine."""
    
    def test_create_twin(self, tmp_path):
        from citymind.digital_twin.twin_engine import DigitalTwinEngine
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        
        # Create a synthetic point cloud
        pc_data = PointCloudUtils.generate_synthetic_building()
        ply_path = str(tmp_path / "test.ply")
        PointCloudUtils.write_ply(ply_path, pc_data["points"], pc_data["colors"])
        
        engine = DigitalTwinEngine()
        twin = engine.create_twin(
            point_cloud_path=ply_path,
            label_stats={"num_points": len(pc_data["points"])},
            frame_defects=[{
                "defects": [
                    {"defect_type": "crack", "severity": 7.5, "confidence": 0.89, "bbox": [120, 200, 280, 350]},
                ]
            }],
            frame_detections=[{
                "detections": [
                    {"structural_type": "column", "confidence": 0.92},
                ]
            }],
            video_metadata={"filename": "test.mp4", "duration": 30.0},
            inspection_metadata={"structure_type": "Test Building"},
        )
        
        assert "twin_id" in twin
        assert "health_index" in twin
        assert "defect_analysis" in twin
        assert twin["health_index"]["score"] >= 0
        assert twin["health_index"]["score"] <= 100
    
    def test_health_index_no_defects(self, tmp_path):
        from citymind.digital_twin.twin_engine import DigitalTwinEngine
        
        engine = DigitalTwinEngine()
        ply_path = str(tmp_path / "test.ply")
        Path(ply_path).touch()
        
        twin = engine.create_twin(
            point_cloud_path=ply_path,
            label_stats={},
            frame_defects=[],
            frame_detections=[],
            video_metadata={},
        )
        
        assert twin["health_index"]["score"] == 95
        assert twin["health_index"]["grade"] == "A"
    
    def test_health_index_critical(self, tmp_path):
        from citymind.digital_twin.twin_engine import DigitalTwinEngine
        
        engine = DigitalTwinEngine()
        ply_path = str(tmp_path / "test.ply")
        Path(ply_path).touch()
        
        # Multiple severe defects
        twin = engine.create_twin(
            point_cloud_path=ply_path,
            label_stats={},
            frame_defects=[{
                "defects": [
                    {"defect_type": "crack", "severity": 9.0, "confidence": 0.95},
                    {"defect_type": "exposed_rebar", "severity": 9.5, "confidence": 0.90},
                    {"defect_type": "spalling", "severity": 8.0, "confidence": 0.85},
                ]
            }],
            frame_detections=[],
            video_metadata={},
        )
        
        assert twin["health_index"]["score"] < 50
        assert twin["health_index"]["grade"] in ("D", "F")


# ═══════════════════════════════════════════════════════════════
# Layer 5: Agent Tests
# ═══════════════════════════════════════════════════════════════

class TestAgentOrchestrator:
    """Test multi-agent orchestrator."""
    
    def test_deterministic_mode(self):
        from citymind.agents.orchestrator import AgentOrchestrator
        
        orch = AgentOrchestrator(llm_provider="deterministic")
        
        twin_data = {
            "twin_id": "TEST-001",
            "health_index": {"score": 45, "grade": "D", "status": "POOR", 
                           "description": "Test", "recommendation": "Test rec"},
            "defect_analysis": {
                "total_defects": 3,
                "critical_defects": [
                    {"id": "DEF-001", "defect_type": "crack", "severity": 8.0, "confidence": 0.9},
                    {"id": "DEF-002", "defect_type": "spalling", "severity": 7.5, "confidence": 0.85},
                ],
                "high_defects": [
                    {"id": "DEF-003", "defect_type": "corrosion", "severity": 6.0, "confidence": 0.7},
                ],
            },
            "structural_elements": [{"type": "column", "count": 4}],
        }
        
        results = orch.run_inspection_pipeline(
            twin_data=twin_data,
            frame_defects=[],
            frame_detections=[],
            rag_context="Test building code context",
        )
        
        assert "inspector" in results
        assert "compliance" in results
        assert "safety" in results
        assert "report" in results
        assert "pipeline_id" in results
    
    def test_inspector_validates_defects(self):
        from citymind.agents.orchestrator import AgentOrchestrator
        
        orch = AgentOrchestrator(llm_provider="deterministic")
        result_str = orch._deterministic_inspector({
            "defect_analysis": {
                "critical_defects": [
                    {"id": "DEF-001", "defect_type": "crack", "severity": 8.0},
                ],
                "total_defects": 1,
            },
            "health_index": {"score": 50},
        })
        
        result = json.loads(result_str)
        assert "validated_defects" in result
        assert len(result["validated_defects"]) >= 1
    
    def test_compliance_maps_codes(self):
        from citymind.agents.orchestrator import AgentOrchestrator
        
        orch = AgentOrchestrator(llm_provider="deterministic")
        result_str = orch._deterministic_compliance({
            "inspector_findings": {
                "validated_defects": [
                    {"id": "DEF-001", "defect_type": "crack", "severity_score": 8.0},
                ],
            },
        })
        
        result = json.loads(result_str)
        assert "compliance_findings" in result
        assert "overall_compliance" in result
    
    def test_safety_scoring(self):
        from citymind.agents.orchestrator import AgentOrchestrator
        
        orch = AgentOrchestrator(llm_provider="deterministic")
        result_str = orch._deterministic_safety({
            "health_index": {"score": 30},
        })
        
        result = json.loads(result_str)
        assert "risk_score" in result
        assert "risk_level" in result
        assert result["risk_score"] > 0


# ═══════════════════════════════════════════════════════════════
# Layer 5B: RAG Tests
# ═══════════════════════════════════════════════════════════════

class TestRAGPipeline:
    """Test RAG document ingestion and retrieval."""
    
    def test_ingest_text_documents(self, tmp_path):
        from citymind.rag.ingest import BuildingCodeIngestor
        
        # Create sample documents
        doc1 = tmp_path / "aci_318.txt"
        doc1.write_text("ACI 318-19: Maximum crack width 0.3mm for interior exposure.")
        
        doc2 = tmp_path / "asce_7.txt"
        doc2.write_text("ASCE 7-22: Seismic base shear V = Cs * W.")
        
        ingestor = BuildingCodeIngestor()
        docs = ingestor.load_documents(str(tmp_path))
        
        assert len(docs) >= 2
    
    def test_chunk_documents(self, tmp_path):
        from citymind.rag.ingest import BuildingCodeIngestor
        
        doc = tmp_path / "long_doc.txt"
        doc.write_text("Section 1 about concrete crack limits. " * 200)  # Long document
        
        ingestor = BuildingCodeIngestor(chunk_size=100, chunk_overlap=20)
        docs = ingestor.load_documents(str(tmp_path))
        chunks = ingestor.chunk_documents(docs)
        
        assert len(chunks) >= 1
    
    def test_full_ingest_pipeline(self):
        from citymind.rag.ingest import BuildingCodeIngestor
        
        codes_dir = Path(__file__).parent.parent / "data" / "building_codes"
        if not codes_dir.exists():
            pytest.skip("Building codes directory not found")
        
        ingestor = BuildingCodeIngestor()
        n_chunks = ingestor.ingest_directory(str(codes_dir))
        
        assert n_chunks > 0
        assert len(ingestor.chunks) > 0
    
    def test_retriever_query(self):
        from citymind.rag.ingest import BuildingCodeIngestor
        from citymind.rag.retriever import BuildingCodeRetriever
        
        codes_dir = Path(__file__).parent.parent / "data" / "building_codes"
        if not codes_dir.exists():
            pytest.skip("Building codes directory not found")
        
        ingestor = BuildingCodeIngestor()
        ingestor.ingest_directory(str(codes_dir))
        
        retriever = BuildingCodeRetriever(ingestor)
        results = retriever.query("crack width limits for concrete")
        
        assert len(results) > 0
        assert "text" in results[0]
    
    def test_retriever_default_context(self):
        from citymind.rag.retriever import BuildingCodeRetriever
        
        retriever = BuildingCodeRetriever(ingestor=None)
        results = retriever.query("crack detection")
        
        assert len(results) > 0  # Should return default context


# ═══════════════════════════════════════════════════════════════
# Layer 6: Visualization Tests
# ═══════════════════════════════════════════════════════════════

class TestReportGenerator:
    """Test report generation."""
    
    def test_markdown_report(self, tmp_path):
        from citymind.visualization.report_pdf import ReportGenerator
        
        gen = ReportGenerator()
        twin_data = {
            "twin_id": "TEST-001",
            "health_index": {"score": 65, "grade": "C", "status": "FAIR",
                           "description": "Test desc", "recommendation": "Test rec"},
            "defect_analysis": {"total_defects": 2, "critical_defects": [], "high_defects": []},
        }
        
        md = gen.generate_markdown(twin_data, {"report": {}})
        assert "CityMind" in md
        assert "TEST-001" in md
    
    def test_pdf_report(self, tmp_path):
        from citymind.visualization.report_pdf import ReportGenerator
        
        gen = ReportGenerator()
        twin_data = {
            "twin_id": "TEST-002",
            "health_index": {"score": 45, "grade": "D", "status": "POOR",
                           "description": "Significant issues", "recommendation": "Get help"},
            "defect_analysis": {
                "total_defects": 3,
                "critical_defects": [
                    {"id": "DEF-001", "defect_type": "crack", "severity": 8.0,
                     "confidence": 0.9, "detection_method": "yolov8",
                     "code_reference": "ACI 318-19 S24.3 -- Crack Control"},
                ],
                "high_defects": [],
            },
            "structure_info": {"type": "Test Bridge"},
        }
        
        pipeline_results = {
            "compliance": {"overall_compliance": "NON-COMPLIANT", "total_violations": 2},
            "safety": {"risk_score": 55.0, "risk_level": "ELEVATED"},
        }
        
        pdf_path = str(tmp_path / "test_report.pdf")
        result = gen.generate_pdf(twin_data, pipeline_results, pdf_path)
        
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


class TestCharts:
    """Test chart generation."""
    
    def test_defect_distribution_chart(self):
        from citymind.visualization.charts import defect_distribution_chart
        
        defect_types = {
            "crack": {"count": 5, "avg_severity": 7.2, "max_severity": 9.0},
            "spalling": {"count": 3, "avg_severity": 6.1, "max_severity": 8.0},
            "corrosion": {"count": 2, "avg_severity": 4.5, "max_severity": 5.0},
        }
        
        fig = defect_distribution_chart(defect_types)
        assert fig is not None
    
    def test_health_gauge(self):
        from citymind.visualization.charts import risk_gauge
        
        fig = risk_gauge(72.5)
        assert fig is not None


# ═══════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Test end-to-end pipeline."""
    
    def test_synthetic_pipeline(self, tmp_path):
        from citymind.pipeline import CityMindPipeline
        
        pipeline = CityMindPipeline(
            output_dir=str(tmp_path),
            llm_provider="deterministic",
            max_frames=5,
        )
        
        results = pipeline._run_synthetic("Test Building")
        
        assert "twin" in results
        assert "agents" in results
        assert "report_path" in results
        assert results["twin"]["twin_id"].startswith("TWIN-")
        assert results["twin"]["health_index"]["score"] >= 0
    
    def test_pipeline_with_no_input(self, tmp_path):
        from citymind.pipeline import CityMindPipeline
        
        pipeline = CityMindPipeline(
            output_dir=str(tmp_path),
            llm_provider="deterministic",
        )
        
        # Should fall back to synthetic
        results = pipeline.run(video_path=None, image_dir=None)
        assert "twin" in results


# ═══════════════════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    """Test configuration."""
    
    def test_config_loads(self):
        from citymind.config import PROJECT_ROOT, OUTPUT_DIR, DEFECT_TYPES
        
        assert PROJECT_ROOT.exists()
        assert OUTPUT_DIR.exists()
        assert len(DEFECT_TYPES) > 0
    
    def test_amd_tech_stack(self):
        from citymind.config import AMD_TECH_STACK
        
        assert "training" in AMD_TECH_STACK
        assert "optimization" in AMD_TECH_STACK
        assert "edge_inference" in AMD_TECH_STACK


# ═══════════════════════════════════════════════════════════════
# Deviation Analysis Tests (Stretch Goal)
# ═══════════════════════════════════════════════════════════════

class TestDeviationAnalysis:
    """Test deviation analysis module."""

    def test_cloud_to_cloud_deviation(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()

        # Two identical clouds → zero deviation
        points = np.random.randn(100, 3)
        result = analyzer.cloud_to_cloud_deviation(points, points.copy())

        assert result["analysis_type"] == "cloud_to_cloud"
        assert result["num_points"] == 100
        assert result["mean_mm"] < 1.0  # Should be ~0
        assert result["rmse_mm"] < 1.0

    def test_cloud_to_cloud_with_offset(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()

        source = np.random.randn(200, 3) * 5
        # Add 10mm offset (0.01m)
        target = source + 0.01

        result = analyzer.cloud_to_cloud_deviation(source, target)
        assert result["rmse_mm"] > 5.0  # Should be ~17mm (sqrt(3)*10)
        assert result["severity"] in ("minor", "moderate", "significant")

    def test_cloud_to_cloud_empty(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        result = analyzer.cloud_to_cloud_deviation(np.array([]).reshape(0, 3), np.array([]).reshape(0, 3))
        assert result["num_points"] == 0

    def test_cloud_to_plane_deviation(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()

        # Points on a plane z=0 with small noise
        n = 200
        points = np.column_stack([
            np.random.randn(n) * 10,
            np.random.randn(n) * 10,
            np.random.randn(n) * 0.002,  # ~2mm noise
        ])

        result = analyzer.cloud_to_plane_deviation(points)
        assert result["analysis_type"] == "cloud_to_plane"
        assert result["rmse_mm"] < 10.0  # Should be small

    def test_deviation_severity_classification(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        assert analyzer._classify_deviation(0.5) == "negligible"
        assert analyzer._classify_deviation(3.0) == "minor"
        assert analyzer._classify_deviation(10.0) == "moderate"
        assert analyzer._classify_deviation(20.0) == "significant"
        assert analyzer._classify_deviation(50.0) == "critical"

    def test_temporal_deviation(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()

        # Create scan history with increasing deviation
        scan_history = []
        base_points = np.random.randn(50, 3) * 5
        for i in range(3):
            noise = np.random.randn(50, 3) * 0.005 * (i + 1)
            scan_history.append({
                "timestamp": f"2026-0{i+1}-01T00:00:00",
                "points": (base_points + noise).tolist(),
            })

        result = analyzer.temporal_deviation(scan_history)
        assert result["analysis_type"] == "temporal"
        assert result["scan_count"] == 3
        assert len(result["pairwise_results"]) == 2

    def test_temporal_deviation_single_scan(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        result = analyzer.temporal_deviation([{"timestamp": "2026-01-01", "points": []}])
        assert "error" in result

    def test_analyze_structure(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        cloud = np.random.randn(500, 3) * 10

        result = analyzer.analyze_structure(cloud)
        assert "flatness" in result
        assert "summary" in result
        assert result["current_points"] == 500

    def test_analyze_structure_with_reference(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        current = np.random.randn(300, 3) * 10
        reference = current + np.random.randn(300, 3) * 0.005

        result = analyzer.analyze_structure(current, reference)
        assert "deviation" in result
        assert result["deviation"]["analysis_type"] == "cloud_to_cloud"

    def test_heatmap_generation(self, tmp_path):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()
        points = np.random.randn(200, 3) * 10
        deviations = np.abs(np.random.randn(200)) * 0.01

        output_path = str(tmp_path / "test_heatmap.png")
        result = analyzer.generate_deviation_heatmap(points, deviations, output_path)
        assert "grid_shape" in result
        assert result["coverage"] > 0

    def test_within_tolerance_percentages(self):
        from citymind.digital_twin.deviation_analysis import DeviationAnalyzer

        analyzer = DeviationAnalyzer()

        # All points very close → high tolerance percentages
        source = np.random.randn(100, 3)
        target = source + np.random.randn(100, 3) * 0.0001  # ~0.1mm

        result = analyzer.cloud_to_cloud_deviation(source, target)
        assert result["within_tolerance"]["1mm"] > 90.0
        assert result["within_tolerance"]["5mm"] > 99.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
