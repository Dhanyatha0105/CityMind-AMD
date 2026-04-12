"""
CityMind End-to-End Pipeline
Orchestrates all 6 layers: Ingestion → Perception → Reconstruction → Twin → Agents → Visualization

AMD Tech: Full AMD technology stack integration
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CityMindPipeline:
    """
    Full end-to-end CityMind pipeline.
    
    Processes infrastructure video through all 6 layers:
    1. Video Ingestion — Keyframe extraction & quality filtering
    2. Edge Perception — Depth, detection, defect analysis (AMD NPU)
    3. 3D Reconstruction — Point cloud generation (AMD iGPU)
    4. Digital Twin — Semantic 3D model with health metrics
    5. Multi-Agent Intelligence — 4-agent inspection (AMD GAIA)
    6. Visualization — Dashboard & reports
    
    AMD Technology Stack:
    - Training: Instinct MI300X, ROCm 7.2, PyTorch TunableOp
    - Optimization: Vitis AI Quantizer, ONNX Runtime + Vitis AI EP
    - Edge: Ryzen AI NPU (XDNA), CVML Library, Ryzen AI Max+ iGPU
    - Intelligence: GAIA Framework, Lemonade, Ryzen AI Software v1.7
    - Simulation: Genesis Simulation Engine
    """
    
    def __init__(
        self,
        output_dir: str = None,
        llm_provider: str = "deterministic",
        api_key: str = None,
        max_frames: int = 30,
        detection_conf: float = 0.35,
        use_npu: bool = False,
        building_codes_dir: str = None,
    ):
        from citymind.config import OUTPUT_DIR, BUILDING_CODES_DIR
        
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.max_frames = max_frames
        self.detection_conf = detection_conf
        self.use_npu = use_npu
        self.building_codes_dir = building_codes_dir or str(BUILDING_CODES_DIR)
        
        self.timing = {}
    
    def run(
        self,
        video_path: str = None,
        image_dir: str = None,
        structure_type: str = "Reinforced Concrete Building",
    ) -> Dict:
        """
        Run the full CityMind pipeline.
        
        Args:
            video_path: Path to infrastructure video
            image_dir: Path to directory of images (alternative to video)
            structure_type: Type of structure being inspected
            
        Returns:
            Complete pipeline results dictionary
        """
        logger.info("=" * 70)
        logger.info("  🏗️  CityMind AI Infrastructure Digital Twin Pipeline")
        logger.info("  Powered by AMD Ryzen AI | GAIA Framework")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        results = {
            "pipeline_id": f"CITY-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "input": {
                "video_path": video_path,
                "image_dir": image_dir,
                "structure_type": structure_type,
            },
        }
        
        # ── Layer 1: Video Ingestion ────────────────────────────
        logger.info("\n📹 LAYER 1: Video Ingestion")
        t0 = time.time()
        frames, video_metadata = self._layer1_ingest(video_path, image_dir)
        self.timing["layer1_ingestion"] = time.time() - t0
        results["frames"] = {"count": len(frames), "paths": frames[:5]}
        results["video_metadata"] = video_metadata
        logger.info(f"  → {len(frames)} quality frames extracted ({self.timing['layer1_ingestion']:.1f}s)")
        
        if not frames:
            logger.warning("No frames extracted, generating synthetic data")
            return self._run_synthetic(structure_type)
        
        # ── Layer 2: Edge Perception ────────────────────────────
        logger.info("\n🔍 LAYER 2: Edge Perception (AMD Ryzen AI NPU)")
        t0 = time.time()
        perception_results = self._layer2_perception(frames)
        self.timing["layer2_perception"] = time.time() - t0
        results["perception"] = perception_results
        logger.info(f"  → Perception complete ({self.timing['layer2_perception']:.1f}s)")
        
        # ── Layer 3: 3D Reconstruction ──────────────────────────
        logger.info("\n🗺️ LAYER 3: 3D Reconstruction (AMD Ryzen AI Max+ iGPU)")
        t0 = time.time()
        reconstruction = self._layer3_reconstruction(frames, perception_results)
        self.timing["layer3_reconstruction"] = time.time() - t0
        results["reconstruction"] = reconstruction
        logger.info(f"  → Reconstruction complete ({self.timing['layer3_reconstruction']:.1f}s)")
        
        # ── Layer 4: Digital Twin ───────────────────────────────
        logger.info("\n🏢 LAYER 4: Digital Twin Engine")
        t0 = time.time()
        twin_data = self._layer4_digital_twin(
            reconstruction, perception_results, video_metadata,
            structure_type=structure_type,
        )
        self.timing["layer4_twin"] = time.time() - t0
        results["twin"] = twin_data
        logger.info(f"  → Digital twin created ({self.timing['layer4_twin']:.1f}s)")
        
        # ── Layer 5: Multi-Agent Intelligence ───────────────────
        logger.info("\n🤖 LAYER 5: Multi-Agent Intelligence (AMD GAIA)")
        t0 = time.time()
        agent_results = self._layer5_agents(twin_data, perception_results)
        self.timing["layer5_agents"] = time.time() - t0
        results["agents"] = agent_results
        logger.info(f"  → Agent pipeline complete ({self.timing['layer5_agents']:.1f}s)")
        
        # ── Layer 6: Output ─────────────────────────────────────
        logger.info("\n📊 LAYER 6: Report & Visualization")
        t0 = time.time()
        report_path = self._layer6_report(twin_data, agent_results)
        self.timing["layer6_report"] = time.time() - t0
        results["report_path"] = report_path
        logger.info(f"  → Report generated ({self.timing['layer6_report']:.1f}s)")
        
        # ── Summary ─────────────────────────────────────────────
        total_time = time.time() - start_time
        self.timing["total"] = total_time
        results["timing"] = self.timing
        results["completed_at"] = datetime.now().isoformat()
        
        # Save full results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"  ✅ Pipeline complete in {total_time:.1f}s")
        logger.info(f"  Results saved to: {results_path}")
        logger.info("=" * 70)
        
        return results
    
    def _layer1_ingest(self, video_path: str, image_dir: str):
        """Layer 1: Extract and filter keyframes."""
        from citymind.ingestion.frame_extractor import FrameExtractor
        from citymind.ingestion.quality_filter import QualityFilter
        from citymind.ingestion.metadata import MetadataExtractor
        
        frames_dir = str(self.output_dir / "frames")
        extractor = FrameExtractor(
            max_frames=self.max_frames,
        )
        quality_filter = QualityFilter()
        metadata_extractor = MetadataExtractor()
        
        frames = []
        video_metadata = {}
        
        if video_path and Path(video_path).exists():
            result = extractor.extract_frames(video_path, output_dir=frames_dir)
            frames = result.get("frame_paths", [])
            video_metadata = metadata_extractor.extract_video_metadata(video_path)
        elif image_dir and Path(image_dir).exists():
            import glob
            frames = sorted(glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png"))
        
        # Quality filter
        if frames:
            filter_result = quality_filter.filter_frames(frames)
            frames = filter_result.get("passed", frames)
        
        return frames, video_metadata
    
    def _layer2_perception(self, frames: List[str]) -> Dict:
        """Layer 2: Run perception models."""
        from citymind.perception.depth_estimation import DepthEstimator
        from citymind.perception.object_detection import StructuralDetector
        from citymind.perception.defect_detection import DefectDetector
        
        depth_estimator = DepthEstimator()
        detector = StructuralDetector(confidence=self.detection_conf)
        defect_detector = DefectDetector(confidence=self.detection_conf)
        
        frame_detections = []
        frame_defects = []
        depth_maps = []
        
        for i, frame_path in enumerate(frames):
            logger.info(f"  Processing frame {i+1}/{len(frames)}: {Path(frame_path).name}")
            
            # Depth estimation
            try:
                depth_map = depth_estimator.estimate(frame_path)
                depth_maps.append({"frame_idx": i, "shape": list(depth_map.shape) if depth_map is not None else []})
            except Exception as e:
                logger.warning(f"  Depth estimation failed for frame {i}: {e}")
            
            # Structural detection
            try:
                detections = detector.detect(frame_path)
                frame_detections.append(detections)
            except Exception as e:
                logger.warning(f"  Detection failed for frame {i}: {e}")
                frame_detections.append({"detections": []})
            
            # Defect detection
            try:
                defects = defect_detector.detect(frame_path)
                frame_defects.append(defects)
            except Exception as e:
                logger.warning(f"  Defect detection failed for frame {i}: {e}")
                frame_defects.append({"defects": []})
        
        return {
            "frame_detections": frame_detections,
            "frame_defects": frame_defects,
            "depth_maps": depth_maps,
            "frames_processed": len(frames),
        }
    
    def _layer3_reconstruction(self, frames: List[str], perception: Dict) -> Dict:
        """Layer 3: 3D reconstruction."""
        from citymind.reconstruction.colmap_pipeline import ColmapPipeline
        from citymind.reconstruction.semantic_projection import SemanticProjector
        
        recon_dir = str(self.output_dir / "reconstruction")
        
        # Run COLMAP SfM
        colmap = ColmapPipeline(output_dir=recon_dir)
        
        frames_dir = str(Path(frames[0]).parent) if frames else ""
        point_cloud_path, sfm_stats = colmap.run(frames_dir)
        
        # Semantic projection
        projector = SemanticProjector()
        label_stats = projector.project_labels(
            point_cloud_path=point_cloud_path,
            frame_detections=perception.get("frame_detections", []),
            frame_defects=perception.get("frame_defects", []),
        )
        
        return {
            "point_cloud_path": point_cloud_path,
            "sfm_stats": sfm_stats,
            "label_stats": label_stats,
        }
    
    def _layer4_digital_twin(
        self,
        reconstruction: Dict,
        perception: Dict,
        video_metadata: Dict,
        structure_type: str = "Unknown",
    ) -> Dict:
        """Layer 4: Create digital twin."""
        from citymind.digital_twin.twin_engine import DigitalTwinEngine
        
        engine = DigitalTwinEngine()
        
        twin = engine.create_twin(
            point_cloud_path=reconstruction.get("point_cloud_path", ""),
            label_stats=reconstruction.get("label_stats", {}),
            frame_defects=perception.get("frame_defects", []),
            frame_detections=perception.get("frame_detections", []),
            video_metadata=video_metadata,
            inspection_metadata={"structure_type": structure_type},
        )
        
        return twin
    
    def _layer5_agents(self, twin_data: Dict, perception: Dict) -> Dict:
        """Layer 5: Multi-agent intelligence."""
        from citymind.agents.orchestrator import AgentOrchestrator
        from citymind.rag.ingest import BuildingCodeIngestor
        from citymind.rag.retriever import BuildingCodeRetriever
        
        # Build RAG context
        rag_context = ""
        try:
            ingestor = BuildingCodeIngestor()
            n_chunks = ingestor.ingest_directory(self.building_codes_dir)
            
            retriever = BuildingCodeRetriever(ingestor)
            all_defects = twin_data.get("defect_analysis", {}).get("critical_defects", [])
            all_defects += twin_data.get("defect_analysis", {}).get("high_defects", [])
            rag_context = retriever.get_context_for_defects(all_defects)
            
            logger.info(f"  RAG: Indexed {n_chunks} chunks, retrieved context for {len(all_defects)} defects")
        except Exception as e:
            logger.warning(f"  RAG setup failed: {e}")
        
        # Run agent pipeline
        orchestrator = AgentOrchestrator(
            llm_provider=self.llm_provider,
            api_key=self.api_key,
        )
        
        agent_results = orchestrator.run_inspection_pipeline(
            twin_data=twin_data,
            frame_defects=perception.get("frame_defects", []),
            frame_detections=perception.get("frame_detections", []),
            rag_context=rag_context,
        )
        
        return agent_results
    
    def _layer6_report(self, twin_data: Dict, agent_results: Dict) -> str:
        """Layer 6: Generate reports."""
        from citymind.visualization.report_pdf import ReportGenerator
        
        generator = ReportGenerator()
        
        # Generate PDF
        pdf_path = str(self.output_dir / "reports" / f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        report_path = generator.generate_pdf(twin_data, agent_results, pdf_path)
        
        # Generate Markdown
        md_path = str(self.output_dir / "reports" / f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        generator.generate_markdown(twin_data, agent_results, md_path)
        
        return report_path
    
    def _run_synthetic(self, structure_type: str) -> Dict:
        """Run pipeline with synthetic demo data."""
        from citymind.reconstruction.point_cloud_utils import PointCloudUtils
        
        logger.info("Running synthetic demo pipeline...")
        
        # Generate synthetic point cloud
        pc_data = PointCloudUtils.generate_synthetic_building()
        
        # Write PLY
        ply_path = str(self.output_dir / "reconstruction" / "synthetic_building.ply")
        PointCloudUtils.write_ply(ply_path, pc_data["points"], pc_data["colors"])
        
        # Create synthetic perception data
        perception = {
            "frame_defects": [{
                "defects": [
                    {"defect_type": "crack", "severity": 7.5, "confidence": 0.89, "bbox": [120, 200, 280, 350], "detection_method": "yolov8"},
                    {"defect_type": "spalling", "severity": 8.2, "confidence": 0.92, "bbox": [400, 150, 550, 300], "detection_method": "yolov8"},
                    {"defect_type": "corrosion", "severity": 6.1, "confidence": 0.78, "bbox": [50, 400, 200, 520], "detection_method": "color_analysis"},
                    {"defect_type": "exposed_rebar", "severity": 9.0, "confidence": 0.95, "bbox": [180, 280, 320, 400], "detection_method": "yolov8"},
                ],
            }],
            "frame_detections": [{
                "detections": [
                    {"structural_type": "column", "confidence": 0.92},
                    {"structural_type": "beam", "confidence": 0.88},
                    {"structural_type": "wall", "confidence": 0.95},
                    {"structural_type": "slab", "confidence": 0.90},
                ],
            }],
            "depth_maps": [],
            "frames_processed": 0,
        }
        
        # Run through remaining layers
        video_metadata = {
            "filename": "synthetic_demo",
            "duration": 30.0,
        }
        
        twin_data = self._layer4_digital_twin(
            {"point_cloud_path": ply_path, "label_stats": {"num_points": len(pc_data["points"])}},
            perception, video_metadata, structure_type,
        )
        
        agent_results = self._layer5_agents(twin_data, perception)
        report_path = self._layer6_report(twin_data, agent_results)
        
        return {
            "pipeline_id": f"CITY-SYNTHETIC-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "mode": "synthetic",
            "twin": twin_data,
            "agents": agent_results,
            "report_path": report_path,
        }


def main():
    """CLI entry point for the CityMind pipeline."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="CityMind — AI-Powered Infrastructure Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with video
  python -m citymind.pipeline --video path/to/video.mp4
  
  # Run with images
  python -m citymind.pipeline --images path/to/images/
  
  # Run synthetic demo
  python -m citymind.pipeline --demo
  
  # Run with OpenAI
  python -m citymind.pipeline --demo --llm openai --api-key sk-...
""",
    )
    
    parser.add_argument("--video", help="Path to infrastructure video")
    parser.add_argument("--images", help="Path to directory of images")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--llm", choices=["deterministic", "openai", "ollama"], default="deterministic")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--structure-type", default="Reinforced Concrete Building")
    
    args = parser.parse_args()
    
    pipeline = CityMindPipeline(
        output_dir=args.output,
        llm_provider=args.llm,
        api_key=args.api_key,
        max_frames=args.max_frames,
    )
    
    if args.demo:
        results = pipeline._run_synthetic(args.structure_type)
    else:
        results = pipeline.run(
            video_path=args.video,
            image_dir=args.images,
            structure_type=args.structure_type,
        )
    
    print(f"\n✅ Pipeline complete!")
    print(f"Results: {json.dumps({k: v for k, v in results.items() if k != 'twin'}, indent=2, default=str)[:1000]}")


if __name__ == "__main__":
    main()
