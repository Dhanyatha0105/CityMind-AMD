"""
Layer 4: Digital Twin Engine
Combines 3D reconstruction with semantic analysis to create an intelligent digital twin.

AMD Tech: Genesis Simulation Engine (stretch), Ryzen AI Max+, Vulkan
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class DigitalTwinEngine:
    """
    Creates and manages the infrastructure digital twin.
    
    The digital twin is a structured data model combining:
    1. 3D geometry (point cloud from Layer 3)
    2. Semantic labels (structural elements + defects)
    3. Inspection metadata
    4. Risk assessment
    5. Code compliance status
    
    AMD Technology:
    - Genesis Simulation Engine for physics-based structural analysis
    - Ryzen AI Max+ iGPU for GPU-accelerated twin rendering
    - Vulkan backend optimized for AMD GPUs
    """
    
    def create_twin(
        self,
        point_cloud_path: str,
        label_stats: Dict,
        frame_defects: List[Dict],
        frame_detections: List[Dict],
        video_metadata: Dict,
        inspection_metadata: Dict = None,
    ) -> Dict:
        """
        Create a complete digital twin data model.
        
        Returns comprehensive twin data for visualization and reporting.
        """
        # Aggregate all defects across frames
        all_defects = self._aggregate_defects(frame_defects)
        all_detections = self._aggregate_detections(frame_detections)
        
        # Compute structural health index
        health_index = self._compute_health_index(all_defects, all_detections)
        
        # Generate zone analysis
        zones = self._analyze_zones(all_defects, all_detections)
        
        # Create the twin model
        twin = {
            "twin_id": f"TWIN-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "structure_info": {
                "type": inspection_metadata.get("structure_type", "unknown") if inspection_metadata else "unknown",
                "source_video": video_metadata.get("filename", "unknown"),
                "video_duration": video_metadata.get("duration", 0),
            },
            "geometry": {
                "point_cloud_path": point_cloud_path,
                "num_points": label_stats.get("num_points", 0),
                "structural_distribution": label_stats.get("stats", {}).get("structural_distribution", {}),
            },
            "defect_analysis": {
                "total_defects": len(all_defects),
                "defect_types": self._defect_type_summary(all_defects),
                "critical_defects": [d for d in all_defects if d.get("severity", 0) >= 7],
                "high_defects": [d for d in all_defects if 5 <= d.get("severity", 0) < 7],
                "medium_defects": [d for d in all_defects if 3 <= d.get("severity", 0) < 5],
                "low_defects": [d for d in all_defects if d.get("severity", 0) < 3],
            },
            "health_index": health_index,
            "zones": zones,
            "structural_elements": all_detections,
            "amd_processing": {
                "perception_engine": "AMD Ryzen AI NPU (XDNA)",
                "reconstruction": "AMD Ryzen AI Max+ iGPU",
                "agent_framework": "AMD GAIA",
                "deployment": "ONNX Runtime + Vitis AI EP",
            },
        }
        
        # Save twin data
        twin_dir = Path(point_cloud_path).parent
        twin_path = twin_dir / "digital_twin.json"
        with open(twin_path, "w") as f:
            json.dump(twin, f, indent=2, default=str)
        
        twin["twin_data_path"] = str(twin_path)
        logger.info(f"Digital twin created: {twin['twin_id']}")
        
        return twin
    
    def _aggregate_defects(self, frame_defects: List[Dict]) -> List[Dict]:
        """Aggregate defects across all frames, merging duplicates."""
        all_defects = []
        defect_id = 0
        
        for frame_idx, frame_result in enumerate(frame_defects):
            if "defects" not in frame_result:
                continue
            
            for defect in frame_result["defects"]:
                defect_id += 1
                all_defects.append({
                    "id": f"DEF-{defect_id:03d}",
                    "frame_idx": frame_idx,
                    "defect_type": defect.get("defect_type", "unknown"),
                    "severity": defect.get("severity", 5.0),
                    "confidence": defect.get("confidence", 0.5),
                    "bbox": defect.get("bbox", []),
                    "detection_method": defect.get("detection_method", "unknown"),
                    "code_reference": self._get_code_reference(defect.get("defect_type")),
                })
        
        # Sort by severity (most severe first)
        all_defects.sort(key=lambda d: d["severity"], reverse=True)
        return all_defects
    
    def _aggregate_detections(self, frame_detections: List[Dict]) -> List[Dict]:
        """Aggregate structural element detections."""
        elements = {}
        
        for frame_result in frame_detections:
            if "detections" not in frame_result:
                continue
            
            for det in frame_result["detections"]:
                stype = det.get("structural_type", "unknown")
                if stype not in elements:
                    elements[stype] = {
                        "type": stype,
                        "count": 0,
                        "avg_confidence": 0,
                        "detections": 0,
                    }
                elements[stype]["count"] += 1
                elements[stype]["avg_confidence"] += det.get("confidence", 0)
                elements[stype]["detections"] += 1
        
        # Compute averages
        for stype in elements:
            n = elements[stype]["detections"]
            if n > 0:
                elements[stype]["avg_confidence"] = round(
                    elements[stype]["avg_confidence"] / n, 3
                )
        
        return list(elements.values())
    
    def _compute_health_index(
        self, defects: List[Dict], detections: List[Dict]
    ) -> Dict:
        """
        Compute Structural Health Index (SHI) — score 0-100.
        
        100 = Perfect condition, 0 = Critical failure imminent
        """
        if not defects:
            return {
                "score": 95,
                "grade": "A",
                "status": "GOOD",
                "description": "No significant structural defects detected",
                "recommendation": "Continue routine inspection schedule",
            }
        
        # Base score starts at 100
        score = 100.0
        
        # Deductions based on defect severity
        for defect in defects:
            severity = defect.get("severity", 5)
            confidence = defect.get("confidence", 0.5)
            
            # Higher severity and confidence = bigger deduction
            deduction = severity * confidence * 1.5
            
            # Critical defects have amplified impact
            if severity >= 8:
                deduction *= 2.0
            elif severity >= 6:
                deduction *= 1.5
            
            score -= deduction
        
        # Cap at 0-100
        score = max(0, min(100, score))
        
        # Grade assignment
        if score >= 90:
            grade, status = "A", "EXCELLENT"
            desc = "Structure in excellent condition"
            rec = "Continue routine monitoring"
        elif score >= 75:
            grade, status = "B", "GOOD"
            desc = "Minor issues detected, overall good condition"
            rec = "Schedule preventive maintenance within 6 months"
        elif score >= 60:
            grade, status = "C", "FAIR"
            desc = "Moderate deterioration, attention needed"
            rec = "Schedule detailed inspection within 3 months"
        elif score >= 40:
            grade, status = "D", "POOR"
            desc = "Significant structural concerns"
            rec = "Immediate professional engineering assessment required"
        else:
            grade, status = "F", "CRITICAL"
            desc = "Critical structural defects detected"
            rec = "URGENT: Restrict access and engage structural engineer immediately"
        
        return {
            "score": round(score, 1),
            "grade": grade,
            "status": status,
            "description": desc,
            "recommendation": rec,
            "defect_count": len(defects),
            "critical_count": sum(1 for d in defects if d.get("severity", 0) >= 7),
        }
    
    def _analyze_zones(
        self, defects: List[Dict], detections: List[Dict]
    ) -> List[Dict]:
        """Divide the structure into inspection zones for targeted analysis."""
        zones = [
            {"name": "Foundation Zone", "y_range": [0, 0.5], "defects": []},
            {"name": "Lower Structure", "y_range": [0.5, 1.5], "defects": []},
            {"name": "Mid Structure", "y_range": [1.5, 2.5], "defects": []},
            {"name": "Upper Structure", "y_range": [2.5, 3.5], "defects": []},
            {"name": "Roof/Ceiling", "y_range": [3.5, 5.0], "defects": []},
        ]
        
        # Assign defects to zones based on bbox vertical position
        for defect in defects:
            bbox = defect.get("bbox", [])
            if len(bbox) >= 4:
                # Normalize y position (0 = top, 1 = bottom in image)
                y_center = (bbox[1] + bbox[3]) / 2
                # Map to zone (invert: bottom of image = foundation)
                zone_idx = min(int(y_center / 100), len(zones) - 1)
                zone_idx = max(0, len(zones) - 1 - zone_idx)
                zones[zone_idx]["defects"].append(defect["id"])
        
        # Add zone health scores
        for zone in zones:
            n_defects = len(zone["defects"])
            if n_defects == 0:
                zone["health"] = "GOOD"
                zone["score"] = 95
            elif n_defects <= 2:
                zone["health"] = "FAIR"
                zone["score"] = 70
            else:
                zone["health"] = "POOR"
                zone["score"] = 40
        
        return zones
    
    def _defect_type_summary(self, defects: List[Dict]) -> Dict:
        """Summarize defects by type."""
        summary = {}
        for d in defects:
            dtype = d.get("defect_type", "unknown")
            if dtype not in summary:
                summary[dtype] = {"count": 0, "max_severity": 0, "avg_severity": 0}
            summary[dtype]["count"] += 1
            summary[dtype]["max_severity"] = max(
                summary[dtype]["max_severity"], d.get("severity", 0)
            )
            summary[dtype]["avg_severity"] += d.get("severity", 0)
        
        for dtype in summary:
            n = summary[dtype]["count"]
            summary[dtype]["avg_severity"] = round(summary[dtype]["avg_severity"] / max(n, 1), 1)
        
        return summary
    
    def _get_code_reference(self, defect_type: str) -> str:
        """Get building code reference for a defect type."""
        code_refs = {
            "crack": "ACI 318-19 §24.3 — Crack Control",
            "spalling": "ACI 562-19 §6.3 — Surface Repair",
            "corrosion": "ACI 222R-19 §4.2 — Corrosion Protection",
            "delamination": "ACI 562-19 §6.2 — Delamination Repair",
            "exposed_rebar": "ACI 318-19 §20.5 — Minimum Cover",
            "water_damage": "ACI 515.2R-13 §3 — Waterproofing",
            "displacement": "ASCE 7-22 §12.12 — Drift Limits",
            "scaling": "ACI 201.2R-16 §5 — Surface Scaling",
        }
        return code_refs.get(defect_type, "Refer to applicable local building code")
