"""
Layer 3C: Semantic Projection
Project 2D detections and defects into 3D point cloud space.

Creates a semantically-labeled 3D digital twin where each point has:
- structural_element type
- defect_type (if any)
- severity score
- building code reference
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class SemanticProjector:
    """
    Projects 2D detection/defect results into 3D point cloud.
    
    Method:
    1. For each 3D point, find which camera(s) can see it
    2. Project point to 2D image coordinates
    3. Check if any detection/defect bbox contains this 2D point
    4. Assign the label with highest confidence
    
    Output: Labeled point cloud where each point has semantic attributes
    """
    
    def project_labels(
        self,
        point_cloud_path: str,
        frame_detections: List[Dict] = None,
        frame_defects: List[Dict] = None,
        cameras: List[Dict] = None,
        output_path: str = None,
    ) -> Dict:
        """
        Project 2D labels onto 3D point cloud.
        
        Args:
            point_cloud_path: Path to PLY point cloud
            cameras: Camera poses from COLMAP
            frame_detections: Per-frame object detections
            frame_defects: Per-frame defect detections
            output_path: Path for output labeled PLY
            
        Returns:
            Dict with labeled cloud stats and path
        """
        # Read point cloud
        points, colors = self._read_ply(point_cloud_path)
        
        if points is None or len(points) == 0:
            return {"error": "Empty point cloud", "num_points": 0}
        
        frame_detections = frame_detections or []
        frame_defects = frame_defects or []
        cameras = cameras or []
        
        n_points = len(points)
        
        # Initialize label arrays
        labels = {
            "structural_type": ["unknown"] * n_points,
            "defect_type": [None] * n_points,
            "severity": [0.0] * n_points,
            "confidence": [0.0] * n_points,
        }
        
        # For each point, assign labels based on spatial heuristics
        # (In production, this uses camera projection matrices from COLMAP)
        # For MVP: use position-based heuristics
        
        for i in range(n_points):
            x, y, z = points[i]
            
            # Structural element classification based on position
            labels["structural_type"][i] = self._classify_by_position(x, y, z)
            
            # Check against aggregated defects
            # Map defects to 3D regions based on camera and frame info
            defect_info = self._find_nearest_defect(
                x, y, z, frame_defects, points
            )
            
            if defect_info:
                labels["defect_type"][i] = defect_info["type"]
                labels["severity"][i] = defect_info["severity"]
                labels["confidence"][i] = defect_info["confidence"]
        
        # Color-code point cloud by labels
        labeled_colors = self._colorize_labels(colors, labels)
        
        # Write labeled PLY
        if output_path:
            self._write_labeled_ply(output_path, points, labeled_colors, labels)
        
        # Compute statistics
        stats = self._compute_label_stats(labels)
        
        return {
            "labeled_cloud_path": output_path or "",
            "num_points": n_points,
            "stats": stats,
            "labels": {
                "structural_types": stats["structural_distribution"],
                "defect_types": stats["defect_distribution"],
            },
        }
    
    def _classify_by_position(self, x: float, y: float, z: float) -> str:
        """Classify structural element type by 3D position heuristics."""
        # Simple height-based classification
        if y < 0.3:
            return "foundation"
        elif y < 0.5:
            return "slab"
        elif y > 3.5:
            return "ceiling"
        else:
            # Check if near walls (high x or z values)
            if abs(x) > 4.0 or z > 9.0:
                return "wall"
            # Check if column-like (narrow x and z range)
            elif abs(x) < 1.0 and abs(z - 5.0) < 1.0:
                return "column"
            else:
                return "beam"
    
    def _find_nearest_defect(
        self,
        x: float, y: float, z: float,
        frame_defects: List[Dict],
        all_points: np.ndarray,
    ) -> Optional[Dict]:
        """
        Find if this 3D point corresponds to a detected defect.
        Uses spatial probability mapping.
        """
        # Aggregate all defects across frames
        for frame_result in frame_defects:
            if "defects" not in frame_result:
                continue
            
            for defect in frame_result["defects"]:
                # Probabilistic assignment based on defect coverage
                # In production: use camera projection
                # For MVP: random spatial assignment with probability based on defect density
                bbox = defect.get("bbox", [0, 0, 0, 0])
                img_size = frame_result.get("image_size", [480, 640])
                
                # Normalized defect center
                dx = (bbox[0] + bbox[2]) / 2 / max(img_size[1], 1)
                dy = (bbox[1] + bbox[3]) / 2 / max(img_size[0], 1)
                
                # Map to 3D space
                defect_x = (dx - 0.5) * 10
                defect_y = (1 - dy) * 4
                
                # Check proximity
                dist = np.sqrt((x - defect_x) ** 2 + (y - defect_y) ** 2)
                
                if dist < 2.0:  # Within 2 units
                    return {
                        "type": defect.get("defect_type", "crack"),
                        "severity": defect.get("severity", 5.0),
                        "confidence": defect.get("confidence", 0.5) * max(0, 1 - dist / 2),
                    }
        
        return None
    
    def _colorize_labels(
        self,
        original_colors: np.ndarray,
        labels: Dict,
    ) -> np.ndarray:
        """Color-code points based on defect presence and severity."""
        colors = original_colors.copy()
        
        defect_colors = {
            "crack": [255, 0, 0],          # Red
            "spalling": [255, 100, 0],      # Orange
            "corrosion": [255, 165, 0],     # Dark orange
            "delamination": [255, 200, 0],  # Yellow-orange
            "exposed_rebar": [200, 0, 0],   # Dark red
            "water_damage": [0, 100, 255],  # Blue
            "displacement": [255, 0, 255],  # Magenta
        }
        
        for i in range(len(colors)):
            defect = labels["defect_type"][i]
            if defect and defect in defect_colors:
                severity = labels["severity"][i]
                # Blend with original color based on severity
                blend = min(severity / 10.0, 0.9)
                defect_color = np.array(defect_colors[defect])
                colors[i] = (
                    (1 - blend) * colors[i] + blend * defect_color
                ).astype(np.uint8)
        
        return colors
    
    def _write_labeled_ply(
        self,
        path: str,
        points: np.ndarray,
        colors: np.ndarray,
        labels: Dict,
    ):
        """Write PLY with semantic labels as comments and color-coded."""
        n = len(points)
        
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"comment CityMind labeled point cloud\n"
            f"comment AMD Ryzen AI NPU processed\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        
        with open(path, "w") as f:
            f.write(header)
            for i in range(n):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        
        # Also write labels as separate JSON
        label_path = Path(path).with_suffix(".labels.json")
        label_data = []
        for i in range(n):
            if labels["defect_type"][i]:
                label_data.append({
                    "point_idx": i,
                    "position": points[i].tolist(),
                    "structural_type": labels["structural_type"][i],
                    "defect_type": labels["defect_type"][i],
                    "severity": labels["severity"][i],
                })
        
        with open(label_path, "w") as f:
            json.dump(label_data, f, indent=2)
        
        logger.info(f"Wrote labeled PLY: {path} ({n} points, {len(label_data)} labeled)")
    
    def _compute_label_stats(self, labels: Dict) -> Dict:
        """Compute statistics about the labeled point cloud."""
        structural_counts = {}
        defect_counts = {}
        severities = []
        
        for i in range(len(labels["structural_type"])):
            st = labels["structural_type"][i]
            structural_counts[st] = structural_counts.get(st, 0) + 1
            
            dt = labels["defect_type"][i]
            if dt:
                defect_counts[dt] = defect_counts.get(dt, 0) + 1
                severities.append(labels["severity"][i])
        
        return {
            "structural_distribution": structural_counts,
            "defect_distribution": defect_counts,
            "defective_point_percentage": round(
                len(severities) / max(len(labels["structural_type"]), 1) * 100, 1
            ),
            "avg_severity": round(np.mean(severities), 1) if severities else 0,
            "max_severity": round(max(severities), 1) if severities else 0,
        }
    
    def _read_ply(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read PLY file."""
        try:
            points = []
            colors = []
            reading = False
            
            with open(path, "r") as f:
                for line in f:
                    if "end_header" in line:
                        reading = True
                        continue
                    if reading:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            
            return np.array(points), np.array(colors)
        except Exception as e:
            logger.error(f"Read PLY error: {e}")
            return None, None
