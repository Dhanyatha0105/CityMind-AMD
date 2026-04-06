"""
Layer 4 (Stretch): Deviation Analysis Module
Compares current scan against a reference model or previous scan
to detect structural displacement, settlement, and deformation.

AMD Tech: AMD Ryzen AI Max+ iGPU for point cloud processing,
          Genesis Simulation Engine for physics-based analysis.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DeviationAnalyzer:
    """
    Computes geometric deviations between two point clouds or between
    a point cloud and a reference BIM/CAD model.

    Analysis types:
    - Cloud-to-Cloud (C2C): Compare two scans (temporal tracking)
    - Cloud-to-Model (C2M): Compare scan against reference BIM
    - Surface deviation heatmap generation
    - Statistical deviation reporting (RMSE, mean, std, percentiles)
    """

    def __init__(self, max_deviation_mm: float = 50.0, grid_resolution: float = 0.1):
        """
        Args:
            max_deviation_mm: Maximum expected deviation in mm (for colormap clamping)
            grid_resolution: Voxel grid size for downsampling (meters)
        """
        self.max_deviation_mm = max_deviation_mm
        self.grid_resolution = grid_resolution

    def cloud_to_cloud_deviation(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        source_normals: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute point-to-point deviation between source and target clouds.

        For each point in source, find the nearest point in target and
        compute the Euclidean distance (deviation).

        Args:
            source_points: Nx3 array (current scan)
            target_points: Mx3 array (reference scan)
            source_normals: Optional Nx3 normals for signed deviation

        Returns:
            Dict with deviation statistics and per-point deviations
        """
        logger.info(f"Computing C2C deviation: {len(source_points)} → {len(target_points)} points")

        if len(source_points) == 0 or len(target_points) == 0:
            return self._empty_result("cloud_to_cloud")

        try:
            from scipy.spatial import KDTree

            tree = KDTree(target_points)
            distances, indices = tree.query(source_points, k=1)

            # Signed deviation if normals available
            if source_normals is not None and len(source_normals) == len(source_points):
                vectors = target_points[indices] - source_points
                signed_distances = np.sum(vectors * source_normals, axis=1)
            else:
                signed_distances = distances

            return self._compute_statistics(distances, signed_distances, "cloud_to_cloud")

        except ImportError:
            logger.warning("scipy not available, using brute-force nearest neighbor")
            return self._brute_force_c2c(source_points, target_points)

    def _brute_force_c2c(self, source: np.ndarray, target: np.ndarray) -> Dict:
        """Fallback brute-force nearest neighbor (for small clouds)."""
        # Process in batches to limit memory
        batch_size = 1000
        all_distances = []

        for i in range(0, len(source), batch_size):
            batch = source[i:i + batch_size]
            # Compute pairwise distances
            diffs = batch[:, np.newaxis, :] - target[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            min_dists = np.min(dists, axis=1)
            all_distances.extend(min_dists.tolist())

        distances = np.array(all_distances)
        return self._compute_statistics(distances, distances, "cloud_to_cloud")

    def cloud_to_plane_deviation(
        self,
        points: np.ndarray,
        plane_normal: np.ndarray = None,
        plane_point: np.ndarray = None,
    ) -> Dict:
        """
        Compute signed deviation of points from a reference plane.
        Useful for flatness analysis of walls, slabs, floors.

        Args:
            points: Nx3 point array
            plane_normal: Normal vector of the reference plane
            plane_point: A point on the reference plane

        Returns:
            Dict with deviation statistics
        """
        if plane_normal is None:
            # Fit a plane to the points using least squares
            plane_normal, plane_point = self._fit_plane(points)

        # Signed distance from point to plane
        plane_normal = np.array(plane_normal, dtype=np.float64)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        plane_point = np.array(plane_point, dtype=np.float64)

        signed_distances = np.dot(points - plane_point, plane_normal)
        abs_distances = np.abs(signed_distances)

        return self._compute_statistics(abs_distances, signed_distances, "cloud_to_plane")

    def _fit_plane(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a plane to points using SVD (least-squares)."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # SVD
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]  # Last right singular vector = normal

        return normal, centroid

    def temporal_deviation(
        self,
        scan_history: List[Dict],
    ) -> Dict:
        """
        Analyze deviation trends across multiple temporal scans.

        Args:
            scan_history: List of dicts with 'timestamp', 'points', 'metadata'

        Returns:
            Dict with temporal trend analysis
        """
        if len(scan_history) < 2:
            return {
                "analysis_type": "temporal",
                "error": "Need at least 2 scans for temporal analysis",
                "scan_count": len(scan_history),
            }

        # Compare each consecutive pair
        pairwise_results = []
        for i in range(1, len(scan_history)):
            prev_scan = scan_history[i - 1]
            curr_scan = scan_history[i]

            result = self.cloud_to_cloud_deviation(
                source_points=np.array(curr_scan.get("points", [])),
                target_points=np.array(prev_scan.get("points", [])),
            )

            result["time_delta_days"] = (
                datetime.fromisoformat(curr_scan["timestamp"])
                - datetime.fromisoformat(prev_scan["timestamp"])
            ).days

            pairwise_results.append(result)

        # Compute trend
        mean_deviations = [r["mean_mm"] for r in pairwise_results]
        max_deviations = [r["max_mm"] for r in pairwise_results]

        trend = "stable"
        if len(mean_deviations) >= 2:
            if mean_deviations[-1] > mean_deviations[0] * 1.5:
                trend = "increasing"
            elif mean_deviations[-1] < mean_deviations[0] * 0.7:
                trend = "decreasing"

        return {
            "analysis_type": "temporal",
            "scan_count": len(scan_history),
            "pairwise_results": pairwise_results,
            "trend": trend,
            "mean_deviation_trend_mm": mean_deviations,
            "max_deviation_trend_mm": max_deviations,
            "alert": trend == "increasing",
            "alert_message": (
                f"ALERT: Deviation increasing — latest mean deviation "
                f"{mean_deviations[-1]:.1f}mm vs initial {mean_deviations[0]:.1f}mm"
                if trend == "increasing"
                else "Deviation within normal range"
            ),
        }

    def generate_deviation_heatmap(
        self,
        points: np.ndarray,
        deviations: np.ndarray,
        output_path: str = None,
    ) -> Dict:
        """
        Generate a 2D deviation heatmap (bird's-eye projection).

        Args:
            points: Nx3 points
            deviations: N deviations (one per point)
            output_path: Optional path to save heatmap image

        Returns:
            Dict with heatmap data
        """
        if len(points) == 0:
            return {"error": "No points for heatmap"}

        # Project to XY plane (bird's-eye)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        # Grid resolution
        grid_size = max(int((x_max - x_min) / self.grid_resolution), 10)
        grid_size_y = max(int((y_max - y_min) / self.grid_resolution), 10)

        # Cap grid size for memory
        grid_size = min(grid_size, 500)
        grid_size_y = min(grid_size_y, 500)

        # Create grid
        heatmap = np.full((grid_size_y, grid_size), np.nan)
        counts = np.zeros((grid_size_y, grid_size), dtype=int)

        for i, (pt, dev) in enumerate(zip(points, deviations)):
            gx = int((pt[0] - x_min) / (x_max - x_min + 1e-8) * (grid_size - 1))
            gy = int((pt[1] - y_min) / (y_max - y_min + 1e-8) * (grid_size_y - 1))
            gx = np.clip(gx, 0, grid_size - 1)
            gy = np.clip(gy, 0, grid_size_y - 1)

            if np.isnan(heatmap[gy, gx]):
                heatmap[gy, gx] = 0
            heatmap[gy, gx] += dev
            counts[gy, gx] += 1

        # Average
        valid = counts > 0
        heatmap[valid] /= counts[valid]

        # Save if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                im = ax.imshow(
                    heatmap,
                    cmap="RdYlGn_r",
                    aspect="auto",
                    vmin=0,
                    vmax=self.max_deviation_mm,
                    origin="lower",
                )
                plt.colorbar(im, ax=ax, label="Deviation (mm)")
                ax.set_title("CityMind — Structural Deviation Heatmap")
                ax.set_xlabel("X (grid)")
                ax.set_ylabel("Y (grid)")
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Deviation heatmap saved: {output_path}")
            except Exception as e:
                logger.warning(f"Could not save heatmap image: {e}")

        return {
            "grid_shape": [grid_size_y, grid_size],
            "coverage": float(np.sum(valid) / (grid_size * grid_size_y)),
            "max_deviation_mm": float(np.nanmax(heatmap)) if np.any(valid) else 0,
            "mean_deviation_mm": float(np.nanmean(heatmap)) if np.any(valid) else 0,
            "output_path": output_path,
        }

    def _compute_statistics(
        self,
        abs_distances: np.ndarray,
        signed_distances: np.ndarray,
        analysis_type: str,
    ) -> Dict:
        """Compute comprehensive deviation statistics."""
        # Convert to mm (assuming meters input)
        abs_mm = abs_distances * 1000.0
        signed_mm = signed_distances * 1000.0

        return {
            "analysis_type": analysis_type,
            "num_points": len(abs_distances),
            "mean_mm": float(np.mean(abs_mm)),
            "std_mm": float(np.std(abs_mm)),
            "min_mm": float(np.min(abs_mm)),
            "max_mm": float(np.max(abs_mm)),
            "rmse_mm": float(np.sqrt(np.mean(abs_mm ** 2))),
            "median_mm": float(np.median(abs_mm)),
            "p95_mm": float(np.percentile(abs_mm, 95)),
            "p99_mm": float(np.percentile(abs_mm, 99)),
            "signed_mean_mm": float(np.mean(signed_mm)),
            "signed_std_mm": float(np.std(signed_mm)),
            "within_tolerance": {
                "1mm": float(np.mean(abs_mm <= 1.0) * 100),
                "5mm": float(np.mean(abs_mm <= 5.0) * 100),
                "10mm": float(np.mean(abs_mm <= 10.0) * 100),
                "25mm": float(np.mean(abs_mm <= 25.0) * 100),
            },
            "severity": self._classify_deviation(float(np.mean(abs_mm))),
            "timestamp": datetime.now().isoformat(),
        }

    def _classify_deviation(self, mean_deviation_mm: float) -> str:
        """Classify deviation severity."""
        if mean_deviation_mm < 2.0:
            return "negligible"
        elif mean_deviation_mm < 5.0:
            return "minor"
        elif mean_deviation_mm < 15.0:
            return "moderate"
        elif mean_deviation_mm < 30.0:
            return "significant"
        else:
            return "critical"

    def _empty_result(self, analysis_type: str) -> Dict:
        """Return empty result for edge cases."""
        return {
            "analysis_type": analysis_type,
            "num_points": 0,
            "error": "Empty point cloud(s)",
            "mean_mm": 0,
            "rmse_mm": 0,
            "severity": "unknown",
        }

    def analyze_structure(
        self,
        current_cloud: np.ndarray,
        reference_cloud: Optional[np.ndarray] = None,
        zones: Optional[Dict] = None,
    ) -> Dict:
        """
        Full structural deviation analysis.

        Args:
            current_cloud: Nx3 current scan
            reference_cloud: Optional Mx3 reference (previous scan or BIM)
            zones: Optional zone definitions for per-zone analysis

        Returns:
            Comprehensive deviation analysis report
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "current_points": len(current_cloud),
        }

        # Overall flatness (fit plane)
        results["flatness"] = self.cloud_to_plane_deviation(current_cloud)

        # C2C deviation if reference available
        if reference_cloud is not None and len(reference_cloud) > 0:
            results["reference_points"] = len(reference_cloud)
            results["deviation"] = self.cloud_to_cloud_deviation(
                current_cloud, reference_cloud
            )
        else:
            results["deviation"] = {"note": "No reference cloud for C2C analysis"}

        # Zone-based analysis
        if zones:
            zone_results = {}
            for zone_name, zone_bounds in zones.items():
                z_min, z_max = zone_bounds.get("z_min", -np.inf), zone_bounds.get("z_max", np.inf)
                mask = (current_cloud[:, 2] >= z_min) & (current_cloud[:, 2] < z_max)
                zone_points = current_cloud[mask]
                if len(zone_points) > 10:
                    zone_results[zone_name] = self.cloud_to_plane_deviation(zone_points)
                    zone_results[zone_name]["point_count"] = len(zone_points)
            results["zone_analysis"] = zone_results

        # Generate severity summary
        main_dev = results.get("deviation", {})
        flatness = results.get("flatness", {})

        results["summary"] = {
            "overall_severity": main_dev.get("severity", flatness.get("severity", "unknown")),
            "flatness_rmse_mm": flatness.get("rmse_mm", 0),
            "deviation_rmse_mm": main_dev.get("rmse_mm", 0),
            "needs_attention": (
                main_dev.get("severity", "negligible") in ("significant", "critical")
                or flatness.get("severity", "negligible") in ("significant", "critical")
            ),
        }

        return results
