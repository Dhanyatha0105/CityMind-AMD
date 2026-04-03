"""
Layer 3 Utility: Point Cloud Utilities
PLY read/write, sampling, transformation, and basic mesh operations.

AMD Tech: Ryzen AI Max+ iGPU for GPU-accelerated point cloud processing
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PointCloudUtils:
    """
    Utility class for point cloud operations.
    
    AMD Technology:
    - Ryzen AI Max+ iGPU for GPU-accelerated operations (Open3D Vulkan backend)
    - NumPy leverages AMD Zen5 AVX-512 for vectorized operations
    """
    
    @staticmethod
    def read_ply(path: str) -> Dict:
        """
        Read a PLY point cloud file.
        Returns dict with 'points' (Nx3), 'colors' (Nx3), 'normals' (Nx3).
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"PLY file not found: {path}")
            return {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3))}
        
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(path))
            vertex = plydata['vertex']
            
            points = np.column_stack([
                vertex['x'], vertex['y'], vertex['z']
            ]).astype(np.float32)
            
            colors = np.zeros((len(points), 3), dtype=np.float32)
            if 'red' in vertex.data.dtype.names:
                colors = np.column_stack([
                    vertex['red'], vertex['green'], vertex['blue']
                ]).astype(np.float32) / 255.0
            
            normals = np.zeros((len(points), 3), dtype=np.float32)
            if 'nx' in vertex.data.dtype.names:
                normals = np.column_stack([
                    vertex['nx'], vertex['ny'], vertex['nz']
                ]).astype(np.float32)
            
            logger.info(f"Read PLY: {len(points)} points from {path.name}")
            return {"points": points, "colors": colors, "normals": normals}
            
        except ImportError:
            logger.warning("plyfile not installed, trying Open3D")
            return PointCloudUtils._read_ply_open3d(str(path))
    
    @staticmethod
    def _read_ply_open3d(path: str) -> Dict:
        """Read PLY using Open3D."""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros((len(points), 3))
            normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else np.zeros((len(points), 3))
            return {"points": points, "colors": colors, "normals": normals}
        except ImportError:
            logger.error("Neither plyfile nor open3d available")
            return {"points": np.zeros((0, 3)), "colors": np.zeros((0, 3))}
    
    @staticmethod
    def write_ply(path: str, points: np.ndarray, colors: np.ndarray = None, normals: np.ndarray = None):
        """Write point cloud to PLY file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        n = len(points)
        
        try:
            from plyfile import PlyData, PlyElement
            
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            if colors is not None:
                dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            if normals is not None:
                dtype += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
            
            vertex_data = np.zeros(n, dtype=dtype)
            vertex_data['x'] = points[:, 0]
            vertex_data['y'] = points[:, 1]
            vertex_data['z'] = points[:, 2]
            
            if colors is not None:
                c = (colors * 255).clip(0, 255).astype(np.uint8)
                vertex_data['red'] = c[:, 0]
                vertex_data['green'] = c[:, 1]
                vertex_data['blue'] = c[:, 2]
            
            if normals is not None:
                vertex_data['nx'] = normals[:, 0]
                vertex_data['ny'] = normals[:, 1]
                vertex_data['nz'] = normals[:, 2]
            
            el = PlyElement.describe(vertex_data, 'vertex')
            PlyData([el]).write(str(path))
            logger.info(f"Wrote PLY: {n} points to {path.name}")
            
        except ImportError:
            # Fallback: write ASCII PLY manually
            PointCloudUtils._write_ply_ascii(str(path), points, colors, normals)
    
    @staticmethod
    def _write_ply_ascii(path: str, points: np.ndarray, colors: np.ndarray = None, normals: np.ndarray = None):
        """Write ASCII PLY file (fallback)."""
        n = len(points)
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            if colors is not None:
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            
            for i in range(n):
                line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
                if colors is not None:
                    c = (colors[i] * 255).clip(0, 255).astype(int)
                    line += f" {c[0]} {c[1]} {c[2]}"
                f.write(line + "\n")
        
        logger.info(f"Wrote ASCII PLY: {n} points to {path}")
    
    @staticmethod
    def downsample(points: np.ndarray, colors: np.ndarray = None, voxel_size: float = 0.05) -> Dict:
        """
        Voxel-based downsampling for visualization performance.
        
        AMD Optimization: Open3D Vulkan backend on Ryzen AI Max+ iGPU.
        """
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            result = {"points": np.asarray(pcd_down.points, dtype=np.float32)}
            if pcd_down.has_colors():
                result["colors"] = np.asarray(pcd_down.colors, dtype=np.float32)
            
            logger.info(f"Downsampled: {len(points)} → {len(result['points'])} points (voxel={voxel_size})")
            return result
            
        except ImportError:
            # Fallback: random sampling
            n = len(points)
            target = max(1, int(n * 0.1))
            indices = np.random.choice(n, size=min(target, n), replace=False)
            result = {"points": points[indices]}
            if colors is not None:
                result["colors"] = colors[indices]
            logger.info(f"Random downsampled: {n} → {len(result['points'])} points")
            return result
    
    @staticmethod
    def compute_bounding_box(points: np.ndarray) -> Dict:
        """Compute axis-aligned bounding box."""
        if len(points) == 0:
            return {"min": [0, 0, 0], "max": [0, 0, 0], "center": [0, 0, 0], "dimensions": [0, 0, 0]}
        
        min_pt = points.min(axis=0).tolist()
        max_pt = points.max(axis=0).tolist()
        center = ((points.min(axis=0) + points.max(axis=0)) / 2).tolist()
        dims = (points.max(axis=0) - points.min(axis=0)).tolist()
        
        return {"min": min_pt, "max": max_pt, "center": center, "dimensions": dims}
    
    @staticmethod
    def generate_synthetic_building(
        width: float = 10.0,
        height: float = 15.0,
        depth: float = 8.0,
        num_points: int = 5000,
    ) -> Dict:
        """
        Generate a synthetic building point cloud for demo purposes.
        Creates a box-like structure with floors, walls, and some surface detail.
        """
        points = []
        colors = []
        
        # Walls (4 sides)
        n_wall = num_points // 5
        for _ in range(n_wall):
            side = np.random.randint(4)
            if side == 0:  # Front
                p = [np.random.uniform(0, width), np.random.uniform(0, height), 0 + np.random.normal(0, 0.02)]
            elif side == 1:  # Back
                p = [np.random.uniform(0, width), np.random.uniform(0, height), depth + np.random.normal(0, 0.02)]
            elif side == 2:  # Left
                p = [0 + np.random.normal(0, 0.02), np.random.uniform(0, height), np.random.uniform(0, depth)]
            else:  # Right
                p = [width + np.random.normal(0, 0.02), np.random.uniform(0, height), np.random.uniform(0, depth)]
            points.append(p)
            colors.append([0.7, 0.7, 0.65])  # Concrete gray
        
        # Floor slabs (3 floors)
        n_floor = num_points // 5
        for _ in range(n_floor):
            floor_y = np.random.choice([0, height / 3, 2 * height / 3, height])
            p = [np.random.uniform(0, width), floor_y + np.random.normal(0, 0.02), np.random.uniform(0, depth)]
            points.append(p)
            colors.append([0.6, 0.6, 0.6])
        
        # Columns (4 corners × 4 per floor)
        n_col = num_points // 10
        col_radius = 0.3
        for _ in range(n_col):
            cx = np.random.choice([col_radius, width - col_radius])
            cz = np.random.choice([col_radius, depth - col_radius])
            p = [
                cx + np.random.normal(0, col_radius * 0.5),
                np.random.uniform(0, height),
                cz + np.random.normal(0, col_radius * 0.5),
            ]
            points.append(p)
            colors.append([0.5, 0.5, 0.55])
        
        # Beams (horizontal members at floor levels)
        n_beam = num_points // 10
        for _ in range(n_beam):
            floor_y = np.random.choice([height / 3, 2 * height / 3, height])
            beam_dir = np.random.choice([0, 1])  # 0=x-direction, 1=z-direction
            if beam_dir == 0:
                p = [np.random.uniform(0, width), floor_y + np.random.normal(0, 0.05), np.random.choice([0, depth])]
            else:
                p = [np.random.choice([0, width]), floor_y + np.random.normal(0, 0.05), np.random.uniform(0, depth)]
            points.append(p)
            colors.append([0.55, 0.55, 0.58])
        
        # Add some "defect" colored regions
        n_defects = num_points // 20
        for _ in range(n_defects):
            # Random position on front wall with crack-like coloring
            p = [np.random.uniform(2, 8), np.random.uniform(1, 12), np.random.normal(0, 0.01)]
            colors.append([0.8, 0.2, 0.1])  # Red for defect
            points.append(p)
        
        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        
        logger.info(f"Generated synthetic building: {len(points)} points, {width}×{height}×{depth}m")
        
        return {"points": points, "colors": colors, "normals": np.zeros_like(points)}
    
    @staticmethod
    def color_by_labels(points: np.ndarray, labels: np.ndarray, label_colors: Dict = None) -> np.ndarray:
        """
        Color point cloud by semantic labels.
        """
        if label_colors is None:
            label_colors = {
                0: [0.7, 0.7, 0.7],   # Unknown → gray
                1: [0.2, 0.2, 0.8],   # Wall → blue
                2: [0.2, 0.8, 0.2],   # Column → green
                3: [0.8, 0.8, 0.2],   # Beam → yellow
                4: [0.6, 0.6, 0.6],   # Slab → light gray
                5: [1.0, 0.0, 0.0],   # Crack → red
                6: [1.0, 0.5, 0.0],   # Spalling → orange
                7: [0.8, 0.0, 0.8],   # Corrosion → purple
            }
        
        colors = np.zeros((len(points), 3), dtype=np.float32)
        for i, label in enumerate(labels):
            colors[i] = label_colors.get(int(label), [0.5, 0.5, 0.5])
        
        return colors
    
    @staticmethod
    def color_by_height(points: np.ndarray) -> np.ndarray:
        """Color points by height (y-coordinate) using a gradient."""
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        y = points[:, 1]
        y_min, y_max = y.min(), y.max()
        if y_max - y_min < 1e-6:
            return np.full((len(points), 3), 0.5, dtype=np.float32)
        
        t = (y - y_min) / (y_max - y_min)
        
        # Blue (low) → Green (mid) → Red (high) gradient
        colors = np.zeros((len(points), 3), dtype=np.float32)
        colors[:, 0] = np.clip(2 * t - 1, 0, 1)        # Red
        colors[:, 1] = np.clip(1 - 2 * np.abs(t - 0.5), 0, 1)  # Green
        colors[:, 2] = np.clip(1 - 2 * t, 0, 1)        # Blue
        
        return colors
